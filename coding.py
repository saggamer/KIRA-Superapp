import base64
import difflib
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class KiraCodingMixin:
    """Provider-backed coding workspace managed by Orchestrator V1."""

    CODING_COMMAND_TIMEOUT_SECONDS = 120
    CODING_COMMAND_OUTPUT_LIMIT_BYTES = 512 * 1024
    CODING_COMMAND_DRAIN_TIMEOUT_SECONDS = 2.0
    CODING_MAX_PLAN_REPAIRS = 1
    CODING_PROVIDERS = {
        "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
        "anthropic": {"label": "Anthropic", "env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com"},
        "gemini": {"label": "Google Gemini", "env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com"},
        "deepseek": {"label": "DeepSeek", "env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
        "nvidia_nim": {"label": "NVIDIA NIM", "env": "NVIDIA_API_KEY", "base_url": "https://integrate.api.nvidia.com/v1"},
        "openrouter": {"label": "OpenRouter", "env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
        "custom": {"label": "OpenAI-compatible", "env": "KIRA_CODING_API_KEY", "base_url": "http://127.0.0.1:1234/v1"},
    }

    CODING_IGNORED_DIRS = {
        ".git", ".hg", ".svn", ".idea", ".pytest_cache", ".mypy_cache",
        "node_modules", "vendor", "dist", "build", "target", "coverage",
        "__pycache__", ".venv", "venv", "env", ".next", ".turbo",
    }
    CODING_TEXT_SUFFIXES = {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
        ".json", ".jsonc", ".toml", ".yaml", ".yml", ".xml", ".md",
        ".txt", ".sh", ".zsh", ".bash", ".fish", ".sql", ".graphql",
        ".go", ".rs", ".java", ".kt", ".kts", ".swift", ".m", ".mm",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php",
        ".dart", ".lua", ".r", ".ex", ".exs", ".tf", ".proto",
    }
    CODING_SAFE_FILENAMES = {
        "dockerfile", "makefile", "procfile", "gemfile", "rakefile",
        "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
        "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod",
    }

    def _init_coding_mode(self):
        self.coding_workspace = os.path.join(self.architect_workspace, "coding")
        self.coding_settings_path = os.path.join(self.coding_workspace, "providers.json")
        self.coding_usage_path = os.path.join(self.coding_workspace, "api_usage.json")
        self.coding_sessions_path = os.path.join(self.coding_workspace, "sessions")
        self.coding_backups_path = os.path.join(self.coding_workspace, "backups")
        self.coding_runs_path = os.path.join(self.coding_workspace, "runs")
        self.coding_lock = threading.RLock()
        self.coding_execution_lock = threading.RLock()
        self.coding_active_executions = {}
        self.coding_execution_history = []
        self.coding_key_view_unlocked_until = 0.0
        os.makedirs(self.coding_workspace, exist_ok=True)
        os.makedirs(self.coding_sessions_path, exist_ok=True)
        os.makedirs(self.coding_backups_path, exist_ok=True)
        os.makedirs(self.coding_runs_path, exist_ok=True)
        defaults = {"default_provider": "", "providers": {}, "key_view_auth": {}}
        if not os.path.exists(self.coding_settings_path):
            self._coding_write_json(self.coding_settings_path, defaults)
        settings = self._coding_read_json(self.coding_settings_path, defaults)
        profiles = settings.setdefault("providers", {})
        changed = False
        if not isinstance(settings.get("key_view_auth"), dict):
            settings["key_view_auth"] = {}
            changed = True
        for provider, meta in self.CODING_PROVIDERS.items():
            profile = profiles.setdefault(provider, {})
            if not str(profile.get("base_url", "")).strip():
                profile["base_url"] = meta["base_url"]
                changed = True
        if changed:
            self._coding_write_json(self.coding_settings_path, settings)
        if not os.path.exists(self.coding_usage_path):
            self._coding_write_json(self.coding_usage_path, {"events": []})

    def _coding_budget_defaults(self):
        return {
            "monthly_cap_usd": 0.0,
            "task_cap_usd": 0.0,
            "input_usd_per_million": 0.0,
            "output_usd_per_million": 0.0,
            "starting_spend_usd": 0.0,
            "policy": "hard_stop",
        }

    def _coding_budget_config(self, provider):
        settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
        saved = settings.get("providers", {}).get(provider, {})
        budget = dict(self._coding_budget_defaults())
        if isinstance(saved.get("budget"), dict):
            budget.update(saved["budget"])
        for key in [
            "monthly_cap_usd", "task_cap_usd", "input_usd_per_million",
            "output_usd_per_million", "starting_spend_usd",
        ]:
            try:
                budget[key] = max(0.0, float(budget.get(key, 0) or 0))
            except (TypeError, ValueError):
                budget[key] = 0.0
        budget["policy"] = "finish_current" if budget.get("policy") == "finish_current" else "hard_stop"
        return budget

    def _coding_month_key(self, timestamp=None):
        return time.strftime("%Y-%m", time.localtime(timestamp or time.time()))

    def get_coding_budget(self, provider):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider."}
        budget = self._coding_budget_config(provider)
        month = self._coding_month_key()
        with self.coding_lock:
            ledger = self._coding_read_json(self.coding_usage_path, {"events": []})
        events = [
            event for event in ledger.get("events", [])
            if isinstance(event, dict) and event.get("provider") == provider and event.get("month") == month
        ]
        api_spend = round(sum(float(event.get("cost_usd", 0) or 0) for event in events), 8)
        total_spend = round(api_spend + budget["starting_spend_usd"], 8)
        cap = budget["monthly_cap_usd"]
        remaining = max(0.0, cap - total_spend) if cap else None
        percent = min(100.0, (total_spend / cap) * 100.0) if cap else 0.0
        return {
            "ok": True,
            **budget,
            "month": month,
            "tracked_api_spend_usd": api_spend,
            "total_spend_usd": total_spend,
            "remaining_usd": remaining,
            "percent": round(percent, 2),
            "request_count": len(events),
            "cap_enabled": cap > 0,
            "pricing_configured": budget["input_usd_per_million"] > 0 or budget["output_usd_per_million"] > 0,
            "blocked": bool(cap and total_spend >= cap),
            "scope_note": "Tracks calls made by KIRA OS plus the user-entered starting spend.",
        }

    def save_coding_budget(
        self, provider, monthly_cap_usd=0, task_cap_usd=0,
        input_usd_per_million=0, output_usd_per_million=0,
        starting_spend_usd=0, policy="hard_stop",
    ):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider."}
        try:
            budget = {
                "monthly_cap_usd": max(0.0, float(monthly_cap_usd or 0)),
                "task_cap_usd": max(0.0, float(task_cap_usd or 0)),
                "input_usd_per_million": max(0.0, float(input_usd_per_million or 0)),
                "output_usd_per_million": max(0.0, float(output_usd_per_million or 0)),
                "starting_spend_usd": max(0.0, float(starting_spend_usd or 0)),
                "policy": "finish_current" if policy == "finish_current" else "hard_stop",
            }
        except (TypeError, ValueError):
            return {"ok": False, "error": "Budget and pricing values must be valid non-negative numbers."}
        if any(value > 1_000_000 for key, value in budget.items() if key != "policy"):
            return {"ok": False, "error": "A budget or pricing value is unreasonably large."}
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
            profile = settings.setdefault("providers", {}).setdefault(provider, {})
            profile["budget"] = budget
            self._coding_write_json(self.coding_settings_path, settings)
        return self.get_coding_budget(provider)

    def reset_coding_usage(self, provider):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider."}
        month = self._coding_month_key()
        with self.coding_lock:
            ledger = self._coding_read_json(self.coding_usage_path, {"events": []})
            ledger["events"] = [
                event for event in ledger.get("events", [])
                if not (isinstance(event, dict) and event.get("provider") == provider and event.get("month") == month)
            ][-2000:]
            self._coding_write_json(self.coding_usage_path, ledger)
        return self.get_coding_budget(provider)

    def _coding_write_json(self, path, payload):
        if hasattr(self, "_atomic_write_json"):
            self._atomic_write_json(path, payload)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def _coding_read_json(self, path, default):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, type(default)) else default
        except Exception:
            return default

    def _coding_key_service(self, provider):
        provider = re.sub(r"[^a-z0-9_-]", "", str(provider or "").lower())
        return f"ai.kiraos.coding.{provider}"

    def _coding_store_key(self, provider, api_key):
        key = str(api_key or "").strip()
        if not key:
            return False
        security = shutil.which("security") or "/usr/bin/security"
        result = subprocess.run(
            [security, "add-generic-password", "-U", "-a", "default", "-s", self._coding_key_service(provider), "-w", key],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError("macOS Keychain rejected the API key. Unlock Keychain Access and try again.")
        return True

    def _coding_load_key(self, provider):
        meta = self.CODING_PROVIDERS.get(provider, {})
        env_key = str(os.environ.get(meta.get("env", ""), "") or "").strip()
        security = shutil.which("security") or "/usr/bin/security"
        try:
            result = subprocess.run(
                [security, "find-generic-password", "-a", "default", "-s", self._coding_key_service(provider), "-w"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip(), "keychain"
        except Exception:
            pass
        return (env_key, "environment") if env_key else ("", "missing")

    def _coding_delete_key(self, provider):
        security = shutil.which("security") or "/usr/bin/security"
        subprocess.run(
            [security, "delete-generic-password", "-a", "default", "-s", self._coding_key_service(provider)],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _coding_redact(self, text, secrets=None):
        clean = str(text or "")
        for secret in secrets or []:
            if secret:
                clean = clean.replace(str(secret), "[REDACTED]")
        clean = re.sub(
            r"([?&](?:key|api_key|token)=)[^&\s]+",
            r"\1[REDACTED]",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\b(sk|key|token)-[A-Za-z0-9._-]{8,}\b", "[REDACTED]", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bnvapi-[A-Za-z0-9._-]{12,}\b", "[REDACTED]", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[REDACTED]", clean)
        clean = re.sub(r"\bAQ\.[A-Za-z0-9_-]{20,}\b", "[REDACTED]", clean)
        return clean

    def get_coding_settings(self):
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
        profiles = []
        saved = settings.get("providers", {}) if isinstance(settings.get("providers"), dict) else {}
        for provider, meta in self.CODING_PROVIDERS.items():
            profile = saved.get(provider, {}) if isinstance(saved.get(provider), dict) else {}
            _, key_source = self._coding_load_key(provider)
            profile_url = str(profile.get("base_url", meta["base_url"]))
            budget = self.get_coding_budget(provider)
            profiles.append({
                "id": provider,
                "label": meta["label"],
                "model": str(profile.get("model", "")),
                "models": [str(item) for item in profile.get("models", []) if str(item).strip()],
                "base_url": profile_url,
                "configured": key_source != "missing" or (provider == "custom" and self._coding_is_local_url(profile_url)),
                "key_source": key_source,
                "updated_at": profile.get("updated_at", 0),
                "budget": budget,
            })
        return {"default_provider": settings.get("default_provider", ""), "providers": profiles}

    def _coding_password_record(self):
        settings = self._coding_read_json(self.coding_settings_path, {"key_view_auth": {}})
        record = settings.get("key_view_auth", {})
        return record if isinstance(record, dict) else {}

    def _coding_password_matches(self, password, record=None):
        record = record or self._coding_password_record()
        try:
            salt = base64.b64decode(str(record.get("salt", "")), validate=True)
            expected = base64.b64decode(str(record.get("hash", "")), validate=True)
            iterations = int(record.get("iterations", 310000))
        except Exception:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", str(password or "").encode("utf-8"), salt, iterations)
        return hmac.compare_digest(candidate, expected)

    def set_key_view_password(self, new_password, current_password=""):
        new_password = str(new_password or "")
        if len(new_password) < 8:
            return {"ok": False, "error": "Use at least 8 characters for the key-view password."}
        record = self._coding_password_record()
        if record.get("hash") and not self._coding_password_matches(current_password, record):
            return {"ok": False, "error": "Current password is incorrect."}
        salt = secrets.token_bytes(16)
        iterations = 310000
        derived = hashlib.pbkdf2_hmac("sha256", new_password.encode("utf-8"), salt, iterations)
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
            settings["key_view_auth"] = {
                "salt": base64.b64encode(salt).decode("ascii"),
                "hash": base64.b64encode(derived).decode("ascii"),
                "iterations": iterations,
                "updated_at": time.time(),
            }
            settings.pop("council", None)
            self._coding_write_json(self.coding_settings_path, settings)
        self.coding_key_view_unlocked_until = time.time() + 300
        return {"ok": True, "unlocked": True, "expires_in": 300}

    def unlock_model_library(self, password):
        record = self._coding_password_record()
        if not record.get("hash"):
            return {"ok": False, "needs_password_setup": True, "error": "Create a key-view password first."}
        if not self._coding_password_matches(password, record):
            return {"ok": False, "error": "Password is incorrect."}
        self.coding_key_view_unlocked_until = time.time() + 300
        return {"ok": True, "unlocked": True, "expires_in": 300}

    def lock_model_library(self):
        self.coding_key_view_unlocked_until = 0.0
        return {"ok": True, "unlocked": False}

    def _model_library_is_unlocked(self):
        return time.time() < float(getattr(self, "coding_key_view_unlocked_until", 0) or 0)

    def get_model_library(self):
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"providers": {}})
            usage = self._coding_read_json(self.coding_usage_path, {"events": []})
        used_by_provider = {}
        for event in usage.get("events", []) if isinstance(usage.get("events"), list) else []:
            if not isinstance(event, dict):
                continue
            provider = str(event.get("provider", "")).strip()
            model = str(event.get("model", "")).strip()
            if provider and model:
                used_by_provider.setdefault(provider, set()).add(model)
        profiles = settings.get("providers", {}) if isinstance(settings.get("providers"), dict) else {}
        rows = []
        for provider, profile in profiles.items():
            if provider not in self.CODING_PROVIDERS or not isinstance(profile, dict):
                continue
            key, key_source = self._coding_load_key(provider)
            base_url = str(profile.get("base_url") or self.CODING_PROVIDERS[provider]["base_url"])
            configured = bool(key) or (provider == "custom" and self._coding_is_local_url(base_url))
            if not configured:
                continue
            current_model = str(profile.get("model", "")).strip()
            models = set(used_by_provider.get(provider, set()))
            if current_model:
                models.add(current_model)
            if not models:
                continue
            rows.append({
                "provider": provider,
                "label": self.CODING_PROVIDERS[provider]["label"],
                "models": sorted(models, key=str.lower),
                "key_source": key_source,
                "has_saved_key": bool(key),
                "key_preview": ("•••• " + key[-4:]) if key else "Local connection",
                "budget": self.get_coding_budget(provider),
            })
        unlocked = self._model_library_is_unlocked()
        return {
            "ok": True,
            "password_set": bool(self._coding_password_record().get("hash")),
            "unlocked": unlocked,
            "expires_in": max(0, int(self.coding_key_view_unlocked_until - time.time())) if unlocked else 0,
            "models": rows,
        }

    def reveal_model_library_key(self, provider):
        if not self._model_library_is_unlocked():
            return {"ok": False, "locked": True, "error": "Unlock the Model Library first."}
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported provider."}
        key, source = self._coding_load_key(provider)
        if not key:
            return {"ok": False, "error": "This connection does not have a saved API key."}
        return {"ok": True, "provider": provider, "api_key": key, "key_source": source}


    def list_coding_models(self, provider, api_key="", base_url=""):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider.", "models": []}
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
        saved = settings.get("providers", {}).get(provider, {})
        base = str(base_url or saved.get("base_url") or self.CODING_PROVIDERS[provider]["base_url"]).strip().rstrip("/")
        key = str(api_key or "").strip()
        if not key:
            key, _ = self._coding_load_key(provider)
        if not key and not (provider == "custom" and self._coding_is_local_url(base)):
            return {"ok": False, "error": "Add this provider's API key before loading models.", "models": []}

        headers = {"Accept": "application/json", "User-Agent": "KIRA-Superapp/1.0"}
        if provider == "gemini":
            url = base + "/v1beta/models?pageSize=1000"
            headers["x-goog-api-key"] = key
        elif provider == "anthropic":
            url = base + ("/models?limit=1000" if base.endswith("/v1") else "/v1/models?limit=1000")
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        else:
            url = base if base.endswith("/models") else base + "/models"
            if key:
                headers["Authorization"] = "Bearer " + key

        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            return {"ok": False, "error": f"Model list returned HTTP {exc.code}: {self._coding_redact(detail, [key])}", "models": []}
        except Exception as exc:
            return {"ok": False, "error": f"Could not load models: {self._coding_redact(exc, [key])}", "models": []}

        rows = payload.get("models", []) if provider == "gemini" else payload.get("data", [])
        models = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            if provider == "gemini":
                methods = row.get("supportedGenerationMethods", [])
                if methods and "generateContent" not in methods:
                    continue
                model_id = str(row.get("name", "")).removeprefix("models/")
                label = str(row.get("displayName") or model_id)
            else:
                model_id = str(row.get("id") or row.get("name") or "")
                label = str(row.get("display_name") or row.get("displayName") or model_id)
            if model_id and not any(item["id"] == model_id for item in models):
                models.append({"id": model_id, "label": label})
        models.sort(key=lambda item: (item["label"].lower(), item["id"].lower()))
        if not models:
            return {"ok": False, "error": "The provider returned no compatible text-generation models.", "models": []}

        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
            profile = settings.setdefault("providers", {}).setdefault(provider, {})
            profile["models"] = [item["id"] for item in models]
            self._coding_write_json(self.coding_settings_path, settings)
        return {"ok": True, "provider": provider, "models": models}

    def save_coding_provider(self, provider, model, api_key="", base_url=""):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider."}
        model = str(model or "").strip()
        if not model:
            return {"ok": False, "error": "Enter the exact model ID supplied by the provider."}
        url = str(base_url or self.CODING_PROVIDERS[provider]["base_url"]).strip().rstrip("/")
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return {"ok": False, "error": "Base URL must begin with http:// or https://."}
        try:
            if str(api_key or "").strip():
                self._coding_store_key(provider, api_key)
            key, source = self._coding_load_key(provider)
            if not key and not (provider == "custom" and self._coding_is_local_url(url)):
                return {"ok": False, "error": "Add an API key or set the provider's environment variable."}
            with self.coding_lock:
                settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
                profile = settings.setdefault("providers", {}).setdefault(provider, {})
                profile.update({
                    "model": model,
                    "base_url": url,
                    "updated_at": time.time(),
                })
                settings["default_provider"] = provider
                self._coding_write_json(self.coding_settings_path, settings)
            return {"ok": True, "provider": provider, "model": model, "key_source": source}
        except Exception as exc:
            return {"ok": False, "error": self._coding_redact(exc, [api_key])}

    def delete_coding_provider(self, provider):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            return {"ok": False, "error": "Unsupported coding provider."}
        self._coding_delete_key(provider)
        with self.coding_lock:
            settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
            settings.setdefault("providers", {}).pop(provider, None)
            if settings.get("default_provider") == provider:
                settings["default_provider"] = ""
            self._coding_write_json(self.coding_settings_path, settings)
        return {"ok": True}

    def test_coding_provider(self, provider, model="", base_url=""):
        try:
            profile = self._coding_provider_profile(provider, model, base_url)
            answer = self._coding_call_provider(
                profile,
                "You are a connectivity test. Reply with exactly KIRA_READY.",
                "Reply exactly KIRA_READY.",
                max_tokens=24,
                timeout=45,
            )
            return {"ok": "KIRA_READY" in answer.upper(), "response": self._coding_redact(answer)[:300]}
        except Exception as exc:
            return {"ok": False, "error": self._coding_redact(exc)}

    def _coding_is_local_url(self, url):
        try:
            return (urllib.parse.urlparse(url).hostname or "") in {"localhost", "127.0.0.1", "::1"}
        except Exception:
            return False

    def _coding_provider_profile(self, provider, model="", base_url=""):
        provider = str(provider or "").strip().lower()
        if provider not in self.CODING_PROVIDERS:
            raise ValueError("Choose and configure a coding provider first.")
        settings = self._coding_read_json(self.coding_settings_path, {"default_provider": "", "providers": {}})
        saved = settings.get("providers", {}).get(provider, {})
        exact_model = str(model or saved.get("model", "")).strip()
        if not exact_model:
            raise ValueError("The coding provider needs an exact model ID.")
        url = str(base_url or saved.get("base_url") or self.CODING_PROVIDERS[provider]["base_url"]).strip().rstrip("/")
        key, source = self._coding_load_key(provider)
        if not key and not (provider == "custom" and self._coding_is_local_url(url)):
            raise ValueError("The coding provider has no API key in macOS Keychain or its environment variable.")
        return {"provider": provider, "model": exact_model, "base_url": url, "api_key": key, "key_source": source}

    def discover_coding_ides(self):
        candidates = [
            ("Cursor", "Cursor.app"), ("Visual Studio Code", "Visual Studio Code.app"),
            ("Windsurf", "Windsurf.app"), ("Zed", "Zed.app"), ("Xcode", "Xcode.app"),
            ("PyCharm", "PyCharm.app"), ("IntelliJ IDEA", "IntelliJ IDEA.app"),
            ("Sublime Text", "Sublime Text.app"), ("Nova", "Nova.app"),
            ("Android Studio", "Android Studio.app"), ("BBEdit", "BBEdit.app"),
        ]
        roots = getattr(self, "coding_application_roots", ["/Applications", os.path.expanduser("~/Applications")])
        found = []
        seen = set()
        for label, bundle in candidates:
            for root in roots:
                path = os.path.join(root, bundle)
                if os.path.isdir(path) and path not in seen:
                    found.append({"id": label, "label": label, "app_name": label, "path": path})
                    seen.add(path)
                    break
        return found

    def choose_coding_project(self, chat_id=None):
        try:
            import webview
            window = webview.windows[0] if getattr(webview, "windows", None) else None
            if window is None:
                return {"ok": False, "error": "KIRA window is not ready."}
            selected = window.create_file_dialog(webview.FileDialog.FOLDER, allow_multiple=False)
            path = selected[0] if selected else ""
            if not path:
                return {"ok": False, "cancelled": True}
            path = os.path.realpath(os.path.expanduser(path))
            if not os.path.isdir(path) or self._coding_project_is_sensitive(path):
                return {"ok": False, "error": "That folder cannot be used as a coding workspace."}
            session = self.configure_coding_session(chat_id, path, "", "", "")
            return {"ok": True, "path": path, "session": session}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _coding_session_file(self, chat_id):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(chat_id or "default"))
        return os.path.join(self.coding_sessions_path, safe + ".json")

    def get_coding_session(self, chat_id=None):
        session = self._coding_read_json(self._coding_session_file(chat_id), {})
        settings = self.get_coding_settings()
        if not session.get("provider"):
            session["provider"] = settings.get("default_provider", "")
        if session.get("provider") and not session.get("model"):
            match = next((p for p in settings["providers"] if p["id"] == session["provider"]), {})
            session["model"] = match.get("model", "")
        session.setdefault("project_path", "")
        session.setdefault("ide", "")
        session.setdefault("model", "")
        session.setdefault("harness", "kira")
        session["ides"] = self.discover_coding_ides()
        session["harnesses"] = self.discover_coding_harnesses()
        return session

    def configure_coding_session(self, chat_id, project_path="", ide="", provider="", model="", harness=""):
        existing = self._coding_read_json(self._coding_session_file(chat_id), {})
        if project_path:
            project = os.path.realpath(os.path.expanduser(str(project_path)))
            if not os.path.isdir(project):
                return {"ok": False, "error": "Project folder does not exist."}
            if self._coding_project_is_sensitive(project):
                return {"ok": False, "error": "Sensitive system and credential folders cannot be coding projects."}
            existing["project_path"] = project
        if ide:
            installed = {item["id"] for item in self.discover_coding_ides()}
            if ide not in installed:
                return {"ok": False, "error": "The selected IDE is not installed."}
            existing["ide"] = str(ide)
        if provider:
            if provider not in self.CODING_PROVIDERS:
                return {"ok": False, "error": "Unsupported provider."}
            existing["provider"] = provider
        if model:
            existing["model"] = str(model).strip()
        if harness:
            available = {item["id"] for item in self.discover_coding_harnesses() if item.get("available")}
            if harness not in available:
                return {"ok": False, "error": "The selected coding harness runtime is unavailable."}
            existing["harness"] = harness
        existing["chat_id"] = str(chat_id or "default")
        existing["updated_at"] = time.time()
        self._coding_write_json(self._coding_session_file(chat_id), existing)
        result = self.get_coding_session(chat_id)
        result["ok"] = True
        return result

    def discover_coding_harnesses(self):
        codex_candidates = [
            shutil.which("codex"),
            "/Applications/ChatGPT.app/Contents/Resources/codex",
        ]
        codex_path = next((path for path in codex_candidates if path and os.path.isfile(path)), "")
        dsh_candidates = [
            shutil.which("dsh"),
            os.path.join(getattr(self, "app_root", os.getcwd()), "external_harnesses", "deepseek-runtime", "node_modules", ".bin", "dsh"),
        ]
        dsh_path = next((path for path in dsh_candidates if path and os.path.isfile(path)), "")
        return [
            {"id": "kira", "label": "KIRA Native", "available": True, "path": "", "license": "KIRA"},
            {"id": "codex", "label": "OpenAI Codex", "available": bool(codex_path), "path": codex_path, "license": "Apache-2.0"},
            {"id": "deepseek", "label": "DeepSeek Harness", "available": bool(dsh_path), "path": dsh_path, "license": "MIT", "preview": True},
        ]

    def open_coding_project(self, chat_id=None):
        session = self.get_coding_session(chat_id)
        project = session.get("project_path", "")
        ide = session.get("ide", "")
        if not project or not os.path.isdir(project):
            return {"ok": False, "error": "Choose a project folder first."}
        if not ide:
            return {"ok": False, "error": "Choose an installed IDE first."}
        result = subprocess.run(["open", "-a", ide, project], capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "IDE failed to open.").strip()}
        return {"ok": True, "message": f"Opened {os.path.basename(project)} in {ide}."}

    def _coding_project_is_sensitive(self, path):
        real = os.path.realpath(path)
        home = os.path.realpath(os.path.expanduser("~"))
        blocked_roots = {
            "/", "/System", "/Library", "/private", "/usr", "/bin", "/sbin", "/etc", "/var",
            home,
            os.path.join(home, ".ssh"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".aws"),
            os.path.join(home, "Library", "Keychains"),
            os.path.join(home, "Library", "Mail"),
            os.path.join(home, "Library", "Messages"),
        }
        if real in blocked_roots:
            return True
        return any(real.startswith(root + os.sep) for root in blocked_roots if root not in {"/", home})

    def _coding_progress(self, chat_id, stage, percent, label):
        self.response_queue.put({
            "type": "progress", "chat_id": chat_id, "stage": stage,
            "percent": int(percent), "label": label,
        })
        self.response_queue.put({"type": "status", "chat_id": chat_id, "content": label})

    def _coding_run_event(self, run_id, phase, detail=None):
        if not run_id:
            return
        event = {
            "run_id": str(run_id),
            "phase": str(phase),
            "timestamp": time.time(),
            "detail": detail if isinstance(detail, dict) else {"message": str(detail or "")},
        }
        path = os.path.join(self.coding_runs_path, re.sub(r"[^A-Za-z0-9_-]", "", str(run_id)) + ".jsonl")
        with self.coding_lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _handle_vibe_coding_prompt(self, raw_prompt, chat_id, attachments=None):
        run_id = "code_" + uuid.uuid4().hex[:16]
        self._coding_run_event(run_id, "started", {"chat_id": chat_id})
        session = self.get_coding_session(chat_id)
        project = session.get("project_path", "")
        provider = session.get("provider", "")
        if not project or not os.path.isdir(project):
            return "Vibe Coding needs a project folder. Use the Project control above the chat, then send this task again."
        if not session.get("ide"):
            return "Vibe Coding needs an installed IDE. Choose one above the chat, then send this task again."
        try:
            profile = self._coding_provider_profile(provider, session.get("model", ""))
        except Exception as exc:
            return f"Vibe Coding provider setup is incomplete: {exc} Open Coding Settings, save a model ID and API key, then retry."

        self._coding_progress(chat_id, "Inspecting", 10, "Reading the selected project safely...")
        context = self._coding_project_context(project, attachments or [], task=raw_prompt)
        self._coding_run_event(run_id, "inspected", {"context_chars": len(context)})
        self._coding_progress(chat_id, "Planning", 28, "Orchestrator V1 is preparing the coding brief...")
        brief = self._coding_orchestrator_brief(raw_prompt, session, context)
        self._coding_progress(chat_id, "Delegating", 48, f"Sending Orchestrator's brief to {self.CODING_PROVIDERS[provider]['label']}...")
        worker_system = self._coding_worker_system(project)
        worker_prompt = brief + "\n\nPROJECT SNAPSHOT:\n" + context
        try:
            validated = self._coding_prepare_plan(
                profile, worker_system, worker_prompt, project, run_id, session.get("harness", "kira")
            )
        except Exception as exc:
            self._coding_run_event(run_id, "failed", {"error": self._coding_redact(exc, [profile.get("api_key")])})
            return "Coding worker failed safely: " + self._coding_redact(exc, [profile.get("api_key")])

        answer = str(validated.get("answer", "")).strip()
        edits = validated.get("files", [])
        commands = validated.get("commands", [])
        if not edits and not commands:
            self._coding_progress(chat_id, "Complete", 100, "Coding analysis complete.")
            return answer or "The coding worker returned no valid edits or project commands. Nothing changed."

        preview = self._coding_plan_preview(project, validated)
        payload = {
            "project_path": project,
            "ide": session.get("ide", ""),
            "provider": provider,
            "model": profile["model"],
            "harness": session.get("harness", "kira"),
            "summary": validated.get("summary", "Coding change"),
            "files": edits,
            "commands": commands,
            "chat_id": chat_id,
            "task": str(raw_prompt),
            "run_id": run_id,
        }
        previous_context = getattr(self, "_active_permission_context", None)
        self._active_permission_context = {"raw_prompt": raw_prompt, "mode": "vibe_coding"}
        try:
            permission_message = self._queue_permission("coding_apply", payload, preview, chat_id=chat_id)
        finally:
            self._active_permission_context = previous_context
        permission_match = re.search(r"`(perm_[^`]+)`", str(permission_message))
        permission_id = permission_match.group(1) if permission_match else "pending coding change"
        self._coding_progress(chat_id, "Approval", 70, "Validated coding plan is waiting for approval.")
        self._coding_run_event(run_id, "awaiting_approval", {
            "files": len(edits), "commands": len(commands), "permission_id": permission_id,
        })
        return (
            f"Orchestrator prepared {len(edits)} file edit(s) and {len(commands)} project command(s). "
            f"Review permission `{permission_id}` to apply them. Nothing has changed yet."
            + (f"\n\n{answer}" if answer else "")
        )

    def _coding_project_context(self, project, attachments=None, task=""):
        rows = []
        total = 0
        max_chars = 90000
        max_files = 90
        task_terms = {
            token for token in re.findall(r"[A-Za-z0-9_./-]{3,}", str(task).lower())
            if token not in {"the", "and", "for", "with", "this", "that", "from", "into", "build", "create"}
        }
        candidates = []
        for root, dirs, files in os.walk(project):
            dirs[:] = sorted(d for d in dirs if d not in self.CODING_IGNORED_DIRS and not d.startswith("."))
            for name in sorted(files):
                relative = os.path.relpath(os.path.join(root, name), project)
                if self._coding_secret_path(relative) or not self._coding_text_file(name):
                    continue
                path = os.path.join(project, relative)
                try:
                    if os.path.getsize(path) > 180000:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        content = handle.read(18000)
                except Exception:
                    continue
                lower_path = relative.lower()
                lower_content = content[:4000].lower()
                score = max(0, 12 - relative.count(os.sep))
                if name.lower() in {"agents.md", "readme.md", "pyproject.toml", "package.json", "cargo.toml", "go.mod"}:
                    score += 80
                score += sum(12 for term in task_terms if term in lower_path)
                score += sum(2 for term in task_terms if term in lower_content)
                candidates.append((score, relative, content))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        repo_map = "\n".join(f"- {relative}" for _, relative, _ in candidates[:250])
        if repo_map:
            rows.append("--- REPOSITORY MAP ---\n" + repo_map[:16000])
            total += len(rows[-1])
        for _, relative, content in candidates[:max_files]:
            block = f"\n--- FILE: {relative} ---\n{content}"
            if total + len(block) > max_chars:
                break
            rows.append(block)
            total += len(block)
        git_status = self._coding_git_status(project)
        if git_status:
            rows.append("\n--- GIT STATUS (do not overwrite unrelated work) ---\n" + git_status)
        if attachments:
            rows.append("\n--- USER ATTACHMENTS ---\n" + json.dumps(attachments, ensure_ascii=False)[:8000])
        return "".join(rows) or "(Project contains no readable source files.)"

    def _coding_git_status(self, project):
        if not os.path.isdir(os.path.join(project, ".git")):
            return ""
        try:
            result = subprocess.run(
                ["git", "status", "--short"], cwd=project, capture_output=True,
                text=True, timeout=8,
            )
            return str(result.stdout or "").strip()[:12000]
        except Exception:
            return ""

    def _coding_text_file(self, name):
        lower = str(name).lower()
        return lower in self.CODING_SAFE_FILENAMES or os.path.splitext(lower)[1] in self.CODING_TEXT_SUFFIXES

    def _coding_secret_path(self, relative):
        lower = str(relative).replace("\\", "/").lower()
        parts = lower.split("/")
        basename = parts[-1]
        return (
            basename.startswith(".env") or basename in {"credentials", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
            or any(part in {".ssh", ".aws", ".gnupg", "keychains"} for part in parts)
            or any(token in basename for token in ["private_key", "api_key", "access_token"])
        )

    def _coding_orchestrator_brief(self, task, session, project_context):
        budget = self.get_coding_budget(session.get("provider", ""))
        budget_note = ""
        if budget.get("ok"):
            remaining = budget.get("remaining_usd")
            remaining_label = "unlimited" if remaining is None else f"${remaining:.4f}"
            budget_note = (
                "\n\nAPI BUDGET STATE: "
                f"tracked total ${budget.get('total_spend_usd', 0):.4f}; remaining {remaining_label}; "
                f"per-task cap ${budget.get('task_cap_usd', 0):.4f}; policy {budget.get('policy')}. "
                "Keep the worker brief focused and avoid unnecessary generations."
            )
        fallback = (
            "Implement the user's task in the supplied project. Preserve existing architecture and style. "
            "Return only the required structured JSON plan. USER TASK: " + str(task) + budget_note
        )
        try:
            self._evict_and_load("orchestrator")
            message = (
                "You are Orchestrator V1 managing a coding worker. Privately reason about the request and project snapshot, "
                "then write a concise implementation brief for another coding model. Include goals, constraints, likely files, "
                "verification, and explicit non-goals. Do not emit tools or JSON.\n\nUSER TASK:\n"
                + str(task) + "\n\nPROJECT EXCERPT:\n" + project_context[:26000]
                + budget_note
            )
            prompt = self._tokenizer_prompt([{"role": "user", "content": message}]) + "<thought_process>\n"
            response = self._generate_with_watchdog(prompt, temperature=0.35, max_tokens=800, timeout_seconds=45, label="coding_brief")
            private, public = self._split_orchestrator_response(response)
            if hasattr(self, "_log_private_thoughts"):
                self._log_private_thoughts(private)
            clean = self._strip_agentic_blocks(self._strip_private_reasoning(public or response)).strip()
            return clean or fallback
        except Exception:
            return fallback

    def _coding_worker_system(self, project):
        return (
            "You are a coding implementation worker inside KIRA's bounded inspect-plan-review-execute-verify loop. "
            "Do not claim files were changed or tests passed. Obey repository instructions such as AGENTS.md. "
            "Analyze the supplied snapshot and return exactly one JSON object, without markdown, using this schema: "
            '{"summary":"...","answer":"optional user-facing note","files":[{"path":"relative/path","content":"complete new file content","reason":"..."}],'
            '"commands":[{"command":"test or build command","reason":"..."}],"notes":["..."]}. '
            "Use project-relative paths only. Include complete file contents, not patches. Never request credentials, inspect secrets, "
            "modify files outside the project, or propose destructive commands. Commands must be bounded verification tests/builds only; "
            "never install packages, start persistent servers, run formatters, or mutate project files from commands. "
            "Keep edits tightly scoped, preserve unrelated user changes, and include the smallest useful verification set. "
            f"The project root is {os.path.basename(project)}."
        )

    def _coding_harness_path(self, harness):
        return next(
            (item.get("path", "") for item in self.discover_coding_harnesses() if item.get("id") == harness and item.get("available")),
            "",
        )

    def _coding_harness_env(self, profile):
        env = dict(os.environ)
        for meta in self.CODING_PROVIDERS.values():
            env.pop(meta.get("env", ""), None)
        env_name = self.CODING_PROVIDERS.get(profile["provider"], {}).get("env")
        if env_name and profile.get("api_key"):
            env[env_name] = profile["api_key"]
        env["KIRA_HARNESS"] = "1"
        return env

    def _coding_external_plan(self, harness, profile, worker_system, worker_prompt, project):
        binary = self._coding_harness_path(harness)
        if not binary:
            raise RuntimeError(f"{harness} harness runtime is not installed.")
        self._coding_budget_guard(profile, worker_system, worker_prompt, 10000)
        env = self._coding_harness_env(profile)
        combined_prompt = worker_system + "\n\n" + worker_prompt
        if harness == "codex":
            if profile["provider"] != "openai":
                raise RuntimeError("OpenAI Codex harness requires an OpenAI provider connection.")
            schema = os.path.join(getattr(self, "app_root", os.getcwd()), "external_harnesses", "kira_plan.schema.json")
            with tempfile.TemporaryDirectory(prefix="kira-codex-") as temporary:
                output_path = os.path.join(temporary, "plan.json")
                command = [
                    binary, "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                    "--ignore-user-config", "--model", profile["model"], "--output-schema", schema,
                    "--output-last-message", output_path, "-",
                ]
                result = subprocess.run(
                    command, cwd=project, env=env, input=combined_prompt, capture_output=True,
                    text=True, timeout=210,
                )
                if result.returncode != 0:
                    raise RuntimeError("Codex harness failed: " + self._coding_redact(result.stderr or result.stdout, [profile.get("api_key")])[:1200])
                with open(output_path, "r", encoding="utf-8") as handle:
                    output = handle.read()
        elif harness == "deepseek":
            if profile["provider"] != "deepseek":
                raise RuntimeError("DeepSeek Harness requires a DeepSeek provider connection.")
            if os.name != "posix" or not shutil.which("sandbox-exec"):
                raise RuntimeError("DeepSeek Harness planning requires the macOS read-only sandbox.")
            with tempfile.TemporaryDirectory(prefix="kira-dsh-") as dsh_home:
                env["DSH_HOME"] = dsh_home
                sandbox = (
                    '(version 1)(deny default)(allow process*)(allow file-read*)'
                    '(allow network-outbound)(allow file-write* (subpath "/private/tmp") (subpath "/tmp"))'
                )
                command = ["sandbox-exec", "-p", sandbox, binary, "--profile", "headless", combined_prompt]
                result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=210)
                if result.returncode != 0:
                    raise RuntimeError("DeepSeek harness failed: " + self._coding_redact(result.stderr or result.stdout, [profile.get("api_key")])[:1200])
                output = result.stdout
        else:
            raise RuntimeError("Unsupported external coding harness.")
        self._coding_record_usage(
            profile,
            self._coding_estimate_tokens(combined_prompt),
            self._coding_estimate_tokens(output),
            estimated=True,
        )
        return output

    def _coding_prepare_plan(self, profile, worker_system, worker_prompt, project, run_id, harness="kira"):
        prompt = worker_prompt
        attempts = max(0, int(self.CODING_MAX_PLAN_REPAIRS)) + 1
        last_error = None
        for attempt in range(attempts):
            raw_plan = (
                self._coding_external_plan(harness, profile, worker_system, prompt, project)
                if harness in {"codex", "deepseek"}
                else self._coding_call_provider(profile, worker_system, prompt, max_tokens=10000, timeout=150)
            )
            try:
                plan = self._coding_parse_plan(raw_plan)
                validated = self._coding_validate_plan(project, plan)
                validated["commands"] = validated.get("commands") or self._coding_infer_checks(
                    project, validated.get("files", [])
                )
                self._coding_run_event(run_id, "planned", {
                    "attempt": attempt + 1,
                    "files": len(validated.get("files", [])),
                    "commands": len(validated.get("commands", [])),
                    "harness": harness,
                })
                return validated
            except Exception as exc:
                last_error = exc
                self._coding_run_event(run_id, "plan_rejected", {
                    "attempt": attempt + 1, "finding": str(exc)[:1000],
                })
                if attempt + 1 >= attempts:
                    break
                prompt = (
                    worker_prompt
                    + "\n\nThe previous plan was rejected by KIRA's deterministic policy: "
                    + str(exc)[:1200]
                    + "\nReturn a corrected complete JSON plan. Do not explain the correction."
                )
        raise ValueError(f"Coding plan remained invalid after bounded repair: {last_error}")

    def _coding_infer_checks(self, project, files):
        paths = [str(item.get("path", "")) for item in files if isinstance(item, dict)]
        checks = []
        python_paths = [path for path in paths if path.endswith((".py", ".pyi"))]
        javascript_paths = [path for path in paths if path.endswith((".js", ".mjs", ".cjs"))]
        if python_paths:
            args = " ".join(shlex.quote(path) for path in python_paths[:12])
            checks.append({"command": f"python3 -m compileall -q {args}", "reason": "KIRA inferred Python syntax verification"})
        for path in javascript_paths[:3]:
            checks.append({"command": "node --check " + shlex.quote(path), "reason": "KIRA inferred JavaScript syntax verification"})
        return checks[:4]

    def _coding_estimate_tokens(self, text):
        return max(1, (len(str(text or "")) + 3) // 4)

    def _coding_cost(self, budget, input_tokens, output_tokens):
        return round(
            (max(0, int(input_tokens or 0)) / 1_000_000.0) * budget["input_usd_per_million"]
            + (max(0, int(output_tokens or 0)) / 1_000_000.0) * budget["output_usd_per_million"],
            8,
        )

    def _coding_budget_guard(self, profile, system_prompt, user_prompt, max_tokens):
        budget = self.get_coding_budget(profile["provider"])
        if not budget.get("ok"):
            raise RuntimeError(budget.get("error", "Could not read the API budget."))
        estimated_input = self._coding_estimate_tokens(system_prompt) + self._coding_estimate_tokens(user_prompt)
        projected = self._coding_cost(budget, estimated_input, max_tokens)
        if budget.get("blocked"):
            raise RuntimeError(
                f"API budget stopped this request. {profile['provider']} has reached its "
                f"${budget['monthly_cap_usd']:.4f} monthly cap."
            )
        task_cap = float(budget.get("task_cap_usd", 0) or 0)
        remaining = budget.get("remaining_usd")
        if budget.get("policy") == "hard_stop":
            if task_cap and projected > task_cap:
                raise RuntimeError(
                    f"API budget stopped this request before billing. Projected maximum ${projected:.4f} "
                    f"exceeds the ${task_cap:.4f} per-task cap."
                )
            if remaining is not None and projected > remaining:
                raise RuntimeError(
                    f"API budget stopped this request before billing. Projected maximum ${projected:.4f} "
                    f"exceeds the ${remaining:.4f} remaining monthly budget."
                )
        return {
            "estimated_input_tokens": estimated_input,
            "projected_max_cost_usd": projected,
            "policy": budget.get("policy"),
        }

    def _coding_usage_from_payload(self, provider, payload, system_prompt, user_prompt, response_text):
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        estimated = False
        if provider == "anthropic":
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
        elif provider == "gemini":
            usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
            input_tokens = usage.get("promptTokenCount")
            output_tokens = usage.get("candidatesTokenCount")
        else:
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None:
            input_tokens = self._coding_estimate_tokens(system_prompt) + self._coding_estimate_tokens(user_prompt)
            estimated = True
        if output_tokens is None:
            output_tokens = self._coding_estimate_tokens(response_text)
            estimated = True
        return max(0, int(input_tokens)), max(0, int(output_tokens)), estimated

    def _coding_record_usage(self, profile, input_tokens, output_tokens, estimated=False):
        budget = self._coding_budget_config(profile["provider"])
        cost = self._coding_cost(budget, input_tokens, output_tokens)
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "month": self._coding_month_key(),
            "provider": profile["provider"],
            "model": profile["model"],
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": cost,
            "estimated_tokens": bool(estimated),
        }
        with self.coding_lock:
            ledger = self._coding_read_json(self.coding_usage_path, {"events": []})
            events = ledger.get("events", []) if isinstance(ledger.get("events"), list) else []
            events.append(event)
            ledger["events"] = events[-2000:]
            self._coding_write_json(self.coding_usage_path, ledger)
        return event

    def _coding_call_provider(self, profile, system_prompt, user_prompt, max_tokens=8000, timeout=120):
        provider = profile["provider"]
        key = profile.get("api_key", "")
        base = profile["base_url"].rstrip("/")
        model = profile["model"]
        self._coding_budget_guard(profile, system_prompt, user_prompt, max_tokens)
        headers = {"Content-Type": "application/json", "User-Agent": "KIRA-OS-Vibe-Coding/1.0"}
        if provider == "anthropic":
            url = base + "/v1/messages" if not base.endswith("/v1") else base + "/messages"
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            body = {"model": model, "max_tokens": max_tokens, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}
        elif provider == "gemini":
            encoded = urllib.parse.quote(model, safe="-_.")
            url = f"{base}/v1beta/models/{encoded}:generateContent"
            headers["x-goog-api-key"] = key
            body = {"systemInstruction": {"parts": [{"text": system_prompt}]}, "contents": [{"role": "user", "parts": [{"text": user_prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}}
        else:
            if base.endswith("/chat/completions"):
                url = base
            else:
                url = base + "/chat/completions"
            if key:
                headers["Authorization"] = "Bearer " + key
            if provider == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/saggamer/KIRA-OS"
                headers["X-Title"] = "KIRA OS"
            body = {"model": model, "temperature": 0.2, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            raise RuntimeError(f"Provider returned HTTP {exc.code}: {self._coding_redact(detail, [key])}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Provider connection failed: {self._coding_redact(exc, [key])}") from exc
        if provider == "anthropic":
            content = "".join(part.get("text", "") for part in payload.get("content", []) if isinstance(part, dict))
        elif provider == "gemini":
            candidates = payload.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            content = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        else:
            choices = payload.get("choices", [])
            if not choices:
                raise RuntimeError("Provider returned no completion choices.")
            content = choices[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        content = str(content or "")
        input_tokens, output_tokens, estimated = self._coding_usage_from_payload(
            provider, payload, system_prompt, user_prompt, content
        )
        self._coding_record_usage(profile, input_tokens, output_tokens, estimated)
        return content

    def _coding_parse_plan(self, text):
        source = str(text or "").strip()
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```$", "", source)
        try:
            plan = json.loads(source)
        except Exception:
            start, end = source.find("{"), source.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Coding worker did not return a JSON plan.")
            plan = json.loads(source[start:end + 1])
        if not isinstance(plan, dict):
            raise ValueError("Coding worker plan must be a JSON object.")
        return plan

    def _coding_validate_plan(self, project, plan):
        files = plan.get("files", [])
        commands = plan.get("commands", [])
        if not isinstance(files, list) or not isinstance(commands, list):
            raise ValueError("Coding plan files and commands must be arrays.")
        if len(files) > 24 or len(commands) > 6:
            raise ValueError("Coding plan exceeds KIRA's bounded change limit.")
        clean_files, clean_commands, total = [], [], 0
        seen_paths, seen_commands = set(), set()
        project_real = os.path.realpath(project)
        for item in files:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("path", "")).strip().replace("\\", "/")
            if not relative or os.path.isabs(relative) or relative.startswith("../") or "/../" in relative or self._coding_secret_path(relative):
                raise ValueError(f"Rejected unsafe coding path: {relative or '(empty)'}")
            if relative in seen_paths:
                raise ValueError(f"Rejected duplicate coding path: {relative}")
            seen_paths.add(relative)
            target = os.path.realpath(os.path.join(project_real, relative))
            if target != project_real and not target.startswith(project_real + os.sep):
                raise ValueError(f"Rejected path outside project: {relative}")
            content = str(item.get("content", ""))
            total += len(content.encode("utf-8"))
            if total > 2_500_000:
                raise ValueError("Coding plan is larger than 2.5 MB.")
            clean_files.append({"path": relative, "content": content, "reason": str(item.get("reason", ""))[:500]})
        for item in commands:
            command = str(item.get("command", "") if isinstance(item, dict) else item).strip()
            if not command:
                continue
            if self._coding_dangerous_command(command):
                raise ValueError(f"Rejected destructive coding command: {command}")
            if self._coding_mutating_command(command):
                raise ValueError(f"Rejected non-verification coding command: {command}")
            if command in seen_commands:
                continue
            seen_commands.add(command)
            clean_commands.append({"command": command, "reason": str(item.get("reason", ""))[:500] if isinstance(item, dict) else ""})
        return {
            "summary": str(plan.get("summary", "Coding change"))[:1000],
            "answer": str(plan.get("answer", ""))[:8000],
            "notes": [str(note)[:500] for note in plan.get("notes", [])[:12]] if isinstance(plan.get("notes", []), list) else [],
            "files": clean_files,
            "commands": clean_commands,
        }

    def _coding_dangerous_command(self, command):
        lower = " " + re.sub(r"\s+", " ", str(command).lower()) + " "
        patterns = [
            r"(^|[;&|]\s*)sudo\b", r"\brm\s+(-[^\n]*[rf]|--recursive|--force)",
            r"\bgit\s+(reset\s+--hard|clean\s+-|checkout\s+--\s+)",
            r"\b(mkfs|diskutil\s+erase|shutdown|reboot|halt|launchctl|killall)\b",
            r"\bcurl\b[^\n|]*\|\s*(sh|bash|zsh)\b", r"\bwget\b[^\n|]*\|\s*(sh|bash|zsh)\b",
            r"\bchmod\s+-r\b", r"\bchown\s+-r\b", r"\bdd\s+if=",
        ]
        return any(re.search(pattern, lower, re.IGNORECASE) for pattern in patterns)

    def _coding_mutating_command(self, command):
        lower = " " + re.sub(r"\s+", " ", str(command).lower()) + " "
        patterns = [
            r"\b(npm|pnpm|yarn|bun)\s+(install|add|remove|update)\b",
            r"\b(pip|pip3|uv|poetry|gem|cargo)\s+(install|add|remove|update)\b",
            r"\b(sed\s+-i|perl\s+-pi|tee|truncate|touch|mkdir|mv|cp)\b",
            r"(^|[^<])>>?\s*[^&]",
            r"\b(prettier|black|ruff\s+format|gofmt|rustfmt)\b",
        ]
        return any(re.search(pattern, lower, re.IGNORECASE) for pattern in patterns)

    def _coding_execution_snapshot(self, execution_id):
        with self.coding_execution_lock:
            active = self.coding_active_executions.get(str(execution_id), {})
            if active:
                return {
                    key: value for key, value in active.items()
                    if key not in {"process", "cancel_event"}
                }
            return next(
                (dict(item) for item in reversed(self.coding_execution_history)
                 if item.get("execution_id") == str(execution_id)),
                {},
            )

    def get_coding_execution(self, execution_id):
        snapshot = self._coding_execution_snapshot(execution_id)
        return snapshot or {
            "ok": False,
            "execution_id": str(execution_id or ""),
            "status": "missing",
            "error": "Coding execution was not found.",
        }

    def cancel_coding_execution(self, execution_id):
        execution_id = str(execution_id or "")
        with self.coding_execution_lock:
            active = self.coding_active_executions.get(execution_id)
            if not active:
                finished = self._coding_execution_snapshot(execution_id)
                if finished:
                    return {"ok": False, **finished, "message": "Execution already finished."}
                return {
                    "ok": False, "execution_id": execution_id, "status": "missing",
                    "error": "Coding execution was not found.",
                }
            active["cancel_event"].set()
            active["status"] = "cancelling"
        return {
            "ok": True, "execution_id": execution_id, "status": "cancelling",
            "message": "Cancellation requested.",
        }

    def _coding_emit_execution(self, result):
        queue = getattr(self, "response_queue", None)
        if queue is None:
            return
        queue.put({
            "type": "coding_execution",
            "execution_id": result.get("execution_id", ""),
            "status": result.get("status", ""),
            "command": result.get("command", ""),
            "cwd": result.get("cwd", ""),
            "duration_ms": result.get("duration_ms", 0),
            "exit_code": result.get("exit_code"),
            "output_truncated": bool(result.get("output_truncated")),
        })

    def _coding_terminate_process_group(self, process):
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except Exception:
                pass

    def _coding_run_bounded_process(self, runner, cwd, timeout, execution_id, seatbelt_mode):
        limit = max(4096, int(self.CODING_COMMAND_OUTPUT_LIMIT_BYTES))
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        buffer_lock = threading.Lock()
        cancel_event = threading.Event()
        started = time.monotonic()

        process = subprocess.Popen(
            runner,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
        active = {
            "ok": True,
            "execution_id": execution_id,
            "status": "running",
            "command": runner[-1] if runner else "",
            "cwd": cwd,
            "started_at": time.time(),
            "seatbelt": seatbelt_mode,
            "process": process,
            "cancel_event": cancel_event,
        }
        with self.coding_execution_lock:
            self.coding_active_executions[execution_id] = active
        self._coding_emit_execution(active)

        def drain(name, stream):
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    with buffer_lock:
                        remaining = limit - len(buffers[name])
                        if remaining > 0:
                            buffers[name].extend(chunk[:remaining])
                        if len(chunk) > max(0, remaining):
                            truncated[name] = True
            except Exception:
                pass

        drains = [
            threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
        ]
        for thread in drains:
            thread.start()

        status = "running"
        deadline = started + max(1.0, float(timeout))
        while process.poll() is None:
            if cancel_event.wait(0.05):
                status = "cancelled"
                self._coding_terminate_process_group(process)
                break
            if time.monotonic() >= deadline:
                status = "timed_out"
                self._coding_terminate_process_group(process)
                break

        try:
            process.wait(timeout=self.CODING_COMMAND_DRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._coding_terminate_process_group(process)
            try:
                process.wait(timeout=1)
            except Exception:
                pass
        for thread in drains:
            thread.join(timeout=self.CODING_COMMAND_DRAIN_TIMEOUT_SECONDS)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except Exception:
                pass

        if status == "running":
            status = "completed" if process.returncode == 0 else "failed"
        exit_code = 124 if status == "timed_out" else 130 if status == "cancelled" else process.returncode
        secrets = [
            os.environ.get(meta.get("env", ""), "")
            for meta in self.CODING_PROVIDERS.values()
            if meta.get("env")
        ]
        result = {
            "ok": status == "completed",
            "execution_id": execution_id,
            "status": status,
            "command": active["command"],
            "cwd": cwd,
            "started_at": active["started_at"],
            "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
            "exit_code": exit_code,
            "stdout": self._coding_redact(bytes(buffers["stdout"]).decode("utf-8", errors="replace"), secrets),
            "stderr": self._coding_redact(bytes(buffers["stderr"]).decode("utf-8", errors="replace"), secrets),
            "output_truncated": bool(truncated["stdout"] or truncated["stderr"]),
            "seatbelt": seatbelt_mode,
        }
        history_item = dict(result)
        history_item["stdout"] = history_item["stdout"][:12000]
        history_item["stderr"] = history_item["stderr"][:12000]
        with self.coding_execution_lock:
            self.coding_active_executions.pop(execution_id, None)
            self.coding_execution_history.append(history_item)
            self.coding_execution_history = self.coding_execution_history[-30:]
        self._coding_emit_execution(result)
        return result

    def _coding_execute_project_command(self, payload):
        cwd = os.path.realpath(os.path.expanduser(str(payload.get("cwd", "") or "")))
        command = str(payload.get("command", "") or "").strip()
        if not command:
            return {"ok": False, "status": "blocked", "error": "Missing project command."}
        if not os.path.isdir(cwd) or self._coding_project_is_sensitive(cwd):
            return {"ok": False, "status": "blocked", "error": "Project command CWD is unavailable or sensitive."}
        if self._coding_dangerous_command(command):
            return {"ok": False, "status": "blocked", "error": "Destructive project command was rejected."}

        timeout = payload.get("timeout", self.CODING_COMMAND_TIMEOUT_SECONDS)
        try:
            timeout = min(600, max(1, int(timeout)))
        except (TypeError, ValueError):
            timeout = self.CODING_COMMAND_TIMEOUT_SECONDS
        execution_id = "exec_" + uuid.uuid4().hex[:16]
        raw_runner = ["/bin/zsh", "-lc", command]
        runner = raw_runner
        seatbelt_mode = "unavailable"
        if hasattr(self, "_seatbelt_available") and self._seatbelt_available():
            profile = self._seatbelt_profile(mode="project", write_paths=[cwd])
            runner = ["sandbox-exec", "-p", profile] + runner
            seatbelt_mode = "project"
        result = self._coding_run_bounded_process(runner, cwd, timeout, execution_id, seatbelt_mode)
        seatbelt_output = str(result.get("stderr", "")) + str(result.get("stdout", ""))
        if seatbelt_mode != "project" or result.get("exit_code") != 71 or "sandbox_apply" not in seatbelt_output:
            return result

        reason = seatbelt_output.strip()[:1000]
        logger = getattr(self, "_log_agentic_event", None)
        if callable(logger):
            logger("seatbelt_unavailable_fallback", {
                "mode": "project",
                "command": command,
                "execution_id": execution_id,
                "reason": reason,
            })

        # Some current macOS builds expose sandbox-exec but reject every
        # profile. Preserve the established KIRA fallback while keeping one
        # stable execution identity and explicit evidence of the downgrade.
        with self.coding_execution_lock:
            self.coding_execution_history = [
                item for item in self.coding_execution_history
                if item.get("execution_id") != execution_id
            ]
        result = self._coding_run_bounded_process(
            raw_runner, cwd, timeout, execution_id, "fallback_unavailable"
        )
        result["seatbelt_fallback"] = True
        result["seatbelt_fallback_reason"] = reason
        with self.coding_execution_lock:
            for item in reversed(self.coding_execution_history):
                if item.get("execution_id") == execution_id:
                    item["seatbelt_fallback"] = True
                    item["seatbelt_fallback_reason"] = reason
                    break
        return result

    def _coding_format_execution(self, result):
        if not isinstance(result, dict):
            return str(result)
        if result.get("status") == "blocked":
            return "PROJECT COMMAND BLOCKED: " + str(result.get("error", "Unknown policy failure."))
        output = result.get("stdout") if result.get("exit_code") == 0 else result.get("stderr") or result.get("stdout")
        output = str(output or "(no output)")[:12000]
        truncation = "\n[output capped by KIRA]" if result.get("output_truncated") else ""
        return (
            f"PROJECT COMMAND `{result.get('command', '')}` in `{result.get('cwd', '')}` "
            f"{result.get('status', 'failed')} with code `{result.get('exit_code')}` "
            f"in {result.get('duration_ms', 0)} ms [Execution: `{result.get('execution_id', '')}`; "
            f"Seatbelt: {result.get('seatbelt', 'unknown')}]:\n```\n{output}{truncation}\n```"
        )

    def _coding_plan_preview(self, project, plan):
        output = [plan.get("summary", "Coding change"), ""]
        for item in plan.get("files", []):
            target = os.path.join(project, item["path"])
            try:
                with open(target, "r", encoding="utf-8", errors="replace") as handle:
                    before = handle.read()
            except FileNotFoundError:
                before = ""
            diff = difflib.unified_diff(
                before.splitlines(), item["content"].splitlines(),
                fromfile="a/" + item["path"], tofile="b/" + item["path"], lineterm="",
            )
            output.append("\n".join(diff)[:8000] or f"Create {item['path']}")
        if plan.get("commands"):
            output.append("\nApproved project commands:")
            output.extend("- " + item["command"] for item in plan["commands"])
        return "\n".join(output)[:30000]

    def _execute_coding_apply(self, payload):
        run_id = str(payload.get("run_id", "") or "code_" + uuid.uuid4().hex[:16])
        project = os.path.realpath(str(payload.get("project_path", "")))
        if not os.path.isdir(project) or self._coding_project_is_sensitive(project):
            self._coding_run_event(run_id, "blocked", {"reason": "project_unavailable_or_sensitive"})
            return "CODING APPLY BLOCKED: project folder is unavailable or sensitive. Nothing changed."
        try:
            plan = self._coding_validate_plan(project, payload)
        except Exception as exc:
            self._coding_run_event(run_id, "blocked", {"reason": str(exc)[:1000]})
            return f"CODING APPLY BLOCKED: {exc}. Nothing changed."
        self._coding_run_event(run_id, "approved", {
            "files": len(plan["files"]), "commands": len(plan["commands"]),
        })
        backup_root = os.path.join(self.coding_backups_path, str(uuid.uuid4()))
        changed, verified, command_results = [], [], []
        original_state = {}
        for item in plan["files"]:
            target = os.path.realpath(os.path.join(project, item["path"]))
            existed = os.path.exists(target)
            original_state[target] = existed
            if existed:
                backup = os.path.join(backup_root, item["path"])
                os.makedirs(os.path.dirname(backup), exist_ok=True)
                shutil.copy2(target, backup)
            result = self._write_file_with_seatbelt(target, item["content"], append=False)
            if str(result).startswith("WRITE FAILED"):
                rollback = self._coding_rollback(project, backup_root, changed + [item["path"]], original_state)
                self._coding_run_event(run_id, "rolled_back", {
                    "reason": "write_failed", "rollback": rollback,
                })
                return (
                    f"CODING APPLY FAILED after {len(changed)} file(s):\n{result}\n"
                    f"Rollback: {rollback}\nBackups: `{backup_root}`"
                )
            changed.append(item["path"])
            try:
                with open(target, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()[:12]
                verified.append(f"{item['path']} sha256:{digest}")
            except Exception:
                verified.append(f"{item['path']} written; hash unavailable")
        failed_checks = []
        for item in plan["commands"]:
            if self._coding_dangerous_command(item["command"]):
                command_results.append(f"BLOCKED after revalidation: {item['command']}")
                continue
            execution = self._coding_execute_project_command({
                "cwd": project,
                "command": item["command"],
                "timeout": self.CODING_COMMAND_TIMEOUT_SECONDS,
            })
            command_results.append(self._coding_format_execution(execution))
            if not execution.get("ok"):
                failed_checks.append(execution)
        if failed_checks:
            rollback = self._coding_rollback(project, backup_root, changed, original_state)
            self._coding_run_event(run_id, "rolled_back", {
                "reason": "verification_failed",
                "failed_checks": len(failed_checks),
                "rollback": rollback,
            })
            chat_id = payload.get("chat_id")
            if chat_id:
                self._coding_progress(chat_id, "Failed", 100, "Verification failed; coding edits were rolled back.")
            return (
                "CODING VERIFICATION FAILED. KIRA restored the pre-task files and did not open the IDE.\n\n"
                + "\n\n".join(command_results)
                + f"\n\nRollback: {rollback}\nBackups: `{backup_root}`"
            )
        ide_result = ""
        ide = str(payload.get("ide", ""))
        if ide:
            primary_file = os.path.realpath(os.path.join(project, changed[0])) if changed else ""
            open_targets = [project] + ([primary_file] if primary_file and os.path.isfile(primary_file) else [])
            opened = subprocess.run(["open", "-a", ide, *open_targets], capture_output=True, text=True, timeout=20)
            if opened.returncode == 0:
                ide_result = (
                    f"Opened `{changed[0]}` in {ide} with the project."
                    if changed else f"Opened project in {ide}."
                )
            else:
                ide_result = f"IDE open failed: {(opened.stderr or opened.stdout).strip()}"
        chat_id = payload.get("chat_id")
        if chat_id:
            self._coding_progress(chat_id, "Complete", 100, "Coding changes applied and verified.")
        self._coding_run_event(run_id, "completed", {
            "files": len(changed), "checks": len(command_results), "ide_opened": bool(ide_result and ide_result.startswith("Opened")),
        })
        return (
            f"Coding task applied and verified {len(changed)} file(s).\n"
            + "\n".join(f"- {row}" for row in verified)
            + ("\n\nProject checks:\n" + "\n\n".join(command_results) if command_results else "")
            + ("\n\n" + ide_result if ide_result else "")
            + (f"\nBackups: `{backup_root}`" if changed else "")
        ).strip()

    def _coding_rollback(self, project, backup_root, changed, original_state):
        failures = []
        for relative in reversed(changed):
            target = os.path.realpath(os.path.join(project, relative))
            backup = os.path.join(backup_root, relative)
            try:
                if original_state.get(target):
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy2(backup, target)
                elif os.path.exists(target):
                    os.remove(target)
            except Exception as exc:
                failures.append(f"{relative}: {exc}")
        return "complete" if not failures else "incomplete (" + "; ".join(failures) + ")"
