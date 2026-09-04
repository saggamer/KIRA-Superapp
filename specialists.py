import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class KiraSpecialistHub:
    """In-process specialists that turn Tree task packets into verified evidence."""

    AGENTS = {
        "research": {
            "name": "Research Agent",
            "purpose": "Fetch readable web sources and return an evidence bundle.",
            "parallel": True,
            "inputs": ["QUERY", "MAX_PAGES"],
            "outputs": ["source text", "source URLs", "fetch count"],
        },
        "media": {
            "name": "Media Agent",
            "purpose": "Find and download usable visual assets for artifacts.",
            "parallel": True,
            "inputs": ["QUERY", "LIMIT"],
            "outputs": ["local image paths", "source URLs"],
        },
        "slides": {
            "name": "Slides Agent",
            "purpose": "Build a PPTX from a structured slide specification and verify it.",
            "parallel": False,
            "inputs": ["REQUEST", "SPEC", "EVIDENCE"],
            "outputs": ["PPTX path", "quality report", "provenance"],
        },
        "document": {
            "name": "Document Agent",
            "purpose": "Build a DOCX from a structured document specification and verify it.",
            "parallel": False,
            "inputs": ["REQUEST", "SPEC", "EVIDENCE"],
            "outputs": ["DOCX path", "quality report", "provenance"],
        },
        "artifact_verifier": {
            "name": "Artifact Verifier",
            "purpose": "Inspect a generated PPTX, DOCX, or PDF before completion is claimed.",
            "parallel": False,
            "inputs": ["PATH", "REQUEST", "EVIDENCE"],
            "outputs": ["verified status", "quality report"],
        },
    }

    def __init__(self, owner):
        self.owner = owner

    def registry(self):
        return {
            "version": "specialists-1",
            "contract": (
                "Orchestrator chooses a specialist. Specialists return evidence; "
                "only a verified artifact may branch into completion."
            ),
            "agents": [
                {"id": agent_id, **profile}
                for agent_id, profile in self.AGENTS.items()
            ],
        }

    def registry_text(self):
        registry = self.registry()
        lines = ["SPECIALIST_REGISTRY:", f"Version: {registry['version']}"]
        for item in registry["agents"]:
            lines.append(
                f"- {item['id']}: {item['name']} | parallel={str(item['parallel']).lower()} | "
                f"{item['purpose']}"
            )
        return "\n".join(lines)

    def execute_many(self, packets, chat_id=""):
        indexed = list(enumerate(packets or []))
        results = {}
        parallel = []
        serial = []
        for index, packet in indexed:
            agent = str(packet.get("agent", "")).strip().lower()
            profile = self.AGENTS.get(agent, {})
            if profile.get("parallel"):
                parallel.append((index, packet))
            else:
                serial.append((index, packet))

        if parallel:
            workers = min(4, len(parallel))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.execute, packet, chat_id): index
                    for index, packet in parallel
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as error:
                        results[index] = self._result(
                            packets[index].get("agent", "unknown"),
                            "failed",
                            error=str(error),
                        )

        for index, packet in serial:
            results[index] = self.execute(packet, chat_id)
        return [results[index] for index, _packet in indexed]

    def execute(self, packet, chat_id=""):
        agent = str(packet.get("agent", "")).strip().lower()
        started = time.perf_counter()
        if agent not in self.AGENTS:
            return self._result(
                agent or "unknown",
                "failed",
                error=f"Unknown specialist. Available: {', '.join(self.AGENTS)}",
            )

        if agent == "research":
            result = self._research(packet)
        elif agent == "media":
            result = self._media(packet)
        elif agent == "slides":
            result = self._slides(packet, chat_id)
        elif agent == "document":
            result = self._document(packet, chat_id)
        else:
            result = self._verify(packet, chat_id)

        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result

    def _research(self, packet):
        query = str(packet.get("query") or packet.get("task") or "").strip()
        if not query:
            return self._result("research", "failed", error="QUERY is required.")
        max_pages = str(packet.get("max_pages") or "5")
        evidence = self.owner._web_research_tool(query, max_pages=max_pages)
        has_source_url = bool(re.search(r"https?://[^\s`'\"<>]+", str(evidence or ""), re.IGNORECASE))
        ok = self.owner._has_web_evidence(evidence) and has_source_url
        return self._result(
            "research",
            "verified" if ok else "failed",
            evidence=evidence,
            error="" if ok else "No fetched page evidence with a direct source URL was returned.",
        )

    def _media(self, packet):
        query = str(packet.get("query") or packet.get("task") or "").strip()
        if not query:
            return self._result("media", "failed", error="QUERY is required.")
        limit = str(packet.get("limit") or "5")
        evidence = self.owner._web_image_search_tool(query, limit=limit)
        evidence_text = str(evidence or "")
        candidates = self._media_candidates(evidence_text)
        query_terms = self._media_query_terms(query)
        valid_images = []
        rejected = []
        seen_hashes = set()
        for candidate in candidates:
            path = self.owner._resolve_path(candidate["path"].rstrip(").,;]"))
            if not os.path.isfile(path):
                rejected.append(f"{path}: missing file")
                continue
            try:
                visual_quality = self.owner._image_quality_report(path)
                visual_ok = bool(visual_quality.get("ok"))
                if not visual_ok:
                    rejected.append(
                        f"{path}: " + "; ".join(visual_quality.get("reasons", []))
                    )
                context = " ".join([
                    os.path.basename(path),
                    candidate.get("source", ""),
                    candidate.get("page", ""),
                    candidate.get("alt", ""),
                ]).lower()
                if query_terms and not any(term in context for term in query_terms):
                    rejected.append(f"{path}: source metadata does not match the query")
                    continue
                if not visual_ok:
                    continue
                with open(path, "rb") as image_stream:
                    content_hash = hashlib.sha256(image_stream.read()).hexdigest()
                if content_hash in seen_hashes:
                    rejected.append(f"{path}: duplicate visual asset")
                    continue
                seen_hashes.add(content_hash)
                valid_images.append(path)
            except Exception as error:
                rejected.append(f"{path}: unreadable ({error})")
                continue
        ok = bool(valid_images)
        quality = {
            "ok": ok,
            "accepted": len(valid_images),
            "rejected": len(rejected),
            "rejection_reasons": rejected[:8],
        }
        return self._result(
            "media",
            "verified" if ok else "failed",
            evidence=evidence,
            assets=valid_images,
            quality=quality,
            error="" if ok else "No relevant, presentation-quality local image asset was downloaded.",
        )

    def _media_candidates(self, evidence):
        entries = []
        pattern = re.compile(
            r"(?ims)^\s*-\s*(?:Path:\s*)?`?(?P<path>(?:~|/)[^\n\r`'\"<>]+?\.(?:png|jpe?g|gif|webp))`?"
            r"(?P<meta>.*?)(?=^\s*-\s*(?:Path:\s*)?`?(?:~|/)|\Z)"
        )
        for match in pattern.finditer(str(evidence or "")):
            meta = match.group("meta") or ""
            values = {}
            for key in ("source", "page", "alt"):
                field = re.search(rf"(?im)^\s*{key}\s*:\s*(.+?)\s*$", meta)
                values[key] = field.group(1).strip().strip("`") if field else ""
            entries.append({"path": match.group("path").strip(), **values})

        if entries:
            return entries
        return [
            {"path": path, "source": "", "page": "", "alt": ""}
            for path in re.findall(
                r"(?:~|/)[^\n\r`'\"<>]+?\.(?:png|jpe?g|gif|webp)",
                str(evidence or ""),
                re.IGNORECASE,
            )
        ]

    def _media_query_terms(self, query):
        generic = {
            "a", "an", "and", "art", "for", "from", "image", "images", "in",
            "of", "official", "photo", "photos", "picture", "pictures", "reference",
            "references", "the", "to", "visual", "visuals", "with",
        }
        terms = [
            term for term in re.findall(r"[a-z0-9]{3,}", str(query or "").lower())
            if term not in generic
        ]
        return terms[:10]

    def _slides(self, packet, chat_id):
        request = str(packet.get("request") or packet.get("task") or "").strip()
        spec = str(packet.get("spec") or "").strip()
        evidence = str(packet.get("evidence") or "").strip()
        if not spec:
            return self._result("slides", "failed", error="SPEC is required.")

        preflight_reasons = self.owner._artifact_spec_quality_reasons(spec, "pptx", request)
        if preflight_reasons:
            return self._result(
                "slides",
                "needs_repair",
                quality={"ok": False, "reasons": preflight_reasons},
                error="; ".join(preflight_reasons),
            )

        generated = self.owner._generate_native_pptx(spec, record_artifact=False)
        path = self._artifact_path(generated, ".pptx")
        if not path or not os.path.exists(path):
            return self._result("slides", "failed", evidence=generated, error="PPTX was not created.")

        quality = self.owner._pptx_quality_report(path, request)
        if self.owner._prompt_needs_web_evidence_for_artifact(request):
            cached = getattr(self.owner, "web_evidence_by_chat", {}).get(chat_id, "")
            if not self.owner._has_web_evidence(evidence, cached):
                quality.setdefault("reasons", []).append("requested web evidence is missing")
                quality["ok"] = False
        if quality.get("ok"):
            self.owner._record_artifact(path, "presentation", self.owner._parse_fields(spec).get("title", "KIRA Deck"))
            self.owner._attach_artifact_provenance(
                path,
                request,
                chat_id,
                evidence=evidence,
                quality=quality,
            )
        return self._result(
            "slides",
            "verified" if quality.get("ok") else "needs_repair",
            path=path,
            evidence=generated,
            quality=quality,
            error="" if quality.get("ok") else "; ".join(quality.get("reasons", [])),
        )

    def _document(self, packet, chat_id):
        request = str(packet.get("request") or packet.get("task") or "").strip()
        spec = str(packet.get("spec") or "").strip()
        evidence = str(packet.get("evidence") or "").strip()
        if not spec:
            return self._result("document", "failed", error="SPEC is required.")

        preflight_reasons = self.owner._artifact_spec_quality_reasons(spec, "docx", request)
        if preflight_reasons:
            return self._result(
                "document",
                "needs_repair",
                quality={"ok": False, "reasons": preflight_reasons},
                error="; ".join(preflight_reasons),
            )

        generated = self.owner._generate_native_docx(spec, record_artifact=False)
        path = self._artifact_path(generated, ".docx")
        if not path or not os.path.exists(path):
            return self._result("document", "failed", evidence=generated, error="DOCX was not created.")

        quality = self.owner._docx_quality_report(path, request)
        if self.owner._prompt_needs_web_evidence_for_artifact(request):
            cached = getattr(self.owner, "web_evidence_by_chat", {}).get(chat_id, "")
            if not self.owner._has_web_evidence(evidence, cached):
                quality.setdefault("reasons", []).append("requested web evidence is missing")
                quality["ok"] = False
        if quality.get("ok"):
            self.owner._record_artifact(path, "word", self.owner._parse_fields(spec).get("title", "KIRA Document"))
            self.owner._attach_artifact_provenance(
                path,
                request,
                chat_id,
                evidence=evidence,
                quality=quality,
            )
        return self._result(
            "document",
            "verified" if quality.get("ok") else "needs_repair",
            path=path,
            evidence=generated,
            quality=quality,
            error="" if quality.get("ok") else "; ".join(quality.get("reasons", [])),
        )

    def _verify(self, packet, chat_id):
        path = self.owner._resolve_path(packet.get("path", ""))
        request = str(packet.get("request") or packet.get("task") or "").strip()
        evidence = str(packet.get("evidence") or "").strip()
        if not path or not os.path.exists(path):
            return self._result("artifact_verifier", "failed", error="Artifact path does not exist.")

        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".pptx":
            quality = self.owner._pptx_quality_report(path, request)
        elif suffix == ".docx":
            quality = self.owner._docx_quality_report(path, request)
        elif suffix == ".pdf":
            quality = self.owner._pdf_quality_report(path, request)
        else:
            quality = {"ok": False, "reasons": [f"Unsupported artifact type: {suffix or 'none'}"]}

        if self.owner._prompt_needs_web_evidence_for_artifact(request):
            cached = getattr(self.owner, "web_evidence_by_chat", {}).get(chat_id, "")
            if not self.owner._has_web_evidence(evidence, cached):
                quality.setdefault("reasons", []).append("requested web evidence is missing")
                quality["ok"] = False
        return self._result(
            "artifact_verifier",
            "verified" if quality.get("ok") else "needs_repair",
            path=path,
            quality=quality,
            error="" if quality.get("ok") else "; ".join(quality.get("reasons", [])),
        )

    def _artifact_path(self, text, suffix):
        for candidate in re.findall(r"`([^`]+)`", str(text or "")):
            if candidate.lower().endswith(suffix):
                return self.owner._resolve_path(candidate)
        return ""

    def _result(self, agent, status, *, path="", evidence="", quality=None, error="", assets=None):
        result = {
            "agent": str(agent or "unknown"),
            "status": status,
            "path": path,
            "evidence": evidence,
            "quality": quality or {},
            "error": error,
        }
        if assets is not None:
            result["assets"] = list(assets)
        return result

    @staticmethod
    def format_result(result):
        agent = str(result.get("agent", "unknown"))
        status = str(result.get("status", "failed"))
        lines = [
            "SPECIALIST_RESULT:",
            f"Agent: {agent}",
            f"Status: {status}",
        ]
        if result.get("path"):
            lines.append(f"Artifact: `{result['path']}`")
        if result.get("quality"):
            lines.append("Quality: " + json.dumps(result["quality"], ensure_ascii=False, sort_keys=True))
        if result.get("evidence"):
            lines.append("Evidence:\n" + str(result["evidence"]))
        if result.get("error"):
            lines.append("Error: " + str(result["error"]))
        if "elapsed_seconds" in result:
            lines.append(f"Elapsed: {result['elapsed_seconds']}s")
        return "\n".join(lines)
