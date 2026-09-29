import gc
import json
import os
import re
import sys
import traceback
import types
from kira_live.privacy import sanitize_public_output


RUNTIME_STOP_TOKEN_IDS = {1, 49, 51, 106}
MAX_OUTPUT_TOKENS = max(256, int(os.environ.get("KIRA_WORKER_MAX_OUTPUT_TOKENS", "1200")))
def sanitize_generated_content(content):
    """Return only public answer text; private reasoning never crosses IPC."""
    return sanitize_public_output(content)


def install_fast_transformers_shim():
    if sys.modules.get("transformers") is not None:
        return

    try:
        from tokenizers import Tokenizer
    except Exception:
        return

    class KiraFastTokenizer:
        def __init__(self, model_path, **_kwargs):
            self.model_path = os.fspath(model_path)
            tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
            config_path = os.path.join(self.model_path, "tokenizer_config.json")
            chat_template_path = os.path.join(self.model_path, "chat_template.jinja")
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self.init_kwargs = {}
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        self.init_kwargs = json.load(handle)
                except Exception:
                    self.init_kwargs = {}

            self.chat_template = self.init_kwargs.get("chat_template")
            if not self.chat_template and os.path.isfile(chat_template_path):
                try:
                    with open(chat_template_path, "r", encoding="utf-8") as handle:
                        self.chat_template = handle.read()
                except Exception:
                    self.chat_template = None

            self.name_or_path = self.model_path
            self.model_max_length = int(self.init_kwargs.get("model_max_length", 8192) or 8192)
            self.bos_token = self._token_text(self.init_kwargs.get("bos_token")) or "<bos>"
            self.eos_token = self._token_text(self.init_kwargs.get("eos_token")) or "<eos>"
            self.unk_token = self._token_text(self.init_kwargs.get("unk_token")) or "<unk>"
            self.pad_token = self._token_text(self.init_kwargs.get("pad_token")) or self.eos_token
            self.bos_token_id = self.convert_tokens_to_ids(self.bos_token)
            self.eos_token_id = self.convert_tokens_to_ids(self.eos_token)
            self.eos_token_ids = set(RUNTIME_STOP_TOKEN_IDS)
            self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
            self.vocab_size = self._tokenizer.get_vocab_size()
            self.vocab = self._tokenizer.get_vocab()

        @staticmethod
        def _token_text(token):
            if isinstance(token, dict):
                return token.get("content") or token.get("token")
            return None if token is None else str(token)

        def encode(self, text, add_special_tokens=True, **_kwargs):
            return self._tokenizer.encode(
                str(text), add_special_tokens=bool(add_special_tokens)
            ).ids

        def decode(self, tokens, skip_special_tokens=False, **_kwargs):
            if tokens is None:
                return ""
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
            return self._tokenizer.decode(
                [int(token) for token in tokens],
                skip_special_tokens=bool(skip_special_tokens),
            )

        def batch_decode(self, batch, skip_special_tokens=False, **kwargs):
            return [
                self.decode(tokens, skip_special_tokens=skip_special_tokens, **kwargs)
                for tokens in batch
            ]

        def convert_tokens_to_ids(self, token):
            token_id = self._tokenizer.token_to_id(str(token))
            return token_id if token_id is not None else 0

        def get_vocab(self):
            return self.vocab

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            **_kwargs,
        ):
            rendered = []
            for message in messages or []:
                role = str(message.get("role", "user")).lower()
                role = "model" if role in ("assistant", "model") else "user"
                rendered.append(
                    "<|turn>" + role + "\n" + str(message.get("content", "")) + "<turn|>\n"
                )
            if add_generation_prompt:
                rendered.append("<|turn>model\n")
            text = "".join(rendered)
            if not tokenize:
                return text
            ids = self.encode(text, add_special_tokens=False)
            return {"input_ids": ids} if return_dict else ids

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return cls(model_path, **kwargs)

        def __call__(self, text, return_tensors=None, add_special_tokens=True, **_kwargs):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if return_tensors == "np":
                import numpy as np

                return {"input_ids": np.array([ids], dtype=np.int64)}
            return {"input_ids": ids}

    class AutoTokenizer:
        @classmethod
        def register(cls, *_args, **_kwargs):
            return None

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return KiraFastTokenizer(model_path, **kwargs)

    shim = types.ModuleType("transformers")
    shim.AutoTokenizer = AutoTokenizer
    shim.PreTrainedTokenizer = KiraFastTokenizer
    shim.PreTrainedTokenizerFast = KiraFastTokenizer
    shim.__version__ = "kira-fast-shim"
    shim._kira_fast_tokenizer_shim = True
    sys.modules["transformers"] = shim


def send(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def clear_mlx_cache():
    try:
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
            return
        metal = getattr(mx, "metal", None)
        legacy_clear = getattr(metal, "clear_cache", None)
        if callable(legacy_clear):
            legacy_clear()
    except Exception:
        pass


def configure_mlx_cache():
    """Bound reusable GPU buffers without flushing them after every answer."""
    try:
        import mlx.core as mx

        physical = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        limit = min(768 * 1024**2, max(128 * 1024**2, physical // 64))
        mx.set_cache_limit(limit)
        return limit
    except Exception:
        return 0


def configure_stop_tokens(active_tokenizer, include_runtime_tokens=True):
    existing = set(getattr(active_tokenizer, "eos_token_ids", []) or [])
    eos_token_id = getattr(active_tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        existing.add(int(eos_token_id))
    if include_runtime_tokens:
        existing.update(RUNTIME_STOP_TOKEN_IDS)
    try:
        active_tokenizer.eos_token_ids = existing
    except Exception:
        pass


def create_sampler(temperature):
    from mlx_lm.sample_utils import make_sampler

    try:
        return make_sampler(temperature=temperature)
    except TypeError:
        return make_sampler(temp=temperature)


def main():
    model = None
    tokenizer = None
    current_brain = None
    kira_native_thinker = False

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        request_id = ""
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id", ""))
            request_type = str(request.get("type", ""))

            if request_type == "shutdown":
                send({"id": request_id, "ok": True, "shutdown": True})
                break

            if request_type == "load":
                current_brain = request.get("brain", "")
                kira_native_thinker = False
                # Fused Orchestrator checkpoints use KIRA's lightweight tokenizer
                # shim. Qwen3.5 and its LoRA adapter need the full Transformers
                # tokenizer/detokenizer contract supplied by MLX-LM.
                if current_brain != "kira":
                    install_fast_transformers_shim()
                clear_mlx_cache()
                gc.collect()
                cache_limit = configure_mlx_cache()
                model_path = request["model_path"]
                adapter_path = request.get("adapter_path") or None
                thinker_addons_path = request.get("thinker_addons_path") or None
                thinker_continuation_path = request.get("thinker_continuation_path") or None
                expert_rank = int(request.get("expert_rank") or 128)
                if current_brain == "kira" and thinker_addons_path:
                    from kira_live.mlx_thinker import KiraMLXThinker
                    from kira_live.expert_surgery import expert_checkpoint_config

                    expert_config = expert_checkpoint_config(thinker_addons_path)
                    expert_rank = expert_config['expert_rank']
                    model = KiraMLXThinker.from_pretrained(model_path, **expert_config)
                    model.load_weights(thinker_addons_path, strict=False)
                    if thinker_continuation_path:
                        model.load_weights(thinker_continuation_path, strict=False)
                    model.eval()
                    tokenizer = model.tokenizer
                    kira_native_thinker = True
                else:
                    from mlx_lm import load

                    try:
                        model, tokenizer = load(
                            model_path,
                            adapter_path=adapter_path,
                            lazy=True,
                        )
                    except TypeError:
                        model, tokenizer = load(model_path, adapter_path=adapter_path)
                configure_stop_tokens(
                    tokenizer,
                    include_runtime_tokens=current_brain != "kira",
                )
                send({
                    "id": request_id,
                    "ok": True,
                    "loaded": True,
                    "brain": current_brain,
                    "adapter": adapter_path,
                    "thinker_addons": thinker_addons_path,
                    "thinker_continuation": thinker_continuation_path,
                    "expert_rank": expert_rank if kira_native_thinker else None,
                    "expert_profile": expert_config['expert_profile'] if kira_native_thinker else None,
                    "native_thinker": kira_native_thinker,
                    "mlx_cache_limit": cache_limit,
                    "stop_token_ids": sorted(RUNTIME_STOP_TOKEN_IDS),
                })
                continue

            if request_type == "generate":
                if model is None or tokenizer is None:
                    raise RuntimeError("Model is not loaded in the isolated worker.")

                prompt = str(request.get("prompt", ""))
                max_tokens = max(32, min(int(request.get("max_tokens", 900)), MAX_OUTPUT_TOKENS))
                temperature = max(0.0, min(float(request.get("temperature", 0.35)), 2.0))
                if kira_native_thinker:
                    import mlx.core as mx

                    token_ids = mx.array(
                        [tokenizer.encode(prompt, add_special_tokens=False)],
                        dtype=mx.int32,
                    )
                    embeddings = model.embed(token_ids)
                    generated = model.generate_from_embeddings(
                        embeddings,
                        ngram_ids=token_ids,
                        max_tokens=min(max_tokens, 256),
                        compute_response_hidden=False,
                    )
                    content = generated.text
                else:
                    from mlx_lm import generate

                    sampler = create_sampler(temperature)
                    # One request must perform exactly one inference call. Retrying is
                    # owned by the parent process after this worker is restarted.
                    content = generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        verbose=False,
                    )
                public_content = sanitize_public_output(content, live=True) if current_brain == "kira" else sanitize_generated_content(content)
                send({"id": request_id, "ok": True, "content": public_content})
                continue

            raise RuntimeError("Unknown model worker request type: " + request_type)
        except Exception as error:
            clear_mlx_cache()
            gc.collect()
            send({
                "id": request_id,
                "ok": False,
                "error": str(error),
                "trace": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
