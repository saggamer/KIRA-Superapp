import json
import os
import re
import subprocess
import sys
import traceback


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def install_transformers_compat():
    import transformers

    auto_tokenizer = transformers.AutoTokenizer
    if getattr(auto_tokenizer, "_kira_string_register_compat", False):
        return
    original_register = auto_tokenizer.register

    def safe_register(cls, config_class, *args, **kwargs):
        if isinstance(config_class, str):
            return None
        return original_register(config_class, *args, **kwargs)

    auto_tokenizer.register = classmethod(safe_register)
    auto_tokenizer._kira_string_register_compat = True


def render_prompt(agent, task):
    system = agent.get("system_prompt") or "You are a focused KIRA OS subagent. Return concise evidence."
    body = task.get("task") or task.get("prompt") or ""
    return f"{system}\n\nTask packet:\n{body}\n\nReturn concise evidence and a recommended next step."


def public_output(text):
    value = str(text or "").strip()
    channel = re.search(
        r"(?is)<\|channel\>\s*(?:thought|analysis|reasoning)?\s*(.*?)<channel\|>(.*)$",
        value
    )
    if channel:
        return channel.group(2).strip()
    thought = re.search(r"(?is)<thought_process>.*?</thought_process>(.*)$", value)
    if thought:
        return thought.group(1).strip()
    if re.search(r"(?is)<\|channel\>\s*(?:thought|analysis|reasoning)", value):
        return ""
    if "<thought_process>" in value:
        return ""
    return value


def run():
    agent = load_json(sys.argv[1])
    task = load_json(sys.argv[2])
    backend = (agent.get("backend") or "").lower()
    model_path = agent.get("model_path") or ""
    prompt = render_prompt(agent, task)

    if backend == "mlx":
        install_transformers_compat()
        from mlx_lm import generate, load
        model, tokenizer = load(model_path)
        try:
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            chat_prompt = prompt
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=float(task.get("temperature", 0.35)))
            output = generate(
                model,
                tokenizer,
                prompt=chat_prompt,
                max_tokens=int(task.get("max_tokens", 600)),
                sampler=sampler,
                verbose=False
            )
        except (ImportError, TypeError):
            output = generate(
                model,
                tokenizer,
                prompt=chat_prompt,
                max_tokens=int(task.get("max_tokens", 600)),
                temperature=float(task.get("temperature", 0.35)),
                verbose=False
            )
    elif backend == "ollama":
        model_ref = agent.get("model_ref") or os.path.basename(model_path)
        proc = subprocess.run(["ollama", "run", model_ref], input=prompt, capture_output=True, text=True, timeout=int(task.get("timeout", 90)))
        if proc.returncode != 0:
            print(json.dumps({"ok": False, "agent": agent.get("name"), "error": (proc.stderr or proc.stdout or "Ollama execution failed.").strip()}, ensure_ascii=False))
            return 2
        output = proc.stdout
    else:
        print(json.dumps({
            "ok": False,
            "agent": agent.get("name"),
            "error": f"Unsupported subagent backend: {backend or 'missing'}"
        }, ensure_ascii=False))
        return 2

    output = public_output(output)
    if not output:
        print(json.dumps({"ok": False, "agent": agent.get("name"), "error": "Subagent returned no output."}, ensure_ascii=False))
        return 3
    print(json.dumps({"ok": True, "agent": agent.get("name"), "output": output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, ensure_ascii=False))
        sys.exit(1)
