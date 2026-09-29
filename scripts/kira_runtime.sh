# Shared macOS launcher resolution. Keep SDK files outside cloud-managed Documents.
kira_runtime_python() {
    if [[ -n "${KIRA_PYTHON:-}" ]]; then
        print -r -- "$KIRA_PYTHON"
    elif [[ -x "$HOME/.local/share/kira-superapp/runtime/bin/python" ]]; then
        print -r -- "$HOME/.local/share/kira-superapp/runtime/bin/python"
    elif [[ -x "$PWD/.venv/bin/python" ]]; then
        print -r -- "$PWD/.venv/bin/python"
    else
        print -r -- "$PWD/.venv-kira-live/bin/python"
    fi
}
