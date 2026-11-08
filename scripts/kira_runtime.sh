# Shared macOS launcher resolution. Keep SDK files outside cloud-managed Documents.
kira_runtime_python() {
    if [[ -n "${KIRA_PYTHON:-}" ]]; then
        printf '%s\n' "$KIRA_PYTHON"
    elif [[ -x "$HOME/.local/share/kira-superapp/runtime/bin/python" ]]; then
        printf '%s\n' "$HOME/.local/share/kira-superapp/runtime/bin/python"
    elif [[ -x "$PWD/.venv/bin/python" ]]; then
        printf '%s\n' "$PWD/.venv/bin/python"
    else
        printf '%s\n' "$PWD/.venv-kira-live/bin/python"
    fi
}
