"""Conservative reply-aware delivery policy; not a trained emotion predictor."""
import re


def response_style(public_text, listener_style):
    text = public_text.casefold().replace("’", "'")
    # Reply meaning takes precedence over mirroring the user's arousal.
    if re.search(r"\b(?:sorry for your loss|grief|grieving|passed away|that sounds painful|that sounds difficult|i'm here with you)\b", text):
        manner = "gentle, compassionate, and grounded; use an unhurried pace and soft natural pauses"
    elif re.search(r"\b(?:can't verify|cannot verify|haven't verified|not sure|uncertain|need your permission|approval required)\b", text):
        manner = "clear, measured, and honest; avoid confident or celebratory delivery"
    elif re.search(r"\b(?:congratulations|well done|happy birthday|that's wonderful|great news)\b", text):
        manner = "warm and lightly joyful; do not shout or exaggerate excitement"
    elif re.search(r"\b(?:let's|first step|next step|here's how|step one)\b", text):
        manner = "calm, focused, and encouraging; keep instructions easy to follow"
    else:
        return listener_style
    return f"Speak in a {manner} manner. Keep the voice coherent and conversational. Do not read style directions aloud."
