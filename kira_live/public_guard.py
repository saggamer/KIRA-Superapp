"""Conservative public-output guard; execution evidence remains backend-owned."""

from __future__ import annotations

import re


def tools_disallowed(prompt: str) -> bool:
    return bool(re.search(
        r"\b(?:without (?:using )?(?:any )?tools|no tools|do not use (?:any )?tools|don't use (?:any )?tools)\b",
        str(prompt or ""), re.I,
    ))


def guard_public_reply(prompt: str, answer: str, *, observed_local_files: bool = False) -> str:
    request = str(prompt or "").casefold()
    reply = str(answer or "").strip()
    lowered = reply.casefold()
    if re.fullmatch(r"\s*(?:who are you|what(?:'s| is) your name|which model are you|what model are you|identify yourself)(?:[.!?\s]*)", request):
        return "I'm KIRA Live 1."
    asks_local_files = (
        any(place in request for place in ("desktop", "downloads", "home folder"))
        and any(word in request for word in ("files", "folders", "contents", "documents"))
        and any(word in request for word in ("which", "what", "list", "tell", "show"))
    )
    disclaims_observation = any(phrase in lowered for phrase in (
        "can't", "cannot", "don't know", "do not know", "haven't inspected", "have not inspected",
        "need to inspect", "need to check", "without inspecting", "without checking",
    ))
    if asks_local_files and not observed_local_files and not disclaims_observation:
        return "I haven't inspected those files, so I can't verify their names or contents. I can check with a read-only tool."
    asks_exact_emotion = (
        any(word in request for word in ("emotion", "feel", "mood"))
        and any(word in request for word in ("exact", "certain", "guarantee", "prove"))
        and any(phrase in request for phrase in ("how i feel", "what i feel", "my emotion", "my mood", "emotion i feel", "i am", "i feel", "i said", "i typed"))
    )
    asserts_certainty = bool(re.search(
        r"\b(?:can tell|know|tell you|understand)\b.{0,25}\bexactly\b|\byou(?:'re| are) definitely\b",
        lowered,
    ))
    calibrated = any(phrase in lowered for phrase in ("can't know", "cannot know", "not certain", "can't be certain", "cannot be certain", "can't tell", "cannot tell"))
    if asks_exact_emotion and (asserts_certainty or not calibrated):
        return "I can't know your exact emotion from a short phrase or tone alone. I can notice possible cues, but I'd rather ask how you feel than assume."
    relation = re.fullmatch(r"\s*(?:what is|what was|what's) my ([a-z' -]{1,60})\??\s*", request)
    if relation:
        attribute = relation.group(1).strip()
        reply = re.sub(rf"^my\s+{re.escape(attribute)}\b", f"Your {attribute}", reply, count=1, flags=re.I)
    return reply


def guard_live_action_reply(prompt: str, answer: str, evidence: str = "") -> str:
    """Seal spoken completion claims when Tree has no success evidence."""
    lowered_evidence = str(evidence or "").casefold()
    if "permission required:" in lowered_evidence:
        return "I haven't completed that action yet. Please review and approve the permission request in KIRA."
    if any(marker in lowered_evidence for marker in ("failed safely", "blocked safely", "agentic step error:", "agentic step timeout:")):
        return "That action did not complete successfully. The app has the error details, and I haven't marked it done."
    action_requested = bool(re.search(r"\b(?:send|delete|move|book|pay|write|create|generate|open|close)\b", str(prompt or ""), re.I))
    completed_claim = any(
        re.search(r"\b(?:sent|booked|paid|deleted|moved|created|generated|opened|closed|completed|done)\b", clause)
        and not re.search(r"\b(?:not|no|cannot|can't|haven't|hasn't)\b", clause)
        for clause in re.split(r"[.!?;\n]", str(answer or "").casefold())
    )
    success_markers = ("file written", "path moved", "moved to trash", "pptx generated", "pdf generated", "docx generated", "opened:", "open:", "close:", "scheduled task:")
    for action, required in ((r"\bsend\b", "message delivered:"), (r"\bbook\b", "booking confirmed:"), (r"\bpay\b", "payment confirmed:")):
        if completed_claim and re.search(action, str(prompt or ""), re.I) and required not in lowered_evidence:
            return "I haven't verified that action was completed. I won't claim it is done without the matching tool result."
    if action_requested and completed_claim and not any(marker in lowered_evidence for marker in success_markers):
        return "I haven't verified that action was completed. I can help with the next safe step, but I won't claim it is done without a tool result."
    return str(answer or "").strip()
