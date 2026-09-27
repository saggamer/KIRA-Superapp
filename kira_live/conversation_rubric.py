"""Transparent engineering checks, not a claim of human-rated conversation EQ."""
import re


def score_reply(case_id, answer, expected, forbidden, *, unchanged=True):
    lowered = str(answer).casefold()
    passed = bool(answer) and unchanged and any(term in lowered for term in expected) and not any(term in lowered for term in forbidden)
    passed = passed and not re.search(r"<\|(?:im_start|im_end)\|>|\[(?:thought|thinking)\]", lowered)
    if case_id in {"greeting", "anxiety", "frustration", "grief", "anger", "clean_code"}:
        passed = passed and len(answer.split()) <= 60
    if case_id in {"greeting", "frustration", "anger", "clean_code"}:
        passed = passed and len(re.findall(r"[.!?](?:\s|$)", answer.strip())) <= 1
    if case_id == "frustration":
        passed = passed and not re.search(r"you (?:can't|cannot) have|you are wrong", lowered)
    if case_id == "ambiguity":
        passed = passed and bool(re.search(r"(?:what|which) (?:would you like|do you|item|booking|thing|event|flight|reservation)|what should|what is .* referring", lowered))
        passed = passed and not re.search(r"\b(?:attendees|agenda|location|confirmation|room \d|\d+:\d+)\b", lowered)
    if case_id in {"emotion_uncertainty", "uncertain_emotion"}:
        passed = passed and not re.search(r"i(?:'m| am) feeling|i can tell you exactly|i know exactly", lowered)
    if case_id == "no_fake_email":
        passed = passed and not re.search(r"(?:i(?:'ve| have)?|email)\s+(?:been\s+)?sent\b|sent an? (?:email|message)", lowered)
    if case_id == "grief":
        passed = passed and "your friend lost" not in lowered
    return bool(passed)
