"""Explicit Live action contracts; no permissions or model-worker bypass."""
import re


def wants_search(text):
    return bool(re.search(r'\b(?:search|look up|browser tool|weather|forecast)\b|\bfind\b.*\b(?:website|stories|best|spot)\b', text, re.I))


def wants_docx(text):
    return bool(re.search(r'\b(?:create|make|write|generate)\b', text, re.I) and
                re.search(r'\b(?:word|docx)\b', text, re.I))


def search_wants_open(text, turns):
    if re.search(r"\b(?:don't|do not|without)\s+open", text, re.I):
        return False
    if re.search(r'\bopen\b', text, re.I):
        return True
    recent = [t['content'] for t in turns if t['role'] == 'user'][-2:]
    return bool(re.search(r'\b(?:it|those|them|other)\b', text, re.I) and
                any(wants_search(t) and re.search(r'\bopen\b', t, re.I) for t in recent))


def search_query(text, turns):
    request = re.sub(r'\b(?:browser tool|using (?:the )?tree|using (?:the )?browser)\b', '', text, flags=re.I)
    context = []
    if re.search(r'\b(?:it|those|them|other|more)\b', request, re.I):
        context = [t['content'] for t in turns if t['role'] == 'user' and len(t['content'].split()) >= 5][-2:]
    query = ' '.join([request, *context])
    return re.sub(r'[\[\]\r\n]', ' ', query).strip()[:1400]


def docx_contract(text, request=''):
    """Wrap generated prose, not a made-up story or an execution promise."""
    body = re.sub(r'\*\*|^#+\s*', '', text.strip(), flags=re.M)
    if len(body.split()) < 90 or re.search(r"\b(?:cannot|can't)\b.{0,35}\b(?:create|document|file|tool)\b", body, re.I):
        raise ValueError('The Live model did not produce enough document content; no file was claimed.')
    sentences = re.split(r'(?<=[.!?])\s+', body)
    if len(sentences) < 4:
        raise ValueError('The document content needs complete paragraphs.')
    groups = [sentences[i * len(sentences) // 4:(i + 1) * len(sentences) // 4] for i in range(4)]
    paragraphs = [' '.join(group).replace('\n', ' ') for group in groups]
    theme = [word.capitalize() for word in ('spooky', 'horror', 'midnight', 'camp') if re.search(r'\b' + word + r'\b', request, re.I)]
    title = ' '.join(theme + ['Story']) if theme else 'KIRA Story'
    return '[NATIVE_DOCX]\nTITLE: ' + title + '\n' + '\n'.join('PARAGRAPH: ' + p for p in paragraphs) + '\n[/NATIVE_DOCX]'


def result_links(evidence):
    section = evidence.split('Real result links:', 1)[-1].split('Fetched top pages:', 1)[0] if 'Real result links:' in evidence else ''
    return re.findall(r'https?://[^\s`<>]+', section)[:3]


def tool_acknowledgement(evidence):
    if any(marker in evidence.lower() for marker in ('blocked safely', 'permission required', 'error:', 'failed safely', 'needs repair')):
        return None  # Existing permission/failure guards still own these cases.
    match = re.search(r'DOCX generated:\s*`([^`]+)`', evidence)
    if match:
        return 'Your Word document is ready. You can open it from the chat: ' + match.group(1)
    links = result_links(evidence)
    if links:
        prefix = 'I found these websites.'
        if re.search(r'(?im)^\s*(?:WEB_OPEN|Opened|OPEN):', evidence):
            prefix += ' I also opened the first result.'
        return prefix + '\n' + '\n'.join(links)
    return None
