"""Make a verified browser screenshot visible in Kira's chat renderer."""
import re


def with_screenshot(answer, path):
    text = str(answer or '').strip()
    path = str(path or '').strip()
    if not path:
        return text
    if re.search(r'!\[[^\]]*\]\(<?' + re.escape(path) + r'>?\)', text):
        return text
    text = re.sub(r'\[Screenshot of [^\n\]]+\]\s*\n\[' + re.escape(path) + r'\]',
                  '', text, flags=re.I).strip()
    text = text.replace('[' + path + ']', '').strip()
    text = re.sub(r'(?m)^' + re.escape(path) + r'\s*$', '', text).strip()
    return text + '\n\n![Browser screenshot](<' + path + '>)'
