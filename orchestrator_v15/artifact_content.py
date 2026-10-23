"""Shared checks on audience-visible content, independent of the renderer."""
import re


def without_asset_paths(text):
    return re.sub(
        r'`(?:/|~)[^`\n]+`|https?://\S+|'
        r'(?<!\w)(?:~?/)[^\n`]*?\.(?:png|jpe?g|webp|gif|bmp|pdf|docx?|pptx?|txt|md|csv)\b|'
        r'(?<!\w)/(?:[^\s`]+)', '', str(text), flags=re.I)


def requested_image_count(request, default=1):
    match = re.search(
        r'\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|single)\s+'
        r'(?:(?:distinct|relevant|original|generated|analyzed|local|edited)\s+)*'
        r'(?:images?|photos?|visuals?|pictures?|illustrations?)\b', without_asset_paths(request), re.I)
    if not match:
        return default
    words = dict(zip(('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'), range(1, 11)))
    words['single'] = 1
    value = match.group(1).lower()
    return min(20, max(1, int(value) if value.isdigit() else words[value]))


IMAGE_INSTRUCTION = re.compile(
    r'\b(?:image[- ]generation prompt|image prompt|prompt to (?:generate|create)|'
    r'(?:generate|create|render) (?:an? )?(?:image|photo|illustration)|'
    r'(?:image|photo) of [^\n]*\b(?:no text|aspect ratio|cinematic lighting))\b', re.I)


def visible_content_issue(body):
    # Paths, direct source links and layout metadata are not audience prose.
    visible = re.sub(r'(?im)^\s*(?:IMAGE|COVER_IMAGE|LINK|THEME)\s*:[^\n]*$', '', body)
    if IMAGE_INSTRUCTION.search(visible):
        return ('Visible content contains an image-making instruction. Put the prompt in '
                'IMAGE_GENERATE or IMAGE_EDIT and write audience-facing content in the artifact.')
    return None


def edited_image_issue(controller, images):
    if controller.wants_edited_images and not set(images).intersection(controller.edited_image_paths):
        return 'Embed the analyzed IMAGE_EDIT result, not only the unedited source.'
    return None
