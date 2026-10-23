"""Separate Kira slide specialist candidate, sharing the promoted v1.5 weights."""
import re

from orchestrator_v15.artifact_content import edited_image_issue, visible_content_issue

from orchestrator_v15.presentation_controller import PresentationController, SLIDE, action_tag, clean_local_image_fields


class KiraPPTCandidate(PresentationController):
    def normalize(self, answer):
        """Translate nested slide fields and remove Kira's redundant cover."""
        raw = str(answer or '').strip()
        if action_tag(raw)[0] is None and raw.count('[NATIVE_PPTX]') == 1 and raw.count('[/NATIVE_PPTX]') == 1:
            match = re.search(r'(?s)\[NATIVE_PPTX\]\s*\n.*?\n\[/NATIVE_PPTX\]', raw)
            if match:
                answer = match.group(0)
        tag, body = action_tag(answer)
        if tag != 'NATIVE_PPTX':
            return answer
        body = re.sub(r'(?im)^SLIDE:[ \t]*\n[ \t]+TITLE:[ \t]*(.+)$',
                      lambda match: 'SLIDE: ' + match.group(1).strip(), body)
        body = re.sub(r'(?im)^[ \t]*BODY:[ \t]*\n((?:[ \t]+-[^\n]*(?:\n|$))+)',
                      lambda match: ''.join('BULLET: ' + line.strip().removeprefix('- ').strip() + '\n'
                                            for line in match.group(1).splitlines() if line.strip()), body)
        body = re.sub(r'(?m)^[ \t]+(IMAGE|COVER_IMAGE|LINK):', r'\1:', body)
        body = re.sub(r'(?im)^\s*VISUAL\s*:\s*\n\s*PATH\s*:\s*([^\n]+)$',
                      lambda match: 'IMAGE: ' + match.group(1).strip(), body)
        slides = list(SLIDE.finditer(body))
        if slides and re.match(
                r'(?i)^(?:\d+(?:\s*/\s*\d+)?\s*[-–:]\s*)?(?:cover|title(?: slide)?)\b',
                slides[0].group(1).strip()) and len(slides) >= 2:
            cover = body[slides[0].end():slides[1].start()]
            image = re.search(r'(?im)^\s*IMAGE\s*:\s*(\S+)', cover)
            prefix = body[:slides[0].start()].rstrip()
            if image and not re.search(r'(?im)^\s*COVER_IMAGE\s*:', prefix):
                prefix += '\nCOVER_IMAGE: ' + image.group(1)
            body = prefix + '\n' + body[slides[1].start():]
        return '[NATIVE_PPTX]\n' + clean_local_image_fields(body).strip() + '\n[/NATIVE_PPTX]'

    def guidance(self):
        return super().guidance() + (
            ' This PPT candidate should work as an autonomous slide producer: inspect the brief, '
            'research when facts or links are requested, generate original images when requested, search sourced images, and analyze every selected image, '
            'then make one complete deck. Plan the story across slides before writing. Each slide needs '
            'a distinct insight, specific supporting content, and a visual only when it explains that insight. '
            'When two images are requested, embed two different analyzed local images on distinct content slides. '
            'Write each image as IMAGE: /observed/local/path.png directly inside its SLIDE; '
            'do not use VISUAL: with a nested PATH. '
            'Image-generation and edit prompts belong only in IMAGE_GENERATE or IMAGE_EDIT actions. '
            'Never put a prompt, rendering instruction, or tool command in TITLE, BODY, or BULLET. '
            'Match each slide claim to the analyzed image actually placed on that slide. '
            'If an edit added an object, describe that object only on the slide using the edited image; '
            'do not attribute it to the unedited source. '
            'Anchor every slide to the user\'s named topic; use image analysis to select and explain visuals, '
            'not to replace the requested story with an image-analysis report. '
            'Keep each content slide substantive enough to stand alone: a specific heading and at least '
            '120 readable characters across its body and bullets. '
            'Use no invented source URLs or image paths. Prefer concise headings and scannable body text. '
            'After Kira verifies the PPTX, report the file path and any known limitations.'
        )

    def preflight(self, answer):
        issue = super().preflight(answer)
        if issue:
            return issue
        tag, body = action_tag(answer)
        if tag != 'NATIVE_PPTX':
            return None
        issue = visible_content_issue(body)
        if issue:
            return issue
        image_fields = [path.strip().strip('`') for path in re.findall(
            r'(?im)^\s*(?:COVER_IMAGE|IMAGE)\s*:\s*([^\n]+)', body)]
        issue = edited_image_issue(self, image_fields)
        if issue:
            return issue
        if self.needs_images:
            if any(path not in self.analyzed_image_paths for path in image_fields):
                return 'Analyze each selected local image with IMAGE_ANALYZE before embedding it.'
            slide_sections = re.split(r'(?im)^\s*SLIDE\s*:', body)[1:]
            if self.required_images > 1 and sum('IMAGE:' in section for section in slide_sections) < 2:
                return 'Place the two relevant analyzed images on distinct content slides.'
            for section in slide_sections:
                image = re.search(r'(?im)^\s*IMAGE\s*:\s*([^\n]+)', section)
                if not image:
                    continue
                image_path = image.group(1).strip().strip('`')
                text = re.sub(r'(?im)^\s*IMAGE\s*:\s*[^\n]+', '', section)
                if (self.edited_image_paths and image_path not in self.edited_image_paths and
                        any(re.search(r'\b' + re.escape(subject) + r's?\b', text, re.I)
                            for subject in self.edited_subjects)):
                    return 'A slide describing the added object must use the edited image, not its unedited source.'
        titles = [match.group(1).strip().casefold() for match in SLIDE.finditer(body)]
        if len(titles) != len(set(titles)):
            return 'Every content slide needs a distinct, descriptive title.'
        if any(re.fullmatch(r'(?:introduction|overview|summary|key points|conclusion)', title)
               for title in titles):
            return 'Replace generic slide titles with specific claims that fit the brief.'
        return None
