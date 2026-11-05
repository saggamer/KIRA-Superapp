"""Standalone Horizon image workflows using the same observed tool handoffs."""
import re

from orchestrator_v15.presentation_controller import PresentationController, action_tag, CREATE


EDIT = re.compile(
    r'\b(?:edit|modify|retouch|inpaint|rework|revise|recolor|brighten|darken|crop|resize)\b|'
    r'\b(?:remove|replace)\b.{0,100}\bbackground\b|'
    r'\bmake\b.{0,100}\b(?:brighter|darker|warmer|transparent)\b|'
    r'\badd\b.{0,100}\bto (?:this|the|my) (?:image|photo|picture)\b', re.I)
IMAGE = re.compile(r'\b(?:image|photo|picture|illustration|artwork)s?\b|\.(?:png|jpe?g|webp)\b', re.I)


def is_image_request(request):
    return bool(IMAGE.search(request) and (CREATE.search(request) or EDIT.search(request)))


class KiraImageCandidate(PresentationController):
    def __init__(self, request, context='', max_repairs=3):
        super().__init__(request, context=context, max_repairs=max_repairs)
        self.needs_images = True
        self.wants_edited_images = bool(EDIT.search(request))
        self.wants_generated_images = not self.wants_edited_images
        self.output_image_paths = set()

    def normalize(self, answer):
        return str(answer or '').strip()

    def guidance(self):
        if getattr(self, 'image_backend', '') == 'horizon-1':
            return ('Horizon 1 image task. Inspect the supplied source with IMAGE_ANALYZE, then return '
                    '[IMAGE_EDIT]\nPATH: observed local source\nPROMPT: original requested change\n[/IMAGE_EDIT]. '
                    'Horizon learns its region from the source and instruction: omit MASK, REGION and STRENGTH. '
                    'It uses a 512 by 512 working canvas. Analyze the raw edited result before reporting its path; '
                    'count and placement errors can persist. For a new image use IMAGE_GENERATE with PROMPT '
                    'and optional SEED; it uses Horizon’s trained creativity expert without a source image. '
                    'Do not claim that image output proves every requested detail was satisfied.')
        return (
            'Standalone Horizon image task. Return one closed action block at a time. '
            'For an edit, IMAGE_ANALYZE the supplied local image to inspect it, then use '
            '[IMAGE_EDIT]\nPATH: observed local source\nPROMPT: precise one-line change\n'
            'REGION: center\n[/IMAGE_EDIT]. Choose a supported REGION (left, right, center, '
            'upper_left, upper_right, lower_left, lower_right) for a localized change, or an '
            'observed MASK instead. Omit REGION for a global edit; optional STRENGTH is 0.1 to 0.85. '
            'For a new image use [IMAGE_GENERATE]\nPROMPT: concrete one-line visual brief\n'
            '[/IMAGE_GENERATE]. Never substitute a new image for an edit. '
            'Analyze every returned image with IMAGE_ANALYZE. After all requested outputs '
            'have been returned and analyzed, report their exact local paths. '
            'An image analysis is an inspection, not proof that every requested detail is correct. '
            'Do not create a slide deck or document unless asked.'
        )

    def preflight(self, answer):
        tag, _body = action_tag(answer)
        if tag is None:
            if self.verified_path:
                return None
            return 'Create or edit the requested image and analyze the returned output before reporting completion.'
        if tag not in {'IMAGE_GENERATE', 'IMAGE_EDIT', 'IMAGE_ANALYZE', 'WEB_IMAGE_DOWNLOAD',
                       'READ_FILE', 'LIST_DIR', 'SEARCH_FILES', 'FIND_IN_COMPUTER'}:
            return 'Use an image action for this request.'
        if self.verified_path:
            return 'The requested image outputs are already analyzed. Report their paths.'
        if tag in {'IMAGE_GENERATE', 'IMAGE_EDIT'} and len(self.output_image_paths) >= self.required_images:
            return 'Analyze the returned output images before generating duplicates.'
        return super().preflight(answer)

    def record(self, answer, result):
        super().record(answer, result)
        tag, _body = action_tag(answer)
        expected = {'IMAGE_GENERATE': 'IMAGE_GENERATED:', 'IMAGE_EDIT': 'IMAGE_EDITED:'}.get(tag)
        if expected and str(result).startswith(expected):
            self.output_image_paths.update(re.findall(
                r'(?im)^Path:\s*`?([^\n`]+\.png)`?\s*$', str(result)))
        complete = self.output_image_paths.intersection(self.analyzed_image_paths)
        if len(complete) >= self.required_images:
            self.verified_path = sorted(complete)[0]

    def result_hint(self):
        if self.verified_path:
            return 'Report these generated and analyzed image paths: ' + ', '.join(sorted(
                self.output_image_paths.intersection(self.analyzed_image_paths)))
        if self.output_image_paths - self.analyzed_image_paths:
            return 'Analyze the returned output image before reporting completion.'
        return 'Continue with the requested image edit or generation using observed source paths.'
