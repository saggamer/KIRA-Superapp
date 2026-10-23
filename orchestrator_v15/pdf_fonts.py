"""Choose an available Unicode font instead of silently replacing PDF text."""
import os
from pathlib import Path


def configure_pdf_font(pdf, text, app_root):
    try:
        text.encode('latin-1')
        return 'Helvetica'
    except UnicodeEncodeError:
        pass
    candidates = [os.environ.get('KIRA_PDF_FONT', ''),
                  str(Path(app_root) / 'assets/fonts/DejaVuSans.ttf'),
                  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                  '/Library/Fonts/Arial Unicode.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    from fontTools.ttLib import TTFont
    required = {ord(character) for character in text if not character.isspace()}
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        with TTFont(candidate) as font:
            if not required.issubset(font.getBestCmap() or {}):
                continue
        for style in ('', 'B'):
            pdf.add_font('KiraUnicode', style=style, fname=candidate)
        return 'KiraUnicode'
    raise ValueError('PDF text needs a Unicode font covering all characters; set KIRA_PDF_FONT to a suitable TTF file')
