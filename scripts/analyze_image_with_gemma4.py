#!/usr/bin/env python3
"""Read one local image with the cached Gemma 4 vision stack."""
import argparse
from pathlib import Path
from colorsys import rgb_to_hsv

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/Users/saidhanvanth/.cache/huggingface/hub/models--mlx-community--gemma-4-E4B-it-4bit/snapshots/deb1db712068b1c9f83fb1c97f08c1204b9459a1')


def visual_profile(path):
    """Give the controller a reproducible palette hint alongside model analysis."""
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        image.thumbnail((160, 160))
        colors = image.quantize(colors=12).convert('RGB').getcolors(image.width * image.height) or []
    colors.sort(reverse=True)
    dominant = colors[0][1] if colors else (245, 245, 245)
    candidates = []
    for count, rgb in colors:
        hue, saturation, value = rgb_to_hsv(*(channel / 255 for channel in rgb))
        if saturation > 0.22 and 0.18 < value < 0.96:
            candidates.append((count * saturation, rgb, hue))
    _, accent, hue = max(candidates, default=(0, (39, 117, 197), 0.6))
    name = ('red' if hue < 0.05 or hue >= 0.96 else
            'orange' if hue < 0.11 else 'gold' if hue < 0.17 else
            'green' if hue < 0.46 else 'cyan' if hue < 0.54 else
            'blue' if hue < 0.72 else 'purple')
    luminance = sum(channel * weight for channel, weight in zip(dominant, (0.2126, 0.7152, 0.0722)))
    tone = 'light' if luminance >= 130 else 'dark'
    color = lambda rgb: '#%02X%02X%02X' % rgb
    return (f'VISUAL_PROFILE:\nDimensions: {width}x{height}; dominant {color(dominant)}; '
            f'accent {color(accent)}.\nSuggested THEME: {tone} {name}. '
            'Use the accent for emphasis, maintain text contrast, and preserve any chart labels and scales.')


def main():
    from mlx_vlm import apply_chat_template, generate, load
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--question', default='Describe visible evidence and give a presentation design insight.')
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    model, processor = load(str(MODEL), adapter_path=str(args.adapter) if args.adapter else None)
    request = (
        'Analyze only what is visible. Separate observations from interpretation. '
        'For charts, read labels and values carefully, preserve units, and state when a trend or cause '
        'cannot be inferred. For photos or slide screenshots, describe subject, composition, hierarchy, '
        'legibility, contrast, and audience fit. Give one practical creative direction and one alternative. '
        'Do not invent text, sources, or visual details. User question: ' + args.question
    )
    prompt = apply_chat_template(processor, model.config, request, num_images=1)
    response = generate(model, processor, prompt, image=str(args.image),
                        max_tokens=350, temperature=0, verbose=False).text.strip()
    print(f'IMAGE_ANALYSIS:\nPath: `{args.image}`\n{response}\n\n{visual_profile(args.image)}')


if __name__ == '__main__':
    main()
