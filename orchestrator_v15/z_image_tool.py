"""Bounded local Z-Image Turbo Q4 generation for Kira artifact previews."""

import os
from pathlib import Path
import subprocess
import uuid


MODEL_DIR = Path('/Volumes/Extreme SSD/KIRA_SUPERAPP_MODELS_2026-10-10/z_image_turbo_4bit')


def _load_raster(path):
    """Decode before allocating diffusion state; reject broken or oversized inputs."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError('image exceeds 25 megapixels')
        image.load()
        return ImageOps.exif_transpose(image).copy()


def generate_image(prompt, workspace, app_root, *, seed=42, timeout=600):
    return _run_image_model(prompt, workspace, app_root, seed=seed, timeout=timeout)


def edit_image(prompt, source, workspace, app_root, *, strength=0.4, mask=None, region=None,
               seed=42, timeout=600):
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise ValueError('PATH must name a local PNG, JPEG, or WebP image')
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('source image exceeds 20 MiB')
    original = _load_raster(source)
    if mask:
        mask = Path(mask).expanduser().resolve()
        if not mask.is_file() or mask.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise ValueError('MASK must name a local raster image')
        if mask.stat().st_size > 20 * 1024 * 1024:
            raise ValueError('mask image exceeds 20 MiB')
    if region and mask:
        raise ValueError('Use either MASK or REGION, not both')
    if region:
        mask = _region_mask(source, workspace, str(region).lower().strip())
    if mask:
        selection = _load_raster(mask).convert('L')
        if selection.size != original.size:
            raise ValueError('MASK dimensions must match the source image')
        if selection.getbbox() is None:
            raise ValueError('MASK must select at least one pixel (white means edit)')
    strength = float(strength)
    if not 0.1 <= strength <= 0.85:
        raise ValueError('STRENGTH must be between 0.1 and 0.85')
    output = _run_image_model(prompt, workspace, app_root, seed=seed, timeout=timeout,
                              source=source, strength=strength, mask=mask)
    # Diffusion can change unselected pixels and resize the canvas. Restore the
    # exact source geometry and composite only the selected region locally.
    from PIL import Image
    edited = _load_raster(output).convert('RGB').resize(original.size, Image.Resampling.LANCZOS)
    if mask:
        edited = Image.composite(edited, original.convert('RGB'), selection)
    if 'A' in original.getbands() or 'transparency' in original.info:
        edited = edited.convert('RGBA')
        edited.putalpha(original.convert('RGBA').getchannel('A'))
    edited.save(output)
    return output


def _region_mask(source, workspace, region):
    """Create a soft local inpainting mask from a bounded spatial instruction."""
    from PIL import Image, ImageDraw, ImageFilter

    boxes = {
        'left': (0.02, 0.15, 0.43, 0.88),
        'right': (0.57, 0.15, 0.98, 0.88),
        'center': (0.25, 0.20, 0.75, 0.82),
        'upper_left': (0.02, 0.02, 0.48, 0.52),
        'upper_right': (0.52, 0.02, 0.98, 0.52),
        'lower_left': (0.02, 0.43, 0.48, 0.91),
        'lower_right': (0.68, 0.43, 0.98, 0.91),
    }
    if region not in boxes:
        raise ValueError('REGION must be left, right, center, upper_left, upper_right, lower_left, or lower_right')
    width, height = _load_raster(source).size
    area = boxes[region]
    output_dir = Path(workspace) / 'generated_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'z_image_mask_{uuid.uuid4().hex[:12]}.png'
    picture = Image.new('L', (width, height), 0)
    ImageDraw.Draw(picture).rectangle(tuple(round(value * size) for value, size in
                                            zip(area, (width, height, width, height))), fill=255)
    picture.filter(ImageFilter.GaussianBlur(max(4, round(min(width, height) * 0.015)))).save(path)
    return path


def _run_image_model(prompt, workspace, app_root, *, seed, timeout, source=None,
                     strength=0.4, mask=None):
    prompt = str(prompt or '').strip()
    if not prompt or len(prompt) > 600 or '\n' in prompt:
        raise ValueError('PROMPT must be one line of 1–600 characters')
    seed = int(seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError('SEED must be a 32-bit unsigned integer')
    default_dir = MODEL_DIR if MODEL_DIR.is_dir() else Path(app_root) / 'models' / 'z_image_turbo_4bit'
    model_dir = Path(os.environ.get('KIRA_Z_IMAGE_MODEL_DIR', str(default_dir)))
    if not (model_dir / 'transformer' / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError(f'Z-Image Turbo Q4 weights are unavailable at {model_dir}')
    executable = Path(app_root) / '.venv-z-image-turbo' / 'bin' / 'mlxgen'
    if not executable.is_file():
        raise FileNotFoundError(f'MLX-Gen runtime is unavailable at {executable}')
    output_dir = Path(workspace) / 'generated_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'z_image_edit' if source else 'z_image_turbo'
    output = output_dir / f'{prefix}_{uuid.uuid4().hex[:12]}.png'
    width, height = 1024, 768
    if source:
        source_width, source_height = _load_raster(source).size
        scale = min(1, 1024 / max(source_width, source_height))
        width, height = (max(64, round(size * scale / 16) * 16)
                         for size in (source_width, source_height))
    command = [str(executable), 'generate', '--model', str(model_dir),
               '--base-model', 'z-image-turbo', '--prompt', prompt,
               '--width', str(width), '--height', str(height), '--steps', '8', '--guidance', '0',
               '--seed', str(seed), '--output', str(output)]
    if source:
        command.extend(['--image', str(source)])
        if mask:
            command.extend(['--mask-path', str(mask)])
        else:
            command.extend(['--image-strength', str(strength)])
    env = os.environ.copy()
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    # Diffusion backends do not all honor EXIF orientation. Pass normalized
    # copies only when needed so the source, selection and working canvas agree.
    normalized_inputs = []
    from PIL import Image
    try:
        for flag in ('--image', '--mask-path'):
            if flag not in command:
                continue
            index = command.index(flag) + 1
            with Image.open(command[index]) as incoming:
                oriented = incoming.getexif().get(274, 1) != 1
            if oriented:
                normalized = output_dir / f'input_{uuid.uuid4().hex[:12]}.png'
                _load_raster(command[index]).save(normalized)
                normalized_inputs.append(normalized)
                command[index] = str(normalized)
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    finally:
        for normalized in normalized_inputs:
            normalized.unlink(missing_ok=True)
    if run.returncode:
        raise RuntimeError((run.stderr or run.stdout).strip()[-1600:])
    if not output.is_file():
        raise RuntimeError('MLX-Gen returned without a usable PNG')
    try:
        picture = _load_raster(output)
        if min(picture.size) < 64:
            raise ValueError('output is smaller than 64 pixels')
    except (OSError, ValueError) as error:
        output.unlink(missing_ok=True)
        raise RuntimeError('MLX-Gen returned an invalid image') from error
    return output
