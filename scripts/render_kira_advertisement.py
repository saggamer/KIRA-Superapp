#!/usr/bin/env python3
"""Render a narrated, one-minute KIRA OS product demo."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice import _stable_espeak_config


DEMO_DIR = ROOT / "demo"
SCRIPT_PATH = DEMO_DIR / "KIRA_OS_Product_Demo_Script.md"
OUTPUT = DEMO_DIR / "KIRA_OS_Product_Demo.mp4"
POSTER = DEMO_DIR / "KIRA_OS_Product_Demo_Poster.png"
NARRATION_PATH = DEMO_DIR / "KIRA_OS_Product_Demo_Narration.wav"
MUSIC_PATH = DEMO_DIR / "KIRA_OS_Product_Demo_Ambient.wav"
SILENT_VIDEO = DEMO_DIR / ".kira_product_demo_silent.mp4"
UI_CAPTURE = ROOT / "stress_lab" / "ui_stress_screenshot_clean.png"
LOGO = ROOT / "assets" / "kira_logo_sample4.png"
VOICE_MODEL = Path(os.environ.get("KIRA_KOKORO_MODEL", ROOT / "models" / "kokoro-v1.0.onnx"))
VOICE_STYLES = Path(os.environ.get("KIRA_KOKORO_VOICES", ROOT / "models" / "voices-v1.0.bin"))

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION = 60

BG = "#070809"
PANEL = "#121416"
PANEL_2 = "#1a1d20"
TEXT = "#f4f5f6"
MUTED = "#8f969f"
LINE = "#363b42"
WHITE = "#ffffff"
GREEN = "#69d89e"
CYAN = "#80d9e8"
VIOLET = "#b8a1e6"
AMBER = "#e8b86b"
RED = "#ed7b73"

REGULAR = "/System/Library/Fonts/SFNS.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, bold: bool = False, mono: bool = False):
    return ImageFont.truetype(MONO if mono else (BOLD if bold else REGULAR), size)


F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F22 = font(22, bold=True)
F28 = font(28, bold=True)
F38 = font(38, bold=True)
F52 = font(52, bold=True)
FM13 = font(13, mono=True)
FM16 = font(16, mono=True)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def progress(t: float, start: float, end: float) -> float:
    value = clamp((t - start) / max(0.001, end - start))
    return 1.0 - (1.0 - value) ** 3


def rounded(draw, box, radius=12, fill=PANEL, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap(draw, text: str, max_width: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, text, max_width, fnt, fill=TEXT, line_gap=7):
    x, y = xy
    for line in wrap(draw, text, max_width, fnt):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def add_header(img: Image.Image, scene: str):
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH, 42), fill="#151619")
    for x, color in ((18, "#ff5f57"), (39, "#febc2e"), (60, "#28c840")):
        draw.ellipse((x, 14, x + 13, 27), fill=color)
    draw.text((88, 12), "KIRA OS", font=F14, fill="#d8dadd")
    draw.text((WIDTH - 240, 12), "VERIFIED PRODUCT FILM", font=F12, fill=MUTED)
    draw.text((32, 58), scene.upper(), font=F12, fill=MUTED)


def actual_ui_base(ui: Image.Image, dim=0.0, zoom=1.0) -> Image.Image:
    source = ui
    if zoom != 1.0:
        width = int(source.width / zoom)
        height = int(source.height / zoom)
        left = max(0, (source.width - width) // 2)
        top = max(0, (source.height - height) // 2)
        source = source.crop((left, top, left + width, top + height))
    image = source.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.rectangle((360, 96, 900, 188), fill="#080909")
    draw.text((400, 123), "Hello, apply your name.", font=F38, fill=TEXT)
    draw.text((855, 134), "edit", font=F14, fill=MUTED)
    if dim:
        shade = Image.new("RGBA", image.size, (0, 0, 0, int(255 * clamp(dim))))
        image.alpha_composite(shade)
    return image


def title_scene(t: float, ui: Image.Image, logo: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.55 - 0.15 * progress(t, 0, 5), zoom=1.04)
    glow = logo.resize((660, 360), Image.Resampling.LANCZOS)
    glow = ImageEnhance.Brightness(glow).enhance(0.6)
    glow = glow.filter(ImageFilter.GaussianBlur(0.5))
    image.alpha_composite(glow, (WIDTH - 705, 110))
    draw = ImageDraw.Draw(image)
    draw.text((66, 112), "KIRA OS", font=F52, fill=WHITE)
    draw.text((70, 180), "Local agentic work,", font=F28, fill=TEXT)
    draw.text((70, 217), "grounded in evidence.", font=F28, fill=TEXT)
    rounded(draw, (70, 303, 400, 352), radius=10, fill="#f1f2f4")
    draw.text((96, 318), "ORCHESTRATOR V1 INSIDE", font=F14, fill="#121315")
    draw.text((70, 636), "Edited from the real KIRA OS interface and test harness", font=F14, fill="#c4c8ce")
    add_header(image, "Local agentic environment")
    return image


def goal_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.2, zoom=1.0)
    draw = ImageDraw.Draw(image)
    rounded(draw, (365, 184, 1188, 364), radius=16, fill="#131518", outline=LINE)
    draw.text((395, 210), "GOAL", font=F12, fill=CYAN)
    prompt = (
        "Research local robotics trends, collect useful visuals, and build a "
        "source-backed presentation. Open it only after verification."
    )
    typed = prompt[: int(len(prompt) * progress(t, 10, 14.1))]
    draw_wrapped(draw, (395, 244), typed, 730, F22, fill=TEXT, line_gap=8)
    rounded(draw, (395, 322, 1138, 328), radius=3, fill="#2a2e34")
    rounded(draw, (395, 322, 395 + int(743 * progress(t, 13, 15)), 328), radius=3, fill=CYAN)
    draw.text((395, 347), "Orchestrator is choosing the work...", font=F14, fill=MUTED)
    add_header(image, "Natural-language goal")
    return image


def personalization_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.24)
    draw = ImageDraw.Draw(image)
    rounded(draw, (408, 150, 908, 458), radius=16, fill="#151719", outline=LINE)
    draw.text((442, 184), "YOUR KIRA GREETING", font=F12, fill=CYAN)
    draw.text((442, 220), "Make the workspace yours.", font=F28, fill=TEXT)
    draw.text((442, 270), "Display name", font=F14, fill=MUTED)
    rounded(draw, (442, 302, 874, 354), radius=8, fill="#0b0c0e", outline="#565c64")
    name = "Alex"
    typed = name[: int(len(name) * progress(t, 6.4, 8.3))]
    draw.text((462, 317), typed, font=F18, fill=TEXT)
    rounded(draw, (734, 382, 874, 426), radius=8, fill="#f0f1f3")
    draw.text((780, 395), "Apply", font=F14, fill="#111214")
    if t >= 8.5:
        draw.text((442, 391), "Saved locally", font=F14, fill=GREEN)
    draw.text((408, 508), "Editable at any time. Stored on this Mac.", font=F16, fill=MUTED)
    add_header(image, "Local personalization")
    return image


def node(draw, x, y, title, detail, color, active=1.0):
    rounded(draw, (x, y, x + 224, y + 72), radius=11, fill=PANEL_2, outline=color if active > 0.2 else LINE, width=2)
    draw.ellipse((x + 15, y + 25, x + 33, y + 43), fill=color)
    draw.text((x + 45, y + 15), title, font=F16, fill=TEXT)
    draw.text((x + 45, y + 41), detail, font=F12, fill=GREEN if active >= 1 else MUTED)


def tree_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.78)
    draw = ImageDraw.Draw(image)
    draw.text((78, 105), "Orchestrator selects. The Tree connects.", font=F38, fill=TEXT)
    draw.text((80, 155), "Only the branches required by this goal become active.", font=F18, fill=MUTED)

    root = (528, 224, 752, 296)
    rounded(draw, root, radius=12, fill="#202126", outline=AMBER, width=2)
    draw.text((566, 242), "ORCHESTRATOR V1", font=F18, fill=TEXT)
    draw.text((604, 269), "delegating", font=F12, fill=AMBER)

    branches = [
        (100, 398, "Research Agent", "5 direct sources", CYAN, progress(t, 15, 21)),
        (376, 398, "Media Agent", "5 workspace visuals", VIOLET, progress(t, 15, 21)),
        (652, 398, "Slides Agent", "7 substantive slides", AMBER, progress(t, 18, 23)),
        (928, 398, "Verifier", "quality gate passed", GREEN, progress(t, 21, 25)),
    ]
    centers = []
    for x, y, title, detail, color, active in branches:
        centers.append((x + 112, y))
        alpha = int(80 + 175 * active)
        draw.line((640, 296, x + 112, y), fill=(*ImageColor.getrgb(color), alpha), width=3)
        node(draw, x, y, title, detail, color, active)

    rounded(draw, (100, 548, 1152, 562), radius=7, fill="#272b31")
    rounded(draw, (100, 548, 100 + int(1052 * progress(t, 15, 25)), 562), radius=7, fill=GREEN)
    draw.text((100, 578), "Execution progression", font=F12, fill=MUTED)
    draw.text((1030, 578), f"{int(progress(t, 15, 25) * 100)}%", font=FM13, fill=TEXT)
    add_header(image, "Parallel specialist execution")
    return image


def evidence_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.82)
    draw = ImageDraw.Draw(image)
    draw.text((74, 98), "Evidence in. Verified work out.", font=F38, fill=TEXT)
    cards = [
        ("WEB RESEARCH", "5 pages fetched", "Readable source text retained", CYAN),
        ("MEDIA", "5 images embedded", "Local asset copies recorded", VIOLET),
        ("PPTX", "7 slides / direct links", "Round-trip package verified", AMBER),
        ("DOCX", "Body + headings + links", "Round-trip package verified", GREEN),
    ]
    for index, (label, value, detail, color) in enumerate(cards):
        x = 74 + (index % 2) * 570
        y = 184 + (index // 2) * 182
        rounded(draw, (x, y, x + 520, y + 145), radius=12, fill=PANEL, outline=LINE)
        draw.text((x + 24, y + 20), label, font=F12, fill=color)
        draw.text((x + 24, y + 51), value, font=F22, fill=TEXT)
        draw.text((x + 24, y + 91), detail, font=F14, fill=MUTED)
        if progress(t, 25 + index * 1.0, 28 + index * 1.0) >= 1:
            draw.text((x + 424, y + 20), "PASS", font=F12, fill=GREEN)
    rounded(draw, (74, 586, 1192, 636), radius=11, fill="#e8eaed")
    draw.text((100, 601), "KIRA reports completion only after the verifier can reopen the artifact.", font=F16, fill="#101113")
    add_header(image, "Artifact branch-in")
    return image


def workspace_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.84)
    draw = ImageDraw.Draw(image)
    draw.text((72, 98), "One workspace. More than slides.", font=F38, fill=TEXT)
    items = [
        ("PPTX", "Visual presentations", AMBER),
        ("DOCX", "Structured reports", CYAN),
        ("PDF", "Portable deliverables", VIOLET),
        ("FILES", "User attachments", GREEN),
    ]
    for index, (kind, label, color) in enumerate(items):
        x = 72 + index * 286
        rounded(draw, (x, 196, x + 250, 382), radius=12, fill=PANEL, outline=LINE)
        draw.text((x + 22, 220), kind, font=F12, fill=color)
        draw.text((x + 22, 260), label, font=F18, fill=TEXT)
        draw.rectangle((x + 22, 315, x + 210, 321), fill="#2d3138")
        draw.rectangle((x + 22, 315, x + 22 + int(188 * progress(t, 34 + index * 0.6, 38 + index * 0.6)), 321), fill=color)
        draw.text((x + 22, 340), "Indexed in KIRA Menu", font=F12, fill=MUTED)
    draw.text((74, 480), "Attachments become scoped evidence.", font=F22, fill=TEXT)
    draw.text((74, 518), "Artifacts stay discoverable by chat and type.", font=F22, fill=TEXT)
    add_header(image, "Artifact workspace")
    return image


def coding_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.86)
    draw = ImageDraw.Draw(image)
    draw.text((72, 94), "Vibe Coding, with engineering controls.", font=F38, fill=TEXT)
    steps = [
        ("1", "Choose installed IDE", "Windsurf", CYAN),
        ("2", "Scope project files", "secrets excluded", VIOLET),
        ("3", "Apply transaction", "rollback ready", AMBER),
        ("4", "Run verification", "tests passed", GREEN),
    ]
    for index, (number, title, detail, color) in enumerate(steps):
        y = 176 + index * 90
        active = progress(t, 41 + index * 1.4, 43 + index * 1.4)
        draw.ellipse((78, y, 120, y + 42), fill=color)
        draw.text((94, y + 10), number, font=F16, fill="#101113")
        draw.text((144, y - 1), title, font=F18, fill=TEXT)
        draw.text((144, y + 25), detail, font=F14, fill=GREEN if active >= 1 else MUTED)
        if index < 3:
            draw.line((99, y + 42, 99, y + 88), fill=LINE, width=3)
    rounded(draw, (680, 190, 1182, 492), radius=14, fill="#0b0d0f", outline=LINE)
    draw.text((708, 218), "API SPEND CONTROL", font=F12, fill=MUTED)
    draw.text((708, 258), "$0.18 / $2.00 task cap", font=F22, fill=TEXT)
    rounded(draw, (708, 309, 1148, 323), radius=7, fill="#272b31")
    rounded(draw, (708, 309, 748, 323), radius=7, fill=GREEN)
    draw.text((708, 349), "Policy: finish current call", font=F14, fill=MUTED)
    draw.text((708, 383), "Projected overage: blocked", font=F14, fill=GREEN)
    draw.text((708, 435), "Secrets stay outside project context.", font=F14, fill=CYAN)
    add_header(image, "Vibe Coding")
    return image


def safety_scene(t: float, ui: Image.Image) -> Image.Image:
    image = actual_ui_base(ui, dim=0.86)
    draw = ImageDraw.Draw(image)
    draw.text((72, 96), "Direct when safe. Permission when it matters.", font=F38, fill=TEXT)
    rounded(draw, (72, 190, 592, 477), radius=14, fill=PANEL, outline=GREEN)
    draw.text((102, 220), "READ-ONLY MAC INSPECTION", font=F12, fill=GREEN)
    rows = [
        ("Applications", "mapped"),
        ("Downloads", "scanned"),
        ("Memory", "reported"),
        ("Storage", "reported"),
    ]
    for index, (label, state) in enumerate(rows):
        y = 270 + index * 46
        draw.text((102, y), label, font=F16, fill=TEXT)
        draw.text((466, y), state, font=F14, fill=GREEN)
    rounded(draw, (640, 190, 1160, 477), radius=14, fill=PANEL, outline=AMBER)
    draw.text((670, 220), "RISKY ACTION", font=F12, fill=AMBER)
    draw.text((670, 266), "Delete selected installer?", font=F22, fill=TEXT)
    draw_wrapped(draw, (670, 308), "The exact target and expected effect are shown before execution.", 430, F14, MUTED)
    rounded(draw, (670, 400, 826, 444), radius=9, fill="#eceef0")
    draw.text((718, 413), "Approve", font=F14, fill="#111214")
    rounded(draw, (842, 400, 982, 444), radius=9, fill="#24282d")
    draw.text((890, 413), "Reject", font=F14, fill=TEXT)
    add_header(image, "macOS control")
    return image


def end_scene(t: float, ui: Image.Image, logo: Image.Image) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    logo_crop = logo.resize((500, 273), Image.Resampling.LANCZOS)
    image.alpha_composite(logo_crop, (700, 58))
    draw.text((70, 104), "50", font=F52, fill=GREEN)
    draw.text((70, 167), "automated app tests passed", font=F22, fill=TEXT)
    draw.text((70, 230), "442 ms", font=F52, fill=CYAN)
    draw.text((70, 293), "simulated UI shell boot", font=F22, fill=TEXT)
    draw.text((70, 366), "SAFE LOCAL HARNESS", font=F14, fill=MUTED)
    rounded(draw, (70, 404, 410, 454), radius=10, fill=PANEL_2, outline=GREEN)
    draw.text((98, 419), "ALL CHECKS PASSED", font=F16, fill=GREEN)
    draw.text((70, 532), "KIRA OS", font=F52, fill=TEXT)
    draw.text((74, 597), "Local intelligence that can finish the work.", font=F22, fill="#d6d9dd")
    draw.text((74, 653), "DIAGONAL AGENTIC STUDIO · PROTOTYPE BUILD", font=F12, fill=MUTED)
    add_header(image, "Verified current build")
    return image


def frame_at(t: float, ui: Image.Image, logo: Image.Image) -> Image.Image:
    if t < 5:
        return title_scene(t, ui, logo)
    if t < 10:
        return personalization_scene(t, ui)
    if t < 15:
        return goal_scene(t, ui)
    if t < 25:
        return tree_scene(t, ui)
    if t < 34:
        return evidence_scene(t, ui)
    if t < 41:
        return workspace_scene(t, ui)
    if t < 49:
        return coding_scene(t, ui)
    if t < 56:
        return safety_scene(t, ui)
    return end_scene(t, ui, logo)


def narration_text() -> str:
    markdown = SCRIPT_PATH.read_text(encoding="utf-8")
    match = re.search(r"## Narration\s+(.*?)(?:\n## |\Z)", markdown, re.DOTALL)
    if not match:
        raise RuntimeError("Narration section is missing.")
    return re.sub(r"\s+", " ", match.group(1)).strip()


def write_wav(path: Path, samples: np.ndarray, sample_rate: int):
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = max(0.001, float(np.max(np.abs(audio))))
    pcm = np.asarray(np.clip(audio / peak * 0.93, -1.0, 1.0) * 32767, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def synthesize_narration():
    if not VOICE_MODEL.exists() or not VOICE_STYLES.exists():
        raise FileNotFoundError("Kokoro voice assets are missing.")
    session = ort.InferenceSession(str(VOICE_MODEL), providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(
        session,
        str(VOICE_STYLES),
        espeak_config=_stable_espeak_config(),
    )
    samples, sample_rate = kokoro.create(
        narration_text(),
        voice="af_sarah",
        speed=1.08,
        lang="en-us",
    )
    write_wav(NARRATION_PATH, samples, sample_rate)


def synthesize_music():
    sample_rate = 24000
    count = sample_rate * DURATION
    t = np.arange(count, dtype=np.float32) / sample_rate
    frequencies = (55.0, 82.41, 110.0, 164.81)
    bed = sum(np.sin(2 * math.pi * frequency * t) / (index + 2) for index, frequency in enumerate(frequencies))
    pulse = 0.68 + 0.32 * np.sin(2 * math.pi * 0.083 * t)
    shimmer = 0.08 * np.sin(2 * math.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.05 * t))
    fade = np.minimum(1.0, t / 2.5) * np.minimum(1.0, (DURATION - t) / 3.0)
    write_wav(MUSIC_PATH, (bed * pulse * 0.14 + shimmer) * fade, sample_rate)


def render_silent_video():
    ui = Image.open(UI_CAPTURE).convert("RGBA")
    logo = Image.open(LOGO).convert("RGBA")
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(SILENT_VIDEO),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable.")
    for index in range(FPS * DURATION):
        frame = frame_at(index / FPS, ui, logo).convert("RGBA")
        process.stdin.write(frame.tobytes())
        if index == 57 * FPS:
            frame.convert("RGB").save(POSTER, quality=95)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Silent video render failed.")


def mix_final_video():
    command = [
        "ffmpeg", "-y",
        "-i", str(SILENT_VIDEO),
        "-i", str(NARRATION_PATH),
        "-i", str(MUSIC_PATH),
        "-filter_complex",
        (
            "[1:a]adelay=900,apad=pad_dur=60,volume=1.0[voice];"
            "[2:a]volume=0.11[music];"
            "[voice][music]amix=inputs=2:duration=longest:normalize=0,"
            "alimiter=limit=0.95[aout]"
        ),
        "-map", "0:v:0", "-map", "[aout]",
        "-t", str(DURATION),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(OUTPUT),
    ]
    subprocess.run(command, check=True)


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    synthesize_narration()
    synthesize_music()
    render_silent_video()
    mix_final_video()
    print(OUTPUT)
    print(POSTER)
    print(NARRATION_PATH)


if __name__ == "__main__":
    main()
