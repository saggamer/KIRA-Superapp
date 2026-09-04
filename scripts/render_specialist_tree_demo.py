#!/usr/bin/env python3
"""Render a short, honest product demo of KIRA OS specialist orchestration."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"
OUTPUT = DEMO_DIR / "KIRA_OS_Specialist_TREE_Demo.mp4"
POSTER = DEMO_DIR / "KIRA_OS_Specialist_TREE_Demo_Poster.png"
LOGO = ROOT / "assets" / "kira_logo_sample4.png"

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION = 18

BG = "#08090b"
PANEL = "#111318"
PANEL_2 = "#171a20"
TEXT = "#f5f7fa"
MUTED = "#9299a6"
LINE = "#343a45"
CYAN = "#7ee8fa"
VIOLET = "#b792ff"
GREEN = "#61d69b"
AMBER = "#f1bd68"

REGULAR = "/System/Library/Fonts/SFNS.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = MONO if mono else (BOLD if bold else REGULAR)
    return ImageFont.truetype(path, size)


F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F22 = font(22, bold=True)
F28 = font(28, bold=True)
F42 = font(42, bold=True)
F54 = font(54, bold=True)
FM14 = font(14, mono=True)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    value = clamp(value)
    return 1 - (1 - value) ** 3


def local_progress(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def rounded(draw: ImageDraw.ImageDraw, box, radius=14, fill=PANEL, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def line_wrap(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt) -> list[str]:
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


def draw_wrapped(draw, pos, text, max_width, fnt, fill=TEXT, spacing=7):
    x, y = pos
    for line in line_wrap(draw, text, max_width, fnt):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


def glow_line(base: Image.Image, points, color, width=3, intensity=1.0):
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(points, fill=color, width=width + 10, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(10))
    if intensity < 1:
        glow.putalpha(glow.getchannel("A").point(lambda a: int(a * intensity)))
    base.alpha_composite(glow)
    ImageDraw.Draw(base).line(points, fill=color, width=width, joint="curve")


def base_ui() -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, WIDTH, 48), fill="#171611")
    for x, color in ((20, "#ff5f57"), (42, "#febc2e"), (64, "#28c840")):
        draw.ellipse((x, 17, x + 14, 31), fill=color)
    draw.text((93, 15), "KIRA OS", font=F14, fill="#c8cbd1")
    draw.text((WIDTH - 174, 16), "SCRIPTED PRODUCT DEMO", font=F12, fill=MUTED)

    draw.rectangle((0, 48, 292, HEIGHT), fill="#111214")
    draw.line((292, 48, 292, HEIGHT), fill="#252932", width=1)
    draw.text((30, 85), "KIRA", font=F28, fill=TEXT)
    draw.text((30, 120), "LOCAL AGENTIC ENVIRONMENT", font=F12, fill=MUTED)

    rounded(draw, (24, 174, 268, 222), radius=12, fill="#f4f5f7")
    draw.text((88, 187), "+  New task", font=F16, fill="#121317")
    draw.text((30, 258), "THE TREE", font=F12, fill=MUTED)
    for i, (name, color) in enumerate(
        (("Agentic workspace", CYAN), ("Specialist agents", VIOLET), ("Artifact verification", GREEN))
    ):
        y = 292 + i * 48
        draw.ellipse((30, y + 6, 41, y + 17), fill=color)
        draw.text((55, y), name, font=F16, fill="#d8dce4")

    rounded(draw, (24, HEIGHT - 75, 268, HEIGHT - 25), radius=10, fill="#0b0d10")
    draw.ellipse((40, HEIGHT - 56, 50, HEIGHT - 46), fill=GREEN)
    draw.text((61, HEIGHT - 61), "Specialists ready", font=F14, fill="#cbd1da")

    draw.text((332, 76), "Orchestrator V1", font=F22, fill=TEXT)
    draw.text((332, 108), "THE TREE / SPECIALIST BRANCH-IN", font=F12, fill=MUTED)
    return img


def intro_frame(t: float, logo: Image.Image) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    p = local_progress(t, 0, 1.2)
    scale = 1.06 - 0.06 * p
    w = int(WIDTH * scale)
    h = int(HEIGHT * scale)
    splash = logo.resize((w, h), Image.Resampling.LANCZOS)
    splash = ImageEnhance.Brightness(splash).enhance(0.65 + 0.35 * p)
    img.alpha_composite(splash, ((WIDTH - w) // 2, (HEIGHT - h) // 2))
    shade = Image.new("RGBA", img.size, (0, 0, 0, int(75 * p)))
    img.alpha_composite(shade)
    draw = ImageDraw.Draw(img)
    draw.text((60, 54), "KIRA OS", font=F54, fill=TEXT)
    draw.text((64, 119), "Specialists converge. Evidence becomes verified work.", font=F18, fill="#d4d9e3")
    draw.text((64, HEIGHT - 48), "A scripted demonstration of the implemented execution flow", font=F14, fill=MUTED)
    return img


def prompt_frame(t: float) -> Image.Image:
    img = base_ui()
    draw = ImageDraw.Draw(img)
    p = local_progress(t, 2.0, 3.4)
    y = 158
    draw.ellipse((338, y, 372, y + 34), fill="#252a32")
    draw.text((349, y + 6), "U", font=F16, fill=TEXT)
    draw.text((390, y - 1), "USER", font=F12, fill=MUTED)
    prompt = (
        "Research local robotics trends, find relevant visuals, and create a "
        "source-backed presentation with direct links."
    )
    chars = int(len(prompt) * p)
    draw_wrapped(draw, (390, y + 24), prompt[:chars], 760, F18, TEXT, 7)
    if t > 3.4:
        rounded(draw, (390, 262, 1110, 307), radius=10, fill=PANEL_2, outline=LINE)
        draw.text((412, 275), "Orchestrator is selecting branches...", font=F16, fill=CYAN)
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        draw.ellipse((1072, 276, 1088, 292), fill=(126, 232, 250, int(90 + 165 * pulse)))
    return img


def node(draw, center, title, subtitle, color, active=1.0, done=False):
    x, y = center
    box = (x - 106, y - 39, x + 106, y + 39)
    fill = "#1a1d24" if active > 0.1 else "#111318"
    outline = color if active > 0.45 else LINE
    rounded(draw, box, radius=12, fill=fill, outline=outline, width=2)
    draw.ellipse((x - 90, y - 12, x - 66, y + 12), fill=color if active > 0.45 else "#303641")
    draw.text((x - 56, y - 22), title, font=F16, fill=TEXT)
    draw.text((x - 56, y + 4), subtitle, font=F12, fill=GREEN if done else MUTED)


def tree_frame(t: float) -> Image.Image:
    img = base_ui()
    draw = ImageDraw.Draw(img)
    draw.text((350, 146), "Parallel discovery", font=F28, fill=TEXT)
    draw.text((350, 183), "Independent branches gather evidence at the same time.", font=F16, fill=MUTED)

    root = (792, 254)
    research = (520, 370)
    media = (1062, 370)
    slides = (792, 505)
    verifier = (792, 620)

    research_p = local_progress(t, 4.8, 7.3)
    media_p = local_progress(t, 4.8, 7.0)
    build_p = local_progress(t, 7.3, 9.0)
    verify_p = local_progress(t, 8.8, 10.1)

    glow_line(img, (root, research), CYAN, intensity=research_p)
    glow_line(img, (root, media), VIOLET, intensity=media_p)
    if build_p > 0:
        glow_line(img, (research, slides), CYAN, intensity=build_p)
        glow_line(img, (media, slides), VIOLET, intensity=build_p)
    if verify_p > 0:
        glow_line(img, (slides, verifier), GREEN, intensity=verify_p)

    draw = ImageDraw.Draw(img)
    node(draw, root, "Orchestrator", "delegating", AMBER, 1.0)
    node(draw, research, "Research Agent", f"{int(research_p * 5)}/5 sources", CYAN, research_p, research_p >= 1)
    node(draw, media, "Media Agent", f"{int(media_p * 4)}/4 visuals", VIOLET, media_p, media_p >= 1)
    node(draw, slides, "Slides Agent", f"{int(build_p * 7)}/7 slides", AMBER, build_p, build_p >= 1)
    node(draw, verifier, "Artifact Verifier", "quality gate passed" if verify_p >= 1 else "checking", GREEN, verify_p, verify_p >= 1)

    progress = clamp((t - 4.6) / 5.6)
    rounded(draw, (350, 660, 1160, 674), radius=7, fill="#242933")
    rounded(draw, (350, 660, 350 + int(810 * progress), 674), radius=7, fill=GREEN)
    return img


def artifact_frame(t: float) -> Image.Image:
    img = base_ui()
    draw = ImageDraw.Draw(img)
    p = local_progress(t, 10.2, 11.2)
    draw.text((350, 146), "Evidence branches in. A specialist builds.", font=F28, fill=TEXT)
    draw.text((350, 184), "The final answer is held until the artifact passes verification.", font=F16, fill=MUTED)

    card = (350, 232, 1160, 610)
    rounded(draw, card, radius=16, fill=PANEL, outline=LINE)
    draw.text((382, 264), "LOCAL ROBOTICS / 2026", font=F12, fill=CYAN)
    draw.text((382, 292), "From sources to a verified deck", font=F28, fill=TEXT)

    for i, (label, value, color) in enumerate(
        (
            ("Sources", "5 direct links", CYAN),
            ("Visuals", "4 embedded", VIOLET),
            ("Slides", "7 substantive", AMBER),
            ("Verification", "PASSED", GREEN),
        )
    ):
        x = 382 + (i % 2) * 360
        y = 365 + (i // 2) * 92
        rounded(draw, (x, y, x + 320, y + 68), radius=10, fill=PANEL_2, outline=LINE)
        draw.text((x + 18, y + 12), label, font=F12, fill=MUTED)
        draw.text((x + 18, y + 32), value, font=F18, fill=color)

    if p > 0.35:
        rounded(draw, (892, 536, 1126, 582), radius=10, fill="#e9edf2")
        draw.text((940, 549), "Open deck", font=F16, fill="#111318")
    return img


def finish_frame(t: float) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.text((90, 122), "THE TREE", font=F14, fill=CYAN)
    draw.text((90, 158), "A better artifact pipeline,", font=F42, fill=TEXT)
    draw.text((90, 213), "because every branch has a job.", font=F42, fill=TEXT)
    draw.text((92, 306), "Research + media run in parallel.", font=F18, fill="#d4dae4")
    draw.text((92, 344), "Slides and documents consume evidence.", font=F18, fill="#d4dae4")
    draw.text((92, 382), "Verification blocks thin or ungrounded artifacts.", font=F18, fill="#d4dae4")
    rounded(draw, (90, 480, 550, 548), radius=14, fill=PANEL_2, outline=GREEN, width=2)
    draw.text((116, 497), "46 tests + full stress harness passed", font=F18, fill=GREEN)
    draw.text((92, 626), "KIRA OS / DIAGONAL AGENTIC STUDIO", font=F14, fill=MUTED)
    return img


def make_frame(t: float, logo: Image.Image) -> Image.Image:
    if t < 2.0:
        return intro_frame(t, logo)
    if t < 4.7:
        return prompt_frame(t)
    if t < 10.2:
        return tree_frame(t)
    if t < 14.1:
        return artifact_frame(t)
    return finish_frame(t)


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    logo = Image.open(LOGO).convert("RGBA")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    poster_written = False
    for index in range(FPS * DURATION):
        t = index / FPS
        frame = make_frame(t, logo).convert("RGBA")
        process.stdin.write(frame.tobytes())
        if not poster_written and t >= 9.7:
            frame.convert("RGB").save(POSTER, quality=95)
            poster_written = True
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed to render the demo")
    print(OUTPUT)
    print(POSTER)


if __name__ == "__main__":
    main()
