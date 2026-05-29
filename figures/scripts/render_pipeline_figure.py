from __future__ import annotations

import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pipeline.png"
ASSETS = ROOT / "pipeline_assets"

W, H = 3200, 1600
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
BLACK = Path("/System/Library/Fonts/Supplemental/Arial Black.ttf")


def font(size: int, bold: bool = False, black: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BLACK if black else (BOLD if bold else FONT)), size)


F_TITLE = font(76, black=True)
F_SUB = font(38)
F_PANEL = font(46, bold=True)
F_BAND = font(43, bold=True)
F_LABEL = font(40, bold=True)
F_SMALL = font(32)
F_TINY = font(31)
F_CHIP = font(29, bold=True)

C_TEXT = "#171b25"
C_MUTED = "#66758d"
C_LINE = "#d7e0ea"
C_TEAL = "#159d8c"
C_TEAL_DARK = "#0d7f74"
C_RED = "#c74b4e"
C_ORANGE = "#e88412"
C_BLUE = "#2d95c4"
C_PURPLE = "#7b66ff"
C_GREEN_FILL = "#e9f7ef"
C_GREEN_STROKE = "#bde8cf"
C_CYAN_FILL = "#eaf9fb"
C_CYAN_STROKE = "#b9e8ee"
C_PURPLE_FILL = "#f0ecff"
C_ORANGE_FILL = "#fff5e8"

img = Image.new("RGB", (W, H), "#f8fafc")
d = ImageDraw.Draw(img)


def rounded(box, r=20, fill="white", outline=C_LINE, width=3):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def shadowed_panel(box, r=24, fill="white"):
    x0, y0, x1, y1 = box
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((x0 + 8, y0 + 12, x1 + 8, y1 + 12), radius=r, fill=(20, 35, 55, 23))
    layer = layer.filter(ImageFilter.GaussianBlur(18))
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))
    d.rounded_rectangle(box, radius=r, fill=fill, outline=C_LINE, width=3)


def paste_cover(src, box, radius=0, border=None, bw=0, bg="#ffffff"):
    x0, y0, x1, y1 = map(int, box)
    tw, th = x1 - x0, y1 - y0
    fitted = ImageOps.fit(src.convert("RGB"), (tw, th), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if radius:
        mask = Image.new("L", (tw, th), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, tw, th), radius=radius, fill=255)
        bgim = Image.new("RGB", (tw, th), bg)
        bgim.paste(fitted, (0, 0), mask)
        img.paste(bgim, (x0, y0))
    else:
        img.paste(fitted, (x0, y0))
    if border and bw:
        d.rounded_rectangle((x0, y0, x1, y1), radius=radius, outline=border, width=bw)


def text_center(xy, text, fnt, fill=C_TEXT):
    x, y = xy
    box = d.textbbox((0, 0), text, font=fnt)
    d.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def text_fit_center(xy, text, size, max_width, fill=C_TEXT, bold=False, min_size=17):
    chosen = size
    fnt = font(chosen, bold=bold)
    while chosen > min_size:
        box = d.textbbox((0, 0), text, font=fnt)
        if box[2] - box[0] <= max_width:
            break
        chosen -= 1
        fnt = font(chosen, bold=bold)
    text_center(xy, text, fnt, fill=fill)


def text_multiline_fit_center(xy, text, size, max_width, fill=C_TEXT, bold=False, min_size=17, line_gap=3):
    lines = [line for line in text.split("\n") if line]
    chosen = size
    fnt = font(chosen, bold=bold)
    while chosen > min_size:
        widths = [d.textbbox((0, 0), line, font=fnt)[2] for line in lines]
        if widths and max(widths) <= max_width:
            break
        chosen -= 1
        fnt = font(chosen, bold=bold)
    heights = [d.textbbox((0, 0), line, font=fnt)[3] - d.textbbox((0, 0), line, font=fnt)[1] for line in lines]
    line_h = max(heights) if heights else chosen
    total_h = line_h * len(lines) + line_gap * max(0, len(lines) - 1)
    x, y = xy
    start_y = y - total_h / 2 + line_h / 2
    for i, line in enumerate(lines):
        text_center((x, start_y + i * (line_h + line_gap)), line, fnt, fill=fill)


def arrow(points, color="#1c9b8e", width=9, head=26):
    pts = [(int(x), int(y)) for x, y in points]
    for a, b in zip(pts[:-1], pts[1:]):
        d.line([a, b], fill=color, width=width)
    if len(pts) < 2:
        return
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (x1, y1)
    p2 = (x1 - ux * head + px * head * 0.48, y1 - uy * head + py * head * 0.48)
    p3 = (x1 - ux * head - px * head * 0.48, y1 - uy * head - py * head * 0.48)
    d.polygon([p1, p2, p3], fill=color)


def node(box, title, sub=None, stroke=C_TEAL, fill="white", title_color=C_TEXT, title_size=35):
    rounded(box, r=16, fill=fill, outline=stroke, width=4)
    x0, y0, x1, _ = box
    text_center(((x0 + x1) / 2, y0 + 54), title, font(title_size, bold=True), fill=title_color)
    if sub:
        text_center(((x0 + x1) / 2, y0 + 100), sub, F_SMALL, fill=C_MUTED)


def chip(x, y, text, stroke, fill="white"):
    box = d.textbbox((0, 0), text, font=F_CHIP)
    w = box[2] - box[0] + 34
    h = box[3] - box[1] + 20
    d.rounded_rectangle((x, y, x + w, y + h), radius=11, fill=fill, outline=stroke, width=3)
    d.text((x + 17, y + 9), text, font=F_CHIP, fill=stroke)


def dropout_overlay(box, label="p_drop"):
    x0, y0, x1, y1 = map(int, box)
    w, h = x1 - x0, y1 - y0
    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    pd.rounded_rectangle((0, 0, w - 1, h - 1), radius=14, fill=(245, 248, 250, 158), outline=(90, 100, 115, 215), width=4)
    for k in range(-h, w + h, 28):
        pd.line((k, h, k + h, 0), fill=(85, 95, 110, 140), width=4)
    label_text = label
    label_size = min(28, max(17, w // 4))
    drop_font = font(label_size, bold=True)
    tb = pd.textbbox((0, 0), label_text, font=drop_font)
    while tb[2] - tb[0] + 30 > w - 10 and label_size > 16:
        label_size -= 1
        drop_font = font(label_size, bold=True)
        tb = pd.textbbox((0, 0), label_text, font=drop_font)
    if tb[2] - tb[0] + 30 > w - 10 and label_text == "p_drop":
        label_text = "drop"
        drop_font = font(max(16, label_size), bold=True)
        tb = pd.textbbox((0, 0), label_text, font=drop_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pd.rounded_rectangle((8, 8, min(w - 8, tw + 30), th + 26), radius=8, fill=(255, 255, 255, 235), outline=(95, 105, 118, 185), width=2)
    pd.text((18, 14), label_text, font=drop_font, fill=(65, 72, 86, 255))
    base = img.crop((x0, y0, x1, y1)).convert("RGBA")
    comp = Image.alpha_composite(base, patch)
    img.paste(comp.convert("RGB"), (x0, y0))


def labeled_tile(src, box, label, stroke, label_color=None, label_size=26, image_frac=0.72, overlay_drop=False):
    x0, y0, x1, y1 = map(int, box)
    rounded((x0, y0, x1, y1), r=14, fill="white", outline=stroke, width=4)
    pad = 11
    im_bottom = y0 + int((y1 - y0) * image_frac)
    paste_cover(src, (x0 + pad, y0 + pad, x1 - pad, im_bottom), radius=9)
    if overlay_drop:
        dropout_overlay((x0 + pad, y0 + pad, x1 - pad, im_bottom), "p_drop")
    if "\n" in label:
        text_multiline_fit_center(((x0 + x1) / 2, y1 - 48), label, label_size, (x1 - x0) - 18, fill=label_color or stroke, bold=True)
    else:
        text_fit_center(((x0 + x1) / 2, y1 - 22), label, label_size, (x1 - x0) - 18, fill=label_color or stroke, bold=True)


def pair_card(box, title, rgb_img, ray_img, stroke, dropped=False, title_size=24):
    x0, y0, x1, y1 = map(int, box)
    rounded((x0, y0, x1, y1), r=16, fill="#fbfdff", outline=stroke, width=4)
    text_fit_center(((x0 + x1) / 2, y0 + 30), title, title_size, (x1 - x0) - 24, fill=stroke, bold=True)
    gap = 14
    inner_top = y0 + 58
    inner_bottom = y1 - 16
    tile_w = int((x1 - x0 - 46 - gap) / 2)
    labeled_tile(rgb_img, (x0 + 16, inner_top, x0 + 16 + tile_w, inner_bottom), "RGB", stroke, label_size=22, overlay_drop=dropped)
    labeled_tile(ray_img, (x0 + 16 + tile_w + gap, inner_top, x0 + 16 + tile_w * 2 + gap, inner_bottom), "ray", stroke, label_size=22, overlay_drop=dropped)


def scene_token_glyph(box, label, stroke=C_TEAL, fill="#def7e3", label_size=22):
    x0, y0, x1, y1 = map(int, box)
    rounded((x0, y0, x1, y1), r=16, fill="white", outline=stroke, width=4)
    token_w, token_h = 38, 62
    start_x = x0 + (x1 - x0 - token_w) // 2
    top = y0 + 18
    d.rounded_rectangle((start_x, top, start_x + token_w, top + token_h), radius=7, fill=fill, outline="#65c878", width=3)
    for yy in (13, 31, 49):
        d.line((start_x + 8, top + yy, start_x + token_w - 8, top + yy), fill="#9ee0aa", width=2)
    if "\n" in label:
        text_multiline_fit_center(((x0 + x1) / 2, y1 - 36), label, label_size, (x1 - x0) - 18, fill=stroke, bold=True, line_gap=1)
    else:
        text_fit_center(((x0 + x1) / 2, y1 - 23), label, label_size, (x1 - x0) - 18, fill=stroke, bold=True)


def plucker(size=(330, 185), phase=0.0, variant="A"):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    if variant == "A":
        cx = w * (0.44 + 0.018 * math.sin(phase * 11.0))
        cy = h * (0.48 + 0.014 * math.cos(phase * 13.0))
        hue_base = 0.48
        ray_sat = 0.82
        ring_color = (255, 255, 255, 78)
    else:
        cx = w * (0.59 + 0.015 * math.cos(phase * 9.0))
        cy = h * (0.56 + 0.018 * math.sin(phase * 15.0))
        hue_base = 0.02
        ray_sat = 0.90
        ring_color = (255, 244, 232, 95)
    nx = (xx - cx) / max(1.0, w * 0.5)
    ny = (yy - cy) / max(1.0, h * 0.5)
    radius = np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 1.2)
    angle = np.arctan2(ny, nx)
    if variant == "A":
        hue = (angle / (2 * np.pi) * 0.55 + hue_base + phase * 0.18) % 1.0
        texture = 0.035 * np.cos(18.0 * radius - phase * 8.0)
    else:
        diagonal = (xx / max(1, w) * 0.75 + yy / max(1, h) * 0.25)
        hue = (hue_base + 0.14 * np.sin(angle * 2.0 + phase * 5.0) + 0.08 * diagonal) % 1.0
        texture = 0.045 * np.sin((xx + 1.35 * yy) * 0.055 + phase * 12.0)
    sat = np.clip(0.58 + 0.34 * radius, 0.0, 1.0)
    val = np.clip(0.98 - 0.16 * radius + texture, 0.0, 1.0)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = colorsys.hsv_to_rgb(float(hue[y, x]), float(sat[y, x]), float(val[y, x]))
            arr[y, x] = (int(r * 255), int(g * 255), int(b * 255))
    out = Image.fromarray(arr, "RGB")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(layer)

    # Perspective-style ray maps. A and B use intentionally different visual
    # structure so the paper figure does not read them as duplicated tiles.
    max_r = int(math.hypot(w, h))
    if variant == "A":
        for frac in (0.24, 0.48, 0.72, 0.96):
            rx = max_r * frac * 0.50
            ry = max_r * frac * 0.32
            od.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=ring_color, width=2)
    else:
        for offset in range(-h, w + h, max(22, w // 12)):
            od.line((offset, h + 8, offset + h * 1.35, -8), fill=ring_color, width=2)

    def edge_point(theta, margin=7):
        c, s = math.cos(theta), math.sin(theta)
        ts = []
        if abs(c) > 1e-5:
            ts.append(((w - margin - cx) / c) if c > 0 else ((margin - cx) / c))
        if abs(s) > 1e-5:
            ts.append(((h - margin - cy) / s) if s > 0 else ((margin - cy) / s))
        t = min(t for t in ts if t > 0)
        return cx + t * c, cy + t * s

    ray_count = 22 if variant == "A" else 16
    for i in range(ray_count):
        if variant == "A":
            theta = 2 * math.pi * i / ray_count + phase * 1.7
        else:
            theta = -0.85 + (1.70 * i / max(1, ray_count - 1)) + phase * 0.35
        ex, ey = edge_point(theta)
        rr, gg, bb = colorsys.hsv_to_rgb((theta / (2 * math.pi) + hue_base + phase * 0.23) % 1.0, ray_sat, 0.55)
        alpha = 178 if variant == "A" else 205
        col = (int(rr * 255), int(gg * 255), int(bb * 255), alpha)
        width = (3 if w >= 300 else 2) if variant == "A" else (4 if w >= 300 else 3)
        od.line((cx, cy, ex, ey), fill=col, width=width)
        ux, uy = math.cos(theta), math.sin(theta)
        px, py = -uy, ux
        head = max(8, min(w, h) * 0.055)
        tip = (ex, ey)
        p1 = (ex - ux * head + px * head * 0.44, ey - uy * head + py * head * 0.44)
        p2 = (ex - ux * head - px * head * 0.44, ey - uy * head - py * head * 0.44)
        od.polygon((tip, p1, p2), fill=col)

    center_outline = (23, 83, 92, 230) if variant == "A" else (128, 46, 42, 230)
    od.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=(255, 255, 255, 245), outline=center_outline, width=3)
    od.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=center_outline)
    return Image.alpha_composite(out.convert("RGBA"), layer).convert("RGB")


def main():
    teaser = Image.open(ROOT / "teaser.png").convert("RGB")
    view_a = Image.open(ASSETS / "inputA_direct.png").convert("RGB")
    view_b = Image.open(ASSETS / "inputB_direct.png").convert("RGB")
    scene_crop = teaser.crop((120, 95, 2050, 950))
    action_crop = teaser.crop((2200, 1375, 3380, 1810))
    ray_a = plucker((420, 235), 0.03, "A")
    ray_b = plucker((420, 235), 0.42, "B")

    d.text((58, 42), "End-to-end camera-pose policy variants", font=F_TITLE, fill=C_TEXT)
    d.text((62, 124), "Each model predicts camera pose inside the policy; action and NVS losses train the geometry module jointly with the encoder.", font=F_SUB, fill=C_MUTED)

    left = (46, 190, 820, 1536)
    right = (850, 190, 3154, 1536)
    shadowed_panel(left)
    shadowed_panel(right)
    d.text((88, 238), "Scene + input views", font=F_PANEL, fill=C_TEXT)
    d.text((902, 238), "Model variants", font=F_PANEL, fill=C_TEXT)

    rounded((88, 318, 778, 910), r=22, fill="#fbfdff", outline=C_LINE, width=3)
    paste_cover(scene_crop, (112, 346, 754, 882), radius=16)
    for box, label, col, im in [
        ((88, 1010, 420, 1510), "View A", C_TEAL, view_a),
        ((448, 1010, 778, 1510), "View B", C_RED, view_b),
    ]:
        rounded(box, r=20, fill="#fbfdff", outline=C_LINE, width=3)
        x0, y0, x1, y1 = box
        paste_cover(im, (x0 + 22, y0 + 28, x1 - 22, y1 - 102), radius=14, border=col, bw=6)
        text_fit_center(((x0 + x1) / 2, y1 - 54), label, 42, (x1 - x0) - 36, fill=col, bold=True)

    def rgb_views_card(box, title="RGB views"):
        x0, y0, x1, y1 = map(int, box)
        rounded((x0, y0, x1, y1), r=18, fill="white", outline=C_LINE, width=4)
        text_fit_center(((x0 + x1) / 2, y0 + 42), title, 39, x1 - x0 - 28, fill=C_TEXT, bold=True)
        tile_w = int((x1 - x0 - 78) / 2)
        labeled_tile(view_a, (x0 + 24, y0 + 86, x0 + 24 + tile_w, y1 - 24), "View A", C_TEAL, label_size=31, image_frac=0.66)
        labeled_tile(view_b, (x0 + 54 + tile_w, y0 + 86, x0 + 54 + tile_w * 2, y1 - 24), "View B", C_RED, label_size=31, image_frac=0.66)

    def ray_maps_card(box):
        x0, y0, x1, y1 = map(int, box)
        rounded((x0, y0, x1, y1), r=18, fill="white", outline=C_BLUE, width=4)
        text_fit_center(((x0 + x1) / 2, y0 + 42), "Predicted rays", 39, x1 - x0 - 28, fill=C_BLUE, bold=True)
        tile_w = int((x1 - x0 - 80) / 2)
        labeled_tile(ray_a, (x0 + 24, y0 + 88, x0 + 24 + tile_w, y1 - 24), "ray A", C_TEAL, label_size=31, image_frac=0.66)
        labeled_tile(ray_b, (x0 + 56 + tile_w, y0 + 88, x0 + 56 + tile_w * 2, y1 - 24), "ray B", C_RED, label_size=31, image_frac=0.66)

    def pose_block(box, aux_y=None):
        node(box, "Pose", "predictor", stroke=C_PURPLE, fill=C_PURPLE_FILL, title_color=C_TEXT, title_size=42)
        if aux_y is not None:
            x0, _, x1, _ = box
            rounded((x0 - 18, aux_y, x1 + 18, aux_y + 84), r=12, fill="white", outline=C_PURPLE, width=3)
            text_multiline_fit_center(((x0 + x1) / 2, aux_y + 43), "semi-supervised\npose loss", 31, (x1 - x0) + 10, fill=C_PURPLE, bold=True)

    band_a = (895, 320, 3110, 820)
    rounded(band_a, r=22, fill=C_GREEN_FILL, outline=C_GREEN_STROKE, width=3)
    d.text((940, 374), "A  E2E camera-pose action policy", font=F_BAND, fill=C_TEAL_DARK)
    rgb_views_card((940, 452, 1276, 748))
    pose_block((1350, 528, 1574, 678), aux_y=714)
    ray_maps_card((1650, 452, 2020, 748))
    node((2100, 528, 2374, 678), "Observation", "encoder", stroke="#35ad67", fill="white", title_size=39)
    node((2454, 528, 2684, 678), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=41)
    labeled_tile(action_crop, (2760, 444, 3026, 752), "action\nobjective", C_ORANGE, label_size=34, image_frac=0.66)
    arrow([(1276, 600), (1350, 600)], color=C_PURPLE, width=9)
    arrow([(1574, 600), (1650, 600)], color=C_BLUE, width=9)
    arrow([(2020, 600), (2100, 600)], color=C_TEAL, width=9)
    arrow([(2374, 600), (2454, 600)], color=C_ORANGE, width=9)
    arrow([(2684, 600), (2760, 600)], color=C_ORANGE, width=9)

    band_b = (895, 885, 3110, 1512)
    rounded(band_b, r=22, fill=C_CYAN_FILL, outline=C_CYAN_STROKE, width=3)
    d.text((940, 942), "B  E2E camera-pose NVS policy", font=F_BAND, fill=C_TEAL_DARK)
    rgb_views_card((940, 1014, 1276, 1304))
    pose_block((1350, 1084, 1574, 1234), aux_y=1272)
    ray_maps_card((1650, 1014, 2020, 1304))
    arrow([(1276, 1158), (1350, 1158)], color=C_PURPLE, width=9)
    arrow([(1574, 1158), (1650, 1158)], color=C_BLUE, width=9)

    node((2072, 1078, 2366, 1234), "Scene-token", "encoder", stroke=C_TEAL, fill="white", title_size=38)
    scene_token_glyph((2130, 928, 2310, 1064), "scene token", label_size=29)
    scene_token_glyph((2428, 1076, 2608, 1232), "updated\nscene token", label_size=31)
    labeled_tile(ray_b, (2422, 928, 2632, 1068), "target ray\nquery", C_RED, label_color=C_RED, label_size=28, image_frac=0.48)
    node((2700, 960, 2884, 1108), "NVS", "decoder", stroke=C_BLUE, fill="#eef9ff", title_size=41)
    labeled_tile(view_b, (2934, 922, 3096, 1148), "NVS\nobjective", C_RED, label_size=28, image_frac=0.54)
    node((2700, 1270, 2884, 1418), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=39)
    labeled_tile(action_crop, (2934, 1228, 3096, 1454), "action\nobjective", C_ORANGE, label_size=28, image_frac=0.54)
    arrow([(2020, 1158), (2072, 1158)], color=C_TEAL, width=9)
    arrow([(2220, 1064), (2220, 1078)], color=C_TEAL, width=8, head=18)
    arrow([(2366, 1156), (2428, 1156)], color=C_TEAL, width=9)
    arrow([(2608, 1188), (2652, 1188), (2652, 1034), (2700, 1034)], color=C_BLUE, width=8)
    arrow([(2632, 998), (2700, 998)], color=C_RED, width=8, head=20)
    arrow([(2884, 1034), (2934, 1034)], color=C_RED, width=8)
    arrow([(2608, 1220), (2652, 1220), (2652, 1344), (2700, 1344)], color=C_ORANGE, width=8)
    arrow([(2884, 1344), (2934, 1344)], color=C_ORANGE, width=8)

    img.save(OUT, quality=98)


if __name__ == "__main__":
    main()
