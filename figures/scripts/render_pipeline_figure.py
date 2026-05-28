from __future__ import annotations

import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pipeline.png"
ASSETS = ROOT / "pipeline_assets"

W, H = 3800, 1850
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
BLACK = Path("/System/Library/Fonts/Supplemental/Arial Black.ttf")


def font(size: int, bold: bool = False, black: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BLACK if black else (BOLD if bold else FONT)), size)


F_TITLE = font(78, black=True)
F_SUB = font(39)
F_PANEL = font(42, bold=True)
F_BAND = font(35, bold=True)
F_LABEL = font(33, bold=True)
F_SMALL = font(29)
F_TINY = font(25)
F_CHIP = font(25, bold=True)

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
    tb = pd.textbbox((0, 0), label, font=F_TINY)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pd.rounded_rectangle((8, 8, tw + 30, th + 26), radius=8, fill=(255, 255, 255, 235), outline=(95, 105, 118, 185), width=2)
    pd.text((18, 14), label, font=F_TINY, fill=(65, 72, 86, 255))
    base = img.crop((x0, y0, x1, y1)).convert("RGBA")
    comp = Image.alpha_composite(base, patch)
    img.paste(comp.convert("RGB"), (x0, y0))


def labeled_tile(src, box, label, stroke, label_color=None, label_size=23, image_frac=0.72, overlay_drop=False):
    x0, y0, x1, y1 = map(int, box)
    rounded((x0, y0, x1, y1), r=14, fill="white", outline=stroke, width=4)
    pad = 11
    im_bottom = y0 + int((y1 - y0) * image_frac)
    paste_cover(src, (x0 + pad, y0 + pad, x1 - pad, im_bottom), radius=9)
    if overlay_drop:
        dropout_overlay((x0 + pad, y0 + pad, x1 - pad, im_bottom), "p_drop")
    text_fit_center(((x0 + x1) / 2, y1 - 22), label, label_size, (x1 - x0) - 18, fill=label_color or stroke, bold=True)


def pair_card(box, title, rgb_img, ray_img, stroke, dropped=False, title_size=24):
    x0, y0, x1, y1 = map(int, box)
    rounded((x0, y0, x1, y1), r=16, fill="#fbfdff", outline=stroke, width=4)
    text_fit_center(((x0 + x1) / 2, y0 + 30), title, title_size, (x1 - x0) - 24, fill=stroke, bold=True)
    gap = 14
    inner_top = y0 + 58
    inner_bottom = y1 - 16
    tile_w = int((x1 - x0 - 46 - gap) / 2)
    labeled_tile(rgb_img, (x0 + 16, inner_top, x0 + 16 + tile_w, inner_bottom), "RGB", stroke, label_size=20, overlay_drop=dropped)
    labeled_tile(ray_img, (x0 + 16 + tile_w + gap, inner_top, x0 + 16 + tile_w * 2 + gap, inner_bottom), "ray", stroke, label_size=20, overlay_drop=dropped)


def plucker(size=(330, 185), phase=0.0):
    w, h = size
    xx = np.linspace(0, 1, w)[None, :]
    yy = np.linspace(0, 1, h)[:, None]
    hue = (0.62 * xx + 0.26 * (1 - yy) + phase) % 1.0
    sat = np.broadcast_to(0.78 + 0.12 * np.sin(2 * np.pi * (xx + yy)), (h, w))
    val = np.broadcast_to(0.96 - 0.10 * yy, (h, w))
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = colorsys.hsv_to_rgb(float(hue[y, x]), float(sat[y, x]), float(val[y, x]))
            arr[y, x] = (int(r * 255), int(g * 255), int(b * 255))
    out = Image.fromarray(arr, "RGB")
    od = ImageDraw.Draw(out)
    for gx in range(0, w, 38):
        od.line((gx, 0, gx, h), fill=(255, 255, 255), width=2)
    for gy in range(0, h, 38):
        od.line((0, gy, w, gy), fill=(255, 255, 255), width=2)
    for y in range(24, h, 38):
        for x in range(24, w, 46):
            ang = -0.55 + 1.25 * (x / w) + 0.25 * math.sin(y * 0.05 + phase * 3)
            length = 16
            x2 = x + length * math.cos(ang)
            y2 = y + length * math.sin(ang)
            od.line((x, y, x2, y2), fill=(35, 45, 60), width=2)
            od.polygon(
                [
                    (x2, y2),
                    (x2 - 5 * math.cos(ang - 0.55), y2 - 5 * math.sin(ang - 0.55)),
                    (x2 - 5 * math.cos(ang + 0.55), y2 - 5 * math.sin(ang + 0.55)),
                ],
                fill=(35, 45, 60),
            )
    return out


def main():
    teaser = Image.open(ROOT / "teaser.png").convert("RGB")
    view_a = Image.open(ASSETS / "inputA_direct.png").convert("RGB")
    view_b = Image.open(ASSETS / "inputB_direct.png").convert("RGB")
    scene_crop = teaser.crop((120, 95, 2050, 950))
    pose_crop = teaser.crop((770, 1280, 1880, 1840))
    action_crop = teaser.crop((2190, 1320, 3380, 1810))
    ray_a = plucker((420, 235), 0.03)
    ray_b = plucker((420, 235), 0.42)

    d.text((78, 48), "End-to-end camera-aware policy pipeline", font=F_TITLE, fill=C_TEXT)
    d.text((80, 132), "RGB observations are paired with predicted Plucker rays; only the NVS branch samples source/target views and applies target dropout.", font=F_SUB, fill=C_MUTED)

    left = (60, 190, 1050, 1778)
    right = (1100, 190, 3740, 1778)
    shadowed_panel(left)
    shadowed_panel(right)
    d.text((105, 235), "Scene + input views", font=F_PANEL, fill=C_TEXT)
    d.text((1150, 235), "Model variants", font=F_PANEL, fill=C_TEXT)

    rounded((105, 305, 1010, 1045), r=22, fill="#fbfdff", outline=C_LINE, width=3)
    paste_cover(scene_crop, (135, 335, 980, 1015), radius=16)
    for box, label, col, im in [
        ((105, 1148, 535, 1648), "View A", C_TEAL, view_a),
        ((585, 1148, 1010, 1648), "View B", C_RED, view_b),
    ]:
        rounded(box, r=20, fill="#fbfdff", outline=C_LINE, width=3)
        x0, y0, x1, y1 = box
        paste_cover(im, (x0 + 26, y0 + 30, x1 - 26, y0 + 380), radius=14, border=col, bw=6)
        text_fit_center(((x0 + x1) / 2, y1 - 46), label, 36, (x1 - x0) - 38, fill=col, bold=True)

    front = (1148, 305, 3690, 695)
    rounded(front, r=22, fill="#fbfdff", outline=C_LINE, width=3)
    d.text((1186, 338), "Per-view camera geometry front-end", font=F_BAND, fill=C_TEAL_DARK)

    def front_card(box, title, rgb_img, ray_img, stroke):
        x0, y0, x1, y1 = box
        rounded(box, r=17, fill="white", outline=stroke, width=4)
        text_fit_center(((x0 + x1) / 2, y0 + 34), title, 30, (x1 - x0) - 28, fill=stroke, bold=True)
        labeled_tile(rgb_img, (x0 + 32, y0 + 78, x0 + 252, y1 - 28), "RGB", stroke, label_size=21)
        node((x0 + 382, y0 + 91, x0 + 625, y1 - 43), "Pose", "pred.", stroke=C_PURPLE, fill=C_PURPLE_FILL, title_color=C_TEXT, title_size=31)
        labeled_tile(ray_img, (x1 - 452, y0 + 78, x1 - 32, y1 - 28), "Plucker ray", C_BLUE, label_color=stroke, label_size=21)
        arrow([(x0 + 252, (y0 + y1) / 2 + 16), (x0 + 382, (y0 + y1) / 2 + 16)], color=C_PURPLE, width=7)
        arrow([(x0 + 625, (y0 + y1) / 2 + 16), (x1 - 452, (y0 + y1) / 2 + 16)], color=C_BLUE, width=7)

    front_card((1210, 395, 2395, 625), "View A", view_a, ray_a, C_TEAL)
    front_card((2475, 395, 3660, 625), "View B", view_b, ray_b, C_RED)

    band_a = (1148, 755, 3690, 1120)
    rounded(band_a, r=22, fill=C_GREEN_FILL, outline=C_GREEN_STROKE, width=3)
    d.text((1195, 805), "A  E2E camera-pose action policy", font=F_BAND, fill=C_TEAL_DARK)
    pair_card((1220, 858, 1538, 1062), "View A input", view_a, ray_a, C_TEAL)
    pair_card((1576, 858, 1894, 1062), "View B input", view_b, ray_b, C_RED)
    node((2050, 890, 2380, 1025), "Observation", "encoder", stroke="#35ad67", fill="white", title_size=34)
    node((2585, 890, 2885, 1025), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=34)
    labeled_tile(action_crop, (3090, 825, 3610, 1070), "action objective", C_ORANGE, label_size=29, image_frac=0.76)
    arrow([(1538, 940), (2050, 940)], color=C_TEAL, width=8)
    arrow([(1894, 980), (2050, 980)], color=C_TEAL, width=8)
    arrow([(2380, 958), (2585, 958)], color=C_ORANGE, width=8)
    arrow([(2885, 958), (3090, 958)], color=C_ORANGE, width=8)

    band_b = (1148, 1200, 3690, 1668)
    rounded(band_b, r=22, fill=C_CYAN_FILL, outline=C_CYAN_STROKE, width=3)
    d.text((1195, 1250), "B  E2E camera-pose NVS policy", font=F_BAND, fill=C_TEAL_DARK)
    sampler = (1218, 1302, 1960, 1615)
    rounded(sampler, r=18, fill="white", outline=C_LINE, width=3)
    text_fit_center(((sampler[0] + sampler[2]) / 2, sampler[1] + 32), "NVS source / target split", 27, sampler[2] - sampler[0] - 32, fill=C_TEXT, bold=True)
    pair_card((1248, 1352, 1582, 1588), "source A", view_a, ray_a, C_TEAL)
    pair_card((1598, 1352, 1932, 1588), "target B dropped", view_b, ray_b, C_RED, dropped=True)
    node((2105, 1458, 2448, 1588), "Scene-token", "encoder", stroke=C_TEAL, fill="white", title_size=33)
    labeled_tile(ray_b, (2130, 1282, 2468, 1418), "target ray query", C_BLUE, label_color=C_BLUE, label_size=22, image_frac=0.66)
    grid_x, grid_y = 2588, 1378
    for row in range(8):
        for col in range(8):
            x = grid_x + col * 32
            y = grid_y + row * 26
            d.rounded_rectangle((x, y, x + 18, y + 18), radius=4, fill="#def7e3", outline="#65c878", width=2)
    text_center((grid_x + 112, grid_y + 232), "scene tokens", F_SMALL, fill=C_MUTED)
    node((2988, 1276, 3268, 1406), "NVS", "decoder", stroke=C_BLUE, fill="#eef9ff", title_size=34)
    labeled_tile(view_b, (3350, 1242, 3610, 1450), "NVS objective", C_BLUE, label_size=24, image_frac=0.72)
    node((2988, 1500, 3268, 1628), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=34)
    labeled_tile(action_crop, (3350, 1472, 3610, 1642), "action objective", C_ORANGE, label_size=24, image_frac=0.66)
    arrow([(1960, 1518), (2105, 1518)], color=C_TEAL, width=8)
    arrow([(2448, 1522), (2588, 1522)], color=C_TEAL, width=8)
    arrow([(2828, 1490), (2925, 1490), (2925, 1342), (2988, 1342)], color=C_BLUE, width=8)
    arrow([(2468, 1350), (2988, 1350)], color=C_BLUE, width=7)
    arrow([(3268, 1342), (3350, 1342)], color=C_BLUE, width=8)
    arrow([(2828, 1518), (2925, 1518), (2925, 1564), (2988, 1564)], color=C_ORANGE, width=8)
    arrow([(3268, 1564), (3350, 1564)], color=C_ORANGE, width=8)

    chip(1148, 1710, "No GT camera pose at inference", C_PURPLE)
    chip(1665, 1710, "NVS decoder is training-time only", C_BLUE)
    chip(2265, 1710, "Action loss supervises both variants", C_TEAL)
    chip(2928, 1710, "Target dropout only in NVS", C_RED)

    img.save(OUT, quality=98)


if __name__ == "__main__":
    main()
