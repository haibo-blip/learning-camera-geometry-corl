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
    d.text((80, 132), "Two rendered RGB views and two predicted Plucker ray maps are encoded jointly; objectives are shown at their heads.", font=F_SUB, fill=C_MUTED)

    left = (60, 190, 1050, 1778)
    right = (1100, 190, 3740, 1778)
    shadowed_panel(left)
    shadowed_panel(right)
    d.text((105, 235), "Scene + direct views", font=F_PANEL, fill=C_TEXT)
    d.text((1150, 235), "Model variants", font=F_PANEL, fill=C_TEXT)

    rounded((105, 305, 1010, 1045), r=22, fill="#fbfdff", outline=C_LINE, width=3)
    paste_cover(scene_crop, (135, 335, 980, 1015), radius=16)
    for box, label, col, im in [
        ((105, 1148, 535, 1648), "source view", C_TEAL, view_a),
        ((585, 1148, 1010, 1648), "target view", C_RED, view_b),
    ]:
        rounded(box, r=20, fill="#fbfdff", outline=C_LINE, width=3)
        x0, y0, x1, y1 = box
        paste_cover(im, (x0 + 26, y0 + 30, x1 - 26, y0 + 380), radius=14, border=col, bw=6)
        text_center(((x0 + x1) / 2, y1 - 46), label, F_LABEL, fill=col)

    front = (1148, 305, 3690, 665)
    rounded(front, r=22, fill="#fbfdff", outline=C_LINE, width=3)
    d.text((1186, 338), "Camera geometry front-end", font=F_BAND, fill=C_TEAL_DARK)
    rgb_box = (1206, 408, 1618, 610)
    rounded(rgb_box, r=14, fill="white", outline=C_LINE, width=3)
    paste_cover(view_a, (1230, 431, 1396, 558), radius=11, border=C_TEAL, bw=4)
    paste_cover(view_b, (1418, 431, 1584, 558), radius=11, border=C_RED, bw=4)
    text_center((1412, 591), "RGB views", F_SMALL, fill=C_TEXT)
    node((1725, 400, 1985, 565), "Pose", "predictor", stroke=C_PURPLE, fill=C_PURPLE_FILL, title_color=C_TEXT, title_size=36)
    pose_img_box = (2130, 365, 2525, 590)
    rounded(pose_img_box, r=14, fill="white", outline=C_PURPLE, width=5)
    paste_cover(pose_crop, (2142, 377, 2513, 555), radius=10)
    text_center(((pose_img_box[0] + pose_img_box[2]) / 2, 577), "cam pose", F_SMALL, fill=C_PURPLE)
    rays_box = (2730, 350, 3600, 620)
    rounded(rays_box, r=18, fill="white", outline=C_BLUE, width=4)
    paste_cover(ray_a, (2760, 382, 3125, 542), radius=10, border=C_TEAL, bw=4)
    paste_cover(ray_b, (3195, 382, 3560, 542), radius=10, border=C_RED, bw=4)
    text_center((3165, 591), "two Plucker ray maps", F_SMALL, fill=C_BLUE)
    arrow([(1618, 503), (1725, 503)], color=C_PURPLE, width=8)
    arrow([(1985, 503), (2130, 503)], color=C_PURPLE, width=8)
    d.text((2548, 481), "pose + intrinsics", font=F_TINY, fill=C_MUTED)
    arrow([(2525, 503), (2730, 503)], color=C_BLUE, width=8)

    obs = (1240, 750, 1845, 1180)
    rounded(obs, r=18, fill="white", outline=C_TEAL, width=5)
    text_center((1542, 804), "RGB + rays", font(42, bold=True), fill=C_TEXT)
    d.text((1340, 838), "2 RGB views + 2 ray maps", font=F_SMALL, fill=C_MUTED)

    # View/ray pairs are labeled below the thumbnails; view B and its ray map are dropped together.
    paste_cover(view_a, (1285, 895, 1408, 982), radius=9, border=C_TEAL, bw=3)
    paste_cover(ray_a, (1435, 895, 1608, 982), radius=9, border=C_TEAL, bw=3)
    text_center((1446, 1013), "view A + ray A", F_TINY, fill=C_TEAL_DARK)
    paste_cover(view_b, (1285, 1035, 1408, 1122), radius=9, border=C_RED, bw=3)
    paste_cover(ray_b, (1435, 1035, 1608, 1122), radius=9, border=C_RED, bw=3)
    dropout_overlay((1285, 1035, 1408, 1122), "p_drop")
    dropout_overlay((1435, 1035, 1608, 1122), "p_drop")
    text_center((1446, 1150), "view B + ray B", F_TINY, fill=C_RED)
    arrow([(1412, 610), (1412, 720), (1490, 720), (1490, 750)], color=C_TEAL, width=8)
    arrow([(3165, 620), (3165, 720), (1685, 720), (1685, 750)], color=C_BLUE, width=8)

    band_a = (1905, 740, 3650, 1135)
    rounded(band_a, r=22, fill=C_GREEN_FILL, outline=C_GREEN_STROKE, width=3)
    d.text((1945, 786), "A  E2E camera-pose action policy", font=F_BAND, fill=C_TEAL_DARK)
    node((2015, 880, 2335, 1015), "Observation", "encoder", stroke="#35ad67", fill="white", title_size=34)
    node((2515, 880, 2815, 1015), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=34)
    action_a = (3005, 824, 3540, 1065)
    rounded(action_a, r=14, fill="white", outline=C_ORANGE, width=5)
    paste_cover(action_crop, (3018, 837, 3527, 1028), radius=9)
    text_center(((action_a[0] + action_a[2]) / 2, 1048), "action", F_SMALL, fill=C_ORANGE)
    arrow([(1845, 930), (2015, 930)], color=C_TEAL, width=10)
    arrow([(2335, 948), (2515, 948)], color=C_ORANGE, width=8)
    arrow([(2815, 948), (3005, 948)], color=C_ORANGE, width=8)

    band_b = (1905, 1210, 3650, 1668)
    rounded(band_b, r=22, fill=C_CYAN_FILL, outline=C_CYAN_STROKE, width=3)
    d.text((1945, 1262), "B  E2E camera-pose NVS policy", font=F_BAND, fill=C_TEAL_DARK)
    node((2015, 1388, 2355, 1526), "Scene-token", "encoder", stroke=C_TEAL, fill="white", title_size=33)
    grid_x, grid_y = 2490, 1348
    for row in range(8):
        for col in range(8):
            x = grid_x + col * 34
            y = grid_y + row * 28
            d.rounded_rectangle((x, y, x + 18, y + 18), radius=4, fill="#def7e3", outline="#65c878", width=2)
    text_center((grid_x + 120, grid_y + 250), "64 scene tokens", F_SMALL, fill=C_MUTED)
    node((2910, 1275, 3200, 1398), "NVS", "decoder", stroke=C_BLUE, fill="#eef9ff", title_size=34)
    nvs_out = (3280, 1238, 3575, 1438)
    rounded(nvs_out, r=14, fill="white", outline=C_BLUE, width=5)
    paste_cover(view_b, (3293, 1251, 3562, 1400), radius=9)
    text_center(((nvs_out[0] + nvs_out[2]) / 2, 1423), "target view", F_TINY, fill=C_BLUE)
    node((2910, 1480, 3200, 1603), "Action", "expert", stroke=C_ORANGE, fill=C_ORANGE_FILL, title_size=34)
    action_b = (3280, 1460, 3575, 1635)
    rounded(action_b, r=14, fill="white", outline=C_ORANGE, width=5)
    paste_cover(action_crop, (3293, 1473, 3562, 1595), radius=9)
    text_center(((action_b[0] + action_b[2]) / 2, 1622), "action", F_TINY, fill=C_ORANGE)
    arrow([(1845, 1088), (1862, 1088), (1862, 1457), (2015, 1457)], color=C_TEAL, width=10)
    arrow([(2355, 1457), (2490, 1457)], color=C_TEAL, width=8)
    arrow([(2762, 1457), (2835, 1457), (2835, 1336), (2910, 1336)], color=C_BLUE, width=8)
    arrow([(3200, 1336), (3280, 1336)], color=C_BLUE, width=8)
    arrow([(2762, 1457), (2835, 1457), (2835, 1542), (2910, 1542)], color=C_ORANGE, width=8)
    arrow([(3200, 1542), (3280, 1542)], color=C_ORANGE, width=8)

    chip(1148, 1708, "No GT camera pose at inference", C_PURPLE)
    chip(1665, 1708, "NVS decoder is training-time only", C_BLUE)
    chip(2265, 1708, "Action loss supervises both variants", C_TEAL)
    chip(2928, 1708, "One view/ray pair can be dropped", C_RED)

    img.save(OUT, quality=98)


if __name__ == "__main__":
    main()
