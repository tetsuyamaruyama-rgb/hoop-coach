"""Generate simple basketball app icons (no external assets)."""
from PIL import Image, ImageDraw

def basketball(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # rounded-square background
    pad = int(size * 0.06)
    d.rounded_rectangle([pad, pad, size - pad, size - pad],
                        radius=int(size * 0.22), fill=(255, 247, 239, 255))
    # ball
    m = int(size * 0.16)
    box = [m, m, size - m, size - m]
    d.ellipse(box, fill=(255, 122, 24, 255))
    cx, cy = size / 2, size / 2
    r = (size - 2 * m) / 2
    lw = max(2, int(size * 0.012))
    line = (60, 30, 10, 255)
    # seams
    d.line([cx, m, cx, size - m], fill=line, width=lw)
    d.line([m, cy, size - m, cy], fill=line, width=lw)
    d.arc([cx - r * 2.0, m, cx + r * 0.2, size - m], 270, 90, fill=line, width=lw)
    d.arc([cx - r * 0.2, m, cx + r * 2.0, size - m], 90, 270, fill=line, width=lw)
    d.ellipse(box, outline=line, width=lw)
    return img

for s in (180, 192, 512):
    basketball(s).save(f"icons/icon-{s}.png")
    print("wrote icons/icon-%d.png" % s)
