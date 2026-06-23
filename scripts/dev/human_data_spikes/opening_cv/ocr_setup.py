import os
import sys

import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(["en"], gpu=False, verbose=False)
LX0, LY0, LX1, LY1 = 0.645, 0.0, 1.0, 0.35
for path in sys.argv[1:]:
    t = os.path.basename(path).split("_")[-1].split(".")[0].replace("g", "")
    im = Image.open(path).convert("RGB")
    w, h = im.size
    crop = im.crop((int(LX0 * w), int(LY0 * h), int(LX1 * w), int(LY1 * h)))
    res = reader.readtext(np.array(crop), detail=0, paragraph=False)
    print(f"=== t={t} ===")
    for line in res:
        print("   ", line)
