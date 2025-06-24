import os, io
import pandas as pd
from PIL import Image

# --- CONFIGURE THESE PATHS ---
PARQUET = "data/train-00000-of-00001.parquet"
OUT_Q   = "data/questions"
OUT_I   = "data/images"
# -----------------------------

os.makedirs(OUT_Q, exist_ok=True)
os.makedirs(OUT_I, exist_ok=True)

df = pd.read_parquet(PARQUET)

for _, row in df.iterrows():
    idx      = row["index"]
    text     = row["question"]
    img_bytes= row["image_0"]["bytes"]  # assuming only one image

    # 1) dump the question
    with open(f"{OUT_Q}/{idx:04d}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # 2) restore + save the PNG
    img = Image.open(io.BytesIO(img_bytes))
    img.save(f"{OUT_I}/{idx:04d}_0.png")
