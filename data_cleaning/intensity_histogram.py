# =============================================================================
# intensity_histogram.py — "Histogram of Mean Intensity" per dataset
# Same style as the IDD reference figure: three overlaid histograms
# (Train_Data / Validation_Data / Test_Data), one figure per dataset.
#
# Reads the CLEANED datasets from Drive:
#   /content/drive/MyDrive/object-detection/cleaned_data/<ds>/{train,val,test}/images
# Saves PNGs to /content/drive/MyDrive/object-detection/figures/ and shows inline.
#
# Optional: set INTENS_SAMPLE (e.g. 2000) to subsample per split for speed.
# =============================================================================
import os, glob, random
from concurrent.futures import ThreadPoolExecutor
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

ROOT   = "/content/drive/MyDrive/object-detection/cleaned_data"
FIGDIR = "/content/drive/MyDrive/object-detection/figures"
SAMPLE = int(os.environ.get("INTENS_SAMPLE", "0"))   # 0 = use every image
WORKERS = 32
SEED = 42

os.makedirs(FIGDIR, exist_ok=True)

def mean_intensity(path):
    """mean grayscale intensity 0-255, via 1x1 BOX resize (fast, exact mean)."""
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            return float(g.resize((1, 1), Image.BOX).getpixel((0, 0)))
    except Exception:
        return None

def split_values(split_dir):
    files = sorted(glob.glob(os.path.join(split_dir, "*.jpg"))
                   + glob.glob(os.path.join(split_dir, "*.png")))
    if SAMPLE and len(files) > SAMPLE:
        random.Random(SEED).shuffle(files)
        files = files[:SAMPLE]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        vals = [v for v in ex.map(mean_intensity, files) if v is not None]
    return vals

for ds in ["codrone", "visdrone", "dota"]:
    data, stats = {}, {}
    for split, label in [("train", "Train_Data"), ("val", "Validation_Data"),
                         ("test", "Test_Data")]:
        d = os.path.join(ROOT, ds, split, "images")
        if not os.path.isdir(d):
            print(f"[{ds}] missing {d}, skipped")
            continue
        vals = split_values(d)
        data[label] = vals
        if vals:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            stats[label] = (len(vals), m, sd)

    if not data:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [i * 2 for i in range(128)]            # 0-255 in bins of 2
    for label, color in [("Train_Data", "tab:blue"),
                         ("Validation_Data", "tab:orange"),
                         ("Test_Data", "tab:green")]:
        if label in data:
            ax.hist(data[label], bins=bins, alpha=0.55, label=label,
                    color=color, edgecolor=color)
    ax.set_title(f"Histogram of Mean Intensity for {ds.upper()} Dataset",
                 fontsize=14)
    ax.set_xlabel("Mean Intensity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"{ds}_mean_intensity_hist.png")
    fig.savefig(out, dpi=200)
    plt.show()
    print(f"[{ds}] saved -> {out}")
    for label, (n, m, sd) in stats.items():
        print(f"    {label:17s} n={n:5d}  mean={m:6.1f}  std={sd:5.1f}")

print("\nAll figures written to:", FIGDIR)
