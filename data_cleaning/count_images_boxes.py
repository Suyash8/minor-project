#!/usr/bin/env python3
"""Count images and boxes per dataset per split in the cleaned data.

Usage (Colab, Drive mounted):
    python data_cleaning/count_images_boxes.py
    CLEANED_DATA_ROOT=/path python data_cleaning/count_images_boxes.py
"""
import os

ROOT = os.environ.get(
    "CLEANED_DATA_ROOT",
    "/content/drive/MyDrive/object-detection/cleaned_data")

def main():
    print(f"{'dataset':<10}{'split':<7}{'images':>8}{'boxes':>9}{'boxes/img':>11}")
    tot_i = tot_b = 0
    for ds in ["codrone", "visdrone", "dota"]:
        ti = tb = 0
        for sp in ["train", "val", "test"]:
            img_d = os.path.join(ROOT, ds, sp, "images")
            lbl_d = os.path.join(ROOT, ds, sp, "labels")
            n_img = len(os.listdir(img_d)) if os.path.isdir(img_d) else 0
            n_box = 0
            if os.path.isdir(lbl_d):
                for f in os.listdir(lbl_d):
                    if f.endswith(".txt"):
                        with open(os.path.join(lbl_d, f)) as fh:
                            n_box += sum(1 for ln in fh if ln.strip())
            print(f"{ds:<10}{sp:<7}{n_img:>8}{n_box:>9}{(n_box/max(n_img,1)):>11.1f}")
            ti += n_img; tb += n_box
        print(f"{ds+' TOTAL':<17}{ti:>8}{tb:>9}{(tb/max(ti,1)):>11.1f}\n")
        tot_i += ti; tot_b += tb
    print(f"{'GRAND TOTAL':<17}{tot_i:>8}{tot_b:>9}{(tot_b/max(tot_i,1)):>11.1f}")

if __name__ == "__main__":
    main()
