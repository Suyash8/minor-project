#!/usr/bin/env python3
# =============================================================================
# clean_build_v4.py — three SEPARATED but cross-compatible cleaned OBB datasets
#
# Output (each dataset identical structure, identical 8-class space):
#   <out>/cleaned_data/
#     codrone /  visdrone /  dota  each with:
#       train/{images,labels}  val/{images,labels}  test/{images,labels}
#       dataset.yaml   (path/train/val/test/names — same names everywhere)
#     dataset_all.yaml      <- merge all three (Ultralytics accepts lists)
#     classes.txt  cleaning_report.json  CLEANING_SUMMARY.md
#
# Cross-compatible: swap any dataset folder for another and your training
# program works unchanged; merge all three by pointing at dataset_all.yaml.
#
# v4 (rewritten):
#   - RECURSIVE discovery: every image under data/<ds> is matched to a label
#     file purely by stem, anywhere under data/<ds>. No hardcoded folder names.
#   - ORIGINAL DOTA format supported: "x1 y1 x2 y2 x3 y3 x4 y4 class difficult"
#     (class-LAST, 10 tokens, absolute pixels, 'imagesource:'/'gsd:' headers)
#   - labels-v1.5 dirs get top label priority (v1.5 has 3.4x more vehicles)
#   - heavy multithreading (default 64) — Drive read latency is the bottleneck
#
# Colab usage:
#   TEST:  paste & run as-is                      (TEST_RUN=1, 60 imgs/source)
#   FULL:  os.environ['CLEAN_TEST_RUN']='0'       then run
# =============================================================================
import os, io, sys, json, glob, hashlib, shutil, random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None          # DOTA source images can be huge

# ------------------------- config -------------------------------------------
DATA_DIR     = os.environ.get("CLEAN_DATA_DIR", "/content/drive/MyDrive/object-detection/data")
OUT_ROOT     = os.environ.get("CLEAN_OUT_ROOT", "/content/drive/MyDrive/object-detection")
TEST_RUN     = os.environ.get("CLEAN_TEST_RUN", "1") == "1"
TEST_N       = int(os.environ.get("CLEAN_TEST_N", "60"))      # per source
MAX_WORKERS  = int(os.environ.get("CLEAN_WORKERS", "64"))     # Drive read bound
COPY_WORKERS = int(os.environ.get("CLEAN_COPY_WORKERS", "16"))
COPY_TO_DRIVE = (not TEST_RUN) or os.environ.get("CLEAN_FORCE_COPY", "0") == "1"
CROSS_SOURCE_DEDUP = os.environ.get("CLEAN_CROSS_DEDUP", "1") == "1"
LOCAL_ROOT   = os.environ.get("CLEAN_LOCAL_ROOT", "/content")
STAGING      = None          # set in main(): local scratch for phase A output
MIN_BOX_PX   = int(os.environ.get("CLEAN_MIN_BOX_PX", "4"))
RESIZE_LONG_SIDE = 1280
JPEG_QUALITY = 90
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_FRAC = {"train": 0.82, "val": 0.09, "test": 0.09}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ------------------------- class space --------------------------------------
UNIFIED = ["pedestrian", "bicycle", "car", "van", "truck", "bus", "tricycle", "motorcyclist"]
UIDX = {n: i for i, n in enumerate(UNIFIED)}
SYNONYMS = {
    "pedestrian": "pedestrian", "people": "pedestrian", "person": "pedestrian",
    "bicycle": "bicycle", "cyclist": "bicycle", "bike": "bicycle",
    "car": "car", "small-vehicle": "car",
    "van": "van",
    "truck": "truck", "large-vehicle": "truck",
    "bus": "bus",
    "tricycle": "tricycle", "awning-tricycle": "tricycle",
    "motorcyclist": "motorcyclist", "motor": "motorcyclist",
    "motorbike": "motorcyclist",
}
# VisDrone RAW annotations are 1-based, 0 = ignored regions, 11 = others
VISDRONE_RAW_MAP = {0: None, 1: "pedestrian", 2: "pedestrian", 3: "bicycle",
                    4: "car", 5: "van", 6: "truck", 7: "tricycle",
                    8: "tricycle", 9: "bus", 10: "motorcyclist", 11: None}
FALLBACK_ID_MAPS = {
    "codrone":  {0: "pedestrian", 1: "bicycle", 2: "car", 3: "van",
                 4: "truck", 5: "bus", 6: "tricycle", 7: "motorcyclist"},
    "visdrone": {0: "pedestrian", 1: "pedestrian", 2: "bicycle", 3: "car",
                 4: "van", 5: "truck", 6: "tricycle", 7: "tricycle",
                 8: "bus", 9: "motorcyclist"},
    "dota":     {9: "truck", 10: "car"},
}

def unify_name(name):
    n = name.strip().lower().replace("_", "-")
    n = SYNONYMS.get(n, n)
    return UIDX.get(n)

# yaml-driven id->name maps per dataset (fallback if no/!parseable yaml)
def load_yaml_names(ds_dir):
    try:
        import yaml
    except ImportError:
        return {}
    maps = {}
    for yf in glob.glob(os.path.join(ds_dir, "*.yaml")) + glob.glob(os.path.join(ds_dir, "*.yml")):
        try:
            y = yaml.safe_load(open(yf, "r")) or {}
        except Exception:
            continue
        names = y.get("names")
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        if not isinstance(names, dict):
            continue
        m = {}
        for k, v in names.items():
            try:
                m[int(k)] = str(v).strip().lower()
            except (TypeError, ValueError):
                pass
        if m:
            maps[os.path.basename(yf)] = m
    return maps

def merge_id_maps(ymaps, fallback):
    merged = dict(fallback)
    conflicts = []
    for yf, m in sorted(ymaps.items()):
        for k, v in m.items():
            if k in merged:
                a = unify_name(merged[k]); a = merged[k] if a is None else UNIFIED[a]
                b = unify_name(v);       b = v if b is None else UNIFIED[b]
                if a != b:   # same unified class (e.g. cyclist==bicycle) is NOT a conflict
                    conflicts.append(f"{yf}: id {k}={v} overrides {merged[k]}")
            merged[k] = v
    return merged, conflicts

# ------------------------- discovery ----------------------------------------
def label_dir_prio(rel):
    """lower rel path of a dir containing txt files -> priority (smaller=best)"""
    r = rel.lower()
    leaf = r.rstrip("/").split("/")[-1]
    if "v1.5" in r or "labeltxt" in leaf:
        return 0          # labels-v1.5 / labelTxt-v1.5 (and v1.0 labelTxt)
    if leaf == "labels":
        return 1
    if leaf == "annfile":
        return 2
    if leaf == "annotations":
        return 3
    if "xml" in leaf:
        return 9
    return 5

def discover_dataset(ds_dir):
    """walk everything; images by extension, label candidates = dirs with txts"""
    images, txt_dirs = [], {}
    for root, _dirs, files in os.walk(ds_dir):
        for f in files:
            p = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                images.append(p)
            elif ext in (".txt", ".gdoc"):
                txt_dirs.setdefault(root, []).append(p)
    images.sort()
    # label index: stem -> [(prio, path)] sorted best-first
    lbl_index = defaultdict(list)
    notes = Counter()
    for d, files in txt_dirs.items():
        rel = os.path.relpath(d, ds_dir)
        prio = label_dir_prio(rel)
        n_empty = n_gdoc = 0
        for p in files:
            base = os.path.basename(p).lower()
            if base.endswith(".gdoc"):
                n_gdoc += 1
                stem = os.path.splitext(os.path.splitext(base)[0])[0]
            else:
                stem = os.path.splitext(base)[0]
            try:
                if os.path.getsize(p) == 0:
                    n_empty += 1
            except OSError:
                pass
            lbl_index[stem].append((prio, p))
        if n_gdoc:
            notes[f"gdoc_placeholders:{rel}"] = n_gdoc
        if n_empty:
            notes[f"empty_label_files:{rel}"] = n_empty
    for stem in lbl_index:
        lbl_index[stem].sort(key=lambda t: (t[0], t[1]))
    return images, lbl_index, notes

# ------------------------- label parsing ------------------------------------
def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_label_file(path, id_map):
    """
    Returns (fmts Counter, rows, notes Counter, id_hist Counter) where each row
    is dict(uid, coords(8 floats, pixel or normalized), normalized, difficult).
    Handles: VisDrone-raw CSV, YOLO-OBB (9 tok, normalized), DOTA class-first
    pixels (9 tok), original DOTA class-last (10 tok + header lines).
    """
    try:
        raw = open(path, "rb").read()
    except OSError:
        return Counter(), [], Counter({"read_error": 1}), Counter()
    if not raw.strip():
        return Counter(), [], Counter({"empty_label_file": 1}), Counter()
    txt = raw.decode("utf-8", errors="replace")
    fmts, rows, notes, id_hist = Counter(), [], Counter(), Counter()
    first = txt.lstrip().split("\n", 1)[0].strip()
    ftoks = [x.strip() for x in first.strip("<>").split(",")] if "," in first else []
    if first.startswith("<") or (len(ftoks) >= 6 and _isnum(ftoks[0])
                                 and _isnum(ftoks[4])):
        fmts["VisDrone-raw"] += 1
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            t = [x.strip() for x in line.strip("<>").split(",")]
            if len(t) < 6:
                notes["bad_row"] += 1
                continue
            try:
                x, y, w, h = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                cat = int(float(t[5]))
            except ValueError:
                notes["bad_row"] += 1
                continue
            id_hist[f"csv{cat}"] += 1
            name = VISDRONE_RAW_MAP.get(cat, f"rawcat{cat}")
            if name is None:
                notes[f"dropped_class:rawcat{cat}"] += 1
                continue
            uid = unify_name(name)
            if uid is None:
                notes[f"dropped_class:{name}"] += 1
                continue
            rows.append(dict(uid=uid, coords=[x, y, x + w, y, x + w, y + h, x, y + h],
                             normalized=False, difficult=False))
        return fmts, rows, notes, id_hist
    for line in txt.splitlines():
        t = line.split()
        if not t or t[0].startswith(("imagesource", "gsd", "#")):
            continue
        if len(t) == 9 and all(_isnum(x) for x in t):
            cid = int(float(t[0]))
            id_hist[f"cls{cid}"] += 1
            name = id_map.get(cid, f"cls{cid}")
            coords = [float(x) for x in t[1:9]]
            normalized = all(abs(c) <= 2.0 for c in coords)
            fmts["YOLO-OBB" if normalized else "DOTA"] += 1
            uid = unify_name(name)
            if uid is None:
                notes[f"dropped_class:{name}"] += 1
                continue
            rows.append(dict(uid=uid, coords=coords, normalized=normalized,
                             difficult=False))
        elif len(t) >= 10 and _isnum(t[0]) and _isnum(t[7]) and not _isnum(t[8]):
            # original DOTA: x1 y1 x2 y2 x3 y3 x4 y4 class difficult
            name = t[8].lower()
            id_hist[f"name:{name}"] += 1
            fmts["DOTA-v1.x"] += 1
            uid = unify_name(name)
            if uid is None:
                notes[f"dropped_class:{name}"] += 1
                continue
            try:
                coords = [float(x) for x in t[0:8]]
            except ValueError:
                notes["bad_row"] += 1
                continue
            difficult = len(t) >= 10 and t[9] in ("1", "1.0")
            if difficult:
                notes["difficult_kept"] += 1
            rows.append(dict(uid=uid, coords=coords, normalized=False,
                             difficult=difficult))
        elif len(t) == 8 and all(_isnum(x) for x in t):
            notes["skipped_8tok_hbb"] += 1
        else:
            notes["unparsed_row"] += 1
    return fmts, rows, notes, id_hist

def finalize_rows(rows, W, H):
    """normalize pixel rows to [0,1], clip, drop degenerate + exact dups"""
    out, notes = [], Counter()
    for r in rows:
        c = r["coords"]
        if not r["normalized"]:
            c = [c[i] / (W if i % 2 == 0 else H) for i in range(8)]
        c = [min(1.0, max(0.0, v)) for v in c]
        xs, ys = c[0::2], c[1::2]
        wpx = (max(xs) - min(xs)) * W
        hpx = (max(ys) - min(ys)) * H
        if wpx < MIN_BOX_PX or hpx < MIN_BOX_PX:
            notes["degenerate"] += 1
            continue
        out.append((r["uid"], tuple(round(v, 6) for v in c), r["difficult"]))
    seen, dedup = set(), []
    for row in out:
        k = (row[0], row[1])
        if k in seen:
            notes["duplicate_box"] += 1
            continue
        seen.add(k)
        dedup.append(row)
    return dedup, notes

# ------------------------- stratified split ---------------------------------
def stratified_assign(items):
    """
    Schedule-anchored stratified split (v4.1).
    Images are processed rarest-class-first. Image counts are anchored to the
    cumulative split targets within a small slack, so the 82/9/9 image
    proportions always hold. Among anchor-feasible splits, the assignment is
    decided by the image's SINGLE rarest class remaining quota (unclamped),
    which spreads rare classes across val/test even when they occur in very
    few, very large images. Validated on real heavy-tailed data (1..10,066
    boxes/image): image proportions within ~1pp, per-class shares within ~2pp
    of target.
    Returns {key: split}.
    """
    n_items = len(items)
    total = Counter()
    for _, cc in items:
        total.update(cc)
    splits = list(SPLIT_FRAC)
    remaining = {s: {c: total[c] * SPLIT_FRAC[s] for c in total} for s in splits}
    count = {s: 0 for s in splits}
    slack = max(3.0, 0.012 * n_items)
    order = sorted(items, key=lambda kv: min(total[c] for c in kv[1]) if kv[1] else 0)
    assign = {}
    for i, (key, cc) in enumerate(order):
        behind = {s: SPLIT_FRAC[s] * i - count[s] for s in splits}
        cands = [s for s in splits if behind[s] > -slack]
        if not cc:
            best = max(cands, key=lambda s: behind[s])
        else:
            rare = min(cc, key=lambda c: total[c])
            def score(s):
                return (remaining[s].get(rare, 0.0) / max(1, total[rare])
                        + 0.1 * behind[s] / max(1.0, SPLIT_FRAC[s] * n_items))
            best = max(cands, key=lambda s: (score(s), behind[s]))
        for c, n in cc.items():
            remaining[best][c] = remaining[best].get(c, 0.0) - n
        count[best] += 1
        assign[key] = best
    return assign

# ------------------------- phase A ------------------------------------------
def process_one(task):
    src, ds_dir, id_map, img_path, key, cands = task
    rec = dict(src=src, key=key, img=img_path, status=None, md5=None,
               W=0, H=0, rows_in=0, kept=0, fmts=Counter(), notes=Counter(),
               id_hist=Counter(), cls_counts=Counter(), rows=[],
               jpg_path=None, txt_path=None)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if not cands:
        rec["status"] = "unlabeled"
        return rec
    # parse the best label candidate that yields rows (prio order)
    parsed, any_content = None, False
    for prio, lp in cands:
        fmts, rows, notes, id_hist = parse_label_file(lp, id_map)
        rec["fmts"].update(fmts)
        rec["id_hist"].update(id_hist)
        rec["notes"].update(notes)
        if not ("empty_label_file" in notes or "read_error" in notes):
            any_content = True
        if rows:
            parsed = (lp, rows)
            break
    if parsed is None:
        rec["status"] = "empty_label_file" if not any_content else "no_label_rows"
        return rec
    # decode image
    try:
        data = open(img_path, "rb").read()
    except OSError:
        rec["status"] = "read_error"
        return rec
    rec["md5"] = hashlib.md5(data).hexdigest()
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        W, H = im.size
        rec["W"], rec["H"] = W, H
    except Exception:
        rec["status"] = "decode_error"
        return rec
    rows, fnote = finalize_rows(parsed[1], W, H)
    rec["notes"].update(fnote)
    rec["rows_in"] = len(parsed[1])
    if not rows:
        rec["status"] = "no_surviving_boxes"
        return rec
    rec["rows"] = rows
    rec["kept"] = len(rows)
    for uid, _c, _d in rows:
        rec["cls_counts"][UNIFIED[uid]] += 1
    # resize + re-encode
    if max(W, H) > RESIZE_LONG_SIDE:
        s = RESIZE_LONG_SIDE / max(W, H)
        im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY)
    # write to local staging (phase B moves files into split dirs)
    sdir = os.path.join(STAGING, src)
    os.makedirs(sdir, exist_ok=True)
    rec["jpg_path"] = os.path.join(sdir, rec["key"] + ".jpg")
    rec["txt_path"] = os.path.join(sdir, rec["key"] + ".txt")
    with open(rec["jpg_path"], "wb") as f:
        f.write(buf.getvalue())
    write_label_txt(rec["txt_path"], rows)
    return rec

# ------------------------- output helpers -----------------------------------
def write_label_txt(path, rows):
    with open(path, "w") as f:
        for uid, c, _d in rows:
            f.write(f"{uid} " + " ".join(f"{v:.6f}" for v in c) + "\n")

def write_yaml(path, ds_path, with_test=True):
    with open(path, "w") as f:
        f.write(f"path: {ds_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        if with_test:
            f.write("test: test/images\n")
        f.write("names:\n")
        for i, n in enumerate(UNIFIED):
            f.write(f"  {i}: {n}\n")

# ------------------------- main ---------------------------------------------
def main():
    global STAGING
    T0 = datetime.now()
    random.seed(42)
    suffix = "_TEST" if TEST_RUN else ""
    out_local = os.path.join(LOCAL_ROOT, "cleaned_data" + suffix)
    out_drive = os.path.join(OUT_ROOT, "cleaned_data" + suffix)
    if os.path.exists(out_local):
        shutil.rmtree(out_local)
    staging = os.path.join(out_local, "_staging")
    os.makedirs(staging, exist_ok=True)
    STAGING = staging
    log(f"TEST_RUN={TEST_RUN}  workers={MAX_WORKERS}  out={out_local}"
        + (f" -> {out_drive}" if COPY_TO_DRIVE else ""))

    sources = ["codrone", "visdrone", "dota"]
    digest = {}
    global_md5 = {}

    for src in sources:
        ds_dir = os.path.join(DATA_DIR, src)
        if not os.path.isdir(ds_dir):
            log(f"[{src}] MISSING dir: {ds_dir} — skipped")
            digest[src] = {"error": "missing_dir"}
            continue
        ymaps = load_yaml_names(ds_dir)
        id_map, conflicts = merge_id_maps(ymaps, FALLBACK_ID_MAPS.get(src, {}))
        for c in conflicts:
            log(f"[{src}] yaml conflict: {c}")
        log(f"[{src}] id map: "
            + ", ".join(f"{k}->{v}" for k, v in sorted(id_map.items())))
        images, lbl_index, disc_notes = discover_dataset(ds_dir)
        for k, v in sorted(disc_notes.items()):
            log(f"[{src}] NOTE {k}: {v}")
        tasks = []
        n_unlabeled = 0
        stem_counts = Counter(
            os.path.splitext(os.path.basename(p))[0].lower() for p in images)
        for p in images:
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            cands = lbl_index.get(stem, [])
            if not cands:
                n_unlabeled += 1
                continue
            if stem_counts[stem] > 1:   # ambiguous stem -> disambiguate by path
                key = os.path.splitext(os.path.relpath(p, ds_dir))[0].replace(os.sep, "__")
            else:
                key = os.path.splitext(os.path.basename(p))[0]
            tasks.append((src, ds_dir, id_map, p, key, cands))
        log(f"[{src}] images: {len(images)}, with label candidates: {len(tasks)}, "
            f"unlabeled: {n_unlabeled}")
        if TEST_RUN and len(tasks) > TEST_N:
            tasks = random.sample(tasks, TEST_N)
        log(f"[{src}] phase A: processing {len(tasks)} images "
            f"with {MAX_WORKERS} threads ...")
        t0 = datetime.now()
        recs, errs = [], 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, rec in enumerate(ex.map(process_one, tasks), 1):
                recs.append(rec)
                if rec["status"] in ("read_error", "decode_error"):
                    errs += 1
                if i % 1000 == 0:
                    log(f"[{src}]   {i}/{len(tasks)}")
        dt = (datetime.now() - t0).total_seconds()
        log(f"[{src}] phase A done in {dt:.0f}s, errors: {errs}")

        # dedup (keep first by key order)
        recs.sort(key=lambda r: r["key"] or "")
        kept, seen_md5, statuses = [], set(), Counter()
        for r in recs:
            st = r["status"]
            if st in ("unlabeled", "no_label_rows", "empty_label_file",
                      "no_surviving_boxes", "read_error", "decode_error"):
                statuses[st] += 1
                continue
            if r["md5"] in seen_md5:
                statuses["duplicate_image"] += 1
                continue
            if CROSS_SOURCE_DEDUP and r["md5"] in global_md5:
                statuses["cross_source_duplicate"] += 1
                continue
            seen_md5.add(r["md5"])
            global_md5[r["md5"]] = f"{src}:{r['key']}"
            kept.append(r)

        # phase B: split + write
        items = [(r["key"], r["cls_counts"]) for r in kept]
        assign = stratified_assign(items)
        src_out = os.path.join(out_local, src)
        for split in SPLIT_FRAC:
            os.makedirs(os.path.join(src_out, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(src_out, split, "labels"), exist_ok=True)
        agg = defaultdict(Counter)
        split_sizes = Counter()
        write_errors = 0
        dropped = Counter()
        per_class = Counter()
        for r in kept:
            split = assign[r["key"]]
            try:
                shutil.copyfile(r["jpg_path"],
                                os.path.join(src_out, split, "images",
                                             r["key"] + ".jpg"))
                shutil.copyfile(r["txt_path"],
                                os.path.join(src_out, split, "labels",
                                             r["key"] + ".txt"))
            except OSError:
                write_errors += 1
                statuses["write_error"] += 1
                continue
            split_sizes[split] += 1
            agg[split].update(r["cls_counts"])
            per_class.update(r["cls_counts"])
            dropped.update({k: v for k, v in r["notes"].items()})
        write_yaml(os.path.join(src_out, "dataset.yaml"),
                   os.path.join(out_drive if COPY_TO_DRIVE else out_local, src))
        with open(os.path.join(src_out, "classes.txt"), "w") as f:
            f.write("\n".join(UNIFIED) + "\n")
        boxes_in = sum(r["rows_in"] for r in kept) + sum(
            r["rows_in"] for r in recs
            if r["status"] in ("no_surviving_boxes",))
        fmts_total = Counter()
        id_hist_total = Counter()
        for r in recs:
            fmts_total.update(r["fmts"])
            id_hist_total.update(r["id_hist"])
        digest[src] = dict(
            phase_a_errors=errs, notes={**disc_notes},
            images_on_disk=len(images), unlabeled=n_unlabeled, pairs=len(tasks),
            kept=len(kept), statuses=dict(statuses), boxes_in=boxes_in,
            boxes_kept=sum(r["kept"] for r in kept),
            dropped_boxes=dict(dropped), per_class=dict(per_class),
            fmts=dict(fmts_total), id_hist=dict(id_hist_total),
            dups=sum(1 for r in recs if r["status"] == "duplicate_image"),
            split_sizes=dict(split_sizes),
            instances_per_class={s: dict(agg[s]) for s in SPLIT_FRAC},
            write_errors=write_errors)
        log(f"[{src}] kept {len(kept)}/{len(tasks)} | "
            f"splits {dict(split_sizes)} | boxes "
            f"{digest[src]['boxes_kept']}/{boxes_in} | statuses {dict(statuses)}")

    # root-level artifacts
    with open(os.path.join(out_local, "dataset_all.yaml"), "w") as f:
        f.write("# merged: all three cleaned datasets in one training set\n")
        f.write(f"path: {out_drive if COPY_TO_DRIVE else out_local}\n")
        f.write("train:\n")
        for s in sources:
            f.write(f"  - {s}/train/images\n")
        f.write("val:\n")
        for s in sources:
            f.write(f"  - {s}/val/images\n")
        f.write("test:\n")
        for s in sources:
            f.write(f"  - {s}/test/images\n")
        f.write("names:\n")
        for i, n in enumerate(UNIFIED):
            f.write(f"  {i}: {n}\n")
    with open(os.path.join(out_local, "classes.txt"), "w") as f:
        f.write("\n".join(UNIFIED) + "\n")
    with open(os.path.join(out_local, "cleaning_report.json"), "w") as f:
        json.dump(dict(generated=datetime.now(timezone.utc).isoformat(),
                       test_run=TEST_RUN, resize_long_side=RESIZE_LONG_SIDE,
                       min_box_px=MIN_BOX_PX, jpeg_quality=JPEG_QUALITY,
                       workers=MAX_WORKERS, sources=digest), f, indent=2)
    with open(os.path.join(out_local, "CLEANING_SUMMARY.md"), "w") as f:
        f.write("# Cleaning summary\n\n")
        f.write(f"- generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- test_run: {TEST_RUN}, workers: {MAX_WORKERS}\n")
        f.write(f"- Output: {out_drive if COPY_TO_DRIVE else out_local}\n\n")
        for s, d in digest.items():
            if "error" in d:
                f.write(f"## {s}: {d['error']}\n\n")
                continue
            f.write(f"## {s}\n- kept {d['kept']}/{d['pairs']} images, "
                    f"boxes {d['boxes_kept']}/{d['boxes_in']}\n"
                    f"- splits: {d['split_sizes']}\n"
                    f"- statuses: {d['statuses']}\n\n")
    shutil.rmtree(staging, ignore_errors=True)

    # copy to Drive
    files = []
    for root, _d, fs in os.walk(out_local):
        for f in fs:
            files.append(os.path.join(root, f))
    if COPY_TO_DRIVE and os.path.realpath(out_drive) != os.path.realpath(out_local):
        if os.path.exists(out_drive):
            shutil.rmtree(out_drive)
        log(f"\n=== copying {len(files)} files -> {out_drive} "
            f"({COPY_WORKERS} threads) ===")
        def cp(srcf):
            dst = os.path.join(out_drive, os.path.relpath(srcf, out_local))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(srcf, dst)
        with ThreadPoolExecutor(max_workers=COPY_WORKERS) as ex:
            for i, _ in enumerate(ex.map(cp, files), 1):
                if i % 2000 == 0:
                    log(f"  copied {i}/{len(files)}")
        log(f"copied {len(files)} files to Drive")

    print("\n================ DIGEST (copy back to the chat) ================")
    print(json.dumps(dict(out_local=out_local,
                          out_drive=out_drive if COPY_TO_DRIVE else None,
                          files_copied_to_drive=len(files) if COPY_TO_DRIVE else 0,
                          elapsed_s=round((datetime.now() - T0).total_seconds(), 1),
                          sources=digest), indent=1, default=dict))
    print("================ END DIGEST ================")

if __name__ == "__main__":
    main()
