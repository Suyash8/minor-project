"""
Synthetic Aerial OBB Dataset Generator for Instant CPU Testing (--test mode).
Generates synthetic aerial road scenes with perspective tilt, dense vehicles,
and challenging road clutter to test the entire pipeline in seconds.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from src.config import DATASET_CLASSES
from src.metrics.geometry import obb_to_corners

def generate_mock_scene(
    img_size: Tuple[int, int] = (640, 640),
    num_objects: int = 15,
    class_names: List[str] = None,
    tilt_angle_deg: float = 30.0,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Synthesize an aerial image with roads, lane lines, shadows, and oriented vehicles.
    """
    if class_names is None:
        class_names = ["car", "van", "truck", "bus", "pedestrian"]

    w, h = img_size
    # Background: asphalt gray with subtle texture
    img = Image.new("RGB", (w, h), color=(55, 60, 65))
    draw = ImageDraw.Draw(img)

    # Road boundaries & lane markings
    road_width = int(w * 0.65)
    road_x1 = (w - road_width) // 2
    road_x2 = road_x1 + road_width
    draw.rectangle([road_x1, 0, road_x2, h], fill=(45, 48, 52))

    # Dashed lane divider
    dash_len = 25
    gap_len = 20
    center_x = w // 2
    y_curr = 0
    while y_curr < h:
        draw.line([(center_x, y_curr), (center_x, min(y_curr + dash_len, h))], fill=(220, 220, 225), width=3)
        y_curr += dash_len + gap_len

    # Diagonal shadow clutter (simulating tall buildings or trees)
    shadow_poly = [(road_x1 - 40, 0), (road_x1 + 180, 0), (road_x1 + 60, h), (road_x1 - 80, h)]
    draw.polygon(shadow_poly, fill=(35, 38, 42))

    objects = []
    # Color palette for synthetic vehicles
    vehicle_colors = [(220, 40, 40), (40, 100, 220), (230, 230, 230), (30, 30, 30), (220, 180, 30), (40, 180, 80)]

    for i in range(num_objects):
        cls_idx = random.randint(0, len(class_names) - 1)
        cls_name = class_names[cls_idx]

        # Vehicle dimensions based on class and altitude scale
        if cls_name in ["pedestrian", "cyclist"]:
            box_w = random.uniform(8.0, 16.0)
            box_h = random.uniform(10.0, 20.0)
        elif cls_name in ["truck", "bus"]:
            box_w = random.uniform(25.0, 40.0)
            box_h = random.uniform(60.0, 95.0)
        else: # car, van
            box_w = random.uniform(18.0, 28.0)
            box_h = random.uniform(35.0, 55.0)

        # Apply foreshortening due to oblique tilt angle
        tilt_factor = math.cos(math.radians(tilt_angle_deg))
        box_h = box_h * tilt_factor

        cx = random.uniform(road_x1 + box_w, road_x2 - box_w)
        cy = random.uniform(box_h, h - box_h)

        # Realistic road orientation: vehicles travel mostly along the road (e.g. around 90 deg or 270 deg with jitter)
        heading_base = random.choice([80.0, 100.0, 260.0, 280.0])
        angle_deg = (heading_base + random.uniform(-15.0, 15.0)) % 180.0

        corners = obb_to_corners(cx, cy, box_w, box_h, angle_deg)
        polygon_pts = [(float(pt[0]), float(pt[1])) for pt in corners]

        # Draw vehicle body
        col = random.choice(vehicle_colors)
        draw.polygon(polygon_pts, fill=col, outline=(15, 15, 15))

        # Windshield marker to indicate front heading
        front_pt1 = corners[0]
        front_pt2 = corners[1]
        mid_front = ((front_pt1[0] + front_pt2[0]) / 2, (front_pt1[1] + front_pt2[1]) / 2)
        draw.ellipse([mid_front[0] - 2, mid_front[1] - 2, mid_front[0] + 2, mid_front[1] + 2], fill=(255, 255, 0))

        objects.append({
            "cx": float(cx),
            "cy": float(cy),
            "w": float(box_w),
            "h": float(box_h),
            "angle_deg": float(angle_deg),
            "class_id": cls_idx,
            "class_name": cls_name,
            "corners": corners,
        })

    return img, objects

def create_mock_dataset(
    dataset_name: str,
    output_dir: Path,
    num_train: int = 5,
    num_val: int = 4,
    img_size: Tuple[int, int] = (640, 640),
) -> Path:
    """
    Generate and save a complete mock dataset with images and YOLO-OBB / DOTA annotations.
    """
    dataset_dir = Path(output_dir) / dataset_name
    class_names = DATASET_CLASSES.get(dataset_name.lower(), ["car", "van", "truck", "bus", "pedestrian"])

    for split, count in [("train", num_train), ("val", num_val)]:
        split_img_dir = dataset_dir / "images" / split
        split_lbl_dir = dataset_dir / "labels" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img_name = f"{dataset_name}_{split}_{idx:04d}"
            img, objects = generate_mock_scene(img_size=img_size, num_objects=random.randint(8, 16), class_names=class_names)

            # Save image
            img_path = split_img_dir / f"{img_name}.jpg"
            img.save(img_path, quality=92)

            # Save YOLO-OBB annotation format (class_id x1 y1 x2 y2 x3 y3 x4 y4 normalized)
            lbl_path = split_lbl_dir / f"{img_name}.txt"
            w_img, h_img = img_size
            lines = []
            for obj in objects:
                corners = obj["corners"]
                norm_coords = []
                for pt in corners:
                    norm_coords.append(f"{pt[0] / w_img:.6f}")
                    norm_coords.append(f"{pt[1] / h_img:.6f}")
                coords_str = " ".join(norm_coords)
                lines.append(f"{obj['class_id']} {coords_str}")

            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))

    # Write dataset yaml for Ultralytics YOLO-OBB compatibility
    yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(dataset_dir / f"{dataset_name}.yaml", "w") as f:
        f.write(yaml_content)

    return dataset_dir
