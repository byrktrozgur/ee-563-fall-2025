import json
import os
import cv2


DATA_DIR = "balloon_dataset"

# Directories for train and val
SPLITS = {
    "train": {
        "img_dir": os.path.join(DATA_DIR, "images", "train"),
        "label_dir": os.path.join(DATA_DIR, "labels", "train"),
        "via_json": os.path.join(DATA_DIR, "labels", "train", "train_via_region_data.json"),
    },
    "val": {
        "img_dir": os.path.join(DATA_DIR, "images", "val"),
        "label_dir": os.path.join(DATA_DIR, "labels", "val"),
        "via_json": os.path.join(DATA_DIR, "labels", "val", "val_via_region_data.json"),
    },
}

# Make sure folders exist
for split in SPLITS.values():
    os.makedirs(split["label_dir"], exist_ok=True)


def load_via_data(path):
    with open(path, "r") as f:
        via_data = json.load(f)
    return via_data


def via_regions_iter(regions):
    return regions.values() if isinstance(regions, dict) else regions


def convert_to_yolo(via_data, img_dir, label_dir, class_id=0):
    print(f"Converting to YOLO TXT: {img_dir}")

    for key, item in via_data.items():
        filename = item["filename"]
        regions = item["regions"]

        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("Missing image:", img_path)
            continue

        h, w = img.shape[:2]

        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        lines = []

        for r in via_regions_iter(regions):
            xs = r["shape_attributes"]["all_points_x"]
            ys = r["shape_attributes"]["all_points_y"]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_c = (x_min + x_max) / (2 * w)
            y_c = (y_min + y_max) / (2 * h)
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            polygon = []
            for x, y in zip(xs, ys):
                polygon += [x / w, y / h]

            line = [str(class_id), f"{x_c:.6f}", f"{y_c:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + \
                   [f"{p:.6f}" for p in polygon]
            lines.append(" ".join(line))

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print("Finished YOLO conversion.\n")


def convert_to_coco(via_data, img_dir, output_json, category_name="balloon"):
    print(f"Converting to COCO: {img_dir}")

    images = []
    annotations = []
    categories = [{"id": 1, "name": category_name, "supercategory": "object"}]

    image_id = 1
    ann_id = 1

    for key, item in via_data.items():
        filename = item["filename"]
        regions = item["regions"]

        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("Missing image:", img_path)
            continue

        h, w = img.shape[:2]

        images.append({
            "id": image_id,
            "file_name": filename,
            "height": h,
            "width": w
        })

        for r in via_regions_iter(regions):
            xs = r["shape_attributes"]["all_points_x"]
            ys = r["shape_attributes"]["all_points_y"]

            seg = []
            for x, y in zip(xs, ys):
                seg.extend([float(x), float(y)])

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bw = float(x_max - x_min)
            bh = float(y_max - y_min)

            # polygon area
            area = 0
            for i in range(len(xs)):
                x1, y1 = xs[i], ys[i]
                x2, y2 = xs[(i + 1) % len(xs)], ys[(i + 1) % len(ys)]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [seg],
                "area": area,
                "bbox": [float(x_min), float(y_min), bw, bh],
                "iscrowd": 0
            })
            ann_id += 1

        image_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print("Finished COCO conversion.\n")


def convert_via(format_choice=None):
    if format_choice is None:
        print("Choose output format:\n 1) YOLO TXT\n 2) COCO JSON")
        choice = input("Enter 1 or 2: ").strip()
    else:
        choice = "1" if format_choice.lower().startswith("yolo") else "2"

    for split, paths in SPLITS.items():
        print(f"=== Processing {split.upper()} split ===")

        if not os.path.exists(paths["via_json"]):
            print(f"Missing VIA file: {paths['via_json']}\n")
            continue

        via_data = load_via_data(paths["via_json"])

        if choice == "1":
            convert_to_yolo(via_data, paths["img_dir"], paths["label_dir"])

        elif choice == "2":
            coco_path = os.path.join(DATA_DIR, f"annotations_coco_{split}.json")
            convert_to_coco(via_data, paths["img_dir"], coco_path)

        else:
            print("Invalid format choice.")
            return


if __name__ == "__main__":
    convert_via()
