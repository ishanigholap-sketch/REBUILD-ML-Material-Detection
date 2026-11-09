
from pathlib import Path
import random
import shutil
import yaml

# --------------------------------------------------
# RE:BUILD - DATASET PREPARATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
DATASETS = BASE / "datasets"
OUTPUT = DATASETS / "rebuild_yolo"

SEED = 42
random.seed(SEED)

CLASSES = [
    "concrete",
    "brick",
    "steel",
    "wood",
    "glass",
    "tiles",
]

CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Original class names -> RE:BUILD class names.
MAPPINGS = {
    "building_components": {
        "concrete": "concrete",
        "brick": "brick",
        "glass": "glass",
        "metal": "steel",       # Provisional mapping
        "wooden": "wood",
    },
    "building_materials": {
        "brick": "brick",
        "tile": "tiles",
        "timber plank": "wood",
    },
    "rmit_demolition": {
        "brick": "brick",
        "glass": "glass",
        "rebar": "steel",
        "tile": "tiles",
        "tiles": "tiles",
        "timber": "wood",
    },
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}


def convert_annotation(line, class_names, mapping):
    """Convert a YOLO box or polygon into a YOLO bounding box."""

    parts = line.strip().split()

    if not parts:
        return None

    try:
        values = [float(x) for x in parts]
        original_id = int(values[0])
    except (ValueError, IndexError):
        return None

    if original_id < 0 or original_id >= len(class_names):
        return None

    original_name = str(class_names[original_id]).strip().lower()
    target_name = mapping.get(original_name)

    if target_name not in CLASS_ID:
        return None

    coords = values[1:]

    # Existing bounding box: class + 4 coordinates.
    if len(coords) == 4:
        cx, cy, width, height = coords

        if width <= 0 or height <= 0:
            return None

        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

    # Polygon: class + 3 or more coordinate pairs.
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

    else:
        return None

    # Validate normalized bounding-box coordinates.
    if not all(0 <= v <= 1 for v in [cx, cy, width, height]):
        return None

    return (
        f"{CLASS_ID[target_name]} "
        f"{cx:.6f} {cy:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def process_dataset(dataset_name):
    dataset_path = DATASETS / dataset_name
    yaml_path = dataset_path / "data.yaml"

    if not yaml_path.exists():
        print(f"WARNING: Missing YAML: {dataset_name}")
        return []

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    class_names = config["names"]

    if isinstance(class_names, dict):
        class_names = [
            class_names[i] for i in range(len(class_names))
        ]

    mapping = MAPPINGS[dataset_name]
    records = []

    for split in ["train", "valid", "test"]:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            converted = []

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():

                result = convert_annotation(
                    line, class_names, mapping
                )

                if result:
                    converted.append(result)

            # Ignore images containing no selected RE:BUILD class.
            if converted:
                records.append({
                    "source": dataset_name,
                    "image": image_path,
                    "annotations": converted,
                })

    print(f"{dataset_name}: {len(records)} useful images")
    return records


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or remove it "
            "only after checking its contents."
        )

    all_records = []

    for dataset_name in MAPPINGS:
        all_records.extend(process_dataset(dataset_name))

    if not all_records:
        raise RuntimeError("No usable images were found.")

    random.shuffle(all_records)

    total = len(all_records)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    splits = {
        "train": all_records[:train_end],
        "valid": all_records[train_end:valid_end],
        "test": all_records[valid_end:],
    }

    counts = {name: [0] * len(CLASSES) for name in splits}

    for split_name, records in splits.items():
        image_dir = OUTPUT / split_name / "images"
        label_dir = OUTPUT / split_name / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(records):
            filename = (
                f"{record['source']}_{index:05d}"
                f"{record['image'].suffix.lower()}"
            )

            label_filename = Path(filename).with_suffix(".txt")

            shutil.copy2(
                record["image"],
                image_dir / filename
            )

            (label_dir / label_filename).write_text(
                "\n".join(record["annotations"]) + "\n",
                encoding="utf-8"
            )

            for annotation in record["annotations"]:
                class_id = int(annotation.split()[0])
                counts[split_name][class_id] += 1

    # Write Ultralytics dataset configuration.
    yaml_content = {
        "path": OUTPUT.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }

    with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    print("\n========== RE:BUILD DATASET ==========")
    print("Total useful images:", total)

    for split_name, records in splits.items():
        print(f"\n{split_name.upper()}: {len(records)} images")

        for class_id, name in enumerate(CLASSES):
            print(f"  {name}: {counts[split_name][class_id]}")

    print("\nDataset created at:", OUTPUT)
    print("======================================")


if __name__ == "__main__":
    main()
