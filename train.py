
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(data_yaml):

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_yaml}"
        )

    print("Starting RE:BUILD YOLO training...")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="rebuild-materials",
        exist_ok=True,
        patience=10,
        workers=0
    )

    best_weights = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    print("Training completed!")
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset data.yaml"
    )

    args = parser.parse_args()

    train_model(args.data)
