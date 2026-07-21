#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    # start from pretrained checkpoint
    model_name = "yolo26n-seg.pt"
    print("Starting from model:", model_name)
    model = YOLO(model_name)
    results = model.train(
        data = "/mnt/storage/images/lunar_dataset/lunar_pits.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 16,
        device = "cuda",
        name = "lunar_pit_v1",
        hsv_h = 0.0,
        hsv_s = 0.0,
        hsv_v = 0.4,
        degrees = 15.0,
        scale = 0.5,
        flipud = 0.5,
        fliplr = 0.5,
    )

    # Path to best checkpoint
    print(results.save_dir)

if __name__ == "__main__":
    main()
