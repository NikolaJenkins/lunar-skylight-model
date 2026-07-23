#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    # start from pretrained checkpoint
    model_name = "yolo26n-seg.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting from model:", model_name)
    model = YOLO(model_name)
    results = model.train(
        data = "/mnt/storage/images/lunar_dataset/lunar_pits.yaml",
        epochs = 150,
        imgsz = 640,
        batch = 16,
        device = device,
        name = "lunar_pit_v1",
        hsv_h = 0.0,
        hsv_s = 0.0,
        hsv_v = 0.4,
        degrees = 15.0,
        scale = 0.5,
        flipud = 0.5,
        fliplr = 0.5,
        box = 12.5,
        cls = 2.0,
    )

    # Path to best checkpoint
    print(results.save_dir)

if __name__ == "__main__":
    main()
