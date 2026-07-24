#!/usr/bin/env python3
from ultralytics import YOLO
from PIL import Image
import argparse
from pathlib import Path
from helper_functions import valid_file, valid_dir
import torch

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--model",
    type = str,
    required = True,
    help = "Path to lunar pit model"
)

parser.add_argument(
    "--images",
    type = str,
    required = True,
    help = "Path to directory of images to run inferences on"
)

parser.add_argument(
    "--output",
    type = str,
    required = True,
    help = "Name of directory to store image inferences"
)

def main(args = argparse.Namespace):
    # Load directory and file paths
    model_path = valid_file(Path(args.model))
    images_dir = valid_dir(Path(args.images))
    print("Using model:", model_path.name)

    # load device gpu and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    image_counter = 0
    for image in images_dir.iterdir():
        if "random" not in image.name and image.suffix == ".png":
            # run inference on images, confirm they're not randomly cropped
            results = model(
                source = image,
                conf = 0.1,
                imgsz = 640,
                device = device,
                max_det = 1,
                name = args.output,
                save = True,
                save_txt = True,
                save_conf = True
            )
            image_counter += 1
        if image_counter == 5:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
