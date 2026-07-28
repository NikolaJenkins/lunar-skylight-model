#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import rasterio
import torch
from convert_16_bit_png import convert_16_bit_image
from helper_functions import valid_dir, valid_file
from ultralytics import YOLO
import time

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
    required = False,
    help = "Name of directory to store image inferences"
)

def main(args = argparse.Namespace):
    # Load directory and file paths
    model_path = valid_file(Path(args.model))
    images_dir = valid_dir(Path(args.images))
    output_dir = valid_dir(Path(args.output))
    print("Using model:", model_path.name)

    # load device gpu and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    image_counter = 0
    for image in images_dir.iterdir():
        if "random" not in image.name and image.suffix == ".IMG":
            with rasterio.open(image) as uncropped_image:
                start_time = time.perf_counter()
                print("Reading image", image.name, type(uncropped_image))
                image_pred_dir = Path(output_dir / image.stem)
                image_pred_dir.mkdir(parents = True, exist_ok = True)
                height, width = uncropped_image.shape
                tile_counter = 0
                for y in range(0, height - 640, 600):
                    for x in range(0, width - 640, 600):
                        tile_counter += 1
                        rast_window = rasterio.windows.Window(x, y, 640, 640)
                        tile = uncropped_image.read(1, window = rast_window)

                        # TODO: look at 4000x4000 tiles to get more general mean and std
                        # convert tile to 8 bit image
                        tile_8_bit = convert_16_bit_image(tile)
                        tile_8_bit_rgb = cv2.cvtColor(tile_8_bit, cv2.COLOR_GRAY2RGB)

                        # run inference on images, confirm they're not randomly cropped
                        results = model(
                            source = tile_8_bit_rgb,
                            conf = 0.80,
                            iou = 0.45,
                            imgsz = 640,
                            device = device,
                            max_det = 1,
                        )
                        # print("Found pit:", results[0].boxes)
                        result = results[0]
                        detected_pit_num = len(result.boxes)
                        if 0 < detected_pit_num <= 2:
                            print(f"pit discovered at ({x}, {y})")
                            # print("box result type:", type(result.boxes))
                            x_min, y_min, x_max, y_max = result.boxes.xyxy.cpu().numpy()[0].astype(int)
                            confidence = result.boxes.conf.item()
                            print("box coordinates:", x_min, y_min, x_max, y_max)
                            cv2.rectangle(tile_8_bit_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(
                                tile_8_bit_rgb,
                                f"Pit: {confidence:.2f}",
                                (x_min, max(20, y_min - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                            cv2.imwrite(image_pred_dir / f"{image.stem}_{x}_{y}.png", tile_8_bit_rgb)

                        tile_counter += 1
                        # if tile_counter == 1:
                            # break
            end_time = time.perf_counter()
            total_time = end_time - start_time
            image_counter += 1
            print(tile_counter, "tiles were produced")
            print(f"Inferencing took {total_time // 60} minutes and {total_time % 60} seconds")
        if image_counter == 10:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
