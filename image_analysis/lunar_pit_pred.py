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
import numpy as np

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

def broad_tile_prediction(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    macro_size = 4000,
    tile_size = 640):
    with rasterio.open(image_path) as uncropped_image:
        print("Reading image", image_path.name, type(uncropped_image))
        # image_pred_dir = Path(output_dir / image_path.stem)
        # image_pred_dir.mkdir(parents = True, exist_ok = True)
        height, width = uncropped_image.shape
        tile_counter = 0
        macro_stride = macro_size - 40
        tile_stride = tile_size - 40
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for macro_y in range(0, height - macro_size, macro_stride):
            for macro_x in range(0, width - macro_size, macro_stride):
                tile_counter += 1
                macro_window = rasterio.windows.Window(macro_x, macro_y, macro_size, macro_size)
                macro = uncropped_image.read(1, window = macro_window)
                macro_mean = macro.mean()
                macro_std = macro.std()
                macro_pixel_min = max(macro.min(), macro_mean - 2 * macro_std)
                macro_pixel_max = min(macro.max(), macro_mean + 2 * macro_std)

                if macro_pixel_max - macro_pixel_min > 0:
                    tile_y = 0
                    while 0 <= tile_y < macro_size:
                        tile_x = 0
                        while 0<= tile_x < macro_size:
                            tile_16_bit = macro[tile_x : tile_x + tile_size, tile_y : tile_y + tile_size]
                            tile_float = tile_16_bit.astype(np.uint32)
                            tile_8_bit = ((np.clip(tile_float, macro_pixel_min, macro_pixel_max) - macro_pixel_min) / (macro_pixel_max - macro_pixel_min) * 255.0).astype(np.uint8)
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
                            result = results[0]
                            detected_pit_num = len(result.boxes)
                            if 0 < detected_pit_num <= 2:
                                print(f"pit discovered at ({macro_x + tile_x}, {macro_y + tile_y})")
                                x_min, y_min, x_max, y_max = result.boxes.xyxy.cpu().numpy()[0].astype(int)
                                confidence = result.boxes.conf.item()
                                print("box coordinates:", x_min, y_min, x_max, y_max)
                                cv2.rectangle(tile_8_bit_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                                cv2.putText(
                                    tile_8_bit_rgb,
                                    f"Pit: {confidence:.2f}",
                                    (x_min, max(20, y_min - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 0, 0),
                                    2
                                )
                                # cv2.imwrite(image_pred_dir / f"{image.stem}_{macro_x + tile_x}_{macro_y + tile_y}.png", tile_8_bit_rgb)
                                cv2.imshow("Image tile", tile_8_bit_rgb)

                            tile_counter += 1
                            if tile_counter == 1:
                                break
                            # ensure tile_x is always less than macro size while not ignoring end strips
                            if macro_size - tile_size < tile_x < macro_size:
                                tile_x = macro_size - tile_size
                            elif tile_x < macro_size:
                                tile_x += tile_stride
                            else:
                                break
                        break
                        # ensure tile_y is always less than macro size while not ignoring end strips
                        if macro_size - tile_size < tile_y < macro_size:
                            tile_y = macro_size - tile_size
                        elif tile_y < macro_size:
                            tile_y += tile_stride
                        else:
                            break
                break
            break
                # if tile_counter == 1:
                    # break
    image_counter += 1
    print(tile_counter, "tiles were produced")
    print(f"Inferencing took {total_time // 60} minutes and {total_time % 60} seconds")

def main(args = argparse.Namespace):
    # Load directory and file paths
    model_path = valid_file(Path(args.model))
    images_dir = valid_dir(Path(args.images))
    output_dir = valid_dir(Path(args.output))
    print("Using model:", model_path.name)

    # load device gpu and model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    image_counter = 0
    for image in images_dir.iterdir():
        if "random" not in image.name and image.suffix == ".IMG":
            start_time = time.perf_counter()
            broad_tile_prediction(
                image_path = image,
                model = model,
                output_dir = output_dir,
            )
            # with rasterio.open(image) as uncropped_image:
            #     start_time = time.perf_counter()
            #     print("Reading image", image.name, type(uncropped_image))
            #     image_pred_dir = Path(output_dir / image.stem)
            #     image_pred_dir.mkdir(parents = True, exist_ok = True)
            #     height, width = uncropped_image.shape
            #     tile_counter = 0
            #     for y in range(0, height - 640, 600):
            #         for x in range(0, width - 640, 600):
            #             tile_counter += 1
            #             rast_window = rasterio.windows.Window(x, y, 640, 640)
            #             tile = uncropped_image.read(1, window = rast_window)

            #             # TODO: look at 4000x4000 tiles to get more general mean and std
            #             # convert tile to 8 bit image
            #             tile_8_bit = convert_16_bit_image(tile)
            #             tile_8_bit_rgb = cv2.cvtColor(tile_8_bit, cv2.COLOR_GRAY2RGB)

            #             # run inference on images, confirm they're not randomly cropped
            #             results = model(
            #                 source = tile_8_bit_rgb,
            #                 conf = 0.80,
            #                 iou = 0.45,
            #                 imgsz = 640,
            #                 device = device,
            #                 max_det = 1,
            #             )
            #             # print("Found pit:", results[0].boxes)
            #             result = results[0]
            #             detected_pit_num = len(result.boxes)
            #             if 0 < detected_pit_num <= 2:
            #                 print(f"pit discovered at ({x}, {y})")
            #                 # print("box result type:", type(result.boxes))
            #                 x_min, y_min, x_max, y_max = result.boxes.xyxy.cpu().numpy()[0].astype(int)
            #                 confidence = result.boxes.conf.item()
            #                 print("box coordinates:", x_min, y_min, x_max, y_max)
            #                 cv2.rectangle(tile_8_bit_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            #                 cv2.putText(
            #                     tile_8_bit_rgb,
            #                     f"Pit: {confidence:.2f}",
            #                     (x_min, max(20, y_min - 10)),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.6,
            #                     (255, 0, 0),
            #                     2
            #                 )
            #                 cv2.imwrite(image_pred_dir / f"{image.stem}_{x}_{y}.png", tile_8_bit_rgb)

            #             tile_counter += 1
            #             # if tile_counter == 1:
            #                 # break
            end_time = time.perf_counter()
            total_time = end_time - start_time
            image_counter += 1
            # print(tile_counter, "tiles were produced")
            print(f"Inferencing took {total_time // 60} minutes and {total_time % 60} seconds")
            if image_counter == 1:
                break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
