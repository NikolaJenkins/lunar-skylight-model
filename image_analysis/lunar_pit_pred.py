#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import rasterio
import torch
import numpy as np
from convert_16_bit_png import convert_16_bit_image
from helper_functions import valid_dir, valid_file
from ultralytics import YOLO

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
    print("Using model:", model_path.name)

    # load device gpu and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    image_counter = 0
    for image in images_dir.iterdir():
        if "random" not in image.name and image.suffix == ".IMG":
            with rasterio.open(image) as uncropped_image:
                print("Reading image", image.name, type(uncropped_image))
                # print("image dimensions:", uncropped_image.shape)
                height, width = uncropped_image.shape
                tile_counter = 0
                for y in range(0, height - 640, 500):
                    for x in range(0, width - 640, 500):
                        tile_counter += 1
                        rast_window = rasterio.windows.Window(x, y, 640, 640)
                        tile = uncropped_image.read(1, window = rast_window)

                        # convert tile to 16 bit image
                        tile_16_bit = tile.astype(np.uint16)
                        tile_mean, tile_std = tile_16_bit.mean(), tile_16_bit.std()
                        pixel_min = max(tile_16_bit.min(), tile_mean - 2 * tile_std)
                        pixel_max = min(tile_16_bit.max(), tile_mean + 2 * tile_std)
                        if pixel_max - pixel_min <= 0:
                            continue
                        # clipped_tile =

                        # convert 16 bit tile to 8 bit tile


                        # run inference on images, confirm they're not randomly cropped
                        # results = model(
                        #     source = image,
                        #     conf = 0.2,
                        #     imgsz = 640,
                        #     device = device,
                        #     max_det = 1,
                        #     name = args.output,
                        #     save = True,
                        #     save_txt = True,
                        #     save_conf = True,
                        # )
                        tile_counter += 1
                        if tile_counter == 1:
                            break
                    break

            image_counter += 1
            print(tile_counter, "tiles were produced")
        if image_counter == 1:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
