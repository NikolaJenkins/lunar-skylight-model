#!/usr/bin/env python3
import argparse
from helper_functions import valid_dir
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description=(
        "Downsample 16 bit png images to 8 bit so YOLO can analyze them"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to directory of 16 bit png images",
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    help="Path to output directory to store 8 bit png images",
)

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))
    png_counter = 0
    for png in input_dir.iterdir():
        print("Reading file:", png.name)
        image_16bit = cv2.imread(png, cv2.IMREAD_UNCHANGED)

        # check that image is actually 16 bit
        if image_16bit.dtype == np.uint16:
            print(f"{png.name} is a 16 bit image")
            # map values between 0 and 255
            image_min = image_16bit.min()
            image_max = image_16bit.max()
            if image_max - image_min > 0:
                image_float = image_16bit.astype(np.float32)
                image_8bit = ((image_float - image_min) / (image_max - image_min) * 255.0).astype(np.uint8)
            else:
                image_8bit = np.zeros_like(image_16bit, dtype = np.uint8)

        # write new 8 bit png to output folder
        cv2.imwrite(output_dir / png.name, image_8bit)

        png_counter += 1
        print(f"{png_counter} images stretched to 8 bit")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
