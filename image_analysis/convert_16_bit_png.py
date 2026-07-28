#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from cv2.typing import MatLike
import numpy as np
from helper_functions import valid_dir

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
    required=True,
    help="Path to output directory to store 8 bit png images",
)

def convert_16_bit_image(image_16_bit: MatLike):
    # map values between 0 and 255
    image_float = image_16_bit.astype(np.float32)
    image_mean, image_std = image_float.mean(), image_float.std()
    image_min = max(image_float.min(), image_mean - 2 * image_std)
    image_max = min(image_float.max(), image_mean + 2 * image_std)
    if image_max - image_min > 0:
        return ((np.clip(image_float, image_min, image_max) - image_min) / (image_max - image_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(image_float, dtype = np.uint8)

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))
    png_counter = 0
    for png in input_dir.iterdir():
        if png.suffix == ".png":
            print("Reading file:", png.name)
            image_16bit = cv2.imread(png, cv2.IMREAD_UNCHANGED)
            image_8bit = convert_16_bit_image(image_16bit)

            # write new 8 bit png to output folder
            cv2.imwrite(output_dir / png.name, image_8bit)

            png_counter += 1
        print(f"{png_counter} images stretched to 8 bit")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
