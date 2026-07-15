#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(
    description=(
        "Convert tiff file to jpg while preserving important details such as colors and contrast"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to directory containing tiff files to convert",
)

def valid_dir(path: str):
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not Path(path).is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def main(args: argparse.Namespace):
    # turn input into image
    image_dir = Path(args.input)

    image_counter = 0
    for image in image_dir.iterdir():
        if image.suffix == ".tif":
            print("tiff path: ", image)
            print(f"Image path: {image_dir}/{image.stem}.png")
            subprocess.run([
                "gdal_translate",
                "-of",
                "PNG",
                "-ot",
                "UInt16",
                image,
                f"{image_dir}/{image.stem}.png"
            ])
            image_counter += 1
            print(f"{image_counter} images have been converted")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
