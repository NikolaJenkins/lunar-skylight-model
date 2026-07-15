#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import random
import csv
import cv2

parser = argparse.ArgumentParser(
    description=(
        "Crop tif files around the given pit coordinates."
        "Otherwise randomly crops them"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to directory of tiff images",
)

parser.add_argument(
    "--coords",
    type=str,
    required=False,
    help="Path to csv containing pit coordinates. Don't include if you want to randomly crop the images."
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    help="Path to output directory",
)

def valid_dir(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def valid_file(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
    # elif not path.stem == ".csv":
    #     raise argparse.ArgumentTypeError(f"{path} is not a valid csv file")
    else:
        print(path.stem)
        return path


def has_pit(*, cropped_image: np.ndarray, base_image: np.ndarray):
    """Use on a 640x640 image to roughly distinguish images with pits from images without pits."""

    # calculate threshold for shadowed pixels based on uncropped image
    base_rep = base_image[::50, ::50]
    valid_pixels = base_rep[base_rep > 0]
    base_median_brightness = np.median(valid_pixels)
    base_min_brightness = valid_pixels.min()
    shadow_threshold = base_min_brightness + base_median_brightness * .05
    print("Darkest cropped pixel:", cropped_image.min())
    print("Shadow threshold:", shadow_threshold)

    shadow_pixel_num = np.sum(cropped_image < shadow_threshold)
    if cropped_image.min() <= shadow_threshold:
        shadow_pixel_num = np.sum(cropped_image <= shadow_threshold)
        return shadow_pixel_num > 15

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))
    tiff_counter = 0
    if args.coords is not None:
        pit_csv = valid_file(Path(args.coords))
        pit_csv = pd.read_csv(pit_csv, index_col = 0)
        pit_dict = pit_csv.to_dict(orient = "split")
        coords_dict = dict(zip(pit_dict["index"], pit_dict["data"]))
        for tiff in input_dir.iterdir():
            image = cv2.imread(tiff, -1)
            print(type(image))
            key = tiff.stem.upper()[:-1]
            pit_y, pit_x = coords_dict[key]
            crop_offset_x = random.randint(20, 620)
            crop_offset_y = random.randint(20, 620)
            cropped_pit_x = -crop_offset_x
            cropped_pit_y = -crop_offset_y
            # subprocess.run([
            #     "magick",
            #     tiff,
            #     "-crop",
            #     f"640x640+{pit_x + crop_offset_x}+{pit_y + crop_offset_y}",
            #     f"{output_dir}/{tiff.stem}_cropped.tif",
            # ])
            cropped_image = image[
                pit_x + crop_offset_x - 320 : pit_x + crop_offset_y + 320,
                pit_y + crop_offset_y - 320 : pit_y + crop_offset_y + 320
            ]
            print(cropped_image.shape)
            # print("Cropped image has pit?:", has_pit(cropped_image = cropped_image, base_image = image))
            # cv2.imwrite(
            #     f"{output_dir}/{tiff.stem}_cropped.png",
            #     cropped_image
            # )

            # save the pits coordinates in the cropped image as a csv
            pit_coords_data = [
                ["sample (x)", "line (y)"],
                [cropped_pit_x, cropped_pit_y]
            ]
            with open(
                f"{output_dir}/{tiff.stem}_pit_coords.csv",
                mode = "w",
                newline = "",
                encoding = "utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerows(pit_coords_data)

            tiff_counter += 1
            print(f"Number of images cropped: {tiff_counter}")
            if tiff_counter == 1:
                break

    # TODO: add random crop if csv isn't provided
    else:
        # for tiff in input_dir.iterdir():
        #     image = cv2.imread(tiff)
        #     height, width = image.shape[:2]
        #     crop_offset_x = random.randint(0, width - 640)
        #     crop_offset_y = random.randint(0, height - 640)
        #     cropped_image = image[
        #         crop_offset_x : crop_offset_x + 640,
        #         crop_offset_y: crop_offset_y + 640
        #     ]
        #     print(f"x, y offsets: {crop_offset_x}, {crop_offset_y}")
        #     print(cropped_image.shape)
        #     # subprocess.run([
        #     #     "magick",
        #     #     tiff,
        #     #     "-crop",
        #     #     f"640x640+{crop_offset_x}+{crop_offset_y}",
        #     #     f"{output_dir}/{tiff.stem}_cropped_random.tif",
        #     # ])
        #     print(f"Cropped image has a pit?: {has_pit(cropped_image = cropped_image, base_image = image)}")
        #     print(f"Image {tiff.name} cropped")
        #     tiff_counter += 1
        #     if tiff_counter == 1:
        #         break
        base_image = cv2.imread(args.input_base, -1)
        cropped_image = cv2.imread(args.input_tif, -1)
        print(f"Cropped image has a pit?: {has_pit(cropped_image = cropped_image, base_image = base_image)}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
