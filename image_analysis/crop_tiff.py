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
    shadow_threshold = base_min_brightness + base_median_brightness * .15

    # check density of shadowed pixels
    shadowed_pixels = (cropped_image < shadow_threshold).astype(np.uint8)
    num_shadowed_pixels = np.sum(shadowed_pixels)
    tile_size = 25
    density_image = cv2.boxFilter(shadowed_pixels.astype(float), -1, (tile_size, tile_size), normalize = False)
    max_shadow_cluster = np.max(density_image)
    is_dense_shadow = max_shadow_cluster >= 15

    return cropped_image.min() <= shadow_threshold and num_shadowed_pixels >= 10 and is_dense_shadow

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
            # read image
            image = cv2.imread(tiff, -1)
            height, width = image.shape[:2]
            key = tiff.stem.upper()[:-1]
            pit_y, pit_x = coords_dict[key]

            # random offset so model doesn't always train on pits in center of image
            crop_offset_x = random.randint(-300, 300)
            crop_offset_y = random.randint(-300, 300)

            # pit coordinates in cropped image
            cropped_pit_x = 320 - crop_offset_x
            cropped_pit_y = 320 - crop_offset_y

            # center of cropped image
            cropped_center_x = pit_x + crop_offset_x
            cropped_center_y = pit_y + crop_offset_y

            # some pits are close to edge of image so this prevents index out of bound errors
            if cropped_center_x < 320:
                cropped_center_x = 320
            elif cropped_center_x > height - 320:
                cropped_center_x = height - 320
            if cropped_center_y < 320:
                cropped_center_y = 320
            elif cropped_center_y > width - 320:
                cropped_center_y = width - 320

            # generate cropped image and write to png file in output directory
            cropped_image = image[
                cropped_center_x - 320 : cropped_center_x + 320,
                cropped_center_y - 320 : cropped_center_y + 320
            ].astype(np.uint16)
            cv2.imwrite(
                f"{output_dir}/{tiff.stem}_cropped.png",
                cropped_image
            )


            # save the pits coordinates in the cropped image as a csv
            pit_coords_data = [
                ["sample (x)", "line (y)"],
                [cropped_pit_y, cropped_pit_x]
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
            print(f"Image {tiff.name} cropped")
            print(f"This image has a pit?: {has_pit(cropped_image = cropped_image, base_image = image)}")
            print(f"Number of images cropped: {tiff_counter}")

    else:
        for tiff in input_dir.iterdir():
            image = cv2.imread(tiff, -1)
            height, width = image.shape[:2]
            pit_exists = True
            while pit_exists:
                crop_offset_x = random.randint(0, height - 640)
                crop_offset_y = random.randint(0, width - 640)
                cropped_image = image[
                    crop_offset_x : crop_offset_x + 640,
                    crop_offset_y: crop_offset_y + 640
                ].astype(np.uint16)
                pit_exists = has_pit(cropped_image = cropped_image, base_image = image)

            cv2.imwrite(
                f"{output_dir}/{tiff.stem}_cropped_random.png",
                cropped_image,
            )

            tiff_counter += 1
            print("Number of images cropped:", tiff_counter)
            print(f"Image {tiff.name} cropped")
            # if tiff_counter == 1:
            #     break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
