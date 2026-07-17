#!/usr/bin/env python3

import argparse
from pathlib import Path
from helper_functions import valid_dir, valid_file, has_pit, read_raw_img
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
    help="Path to directory of IMG files",
)

parser.add_argument(
    "--col-1",
    type=str,
    required=False,
    help="Path to directory storing column 1 images and corresponding csv files"
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    help="Path to output directory",
)

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    col_1_dir = valid_dir(Path(args.col_1))
    output_dir = valid_dir(Path(args.output))

    tiff_counter = 0
    if col_1_dir is not None:
        for img in input_dir.iterdir():
            # read images
            np_array = read_raw_img(img_path = img, width = 5064)
            img_stem = img.stem.lower()
            tiff = cv2.imread(f"{col_1_dir}/{img_stem}_cropped.png", -1)
            # coords = pd.read_csv(f"{col_1_dir}/{img_stem}_pit_coords.csv")
            # x, y = coords.iloc[1]

            # random offset so model doesn't always train on pits in center of image
            # crop_offset_x = random.randint(-300, 300)
            # crop_offset_y = random.randint(-300, 300)

            # pit coordinates in cropped image
            # cropped_pit_x = 320 - crop_offset_x
            # cropped_pit_y = 320 - crop_offset_y

            # center of cropped image
            # cropped_center_x = pit_x + crop_offset_x
            # cropped_center_y = pit_y + crop_offset_y

            # some pits are close to edge of image so this prevents index out of bound errors
            # if cropped_center_x < 320:
                # cropped_center_x = 320
            # elif cropped_center_x > height - 320:
                # cropped_center_x = height - 320
            # if cropped_center_y < 320:
                # cropped_center_y = 320
            # elif cropped_center_y > width - 320:
                # cropped_center_y = width - 320

            # generate cropped image and write to png file in output directory
            # cropped_image = image[
            #     cropped_center_x - 320 : cropped_center_x + 320,
            #     cropped_center_y - 320 : cropped_center_y + 320
            # ].astype(np.uint16)
            # cv2.imwrite(
            #     f"{output_dir}/{img.stem}_cropped.png",
            #     cropped_image
            # )


            # save the pits coordinates in the cropped image as a csv
            # pit_coords_data = [
            #     ["sample (x)", "line (y)"],
            #     [cropped_pit_y, cropped_pit_x]
            # ]
            # with open(
            #     f"{output_dir}/{img.stem}_pit_coords.csv",
            #     mode = "w",
            #     newline = "",
            #     encoding = "utf-8"
            # ) as file:
            #     writer = csv.writer(file)
            #     writer.writerows(pit_coords_data)

            tiff_counter += 1
            print(f"Image {img.name} cropped")
            # print(f"This image has a pit?: {has_pit(cropped_image = cropped_image, base_image = image)}")
            print(f"Number of images cropped: {tiff_counter}")
            if tiff_counter == 5:
                break

    else:
        for img in input_dir.iterdir():
            image = cv2.imread(img, -1)
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
                f"{output_dir}/{img.stem}_cropped_random.png",
                cropped_image,
            )

            tiff_counter += 1
            print("Number of images cropped:", tiff_counter)
            print(f"Image {img.name} cropped")
            # if tiff_counter == 1:
            #     break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
