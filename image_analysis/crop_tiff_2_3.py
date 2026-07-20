#!/usr/bin/env python3

import argparse
from pathlib import Path
from helper_functions import valid_dir, valid_file, has_pit, read_raw_img, sliding_window_pit_search, random_crop
import pandas as pd
import numpy as np
import random
import csv
import cv2
import matplotlib.pyplot as plt

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
    help="Path to directory of column 2/3 raw img files",
)

parser.add_argument(
    "--col-1-img",
    type=str,
    required=True,
    help="Path to directory storing column 1 raw img files"
)

parser.add_argument(
    "--col-1-cropped",
    type = str,
    required = True,
    help = "Path to directory storing cropped column 1 pngs and corresponding coordinates csv files"
)

parser.add_argument(
    "--csv",
    type = str,
    required = True,
    help = "Path to csv matching column 2/3 images with corresponding column 1 images"
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    help="Path to output directory",
)

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    col_1_img_dir = valid_dir(Path(args.col_1_img))
    col_1_cropped_dir = valid_dir(Path(args.col_1_cropped))
    pits_csv_path = valid_file(Path(args.csv))
    # output_dir = valid_dir(Path(args.output))

    img_counter = 0
    pit_csv = pd.read_csv(pits_csv_path, index_col = 0)
    pit_dict = pit_csv.to_dict(orient = "split")
    coords_dict = dict(zip(pit_dict["index"], pit_dict["data"]))

    if input_dir is not None:
        for img in input_dir.iterdir():
            # read images
            img_stem = img.stem[:-1]
            col_1_stem = coords_dict[img_stem][0].lower()
            col_1_img_path = f"{col_1_img_dir}/{col_1_stem}c.img"
            print("Col 1 image path:", col_1_img_path)
            col_1_csv_path = f"{col_1_cropped_dir}/{col_1_stem}c_pit_coords.csv"
            col_1_img = read_raw_img(img_path = col_1_img_path)
            col_2_3_img = read_raw_img(img_path = img)
            col_1_csv = pd.read_csv(col_1_csv_path)
            x, y = col_1_csv.iloc[0].tolist()
            print("Column 1 pit x, y:", x, y)


            # for name, matrix in [("Column 1", col_1_img), ("Column 2/3", col_2_3_img)]:
            #     print(f"--- Checking {name} Matrix Integrity ---")
            #     print(f"Data type: {matrix.dtype}")
            #     print(f"Array Shape (Height x Width): {matrix.shape}")
            #     print(f"Minimum Pixel Intensity: {matrix.min()}")
            #     print(f"Maximum Pixel Intensity: {matrix.max()}")
            #     print(f"Average Pixel Intensity: {matrix.mean():.2f}")
            #     print("-" * 40)
            pit_y, pit_x = sliding_window_pit_search(
                img_1 = col_1_img,
                img_2_3 = col_2_3_img,
                pit_sample = x,
                pit_line = y,
            )

            col_2_3_y, col_2_3_x = col_2_3_img.shape
            cropped_start_x = max(0, min(pit_x, col_2_3_x) - 320)
            cropped_start_y = max(0, min(pit_y, col_2_3_y) - 320)
            print("Check cropped origin:", cropped_start_x, cropped_start_y)
            cropped_img = col_2_3_img[
                cropped_start_x : cropped_start_x + 640,
                cropped_start_y : cropped_start_y + 640
            ]
            print("Check dimensions", cropped_img.shape)

            # cropped_img, cropped_pit_x, cropped_pit_y = random_crop(
            #     img = col_2_3_img,
            #     pit_sample = pit_x,
            #     pit_line = pit_y
            # )

            plt.figure(figsize = (8,8))
            plt.imshow(cropped_img, cmap = "gray", aspect = 'equal')
            plt.title(f"Crop check")
            plt.show()

            # crop_origin_x = max(0, pi)

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

            img_counter += 1
            print(f"Image {img.name} cropped")
            # print(f"This image has a pit?: {has_pit(cropped_image = cropped_image, base_image = image)}")
            print(f"Number of images cropped: {img_counter}")
            if img_counter == 1:
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

            img_counter += 1
            print("Number of images cropped:", img_counter)
            print(f"Image {img.name} cropped")
            # if tiff_counter == 1:
            #     break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
