#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import pandas as pd
import random
import csv

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
    required=True,
    help="Path to output directory",
)

def valid_dir(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def valid_csv(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
    # elif not path.stem == ".csv":
    #     raise argparse.ArgumentTypeError(f"{path} is not a valid csv file")
    else:
        print(path.stem)
        return path

def main(args: argparse.Namespace):
    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))
    tiff_counter = 0
    if args.coords is not None:
        pit_csv = valid_csv(Path(args.coords))
        pit_csv = pd.read_csv(pit_csv, index_col = 0)
        pit_dict = pit_csv.to_dict(orient = "split")
        coords_dict = dict(zip(pit_dict["index"], pit_dict["data"]))
        for tiff in input_dir.iterdir():
            key = tiff.stem.upper()[:-1]
            pit_x, pit_y = coords_dict[key]
            crop_offset_x = random.randint(-620, -20)
            crop_offset_y = random.randint(-620, -20)
            cropped_pit_x = 320 - (crop_offset_x + 320)
            cropped_pit_y = 320 - (crop_offset_y + 320)
            subprocess.run([
                "magick",
                tiff,
                "-crop",
                f"640x640+{pit_x + crop_offset_x}+{pit_y + crop_offset_y}",
                f"{output_dir}/{tiff.stem}_cropped.tif",
            ])
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
    # TODO: add random crop if csv isn't provided

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
