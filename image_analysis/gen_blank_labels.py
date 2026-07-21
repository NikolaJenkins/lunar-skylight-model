#!/usr/bin/env python3

import argparse
from pathlib import Path
from helper_functions import valid_dir

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
    help="Path to directory of cropped background pngs",
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    help="Path to output directory for blank txt files",
)

def main(args: argparse.Namespace):
    background_png_dir = valid_dir(Path(args.input))
    background_labels_dir = valid_dir(Path(args.output))
    # for filename in os.listdir(background_png_dir):
        # stem, _ = os.path.spli
    png_counter = 0
    for png in background_png_dir.iterdir():
        stem = png.stem
        blank_labels_path = f"{background_labels_dir}/{stem}.txt"
        with open(blank_labels_path, "w") as file:
            pass
        png_counter += 1
        print(f"{png_counter} label files generated")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
