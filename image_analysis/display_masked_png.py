#!/usr/bin/env python3

import argparse
from pathlib import Path
from ultralytics.utils import

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input-png",
    type=str,
    required=True,
    help="Path to a single input png",
)

parser.add_argument(
    "--input-txt",
    type=str,
    required=True,
    help="Path to a single input mask txt",
)

def valid_file(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
    else:
        return path

def main(args: argparse.Namespace):
    image_path = valid_file(Path(args.input_png))
    csv_path = valid_file(Path(args.input_csv))
    classes = {
        0: "Lunar pit"
    }
    # do the thing!



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
