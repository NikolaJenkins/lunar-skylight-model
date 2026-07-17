#!/usr/bin/env python3

import argparse
from pathlib import Path
from ultralytics.data.utils import visualize_image_annotations

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

def main(args: argparse.Namespace):
    classes = {
        0: "Lunar pit"
    }
    visualize_image_annotations(
        args.input_png,
        args.input_txt,
        classes
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
