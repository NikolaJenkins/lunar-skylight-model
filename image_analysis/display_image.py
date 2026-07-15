#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to a single input image",
)

def main(args: argparse.Namespace):
    image = cv2.imread(args.input, -1)
    assert image.dtype == "uint16", f"Error: image downsampled to {image.dtype}"
    assert image.min() >= 0, "Error: image contains negative pixel values"
    assert image.max() <= 65535, "Error: image data exceeds 16-bit bounds"
    plt.figure(figsize = (8,8))
    plt.imshow(image, cmap = "gray", vmin = image.min(), vmax = image.max())
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
