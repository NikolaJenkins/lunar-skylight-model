#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
    image_float = image.astype(float)
    image_min = image_float[image_float > 0].min()
    image_max = image_float[image_float > 0].max()
    brightened_image = ((image_float - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    cv2.namedWindow(args.input, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.input, 600, 600)
    print(type(brightened_image))
    # cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB)
    plt.imshow(brightened_image, cmap = 'gray')
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
