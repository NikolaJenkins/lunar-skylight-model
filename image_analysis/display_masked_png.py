#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from helper_functions import valid_dir

parser = argparse.ArgumentParser(
    description=(
        "Display cropped png image and its label"
    )
)

parser.add_argument(
    "--input-images",
    type=str,
    required=True,
    help="Path to directory of cropped pit images",
)

parser.add_argument(
    "--input-labels",
    type=str,
    required=True,
    help="Path to directory of label files",
)

def main(args: argparse.Namespace):
    image_dir = valid_dir(Path(args.input_images))
    label_dir = valid_dir(Path(args.input_labels))
    png_counter = 0
    for png in image_dir.iterdir():
        stem = png.stem
        if "random" not in stem:
            # read image and corresponding label
            image = cv2.imread(png, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_overlay = image_rgb.copy()
            height, width, _ = image_rgb.shape
            label_path = label_dir / (stem + ".txt")
            labels = label_path.read_text().strip().split()

            if len(labels) == 5:
                print("This is a bounding box!")
                break

            # unnormalize label file
            annotator = Annotator(image_rgb, line_width = 2, example = png.name)
            class_id = int(labels[0])
            normalized_labels = np.array([float(x) for x in labels[1:]])
            normalized_labels[0::2] *= width
            normalized_labels[1::2] *= height
            normalized_labels = normalized_labels
            coords = normalized_labels.reshape((-1, 1, 2)).astype(np.int32)
            print("# of coords:", len(coords))

            # create masks
            class_color = colors(class_id, True)
            cv2.fillPoly(image_overlay, [coords], class_color)
            cv2.polylines(image_rgb, [coords], True, class_color)
            masked_image = cv2.addWeighted(image_overlay, 0.4, image_rgb, 0.6, 0)

            # display masked image
            plt.figure(figsize = (10, 10))
            plt.imshow(masked_image)
            plt.title(f"Labels check: {png.name}")
            plt.axis("on")
            plt.show()
            png_counter += 1
            if png_counter == 10:
                break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
