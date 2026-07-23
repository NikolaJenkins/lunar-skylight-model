#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
from helper_functions import valid_dir, valid_file, show_points, show_mask
import csv
from segment_anything import SamPredictor, sam_model_registry
import sys
import time

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
    help="Path to directory storing cropped png files and their respective pit coords csv files.",
)

parser.add_argument(
    "--output",
    type = str,
    required = True,
    help = "Path to directory to store masks"
)

parser.add_argument(
    "--model",
    type = str,
    required = True,
    help = "Path to model"
)

def main(args: argparse.Namespace):
    sys.path.append("..")

    model_type = "vit_h"
    checkpoint_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))

    img_counter = 0
    mask_counter = 0
    low_qual_masks = []
    for img in input_dir.iterdir():
        if img.suffix == ".png" and "random" not in img.name:
            start_time = time.perf_counter()

            # read img and prepare for generating mask
            image_16bit = cv2.imread(img, -1)
            image_min = float(image_16bit.min())
            image_max = float(image_16bit.max())
            image_8bit = ((image_16bit.astype(float) - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            image_8bit_rgb = cv2.merge([image_8bit, image_8bit, image_8bit])

            # search for corresponding pit coordinates csv file
            coords_path = valid_file(Path(f"{input_dir}/{img.stem.split("_")[0]}_pit_coords.csv"))
            with open(coords_path, mode = 'r', newline = '', encoding = 'utf-8') as file:
                reader = csv.reader(file)
                coords = np.array([[int(x) for x in list(reader)[1]]])
            input_label = np.array([1])

            # generate mask centered on target point
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            sam.float()
            with torch.inference_mode():
                predictor = SamPredictor(sam)
                predictor.set_image(image_8bit_rgb)
                masks, scores, logits = predictor.predict(
                    point_coords = coords,
                    point_labels = input_label,
                    multimask_output = True,
                )

            # create side by side plot with original image
            fig, ax = plt.subplots(1, 4, figsize = (24, 6))
            og_plot = ax[0]
            fig.suptitle("Mask candidates")
            og_plot.imshow(image_8bit_rgb)
            show_points(coords, input_label, ax[0], marker_size = 100)
            og_plot.set_title("Original Image", fontsize=18)
            og_plot.axis('on')

            # add 3 masked plots
            for i, (mask, score) in enumerate(zip(masks, scores)):
                subplot = ax[i + 1]
                subplot.imshow(image_8bit_rgb)
                show_mask(mask, subplot)
                subplot.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                subplot.axis('on')
            plt.tight_layout()
            plt.show(block = False)
            plt.pause(0.1)

            selected_mask = None

            # take user input to determine which mask to take
            choice = input(f"Select one of the masks for image {img.name} by typing 1, 2, or 3. If no masks were generated or they're all low quality, type 's' to skip.").strip().lower()
            while True:
                if choice in ["1", "2", "3"]:
                    selected_mask = masks[int(choice) - 1]
                    print(f"Chose mask {choice} for image {img.name}")
                    mask_counter += 1
                    break
                elif choice == "s":
                    print(f"Skipped mask generation for image {img.name}")
                    low_qual_masks.append(img.name)
                    break
            plt.close()

            # write mask to yolo compatible file
            if selected_mask is not None:
                binary_mask = selected_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                blank_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype = np.uint8)
                for _, contour in enumerate(contours):
                    cv2.drawContours(blank_mask, [contour], -1, [30, 144, 255], -1)
                print(len(contours))
                yolo_lines = []
                normalized_points = []
                contour = contours[0]
                for point in contour:
                    x, y = point[0]
                    normalized_points.append(f"{x / 640:.5f} {y / 640:.4f}")
                yolo_lines.append(f"0 {" ".join(normalized_points)}")
                with open(f"{output_dir}/{img.stem}.txt", "w") as file:
                    file.write(yolo_lines[0])

            print(f"{mask_counter} images have been masked")

            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Executed in {execution_time // 60} minutes and {execution_time % 60:.3f} seconds")

            print("Unmasked files:", low_qual_masks)
            img_counter += 1

    # record images that were skipped
    if low_qual_masks:
        skipped_images = f"{output_dir}/low_qual_images.txt"
        with open(skipped_images, "w") as file:
            for item in low_qual_masks:
                file.write(f"{item}\n")
    else:
        print("All images had satisfactory masks!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
