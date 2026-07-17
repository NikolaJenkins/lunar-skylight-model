#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
from helper_functions import valid_dir, valid_file
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
    required = False,
    help = "Path to directory to store masks"
)

parser.add_argument(
    "--model",
    type = str,
    help = "Path to model"
)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main(args: argparse.Namespace):
    sys.path.append("..")

    model_type = "vit_h"
    checkpoint_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = valid_dir(Path(args.input))
    output_dir = valid_dir(Path(args.output))

    img_counter = 0
    for img in input_dir.iterdir():
        if img.suffix == ".png":
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
                masks, _, _ = predictor.predict(
                    point_coords = coords,
                    point_labels = input_label,
                    multimask_output = False,
                )
            mask = masks[0]

            # plt.figure(figsize = (8,8))
            # plt.imshow(image_8bit_rgb)
            # show_mask(masks, plt.gca())
            # show_points(coords, input_label, plt.gca())
            # plt.axis('on')
            # plt.show()

            # write mask to yolo compatible file
            binary_mask = mask.astype(np.uint8) * 255
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
                normalized_points.append(f"{x / 640:.4f} {y / 640:.4f}")
            yolo_lines.append(f"0 {" ".join(normalized_points)}")
            print(yolo_lines)
            with open(f"{output_dir}/{img.stem}.txt", "w") as file:
                file.write(yolo_lines[0])

            print(f"Masked image {img.name}")
            img_counter += 1
            print(f"{img_counter} images have been masked")

            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Executed in {execution_time // 60} minutes and {execution_time % 60:.3f} seconds")

            # if img_counter == 1:
            #     break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
