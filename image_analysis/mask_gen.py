#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import csv
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import sys

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
    required=False,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--coords",
    type = str,
    help="Path to csv containing coordinates for pit to be masked",
)

def valid_dir(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def valid_file(path: Path):
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
    else:
        return path

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
    checkpoint_path = "sam_vit_h_4b8939.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path: str = valid_file(Path(args.input))
    coords_path: str = valid_file(Path(args.coords))
    image_16bit = cv2.imread(image_path, -1)
    image_min = float(image_16bit.min())
    image_max = float(image_16bit.max())
    image_8bit = ((image_16bit.astype(float) - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    image_8bit_rgb = cv2.merge([image_8bit, image_8bit, image_8bit])

    # display image
    print("Image dimensions: ", image_8bit_rgb.shape)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_8bit, cmap = "gray")
    plt.xlim(0, 640)
    plt.ylim(640, 0)
    plt.axis('on')
    plt.show()

    # display image with pit coordinates visible
    with open(coords_path, mode = 'r', newline = '', encoding = 'utf-8') as file:
        reader = csv.reader(file)
        coords = np.array([[int(x) for x in list(reader)[1]]])
        # print(coords)
    input_label = np.array([1])

    plt.figure(figsize = (8,8))
    plt.imshow(image_8bit_rgb, cmap = "gray")
    show_points(coords, input_label, plt.gca())
    plt.xlim(0, 640)
    plt.ylim(640, 0)
    plt.axis('on')
    plt.show()

    # generate mask centered on target point
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.float()
    with torch.inference_mode():
    #     # mask_generator = SamAutomaticMaskGenerator(
    #     #     model = sam
    #     # )
    #     # masks = mask_generator.generate(image)
        predictor = SamPredictor(sam)
        predictor.set_image(image_8bit_rgb)
        masks, scores, logits = predictor.predict(
            point_coords = coords,
            point_labels = input_label,
            multimask_output = True,
        )
    print(masks.shape)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(8,8))
        plt.imshow(image_8bit_rgb, cmap = "gray")
        show_mask(mask, plt.gca())
        show_points(coords, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
