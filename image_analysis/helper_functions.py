import argparse
from pathlib import Path
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def valid_dir(path: Path):
    """Use on a provided path to confirm that it's a valid directory"""
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def valid_file(path: Path):
    """Use on a provided path to confirm that it's a valid file"""
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
    else:
        print(path.stem)
        return path

# based on pitscan formula developed by R. V. Wagner
def has_pit(*, cropped_image: np.ndarray, base_image: np.ndarray):
    """Use on a 640x640 image to roughly distinguish images with pits from images without pits."""

    # calculate threshold for shadowed pixels based on uncropped image
    base_rep = base_image[::50, ::50]
    valid_pixels = base_rep[base_rep > 0]
    base_median_brightness = np.median(valid_pixels)
    base_min_brightness = valid_pixels.min()
    shadow_threshold = base_min_brightness + base_median_brightness * .15

    # check density of shadowed pixels
    shadowed_pixels = (cropped_image < shadow_threshold).astype(np.uint8)
    num_shadowed_pixels = np.sum(shadowed_pixels)
    tile_size = 25
    density_image = cv2.boxFilter(shadowed_pixels.astype(float), -1, (tile_size, tile_size), normalize = False)
    max_shadow_cluster = np.max(density_image)
    is_dense_shadow = max_shadow_cluster >= 15

    return cropped_image.min() <= shadow_threshold and num_shadowed_pixels >= 10 and is_dense_shadow

def read_raw_img(img_path: str):
    with rasterio.open(img_path) as dataset:
        print("Reading file:", img_path)
        print("Uncompressed grid dimensions:", dataset.width, dataset.height)
        raw_matrix = dataset.read(1)
        matrix_32 = raw_matrix.astype(np.float32)
        avg = matrix_32.mean()
        std = matrix_32.std()
        img_min = max(matrix_32.min(), avg - 2 * std)
        img_max = min(matrix_32.max(), avg + 2 * std)
        if img_max - img_min > 0:
            return ((raw_matrix - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            return np.zeros_like(raw_matrix, dtype = np.uint8)

def random_crop(img, pit_sample, pit_line, crop_size = 640):
    offset_radius = crop_size // 2 - 20
    crop_offset = random.randint(-offset_radius, offset_radius)
    crop_origin_x = pit_sample + crop_offset
    crop_origin_y = pit_line + crop_offset
    cropped_pit_x = 320 - crop_offset
    cropped_pit_y = 320 - crop_offset
    crop_radius = crop_size // 2
    image_crop = img[
        crop_origin_x - crop_radius : crop_origin_x + crop_radius,
        crop_origin_y - crop_radius : crop_origin_y + crop_radius
    ]
    return image_crop, cropped_pit_x, cropped_pit_y

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# copied from SAM detection script for drawing star on image
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# copied from SAM detection script for drawing bounding box on image
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# copied from SAM detection script for drawing mask label on image
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
