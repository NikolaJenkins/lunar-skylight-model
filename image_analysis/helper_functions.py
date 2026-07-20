import argparse
from pathlib import Path
import numpy as np
import cv2
import rasterio
import random

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
    # elif not path.stem == ".csv":
    #     raise argparse.ArgumentTypeError(f"{path} is not a valid csv file")
    else:
        print(path.stem)
        return path

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

def sliding_window_pit_search(
    img_1,
    img_2_3,
    pit_sample,
    pit_line,
    window_size = 150,
    search_radius = 2000,
):
    pit_template = img_1[
        pit_sample - (window_size // 2) : pit_sample + (window_size // 2),
        pit_line - (window_size // 2) : pit_line + (window_size // 2)
    ]

    img_height, img_width = img_2_3.shape
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    search_x_min = max(0, img_center_x - search_radius)
    search_x_max = min(img_width, img_center_x + search_radius)
    search_y_min = max(0, img_center_y - search_radius)
    search_y_max = min(img_height, img_center_y + search_radius)
    search_area = img_2_3[
        search_x_min: search_x_max,
        search_y_min : search_y_max
    ]

    search_area = img_2_3
    res = cv2.matchTemplate(search_area, pit_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.60:
        tile_match_x, tile_match_y = max_loc
        col_2_pit_x = tile_match_x + window_size // 2
        col_2_pit_y = tile_match_y + window_size // 2
        print(f"Match found! Confidence: {max_val:.4f}, pit pixels: x: {col_2_pit_x}, y: {col_2_pit_y}")
        return col_2_pit_x, col_2_pit_y
    print(f"Couldn't find a pit match, highest confidence: {max_val:.4f}")
    return None, None

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
