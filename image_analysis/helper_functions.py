import argparse
from pathlib import Path
import numpy as np
import cv2
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

def sift_pit_search(
    img_1_path,
    img_2_3_path,
    pit_sample,
    pit_line,
    radius = 200
):
    # read images
    img_1 = read_raw_img(img_1_path)
    img_2_3 = read_raw_img(img_2_3_path)

    # extract pit template from img 1
    img_height, img_width = img_1.shape
    pit_template_origin_x = max(0, min(pit_sample - radius, img_width - 2 * radius))
    pit_template_origin_y = max(0, min(pit_line - radius, img_height - 2 * radius))
    pit_template = img_1[
        pit_template_origin_y : pit_template_origin_y + 2 * radius,
        pit_template_origin_x : pit_template_origin_x + 2 * radius
    ]
    print(f"Template origin coordinates: ({pit_template_origin_x}, {pit_template_origin_y})")

    # initialize SIFT
    sift = cv2.SIFT_create()
    # actually ended up freezing my laptop, this process had to be killed
    kp_1, descriptor_1 = sift.detectAndCompute(pit_template, None)
    kp_2_3, descriptor_2_3 = sift.detectAndCompute(img_2_3, None)

    if descriptor_1 is None or descriptor_2_3 is None:
        print("SIFT couldn't calculate texture descriptors")

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
