import argparse
from pathlib import Path
import numpy as np
import cv2
from osgeo import gdal

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

def gdal_to_pixel_coords(tiff_path, pit_long, pit_lat):
    """Obtain pixel coordinates of lunar pit in image with provided latitude and longitude values"""
    with gdal.Open(tiff_path) as ds:
        gt = ds.GetGeoTransform()
        print("Forward Geotransform:", gt)
        inv_gt = gdal.InvGeoTransform(gt)
        if inv_gt is None:
            raise ValueError("The inverse transform didn't succeed.")
        pixel_coords = gdal.ApplyGeoTransform(inv_gt, pit_long, pit_lat)
        pit_sample = int(pixel_coords[0])
        pit_line = int(pixel_coords[1])
        print(f"The pit is located at sample (x): {pit_sample} and line (y): {pit_line}")
        return (pit_sample, pit_line)

def read_raw_img(img_path: str, width: int):
    with open(img_path, 'rb') as f:
        header = f.read(10000)
        byte_offset = 0
        end_index = header.find(b'END')
        if end_index != -1:
            header_size = 2532 # mandated by LROC SIS
            byte_offset = ((end_index // header_size) + 1) * 2532
            print("Found END!")
    raw_data = np.fromfile(img_path, dtype = np.uint8, offset = byte_offset)
    print("Did metadata check work?:", len(raw_data) / width)
    height = len(raw_data) // width
    raw_data = raw_data[:height * width].reshape((height, width))
    return raw_data
