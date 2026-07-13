import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(
    description=(
        "Convert tiff file to jpg while preserving important details such as colors and contrast"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to output image",
)

def main(args: argparse.Namespace):
    # turn input into image
    img_path: str = args.input
    img: np.ndarray = cv2.imread(img_path)

    # check for redundant color channel dimension, then remove
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, 0]

    # check for extraneous noise
    p05, p995 = np.percentile(img, [0.5, 99.5])
    clipped = np.clip(img, p05, p995)
    base_8bit = ((clipped - p05) / (p995 - p05) * 255).astype(np.uint8)

    # divides image into 8x8 grids then enhances contrast in grids
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_surroundings = clahe.apply(base_8bit)

    # create converted file
    cv2.imwrite(args.output, enhanced_surroundings)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
