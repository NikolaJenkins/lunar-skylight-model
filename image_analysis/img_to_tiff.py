#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(
    description=(
        "Converts all .img files in given directory to .tif files in another directory"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to folder of images and their respective xmf files.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to folder to store tiff images.",
)

def valid_dir(path: str):
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    elif not Path(path).is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    else:
        return path

def main(args: argparse.Namespace):
    input_dir = Path(valid_dir(args.input))
    output_dir = Path(valid_dir(args.output))
    img_counter = 0

    for file in input_dir.iterdir():
        if file.suffix == ".img":
            subprocess.run([
                "ln",
                "-s",
                file,
                f"{input_dir}/{file.name.upper()}",
            ])
            subprocess.run([
                "gdal_translate",
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LZW",
                file,
                f"{output_dir}/{file.stem}.tif",
            ])
            subprocess.run([
                "rm",
                f"{input_dir}/{file.name.upper()}"
            ])
            img_counter += 1
            print(f"File {file.name} processed, {img_counter} files have been processed")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
