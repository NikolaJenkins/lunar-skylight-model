#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
from helper_functions import valid_dir, valid_file
import random

parser = argparse.ArgumentParser(
    description=(
        "Split images and labels into train, valid and test folders"
    )
)

parser.add_argument(
    "--input-images",
    type=str,
    required=True,
    help="Path to directory of cropped pngs",
)

parser.add_argument(
    "--input-labels",
    type=str,
    required=True,
    help="Path to directory of labels",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to output directory for organized dataset. Should contain 'images' and 'labels' folder",
)

def split_dataset(
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    background_image_ratio = 0.15,
    random_seed: int = 42
):
    random.seed(random_seed)

    # split images into pits and backgrounds
    dataset_images = valid_dir(output_dir / 'images')
    dataset_labels = valid_dir(output_dir / 'labels')
    for split in ['train', 'val', 'test']:
        (dataset_images / split).mkdir(parents = True, exist_ok = True)
        (dataset_labels / split).mkdir(parents = True, exist_ok = True)
    low_qual_images_path = valid_file(label_dir / "low_qual_images.txt")
    low_qual_images = []
    with open(low_qual_images_path, "r", encoding="utf-8") as file:
        for line in file:
            file = line.rstrip("\n")
            low_qual_images.append(file)
    print("# of low quality images:", len(low_qual_images))
    pit_images = [png for png in image_dir.iterdir() if "random" not in png.name.lower() and png.suffix == ".png" and png.name not in low_qual_images]
    background_images = [png for png in image_dir.iterdir() if "random" in png.name.lower() and png.suffix == ".png"]
    print("# of pit images:", len(pit_images))
    print("# of background images:", len(background_images))

    # split pit images into train, val, test
    random.shuffle(pit_images)
    total_pits = len(pit_images)
    train_mark_pits = int(total_pits * train_ratio)
    val_mark_pits = train_mark_pits + int(total_pits * val_ratio)
    train_pits = pit_images[:train_mark_pits]
    val_pits = pit_images[train_mark_pits:val_mark_pits]
    test_pits = pit_images[val_mark_pits:]
    print("Splits by number:", len(train_pits), len(val_pits), len(test_pits))
    pit_splits = {
        'train': train_pits,
        'val': val_pits,
        'test': test_pits,
    }

    # split background images into train, val, test
    random.shuffle(background_images)
    total_backgrounds = len(background_images)
    included_backgrounds = background_images[:int(total_backgrounds * background_image_ratio)]
    included_background_num = len(included_backgrounds)
    print(f"Keeping {included_background_num} background images in dataset")
    train_mark_backgrounds = int(included_background_num * train_ratio)
    val_mark_backgrounds = train_mark_backgrounds + int(included_background_num * val_ratio)
    train_backgrounds = included_backgrounds[:train_mark_backgrounds]
    val_backgrounds = included_backgrounds[train_mark_backgrounds:val_mark_backgrounds]
    test_backgrounds = included_backgrounds[val_mark_backgrounds:]
    background_splits = {
        'train': train_backgrounds,
        'val': val_backgrounds,
        'test': test_backgrounds,
    }
    print("Splits by number:", len(train_backgrounds), len(val_backgrounds), len(test_backgrounds))

    # move split images and corresponding labels to train, val, test folders
    image_counter = 0
    label_counter = 0
    for split in ['train', 'val', 'test']:
        for img_path in pit_splits[split]:
            shutil.copy(img_path, output_dir / 'images' / split / img_path.name)
            image_counter += 1
            print(f"Moved image {img_path.name}")
            label_name = img_path.stem + '.txt'
            label_path = valid_file(label_dir / label_name)
            shutil.copy(label_path, output_dir / 'labels' / split / label_name)
            label_counter += 1
            print(f"Moved label {label_name}")
            print(f"Moved {image_counter} images and {label_counter} labels")
        for img_path in background_splits[split]:
            shutil.copy(img_path, output_dir / 'images' / split / img_path.name)
            image_counter += 1
            print(f"Moved image {img_path.name}")
            label_name = img_path.stem + '.txt'
            label_path = valid_file(label_dir / label_name)
            shutil.copy(label_path, output_dir / 'labels' / split / label_name)
            label_counter += 1
            print(f"Moved label {label_name}")
            print(f"Moved {image_counter} images and {label_counter} labels")



def main(args: argparse.Namespace):
    input_images = valid_dir(Path(args.input_images))
    input_labels = valid_dir(Path(args.input_labels))
    output_dataset = valid_dir(Path(args.output))
    split_dataset(
        image_dir = input_images,
        label_dir = input_labels,
        output_dir = output_dataset,
    )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
