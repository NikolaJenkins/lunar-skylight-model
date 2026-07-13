import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
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

def main(args: argparse.Namespace):
    sys.path.append("..")

    model_type = "vit_h"
    checkpoint_path = "sam_vit_h_4b8939.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = args.input

    print(f"Image path: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.float()
    # predictor = SamPredictor(sam)
    # predictor.set_image(<your_imagemask_generator = SamAutomaticMaskGenerator(sam)>)
    # masks, _, _ = predictor.predict(<input_prompts>)
    with torch.inference_mode():
        mask_generator = SamAutomaticMaskGenerator(
            model = sam
        )
        masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

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

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
