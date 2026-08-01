# lunar-skylight-model

This is a project for creating an object detection model and identifying lunar skylight candidates. Since lunar skylights are a subcategory of lunar pits and are hard to distinguish from them, this model just identifies candidates that can later be reviewed by a human. 

## Installation

### Clone the repository
1. Clone this repository using 
```
git clone git@github.com:NikolaJenkins/lunar-skylight-model.git
```

### Set up the environment
1. Move to the repository using 
```
cd lunar-skylight-model
```

2. Create a virtual environment by running 
```
python3 -m venv .lunar
```

3. Activate the environment by running 
```
source .lunar/bin/activate
```

4. Download the requirements using 
```
python -m pip install -r requirements.txt
```

5. Download Pytorch and Torchvision with:
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
```

6. Download the SAM model by running:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Then move the model to the image_analysis directory using
```
mv ~/lunar-skylight-model/sam_vit_h_4b8939.pth ~/lunar-skylight-model/image_analysis
```

### Install images (skip to generating labels if using sample images)

1. Create a directory to store the column 1 IMG files and their cooresponding xml files by running
```
mkdir ~/col_1_imgs && cd ~/col_1_imgs
```
2. Copy the links to a txt file. Then run the commands
```
grep -oE '\S*c\.img' ~/lunar-skylight-model/lunar_pit_order/col_1_order.txt > col_1_img_links.txt
``` 
```
grep -oE '\S*c\.xml' ~/lunar-skylight-model/lunar_pit_order/col_1_order.txt > col_1_xml_links.txt
```
3. Download the images and xml files with:
```
wget -i col_1_img_links.txt
```
```
wget -i col_1_xml_links.txt
```

4. Create a directory to store the column 2 and 3 IMG files with:
```
mkdir ~/col_2_3_imgs && cd ~/col_2_3_imgs
```

5. Download the images with:
```
wget -i ~/lunar-skylight-model/lunar_pit_order/img_order_2_3.txt
```

### Convert .img files to .tif files
1. Run 
```
chmod +x ~/lunar-skylight-model/image_analysis/img_to_tiff.py
```

2. Create directory to store tiff images by running
```
mkdir ~/col_1_tifs
```

3. Run 
```
~/lunar-skylight-model/image_analysis/img_to_tiff.py --input ~/col_1_imgs --output ~/col_1_tifs
```

### Crop .tif images
1. Run 
```
chmod +x image_analysis/crop_tiff.py
```

2. Create directory to store cropped images and corresponding pit coordinate csv files by running
```
mkdir ~/all_images
```

3. Run 
```
~/lunar-skylight-model/image_analysis/crop_tiff.py --input ~/col_1_tifs --coords lunar_pit_order_pit_pixel_coords.csv --output ~/all_images
```

4. Run 
```
~/lunar-skylight-model/image_analysis/crop_tiff.py --input ~/col_1_tifs --output ~/all_images
```

5. Create another directory to store the images stretched to 8 bits for the YOLO model by running the commands
```
mkdir ~/all_images_8_bit
```

```
chmod +x image_analysis/convert_16bit_png.py
```

Then run 
```
image_analysis/convert_16bit_png.py --input ~/all_images --output all_images_8_bit
```

6. (If using downloaded images) Copy the pit coordinate files to the new directory by running 
```
cp ~/all_images/*.csv ~/all_images_8_bit
```

### Generate labels
1. Create a directory to store image labels.
```
mkdir ~/all_masks
```
2. Run
```
chmod +x ~/lunar-skylight-model/image_analysis/mask_gen.py
``` 
Then run this command if you downloaded the images. Follow the instructions in the terminal to select the best mask.
```
~/lunar-skylight-model/image_analysis/mask_gen.py --input ~/all_images_8_bit --output ~/all_masks --model ~/lunar-skylight-model/image_analysis/sam_vit_h_4b8939.pth
```
Otherwise, run this command if you're using sample_imgs.
```
~/lunar-skylight-model/image_analysis/mask_gen.py --input ~/lunar-skylight-model/sample_imgs --output ~/all_masks --model ~/lunar-skylight-model/image_analysis/sam_vit_h_4b8939.pth
```
3. Run 
```
chmod +x ~/lunar-skylight-model/image_analysis/gen_blank_labels.py
```
Then run this command if you downloaded the images.
```
~/lunar-skylight-model/image_analysis/gen_blank_labels.py --input ~/all_images --output ~/all_masks
```
Otherwise, run this command if you're using sample_imgs.
```
~/lunar-skylight-model/image_analysis/gen_blank_labels.py --input ~/lunar-skylight-model/sample_imgs --output ~/all_masks
```

### Create dataset
1. Create a directory to store the dataset by running
```
mkdir ~/lunar_dataset
```
Inside, create directories called 'images' and 'labels' by running
```
mkdir ~/lunar_dataset/images
```
```
mkdir ~/lunar_dataset/labels
```
2. Run 
```
chmod +x ~/lunar-skylight-model/image_analysis/split_data.py
```
3. Run this command if you downloaded the images.
```
~/lunar-skylight-model/image_analysis/split_data.py --input-images ~/all_images_8_bit --input-labels ~/all_masks --output ~/lunar_dataset
```
Otherwise run this command if you're using sample_imgs.
```
~/lunar-skylight-model/image_analysis/split_data.py --input-images ~/lunar-skylight-model/sample_imgs --input-labels ~/all_masks --output ~/lunar_dataset
```

### Train model
1. Run 
```
chmod +x ~/lunar-skylight-model/image_analysis/lunar_pit_model.py
```
2. Run 
```
cd ~/lunar_dataset
```
3. Run 
```
~/lunar-skylight-model/image_analysis/lunar_pit_model.py
```
This may take a long time depending on how much data you're training on.

### Run inferences
1. Run 
```
chmod +x ~/lunar-skylight-model/image_analysis/lunar_pit_pred.py
```
2. Create a directory to store predictions with
```
mkdir ~/lunar_pit_predictions
```
4. To test the model on images, run 
```
~/lunar-skylight-model/image_analysis/lunar_pit_pred.py --model ~/lunar_dataset/runs/segment/lunar_pit_v1/weights/best.pt --images col_2_3_imgs --output ~/lunar_pit_predictions
```

## Building the paper and slides
Run 
```
pdflatex ~/lunar-skylight-model/NaoJoy_Lunar_Skylight_Detection/main.tex
```
and 
```
pdflatex ~/lunar-skylight-model/NaoJoy_Lunar_Skylight_Detection_slides/main.tex
```
