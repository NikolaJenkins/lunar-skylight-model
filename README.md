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
```source .lunar/bin/activate```

4. Download the requirements using 
```python -m pip install -r requirements.txt```

5. Install torch and torchvision by following the steps in this link: https://pytorch.org/get-started/locally/. The command will change depending on the GPU your computer has. You may have to add the tag ```--no-cache-dir``` to the end of the install command to force pip to bypass the storage space limit on your computer.
6. Download the SAM model by clicking on this link:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth.
Then move the model from Downloads to the ```image_analysis``` directory using
```mv ~/Downloads/sam_vit_h_4b8939.pth ~/lunar-skylight-model/image_analysis```

### Install images (skip to generating labels if using sample images)
1. Go to https://ode.rsl.wustl.edu/moon/productsearch and select Lunar Reconnaissance Orbiter -> LROC -> PDS4 Calibrated Data Record Narrow Angle Camera under "Step 1. Select Data Sets to Search (A Selection is Required)".

2. Copy 50 lines from lunar_pit_order/lunar_pit_ids_1_order.txt into the Product ID text box under "Step 2. Set Additional Filtering Parameters (Optional)". Then click "View Results in Table" under step 4 and add the images to the cart. Repeat for the rest of the txt file in batches of 50 images.

3. Submit the order under the 'Download' button in the upper bar, and either wait for a confirmation email or find the list of direct download links under 'Advanced user options'.

4. Create a directory to store the IMG files and their cooresponding xml files by running
```mkdir ~/col_1_imgs```
5. Copy the links to a txt file. Then run the commands
```grep '\.img' > ~/col_1_imgs/col_1_img_links.txt``` 
```grep 'c\.xml' > ~/col_1_imgs/col_1_xml_links.txt.```

5. Download the img and xml files using wget to a directory of your choice.

6. Repeat steps 2-5 but with lunar_pit_order/lunar_pit_ids_2_3_order.txt instead, and downloading the images to a different directory.

### Convert .img files to .tif files
1. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/img_to_tiff.py```
2. Create directory to store tiff images by running
```mkdir ~/col_1_tifs```
3. Run 
```~/lunar-skylight-model/image_analysis/img_to_tiff.py --input ~/col_1_imgs --output ~/col_1_tifs```

### Crop .tif images
1. Run 
```chmod +x image_analysis/crop_tiff.py```

2. Create directory to store cropped images and corresponding pit coordinate csv files by running
```mkdir ~/all_images```

3. Run 
```~/lunar-skylight-model/image_analysis/crop_tiff.py --input ~/col_1_tifs --coords lunar_pit_order_pit_pixel_coords.csv --output ~/all_images```

4. Run 
```~/lunar-skylight-model/image_analysis/crop_tiff.py --input ~/col_1_tifs --output ~/all_images```

5. Create another directory to store the images stretched to 8 bits for the YOLO model by running 
```mkdir ~/all_images_8_bit```
Run 
```chmod +x image_analysis/convert_16bit_png.py```
Then run 
```image_analysis/convert_16bit_png.py --input ~/all_images --output all_images_8_bit```

6. (If using downloaded images) Copy the pit coordinate files to the new directory by running 
```cp ~/all_images/*.csv ~/all_images_8_bit```

### Generate labels
1. Create a directory to store image labels.
```mkdir ~/all_masks```
2. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/mask_gen``` Then run 
```~/lunar-skylight-model/image_analysis/mask_gen --input ~/all_images_8_bit --output ~/all_masks --model ~/lunar-skylight-model/image_analysis/sam_vit_h_4b8939.pth```
3. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/gen_blank_labels.py```
Then run
```image_analysis/gen_blank_labels --input ~/all_images --output ~/all_masks```
4. Create a directory to store all the labels. Move the masked image labels and blank labels to the new directory.

### Create dataset
1. Create a directory to store the dataset by running
```mkdir ~/lunar_dataset```
Inside, create directories called 'images' and 'labels' by running
```mkdir ~/lunar_dataset/images```
```mkdir ~/lunar_dataset/labels```
2. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/split_data.py```
3. Run 
```~/lunar-skylight-model/image_analysis/split_data.py --input-images ~/all_images_8_bit --input-labels ~/all_masks --output ~/lunar_dataset```

### Train model
1. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/lunar_pit_model.py```.
2. Run 
```cd ~/lunar_dataset```.
3. Run 
```~/lunar-skylight-model/image_analysis/lunar_pit_model.py``` This may take a long time depending on how much data you're training on.

### Run inferences (work in progress)
1. Run 
```chmod +x ~/lunar-skylight-model/image_analysis/lunar_pit_pred.py```
2. To test the model on images, run 
```image_analysis_lunar_pit_pred.py --model ~/lunar_dataset/runs/segment/lunar_pit_v1/weights/best.pt --images [any directory with cropped 640x640 images]```
