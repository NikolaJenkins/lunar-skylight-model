# lunar-skylight-model

## Steps to replicate

### Clone the repository
1. Clone this repository using ```git clone git@github.com:NikolaJenkins/lunar-skylight-model.git```.
2. Move to the repository using ```mv lunar-skylight-model```.
3. Create a virtual environment by running ```python3 -m venv .lunar```.
4. Activate the environment by running ```source .lunar/bin/activate```.
5. Download the requirements using ```pip install -r requirements.txt```.

### Install images
1. Go to https://ode.rsl.wustl.edu/moon/productsearch and select Lunar Reconnaissance Orbiter -> LROC -> PDS4 Calibrated Data Record Narrow Angle Camera under "Step 1. Select Data Sets to Search (A Selection is Required)".

2. Copy 50 lines from lunar_pit_order/lunar_pit_ids_1_order.txt into the Product ID text box under "Step 2. Set Additional Filtering Parameters (Optional)". Then click "View Results in Table" under step 4 and add the images to the cart. Repeat for the rest of the txt file in batches of 50 images.

3. Submit the order under the 'Download' button in the upper bar, and either wait for a confirmation email or find the list of direct download links under 'Advanced user options'.

4. Copy the links to a txt file. Then run the commands
```grep '\.img' > col_1_img_links.txt``` 
```grep 'c\.xml' > col_1_xml_links.txt.```

5. Download the img and xml files using wget to a directory of your choice.

6. Repeat steps 2-5 but with lunar_pit_order/lunar_pit_ids_2_3_order.txt instead, and downloading the images to a different directory.

### Convert .img files to .tif files
1. Run ```chmod +x image_analysis/img_to_tiff.py```
2. Create directory to store tiff images.
3. Run ```image_analysis/img_to_tiff.py --input [directory storing img and xmf files] --output [target directory to store tiff images]```

### Crop .tif images
1. Run ```chmod +x image_analysis/crop_tiff.py```
2. Create directory to store cropped images and corresponding pit coordinate csv files.
3. Create another directory to store background images.
4. Run ```image_analysis/crop_tiff.py --input [directory storing tiff images] --coords lunar_pit_order_pit_pixel_coords.csv --output [target directory created in step 2]```
5. Run ```image_analysis/crop_tiff.py --input [directory storing tiff images] --output [target directory created in step 3]```
6. Create a directory to store all the images. Move the cropped png images from the directories created in steps 2 and 3 to the directory just created.
7. Create another directory to store the images stretched to 8 bits for the YOLO model. Run ```chmod +x image_analysis/convert_16bit_png.py```. Then run ```image_analysis/convert_16bit_png.py --input [directory storing images from step 6] --output [directory to store 8 bit images]```

### Generate labels
1. Create a directory to store masked image labels. Create another directory to store background image labels.
2. Run ```chmod +x image_analysis/mask_gen```. Then run ```image_analysis/mask_gen --input [directory storing cropped png images] --output [directory to store mask labels] --model image_analysis/sam_vit_h_4b8939.pth```
3. Run ```chmod +x image_analysis/gen_blank_labels.py```. Then run ```image_analysis/gen_blank_labels --input [directory storing background images] --output [directory to store blank labels]```.
4. Create a directory to store all the labels. Move the masked image labels and blank labels to the new directory.

### Create dataset
1. Create a directory to store the dataset. Inside, create directories called 'images' and 'labels'.
2. Run ```chmod +x image_analysis/split_data.py```
3. Run ```image_analysis/split_data.py --input-images [directory storing all the cropped pngs] --input-labels [directory storing all the labels] --output [directory storing dataset]```

### Train model
1. 
2. Run ```chmod +x image_analysis/lunar_pit_model.py```
