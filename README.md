# lunar-skylight-model

## Steps to replicate

### Install images
1. Go to https://ode.rsl.wustl.edu/moon/productsearch and select Lunar Reconnaissance Orbiter -> LROC -> PDS4 Calibrated Data Record Narrow Angle Camera under "Step 1. Select Data Sets to Search (A Selection is Required)".

2. Copy 50 lines from lunar_pit_order/lunar_pit_ids_1_order.txt into the Product ID text box under "Step 2. Set Additional Filtering Parameters (Optional)". Then click "View Results in Table" under step 4 and add the images to the cart. Repeat for the rest of the txt file in batches of 50 images.

3. Submit the order under the 'Download' button in the upper bar, and either wait for a confirmation email or find the list of direct download links under 'Advanced user options'.

4. Copy the links to a txt file. Then run the commands
grep '\.img' > col_1_img_links.txt
and
grep 'c\.xml' > col_1_xml_links.txt.

5. Download the img and xml files using wget to a directory of your choice.

6. Repeat steps 2-5 but with lunar_pit_order/lunar_pit_ids_2_3_order.txt instead, and downloading the images to a different directory.

### Convert .img files to .tif files
1. Run chmod +x img_to_tiff.py
2. Create directory to store tiff images.
3. Run ~/LunarSkylights/image_analysis/img_to_tiff.py --input \[directory storing img and xmf files\] --output \[target directory to store tiff images\]

### Crop .tif images
1. Run chmod +x crop_tiff.py
2. Create directory to store cropped images and corresponding pit coordinate csv files.
3. Create another directory to store randomly cropped images.
4. Run ~/LunarSkylights/image_analysis/crop_tiff.py --input \[directory storing tiff images\] --coords ~/LunarSkylights/lunar_pit_order_pit_pixel_coords.csv --output \[target directory created in step 2\]
5. Run ~/LunarSkylights/image_analysis/crop_tiff.py --input \[directory storing tiff images\] --output \[target directory created in step 3\]
