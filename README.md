# Synthetic-Label-Generation

## Purpose
This repo contains files that are used to generate YOLOv3 formatted labels from the black-and-white ground truth images generated from the CityEngine script.

## Using this repo
The only script that should be edited and used is preprocess_syn_xview_background_gt_labels. This file takes 4 arguments (through an argparser) that you need to worry about:
* syn_data_dir - path to the directory that contains the black and white images generated from the CityEngine script
* syn_annos_dir - the directory where it will output the YOLOv3 formatted labels for the synthetic data
* syn_box_dir - the directory where it will output the black and white images with bounding boxes drawn on them after grouping the black pixels
* syn_txt_dir - the directory where it will output any references to the images/labels (such as when you call create_paths())

Additionally, this file has two functions that are important for our use
* group_object_annotation_and_draw_bbox() - this looks in the syn_data_dir to find all of the black and white overhead images. It then groups the black pixels in the images and creates a label for each images, which is then saved in the syn_annos_dir. Additionally, it is supposed to draw bounding boxes on the black and white images and saves these new images to syn_box_dir, but we need to debug it. If there is a permission error when running the script, it has likely deleted the syn_box_dir but should have created the labels beforehand.
* create_paths() - generates paths for the synthetic images and labels. Pulls all of the black and white images in the syn_data_dir and outputs two files in the syn_txt_dir: one containing all of the paths for the synthetic images in the format ../synthetic_images/image_name.jpg, and another containing all of the paths for the synthetic labels in the format ../synthetic_labels/image_name.txt. The synthetic image paths can then be copy and pasted into the .txt file for the training image paths and the synthetic label paths can be copy and pasted into the .txt file for the training label paths.
