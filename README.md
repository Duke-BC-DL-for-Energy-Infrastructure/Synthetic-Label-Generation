# Synthetic-Label-Generation

## Purpose
This repo contains files that are used to generate YOLOv3 formatted labels from the black-and-white ground truth images generated from the CityEngine script.

## Using this repo
The only script that should be edited and run is preprocess_syn_xview_background_gt_labels. This file takes 4 arguments (through an argparser) that you need to worry about:
* syn_data_dir - path to the directory that contains the black and white images generated from the CityEngine script
* syn_annos_dir - the directory where it will output the YOLOv3 formatted labels for the synthetic data
* syn_box_dir - the directory where it will output the black and white images with bounding boxes drawn on them after grouping the black pixels
* syn_txt_dir - the directory where it will output any references to the images/labels (such as when you call create_paths())

This file has one important function:
* group_object_annotation_and_draw_bbox() - this looks in the syn_data_dir to find all of the black and white overhead images. It then groups the black pixels in the images and creates a label for each images, which is then saved in the syn_annos_dir. Additionally, it is supposed to draw bounding boxes on the black and white images and saves these new images to syn_box_dir, but we need to debug it. If there is a permission error when running the script, it has likely deleted the syn_box_dir but should have created the labels beforehand.

To use this, change the argparser arguments to your specific paths, and then you can run the file. We have found a issue where it deletes the black and white annotation folder after generating the labels, so it would be safest to use this with a copy of the black and white annotations.
