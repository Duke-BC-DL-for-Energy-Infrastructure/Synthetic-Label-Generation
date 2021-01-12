'''
creater xuyang_ustb@163.com
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
import pandas as pd
import shutil
import get_bbox_coords_from_annos_with_object_score as gbc

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def merge_clean_origin_syn_image_files(ct, st, dt):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    and remove gt images that are all white pixels
    :return:
    '''
    step = syn_args.tile_size * syn_args.resolution
    image_folder_name = 'syn_{}_{}_{}_images_step{}'.format(dt, ct, st, step)
    label_folder_name = 'syn_{}_{}_{}_annos_step{}'.format(dt, ct, st, step)
    file_path = syn_args.syn_data_dir

    new_img_folder = '{}_all_images_step{}'.format(dt, step)
    new_lbl_folder = '{}_all_annos_step{}'.format(dt, step)
    des_img_path = os.path.join(file_path, new_img_folder)
    des_lbl_path = os.path.join(file_path, new_lbl_folder)
    if not os.path.exists(des_img_path):
        os.mkdir(des_img_path)
    else:
        shutil.rmtree(des_img_path)
        os.mkdir(des_img_path)
    if not os.path.exists(des_lbl_path):
        os.mkdir(des_lbl_path)
    else:
        shutil.rmtree(des_lbl_path)
        os.mkdir(des_lbl_path)

    image_path = os.path.join(file_path, image_folder_name)
    image_files = np.sort(glob.glob(os.path.join(image_path, '*{}'.format(IMG_FORMAT))))
    for img in image_files:
        shutil.copy(img, des_img_path)

    lbl_path = os.path.join(file_path, label_folder_name)
    lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))

    for lbl in lbl_files:
        shutil.copy(lbl, des_lbl_path)


def group_object_annotation_and_draw_bbox(dt, px_thresh=20, whr_thres=4):
    """ Takes black and white images from syn_data_dir and creates labels, saving them in syn_annos_dir
    Also plots the bboxs on the black and white images and saves those new images in syn_box_dir

    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object, and draw bbox for each ground truth files
    """

    step = syn_args.tile_size * syn_args.resolution
    lbl_path = syn_args.syn_data_dir # This path should point to the folder containing all of the black-and-white label images
    print('lbl_path', lbl_path)
    save_txt_path = syn_args.syn_annos_dir
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)

    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region,
                                    link_r=syn_args.link_r, px_thresh=px_thresh, whr_thres=whr_thres)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))
    print(gt_files)
    save_bbx_path = syn_args.syn_box_dir
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
    for g in gt_files:
        gt_name = g.split('/')[-1]
        txt_name = gt_name.replace(IMG_FORMAT, TXT_FORMAT)
        txt_file = os.path.join(save_txt_path, txt_name)
        gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)
    print('DONE!')


def draw_bbx_on_rgb_images(dt, px_thresh=20, whr_thres=4):
    """ Draws the bounding boxes on the rgb images

    """

    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(dt, step)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                                dt, step)
    annos_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_images_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                              dt, step)
    save_bbx_path = os.path.join(syn_args.syn_box_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files):
        txt_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
        gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=False)



def create_paths(seed=17, comment=''):
    """ Generates paths for the images and labels so that they can be copy and pasted into the paths for the training data
        Collects the images from syn_data_dir, and writes

    """

    all_syn_files = np.sort(glob.glob(syn_args.syn_data_dir + '/*/*.png'))
    np.random.shuffle(all_syn_files)
    data_txt_dir = syn_args.syn_txt_dir
    img_paths = open(os.path.join(data_txt_dir, 'synthetic_image_paths.txt', 'w')
    lbl_paths = open(os.path.join(data_txt_dir, 'synthetic_label_paths.txt', 'w')
    img_dir = '../data/synthetic_images/'
    lbl_dir = '../data/synthetic_labels/'
    for image in all_syn_files:
        img_paths.write('%s\n' % os.path.join(img_dir, os.path.basename(image)))
        lbl_paths.write('%s\n' % os.path.join(lbl_dir, os.path.basename(image).replace(IMG_FORMAT, TXT_FORMAT)))
    img_paths.close()
    lbl_paths.close()

def split_trn_val_for_syn_and_real(seed=17, comment='wnd_syn_real', pxwhr='px5whr6', real_img_dir='', real_lbl_dir=''):
    """ Splits the synthetic and real data. Not really useful since the synthetic data should just go into the training set

    """

    step = syn_args.tile_size * syn_args.resolution
    all_syn_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, 'color_all_images_step{}'.format(step), '*.png')))
    num_syn_files = len(all_syn_files)
    all_real_imgs = np.sort(glob.glob(os.path.join(real_img_dir,'*.jpg')))
    num_real_files = len(all_real_imgs)
    np.random.seed(seed)
    syn_indices = np.random.permutation(num_syn_files)
    real_indices = np.random.permutation(num_real_files)
    data_txt_dir = syn_args.syn_data_dir

    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    num_val_syn = int(num_syn_files*syn_args.val_percent)
    lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_color_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, step))
    for i in syn_indices[:num_val_syn]:
        val_img_txt.write('%s\n' % all_syn_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_syn_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))

    num_val_real = int(num_real_files*syn_args.val_percent)
    for i in real_indices[:num_val_real]:
        val_img_txt.write('%s\n' % all_real_imgs[i])
        val_lbl_txt.write('%s\n' % os.path.join(real_lbl_dir, os.path.basename(all_real_imgs[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()

    for j in syn_indices[num_val_syn:]:
        trn_img_txt.write('%s\n' % all_syn_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_syn_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))

    for j in real_indices[num_val_real:]:
        trn_img_txt.write('%s\n' % all_real_imgs[j])
        trn_lbl_txt.write('%s\n' % os.path.join(real_lbl_dir, os.path.basename(all_real_imgs[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()


def get_args(cmt=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing the black and white ground truth synthetic annotations",
                        default='C:/Users/Tyler Feldman/Box/Bass Connections 2020-2021/Wind Turbine Object Detection Dataset/synthetic_wind_turbine_images/synthetic_image_labels_Oct28/')
    parser.add_argument("--syn_annos_dir", type=str, default='C:/Users/Tyler Feldman/Documents/Fall 2020 Classes/BassConnections/synthetic data/labels',
                        help="Directory where it will output the labels for each image")
    parser.add_argument("--syn_box_dir", type=str, default='C:/Users/Tyler Feldman/Documents/Fall 2020 Classes/BassConnections/synthetic data/bbox',
                        help="Directory where it will output any bbox images. This doesn't matter unless you're running draw_bbx_on_rgb_images")
    parser.add_argument("--syn_txt_dir", type=str, default='C:/Users/Tyler Feldman/Documents/Fall 2020 Classes/BassConnections/synthetic data/labels',
                        help="Directory where it will output the image/label paths along with other .txt files that reference the raw data (images/labels)")


    # Don't need to worry about these arguments
    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    parser.add_argument("--min_region", type=int, default=20, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=2,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=1, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.25, help="train:val=0.75:0.25")

    args = parser.parse_args()

    if not os.path.exists(args.syn_annos_dir):
        os.makedirs(args.syn_annos_dir)
    if not os.path.exists(args.syn_txt_dir):
        os.makedirs(args.syn_txt_dir)
    if not os.path.exists(args.syn_box_dir):
        os.makedirs(args.syn_box_dir)

    return args


if __name__ == '__main__':

    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    # px_thres: threshold for the length of edge lenght of b-box (at the margin)
    # whr_thres: threshold for width/height or height/width
    px_thres= 30 # 23
    whr_thres= 6 # 3
    display_types = ['color'] # 'mixed'
    cmt = 'wnd_syn'
    syn_args = get_args(cmt)
    for dt in display_types:
        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)

    
    '''
    create paths for the syn data to copy and paste into other txt files
    '''
    create_paths(comment=cmt)

    '''
    draw bbox on rgb images for syn_background data
    '''

    '''
    px_thres= 5 # 5 # 23 #20 #30
    whr_thres= 6 # 3
    display_types = ['color'] # 'mixed'
    cmt = ''
    syn_args = get_args(cmt)
    for dt in display_types:
        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)'''


