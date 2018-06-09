import os
import cv2
import tensorflow as tf
import numpy as np
import skimage
import skimage.io
import skimage.transform
import math
import time
import pandas as pd
import pickle
from scipy.stats.stats import pearsonr
from os import listdir
from os.path import isfile, join, basename
pd.options.mode.chained_assignment = None  # default='warn'


# change the raw_data_dir, crop_dir and crop_with_mask_dir to fit your need.
raw_data_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/'
crop_dir = '/home/amanda/Documents/cropped_faces/vc/'
crop_with_mask_dir = '/home/amanda/Documents/cropped_faces/vc_with_mask/'


# change these (or rename your folders to match the dir paths)
e_path = ''
vc_path = ''
vc_df_save_path = '../pretrain_first_impression/tmp_data/vc_df.pkl'
vc_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/'


def prepare_im_df(e_df_save_path='../pretrain_first_impression/tmp_data/e_df.pkl',
                  vc_df_save_path='../pretrain_first_impression/tmp_data/vc_df.pkl'):
    """
    Convert the raw uuid_dir_with_ranking.xlsx into Entrepreneurs and VC df.
    :param e_df_save_path:
    :param vc_df_save_path:
    :return:
    """
    vc_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/'
    ranking_file_name = 'uuid_dir_with_rankings.xlsx'

    # prepare two separate excel files. One for VC. One for E.
    df_full = pd.read_excel(open(vc_dir + ranking_file_name, 'rb'), sheet_name='uuid_dir')
    df_full['tw_img_path'] = df_full['directory'] + '/' + df_full['short_id'] + '_tw.jpeg'
    df_full['cb_img_path'] = df_full['directory'] + '/' + df_full['short_id'] + '_cb.jpeg'

    # vc data frame
    vc_index = df_full['VC'] == 1
    vc_df = df_full[vc_index]
    vc_feat_lst = ['uuid', 'short_id', 'directory', 'tw_img_path', 'cb_img_path',
                   'VC - Investment # rank', 'VC - Investment $ rank']
    vc_df = vc_df[vc_feat_lst]

    # E data frame
    e_index = df_full['VC'] == 0
    e_df = df_full[e_index]
    e_feat_lst = ['uuid', 'short_id', 'directory', 'tw_img_path', 'cb_img_path',
                  'E - Employees (Log)', 'E - VC Raised (log)', 'E - VC Rds',
                  'E - Total Raised (log)', 'E - Total Rds', 'E - IPO']
    e_df = e_df[e_feat_lst]

    # save the data frames separately
    vc_df.to_pickle(vc_df_save_path)
    e_df.to_pickle(e_df_save_path)
    return

# prepare_im_df()


def fetch_file(path_to_folder, flag, key_word):
    '''
    fetch_files() requires three arguments: pathToFolder, flag and keyWord

    flag must be 'start_with' or 'end_with'
    keyWord is a string to search the file's name

    Be careful, the keyWord is case sensitive and must be exact

    Example: fetch_file('/Documents/Photos/','end_with','.jpg')

    returns: _pathToFiles and _fileNames
    '''

    _path_to_files = []
    _file_names = []

    for dir_path, dir_names, file_names in os.walk(path_to_folder):
        if flag == 'end_with':
            raw_file = [item for item in file_names if item.endswith(key_word)]
            selected_file = [item for item in raw_file if not item.startswith('._')]
            selected_file = list(set(selected_file))
            _file_names.extend(selected_file)

            selected_path = [os.path.join(dir_path, item) for item in selected_file]
            _path_to_files.extend(selected_path)

        elif flag == 'start_with':
            raw_file = [item for item in file_names if item.startswith(key_word)]
            selected_file = [item[2:] if item.startswith('._') else item for item in raw_file]
            selected_file = list(set(selected_file))
            _file_names.extend(selected_file)

            selected_path = [os.path.join(dir_path, item) for item in selected_file]
            _path_to_files.extend(selected_path)

        else:
            print fetch_file.__doc__
            break

        # try to remove empty entries if none of the required files are in the directory
        try:
            _path_to_files.remove('')
            _file_names.remove('')
        except ValueError:
            pass

        # warn if nothing was found in the given path
        if selected_file == []:
            print('No files with given parameters were found in {}\n'.format(dir_path))

    print('{} files are found in searched folders.'.format(len(_file_names)))
    return _path_to_files, _file_names


def crop_faces(raw_data_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
               crop_dir='/home/amanda/Documents/cropped_faces/vc_narrow/',
               crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask_narrow/'):

    '''
        crop_faces() requires three arguments: raw_data_dir, crop_dir and crop_with_mask_dir

        raw_data_dir is the root dir that you want to run the analysis on. e.g.
        '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/' will search for all the *.jpg
        files under the subfolder of vc recursively. namely, all the vc images.

        crop_dir is a folder where you want to save the cropped faces (with no mask)

        crop_with_mask_dir is where you want to save the cropped and masked faces.

        Example: raw_data_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
        crop_dir='/home/amanda/Documents/cropped_faces/vc/',
        crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask/'
    '''

    path_to_files, file_names = fetch_file(raw_data_dir, 'end_with', 'jpeg')  # get all the files in the sub directories

    total_file_len = len(file_names)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    if not os.path.exists(crop_with_mask_dir):
        os.makedirs(crop_with_mask_dir)

    face_cascade = cv2.CascadeClassifier('../Cleaning/haarcascade_frontalface_default.xml')
    count = 0
    for path, file_name in zip(path_to_files, file_names):
        img = cv2.imread(path)
        if img is not None:
            height, width, channel = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    if w * h > 900 and int(y - 0.1 * h) > 0 and int(y + 1.1 * h) < height and int(x - 0.05 * w) > 0 \
                            and int(x + 1.05 * w) < width:
                        count += 1
                        roi_color = img[int(y - 0.1 * h):int(y + 1.1 * h), int(x - 0.05 * w):int(x + 1.05 * w), :]
                        img_save_name = crop_dir + file_name
                        cv2.imwrite(img_save_name, roi_color)

                        # add mask and save masked image.
                        mask = np.zeros_like(roi_color)
                        rows, cols, _ = mask.shape

                        # create a black filled ellipse
                        cen_x = int(rows // 2)
                        cen_y = int(cols // 2)
                        cv2.ellipse(mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0,
                                    endAngle=360.0, color=(255, 255, 255), thickness=-1)
                        # bitwise
                        result = np.bitwise_and(roi_color, mask)

                        # flipped mask
                        new_mask = 255 * np.ones_like(roi_color)
                        cv2.ellipse(new_mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0,
                                    endAngle=360,
                                    color=(0, 0, 0), thickness=-1)

                        # final result
                        with_mask = result + new_mask
                        cv2.imwrite(crop_with_mask_dir + file_name, with_mask)

                        if count % 100 == 0:
                            print count, total_file_len
    # libpng warning: iCCP: known incorrect sRGB profile
    # libpng warning: iCCP: profile 'icc': 1000000h: invalid rendering intent
    return


def sort_face(crop_with_mask_dir=crop_with_mask_dir,
              vc_filter_df_save_path='../pretrain_first_impression/tmp_data/vc_filtered_df.pkl'):
    """
    Create a df of VCs who have either tw or cb photos, and their corresponding ranking.
    the new vc_filtered_df includes the following columns, 'short_id' of vc, 'tw_img_path', 'tw_color' 1- color, 0 -gray

    :param crop_with_mask_dir:
    :param vc_filter_df_save_path:
    :return:
    """
    start_t = time.time()
    vc_df = pd.read_pickle(vc_df_save_path)

    # list all the files in crop_with_mask_dir.
    all_face_lst = [f for f in listdir(crop_with_mask_dir) if isfile(join(crop_with_mask_dir, f))]

    vc_filter_df = pd.DataFrame([], index=np.arange(len(vc_df)),
                                columns=['short_id', '#rank', '$rank',
                                         'tw_img_exist', 'tw_img_path', 'tw_shape', 'tw_color',
                                         'cb_img_exist', 'cb_img_path', 'cb_shape', 'cb_color'])

    for i, row in vc_df.iterrows():
        if i % 200 == 0:
            print i, len(vc_df)
        vc_filter_df['short_id'].loc[i] = row['short_id']
        tw_file = basename(row['tw_img_path']).encode('ascii', 'ignore')
        cb_file = basename(row['cb_img_path']).encode('ascii', 'ignore')

        if tw_file in all_face_lst:
            vc_filter_df['tw_img_exist'].loc[i] = 1
            img_full_path = crop_with_mask_dir + tw_file
            vc_filter_df['tw_img_path'].loc[i] = img_full_path

            # check gray scale or not
            img = cv2.imread(img_full_path)
            vc_filter_df['tw_shape'].loc[i] = img.shape[:2]
            residual = np.mean(img, axis=2) - img[:, :, 0]
            if np.sum(np.sum(residual, axis=0), axis=0) != 0:
                vc_filter_df['tw_color'].loc[i] = 1
            else:
                vc_filter_df['tw_color'].loc[i] = 0
        else:
            vc_filter_df['tw_img_exist'].loc[i] = 0

        if cb_file in all_face_lst:
            vc_filter_df['cb_img_exist'].loc[i] = 1
            img_full_path = crop_with_mask_dir + cb_file
            vc_filter_df['cb_img_path'].loc[i] = img_full_path

            img = cv2.imread(img_full_path)
            vc_filter_df['cb_shape'].loc[i] = img.shape[:2]
            residual = np.mean(img, axis=2) - img[:, :, 0]
            if np.sum(np.sum(residual, axis=0), axis=0) != 0:
                vc_filter_df['cb_color'].loc[i] = 1
            else:
                vc_filter_df['cb_color'].loc[i] = 0
        else:
            vc_filter_df['cb_img_exist'].loc[i] = 0

        vc_filter_df['#rank'].loc[i] = row['VC - Investment # rank']
        vc_filter_df['$rank'].loc[i] = row['VC - Investment $ rank']

    vc_filter_df['exist'] = vc_filter_df['tw_img_exist'] + vc_filter_df['cb_img_exist']
    print(sum(vc_filter_df['exist'] == 0))
    vc_filter_df = vc_filter_df[vc_filter_df['exist'] != 0]
    print(len(vc_filter_df))
    vc_filter_df.to_pickle(vc_filter_df_save_path)
    print(time.time()-start_t)
    return


crop_faces()
# sort_face()


