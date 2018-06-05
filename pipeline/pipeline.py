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
from os.path import isfile, join


class CleanOldData:
    def __init__(self,
                 im_df_path='../pretrain_first_impression/tmp_data/vc_df.pkl',
                 root_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
                 crop_dir='/home/amanda/Documents/cropped_faces/vc/',
                 crop_df_path='../pretrain_first_impression/tmp_data/vc_cropped_df.pkl'):
        # im_df is prepared by crop_and_prepare_im_lst.py in Cleaning dir. (prepare_im_df)
        self.im_df_path = im_df_path
        self.root_dir = root_dir
        self.crop_dir = crop_dir
        self.crop_df_path = crop_df_path
        self.face_cascade = cv2.CascadeClassifier('../ Cleaning/haarcascade_frontalface_default.xml')
        # load haar cascade filters

    def crop_face(self):
        df = pd.read_pickle(self.im_df_path)
        df['tw_exist'] = 0
        df['cb_exist'] = 0

        if not os.path.exists(self.crop_dir):
            os.makedirs(self.crop_dir)

        df_len = len(df)

        for ind, row in df.iterrows():
            if ind % 100 == 0:
                print('{} out of {} people...'.format(ind, df_len))

            tw_img_path = self.root_dir + row['tw_img_path']
            cb_img_path = self.root_dir + row['cb_img_path']

            # df.iloc[ind]['tw_img_path'] = ''
            # df.iloc[ind]['cb_img_path'] = ''

            # Deal with twitter data.
            if os.path.isfile(tw_img_path):
               img = cv2.imread(tw_img_path)
               if img is not None:
                   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                   if len(faces) == 1:
                       for (x, y, w, h) in faces:
                           if w * h > 100:
                               roi_color = img[int(y - 0.1 * h):int(y + 1.1 * h), int(x - 0.1 * w):int(x + 1.1 * w), :]
                               img_save_name = self.crop_dir + row['short_id'] + '_tw.jpg'
                               df.iloc[ind]['tw_img_path'] = img_save_name
                               df.iloc[ind]['tw_exist'] = 1
                               cv2.imwrite(img_save_name, roi_color)

            # Deal with crunchbase data.
            if os.path.isfile(cb_img_path):
               img = cv2.imread(cb_img_path)
               if img is not None:
                   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                   if len(faces) == 1:
                       for (x, y, w, h) in faces:
                           if w * h > 100:
                               roi_color = img[int(y - 0.1 * h):int(y + 1.1 * h), int(x - 0.1 * w):int(x + 1.1 * w), :]
                               img_save_name = self.crop_dir + row['short_id'] + '_cb.jpg'
                               df.iloc[ind]['cb_img_path'] = img_save_name
                               df.iloc[ind]['cb_exist'] = 1
                               cv2.imwrite(img_save_name, roi_color)

        df = df[df['tw_exist']+df['cb_exist'] != 0]
        print('Done. {} people have at least one photo. Total photo number = {}'.
              format(len(df), sum(df['cb_exist']+df['tw_exist'])))
        df.to_pickle(self.crop_df_path)


def fetch_file(path_to_folder, flag, key_word):
    '''
    fetchFiles() requires three arguments: pathToFolder, flag and keyWord

    flag must be 'start_with' or 'end_with'
    keyWord is a string to search the file's name

    Be careful, the keyWord is case sensitive and must be exact

    Example: fetchFiles('/Documents/Photos/','ENDS_WITH','.jpg')

    returns: _pathToFiles and _fileNames
    '''

    _path_to_files = []
    _file_names = []

    for dir_path, dir_names, file_names in os.walk(path_to_folder):
        if flag == 'end_with':
            raw_file = [item for item in file_names if item.endswith(key_word)]
            selected_file = [item[2:] if item.startswith('._') else item for item in raw_file]
            _file_names.extend(selected_file)

            selected_path = [os.path.join(dir_path, item) for item in selected_file]
            _path_to_files.extend(selected_path)

        elif flag == 'start_with':
            raw_file = [item for item in file_names if item.startswith(key_word)]
            selected_file = [item[2:] if item.startswith('._') else item for item in raw_file]
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


def crop_faces(crop_dir='/home/amanda/Documents/cropped_faces/vc/',
               crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask/'):
    path_to_files, file_names = fetch_file(
        '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
        'end_with', 'jpeg')

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
                    if (w * h > 100 and int(y - 0.1 * h)>0 and int(y + 1.1 * h)<height and int(x - 0.1 * w)>0 and int(x + 1.1 * w)<width):
                        count += 1
                        roi_color = img[int(y - 0.1 * h):int(y + 1.1 * h), int(x - 0.1 * w):int(x + 1.1 * w), :]
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
    return


# crop_faces()
# libpng warning: iCCP: known incorrect sRGB profile
# libpng warning: iCCP: profile 'icc': 1000000h: invalid rendering intent
