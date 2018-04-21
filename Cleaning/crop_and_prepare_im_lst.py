import glob
import os
import cv2
import numpy as np
import pandas as pd
import os.path

cwd = os.getcwd()

# change these (or rename your folders to match the dir paths)
e_path = ''
vc_path = ''
vc_df_save_path='../pretrain_first_impression/tmp_data/vc_df.pkl'
vc_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/'


def prepare_im_df(e_df_save_path, vc_df_save_path):
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


prepare_im_df(e_df_save_path='../pretrain_first_impression/tmp_data/e_df.pkl',
              vc_df_save_path='../pretrain_first_impression/tmp_data/vc_df.pkl')


def crop_face():
    vc_cropped_dir = '/home/amanda/Documents/cropped_v/'
    vc_df = pd.read_pickle(vc_df_save_path)
    vc_df['cropped_cb_photo_path'] = ''
    vc_df['cropped_cb_photo_notes'] = 0

    # load haar cascade filters
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for index, row in vc_df.iterrows():
        with open('../pretrain_first_impression/tmp_data/vc_df_ind_counter.txt', 'w') as f:
            f.write('%d' % index)

        if index % 10 == 0:
            print index

        cur_img_path = vc_dir + 'Extracted Faces/vc/' + row['cb_img_path']
        if not os.path.isfile(cur_img_path):
            row['cropped_cb_photo_notes'] = '0: not exist'
        else:
            img = cv2.imread(cur_img_path)

            if img is None:
                row['cropped_cb_photo_notes'] = '1: cannot open'

            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) < 1:
                    row['cropped_cb_photo_notes'] = '2: contain no face'
                elif len(faces) > 1:
                    row['cropped_cb_photo_notes'] = '3: contain multiple faces'
                else:
                    for (x, y, w, h) in faces:
                        roi_color = img[int(y-0.1*h):int(y+1.1*h), int(x-0.1*w):int(x+1.1*w), :]

                        white_mask = 255 * np.ones_like(roi_color)
                        a = 12
                        width_x = white_mask.shape[1]
                        height_y = white_mask.shape[0]
                        center_x = int(width_x // 2)
                        center_y = int(height_y // 2)

                        white_mask = cv2.ellipse(white_mask, (center_x, center_y), (width_x, height_y), 0, 0, 360, (0, 0, 0), -1, 8)

                        final_crop = cv2.bitwise_or(roi_color, white_mask)

                        img_save_name = vc_cropped_dir + row['short_id'] + '_cb.jpg'
                        row['cropped_cb_photo_path'] = img_save_name
                        row['cropped_cb_photo_notes'] = '4: successful crop'
                        cv2.imwrite(img_save_name, final_crop)
    # there are 13160 cropped faces generated. but not all of the .jpeg file are readable.
    return


def filter_readable_faces():
    return

crop_face()




