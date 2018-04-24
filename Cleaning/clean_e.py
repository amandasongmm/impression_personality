import os
import cv2
import numpy as np
import pandas as pd
import os.path


# change the dir path to fit your local setting.
vc_data_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/'
e_save_dir = '/home/amanda/Documents/cropped_face/e_no_mask/'


def create_clean_face_for_e():
    # if you pull the latest git hub repo, e_df.pkl should already be there. No need to change.
    e_df_save_path = '../pretrain_first_impression/tmp_data/e_df.pkl'

    e_df = pd.read_pickle(e_df_save_path)
    # the pre-saved fields are as follows
    # e_feat_lst = ['uuid', 'short_id', 'directory', 'tw_img_path', 'cb_img_path',
    #                   'E - Employees (Log)', 'E - VC Raised (log)', 'E - VC Rds',
    #                   'E - Total Raised (log)', 'E - Total Rds', 'E - IPO']

    # prepare new columns to fill in the new paths
    e_df['cropped_cb_photo_path'] = ''
    e_df['cropped_cb_photo_note'] = 0

    e_df['cropped_tw_photo_path'] = ''
    e_df['cropped_tw_photo_note'] = 0

    # load haar cascade filters
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    row_data_len = len(e_df)
    e_df = e_df.reset_index()
    i = 0
    for index, row in e_df.iterrows():

        if index % 100 == 0:
            print('Processing {} out of {}'.format(index+1, row_data_len))

        tw_img_path = vc_data_dir + 'Extracted Faces/e/' + row['tw_img_path']
        cb_img_path = vc_data_dir + 'Extracted Faces/e/' + row['cb_img_path']

        if not os.path.isfile(tw_img_path):
            row['cropped_tw_photo_note'] = '0: not exist'
        else:
            img = cv2.imread(tw_img_path)

            if img is None:
                row['cropped_tw_photo_note'] = '1: cannot open a file'
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) < 1:
                    row['cropped_tw_photo_note'] = '2: contain no face'
                elif len(faces) > 1:
                    row['cropped_tw_photo_note'] = '3: contain multiple faces'
                else:
                    i += 1
                    for (x, y, w, h) in faces:
                        roi_color = img[int(y-0.1*h):int(y+1.1*h), int(x-0.1*w):int(x+1.1*w), :]

                        img_save_name = e_save_dir + row['short_id'] + '_tw.jpg'
                        row['cropped_tw_photo_path'] = img_save_name
                        row['cropped_tw_photo_note'] = '4: successful crop'
                        cv2.imwrite(img_save_name, roi_color)

        if not os.path.isfile(cb_img_path):
            row['cropped_cb_photo_note'] = '0: not exist'
        else:
            img = cv2.imread(cb_img_path)

            if img is None:
                row['cropped_cb_photo_note'] = '1: cannot open a file'
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) < 1:
                    row['cropped_cb_photo_note'] = '2: contain no face'
                elif len(faces) > 1:
                    row['cropped_cb_photo_note'] = '3: contain multiple faces'
                else:
                    i += 1
                    for (x, y, w, h) in faces:
                        roi_color = img[int(y-0.1*h):int(y+1.1*h), int(x-0.1*w):int(x+1.1*w), :]

                        img_save_name = e_save_dir + row['short_id'] + '_cb.jpg'
                        row['cropped_cb_photo_path'] = img_save_name
                        row['cropped_cb_photo_note'] = '4: successful crop'
                        print(img_save_name)
                        print(roi_color.shape)
                        print(img.shape)
                        a = 12

                        cv2.imwrite(img_save_name, roi_color)

    # save the new df.
    e_df.to_pickle('../pretrain_first_impression/tmp_data/e_df_with_crops.pkl')

    return



if __name__ == '__main__':
    create_clean_face_for_e()
