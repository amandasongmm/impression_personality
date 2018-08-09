import pandas as pd
import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from os import listdir
from os.path import isfile, join
import time

# todo: test how accurate the gender prediciton and age prediction are. Save the number and report it to Will.
start_t = time.time()
# # prepare gender and age ground truth.
# df = pd.read_excel('../static/demographic-others-labels.xlsx', sheet_name='Final Values')
# keep_names = ['Filename', 'Age', 'Gender']
# df_select = df[keep_names]
# df_select.to_pickle('../static/gender_age_2k_gt.pkl')
df = pd.read_pickle('../static/gender_age_2k_gt.pkl')

mypath = '/home/amanda/Documents/VC and Entrepreneur Faces Project/New_data/crunchbase/0/0a0/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[0] != '.']
two_k_face_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces/'

# face detection
detector = dlib.get_frontal_face_detector()
face_num = len(df)
# load model and weights
batch_size = face_num
x_left_margin = 0.15
x_right_margin = 0.15
y_top_margin = 0.35
y_bottom_margin = 0.12
margin = 0.25
img_size = 64
faces = np.empty((batch_size, img_size, img_size, 3))
depth = 16
k = 8
weight_file = '/home/amanda/Desktop/weights.18-4.06.hdf5'
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

count = 0

# for index, row in df[:20].iterrows():
crop_with_mask_dir = ''

for index, row in enumerate(onlyfiles):
    if index % 100 == 0:
        print index + 1
    # cur_file_name = row['Filename']
    cur_file_name = row
    img_path = mypath + cur_file_name
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(img)

    # detect faces using dlib detector
    detected = detector(img, 1)

    if len(detected) == 0:
        print('No face detected in this photo')
        print(cur_file_name)
        #todo: save this image into a separate folder. Keep a list of faces that belong to this category.

    elif len(detected) > 1:
        print('More than one face.')
        # todo: save this image into a separate folder. Keep a list of faces that belong to this category.
    else:
        for d in detected:
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - x_left_margin * w), 0)
            yw1 = max(int(y1 - y_top_margin * h), 0)
            xw2 = min(int(x2 + x_right_margin * w), img_w - 1)
            yw2 = min(int(y2 + y_bottom_margin * h), img_h - 1)
            # faces[index, :, :, :] = cv2.resize(img, (img_size, img_size))

            roi_color = img[yw1: yw2+1, xw1: xw2+1, :]

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

            # save the cropped images in a folder.
            cv2.imwrite(crop_with_mask_dir + file_name, with_mask)







            faces[index, :, :, :] = cv2.resize(img[yw1:yw2+1, xw1:xw2+1, :], (img_size, img_size))

            # save cropped faces in another folder.
            save_dir = '/home/amanda/Documents/cropped_test/'

            save_name = save_dir + cur_file_name
            cv2.imwrite(save_name, img[yw1:yw2+1, xw1:xw2+1, :])

            count += 1

print count

# # predict ages and genders of the detected faces.
# results = model.predict(faces)
# predicted_genders = results[0]
# ages = np.arange(0, 101).reshape(101, 1)
# predicted_ages = results[1].dot(ages).flatten()
# # print predicted_genders
# # print predicted_ages
#
# np.savez('predicted_age_gen.npz', age=predicted_ages, gender=predicted_genders)

print('time = {}'.format(time.time()-start_t))
