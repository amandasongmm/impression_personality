import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


# face detection
detector = dlib.get_frontal_face_detector()
print('Done')

# load model and weights
batch_size = 1
margin = 0.1
img_size = 64
faces = np.empty((batch_size, img_size, img_size, 3))
depth = 16
k = 8
weight_file = '/home/amanda/Desktop/weights.18-4.06.hdf5'
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

count = 0
img_path = '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/e/e0/e183_cb.jpeg'
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = np.shape(img)

# detect faces using dlib detector
detected = detector(img, 1)

if len(detected) == 0:
    print('No face detected in this photo')

    #todo: save this image into a separate folder. Keep a list of faces that belong to this category.

elif len(detected) > 1:
    print('More than one face.')
    # todo: save this image into a separate folder. Keep a list of faces that belong to this category.

else:
    print('One face')
    for d in detected:
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), img_w - 1)
        yw2 = min(int(y2 + margin * h), img_h - 1)
        faces[count, :, :, :] = cv2.resize(img[yw1:yw2+1, xw1:xw2+1, :], (img_size, img_size))
        # save cropped faces in another folder.
        save_dir = '/home/amanda/Documents/cropped_2k_test/'
        count += 1

# predict ages and genders of the detected faces.
results = model.predict(faces)
predicted_genders = results[0]
ages = np.arange(0, 101).reshape(101, 1)
predicted_ages = results[1].dot(ages).flatten()
print predicted_genders
print predicted_ages



