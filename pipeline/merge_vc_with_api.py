import pandas as pd
from os import listdir
from os.path import isfile, join, basename
import numpy as np
import cv2


api_data = pd.read_csv('/home/amanda/Github/impression_personality/vc-labels.txt', sep="\t", header=None)
api_data.columns = ["faceId", "faceTopDimension", "faceLeftDimension", "faceWidthDimension", "faceHeightDimension",
                    "smile", "pitch", "roll", "yaw", "gender", "age", "moustache", "beard", "sideburns", "glasses",
                    "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "blurlevel",
                    "blurvalue", "exposurelevel", "exposurevalue", "noiselevel", "noisevalue", "eymakeup", "lipmakeup",
                    "foreheadoccluded", "eyeoccluded", "mouthoccluded", "hair-bald", "hair-invisible", "img_name"]
api_data = api_data.drop(api_data.index[0])
api_data = api_data[['img_name', 'gender', 'age']]
api_data = api_data.drop_duplictates()  # there are some repetitive images.

print('Length of original API data is {}'.format(len(api_data)))

# calculate the intersection and unique set of cropped faces and api-processed faces.
api_face_lst = api_data['img_name'].values
crop_with_mask_dir = '/home/amanda/Documents/cropped_faces/vc_with_mask/'
all_face_lst = [f for f in listdir(crop_with_mask_dir) if isfile(join(crop_with_mask_dir, f))]

only_in_api = np.setdiff1d(api_face_lst, all_face_lst)
only_in_all_faces = np.setdiff1d(all_face_lst, api_face_lst)
intersect = list(set(api_face_lst) & set(all_face_lst))






