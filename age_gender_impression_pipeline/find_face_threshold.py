from __future__ import division
import cv2
import dlib
import time
import pandas as pd
from os.path import isfile, join
from os import listdir


def crop_faces():
    feat_prefixs = ['crunchbase', 'twitter', 'linkedin']
    detector = dlib.get_frontal_face_detector()  # detect faces using dlib detector.

    for feat_prefix in feat_prefixs:

        print('Current folder = {}'.format(feat_prefix))
        crop_with_mask_root_dir = '/home/amanda/Documents/cropped_new_' + feat_prefix + '/'
        csv_result_dir = '/home/amanda/Documents/predicted_results/' + feat_prefix + '_new_data/to_will/'
        csv_result_dropbox_dir = '/home/amanda/Dropbox/VC and Entrepreneur Faces Project/August_2018_result/'\
                                 + feat_prefix + '_data/'
        csv_files = [f for f in listdir(csv_result_dir) if isfile(join(csv_result_dir, f))]

        start_t = time.time()

        for ind, cur_csv_file in enumerate(csv_files):
            print cur_csv_file
            tmp, post_fix = cur_csv_file.split('age_')
            post_fix, _ = post_fix.split('.csv')
            crop_with_mask_cur_dir = crop_with_mask_root_dir + post_fix + '/'
            df = pd.read_csv(join(csv_result_dir, cur_csv_file))

            print('Total number of files in this folder is {}'.format(len(df)))
            prob_array = []
            face_num_array = []

            no_face_count = 0
            multi_face_count = 0
            one_face_count = 0

            for cur_im_name in df['filename']:
                cur_im_path = join(crop_with_mask_cur_dir, cur_im_name)
                img = cv2.imread(cur_im_path)
                dets, scores, indx = detector.run(img, 1, -1)

                if len(scores) == 0:
                    print('No face detected'.format(no_face_count))
                    prob_array.append(0)
                    no_face_count += 1
                    face_num_array.append(-3)
                elif len(scores) > 1:
                    if multi_face_count % 50 == 0:
                        print('{} have multiple faces detected'.format(multi_face_count))
                    prob_array.append(scores[0])
                    multi_face_count += 1
                    face_num_array.append(2)
                else:
                    face_num_array.append(1)
                    prob_array.append(scores[0])
                    one_face_count += 1
                    if one_face_count % 100 == 0:
                        print('{} one face out of {}, csv = {}, feat = {}\n\n'.format(one_face_count, len(df),
                                                                                      cur_csv_file, feat_prefix))

            df['face prob'] = prob_array
            csv_file_new_name = 'prob_' + cur_csv_file
            csv_file_save_path = join(csv_result_dir, csv_file_new_name)

            df.to_csv(csv_file_save_path)

            df.to_csv(join(csv_result_dropbox_dir, csv_file_new_name))
            print('Elapsed time so far = {}. csv ind = {}, total csv file number = {}'.format(time.time()-start_t, ind+1, len(csv_files)))


crop_faces()



