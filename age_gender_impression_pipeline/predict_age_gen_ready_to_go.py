import os
import cv2
import dlib
import numpy as np
from wide_resnet import WideResNet
import pandas as pd
import time


def predict_gen_and_age(current_folder_post_fix='0/',
                        crop_with_mask_root_dir='/home/amanda/Documents/cropped_new_crunchbase/',
                        save_result_root_dir='/home/amanda/Documents/predicted_results/crunchbase_new_data/'):
    # face detection
    start_t = time.time()
    crop_with_mask_dir = crop_with_mask_root_dir + current_folder_post_fix

    save_result_dir = save_result_root_dir + current_folder_post_fix
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    file_lst = [f for f in os.listdir(crop_with_mask_dir)if f[-4:] == '.jpg' or f[-4:] == '.png']

    file_num = len(file_lst)

    # load model and weights.
    img_size = 64
    faces = np.empty((file_num, img_size, img_size, 3))
    depth = 16
    k = 8
    weight_file = '/home/amanda/Documents/age_gender_model/weights.18-4.06.hdf5'

    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    img_height_lst = []
    img_width_lst = []
    for i, cur_im_name in enumerate(file_lst):
        if (i+1) % 100 == 0:
            print('{} out of {}'.format(i+1, len(file_lst)))
        cur_im_path = os.path.join(crop_with_mask_dir, cur_im_name)
        img = cv2.imread(cur_im_path)
        img_h, img_w, _ = np.shape(img)
        img_height_lst.append(img_h)
        img_width_lst.append(img_w)
        faces[i, :, :, :] = cv2.resize(img[:, :, :], (img_size, img_size))

    print('Start to make inferences...')

    results = model.predict(faces)
    print('Done!')

    predicted_genders = results[0]
    predicted_gender_prob = predicted_genders[:, 1]  # the probability of being a male, (0, 1).
    predicted_gender_binary = [round(i) for i in predicted_gender_prob]

    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    result_df = pd.DataFrame(columns=['filename', 'age', 'gender binary', 'gender probability', 'img height', 'img width'])
    result_df['gender probability'] = predicted_gender_prob
    result_df['gender binary'] = predicted_gender_binary
    result_df['age'] = predicted_ages
    result_df['filename'] = file_lst
    result_df['img height'] = img_height_lst
    result_df['img width'] = img_width_lst

    file_name = 'gender_age_only_ind_' + current_folder_post_fix[:-1] + '.pkl'
    file_full_path = save_result_dir + file_name
    result_df.to_pickle(file_full_path)

    print('Done. Total time = {}'.format(time.time()-start_t))

    # Lastly, visualize the results by separating photos into male and female groups. And draw labels and ages on them.


def viz_sample(current_folder_post_fix='0/',
               crop_with_mask_root_dir='/home/amanda/Documents/cropped_new_crunchbase/',
               save_result_root_dir='/home/amanda/Documents/predicted_results/crunchbase_new_data/'):

    def draw_label(image, point, label, gender_str, file_name, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        if gender_str == 'F':
            save_dir = '/home/amanda/Documents/cropped_new_crunchbase/test_sample/female/'
        else:
            save_dir = '/home/amanda/Documents/cropped_new_crunchbase/test_sample/male/'
        cv2.imwrite(save_dir+file_name, image)

    file_name = 'gender_age_only_ind_' + current_folder_post_fix[:-1] + '.pkl'
    save_result_dir = save_result_root_dir + current_folder_post_fix
    file_full_path = save_result_dir + file_name
    crop_with_mask_dir = crop_with_mask_root_dir + current_folder_post_fix

    df = pd.read_pickle(file_full_path)

    for ind, row in df.iterrows():

        if (ind+1) % 100 == 0:
            print ind+1, len(df)
        file_name = row['filename']
        img = cv2.imread(crop_with_mask_dir+file_name)
        if str(row['gender binary']) == '0.0':
            gender_str = "F"
        else:
            gender_str = "M"
        label = '{}, {}'.format(row['age'], gender_str)
        draw_label(img, (100, 100), label, gender_str, file_name)


predict_gen_and_age()
# viz_sample()

