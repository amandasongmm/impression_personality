from __future__ import division
import skimage.io
import skimage.transform
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import cv2
import dlib
from wide_resnet import WideResNet
import os
import tensorflow as tf
import numpy as np
import skimage
import skimage.io
import skimage.transform
import time
import pandas as pd
import pickle
from sklearn import preprocessing
from keras import backend as K

# global parameters that don't change
vgg_weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
default_device = '/gpu:0'
vgg_mean = [103.939, 116.779, 123.68]
feat_save_dir = '/home/amanda/Documents/feat_folder/'  # extracted feature from either 2k dir, or new dataset dir.
model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'  # save pca and regression model trained on 2k.
feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']

if not os.path.exists(feat_save_dir):
    os.makedirs(feat_save_dir)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)


# common function
def load_image(image_path, mean=vgg_mean):
    image = skimage.io.imread(image_path)
    resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
    bgr = resized_image[:, :, ::-1] - mean
    return bgr


# the maxpool1 is using conv1_1, instead of conv1_2. somehow it gives better performance.
class Vgg16Model:

    def __init__(self, weights_path):
        self.weights = np.load(weights_path, encoding='latin1')
        self.activation_fn = tf.nn.relu
        self.conv_padding = "SAME"
        self.pool_padding = "SAME"
        self.use_bias = True

    def build(self, input_sensor, trainable=False):
        self.conv1_1 = self.conv2d(input_sensor, 'conv1_1', 64, trainable)
        self.conv1_2 = self.conv2d(self.conv1_1, 'conv1_2', 64, trainable)

        # max-pooling is performed over a 2x2 pixel window, with stride 2
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_1, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv2_1 = self.conv2d(self.max_pool1, 'conv2_1', 128, trainable)
        self.conv2_2 = self.conv2d(self.conv2_1, 'conv2_2', 128, trainable)

        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv3_1 = self.conv2d(self.max_pool2, 'conv3_1', 256, trainable)
        self.conv3_2 = self.conv2d(self.conv3_1, 'conv3_2', 256, trainable)
        self.conv3_3 = self.conv2d(self.conv3_2, 'conv3_3', 256, trainable)

        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv4_1 = self.conv2d(self.max_pool3, 'conv4_1', 512, trainable)
        self.conv4_2 = self.conv2d(self.conv4_1, 'conv4_2', 512, trainable)
        self.conv4_3 = self.conv2d(self.conv4_2, 'conv4_3', 512, trainable)

        self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv5_1 = self.conv2d(self.max_pool4, 'conv5_1', 512, trainable)
        self.conv5_2 = self.conv2d(self.conv5_1, 'conv5_2', 512, trainable)
        self.conv5_3 = self.conv2d(self.conv5_2, 'conv5_3', 512, trainable)

        self.reshaped_conv52 = tf.reshape(self.conv5_2, shape=(-1, 14 * 14 * 512))

        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)
        self.reshaped = tf.reshape(self.max_pool5, shape=(-1, 7 * 7 * 512))

        self.fc6 = self.fc(self.reshaped, 'fc6', 4096, trainable)
        self.fc7 = self.fc(self.fc6, 'fc7', 4096, trainable)

        self.fc8 = self.fc(self.fc7, 'fc8', 1000, trainable)
        self.predictions = tf.nn.softmax(self.fc8, name='predictions')

    def conv2d(self, layer, name, n_filters, trainable, k_size=3):
        weights = self.weights[name + '_W']
        bias = self.weights[name + '_b']
        return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                activation=self.activation_fn,
                                padding=self.conv_padding,
                                name=name,
                                trainable=trainable,
                                kernel_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),
                                use_bias=self.use_bias)

    def fc(self, layer, name, size, trainable):
        weights = self.weights[name + '_W']
        bias = self.weights[name + '_b']
        return tf.layers.dense(layer, size, activation=self.activation_fn,
                               name=name, trainable=trainable,
                               kernel_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                               bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),
                               use_bias=self.use_bias)


# common function
def extract_features(img_dir, file_names, feat_prefix, batch_size):
    start_zero = time.time()
    tf.reset_default_graph()

    # create mappings of filenames --> vgg features
    num_files = len(file_names)
    num_batches = int(np.ceil(num_files / batch_size))

    conv52_arr = np.empty((0, 100352), dtype=np.float32)

    valid_file_lst = []

    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:
            _input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='images')
            vgg = Vgg16Model(vgg_weights_path)
            vgg.build(_input)

            sess.run(tf.global_variables_initializer())

            for i in range(num_batches):

                if i == num_batches - 1:
                    batch_filenames = file_names[i * batch_size:]
                else:
                    batch_filenames = file_names[i * batch_size: (i + 1) * batch_size]

                if i % 5 == 0:
                    print('batch {} out of {}. '.format(i + 1, num_batches))
                start_t = time.time()
                images = np.array([load_image(os.path.join(img_dir, f)) for f in batch_filenames])

                batch_conv52 = sess.run(vgg.reshaped_conv52, feed_dict={_input: images})

                conv52_arr = np.append(conv52_arr, np.array(batch_conv52), axis=0)

                valid_file_lst += batch_filenames
                end_t = time.time()
                print('\t this batch takes {:.2f} sec'.format(end_t - start_t))

            print('valid file num = {}'.format(len(valid_file_lst)))

            np.save(os.path.join(feat_save_dir, feat_prefix + '_conv52.npy'), conv52_arr)
            np.save(os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy'), valid_file_lst)

    print('Feature extraction done. time elapsed = {:.2f} sec.'.format(time.time() - start_zero))

    return conv52_arr, valid_file_lst


def crop_faces(raw_data_dir_postfix, raw_data_root_dir, crop_with_mask_root_dir
               ):
    #     crop_faces() requires two arguments: raw_data_dir and crop_with_mask_dir
    #
    #     raw_data_dir is the root dir that you want to run the analysis on. e.g.
    #     '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/' will search for all the *.jpg
    #     files under the subfolder of vc recursively. namely, all the vc images.
    #
    #     crop_with_mask_dir is where you want to save the cropped faces with an oval mask.
    #
    #     Example: raw_data_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
    #     crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask/'

    def fetch_file(path_to_folder, flag, key_word):
        # fetch_files() requires three arguments: pathToFolder, flag and keyWord
        #
        # flag must be 'start_with' or 'end_with'
        # keyWord is a string to search the file's name
        #
        # Be careful, the keyWord is case sensitive and must be exact
        #
        # Example: fetch_file('/Documents/Photos/','end_with','.jpg')
        #
        # returns: _pathToFiles and _fileNames

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

    start_t = time.time()
    raw_data_dir = raw_data_root_dir + raw_data_dir_postfix

    crop_with_mask_dir = crop_with_mask_root_dir + raw_data_dir_postfix
    if not os.path.exists(crop_with_mask_dir):
        os.makedirs(crop_with_mask_dir)

    no_face_dir = crop_with_mask_root_dir + 'No_face/' + raw_data_dir_postfix
    if not os.path.exists(no_face_dir):
        os.makedirs(no_face_dir)

    multi_face_dir = crop_with_mask_root_dir + 'Multi_face/' + raw_data_dir_postfix
    if not os.path.exists(multi_face_dir):
        os.makedirs(multi_face_dir)

    cannot_read_dir = crop_with_mask_root_dir + 'cannot_read/' + raw_data_dir_postfix
    if not os.path.exists(cannot_read_dir):
        os.makedirs(cannot_read_dir)

    end_with_key_word = 'g'

    path_to_files, file_names = fetch_file(raw_data_dir, 'end_with', end_with_key_word)  # all files in subdirectories
    total_file_len = len(file_names)
    print('Total number of files in this folder is {}'.format(total_file_len))

    detector = dlib.get_frontal_face_detector()  # detect faces using dlib detector.

    # Face detection related parameters.
    x_left_margin = 0.15
    x_right_margin = 0.15
    y_top_margin = 0.35
    y_bottom_margin = 0.12

    count = 0
    cannot_read_face_count = 0
    no_face_count = 0
    multi_face_count = 0
    for i, (file_name_only, file_path) in enumerate(zip(file_names, path_to_files)):

        #todo: for debug purpose.
        print file_name_only



        img = cv2.imread(file_path)
        if img is None:
            cannot_read_face_count += 1
            if cannot_read_face_count % 5 == 0:
                print('So far #{} photos cannot be read'.format(cannot_read_face_count))
            cv2.imwrite(cannot_read_dir + file_name_only, img)
            continue

        img_h, img_w, _ = np.shape(img)

        detected = detector(img, 1)

        if len(detected) == 0:
            no_face_count += 1
            if no_face_count % 5 == 0:
                print('So far #{} photos have no face'.format(no_face_count))
            cv2.imwrite(no_face_dir + file_name_only, img)

        elif len(detected) > 1:
            multi_face_count += 1
            if multi_face_count % 5 == 0:
                print('So far #{} photos have multiple faces'.format(multi_face_count))
            cv2.imwrite(multi_face_dir + file_name_only, img)

        else:
            count += 1
            if count % 50 == 0:
                print('So far {} single face detected out of {}, elapsed time = {}'.format(
                    count, total_file_len, time.time()-start_t))

            for d in detected:
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - x_left_margin * w), 0)
                yw1 = max(int(y1 - y_top_margin * h), 0)
                xw2 = min(int(x2 + x_right_margin * w), img_w - 1)
                yw2 = min(int(y2 + y_bottom_margin * h), img_h - 1)

                roi_color = img[yw1: yw2 + 1, xw1: xw2 + 1, :]

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
                cv2.ellipse(new_mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0,
                            startAngle=0.0, endAngle=360, color=(0, 0, 0), thickness=-1)

                # final result
                with_mask = result + new_mask

                # save the cropped images in a folder.
                cv2.imwrite(crop_with_mask_dir + file_name_only, with_mask)

    print('Current folder is {}\n Total face # = {}, detected faces = {}, no face # = {}, multi face # = {}, cannot read # = {}'.
          format(raw_data_dir, total_file_len, count, no_face_count, multi_face_count, cannot_read_face_count))
    print('Time elapsed for face cropping in this folder = {:.2f} seconds.'.format(time.time()-start_t))


def train_on_full_2k():
    start_t = time.time()
    pca_num = 180
    vgg_batch_size = 100

    if not os.path.exists(feat_save_dir):
        os.makedirs(feat_save_dir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # extract 2k feature.
    img_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces/'
    file_names = [f for f in os.listdir(img_dir) if f[-3:] == 'jpg' or f[-3:] == 'png']
    feat_prefix = '2k_face'
    feat_arr, valid_img_lst = extract_features(img_dir, file_names, feat_prefix, vgg_batch_size)

    # load selected impression attribute scores
    select_rating_path = '../pretrain_first_impression/tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    rating_df = rating_df.set_index('Filename')
    rating_df = rating_df.loc[valid_img_lst]
    rating_arr = rating_df.as_matrix()

    additional_feat = pd.read_pickle('../static/gender_age_2k_gt.pkl')
    add_feat_df = additional_feat.fillna(additional_feat['Age'].mean())
    add_feat_df = add_feat_df.set_index('Filename')
    add_feat_df = add_feat_df.loc[valid_img_lst]
    add_feat_arr = add_feat_df.as_matrix()  # age, then gender.

    # apply normalization here.
    min_max_scaler = preprocessing.MinMaxScaler()
    add_feat_scale = min_max_scaler.fit_transform(add_feat_arr)

    # do PCA on the whole data set.
    print('Start pca with full data ...')
    time_1 = time.time()
    pca = PCA(n_components=pca_num)
    pca.fit(feat_arr)
    print('PCA done. Takes {}'.format(time.time() - time_1))

    # save the pca model
    feat_layer_name = 'conv52_2k'
    pca_save_name = '{}/{}_all_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)
    pickle.dump(pca, open(pca_save_name, 'wb'))

    x_pca = pca.transform(feat_arr)
    x_all = np.hstack((x_pca, add_feat_scale))

    multi_regressor = MultiOutputRegressor(LinearRegression())
    multi_regressor.fit(x_all, rating_arr)

    # save the regressor model
    regressor_save_name = '{}/{}_all_data_regressor_model.pkl'.format(model_save_dir, feat_layer_name)
    # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
    pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

    y_predict = multi_regressor.predict(x_all)

    for i, cur_feat in enumerate(feature_lst):
        [cor, _] = pearsonr(y_predict[:, i], rating_arr[:, i])
        print('Feature = {}, cor = {}'.format(cur_feat, cor))

    print('Total time = {}'.format(time.time() - start_t))


def predict_gen_and_age(current_folder_post_fix, crop_with_mask_root_dir, save_result_root_dir):
    # face detection
    start_t = time.time()
    print('Start estimating age and gender...')
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
        if (i+1) % 500 == 0:
            print('{} images out of {} images have been processed.'.format(i+1, len(file_lst)))
        cur_im_path = os.path.join(crop_with_mask_dir, cur_im_name)
        img = cv2.imread(cur_im_path)
        img_h, img_w, _ = np.shape(img)
        img_height_lst.append(img_h)
        img_width_lst.append(img_w)
        faces[i, :, :, :] = cv2.resize(img[:, :, :], (img_size, img_size))

    print('Start to make inferences...')
    results = model.predict(faces)
    predicted_genders = results[0]
    predicted_gender_prob = predicted_genders[:, 1]  # the probability of being a male, (0, 1).
    predicted_gender_binary = [round(i) for i in predicted_gender_prob]

    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    result_df = pd.DataFrame(columns=['filename', 'age', 'gender binary', 'gender probability',
                                      'img height', 'img width'])
    result_df['gender probability'] = predicted_gender_prob
    result_df['gender binary'] = predicted_gender_binary
    result_df['age'] = predicted_ages
    result_df['filename'] = file_lst
    result_df['img height'] = img_height_lst
    result_df['img width'] = img_width_lst

    file_name = 'gender_age_only_ind_' + current_folder_post_fix[:-1] + '.pkl'
    file_full_path = save_result_dir + file_name
    result_df.to_pickle(file_full_path)

    print('Age and gender estimation Done. Total time = {}'.format(time.time()-start_t))
    K.clear_session()


def predict_impression(current_folder_post_fix, crop_with_mask_root_dir, save_result_root_dir, feat_prefix, csv_save_dir):
    start_t = time.time()
    print('Start predicting impressions...')
    vgg_batch_size = 100

    # extract all the images' features in one array in a specified folder.
    crop_with_mask_dir = crop_with_mask_root_dir + current_folder_post_fix
    file_names = [f for f in os.listdir(crop_with_mask_dir) if f[-3:] == 'jpg' or f[-3:] == 'png']

    save_result_dir = save_result_root_dir + current_folder_post_fix

    feat_arr, valid_img_lst = extract_features(img_dir=crop_with_mask_dir, file_names=file_names,
                                               feat_prefix=feat_prefix+current_folder_post_fix[:-1],
                                               batch_size=vgg_batch_size)

    add_feat_path = save_result_dir + 'gender_age_only_ind_' + current_folder_post_fix[:-1] + '.pkl'
    add_feat_df = pd.read_pickle(add_feat_path)

    add_feat_df = add_feat_df.set_index('filename')
    add_feat_df = add_feat_df.loc[valid_img_lst]
    add_feat_arr = add_feat_df.as_matrix()  # age, then gender.

    # apply the same normalization here.
    min_max_scaler = preprocessing.MinMaxScaler()
    add_feat_scale = min_max_scaler.fit_transform(add_feat_arr[:, :2])

    # do PCA on the whole data set.
    print('\n Start loading PCA model ...')

    feat_layer_name = 'conv52_2k'
    pca_save_name = '{}/{}_all_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)

    with open(pca_save_name, 'rb') as f:
        pca = pickle.load(f)

    x_pca = pca.transform(feat_arr)
    x_all = np.hstack((x_pca, add_feat_scale))

    print('\n Start running regression...')

    regressor_model_name = '{}/{}_all_data_regressor_model.pkl'.format(model_save_dir, feat_layer_name)
    with open(regressor_model_name, 'rb') as f:
        regressor = pickle.load(f)

    y_predict = regressor.predict(x_all)

    predict_df = pd.DataFrame(data=y_predict, columns=feature_lst)

    predict_df['filename'] = valid_img_lst
    predict_df['age'] = add_feat_arr[:, 0]
    predict_df['gender'] = add_feat_arr[:, 1]
    predict_df['male probability'] = add_feat_arr[:, 2]
    predict_df['img height'] = add_feat_arr[:, 3]
    predict_df['img width'] = add_feat_arr[:, 4]

    predict_df = predict_df[['filename', 'img height', 'img width', 'age', 'gender', 'male probability',
                             'trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']]

    save_file_name = 'impression_gen_age_' + current_folder_post_fix[:-1] + '.pkl'
    file_save_full_path = save_result_dir + save_file_name
    predict_df.to_pickle(file_save_full_path)

    save_csv_name = 'impression_gen_age_' + current_folder_post_fix[:-1] + '.csv'
    predict_df.to_csv(csv_save_dir+save_csv_name)

    print('Total time = {}'.format(time.time()-start_t))


def main():
    start_t = time.time()

# crunch base is all done, except the special case 2. its face crop process is not done yet. need to handle.

    feat_prefix = 'crunchbase'
    raw_data_root_dir = '/home/amanda/Documents/VC and Entrepreneur Faces Project/New_data/'+feat_prefix+'/'

    crop_with_mask_root_dir = '/home/amanda/Documents/cropped_new_'+feat_prefix+'/'
    save_result_root_dir = '/home/amanda/Documents/predicted_results/'+feat_prefix+'_new_data/'
    csv_save_dir = save_result_root_dir + 'to_will/'

    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir)

        special_case = ['2/']  #
        '''
        '2/'
        So far 1896 single face detected out of 3606, elapsed time = 579.934873104
    Traceback (most recent call last):
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 567, in <module>
        main()
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 549, in main
        crop_with_mask_root_dir=crop_with_mask_root_dir)
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 279, in crop_faces
        detected = detector(img, 1)
    RuntimeError: Unsupported image type, must be 8bit gray or RGB image.


        '4/'
        Start estimating age and gender...
    Traceback (most recent call last):
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 580, in <module>
        main()
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 566, in main
        save_result_root_dir=save_result_root_dir)
      File "/home/amanda/Github/impression_personality/age_gender_impression_pipeline/pipeline_ready_to_go.py", line 421, in predict_gen_and_age
        model.load_weights(weight_file)
      File "/usr/local/lib/python2.7/dist-packages/keras/engine/network.py", line 1161, in load_weights
        f, self.layers, reshape=reshape)
      File "/usr/local/lib/python2.7/dist-packages/keras/engine/saving.py", line 928, in load_weights_from_hdf5_group
        K.batch_set_value(weight_value_tuples)
      File "/usr/local/lib/python2.7/dist-packages/keras/backend/tensorflow_backend.py", line 2440, in batch_set_value
        get_session().run(assign_ops, feed_dict=feed_dict)
      File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 900, in run
        run_metadata_ptr)
      File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1078, in _run
        'Cannot interpret feed_dict key as Tensor: ' + e.args[0])
    TypeError: Cannot interpret feed_dict key as Tensor: Tensor Tensor("Placeholder_68:0", shape=(131072, 2), dtype=float32) is not an element of this graph.
        When a new number of image appears, it should restart the graph. Find out how to do it. 
        '''

    # to_do_lst = ['0/', '1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/', '10/',
    #              '20/', '30/', '40/', '50/', '60/', '70/', '80/', '90/', '100/',
    #              '200/', '300/', '400/', '500/', '600/', '700/', '800/', '900/',
    #              'a1/', 'a10/', 'a100/',
    #              'b1/', 'b10/', 'b100/',
    #              'c1/', 'c10/', 'c100/',
    #              'd1/', 'd10/', 'd100/',
    #              'e1/', 'e10/', 'e100/',
    #              'f1/', 'f10/', 'f100/',
    #              # 'g0/', 'g1/', 'g2/', 'g3/', 'g4/', 'g5/', 'g6/', 'g7/', 'g8/', 'g9/', 'g10/',
    #              # 'g20/', 'g30/', 'g40/', 'g50/', 'g60/', 'g70/', 'g80/', 'g90/', 'g100/',
    #              # 'g200/', 'g300/', 'g400/', 'g500/', 'g600/', 'g700/', 'g800/', 'g900/',
    #              # 'ga1/', 'ga10/', 'ga100/',
    #              # 'gb1/', 'gb10/', 'gb100/',
    #              # 'gc1/', 'gc10/', 'gc100/',
    #              # 'gd1/', 'gd10/', 'gd100/',
    #              # 'ge1/', 'ge10/', 'ge100/',
    #              # 'gf1/', 'gf10/', 'gf100/'
    #              ]
    # 2
    # ccc01c7 - dbcf - 4
    # fa9 - bfdc - b4c613290dcb_CB.jpg
    to_do_lst = ['2/']

    for cur_folder_post_fix in to_do_lst:

        crop_faces(raw_data_dir_postfix=cur_folder_post_fix,
                   raw_data_root_dir=raw_data_root_dir,
                   crop_with_mask_root_dir=crop_with_mask_root_dir)

        # predict_gen_and_age(current_folder_post_fix=cur_folder_post_fix,
        #                     crop_with_mask_root_dir=crop_with_mask_root_dir,
        #                     save_result_root_dir=save_result_root_dir)
        #
        # predict_impression(current_folder_post_fix=cur_folder_post_fix,
        #                    crop_with_mask_root_dir=crop_with_mask_root_dir,
        #                    save_result_root_dir=save_result_root_dir,
        #                    feat_prefix=feat_prefix,
        #                    csv_save_dir=csv_save_dir)

        print('\nTotal time elapsed after directory {} is {:.2f} seconds\n\n'.format(cur_folder_post_fix, time.time()-start_t))

    return


if __name__ == '__main__':
    main()
