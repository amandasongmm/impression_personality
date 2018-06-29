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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from shutil import copyfile
from scipy.stats import ttest_ind

import matplotlib
import matplotlib.pyplot as plt

default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
batch_size = 100
feat_save_dir = '/home/amanda/Documents/feat_folder/'
model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'


# the maxpool1 is using conv1_1, instead of conv1_2. somehow it gives better performance.
class Vgg16Model:

    def __init__(self):
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

        self.reshaped_conv51 = tf.reshape(self.conv5_1, shape=(-1, 14 * 14 * 512))
        self.reshaped_conv52 = tf.reshape(self.conv5_2, shape=(-1, 14 * 14 * 512))
        self.reshaped_conv53 = tf.reshape(self.conv5_3, shape=(-1, 14 * 14 * 512))

        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)
        self.reshaped = tf.reshape(self.max_pool5, shape=(-1, 7*7*512))

        self.fc6 = self.fc(self.reshaped, 'fc6', 4096, trainable)
        self.fc7 = self.fc(self.fc6, 'fc7', 4096, trainable)

        self.fc8 = self.fc(self.fc7, 'fc8', 1000, trainable)
        self.predictions = tf.nn.softmax(self.fc8, name='predictions')

    def conv2d(self, layer, name, n_filters, trainable, k_size=3):
        weights = self.weights[name+'_W']
        bias = self.weights[name+'_b']
        return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                activation=self.activation_fn,
                                padding=self.conv_padding,
                                name=name,
                                trainable=trainable,
                                kernel_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),
                                use_bias=self.use_bias)

    def fc(self, layer, name, size, trainable):
        weights = self.weights[name+'_W']
        bias = self.weights[name+'_b']
        return tf.layers.dense(layer, size, activation=self.activation_fn,
                               name=name, trainable=trainable,
                               kernel_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                               bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),
                               use_bias=self.use_bias)


# common function
def load_image(image_path, mean=vgg_mean):
    image = skimage.io.imread(image_path)
    resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
    bgr = resized_image[:, :, ::-1] - mean
    return bgr


# common function
def extract_features(img_dir, file_names, feat_save_dir, feat_prefix, batch_size):
    start_zero = time.time()
    tf.reset_default_graph()

    # create mappings of filenames --> vgg features
    num_files = len(file_names)
    num_batches = int(math.ceil(num_files / batch_size))

    conv51_arr = np.empty((0, 100352), dtype=np.float32)
    conv52_arr = np.empty((0, 100352), dtype=np.float32)
    conv53_arr = np.empty((0, 100352), dtype=np.float32)

    valid_file_lst = []

    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:
            _input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='images')
            vgg = Vgg16Model()
            vgg.build(_input)

            sess.run(tf.global_variables_initializer())

            for i in range(num_batches):
                batch_filenames = file_names[i * batch_size: (i + 1) * batch_size]

                print('batch {} out of {}. '.format(i+1, num_batches))
                start_t = time.time()
                images = np.array([load_image(os.path.join(img_dir, f)) for f in batch_filenames])

                batch_conv51, batch_conv52, batch_conv53 = \
                    sess.run([vgg.reshaped_conv51, vgg.reshaped_conv52, vgg.reshaped_conv53], feed_dict={_input: images})

                conv51_arr = np.append(conv51_arr, np.array(batch_conv51), axis=0)
                conv52_arr = np.append(conv52_arr, np.array(batch_conv52), axis=0)
                conv53_arr = np.append(conv53_arr, np.array(batch_conv53), axis=0)

                valid_file_lst += batch_filenames
                end_t = time.time()
                print('\t this batch takes {:.2f} sec'.format(end_t-start_t))

            print('valid file num = {}'.format(len(valid_file_lst)))

            np.save(os.path.join(feat_save_dir, feat_prefix + '_conv51.npy'), conv51_arr)
            np.save(os.path.join(feat_save_dir, feat_prefix + '_conv52.npy'), conv52_arr)
            np.save(os.path.join(feat_save_dir, feat_prefix + '_conv53.npy'), conv53_arr)

            np.save(os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy'), valid_file_lst)

    print('Total time elapsed: {:.2f} sec.'.format(time.time()-start_zero))


# specific function
def extract_vc_feat():
    img_dir = '/home/amanda/Documents/cropped_faces/vc_with_mask_narrow/'
    file_names = [f for f in os.listdir(img_dir) if f[-4:] == 'jpeg']
    id_lst = [f[:-8] for f in file_names]
    unique_id_lst = set(id_lst)

    im_lst = [im_id + '_cb.jpeg' if im_id + '_cb.jpeg' in file_names else im_id + '_tw.jpeg' for im_id in unique_id_lst]
    print('VC img length = {}'.format(len(im_lst)))
    extract_features(img_dir=img_dir, file_names=im_lst[:9700], feat_save_dir=feat_save_dir, feat_prefix='vc_with_bug',
                     batch_size=batch_size)

    return


def make_new_prediction():
    start_t = time.time()

    # load valid img lst.
    valid_img_file_name = '2kface_valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(feat_save_dir, valid_img_file_name))

    # load extracted features
    layer_name = 'conv52'

    pca_num = 200
    rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    pca_save_name = '{}/{}_predict_mode_pca_{}_model.pkl'.format(feat_save_dir, layer_name, str(pca_num))
    regressor_save_name = '{}/2k_final_model_{}_keep_{}_regressor.pkl'.format(feat_save_dir, layer_name, str(pca_num))

    if os.path.isfile(pca_save_name) and os.path.isfile(regressor_save_name):
        print('Load saved models')
        with open(pca_save_name, 'rb') as f:
            pca = pickle.load(f)

        with open(regressor_save_name, 'rb') as f:
            multi_regressor = pickle.load(f)

    else:
        feature_file_name = os.path.join(feat_save_dir, '2kface_with_bug__' + layer_name + '.npy')
        feat_arr = np.load(feature_file_name)  # 2200 * 100352, for example.

        # load selected impression traits.
        select_rating_path = 'tmp_data/selected_score.pkl'
        rating_df = pd.read_pickle(select_rating_path)

        filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
        filtered_rating_df = filtered_rating_df.set_index('Filename')
        filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

        rating_arr = filtered_rating_df.as_matrix()

        # apply PCA on training data.
        pca = PCA(n_components=pca_num)

        pca.fit(feat_arr)
        print('PCA takes {}'.format(time.time() - start_t))

        cum_ratio = np.cumsum(pca.explained_variance_ratio_)[-1]
        print('number of pc: {},  explained variance = {}\n'.format(pca_num, cum_ratio))  # 0.98

        # save the pca model
        print('Saving pca model...\n')
        pickle.dump(pca, open(pca_save_name, 'wb'))

        # apply pca on the whole 2K dataset.
        two_k_pca = pca.transform(feat_arr)

        # now train the regression model.
        multi_regressor = MultiOutputRegressor(LinearRegression())
        multi_regressor.fit(two_k_pca, rating_arr)

        # save the regressor model
        pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

        # predict on training data.
        y_train_predict = multi_regressor.predict(two_k_pca)

        txt_file_name = '{}/2k_{}_use_all_data_performance.txt'.format(feat_save_dir, layer_name)
        with open(txt_file_name, 'w') as txt_file:
            for ind, cur_name in enumerate(rating_names):
                line = 'Cur impression: {}\n'.format(cur_name)
                print(line)
                txt_file.write(line)

                train_gt = rating_arr[:, ind]
                train_pred = y_train_predict[:, ind]
                [train_cor, p] = pearsonr(train_gt, train_pred)

                line = 'train cor {}\n\n'.format(train_cor)
                print(line)
                txt_file.write(line)

    # Load VC feature matrix.
    vc_feat = np.load('{}/vc_with_bug_conv52.npy'.format(feat_save_dir))
    vc_pca = pca.transform(vc_feat)

    # Now make predictions on the VC dataset.
    vc_prediction = multi_regressor.predict(vc_pca)
    np.save(feat_save_dir+'vc_prediction.txt', vc_prediction)

    feat_prefix = 'vc_with_bug'
    valid_img_lst = np.load(feat_save_dir + feat_prefix + '_valid_img_lst.npy')

    prediction_df = pd.DataFrame(columns=[rating_names], index=valid_img_lst, data=vc_prediction)
    prediction_df.to_pickle(feat_save_dir+'vc_prediction_df.pkl')
    return


def merge_with_funding_and_gender_data():
    # merge impression with VC ranking data.
    # vc_impression_data = pd.read_pickle(feat_save_dir+'vc_prediction_df.pkl')
    # vc_impression_data.to_csv('tmp_data/vc_raw_prediction.csv')

    vc_impression_data = pd.read_csv('tmp_data/vc_raw_prediction.csv')
    vc_impression_data['vc_id'] = pd.Series([s[:-8] for s in vc_impression_data['Unnamed: 0'].values],
                                            index=vc_impression_data.index)

    vc_funding_data = pd.read_pickle('tmp_data/vc_df.pkl')
    short_vc_fund_df = vc_funding_data[vc_funding_data['short_id'].isin(vc_impression_data['vc_id'])]
    short_vc_fund_df = short_vc_fund_df.set_index('short_id')
    short_vc_fund_df = short_vc_fund_df.reindex(index=vc_impression_data['vc_id'])
    short_vc_fund_df = short_vc_fund_df.reset_index()

    vc_impression_data['VC - Investment # rank'] = short_vc_fund_df['VC - Investment # rank']
    vc_impression_data['VC - Investment $ rank'] = short_vc_fund_df['VC - Investment $ rank']
    vc_impression_data['uuid'] = short_vc_fund_df['uuid']

    # merge impression with gender and age data.
    api_data = pd.read_csv('/home/amanda/Github/impression_personality/vc-labels.txt', sep="\t", header=None)
    api_data.columns = ["faceId", "faceTopDimension", "faceLeftDimension", "faceWidthDimension", "faceHeightDimension",
                        "smile", "pitch", "roll", "yaw", "gender", "age", "moustache", "beard", "sideburns", "glasses",
                        "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
                        "blurlevel",
                        "blurvalue", "exposurelevel", "exposurevalue", "noiselevel", "noisevalue", "eymakeup",
                        "lipmakeup",
                        "foreheadoccluded", "eyeoccluded", "mouthoccluded", "hair-bald", "hair-invisible", "img_name"]
    api_data = api_data.drop(api_data.index[0])
    api_data = api_data[['img_name', 'gender', 'age']]

    api_data = api_data.drop_duplicates()  # there are some repetitive images.

    print('api-data len={}'.format(len(api_data)))
    print('vc-impresion len={}'.format(len(vc_impression_data)))

    vc_imp_only = vc_impression_data['Unnamed: 0'].values
    api_only = api_data['img_name'].values
    common_lst = set(vc_imp_only) & set(api_only)  # 8892

    print(len(common_lst))

    vc_impression_data = vc_impression_data[vc_impression_data['Unnamed: 0'].isin(common_lst)]
    vc_impression_data.reset_index()

    api_data = api_data[api_data['img_name'].isin(common_lst)]

    api_data = api_data.set_index('img_name')
    api_data = api_data.reindex(index=vc_impression_data['Unnamed: 0'])
    api_data = api_data.reset_index()

    vc_impression_data['gender'] = api_data['gender']
    vc_impression_data['age'] = api_data['age']
    vc_impression_data.to_pickle('tmp_data/vc_merged_prediction.pkl')
    vc_impression_data.to_csv(feat_save_dir+'vc_merged_prediction.csv', index=False)

    return


# extract_vc_feat()
# make_new_prediction()
merge_with_funding_and_gender_data()
