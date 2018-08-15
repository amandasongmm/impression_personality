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
from sklearn.model_selection import train_test_split
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
    fname = os.path.join(feat_save_dir, feat_prefix + '_conv52.npy')
    if os.path.isfile(fname):
        conv52_arr = np.load(fname)
        valid_file_lst = np.load(os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy'))
    else:
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


def train_test_split_2k():
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

    x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=0.1, random_state=42)
    x_added_train, x_added_test, y_train, y_test = train_test_split(add_feat_scale, rating_arr, test_size=.1,
                                                                    random_state=42)

    # do PCA on the whole data set.
    print('Start pca with full data ...')
    time_1 = time.time()
    pca = PCA(n_components=pca_num)
    pca.fit(x_train)
    print('PCA done. Takes {}'.format(time.time() - time_1))

    # # save the pca model
    # feat_layer_name = 'conv52_2k'
    # pca_save_name = '{}/{}_all_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)
    # pickle.dump(pca, open(pca_save_name, 'wb'))

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    x_train_full = np.hstack((x_train_pca, x_added_train))
    x_test_full = np.hstack((x_test_pca, x_added_test))

    multi_regressor = MultiOutputRegressor(LinearRegression())
    multi_regressor.fit(x_train_full, y_train)


    # # save the regressor model
    # regressor_save_name = '{}/{}_all_data_regressor_model.pkl'.format(model_save_dir, feat_layer_name)
    # # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
    # pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

    y_train_predict = multi_regressor.predict(x_train_full)
    y_test_predict = multi_regressor.predict(x_test_full)

    for i, cur_feat in enumerate(feature_lst):
        [cor, _] = pearsonr(y_train_predict[:, i], y_train[:, i])
        [cor2, _] = pearsonr(y_test_predict[:, i], y_test[:, i])
        print('Feature = {}, cor = {}, cor test = {}'.format(cur_feat, cor, cor2))

    print('Total time = {}'.format(time.time() - start_t))


train_test_split_2k()