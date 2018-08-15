from __future__ import division
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

# In this experiment, keep max_pool1 = conv1_1.
# And change max_pool2 from = conv2_2 to conv2_1.

default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
batch_size = 100

feat_save_dir = '/home/amanda/Documents/feat_folder/'
model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'

if not os.path.exists(feat_save_dir):
    os.makedirs(feat_save_dir)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)


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

        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_1, (2, 2), (2, 2), padding=self.pool_padding)

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
    num_batches = int(np.ceil(num_files / batch_size))

    conv52_arr = np.empty((0, 100352), dtype=np.float32)

    valid_file_lst = []

    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:
            _input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='images')
            vgg = Vgg16Model()
            vgg.build(_input)

            sess.run(tf.global_variables_initializer())

            for i in range(num_batches):

                if i == num_batches - 1:
                    batch_filenames = file_names[i * batch_size:]
                else:
                    batch_filenames = file_names[i * batch_size: (i + 1) * batch_size]

                print('batch {} out of {}. '.format(i+1, num_batches))
                start_t = time.time()
                images = np.array([load_image(os.path.join(img_dir, f)) for f in batch_filenames])

                batch_conv52 = sess.run(vgg.reshaped_conv52, feed_dict={_input: images})

                conv52_arr = np.append(conv52_arr, np.array(batch_conv52), axis=0)

                valid_file_lst += batch_filenames
                end_t = time.time()
                print('\t this batch takes {:.2f} sec'.format(end_t-start_t))

            print('valid file num = {}'.format(len(valid_file_lst)))

            np.save(os.path.join(feat_save_dir, feat_prefix + '_conv52.npy'), conv52_arr)
            np.save(os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy'), valid_file_lst)

    print('Total time elapsed: {:.2f} sec.'.format(time.time()-start_zero))


def extract_2k_feat():
    img_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces/'
    file_names = [f for f in os.listdir(img_dir) if f[-3:] == 'jpg']
    feat_prefix = '2k_face_exp1_'
    extract_features(img_dir, file_names, feat_save_dir, feat_prefix, batch_size)
    return


def train_2k():
    # load valid img lst.
    start_t = time.time()
    pca_num = 200
    feat_prefix = '2k_face'
    valid_img_file_name = os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy')
    valid_img_lst = np.load(valid_img_file_name)

    # load extracted features.
    feature_file_name = os.path.join(feat_save_dir, feat_prefix + '_conv52.npy')
    feat_arr = np.load(feature_file_name)  # 2222 * 100352, for example.

    # load selected impression attribute scores
    select_rating_path = '../pretrain_first_impression/tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_df = rating_df.set_index('Filename')
    rating_df = rating_df.loc[valid_img_lst]
    rating_arr = rating_df.as_matrix()

    additional_feat = pd.read_pickle('../static/gender_age_2k_gt.pkl')
    add_feat_df = additional_feat.fillna(additional_feat['Age'].mean())
    add_feat_df = add_feat_df.set_index('Filename')
    add_feat_df = add_feat_df.loc[valid_img_lst]
    add_feat_arr = add_feat_df.as_matrix()

    train_flag = 1
    if train_flag == 1:
        # split on a 64/16/20 ratio. split on a 90/10 ratio.
        x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=0.1, random_state=42)

        # do PCA on the training set.
        print('Split done. Start pca...')
        pca = PCA(n_components=pca_num)
        pca.fit(x_train)
        print('PCA done. Takes {}'.format(time.time()-start_t))

        cum_ratio = np.cumsum(pca.explained_variance_ratio_)
        print(cum_ratio[-1])

        # save the pca model
        feat_layer_name = 'conv52_2k'
        # pca_save_name = '{}/{}_90_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)
        # pickle.dump(pca, open(pca_save_name, 'wb'))

        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        # Linear regression
        multi_regressor = MultiOutputRegressor(LinearRegression())
        multi_regressor.fit(x_train_pca, y_train)

        # save the regressor model
        # regressor_save_name = '{}/{}_keep_{}_regressor_model.pkl'.format(model_save_dir, feat_layer_name, str(pca_num))
        # # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
        # pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

        y_train_predict = multi_regressor.predict(x_train_pca)
        y_test_predict = multi_regressor.predict(x_test_pca)

        for i, cur_feat in enumerate(feature_lst):
            [train_cor, p] = pearsonr(y_train[:, i], y_train_predict[:, i])
            [test_cor, p] = pearsonr(y_test[:, i], y_test_predict[:, i])
            print('Feature = {}, Train, test cor = {}, {}'.format(cur_feat, train_cor, test_cor))

    else:
        # do PCA on the whole data set.
        print('Split done. Start pca...')
        pca = PCA(n_components=pca_num)
        pca.fit(feat_arr)
        print('PCA done. Takes {}'.format(time.time() - start_t))

        # save the pca model
        feat_layer_name = 'conv52_2k'
        pca_save_name = '{}/{}_all_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)
        pickle.dump(pca, open(pca_save_name, 'wb'))

        x_pca = pca.transform(feat_arr)
        x_all = np.hstack((x_pca, add_feat_arr))
        x_with_gender = np.hstack((x_pca, add_feat_arr[:, 1:]))

        print('Without age and gender info.')
        # Linear regression
        multi_regressor = MultiOutputRegressor(LinearRegression())
        multi_regressor.fit(x_pca, rating_arr)
        y_predict = multi_regressor.predict(x_pca)

        for i, cur_feat in enumerate(feature_lst):
            [cor, p] = pearsonr(y_predict[:, i], rating_arr[:, i])
            print('Feature = {}, cor = {}'.format(cur_feat, cor))

        print('With age and gender info.')
        # Linear regression
        multi_regressor = MultiOutputRegressor(LinearRegression())
        multi_regressor.fit(x_all, rating_arr)
        y_predict = multi_regressor.predict(x_all)

        for i, cur_feat in enumerate(feature_lst):
            [cor, p] = pearsonr(y_predict[:, i], rating_arr[:, i])
            print('Feature = {}, cor = {}'.format(cur_feat, cor))

        print('With gender info only.')
        # Linear regression
        multi_regressor = MultiOutputRegressor(LinearRegression())
        multi_regressor.fit(x_with_gender, rating_arr)
        y_predict = multi_regressor.predict(x_with_gender)

        for i, cur_feat in enumerate(feature_lst):
            [cor, p] = pearsonr(y_predict[:, i], rating_arr[:, i])
            print('Feature = {}, cor = {}'.format(cur_feat, cor))

    print('Total time = {}'.format(time.time()-start_t))
    return


if __name__ == '__main__':
    extract_2k_feat()
    train_2k()