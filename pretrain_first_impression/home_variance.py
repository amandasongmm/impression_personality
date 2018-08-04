import pandas as pd
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
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
import numpy as np
import skimage
import skimage.io
import skimage.transform
import math
import time
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
model_save_dir = '/home/amanda/Documents/vgg_model/Aug_variance/'


def gen_raw_data_pkl():
    df = pd.read_excel('../static/psychology-attributes.xlsx', sheet_name='StDev Values')
    keep_names = ['Filename', 'atypical', 'boring', 'calm', 'cold', 'common', 'confident',
                  'egotistic', 'emotUnstable', 'forgettable', 'intelligent', 'introverted',
                  'kind', 'responsible', 'trustworthy', 'unattractive', 'unemotional',
                  'unfamiliar', 'unfriendly', 'unhappy', 'weird', 'aggressive', 'attractive',
                  'caring', 'emotStable', 'emotional', 'familiar', 'friendly', 'happy',
                  'humble', 'interesting', 'irresponsible', 'mean', 'memorable', 'normal',
                  'sociable', 'typical', 'uncertain', 'uncommon', 'unintelligent', 'untrustworthy']
    df_select = df[keep_names]
    df_select.to_pickle("../static/variance_raw_2k.pkl")

    df = pd.read_excel('../static/psychology-attributes.xlsx', sheet_name='Final Values')
    keep_names = ['Filename', 'atypical', 'boring', 'calm', 'cold', 'common', 'confident',
                  'egotistic', 'emotUnstable', 'forgettable', 'intelligent', 'introverted',
                  'kind', 'responsible', 'trustworthy', 'unattractive', 'unemotional',
                  'unfamiliar', 'unfriendly', 'unhappy', 'weird', 'aggressive', 'attractive',
                  'caring', 'emotStable', 'emotional', 'familiar', 'friendly', 'happy',
                  'humble', 'interesting', 'irresponsible', 'mean', 'memorable', 'normal',
                  'sociable', 'typical', 'uncertain', 'uncommon', 'unintelligent', 'untrustworthy']
    df_select = df[keep_names]
    df_select.to_pickle("../static/final_val_raw_2k.pkl")


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


def main():
    start_t = time.time()
    var_data = pd.read_pickle("../static/variance_raw_2k.pkl")
    final_val_data = pd.read_pickle("../static/final_val_raw_2k.pkl")
    img_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces'
    file_names = list(var_data['Filename'].values)
    feat_save_dir = '/home/amanda/Documents/vgg_model/Aug_variance/'
    feat_prefix = '2k22_'
    batch_size = 101
    if not os.path.exists(feat_save_dir):
        os.makedirs(feat_save_dir)

    if not os.path.exists(os.path.join(feat_save_dir, feat_prefix + '_conv51.npy')):
        extract_features(img_dir, file_names, feat_save_dir, feat_prefix, batch_size)


    var_check_feat_lst = ['atypical', 'boring', 'calm', 'cold', 'common', 'confident',
                  'egotistic', 'emotUnstable', 'forgettable', 'intelligent', 'introverted',
                  'kind', 'responsible', 'trustworthy', 'unattractive', 'unemotional',
                  'unfamiliar', 'unfriendly', 'unhappy', 'weird', 'aggressive', 'attractive',
                  'caring', 'emotStable', 'emotional', 'familiar', 'friendly', 'happy',
                  'humble', 'interesting', 'irresponsible', 'mean', 'memorable', 'normal',
                  'sociable', 'typical', 'uncertain', 'uncommon', 'unintelligent', 'untrustworthy']

    two_k_conv51 = np.load(os.path.join(feat_save_dir, feat_prefix + '_conv51.npy'))
    two_k_final = final_val_data[var_check_feat_lst].values

    # var_check_feat_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    two_k_y = var_data[var_check_feat_lst].values

    train_val_test_split_data_save_name = '/home/amanda/Documents/vgg_model/Aug_variance/final_val_train_test_pca_data.npz'
    if not os.path.exists(train_val_test_split_data_save_name):
        # x_train, x_test, y_train, y_test = train_test_split(two_k_conv51, two_k_y, test_size=0.1, random_state=42)
        x_final_val_train, x_final_va_test, y_train, y_test = train_test_split(two_k_final, two_k_y, test_size=0.1,
                                                                               random_state=42)
        print('Split done. Start pca...')
        pca = PCA(n_components=40)
        pca.fit(x_final_val_train)

        cum_ratio = np.cumsum(pca.explained_variance_ratio_)

        x_train_pca = pca.transform(x_final_val_train)
        x_test_pca = pca.transform(x_final_va_test)

        # save the split feature data.

        np.savez(train_val_test_split_data_save_name, x_train_pca=x_train_pca, x_test_pca=x_test_pca,
                 y_train=y_train, y_test=y_test, cum_ratio=cum_ratio, feature_name=var_check_feat_lst)

        print('pca done. Split processed data saved...\n Elapsed time = {}'.format(time.time() - start_t))

    print('Load feature after PCA.')
    data = np.load(train_val_test_split_data_save_name)
    x_train_pca, x_test_pca = data['x_train_pca'], data['x_test_pca']
    y_train, y_test = data['y_train'], data['y_test']
    cum_ratio, feature_lst = data['cum_ratio'], data['feature_name']

    # pca_nums = np.linspace(10, 300, num=31, dtype=int)
    pca_nums = np.linspace(1, 40, num=40, dtype=int)

    # Test one dimension at a time.
    feat_num = len(feature_lst)
    for feat_dim in range(0, feat_num):

        cur_feat = feature_lst[feat_dim]
        print('\n\nFeature = {}'.format(cur_feat))
        df = pd.DataFrame(columns=['pca number', 'train correlation', 'test correlation'])
        high_test = 0
        for ind, pca_num in enumerate(pca_nums):
            # for a in alphas:
            linear_regressor = LinearRegression()
            x_train, x_test = x_train_pca[:, :pca_num], x_test_pca[:, :pca_num]

            linear_regressor.fit(x_train, y_train[:, feat_dim])

            y_train_predict = linear_regressor.predict(x_train)
            y_test_predict = linear_regressor.predict(x_test)

            # print('\npca number = {}'.format(pca_num))
            [train_cor, p] = pearsonr(y_train[:, feat_dim], y_train_predict)
            [test_cor, p] = pearsonr(y_test[:, feat_dim], y_test_predict)
            if test_cor > high_test:
                high_test = test_cor
                print(test_cor, pca_num)
            # print('Train, test cor = {:.3f}, {:.3f}'.format(train_cor, test_cor))
            df.loc[ind] = [pca_num, train_cor, test_cor]

        df.to_pickle(model_save_dir + cur_feat + 'final_performance.pkl')

        # visualize the performance change curves.
        fig = plt.figure()
        fig.suptitle(cur_feat, fontsize=20)
        plt.scatter(df['pca number'], df['train correlation'], label='train')
        plt.scatter(df['pca number'], df['test correlation'], label='test')
        plt.xlabel('PCA number', fontsize=18)
        plt.ylabel('Correlation', fontsize=16)
        plt.grid()
        plt.legend()
        fig.savefig(model_save_dir + cur_feat + '_final.jpg')
        plt.close()


if __name__ == '__main__':
    main()