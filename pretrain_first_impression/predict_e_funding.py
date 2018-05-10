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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


default_device = '/gpu:0'
num_hidden_neurons = 256
my_batch_size = 130  # total image num = 7800. so it can be done in 60 batches.

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'


class Vgg16Model:

    def __init__(self):
        self.weights = np.load(weights_path, encoding='latin1')
        self.activation_fn = tf.nn.relu
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.use_bias = True

    def build(self, input_tensor, trainable=False):
        self.conv1_1 = self.conv2d(input_tensor, 'conv1_1', 64, trainable)  # batch_size * 224 * 224 * 64
        self.conv1_2 = self.conv2d(self.conv1_1, 'conv1_2', 64, trainable)  # batch_size * 224 * 224 * 64

        # max-pooling is performed over a 2x2 pixel window, with stride 2. batch_size * 112 * 112 * 64
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_1, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv2_1 = self.conv2d(self.max_pool1, 'conv2_1', 128, trainable)  # batch_size * 112 * 112 * 128
        self.conv2_2 = self.conv2d(self.conv2_1, 'conv2_2', 128, trainable)  # batch_size * 112 * 112 * 128

        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, (2, 2), (2, 2), padding=self.pool_padding)
        # batch_size * 56 * 56 * 128

        self.conv3_1 = self.conv2d(self.max_pool2, 'conv3_1', 256, trainable)  # batch_size * 56 * 56 * 256
        self.conv3_2 = self.conv2d(self.conv3_1, 'conv3_2', 256, trainable)  # batch_size * 56 * 56 * 256
        self.conv3_3 = self.conv2d(self.conv3_2, 'conv3_3', 256, trainable)  # batch_size * 56 * 56 * 256

        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3, (2, 2), (2, 2), padding=self.pool_padding)
        # batch_size * 28 * 28 * 256

        self.conv4_1 = self.conv2d(self.max_pool3, 'conv4_1', 512, trainable)  # batch_size * 28 * 28 * 512
        self.conv4_2 = self.conv2d(self.conv4_1, 'conv4_2', 512, trainable)  # batch_size * 28 * 28 * 512
        self.conv4_3 = self.conv2d(self.conv4_2, 'conv4_3', 512, trainable)  # batch_size * 28 * 28 * 512

        self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3, (2, 2), (2, 2), padding=self.pool_padding)
        # batch_size * 14 * 14 * 512

        self.conv5_1 = self.conv2d(self.max_pool4, 'conv5_1', 512, trainable)
        self.conv5_2 = self.conv2d(self.conv5_1, 'conv5_2', 512, trainable)
        self.conv5_3 = self.conv2d(self.conv5_2, 'conv5_3', 512, trainable)

        self.reshaped_conv11 = tf.reshape(self.conv1_1, shape=(-1, 224 * 224 * 64))
        self.reshaped_conv12 = tf.reshape(self.conv1_2, shape=(-1, 224 * 224 * 64))
        self.reshaped_maxp1 = tf.reshape(self.max_pool1, shape=(-1, 112 * 112 * 64))

        self.reshaped_conv21 = tf.reshape(self.conv2_1, shape=(-1, 112 * 112 * 128))
        self.reshaped_conv22 = tf.reshape(self.conv2_2, shape=(-1, 112 * 112 * 128))
        self.reshaped_maxp2 = tf.reshape(self.max_pool2, shape=(-1, 56 * 56 * 128))

        self.reshaped_conv31 = tf.reshape(self.conv3_1, shape=(-1, 56 * 56 * 256))
        self.reshaped_conv32 = tf.reshape(self.conv3_2, shape=(-1, 56 * 56 * 256))
        self.reshaped_conv33 = tf.reshape(self.conv3_3, shape=(-1, 56 * 56 * 256))
        self.reshaped_maxp3 = tf.reshape(self.max_pool3, shape=(-1, 28 * 28 * 256))

        self.reshaped_conv41 = tf.reshape(self.conv4_1, shape=(-1, 28 * 28 * 512))
        self.reshaped_conv42 = tf.reshape(self.conv4_2, shape=(-1, 28 * 28 * 512))
        self.reshaped_conv43 = tf.reshape(self.conv4_3, shape=(-1, 28 * 28 * 512))
        self.reshaped_maxp4 = tf.reshape(self.max_pool4, shape=(-1, 14 * 14 * 512))

        self.reshaped_conv51 = tf.reshape(self.conv5_1, shape=(-1, 14*14*512))
        self.reshaped_conv52 = tf.reshape(self.conv5_2, shape=(-1, 14*14*512))
        self.reshaped_conv53 = tf.reshape(self.conv5_3, shape=(-1, 14*14*512))

        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.reshaped_maxp5 = tf.reshape(self.max_pool5, shape=(-1, 7 * 7 * 512))

        self.fc6 = self.fc(self.reshaped_maxp5, 'fc6', 4096, trainable)
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
                                # kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                                # bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                                use_bias=self.use_bias
                                )

    def fc(self, layer, name, size, trainable):
        weights = self.weights[name+'_W']
        bias = self.weights[name+'_b']
        return tf.layers.dense(layer, size, activation=self.activation_fn,
                               name=name, trainable=trainable,
                               kernel_initializer=tf.constant_initializer(weights, dtype=tf.float32),
                               bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),
                               # kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                               # bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                               use_bias=self.use_bias)


def get_batches(x, y, batch_size=my_batch_size):
    num_rows = y.shape[0]
    num_batches = num_rows // batch_size

    if num_rows % batch_size != 0:
        num_batches = num_batches + 1

    for batch in range(num_batches):
        yield x[batch_size * batch: batch_size * (batch + 1)], y[batch_size * batch: batch_size * (batch + 1)]


def load_image(image_path, mean=vgg_mean):
    image = skimage.io.imread(image_path)
    resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
    bgr = resized_image[:, :, ::-1] - mean
    return bgr


def extract_features(image_directory, filenames, model_save_dir, batch_size=my_batch_size):
    start_zero = time.time()
    tf.reset_default_graph()

    # create mapping of filename -> vgg features

    num_files = len(filenames)
    num_batches = int(math.ceil(num_files / batch_size))

    orig_im_arr = np.empty((0, 150528), dtype=np.float32)
    # conv11_arr = np.empty((0, 3211264), dtype=np.float32)
    # conv12_arr = np.empty((0, 3211264), dtype=np.float32)
    # maxp1_arr = np.empty((0, 802816), dtype=np.float32)
    #
    # conv21_arr = np.empty((0, 1605632), dtype=np.float32)
    # conv22_arr = np.empty((0, 1605632), dtype=np.float32)
    # maxp2_arr = np.empty((0, 401408), dtype=np.float32)
    #
    # conv31_arr = np.empty((0, 802816), dtype=np.float32)
    # conv32_arr = np.empty((0, 802816), dtype=np.float32)
    # conv33_arr = np.empty((0, 802816), dtype=np.float32)
    # maxp3_arr = np.empty((0, 200704), dtype=np.float32)
    #
    # conv41_arr = np.empty((0, 401408), dtype=np.float32)
    # conv42_arr = np.empty((0, 401408), dtype=np.float32)
    # conv43_arr = np.empty((0, 401408), dtype=np.float32)
    # maxp4_arr = np.empty((0, 100352), dtype=np.float32)

    # conv51_arr = np.empty((0, 100352), dtype=np.float32)
    # conv52_arr = np.empty((0, 100352), dtype=np.float32)
    conv53_arr = np.empty((0, 100352), dtype=np.float32)
    maxp5_arr = np.empty((0, 25088), dtype=np.float32)

    fc6_arr = np.empty((0, 4096), dtype=np.float32)
    fc7_arr = np.empty((0, 4096), dtype=np.float32)

    # valid_file_lst = []

    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:
            _input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='images')
            vgg = Vgg16Model()
            vgg.build(_input)

            sess.run(tf.global_variables_initializer())

            for i in range(num_batches):
                batch_filenames = filenames[i*batch_size: ((i+1)*batch_size)]

                print('batch {} of {}'.format(i+1, num_batches))
                start = time.time()
                images = np.array([load_image(os.path.join(image_directory, f)) for f in batch_filenames])
                reshaped_images = np.reshape(images, (-1, 224*224*3))
                orig_im_arr = np.append(orig_im_arr, reshaped_images, axis=0)

                if images.shape[0] != 64:
                    print images.shape
                a = 12
                end = time.time()
                print("\timage loading took {:.4f} sec".format(end - start))

                start = end

                # batch_conv11, batch_conv12, batch_maxp1, \
                # batch_conv21, batch_conv22, batch_maxp2, \
                # batch_conv31, batch_conv32, batch_conv33, batch_maxp3, \
                # batch_conv41, batch_conv42, batch_conv43, batch_maxp4, \
                # batch_conv51, batch_conv52, batch_conv53, batch_maxp5, \
                # batch_fc6, batch_fc7 = \
                #     sess.run([vgg.reshaped_conv11, vgg.reshaped_conv12, vgg.reshaped_maxp1,
                #               vgg.reshaped_conv21, vgg.reshaped_conv22, vgg.reshaped_maxp2,
                #               vgg.reshaped_conv31, vgg.reshaped_conv32, vgg.reshaped_conv33, vgg.reshaped_maxp3,
                #               vgg.reshaped_conv41, vgg.reshaped_conv42, vgg.reshaped_conv43, vgg.reshaped_maxp4,
                #               vgg.reshaped_conv51, vgg.reshaped_conv52, vgg.reshaped_conv53, vgg.reshaped_maxp5,
                #               vgg.fc6, vgg.fc7], feed_dict={_input: images})
                batch_conv53, batch_maxp5, batch_fc6, batch_fc7 = \
                    sess.run([vgg.reshaped_conv53, vgg.reshaped_maxp5, vgg.fc6, vgg.fc7], feed_dict={_input: images})

                # print('conv51 shape', batch_conv51.shape)
                # print('conv11 shape', batch_conv11.shape)
                # print('conv12 shape', batch_conv12.shape)
                # print('max p1 shape', batch_maxp1.shape)
                #
                # print('conv21 shape', batch_conv21.shape)
                # print('conv22 shape', batch_conv22.shape)
                # print('max p2 shape', batch_maxp2.shape)
                #
                # print('conv31 shape', batch_conv31.shape)
                # print('conv32 shape', batch_conv32.shape)
                # print('conv33 shape', batch_conv33.shape)
                # print('max p3 shape', batch_maxp3.shape)
                #
                # print('conv41 shape', batch_conv41.shape)
                # print('conv42 shape', batch_conv42.shape)
                # print('conv43 shape', batch_conv43.shape)
                # print('max p4 shape', batch_maxp4.shape)


                # conv11_arr = np.append(conv11_arr, np.array(batch_conv11), axis=0)
                # conv12_arr = np.append(conv12_arr, np.array(batch_conv12), axis=0)
                # maxp1_arr = np.append(maxp1_arr, np.array(batch_maxp1), axis=0)
                #
                # conv21_arr = np.append(conv21_arr, np.array(batch_conv21), axis=0)
                # conv22_arr = np.append(conv22_arr, np.array(batch_conv22), axis=0)
                # maxp2_arr = np.append(maxp2_arr, np.array(batch_maxp2), axis=0)
                #
                # conv31_arr = np.append(conv31_arr, np.array(batch_conv31), axis=0)
                # conv32_arr = np.append(conv32_arr, np.array(batch_conv32), axis=0)
                # conv33_arr = np.append(conv33_arr, np.array(batch_conv33), axis=0)
                # maxp3_arr = np.append(maxp3_arr, np.array(batch_maxp3), axis=0)
                #
                # conv41_arr = np.append(conv41_arr, np.array(batch_conv41), axis=0)
                # conv42_arr = np.append(conv42_arr, np.array(batch_conv42), axis=0)
                # conv43_arr = np.append(conv43_arr, np.array(batch_conv43), axis=0)
                # maxp4_arr = np.append(maxp4_arr, np.array(batch_maxp4), axis=0)

                # conv51_arr = np.append(conv51_arr, np.array(batch_conv51), axis=0)
                # conv52_arr = np.append(conv52_arr, np.array(batch_conv52), axis=0)
                conv53_arr = np.append(conv53_arr, np.array(batch_conv53), axis=0)
                maxp5_arr = np.append(maxp5_arr, np.array(batch_maxp5), axis=0)

                fc6_arr = np.append(fc6_arr, np.array(batch_fc6), axis=0)
                fc7_arr = np.append(fc7_arr, np.array(batch_fc7), axis=0)

                # valid_file_lst += batch_filenames

                end = time.time()
                print("\tprediction took {:.4f} sec".format(end - start))

            # np.save(os.path.join(model_save_dir, 'conv11.npy'), conv11_arr)
            # np.save(os.path.join(model_save_dir, 'conv12.npy'), conv12_arr)
            # np.save(os.path.join(model_save_dir, 'maxpool1.npy'), maxp1_arr)
            #
            # np.save(os.path.join(model_save_dir, 'conv21.npy'), conv21_arr)
            # np.save(os.path.join(model_save_dir, 'conv22.npy'), conv22_arr)
            # np.save(os.path.join(model_save_dir, 'maxpool2.npy'), maxp2_arr)
            #
            # np.save(os.path.join(model_save_dir, 'conv31.npy'), conv31_arr)
            # np.save(os.path.join(model_save_dir, 'conv32.npy'), conv32_arr)
            # np.save(os.path.join(model_save_dir, 'conv33.npy'), conv33_arr)
            # np.save(os.path.join(model_save_dir, 'maxpool3.npy'), maxp3_arr)
            #
            # np.save(os.path.join(model_save_dir, 'conv41.npy'), conv41_arr)
            # np.save(os.path.join(model_save_dir, 'conv42.npy'), conv42_arr)
            # np.save(os.path.join(model_save_dir, 'conv43.npy'), conv43_arr)
            # np.save(os.path.join(model_save_dir, 'maxpool4.npy'), maxp4_arr)
            #
            # np.save(os.path.join(model_save_dir, 'conv51.npy'), conv51_arr)
            # np.save(os.path.join(model_save_dir, 'conv52.npy'), conv52_arr)
            np.save(os.path.join(model_save_dir, 'full_conv53.npy'), conv53_arr)
            np.save(os.path.join(model_save_dir, 'full_maxpool5.npy'), maxp5_arr)

            np.save(os.path.join(model_save_dir, 'full_fc6.npy'), fc6_arr)
            np.save(os.path.join(model_save_dir, 'full_fc7.npy'), fc7_arr)
            np.save(os.path.join(model_save_dir, 'full_orig_im_arr.npy'), orig_im_arr)

            print('Length of valid image files: {}'.format(orig_im_arr.shape[0]))

            end = time.time()
            print("\tTotal feedforward took {:.4f} sec".format(end - start_zero))
            # load_feats = np.load(feat_path, encoding='latin1').item()

    return


def extract_e_7800_features():
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)

    im_names = df['img_name'].values

    e_dir = '/home/amanda/Documents/E_faces/common_lst/'
    model_save_dir = '/home/amanda/Documents/vgg_model/e_7800_feat/'
    extract_features(image_directory=e_dir, filenames=im_names, model_save_dir=model_save_dir)
    return


def unbalanced_train_test_split():
    # 3000 s, 3000 f = train. 97 s + 1703 f = 1800 test.

    start_t = time.time()
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)

    success_num = sum(df['success'] == 1)
    fail_num = sum(df['success'] == 0)

    all_fail = df[df['success'] == 0]
    all_success = df[df['success'] == 1]

    train_fail = all_fail.iloc[:3000]
    train_success = all_success.iloc[:3000]

    train_ind = train_fail.index.tolist() + train_success.index.tolist()
    all_ind = df.index.tolist()
    test_ind = [x for x in all_ind if x not in train_ind]

    train_df = df.iloc[train_ind]
    test_df = df.iloc[test_ind]
    train_df.to_pickle('/home/amanda/Documents/vgg_model/e_fund_predict_exp/train_df.pkl')
    test_df.to_pickle('/home/amanda/Documents/vgg_model/e_fund_predict_exp/test_df.pkl')
    print('time elapsed = {}'.format(time.time()-start_t))
    return


def male_only_train_test_split():
    # 3000 s, 3000 f = train. 97 s + 1703 f = 1800 test.

    start_t = time.time()
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)

    male_df = df[df['gender'] == 'male']

    success_num = sum(male_df['success'] == 1)
    fail_num = sum(male_df['success'] == 0)

    all_fail = male_df[male_df['success'] == 0]
    all_success = male_df[male_df['success'] == 1]

    train_same_num = 2500
    fail_rand_seed = np.arange(all_fail.shape[0])
    np.random.shuffle(fail_rand_seed)
    train_fail = all_fail.iloc[fail_rand_seed[:train_same_num]]

    success_rand_seed = np.arange(all_success.shape[0])
    np.random.shuffle(success_rand_seed)
    train_success = all_success.iloc[success_rand_seed[:train_same_num]]

    train_ind = train_fail.index.tolist() + train_success.index.tolist()
    all_ind = male_df.index.tolist()
    test_ind = [x for x in all_ind if x not in train_ind]

    train_df = df.iloc[train_ind]
    test_df = df.iloc[test_ind]
    # train_df.to_pickle('/home/amanda/Documents/vgg_model/e_fund_predict_exp/train_male_df.pkl')
    # test_df.to_pickle('/home/amanda/Documents/vgg_model/e_fund_predict_exp/test_male_df.pkl')
    print('time elapsed = {}'.format(time.time()-start_t))
    return train_df, test_df

# todo: how to create more test samples: create mirror faces.
# todo: predict only on male faces.

# analyze if the E group is different from the general population in terms of first impression or anything else.


def do_pca_then_logistic_regression(img_feat_file_name):
    exp_dir = '/home/amanda/Documents/vgg_model/e_fund_predict_exp/'
    cur_feat = img_feat_file_name[5:-4]

    print('Load image feature from {}...'.format(img_feat_file_name))
    img_feat_dir = '/home/amanda/Documents/vgg_model/e_7800_feat/'
    img_feat = np.load(img_feat_dir + img_feat_file_name)  # img_num * feat_num.

    print('Load train-test split and y data.')

    train_df = pd.read_pickle(exp_dir + 'train_df.pkl')
    test_df = pd.read_pickle(exp_dir + 'test_df.pkl')
    my_map = {'female': 0, 'male': 1}
    train_df = train_df.applymap(lambda s: my_map.get(s) if s in my_map else s)
    test_df = test_df.applymap(lambda s: my_map.get(s) if s in my_map else s)

    train_ind = train_df.index.tolist()
    test_ind = test_df.index.tolist()

    x_train = img_feat[train_ind, :]
    x_test = img_feat[test_ind, :]
    y_train = train_df['success'].values
    y_test = test_df['success'].values

    # # train PCA feature.
    # print('start pca. orig shape = {}'.format(img_feat.shape))
    # pca = PCA(n_components=1000)
    # pca.fit(x_train)
    #
    # cum_ratio = np.cumsum(pca.explained_variance_ratio_)
    #
    # pca_save_name = os.path.join(exp_dir, img_feat_file_name[5:-4]+'_pca_model.pkl')
    # pickle.dump(pca, open(pca_save_name, 'wb'))
    # print('PCA saved...')

    # load pca model.
    file_name = exp_dir + img_feat_file_name[5:-4]+'_pca_model.pkl'
    with open(file_name, 'rb') as f:
        pca = pickle.load(f)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    add_feature_name = 'smile'

    add_train_feat = train_df[add_feature_name].values
    add_train_feat = add_train_feat[:, np.newaxis]

    add_test_feat = test_df[add_feature_name].values
    add_test_feat = add_test_feat[:, np.newaxis]

    performance_df = pd.DataFrame(columns=['pca number', 'train acc', 'test acc'])

    print('Start to fine-tune PCA number...')
    pca_start = 300
    pca_end = 600
    pca_nums = np.linspace(pca_start, pca_end, num=31, dtype=int)
    log_regressor = LogisticRegression()

    highest_test = 0
    for ind, pca_num in enumerate(pca_nums):
        log_regressor.fit(x_train_pca[:, :pca_num], y_train)
        x_train_merge = x_train_pca[:, :pca_num]
        x_test_merge = x_test_pca[:, :pca_num]

        x_train_merge = np.append(x_train_merge, add_train_feat, 1)
        x_test_merge = np.append(x_test_merge, add_test_feat, 1)

        train_acc = log_regressor.score(x_train_merge, y_train)
        test_acc = log_regressor.score(x_test_merge, y_test)

        if test_acc > highest_test:
            highest_test = test_acc
            print('pca = {}, train score = {}, test = {}'.format(pca_num, train_acc, test_acc))

        performance_df.loc[ind] = [pca_num, train_acc, test_acc]

    # visualize the performance change curves.
    fig = plt.figure()
    fig.suptitle(cur_feat, fontsize=20)
    plt.scatter(performance_df['pca number'], performance_df['train acc'], label='train', alpha=0.5)
    plt.scatter(performance_df['pca number'], performance_df['test acc'], label='test', alpha=0.5)
    plt.xlabel('PCA number', fontsize=18)
    plt.ylabel('Correlation', fontsize=16)
    plt.ylim(0, 1)
    plt.plot([pca_start, pca_end], [0.5, 0.5], linestyle='--', alpha=0.5)
    plt.grid()
    plt.legend()
    fig.savefig(exp_dir + cur_feat + '_' + add_feature_name + '.jpg')

    return


def predict_on_male_only():
    start_t = time.time()
    print('Generating a new randomly split of train and test data for males only...')
    train_df, test_df = male_only_train_test_split()
    exp_dir = '/home/amanda/Documents/vgg_model/e_fund_predict_exp/'
    img_feat_load_dir = '/home/amanda/Documents/vgg_model/e_7800_feat/'
    img_feat_file_name = 'full_orig_im_arr.npy'

    img_feat = np.load(img_feat_load_dir + img_feat_file_name)
    cur_feat = img_feat_file_name[5:-4]

    train_ind = train_df.index.tolist()
    test_ind = test_df.index.tolist()

    x_train = img_feat[train_ind, :]
    x_test = img_feat[test_ind, :]
    y_train = train_df['success'].values
    y_test = test_df['success'].values

    # load pca model.
    print('Elasped time = {}. Now loading pca model...'.format(time.time()-start_t))
    file_name = exp_dir + img_feat_file_name[5:-4]+'_pca_model.pkl'
    with open(file_name, 'rb') as f:
        pca = pickle.load(f)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    x_start = 60
    x_end = 160
    num_pc_pts = 20
    pca_nums = np.linspace(x_start, x_end, num=num_pc_pts, dtype=int)

    log_regressor = LogisticRegression()

    add_feature_name = ['age', 'trustworthy', 'attractive', 'aggressive', 'responsible', 'intelligent', 'sociable']
    add_train_feat = train_df[add_feature_name].values
    add_test_feat = test_df[add_feature_name].values

    performance_df = pd.DataFrame(columns=['pca number', 'train acc', 'test acc'])
    highest_test = 0
    best_pc_num = 0

    for ind, pca_num in enumerate(pca_nums):
        x_train_merge = x_train_pca[:, :pca_num]
        x_test_merge = x_test_pca[:, :pca_num]

        x_train_merge = np.append(x_train_merge, add_train_feat, 1)
        x_test_merge = np.append(x_test_merge, add_test_feat, 1)
        #     x_train_merge = add_train_feat
        #     x_test_merge = add_test_feat

        log_regressor.fit(x_train_merge, y_train)

        score_1 = log_regressor.score(x_train_merge, y_train)
        score_2 = log_regressor.score(x_test_merge, y_test)

        if score_2 > highest_test:
            highest_test = score_2
            best_pc_num = pca_num
        print('pca = {}, train score = {}, test = {}'.format(pca_num, score_1, score_2))

        performance_df.loc[ind] = [pca_num, score_1, score_2]
    print('Test score highest {}, pca_num = {}'.format(highest_test, best_pc_num))

    # # visualize the performance change curves.
    # fig = plt.figure()
    # fig.suptitle(cur_feat, fontsize=20)
    # plt.scatter(performance_df['pca number'], performance_df['train acc'], label='train', alpha=0.5)
    # plt.scatter(performance_df['pca number'], performance_df['test acc'], label='test', alpha=0.5)
    # plt.xlabel('PCA number', fontsize=18)
    # plt.ylabel('Correlation', fontsize=16)
    # plt.ylim(0.4, 0.7)
    # plt.plot([x_start, x_end], [0.5, 0.5], linestyle='--', alpha=0.5)
    # plt.grid()
    # plt.legend()
    # fig.savefig(exp_dir + cur_feat + '_' + 'multi_feature_select' + '.jpg')
    return

# todo: Add friendly, happy feature into the impression prediction list.
# todo: to be added: happy, friendly, kind, caring, interesting.


def general_viz(tar_impression):
    print(tar_impression)
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)

    plt.figure(figsize=(12, 9))
    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(range(1, 9, 1), fontsize=14)
    plt.yticks(range(50, 300, 50), fontsize=14)

    # Along the same vein, make sure your axis labels are large
    # enough to be easily read as well. Make them slightly larger
    # than your axis tick labels so they stand out.
    plt.xlabel(tar_impression, fontsize=16)
    plt.ylabel("Count", fontsize=16)

    bins = np.linspace(1, 9, 80)
    succ = df[df['success'] == 1]
    fail = df[df['success'] == 0]

    alpha = .3
    plt.hist(fail[tar_impression], bins, alpha=alpha, label='not funded', density=True)
    plt.hist(succ[tar_impression], bins, alpha=alpha, label='funded', density=True)
    plt.legend(loc='upper right', prop={'size': 16})

    save_name = '/home/amanda/Documents/E_faces/gender_viz/general_viz/' + tar_impression + '_by_fund.jpg'
    plt.savefig(save_name, bbox_inches="tight")
    print('Done.')

# extract_e_7800_features()
# unbalanced_train_test_split()
# male_only_train_test_split()
# predict_on_male_only()


def vis_gender_and_rating(tar_impression):
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)
    f_ind = df['gender'] == 'female'
    m_ind = df['gender'] == 'male'
    fund_ind = df['success'] == 1
    fail_ind = df['success'] == 0

    plt.figure(figsize=(12, 9))
    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(range(1, 9, 1), fontsize=14)
    plt.yticks(range(50, 300, 50), fontsize=14)

    # Along the same vein, make sure your axis labels are large
    # enough to be easily read as well. Make them slightly larger
    # than your axis tick labels so they stand out.
    plt.xlabel(tar_impression, fontsize=16)
    plt.ylabel("Count", fontsize=16)

    bins = np.linspace(1, 9, 80)

    alpha = .3
    density_ind = False
    plt.hist(df[f_ind & fund_ind][tar_impression], bins, alpha=alpha, label='female funded', density=density_ind)
    plt.hist(df[f_ind & fail_ind][tar_impression], bins, alpha=alpha, label='female not funded', density=density_ind)
    plt.hist(df[m_ind & fund_ind][tar_impression], bins, alpha=alpha, label='male funded', density=density_ind)
    plt.hist(df[m_ind & fail_ind][tar_impression], bins, alpha=alpha, label='male not funded', density=density_ind)
    plt.legend(loc='upper right', prop={'size': 16})

    save_name = '/home/amanda/Documents/E_faces/gender_viz/general_viz/' + tar_impression + '_by_gender.jpg'
    plt.savefig(save_name, bbox_inches="tight")
    return

# general_viz(tar_impression='aggressive')


# vis_gender_and_rating('aggressive')
# vis_gender_and_rating('attractive')
# vis_gender_and_rating('trustworthy')
# vis_gender_and_rating('sociable')
# vis_gender_and_rating('intelligent')
# vis_gender_and_rating('responsible')