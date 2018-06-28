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

default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
batch_size = 100
feat_save_dir = '/home/amanda/Documents/feat_folder/'
model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'


# common function
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
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2, (2, 2), (2, 2), padding=self.pool_padding)

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
def get_batches(x, y, batch_size):
    num_rows = y.shape[0]
    num_batches = num_rows // batch_size

    if num_rows % batch_size != 0:
        num_batches = num_batches + 1

    for batch in range(num_batches):
        yield x[batch_size * batch : batch_size * (batch+1)], y[batch_size * batch : batch_size * (batch+1)]


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
def extract_2k_feat():
    img_dir = '/home/amanda/Documents/2kfaces/'

    file_names = [f for f in os.listdir(img_dir) if f[-4:] == '.jpg']
    extract_features(img_dir=img_dir, file_names=file_names, feat_save_dir=feat_save_dir, feat_prefix='2kface',
                     batch_size=batch_size)


# specific function
def extract_vc_feat():
    img_dir = '/home/amanda/Documents/cropped_faces/vc_with_mask_narrow/'
    file_names = [f for f in os.listdir(img_dir) if f[-4:] == 'jpeg']
    id_lst = [f[:-8] for f in file_names]
    unique_id_lst = set(id_lst)

    im_lst = [im_id + '_cb.jpeg' if im_id + '_cb.jpeg' in file_names else im_id + '_tw.jpeg' for im_id in unique_id_lst]
    print('VC img length = {}'.format(len(im_lst)))
    extract_features(img_dir=img_dir, file_names=im_lst[:9700], feat_save_dir=feat_save_dir, feat_prefix='vc',
                     batch_size=batch_size)

    return


def split_on_2k(layer_name='conv53', random_state=30):
    start_t = time.time()
    print('Split the 2k dataset into training and testing sets...\n')

    train_test_split_data_save_name = os.path.join(feat_save_dir, '2kface_'+layer_name+'_train_test_data.npz')

    # load valid img lst.
    valid_img_file_name = '2kface_valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(feat_save_dir, valid_img_file_name))

    # load extracted features
    feature_file_name = os.path.join(feat_save_dir, '2kface_'+layer_name+'.npy')
    feat_arr = np.load(feature_file_name)  # 2200 * 100352, for example.

    # load selected impression traits.
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_arr = filtered_rating_df.as_matrix()

    # split all data into training and testing set.
    x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=0.2, random_state=random_state)
    np.savez(train_test_split_data_save_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('Done! Elapsed time = {}'.format(time.time()-start_t))


def pca_and_regression_on_2k(pca_keep_dim=250, layer_name='conv53'):
    start_t = time.time()
    data = np.load(os.path.join(feat_save_dir, '2kface_'+layer_name+'_train_test_data.npz'))
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

    # apply PCA on training data.
    pca = PCA(n_components=pca_keep_dim)
    pca.fit(x_train)
    print('PCA takes {}'.format(time.time()-start_t))

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)[-1]
    print('number of pc: {},  explained variance = {}\n'.
          format(pca_keep_dim, cum_ratio))  # 0.98

    # save the pca model
    pca_save_name = '{}/{}_keep_{}_ratio_{}_pca_model.pkl'.format(
        feat_save_dir, layer_name, str(pca_keep_dim), str(cum_ratio)[2:5])

    print('Saving pca model...\n')

    pickle.dump(pca, open(pca_save_name, 'wb'))

    # apply pca on raw feature matrix
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # multi_regressor = MultiOutputRegressor(Ridge())
    multi_regressor = MultiOutputRegressor(LinearRegression())
    multi_regressor.fit(x_train_pca, y_train)

    # save the regressor model
    regressor_save_name = '{}/{}_keep_{}_regressor_model.pkl'.format(feat_save_dir, layer_name, str(pca_keep_dim))
    # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
    pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

    # predict on training data
    y_train_predict = multi_regressor.predict(x_train_pca)

    # predict on test data
    y_test_predict = multi_regressor.predict(x_test_pca)

    # compare performance
    rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    txt_file_name = '{}/2k_{}_keep_{}_pca_performance.txt'.format(feat_save_dir, layer_name, str(pca_keep_dim))
    with open(txt_file_name, 'w') as txt_file:
        for ind, cur_name in enumerate(rating_names):
            line = 'Cur impression: {}\n'.format(cur_name)
            print(line)
            txt_file.write(line)

            train_gt = y_train[:, ind]
            train_pred = y_train_predict[:, ind]
            [train_cor, p] = pearsonr(train_gt, train_pred)

            test_gt = y_test[:, ind]
            test_pred = y_test_predict[:, ind]
            [test_cor, p] = pearsonr(test_gt, test_pred)
            line = 'train cor {}, test cor = {}\n\n'.format(train_cor, test_cor)
            print(line)
            txt_file.write(line)
    print('total time = {}'.format(time.time()-start_t))
    return


def repeat_split(layer_name, linspace_start, linspace_end):
    start_t = time.time()
    performance_df = pd.DataFrame(
        columns=['feat_layer', 'rand_num', 'test_ratio', 'pca_num',
                 'imp1', 'train1', 'test1', 'imp2', 'train2', 'test2', 'imp3', 'train3', 'test3',
                 'imp4', 'train4', 'test4', 'imp5', 'train5', 'test5', 'imp6', 'train6', 'test6'])

    print('Split the 2k dataset into training and testing sets...\n')

    # load valid img lst.
    valid_img_file_name = '2kface_valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(feat_save_dir, valid_img_file_name))

    # load extracted features
    feature_file_name = os.path.join(feat_save_dir, '2kface_'+layer_name+'.npy')
    feat_arr = np.load(feature_file_name)  # 2200 * 100352, for example.

    # load selected impression traits.
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_arr = filtered_rating_df.as_matrix()
    pca_keep_dim = 500
    test_ratio = 0.2
    itr_count = 0

    linspace_pt_num = (linspace_end-linspace_start)/10 + 1
    repeat_num = 10
    for rand in range(0, repeat_num):

        x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=test_ratio, random_state=rand)

        new_row = []
        new_row += [layer_name]
        new_row += [str(rand)]
        new_row += [str(test_ratio)]

        # apply PCA on training data.
        pca = PCA(n_components=pca_keep_dim)
        pca.fit(x_train)
        print('PCA takes {}'.format(time.time() - start_t))

        cum_ratio = np.cumsum(pca.explained_variance_ratio_)[-1]
        print('number of pc: {},  explained variance = {}\n'.
              format(pca_keep_dim, cum_ratio))  # 0.98

        # apply pca on raw feature matrix
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        fixed_part = new_row[:]

        for actual_pca in np.linspace(linspace_start, linspace_end, linspace_pt_num, dtype=int):
            new_row_cont = fixed_part[:]
            new_row_cont += [str(actual_pca)]
            # multi_regressor = MultiOutputRegressor(Ridge())
            multi_regressor = MultiOutputRegressor(LinearRegression())

            multi_regressor.fit(x_train_pca[:, :actual_pca], y_train)

            # predict on training data
            y_train_predict = multi_regressor.predict(x_train_pca[:, :actual_pca])

            # predict on test data
            y_test_predict = multi_regressor.predict(x_test_pca[:, :actual_pca])

            # compare performance
            rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
            print('cur iter = {} out of {}. actual pca num = {}'.format(rand+1, repeat_num, actual_pca))
            for ind, cur_name in enumerate(rating_names):
                line = 'Cur impression: {}\n'.format(cur_name)
                new_row_cont += [cur_name]
                # print(line)

                train_gt = y_train[:, ind]
                train_pred = y_train_predict[:, ind]
                [train_cor, p] = pearsonr(train_gt, train_pred)
                new_row_cont += [str(train_cor)]

                test_gt = y_test[:, ind]
                test_pred = y_test_predict[:, ind]
                [test_cor, p] = pearsonr(test_gt, test_pred)
                new_row_cont += [str(test_cor)]
                line = 'train cor {}, test cor = {}\n\n'.format(train_cor, test_cor)
                # print(line)

            performance_df.loc[itr_count] = new_row_cont
            itr_count += 1

        print('total time = {}'.format(time.time() - start_t))

    performance_df.to_csv('tmp_data/'+layer_name+'_2k_performance_'+str(linspace_start)+'-'+str(linspace_end)+'.csv')

    for feat_num in range(1, 7):
        att = performance_df[['rand_num', 'pca_num', 'imp' + str(feat_num), 'train' + str(feat_num), 'test' + str(feat_num)]]
        print(att.loc[0]['imp' + str(feat_num)])
        for i in np.linspace(linspace_start, linspace_end, linspace_pt_num, dtype=int):
            print('cur lay = {}, cur pca = {}, train, test cor = {:.2f}, {:.2f}'.format(layer_name, i,
                                att[att['pca_num'] == str(i)]['train' + str(feat_num)].mean(),
                                att[att['pca_num'] == str(i)]['test' + str(feat_num)].mean()))


# extract_2k_feat()
# extract_vc_feat()
# split_on_2k()
# pca_and_regression_on_2k()
# repeat_split(layer_name='conv52')
# repeat_split(layer_name='conv51')
repeat_split(layer_name='conv52', linspace_start=150, linspace_end=350)

