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


default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
all_im_dir = '/home/amanda/Documents/cropped_v_valid/'
# batch_size = 64


def filter_crop_valid_face():
    vc_df_save_path = 'tmp_data/vc_df.pkl'
    vc_cropped_dir = '/home/amanda/Documents/cropped_v/'
    vc_cropped_valid_dir = '/home/amanda/Documents/cropped_v_valid/'

    if not os.path.exists(vc_cropped_dir):
        os.makedirs(vc_cropped_dir)

    filenames = os.listdir(vc_cropped_dir)
    num_files = len(filenames)

    i = 0
    j = 0
    for cur_file_name in filenames:
        img = cv2.imread(vc_cropped_dir + cur_file_name)
        if img is None:
            i += 1
            print('{} img cannot be open. count {}'.format(cur_file_name, i))
        else:
            img_save_name = vc_cropped_valid_dir + cur_file_name
            j += 1
            cv2.imwrite(img_save_name, img)
    print(len(filenames))
    print(j)  # 9620.
    print(i+j)
    return


def filter_validate():
    vc_cropped_valid_dir = '/home/amanda/Documents/cropped_v_valid/'
    filenames = os.listdir(vc_cropped_valid_dir)
    for cur_file in filenames:
        img = cv2.imread(vc_cropped_valid_dir+cur_file)
        if img is None:
            print('{} img cannot be open.'.format(cur_file))

        if img.shape[0] * img.shape[1] < 400:
            print('{} img is too small.'.format(cur_file))
            os.remove(vc_cropped_valid_dir+cur_file)

    # vc6049_cb.jpg too small. should use image size to filter out some invalid faces later.
    return


def get_batches(x, y, batch_size=64):
    num_rows = y.shape[0]
    num_batches = num_rows // batch_size

    if num_rows % batch_size != 0:
        num_batches = num_batches + 1

    for batch in range(num_batches):
        yield x[batch_size * batch: batch_size * (batch+1)], y[batch_size * batch: batch_size * (batch+1)]


def load_image(image_path, mean=vgg_mean):
    image = skimage.io.imread(image_path)
    resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
    bgr = resized_image[:, :, ::-1] - mean
    return bgr


class Vgg16Model:

    def __init__(self):
        self.weights = np.load(weights_path, encoding='latin1')
        self.activation_fn = tf.nn.relu
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.use_bias = True

    def build(self, input_tensor, trainable=False):
        self.conv1_1 = self.conv2d(input_tensor, 'conv1_1', 64, trainable)
        self.conv1_2 = self.conv2d(self.conv1_1, 'conv1_2', 64, trainable)

        # max-pooling is performed over a 2x2 pixel window, with stride 2.
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

        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)

        reshaped = tf.reshape(self.max_pool5, shape=(-1, 7*7*512))

        self.fc6 = self.fc(reshaped, 'fc6', 4096, trainable)
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


def extract_features(image_directory, batch_size=64):
    tf.reset_default_graph()

    # create mapping of filename -> vgg features
    codes_fc6 = {}
    codes_fc7 = {}
    predictions = {}

    filenames = os.listdir(image_directory)
    num_files = len(filenames)
    num_batches = int(math.ceil(num_files / batch_size))

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
                images = np.array([load_image(image_directory + f) for f in batch_filenames])
                end = time.time()
                print("\timage loading took {:.4f} sec".format(end - start))

                start = end

                batch_codes_fc6, batch_codes_fc7 = sess.run([vgg.fc6, vgg.fc7], feed_dict={_input: images})

                end = time.time()
                print("\tprediction took {:.4f} sec".format(end - start))

                for i, filename in enumerate(batch_filenames):
                    codes_fc6[filename] = batch_codes_fc6[i]
                    codes_fc7[filename] = batch_codes_fc7[i]

        return codes_fc6, codes_fc7


def extract_f67_feat():
    print('Extracting training codes for fc6 and fc7')
    all_im_feature_fc6, all_im_feature_fc7 = extract_features(all_im_dir)
    np.save('/home/amanda/Documents/vgg_model/all_vc_vgg_fc6.npy', all_im_feature_fc6)
    np.save('/home/amanda/Documents/vgg_model/all_vc_vgg_fc7.npy', all_im_feature_fc7)


def match_feat_with_rating(vc_feat_path='/home/amanda/Documents/vgg_model/all_vc_vgg_fc7.npy'):
    start_t = time.time()
    vc_feats_ranking_save_path = '/home/amanda/Documents/vgg_model/vc_feat_fc7_and_ranking.npz'

    vc_df_save_path = '../pretrain_first_impression/tmp_data/vc_df.pkl'
    vc_df = pd.read_pickle(vc_df_save_path)

    vc_feats = np.load(vc_feat_path, encoding='latin1').item()

    feat_num = 4096
    feat_arr = np.empty((0, feat_num), dtype=np.float32)
    rank_num = 2
    ranking_arr = np.empty((0, rank_num), dtype=np.float32)
    name_lst = open('/home/amanda/Documents/vgg_model/vc_img_valid_lst.txt', 'w')

    i = 0
    for ind, row in vc_df.iterrows():
        if ind % 1000 == 0:
            print('{} out of {}'.format(ind, len(vc_df)))

        vc_im_name = row['short_id'] + '_cb.jpg'
        if vc_im_name in vc_feats.keys():
            cur_feat = vc_feats[vc_im_name]
            cur_rank = [row['VC - Investment # rank'], row['VC - Investment $ rank']]
            name_lst.write(vc_im_name)
            name_lst.write('\n')
            feat_arr = np.append(feat_arr, np.array([cur_feat]), axis=0)
            ranking_arr = np.append(ranking_arr, np.array([cur_rank]), axis=0)
        else:
            i += 1
    name_lst.close()
    np.savez(vc_feats_ranking_save_path, feat_arr=feat_arr, ranking_arr=ranking_arr)
    print('elapsed time = {}'.format(time.time()-start_t))
    print i

    return


def predict_impressions_with_pretrained_model():
    start_t = time.time()
    vc_feats_ranking_save_path = '/home/amanda/Documents/vgg_model/vc_feat_fc7_and_ranking.npz'
    vc_feats_and_ranking_data = np.load(vc_feats_ranking_save_path)

    vc_feats = vc_feats_and_ranking_data['feat_arr']
    vc_ranking = vc_feats_and_ranking_data['ranking_arr']
    rank1 = vc_ranking[:, 0]
    rank2 = vc_ranking[:, 1]

    # load pca model
    pca_model_path = 'tmp_data/pca_fc7_keep_num_300_model.sav'
    with open(pca_model_path, 'rb') as cur_file:
        loaded_pca_model = pickle.load(cur_file)

    vc_feats_after_pca = loaded_pca_model.transform(vc_feats)
    print('feat shape after transformation, {}'.format(vc_feats_after_pca.shape))

    # load regression model
    regressor_path = 'tmp_data/regressor_fc7_model.sav'
    with open(regressor_path, 'rb') as cur_file:
        regressor_model = pickle.load(cur_file)

    # predict on vc feats data
    vc_impression_predict = regressor_model.predict(vc_feats_after_pca)
    np.savez('/home/amanda/Documents/vgg_model/vc_impression_prediction.npz', vc_impression_predict=vc_impression_predict)
    print('elapsed time = {}'.format(time.time()-start_t))

    # compare correlations.
    impression_names = ['attractive', 'aggressive', 'trustworthy', 'intelligent', 'sociable', 'responsible']

    for ind, cur_trait_name in enumerate(impression_names):
        print('\n\nCur impression: {}'.format(cur_trait_name))
        cur_impression = vc_impression_predict[:, ind]

        [cor_1, p_1] = pearsonr(cur_impression, rank1)
        [cor_2, p_2] = pearsonr(cur_impression, rank2)
        print('Rank 1: cor 1 = {}, p 1 = {}\n'.format(cor_1, p_1))
        print('Rank 2: cor 2 = {}, p 2 = {}\n'.format(cor_2, p_2))

    return


if __name__ == '__main__':
    # filter_crop_valid_face()
    # filter_validate()
    # extract_f67_feat()
    # match_feat_with_rating()
    predict_impressions_with_pretrained_model()