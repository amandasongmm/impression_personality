from __future__ import division
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


def predict_impression(current_folder_post_fix='0/',
                       crop_with_mask_root_dir='/home/amanda/Documents/cropped_new_crunchbase/',
                       save_result_root_dir='/home/amanda/Documents/predicted_results/crunchbase_new_data/',
                       feat_prefix='crunchbase',
                       csv_save_dir='/home/amanda/Documents/predicted_results/crunchbase_new_data/to_will/'
                       ):
    start_t = time.time()
    batch_size = 100

    default_device = '/gpu:0'
    num_hidden_neurons = 256

    vgg_mean = [103.939, 116.779, 123.68]
    weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'

    feat_save_dir = '/home/amanda/Documents/feat_folder/'
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'

    if not os.path.exists(feat_save_dir):
        os.makedirs(feat_save_dir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir)

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

        print('Total time elapsed: {:.2f} sec.'.format(time.time() - start_zero))

        return conv52_arr, valid_file_lst

    # extract all the images' features in one array in a specified folder.
    crop_with_mask_dir = crop_with_mask_root_dir + current_folder_post_fix
    file_names = [f for f in os.listdir(crop_with_mask_dir) if f[-3:] == 'jpg' or f[-3:] == 'png']

    save_result_dir = save_result_root_dir + current_folder_post_fix

    feat_arr, valid_img_lst = extract_features(img_dir=crop_with_mask_dir, file_names=file_names,
                                               feat_save_dir=feat_save_dir,
                                               feat_prefix=feat_prefix+current_folder_post_fix[:-1],
                                               batch_size=batch_size)

    add_feat_path = save_result_dir + 'gender_age_only_ind_' + current_folder_post_fix[:-1] + '.pkl'
    add_feat_df = pd.read_pickle(add_feat_path)
    # add_feat_df = add_feat_df[['filename', 'age', 'gender binary']]

    add_feat_df = add_feat_df.set_index('filename')
    add_feat_df = add_feat_df.loc[valid_img_lst]
    add_feat_arr = add_feat_df.as_matrix()  # age, then gender.

    # apply the same normalization here.
    min_max_scaler = preprocessing.MinMaxScaler()
    add_feat_scale = min_max_scaler.fit_transform(add_feat_arr[:, :2])

    # do PCA on the whole data set.
    print('Start loading PCA model ...')

    feat_layer_name = 'conv52_2k'
    pca_save_name = '{}/{}_all_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)

    with open(pca_save_name, 'rb') as f:
        pca = pickle.load(f)

    x_pca = pca.transform(feat_arr)
    x_all = np.hstack((x_pca, add_feat_scale))

    print('Start do regression...')

    regressor_model_name = '{}/{}_all_data_regressor_model.pkl'.format(model_save_dir, feat_layer_name)
    with open(regressor_model_name, 'rb') as f:
        regressor = pickle.load(f)

    y_predict = regressor.predict(x_all)

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
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
    predict_df.to_csv(csv_save_dir+save_csv_name, sep='\t')

    print('Total time = {}'.format(time.time()-start_t))


predict_impression()
