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

        self.reshaped_conv51 = tf.reshape(self.conv5_1, shape=(-1, 14*14*512))
        self.reshaped_conv52 = tf.reshape(self.conv5_2, shape=(-1, 14*14*512))
        self.reshaped_conv53 = tf.reshape(self.conv5_3, shape=(-1, 14*14*512))

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


def get_batches(x, y, batch_size=64):
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


def extract_features(image_directory, filenames, model_save_dir, batch_size=64):
    start_zero = time.time()
    tf.reset_default_graph()

    # create mapping of filename -> vgg features

    num_files = len(filenames)
    num_batches = int(math.ceil(num_files / batch_size))

    conv51_arr = np.empty((0, 100352), dtype=np.float32)
    conv52_arr = np.empty((0, 100352), dtype=np.float32)
    conv53_arr = np.empty((0, 100352), dtype=np.float32)
    maxpool5_arr = np.empty((0, 25088), dtype=np.float32)
    fc6_arr = np.empty((0, 4096), dtype=np.float32)
    fc7_arr = np.empty((0, 4096), dtype=np.float32)
    valid_file_lst = []

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
                end = time.time()
                print("\timage loading took {:.4f} sec".format(end - start))

                start = end

                batch_conv51, batch_conv52, batch_conv53, batch_max_pool5, batch_fc6, batch_fc7 = \
                    sess.run([vgg.reshaped_conv51, vgg.reshaped_conv52, vgg.reshaped_conv53, vgg.reshaped,
                              vgg.fc6, vgg.fc7], feed_dict={_input: images})

                # print('conv51 shape', batch_conv51.shape)

                conv51_arr = np.append(conv51_arr, np.array(batch_conv51), axis=0)
                conv52_arr = np.append(conv52_arr, np.array(batch_conv52), axis=0)
                conv53_arr = np.append(conv53_arr, np.array(batch_conv53), axis=0)
                maxpool5_arr = np.append(maxpool5_arr, np.array(batch_max_pool5), axis=0)
                fc6_arr = np.append(fc6_arr, np.array(batch_fc6), axis=0)
                fc7_arr = np.append(fc7_arr, np.array(batch_fc7), axis=0)

                valid_file_lst += batch_filenames

                end = time.time()
                print("\tprediction took {:.4f} sec".format(end - start))

            np.save(os.path.join(model_save_dir, 'conv51.npy'), conv51_arr)
            np.save(os.path.join(model_save_dir, 'conv52.npy'), conv52_arr)
            np.save(os.path.join(model_save_dir, 'conv53.npy'), conv53_arr)
            np.save(os.path.join(model_save_dir, 'maxpool5.npy'), maxpool5_arr)
            np.save(os.path.join(model_save_dir, 'fc6.npy'), fc6_arr)
            np.save(os.path.join(model_save_dir, 'fc7.npy'), fc7_arr)
            np.save(os.path.join(model_save_dir, 'valid_img_lst.npy'), valid_file_lst)

            end = time.time()
            print("\tTotal feedforward took {:.4f} sec".format(end - start_zero))
            # load_feats = np.load(feat_path, encoding='latin1').item()

    return


def extract_e_feature():
    print('Start extracting features from various layers in VGG in E-dataset...')
    e_dir = '/home/amanda/Documents/cropped_face/e_with_mask'
    model_save_dir = '/home/amanda/Documents/vgg_model/e_with_mask_feat'

    im_name_lst = [f for f in os.listdir(e_dir) if os.path.isfile(os.path.join(e_dir, f))]
    id_lst = [f[:-7] for f in im_name_lst]
    unique_id_lst = set(id_lst)

    e_im_lst = [im_id + '_cb.jpg' if im_id + '_cb.jpg' in im_name_lst else im_id + '_tw.jpg' for im_id in unique_id_lst]
    extract_features(image_directory=e_dir, filenames=e_im_lst, model_save_dir=model_save_dir)


def extract_2kface_feature():
    # img_dir = '/home/amanda/Downloads/2kfaces'  # Amanda office PC
    img_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces'  # Amanda home pc
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat'

    file_names = [f for f in os.listdir(img_dir) if f[-4:] == '.jpg']  # make sure every file is readable.
    extract_features(image_directory=img_dir, filenames=file_names, model_save_dir=model_save_dir)


def split_on_2k(feat_layer_file_name='maxpool5.npy'):
    start_t = time.time()
    print('Start spliting the 2k dataset into training and testing sets...\n')
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat'
    feat_layer_name = feat_layer_file_name[:-4]
    train_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_data.npz')

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(model_save_dir, valid_img_file_name))  #

    # load extracted feature from model_save_dir. Feature extraction layers include conv51,2,3, maxpool5, fc6, fc7.
    feature_file_name = os.path.join(model_save_dir, feat_layer_file_name)
    feat_arr = np.load(feature_file_name)  # 2176 * 100352, for example.

    # load selected impression attribute scores
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_arr = filtered_rating_df.as_matrix()

    # split all data into training and testing set.
    x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=0.1, random_state=42)
    np.savez(train_test_split_data_save_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('Done! Elapsed time = {}'.format(time.time()-start_t))
    return


def pca_and_regression_on_2k(feat_layer_file_name='conv53.npy', pca_keep_dim=150):

    print('\nStart running pca on 2k feature array...feat layer={}\n'.format(feat_layer_file_name))

    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat'
    feat_layer_name = feat_layer_file_name[:-4]
    train_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_data.npz')

    data = np.load(train_test_split_data_save_name)  # the data is generated by my_train_test_split func.
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

    # apply PCA on training data.
    pca = PCA(n_components=pca_keep_dim)
    pca.fit(x_train)

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)[-1]
    print('number of pc: {},  explained variance = {}\n'.
          format(pca_keep_dim, cum_ratio))  # 0.98

    # save the pca model
    pca_save_name = '{}/{}_keep_{}_ratio_{}_pca_model.pkl'.format(
        model_save_dir, feat_layer_name, str(pca_keep_dim), str(cum_ratio)[2:5])

    print('Saving pca model...\n')

    pickle.dump(pca, open(pca_save_name, 'wb'))

    # # # when loading the model: loaded_model = pickle.load(open(file_name), 'wb')
    # file_name = '/home/amanda/Documents/vgg_model/2k_feat/conv51_keep_num_200_keep_ratio_98_pca_model.pkl'
    # with open(file_name, 'rb') as f:
    #     pca = pickle.load(f)

    # apply pca on raw feature matrix
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # multi_regressor = MultiOutputRegressor(Ridge())
    multi_regressor = MultiOutputRegressor(LinearRegression())
    multi_regressor.fit(x_train_pca, y_train)

    # save the regressor model
    regressor_save_name = '{}/{}_keep_{}_regressor_model.pkl'.format(model_save_dir, feat_layer_name, str(pca_keep_dim))
    # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
    pickle.dump(multi_regressor, open(regressor_save_name, 'wb'))

    # predict on training data
    y_train_predict = multi_regressor.predict(x_train_pca)

    # predict on test data
    y_test_predict = multi_regressor.predict(x_test_pca)

    # compare performance
    rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    for ind, cur_name in enumerate(rating_names):
        print('Cur impression: {}'.format(cur_name))
        train_gt = y_train[:, ind]
        train_pred = y_train_predict[:, ind]
        [train_cor, p] = pearsonr(train_gt, train_pred)

        test_gt = y_test[:, ind]
        test_pred = y_test_predict[:, ind]
        [test_cor, p] = pearsonr(test_gt, test_pred)

        print('train cor {}, test cor = {}\n'.format(train_cor, test_cor))

    return


def predict_on_e(feat_layer_file_name='conv53.npy'):
    start_t = time.time()
    # load extracted feature from feature_save_dir. Feature extraction layers include conv51,2,3, maxpool5, fc6, fc7.
    feature_save_dir = '/home/amanda/Documents/vgg_model/e_with_mask_feat'
    feature_file_name = os.path.join(feature_save_dir, feat_layer_file_name)
    feat_arr = np.load(feature_file_name)  # xxxx * 100352, for example.
    print('feature array shape = {}'.format(feat_arr.shape))

    # Load pretrained model (pca model and the corresponding regressor)
    model_load_dir = '/home/amanda/Documents/vgg_model/2k_feat'
    pca_model_name = 'conv53_keep_150_ratio_986_pca_model.pkl'
    regressor_model_name = 'conv53_keep_150_regressor_model.pkl'

    file_name = os.path.join(model_load_dir, pca_model_name)
    with open(file_name, 'rb') as f:
        pca = pickle.load(f)

    x_pca = pca.transform(feat_arr)

    file_name = os.path.join(model_load_dir, regressor_model_name)
    with open(file_name, 'rb') as f:
        regressor = pickle.load(f)

    y_predict = regressor.predict(x_pca)
    np.save(os.path.join(feature_save_dir, 'conv51_prediction.npy'), y_predict)
    print('Prediction saved.')
    print('Done! Elapsed time = {}'.format(time.time() - start_t))


def merge_data_e_pred_funding():
    e_prediction_dir = '/home/amanda/Documents/vgg_model/e_with_mask_feat'

    impression_prediction_file_name = 'conv53_prediction.npy'
    impression_prediction_file_path = os.path.join(e_prediction_dir, impression_prediction_file_name)
    impression_array = np.load(impression_prediction_file_path)

    valid_im_lst_file_name = 'valid_img_lst.npy'
    valid_im_file_path = os.path.join(e_prediction_dir, valid_im_lst_file_name)
    valid_im_lst = np.load(valid_im_file_path)

    im_lst = pd.Series(valid_im_lst)

    e_pred_df = pd.DataFrame()  # put first impression predictions into this dataframe.
    e_pred_df['img_name'] = im_lst

    rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    img_save_root_dir = '/home/amanda/Documents/cropped_face'

    for ind, impression_name in enumerate(rating_names):
        e_pred_df[impression_name] = impression_array[:, ind]
        attribute_dir = os.path.join(img_save_root_dir, impression_name)
        if not os.path.exists(attribute_dir):
            os.makedirs(attribute_dir)

    # cut the post-fix of the image name to get the e-id
    e_pred_df['e_id'] = pd.Series([s[:-7] for s in e_pred_df['img_name'].values], index=e_pred_df.index)


    # create a copy with images with their rating as the file name in different folder.
    # df_len = len(df)
    # img_src_dir = '/home/amanda/Documents/cropped_face/e_with_mask'
    # for ind, row in df.iterrows():
    #     cur_img_name = row['img_name']
    #     src_path = os.path.join(img_src_dir, cur_img_name)
    #     if ind % 100 == 0:
    #         print('{} out of {}'.format(ind+1, df_len))
    #
    #     for cur_attribute in rating_names:
    #         cur_rating = row[cur_attribute]
    #         dst_path = os.path.join(img_save_root_dir, cur_attribute, str(cur_rating)+'.jpg')
    #         copyfile(src_path, dst_path)

    e_fund_data_df = pd.read_pickle('tmp_data/e_df.pkl')
    short_e_fund_df = e_fund_data_df[e_fund_data_df['short_id'].isin(e_pred_df['e_id'])]
    short_e_fund_df = short_e_fund_df.set_index('short_id')
    short_e_fund_df = short_e_fund_df.loc[e_pred_df['e_id']]
    short_e_fund_df['binary'] = (short_e_fund_df['E - VC Raised (log)'] != 0).astype(int)

    e_pred_df['binary'] = short_e_fund_df['binary'].values
    e_pred_df['fund'] = short_e_fund_df['E - VC Raised (log)'].values

    e_pred_df.to_csv('./tmp_data/e_impression_predict.csv')

    return


def analyze_e_pred():
    e_pred = pd.read_csv('./tmp_data/e_impression_predict.csv')
    attribute_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']

    success = e_pred['binary'] == 1
    fail = e_pred['binary'] == 0
    cat_success = e_pred[success]
    cat_fail = e_pred[fail]

    for cur_attri in attribute_names:
        print cur_attri
        print ttest_ind(cat_success[cur_attri], cat_fail[cur_attri])
        print e_pred[cur_attri][success].mean(), e_pred[cur_attri][fail].mean()
        print e_pred[cur_attri][success].std(), e_pred[cur_attri][fail].std()

    return


def merge_gender_api():
    api_data = pd.read_csv('/home/amanda/Github/impression_personality/ent-labels.txt', sep="\t", header=None)
    api_data.columns = ["faceId", "faceTopDimension", "faceLeftDimension", "faceWidthDimension", "faceHeightDimension",
                    "smile", "pitch", "roll", "yaw", "gender", "age", "moustache", "beard", "sideburns", "glasses",
                    "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "blurlevel",
                    "blurvalue", "exposurelevel", "exposurevalue", "noiselevel", "noisevalue", "eymakeup", "lipmakeup",
                    "foreheadoccluded", "eyeoccluded", "mouthoccluded", "hair-bald", "hair-invisible", "img_name", "nan"]

    api_data = api_data.drop(api_data.index[0])
    api_data = api_data[['img_name', 'smile', 'gender', 'age', 'glasses', 'anger', 'contempt', 'disgust', 'fear', 'happiness',
                 'neutral', 'sadness', 'surprise']]
    return



# extract_e_feature()
# extract_2kface_feature()
# split_on_2k()
# pca_and_regression_on_2k()
# predict_on_e()
# merge_data_e_pred_funding()
analyze_e_pred()



