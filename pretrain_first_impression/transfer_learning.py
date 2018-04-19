import time
import tensorflow as tf
import numpy as np
import skimage
import skimage.io
import skimage.transform
import os
import math
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import time
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS


# This script is adapted based on
# https://github.com/mdietrichstein/vgg16-transfer-learning/blob/master/vgg16_transfer_learning.ipynb

default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]

model_version = 3
model_path = 'models/model-{}/'.format(model_version)
# weights_path = '/home/amanda/Github/vgg_models/vgg16_weights.npz'
# all_im_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/2kfaces/'

weights_path = '/home/amanda/Documents/vgg_dataset/vgg16_weights.npz'
all_im_dir = '/home/amanda/Documents/mit2k/'


def get_batches(x, y, batch_size=32):
    num_rows = y.shape[0]
    num_batches = num_rows // batch_size

    if num_rows % batch_size != 0:
        num_batches = num_batches + 1

    for batch in range(num_batches):
        yield x[batch_size * batch: batch_size * (batch+1)], y[batch_size * batch: batch_size * (batch+1)]

    # this function returns a generator: return a set of values that you will only need to read once.


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


def load_image(image_path, mean=vgg_mean):

    image = skimage.io.imread(image_path)
    resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
    bgr = resized_image[:, :, ::-1] - mean

    return bgr


# extract vgg16 features

def extract_features(image_directory, batch_size=32):
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
    np.save('all_im_fc6.npy', all_im_feature_fc6)
    np.save('all_im_fc7.npy', all_im_feature_fc7)


def train_test_split():
    select_rating_path = 'selected_score.pkl'
    feats = np.load('all_im_fc6.npy')
    df = pd.read_pickle(select_rating_path)

    feat_num = 4096
    rating_impression_num = 6
    feat_arr = np.empty((0, feat_num), dtype=np.float32)
    rating_arr = np.empty((0, rating_impression_num), dtype=np.float32)

    for index, row in df.iterrows():
        file_name = row['Filename']
        if file_name in feats.keys():
            cur_feat = feats[file_name]
            cur_rating = [row['attractive'], row['aggressive'], row['trustworthy'], row['intelligent'], row['sociable'],
                          row['responsible']]
            feat_arr = np.append(feat_arr, np.array([cur_feat]), axis=0)
            rating_arr = np.append(rating_arr, np.array([cur_rating]), axis=0)
        else:
            print('Image {} is not loaded correctly in vgg network. Check what happened.'.format(file_name))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    train_indices, test_indices = next(splitter.split(feat_arr, rating_arr))

    train_feat, train_rating = feat_arr[train_indices], rating_arr[train_indices]
    test_feat, test_rating = feat_arr[test_indices], rating_arr[test_indices]

    return train_feat, train_rating, test_feat, test_rating


def transfer_exp_1():
    """Use a small NN with a single hidden layer"""
    if os.path.exists(model_path):
        raise Exception('Directory "{}" already exists. Delete or move it.'.format(model_path))

    num_epochs = 5
    learning_rate = 0.01
    keep_prob = 0.5
    batch_size = 64
    accuracy_print_steps = 10
    iterations = 0

    tf.reset_default_graph()

    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:

            with tf.name_scopes('inputs'):
                _images = tf.placeholder(tf.float32, shape=(None, 4096), name='images')
                _keep_prob = tf.placeholder(tf.float32, name='keep_probability')

            with tf.name_scope('targets'):
                _ratings = tf.placeholder(tf.float32, shape=(None, 6), name='ratings')
            with tf.name_scope('hidden_layer'):
                hidden_weights = tf.Variable(initial_value=tf.truncated_normal([4096, num_hidden_neurons], mean=0.0,
                                                                               stddev=0.01),
                                             dtype=tf.float32, name='hidden_weights')




    return


if __name__ == '__main__':
    extract_f67_feat()


