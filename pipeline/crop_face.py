import skimage.io
import skimage.transform
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


raw_data_dir = '/home/amanda/Desktop/crunch_base/0/'
crop_with_mask_dir = '/home/amanda/Documents/cropped_faces/crunchbase_new/0/'
end_with_key_word = 'g'


default_device = '/gpu:0'
num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
weights_path = '/home/amanda/Documents/vgg_model/vgg16_weights.npz'
batch_size = 100
feat_save_dir = '/home/amanda/Documents/feat_folder/'
model_save_dir = '/home/amanda/Documents/vgg_model/2k_summer_analysis/'


def crop_faces(raw_data_dir, crop_with_mask_dir, end_with_key_word):
    '''
        crop_faces() requires two arguments: raw_data_dir and crop_with_mask_dir

        raw_data_dir is the root dir that you want to run the analysis on. e.g.
        '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/' will search for all the *.jpg
        files under the subfolder of vc recursively. namely, all the vc images.

        crop_with_mask_dir is where you want to save the cropped faces with an oval mask.

        Example: raw_data_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
        crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask/'
    '''

    def fetch_file(path_to_folder, flag, key_word):
        '''
        fetch_files() requires three arguments: pathToFolder, flag and keyWord

        flag must be 'start_with' or 'end_with'
        keyWord is a string to search the file's name

        Be careful, the keyWord is case sensitive and must be exact

        Example: fetch_file('/Documents/Photos/','end_with','.jpg')

        returns: _pathToFiles and _fileNames
        '''

        _path_to_files = []
        _file_names = []

        for dir_path, dir_names, file_names in os.walk(path_to_folder):
            if flag == 'end_with':
                raw_file = [item for item in file_names if item.endswith(key_word)]
                selected_file = [item for item in raw_file if not item.startswith('._')]
                selected_file = list(set(selected_file))
                _file_names.extend(selected_file)

                selected_path = [os.path.join(dir_path, item) for item in selected_file]
                _path_to_files.extend(selected_path)

            elif flag == 'start_with':
                raw_file = [item for item in file_names if item.startswith(key_word)]
                selected_file = [item[2:] if item.startswith('._') else item for item in raw_file]
                selected_file = list(set(selected_file))
                _file_names.extend(selected_file)

                selected_path = [os.path.join(dir_path, item) for item in selected_file]
                _path_to_files.extend(selected_path)

            else:
                print fetch_file.__doc__
                break

            # try to remove empty entries if none of the required files are in the directory
            try:
                _path_to_files.remove('')
                _file_names.remove('')
            except ValueError:
                pass

            # warn if nothing was found in the given path
            if selected_file == []:
                print('No files with given parameters were found in {}\n'.format(dir_path))

        print('{} files are found in searched folders.'.format(len(_file_names)))
        return _path_to_files, _file_names

    path_to_files, file_names = fetch_file(raw_data_dir, 'end_with', end_with_key_word)  # get all the files in the sub directories

    total_file_len = len(file_names)

    if not os.path.exists(crop_with_mask_dir):
        os.makedirs(crop_with_mask_dir)

    face_cascade = cv2.CascadeClassifier('../Cleaning/haarcascade_frontalface_default.xml')
    count = 0
    for path, file_name in zip(path_to_files, file_names):
        img = cv2.imread(path)
        if img is not None:
            height, width, channel = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    if w * h > 900 and int(y - 0.1 * h) > 0 and int(y + 1.1 * h) < height and int(x - 0.05 * w) > 0 \
                            and int(x + 1.05 * w) < width:
                        count += 1
                        roi_color = img[int(y - 0.1 * h):int(y + 1.1 * h), int(x - 0.05 * w):int(x + 1.05 * w), :]

                        # add mask and save masked image.
                        mask = np.zeros_like(roi_color)
                        rows, cols, _ = mask.shape

                        # create a black filled ellipse
                        cen_x = int(rows // 2)
                        cen_y = int(cols // 2)
                        cv2.ellipse(mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0,
                                    endAngle=360.0, color=(255, 255, 255), thickness=-1)
                        # bitwise
                        result = np.bitwise_and(roi_color, mask)

                        # flipped mask
                        new_mask = 255 * np.ones_like(roi_color)
                        cv2.ellipse(new_mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0,
                                    endAngle=360,
                                    color=(0, 0, 0), thickness=-1)

                        # final result
                        with_mask = result + new_mask
                        cv2.imwrite(crop_with_mask_dir + file_name, with_mask)

                        if count % 100 == 0:
                            print count, total_file_len
    # libpng warning: iCCP: known incorrect sRGB profile
    # libpng warning: iCCP: profile 'icc': 1000000h: invalid rendering intent
    return


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


def predict_features(img_dir, feat_save_dir, feat_prefix, batch_size):
    def load_image(image_path, mean=vgg_mean):
        image = skimage.io.imread(image_path)
        resized_image = skimage.transform.resize(image, (224, 224), mode='constant')
        bgr = resized_image[:, :, ::-1] - mean
        return bgr

    file_names = [f for f in os.listdir(img_dir)]
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

            np.save(os.path.join(feat_save_dir, feat_prefix + '_valid_img_lst.npy'), valid_file_lst)

    # load pretrained models.
    layer_name = 'conv52'
    pca_num = 200
    rating_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    pca_save_name = '{}/{}_predict_mode_pca_{}_model.pkl'.format(feat_save_dir, layer_name, str(pca_num))
    regressor_save_name = '{}/2k_final_model_{}_keep_{}_regressor.pkl'.format(feat_save_dir, layer_name, str(pca_num))

    print('Load saved models')
    with open(pca_save_name, 'rb') as f:
        pca = pickle.load(f)

    with open(regressor_save_name, 'rb') as f:
        multi_regressor = pickle.load(f)

    # Transform extracted features.
    feat_after_pca = pca.transform(conv52_arr)
    new_prediction = multi_regressor.predict(feat_after_pca)  # Now make predictions on the VC dataset.

    valid_img_lst = np.load(feat_save_dir + feat_prefix + '_valid_img_lst.npy')

    prediction_df = pd.DataFrame(columns=[rating_names], index=valid_img_lst, data=new_prediction)
    prediction_df.to_pickle(feat_save_dir+'impression_prediction_df.pkl')
    return


def main():
    crop_faces(raw_data_dir='/home/amanda/Desktop/crunch_base/0/',
               crop_with_mask_dir=crop_with_mask_dir,
               end_with_key_word='g')  # end_with_key_word = 'jpg' or 'jpeg' or 'png'

    predict_features(img_dir=crop_with_mask_dir,
                     feat_save_dir=feat_save_dir,
                     feat_prefix='impression_prediction',
                     batch_size=batch_size)

    return
