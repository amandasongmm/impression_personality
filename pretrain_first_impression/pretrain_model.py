import pandas as pd
import numpy as np
import time
import tensorflow as tf
import cv2
import os
from tensorflow.python.platform import gfile

author = 'amanda'
# part of the codes are based on
#  https://github.com/burliEnterprises/tensorflow-image-classifier/blob/master/retrain.py
# and https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/get_started/regression/dnn_regression.py


# select only the attractiveness score from the full first impression data.
select_save_path = 'attract_score.pkl'
data_dir = '/home/amanda/Github/attractiveness_datamining/MIT2kFaceDataset/'


def load_from_full_data(attr_lst):
    # The first impression dataset comes from MIT, details can be found here:
    # https://github.com/amandasongmm/attractiveness_datamining

    impression_score_path = data_dir + 'Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'

    # load impression scores
    df_full = pd.read_excel(open(impression_score_path, 'rb'), sheet_name='Final Values')
    keep_lst = 'Filename' + attr_lst
    df_select = df_full[keep_lst]
    # df_select = df_full[['Filename', attr]]
    df_select.to_pickle(select_save_path)
    return


def read_data():
    # read the impression scores
    df = pd.read_pickle(select_save_path)
    impression_scores = df['attractive']
    im_lst = df['Filename']
    im_num = len(im_lst)

    # Read the image data
    face_folder_path = data_dir + '2kfaces/'
    tf_im_w = 224
    tf_im_h = 224

    for cur_im_name in im_lst[0:10]:
        cur_im_path = face_folder_path + cur_im_name
        cur_im = cv2.imread(cur_im_path)
        resized_im = cv2.resize(cur_im, (tf_im_h, tf_im_w), interpolation=cv2.INTER_AREA)

    print('Done')


def create_image_lists(img_dir, test_ratio, validation_ratio):
    """Build a list of training images from the files"""


    return


def retrain():
    # We take Inception v3 architecture model trained on ImageNet datasets, and train a new top layer that can estimate
    # the attractiveness score of the image.


    return


def create_inception_graph(FLAGS):
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    jpeg_data_tensor_name = 'DecodeJpeg/contents:0'
    resized_input_tensor_name = 'ResizeBilinear:0'
    bottleneck_tensor_size = 2048
    model_input_width = 224
    model_input_height = 224
    model_input_depth = 3
    """Creates a graph from saved graphDef file and returns a graph object.
    Returns: graph holding the trained inception network, and various tensors we'll be manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[bottleneck_tensor_name, jpeg_data_tensor_name,
                                                                     resized_input_tensor_name])
        )
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the bottleneck summary layers
    Args:
        sess: current active tf session
        image_data: string of raw jpeg data
        image_data_sensor: input data layer in the graph
        bottleneck_tensor: layer before the final softmax
    Returns:
        numpy array of bottleneck values.
        """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def variable_summaries(var):
    """Tensorboard visualization"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(final_tensor_name, bottleneck_tensor, bottleneck_tensor_size, attribute_count):
    """We need to retrain the top layer to return regression scores for the faces for multi-class attributes
    so this function adds the right operation to the graph, along with some variables to hold the weights, and then
    set up all the gradients for the backward pass.

    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder'
        )
        ground_truth_input = tf.placeholder(tf.float32, [None, attribute_count], name='GroundTruthInput')

    # Organizing the following ops as 'final_training_ops' so they're easier to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, attribute_count], stddev=0.001)
            layer_weights = tf.variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([attribute_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('final_activation', logits)














if __name__ == '__main__':
    read_data()
