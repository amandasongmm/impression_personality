import pandas as pd
import cv2
import numpy as np
import time
import os.path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import face_recognition
import time
import os
import tensorflow as tf
import inception_resnet_v1


#todo: Work on the face detection codes from the face_recognition repo.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
image = face_recognition.load_image_file('../../Age-Gender-Estimate-TF/demo/demo2.jpg')
model_path = '../../Age-Gender-Estimate-TF/models'
start = time.time()
face_locations = face_recognition.face_locations(image)
print(time.time()-start)
print(face_locations)

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()

    # face_image = face_image.resize((160, 160), Image.NEAREST)
    face_image = tf.image.resize_images(face_image, [160, 160])
    with tf.Graph().as_default():
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass

        print sess.run([age, gender], feed_dict={images_pl: face_image, train_mode: False})









# load model and weights for the age and gender estimation network.
img_size = 64
depth = 16
k = 8
weight_file = 'weights.18-4.06.hdf5'
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

# prepare faces.
faces = []
results = model.predict(faces)
predicted_genders = results[0]

gender_probability = predicted_genders[:, 1]  # the likelihood of a photo being a male. # 0 - female, 1 - male
predicted_genders = [int(round(i)) for i in gender_probability]

ages = np.arange(0, 101).reshape(101, 1)
predicted_ages = results[1].dot(ages).flatten()
