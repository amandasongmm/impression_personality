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


def train_val_test_split_and_pca(feat_layer_file_name='conv53.npy'):
    start_t = time.time()
    print('Started...')
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat'

    feat_layer_name = feat_layer_file_name[:-4]
    train_val_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_after_pca_data.npz')

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(model_save_dir, valid_img_file_name))  #

    # load extracted features.
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

    # split on a 64/16/20 ratio. split on a 90/10 ratio.
    x_train, x_test, y_train, y_test = train_test_split(feat_arr, rating_arr, test_size=0.1, random_state=42)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # do PCA on the training set.
    print('Split done. Start pca...')
    pca = PCA(n_components=1000)
    pca.fit(x_train)

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)

    # save the pca model
    pca_save_name = '{}/{}_9_1_split_pca_model.pkl'.format(model_save_dir, feat_layer_name)
    pickle.dump(pca, open(pca_save_name, 'wb'))

    x_train_pca = pca.transform(x_train)
    # x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    # save the split feature data.
    np.savez(train_val_test_split_data_save_name, x_train_pca=x_train_pca, x_test_pca=x_test_pca,
             y_train=y_train, y_test=y_test, cum_ratio=cum_ratio, feature_lst=feature_lst)
    # np.savez(train_val_test_split_data_save_name, x_train_pca=x_train_pca, x_val_pca=x_val_pca, x_test_pca=x_test_pca,
    #          y_train=y_train, y_val=y_val, y_test=y_test, cum_ratio=cum_ratio, feature_lst=feature_lst)
    print('pca done. Split processed data saved...\n Elapsed time = {}'.format(time.time()-start_t))
    return


def hyper_para_tuning(feat_layer_file_name='conv53.npy'):
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat'

    feat_layer_name = feat_layer_file_name[:-4]
    train_val_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_after_pca_data.npz')
    data = np.load(train_val_test_split_data_save_name)
    x_train_pca, x_test_pca = data['x_train_pca'], data['x_test_pca']
    y_train, y_test = data['y_train'], data['y_test']
    cum_ratio, feature_lst = data['cum_ratio'], data['feature_lst']

    # x_train, x_val, x_test = x_train_pca[:, :pca_num], x_val_pca[:, :pca_num], x_test_pca[:, :pca_num]

    # prepare the hyper parameter alpha list.
    # n_alphas = 10
    # alphas = np.logspace(-10, -2, n_alphas)
    # alpha = 1e-8
    pca_nums = np.linspace(180, 280, num=21, dtype=int)

    # use only one dimension to test.
    feat_dim = 1
    print('\n\nFeature = {}'.format(feature_lst[feat_dim]))

    for pca_num in pca_nums:
        # for a in alphas:
        linear_regressor = LinearRegression()
        x_train, x_test = x_train_pca[:, :pca_num], x_test_pca[:, :pca_num]

        linear_regressor.fit(x_train, y_train[:, feat_dim])

        y_train_predict = linear_regressor.predict(x_train)
        # y_val_predict = linear_regressor.predict(x_val)
        y_test_predict = linear_regressor.predict(x_test)

        print('\npca number = {}'.format(pca_num))
        [train_cor, p] = pearsonr(y_train[:, feat_dim], y_train_predict)
        # [val_cor, p] = pearsonr(y_val[:, feat_dim], y_val_predict)
        [test_cor, p] = pearsonr(y_test[:, feat_dim], y_test_predict)
        print('Train, test cor = {:.3f}, {:.3f}'.format(train_cor, test_cor))

    return


# train_val_test_split_and_pca()
hyper_para_tuning()