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
import matplotlib.pyplot as plt
import seaborn as sns


def train_val_test_split_and_pca(feat_layer_file_name='conv51.npy'):
    start_t = time.time()
    print('Started...')
    data_load_dir = '/home/amanda/Documents/vgg_model/2k_feat/'
    feat_layer_name = feat_layer_file_name[:-4]
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/' + feat_layer_name + '/'

    train_val_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_after_pca_data.npz')

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(data_load_dir, valid_img_file_name))  #

    # load extracted features.
    feature_file_name = os.path.join(data_load_dir, feat_layer_file_name)
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

    # do PCA on the training set.
    print('Split done. Start pca...')
    pca = PCA(n_components=1000)
    pca.fit(x_train)

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)

    # save the pca model
    pca_save_name = '{}/{}_9_1_split_pca_model.pkl'.format(model_save_dir, feat_layer_name)
    pickle.dump(pca, open(pca_save_name, 'wb'))

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # save the split feature data.
    np.savez(train_val_test_split_data_save_name, x_train_pca=x_train_pca, x_test_pca=x_test_pca,
             y_train=y_train, y_test=y_test, cum_ratio=cum_ratio, feature_lst=feature_lst)

    print('pca done. Split processed data saved...\n Elapsed time = {}'.format(time.time()-start_t))
    return


def merge_multi_layers():
    start_t = time.time()
    print('Started merging features from con51,52 and 53...')
    data_load_dir = '/home/amanda/Documents/vgg_model/2k_feat/'
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/conv5_merge/'

    train_val_test_split_data_save_name = os.path.join(model_save_dir, 'conv5_123_train_test_after_pca_data.npz')

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(data_load_dir, valid_img_file_name))  #

    # load extracted features.
    feat_arr_51 = np.load(data_load_dir + 'conv51.npy')  # 2176 * 100352, for example.
    feat_arr_52 = np.load(data_load_dir + 'conv52.npy')
    feat_arr_53 = np.load(data_load_dir + 'conv53.npy')
    feat_merge = np.concatenate((feat_arr_51, feat_arr_52), axis=1)
    feat_merge = np.concatenate((feat_merge, feat_arr_53), axis=1)

    # load selected impression attribute scores
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_arr = filtered_rating_df.as_matrix()

    # split on a 64/16/20 ratio. split on a 90/10 ratio.
    x_train, x_test, y_train, y_test = train_test_split(feat_merge, rating_arr, test_size=0.1, random_state=42)

    # do PCA on the training set.
    print('Split done. Start pca...')
    pca = PCA(n_components=1700)
    pca.fit(x_train)

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)

    # save the pca model
    pca_save_name = '{}merge_pca_model.pkl'.format(model_save_dir)
    pickle.dump(pca, open(pca_save_name, 'wb'))

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # save the split feature data.
    np.savez(train_val_test_split_data_save_name, x_train_pca=x_train_pca, x_test_pca=x_test_pca,
             y_train=y_train, y_test=y_test, cum_ratio=cum_ratio, feature_lst=feature_lst)

    print('pca done. Split processed data saved...\n Elapsed time = {}'.format(time.time() - start_t))
    return


def hyper_para_tuning(feat_layer_file_name='conv51.npy'):
    feat_layer_name = feat_layer_file_name[:-4]
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/' + feat_layer_name + '/'

    train_val_test_split_data_save_name = os.path.join(model_save_dir, feat_layer_name+'_train_test_after_pca_data.npz')
    data = np.load(train_val_test_split_data_save_name)
    x_train_pca, x_test_pca = data['x_train_pca'], data['x_test_pca']
    y_train, y_test = data['y_train'], data['y_test']
    cum_ratio, feature_lst = data['cum_ratio'], data['feature_lst']

    pca_nums = np.linspace(70, 370, num=101, dtype=int)

    # Test one dimension at a time.

    for feat_dim in range(0, 6):

        cur_feat = feature_lst[feat_dim]
        print('\n\nFeature = {}'.format(cur_feat))
        df = pd.DataFrame(columns=['pca number', 'train correlation', 'test correlation'])
        for ind, pca_num in enumerate(pca_nums):
            # for a in alphas:
            linear_regressor = LinearRegression()
            x_train, x_test = x_train_pca[:, :pca_num], x_test_pca[:, :pca_num]

            linear_regressor.fit(x_train, y_train[:, feat_dim])

            y_train_predict = linear_regressor.predict(x_train)
            y_test_predict = linear_regressor.predict(x_test)

            # print('\npca number = {}'.format(pca_num))
            [train_cor, p] = pearsonr(y_train[:, feat_dim], y_train_predict)
            [test_cor, p] = pearsonr(y_test[:, feat_dim], y_test_predict)
            # print('Train, test cor = {:.3f}, {:.3f}'.format(train_cor, test_cor))
            df.loc[ind] = [pca_num, train_cor, test_cor]

        df.to_pickle(model_save_dir+cur_feat+'_performance.pkl')

        # visualize the performance change curves.
        fig = plt.figure()
        fig.suptitle(cur_feat, fontsize=20)
        plt.scatter(df['pca number'], df['train correlation'], label='train')
        plt.scatter(df['pca number'], df['test correlation'], label='test')
        plt.xlabel('PCA number', fontsize=18)
        plt.ylabel('Correlation', fontsize=16)
        plt.grid()
        plt.legend()
        fig.savefig(model_save_dir + cur_feat + '.jpg')

        #
        # df.sort_values('test correlation', ascending=False)
    return


def hyper_para_tuning_merge():
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/conv5_merge/'

    train_val_test_split_data_save_name = os.path.join(model_save_dir, 'conv5_123_train_test_after_pca_data.npz')
    data = np.load(train_val_test_split_data_save_name)
    x_train_pca, x_test_pca = data['x_train_pca'], data['x_test_pca']
    y_train, y_test = data['y_train'], data['y_test']
    cum_ratio, feature_lst = data['cum_ratio'], data['feature_lst']

    pca_nums = np.linspace(70, 370, num=101, dtype=int)

    # Test one dimension at a time.

    for feat_dim in range(0, 6):

        cur_feat = feature_lst[feat_dim]
        print('\n\nFeature = {}'.format(cur_feat))
        df = pd.DataFrame(columns=['pca number', 'train correlation', 'test correlation'])
        for ind, pca_num in enumerate(pca_nums):
            # for a in alphas:
            linear_regressor = LinearRegression()
            x_train, x_test = x_train_pca[:, :pca_num], x_test_pca[:, :pca_num]

            linear_regressor.fit(x_train, y_train[:, feat_dim])

            y_train_predict = linear_regressor.predict(x_train)
            y_test_predict = linear_regressor.predict(x_test)

            # print('\npca number = {}'.format(pca_num))
            [train_cor, p] = pearsonr(y_train[:, feat_dim], y_train_predict)
            [test_cor, p] = pearsonr(y_test[:, feat_dim], y_test_predict)
            # print('Train, test cor = {:.3f}, {:.3f}'.format(train_cor, test_cor))
            df.loc[ind] = [pca_num, train_cor, test_cor]

        df.to_pickle(model_save_dir + cur_feat + '_performance.pkl')

        # visualize the performance change curves.
        fig = plt.figure()
        fig.suptitle(cur_feat, fontsize=20)
        plt.scatter(df['pca number'], df['train correlation'], label='train')
        plt.scatter(df['pca number'], df['test correlation'], label='test')
        plt.xlabel('PCA number', fontsize=18)
        plt.ylabel('Correlation', fontsize=16)
        plt.grid()
        plt.legend()
        fig.savefig(model_save_dir + cur_feat + '.jpg')

    return


def use_full_data_to_pca(feat_layer_file_name):
    start_t = time.time()
    print('Started...')
    data_load_dir = '/home/amanda/Documents/vgg_model/2k_feat/'
    feat_layer_name = feat_layer_file_name[:-4]
    print(feat_layer_name)
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/full_data_model/'

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(data_load_dir, valid_img_file_name))  #

    # load extracted features.
    feature_file_name = os.path.join(data_load_dir, feat_layer_file_name)
    feat_arr = np.load(feature_file_name)  # 2176 * 100352, for example.

    # load selected impression attribute scores
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    rating_arr = filtered_rating_df.as_matrix()

    # do PCA on the training set.
    print('Start pca...')
    pca = PCA(n_components=500)
    pca.fit(feat_arr)

    cum_ratio = np.cumsum(pca.explained_variance_ratio_)

    # save the pca model
    pca_save_name = '{}/{}_full_data_pca_model.pkl'.format(model_save_dir, feat_layer_name)
    pickle.dump(pca, open(pca_save_name, 'wb'))

    x_pca = pca.transform(feat_arr)

    # save the processed feature data.
    processed_data_name = model_save_dir + feat_layer_name + '_full_processed_data.npz'
    np.savez(processed_data_name, x_pca=x_pca, rating_arr=rating_arr, cum_ratio=cum_ratio, feature_lst=feature_lst)

    print('PCA done. Processed data saved...\n Elapsed time = {}'.format(time.time()-start_t))
    return


def make_single_prediction_on_2k(feat, pca_num, conv_name):
    # conv_name = 'conv53', 'conv52', or 'conv51'
    # make predictions for trustworthy, intelligent, and attractive.
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/full_data_model/'
    data = np.load(model_save_dir + conv_name + '_full_processed_data.npz')
    x_pca = data['x_pca']
    rating_arr = data['rating_arr']
    feature_lst = data['feature_lst'].tolist()
    feat_ind = feature_lst.index(feat)

    y = rating_arr[:, feat_ind]
    x = x_pca[:, :pca_num]
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)

    # save the regressor model
    regressor_save_name = '{}/{}_keep_{}_regressor_model.pkl'.format(model_save_dir, feat, str(pca_num))
    # regressor_save_name = model_save_dir + '/' + feat_layer_name + '_regressor_model.pkl'
    pickle.dump(linear_regressor, open(regressor_save_name, 'wb'))

    y_pred = linear_regressor.predict(x)

    [train_cor, p] = pearsonr(y, y_pred)
    print('Train cor, p = {:.3f}, {:.3f}'.format(train_cor, p))

    fig = plt.figure()

    title_str = '{}:{:.3f}'.format(feat, train_cor)
    fig.suptitle(title_str, fontsize=20)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel('Human', fontsize=16)
    plt.ylabel('Model', fontsize=16)
    plt.grid()
    plt.legend()
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.plot([1, 9], [1, 9], linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(model_save_dir+feat+'.jpg')
    return


def visualize_heatmap():

    start_t = time.time()
    print('Started...')
    data_load_dir = '/home/amanda/Documents/vgg_model/2k_feat/'

    # load valid img lst
    valid_img_file_name = 'valid_img_lst.npy'
    valid_img_lst = np.load(os.path.join(data_load_dir, valid_img_file_name))  #

    # load selected impression attribute scores
    select_rating_path = 'tmp_data/selected_score.pkl'
    rating_df = pd.read_pickle(select_rating_path)

    filtered_rating_df = rating_df[rating_df['Filename'].isin(valid_img_lst)]
    filtered_rating_df = filtered_rating_df.set_index('Filename')
    filtered_rating_df = filtered_rating_df.loc[valid_img_lst]

    f, ax = plt.subplots(figsize=(10, 8))
    corr = filtered_rating_df.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #             square=True, ax=ax)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, vmin=-1, vmax=1, annot=True)
    plt.savefig('/home/amanda/Documents/vgg_model/2k_feat/may_7/heat_map_of_impression_ratings.jpg')
    return


def predict_e_new_one(feat, pca_num, conv_name):
    # df_name = './tmp_data/merged_api_impression.csv'
    # df = pd.read_csv(df_name)
    # im_names = df['img_name'].values

    feature_save_dir = '/home/amanda/Documents/vgg_model/e_with_mask_feat'
    feature_file_name = os.path.join(feature_save_dir, conv_name+'.npy')
    feat_arr = np.load(feature_file_name)  # xxxx * 100352, for example.

    # load pre-trained model. (pca and the corresponding regressor)
    model_save_dir = '/home/amanda/Documents/vgg_model/2k_feat/may_7/full_data_model/'
    regressor_save_name = '{}/{}_keep_{}_regressor_model.pkl'.format(model_save_dir, feat, str(pca_num))

    with open(regressor_save_name, 'rb') as f:
        regressor = pickle.load(f)

    y_pred = regressor.predict(feat_arr[:, :pca_num])

    return y_pred


def predict_e_new_all():
    start_t = time.time()
    feature_lst = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable']
    pca_nums = [154, 122, 173, 361, 301, 320]
    conv_names = ['conv53', 'conv53', 'conv53', 'conv51', 'conv51', 'conv52']

    df = pd.DataFrame(columns=feature_lst)

    for cur_feat, cur_pca_num, cur_conv_name in zip(feature_lst, pca_nums, conv_names):
        print('Predicting feature {}'.format(cur_feat))
        y_pred = predict_e_new_one(cur_feat, cur_pca_num, cur_conv_name)
        df[cur_feat] = y_pred

    e_prediction_dir = '/home/amanda/Documents/vgg_model/e_with_mask_feat'
    valid_im_lst_file_name = 'valid_img_lst.npy'
    valid_im_file_path = os.path.join(e_prediction_dir, valid_im_lst_file_name)
    valid_im_lst = np.load(valid_im_file_path)

    im_lst = pd.Series(valid_im_lst)

    df['img_name'] = im_lst

    # only save a subset of images.
    df_name = './tmp_data/merged_api_impression.csv'
    big_df = pd.read_csv(df_name)
    im_names = big_df['img_name'].values

    short_df = df[df['img_name'].isin(im_names)]
    short_df = short_df.set_index('img_name')
    short_df = short_df.loc[im_names]

    # save
    short_df.to_pickle('./tmp_data/impression_only.pkl')
    print('Elapsed time = {}'.format(time.time()-start_t))
    return


def mv_face(gender_match_word, other_criteria_word, other_criteria_value, save_dir_key_word):
    df_name = './tmp_data/merged_api_impression.csv'
    df = pd.read_csv(df_name)

    target_gender_df = df[df['gender'] == gender_match_word]
    additional_criteria_df = target_gender_df[target_gender_df[other_criteria_word] == other_criteria_value]

    # Amanda Mac
    im_root_dir = '/Users/amanda/Documents/E_faces/common_lst/'
    im_dst_dir = '/Users/amanda/Documents/E_faces/gender_viz/'

    # Amanda home pc
    # im_root_dir = '/home/amanda/Documents/cropped_face/e_with_mask/'
    # im_dst_dir = '/home/amanda/Documents/gender_viz/'

    print save_dir_key_word
    dst_dir = os.path.join(im_dst_dir, save_dir_key_word)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for im in additional_criteria_df['img_name'].values:
        copyfile(im_root_dir + im, os.path.join(dst_dir, im))

    return


# train_val_test_split_and_pca()
# hyper_para_tuning()

# merge_multi_layers()
# hyper_para_tuning_merge()

# use_full_data_to_pca('conv51.npy')
# use_full_data_to_pca('conv52.npy')
# use_full_data_to_pca('conv53.npy')

# make_single_prediction_on_2k(feat='trustworthy', pca_num=154, conv_name='conv53')
# make_single_prediction_on_2k(feat='intelligent', pca_num=122, conv_name='conv53')
# make_single_prediction_on_2k(feat='attractive', pca_num=173, conv_name='conv53')
# make_single_prediction_on_2k(feat='aggressive', pca_num=361, conv_name='conv51')
# make_single_prediction_on_2k(feat='responsible', pca_num=301, conv_name='conv51')
# make_single_prediction_on_2k(feat='sociable', pca_num=320, conv_name='conv52')

# predict_e_new_all()
mv_face(gender_match_word='female',
        other_criteria_word='IPO',
        other_criteria_value=1,
        save_dir_key_word='female-IPO-success')

