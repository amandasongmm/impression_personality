import pandas as pd
import numpy as np
import pickle
import os
import shutil
import sys


def gen_txt_file():
    rating_df = pd.read_csv('../static/csvs/celeb_ratings.csv')
    trait = 'attractive'
    n_unique_num = 90

    attract_df = rating_df[['Filename', 'attractive']]
    att_min, att_max = attract_df[trait].min(), attract_df[trait].max()

    anchor_lst = np.linspace(att_min, att_max, num=n_unique_num)
    n_names, n_values = [], []

    for i in range(n_unique_num):
        anchor_value = anchor_lst[i]
        attract_df['distance'] = (attract_df[trait]-anchor_value).abs()
        attract_df = attract_df.sort_values(by=['distance'])

        n_names.append(attract_df.iloc[0]['Filename'])
        n_values.append(attract_df.iloc[0][trait])

    save_attract_df = pd.DataFrame(columns=[['Filename', 'attractive']])
    save_attract_df['Filename'] = n_names
    save_attract_df['Score'] = n_values
    save_attract_df.to_csv('100_attractive_lst.csv')

    with open('gt-100.txt', 'wb') as fp:
        pickle.dump(n_names, fp)

    print('Done')

    return


def copy_select_gt():

    with open('gt-100.txt', 'rb') as fp:
        file_lst = pickle.load(fp)

    dst_dir = '/raid/amanda/gt_100/attractive/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for cur_file in file_lst:
        src_file = '/raid/SAGAN/CelebA/images/' + cur_file
        try:
            shutil.copy(src_file, dst_dir)
        except IOError as e:
            print('unable to copy file. %s' % e)
        except:
            print('Unexpected error:', sys)

    return


if __name__ == '__main__':
    # gen_txt_file()
    copy_select_gt()