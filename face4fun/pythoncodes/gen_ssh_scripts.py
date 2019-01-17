import pandas as pd
import numpy as np
import shutil
import sys
import os
import pickle


def gen_script_txt():

    script = 'scp -r amanda@neil.ucsd.edu:/raid/SAGAN/CelebA/images/{'


    for trait in ['attractive', 'trustworthy', 'aggressive', 'intelligent']:
        for flag in ['level-1-3', 'level-2-4']:
            file_path = '../../for_amt/temp_img_lst/gt/'+trait+'-'+flag+'.csv'
            df = pd.read_csv(file_path)

            lst1 = df['high name'].values
            lst2 = df['low name'].values
            lst = np.concatenate((lst1, lst2))

            for i in lst:
                script += i+','

    script += '} '
    dst_path = '/Users/amanda/Github/impression_personality/for_amt/gt/'
    script += dst_path

    txt_file = open('test_script.txt', 'w')
    txt_file.write(script)
    txt_file.close()


def choose_selective_ground_truth():

    file_lst = []

    for trait in ['attractive', 'trustworthy', 'aggressive', 'intelligent']:
        for flag in ['level-1-3', 'level-2-4']:
            file_path = '../../for_amt/temp_img_lst/gt/' + trait + '-' + flag + '.csv'
            df = pd.read_csv(file_path)

            lst1 = df['high name'].values
            lst2 = df['low name'].values
            lst = np.concatenate((lst1, lst2))
            file_lst = np.concatenate((file_lst, lst))
            print(len(file_lst))

    file_lst = list(set(file_lst))
    print(len(file_lst))

    with open('set-gt.txt', 'wb') as fp:
        pickle.dump(file_lst, fp)


def copy_selected_gt():
    with open('set-gt.txt', 'rb') as fp:
        file_lst = pickle.load(fp)

    dst_dir = '/raid/amanda/new_gt_4_features/'
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
    choose_selective_ground_truth()
    # copy_selected_gt()





