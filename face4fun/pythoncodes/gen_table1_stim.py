import pandas as pd
import random
import os
import glob


# In table 1, we will pick up one face of lowest score and one face of highest score to compose a pair.
# For each trait, we will pick up 100 unique pairs, and for 20 repetitions.

unique_pair_num = 100
repeat_pair_num = 20
trait_lst = ['attractive', 'trustworthy', 'aggressive', 'intelligent']
stim_columns = ['im1_name', 'im2_name', 'pair_type', 'pair_distance', 'pair_order', 'repeat']
gt_im_dir = 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/for_amt/gt/'


def gen_stim():

    rating_df = pd.read_csv('../static/csvs/celeb_ratings.csv')

    for trait in trait_lst:

        stim_df = pd.DataFrame(columns=['im1_name', 'im2_name'])

        sorted_names = rating_df.sort_values(by=[trait])['Filename']

        # pick up the first k and the past k faces, and pair them together.
        stim_df['im1_name'] = sorted_names[:unique_pair_num].values
        stim_df['im2_name'] = sorted_names[-unique_pair_num:].values

        stim_df['pair_type'] = 'gt'

        # for attractive and aggressive traits, the range is about 2-8, the other two are about 3-7
        if trait in ('attractive', 'aggressive'):
            stim_df['pair_distance'] = '2-8'
        elif trait in ('trustworthy', 'intelligent'):
            stim_df['pair_distance'] = '3-7'

        # the starting state is im1_name = low, im2_name = high, later on, we will randomly flip it.
        stim_df['pair_order'] = 'low-high'
        stim_df['repeat'] = 0

        # randomly sample # repetitions, mark these rows onto repeat pairs.
        stim_repeat = stim_df.sample(repeat_pair_num)
        stim_repeat['repeat'] = 1

        stim_df = pd.concat([stim_df, stim_repeat], ignore_index=True)
        print(len(stim_df))

        rand_lst = [random.randint(0, 1) for i in range(0, len(stim_df))]
        stim_df.is_copy = False

        for index, row in stim_df.iterrows():
            im1_name = row['im1_name']
            im2_name = row['im2_name']

            # switch left-right order if the current rand ind is 0
            if rand_lst[index] == 0:
                stim_df['im1_name'] = gt_im_dir + im2_name
                stim_df['im2_name'] = gt_im_dir + im1_name
                stim_df['pair_order'] = 'high-low'
            else:
                stim_df['im1_name'] = gt_im_dir + im1_name
                stim_df['im2_name'] = gt_im_dir + im2_name

        stim_df = stim_df.sample(frac=1, random_state=1)

        # write the df into txt format, list of lists.
        task_lst = []
        for ind, row in stim_df.iterrows():
            task_lst.append(row.values.tolist())

        txt_save_name = '../static/txt/table1/' + trait + '_gt.txt'

        with open(txt_save_name, 'w') as file_handler:
            for item in task_lst:
                file_handler.write('{},\n'.format(item))

    print('Done!')

    return


if __name__ == '__main__':
    gen_stim()
