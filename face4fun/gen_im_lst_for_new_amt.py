import pandas as pd
import random
import os
import glob

# generate a list of [low, high] * same id, [low, high] * different id, [low, high] * ground truth
# get a set of the unique id list.


def gen_four_level_im_df(tar_dir='../for_amt/aggressive/'):
    # When new modifAE is ready, we used it to generate 4 levels of the same image, in 4 trait dimensions.
    # For each dimension, we use the same set of original images (identities). And they are of the same img_name,
    # but are under different trait/ folders.
    # the naming format is xxxx(id)_-0.75, xxxx(id)_-0.25, xxxx(id)_0.25,xxxx(id)_0.75,
    # (corresponding to original scores 2, 4, 6, 8)

    # This code outputs an ordered_df, with 4 columns [level 1, level 2, level 3, level 4]
    # and each row contains the same id files, but sorted in the 4 levels.
    # e.g. 136220_-0.75.png	136220_-0.25.png	136220_0.25.png	136220_0.75.png

    ordered_im_df_path = '../for_amt/ordered_im.csv'
    if os.path.isfile(ordered_im_df_path):
        ordered_im_df = pd.read_csv(ordered_im_df_path, header=None)
    else:
        img_lst = glob.glob(tar_dir + '*.png')

        id_lst = []
        for i in img_lst:
            id_lst.append(os.path.basename(i).split("_")[0])

        ordered_im_df = pd.DataFrame(columns=['level 1', 'level 2', 'level 3', 'level 4'],
                                     index=list(set(id_lst)))

        for index, row in ordered_im_df.iterrows():
            row['level 1'] = index + '_-0.75.png'
            row['level 2'] = index + '_-0.25.png'
            row['level 3'] = index + '_0.25.png'
            row['level 4'] = index + '_0.75.png'

        ordered_im_df.to_csv('../for_amt/ordered_im.csv')

    return ordered_im_df


def gen_ground_truth_pairs():
    # Since the modifAE images will be compared at 4 levels, we pick up ground truth (unmodified images)
    # that can make similar pairs (one of high score, one of low score, the difference in scores is near 4)

    # flag, gap, num_pts, trait
    df = pd.read_csv('./static/csvs/celeb_ratings.csv')  # the trait scores predicted by the model for each trait.
    gap = 4  # the difference between high and low score pairs of photos.
    num_pts = 100  # find 100 pairs for each starting position (2 --> 6, 4 --> 8)

    for trait in ['attractive', 'trustworthy', 'aggressive', 'intelligent']:  # iterate over 4 traits

        for flag in ['level-1-3', 'level-2-4']:  # iterate over two starting score, 2 --> 6, and 4 --> 8

            print(trait, flag, num_pts)

            temp_df = df[['Filename', trait]]

            if flag == 'level-1-3':
                select_df = df[['Filename', trait]].sort_values(by=[trait])[:num_pts]  # select the lowest N faces
            else:
                select_df = df[['Filename', trait]].sort_values(by=[trait])[-num_pts:]  # select the highest N faces

            pair_values = []
            pair_names = []

            for i in range(num_pts):
                cur_value = select_df.iloc[i][trait]
                temp_df['distance'] = ((temp_df[trait]-cur_value).abs() - gap).abs()
                # for each face, find a face that is closest to 4 pts away from it

                temp_df = temp_df.sort_values(by=['distance'])

                pair_values.append(temp_df.iloc[0][trait])
                pair_names.append(temp_df.iloc[0]['Filename'])

                temp_df.drop(temp_df.index[0], inplace=True)  # remove the selected faces so it won't appear twice.

            if flag == 'level-1-3':  # for level 1-3, we choose a low-value-face first, then find its high value pair
                select_df['high name'] = pair_names
                select_df['high value'] = pair_values

            elif flag == 'level-2-4':  # for level 2-4, it's the opposite.
                select_df['low name'] = pair_names
                select_df['low value'] = pair_values

            if flag == 'level-1-3':
                select_df['low name'] = select_df['Filename']
                select_df['low value'] = select_df[trait]

            elif flag == 'level-2-4':
                select_df['high name'] = select_df['Filename']
                select_df['high value'] = select_df[trait]

            select_df['dif'] = select_df['high value'] - select_df['low value']

            select_df.to_csv('../for_amt/temp_img_lst/gt/'+trait+'-'+flag+'.csv', index=False)


# Now we will mix ground-truth pairs and modified face pairs to make the final version of the task.


def gen_gt_only_task_lst():

    # if the trait is aggressive or attractive, then we choose 50 pairs from the level-1-3, and 50 pairs from level-2-4
    # if the trait is intelligent or trustworthy, then we directly choose 100 pairs from level 2-4 caz 1-3 is identical

    # one thing to note is that we want to make another field indicating whether the pair is from 1-3, or 2-4,
    # as a reference later on potential use.

    # in total there are 4 fields that we will use.
    # 1. img1_file_path,
    # 2. img2_file_path,
    # 3. pair type ('gt' or 'modifae' or 'star')
    # 4. pair distance-type
    #    modifae pairs: 'score 2-6' or 'score 4-8'
    #    modifae gts: 'score 2-6' or 'score 4-8' or 'score 3-7'
    #    starGAN pairs: 'level low-high'
    #    starGAN gts: 'level low-high'
    # 5: pair low-high order: 'low-high', 'high-low'
    # 6: repeat pairs: 1 or 0.

    trait_lst = ['aggressive', 'attractive', 'trustworthy', 'intelligent']

    for trait in trait_lst:
        #
        #
        # if trait == 'attractive' or trait == 'aggressive':
        if trait in {'attractive', 'aggressive'}:
            print 'right. cur trait = {}'.format(trait)
            # get 50 pairs from level-1-3, and level-2-4, respectively.
            # read level-1-3 csv file.
            level13_df = pd.read_csv('../for_amt/temp_img_lst/gt/'+trait+'-'+'level-1-3.csv')
            level24_df = pd.read_csv('../for_amt/temp_img_lst/gt/'+trait+'-'+'level-2-4.csv')

            stim_first_half = pd.DataFrame(columns=[['im1_name', 'im2_name', 'pair_type', 'pair_distance_type']])
            stim_first_half['im1_name'] = level13_df['low name'][:50]
            stim_first_half['im2_name'] = level13_df['high name'][:50]
            stim_first_half['pair_type'] = 'gt'
            stim_first_half['pair_distance_type'] = 'score 2-6'

            stim_second_half = pd.DataFrame(columns=[['im1_name', 'im2_name', 'pair_type', 'pair_distance_type']])
            stim_second_half['im1_name'] = level24_df['low name'][:50]
            stim_second_half['im2_name'] = level24_df['high name'][:50]
            stim_second_half['pair_type'] = 'gt'
            stim_second_half['pair_distance_type'] = 'score 4-8'

            stim_df = pd.concat([stim_first_half, stim_second_half])

        else:  # trait == ('intelligent' or 'trustworthy'):
            print 'wrong. cur trait = {}'.format(trait)
            # get 100 pairs from level-2-4 directly.
            level24_df = pd.read_csv('../for_amt/temp_img_lst/gt/' + trait + '-' + 'level-2-4.csv')
            stim_df = pd.DataFrame(columns=[['im1_name', 'im2_name', 'pair_type', 'pair_distance_type']])
            stim_df['im1_name'] = level24_df['low name'][:100]
            stim_df['im2_name'] = level24_df['high name'][:100]
            stim_df['pair_type'] = 'gt'
            stim_df['pair_distance_type'] = 'score 3-7'

        stim_df['pair_order'] = 'low-high'
        stim_df['repeat'] = 0

        stim_df.to_csv('../for_amt/temp_img_lst/gt_stim_lst/'+trait+'_stim_raw.csv', index=False)

    return


def shuffle_and_repeat_gt_only_lst():

    gt_im_dir = 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/for_amt/gt/'
    trait_lst = ['aggressive', 'attractive', 'trustworthy', 'intelligent']
    for trait in trait_lst:
        print trait
        df = pd.read_csv('../for_amt/temp_img_lst/gt_stim_lst/'+trait+'_stim_raw.csv')
        repeat_num = 10
        df_repeat = df.sample(repeat_num)
        df_repeat['repeat'] = 1
        merged_df = pd.concat([df, df_repeat], ignore_index=True)

        rand_lst = [random.randint(0, 1) for i in range(0, len(merged_df))]

        # randomize the left-right order of the pair.
        merged_df.is_copy = False
        for index, row in merged_df.iterrows():
            im1_name = row['im1_name']
            im2_name = row['im2_name']

            if rand_lst[index] == 0:
                # switch left-right order.
                merged_df['im1_name'] = gt_im_dir + im2_name
                merged_df['im2_name'] = gt_im_dir + im1_name
                merged_df['pair_order'] = 'high-low'
            else:
                merged_df['im1_name'] = gt_im_dir + im1_name
                merged_df['im2_name'] = gt_im_dir + im2_name

        # shuffle the row order.
        merged_df = merged_df.sample(frac=1, random_state=1)

        # write the df into txt format. list of lists.
        task_lst = []
        for ind, row in merged_df.iterrows():
            task_lst.append(row.values.tolist())

        txt_save_name = '../for_amt/temp_img_lst/gt_txt_lst/' + trait + '.txt'

        with open(txt_save_name, 'w') as file_handler:
            for item in task_lst:
                file_handler.write("{},\n".format(item))
    print 'Done!'
    return


if __name__ == '__main__':
    # gen_im_lst()
    # gen_ground_truth_pairs()
    # gen_gt_only_task_lst()
    shuffle_and_repeat_gt_only_lst()
