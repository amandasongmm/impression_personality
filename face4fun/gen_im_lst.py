import pandas as pd
import random
pd.options.mode.chained_assignment = None  # default='warn'

# trustworthy_gen.csv: [less_trustworthy, more_trustworthy]
# trustworthy_groundtruth.csv: [trustworthy_max, trustworthy_min]
# aggressive_gen.csv: [less_aggressive, more_aggressive]
# aggressive_groundtruth.csv: [aggressive_max, aggressive_min]


# generate list for trustworthy only.


def gen_attr_lst(attr, stargan):
    if stargan:
        im_pub_dir_prefix = 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_stargan_'
    else:
        im_pub_dir_prefix = 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_'

    im_pub_dir = im_pub_dir_prefix + attr + '/'
    if stargan:
        generated_img_lst = '../static/csv/stargan_' + attr + '_gen.csv'
        real_img_lst = '../static/csv/stargan_' + attr + '_ground_truth.csv'
    else:
        generated_img_lst = '../static/csv/'+attr+'_gen.csv'
        real_img_lst = '../static/csv/'+attr+'_ground_truth.csv'

    gen_im_df = pd.read_csv(generated_img_lst)
    real_im_df = pd.read_csv(real_img_lst)

    gen_im_df.columns = ['im1_name', 'im2_name', 'Task type']
    gen_im_df['im1'] = '0'
    gen_im_df['im2'] = '1'
    gen_im_df['pair_ind'] = gen_im_df.index

    random.seed(3)
    rep_total = 10
    gen_orig_len = len(gen_im_df)
    gen_new_len = gen_orig_len + rep_total

    # add some repetition to the generated images.
    rep_lst = random.sample(range(0, len(gen_im_df)), rep_total)

    for count, ind in enumerate(rep_lst):
        gen_im_df.loc[gen_orig_len + count] = gen_im_df.iloc[ind].values

    gen_im_df['rep'] = 0
    gen_im_df['rep'].iloc[gen_orig_len:gen_new_len] = 1

    # the original order of the real_im_df.
    real_im_df.columns = ['more', 'less']
    # change the order of the columns.
    real_im_df = real_im_df[['less', 'more']]
    real_im_df.columns = ['im1_name', 'im2_name']

    # add the same additional labels.
    real_im_df['Task type'] = 'ground_truth'
    real_im_df['im1'] = '0'
    real_im_df['im2'] = '1'
    real_im_df['pair_ind'] = real_im_df.index + gen_orig_len
    real_im_df['rep'] = 0

    # merge two data frames
    merge_df = real_im_df.append(gen_im_df, ignore_index=True)
    rand_lst = [random.randint(0, 1) for i in range(0, len(merge_df))]

    # change the left-right order of the pair.
    merge_df.is_copy = False
    for index, row in merge_df.iterrows():
        im1_name = row['im1_name']
        im2_name = row['im2_name']

        if rand_lst[index] == 0:
            merge_df['im1'].iloc[index] = '1'
            merge_df['im2'].iloc[index] = '0'
            merge_df['im1_name'].iloc[index] = im_pub_dir+im2_name
            merge_df['im2_name'].iloc[index] = im_pub_dir+im1_name
        else:
            merge_df['im1_name'].iloc[index] = im_pub_dir+im1_name
            merge_df['im2_name'].iloc[index] = im_pub_dir+im2_name

    # shuffle the row orders.
    new_merge_df = merge_df.sample(frac=1, random_state=1)

    # write the df into txt format. list of lists.
    test_lst = []
    for ind, row in new_merge_df.iterrows():
        test_lst.append(row.values.tolist())

    txt_save_name = attr + '_shuffled_img_lst.txt'
    csv_save_name = attr + '_shuffled_img_lst.csv'
    if stargan == 1:
        save_dir = '../static/csv/stargan_'
    else:
        save_dir = '../static/csv/SAGAN_'

    with open(save_dir + txt_save_name, 'w') as file_handler:
        for item in test_lst:
            file_handler.write("{},\n".format(item))

    new_merge_df.to_csv(save_dir+csv_save_name, index=False)


def gen_from_one_df(attr):

    im_pub_dir_prefix = 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_'

    im_pub_dir = im_pub_dir_prefix + attr + '/'

    real_img_lst = '../static/csv/'+attr+'_ground_truth.csv'

    real_im_df = pd.read_csv(real_img_lst)  # [min, max]
    real_im_df.columns = ['im1_name', 'im2_name']
    real_im_df['Task type'] = 'ground_truth'
    real_im_df['im1'] = '0'
    real_im_df['im2'] = '1'
    real_im_df['pair_ind'] = real_im_df.index
    real_im_df['rep'] = 0

    random.seed(3)
    rep_total = 20
    orig_len = len(real_im_df)
    new_len = orig_len + rep_total

    # add some repetition to the generated images.
    rep_lst = random.sample(range(0, len(real_im_df)), rep_total)

    for count, ind in enumerate(rep_lst):
        real_im_df.loc[orig_len + count] = real_im_df.iloc[ind].values

    real_im_df['rep'].iloc[orig_len:new_len] = 1

    # the original order of the real_im_df.

    rand_lst = [random.randint(0, 1) for i in range(0, len(real_im_df))]

    # change the left-right order of the pair.
    real_im_df.is_copy = False
    for index, row in real_im_df.iterrows():
        im1_name = row['im1_name']
        im2_name = row['im2_name']

        if rand_lst[index] == 0:
            real_im_df['im1'].iloc[index] = '1'
            real_im_df['im2'].iloc[index] = '0'
            real_im_df['im1_name'].iloc[index] = im_pub_dir+im2_name
            real_im_df['im2_name'].iloc[index] = im_pub_dir+im1_name
        else:
            real_im_df['im1_name'].iloc[index] = im_pub_dir+im1_name
            real_im_df['im2_name'].iloc[index] = im_pub_dir+im2_name

    # shuffle the row orders.
    new_real_im_df = real_im_df.sample(frac=1, random_state=1)

    # write the df into txt format. list of lists.
    test_lst = []
    for ind, row in new_real_im_df.iterrows():
        test_lst.append(row.values.tolist())

    txt_save_name = attr + '_shuffled_img_lst.txt'
    csv_save_name = attr + '_shuffled_img_lst.csv'

    save_dir = '../static/csv/'

    with open(save_dir + txt_save_name, 'w') as file_handler:
        for item in test_lst:
            file_handler.write("{},\n".format(item))

    new_real_im_df.to_csv(save_dir+csv_save_name, index=False)
    return

# gen_attr_lst(attr='trustworthy', stargan=0)
# gen_attr_lst(attr='aggressive', stargan=0)
# gen_attr_lst(attr='trustworthy', stargan=1)
# gen_attr_lst(attr='aggressive', stargan=1)


gen_from_one_df('responsible')
