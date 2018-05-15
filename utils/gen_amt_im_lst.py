import pandas as pd
import random
pd.options.mode.chained_assignment = None

# Load img pair lst (generated images and real images)
gen_img_lst = '../static/csv/generated.csv'
real_img_lst = '../static/csv/groundtruth.csv'

gen_im_df = pd.read_csv(gen_img_lst)
real_im_df = pd.read_csv(real_img_lst)

# the first column is the face of lower rating. the second columns are of higher ratings.
gen_im_df.columns = ['im1_name', 'im2_name']
gen_im_df['Task type'] = 'test'
gen_im_df['im1'] = 0
gen_im_df['im2'] = 1
gen_im_df['pair_ind'] = gen_im_df.index  # give each pair a unique id

# randomly pick up some pairs and repeat them once.
random.seed(3)
rep_total = 10
gen_orig_len = len(gen_im_df)
gen_new_len = gen_orig_len + rep_total
rep_lst = random.sample(range(0, len(gen_im_df)), rep_total)

for count, ind in enumerate(rep_lst):
    gen_im_df.loc[gen_orig_len + count] = gen_im_df.iloc[ind].values

# the ones that appear twice are marked as rep = 1.
gen_im_df['rep'] = 0
gen_im_df['rep'].iloc[gen_orig_len:gen_new_len] = 1

# change column names to keep the generated and the GT's column names the same.
real_im_df.columns = ['more_trustworthy', 'less_trustworthy']

# change the order of the columns.
real_im_df = real_im_df[['less_trustworthy', 'more_trustworthy']]
real_im_df.columns = ['im1_name', 'im2_name']

# add the same additional labels.
real_im_df['Task type'] = 'gt'
real_im_df['im1'] = 0
real_im_df['im2'] = 1
real_im_df['pair_ind'] = real_im_df.index + gen_orig_len
real_im_df['rep'] = 0

# merge two data frames
merge_df = real_im_df.append(gen_im_df, ignore_index=True)
rand_lst = [random.randint(0, 1) for i in range(0, len(merge_df))]

# change the left-right order of the pair.
merge_df.is_copy = False
dir_name = 'https://github.com/amandasongmm/impression_personality/tree/master/static/img/'
post_fix = '?raw=true'
for index, row in merge_df.iterrows():
    if rand_lst[index] == 0:
        merge_df['im1'].iloc[index] = 1
        merge_df['im2'].iloc[index] = 0
        im1_name = row['im1_name']
        im2_name = row['im2_name']
        merge_df['im1_name'].iloc[index] = dir_name+im2_name+post_fix
        merge_df['im2_name'].iloc[index] = dir_name+im1_name+post_fix
    else:
        im1_name = row['im1_name']
        im2_name = row['im2_name']
        merge_df['im1_name'].iloc[index] = dir_name+im1_name+post_fix
        merge_df['im2_name'].iloc[index] = dir_name+im2_name+post_fix

# shuffle the merged df.
new_merge_df = merge_df.sample(frac=1, random_state=1)

# write the df into txt format. list of lists.
test_lst = []
for ind, row in new_merge_df.iterrows():
    test_lst.append(row.values.tolist())

with open('../static/csv/img_lst.txt', 'w') as file_handler:
    for item in test_lst:
        file_handler.write("{},\n".format(item))

new_merge_df.to_csv('../static/csv/img_lst.csv', index=False)

