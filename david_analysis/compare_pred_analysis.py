import time
import pandas as pd
from scipy.stats import spearmanr, kendalltau
pd.options.mode.chained_assignment = None


start_t = time.time()
feature_name = 'attractive'
sheet_name = '5.attractiveness'

# Prepare data frame from prediction.
predict_path = '/home/amanda/Documents/predicted_results/david_data/to_david/impression_gen_age_ProfileImageDataset.csv'
predict_df = pd.read_csv(predict_path)

predict_df = predict_df[['filename', feature_name]]

participant_lst, photo_id_lst = [], []
for index, row in predict_df.iterrows():
    participant_id, photo_id = row['filename'].split('_')
    photo_id = photo_id[:-4]
    participant_lst.append(int(participant_id))
    photo_id_lst.append(photo_id)

predict_df['participant_id'] = participant_lst
predict_df['photo_id'] = photo_id_lst

predict_df.sort_values(['participant_id', 'photo_id'], ascending=[True, True], inplace=True)

participant_set = list(set(predict_df['participant_id']))
participant_set.sort()

photo_id_set = list(set(predict_df['photo_id']))
photo_id_set.sort()

resort_predict_df = pd.DataFrame(index=participant_set, columns=photo_id_set)

for index, row in predict_df.iterrows():
    resort_predict_df.set_value(index=row['participant_id'], col=row['photo_id'], value=row[feature_name])

print('Resorted prediction ready.')


# Prepare data frame from human ratings.
gt_path = '/home/amanda/Github/ProfileImageDataset/calibrationExperiment_ItemRatingData.xlsx'
gt_df = pd.read_excel(gt_path, sheet_name=sheet_name)

own_gt_df = gt_df[gt_df.columns[0:12]]
other_gt_df = gt_df[gt_df.columns[12:24]]
amt_gt_df = gt_df[gt_df.columns[24:36]]

print('Segmented ground truth ready.')


def drop_multi_level_index(old_df):
    new_df = old_df.reset_index(level=0, drop=True)
    new_df.columns = new_df.iloc[0]
    new_df.drop('participantID', inplace=True)
    return new_df


def calculate_rank_correlation(gt, index_label, predict_arr, c_s_arr, p_s_arr, c_k_arr, p_k_arr):
    gt_arr = gt.loc[index_label].values

    accept_ind = ~pd.isnull(predict_arr)

    accept_predict = predict_arr[accept_ind]
    accept_gt = gt_arr[accept_ind]

    c_spear, p_spear = spearmanr(accept_predict, accept_gt)
    c_kendall, p_kendall = kendalltau(accept_predict, accept_gt)

    c_s_arr.append(c_spear)
    p_s_arr.append(p_spear)
    c_k_arr.append(c_kendall)
    p_k_arr.append(p_kendall)
    return


new_own_gt_df = drop_multi_level_index(own_gt_df)
new_other_gt_df = drop_multi_level_index(other_gt_df)
new_amt_gt_df = drop_multi_level_index(amt_gt_df)

own_c_spear_arr, own_p_spear_arr, own_c_kendall_arr, own_p_kendall_arr = [], [], [], []
other_c_spear_arr, other_p_spear_arr, other_c_kendall_arr, other_p_kendall_arr = [], [], [], []
ave_c_spear_arr, ave_p_spear_arr, ave_c_kendall_arr, ave_p_kendall_arr = [], [], [], []

for ind, row in resort_predict_df.iterrows():
    predict_arr = row.values
    calculate_rank_correlation(new_own_gt_df, ind, predict_arr,
                               own_c_spear_arr, own_p_spear_arr, own_c_kendall_arr, own_p_kendall_arr)

    calculate_rank_correlation(new_other_gt_df, ind, predict_arr,
                               other_c_spear_arr, other_p_spear_arr, other_c_kendall_arr, other_p_kendall_arr)

    calculate_rank_correlation(new_amt_gt_df, ind, predict_arr,
                               ave_c_spear_arr, ave_p_spear_arr, ave_c_kendall_arr, ave_p_kendall_arr)


resort_predict_df['own_spear_c'] = own_c_spear_arr
resort_predict_df['own_spear_p'] = own_p_spear_arr
resort_predict_df['own_kendall_c'] = own_c_kendall_arr
resort_predict_df['own_kendall_p'] = own_p_kendall_arr

resort_predict_df['other_spear_c'] = other_c_spear_arr
resort_predict_df['other_spear_p'] = other_p_spear_arr
resort_predict_df['other_kendall_c'] = other_c_kendall_arr
resort_predict_df['other_kendall_p'] = other_p_kendall_arr


resort_predict_df['ave_spear_c'] = ave_c_spear_arr
resort_predict_df['ave_spear_p'] = ave_p_spear_arr
resort_predict_df['ave_kendall_c'] = ave_c_kendall_arr
resort_predict_df['ave_kendall_p'] = ave_p_kendall_arr


resort_predict_df.to_csv('attractive_comp_result.csv')
print('Done! Elapsed time = {:.2f} seconds.'.format(time.time()-start_t))