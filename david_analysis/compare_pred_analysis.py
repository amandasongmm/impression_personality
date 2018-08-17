import pandas as pd
pd.options.mode.chained_assignment = None

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


# Prepare data frame from human ratings.
gt_path = '/home/amanda/Github/ProfileImageDataset/calibrationExperiment_ItemRatingData.xlsx'
gt_df = pd.read_excel(gt_path, sheet_name=sheet_name)

own_gt_df = gt_df[gt_df.columns[0:12]]
other_gt_df = gt_df[gt_df.columns[12:24]]
amt_gt_df = gt_df[gt_df.columns[24:36]]


new_df = own_gt_df.reset_index(level=0, drop=True)
new_df.columns = new_df.iloc[0]
new_df.drop('participantID', inplace=True)

