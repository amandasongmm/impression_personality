import pandas as pd
from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

df_name = './tmp_data/merged_api_impression.csv'
df = pd.read_csv(df_name)

# fund_ind = df['success'] == 1
# fail_ind = df['success'] == 0

max_fund = df['total fund'].max()
df.loc[df['IPO'] == 1, 'total fund'] = max_fund + 0.5

df_with_fund = df[df['total fund'] != 0]
f_ind = df_with_fund['gender'] == 'female'
m_ind = df_with_fund['gender'] == 'male'
# df_with_fund = df
attribute_names = ['trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable', 'age']
my_map = {'female': 0, 'male': 1}
df_with_fund = df_with_fund.applymap(lambda s: my_map.get(s) if s in my_map else s)


def correlation_analysis():
    correlation_df = pd.DataFrame(columns=['Attr', 'cor', 'p', 'female-cor', 'female-p', 'male-cor', 'male-p'])
    # include gender-specific analysis.
    for id, cur_name in enumerate(attribute_names):
        [cor, p] = pearsonr(df_with_fund['total fund'], df_with_fund[cur_name])
        print('\n\ncur name {}, cor={:.4f}, p={:.4f}'.format(cur_name, cor, p))
        [f_cor, f_p] = pearsonr(df_with_fund[f_ind]['total fund'], df_with_fund[f_ind][cur_name])
        print('cur name {}, cor={:.4f}, p={:.4f}'.format('female', f_cor, f_p))
        [m_cor, m_p] = pearsonr(df_with_fund[m_ind]['total fund'], df_with_fund[m_ind][cur_name])
        print('cur name {}, cor={:.4f}, p={:.4f}'.format('male', m_cor, m_p))

        correlation_df.loc[id] = [cur_name, cor, p, f_cor, f_p, m_cor, m_p]

    correlation_df.to_pickle('./tmp_data/correlation_result.pkl')
    return


def draw_heatmap():
    df_viz = df_with_fund[['age', 'trustworthy', 'intelligent', 'attractive', 'aggressive', 'responsible', 'sociable', 'total fund']]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_viz.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, vmin=-1, vmax=1, annot=True)
    plt.savefig('/home/amanda/Documents/E_faces/gender_viz/general_viz/funded_heatmap.jpg')

    df_female = df_viz[f_ind]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_female.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, vmin=-1, vmax=1, annot=True)
    plt.savefig('/home/amanda/Documents/E_faces/gender_viz/general_viz/female_heatmap.jpg')

    df_male = df_viz[m_ind]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_male.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, vmin=-1, vmax=1, annot=True)
    plt.savefig('/home/amanda/Documents/E_faces/gender_viz/general_viz/male_heatmap.jpg')

    return


def correlation_control_age():
    correlation_df = pd.DataFrame(columns=['Attr', 'cor', 'p', 'start-age', 'end-age'])
    df_with_fund = df[df['total fund'] != 0]
    df_with_fund = df_with_fund[m_ind]
    # include gender-specific analysis.
    count = 0
    for id, cur_name in enumerate(attribute_names):
        [cor, p] = pearsonr(df_with_fund['total fund'], df_with_fund[cur_name])
        print('\n\ncur name {}, cor={:.4f}, p={:.4f}'.format(cur_name, cor, p))

        for i in np.linspace(10, 80, num=8, dtype=int):
            ind = (df_with_fund['age'] > i) & (df_with_fund['age'] < i + 10)

            age_df = df_with_fund[ind]
            [cor, p] = pearsonr(age_df['total fund'], age_df[cur_name])
            if p < 0.01:
                print('Start [{},{}], num = {}. cor={:.4f}, p={:.4f}'.format(i, i+10, sum(ind), cor, p))
            correlation_df.loc[count] = [cur_name, cor, p, i, i+10]
            count += 1
    correlation_df.to_pickle('./tmp_data/correlation_age_control_male.pkl')
    return


def regression_on_funding(x_feat_lst):
    x = df_with_fund[x_feat_lst].values
    y = df_with_fund[['total fund']].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train, y_train)

    y_train_predict = linear_regressor.predict(x_train)
    y_test_predict = linear_regressor.predict(x_test)

    [train_cor, p] = pearsonr(y_train, y_train_predict)
    [test_cor, p] = pearsonr(y_test, y_test_predict)
    print train_cor, test_cor

    for ind, feat in enumerate(x_feat_lst):
        print('{}, {:.3f}'.format(feat, linear_regressor.coef_[0][ind]))

    return

# draw_heatmap()
# correlation_control_age()
# x_feat_lst = ['age', 'gender', 'trustworthy', 'intelligent', 'aggressive', 'attractive', 'responsible', 'sociable']
# regression_on_funding(x_feat_lst)

# x_feat_lst = ['gender', 'trustworthy', 'intelligent', 'aggressive', 'attractive', 'responsible', 'sociable']
# regression_on_funding(x_feat_lst)

# x_feat_lst = ['trustworthy', 'intelligent', 'aggressive', 'attractive', 'responsible', 'sociable']
x_feat_lst = ['age', 'gender']
for i in x_feat_lst:
    print('\n')
    regression_on_funding([i])