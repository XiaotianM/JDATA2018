# -*- coding: utf-8 -*-
# author：ShawnMa, Cookly
from util import DataLoader, Features
from SBBTree_ONLINE import SBBTree
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import lightgbm as lgb

# test code
Data = DataLoader(
				FILE_jdata_sku_basic_info='../data/jdata_sku_basic_info.csv',
				FILE_jdata_user_action='../data/jdata_user_action.csv',
				FILE_jdata_user_basic_info='../data/jdata_user_basic_info.csv',
				FILE_jdata_user_comment_score='../data/jdata_user_comment_score.csv',
				FILE_jdata_user_order='../data/jdata_user_order.csv',
				startMonth = datetime(2017, 5, 1),
				endMonth = datetime(2017, 7, 31)
			)
# train data
TrainFeatures = Features(
						DataLoader=Data,
						PredMonthBegin = datetime(2017, 8, 1),
						PredMonthEnd = datetime(2017, 8, 30),
						FeatureMonthList = [
									(datetime(2017, 7, 31)-timedelta(7)*1+timedelta(1),   datetime(2017, 7, 31), 0.25),\
									(datetime(2017, 7, 31)-timedelta(7)*2+timedelta(1),   datetime(2017, 7, 31), 0.5),\
									(datetime(2017, 7, 31)-timedelta(7)*3+timedelta(1),   datetime(2017, 7, 31), 0.7),\
									(datetime(2017, 7, 31)-timedelta(30)*1+timedelta(1),  datetime(2017, 7, 31), 1),\
									(datetime(2017, 7, 31)-timedelta(30)*3+timedelta(1),  datetime(2017, 7, 31), 3),\
									(datetime(2017, 7, 31)-timedelta(30)*6+timedelta(1),  datetime(2017, 7, 31), 6),\
									(datetime(2017, 7, 31)-timedelta(30)*9+timedelta(1),  datetime(2017, 7, 31), 9),\
									(datetime(2017, 7, 31)-timedelta(30)*11+timedelta(1), datetime(2017, 7, 31), 11)], # begin:2016,05,06
						MakeLabel = True
					)

# 为了与线上保持一致，理论上仅由8w人进行参与
Offline_Pred = pd.DataFrame(Data.df_user_info["user_id"])

################################################
# train code
Data = DataLoader(
				FILE_jdata_sku_basic_info='../data/jdata_sku_basic_info.csv',
				FILE_jdata_user_action='../data/jdata_user_action.csv',
				FILE_jdata_user_basic_info='../data/jdata_user_basic_info.csv',
				FILE_jdata_user_comment_score='../data/jdata_user_comment_score.csv',
				FILE_jdata_user_order='../data/jdata_user_order.csv',
				startMonth = datetime(2017, 6, 1),
				endMonth = datetime(2017, 8, 31)
			)

# pred data
PredFeatures = Features(
					DataLoader=Data,
					PredMonthBegin = datetime(2017, 9, 1),
					PredMonthEnd = datetime(2017, 9, 30),
					FeatureMonthList = [
									(datetime(2017, 8, 31)-timedelta(7)*1+timedelta(1),   datetime(2017, 8, 31), 0.25),\
									(datetime(2017, 8, 31)-timedelta(7)*2+timedelta(1),   datetime(2017, 8, 31), 0.5),\
									(datetime(2017, 8, 31)-timedelta(7)*3+timedelta(1),   datetime(2017, 8, 31), 0.7),\
									(datetime(2017, 8, 31)-timedelta(30)*1+timedelta(1),  datetime(2017, 8, 31), 1),\
									(datetime(2017, 8, 31)-timedelta(30)*3+timedelta(1),  datetime(2017, 8, 31), 3),\
									(datetime(2017, 8, 31)-timedelta(30)*6+timedelta(1),  datetime(2017, 8, 31), 6), \
									(datetime(2017, 8, 31)-timedelta(30)*9+timedelta(1),  datetime(2017, 8, 31), 9), \
									(datetime(2017, 8, 31)-timedelta(30)*11+timedelta(1), datetime(2017, 8, 31), 11)], # begin:2016,06,05
					MakeLabel = False
				)

# train_features = TrainFeatures.TrainColumns
# cols = TrainFeatures.IDColumns + TrainFeatures.LabelColumns + train_features


###========================================================###
###======================S1 Predict========================###
###========================================================###
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
###############################################################
model = SBBTree(params=params, stacking_num=5, bagging_num=3,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=100)

# train 下个月购买次数预测 回归模型
train_features = TrainFeatures.TrainColumns
train_label_BuyNum = 'Label_30_101_BuyNum'

X = TrainFeatures.data_BuyOrNot_FirstTime[train_features].values 
y = TrainFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum].values

X_pred = PredFeatures.data_BuyOrNot_FirstTime[train_features].values

model.fit(X,y)
Offline_Pred[train_label_BuyNum] = model.predict(X)  # 线下 S1
PredFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum] = model.predict(X_pred)  # 线上 S1


###========================================================###
###======================S2 Predict========================###
###========================================================###
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
model = SBBTree(params=params, stacking_num=5, bagging_num=3,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=100)

# train 当月首次购买时间预测 回归模型
train_features = TrainFeatures.TrainColumns_S2
train_label_FirstTime = 'Label_30_101_FirstTime'

X = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime["Label_30_101_BuyNum"]>0][train_features].values 
y = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime["Label_30_101_BuyNum"]>0][train_label_FirstTime].values 

X_pred = PredFeatures.data_BuyOrNot_FirstTime[train_features].values 

model.fit(X,y)
Offline_Pred[train_label_FirstTime] = model.predict(TrainFeatures.data_BuyOrNot_FirstTime[train_features].values)  # 线下S2
PredFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] = model.predict(X_pred)  # 线上S2


#######################################################

# 线上提交部分
# submit
columns = ['user_id'] + [train_label_BuyNum] + [train_label_FirstTime]
out_submit = PredFeatures.data_BuyOrNot_FirstTime[columns].sort_values([train_label_BuyNum],ascending=False)
out_submit[train_label_FirstTime] = out_submit[train_label_FirstTime].map(lambda day: datetime(2017, 9, 1)+timedelta(days=int(day+0.49-1)))

out_submit = out_submit[['user_id']+[train_label_FirstTime]]
out_submit.columns = ['user_id','pred_date']
out_submit.head(50000).to_csv('../submit/predict_sortpro.csv',index=False,header=True)


# 线下评分
def score(pred, real):
    # pred: user_id, pre_date | real: user_id, o_date
    # wi与oi的定义与官网相同
#     pred['Label_30_101_FirstTime'] = pd.to_datetime(pred['Label_30_101_FirstTime']).dt.day
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    real['oi'] = 1
    compare = pd.merge(pred, real, how='left', on='user_id')
    compare.fillna(0, inplace=True)  # 实际上没有购买的用户，correct_for_S1列的值为nan，将其赋为0
    S1 = np.sum(compare['oi'] * compare['wi']) / np.sum(compare['wi'])

    compare_for_S2 = compare[compare['oi'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['label'] - compare_for_S2['Label_30_101_FirstTime']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)


pred = Offline_Pred.sort_values([train_label_BuyNum],ascending=False).head(50000)
pred[train_label_FirstTime] = pred[train_label_FirstTime].map(lambda day: int(day+0.49-1))
real = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime["Label_30_101_BuyNum"]>0][['user_id']+[train_label_FirstTime]].\
                                    rename(columns={'user_id':'user_id','Label_30_101_FirstTime':'label'}) # real
score(pred, real)

