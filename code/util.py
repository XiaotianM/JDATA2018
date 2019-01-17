# -*- coding: utf-8 -*-
# author：Cookly, ShawnMa
import pandas as pd 
import numpy as np 
from datetime import datetime
import lightgbm as lgb

class DataLoader(object):
	def __init__(self,
				FILE_jdata_sku_basic_info,
				FILE_jdata_user_action,
				FILE_jdata_user_basic_info,
				FILE_jdata_user_comment_score,
				FILE_jdata_user_order,
				startMonth,
				endMonth
				):
		self.FILE_jdata_sku_basic_info = FILE_jdata_sku_basic_info
		self.FILE_jdata_user_action = FILE_jdata_user_action
		self.FILE_jdata_user_basic_info = FILE_jdata_user_basic_info
		self.FILE_jdata_user_comment_score = FILE_jdata_user_comment_score
		self.FILE_jdata_user_order = FILE_jdata_user_order

		self.df_sku_info = pd.read_csv(self.FILE_jdata_sku_basic_info)
		self.df_user_info = pd.read_csv(self.FILE_jdata_user_basic_info)
		self.df_user_comment = pd.read_csv(self.FILE_jdata_user_comment_score, parse_dates=['comment_create_tm'])
		self.df_user_order = pd.read_csv(self.FILE_jdata_user_order, parse_dates=['o_date'])
		# B榜中存在sku_id表单中为空的值
		self.df_user_action = pd.read_csv(self.FILE_jdata_user_action, parse_dates=['a_date']).dropna(subset=['sku_id'])
		atype = pd.get_dummies(self.df_user_action["a_type"], prefix="a_type") #浏览类型更改为on-hot
		self.df_user_action = pd.concat([self.df_user_action, atype], axis=1)

		# 对sku_info 增加额外的特征
		# 每个sku_id 被浏览次数 sku_act_num
		sku_feature_tmp = self.df_user_action.groupby(['sku_id'])['a_num'].sum().reset_index().\
										rename(columns={"sku_id":"sku_id", 'a_num':'sku_act_num'})
		self.df_sku_info = self.df_sku_info.merge(sku_feature_tmp, on=['sku_id'], how='left')
		# 每个sku_id 浏览人数 sku_act_user_num
		sku_feature_tmp = self.df_user_action.groupby(['sku_id'])['user_id'].nunique().reset_index().\
										rename(columns={"sku_id":"sku_id", "user_id":"sku_act_user_num"})
		self.df_sku_info = self.df_sku_info.merge(sku_feature_tmp, on=['sku_id'], how='left')
		# 每个sku_id 被购买次数 sku_buy_num
		sku_feature_tmp = self.df_user_order.groupby(['sku_id'])['o_sku_num'].sum().reset_index().\
										rename(columns={"sku_id":"sku_id", 'o_sku_num':'sku_buy_num'})
		self.df_sku_info = self.df_sku_info.merge(sku_feature_tmp, on=['sku_id'], how='left')
		# 每个sku_id 购买人数 sku_ord_user_num
		sku_feature_tmp = self.df_user_order.groupby(['sku_id'])['user_id'].nunique().reset_index().\
										rename(columns={"sku_id":"sku_id", "user_id":"sku_buy_user_num"})
		self.df_sku_info = self.df_sku_info.merge(sku_feature_tmp, on=['sku_id'], how='left')

		# # 计算商品性价比
		# self.df_sku_info['para1_price'] = self.df_sku_info['para_1'] / self.df_sku_info['price']

		# Clea Data 
		# select alignment 3 month as test
		self.df_Order_Sku = self.df_user_order.merge(self.df_sku_info,on='sku_id',how='left') 							
		self.df_Order_Sku = self.df_Order_Sku[(self.df_Order_Sku.cate == 30) | (self.df_Order_Sku.cate == 101)]
		self.user_list = self.df_Order_Sku[(self.df_Order_Sku.o_date <= endMonth) & (self.df_Order_Sku.o_date >= startMonth)].\
										drop_duplicates('user_id', keep='first')
		self.user_list = pd.DataFrame(self.user_list["user_id"])
		self.df_user_info = pd.merge(self.df_user_info, self.user_list, on="user_id")
		self.df_user_action = pd.merge(self.df_user_action, self.user_list, on="user_id")
		self.df_user_comment = pd.merge(self.df_user_comment, self.user_list, on="user_id")
		self.df_user_order = pd.merge(self.df_user_order, self.user_list, on="user_id")

		# sort by datetime
		self.df_user_action = self.df_user_action.sort_values(['user_id','a_date'])
		self.df_user_order = self.df_user_order.sort_values(['user_id','o_date'])
		self.df_user_comment = self.df_user_comment.sort_values(['user_id','comment_create_tm'])

		# year month day
		self.df_user_order['year'] = self.df_user_order['o_date'].dt.year
		self.df_user_order['month'] = self.df_user_order['o_date'].dt.month
		self.df_user_order['day'] = self.df_user_order['o_date'].dt.day

		self.df_user_action['year'] = self.df_user_action['a_date'].dt.year
		self.df_user_action['month'] = self.df_user_action['a_date'].dt.month
		self.df_user_action['day'] = self.df_user_action['a_date'].dt.day



class Features(object):
	def __init__(self,
				DataLoader,
				PredMonthBegin,
				PredMonthEnd,
				FeatureMonthList,
				MakeLabel = True
				):
		self.DataLoader = DataLoader
		self.PredMonthBegin = PredMonthBegin
		self.PredMonthEnd = PredMonthEnd
		self.FeatureMonthList = FeatureMonthList
		self.MakeLabel = MakeLabel

		# label columns
		self.LabelColumns = ['Label_30_101_BuyNum','Label_30_101_FirstTime','Label_30_101_BuyOrNot']
		self.IDColumns = ['user_id']
		self.S2Columns = []

		# merge feature table
		# Order Comment User Sku
		# print(self.DataLoader.df_user_order.head())
		# print(self.DataLoader.df_user_info.head())
		self.df_Order_Comment_User_Sku = self.DataLoader.df_user_order.\
												merge(self.DataLoader.df_user_comment,on=['user_id','o_id'],how='left').\
												merge(self.DataLoader.df_user_info,on='user_id',how='left').\
												merge(self.DataLoader.df_sku_info,on='sku_id',how='left')

		# Action User Sku
		self.df_Action_User_Sku = self.DataLoader.df_user_action.\
												merge(self.DataLoader.df_user_info,on='user_id',how='left').\
												merge(self.DataLoader.df_sku_info,on='sku_id',how='left')

		# Make Label 
		self.data_BuyOrNot_FirstTime = self.MakeLabel_()

		# MakeFeature_Order_Comment_
		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			self.MakeFeature_Order_Comment_(
									FeatureMonthBegin = FeatureMonthBegin,
									FeatureMonthEnd = FeatureMonthEnd,
									BetweenFlag = 'OM'+str(month)+'_',
									month = month
				)

		# MakeFeature_Action_
		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			self.MakeFeature_Action_(
									FeatureMonthBegin = FeatureMonthBegin,
									FeatureMonthEnd = FeatureMonthEnd,
									BetweenFlag = 'AM'+str(month)+'_'
				)		

		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			# 时间太久远的转化率可能并没什么用
			if month <= 1: 
				self.MakeFeature_Action_Order_(
										FeatureMonthBegin = FeatureMonthBegin,
										FeatureMonthEnd = FeatureMonthEnd,
										BetweenActFlag = 'AM'+str(month)+'_',
										BetweenOrdFlag = 'OM'+str(month)+'_',
										BetweenFlag = 'Rate'+str(month)+'_',
										month = month
					)	 

		self.TrainColumns = [col for col in self.data_BuyOrNot_FirstTime.columns if col not in self.IDColumns + self.LabelColumns + self.S2Columns]
		self.TrainColumns_S2 = [col for col in self.data_BuyOrNot_FirstTime.columns if col not in self.IDColumns + self.LabelColumns]

	def MakeLabel_(self):
		self.data_BuyOrNot_FirstTime = self.DataLoader.df_user_info

		if self.MakeLabel:
			df_user_order_sku = self.DataLoader.df_user_order.merge(self.DataLoader.df_sku_info,on='sku_id',how='left')

			label_temp_ = df_user_order_sku[(df_user_order_sku['o_date']>=self.PredMonthBegin) &\
										(df_user_order_sku['o_date']<=self.PredMonthEnd)]

			label_temp_30_101 = label_temp_[(label_temp_['cate']==30) | (label_temp_['cate']==101)]

			# 统计用户当月下单数 回归建模
			BuyOrNotLabel_30_101 = label_temp_30_101.groupby(['user_id'])['o_id'].\
										nunique().\
										reset_index().\
										rename(columns={'user_id':'user_id','o_id':'Label_30_101_BuyNum'})
					
			# 用户首次下单时间 回归建模
			# keep first 获得首次下单时间 - 月初时间 = 下单在当月第几天购买
			FirstTimeLabel_30_101 = label_temp_30_101.\
			 						drop_duplicates('user_id', keep='first')[['user_id','o_date']].\
			 						rename(columns={'user_id':'user_id','o_date':'Label_30_101_FirstTime'})
			FirstTimeLabel_30_101['Label_30_101_FirstTime'] = (FirstTimeLabel_30_101['Label_30_101_FirstTime'] - self.PredMonthBegin).dt.days

			# FirstTimeLabel_30_101 = label_temp_30_101.groupby(['user_id'])['day'].\
			#										min().\
			#										reset_index().\
			#										rename(columns={'user_id':'user_id','day':'Label_30_101_FirstTime'})

			# merge label
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(BuyOrNotLabel_30_101,on='user_id', how='left').\
																		merge(FirstTimeLabel_30_101, on='user_id', how='left')
			# fillna 0
			self.data_BuyOrNot_FirstTime.fillna(0,inplace=True)
			self.data_BuyOrNot_FirstTime['Label_30_101_BuyOrNot'] = self.data_BuyOrNot_FirstTime['Label_30_101_BuyNum'].apply(lambda x:1 if x else 0) 
		else:
			self.data_BuyOrNot_FirstTime['Label_30_101_BuyNum'] = -1
			self.data_BuyOrNot_FirstTime['Label_30_101_FirstTime'] = -1
			self.data_BuyOrNot_FirstTime['Label_30_101_BuyOrNot'] = -1
 
		return self.data_BuyOrNot_FirstTime

	def MakeFeature_Order_Comment_(self, 
								FeatureMonthBegin,
								FeatureMonthEnd,
								BetweenFlag,
								month
									):
		# Order Comment User Sku
		'''
		self.df_Order_Comment_User_Sku
		user_id sku_id o_id o_date o_area o_sku_num comment_create_tm score_level age sex user_lv_cd price cate para_1 para_2 para_3
		'''
		features_temp_Order_ = self.df_Order_Comment_User_Sku[(self.df_Order_Comment_User_Sku['o_date']>=FeatureMonthBegin) &\
															(self.df_Order_Comment_User_Sku['o_date']<=FeatureMonthEnd)]

		# make features	
		#################################################################################################################												
		########### papa_1 ######################
		# 购买的30品类的para1 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['para_1'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_para_1_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的30品类的para1 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['para_1'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_para_1_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的30品类的para1 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['para_1'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_para_1_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30品类的para1 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['para_1'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_para_1_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		# 购买的101品类的para1 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_101_para_1_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的101品类的para1 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_101_para_1_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的101品类的para1 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_101_para_1_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的101品类的para1 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_101_para_1_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		# 购买的30 101品类的para1 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_101_para_1_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的30 101品类的para1 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_101_para_1_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 购买的30 101品类的para1 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_101_para_1_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30 101品类的para1 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['para_1'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_30_101_para_1_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类的para1 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['para_1'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_other_para_1_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类的para1 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['para_1'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_other_para_1_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类的para1 最大
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['para_1'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_other_para_1_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类的para1 最小
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['para_1'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','para_1':BetweenFlag+'_other_para_1_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# ###########价格#####################
		# 购买的30 101品类价格 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_101_mean_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30品类价格 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['price'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_mean_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的101品类价格 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_101_mean_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30,101品类价格 平均
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['price'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_other_mean_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30 101品类价格 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_101_std_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30品类价格 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['price'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_std_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的101品类价格 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_101_std_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30,101品类价格 方差
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['price'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_other_std_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30 101品类价格 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_101_max_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30品类价格 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['price'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_max_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30 101品类价格 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_101_max_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类价格 最大值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['price'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_other_max_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30 101品类价格 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_101_min_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的101品类价格 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['price'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_101_min_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的30品类价格 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['price'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_30_min_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 购买的非30 101品类价格 最小值
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['price'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','price':BetweenFlag+'o_id_cate_other_min_price'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 30品类 的平均性价比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'30_para_price'] = self.data_BuyOrNot_FirstTime[BetweenFlag+'_30_para_1_mean'] \
															/ self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_30_mean_price']
		
		# 101品类 的平均性价比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'101_para_price'] = self.data_BuyOrNot_FirstTime[BetweenFlag+'_101_para_1_mean'] \
															/ self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_101_mean_price']
		
		# 30 101品类 的平均性价比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'30_101_para_price'] = self.data_BuyOrNot_FirstTime[BetweenFlag+'_30_101_para_1_mean'] \
															/ self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_30_101_mean_price']
		
		# 非30品类 的平均性价比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'other_para_price'] = self.data_BuyOrNot_FirstTime[BetweenFlag+'_other_para_1_mean'] \
															/ self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_other_mean_price']
		

		# ###########购买区域#####################
		if month >= 6: #短期地点变化不会很大
			# 购买 30 101的 地点变化的次数
			features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
												groupby(['user_id'])['o_area'].\
												nunique().\
												reset_index().\
												rename(columns={'user_id':'user_id', 'o_area': BetweenFlag+'o_area_30_101_cnt'})
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
			
			# 购买30 101的地点众数
			features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
												groupby(['user_id'])['o_area'].\
												agg(lambda x: x.value_counts().index[0]).\
												reset_index().\
												rename(columns={'user_id':'user_id', 'o_area': BetweenFlag+'o_area_30_101_mode'})
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

			# 购买非 30 101的 地点变化的次数
			features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
												groupby(['user_id'])['o_area'].\
												nunique().\
												reset_index().\
												rename(columns={'user_id':'user_id', 'o_area': BetweenFlag+'o_area_other_cnt'})
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

			# 购买非30 101的地点众数
			features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
												groupby(['user_id'])['o_area'].\
												agg(lambda x: x.value_counts().index[0]).\
												reset_index().\
												rename(columns={'user_id':'user_id', 'o_area': BetweenFlag+'o_area_other_mode'})
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# ###########订单数#####################
		# o_id cate 30 101 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# o_id cate 30 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# o_id cate 101 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# o_id cate 非 30 101 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_other_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		# ###########购买商品次数#####################
		# sku_id cate 30 101 购买商品种数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 购买商品种数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')		

		# sku_id cate 101 购买商品种数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 非 30 101 购买商品种数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_other_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		# ###########购买天数#####################
		# o_date cate 30 101 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_30_101_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')		

		# o_date cate 30 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_30_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# o_date cate 101 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_101_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')									

		# o_date cate 非30 101 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_other_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')		

		# ###########用户购买件数#####################
		# o_sku_num cate 30 101 用户购买件数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_cate_30_101_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# o_sku_num cate 30 用户购买件数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_cate_30_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# o_sku_num cate 101 用户购买件数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_cate_101_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')				

		# o_sku_num cate 非30 101 用户购买件数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_cate_other_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	
		'''
		继续添加
		'''
		#################################################################################################################	
		# 购买当月平均第几天
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['day'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_30_101_meanday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 最后一次购买时间距离预测日期的时间距离
		if month == 11:
			# 购买月份数
			features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
												groupby(['user_id'])['month'].\
												nunique().\
												reset_index().\
												rename(columns={'user_id':'user_id','month':BetweenFlag+'month_cate_30_101_monthnum'})
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 时间间隔
		# 购买时间间隔 mean 30 101：
		features_temp_Order_ = features_temp_Order_.copy()
		features_temp_Order_['diff'] = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_date'].diff().dt.days
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_mean')
		# 购买时间间隔 std 30 101：
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_std')
		# 购买时间间隔 max 30 101：
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_max')
		# 购买时间间隔 min 30 101：
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_min')
		# ########## 评价天数 ####################
		# # o_date cate 30 101 评价天数
		# features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
		# 									groupby(['user_id'])['comment_create_tm'].\
		# 									nunique().\
		# 									reset_index().\
		# 									rename(columns={'user_id':'user_id','comment_create_tm':BetweenFlag+'comment_create_tm_30_101_nuique'})
		# self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')		
		'''
		继续添加
		'''
		

	def MakeFeature_Action_(self,
							FeatureMonthBegin,
							FeatureMonthEnd,
							BetweenFlag
							):
		# Action User Sku
		'''
		self.df_Action_User_Sku
		user_id sku_id a_date a_num a_type age sex user_lv_cd price cate para_1 para_2 para_3
		'''
		features_temp_Action_ = self.df_Action_User_Sku[(self.df_Action_User_Sku['a_date']>=FeatureMonthBegin) &\
														(self.df_Action_User_Sku['a_date']<=FeatureMonthEnd)]
		# 用户浏览特征
		# sku_id cate 30 浏览行为次数
		features_temp_Action_['a_type_1'] = features_temp_Action_['a_type_1'] * features_temp_Action_['a_num']
		features_temp_Action_['a_type_2'] = features_temp_Action_['a_type_2'] * features_temp_Action_['a_num']
		
		# sku_id cate 30 浏览sku次数 act1
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30)].\
											groupby(['user_id'])['a_type_1'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_1':BetweenFlag+'a_type1_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 浏览sku次数 act2
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30)].\
											groupby(['user_id'])['a_type_2'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_2':BetweenFlag+'a_type2_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 101 浏览sku次数 act1
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_type_1'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_1':BetweenFlag+'a_type1_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 101 浏览sku次数 act2
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_type_2'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_2':BetweenFlag+'a_type2_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 101 浏览sku次数 act1
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_type_1'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_1':BetweenFlag+'a_type1_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 101 浏览sku次数 act2
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_type_2'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_type_2':BetweenFlag+'a_type2_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 浏览sku种类数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 101 浏览sku种类数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 101 浏览sku种类数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# a_date cate 30 天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'a_date_cate_30_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# a_date cate 101 天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'a_date_cate_101_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# a_date cate 30 101 天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'a_date_cate_30_101_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# 浏览当月平均第几天
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['day'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'act_day_cate_30_101_meanday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		'''
		# 继续整加特征
		'''
		# 浏览时间间隔 mean 30 101 
		features_temp_Action_ = features_temp_Action_.copy()
		features_temp_Action_['diff'] = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
										groupby(['user_id'])['a_date'].diff().dt.days
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_mean'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_mean')
		# 浏览时间间隔 std 30 101 
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											std().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_std'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_std')
		# 浏览时间间隔 max 30 101 
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_max'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_max')
		# 浏览时间间隔 min 30 101 
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['diff'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','diff':BetweenFlag+'day_cate_30_101_gapday_min'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		self.S2Columns.append(BetweenFlag+'day_cate_30_101_gapday_min')
		

	def MakeFeature_Action_Order_(self,
							FeatureMonthBegin,
							FeatureMonthEnd,
							BetweenActFlag,
							BetweenOrdFlag,
							BetweenFlag,
							month
							):
		# Action_Order User Sku

		# 浏览时间-购买时间
		if month <= 1:
			self.data_BuyOrNot_FirstTime[BetweenFlag + 'diftime_act_buy'] = self.data_BuyOrNot_FirstTime[BetweenOrdFlag+"day_cate_30_101_meanday"] \
												- self.data_BuyOrNot_FirstTime[BetweenActFlag+"act_day_cate_30_101_meanday"]
############################		
		# 浏览次数act1用户购买次数转化率 30
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_30_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_30_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_30_cnt']

		# 浏览次数act2用户购买次数转化率 30
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_30_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_30_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_30_cnt']

		# 浏览次数act1用户购买次数转化率 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_101_cnt']
											
		# 浏览次数act2用户购买次数转化率 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_101_cnt']

		# 浏览次数act1用户购买次数转化率 30 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_30_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_30_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_30_101_cnt']
											
		# 浏览次数act2用户购买次数转化率 30 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_30_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_30_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'sku_id_cate_30_101_cnt']

#########################
		# 浏览次数act1用户购买订单转化率 30
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_ord_30_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_30_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_30_cnt']

		# 浏览次数act2用户购买订单转化率 30
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_ord_30_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_30_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_30_cnt']

		# 浏览次数act1用户购买订单转化率 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_ord_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_101_cnt']
											
		# 浏览次数act2用户购买订单转化率 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_ord_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_101_cnt']

		# 浏览次数act1用户购买订单转化率 30 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type1_ord_30_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type1_30_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_30_101_cnt']
											
		# 浏览次数act2用户购买订单转化率 30 101
		self.data_BuyOrNot_FirstTime[BetweenFlag+'act_type2_ord_30_101_buytime'] = self.data_BuyOrNot_FirstTime[BetweenActFlag+'a_type2_30_101_cnt'] \
											/ self.data_BuyOrNot_FirstTime[BetweenOrdFlag+'o_id_cate_30_101_cnt']

		# '''
		# # 继续整加特征
		# '''













