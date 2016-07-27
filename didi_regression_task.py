# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 05:46:50 2016

@author: easypc
"""
from time import time

import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

import xgboost as xgb


train = "TRAIN.csv"

train_df = pd.read_csv(train,header = 0)

 
feature_list = [#'ID', 'ord_cnt', 'district_id', 'time', 'Time_slot', 'wday', 'date',
       #'gap', 'psg_goodness', 'NotLocal', 'ord_ratio', 'avg_Prc',
       #'NotLocal_ratio',
       'driversArriving', #'driversArrived',
       
       'Weather',
       'temperature',
       'PM2.5',
       #'T1', 'T2', 'T3', 'T4', 'sumT', 'T1ratio',
       #'T2ratio', 'T3ratio', 'T4ratio',
       'gap10minAgo', 'gap20minAgo','gap30minAgo',       
       'ordCnt10minAgo','ordCnt20minAgo', 'ordCnt30minAgo',     
       'NotLocal10minAgo', 'NotLocal20minAgo', 'NotLocal30minAgo',
       'NotLocal_ratio10minAgo', 'NotLocal_ratio20minAgo',
       'NotLocal_ratio30minAgo',
       'psg_goodness10minAgo',
       'psg_goodness20minAgo', 'psg_goodness30minAgo',
       'ord_ratio10minAgo','ord_ratio20minAgo', 'ord_ratio30minAgo',     
       'avg_Prc10minAgo','avg_Prc20minAgo', 'avg_Prc30minAgo',
       
      # 'T1_10minAgo', 'T1_20minAgo',
      # 'T1_30minAgo', 'T2_10minAgo', 'T2_20minAgo', 'T2_30minAgo',
      # 'T3_10minAgo', 'T3_20minAgo', 'T3_30minAgo', 
      # 'T4_10minAgo',
      # 'T4_20minAgo', 'T4_30minAgo', 
       'T1ratio_10minAgo',
       'T1ratio_20minAgo', 'T1ratio_30minAgo', 'T2ratio_10minAgo',
       'T2ratio_20minAgo', 'T2ratio_30minAgo', 'T3ratio_10minAgo',
       'T3ratio_20minAgo', 'T3ratio_30minAgo', 'T4ratio_10minAgo',
       'T4ratio_20minAgo', 'T4ratio_30minAgo', 
       'delta12_gap','delta23_gap', 'delta13_gap', 
       'delta23ord_ratio', 'delta13ord_ratio', 'delta12ord_ratio',
       'delta23avg_Prc', 'delta13avg_Prc','delta12avg_Prc',
    #   'delta12T1', 'delta23T1', 'delta13T1',
    #   'delta12T2', 'delta23T2', 'delta13T2',
    #   'delta12T3', 'delta23T3', 'delta13T3',
    #   'delta12T4', 'delta23T4', 'delta13T4',
       'delta12_ordCnt', 'delta23_ordCnt', 'delta13_ordCnt']

X1 = train_df[feature_list]

district_dummies = np.array(pd.get_dummies(train_df['district_id'],drop_first=True))
time_dummies = np.array(pd.get_dummies(train_df['time'],drop_first=True))
Wday_dummies = np.array(pd.get_dummies(train_df['wday'],drop_first=True))


X = np.concatenate([X1,time_dummies,district_dummies, 
                    Wday_dummies], axis=1)

y1 = train_df.gap.values
y = np.array(y1)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(X, y, test_size=0.27, random_state=42)



print "features_train dimensions",np.shape(features_train)

t0 = time()
reg = xgb.XGBRegressor(n_estimators=2000,learning_rate=0.115)

reg.fit(features_train,labels_train)
y_predict_xgb = reg.predict(features_test)

mae=mean_absolute_error(labels_test, y_predict_xgb)
evs=explained_variance_score(labels_test, y_predict_xgb)


print "##################################"
print "Xgb mean_absolute_error", mae
print "Xgb explained_variance_score", evs
print "Xgb done in %0.3fs" % (time() - t0)
print "##################################"


feat_imp = pd.Series(reg.booster().get_fscore()).sort_values(ascending=False)


#function below needs updating
#if a dummie feature gets into top 30 features ,out of bounds error will be shown 
print zip(map(lambda x: feature_list[int(feat_imp[[x]].index[0].strip('f'))],
                                 range(len(feat_imp))[:20]),
                                 feat_imp[:20])


