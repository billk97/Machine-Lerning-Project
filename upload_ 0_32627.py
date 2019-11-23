import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
from scipy import stats
from numpy import median
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def data(my_pred):
    True in (my_pred < 0)
    for i, Y  in enumerate(my_pred):
        my_pred[i]=int(np.around([my_pred[i]]))
        if my_pred[i] < 0:
            my_pred[i] = 0
    return my_pred
print('RMSLE:', np.sqrt(mean_squared_log_error(Y_test, pred)))
print('R2:', r2_score(Y_test, pred))

def evaluate(my_pred):
    print('RMSLE:', np.sqrt(mean_squared_log_error(Y_test, my_pred)))
    print('R2:', r2_score(Y_test, my_pred))


filename2 = '../input/inf131-2019/train.csv' 
dataframe_train = pd.read_csv(filename2)
dataframe_train.head(1)
dataframe_train.isnull().sum()
dataframe_train.rename(columns={'weathersit':'weather','mnth':'month','hr':'hour','yr':'year','hum': 'humidity','cnt':'count'},inplace=True)

dataframe_train['season'] = dataframe_train.season.astype('category')
dataframe_train['year'] = dataframe_train.year.astype('category')
dataframe_train['month'] = dataframe_train.month.astype('category')
dataframe_train['hour'] = dataframe_train.hour.astype('category')
dataframe_train['holiday'] = dataframe_train.holiday.astype('category')
dataframe_train['weekday'] = dataframe_train.weekday.astype('category')
dataframe_train['workingday'] = dataframe_train.workingday.astype('category')
#sbinoume to weather gt sta traing den exei weather 4
#xoris na ginoun kategory beltioueike elaxista
#season yr mnth hr holiday weekday workingday weathersit temp atemp hum windspeed casual registered cnt
dataframe_train = dataframe_train.drop(['atemp', 'casual', 'registered','humidity'], axis=1)
# Training and test data is created by splitting the main data. 30% of test data is considered
#dataframe_train=pd.get_dummies(dataframe_train,prefix=['co1-','col2-'])
#print(dataframe_train)
dataframe_train.head(1)


dataframe_train = pd.get_dummies(dataframe_train)
dataframe_train.head(1)
dataframe_train.columns


X = dataframe_train[['temp', 'windspeed', 'season_1', 'season_2', 'season_3',
       'season_4', 'year_0', 'year_1', 'month_1', 'month_2', 'month_3',
       'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
       'month_10', 'month_11', 'month_12', 'hour_0', 'hour_1', 'hour_2',
       'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
       'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
       'hour_22', 'hour_23', 'holiday_0', 'holiday_1', 'weekday_0',
       'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
       'weekday_6', 'workingday_0', 'workingday_1', 'weather']]
Y = dataframe_train['count']
print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#ElasticNet
print("Elastic")
reg = ElasticNet(random_state=0)
reg.fit(X_train , Y_train)
pred = reg.predict(X_test)
pred = data(pred)
evaluate(pred)
# RandomeRorest
print("random forest")
clf = RandomForestRegressor()
clf.fit(X, Y)
Y_pred = clf.predict(X_test)
Y_pred = data(Y_pred)
evaluate(Y_pred)
# RandomeRorest1
print("random forest1")
clf1 = RandomForestRegressor()
clf1.fit(X, Y)
Y_pred1 = clf1.predict(X_test)
Y_pred1 = data(Y_pred1)
evaluate(Y_pred1)
#bays
print("bays")
bays = linear_model.BayesianRidge()
bays.fit(X_train , Y_train) 
bays_pred = bays.predict(X_test)
bays_pred = data(bays_pred)
evaluate(bays_pred)
##xgb
print("xgb")
#xgb = xgb.XGBRegressor(booster ="gbtree",subsample = 0.5,colsample_by=0.6, num_parallel_tree =200 ,num_boost_round =1, eta = 1, random_state =42)
#xgb.fit(X_train , Y_train)
#xgb_pred=xgb.predict(X_test)
#xgb_pred = data(xgb_pred)
#evaluate(xgb_pred)
## neural netwok
print("neural netwok")
nn = MLPClassifier(solver='adam',hidden_layer_sizes=(30,30,30,30,30,30,30,30), random_state=49,warm_start=False,activation='relu', max_iter=250,alpha=0.001 )
nn.fit(X , Y)
nn_pred=nn.predict(X_test)
nn_pred = data(nn_pred)
evaluate(nn_pred)


#clf_cv_score = cross_val_score(clf, X, Y, cv = 10, scoring = 'roc_auc', error_score = 'raise')
print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, Y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, Y_pred))
print('\n')
print("=== All AUC Scores ===")
#print(clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")

#print("Mean AUC Score - Random Forest: ", clf_cv_score.mean())

filename = '../input/inf131-2019/test.csv' 
df_test = pd.read_csv(filename)
df_test.head(1)
df_test.rename(columns={'weathersit':'weather','mnth':'month','hr':'hour','yr':'year','hum': 'humidity','cnt':'count'},inplace=True)
df_test = df_test.drop(['atemp','humidity'], axis=1)
df_test['season'] = df_test.season.astype('category')
df_test['year'] = df_test.year.astype('category')
df_test['month'] = df_test.month.astype('category')
df_test['hour'] = df_test.hour.astype('category')
df_test['holiday'] = df_test.holiday.astype('category')
df_test['weekday'] = df_test.weekday.astype('category')
df_test['workingday'] = df_test.workingday.astype('category')
df_test = pd.get_dummies(df_test)
print(df_test.head(1))
print(df_test.columns)
df_test = df_test[['temp', 'windspeed', 'season_1', 'season_2', 'season_3',
       'season_4', 'year_0', 'year_1', 'month_1', 'month_2', 'month_3',
       'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
       'month_10', 'month_11', 'month_12', 'hour_0', 'hour_1', 'hour_2',
       'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
       'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
       'hour_22', 'hour_23', 'holiday_0', 'holiday_1', 'weekday_0',
       'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
       'weekday_6', 'workingday_0', 'workingday_1', 'weather']]

Y_pred = np.zeros((5214,10))
Y_pred[:,0] = data(nn.predict(df_test))
Y_pred[:,1] = data(clf.predict(df_test))
Y_pred[:,2] = data(clf1.predict(df_test))
Y_pred[:,3] 
Y_pred[:,4]
Y_pred[:,5]
Y_pred[:,6]
Y_pred[:,7]
Y_pred[:,8]
Y_pred[:,9]
print(Y_pred)
print(type(Y_pred))
print(Y_pred.shape)
print(df_test.shape)
print(X_train.shape)
print(X.shape)
print(Y_pred.shape)
result=Y_pred[:,0]
for i in range(Y_pred.shape[0]):
        result[i]=(Y_pred[i,0]+(Y_pred[i,1]+Y_pred[i,2])/2)/2

submission = pd.DataFrame()
submission['Id'] = range(Y_pred.shape[0])
submission['Predicted'] = result
submission.shape
submission.to_csv("submission.csv", index=False)