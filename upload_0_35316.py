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
#xoris na ginoun kategory beltioueike elaxista
#season yr mnth hr holiday weekday workingday weathersit temp atemp hum windspeed casual registered cnt
dataframe_train = dataframe_train.drop(['atemp', 'casual', 'registered','humidity'], axis=1)
# Training and test data is created by splitting the main data. 30% of test data is considered
#dataframe_train=pd.get_dummies(dataframe_train,prefix=['co1-','col2-'])
#print(dataframe_train)
X = dataframe_train[['season','year','month','holiday','workingday', 'weather', 'temp','hour', 'windspeed']]
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
clf = RandomForestRegressor()

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



clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
True in (Y_pred < 0)
for i, Y  in enumerate(Y_pred):
    Y_pred[i]=int(np.around([Y_pred[i]]))
    if Y_pred[i] < 0:
        Y_pred[i] = 0
        
print('RMSLE:', np.sqrt(mean_squared_log_error(Y_test, Y_pred)))
print('R2:', r2_score(Y_test, Y_pred))


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
df_test = df_test[['season','year','month','holiday','workingday', 'weather', 'temp','hour', 'windspeed']]
#clf.fit(X, Y)
kf = KFold(n_splits=2)
for i, j in kf.split(X):
    print("%s %s" % (i, j))
Y_pred = np.zeros((5214,10))
Y_pred[:,0] = data(nn.predict(df_test))
Y_pred[:,1] = data(clf.predict(df_test))
#Y_pred[:,2] = bays_pred.predict(df_test)
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
        result[i]=(Y_pred[i,0]+Y_pred[i,1])/2

submission = pd.DataFrame()
submission['Id'] = range(Y_pred.shape[0])
submission['Predicted'] = result
submission.shape
submission.to_csv("submission.csv", index=False)

