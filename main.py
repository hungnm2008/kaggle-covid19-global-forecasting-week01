import pandas as pd
import xgboost as xgb
from matplotlib.projections import geo
from sklearn import preprocessing as pre
import numpy as np
from sklearn.metrics import mean_squared_log_error
import reverse_geocode as rg
from sklearn.model_selection import GridSearchCV
from datetime import datetime,timedelta
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
# Console ouput configuration
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


#----------Read input csv----------#
train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
submission = pd.read_csv('test.csv',index_col=False)

#----------Correlation analysis----------#
correlation = train.corr(method='pearson')
corr_cases = correlation.nlargest(6, 'ConfirmedCases').index
corr_fatalities = correlation.nlargest(6, 'Fatalities').index

correlation_map_cases = np.corrcoef(train[corr_cases].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map_cases, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=corr_cases.values, xticklabels=corr_cases.values)
plt.show()

#----------Features engineering----------#
train = train.drop(['Id'], axis=1)
train['ConfirmedCases'] = train['ConfirmedCases'].astype(int)
train['Fatalities'] = train['Fatalities'].astype(int)
#copy
submission = submission.drop(['ForecastId'], axis=1)


# Date extraction
FMT = '%Y-%m-%d'
date = train['Date']
train['day'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01", FMT)).days)
train = train.drop(['Date'], axis=1)
#copy
date = submission['Date']
submission['day'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01", FMT)).days)
submission = submission.drop(['Date'], axis=1)

# Get city from lat and long
coords = train[['Lat', 'Long']].apply(tuple, axis=1).tolist()
results = rg.search(coords)
train['city'] = [x['city'] for x in results]
train = train.drop(['Lat'], axis=1)
train = train.drop(['Long'], axis=1)
#copy
coords = submission[['Lat', 'Long']].apply(tuple, axis=1).tolist()
results = rg.search(coords)
submission['city'] = [x['city'] for x in results]
submission = submission.drop(['Lat'], axis=1)
submission = submission.drop(['Long'], axis=1)

# Label encoder
labelencoder = LabelEncoder()
train['Province/State'] = labelencoder.fit_transform(train['Province/State'].astype(str))
train['Country/Region'] = labelencoder.fit_transform(train['Country/Region'].astype(str))
train['city'] = labelencoder.fit_transform(train['city'].astype(str))
# copy
labelencoder = LabelEncoder()
submission['Province/State'] = labelencoder.fit_transform(submission['Province/State'].astype(str))
submission['Country/Region'] = labelencoder.fit_transform(submission['Country/Region'].astype(str))
submission['city'] = labelencoder.fit_transform(submission['city'].astype(str))

## Regressor
# y_test_predict = dic
y_msle = dict()
predictor= dict()
print(predictor)
print(submission)
i=0
for country in train['Country/Region'].unique():
    print('training model for country: '+str(country))
    country_train = train['Country/Region']==country
    country_train = train[country_train]
    province_train = country_train['Province/State']==128

    if province_train.empty:
        # Train/test split
        X = country_train.iloc[:, [0, 1, 4, 5]]
        y = country_train.iloc[:, [2, 3]]
        X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state=42)

        # Scaling
        scaler = pre.MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        # param_grid = {'n_estimators': range(9, 11),
        #               'max_depth': range(6, 10)
        #               }
        #
        # clf = GridSearchCV(estimator,
        #                    param_grid,
        #                    scoring='neg_mean_squared_log_error',
        #                    cv=5, n_jobs=4, verbose=1)
        # clf.fit(X_train_scaled, y_train)
        # print(clf.best_params_)

        estimator = RandomForestRegressor(random_state=42, min_samples_split=5, min_samples_leaf=5, max_features=3,
                                          max_samples=0.9, max_depth=18, n_estimators=12)
        estimator.fit(X_train_scaled, y_train)  # fit() with instantiated object
        print(estimator.predict(X_test_scaled))
        predictor[country][province] = estimator
        y_msle[i] = np.sqrt(mean_squared_log_error(y_test, estimator.predict(X_test_scaled)))
        print(y_msle[i])
        i += 1
        strg = str(country)+ str(128)
        predictor[strg] = estimator


    else:
        for province in country_train['Province/State'].unique():
            print('training model for province: ' + str(country))
            province_train = country_train['Province/State'] == province
            province_train = country_train[province_train]
            # Train/test split
            X = province_train.iloc[:, [0, 1, 4, 5]]
            y = province_train.iloc[:, [2, 3]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Scaling
            scaler = pre.MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)

            # param_grid = {'n_estimators': range(9, 20),
            #               'max_depth': range(6, 20)
            #               }
            #
            # clf = GridSearchCV(estimator,
            #                    param_grid,
            #                    scoring='neg_mean_squared_log_error',
            #                    cv=5, n_jobs=4, verbose=1)
            # clf.fit(X_train_scaled, y_train)
            # print(clf.best_params_)
            estimator = RandomForestRegressor(random_state=42, min_samples_split=5, min_samples_leaf=5, max_features=3,
                                              max_samples=0.9, max_depth=18, n_estimators=12)

            estimator.fit(X_train_scaled, y_train)  # fit() with instantiated object

            print(estimator.predict(X_test_scaled))
            y_msle[i] = np.sqrt(mean_squared_log_error(y_test, estimator.predict(X_test_scaled)))
            print(y_msle[i])
            i += 1
            strg = str(country) + str(province)
            predictor[strg] = estimator

#-------Predicting------#

submission.reset_index(drop=True, inplace=True)
print(submission)
result = pd.DataFrame(columns = ['ConfirmedCases', 'Fatalities'])
for a in submission.itertuples():
    x = a
    predictor_id = str(x[2]) + str(x[1])
    # a = pd.DataFrame(a).iloc[:,[0,1,2,3]]
    x = np.reshape(x, (1, -1))
    x = np.delete(x, 1, 1)
    out = pd.DataFrame(predictor[predictor_id].predict(x))
    result = pd.concat([result,out], axis=0)
    print(result)

submission_ = pd.concat([submission,result], axis=1)
submission_.to_csv('submission.csv',header=True,index=False)
    # test = test.iloc[:,[0,7,8]].astype(int)
    # test.columns=['ForecastId', 'ConfirmedCases','Fatalities']
    # test = test.round()



#----------Parameters tuning----------#
# param_grid = {'max_depth': range(8,40),
#               'min_samples_split': range(8,10),
#               'min_samples_leaf':range(8,10),
#               'max_features':range(5,6)
#               }
#
# clf = GridSearchCV(DecisionTreeRegressor(),
#                    param_grid,
#                    scoring='neg_mean_squared_log_error',
#                    cv=5, n_jobs=1, verbose=1)
#
# clf.fit(X_train, y_train)
#
# print(clf.best_score_)
# print(clf.best_params_)
# print(clf.error_score)
# print(clf.cv_results_)



#
# GBR = MultiOutputRegressor(GradientBoostingRegressor(random_state=42,
#                                                      learning_rate=0.01,
#                                                      n_estimators = 500,
#                                                      max_depth=3))
# # RFR = RandomForestRegressor()
# GBR.fit(X_train_scaled, y_train)
#
# ## Evaluate
# y_pred = GBR.predict(X_test)
# scores = cross_val_score(GBR, X_test_scaled, y_test, cv=10)# print(train.describe())
# print(scores.mean())
# print(scores)


# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print(roc_auc)

