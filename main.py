import pandas as pd
import xgboost as xgb
from matplotlib.projections import geo
from sklearn import preprocessing as pre
import numpy as np
from sklearn.metrics import mean_squared_log_error
import reverse_geocode as rg
from sklearn.model_selection import GridSearchCV

import sklearn
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
submission = pd.read_csv('submission_sample.csv')

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

# Date extraction
train['Date'] =pd.to_datetime(train['Date'])
train['year'] = train['Date'].dt.year
train['month'] = train['Date'].dt.month
train['date'] = train['Date'].dt.day
train = train.drop(['Date'], axis=1)

### copy
# test = test.drop(['ForecastId'], axis=1)
test['Date'] =pd.to_datetime(test['Date'])
test['year'] = test['Date'].dt.year
test['month'] = test['Date'].dt.month
test['date'] = test['Date'].dt.day
test = test.drop(['Date'], axis=1)

# Get city from lat and long
coords = train[['Lat', 'Long']].apply(tuple, axis=1).tolist()
results = rg.search(coords)
train['city'] = [x['city'] for x in results]
train = train.drop(['Lat'], axis=1)
train = train.drop(['Long'], axis=1)
##copy
coords = test[['Lat', 'Long']].apply(tuple, axis=1).tolist()
results = rg.search(coords)
test['city'] = [x['city'] for x in results]
test = test.drop(['Lat'], axis=1)
test = test.drop(['Long'], axis=1)


# Label encoder
labelencoder = LabelEncoder()
train['Province/State'] = labelencoder.fit_transform(train['Province/State'].astype(str))
train['Country/Region'] = labelencoder.fit_transform(train['Country/Region'].astype(str))
train['city'] = labelencoder.fit_transform(train['city'].astype(str))
##copy
labelencoder = LabelEncoder()
test['Province/State'] = labelencoder.fit_transform(test['Province/State'].astype(str))
test['Country/Region'] = labelencoder.fit_transform(test['Country/Region'].astype(str))
test['city'] = labelencoder.fit_transform(test['city'].astype(str))

X_validation = test.iloc[:,[1,2,3,4,5,6]]

# Train/test split
X = train.iloc[:,[0,1,4,5,6,7]]
Y = train.iloc[:,[2,3]]
X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.20, random_state=42)

print(X_train)
print(len(X_train))
scaler = pre.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_test_scaled)
## Regressor
# xgbr = xgb.XGBRegressor()
# reg = MLPRegressor(solver='lbfgs',alpha=0.001,hidden_layer_sizes=(150,))


ESTIMATORS = {
    # "MultiO/P AdaB": MultiOutputRegressor(AdaBoostRegressor(n_estimators=5))
    # "MultiO/P GBR" :MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
    # "RandomForestRegressor": RandomForestRegressor(max_depth=4, random_state=2),
    "Decision Tree Regressor":DecisionTreeRegressor(random_state=42,max_depth=25, max_features=5,
                                                    min_samples_leaf=8,min_samples_split=8)
    # "GBR" : MultiOutputRegressor(GradientBoostingRegressor(random_state=42,
    #                                                  learning_rate=0.01,
    #                                                  n_estimators = 500,
    #                                                  max_depth=3)),
}

y_test_predict = dict()
y_msle = dict()
#
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)                    # fit() with instantiated object
    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name
    print(estimator.predict(X_test_scaled))
    y_msle[name] = np.sqrt(mean_squared_log_error(y_test, estimator.predict(X_test)))
    print(estimator.predict(X_validation))
    results = pd.DataFrame(estimator.predict(X_validation))
    test = pd.concat([test,results], axis=1)
    test = test.iloc[:,[0,7,8]].astype(int)
    test.columns=['ForecastId', 'ConfirmedCases','Fatalities']
    test = test.round()


print(test)
test.to_csv('submission.csv',header=True,index=False)

#----------Parameters tuning----------#
param_grid = {'max_depth': range(8,40),
              'min_samples_split': range(8,10),
              'min_samples_leaf':range(8,10),
              'max_features':range(5,6)
              }

clf = GridSearchCV(DecisionTreeRegressor(),
                   param_grid,
                   scoring='neg_mean_squared_log_error',
                   cv=5, n_jobs=1, verbose=1)

clf.fit(X_train, y_train)

print(clf.best_score_)
print(clf.best_params_)
print(clf.error_score)
print(clf.cv_results_)



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

