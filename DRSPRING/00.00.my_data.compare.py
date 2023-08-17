
# 다른 방법론 비교?

1) RandomForest 
2) SVM 
3) MLP
4) GBM
5) XGB
6) adaboost 

1) audnnsynergy
2) elastic net
3) svr 
4) random forest 
5) xgboost 

xgboost 
MLP
SVM
random forest 
lasso
elastic net 

prodeepsyn 에서 쓴 내용
Elastic Net alpha  {0.1; 1; 10; 100}
Elastic Net lambda {0.3; 0.5; 0.7; 0.9}
SVR v {0.01; 0.05; 0.1; 0.5}
SVR C {0.001; 0.01; 1; 10}
Random Forest number of trees {128; 256; 512; 1024}
Random Forest number of features (n) {√n; log2n; 256; 512}
XGBoost number of trees {128; 256; 512; 1024}
XGBoost learning rate {0.001; 0.01; 0.1; 1}


그래도 혹시 모르니 하나씩은 짜둘까 싶기도 하고 
다른 논문에서는 이런 표가 들어가니까.. 



[ Random Forest ]
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
X = data.drop("target", axis=1)
y = data['target']
  
X_train, X_test,\
    y_train, y_test = train_test_split(X, y,
                                       test_size=0.25,
                                       random_state=42)
X_train.shape, X_test.shape

model = RandomForestClassifier()
model.fit(X_train, y_train)
  
# predict the mode
y_pred = model.predict(X_test)
  
# performance evaluatio metrics
print(classification_report(y_pred, y_test))

param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

model_grid = RandomForestClassifier(max_depth=9,
                                    max_features="log2",
                                    max_leaf_nodes=9,
                                    n_estimators=25)
model_grid.fit(X_train, y_train)
y_pred_grid = model.predict(X_test)
print(classification_report(y_pred_grid, y_test))



[ XGBOOST]

import pandas as pd
import random
import os
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import sklearn.metrics as metrics

import xgboost as xgb



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정


DATA_PATH = 'D:\\Data\\LGAI_AutoDriveSensors\\'

# DATA
train_df = pd.read_csv(DATA_PATH + 'train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

def modelfit(pip_xgb, grid_param_xgb, x, y) : 
    gs_xgb = (GridSearchCV(estimator=pip_xgb,
                        param_grid=grid_param_xgb,
                        cv=4,
                        # scoring='neg_mean_squared_error',
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        verbose=10))

    gs_xgb = gs_xgb.fit(x, y)
    print('Train Done.')

    #Predict training set:
    y_pred = gs_xgb.predict(x)

    #Print model report:
    print("\nModel Report")
    print("\nCV 결과 : ", gs_xgb.cv_results_)
    print("\n베스트 정답률 : ", gs_xgb.best_score_)
    print("\n베스트 파라미터 : ", gs_xgb.best_params_)


pip_xgb1 = Pipeline([('scl', StandardScaler()),
    ('reg', MultiOutputRegressor(xgb.XGBRegressor()))])
grid_param_xgb1 = {
    'reg__estimator__max_depth' : [5, 6, 7],
    #'reg__estimator__gamma' : [1, 0.1, 0.01, 0.001, 0.0001, 0],
    # 'reg__estimator__learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.08],
    # 'reg__estimator__subsample' : [0.4, 0.6, 0.8],
    # 'reg__estimator__colsample_bytree' : [0.2, 0.6, 0.8]
}

modelfit(pip_xgb1, grid_param_xgb1, train_x, train_y)







[ SVR ]

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVR

estimator=SVR(kernel='rbf')

param_grid={
            'C': [1.1, 5.4, 170, 1001],
            'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
            'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
        }


grid = GridSearchCV(

estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [1.1, 5.4, 170, 1001],
            'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
            'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

