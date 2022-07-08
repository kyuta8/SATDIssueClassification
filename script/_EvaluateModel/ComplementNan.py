import os
import gc

from pickle import dump, load
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class ComplementNan(object):

    def __init__(self) -> None:
        print('Start Complementig values...')
        self.model = {'linear': LinearRegression,
                      'ridge': Ridge,
                      'lasso': Lasso,
                      'elasticnet': ElasticNet}

        self.ROOT_PATH = '.'
        while True:
            dirs = os.listdir(self.ROOT_PATH)
            if 'data' in dirs:
                self.ROOT_PATH = os.path.join(self.ROOT_PATH, 'data')
                break
            else:
                self.ROOT_PATH = os.path.join(self.ROOT_PATH, '..')


    def lr_comp(self, df: pd.DataFrame, project: str, how='linear', types='single', drop='without_outlier', tfidf=False, **param) -> pd.DataFrame:
        """
            (parameter)[type] : <description>
            - train_metrics[set|list] : 
            - nan_metrics[set|list] : 
        """
        tfidf = '_tfidf' if tfidf else ''

        train_metrics = set(df.dropna(how='any', axis=1).columns)
        nan_metrics = set(df.columns) - train_metrics
        print(nan_metrics)
        for metrics in nan_metrics:
            print('Metrics:', metrics)

            train_index = set(df.dropna(subset=[metrics], how='any', axis=0).index)
            nan_index = set(df.index) - train_index

            train_data_X = df.loc[train_index, train_metrics].values
            train_data_Y = df.loc[train_index, metrics].values
            test_data_X = df.loc[nan_index, train_metrics].values

            """ Validation for Linear Regression Model """
            # mae_mean = []
            # rmse_mean = []
            # score_mean = []
            # for _ in range(10):
            #     val_train_X, val_test_X, val_train_Y, val_test_Y = train_test_split(train_data_X, train_data_Y, train_size=0.9)

            #     lr = self.model[how](**param)
            #     lr.fit(val_train_X, val_train_Y)
            #     pred_Y = lr.predict(val_test_X)

            #     mae = mean_absolute_error(val_test_Y, pred_Y) # 平均絶対誤差(MAE: Mean Absolute Error)
            #     rmse = np.sqrt(mean_squared_error(val_test_Y, pred_Y)) # 平方根平均二乗誤差（RMSE: Root Mean Squared Error）
            #     score = lr.score(val_test_X, val_test_Y) # score

            #     print("MAE = %.2f,  RMSE = %.2f,  score = %.2f" % (mae, rmse, score))
            #     print("Coef = ", lr.coef_)
            #     print("Intercept =", lr.intercept_)
            #     print()

            #     mae_mean.append(mae)
            #     rmse_mean.append(rmse)
            #     score_mean.append(score)

            # print('---Average---')
            # print("MAE = %.2f,  RMSE = %.2f,  score = %.2f" % (round(np.mean(mae_mean), 3), round(np.mean(rmse_mean), 3), round(np.mean(score_mean), 3)))
            # print()

            if types == 'single':
                lr = self.model[how](**param)
                lr.fit(train_data_X, train_data_Y)
                with open(os.path.join(self.ROOT_PATH, f'{project}/single/complement{tfidf}.{metrics}.{drop}.pkl'), 'wb') as f:
                    dump(lr, f)
            
            elif types == 'cross':
                with open(os.path.join(self.ROOT_PATH, f'{project}/single/complement{tfidf}.{metrics}.{drop}.pkl'), 'rb') as f:
                    lr = load(f) 

            print('Complement NaN...')
            comp_Y = lr.predict(test_data_X)

            df.loc[nan_index, metrics] = comp_Y
            print('-> Finished')
            print()

        return df


