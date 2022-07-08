from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.svm import LinearSVC # Linear Support Vector Classification
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier # 不均衡データに対処したRandom Forest
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier # GradientBoost（勾配ブースティング）
from xgboost import XGBClassifier # XGBoost
from lightgbm import LGBMClassifier # light GradientBoost
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score

from time import time


class Model(object):

    def __init__(self) -> None:
        self.algorithm = {'SVM': LinearSVC, 
                          'LR': LogisticRegression, 
                          'DT': DecisionTreeClassifier, 
                          'RF': RandomForestClassifier, 
                          'BRF': BalancedRandomForestClassifier, 
                          'AB': AdaBoostClassifier, 
                          'GB': GradientBoostingClassifier, 
                          'XGB': XGBClassifier, 
                          'LGB': LGBMClassifier, 
                          'NB': MultinomialNB,
                          'MLP': MLPClassifier}

        self.evaluate_dict = defaultdict()

    
    def build(self, X: list or pd.DataFrame, y: list or pd.DataFrame, model: str, **param) -> None:
        X = X if type(X) in [list, np.ndarray] else X.values
        y = y if type(y) in [list, np.ndarray] else y.values
        self.model = self.algorithm[model](**param)
        print('Model Building...')
        start = time()
        self.model.fit(X, y)
        self.evaluate_dict['train_accuracy'] = self.model.score(X, y)
        end = time() - start
        print('-> Finish: {:,}m {:.2f}s'.format(int(end/60), end%60))
        print('Train Accuracy:', round(self.evaluate_dict['train_accuracy'], 3))


    def evaluate(self, X: list or pd.DataFrame, y: list or pd.DataFrame, label=None) -> None:
        X = X if type(X) in [list, np.ndarray] else X.values
        pred_y = self.model.predict(X)
        self.evaluate_dict['accuracy'] = accuracy_score(y, pred_y)
        self.evaluate_dict['precision'] = precision_score(y, pred_y)
        self.evaluate_dict['recall'] = recall_score(y, pred_y)
        self.evaluate_dict['f1'] = f1_score(y, pred_y)
        self.evaluate_dict['predicted_label'] = pred_y

        if (str(type(self.model)).split('.')[1] in ['tree', 'ensemble']) or (str(type(self.model)) in ['xgboost', 'lightgbm']):
            _imp = pd.DataFrame({'metrics': X.columns if type(X)==pd.DataFrame else label, 'importance': self.model.feature_importances_})
            _imp = _imp.sort_values(by='importance', ascending=False)
            print('Importance')
            print(_imp.head(n=5))
            print()
            importance = {}
            for w, v in _imp.values: importance[str(w)] = v
            self.evaluate_dict['importance'] = importance


    def result(self, output=True) -> dict:
        print('-'*30)
        print('Test Accuracy:', round(self.evaluate_dict['accuracy'], 3))
        print('Precision:', round(self.evaluate_dict['precision'], 3))
        print('Recall:', round(self.evaluate_dict['recall'], 3))
        print('F1:', round(self.evaluate_dict['f1'], 3))
        print('-'*30)
        print()

        if output: return self.evaluate_dict