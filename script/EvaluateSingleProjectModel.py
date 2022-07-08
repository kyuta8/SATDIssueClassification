import os
import argparse

import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold

from _EvaluateModel.DataLoader import DataLoader
from _EvaluateModel.ComplementNan import ComplementNan
# from _EvaluateModel.ReduceMemory import ReduceMemory
from _EvaluateModel.DataProcessing import DataProcessing
from _EvaluateModel.ClassificationModel import Model

from _common.CommonFunctions import *

import gc


ROOT_PATH = '.'
while True:
    dirs = os.listdir(ROOT_PATH)
    if 'data' in dirs:
        ROOT_PATH = os.path.join(ROOT_PATH, 'data')
        break
    else:
        ROOT_PATH = os.path.join(ROOT_PATH, '..')


def main(project: str, algorithm: str, k: int, iterate: int, **param):

    """ Load Dataset and Label """
    dl = DataLoader(**param['Loader'])
    dataset, label_df = dl.load(project, test_project='')
    class_label = label_df['Class'].tolist()

    tfidf = '_tfidf' if param['Loader']['tfidf'] else ''

    if not(os.path.exists('./../data/{}/single/dataset_{}{}.tmp.{}.pkl'.format(project, param['Loader']['n'], tfidf, param['Loader']['type']))):
        """ Complement Nan Values """
        cn = ComplementNan()
        dataset = cn.lr_comp(df=dataset, project=project, how='lasso', types='single', drop='without_outlier', tfidf=param['Loader']['tfidf'], random_state=42)
        # rm = ReduceMemory(dataset)
        # dataset = rm.execute()
        dataset = dataset.astype(np.float16)
        dataset.to_pickle('./../data/{}/single/dataset_{}{}.tmp.{}.pkl'.format(project, param['Loader']['n'], tfidf, param['Loader']['type']))
    
    """ Cross Validation """
    SKF = StratifiedKFold(n_splits=k, shuffle=True)
    cross_val = {i: list(SKF.split(list(dataset.values), class_label)) for i in range(iterate)}
    evaluate_dict = defaultdict(list)
    importance = defaultdict(list)
    classified_issues = pd.DataFrame(index=list(label_df['Issue']))
    for i in range(iterate):
        j = 0
        for train_index, test_index in cross_val[i]:
            index = k*i+j
            print('--{}--'.format(index+1))
            print('Train Data')
            print('  Positive:', label_df['Class'].loc[train_index].sum())
            print('  Negative:', len(train_index) - label_df['Class'].loc[train_index].sum())
            print()
            print('Train Data')
            print('  Positive:', label_df['Class'].loc[test_index].sum())
            print('  Negative:', len(test_index) - label_df['Class'].loc[test_index].sum())
            print()
            print('Features:', len(dataset.columns))
            print()

            """ Dataset """
            train_X = dataset.query("index in @train_index").values
            train_y = list(label_df['Class'].loc[train_index])
            test_X = dataset.query("index in @test_index").values
            metrics = list(dataset.columns)
            del dataset
            gc.collect()

            if param['Model']['how'] == 'over':
                dp = DataProcessing()
                train_X, train_y = dp.oversampling(train_x=train_X, train_label=train_y, ratio=0.5)
            elif param['Model']['how'] == 'under':
                dp = DataProcessing()
                train_X, train_y = dp.undersampling(train_x=train_X, train_label=train_y, ratio=0.1)
            elif param['Model']['how'] == 'smote':
                dp = DataProcessing()
                train_X, train_y = dp.SMOTE(train_x=train_X, train_label=train_y, ratio=0.1)
            elif param['Model']['how'] == 'smoteenn':
                dp = DataProcessing()
                train_X, train_y = dp.SMOTEENN(train_x=train_X, train_label=train_y, ratio=0.5)
            elif param['Model']['how'] == 'both':
                dp = DataProcessing()
                train_X, train_y = dp.undersampling(train_x=train_X, train_label=train_y, ratio=0.1)
                train_X, train_y = dp.SMOTE(train_x=train_X, train_label=train_y, ratio=0.5)

            """ Build Classification Model """
            model = Model()
            model.build(X=train_X, y=train_y, model=algorithm, **param['Model']['param'])

            """ Evaluate Classification Model """
            model.evaluate(X=test_X, y=label_df['Class'].loc[test_index], label=metrics)
            result = model.result(output=True)

            """ Save Evaluation Result """
            for key in result.keys():
                if key in ['train_accuracy', 'accuracy', 'precision', 'recall', 'f1']: evaluate_dict[key].append(result[key])
            classified_issues = pd.concat([classified_issues, pd.DataFrame(result['predicted_label'], index=list(label_df['Issue'].loc[test_index]))], axis=1)
            if 'importance' in result.keys(): 
                for w, v in result['importance'].items(): importance[w].append(v)

            j += 1
            if not(index+1==k*iterate):
                del train_X, test_X
                gc.collect
                dataset, _ = dl.load(project, test_project='')
                # dataset = pd.read_pickle('./../data/{}/single/dataset.tmp.{}.pkl'.format(project, param['Loader']['type']))

    """ Final Result """
    print('--Average--')
    print('Precision:', round(np.mean(evaluate_dict['precision']), 3))
    print('Recall:', round(np.mean(evaluate_dict['recall']), 3))
    print('F1:', round(np.mean(evaluate_dict['f1']), 3))
    print()
    
    """ Output Result """
    OPTION = ''
    if param['Loader']['type'] == 'without_outlier': OPTION += '/drop' 
    OPTION += '/{}gram'.format(param['Loader']['n'])

    SAVEPATH = os.path.join(ROOT_PATH, f'result/{project}/single{OPTION}/{algorithm}')
    os.makedirs(SAVEPATH, exist_ok=True)
    result_df = pd.DataFrame({'train_accuracy': evaluate_dict['train_accuracy'], 'accuracy': evaluate_dict['accuracy'], 
                              'precision': evaluate_dict['precision'], 'recall': evaluate_dict['recall'], 'f1': evaluate_dict['f1']})
    average = pd.DataFrame(columns=['train_accuracy', 'accuracy', 'precision', 'recall', 'f1'])
    for key in ['train_accuracy', 'accuracy', 'precision', 'recall', 'f1']: average.at[0, key] = np.mean(evaluate_dict[key])
    result_df = pd.concat([result_df, average])
    result_df.to_csv(SAVEPATH+'/text{}_reporter{}_process{}_change{}{}{}_{}.csv'.format(param['Loader']['text'], param['Loader']['reporter'], param['Loader']['process'], param['Loader']['code'], param['Model']['how'], tfidf, int(iterate*k)), index=None)
    # write_json(data=evaluate_dict, path=SAVEPATH+'/text{}_reporter{}_process{}_change{}{}_{}.json'.format(param['Loader']['text'], param['Loader']['reporter'], param['Loader']['process'], param['Loader']['code'], param['Model']['how'], int(iterate*k)))
    os.makedirs(SAVEPATH+'/classified_result', exist_ok=True)
    classified_issues.to_csv(SAVEPATH+'/classified_result/text{}_reporter{}_process{}_change{}{}{}_{}.csv'.format(param['Loader']['text'], param['Loader']['reporter'], param['Loader']['process'], param['Loader']['code'], param['Model']['how'], tfidf, int(iterate*k)), index=None)
    if 'importance' in result.keys(): 
        os.makedirs(SAVEPATH+'/importance', exist_ok=True)
        write_json(data=importance, path=SAVEPATH+'/importance/text{}_reporter{}_process{}_change{}{}{}_{}.json'.format(param['Loader']['text'], param['Loader']['reporter'], param['Loader']['process'], param['Loader']['code'], param['Model']['how'], tfidf, int(iterate*k)))

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TD-Issue Classification')
    parser.add_argument('-n', '--ngram', choices=['1', '2', 'n', 'embeded'], required=True, help='学習時のN-gramを指定')
    parser.add_argument('-m', '--model', help='Select Machine Learning Algorithm')
    parser.add_argument('-t', '--text', action='store_true', help='Use Text Metrics')
    parser.add_argument('-r', '--reporter', action='store_true', help='Use Reporter Metrics')
    parser.add_argument('-p', '--process', action='store_true', help='Use Process Metrics')
    parser.add_argument('-c', '--code', action='store_true', help='Use Source Code Metrics')
    parser.add_argument('-s', '--sampling', help='Select Imbalanced Data Processing')
    parser.add_argument('-tf', '--tfidf', action='store_true', help='Use tf-idf')
    parser.add_argument('--drop', action='store_true', help='Drop Data with Outliers')
    parser.add_argument('--project', help='Select Project')
    args = parser.parse_args()

    sampling = args.sampling if args.sampling else ''

    print('='*30)
    print('<CONFIGURATION>')
    print('Project:', args.project)
    print('Model:', args.model)
    print('Data-drop:', args.drop)
    print('Metrics:')
    print(' '*2, 'Text', args.text)
    print(' '*2, 'Reporter:', args.reporter)
    print(' '*2, 'Process:', args.process)
    print(' '*2, 'Change:', args.code)
    print('Sampling:')
    if sampling == 'under':
        print(' '*2, 'Undersampling')
    elif sampling == 'over':
        print(' '*2, 'Oversampling')
    elif sampling == 'smote':
        print(' '*2, 'SMOTE')
    elif sampling == 'smoteenn':
        print(' '*2, 'SMOTE-ENN')
    elif sampling == 'both':
        print(' '*2, 'Undersampling')
        print(' '*2, 'SMOTE')
    else:
        print(' '*2, 'Non-processing')
    print('='*30)
    print()

    if args.model == 'RF':
        # 'class_weight': 'balanced',
        model_param = {'n_estimators': 200, 
                        'bootstrap': True, 
                        'random_state': 42, 
                        'n_jobs': -1,
                        'max_features': 'sqrt',
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_depth': None}

    elif args.model == 'LR':
        # 'class_weight': 'blanced',
        model_param = {'solver': 'lbfgs', 
                       'penalty': 'l2',
                       'dual': False,  
                       'random_state': 42, 
                       'max_iter': 2000}

    elif args.model == 'SVM':
        # 'class_weight': 'balanced',
        model_param = {'max_iter': 2000, 
                       'dual': True, 
                       'penalty': 'l2', 
                       'random_state': 42}

    elif args.model == 'MLP':
        model_param = {'hidden_layer_sizes': (100, 100), 
                       'max_iter': 200,
                       'activation': 'logistic',
                       'verbose': False, 
                       'warm_start': False,
                       'random_state': 42
                       }

    else:
        model_param = {}

    type_ = 'without_outlier' if args.drop else 'plane'
    ngram_ = False if args.ngram==1 else True

    param = {'Loader': {'tfidf': args.tfidf, 'n': args.ngram, 'ngram': ngram_, 
                        'text': args.text, 'reporter': args.reporter, 'process': args.process, 'code': args.code, 'type': type_, 'how': 'single'}, 
             'Model': {'param': model_param, 'how': sampling}}

    k = 5
    iterate = int(100 / k)
    main(project=args.project, algorithm=args.model, k=k, iterate=iterate, **param)