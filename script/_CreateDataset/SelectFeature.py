'''
This is for selecting (extracting) effective features in each project's dataset.

Two main functions are implemented:
 - _test() performs a statistical test and returns effective features.
 - _vif() calculates the correlation and returns independent variables (explanatory variables).
'''


import os
import sys
sys.path.append(os.path.abspath('.').rsplit('/', 1)[0])
from pprint import pprint
pprint(sys.path)

import json
import pickle

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter

from typing import Tuple


REPORTER = ['Experience', 'OpenIssueNum', 'OpenPullRequestNum', 'CommitNum', 'Member', 'Contributor', 'Collaborator']
PROCESS = ['TitleLen', 'DescriptionLen', 'AssigneeNum', 'ParticipantNum', 'CommentNum', 'SelfAssign', 'pCommitNum', 'ChangeFileNum', 'ResolutionTime']
# PROCESS = ['CommentNum', 'ChangeFileNum', 'SelfAssign', 'Link', 'ResolutionTime', 'Image', 'DescriptionLen', 'CodeLink', 'AssigneeNum', 'pCommitNum', 'CodeSnipet', 'ParticipantNum', 'TitleLen']
CODE = ['CountDeclClassAdd', 'CountDeclClassDel', 'CountDeclFunctionAdd', 'CountDeclFunctionDel', 'CountLineCodeAdd', 'CountLineCodeDel', 'CodeChurnAdd', 'CodeChurnDel', 'CountLineCommentAdd', 'CountLineCommentDel', 'CyclomaticAdd', 'CyclomaticDel', 'CyclomaticModifiedAdd', 'CyclomaticModifiedDel', 'CyclomaticStrictAdd', 'CyclomaticStrictDel', 'EssentialAdd', 'EssentialDel', 'MaxNestingAdd', 'MaxNestingDel']


class SelectFeature(object):

    def __init__(self) -> None:
        pass
        

    def _test(self, dataset, project, NORM, OUTPUT, OUTPUT_PATH) -> list:
        columns = list(dataset.columns)
        types = list(dataset.dtypes)
        _dataset = dataset.copy()
        OUTPUT_OPTION = ''

        # Normalization: Aligns a scale of each metrics
        if NORM:
            OUTPUT_OPTION = '_normalization'
        else:
            OUTPUT_OPTION = '_without_outlier'

            # scale_col = []
            # for col, t in zip(columns, types):
            #     if (col in ['Issue', 'Class', 'AuthorType']) or (t == bool):
            #         pass
            #     else:
            #         scale_col.append(col)
                    
            # mmscaler = MinMaxScaler()
            # _dataset.loc[:, scale_col] = mmscaler.fit_transform(_dataset.loc[:, scale_col].T.values)

            # sc = StandardScaler()
            # _dataset.loc[:, scale_col] = sc.fit_transform(_dataset.loc[:, scale_col].values)

        td = _dataset[_dataset['Class']==True]
        ntd = _dataset[_dataset['Class']==False]

        select = {}
        result = []
        outlier = {}
        for col, t in zip(columns, types):
            try:
                if col in ['Issue', 'Class']:
                    continue

                flag = False # Significant difference
                p_sign = '-'
                size = '-'

                print(col)
                td_ = list(td[col].dropna())
                ntd_ = list(ntd[col].dropna())

                if t == bool:
                    print('  - TD:', len(td_), np.median(td_))
                    print('  - Non-TD:', len(ntd_), np.median(ntd_))

                    data = np.matrix([ [ sum(td_), len(td_)-sum(td_) ], [ sum(ntd_), len(ntd_)-sum(ntd_) ] ])
                    chi2, p, ddof, expected = stats.chi2_contingency( data , correction=False)
                    delta = np.sqrt(chi2 / (len(td_) + len(ntd_)))
                    msg = "Test Statistic: {}\np-value: {}\nDegrees of Freedom: {}\nEffect Size: {}"
                    print( msg.format( chi2, p, ddof, delta ) )
                    # print( expected )
                    if abs(delta) >= 0.1:
                        flag = True
                        effect = abs(delta)
                        if 0.5 <= abs(delta):
                            size = 'Large'
                        elif (0.3 <= abs(delta)) and (abs(delta) < 0.5):
                            size = 'Medium'
                        elif (0.1 <= abs(delta)) and (abs(delta) < 0.3):
                            size = 'Small'
                    else:
                        size = 'Negligible'

                else:
                    if not(NORM):
                        _td = pd.DataFrame({col: td_})
                        below = _td[col].quantile(0.25) - (_td[col].quantile(0.75) - _td[col].quantile(0.25)) * 1.5
                        above = _td[col].quantile(0.75) + (_td[col].quantile(0.75) - _td[col].quantile(0.25)) * 1.5
                        td_ = [i for i in td_ if below <= i if i <= above]

                        _ntd = pd.DataFrame({col: ntd_})
                        below = _ntd[col].quantile(0.25) - (_ntd[col].quantile(0.75) - _ntd[col].quantile(0.25)) * 1.5
                        above = _ntd[col].quantile(0.75) + (_ntd[col].quantile(0.75) - _ntd[col].quantile(0.25)) * 1.5
                        ntd_ = [i for i in ntd_ if below <= i if i <= above]

                        outlier[col] = (len(_td) - len(td_)) + (len(_ntd) - len(ntd_))

                    print('  - TD:', len(td_))
                    print('    -> Median: {}'.format(np.median(td_)))
                    print('    -> Variance: {}'.format(np.var(td_)))
                    print('  - Non-TD:', len(ntd_))
                    print('    -> Median: {}'.format(np.median(ntd_)))
                    print('    -> Variance: {}'.format(np.var(ntd_)))

                    if (len(td_) < 10) or (len(ntd_) < 10):
                        p, delta = np.nan, np.nan
                    else:
                        U, p = stats.mannwhitneyu(td_, ntd_, alternative='two-sided')
                        delta = 2 * U / (len(td_) * len(ntd_)) - 1
                        msg = "Test Statistic: {}\np-value: {}\nEffect Size: {}"
                        print( msg.format( U, p, delta ) )
                        if abs(delta) >= 0.147:
                            flag = True
                            effect = abs(delta)
                            if 0.474 <= abs(delta):
                                size = 'Large'
                            elif (0.330 <= abs(delta)) and (abs(delta) < 0.474):
                                size = 'Medium'
                            elif (0.147 <= abs(delta)) and (abs(delta) < 0.330):
                                size = 'Small'
                        else:
                            size = 'Negligible'

                if p < 0.01:
                    print('Significant Difference: **')
                    p_sign = '**'
                    if flag:
                        print("***This metrics is valid.***")
                        select[col] = effect
                elif (0.01 < p) and (p < 0.05):
                    p_sign = '*'
                    print('Significant Difference: *')
                else:
                    p_sign = ''
                    print('Significant Difference: ')

                result.append({'metrics': col, 'td': len(td_), 'non_td': len(ntd_), 'p': round(p, 3), 'delta': round(delta, 3), 'sign': p_sign, 'size': size, 'significance': flag})

                print('-'*40)

            except Exception as e:
                print(e)
                result.append({'metrics': col, 'td': len(td_), 'non_td': len(ntd_), 'p': np.nan, 'delta': np.nan, 'sign': p_sign, 'size': size, 'significance': flag})
                print("This metrics is invalid.")
                print('-'*40)

        # if OUTPUT:
        tmp_df = pd.json_normalize(result)
        df = pd.DataFrame()
        cols = REPORTER + PROCESS + CODE
        for col in cols: df = pd.concat([df, tmp_df[tmp_df['metrics']==col]])
        df.to_csv(OUTPUT_PATH+'/{}/statistical_test_result{}.csv'.format(project, OUTPUT_OPTION), index=None)
            # df.to_csv(OUTPUT_PATH+'/additional/statistical_test_result_{}{}.csv'.format(project, OUTPUT_OPTION), index=None)

        return list(select.keys())
    
    
    def _vif(self, dataset, _vif=10) -> list:
        _cols = set(dataset.columns) - set(['Issue', 'Class'])
        data_x = dataset.loc[:, _cols]
        data_x = data_x.fillna(data_x.median())
        data_x = data_x.astype(float)
        data_y = pd.DataFrame(list(dataset['Class']), columns=['target'])

        #vifを計算する
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(data_x.values, i) for i in range(data_x.shape[1])]
        vif["features"] = data_x.columns

        #vifを計算結果を出力する
        # print('Before')
        # print(vif)

        while vif['VIF Factor'].max() >= _vif:
            drop_col = vif['features'][vif['VIF Factor']==vif['VIF Factor'].max()].iat[0]
            data_x = data_x.drop([drop_col], axis=1)

            #vifを計算する
            vif = pd.DataFrame()
            vif["VIF Factor"] = [variance_inflation_factor(data_x.values, i) for i in range(data_x.shape[1])]
            vif["features"] = data_x.columns
        
        return list(vif['features'])
    

    def _remove_outlier(self, dataset) -> Tuple[list, list, list, list, list]:
        columns = list(dataset.columns)
        types = list(dataset.dtypes)
        _dataset = dataset.copy()

        p_index = dataset[dataset['Class']==True].index
        n_index = dataset[dataset['Class']==False].index
        # for col in dataset.columns:
        #     if (col in CODE) or (col in ['CommitNum', 'Experience']):
        #         try:
        #             dataset.loc[p_index, col] = dataset.loc[p_index, col].fillna(dataset.loc[p_index, col].median())
        #         except:
        #             dataset.loc[p_index, col] = 0

        #         try:
        #             dataset.loc[n_index, col] = dataset.loc[n_index, col].fillna(dataset.loc[n_index, col].median())
        #         except:
        #             dataset.loc[n_index, col] = 0

        td_ = _dataset[_dataset['Class']==True]
        ntd_ = _dataset[_dataset['Class']==False]
        # dataset = dataset.fillna(dataset.median())

        _drop_index = set([])
        drop_reporter = []
        drop_process = []
        drop_code = []
        for col, t in zip(columns, types):
            if (col in ['Issue', 'Class', 'AuthorType']) or (t == bool):
                continue

            print(col)
            drop_index = set([])
            td = td_.dropna(subset=[col])
            below = td[col].quantile(0.25) - (td[col].quantile(0.75) - td[col].quantile(0.25)) * 1.5
            above = td[col].quantile(0.75) + (td[col].quantile(0.75) - td[col].quantile(0.25)) * 1.5
            _drop_index = _drop_index | set(td[(td[col] < below) | (above < td[col])].index)
            tmp = list(td[(td[col] < below) | (above < td[col])].index)
            drop_index = drop_index | set(tmp)
            if col in REPORTER:
                drop_reporter += tmp
            elif col in PROCESS:
                drop_process += tmp
            elif col in CODE:
                drop_code += tmp

            ntd = ntd_.dropna(subset=[col])
            below = ntd[col].quantile(0.25) - (ntd[col].quantile(0.75) - ntd[col].quantile(0.25)) * 1.5
            above = ntd[col].quantile(0.75) + (ntd[col].quantile(0.75) - ntd[col].quantile(0.25)) * 1.5
            _drop_index = _drop_index | set(ntd[(ntd[col] < below) | (above < ntd[col])].index)
            tmp = list(ntd[(ntd[col] < below) | (above < ntd[col])].index)
            drop_index = drop_index | set(tmp)
            if col in REPORTER:
                drop_reporter += tmp
            elif col in PROCESS:
                drop_process += tmp
            elif col in CODE:
                drop_code += tmp

            # 外れ値の置換
            dataset.loc[drop_index, col] = None

        # 外れ値の除去
        if _drop_index:
            _drop_index = list(_drop_index)
            # outlier_df = dataset.query('index in @_drop_index')
            # dataset = dataset.drop(_drop_index, axis=0)

        return dataset, drop_reporter, drop_process, drop_code, _drop_index


    def select(self, dataset, project, test_dataset, test_project, NORM=False, OUTPUT=False) -> None:
        print('='*40)
        print('Project:', project)
        print('Test Project:', test_project)
        print('-'*40)

        OUTPUT_PATH = '.'
        while True:
            dirs = os.listdir(OUTPUT_PATH)
            if 'data' in dirs:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'data')
                break
            else:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, '..')

        types = list(dataset.dtypes)
        pprint(dataset.dtypes)

        # """ Log Transformation """
        # ss_cols = []
        # for c, t in zip(dataset.columns, types):
        #     if (t != bool) and not(c in ['Issue', 'Class']):
        #         ss_cols += [c]
        #         # dataset[c] = dataset[c].apply(lambda x: np.log(x+1) if type(x) in [int, float] else x)
        
        # ss = StandardScaler()
        # dataset.loc[:, ss_cols] = ss.fit_transform(dataset.loc[:, ss_cols].values)

        _select_features = self._test(dataset, project, NORM, OUTPUT, OUTPUT_PATH)
        __select_features = set(_select_features) | set(['Issue', 'Class'])
        _vif_features = self._vif(dataset.loc[:, __select_features], _vif=3)

        dataset.loc[:, set(_select_features) & set(_vif_features)].to_csv(OUTPUT_PATH+'/{}/metrics_plane.csv'.format(project), index=None)
        
        if OUTPUT:
            if NORM:
                OUTPUT_OPTION = '_normalization'
            else:
                OUTPUT_OPTION = '_without_outlier'

            final_cols = set(_select_features) & set(_vif_features) | set(['Issue', 'Class'])
            # final_cols = set(_select_features) | set(['Issue', 'Class'])
            

            """ Train Data Creation """
            dataset = dataset.loc[:, final_cols]
            dataset, drop_reporter, drop_process, drop_code, drop_index = self._remove_outlier(dataset=dataset)

            dataset['ReporterOutlier'] = False
            dataset['ProcessOutlier'] = False
            dataset['CodeOutlier'] = False
            dataset.loc[set(drop_reporter), 'ReporterOutlier'] = True
            dataset.loc[set(drop_process), 'ProcessOutlier'] = True
            dataset.loc[set(drop_code), 'CodeOutlier'] = True
            dataset.loc[:, ['ReporterOutlier', 'ProcessOutlier', 'CodeOutlier']].to_csv(OUTPUT_PATH+'/{}/single/outliers.csv'.format(project), index=None)

            # dataset = dataset.drop(drop_index, axis=0)
            dataset.loc[:, ['Issue', 'Class']].to_csv(OUTPUT_PATH+'/{}/single/class_labels.csv'.format(project), index=None)
            cols = set(_select_features) & set(_vif_features)
            dataset.loc[:, cols].to_csv(OUTPUT_PATH+'/{}/single/metrics{}.csv'.format(project, OUTPUT_OPTION), index=None)
            print('Feature Set')
            print(cols)
            print()

            issues = list(dataset.index)

            with open(OUTPUT_PATH+'/{}/dataset_1.pkl'.format(project), 'rb') as f:
                df = pickle.load(f)

            print('--- Train Data ---')
            print(f'Before: {len(df)}')
            print(df['Class'].value_counts())
            
            df = df.query('index in @issues')
            print(f'After: {len(df)}')
            print(df['Class'].value_counts())
            df.to_pickle(OUTPUT_PATH+'/{}/dataset_1{}.pkl'.format(project, OUTPUT_OPTION))
            print()


            """ Test Data Creation """
            test_dataset.loc[:, set(_select_features) & set(_vif_features)].to_csv(OUTPUT_PATH+'/{}/cross/{}/metrics_plane.csv'.format(project, test_project), index=None)
            test_dataset = test_dataset.loc[:, final_cols]
            test_dataset, test_drop_reporter, test_drop_process, test_drop_code, test_drop_index = self._remove_outlier(dataset=test_dataset)

            test_dataset['ReporterOutlier'] = False
            test_dataset['ProcessOutlier'] = False
            test_dataset['CodeOutlier'] = False
            test_dataset.loc[set(test_drop_reporter), 'ReporterOutlier'] = True
            test_dataset.loc[set(test_drop_process), 'ProcessOutlier'] = True
            test_dataset.loc[set(test_drop_code), 'CodeOutlier'] = True
            test_dataset.loc[:, ['ReporterOutlier', 'ProcessOutlier', 'CodeOutlier']].to_csv(OUTPUT_PATH+'/{}/cross/{}/outliers.csv'.format(project, test_project), index=None)

            test_dataset.loc[:, ['Issue', 'Class']].to_csv(OUTPUT_PATH+'/{}/cross/{}/class_labels.csv'.format(project, test_project), index=None)
            cols = set(_select_features) & set(_vif_features)
            test_dataset.loc[:, cols].to_csv(OUTPUT_PATH+'/{}/cross/{}/metrics{}.csv'.format(project, test_project, OUTPUT_OPTION), index=None)
            print()
            print('Feature Set')
            print(cols)
            print()

            issues = list(test_dataset.index)

            with open(OUTPUT_PATH+'/{}/dataset_1.pkl'.format(test_project), 'rb') as f:
                df = pickle.load(f)

            print('--- Test Data ---')
            print(f'Before: {len(df)}')
            print(df['Class'].value_counts())
            
            df = df.query('index in @issues')
            print(f'After: {len(df)}')
            print(df['Class'].value_counts())
            df.to_pickle(OUTPUT_PATH+'/{}/cross/{}/dataset_1{}.pkl'.format(project, test_project, OUTPUT_OPTION))
            print()


class CrossProjectSelectFeature(SelectFeature):

    def select(self, dataset, project, test_dataset, test_project, NORM=False, OUTPUT=False) -> None:
        print('='*40)
        print('Project:', project)
        print('Test Project:', test_project)
        print('-'*40)

        OUTPUT_PATH = '.'
        while True:
            dirs = os.listdir(OUTPUT_PATH)
            if 'data' in dirs:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'data')
                break
            else:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, '..')

        # for col in dataset.columns:
        #     if ('Add' in col) or ('Del' in col) or (col == 'ChangeFileNum'):
        #     # if col in ['ChangeFileNum', 'CodeChurnAdd', 'CodeChurnDel']:
        #         dataset[col] = dataset[col].fillna(0)
        # rep_index = dataset[dataset['pCommitNum']==0].index
        # dataset.loc[rep_index, ['pCommitNum', 'ChangeFileNum']] = None

        _select_features = super()._test(dataset, project, NORM, OUTPUT, OUTPUT_PATH)
        _vif_features = super()._vif(dataset, _vif=3)

        dataset.loc[:, set(_select_features) & set(_vif_features)].to_csv(OUTPUT_PATH+'/{}/metrics_.csv'.format(project), index=None)
        
        if OUTPUT:
            if NORM:
                OUTPUT_OPTION = '_normalization'
            else:
                OUTPUT_OPTION = '_without_outlier'

            final_cols = set(_select_features) & set(_vif_features) | set(['Issue', 'Class'])
            

            """ Train Data Creation """
            dataset = dataset.loc[:, final_cols]
            dataset, drop_reporter, drop_process, drop_code, drop_index = self._remove_outlier(dataset=dataset)

            dataset['ReporterOutlier'] = False
            dataset['ProcessOutlier'] = False
            dataset['CodeOutlier'] = False
            dataset.loc[set(drop_reporter), 'ReporterOutlier'] = True
            dataset.loc[set(drop_process), 'ProcessOutlier'] = True
            dataset.loc[set(drop_code), 'CodeOutlier'] = True
            dataset.loc[:, ['ReporterOutlier', 'ProcessOutlier', 'CodeOutlier']].to_csv(OUTPUT_PATH+'/{}/outliers.csv'.format(project), index=None)

            dataset = dataset.drop(drop_index, axis=0)
            dataset.loc[:, ['Issue', 'Class']].to_csv(OUTPUT_PATH+'/{}/class_labels{}.csv'.format(project, OUTPUT_OPTION), index=None)
            cols = set(_select_features) & set(_vif_features)
            dataset.loc[:, cols].to_csv(OUTPUT_PATH+'/{}/metrics{}.csv'.format(project, OUTPUT_OPTION), index=None)
            print('Feature Set')
            print(cols)
            print()

            issues = list(dataset.index)

            with open(OUTPUT_PATH+'/{}/dataset_1.pkl'.format(project), 'rb') as f:
                df = pickle.load(f)

            print('--- Train Data ---')
            print(f'Before: {len(df)}')
            print(df['Class'].value_counts())
            
            df = df.query('index in @issues')
            print(f'After: {len(df)}')
            print(df['Class'].value_counts())
            df.to_pickle(OUTPUT_PATH+'/{}/dataset_1{}.pkl'.format(project, OUTPUT_OPTION))
            print()


            """ Test Data Creation """
            test_dataset = test_dataset.loc[:, final_cols]
            test_dataset, test_drop_reporter, test_drop_process, test_drop_code, test_drop_index = self._remove_outlier(dataset=test_dataset)

            test_dataset['ReporterOutlier'] = False
            test_dataset['ProcessOutlier'] = False
            test_dataset['CodeOutlier'] = False
            test_dataset.loc[set(test_drop_reporter), 'ReporterOutlier'] = True
            test_dataset.loc[set(test_drop_process), 'ProcessOutlier'] = True
            test_dataset.loc[set(test_drop_code), 'CodeOutlier'] = True
            test_dataset.loc[:, ['ReporterOutlier', 'ProcessOutlier', 'CodeOutlier']].to_csv(OUTPUT_PATH+'/cross/{}/outliers.csv'.format(test_project), index=None)

            test_dataset = test_dataset.drop(test_drop_index, axis=0)
            test_dataset.loc[:, ['Issue', 'Class']].to_csv(OUTPUT_PATH+'/cross/{}/class_labels{}.csv'.format(test_project, OUTPUT_OPTION), index=None)
            cols = set(_select_features) & set(_vif_features)
            test_dataset.loc[:, cols].to_csv(OUTPUT_PATH+'/cross/{}/metrics{}.csv'.format(test_project, OUTPUT_OPTION), index=None)
            print()
            print('Feature Set')
            print(cols)
            print()

            issues = list(dataset.index)

            with open(OUTPUT_PATH+'/cross/{}/dataset_1.pkl'.format(test_project), 'rb') as f:
                df = pickle.load(f)

            print('--- Test Data ---')
            print(f'Before: {len(df)}')
            print(df['Class'].value_counts())
            
            df = df.query('index in @issues')
            print(f'After: {len(df)}')
            print(df['Class'].value_counts())
            df.to_pickle(OUTPUT_PATH+'/cross/{}/dataset_1{}.pkl'.format(test_project, OUTPUT_OPTION))
            print()