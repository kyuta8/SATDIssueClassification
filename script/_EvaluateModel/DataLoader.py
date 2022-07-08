import os
import gc
from typing import Tuple

from pickle import dump, load
import pandas as pd
import numpy as np
import datatable as dt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer

from tqdm import tqdm

class DataLoader(object):

    def __init__(self, **param) -> None:
        """
            (parameter) : <Description>
            - TEXT : This parameter is whether text metrics are used or not. If you use text metrics for building a model, you set "True" value to "param['text']"
            - REPORTER : This parameter is whether reporter metrics are used or not. If you use reporter metrics for building a model, you set "True" value to "param['reporter']"
            - PROCESS : This parameter is whether process metrics are used or not. If you use process metrics for building a model, you set "True" value to "param['process']"
            - CODE : This parameter is whether source code metrics are used or not. If you use source code metrics for building a model, you set "True" value to "param['code']"
            - TYPE : This parameter is data type. You can change "TYPE" according to the type of data you want to learn. 
                     ex.) You set "without_outlier", if you want to drop data including outliers.

            - REPORTER_SET : This parameter is the set of reporter metrics
            - PROCESS_SET : This parameter is the set of process metrics
            - CODE_SET : This parameter is the set of source code metrics

            - ROOT_PATH : This parameter is the root path when loading or outputing data
        """
        self.TEXT = param['text']
        self.REPORTER = param['reporter']
        self.PROCESS = param['process']
        self.CODE = param['code']
        self.TYPE = param['type']
        self.HOW = param['how']
        self.ngram = param['n']
        self.tfidf = param['tfidf']
        # print('Text:', self.TEXT)
        # print('Reporter:', self.REPORTER)
        # print('Process:', self.PROCESS)
        # print('Source Code:', self.CODE)
        # print('Dataset Type:', self.TYPE)
        # print('How:', self.HOW)

        self.REPORTER_SET = set(['Experience', 'OpenIssueNum', 'OpenPullRequestNum', 'CommitNum', 'Member', 'Contributor', 'Collaborator'])
        self.PROCES_SET = set(['CommentNum', 'ChangeFileNum', 'SelfAssign', 'ResolutionTime', 'DescriptionLen', 'AssigneeNum', 'pCommitNum', 'ParticipantNum', 'TitleLen'])
        self.CODE_SET = set(['CountDeclFunctionAdd', 'CyclomaticAdd', 'CountDeclClassDel', 'CyclomaticStrictDel', 'EssentialDel', 'CountLineCodeDel', 'CountLineCommentDel', 
                             'CyclomaticDel', 'EssentialAdd', 'CountLineCommentAdd', 'CyclomaticModifiedDel', 'CyclomaticStrictAdd', 'CyclomaticModifiedAdd', 'MaxNestingAdd', 
                             'CodeChurnDel', 'MaxNestingDel', 'CountLineCodeAdd', 'CodeChurnAdd', 'CountDeclClassAdd', 'CountDeclFunctionDel'])

        self.ROOT_PATH = '.'
        while True:
            dirs = os.listdir(self.ROOT_PATH)
            if 'data' in dirs:
                self.ROOT_PATH = os.path.join(self.ROOT_PATH, 'data')
                break
            else:
                self.ROOT_PATH = os.path.join(self.ROOT_PATH, '..')

    
    def _load_text_metrics(self) -> pd.DataFrame:
        text_df = pd.DataFrame()

        """ Load Description Data """
        text_df = pd.concat([text_df, dt.fread(os.path.join(self.PATH, 'body_{}gram.csv'.format(self.ngram))).to_pandas()], axis=1)

        """ Load Title Data """
        text_df = pd.concat([text_df, dt.fread(os.path.join(self.PATH, 'title_{}gram.csv'.format(self.ngram))).to_pandas()], axis=1)

        # """ text """
        # text_df = pd.concat([text_df, pd.read_csv(os.path.join(self.PATH, 'title_text.csv'))], axis=1)
        # text_df = pd.concat([text_df, pd.read_csv(os.path.join(self.PATH, 'body_text.csv'))], axis=1)

        return text_df


    def _load_other_metrics(self) -> pd.DataFrame:
        """ Load Reporter, Process and Source Code Data """
        if self.HOW == 'cross':
            metrics_df = dt.fread(os.path.join(self.PATH, f'metrics_plane.csv')).to_pandas()
        elif self.HOW == 'single':
            metrics_df = dt.fread(os.path.join(self.PATH, f'metrics_{self.TYPE}.csv')).to_pandas()

        """ Drop Unuse Metrics """
        if not(self.REPORTER) or not(self.PROCESS) or not(self.CODE):
            drop_columns = set([])
            if not(self.REPORTER): drop_columns = drop_columns | self.REPORTER_SET
            if not(self.PROCESS): drop_columns = drop_columns | self.PROCES_SET
            if not(self.CODE): drop_columns = drop_columns | self.CODE_SET

            use_columns = set(metrics_df.columns) - drop_columns
            metrics_df = metrics_df.loc[:, use_columns]

        return metrics_df


    def _load_labels(self) -> pd.DataFrame:
        """ Load Reporter, Process and Source Code Data """
        label_df = dt.fread(os.path.join(self.PATH, 'class_labels.csv')).to_pandas()

        return label_df


    def _remove_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Load Outlier Data """
        outlier_df = dt.fread(os.path.join(os.path.join(self.ROOT_PATH, f'{self.project}/single'), 'outliers.csv')).to_pandas()

        """ Set Outlier Index """
        outlier_index = list(outlier_df[(outlier_df['ReporterOutlier']==True)|(outlier_df['ProcessOutlier']==True)|(outlier_df['CodeOutlier']==True)].index)

        """ Drop Data with Outliers """
        df.drop(outlier_index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


    def load(self, project: str, test_project: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if self.HOW == 'single':
            self.project = project
            self.PATH = os.path.join(self.ROOT_PATH, f'{project}/{self.HOW}')
        elif self.HOW == 'cross':
            self.project = test_project
            self.PATH = os.path.join(self.ROOT_PATH, f'{project}/{self.HOW}/{test_project}')

        tfidf = '_tfidf' if self.tfidf else ''

        if os.path.exists(self.PATH+f'/dataset_{self.ngram}{tfidf}.tmp.{self.TYPE}.pkl'):
            dataset = pd.read_pickle(self.PATH+f'/dataset_{self.ngram}{tfidf}.tmp.{self.TYPE}.pkl')
            """ Drop Unuse Metrics """
            # if not(self.REPORTER) or not(self.PROCESS) or not(self.CODE):
            drop_columns = set([])
            if not(self.REPORTER): drop_columns = drop_columns | self.REPORTER_SET
            if not(self.PROCESS): drop_columns = drop_columns | self.PROCES_SET
            if not(self.CODE): drop_columns = drop_columns | self.CODE_SET

            # print(drop_columns)
            drop_columns = set(dataset.columns) & drop_columns
            # dataset = dataset.loc[:, use_columns]
            dataset.drop(drop_columns, axis=1, inplace=True)

        else:
            dataset = pd.DataFrame()

            """ Text Metrics Data Loader """
            text_df = self._load_text_metrics()
            if self.tfidf:
                # sc = TfidfTransformer()
                # text_df.loc[:, :] = sc.fit_transform(text_df.values).toarray()

                if self.HOW == 'single':
                    sc = TfidfTransformer()
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/text_scaler_{self.ngram}{tfidf}.{self.TYPE}.pkl'), "wb") as f:
                        sc.fit(text_df.values)
                        dump(sc, f)
                    
                elif self.HOW == 'cross':
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/text_scaler_{self.ngram}{tfidf}.without_outlier.pkl'), "rb") as f:
                        sc = load(f)

                text_df.loc[:, :] = sc.transform(text_df.values).toarray()

            else:
                # sc = MinMaxScaler()
                # text_df.loc[:, :] = sc.fit_transform(text_df.values)

                if self.HOW == 'single':
                    sc = MinMaxScaler()
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/text_scaler_{self.ngram}{tfidf}.{self.TYPE}.pkl'), "wb") as f:
                            sc.fit(text_df.values)
                            dump(sc, f)
                    
                elif self.HOW == 'cross':
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/text_scaler_{self.ngram}{tfidf}.without_outlier.pkl'), "rb") as f:
                        sc = load(f)

                text_df.loc[:, :] = sc.transform(text_df.values)

            if self.TYPE=='without_outlier': 
                text_df = self._remove_outlier(text_df)
                if self.HOW == 'single':
                    sum_df = pd.DataFrame(list(dict(text_df.sum(axis=0)).items()), columns=['Column', 'Sum'])
                    sum_df.query('Sum==0').to_csv(self.PATH+'/drop_texts.csv', index=None)

                elif self.HOW == 'cross':
                    drop_text = pd.read_csv(self.ROOT_PATH+f'{project}/single/drop_texts.csv')
                    text_df = text_df.drop(list(drop_text['Column']), axis=1)
                    gc.collect()

            """ Other Metrics Data Loader """
            if self.REPORTER or self.PROCESS or self.CODE:
                metrics_df = self._load_other_metrics()
                if self.TYPE=='without_outlier': metrics_df = self._remove_outlier(metrics_df)
                # sc = MinMaxScaler()
                # for c, t in zip(metrics_df.columns, list(metrics_df.dtypes)):
                #         if not(t in [int, float, bool]):
                #             metrics_df[c] = metrics_df[c].apply(lambda x: float(x) if type(x) in [int, float, bool] else None)
                # metrics_df.loc[:, :] = sc.fit_transform(metrics_df.astype(float).values)
                
                if self.HOW == 'single':
                    sc = MinMaxScaler()
                    std = sc.fit_transform(metrics_df.astype(float).values)
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/minmax_scaler_{self.ngram}{tfidf}.{self.TYPE}.pkl'), "wb") as f:
                        dump(sc, f)
                    metrics_df.loc[:, :] = std
                elif self.HOW == 'cross':
                    with open(os.path.join(self.ROOT_PATH, f'{project}/single/minmax_scaler_{self.ngram}{tfidf}.without_outlier.pkl'), "rb") as f:
                        sc = load(f)
                    for c, t in zip(metrics_df.columns, list(metrics_df.dtypes)):
                        if not(t in [int, float, bool]):
                            metrics_df[c] = metrics_df[c].apply(lambda x: float(x) if type(x) in [int, float, bool] else None)
                    metrics_df.loc[:, :] = sc.transform(metrics_df.values)

                    # sc = MinMaxScaler()
                    # metrics_df.loc[:, :] = sc.fit_transform(metrics_df.values)

                dataset = pd.concat([text_df, metrics_df], axis=1)
                del metrics_df
            else:
                dataset = text_df

            del text_df
            gc.collect()

        """ Ground Truth Label Loader """
        label_df = self._load_labels()
        if self.TYPE=='without_outlier': label_df = self._remove_outlier(label_df)

        return dataset, label_df
