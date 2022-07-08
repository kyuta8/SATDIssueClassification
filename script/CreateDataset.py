"""
    データ収集にはGitHub REST APIを利用
"""


import os
import shutil
# import sys
import argparse
import gc


import random
import pickle
import json
import pandas as pd
import datatable as dt
import numpy as np
from scipy.sparse import data

from tqdm import tqdm

from _CreateDataset.Preprocessing import Preprocessing
from _CreateDataset.SelectFeature import *
from _CreateDataset.Vectorize import Vectorizer



# Creates a new directory
def _create_dir(path:str):
    _dirs = path.split('/')
    _path = _dirs[0]
    _epoch = 1
    _max_count = len(_dirs)
    while _epoch < _max_count:
        _path = os.path.join(_path, _dirs[_epoch])
        if not(os.path.exists(_path)):
            os.mkdir(_path)
        _epoch += 1


def read_json(path:str):
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def write_json(path:str, data:dict):
    with open(path, mode='w') as f:
        json.dump(data, f, indent=4)


def read_issues(project: str):
    path = './../data/issue/{}'.format(project)
    read_list = os.listdir(path)
    _issues = {}
    for f in read_list:
        if '.json' in f:
            _issues.update(read_json(os.path.join(path, f)))
            
    index = 0
    issues = {}
    for i in _issues:
        if (_issues[i]['state'] == 'closed') and not(_issues[i].get('pull_request')):
            issues.update({index: _issues[i]})
            index += 1
    return issues


def rand_ints_nodup(start, end, k):
    ns = []
    while len(ns) < k:
        num = random.randint(start, end)
        if not num in ns:
            ns.append(num)
    return ns


class createDataset(object):

    def __init__(self, TOKENIZE, LEMMA, NORM, TYPE, OUTPUT):
        self.TOKENIZE = TOKENIZE
        self.LEMMA = LEMMA
        self.NORM = NORM
        self.TYPE = TYPE
        self.OUTPUT = OUTPUT

    def _tokenize(self, project, **kwargs):
        pre = Preprocessing('large')
        with open('./../data/{}/dataset_1.pkl'.format(project), 'rb') as f:
            dataset = pickle.load(f)
        # print(dataset)
        if self.TOKENIZE:
            issues = read_issues(project)
            for i in tqdm(issues):
                num = str(issues[i]['number'])
                index = dataset[dataset['Issue']==num]#.query('@num == Issue')
                if len(index) == 0:
                    continue
                else:
                    index = index.index[0]

                words, word_num = pre.prep(issues[i]['title'], lemma=self.LEMMA, add=False, **kwargs)
                dataset['Title'].at[index] = words
                dataset['TitleLen'].at[index] = word_num

                body = issues[i]['body']
                if body:
                    words, word_num = pre.prep(body, lemma=self.LEMMA, add=True, **kwargs)
                    dataset['Description'].at[index] = words
                    dataset['DescriptionLen'].at[index] = word_num
                else:
                    dataset['Description'].at[index] = []
                    dataset['DescriptionLen'].at[index] = 0

        return dataset


    def _metrics(self, project):
        m_df = dt.fread('./../data/{}/metrics.csv'.format(project))
        m_df = m_df.to_pandas()

        """ Convert Data Types """
        convert_col = set(m_df.columns) - set(['AuthorType', 'SelfAssign'])
        for c_col in convert_col:
            if c_col in ['DescriptionLen', 'CommentNum', 'AssigneeNum', 'ParticipantNum', 'pCommitNum', 'ChangeFileNum']:
                m_df[c_col] = m_df[c_col].fillna(0)
                m_df[c_col] = m_df[c_col].astype(float)
            elif c_col in ['Member', 'Contributor', 'Collaborator']:
                m_df[c_col] = m_df[c_col].astype(bool)
            else:
                m_df[c_col] = m_df[c_col].apply(lambda x: x if type(x) in [float, int] else None)

        try:
            m_df = m_df.drop(['CheckBox', 'Link', 'Image', 'CodeSnipet', 'CodeLink'], axis=1)
            # m_df.loc[m_df[m_df['pCommitNum']>0].index, 'pCommitNum'] = True
            # m_df.loc[m_df[m_df['pCommitNum']==0].index, 'pCommitNum'] = False
            m_df.to_csv('./../data/{}/metrics.csv'.format(project), index=None)
        except:
            pass

        select = set(m_df.columns) - set(['CyclomaticAdd', 'CyclomaticDel', 'CyclomaticModifiedAdd', 'CyclomaticModifiedDel', 'CyclomaticStrictAdd', 'CyclomaticStrictDel'])
        m_df = m_df.loc[:, select]

        class_df = dt.fread('./../data/{}/class_labels.csv'.format(project))
        class_df = class_df.to_pandas()

        return m_df, class_df


    def main(self, project):
        """
            Type of Dataset Creation
                - single : create a dataset for each project
                - part : separate projects into 3 projects for a training data and 1 project for a testing data, and create each dataset
                - all : concatenate all projects and create a dataset, but output it after separating datasets for each project.
        """

        OUTPUT_OPTION = ''
        # Normalization: Aligns a scale of each metrics
        if self.NORM:
            OUTPUT_OPTION = '_normalization'
        # else:
        #     OUTPUT_OPTION = '_without_outlier'

        vr = Vectorizer()
        n = 1
        if (self.TYPE == 'single') or not(self.TYPE):
            for p in project:
                _create_dir('./../data/{}/single'.format(p))
                print('Project:', p)
                if self.OUTPUT:
                    dataset = self._tokenize(p, n=n, n_gram=True, variable=True)
                    m_df, _ = self._metrics(p)
                    m_df.loc[:, ['TitleLen', 'DescriptionLen']] = dataset.loc[:, ['TitleLen', 'DescriptionLen']]
                    m_df.to_csv('./../data/{}/metrics.csv'.format(p), index=None)
                    if self.LEMMA:
                        dataset.to_pickle('./../data/{}/dataset_{}{}.pkl'.format(p, n, OUTPUT_OPTION))
                    else:
                        dataset.to_pickle('./../data/{}/dataset_{}_for_bert{}.pkl'.format(p, n, OUTPUT_OPTION))
                    print('Text Vectorization...')
                    vr.vectorize(project=p, NORM=self.NORM, no_below=2, n=n)
                    gc.collect()

            for p in project:
                m_df, class_df = self._metrics(p)
                # m_df = m_df.drop(['pCommitNum'], axis=1)
                metrics = pd.concat([m_df, class_df], axis=1)
                _project = [_p for _p in project if not(p==_p)]
                for _p in _project:
                    _create_dir('./../data/{}/cross/{}'.format(p, _p))
                    m_df, class_df = self._metrics(_p)
                    test_dataset = pd.concat([m_df, class_df], axis=1)
                    SF = SelectFeature()
                    SF.select(dataset=metrics, project=p, test_dataset=test_dataset, test_project=_p, NORM=self.NORM, OUTPUT=self.OUTPUT)

                    drop_df = dt.fread('./../data/{}/cross/{}/outliers.csv'.format(p, _p))
                    drop_df = drop_df.to_pandas()
                    drop_outlier_index = list(drop_df[(drop_df['ReporterOutlier']==True)|(drop_df['ProcessOutlier']==True)|(drop_df['CodeOutlier']==True)].index)
                    if not(os.path.exists('./../data/{}/cross/{}/title_{}gram.csv'.format(p, _p, n))):
                        header = pd.read_csv('./../data/{}/single/title_{}gram.csv'.format(p, n), chunksize=2)
                        h_col = next(header).columns
                        df = dt.fread('./../data/{}/single/title_{}gram.csv'.format(_p, n)).to_pandas()
                        print('Drop Title Data...')
                        # index = set(df.index) - set(drop_outlier_index)
                        # df = df.query('index in @index')
                        cols = set(h_col) & set(df.columns)
                        df = pd.concat([pd.DataFrame(columns=h_col), df.loc[:, cols]], axis=0)
                        df = df.fillna(0)
                        df.to_csv('./../data/{}/cross/{}/title_{}gram.csv'.format(p, _p, n), index=None)
                        print('-> Finished')
                        del df
                        gc.collect()
                    
                    if not(os.path.exists('./../data/{}/cross/{}/body_{}gram.csv'.format(p, _p, n))):
                        header = pd.read_csv('./../data/{}/single/body_{}gram.csv'.format(p, n), chunksize=2)
                        h_col = next(header).columns
                        df = dt.fread('./../data/{}/single/body_{}gram.csv'.format(_p, n)).to_pandas()
                        print('Drop Description Data...')
                        # index = set(df.index) - set(drop_outlier_index)
                        # df = df.query('index in @index')
                        cols = set(h_col) & set(df.columns)
                        df = pd.concat([pd.DataFrame(columns=h_col), df.loc[:, cols]], axis=0)
                        df = df.fillna(0)
                        df.to_csv('./../data/{}/cross/{}/body_{}gram.csv'.format(p, _p, n), index=None)
                        print('-> Finished')
                        del df
                        gc.collect()

                    # print('Text Vectorization...')
                    # vr.cross_vectorize(project=p, test_project=_p, NORM=self.NORM)

        elif self.TYPE == 'part':
            """ Get Test Project """
            for p in project:
                dataset = self._tokenize(p, 1)
                m_df, class_df = self._metrics(p)
                convert_col = set(m_df.columns) - set(['AuthorType', 'SelfAssign'])

                # """ Convert Data Types """
                # for c_col in convert_col:
                #     try:
                #         if c_col in ['DescriptionLen', 'CommentNum', 'AssigneeNum', 'ParticipantNum']:
                #             m_df[c_col] = m_df[c_col].fillna(0)
                #         m_df[c_col] = m_df[c_col].astype(float)
                #     except:
                #         pass
                #     m_df[c_col] = m_df[c_col].apply(lambda x: x if type(x) in [float, int] else None)

                if self.OUTPUT:
                    print('Create a Directory:', './../data/cross/{}'.format(p))
                    _create_dir('./../data/cross/{}'.format(p))
                    
                    if self.LEMMA:
                        dataset.to_pickle('./../data/cross/{}/dataset_{}{}.pkl'.format(p, 1, OUTPUT_OPTION))
                    else:
                        dataset.to_pickle('./../data/cross/{}/dataset_{}_for_bert{}.pkl'.format(p, 1, OUTPUT_OPTION))

                    dataset.to_pickle('./../data/cross/{}/dataset_1{}.pkl'.format(p, OUTPUT_OPTION))
                    m_df.to_csv('./../data/cross/{}/metrics{}.csv'.format(p, OUTPUT_OPTION), index=None)
                    class_df.to_csv('./../data/cross/{}/class_labels{}.csv'.format(p, OUTPUT_OPTION), index=None)

                vr = Vectorizer()
                if not(os.path.exists('./../data/cross/{}/body_1gram_without_outlier.csv'.format(p))):
                    if not(os.path.exists('./../data/{}/body_1gram_without_outlier.csv'.format(p))):
                        print('Text Vectorization...')
                        vr.vectorize(project=p, NORM=self.NORM)
                    else:
                        print('Copy', './../data/{}/body_1gram_without_outlier.csv'.format(p), '->', './../data/cross/{}/body_1gram.csv'.format(p))
                        shutil.copy('./../data/{}/body_1gram_without_outlier.csv'.format(p), './../data/cross/{}/body_1gram.csv'.format(p))

                if not(os.path.exists('./../data/cross/{}/title_1gram_without_outlier.csv'.format(p))):
                    if not(os.path.exists('./../data/{}/title_1gram_without_outlier.csv'.format(p))):
                        print('Text Vectorization...')
                        vr.vectorize(project=p, NORM=self.NORM)
                    else:
                        print('Copy', './../data/{}/title_1gram_without_outlier.csv'.format(p), '->', './../data/cross/{}/title_1gram.csv'.format(p))
                        shutil.copy('./../data/{}/title_1gram_without_outlier.csv'.format(p), './../data/cross/{}/title_1gram.csv'.format(p))

                test_dataset = pd.concat([m_df, class_df], axis=1)

                """ Create Traind Data """
                all_text = pd.DataFrame()
                all_metrics = pd.DataFrame()
                all_class_labels = pd.DataFrame()
                _project = [_p for _p in project if not(_p == p)]
                print('Concat Each Project Data...')
                for _p in _project:
                    print(_p)

                    # Concatenates the text data of each poroject
                    for n in [1]:
                        dataset = self._tokenize(_p, n)
                        dataset['Project'] = _p
                        all_text = pd.concat([all_text, dataset])

                    # Concatenates the metrics data of each poroject
                    m_df, class_df = self._metrics(_p)
                    # convert_col = set(m_df.columns) - set(['AuthorType', 'SelfAssign'])
                    # for c_col in convert_col:
                    #     try:
                    #         if c_col in ['DescriptionLen', 'CommentNum', 'AssigneeNum', 'ParticipantNum']:
                    #             m_df[c_col] = m_df[c_col].fillna(0)
                    #         m_df[c_col] = m_df[c_col].astype(float)
                    #     except:
                    #         pass
                    #     m_df[c_col] = m_df[c_col].apply(lambda x: x if type(x) in [float, int] else None)

                    all_metrics = pd.concat([all_metrics, m_df])
                    all_class_labels = pd.concat([all_class_labels, class_df])

                all_text.reset_index(drop=True, inplace=True)
                all_metrics.reset_index(drop=True, inplace=True)
                all_class_labels.reset_index(drop=True, inplace=True)

                print('Train Data:', _project)
                print(all_metrics.dtypes)
                print()
                print('Test Data:', p)
                print(test_dataset.dtypes)
                print()

                if self.OUTPUT:
                    _create_dir('./../data/cross/{}'.format('_'.join(_project)))

                    if self.LEMMA:
                        all_text.to_pickle('./../data/cross/{}/dataset_{}{}.pkl'.format('_'.join(_project), n, OUTPUT_OPTION))
                    else:
                        all_text.to_pickle('./../data/cross/{}/dataset_{}_for_bert{}.pkl'.format('_'.join(_project), n, OUTPUT_OPTION))

                    all_text.to_pickle('./../data/cross/{}/dataset_1{}.pkl'.format('_'.join(_project), OUTPUT_OPTION))
                    all_metrics.to_csv('./../data/cross/{}/metrics{}.csv'.format('_'.join(_project), OUTPUT_OPTION), index=None)
                    all_class_labels.to_csv('./../data/cross/{}/class_labels{}.csv'.format('_'.join(_project), OUTPUT_OPTION), index=None)

                print(all_class_labels.columns)

                dataset = pd.concat([all_metrics, all_class_labels], axis=1)
                print('Dataset:')
                print(dataset['Class'].value_counts())
                print()

                CSF = CrossProjectSelectFeature()
                CSF.select(dataset=dataset, project=os.path.join('cross', '_'.join(_project)), test_dataset=test_dataset, test_project=p, NORM=self.NORM, OUTPUT=self.OUTPUT)

                del all_text, all_metrics, all_class_labels
                gc.collect()

                drop_df = dt.fread('./../data/{}/outliers.csv'.format(os.path.join('cross', '_'.join(_project))))
                drop_df = drop_df.to_pandas()
                drop_outlier_index = list(drop_df[(drop_df['ReporterOutlier']==True)|(drop_df['ProcessOutlier']==True)|(drop_df['CodeOutlier']==True)].index)

                if not(os.path.exists('./../data/cross/{}/title_1gram_without_outlier.csv'.format('_'.join(_project)))):
                    print('Concat Title Data...')
                    all_title = pd.DataFrame()
                    for _p in _project:
                        print(_p)
                        df = dt.fread('./../data/{}/title_1gram_without_outlier.csv'.format(_p))
                        df = df.to_pandas()
                        all_title = pd.concat([all_title, df])

                    print('Reset Index...')
                    all_title.reset_index(drop=True, inplace=True)
                    print('Drop Title Data...')
                    all_title = all_title.drop(drop_outlier_index, axis=0)
                    all_title = all_title.fillna(0)
                    all_title.to_csv('./../data/cross/{}/title_1gram_without_outlier.csv'.format('_'.join(_project)), index=None)
                    print('-> Finished')
                    del all_title
                    gc.collect()

                if not(os.path.exists('./../data/cross/{}/body_1gram_without_outlier.csv'.format('_'.join(_project)))):
                    print('Concat Description Data...')
                    all_body = pd.DataFrame()
                    for _p in _project:
                        print(_p)
                        df = dt.fread('./../data/{}/body_1gram_without_outlier.csv'.format(_p))
                        df = df.to_pandas()
                        all_body = pd.concat([all_body, df])

                    print('Reset Index...')
                    all_body.reset_index(drop=True, inplace=True)
                    print('Drop Description Data...')
                    all_body = all_body.drop(drop_outlier_index, axis=0)
                    all_body = all_body.fillna(0)
                    all_body.to_csv('./../data/cross/{}/body_1gram_without_outlier.csv'.format('_'.join(_project)), index=None)    
                    print('-> Finished')
                    del all_body
                    gc.collect()         

                # print('Text Vectorization...')
                # vectorize(project=os.path.join('cross', '_'.join(_project)), NORM=self.NORM)

                """" Create Test Data """
                print()
                print(p)
                drop_df = dt.fread('./../data/{}/outliers.csv'.format(os.path.join('cross', p)))
                drop_df = drop_df.to_pandas()
                drop_outlier_index = list(drop_df[(drop_df['ReporterOutlier']==True)|(drop_df['ProcessOutlier']==True)|(drop_df['CodeOutlier']==True)].index)
                if not(os.path.exists('./../data/cross/{}/title_1gram_without_outlier.csv'.format(p))):
                    header = pd.read_csv('./../data/cross/{}/title_1gram_without_outlier.csv'.format('_'.join(_project)), chunksize=2)
                    h_col = next(header).columns
                    df = dt.fread('./../data/cross/{}/title_1gram.csv'.format(p))
                    df = df.to_pandas()
                    print('Drop Title Data...')
                    df = df.drop(drop_outlier_index, axis=0)
                    cols = set(h_col) & set(df.columns)
                    df = pd.concat([pd.DataFrame(columns=h_col), df.loc[:, cols]], axis=0)
                    df = df.fillna(0)
                    df.to_csv('./../data/cross/{}/title_1gram_without_outlier.csv'.format(p), index=None)
                    print('-> Finished')
                    del df
                    gc.collect()
                
                if not(os.path.exists('./../data/cross/{}/body_1gram_without_outlier.csv'.format(p))):
                    header = pd.read_csv('./../data/cross/{}/body_1gram_without_outlier.csv'.format('_'.join(_project)), chunksize=2)
                    h_col = next(header).columns
                    df = dt.fread('./../data/cross/{}/body_1gram.csv'.format(p))
                    df = df.to_pandas()
                    print('Drop Description Data...')
                    df = df.drop(drop_outlier_index, axis=0)
                    cols = set(h_col) & set(df.columns)
                    df = pd.concat([pd.DataFrame(columns=h_col), df.loc[:, cols]], axis=0)
                    df = df.fillna(0)
                    df.to_csv('./../data/cross/{}/body_1gram_without_outlier.csv'.format(p), index=None)
                    print('-> Finished')
                    del df
                    gc.collect()
                    

        elif self.TYPE == 'all':
            all_text = pd.DataFrame()
            all_metrics = pd.DataFrame()
            all_class_labels = pd.DataFrame()
            for p in project:
                # Concatenates the text data of each poroject
                for n in [1]:
                    dataset = self._tokenize(p, n)
                    dataset['Project'] = p
                    all_text = pd.concat([all_text, dataset])

                # Concatenates the metrics data of each poroject
                m_df, class_df = self._metrics(p)
                class_df['Project'] = p

                all_metrics = pd.concat([all_metrics, m_df])
                all_class_labels = pd.concat([all_class_labels, class_df])

            all_text.reset_index(drop=True, inplace=True)
            all_metrics.reset_index(drop=True, inplace=True)
            all_class_labels.reset_index(drop=True, inplace=True)

            if self.OUTPUT:
                _create_dir('./../data/all')
                if self.LEMMA:
                    all_text.to_pickle('./../data/all/dataset_{}{}.pkl'.format(n, OUTPUT_OPTION))
                else:
                    all_text.to_pickle('./../data/all/dataset_{}_for_bert{}.pkl'.format(n, OUTPUT_OPTION))

                all_text.to_pickle('./../data/all/dataset_1{}.pkl'.format(OUTPUT_OPTION))
                all_metrics.to_csv('./../data/all/metrics{}.csv'.format(OUTPUT_OPTION), index=None)
                all_class_labels.to_csv('./../data/all/class_labels{}.csv'.format(OUTPUT_OPTION), index=None)

            print(all_class_labels.columns)

            dataset = pd.concat([all_metrics, all_class_labels], axis=1)
            print('Dataset:')
            print(dataset['Class'].value_counts())
            print()

            ex_df = pd.DataFrame()
            for _ in range(100):
                for p in dataset['Project'].unique():
                    index_ = []
                    p_index = list(dataset[(dataset['Project']==p)&(dataset['Class']==True)].index)
                    index_ += rand_ints_nodup(np.min(p_index), np.max(p_index), k=50)
                    n_index = list(dataset[(dataset['Project']==p)&(dataset['Class']==False)].index)
                    index_ += rand_ints_nodup(np.min(n_index), np.max(n_index), k=50)
                    ex_df = pd.concat([ex_df, dataset.loc[index_, :]])  

            SF = SelectFeature()
            SF.select(dataset.drop(['Project'], axis=1), project='all', NORM=self.NORM, OUTPUT=self.OUTPUT, all_dataset=ex_df.drop(['Project'], axis=1))

            # print('Text Vectorization...')
            # vectorize(project='all', NORM=self.NORM)
        

        
if __name__ == '__main__':
    """
        Parse Inputed Argument
    """
    parser = argparse.ArgumentParser(description='データセット作成プログラム')
    parser.add_argument('--tokenize', action='store_true', help='テキストのトークナイズ')
    parser.add_argument('--lemma', action='store_true', help='テキストのレマタイズ')
    parser.add_argument('--norm', action='store_true', help='特徴量の正規化・標準化')
    parser.add_argument('--output', action='store_true', help='データの出力')
    parser.add_argument('--type', help='作成したいデータセットの種別')
    args = parser.parse_args()

    TOKENIZE = args.tokenize
    LEMMA = args.lemma
    NORM = args.norm
    OUTPUT = args.output
    TYPE = args.type

    print(f'TOKENIZE = {args.tokenize}')
    print(f'LEMMA = {args.lemma}')
    print(f'NORM = {args.norm}')
    print(f'OUTPUT = {args.output}')
    print(f'TYPE = {args.type}')

    cd = createDataset(TOKENIZE=TOKENIZE, LEMMA=LEMMA, NORM=NORM, TYPE=TYPE, OUTPUT=OUTPUT)
    cd.main(project=['influxdb', 'saleor', 'server', 'vscode'])