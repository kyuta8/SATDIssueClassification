import os

import pandas as pd
import datatable as dt
import numpy as np
import pickle
from gensim import corpora, matutils
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm


class Vectorizer(object):

    def __init__(self) -> None:
        pass

    def vectorize(self, project, NORM=False, no_below=1, n=1):
        OUTPUT_PATH = '.'
        while True:
            dirs = os.listdir(OUTPUT_PATH)
            if 'data' in dirs:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'data')
                break
            else:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, '..')

        print('  - {}-gram'.format(n))

        with open(OUTPUT_PATH+'/{}/dataset_{}.pkl'.format(project, n), 'rb') as f:
            df1 = pickle.load(f)

        if not(os.path.exists(OUTPUT_PATH+'/{}/single/title_{}gram.csv'.format(project, n))):
            title_dic = corpora.Dictionary(list(df1['Title']))
            title_dic.filter_extremes(no_below=no_below, no_above=1, keep_n=1000000, keep_tokens=None)

            title_bow_data = [title_dic.doc2bow(data) if data else title_dic.doc2bow(['']) for data in list(df1['Title'])]
            pd.DataFrame(columns=['t_'+title_dic[l] for l in list(title_dic.keys())]).to_csv(OUTPUT_PATH+'/{}/single/title_{}gram.csv'.format(project, n), index=None)
            for bow in tqdm(title_bow_data):
                pd.DataFrame(list(matutils.corpus2dense([bow], num_terms=len(title_dic)).T[0])).T.to_csv(OUTPUT_PATH+'/{}/single/title_{}gram.csv'.format(project, n), mode='a', header=None, index=None)

        if not(os.path.exists(OUTPUT_PATH+'/{}/single/body_{}gram.csv'.format(project, n))):
            body_dic = corpora.Dictionary(list(df1['Description']))
            body_dic.filter_extremes(no_below=no_below, no_above=1, keep_n=1000000, keep_tokens=None)

            body_bow_data = [body_dic.doc2bow(data) if data else body_dic.doc2bow(['']) for data in list(df1['Description'])]
            pd.DataFrame(columns=['d_'+body_dic[l] for l in list(body_dic.keys())]).to_csv(OUTPUT_PATH+'/{}/single/body_{}gram.csv'.format(project, n), index=None)
            for bow in tqdm(body_bow_data):
                pd.DataFrame(list(matutils.corpus2dense([bow], num_terms=len(body_dic)).T[0])).T.to_csv(OUTPUT_PATH+'/{}/single/body_{}gram.csv'.format(project, n), mode='a', header=None, index=None)


    def cross_vectorize(self, project: str, test_project: str, NORM=False, no_below=1):
        OUTPUT_PATH = '.'
        while True:
            dirs = os.listdir(OUTPUT_PATH)
            if 'data' in dirs:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'data')
                break
            else:
                OUTPUT_PATH = os.path.join(OUTPUT_PATH, '..')

        if NORM:
            OUTPUT_OPTION = '_normalization'
        else:
            OUTPUT_OPTION = '_without_outlier'

        for n in [1]:
            print('  - {}-gram'.format(n))

            with open(OUTPUT_PATH+'/{}/dataset_{}{}.pkl'.format(project, n, OUTPUT_OPTION), 'rb') as f:
                df1 = pickle.load(f)

            with open(OUTPUT_PATH+'/{}/dataset_{}{}.pkl'.format(test_project, n, OUTPUT_OPTION), 'rb') as f:
                test_df = pickle.load(f)


            if not(os.path.exists(OUTPUT_PATH+'/{}/cross/{}/title_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION))):
                title_dic = corpora.Dictionary(list(df1['Title']))
                title_dic.filter_extremes(no_below=no_below, no_above=1, keep_n=1000000, keep_tokens=None)

                title_bow_data = [title_dic.doc2bow(data) if data else title_dic.doc2bow(['']) for data in list(test_df['Title'])]
                pd.DataFrame(columns=['t_'+title_dic[l] for l in list(title_dic.keys())]).to_csv(OUTPUT_PATH+'/{}/cross/{}/title_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION), index=None)
                for bow in tqdm(title_bow_data):
                    pd.DataFrame(list(matutils.corpus2dense([bow], num_terms=len(title_dic)).T[0])).T.to_csv(OUTPUT_PATH+'/{}/cross/{}/title_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION), mode='a', header=None, index=None)

            if not(os.path.exists(OUTPUT_PATH+'/{}/cross/{}/body_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION))):
                body_dic = corpora.Dictionary(list(df1['Description']))
                body_dic.filter_extremes(no_below=no_below, no_above=1, keep_n=1000000, keep_tokens=None)

                body_bow_data = [body_dic.doc2bow(data) if data else body_dic.doc2bow(['']) for data in list(test_df['Description'])]
                pd.DataFrame(columns=['d_'+body_dic[l] for l in list(body_dic.keys())]).to_csv(OUTPUT_PATH+'/{}/cross/{}/body_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION), index=None)
                for bow in tqdm(body_bow_data):
                    pd.DataFrame(list(matutils.corpus2dense([bow], num_terms=len(body_dic)).T[0])).T.to_csv(OUTPUT_PATH+'/{}/cross/{}/body_{}gram{}.csv'.format(project, test_project, n, OUTPUT_OPTION), mode='a', header=None, index=None)
