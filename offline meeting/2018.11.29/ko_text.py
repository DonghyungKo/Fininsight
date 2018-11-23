# -*- coding: utf-8 -*-
import sys
try: sys.path.remove('/home/donghyungko/anaconda3/lib/python3.7/site-packages')
except: pass
import ast

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
import matplotlib as mpl
from matplotlib import font_manager, rc

# 그래프에서 마이너스 폰트 꺠지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 한글 문제 해결
try:
    path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    fontprop = font_manager.FontProperties(fname=path, size=18).get_name()
    rc('font', family='NanumGothicCoding')
except: pass

import multiprocessing
import time
import pandas as pd
import re
import datetime
from collections import OrderedDict
import konlpy

import jpype

from konlpy.tag import *
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    
from collections import namedtuple
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.metrics import accuracy_score


import os
from multiprocessing import Process,Queue, Pool
import functools
from threading import Thread
import queue
import pickle

from konlpy import jvm


class NLP(object):
    
    '''
    크롤링한 데이터에 대한 텍스트분석을 위한 클래스입니다.
    
    - 클렌징, 명사&어근 추출 
    - TF-IDF 행렬 반환, 키워드 추출,
    - Doc2Vec 모델 생성 및 학습,
    - Topic 모델링의 기능을 제공합니다.
    '''
    
    
    def __init__(self):
        self.twit = Okt()
        self.kkma = Kkma()
        self.stopwords = []
        
    
    
    #####################################################
    ################## Preprocessing ####################
    #####################################################
    
    # 크롤링한 text에서 불필요한 부분들을 제거하는 함수입니다.
    def _clean_text(self,text):
        text = re.sub('\(.+?\)', '',string = text)
        text = re.sub('\[.+?\]', '',string = text)
        text = re.sub('\<.+?\>', '',string = text)
        text = re.sub('◀.+?▶', '',string = text)
        text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@▲▶◆\#$%┌─┐&\\\=\(\'\"├┼┤│┬└┴┘|ⓒ]', '' , text) # 특수문자 제거
        text = re.sub('[\t\n\r\f\v]', ' ' , text)   # 공백 제거

        # 년월일, 시분 제거 및 숫자 제거
        text = re.sub('[0-9]+[년월분일시조억만천원]*', '',text)

        # 문서의 특성에 맞춰 따로 제거 
        text = re.sub('기자', '', text)   # 기자라는 단어도 제거
        text = re.sub('여기를 누르시면 크게 보실 수 있습니다', '', text)   # 여기를 누르시면 크게 보실 수 있습니다 제거
        text = re.sub('[a-zA-Z]+', '', text)
        text = re.sub('연합뉴스', '', text)
        text = re.sub('무단.+금지', '', text)

        return text
    
    
    def clean_doc(self, doc_ls):
        return [self._clean_text(doc) for doc in doc_ls]
    
    
    def remove_stopwords(self, text_ls, stopword_ls):
        for word in remove_word_ls:
            text_ls = [text.replace(word,'') for text in text_ls]
        return text_ls

    
    def _extract_nouns_for_single_doc(self, doc):
        return [x for x in self.twit.nouns(doc) if len(x) > 1]

    def _extract_morphs_for_single_doc(self, doc):
        return [x for x in self.twit.morphs(doc) if len(x) > 1]

    
    
    def extract_nouns_for_all_document(self,doc_ls):
        jpype.attachThreadToJVM()
        clean_doc_ls = self.clean_doc(doc_ls)
        return [self._extract_nouns_for_single_doc(doc) for doc in clean_doc_ls]
 

    def extract_morphs_for_all_document(self, doc_ls):
        jpype.attachThreadToJVM()
        clean_doc_ls = self.clean_doc(doc_ls)
        return [self._extract_morphs_for_single_doc(doc) for doc in clean_doc_ls]
        

    
    def _extract_nouns_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_nouns_for_single_doc(doc)) for idx, doc in tuple_ls]
        
        
    def _extract_morphs_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_morphs_for_single_doc(doc)) for idx, doc in tuple_ls]
                  
        
                  
        
    def extract_tokens_for_all_document_FAST_VERSION(self, 
                                                     doc_ls, 
                                                     n_thread = 4,
                                                     if_morphs = True):
        jpype.attachThreadToJVM()

        '''
        멀티쓰레딩을 적용하여 속도가 개선된 버전입니다.
        morphs 추출을 위한 원문을 input으로 받습니다. (전처리 기능 포함)
        사용하실 쓰레드의 갯수를 input으로 받습니다. [default : 4]
        '''
                
        # 텍스트 클렌징 작업 수행
        # [(idx, clean_doc)] 형태로 저장 (나중에 sorting을 위해)
        clean_tuple_ls = [(idx, clean_doc) for idx, clean_doc in zip(range(len(doc_ls)), self.clean_doc(doc_ls))]
        
        # 멀티쓰레딩을 위한 작업(리스트)분할
        length = len(clean_tuple_ls)
        splited_clean_tuple_ls = self.split_list(clean_tuple_ls, length//n_thread)
        
        que = queue.Queue()
        thread_ls = []
        
        for tuple_ls in splited_clean_tuple_ls:
            if if_morphs:
                temp_thread = Thread(target= lambda q, arg1: q.put(self._extract_morphs_for_multiprocessing(arg1)),  args = (que, tuple_ls))
            else:
                temp_thread = Thread(target= lambda q, arg1: q.put(self._extract_nouns_for_multiprocessing(arg1)),  args = (que, tuple_ls))
                
            temp_thread.start()
            thread_ls.append(temp_thread)

        for thread in thread_ls:
            thread.join()
        
        # 정렬을 위한 index_ls와 token_ls를 사용
        index_ls = []
        token_ls = []
        
        # thread의 return 값을 결합
        while not que.empty():
            
            result = que.get() # [(idx, token), (idx, token)...] 형태를 반환
            index_ls += [idx for idx, _ in result]
            token_ls += [token for _, token in result]
                       
        return [token for idx, token in sorted(zip(index_ls, token_ls))]
    

    
    
    def split_list(self, ls, n):
        '''
        병렬처리를 위해, 리스트를 원하는 크기(n)로 분할해주는 함수입니다.
        '''
        result_ls = []
        
        for i in range(0, len(ls), n):
            result_ls += [ls[i:i+n]]
        return result_ls    
    
    
    def extract_a_equally_splited_batch(self, X_ls, Y_ls, size):
        
        if type(X_ls) == list:
            pass
        else:
            try: X_ls.tolist()
            except: pass
            
        if size == len(X_ls):
            return X_ls,Y_ls
        
        unique_y_ls = list(set(Y_ls))
        
        temp_dict = {}
        
        for x, y in zip(X_ls,Y_ls):
            try: temp_dict[y] += [x]
            except: temp_dict[y] = [x]
        
        
        # 한 섹션별로 k개씩 뽑아서 하나의 batch를 만든다.
        k = size // len(unique_y_ls)
        
        batch_X_ls = []
        batch_y_ls = []
        
        for key in temp_dict.keys():
            try:
                batch_X_ls += temp_dict[key][:k]
                batch_y_ls += [key] * len(temp_dict[key][:k])
            except:
                batch_X_ls += temp_dict[key]
                batch_y_ls += [key] * len(temp_dict[key])
                
                
        return batch_X_ls, batch_y_ls
        
    
    
    
    #####################################################
    ###################   TF-IDF   ######################
    #####################################################
        
    def doc_to_tfidf_df(self, doc_ls, 
                        min_df = 2, 
                        max_df = 0.3,
                        max_features = 300, 
                        if_tokenized = True,
                        if_morphs = True,
                       ):
        
        '''
        각 문서에 대한 TF-IDF vector를 pandas_DataFrame으로 반환하는 함수입니다.
        
        Inputs
         - doc_ls : iterable, list of documents
         - min_df : int or float, minimum occurance in a doc (at least bigger than min_df) if float, represents minimum ratio 
         - max_df : int or float, maximum occurange in a doc (at best smaller than max_df)
         - if_tokenized : Boolean, True if input document is tokenized [default = True]
         - if_morphs : Boolean, 
                       True : if not tokenized, tokenized with morphs,
                       False : if not tokenized, tokenized with nouns.
        
        Return
         - TF-IDF matrix (pandas.DataFrame)
        '''
        
        if if_tokenized:
            tokenized_doc_ls = doc_ls
        else:
            if if_morphs :
                tokenized_doc_ls = self.extract_tokens_for_all_document_FAST_VERSION(doc_ls, if_morphs = True)
            else :
                tokenized_doc_ls = self.extract_tokens_for_all_document_FAST_VERSION(doc_ls, if_morphs = False)

        corpus_for_tfidf_ls = [' '.join(x) for x in tokenized_doc_ls]
        
        tfidf_vectorizer = TfidfVectorizer(max_df = max_df,
                                           min_df = min_df,
                                           max_features = max_features).fit(corpus_for_tfidf_ls)
        
        tfidf_array = tfidf_vectorizer.transform(corpus_for_tfidf_ls).toarray()
        vocab = tfidf_vectorizer.get_feature_names()
        
        self.tfidf_df = pd.DataFrame(tfidf_array, columns = vocab)
        return self.tfidf_df
    
    
    def keyword_ls_from_tfidf_df(self):
        
        ''' 
        doc_to_tfidf_df 함수를 실행한 후, 사용 가능합니다.
        TF-IDF 행렬을 기반으로, 1순위, 2순위 키워드를 반환합니다.
        '''
        
        # 각 documents 별로, 1순위 키워드가 담긴 list
        first_keyword_ls = self.tfidf_df.idxmax(axis=1)
        
        # 각 documents 별로, 2순위 키워드가 담긴 list
        second_keyword_ls = [row.sort_values(ascending =False).index[1] for index, row in self.tfidf_df.iterrows()]
        second_keyword_ls.append(self.tfidf_df.iloc[-1,:].sort_values(ascending =False).index[1])
    
        return first_keyword_ls, second_keyword_ls
    

    
    
    
    #####################################################
    ##################   Doc2Vec   ######################
    #####################################################
    
    
    def make_Doc2Vec_model(self,
                           dm = 1,
                           dbow_words = 0, 
                           window = 15,
                           vector_size = 300,
                           sample = 1e-5,
                           min_count = 5,
                           hs = 0,
                           negative = 5,
                           dm_mean = 0,
                           dm_concat = 0):
        '''
        Doc2Vec 모델의 초기설정을 입력하는 함수입니다.
        기존에 만들어진 모델을 load하여 사용할 수 있습니다. (load_Doc2Vec_model 함수를 사용)
        
        Inputs
         - dm : PV-DBOW / default 1
         - dbow_words : w2v simultaneous with DBOW d2v / default 0
         - window : distance between the predicted word and context words
         - vector size : vector_size
         - min_count : ignore with freq lower
         - workers : multi cpu
         - hs : hierarchical softmax / default 0
         - negative : negative sampling / default 5
        
        Return
         - None
        '''
        cores = multiprocessing.cpu_count()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        self.Doc2Vec_model = Doc2Vec(
            dm= dm,                     # PV-DBOW / default 1
            dbow_words= dbow_words,     # w2v simultaneous with DBOW d2v / default 0
            window= window,             # distance between the predicted word and context words
            vector_size= vector_size,   # vector size
            sample = sample,
            min_count = min_count,      # ignore with freq lower
            workers= cores,             # multi cpu
            hs = hs,                    # hierarchical softmax / default 0
            negative = negative,        # negative sampling / default 5
            dm_mean = dm_mean,
            dm_concat = dm_concat,
        )
        
        return
            
        
        
    def build_and_train_Doc2Vec_model(self,
                                      train_doc_ls,
                                      train_tag_ls,
                                      n_epochs = 10,
                                      if_tokenized = True,
                                      if_morphs = True):
        
        ''' 
        Doc2Vec 모델을 생성 혹은 Load한 다음 작업으로, Doc2Vec을 build하고 학습을 수행합니다.
        
        Inputs
         - train_doc_ls : iterable, documents(tokenized or not tokenized)
         - train_tag_ls : iterable, tags of each documents
         - n_epochs : int, numbers of iteration
         - if_morphs : 원문에 대한 tokenizing을 수행할 때, morphs를 추출 (defaulf = True),
         - if_tokenized : Boolean, True if input document is tokenized [default = True]
         - if_morphs : Boolean, 
                       True : if not tokenized, tokenized with morphs,
                       False : if not tokenized, tokenized with nouns.
        
        Return
         - None
        '''
        
                
        # tokenized된 input데이터를 받으면 tokenizing skip
        if if_tokenized :
            train_token_ls = train_doc_ls
        else:
            clean_train_doc_ls = self.clean_doc(train_doc_ls)
            if if_morphs:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = True)
            else:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = False)
        
        # train_tag_ls를 리스트 형태로 변환 (Series 형태로 들어올 경우)
        try:  train_tag_ls = train_tag_ls.tolist()
        except:   pass
            
        # words와 tags로 구성된 namedtuple 형태로 데이터 변환 (tagging 작업)
        tagged_train_doc_ls = [TaggedDocument(tuple_[0], [tuple_[1]]) for i, tuple_ in enumerate(zip(train_token_ls, train_tag_ls))]
        
        # Doc2Vec 모델에 단어 build작업 수행
        self.Doc2Vec_model.build_vocab(tagged_train_doc_ls)

        # 학습 수행
        self.Doc2Vec_model.train(tagged_train_doc_ls,
                                 total_examples= self.Doc2Vec_model.corpus_count,
                                 epochs= n_epochs)
        return
    
    
    def train_Doc2Vec_model(self,
                            train_doc_ls,
                            train_tag_ls,
                            n_epochs = 10,
                            if_tokenized = True,
                            if_morphs = True,
                            ):
        ''' 
        built된 Doc2Vec 모델에 추가적인 학습을 수행합니다.
        
        Inputs
         - train_doc_ls : iterable, documents(tokenized or not tokenized)
         - train_tag_ls : iterable, tags of each documents
         - n_epochs : int, numbers of iteration
         - if_morphs : 원문에 대한 tokenizing을 수행할 때, morphs를 추출 (defaulf = True),
         - if_tokenized : Boolean, True if input document is tokenized [default = True]
         - if_morphs : Boolean, 
                       True : if not tokenized, tokenized with morphs,
                       False : if not tokenized, tokenized with nouns.
        
        Return
         - None
        '''
        
                
        # tokenized된 input데이터를 받으면 tokenizing skip
        if if_tokenized :
            train_token_ls = train_doc_ls
        else:
            clean_train_doc_ls = self.clean_doc(train_doc_ls)
            if if_morphs:
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = True)
            else :
                train_token_ls = self.extract_tokens_for_all_document_FAST_VERSION(clean_train_doc_ls, if_morphs = False)
        
        # train_tag_ls를 리스트 형태로 변환 (Series 형태로 들어올 경우)
        try:  train_tag_ls = train_tag_ls.tolist()
        except:   pass
            
        # words와 tags로 구성된 namedtuple 형태로 데이터 변환 (tagging 작업)
        tagged_train_doc_ls = [TaggedDocument(tuple_[0], [tuple_[1]]) for i, tuple_ in enumerate(zip(train_token_ls, train_tag_ls))]
        
        # 학습 수행
        self.Doc2Vec_model.train(tagged_train_doc_ls,
                                 total_examples= self.Doc2Vec_model.corpus_count,
                                 epochs= n_epochs)
        return
    
    
    
    def infer_vectors_with_Doc2Vec(self,doc_ls, 
                                   alpha = 0.05,
                                   steps = 20):
        
        '''
        Doc2Vec을 사용하여, documents를 vectorize하는 함수입니다.
        
        Inputs
         - doc_ls : iterable or str, array of tokenized documents
         
        return
         - matrix of documents inferred by Doc2Vec
        '''
        
        return_ls = []
        
        if type(doc_ls) == str:
            return self.Doc2Vec_model.infer_vector(doc, 
                                                   alpha = alpha, 
                                                   min_alpha = self.Doc2Vec_model.min_alpha,
                                                   steps = steps)
        
        else:
            return [self.Doc2Vec_model.infer_vector(doc,
                                                    alpha = alpha,
                                                    min_alpha = self.Doc2Vec_model.min_alpha,
                                                    steps = steps) \
                    for doc in doc_ls]
        
        
    def load_Doc2Vec_model(self, model_name):
        self.Doc2Vec_model = Doc2Vec.load(model_name)
        return  self.Doc2Vec_model
    
    

