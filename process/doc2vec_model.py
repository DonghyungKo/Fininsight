# Y 더미화
from collections import defaultdict

# CBOW 계산
import numpy as np

# 임베딩
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 병렬처리
import multiprocessing
import logging
from multiprocessing import Process,Queue, Pool
import functools
from threading import Thread
import queue

# 분류기 학습 결과 계산
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# 결과 저장
import pandas as pd


class D2V(object):
    #####################################################
    ##################   Doc2Vec   ######################
    #####################################################

    def __init__(self, path_to_model = ''):
        '''
        기존에 학습된 모델을 불러옵니다.

        Inputs
        =========================
        path_to_model : str,
            학습된 모델이 저장된 path
        '''
        try:
            self.Doc2Vec_model = Doc2Vec.load(path_to_model)
            print('학습된 모델을 성공적으로 불러왔습니다.')
            return
        except:
            print('모델을 불러오지 못하였습니다.')
            print('새로운 모델을 생성해주시기 바랍니다.')
            return


    def make_Doc2Vec_model(self,
                           dm = 1,
                           dbow_words = 0,
                           window = 15,
                           vector_size = 100,
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
        =================================
        train_doc_ls : iterable,
            documents(tokenized or not tokenized)

        train_tag_ls : iterable,
            tags of each documents

        n_epochs : int
            numbers of iteration

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
         - train_doc_ls : iterable,
            documents(tokenized or not tokenized)

         - train_tag_ls : iterable,
            tags of each documents

         - n_epochs : int
            numbers of iteration

         - if_morphs :  boolean
            원문에 대한 tokenizing을 수행할 때, morphs를 추출 (defaulf = True),

         - if_tokenized : boolean
            True if input document is tokenized [default = True]

         - if_morphs : boolean,
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


    # 문서 벡터를 추정하기 위한 함수 (병렬처리에 사용)
    def _infer_vector(self,
            doc_ls,
            alpha=0.1,
            steps=30,
            queue = False):

        '''
        Doc2Vec을 사용하여, documents를 vectorize하는 함수입니다.
        본 함수는 병렬처리를 위해 사용합니다.

        Inputs
        doc_ls : iterable or str
            array of tokenized documents

        alpha : int
        steps : int

        return
         - matrix of documents inferred by Doc2Vec
        '''
        return_ls = []

        # 문서 1개가 들어온 경우,
        if type(doc_ls) == str:
            return self.Doc2Vec_model.infer_vector(doc_ls,
                                                   alpha = alpha,
                                                   min_alpha = self.Doc2Vec_model.min_alpha,
                                                   steps = steps)

        # 복수 개의 문서가 input으로 들어온 경우,
        else:
            return [self.Doc2Vec_model.infer_vector(doc,
                                                    alpha = alpha,
                                                    min_alpha = self.Doc2Vec_model.min_alpha,
                                                    steps = steps) \
                    for doc in doc_ls]


    ###################################################
    ################### 병렬처리 적용 ####################
    ###################################################
    def _multiprocessing_queue_put(self, func, queue, **kwargs):
        queue.put(func(**kwargs))
        return


    def infer_vectors_multiprocessing(self, doc_ls):
        queue_ls = []
        procs = []
        result_ls = []
        batch_size = len(doc_ls) // 10

        # process에 작업들을 할당
        for i, idx in enumerate(range(0, len(doc_ls), batch_size)):
            try:
                batch_ls = doc_ls[idx : idx + batch_size]
            except:
                batch_ls = doc_ls[idx :]

            queue_ls.append(Queue())
            proc = Process(
                    target= self._multiprocessing_queue_put,
                    kwargs = {
                        'func' : self._infer_vector,
                        'queue' : queue_ls[i],
                        'doc_ls' : batch_ls})

            procs.append(proc)
            proc.start()

        for queue in queue_ls:
            temp_result_ls = queue.get()
            queue.close()
            del queue

            result_ls += temp_result_ls

        for proc in procs:
            proc.join()
            del proc

        return np.array(result_ls)
