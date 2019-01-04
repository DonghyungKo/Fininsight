
# Y 더미화
from collections import defaultdict

# CBOW 계산
import numpy as np

# 임베딩
import os
from gensim.models import Word2Vec, Doc2Vec
import logging

# 분류기 학습 결과 계산
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# 결과 저장
import pandas as pd

# 데이터를 읽는 함
from data_loader import read_txt


# Word2Vec 성능 평가를 위한 CBOW
def make_cbow(token_ls, word2vec):
    cbow_ls = []

    for tokens in token_ls:
        cbow = np.zeros((word2vec.vector_size))

        for token in tokens:
            cbow += word2vec.wv.get_vector(token)

        cbow_ls.append(cbow)
    return cbow_ls

def main():
    # data load
    token_ls, category_ls = read_txt('news_tokenized.txt')
    token_ls = [tokens.split() for tokens in token_ls]

    # train test split
    train_size = int(round(len(token_ls) * 0.8))

    x_train = token_ls[:train_size]
    y_train = category_ls[:train_size]

    x_test = token_ls[train_size:]
    y_test = category_ls[train_size:]

    # Y 더미화
    cat2idx = defaultdict(lambda : len(cat2idx))
    y_train_idx = [cat2idx[x] for x in y_train]
    y_test_idx = [cat2idx[x] for x in y_test]

    # train
    result_dict = defaultdict(lambda: [])
    best_loss = 1e10

    # training
    for sg in [1,0]:
        for sample in [1e-05, 1e-06]:
            for min_count in [1, 10]:
                for alpha in [0.025, 0.1, 0.25]:
                    for vector_size in [100,300]:
                        for negative in [5,10,15]:
                            for n_epochs in [10,30,50]:
                                print('==============================================================================')
                                print('Model Training Started')

                                # word2vec 모델 생성
                                word2vec = Word2Vec(
                                    sentences = token_ls,
                                    sg = sg,
                                    size = vector_size,
                                    alpha = alpha,
                                    min_count = min_count,
                                    sample = sample,
                                    workers = 4,
                                    negative = negative,
                                    iter = n_epochs,
                                )

                                # CBOW 생성
                                X_train = make_cbow(x_train, word2vec)
                                X_test = make_cbow(x_test, word2vec)

                                # 분류기 학습
                                clf = LogisticRegression(solver = 'sag', multi_class = 'multinomial')

                                clf.fit(X_train, y_train_idx)
                                y_prob = clf.predict_log_proba(X_test)
                                y_pred = [prob.argmax() for prob in y_prob]

                                acc = accuracy_score(y_test_idx, y_pred)
                                loss = log_loss(y_test_idx, y_prob, normalize=False)

                                print('Accuracy : ', acc)

                                # 모델 저장할 폴더 생성
                                if not os.path.exists('Word2Vec_model'):
                                    os.mkdir('Word2Vec_model')

                                # 분류 결과가 best인 모델만 저장
                                if loss < best_loss:
                                    best_loss = loss
                                    word2vec.save('Word2Vec_model/best_w2v_model')

                                # 결과 저장
                                result_dict['sg'].append(sg)
                                result_dict['corpus_count'].append(len(token_ls))
                                result_dict['min_count'].append(min_count)
                                result_dict['vector_size'].append(vector_size)
                                result_dict['n_epochs'].append(n_epochs)
                                result_dict['sample'].append(sample)
                                result_dict['accuracy'].append(acc)
                                result_dict['negative'].append(negative)

                                # 분류 결과 저장
                                pd.DataFrame(result_dict).to_csv('Word2Vec_model/word2Vec_tuning_result.csv', index=False)

                                print('Model Training Finished')
    return

if __name__=='__main__':
    main()
