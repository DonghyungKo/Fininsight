# 데이터를 읽어오는 함수 호출
from data_loader import read_txt

# Doc2Vec 모델 학습에 사용하는 클래스 및 함수 호출
from doc2vec_model import *

# 학습 결과 TSNE로 시각화
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 그래프에서 마이너스 폰트 꺠지는 문제에 대한 대처
import matplotlib as mpl
from matplotlib import font_manager, rc

mpl.rcParams['axes.unicode_minus'] = False

# 한글 문제 해결
try:
    path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    fontprop = font_manager.FontProperties(fname=path, size=18).get_name()
    rc('font', family='NanumGothicCoding')
except:
    pass


def main():
    # data load
    token_ls, category_ls = read_txt('news_tokenized.txt')

    # token 분리
    token_ls = [tokens.split() for tokens in token_ls]

    # train test split
    train_size = int(round(len(token_ls) * 0.8))

    x_train, x_test = token_ls[:train_size], token_ls[train_size:]
    y_train, y_test = category_ls[:train_size], category_ls[train_size:]

    # Y 더미화
    cat2idx = defaultdict(lambda : len(cat2idx))
    y_train_idx = [cat2idx[x] for x in y_train]
    y_test_idx = [cat2idx[x] for x in y_test]

    # 모델 저장할 폴더 생성
    if not os.path.exists('Doc2Vec_model'):
        os.mkdir('Doc2Vec_model')

    # 기존에 학습하던 결과가 있으면 불러온다.
    if os.path.isfile('Doc2Vec_model/Doc2Vec_tuning_result.csv'):
        result_dict = pd.read_csv('Doc2Vec_model/Doc2Vec_tuning_result.csv').to_dict('list')
        print('기존의 튜닝 결과를 불러옵니다.')
    else:
        result_dict = defaultdict(lambda : [])


    #train
    d2v = D2V()
    best_loss = 1e10

    # training
    for dm in [1]:
        for sample in [1e-05, 1e-06]:
            for min_count in [1, 5, 15]:
                for vector_size in [100,300]:
                    for negative in [5,10,15]:
                        for n_epochs in [10,30]:
                            print('==============================================================================')
                            print('Model Training Started')

                            # Doc2Vec 모델 생성
                            d2v.make_Doc2Vec_model(
                                dm = dm,
                                min_count = min_count,
                                sample = sample,
                                vector_size = vector_size)

                            # build and train Doc2Vec
                            d2v.build_and_train_Doc2Vec_model(x_train, category_ls, n_epochs = n_epochs)

                            '''
                            # 단어 벡터 추정량 분포 시각화
                            X =d2v.infer_vectors_multiprocessing(x_test)

                            tsne= TSNE(n_components=2)
                            X_tsne = tsne.fit_transform(X)
                            scatter_df = pd.DataFrame(X_tsne,
                                                      index = y_test,
                                                      columns = ['x','y'])

                            plt.figure(figsize = (10, 10))

                            for i,section in enumerate(set(category_ls)):
                                temp_df = scatter_df[scatter_df.index == section]
                                plt.scatter(temp_df['x'].values, temp_df['y'].values, label = section, c = np.random.rand(3,))

                            plt.legend(loc = 'best')
                            plt.savefig('추정된 벡터 분포 t-sne ver')

                            '''

                            # 벡터 추정 후 학습 및 성과 평가
                            X_train = d2v.infer_vectors_multiprocessing(x_train)
                            X_test = d2v.infer_vectors_multiprocessing(x_test)

                            # 분류기 학습
                            clf = LogisticRegression(solver = 'sag', multi_class = 'multinomial')

                            clf.fit(X_train, y_train_idx)
                            y_prob = clf.predict_log_proba(X_test)
                            y_pred = [prob.argmax() for prob in y_prob]

                            acc = accuracy_score(y_test_idx, y_pred)
                            loss = log_loss(y_test_idx, y_prob, normalize=False)

                            print('Accuracy : ', acc)

                            # 분류 결과가 best인 모델만 저장
                            if loss < best_loss:
                                best_loss = loss
                                d2v.Doc2Vec_model.save('Doc2Vec_model/best_d2v_model')

                            # 결과 저장
                            result_dict['dm'].append(dm)
                            result_dict['corpus_count'].append(len(token_ls))
                            result_dict['min_count'].append(min_count)
                            result_dict['vector_size'].append(vector_size)
                            result_dict['negative'].append(negative)
                            result_dict['n_epochs'].append(n_epochs)
                            result_dict['sample'].append(sample)
                            result_dict['accuracy'].append(acc)
                            result_dict['loss'].append(loss)

                            # 분류 결과 저장
                            pd.DataFrame(result_dict).to_csv('Doc2Vec_model/Doc2Vec_tuning_result.csv', index=False)

                            print('Model Training Finished')

if __name__=='__main__':
    main()
