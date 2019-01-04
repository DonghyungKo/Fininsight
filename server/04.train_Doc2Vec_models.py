from ko_text import *
import ko_text
from sklearn.model_selection import train_test_split


def split_train_test(token_df, test_size=0.2):
    # train, test split
    train_token_ls, test_token_ls, train_Y, test_Y = train_test_split(token_df['Token'],token_df['Section'], test_size=test_size)

    train_token_ls, test_token_ls = np.array(train_token_ls), np.array(test_token_ls)
    train_Y, test_Y = np.array(train_Y), np.array(test_Y)

    return train_token_ls, test_token_ls, train_Y, test_Y


def train_and_save_Doc2Vec_model(token_df):
    nlp = NLP()
    d2v = D2V()

    # Doc2Vec은 전체 문서로 학습을 수행합니다.
    token_ls = np.array(token_df['Token'])
    label_ls = np.array(token_df['Section'])

    # 분류기를 학습하기 위한 train, test, split입니다.
    train_token_ls, test_token_ls, train_Y, test_Y = split_train_test(token_df)

    # 분류기를 학습하기 위해 샘플링을 수행합니다.
    # Train : Oversampling , Test : Undersampling
    print('label에 따른 Train 데이터 갯수입니다.')
    [print(x) for x in Counter(train_Y).most_common()]


    # 기존에 학습하던 결과가 있으면 불러온다.
    if os.path.isfile('Doc2Vec_model/Doc2Vec_tuning_result.csv'):
        result_dict = pd.read_csv('Doc2Vec_model/Doc2Vec_tuning_result.csv').to_dict('list')
        print('기존의 튜닝 결과를 불러옵니다.')
    else:
        result_dict = defaultdict(lambda : [])

    # training
    for dm in [1]:
        for dm_mean in [0,1]:
            for sample in [1e-04, 1e-05, 1e-06]:
                for min_count in [1, 5, 15]:
                    for vector_size in [100,300]:
                        for window in [5,15]:
                            for n_epochs in [10,30]:
                                print('==============================================================================')
                                print('Model Training Started')

                                # Doc2Vec 모델 생성
                                d2v.make_Doc2Vec_model(
                                    dm = dm,
                                    min_count = min_count,
                                    sample = sample,
                                    vector_size = vector_size,
                                    window = window,
                                    dm_mean = dm_mean,
                                    dm_concat = 0)


                                model_name = 'Doc2Vec_dm=%s&cc=%s&vs=%s&win=%s&min=%s&sample=%s&epochs=%s&dm_mean=%s'%(\
                                    d2v.Doc2Vec_model.dm,
                                    len(token_ls),
                                    d2v.Doc2Vec_model.vector_size,
                                    d2v.Doc2Vec_model.window,
                                    d2v.Doc2Vec_model.min_count,
                                    d2v.Doc2Vec_model.sample,
                                    n_epochs,
                                    dm_mean)

                                # 이미 모델이 존재하면 전체 과정 생략
                                if os.path.isfile('Doc2Vec_model/' + model_name):
                                    print('해당 파라미터로 학습된 모델이 이미 존재합니다. 과정을 생략합니다.')
                                    continue

                                # build and train Doc2Vec
                                d2v.build_and_train_Doc2Vec_model(
                                    token_ls,
                                    label_ls,
                                    n_epochs = n_epochs)

                                # Doc2Vec 모델 저장
                                if not os.path.exists('Doc2Vec_model'):
                                    os.mkdir('Doc2Vec_model')
                                d2v.Doc2Vec_model.save('Doc2Vec_model/' + model_name)

                                # 단어 벡터 추정량 분포 시각화
                                '''
                                X =d2v.infer_vectors_with_Doc2Vec(train_token_ls_split, alpha = 0.1)

                                tsne= TSNE(n_components=2)
                                X_tsne = tsne.fit_transform(X)
                                scatter_df = pd.DataFrame(X_tsne,
                                                          index = train_tag_ls_split,
                                                          columns = ['x','y'])

                                plt.figure(figsize = (10, 10))

                                for i,section in enumerate(set(test_df['Section'])):
                                    temp_df = scatter_df[scatter_df.index == section]
                                    plt.scatter(temp_df['x'].values, temp_df['y'].values, label = section, c = np.random.rand(3,))

                                plt.legend(loc = 'best')
                                plt.savefig('추정된 벡터 분포 t-sne ver')
                                '''


                                # train size를 나눠서 학습 비교
                                for train_batch_size in [1000,10000]:
                                    test_batch_size = 200

                                    # 학습을 위한 샘플링
                                    train_token_ls_split, train_label_ls_split = nlp.oversample_batch(train_token_ls, train_Y, train_batch_size)
                                    test_token_ls_split, test_label_ls_split = nlp.undersample_batch(test_token_ls, test_Y, size = train_batch_size)

                                    # clf를 각 레이블별 train_batch_size개씩 학습,
                                    X_train = d2v.infer_vectors_multiprocessing(train_token_ls_split)
                                    y_train = train_label_ls_split

                                    X_test = d2v.infer_vectors_multiprocessing(test_token_ls_split)
                                    y_test = test_label_ls_split

                                    clf = LogisticRegression(
                                        solver = 'sag',
                                        multi_class = 'multinomial')

                                    # fit and predict
                                    clf.fit(X_train, y_train)
                                    y_pred = clf.predict(X_test)
                                    print('Test Accuracy : %s'%accuracy_score(y_pred, y_test))
                                    print('==================================================================')

                                    # 결과 저장
                                    result_dict['dm'].append(d2v.Doc2Vec_model.dm)
                                    result_dict['dm_mean'].append(dm_mean)
                                    result_dict['corpus_count'].append(d2v.Doc2Vec_model.corpus_count)
                                    result_dict['min_count'].append(d2v.Doc2Vec_model.min_count)
                                    result_dict['vector_size'].append(d2v.Doc2Vec_model.vector_size)
                                    result_dict['window'].append(d2v.Doc2Vec_model.window)
                                    result_dict['n_epochs'].append(d2v.Doc2Vec_model.epochs)
                                    result_dict['sample'].append(d2v.Doc2Vec_model.sample)
                                    result_dict['train_size'].append(train_batch_size)
                                    result_dict['accuracy'].append(accuracy_score(y_pred, y_test))

                                    # 분류 결과 저장
                                    pd.DataFrame(result_dict).to_csv('Doc2Vec_tuning_result.csv', index=False)

                                print('Model Training Finished')
                                print('Model : %s'%model_name)



def main():
    token_df = load_data('data/data_merged.csv', usecols = ['Token','Section'])
    train_and_save_Doc2Vec_model(token_df)

if __name__=='__main__':
    main()
