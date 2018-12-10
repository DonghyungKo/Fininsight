import pandas as pd
from ko_text import *

def load_data(path_to_file):
    data = pd.read_csv(open(path_to_file,'r'), encoding='utf-8', engine='c')

    # Token 칼럼이 존재하는 경우
    if 'Token' in data.columns:
        # 용량을 줄이기 위해 '단어 단어' 꼴로 묶어둔 token을 ['단어', '단어']의 리스트 형태로 풀기
        data['Token'] = [token.split() for token in data['Token']]

    print('Data Loaded')
    print('=========================================')
    return data


def save_data(data, path_to_file):
    # Token 칼럼이 존재하는 경우
    if 'Token' in data.columns:
        # 용량을 줄이기 위해 ['단어', '단어']형태의 토큰을 '단어 단어' 형태로 묶기
        data['Token'] = [' '.join(doc) for doc in data['Token'].tolist()]

    data.to_csv(path_to_file, index = False)
    print('Data Saved')
    return

def remove_overlapped_articles(data):
    '''
    동일한 내용이 중복되는 기사를 제거하는 함수입니다.
    '''

    # index 초기화
    data.index = np.arange(len(data))
    text_ls = []
    unique_idx_ls = []

    for idx, text in enumerate(data['Text']):
        if not text in text_ls:
            text_ls.append(text)
            unique_idx_ls.append(idx)

    print('중복되는 기사들을 제거하였습니다')
    print('원본 기사의 수 : %s'%len(data))
    print('중복을 제거한 기사의 수 : %s'%len(unique_idx_ls))
    print('========================================')

    # 고유한 기사만 추출
    data = data.loc[unique_idx_ls]
    data.index = np.arange(len(data))
    return data




def main():

    # Data loading
    data = load_data('data/data_merged.csv')

    # 중복되는 기사 제거
    data = remove_overlapped_articles(data)

    # 자연어처리 클래스 생성
    nlp = NLP()

    # 정규식, 불용어 추가
    regex_ls = ['여기를 누르시면 크게 보실 수 있습니다']
    nlp.add_regex(regex_ls)


    # 토크나이징
    print('Tokenizer Started')
    start_time = time.time()
    token_doc_ls = nlp.extract_morphs_for_all_document_FAST_VERSION(data['Text'].tolist(),
                                                                n_thread = 4)
    print('Tokenizer Finished')
    print('Tokenizing 총 소요시간 %s(초)'%(time.time() - start_time))
    data['Token'] = token_doc_ls


    # 토큰의 길이가 30개 이상인 단어만 추출
    len_ls = []
    for token in data['Token']:
        len_ls.append(len(token))

    data['num_token'] = len_ls
    data = data[data['num_token'] > 30]

    # Data save
    save_data(data, 'data/data_tokenized.csv')
    return









if __name__ == '__main__':
    main()
