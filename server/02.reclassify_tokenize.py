import pandas as pd
from collections import Counter, defaultdict
import pickle
import ko_text as kotxt
from ko_text import *

'''
def drop_uselsess_data(data):

    #기사 제목에 원하지 않는 단어를 포함한 기사를 제거하는 함수입니다.
    #
    #inputs
    #=================================
    #data : pandas.DataFrame
    #    크롤링을 마친 raw data 상태의 DataFrame


    # 제거 목록 키워드 리스트
    useless_keyword_ls = ['신년사','[인사]','[포토]','포토','MK포토']

    index_ls = data.index.tolist()
    title_ls = data['Title'].tolist()

    drop_index_ls = []

    for idx, title in zip(index_ls, title_ls):
        if any(keyword in title for keyword in useless_keyword_ls):
            drop_index_ls.append(idx)
    return data.drop(drop_index_ls)
'''

def reclassify_categories(data, input_category, output_category):
    '''
    input_category에 해당하는 카테고리를 output_category로 변환하는 함수입니다.

    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame

    input_category : str, list
        재분류 전 카테고리

    output_category : str, list
        재분류 후 카테고리
    '''
    if type(input_category) == str:
        input_category = [input_category]

    # category reclassification
    data.loc[data['Section'].isin(input_category), 'Section'] = output_category
    return data

# business로 재분류 하는 함수
def to_business(data, stock_name_ls):
    '''
    economy, special_edition, health의 일부 기사를 business로 재분류합니다.
    retail, it, financial, electronics, autos, chemistry, heavy_industries의 모든 기사를 기업(business)로 재분류합니다.

    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame

    stock_name_ls : list,
        기업명이 str형태로 저장된 list
    '''
    # economy와 special_edition, health 카테고리 기사에서
    # 제목에 상장기업명이 포함된 경우 business로 재분류
    section_ls = ['economy','special_edition', 'health']
    temp_df = data.loc[data['Section'].isin(section_ls)]

    index_ls = temp_df.index
    title_ls = temp_df['Title'].tolist()

    reclassification_idx_ls = []

    for idx, title in zip(index_ls, title_ls):
        if any(stock_name in title for stock_name in stock_name_ls):
            reclassification_idx_ls.append(idx)

    data.loc[reclassification_idx_ls, 'Section'] = 'business'

    return data


def to_stock(data):
    '''
    경제, 기업, health, special_edition 기사의 일부를
    증권(stock)기사로 재분류 하는 함수입니다.

    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame
    '''
    section_ls = ['economy', 'business','health','special_edition']
    temp_df = data.loc[data['Section'].isin(section_ls)]

    index_ls = temp_df.index
    title_ls = temp_df['Title'].tolist()

    to_stock_idx_ls = []
    keyword_ls = ['코스피', '코스닥', '증시','주가','주식','목표가','상장','특징주', '증자',
                  '영업익', '공시', '지분', '매출','이익', 'Hot-Line', '펀드',
                  '키움증권','NH투자','KB증권','미래에셋대우','신한금투','대신증권', 'KTB투자증권',
                  '한투증권', '현대차투자증권', '유안타증권', '유진투자', '메리츠종금']

    for idx, title in zip(index_ls, title_ls):
        if any(keyword in title for keyword in keyword_ls):
            to_stock_idx_ls.append(idx)

    data.loc[to_stock_idx_ls, 'Section'] = 'stock'
    return data


def to_economy(data):
    '''
    기업, 증권 기사에서 경제(economy)기사로 재분류 하는 함수입니다.

    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame
    '''
    section_ls = ['stock', 'business', 'special_edition']
    temp_df = data.loc[data['Section'].isin(section_ls)]

    index_ls = temp_df.index
    title_ls = temp_df['Title'].tolist()

    reclassification_ls = []
    keyword_ls = ['경제', '업종','환율','핀테크','산업혁명','가상화폐','비트코인', '금리','유가',
                  '임금',]

    for idx, title in zip(index_ls, title_ls):
        if any(keyword in title for keyword in keyword_ls):
            reclassification_ls.append(idx)

    data.loc[reclassification_ls, 'Section'] = 'stock'
    return data



def reclassify_culture(data):
    '''
    entertainment와 culture 기사에서 culture & art로 재분류 하는 함수입니다.
    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame
    '''
    # 1. 일기예보(weather-forecast) 분류
    temp_df = data[data['Section'] == 'culture']
    index_ls = temp_df.index
    title_ls = temp_df['Title'].tolist()

    keyword_ls = ['기온', '날씨','온도', '영하', '한파', '눈', '추위', '폭설', '적설량', '대설',
                  '영상', '낮','폭염', '비','더위','폭우','강수량', '장마',
                  '쌀쌀','맑고','구름','미세먼지','안개','바람','찜통']

    reclassification_index_ls = []
    for idx, title in zip(index_ls, title_ls):
        if any(keyword in title for keyword in keyword_ls):
            reclassification_index_ls.append(idx)

    data.loc[reclassification_index_ls, 'Section'] = 'weather-forecast'

    # 2. culture & art 분류
    section_ls = ['entertainment','culture']
    temp_df = data[data['Section'].isin(section_ls)]

    keyword_ls = ['박물관','미술','전시','신간','작품','피아니스트','바이올','아트',
              '예술','유물','展','소설','수필','문학','에세이','발간','출간','사진',
              '뮤지컬','영화','개봉','완간']

    index_ls = temp_df.index
    title_ls = temp_df['Title'].tolist()

    reclassfication_ls = []
    for idx, title in zip(index_ls, title_ls):
        if any(keyword in title for keyword in keyword_ls):
            reclassfication_ls.append(idx)

    data.loc[reclassfication_ls, 'Section'] = 'culture & art'

    return data.drop(data[data['Section'] == 'culture'].index)


def drop_categories(data, drop_category_ls):
    '''
    ' ', people, opinion, special_edition 카테고리 및, 전체 비중의 0.1% 이하를 차지하는 기사를 전부 제거

    inputs
    =================================
    data : pandas.DataFrame
        크롤링을 마친 raw data 상태의 DataFrame

    drop_category_ls : str, list
        제거 대상 category 목록
    '''

    # 원하지 않는 카테고리는 제거
    if type(drop_category_ls) == str:
        drop_category_ls = [drop_category_ls]

    drop_index_ls = data[data['Section'].isin(drop_category_ls)].index
    data.drop(drop_index_ls, inplace = True)

    # 전체 비중의 0.1% 이하의 카테고리 제거
    ratio_huddle = len(data) // 1000

    total_category_ls = list(set(data['Section']))
    counter = Counter(data['Section'])

    useful_category_ls = []
    for category in total_category_ls:
        if counter[category] > ratio_huddle:
            useful_category_ls.append(category)

    return data.loc[data['Section'].isin(useful_category_ls)]


def drop_duplicated_rows(data):
    '''
    동일한 내용이 중복되는 기사를 제거하는 함수입니다.
    '''

    print('원본 기사의 수 : %s'%len(data))

    data = data.drop_duplicates(subset = 'Text', keep = 'first')
    print('중복되는 기사들을 제거하였습니다')
    print('중복을 제거한 기사의 수 : %s'%len(data))

    return data





def main_reclassify(data):
    # 상장된 기업명이 담긴 list 파일 load
    with open('stock_name_ls.pickle', 'rb') as f:
        stock_name_ls = pickle.load(f)

    # 카테고리 정리, 통합
    #data = drop_uselsess_data(data)
    data = reclassify_categories(data,
                                ['tv_broadcasting', 'entertainment_topic', 'broadcasting', 'hot_issue', 'music', 'overseas_etn'],
                                'entertainment')

    data = reclassify_categories(data, 'golf', 'sports')
    data = reclassify_categories(data, ['movie','performance'], 'culture & art')
    data = reclassify_categories(data, 'patent', 'technology')
    data = reclassify_categories(data,
                                ['retail', 'it', 'financial', 'electronics', 'autos', 'chemistry', 'heavy_industries'],
                                'business')

    data = to_business(data, stock_name_ls)
    data = to_stock(data)
    data = to_economy(data)
    data = reclassify_culture(data)

    data = drop_categories(data, [' ','opinion','people','special_edition','photo_news'])
    return data




def main_tokenize(data):
    # 자연어처리 클래스 생성
    nlp = kotxt.NLP()

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
    return data



if __name__ == '__main__':

    # 저장 위치의 폴더가 존재하지 않으면, 폴더 생성
    save_folder_path = 'data/tokenized/'
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    # 10,000개씩 쪼개진 데이터에 대한 Reclassfication, Tokenizing
    for year in [2018, 2017]:
        for start,end in zip(range(0,100), range(1,101)):
            try:
                file_name = 'MK_%s_No_%s_to_%s.csv'%(year, start*10000, end*10000)
                # 이미 파일이 있으면 건너뜀
                if os.path.isfile(save_folder_path + file_name):
                    continue

                # 데이터 loading
                path_to_file = 'data/raw/' + file_name
                temp_df = kotxt.load_data(path_to_file)

                # 카테고리 재분류
                temp_df = main_reclassify(temp_df)

                # 중복 기사 제거
                temp_df = drop_duplicated_rows(temp_df)

                # Tokenize
                temp_df = main_tokenize(temp_df)


                # 결과 저장
                save_file_path = save_folder_path + file_name
                kotxt.save_data(data = temp_df, path_to_file = save_file_path, index = False)
                print('%s년도의 %s번째 batch에 대한 reclassification & tokenizing을 완료하였습니다.'%(year,end))
                print('=========================================================')
            except:
                break
