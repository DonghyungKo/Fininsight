# 데이터 불러오기
from data_loader import read_txt

# 전처리(클렌징)
import re

# 형태소 분석
from konlpy.tag import Okt
import konlpy

# 병렬처리
from multiprocessing import Process,Queue, Pool
import functools
from threading import Thread
from konlpy import jvm
import jpype
import queue


class Preprocessor(object):

    def __init__(self):
        '''
        전처리를 위한 함수들이 저장된 클래스입니다.
        '''

        self.twit = Okt()# 한 개의 문서에서 명사(noun)만 추출하는 함수

        # 정규 표현식 리스트
        self.regex_ls = [
                    '[\t\n\r\f\v]', #공백 제거,
            '\(.+?\)', '\[.+?\]', '\<.+?\>',  '◀.+?▶',  '=.+=', #특수문자 사이에 오는 단어 제거
            '(?<=▶).+', '(?<=▷).+', '(?<=※).+', #특수문자 다음으로 오는 단어 제거
            #'(?<=\xa0).+', # \xa0(증권 기사) 다음으로 오는 단어 제거
            '(?<=\Copyrights).+', # Copyrights 다음에 오는 기사 제거
            '[\w]+@[a-zA-Z]+\.[a-zA-Z]+[\.]?[a-z]*', #이메일 제거
            '[가-힣]+ 기자', '[가-힣]+ 선임기자',
            '[\{\}\[\]\/?,;·:“‘|\)*~`!^\-_+<>@○▲▶■◆\#$┌─┐&\\\=\(\'\"├┼┤│┬└┴┘|ⓒ]', #특수문자 제거
            #'[0-9]+[년월분일시조억만천원]*' , #숫자 단위 제거
        ]

        # 2. 제거 대상 단어 리스트
        self.word_to_be_cleaned_ls = ['한기재','헬스경향','디지털뉴스팀', '\u3000','Copyrights','xa0',
                                      'googletagdisplay', 'windowjQuery', 'documentwrite']

        # 3. 불용어 리스트
        self.stopword_ls = ['에서','으로','했다','하는','이다','있다','하고','있는','까지','이라고','에는',
                            '한다','지난','관련','대한','됐다','부터','된다','위해','이번','통해','대해',
                            '애게','했다고','보다','되는','에서는','있다고','한다고','하기','에도','때문',
                            '하며','하지','해야','이어','하면','따라','하면서','라며','라고','되고','단지',
                            '이라는','이나','한다는','있따는','했고','이를','있도록','있어','하게','있다며',
                            '하기로','에서도','오는','라는','이런','하겠다고','만큼','이는','덧붙였다','있을',
                            '이고','이었다','이라','있으며','있고','이며','했다며','됐다고','나타났다','한다며',
                            '하도록','있지만','된다고','되면서','그러면서','그동안','해서는','에게','밝혔다', '한편',
                            '최근', '있다는','보이','되지','정도','지난해','매년','오늘','되며','하기도', '지난달',
                            '하겠다는','했다세라','올해', '바로', '바랍니다', '함께','이후','따르면','같은','오후','모두',
                            '로부터','전날','면서','했다는','그리고','있던'
                           ]

    # text에 정규표현식을 적용하는 함수입니다.
    def clean_text(self,text):
        try:
            for regex in self.regex_ls:
                text = re.sub(regex, '', text)
        except:
            text = ' '
        return text

    # 복수 개의 문서를 클렌징하는 함수입니다.
    def clean_doc(self, doc_ls):
        '''
        정규표현식으로 문서를 전처리하는 함수입니다.

        input
        doc_ls : list(iterable)
            str형태의 text를 원소로 갖는 list type
        '''
        return [self.clean_text(doc) for doc in doc_ls]


    # 불용어를 제거하는 함수입니다.
    def remove_stopwords(self, token_doc_ls):
        '''
        불용어를 제거하는 함수입니다.

        input
        token_doc_ls : str, iterable
            token 형태로 구분된 문서가 담긴 list 형식
        '''

        total_stopword_set = set(self.stopword_ls)

        # input이 복수 개의 문서가 담긴 list라면, 개별 문서에 따라 단어를 구분하여 불용어 처리
        return_ls = []

        if token_doc_ls:
            if type(token_doc_ls[0]) == list:
                for doc in token_doc_ls:
                    return_ls += [[token for token in doc if not token in total_stopword_set]]

            elif type(token_doc_ls[0]) == str:
                return_ls = [token for token in token_doc_ls if not token in total_stopword_set]

        return return_ls

    def _extract_nouns_for_single_doc(self, doc):
        '''
        명사 추출
        '''
        clean_doc = self.clean_text(doc) # 클렌징
        token_ls = [x for x in self.twit.nouns(clean_doc) if len(x) > 1] # 토크나이징
        return self.remove_stopwords(token_ls) # 불용어 제거


    # 한 개의 문서에서 형태소(morphs)만 추출하는 함수
    def _extract_morphs_for_single_doc(self, doc):
        '''
        형태소 추출
        '''
        clean_doc = self.clean_text(doc) # 클렌징
        token_ls = [x for x in self.twit.morphs(clean_doc) if len(x) > 1] # 토크나이징
        return self.remove_stopwords(token_ls) # 불용어 제거



    # 모든 문서에서 명사(nouns)을 추출하는 함수.
    def extract_nouns_for_all_document(self,doc_ls, stopword_ls = []):
        '''
        모든 문서에서 명사를 추출하는 함수입니다.
        전처리를 적용하고 불용어를 제거한 결과를 반환합니다.

        input
        doc_ls : iterable, 원문이 str형태로 저장된 list

        '''
        jpype.attachThreadToJVM()
        # 전처리
        clean_doc_ls = self.clean_doc(doc_ls)

        # 명사 추출
        token_doc_ls = [self._extract_nouns_for_single_doc(doc) for doc in clean_doc_ls]

        # 불용어 제거
        return self.remove_stopwords(token_doc_ls)



    # 모든 문서에서 형태소(morph)를 추출하는 함수.
    def extract_morphs_for_all_document(self,doc_ls, stopword_ls = []):
        '''
        모든 문서에서 형태소(morph)를 추출하는 함수입니다.
        전처리를 적용하고 불용어를 제거한 결과를 반환합니다.

        input
        doc_ls : iterable, 원문이 str형태로 저장된 list

        return
        list : 각각의 문서가 토크나이징 된 결과를 list형태로 반환
        '''
        jpype.attachThreadToJVM()
        # 전처리
        clean_doc_ls = self.clean_doc(doc_ls)

        # 형태소(morph) 추출
        token_doc_ls = [self._extract_morphs_for_single_doc(doc) for doc in clean_doc_ls]

        # 불용어 제거
        return self.remove_stopwords(token_doc_ls)



    # 토크나이징을 병렬처리 하는데 사용되는 함수
    def _extract_nouns_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_nouns_for_single_doc(doc)) for idx, doc in tuple_ls]

    # 토크나이징을 병렬처리 하는데 사용되는 함수
    def _extract_morphs_for_multiprocessing(self, tuple_ls):
        jpype.attachThreadToJVM()
        # 멀티프로세싱의 경우, 병렬처리시 순서가 뒤섞이는 것을 방지하기위해,
        # [(idx, doc)] 형태의 tuple이 들어온다.
        return [(idx, self._extract_morphs_for_single_doc(doc)) for idx, doc in tuple_ls]

    # 토크나이징을 병렬처리 하기 위해, 작업을 분할하는 함수
    def split_list(self, ls, n):
        '''
        병렬처리를 위해, 리스트를 원하는 크기(n)로 분할해주는 함수입니다.
        '''
        result_ls = []
        for i in range(0, len(ls), n):
            result_ls += [ls[i:i+n]]

        return result_ls

    def extract_morphs_for_all_document_FAST_VERSION(self,
                                                     doc_ls,
                                                     n_thread = 4):
        jpype.attachThreadToJVM()

        '''
        멀티쓰레딩을 적용하여 속도가 개선된 버전입니다.
        문서들을 전처리하고 불용어(stopwords)를 제거한 후, Tokenzing하는 함수입니다.

        inputs
        doc_ls : iterable, 원문이 str 형태로 담겨있는 list를 input으로 받습니다.
        n_thread: int[default : 4], 사용하실 쓰레드의 갯수를 input으로 받습니다.
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

            temp_thread = Thread(target= lambda q, arg1: q.put(self._extract_morphs_for_multiprocessing(arg1)),  args = (que, tuple_ls))

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

        token_ls = [token for idx, token in sorted(zip(index_ls, token_ls))]
        token_ls = [' '.join(tokens) for tokens in token_ls]

        return token_ls



def main():
    # 데이터 로드
    print('데이터를 불러옵니다')
    body_ls, category_ls = read_txt('news.txt')

    # 전처리 클래스 생성
    pre = Preprocessor()

    # 전처리 수행
    print('문서 전처리와 형태소 분석을 수행합니다')
    print('데이터의 양에 따라 1 ~ 10분 정도 소요됩니다.')
    token_ls = pre.extract_morphs_for_all_document_FAST_VERSION(body_ls)

    # 결과 저장
    with open('news_tokenized.txt', 'w') as f:
    for category, body in zip(category_ls, clean_body_ls):
        f.write('%s\t%s\n'%(category, body))

    print('문서 전처리와 형태소 분석을 마쳤습니다')
    print('결과를 저장합니다.')
