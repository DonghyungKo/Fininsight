import time
from selenium import webdriver
import pandas as pd
import re
import datetime
from collections import OrderedDict
import konlpy

        
from konlpy.tag import *
import konlpy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class NaverNewsCrawler(object):
    
    '''
    입력한 키워드로 검색한 뉴스를 크롤링하는 클래스입니다.
    제목, 본문요약, 등록일, 언론사명를 기록합니다.
    
    crawl_process 매서드를 사용하시면 됩니다.
    '''
    
    
    def __init__(self, keyword):
        
        self.keyword = keyword
        self.now = datetime.datetime.now()

        self.item = OrderedDict(\
                       {'title' : [],
                        'text' : [],
                        'date' : [],
                        'publisher' : []
                       })
        
        
        # ?분 전, ?시간 전, ?일 전으로 표시되는 날짜를 정확히 계산하기 위한 정규표현식
        self._p_today = re.compile('.시간|.분')
        self._p_days_age = re.compile('.일 전')
        self._p_date = re.compile('[0-9]*\.[0-9]*\.[0-9]*')
        
        
        # Driver 열기
        path = "D:\chromedriver.exe"
        self.driver = webdriver.Chrome(path)
        self.driver.get("https://search.naver.com/search.naver?where=news&sm=tab_jum&query=%s"%keyword)
        time.sleep(3)
        
        
    
    
    
    
    # 크롤링한 title에서 특수문자를 제거하는 함수입니다.
    def _clean_title(self,text):
        text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]', '' , text)
        return text
    
    
    
    
    
    
    # 크롤링한 text에서 불필요한 부분들을 제거하는 함수입니다.
    def _clean_text(self,text):
        
        # () 와 [], <>로 감싸진 부분들은 전부 제거
        text = re.sub('\(.+?\)', '',string = text)
        text = re.sub('\[.+?\]', '',string = text)
        text = re.sub('\<.+?\>', '',string = text)
        text = re.sub('◀.+?▶', '',string = text)
        text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]▲', '' , text) # 특수문자 제거
        text = re.sub('... 기자', '', text)   # 기자라는 단어도 제거
        
        return text
        
        
        
        
        
    # 크롤링한 날짜를 YYYYmmdd 형식으로 반환하는 함수입니다.
    def _clean_date(self, text):
        
        # ?분 전, ?시간 전으로 표시되어 있다면 오늘 날짜로 치환
        if self._p_today.search(text):
            date = self.now.strftime("%Y%m%d")
        
        # ?일 전으로 표시되어 있다면 n일 전의 날짜로 치환
        elif self._p_days_age.search(text):
            n = self._p_days_age.search(text).group()[0] # n일 전의 n을 추출
            
            date = self.now - datetime.timedelta(days = int(n))
            date = date.strftime("%Y%m%d")
            
        else:
            date = self._p_date.search(text).group().replace('.','')
            
        return date
            
        
        
        
        
    # 크롤링한 publisher에서 불필요한 단어들을 제거하는 함수입니다.
    def _clean_publisher(self, text):
        return text.replace('언론사', '').replace('선정', '').replace('  ', '')
    
    
    
    
    # 타이틀을 크롤링하는 함수입니다.
    def _crawl_title(self):
        title_ls = [self._clean_title(title.text) for title in self.driver.find_elements_by_class_name(' _sp_each_title')]
        return title_ls
        
        
    # 본문 요약(text)을 크롤링하는 함수입니다.
    def _crawl_text(self):
        text_ls = [self._clean_text(text.text) for text in self.driver.find_elements_by_css_selector('dl > dd:nth-child(3)') if not text.text == '']
        return text_ls
    
    
    # 날짜를 크롤링하는 함수입니다.
    def _crawl_date(self):
        date_ls = [self._clean_date(date.text) for date in self.driver.find_elements_by_css_selector('dl > dd.txt_inline')]
        return date_ls
        
        
    # 언론사명를 크롤링하는 함수입니다.
    def _crawl_publisher(self):
        publisher_ls = [self._clean_publisher(title.text) for title in self.driver.find_elements_by_class_name(' _sp_each_source')]
        return publisher_ls
    
    
    
    # 메인함수입니다.
    def crawl_process(self, n_page, sleep_time = 0.5):
        self.item = OrderedDict(\
                       {'title' : [],
                        'text' : [],
                        'date' : [],
                        'publisher' : []
                       })
        
        for page_cnt in range(n_page):
            
            # 크롤링을 수행합니다.
            self.item['title'] += self._crawl_title()
            self.item['text'] += self._crawl_text()
            self.item['date'] += self._crawl_date()
            self.item['publisher'] += self._crawl_publisher()
            
            
            # 다음 페이지로 넘어갑니다.
            try:
                self.driver.find_element_by_css_selector('#main_pack > div.news.mynews.section._prs_nws > div.paging > a.next').click()
                time.sleep(sleep_time)
            except:
                self.data = pd.DataFrame(self.item)
                self.data.to_csv('%s_%s.csv'%(self.keyword, self.now.strftime('%Y%m%d')), index = False)
                return self.item
        
        self.data = pd.DataFrame(self.item)
        self.data.to_csv('%s_%s.csv'%(self.keyword, self.now.strftime('%Y%m%d')), index = False)
        return self.item

    
    
    
    
    
    
    

class NLP(object):
    
    '''
    크롤링한 데이터에 대한 전처리를 위해 사용하는 클래스입니다.
    명사 추출, 어근 추출 및 tf-idf 기반의 키워드 추출을 제공합니다.
    '''
    
    
    def __init__(self):
        self.twit = Twitter()
        self.kkma = Kkma()
        
        
    def remove_words(self, word_ls,text_ls):
        for word in word_ls:
            text_ls = [text.replace(word,'') for text in text_ls]

        return text_ls

    
    
    def extract_nouns_twitter(self, text_ls):

        temp_nouns = self.twit.nouns(' '.join(text_ls))
        
        # 한글자 단어는 제외
        noun_ls = [x for x in temp_nouns if len(x) > 1]

        print('총 단어 수 :', len(nouns))
        print('중복을 제거한 단어의 수 :', len(set(nouns)))

        return noun_ls
    
    def extract_morphs_twitter(self, text_ls):

        temp_morphs = self.twit.morphs(' '.join(text_ls))
        morphs = [x for x in temp_morphs if len(x) > 1]

        print('총 단어 수 :',len(morphs))
        print('중복을 제거한 단어의 수 :', len(set(morphs)))

        return morphs

    
    def extract_nowns_for_each_document(self, doc_ls):

        total_ls = []

        # ls는 하나의 게시글을 개별 값으로 담고있는 리스트이다.
        for doc in doc_ls:
            noun_ls = self.twit.nouns(doc)

            total_ls += [[x for x in noun_ls if len(x) > 1]]

        return total_ls
    
    
    def extract_morphs_for_each_document(self, doc_ls):

        total_ls = []

        # ls는 하나의 게시글을 개별 값으로 담고있는 리스트이다.
        for i,doc in enumerate(doc_ls):
            
            try:
                morph_ls = self.twit.morphs(doc)
                total_ls += [[x for x in morph_ls if len(x) > 1]]
            except:
                print(i)

        return total_ls
    
    
    
    def make_tf_idf_df(self, doc_ls, min_df = 0.02, max_df = 0.2, if_nown = True):
        
        if if_nown :
            word_by_document_ls = self.extract_nowns_for_each_document(doc_ls)
        else :
            word_by_document_ls = self.extract_morphs_for_each_document(doc_ls)
        
        corpus_for_tfidf_ls = [' '.join(x) for x in word_by_document_ls]
        
        tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df).fit(corpus_for_tfidf_ls)
        tfidf_array = tfidf_vectorizer.transform(corpus_for_tfidf_ls).toarray()

        vocab = tfidf_vectorizer.get_feature_names()

        tfidf_df = pd.DataFrame(tfidf_array, columns = vocab)
        keyword_ls = tfidf_df.idxmax(axis=1)
        
        return tfidf_df, keyword_ls