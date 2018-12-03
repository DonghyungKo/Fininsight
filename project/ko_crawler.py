# -*- coding: utf-8 -*-
import sys

import time
from selenium import webdriver
import pandas as pd
import re
import datetime
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import os
from multiprocessing import Process,Queue, Pool
import functools
import warnings
warnings.filterwarnings(action='ignore')


    
    
    
    
    
class MKNewsCrawler(object):
    
    '''
    매일경제 전체기사를 일별로 크롤링하는 클래스입니다.
    제목, 본문요약, 등록일을 기록합니다.
    
    crawl_process 매서드를 사용하시면 됩니다.
    '''
    
    
    def __init__(self):
        
        self.today = datetime.datetime.today()
        
        # 데이터가 있으면 불러와서 축적한다
        try:
            self.item = pd.read_csv('Data/MK/news_data.csv', encoding = 'utf-8')
            self.crawled_date = set(self.item['Date'])
        
        # 기존에 데이터가 없으면 새로 만들어서 크롤링
        except:
            self.item = OrderedDict(\
                           {'Title' : [],
                            'Text' : [],
                            'Date' : [],
                            'Section' : [],
                            
                           })
            self.crawled_date = []
            self._pool = Pool(processes = 4)
        # Driver 열기
        #path = "chromedriver"
        #self.driver = webdriver.Chrome(path)
    
    
    

    # 타이틀을 크롤링하는 함수입니다.
    # 한 페이지에 있는 25개의 타이틀을 한번에 크롤링합니다.
    def _crawl_title(self, soup):
        return soup.find(class_='top_title').get_text()
    
    # 본문 요약(text)을 크롤링하는 함수입니다.
    # 개별 뉴스 본문에 접근한 후, 크롤링을 수행합니다.
    def _crawl_text(self, soup):
        return soup.find(class_='art_text').get_text()
        
        
    def _crawl_one_article(self, link):
        
        # Requesets와 BeautifulSoup를 사용하여 빠른 크롤링 수행
        # html를 css태그로 찾으면 class로 찾을 때보다 약 2.5배 빠름
        req = requests.get(link)
        soup = BeautifulSoup(req.content.decode('euc-kr','replace'), 'html.parser')
        
        title = soup.select('#top_header > div > div > h1')[0].get_text() 
        text = soup.select('#article_body > div')[0].get_text()
        date = int(soup.find('meta', {'property' :'article:published'})['content'].replace('-',''))
        
        # section의 분류의 경우, 
        # html을 뜯어보면 세부 카테고리가 존재함..
        section = soup.find('meta', {'name' :'classification'})['content']
        
        return title, text, date, section
    
    
    # YYYYMMDD일의 페이지에 접근한 후 해당 일자의 자료 수집하는 함수입니다.
    def _crawl_one_day(self, n_pages, base_url, start_page, section):
        
        title_ls = []
        text_ls = []
        date_ls = []
        section_ls = []
        
        #페이지 접속 (n_pages 까지)
        for i in range(start_page, start_page + n_pages):
            # 해당 페이지의 html 읽어오기
            req_page = requests.get(base_url + '&page=%s'%i)
            soup_page = BeautifulSoup(req_page.content.decode('euc-kr','replace'), 'html.parser')
            
            # 개별 뉴스 원문의 링크 수집
            temp_link_ls = [link['href'] for link in soup_page.select('#container_left > div.list_area > dl > dt > a')]
            
            
            # 개별 링크로 접속하여, 기사 제목과 본문 수집
            temp_title_ls = []
            temp_text_ls = []
            temp_date_ls = []
            temp_section_ls = []
            
            for link in temp_link_ls:
                try:
                    temp_title, temp_text, temp_date, temp_section = self._crawl_one_article(link)
                    temp_title_ls.append(temp_title)
                    temp_text_ls.append(temp_text)
                    temp_date_ls.append(temp_date)
                    temp_section_ls.append(temp_section)
                except:
                    pass
                
            title_ls += temp_title_ls
            text_ls +=  temp_text_ls
            date_ls += temp_date_ls
            section_ls += temp_section_ls
            
            if (i +1) %10 == 0:
                print('%s 칼럼의 %s페이지부터 %s페이지까지 수집하였습니다.'%(section, start_page, (i+1)))
                
        return title_ls, text_ls, date_ls, section_ls


            
    # 메인함수입니다.
    def crawl_process(self, section, queue = False,  
                      n_days = 1, 
                      n_pages = 1, 
                      start_date = datetime.datetime.today(), 
                      start_page = 0):
        
        ''' 
        메인 함수입니다.
        
        section, 시작일, 크롤링할 일 수, 일당 페이지 수의 입력변수를 받습니다.
        section 명 : [경제, 기업, 사회, 국제, 부동산, 증권, 정치, IT과학, 문화]

        ※만약 전체기사 이외의 다른 카테고리를 크롤링하시려면, n_days 파라미터를 1로 맞추고, start_page를 설정해주시기 바랍니다.
        
        section은 iterable 혹은 str타입의 변수를 입력받습니다.        
        start_date = "YYYYMMDD" 형태로 입력
        n_days = 크롤링 할 과거 n일
        n_page = 하루에 크롤링 할 최대 페이지 수 (한 페이지에 25개의 뉴스)
        '''
        
        section_code_dict = {'전체기사': '30000001',
                             '경제' : '30000016',
                             '기업' : '30000017',
                             '사회' : '30000022',
                             '국제' : '30000018',
                             '부동산' : '30000020',
                             '증권' : '30000019',
                             '정치' : '30000021',
                             'IT과학' : '30000037',
                             '문화' : '50500001',
                            }
        
        temp_dict = self.item.copy()

        # 입력받은 section 변수가 str 타입이면 list형태로 변환해준다
        if type(section) == str:
            section_ls = [section]
        else:
            section_ls = section
        
        
        # 입력받은 섹션의 뉴스에 접근
        for section in section_ls:
            section_code = section_code_dict[section]
            
            print('%s칼럼의 뉴스수집을 시작합니다.'%(section))
            # 과거 n일의 페이지에 접근
            for n in range(n_days):
                
                # YYYYMMDD 일의 뉴스가 있는 페이지에 접근
                crawling_day = (pd.to_datetime(start_date, format = '%Y%m%d') - datetime.timedelta(days = n)).strftime('%Y%m%d')
                
                if not n_days == 1:
                    print('현재 %s 칼럼의 %s일의 자료를 수집하고 있습니다.'%(section, crawling_day))
                
                base_url = 'http://news.mk.co.kr/newsList.php?sc=%s&thisDate=%s'%(section_code,crawling_day)       

                # 해당 일자에 대한 크롤링을 수행합니다.
                # 현재 페이지를 크롤링하며, n_pages까지 자료를 수집합니다.
                title_ls, text_ls, date_ls,section_ls  = self._crawl_one_day(n_pages = n_pages, 
                                                                             base_url = base_url, 
                                                                             start_page = start_page, 
                                                                             section = section)
                
                temp_dict['Title'] += title_ls
                temp_dict['Text'] += text_ls
                temp_dict['Date'] += date_ls
                temp_dict['Section'] += section_ls
        
        if queue:
            queue.put(temp_dict)
        
        return self.item

    
    def multiple_crawl_process(self, section_ls, start_date = datetime.datetime.today() , n_days = 1, start_page = 0, n_pages = 1):

        '''
            # multiprocessing을 적용하여 수집 속도를 향상시킨 함수입니다.

            section 명 : [전체기사, 경제, 기업, 사회, 국제, 부동산, 증권, 정치, IT과학, 문화]

            ※만약 전체기사 이외의 다른 카테고리를 크롤링하시려면, n_days 파라미터를 1로 맞추고, start_page를 설정해주시기 바랍니다.

            section은 iterable 혹은 str타입의 변수를 입력받습니다.        
            start_date = "YYYYMMDD" 형태로 입력
            n_days = 크롤링 할 과거 n일
            start_pages = 시작하는 페이지 수
            n_page = 일별 크롤링 할 최대 페이지 수 (한 페이지에 25개의 뉴스)
            
            return: dictionary, key_ls = [title, text, date, section]
        '''    

        queue_ls = []
        procs = []
        total_dict = {'Title' : [],
                      'Text' : [],
                      'Date' : [],
                      'Section' : []}


        for idx, section in enumerate(section_ls):
            queue_ls.append(Queue())
            proc = Process(target=functools.partial(self.crawl_process,
                                                          queue = queue_ls[idx],
                                                          start_date = start_date,
                                                          n_days = n_days,
                                                          start_page = start_page,
                                                          n_pages = n_pages),

                                                          args=(section,))
            procs.append(proc)
            proc.start()

        for queue in queue_ls:
            result = queue.get()

            for key in result.keys():
                total_dict[key] += result[key]

        for proc in procs:
            proc.join()
            #proc.close()

        return total_dict

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    
    