# -*- coding: utf-8 -*-
import pandas as pd
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import os
from multiprocessing import Process,Queue, Pool
import functools
import warnings
warnings.filterwarnings(action='ignore')
import time

class MKNewsCrawler(object):

    def __init__(self):
        # 기존에 데이터가 없으면 새로 만들어서 크롤링

        # 자료가 저장되는 형태
        self.item = OrderedDict(\
                       {'Title' : [],
                        'Text' : [],
                        'Date' : [],
                        'Section' : [],
                        'Link' : [],
                       })

        #self._pool = Pool(processes = 8)



    def make_link_map(self, start_num, end_num, year = 2018):
        '''
        크롤링할 link map을 만드는 함수입니다.

        inputs
        ======================
        start_num : int,
            크롤링을 시작하는 article_number

        end_num : int,
            크롤링을 마치는 article_number

        year : int,
            크롤링을 수행하는 기사년도

        return
        ==========================
        link_map : list,
            크롤링을 수행할 link가 저장된 리스트
        '''

        link_map_ls = []

        for article_num in range(start_num, end_num):
            url = 'http://news.mk.co.kr/newsRead.php?&year=%s&no=%s'%(year,article_num)
            link_map_ls.append(url)

        return link_map_ls



    def _crawl_one_article(self, link):
        '''
        해당 링크에 request를 보내, html에서 css태그로 원하는 정보를 추출하는 함수입니다.

        inputs
        ====================
        link : str
         - 기사 원문이 있는 url
        '''
        try:
            req = requests.get(link)
            soup = BeautifulSoup(req.content.decode('euc-kr','replace'), 'html.parser')

            title = soup.select('#top_header > div > div > h1')[0].get_text()
            text = soup.select('#article_body > div')[0].get_text()
            date = int(soup.find('meta', {'property' :'article:published'})['content'].replace('-',''))

            # section의 분류의 경우,
            # html을 뜯어보면 세부 카테고리가 존재함..
            section = soup.find('meta', {'name' :'classification'})['content']
            return title, text, date, section, link
        except:
            return '','','','',''



    def crawl_process(self, link_map, queue = False):
        '''
        입력받은 link_map에서 개별 링크에서 정보를 수집하는 함수입니다.

        inputs
        ======================
        link_map : list,
            크롤링할 링크가 담겨있는 리스트

        queue : Queue, [optional]
            병렬처리를 위한 Queue

        return
        ======================

        '''
        temp_dict = self.item.copy()

        # 링크 맵을 돌면서 기사를 하나씩 수집
        for link in link_map:
            title, text, date, section, link = self._crawl_one_article(link)

            if not title == '':
                temp_dict['Title'].append(title)
                temp_dict['Text'].append(text)
                temp_dict['Date'].append(date)
                temp_dict['Section'].append(section)
                temp_dict['Link'].append(link)

        # 병렬처리를 위해 queue에 쌓음
        if queue:
            queue.put(temp_dict)

        print('batch done!')
        return temp_dict



    def multiprocess_crawling(self, link_map, n_batch = 30):
        '''
        입력받은 link_map에서 개별 링크에서 정보를 수집하는 함수입니다.
        병렬처리가 적용되었습니다. [os.core_count() * 2]

        inputs
        ======================
        link_map : list,
            크롤링할 링크가 담겨있는 리스트

        n_batch : int,
            병렬처리할 Queue의 수
        return
        ======================

        '''
        queue_ls = []
        procs = []

        result_dict = self.item.copy()

        if n_batch < len(link_map):
            batch_size = len(link_map)// (n_batch)
        else:
            batch_size = 1
        print('batch size : %s'%batch_size)

        # process에 작업들을 할당
        for i, idx in enumerate(range(0, len(link_map), batch_size)):
            try:
                batch_link_map = link_map[idx : idx + batch_size]
            except:
                batch_link_map = link_map[idx :]

            queue_ls.append(Queue())
            proc = Process(target=functools.partial(
                                                    self.crawl_process,
                                                    queue = queue_ls[i],
                                                    link_map= batch_link_map))
            procs.append(proc)
            proc.start()

        for queue in queue_ls:
            temp_result_dict = queue.get()
            queue.close()

        for key in result_dict.keys():
            result_dict[key] += temp_result_dict[key]

        for proc in procs:
            proc.join()
            #proc.close()

        return result_dict

if __name__=='__main__':
    start_time = time.time()

    mk_crawler = MKNewsCrawler()
    start_num = int(input('크롤링을 시작할 article number를 입력하세요 :  '))
    end_num = int(input('크롤링을 마칠 article number를 입력하세요 :  '))
    year = int(input('크롤링할 년도를 입력하세요 (YYYY) :  '))
    n_batch = int(input('병렬처리할 batch의 수를 입력하세요 (default = 50) : '))

    # 크롤링 할 기사가 10,000개 이상인 경우, 10,000개씩 끊은 후, 순차적으로 병렬처리 적용
    batch_size = 100

    for start_idx, end_idx in zip(range(start_num, end_num +1, batch_size),
                                  range(start_num + batch_size, end_num +1, batch_size)):

        link_map = mk_crawler.make_link_map(start_idx, end_idx, year = year)
        print('Start Crawling from %s to %s in %s'%(start_idx, end_idx, year))

        result_dict = mk_crawler.multiprocess_crawling(link_map, n_batch = n_batch)
        pd.DataFrame(result_dict).to_csv('../data/MK_%s_No_%s_to_%s.csv'%(year,start_idx, end_idx))
        print('Data Saved as ../data/MK_%s_No_%s_to_%s.csv'%(year, start_idx, end_idx))


    print('총 소요시간 : %s'%(time.time() - start_time))
