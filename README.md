## Fininsight
 - Research project with Fininsight Co.
 

## Data Crawling (데이터 수집)
 + 네이버 뉴스에 키워드로 검색한 결과를 수집
   ko_crawler.py모듈의 NaverNewsCrawler 클래스
     - 뉴스 제목, 본문 요약, 발행일, 언론사명 
   
 
 + ### 매일경제의 전체기사를 일별로 수집
   #### ko_crawler.py모듈의 MKCrawler 클래스
     - 매일경제의 뉴스를 수집합니다. 
     - 메인 함수 crawl_process를 사용하시면 됩니다. 
     
     - section, 시작일, 크롤링할 일 수, 일당 페이지 수의 입력변수를 받습니다.
       section 명 : [전체기사, 경제, 기업, 사회, 국제, 부동산, 증권, 정치, IT과학, 문화]

     - section은 iterable 혹은 str타입의 변수를 입력받습니다.        
       start_date = "YYYYMMDD" 형태로 입력
       n_days = 크롤링 할 과거 n일
       n_page = 하루에 크롤링 할 최대 페이지 수 (한 페이지에 25개의 뉴스)

   
   
   
