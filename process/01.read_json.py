import requests
import json
from collections import Counter


def read_json(size=10000):
    req = requests.get('http://52.231.65.246:9200/test/news/_search/?size=%s'%size)

    json_dict = json.loads(req.content.decode())

    body_ls = []
    category_ls = []
    try:
        for article in json_dict['hits']['hits']:
            body = article['_source']['body'].replace('\t',' ').replace('\n',' ')
            body_ls.append(body)

            category = article['_source']['category']
            category_ls.append(category)
    except:
        pass

    return body_ls, category_ls

def main():
    body_ls, category_ls = read_json(size=10000)
    with open('news.txt', 'w') as f:
        for category, body in zip(category_ls, body_ls):
            f.write('%s\t%s\n'%(category, body))
    return

if __name__=='__main__':
    main()
    print('데이터를 성공적으로 불러왔습니다.')
