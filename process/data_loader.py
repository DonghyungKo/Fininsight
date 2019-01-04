
# 데이터 불러옴
def read_txt(path_to_file):
    body_ls = []
    category_ls = []

    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                category, body = line.split('\t')
                category_ls.append(category)
                body_ls.append(body)
            except:
                print(i)
    return body_ls, category_ls
