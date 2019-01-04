from ko_text import *
import ko_text

def main():
    # tokenized된 파일을 읽어서 병합하는 함수입니다.
    result_dict = defaultdict(lambda : [])

    folder_path = 'data/tokenized/'
    file_ls = os.listdir(folder_path)

    for file_name in file_ls:
        file_path = folder_path + file_name
        temp_df = ko_text.load_data(file_path)

        for column in temp_df.columns:
            result_dict[column] += temp_df[column].tolist()

    df = pd.DataFrame(result_dict)
    df.drop(df[df['Section'] == 'photo_news'].index, inplace = True)
    df.to_csv('data/data_merged.csv', index = False)

if __name__=='__main__':
    main()
