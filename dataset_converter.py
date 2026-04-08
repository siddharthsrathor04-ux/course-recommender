import pandas as pd
import os
import zipfile

def extract_ml_100k(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_and_transform_data(data_dir):
    # Load u.data
    u_data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', header=None,
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Load u.item
    u_item = pd.read_csv(os.path.join(data_dir, 'u.item'), sep='|', encoding='latin-1', header=None,
                         names=['item_id', 'movie_title'] + [f'genre_{i}' for i in range(19)])

    # Rename columns
    u_data.rename(columns={'item_id': 'course_id'}, inplace=True)
    u_item.rename(columns={'movie_title': 'course_title'}, inplace=True)

    # Convert genre columns to single category column
    genre_columns = [f'genre_{i}' for i in range(19)]
    u_item['genres'] = u_item[genre_columns].apply(lambda row: '|'.join([str(i) for i in row if i == 1]), axis=1)
    u_item = u_item[['item_id', 'course_title', 'genres']]

    return u_data, u_item


def save_datasets(ratings, courses, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ratings.to_csv(os.path.join(output_dir, 'ratings.csv'), index=False)
    courses.to_csv(os.path.join(output_dir, 'courses.csv'), index=False)


def main():
    zip_path = 'ml-100k.zip'
    extract_to = 'ml-100k'
    data_dir = 'ml-100k'
    output_dir = 'data'

    extract_ml_100k(zip_path, extract_to)
    ratings, courses = load_and_transform_data(data_dir)
    save_datasets(ratings, courses, output_dir)


if __name__ == '__main__':
    main()