import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_coursera_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_course_data(df):
    df = df.dropna(subset=['course_id', 'course_name'])
    df['course_id'] = df['course_id'].astype(str)
    df['course_name'] = df['course_name'].astype(str)
    
    if 'difficulty_level' in df.columns:
        df['difficulty_level'] = df['difficulty_level'].fillna('Intermediate')
    
    if 'skills' in df.columns:
        df['skills'] = df['skills'].fillna('')
    
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
    else:
        df['rating'] = 3.0
    
    if 'course_description' in df.columns:
        df['course_description'] = df['course_description'].fillna('')
    else:
        df['course_description'] = ''
    
    return df.drop_duplicates(subset=['course_id'])

def normalize_ratings(df):
    scaler = MinMaxScaler(feature_range=(1, 5))
    if 'rating' in df.columns:
        df['rating'] = scaler.fit_transform(df[['rating']])
    return df

def preprocess_pipeline(filepath):
    df = load_coursera_data(filepath)
    df = clean_course_data(df)
    df = normalize_ratings(df)
    return df

def create_sample_dataset():
    sample_data = {
        'course_id': [f'C{i:04d}' for i in range(1, 51)],
        'course_name': [f'Course {i}' for i in range(1, 51)],
        'difficulty_level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 50),
        'skills': [', '.join(np.random.choice(['Python', 'ML', 'Data Science', 'Web Dev', 'Statistics'], size=np.random.randint(1, 4), replace=False)) for _ in range(50)],
        'rating': np.random.uniform(2.5, 5.0, 50),
        'course_description': [f'This course covers fundamental concepts in topic {i}' for i in range(1, 51)]
    }
    df = pd.DataFrame(sample_data)
    return clean_course_data(df)