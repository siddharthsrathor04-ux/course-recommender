import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBaseline
from surprise.model_selection import train_test_split

class CollaborativeFilteringRecommender:
    def __init__(self, interactions_df):
        self.interactions_df = interactions_df.copy()
        self.user_item_matrix = None
        self.user_based_model = None
        self.item_based_model = None
        self.build_models()
    
    def build_models(self):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            self.interactions_df[['user_id', 'course_id', 'rating']],
            reader
        )
        
        trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
        
        self.user_based_model = KNNBaseline(
            k=10,
            sim_options={'name': 'cosine', 'user_based': True}
        )
        self.user_based_model.fit(trainset)
        
        self.item_based_model = KNNBaseline(
            k=10,
            sim_options={'name': 'cosine', 'user_based': False}
        )
        self.item_based_model.fit(trainset)
        
        self.trainset = trainset
    
    def get_user_based_recommendations(self, user_id, courses_df, top_n=5):
        user_courses = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].values
        all_courses = courses_df['course_id'].values
        unseen_courses = [c for c in all_courses if c not in user_courses]
        
        predictions = []
        for course_id in unseen_courses:
            try:
                pred = self.user_based_model.predict(user_id, course_id)
                predictions.append({
                    'course_id': course_id,
                    'predicted_rating': pred.est
                })
            except:
                pass
        
        if not predictions:
            return pd.DataFrame()
        
        recommendations = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False).head(top_n)
        
        return recommendations.merge(
            courses_df[['course_id', 'course_name', 'difficulty_level']],
            on='course_id'
        ).reset_index(drop=True)
    
    def get_item_based_recommendations(self, user_id, courses_df, top_n=5):
        user_courses = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].values
        all_courses = courses_df['course_id'].values
        unseen_courses = [c for c in all_courses if c not in user_courses]
        
        predictions = []
        for course_id in unseen_courses:
            try:
                pred = self.item_based_model.predict(user_id, course_id)
                predictions.append({
                    'course_id': course_id,
                    'predicted_rating': pred.est
                })
            except:
                pass
        
        if not predictions:
            return pd.DataFrame()
        
        recommendations = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False).head(top_n)
        
        return recommendations.merge(
            courses_df[['course_id', 'course_name', 'difficulty_level']],
            on='course_id'
        ).reset_index(drop=True)