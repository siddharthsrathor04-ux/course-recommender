import pandas as pd
import numpy as np
from data_preprocessing import create_sample_dataset
from synthetic_data import create_interaction_dataset
from content_model import ContentBasedRecommender
from collaborative_model import CollaborativeFilteringRecommender

def baseline_recommendations(interactions_df, courses_df, top_n=5):
    course_stats = interactions_df.groupby('course_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    course_stats.columns = ['course_id', 'avg_rating', 'num_ratings']
    course_stats = course_stats[course_stats['num_ratings'] >= 2].sort_values('avg_rating', ascending=False).head(top_n)
    
    return course_stats.merge(
        courses_df[['course_id', 'course_name', 'difficulty_level']],
        on='course_id'
    )[['course_id', 'course_name', 'difficulty_level', 'avg_rating']].reset_index(drop=True)

def get_hybrid_recommendations(user_id, interactions_df, courses_df, 
                               content_model, collab_model, top_n=5):
    content_recs = content_model.get_user_recommendations(
        interactions_df[interactions_df['user_id'] == user_id],
        top_n=10
    )
    
    user_based_recs = collab_model.get_user_based_recommendations(user_id, courses_df, top_n=10)
    item_based_recs = collab_model.get_item_based_recommendations(user_id, courses_df, top_n=10)
    
    all_recs = []
    
    if not content_recs.empty:
        content_recs['score'] = content_recs.get('weighted_score', 3.0)
        content_recs['method'] = 'content'
        all_recs.append(content_recs[['course_id', 'course_name', 'difficulty_level', 'score', 'method']])
    
    if not user_based_recs.empty:
        user_based_recs['score'] = user_based_recs['predicted_rating']
        user_based_recs['method'] = 'user_based_cf'
        all_recs.append(user_based_recs[['course_id', 'course_name', 'difficulty_level', 'score', 'method']])
    
    if not item_based_recs.empty:
        item_based_recs['score'] = item_based_recs['predicted_rating']
        item_based_recs['method'] = 'item_based_cf'
        all_recs.append(item_based_recs[['course_id', 'course_name', 'difficulty_level', 'score', 'method']])
    
    if not all_recs:
        return baseline_recommendations(interactions_df, courses_df, top_n)
    
    combined = pd.concat(all_recs, ignore_index=True)
    combined['score'] = (combined['score'] - combined['score'].min()) / (combined['score'].max() - combined['score'].min() + 1e-6)
    
    aggregated = combined.groupby('course_id').agg({
        'score': 'mean',
        'course_name': 'first',
        'difficulty_level': 'first'
    }).reset_index().sort_values('score', ascending=False).head(top_n)
    
    return aggregated.rename(columns={'score': 'hybrid_score'})[['course_id', 'course_name', 'difficulty_level', 'hybrid_score']].reset_index(drop=True)

def main():
    print("Loading and preprocessing courses...")
    courses_df = create_sample_dataset()
    print(f"Loaded {len(courses_df)} courses")
    
    print("\nGenerating synthetic users and interactions...")
    users_df, _, interactions_df = create_interaction_dataset(
        None,
        num_users=750,
        interaction_sparsity=0.15
    )
    print(f"Generated {len(users_df)} users and {len(interactions_df)} interactions")
    
    print("\nBuilding content-based recommender...")
    content_model = ContentBasedRecommender(courses_df)
    
    print("Building collaborative filtering models...")
    collab_model = CollaborativeFilteringRecommender(interactions_df)
    
    print("\nBaseline Recommendations (Top-Rated Courses):")
    baseline_recs = baseline_recommendations(interactions_df, courses_df, top_n=5)
    print(baseline_recs.to_string(index=False))
    
    sample_user = users_df.iloc[0]['user_id']
    print(f"\n\nHybrid Recommendations for {sample_user}:")
    hybrid_recs = get_hybrid_recommendations(
        sample_user, interactions_df, courses_df, content_model, collab_model, top_n=5
    )
    print(hybrid_recs.to_string(index=False))
    
    sample_user_2 = users_df.iloc[50]['user_id']
    print(f"\n\nHybrid Recommendations for {sample_user_2}:")
    hybrid_recs_2 = get_hybrid_recommendations(
        sample_user_2, interactions_df, courses_df, content_model, collab_model, top_n=5
    )
    print(hybrid_recs_2.to_string(index=False))

if __name__ == '__main__':
    main()