import pandas as pd
import numpy as np

def generate_synthetic_users(num_users=750):
    skill_preferences = ['Python', 'ML', 'Data Science', 'Web Dev', 'Statistics', 'Cloud']
    difficulty_preferences = ['Beginner', 'Intermediate', 'Advanced']
    
    users = []
    for user_id in range(1, num_users + 1):
        user = {
            'user_id': f'U{user_id:05d}',
            'preferred_skill': np.random.choice(skill_preferences),
            'preferred_difficulty': np.random.choice(difficulty_preferences),
            'learning_speed': np.random.choice(['Slow', 'Medium', 'Fast'])
        }
        users.append(user)
    
    return pd.DataFrame(users)

def generate_interactions(users_df, courses_df, interaction_sparsity=0.15):
    interactions = []
    num_users = len(users_df)
    num_courses = len(courses_df)
    num_interactions = int(num_users * num_courses * interaction_sparsity)
    
    user_ids = users_df['user_id'].values
    course_ids = courses_df['course_id'].values
    
    sampled_pairs = np.random.choice(num_users * num_courses, size=num_interactions, replace=False)
    
    for pair_idx in sampled_pairs:
        user_idx = pair_idx // num_courses
        course_idx = pair_idx % num_courses
        
        user = users_df.iloc[user_idx]
        course = courses_df.iloc[course_idx]
        
        rating = calculate_rating(user, course)
        
        interactions.append({
            'user_id': user['user_id'],
            'course_id': course['course_id'],
            'rating': rating,
            'timestamp': np.random.randint(1, 100)
        })
    
    return pd.DataFrame(interactions)

def calculate_rating(user, course):
    base_rating = 2.5
    
    if user['preferred_skill'] in str(course['skills']):
        base_rating += 1.0
    
    difficulty_match = {
        'Beginner': {'Beginner': 1.2, 'Intermediate': 0.8, 'Advanced': 0.3},
        'Intermediate': {'Beginner': 0.7, 'Intermediate': 1.2, 'Advanced': 0.8},
        'Advanced': {'Beginner': 0.3, 'Intermediate': 0.8, 'Advanced': 1.2}
    }
    
    difficulty_factor = difficulty_match.get(user['preferred_difficulty'], {}).get(course['difficulty_level'], 1.0)
    base_rating *= difficulty_factor
    
    noise = np.random.normal(0, 0.3)
    final_rating = np.clip(base_rating + noise, 1.0, 5.0)
    
    return round(final_rating, 2)

def create_interaction_dataset(filepath, num_users=750, interaction_sparsity=0.15):
    from data_preprocessing import load_coursera_data, clean_course_data, create_sample_dataset
    
    if filepath:
        courses_df = load_coursera_data(filepath)
        courses_df = clean_course_data(courses_df)
    else:
        courses_df = create_sample_dataset()
    
    users_df = generate_synthetic_users(num_users)
    interactions_df = generate_interactions(users_df, courses_df, interaction_sparsity)
    
    return users_df, courses_df, interactions_df