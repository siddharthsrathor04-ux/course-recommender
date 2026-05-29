import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ColdStartHandler:
    """
    Handles cold start problems in recommendation systems.
    
    Cold Start Scenarios:
    1. New User: No interaction history available
    2. New Course: No ratings available
    
    Solutions:
    - New User: Recommend popular/highly-rated courses and similarity-based courses
    - New Course: Use content similarity and metadata features
    
    Attributes:
        courses_df: Course metadata dataframe
        interactions_df: User-course interactions dataframe
        min_interactions: Threshold for classifying user as cold start
    """
    
    def __init__(self, courses_df: pd.DataFrame, interactions_df: pd.DataFrame,
                 min_interactions: int = 2):
        """
        Initialize the cold start handler.
        
        Args:
            courses_df: Course metadata dataframe
            interactions_df: User-course interactions dataframe
            min_interactions: Minimum interactions to not be considered cold start
        """
        self.courses_df = courses_df.copy()
        self.interactions_df = interactions_df.copy()
        self.min_interactions = min_interactions
        
        # Precompute course statistics
        self.course_stats = self._compute_course_stats()
        
        # Precompute TF-IDF matrix for content similarity
        self.tfidf_matrix = self._compute_tfidf_matrix()
    
    def _compute_course_stats(self) -> pd.DataFrame:
        """
        Compute popularity and rating statistics for all courses.
        
        Returns:
            DataFrame with course statistics
        """
        stats = self.interactions_df.groupby('course_id').agg({
            'rating': ['mean', 'std', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        
        stats.columns = ['course_id', 'avg_rating', 'rating_std', 'num_ratings', 'num_users']
        stats['popularity_score'] = (stats['num_ratings'] * 0.6 + stats['avg_rating'] * 0.4)
        stats['rating_std'] = stats['rating_std'].fillna(0)
        
        return stats
    
    def _compute_tfidf_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute TF-IDF representation of courses for content similarity.
        
        Uses course names, descriptions, and skills as features.
        
        Returns:
            Tuple of (TF-IDF matrix, course IDs)
        """
        # Combine text features
        text_features = []
        for _, row in self.courses_df.iterrows():
            combined_text = f"{str(row.get('course_name', ''))} {str(row.get('course_description', ''))} {str(row.get('skills', ''))}"
            text_features.append(combined_text.lower())
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(text_features)
        except:
            tfidf_matrix = None
        
        course_ids = self.courses_df['course_id'].tolist()
        return (tfidf_matrix, course_ids)
    
    def is_cold_start_user(self, user_id: str) -> bool:
        """
        Check if user is in cold start scenario.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user has fewer interactions than threshold
        """
        user_interactions = len(self.interactions_df[self.interactions_df['user_id'] == user_id])
        return user_interactions < self.min_interactions
    
    def is_cold_start_course(self, course_id: str) -> bool:
        """
        Check if course is in cold start scenario.
        
        Args:
            course_id: Course identifier
            
        Returns:
            True if course has fewer ratings than threshold
        """
        course_ratings = len(self.interactions_df[self.interactions_df['course_id'] == course_id])
        return course_ratings < self.min_interactions
    
    def recommend_for_new_user(self, user_id: str, top_n: int = 5,
                              strategy: str = 'hybrid') -> pd.DataFrame:
        """
        Generate recommendations for a new user with no interaction history.
        
        Strategies:
        - 'popularity': Top-rated and popular courses
        - 'diverse': Mix of popular and diverse difficulty levels
        - 'beginner': Courses suitable for beginners
        - 'hybrid': Weighted combination of popularity and diversity
        
        Args:
            user_id: New user identifier
            top_n: Number of recommendations
            strategy: Recommendation strategy to use
            
        Returns:
            DataFrame with recommended courses
        """
        if strategy == 'popularity':
            return self._popular_courses(top_n)
        elif strategy == 'diverse':
            return self._diverse_courses(top_n)
        elif strategy == 'beginner':
            return self._beginner_courses(top_n)
        elif strategy == 'hybrid':
            return self._hybrid_new_user_recommendations(top_n)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _popular_courses(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get top-N most popular and highly-rated courses.
        
        Args:
            top_n: Number of courses to recommend
            
        Returns:
            DataFrame with popular courses
        """
        popular = self.course_stats.nlargest(top_n, 'popularity_score')
        
        result = popular.merge(
            self.courses_df[['course_id', 'course_name', 'difficulty_level', 'skills']],
            on='course_id'
        )
        
        return result[[
            'course_id', 'course_name', 'difficulty_level', 'avg_rating',
            'num_ratings', 'popularity_score'
        ]].rename(columns={'popularity_score': 'score'})
    
    def _diverse_courses(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get courses with diverse difficulty levels, prioritizing quality.
        
        Args:
            top_n: Number of courses to recommend
            
        Returns:
            DataFrame with diverse courses
        """
        diverse_courses = []
        
        for difficulty in ['Beginner', 'Intermediate', 'Advanced']:
            difficulty_courses = self.course_stats[
                self.courses_df.loc[self.course_stats['course_id'].index, 'difficulty_level'] == difficulty
            ] if 'difficulty_level' in self.courses_df.columns else self.course_stats
            
            top_course = difficulty_courses.nlargest(1, 'avg_rating')
            if not top_course.empty:
                diverse_courses.append(top_course)
        
        # Fill remaining slots with top-rated courses
        while len(diverse_courses) < top_n:
            used_courses = set()
            for df in diverse_courses:
                used_courses.update(df['course_id'].values)
            
            remaining = self.course_stats[~self.course_stats['course_id'].isin(used_courses)]
            if remaining.empty:
                break
            
            next_course = remaining.nlargest(1, 'avg_rating')
            diverse_courses.append(next_course)
        
        result = pd.concat(diverse_courses, ignore_index=True)
        result = result.merge(
            self.courses_df[['course_id', 'course_name', 'difficulty_level']],
            on='course_id'
        )
        
        return result[[
            'course_id', 'course_name', 'difficulty_level', 'avg_rating', 'num_ratings'
        ]].rename(columns={'avg_rating': 'score'}).head(top_n)
    
    def _beginner_courses(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get top-rated beginner-friendly courses.
        
        Args:
            top_n: Number of courses to recommend
            
        Returns:
            DataFrame with beginner courses
        """
        beginner_courses = self.courses_df[
            self.courses_df['difficulty_level'] == 'Beginner'
        ]['course_id'].values
        
        beginner_stats = self.course_stats[
            self.course_stats['course_id'].isin(beginner_courses)
        ].nlargest(top_n, 'avg_rating')
        
        result = beginner_stats.merge(
            self.courses_df[['course_id', 'course_name', 'difficulty_level', 'skills']],
            on='course_id'
        )
        
        return result[[
            'course_id', 'course_name', 'difficulty_level', 'avg_rating'
        ]].rename(columns={'avg_rating': 'score'})
    
    def _hybrid_new_user_recommendations(self, top_n: int = 5) -> pd.DataFrame:
        """
        Hybrid strategy: Combine popularity, quality, and diversity.
        
        Scoring: score = 0.5 * normalized_popularity + 0.3 * normalized_rating + 0.2 * diversity_bonus
        
        Args:
            top_n: Number of courses to recommend
            
        Returns:
            DataFrame with recommended courses
        """
        scores = self.course_stats.copy()
        
        # Normalize metrics to [0, 1]
        scores['norm_popularity'] = (scores['popularity_score'] - scores['popularity_score'].min()) / \
                                   (scores['popularity_score'].max() - scores['popularity_score'].min() + 1e-6)
        scores['norm_rating'] = (scores['avg_rating'] - scores['avg_rating'].min()) / \
                               (scores['avg_rating'].max() - scores['avg_rating'].min() + 1e-6)
        
        # Compute combined score
        scores['combined_score'] = (
            0.5 * scores['norm_popularity'] +
            0.3 * scores['norm_rating'] +
            0.2 * np.random.random(len(scores))  # Diversity boost
        )
        
        top_courses = scores.nlargest(top_n, 'combined_score')
        
        result = top_courses.merge(
            self.courses_df[['course_id', 'course_name', 'difficulty_level']],
            on='course_id'
        )
        
        return result[[
            'course_id', 'course_name', 'difficulty_level', 'combined_score'
        ]].rename(columns={'combined_score': 'score'})
    
    def recommend_similar_new_course(self, course_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Get courses similar to a new course using content-based similarity.
        
        Uses TF-IDF vectorization and cosine similarity to find similar courses.
        
        Args:
            course_id: Reference course ID
            top_n: Number of similar courses to recommend
            
        Returns:
            DataFrame with similar courses
        """
        if self.tfidf_matrix is None or self.tfidf_matrix[0] is None:
            return pd.DataFrame()
        
        tfidf_matrix, course_ids = self.tfidf_matrix
        
        # Find reference course index
        try:
            course_idx = course_ids.index(course_id)
        except ValueError:
            return pd.DataFrame()
        
        # Compute similarity with all other courses
        reference_vector = tfidf_matrix[course_idx]
        similarities = cosine_similarity(reference_vector, tfidf_matrix).flatten()
        
        # Get top-N similar courses (excluding the reference)
        similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
        similar_course_ids = [course_ids[i] for i in similar_indices]
        similar_scores = similarities[similar_indices]
        
        result_df = pd.DataFrame({
            'course_id': similar_course_ids,
            'similarity_score': similar_scores
        })
        
        result_df = result_df.merge(
            self.courses_df[['course_id', 'course_name', 'difficulty_level', 'skills']],
            on='course_id'
        )
        
        return result_df[[
            'course_id', 'course_name', 'difficulty_level', 'similarity_score'
        ]].rename(columns={'similarity_score': 'score'})
    
    def get_personalized_new_user_recommendations(self, user_id: str, 
                                                 user_preferences: Dict = None,
                                 top_n: int = 5) -> pd.DataFrame:
        """
        Get recommendations for new user with optional preference hints.
        
        Preferences can include:
        - preferred_difficulty: 'Beginner', 'Intermediate', 'Advanced'
        - preferred_skills: List of skill names
        - learning_speed: 'Slow', 'Medium', 'Fast'
        
        Args:
            user_id: New user identifier
            user_preferences: Dictionary with user preference hints
            top_n: Number of recommendations
            
        Returns:
            DataFrame with personalized recommendations
        """
        if user_preferences is None:
            return self.recommend_for_new_user(user_id, top_n, strategy='hybrid')
        
        candidates = self.courses_df.copy()
        scores = pd.DataFrame({'course_id': candidates['course_id']})
        
        # Filter by difficulty if specified
        if 'preferred_difficulty' in user_preferences:
            difficulty = user_preferences['preferred_difficulty']
            candidates = candidates[candidates['difficulty_level'] == difficulty]
            if candidates.empty:
                return self.recommend_for_new_user(user_id, top_n, strategy='hybrid')
        
        # Score based on skills if specified
        if 'preferred_skills' in user_preferences:
            preferred_skills = user_preferences['preferred_skills']
            skill_scores = []
            for _, row in candidates.iterrows():
                skills_str = str(row.get('skills', ''))
                match_count = sum(1 for skill in preferred_skills if skill.lower() in skills_str.lower())
                skill_scores.append(match_count / len(preferred_skills) if preferred_skills else 0)
            
            candidates['skill_score'] = skill_scores
        else:
            candidates['skill_score'] = 0.5
        
        # Merge with popularity stats
        candidates = candidates.merge(self.course_stats, on='course_id')
        candidates['final_score'] = (
            0.4 * (candidates['avg_rating'] / 5.0) +  # Rating
            0.4 * (candidates['skill_score']) +        # Skill match
            0.2 * (candidates['popularity_score'] / candidates['popularity_score'].max())  # Popularity
        )
        
        top_courses = candidates.nlargest(top_n, 'final_score')
        
        return top_courses[[
            'course_id', 'course_name', 'difficulty_level', 'final_score'
        ]].rename(columns={'final_score': 'score'})
