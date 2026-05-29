import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    """
    Hybrid recommendation system combining content-based and collaborative filtering approaches.
    
    Implements a weighted hybrid recommendation strategy where:
    - Content-based recommendations capture item similarities
    - Collaborative filtering captures user-item interaction patterns
    - Scores are normalized and combined using configurable weights
    
    Attributes:
        content_model: Content-based recommendation engine
        collab_model: Collaborative filtering recommendation engine
        alpha (float): Weight for collaborative filtering score (default: 0.7)
        beta (float): Weight for content-based score (default: 0.3)
        scaler: MinMaxScaler for score normalization
    """
    
    def __init__(self, content_model, collab_model, alpha: float = 0.7, beta: float = 0.3):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_model: Fitted ContentBasedRecommender instance
            collab_model: Fitted CollaborativeFilteringRecommender instance
            alpha (float): Weight for collaborative filtering score (0-1)
            beta (float): Weight for content-based score (0-1)
            
        Raises:
            ValueError: If alpha + beta != 1.0
        """
        if not (0.99 <= alpha + beta <= 1.01):  # Allow for floating-point precision
            raise ValueError(f"Weights must sum to 1.0, got alpha={alpha} + beta={beta} = {alpha + beta}")
        
        self.content_model = content_model
        self.collab_model = collab_model
        self.alpha = alpha
        self.beta = beta
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.recommendation_cache = {}
        
    def set_weights(self, alpha: float, beta: float):
        """
        Update recommendation weights dynamically.
        
        Args:
            alpha (float): New weight for collaborative filtering
            beta (float): New weight for content-based
            
        Raises:
            ValueError: If weights do not sum to 1.0
        """
        if not (0.99 <= alpha + beta <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got alpha={alpha} + beta={beta} = {alpha + beta}")
        
        self.alpha = alpha
        self.beta = beta
        self.recommendation_cache.clear()
        
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using Min-Max scaling.
        
        Args:
            scores: Array of scores to normalize
            
        Returns:
            Normalized scores in [0, 1] range
        """
        if len(scores) == 0:
            return np.array([])
        
        if np.std(scores) == 0:  # All scores are identical
            return np.ones_like(scores) * 0.5
        
        return self.scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    
    def get_content_scores(self, user_id: str, interactions_df: pd.DataFrame, 
                          courses_df: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
        """
        Get content-based recommendation scores for a user.
        
        Args:
            user_id: User identifier
            interactions_df: User-course interaction dataframe
            courses_df: Course metadata dataframe
            top_n: Number of recommendations to generate
            
        Returns:
            Dictionary mapping course_id to content-based score
        """
        try:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            content_recs = self.content_model.get_user_recommendations(
                user_interactions, top_n=top_n
            )
            
            if content_recs.empty:
                return {}
            
            # Use weighted_score if available, otherwise use similarity score
            score_col = 'weighted_score' if 'weighted_score' in content_recs.columns else 'similarity_score'
            return dict(zip(content_recs['course_id'], content_recs[score_col]))
        
        except Exception as e:
            print(f"Error in content-based scoring: {str(e)}")
            return {}
    
    def get_collaborative_scores(self, user_id: str, courses_df: pd.DataFrame, 
                                 top_n: int = 10) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get collaborative filtering recommendation scores from both user-based and item-based approaches.
        
        Args:
            user_id: User identifier
            courses_df: Course metadata dataframe
            top_n: Number of recommendations to generate
            
        Returns:
            Tuple of (user_based_scores, item_based_scores) dictionaries
        """
        user_based_scores = {}
        item_based_scores = {}
        
        try:
            # User-based collaborative filtering
            user_recs = self.collab_model.get_user_based_recommendations(user_id, courses_df, top_n=top_n)
            if not user_recs.empty:
                user_based_scores = dict(zip(user_recs['course_id'], user_recs['predicted_rating']))
            
            # Item-based collaborative filtering
            item_recs = self.collab_model.get_item_based_recommendations(user_id, courses_df, top_n=top_n)
            if not item_recs.empty:
                item_based_scores = dict(zip(item_recs['course_id'], item_recs['predicted_rating']))
        
        except Exception as e:
            print(f"Error in collaborative filtering scoring: {str(e)}")
        
        return user_based_scores, item_based_scores
    
    def compute_hybrid_score(self, user_id: str, interactions_df: pd.DataFrame,
                            courses_df: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
        """
        Compute hybrid recommendation scores combining all approaches.
        
        Algorithm:
        1. Get content-based scores
        2. Get collaborative filtering scores (average of user-based and item-based)
        3. Normalize both score distributions
        4. Combine using: hybrid_score = alpha * normalized_cf + beta * normalized_content
        
        Args:
            user_id: User identifier
            interactions_df: User-course interaction dataframe
            courses_df: Course metadata dataframe
            top_n: Number of recommendations to generate
            
        Returns:
            Dictionary mapping course_id to hybrid score
        """
        cache_key = f"{user_id}_{top_n}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        # Get scores from all approaches
        content_scores = self.get_content_scores(user_id, interactions_df, courses_df, top_n)
        user_cf_scores, item_cf_scores = self.get_collaborative_scores(user_id, courses_df, top_n)
        
        # Merge collaborative filtering scores (average of user-based and item-based)
        all_courses = set(content_scores.keys()) | set(user_cf_scores.keys()) | set(item_cf_scores.keys())
        
        if not all_courses:
            return {}
        
        # Compute average CF score for each course
        cf_scores = {}
        for course_id in all_courses:
            scores = []
            if course_id in user_cf_scores:
                scores.append(user_cf_scores[course_id])
            if course_id in item_cf_scores:
                scores.append(item_cf_scores[course_id])
            
            cf_scores[course_id] = np.mean(scores) if scores else 0.0
        
        # Normalize scores to [0, 1]
        content_array = np.array([content_scores.get(c, 0.0) for c in all_courses])
        cf_array = np.array([cf_scores.get(c, 0.0) for c in all_courses])
        
        norm_content = self._normalize_scores(content_array)
        norm_cf = self._normalize_scores(cf_array)
        
        # Compute hybrid scores
        hybrid_scores = {}
        for idx, course_id in enumerate(all_courses):
            hybrid_score = (self.alpha * norm_cf[idx] + self.beta * norm_content[idx])
            hybrid_scores[course_id] = hybrid_score
        
        self.recommendation_cache[cache_key] = hybrid_scores
        return hybrid_scores
    
    def get_top_recommendations(self, user_id: str, interactions_df: pd.DataFrame,
                               courses_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Get top-N hybrid recommendations for a user.
        
        Args:
            user_id: User identifier
            interactions_df: User-course interaction dataframe
            courses_df: Course metadata dataframe
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with columns: course_id, course_name, difficulty_level, hybrid_score, 
                                   content_score, cf_score, explanation
        """
        # Get user's already-rated courses
        user_courses = interactions_df[interactions_df['user_id'] == user_id]['course_id'].values
        
        # Compute hybrid scores
        hybrid_scores = self.compute_hybrid_score(user_id, interactions_df, courses_df, top_n * 2)
        
        if not hybrid_scores:
            return pd.DataFrame()
        
        # Filter out already-rated courses
        filtered_scores = {cid: score for cid, score in hybrid_scores.items() 
                          if cid not in user_courses}
        
        if not filtered_scores:
            return pd.DataFrame()
        
        # Sort and get top-N
        sorted_courses = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        course_ids = [c[0] for c in sorted_courses]
        hybrid_vals = [c[1] for c in sorted_courses]
        
        # Get detailed course information
        course_info = courses_df[courses_df['course_id'].isin(course_ids)].copy()
        
        # Get component scores for explainability
        content_scores_dict = self.get_content_scores(user_id, interactions_df, courses_df, len(course_ids) * 2)
        user_cf_scores, item_cf_scores = self.get_collaborative_scores(user_id, courses_df, len(course_ids) * 2)
        
        # Reorder to match sorted list
        results = []
        for course_id, hybrid_score in sorted_courses:
            course_row = course_info[course_info['course_id'] == course_id]
            if not course_row.empty:
                # Compute individual component scores
                content_score = content_scores_dict.get(course_id, 0.0)
                cf_score = np.mean([user_cf_scores.get(course_id, 0.0), item_cf_scores.get(course_id, 0.0)])
                
                # Determine which component contributed most
                if cf_score > content_score:
                    explanation = "Recommended based on user-user similarity"
                else:
                    explanation = "Recommended based on course content similarity"
                
                results.append({
                    'course_id': course_id,
                    'course_name': course_row.iloc[0]['course_name'],
                    'difficulty_level': course_row.iloc[0]['difficulty_level'],
                    'hybrid_score': hybrid_score,
                    'content_score': content_score,
                    'cf_score': cf_score,
                    'explanation': explanation
                })
        
        return pd.DataFrame(results)
    
    def get_recommendations_with_details(self, user_id: str, interactions_df: pd.DataFrame,
                                        courses_df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Get detailed hybrid recommendations including component breakdown.
        
        Args:
            user_id: User identifier
            interactions_df: User-course interaction dataframe
            courses_df: Course metadata dataframe
            top_n: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations dataframe and metadata
        """
        recommendations = self.get_top_recommendations(user_id, interactions_df, courses_df, top_n)
        
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'weights': {
                'collaborative_filtering_alpha': self.alpha,
                'content_based_beta': self.beta
            },
            'recommendation_count': len(recommendations),
            'avg_score': recommendations['hybrid_score'].mean() if len(recommendations) > 0 else 0.0
        }
