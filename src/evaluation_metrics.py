import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for recommendation systems.
    
    Metrics:
    - Precision@K: Proportion of recommended items that are relevant
    - Recall@K: Proportion of relevant items that are recommended
    - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
    - MAP: Mean Average Precision (considers all K values)
    - Coverage: Catalog coverage (diversity of recommendations)
    - Diversity: Intra-list diversity of recommendations
    - Novelty: Average popularity of recommended items
    
    Attributes:
        interactions_df: Ground truth interactions dataframe
        predictions_df: Predicted recommendations dataframe
    """
    
    def __init__(self, interactions_df: pd.DataFrame, predictions_df: pd.DataFrame = None):
        """
        Initialize evaluation metrics calculator.
        
        Args:
            interactions_df: Ground truth user-course interactions
            predictions_df: Optional pre-computed predictions
        """
        self.interactions_df = interactions_df.copy()
        self.predictions_df = predictions_df
        
        # Build test set from interactions
        self.test_set = self._build_test_set()
        
    def _build_test_set(self) -> Dict[str, set]:
        """
        Build ground truth test set from interactions.
        
        Returns:
            Dictionary mapping user_id to set of relevant course_ids
        """
        test_set = defaultdict(set)
        
        for _, row in self.interactions_df.iterrows():
            user_id = row['user_id']
            course_id = row['course_id']
            rating = row['rating']
            
            # Consider courses with rating >= 3.5 as relevant
            if rating >= 3.5:
                test_set[user_id].add(course_id)
        
        return dict(test_set)
    
    @staticmethod
    def precision_at_k(recommendations: List[str], relevant_items: set, k: int = 5) -> float:
        """
        Calculate Precision@K.
        
        Formula: P@K = (# relevant items in top-K) / K
        
        Args:
            recommendations: Ranked list of recommended course IDs
            relevant_items: Set of relevant course IDs
            k: Cutoff for ranking
            
        Returns:
            Precision@K score in [0, 1]
        """
        if k <= 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(recommendations: List[str], relevant_items: set, k: int = 5) -> float:
        """
        Calculate Recall@K.
        
        Formula: R@K = (# relevant items in top-K) / (total # relevant items)
        
        Args:
            recommendations: Ranked list of recommended course IDs
            relevant_items: Set of relevant course IDs
            k: Cutoff for ranking
            
        Returns:
            Recall@K score in [0, 1]
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommendations: List[str], relevant_items: set, k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Rewards relevant items appearing early in the ranked list.
        
        Formula:
        DCG@K = sum(rel_i / log2(i+1)) for i in 1..k
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            recommendations: Ranked list of recommended course IDs
            relevant_items: Set of relevant course IDs
            k: Cutoff for ranking
            
        Returns:
            NDCG@K score in [0, 1]
        """
        if len(relevant_items) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommendations[:k], 1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG with all relevant items at top)
        idcg = 0.0
        for i in range(1, min(k, len(relevant_items)) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def average_precision(recommendations: List[str], relevant_items: set, k: int = 10) -> float:
        """
        Calculate Average Precision@K.
        
        Considers precision at each position where a relevant item appears.
        
        Formula: AP@K = (1/min(k, |relevant|)) * sum(P@i * rel_i) for i in 1..k
        
        Args:
            recommendations: Ranked list of recommended course IDs
            relevant_items: Set of relevant course IDs
            k: Cutoff for ranking
            
        Returns:
            Average Precision score in [0, 1]
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(top_k, 1):
            if item in relevant_items:
                num_hits += 1
                precision_at_i = num_hits / i
                score += precision_at_i
        
        return score / min(k, len(relevant_items))
    
    def map_score(self, recommendations_dict: Dict[str, List[str]], k: int = 10) -> float:
        """
        Calculate Mean Average Precision across all users.
        
        Args:
            recommendations_dict: Dict mapping user_id to ranked list of course_ids
            k: Cutoff for ranking
            
        Returns:
            Mean Average Precision score
        """
        ap_scores = []
        
        for user_id, recommendations in recommendations_dict.items():
            if user_id in self.test_set:
                relevant_items = self.test_set[user_id]
                ap = self.average_precision(recommendations, relevant_items, k)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def coverage(self, recommendations_dict: Dict[str, List[str]]) -> float:
        """
        Calculate Catalog Coverage.
        
        Proportion of all unique courses that are recommended.
        
        Formula: Coverage = (# unique recommended courses) / (total # courses)
        
        Args:
            recommendations_dict: Dict mapping user_id to list of recommended course_ids
            
        Returns:
            Coverage score in [0, 1]
        """
        all_recommended = set()
        for courses in recommendations_dict.values():
            all_recommended.update(courses)
        
        total_courses = len(self.interactions_df['course_id'].unique())
        
        return len(all_recommended) / total_courses if total_courses > 0 else 0.0
    
    @staticmethod
    def diversity_intra_list(recommendations: List[str], 
                            similarity_matrix: np.ndarray,
                            course_ids: List[str]) -> float:
        """
        Calculate intra-list diversity (dissimilarity between recommended items).
        
        Uses pairwise similarity and computes average dissimilarity.
        
        Args:
            recommendations: List of recommended course IDs
            similarity_matrix: Course similarity matrix
            course_ids: Mapping of course index to ID
            
        Returns:
            Average dissimilarity in [0, 1]
        """
        if len(recommendations) < 2:
            return 1.0
        
        # Build index mapping
        id_to_idx = {cid: i for i, cid in enumerate(course_ids)}
        
        dissimilarities = []
        for i, course_i in enumerate(recommendations):
            for course_j in recommendations[i+1:]:
                try:
                    idx_i = id_to_idx[course_i]
                    idx_j = id_to_idx[course_j]
                    similarity = similarity_matrix[idx_i, idx_j]
                    dissimilarity = 1 - similarity
                    dissimilarities.append(dissimilarity)
                except (KeyError, IndexError):
                    continue
        
        return np.mean(dissimilarities) if dissimilarities else 0.5
    
    @staticmethod
    def novelty(recommendations: List[str], popularity: Dict[str, float]) -> float:
        """
        Calculate novelty (average unpopularity) of recommendations.
        
        Less popular items are considered more novel/serendipitous.
        
        Formula: Novelty = (1 / |R|) * sum(-log(popularity[i])) for i in R
        
        Args:
            recommendations: List of recommended course IDs
            popularity: Dict mapping course_id to popularity score [0, 1]
            
        Returns:
            Novelty score (higher is more novel)
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for course_id in recommendations:
            if course_id in popularity:
                pop = max(popularity[course_id], 1e-6)  # Avoid log(0)
                novelty_scores.append(-np.log(pop))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def evaluate_recommendations(self, recommendations_dict: Dict[str, List[str]], 
                                k: int = 5) -> Dict[str, float]:
        """
        Comprehensive evaluation of recommendations.
        
        Args:
            recommendations_dict: Dict mapping user_id to ranked list of course_ids
            k: Cutoff for ranking metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        for user_id, recommendations in recommendations_dict.items():
            if user_id in self.test_set:
                relevant_items = self.test_set[user_id]
                
                # Calculate metrics
                p_at_k = self.precision_at_k(recommendations, relevant_items, k)
                r_at_k = self.recall_at_k(recommendations, relevant_items, k)
                ndcg_at_k = self.ndcg_at_k(recommendations, relevant_items, k)
                
                precision_scores.append(p_at_k)
                recall_scores.append(r_at_k)
                ndcg_scores.append(ndcg_at_k)
        
        metrics = {
            'precision_at_k': np.mean(precision_scores) if precision_scores else 0.0,
            'recall_at_k': np.mean(recall_scores) if recall_scores else 0.0,
            'ndcg_at_k': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'map': self.map_score(recommendations_dict, k),
            'coverage': self.coverage(recommendations_dict),
        }
        
        return metrics
    
    def get_metrics_per_user(self, recommendations_dict: Dict[str, List[str]], 
                            k: int = 5) -> pd.DataFrame:
        """
        Get evaluation metrics broken down per user.
        
        Args:
            recommendations_dict: Dict mapping user_id to ranked list of course_ids
            k: Cutoff for ranking metrics
            
        Returns:
            DataFrame with per-user metrics
        """
        results = []
        
        for user_id, recommendations in recommendations_dict.items():
            if user_id in self.test_set:
                relevant_items = self.test_set[user_id]
                
                results.append({
                    'user_id': user_id,
                    'precision_at_k': self.precision_at_k(recommendations, relevant_items, k),
                    'recall_at_k': self.recall_at_k(recommendations, relevant_items, k),
                    'ndcg_at_k': self.ndcg_at_k(recommendations, relevant_items, k),
                    'map': self.average_precision(recommendations, relevant_items, k),
                    'num_relevant': len(relevant_items),
                    'num_recommendations': len(recommendations[:k])
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], name: str = "Recommendations"):
        """
        Pretty-print evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            name: Name of the evaluated system
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {name}")
        print(f"{'='*60}")
        
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"{metric_name:.<40} {value:.4f}")
        
        print(f"{'='*60}\n")


class RecommendationEvaluator:
    """
    High-level evaluator that compares multiple recommendation approaches.
    """
    
    def __init__(self, interactions_df: pd.DataFrame, courses_df: pd.DataFrame):
        """
        Initialize evaluator.
        
        Args:
            interactions_df: Ground truth interactions
            courses_df: Course metadata
        """
        self.interactions_df = interactions_df
        self.courses_df = courses_df
        self.evaluator = EvaluationMetrics(interactions_df)
    
    def evaluate_system(self, recommendations_dict: Dict[str, List[str]], 
                       system_name: str, k_values: List[int] = None) -> Dict:
        """
        Evaluate a recommendation system at multiple K values.
        
        Args:
            recommendations_dict: Dict mapping user_id to ranked list of course_ids
            system_name: Name of the system being evaluated
            k_values: List of K values to evaluate (default: [1, 5, 10])
            
        Returns:
            Dictionary with results for each K value
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        results = {'system_name': system_name}
        
        for k in k_values:
            metrics = self.evaluator.evaluate_recommendations(recommendations_dict, k)
            for metric_name, value in metrics.items():
                results[f'{metric_name}@{k}'] = value
        
        return results
    
    def compare_systems(self, systems: Dict[str, Dict[str, List[str]]], 
                       k_values: List[int] = None) -> pd.DataFrame:
        """
        Compare multiple recommendation systems.
        
        Args:
            systems: Dict mapping system_name to recommendations_dict
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame comparing all systems
        """
        results = []
        
        for system_name, recommendations in systems.items():
            result = self.evaluate_system(recommendations, system_name, k_values)
            results.append(result)
        
        return pd.DataFrame(results)
