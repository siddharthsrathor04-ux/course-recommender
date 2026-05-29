"""
Hybrid Recommendation System Package

Modules:
- hybrid_recommender: Core hybrid recommendation engine
- cold_start: Cold start problem handling
- evaluation_metrics: Comprehensive evaluation metrics
- visualizations: Visualization and reporting tools
- explainability: Recommendation explainability and interpretability
- main: Main pipeline orchestration
"""

from .hybrid_recommender import HybridRecommender
from .cold_start import ColdStartHandler
from .evaluation_metrics import EvaluationMetrics, RecommendationEvaluator
from .visualizations import RecommendationVisualizer
from .explainability import HybridRecommendationExplainer, ExplainabilityDashboard
from .main import HybridRecommendationPipeline

__version__ = '2.0'
__author__ = 'ML Team'

__all__ = [
    'HybridRecommender',
    'ColdStartHandler',
    'EvaluationMetrics',
    'RecommendationEvaluator',
    'RecommendationVisualizer',
    'HybridRecommendationExplainer',
    'ExplainabilityDashboard',
    'HybridRecommendationPipeline',
]
