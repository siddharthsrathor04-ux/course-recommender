import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class RecommendationVisualizer:
    """
    Visualization generator for recommendation system evaluation and analysis.
    
    Generates publication-ready visualizations for:
    - Precision@K, Recall@K, NDCG@K metrics
    - System comparison charts
    - User distribution analysis
    - Recommendation quality heatmaps
    """
    
    def __init__(self, output_dir: str = 'outputs', figsize: Tuple[int, int] = (12, 6),
                 style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size (width, height)
            style: Matplotlib style
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_metrics_at_k(self, metrics_dict: Dict[str, float], k_values: List[int] = None,
                         system_name: str = "Recommendation System") -> str:
        """
        Plot Precision@K, Recall@K, NDCG@K across different K values.
        
        Args:
            metrics_dict: Dictionary with metrics in format {metric_name@k: value}
            k_values: K values to plot (extracted from metrics_dict if None)
            system_name: Name of the system
            
        Returns:
            Path to saved figure
        """
        # Extract metrics by K value
        if k_values is None:
            k_values = sorted(set(int(v.split('@')[1]) for v in metrics_dict.keys() if '@' in v))
        
        metric_types = {}
        for metric_name, value in metrics_dict.items():
            if '@' in metric_name:
                base_metric = metric_name.split('@')[0]
                k_val = int(metric_name.split('@')[1])
                if base_metric not in metric_types:
                    metric_types[base_metric] = {}
                metric_types[base_metric][k_val] = value
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)
        fig.suptitle(f'Metrics vs K Value - {system_name}', fontsize=14, fontweight='bold')
        
        metrics_to_plot = ['precision_at_k', 'recall_at_k', 'ndcg_at_k']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (metric, ax, color) in enumerate(zip(metrics_to_plot, axes, colors)):
            if metric in metric_types:
                k_vals = sorted(metric_types[metric].keys())
                values = [metric_types[metric][k] for k in k_vals]
                
                ax.plot(k_vals, values, marker='o', linewidth=2.5, markersize=8, color=color)
                ax.fill_between(k_vals, values, alpha=0.3, color=color)
                
                ax.set_xlabel('K', fontsize=11, fontweight='bold')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                
                # Add value labels
                for k, v in zip(k_vals, values):
                    ax.annotate(f'{v:.3f}', xy=(k, v), xytext=(0, 5),
                              textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / 'metrics_at_k.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_precision_recall_curve(self, metrics_per_k: Dict[int, Tuple[float, float]],
                                   system_name: str = "Recommendation System") -> str:
        """
        Plot Precision-Recall curve.
        
        Args:
            metrics_per_k: Dict mapping K to (precision, recall) tuple
            system_name: Name of the system
            
        Returns:
            Path to saved figure
        """
        k_values = sorted(metrics_per_k.keys())
        precisions = [metrics_per_k[k][0] for k in k_values]
        recalls = [metrics_per_k[k][1] for k in k_values]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.plot(recalls, precisions, marker='o', linewidth=2.5, markersize=8,
               color='#2E86AB', label=system_name)
        ax.fill_between(recalls, precisions, alpha=0.3, color='#2E86AB')
        
        # Add K annotations
        for k, rec, prec in zip(k_values, recalls, precisions):
            ax.annotate(f'K={k}', xy=(rec, prec), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve - {system_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        filepath = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_system_comparison(self, results_df: pd.DataFrame, metric: str = 'ndcg_at_k',
                              k: int = 5) -> str:
        """
        Compare multiple recommendation systems.
        
        Args:
            results_df: DataFrame with systems as rows and metrics as columns
            metric: Base metric to plot (will look for {metric}@{k})
            k: K value to compare
            
        Returns:
            Path to saved figure
        """
        metric_col = f'{metric}@{k}'
        
        if metric_col not in results_df.columns:
            raise ValueError(f"Metric {metric_col} not found in results")
        
        # Sort by metric value
        results_sorted = results_df.sort_values(metric_col, ascending=True)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        bars = ax.barh(results_sorted['system_name'], results_sorted[metric_col],
                       color=sns.color_palette("husl", len(results_sorted)))
        
        ax.set_xlabel(f'{metric.replace("_", " ").title()}@{k}', fontsize=12, fontweight='bold')
        ax.set_title(f'System Comparison - {metric.replace("_", " ").title()}@{k}',
                    fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(results_sorted.iterrows()):
            value = row[metric_col]
            ax.text(value + 0.02, i, f'{value:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        filepath = self.output_dir / f'system_comparison_{metric}@{k}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_metric_heatmap(self, results_df: pd.DataFrame, metric_prefix: str = None) -> str:
        """
        Plot heatmap of metrics across systems and K values.
        
        Args:
            results_df: DataFrame with systems and metrics
            metric_prefix: Prefix to filter metrics (e.g., 'ndcg_at_k')
            
        Returns:
            Path to saved figure
        """
        # Extract metric columns
        metric_cols = [col for col in results_df.columns if col != 'system_name']
        
        if metric_prefix:
            metric_cols = [col for col in metric_cols if col.startswith(metric_prefix)]
        
        # Create matrix for heatmap
        heatmap_data = results_df.set_index('system_name')[metric_cols]
        
        fig, ax = plt.subplots(figsize=(10, len(results_df) * 0.8 + 1), dpi=self.dpi)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu',
                   cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
        
        ax.set_title('Metrics Heatmap Across Systems', fontsize=13, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('System', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'metrics_heatmap.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_per_user_distribution(self, metrics_per_user_df: pd.DataFrame,
                                  metric: str = 'ndcg_at_k') -> str:
        """
        Plot distribution of metric scores across users.
        
        Args:
            metrics_per_user_df: DataFrame with per-user metrics
            metric: Metric to plot distribution for
            
        Returns:
            Path to saved figure
        """
        if metric not in metrics_per_user_df.columns:
            raise ValueError(f"Metric {metric} not found in dataframe")
        
        scores = metrics_per_user_df[metric].values
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        fig.suptitle(f'Distribution of {metric.replace("_", " ").title()} Across Users',
                    fontsize=13, fontweight='bold')
        
        # Histogram
        ax = axes[0]
        ax.hist(scores, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.4f}')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax = axes[1]
        bp = ax.boxplot(scores, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2E86AB')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Mean: {scores.mean():.4f}\nStd: {scores.std():.4f}\nMin: {scores.min():.4f}\nMax: {scores.max():.4f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filepath = self.output_dir / f'distribution_{metric}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_coverage_analysis(self, coverage_data: Dict[str, float],
                              system_names: List[str] = None) -> str:
        """
        Plot catalog coverage for different systems.
        
        Args:
            coverage_data: Dict mapping system name to coverage score
            system_names: Optional specific systems to plot
            
        Returns:
            Path to saved figure
        """
        if system_names:
            coverage_data = {k: v for k, v in coverage_data.items() if k in system_names}
        
        systems = sorted(coverage_data.items(), key=lambda x: x[1], reverse=True)
        sys_names = [s[0] for s in systems]
        coverage_scores = [s[1] for s in systems]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        bars = ax.bar(sys_names, coverage_scores, color=sns.color_palette("husl", len(systems)))
        
        ax.set_ylabel('Coverage Score', fontsize=12, fontweight='bold')
        ax.set_title('Catalog Coverage by System', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, coverage_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filepath = self.output_dir / 'coverage_analysis.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_recommendation_distribution(self, recommendations_dict: Dict[str, List[str]],
                                       courses_df: pd.DataFrame) -> str:
        """
        Plot distribution of recommended courses (popularity).
        
        Args:
            recommendations_dict: Dict mapping user to recommended courses
            courses_df: Course metadata
            
        Returns:
            Path to saved figure
        """
        # Count course recommendations
        course_counts = {}
        for courses in recommendations_dict.values():
            for course in courses:
                course_counts[course] = course_counts.get(course, 0) + 1
        
        # Sort by count
        sorted_courses = sorted(course_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        course_ids = [c[0] for c in sorted_courses]
        counts = [c[1] for c in sorted_courses]
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        bars = ax.barh(course_ids, counts, color='#2E86AB')
        ax.set_xlabel('Number of Recommendations', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Recommended Courses', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / 'recommendation_distribution.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def generate_summary_report(self, metrics_dict: Dict[str, float],
                               system_name: str = "Recommendation System") -> str:
        """
        Generate a summary figure with all key metrics.
        
        Args:
            metrics_dict: Dictionary with evaluation metrics
            system_name: Name of the system
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(14, 8), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Recommendation System Summary - {system_name}',
                    fontsize=14, fontweight='bold')
        
        # Extract available metrics
        metrics_list = sorted([(k, v) for k, v in metrics_dict.items() if isinstance(v, (int, float))])
        
        # Create text summary
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')
        
        summary_text = f"System: {system_name}\n\n"
        summary_text += "Key Metrics:\n"
        for metric_name, value in metrics_list[:6]:
            summary_text += f"  • {metric_name}: {value:.4f}\n"
        
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot metrics as gauges (simplified bars)
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        
        axes = [ax1, ax2, ax3, ax4]
        metric_pairs = [('precision_at_k', 'recall_at_k'),
                       ('ndcg_at_k', 'map'),
                       ('coverage', None),
                       (None, None)]
        
        for ax, (metric1, metric2) in zip(axes, metric_pairs):
            if metric1 and metric1 in dict(metrics_list):
                val1 = dict(metrics_list)[metric1]
                ax.barh([metric1], [val1], color='#2E86AB')
                ax.set_xlim([0, 1])
                ax.text(val1 + 0.02, 0, f'{val1:.4f}', va='center')
            
            if metric2 and metric2 in dict(metrics_list):
                val2 = dict(metrics_list)[metric2]
                if metric1:
                    ax.barh([metric2], [val2], color='#A23B72')
                else:
                    ax.barh([metric2], [val2], color='#2E86AB')
                ax.set_xlim([0, 1])
                ax.text(val2 + 0.02, 0 if not metric1 else 1, f'{val2:.4f}', va='center')
            
            ax.set_xlim([0, 1.2])
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = self.output_dir / 'summary_report.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
