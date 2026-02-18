#!/usr/bin/env python3
"""
Plot Results Alternative - Signal Optimizer
============================================

Alternative visualization using matplotlib (without kaleido dependency).
Generates optimization history and score distribution plots.

Author: ML Trading Strategy Team
License: MIT
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class OptimizationPlotter:
    """
    Creates visualization plots for optimization results.
    """
    
    def __init__(self, trials_csv: str, output_dir: Optional[str] = None):
        """
        Initialize plotter.
        
        Args:
            trials_csv: Path to optimization trials CSV file
            output_dir: Output directory for plots (optional)
        """
        self.trials_csv = trials_csv
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(trials_csv).parent
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load trials data
        self.trials_df = self._load_trials()
    
    def _load_trials(self) -> pd.DataFrame:
        """Load trials data from CSV."""
        if not os.path.exists(self.trials_csv):
            raise FileNotFoundError(f"Trials file not found: {self.trials_csv}")
        
        df = pd.read_csv(self.trials_csv)
        print(f"✅ Loaded {len(df)} trials from: {self.trials_csv}")
        
        return df
    
    def plot_optimization_history(self, save: bool = True) -> None:
        """
        Plot optimization history showing score progression.
        
        Args:
            save: Whether to save the plot to file
        """
        print("📊 Generating optimization history plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot all trials
        ax.plot(self.trials_df['number'], self.trials_df['value'], 
               'o-', alpha=0.5, linewidth=1, markersize=4, 
               label='Trial Score', color='#3498db')
        
        # Plot best score line
        best_scores = self.trials_df['value'].cummax()
        ax.plot(self.trials_df['number'], best_scores,
               'r-', linewidth=2, label='Best Score', color='#e74c3c')
        
        # Styling
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
        ax.set_title('Optimization History - Score Progression', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        best_score = self.trials_df['value'].max()
        mean_score = self.trials_df['value'].mean()
        std_score = self.trials_df['value'].std()
        
        stats_text = (
            f"Best Score: {best_score:.2f}\n"
            f"Mean Score: {mean_score:.2f}\n"
            f"Std Dev: {std_score:.2f}"
        )
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "optimization_history_matplotlib.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_score_distribution(self, save: bool = True) -> None:
        """
        Plot distribution of scores across all trials.
        
        Args:
            save: Whether to save the plot to file
        """
        print("📊 Generating score distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        scores = self.trials_df['value'].dropna()
        
        # Histogram
        ax1.hist(scores, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(scores.mean(), color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'Mean: {scores.mean():.2f}')
        ax1.axvline(scores.median(), color='#2ecc71', linestyle='--', 
                   linewidth=2, label=f'Median: {scores.median():.2f}')
        
        ax1.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Score Distribution - Histogram', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        bp = ax2.boxplot([scores], vert=True, patch_artist=True,
                        labels=['Composite Score'],
                        boxprops=dict(facecolor='#3498db', alpha=0.7),
                        medianprops=dict(color='#e74c3c', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax2.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
        ax2.set_title('Score Distribution - Box Plot', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        q1, q3 = scores.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        stats_text = (
            f"Min: {scores.min():.2f}\n"
            f"Q1: {q1:.2f}\n"
            f"Median: {scores.median():.2f}\n"
            f"Q3: {q3:.2f}\n"
            f"Max: {scores.max():.2f}\n"
            f"IQR: {iqr:.2f}"
        )
        
        ax2.text(1.15, 0.5, stats_text,
                transform=ax2.transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "score_distribution_matplotlib.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_parameter_correlations(self, save: bool = True) -> None:
        """
        Plot correlations between parameters and score.
        
        Args:
            save: Whether to save the plot to file
        """
        print("📊 Generating parameter correlation plot...")
        
        # Get parameter columns
        param_cols = [col for col in self.trials_df.columns 
                     if col.startswith('params_')]
        
        if len(param_cols) == 0:
            print("   ⚠️  No parameter columns found")
            return
        
        # Calculate correlations
        correlations = {}
        for col in param_cols:
            if self.trials_df[col].dtype in ['float64', 'int64']:
                corr = self.trials_df[col].corr(self.trials_df['value'])
                if not np.isnan(corr):
                    param_name = col.replace('params_', '')
                    correlations[param_name] = corr
        
        if len(correlations) == 0:
            print("   ⚠️  No valid correlations found")
            return
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), 
                                 key=lambda x: abs(x[1]), reverse=True))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_corr) * 0.4)))
        
        params = list(sorted_corr.keys())
        corrs = list(sorted_corr.values())
        colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in corrs]
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, corrs, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Correlation with Score', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Importance - Correlation with Score', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / "parameter_correlations_matplotlib.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_plots(self) -> None:
        """Generate all available plots."""
        print("\n" + "="*70)
        print("📊 Generating Visualization Plots")
        print("="*70 + "\n")
        
        self.plot_optimization_history()
        self.plot_score_distribution()
        self.plot_parameter_correlations()
        
        print("\n✅ All plots generated successfully!")
        print(f"📁 Saved to: {self.output_dir}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate alternative plots for optimization results'
    )
    parser.add_argument(
        'trials_csv',
        help='Path to optimization trials CSV file'
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='Output directory for plots (optional)',
        default=None
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots instead of saving'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 Optimization Results Plotter (Matplotlib)")
    print("="*70 + "\n")
    
    try:
        plotter = OptimizationPlotter(args.trials_csv, args.output_dir)
        
        if args.show:
            plotter.plot_optimization_history(save=False)
            plotter.plot_score_distribution(save=False)
            plotter.plot_parameter_correlations(save=False)
        else:
            plotter.generate_all_plots()
        
        print("✅ Plotting completed successfully")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
