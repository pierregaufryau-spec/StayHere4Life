#!/usr/bin/env python3
"""
Verify Overfitting - Signal Optimizer
======================================

Validates optimized parameters on real market data to detect overfitting.
Compares performance between synthetic/training data and real test data.

Author: ML Trading Strategy Team
License: MIT
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Try to import from signal_optimizer
try:
    from signal_optimizer import (
        SignalGeneratorWrapper,
        LightGBMEvaluator,
        SignalQualityMetrics,
        DataLoader
    )
except ImportError:
    print("⚠️  Could not import from signal_optimizer.py")
    print("   Make sure signal_optimizer.py is in the same directory")
    sys.exit(1)


class OverfittingVerifier:
    """
    Verifies if optimized parameters are overfitted to training data.
    """
    
    def __init__(self, params_file: str, test_data_path: Optional[str] = None):
        """
        Initialize verifier.
        
        Args:
            params_file: Path to optimized parameters JSON file
            test_data_path: Path to test data CSV (optional)
        """
        self.params_file = params_file
        self.test_data_path = test_data_path
        
        # Load optimized parameters
        self.optimized_data = self._load_parameters()
        self.params = self.optimized_data['best_parameters']
        self.train_metrics = self.optimized_data['best_metrics']
        
    def _load_parameters(self) -> Dict[str, Any]:
        """Load optimized parameters from JSON."""
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"Parameters file not found: {self.params_file}")
        
        with open(self.params_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded parameters from: {self.params_file}")
        return data
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data for validation.
        
        Returns:
            Test DataFrame with OHLCV data
        """
        if not self.test_data_path:
            print("⚠️  No test data provided, using synthetic data")
            return self._generate_test_data()
        
        if not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        df = pd.read_csv(self.test_data_path)
        print(f"✅ Loaded {len(df)} test samples from: {self.test_data_path}")
        
        # Validate columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.columns = [col.capitalize() for col in df.columns]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df.dropna()
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data (different from training)."""
        n_samples = 2000
        np.random.seed(123)  # Different seed than training
        
        # Different market conditions
        trend = np.linspace(120, 180, n_samples)
        noise = np.cumsum(np.random.randn(n_samples) * 3)
        close = trend + noise
        
        high = close * (1 + np.abs(np.random.randn(n_samples) * 0.025))
        low = close * (1 - np.abs(np.random.randn(n_samples) * 0.025))
        open_price = close + np.random.randn(n_samples) * 1.5
        volume = np.random.lognormal(11, 1.2, n_samples)
        
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        
        print(f"✅ Generated {len(df)} synthetic test samples")
        return df
    
    def verify(self) -> Dict[str, Any]:
        """
        Verify parameters on test data.
        
        Returns:
            Dictionary with verification results
        """
        print("\n" + "="*70)
        print("🔍 Overfitting Verification Started")
        print("="*70 + "\n")
        
        # Load test data
        test_df = self.load_test_data()
        
        # Generate signals with optimized parameters
        print("🔧 Generating signals with optimized parameters...")
        signal_gen = SignalGeneratorWrapper(test_df)
        df_signals = signal_gen.generate_signals(self.params)
        
        # Evaluate on test data
        print("📊 Evaluating on test data...")
        evaluator = LightGBMEvaluator(n_splits=3)
        test_results = evaluator.evaluate(df_signals)
        
        if not test_results['success']:
            print("❌ Evaluation failed on test data")
            return {'success': False}
        
        test_metrics = test_results['metrics']
        test_n_signals = test_results['n_signals']
        
        # Compare with training metrics
        comparison = self._compare_metrics(test_metrics, test_n_signals)
        
        # Diagnose overfitting
        diagnosis = self._diagnose_overfitting(comparison)
        
        # Print results
        self._print_results(comparison, diagnosis)
        
        # Export report
        self._export_report(comparison, diagnosis)
        
        return {
            'success': True,
            'test_metrics': test_metrics,
            'train_metrics': self.train_metrics,
            'comparison': comparison,
            'diagnosis': diagnosis
        }
    
    def _compare_metrics(self, test_metrics: Dict[str, float], 
                        test_n_signals: int) -> Dict[str, Any]:
        """Compare training and test metrics."""
        train_acc = self.train_metrics.get('accuracy', 0.0)
        train_f1 = self.train_metrics.get('f1_score', 0.0)
        train_n_signals = self.train_metrics.get('n_signals', 0)
        
        test_acc = test_metrics.get('accuracy', 0.0)
        test_f1 = test_metrics.get('f1_score', 0.0)
        
        return {
            'accuracy_train': train_acc,
            'accuracy_test': test_acc,
            'accuracy_diff': train_acc - test_acc,
            'f1_train': train_f1,
            'f1_test': test_f1,
            'f1_diff': train_f1 - test_f1,
            'signals_train': train_n_signals,
            'signals_test': test_n_signals,
            'signals_diff': train_n_signals - test_n_signals,
        }
    
    def _diagnose_overfitting(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose level of overfitting.
        
        Returns:
            Diagnosis with severity level and recommendations
        """
        acc_diff = comparison['accuracy_diff']
        f1_diff = comparison['f1_diff']
        
        # Determine severity
        if acc_diff < 0.05 and f1_diff < 0.05:
            severity = "NONE"
            level = "✅ No significant overfitting detected"
            color = "green"
        elif acc_diff < 0.10 and f1_diff < 0.10:
            severity = "LOW"
            level = "⚠️  Low overfitting - acceptable for production"
            color = "yellow"
        elif acc_diff < 0.15 and f1_diff < 0.15:
            severity = "MODERATE"
            level = "⚠️  Moderate overfitting - use with caution"
            color = "orange"
        else:
            severity = "HIGH"
            level = "❌ High overfitting - DO NOT use in production"
            color = "red"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(severity, comparison)
        
        return {
            'severity': severity,
            'level': level,
            'color': color,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, severity: str, 
                                 comparison: Dict[str, Any]) -> list:
        """Generate recommendations based on overfitting level."""
        recommendations = []
        
        if severity == "NONE":
            recommendations.append("✅ Parameters perform well on unseen data")
            recommendations.append("✅ Safe to use in production")
            recommendations.append("💡 Monitor performance on live data regularly")
        
        elif severity == "LOW":
            recommendations.append("✅ Parameters are reasonably robust")
            recommendations.append("⚠️  Consider additional validation on different time periods")
            recommendations.append("💡 Monitor performance closely in production")
        
        elif severity == "MODERATE":
            recommendations.append("⚠️  Parameters may be too specialized to training data")
            recommendations.append("💡 Try re-optimizing with more diverse data")
            recommendations.append("💡 Consider reducing parameter search space")
            recommendations.append("💡 Use ensemble methods or parameter averaging")
        
        else:  # HIGH
            recommendations.append("❌ DO NOT deploy these parameters to production")
            recommendations.append("🔧 Re-optimize with:")
            recommendations.append("   - More diverse training data")
            recommendations.append("   - Stronger regularization")
            recommendations.append("   - Simpler parameter space")
            recommendations.append("   - Cross-validation on multiple time periods")
        
        # Specific recommendations based on metrics
        if comparison['accuracy_test'] < 0.55:
            recommendations.append("⚠️  Test accuracy is below minimum threshold (55%)")
        
        if abs(comparison['signals_diff']) > comparison['signals_train'] * 0.3:
            recommendations.append("⚠️  Signal count varies significantly between train/test")
        
        return recommendations
    
    def _print_results(self, comparison: Dict[str, Any], 
                      diagnosis: Dict[str, Any]):
        """Print verification results."""
        print("\n" + "="*70)
        print("📊 VERIFICATION RESULTS")
        print("="*70 + "\n")
        
        # Metrics comparison
        print("Training vs Test Metrics:")
        print("-"*70)
        print(f"  Accuracy:  Train={comparison['accuracy_train']:.1%}  "
              f"Test={comparison['accuracy_test']:.1%}  "
              f"Diff={comparison['accuracy_diff']:+.1%}")
        print(f"  F1-Score:  Train={comparison['f1_train']:.1%}  "
              f"Test={comparison['f1_test']:.1%}  "
              f"Diff={comparison['f1_diff']:+.1%}")
        print(f"  Signals:   Train={comparison['signals_train']}  "
              f"Test={comparison['signals_test']}  "
              f"Diff={comparison['signals_diff']:+d}")
        
        print("\n" + diagnosis['level'])
        print("-"*70)
        
        print("\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "="*70 + "\n")
    
    def _export_report(self, comparison: Dict[str, Any], 
                      diagnosis: Dict[str, Any]):
        """Export verification report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.params_file).parent
        filepath = output_dir / f"overfitting_report_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("OVERFITTING VERIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters File: {self.params_file}\n")
            f.write(f"Test Data: {self.test_data_path or 'Synthetic'}\n\n")
            
            f.write("METRICS COMPARISON:\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy Train: {comparison['accuracy_train']:.4f}\n")
            f.write(f"Accuracy Test:  {comparison['accuracy_test']:.4f}\n")
            f.write(f"Difference:     {comparison['accuracy_diff']:+.4f}\n\n")
            
            f.write(f"F1-Score Train: {comparison['f1_train']:.4f}\n")
            f.write(f"F1-Score Test:  {comparison['f1_test']:.4f}\n")
            f.write(f"Difference:     {comparison['f1_diff']:+.4f}\n\n")
            
            f.write(f"Signals Train:  {comparison['signals_train']}\n")
            f.write(f"Signals Test:   {comparison['signals_test']}\n")
            f.write(f"Difference:     {comparison['signals_diff']:+d}\n\n")
            
            f.write("DIAGNOSIS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Severity: {diagnosis['severity']}\n")
            f.write(f"{diagnosis['level']}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*70 + "\n")
            for rec in diagnosis['recommendations']:
                f.write(f"{rec}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"📄 Report saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify optimized parameters for overfitting'
    )
    parser.add_argument(
        'params_file',
        help='Path to optimized parameters JSON file'
    )
    parser.add_argument(
        '--test-data',
        dest='test_data',
        help='Path to test data CSV file (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 Overfitting Verification Tool")
    print("="*70 + "\n")
    
    try:
        verifier = OverfittingVerifier(args.params_file, args.test_data)
        results = verifier.verify()
        
        if results['success']:
            print("✅ Verification completed successfully")
            
            # Exit with appropriate code
            severity = results['diagnosis']['severity']
            if severity == "HIGH":
                sys.exit(2)  # High overfitting
            elif severity == "MODERATE":
                sys.exit(1)  # Moderate overfitting
            else:
                sys.exit(0)  # Low or no overfitting
        else:
            print("❌ Verification failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
