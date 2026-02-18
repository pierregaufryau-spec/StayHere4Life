#!/usr/bin/env python3
"""
Example Usage of Signal Optimizer
==================================

This script demonstrates various ways to use the signal optimizer.

Run this script to see the optimizer in action with different configurations.
"""

from signal_optimizer import SignalOptimizerMain, DEFAULT_CONFIG
import sys


def example_1_basic():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Optimization (Default Settings)")
    print("="*70 + "\n")
    
    config = DEFAULT_CONFIG.copy()
    config['data']['n_samples'] = 2000  # Moderate dataset
    config['optimization']['n_trials'] = 10  # Quick test
    config['output']['generate_plots'] = False  # Skip plots for speed
    
    print("Configuration:")
    print(f"  - Data: Synthetic ({config['data']['n_samples']} samples)")
    print(f"  - Trials: {config['optimization']['n_trials']}")
    print(f"  - Target Signals: {config['optimization']['target_samples']}")
    print()
    
    optimizer = SignalOptimizerMain(config)
    study = optimizer.optimize()
    optimizer.export_results()
    
    print("\n✅ Example 1 completed!")
    return optimizer


def example_2_high_quality():
    """Example 2: Focus on high quality signals."""
    print("\n" + "="*70)
    print("EXAMPLE 2: High Quality Focus (70% Quality Weight)")
    print("="*70 + "\n")
    
    config = DEFAULT_CONFIG.copy()
    config['data']['n_samples'] = 2000
    config['optimization']['n_trials'] = 10
    config['optimization']['quality_weight'] = 0.7  # Prioritize quality
    config['optimization']['min_accuracy'] = 0.60  # Higher threshold
    config['optimization']['target_samples'] = 1500  # Fewer but better
    config['output']['generate_plots'] = False
    
    print("Configuration:")
    print(f"  - Quality Weight: {config['optimization']['quality_weight']*100}%")
    print(f"  - Min Accuracy: {config['optimization']['min_accuracy']*100}%")
    print(f"  - Target Signals: {config['optimization']['target_samples']}")
    print()
    
    optimizer = SignalOptimizerMain(config)
    study = optimizer.optimize()
    optimizer.export_results()
    
    print("\n✅ Example 2 completed!")
    return optimizer


def example_3_high_volume():
    """Example 3: Focus on generating more signals."""
    print("\n" + "="*70)
    print("EXAMPLE 3: High Volume Focus (40% Quality Weight)")
    print("="*70 + "\n")
    
    config = DEFAULT_CONFIG.copy()
    config['data']['n_samples'] = 3000  # More data
    config['optimization']['n_trials'] = 10
    config['optimization']['quality_weight'] = 0.4  # Prioritize volume
    config['optimization']['min_accuracy'] = 0.52  # Lower threshold
    config['optimization']['target_samples'] = 3000  # More signals
    config['output']['generate_plots'] = False
    
    print("Configuration:")
    print(f"  - Quality Weight: {config['optimization']['quality_weight']*100}%")
    print(f"  - Volume Weight: {(1-config['optimization']['quality_weight'])*100}%")
    print(f"  - Target Signals: {config['optimization']['target_samples']}")
    print()
    
    optimizer = SignalOptimizerMain(config)
    study = optimizer.optimize()
    optimizer.export_results()
    
    print("\n✅ Example 3 completed!")
    return optimizer


def example_4_with_plots():
    """Example 4: Generate visualization plots."""
    print("\n" + "="*70)
    print("EXAMPLE 4: With Visualization Plots")
    print("="*70 + "\n")
    
    config = DEFAULT_CONFIG.copy()
    config['data']['n_samples'] = 1500
    config['optimization']['n_trials'] = 15
    config['output']['generate_plots'] = True  # Enable plots
    
    print("Configuration:")
    print(f"  - Trials: {config['optimization']['n_trials']}")
    print(f"  - Plots: Enabled")
    print()
    
    optimizer = SignalOptimizerMain(config)
    study = optimizer.optimize()
    optimizer.export_results()
    optimizer.generate_plots()
    
    print("\n✅ Example 4 completed!")
    print("📊 Check the optimized_results folder for visualization plots")
    return optimizer


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*70)
    print("Signal Optimizer - Example Usage")
    print("="*70 + "\n")
    
    print("Choose an example to run:")
    print("  1. Basic optimization (default settings)")
    print("  2. High quality focus (70% quality weight)")
    print("  3. High volume focus (more signals)")
    print("  4. With visualization plots")
    print("  5. Run all examples")
    print("  0. Exit")
    print()
    
    choice = input("Enter your choice (0-5): ").strip()
    return choice


def main():
    """Main function."""
    print("\n" + "="*70)
    print("🚀 Signal Optimizer Examples")
    print("="*70)
    print("\nThese examples demonstrate different optimization strategies.")
    print("Each example runs quickly with reduced trials for demonstration.")
    print("\nFor production use, increase n_trials to 100-200.")
    
    # Check if running non-interactively
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_1_basic()
        elif example == "2":
            example_2_high_quality()
        elif example == "3":
            example_3_high_volume()
        elif example == "4":
            example_4_with_plots()
        elif example == "all":
            example_1_basic()
            example_2_high_quality()
            example_3_high_volume()
            example_4_with_plots()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_usage.py [1|2|3|4|all]")
            sys.exit(1)
    else:
        # Interactive mode
        while True:
            choice = show_menu()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                example_1_basic()
            elif choice == "2":
                example_2_high_quality()
            elif choice == "3":
                example_3_high_volume()
            elif choice == "4":
                example_4_with_plots()
            elif choice == "5":
                print("\n🚀 Running all examples...")
                example_1_basic()
                example_2_high_quality()
                example_3_high_volume()
                example_4_with_plots()
                print("\n✅ All examples completed!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
    
    print("\n" + "="*70)
    print("📚 For more information, see README_OPTIMIZER.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
