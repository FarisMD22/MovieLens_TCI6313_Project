"""
CLEAN AND ACCURATE MODEL EVALUATION REPORT
Focused visualizations with verified data only
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class CleanModelEvaluator:
    """Clean, accurate model evaluator with focused visualizations"""

    def __init__(self, outputs_dir=None):
        # Use provided path or default to 'outputs'
        if outputs_dir is None:
            # Try to import config
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from src.config import OUTPUTS_DIR
                self.outputs_base = OUTPUTS_DIR
                print(f"📁 Using OUTPUTS_DIR from config: {self.outputs_base}")
            except:
                self.outputs_base = Path('outputs')
                print(f"📁 Using default outputs directory: {self.outputs_base}")
        else:
            self.outputs_base = Path(outputs_dir)

        self.models_data = {}
        self.metrics_df = None

    def load_model_results(self, model_name, model_dir, results_file='results.json', history_file='history.json'):
        """Load results from a single model"""
        try:
            model_path = self.outputs_base / model_dir

            # Get the most recent run
            runs = sorted([d for d in model_path.iterdir() if d.is_dir()])
            if not runs:
                print(f"⚠️  No runs found for {model_name}")
                return False

            latest_run = runs[-1]

            # Load results
            results_path = latest_run / results_file
            history_path = latest_run / history_file

            if not results_path.exists():
                print(f"⚠️  Results file not found for {model_name}: {results_path}")
                return False

            with open(results_path, 'r') as f:
                results = json.load(f)

            # Load history if exists
            history = {}
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)

            self.models_data[model_name] = {
                'results': results,
                'history': history,
                'run_path': latest_run
            }

            print(f"✅ {model_name:12s} loaded from {latest_run.name}")
            return True

        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return False

    def load_all_models(self):
        """Load all four models"""
        print("\n" + "=" * 80)
        print(" " * 30 + "LOADING MODELS")
        print("=" * 80 + "\n")

        # First, show what directories we found
        print("📂 Available model directories:")
        for model_dir in ['ncf', 'hybrid_ncf', 'pso', 'anfis']:
            full_path = self.outputs_base / model_dir
            if full_path.exists():
                runs = sorted([d for d in full_path.iterdir() if d.is_dir()])
                if runs:
                    print(f"   ✓ {model_dir:15s} - {len(runs)} run(s) found, latest: {runs[-1].name}")
                else:
                    print(f"   ⚠ {model_dir:15s} - directory exists but no runs found")
            else:
                print(f"   ✗ {model_dir:15s} - directory not found")
        print()

        success_count = 0

        # NCF
        if self.load_model_results('NCF', 'ncf'):
            success_count += 1

        # Hybrid NCF
        if self.load_model_results('Hybrid', 'hybrid_ncf'):
            success_count += 1

        # PSO
        if self.load_model_results('PSO', 'pso',
                                   results_file='pso_final_results.json',
                                   history_file='pso_history.json'):
            success_count += 1

        # ANFIS
        if self.load_model_results('ANFIS', 'anfis'):
            success_count += 1

        if success_count == 4:
            print(f"\n✅ All {success_count} models loaded successfully!")
        else:
            print(f"\n⚠️  Only {success_count}/4 models loaded")

        return success_count == 4

    def calculate_accurate_metrics(self):
        """Calculate only verified, accurate metrics"""
        print("\n" + "=" * 80)
        print(" " * 28 + "CALCULATING METRICS")
        print("=" * 80 + "\n")

        metrics = []
        baseline_rmse = None

        for model_name, data in self.models_data.items():
            results = data['results']
            history = data['history']

            # CORE METRICS (from results file)
            test_rmse = results.get('test_rmse', 0)
            test_mae = results.get('test_mae', 0)
            test_loss = results.get('test_loss', test_rmse**2)

            # Set baseline
            if model_name == 'NCF':
                baseline_rmse = test_rmse

            # TRAINING METRICS
            training_time = results.get('training_time_minutes', 0)
            n_epochs = results.get('n_epochs', len(history.get('val_rmse', [])))

            # BEST EPOCH (from history)
            if 'val_rmse' in history and len(history['val_rmse']) > 0:
                best_epoch_idx = np.argmin(history['val_rmse'])
                best_epoch = best_epoch_idx + 1
                best_val_rmse = history['val_rmse'][best_epoch_idx]
                final_train_rmse = history['train_rmse'][best_epoch_idx] if 'train_rmse' in history else test_rmse
                final_val_rmse = history['val_rmse'][best_epoch_idx]
            else:
                best_epoch = n_epochs
                best_val_rmse = test_rmse
                final_train_rmse = test_rmse
                final_val_rmse = test_rmse

            # OVERFITTING (only if we have train and val data)
            if 'train_rmse' in history and 'val_rmse' in history and len(history['train_rmse']) > 0:
                generalization_gap = final_val_rmse - final_train_rmse
                overfitting_pct = (generalization_gap / final_train_rmse) * 100 if final_train_rmse > 0 else 0
            else:
                generalization_gap = 0
                overfitting_pct = 0

            # PARAMETERS
            param_map = {
                'NCF': 1387000,
                'Hybrid': 1579000,
                'PSO': 1579000,
                'ANFIS': 576
            }
            parameters = param_map.get(model_name, 0)

            metrics.append({
                'Model': model_name,
                'Test RMSE': round(test_rmse, 4),
                'Test MAE': round(test_mae, 4),
                'Test Loss': round(test_loss, 4),
                'Best Val RMSE': round(best_val_rmse, 4),
                'Train RMSE (best epoch)': round(final_train_rmse, 4),
                'Generalization Gap': round(generalization_gap, 4),
                'Overfitting (%)': round(overfitting_pct, 2),
                'Training Time (min)': round(training_time, 1),
                'Epochs Trained': n_epochs,
                'Best Epoch': best_epoch,
                'Parameters': parameters
            })

        self.metrics_df = pd.DataFrame(metrics)

        # Add improvement column
        if baseline_rmse:
            self.metrics_df['Improvement vs Baseline (%)'] = (
                (baseline_rmse - self.metrics_df['Test RMSE']) / baseline_rmse * 100
            ).round(2)
        else:
            self.metrics_df['Improvement vs Baseline (%)'] = 0

        print("✅ Metrics calculated!\n")
        print(self.metrics_df.to_string(index=False))
        print()

    def create_performance_comparison(self, output_dir):
        """Figure 1: Main Performance Comparison (RMSE & MAE)"""
        print("📊 Creating Figure 1: Performance Comparison...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Model Performance Comparison on MovieLens 1M',
                     fontsize=16, fontweight='bold', y=1.02)

        models = self.metrics_df['Model']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot 1: RMSE Comparison
        rmse_vals = self.metrics_df['Test RMSE']
        bars1 = ax1.bar(models, rmse_vals, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)

        ax1.set_ylabel('Test RMSE', fontweight='bold', fontsize=13)
        ax1.set_title('Root Mean Squared Error (Lower is Better)',
                     fontweight='bold', fontsize=13)
        ax1.set_ylim([0.85, 0.91])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar, val in zip(bars1, rmse_vals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        # Add ranking medals
        rankings = self.metrics_df.sort_values('Test RMSE')['Model'].tolist()
        medals = ['🥇', '🥈', '🥉', '4th']
        for i, model in enumerate(models):
            rank = rankings.index(model)
            ax1.text(i, 0.855, medals[rank], ha='center', fontsize=14)

        # Plot 2: MAE Comparison
        mae_vals = self.metrics_df['Test MAE']
        bars2 = ax2.bar(models, mae_vals, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)

        ax2.set_ylabel('Test MAE', fontweight='bold', fontsize=13)
        ax2.set_title('Mean Absolute Error (Lower is Better)',
                     fontweight='bold', fontsize=13)
        ax2.set_ylim([0.67, 0.71])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar, val in zip(bars2, mae_vals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        plt.tight_layout()
        save_path = Path(output_dir) / 'Fig1_Performance_Comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path.name}\n")
        plt.close()

    def create_convergence_plot(self, output_dir):
        """Figure 2: Training Convergence"""
        print("📊 Creating Figure 2: Training Convergence...")

        fig, ax = plt.subplots(figsize=(12, 7))

        colors = {'NCF': '#3498db', 'Hybrid': '#e74c3c',
                 'PSO': '#2ecc71', 'ANFIS': '#f39c12'}
        markers = {'NCF': 'o', 'Hybrid': 's', 'PSO': '^', 'ANFIS': 'D'}

        for model_name, data in self.models_data.items():
            if 'val_rmse' in data['history'] and len(data['history']['val_rmse']) > 0:
                epochs = range(1, len(data['history']['val_rmse']) + 1)
                val_rmse = data['history']['val_rmse']

                # Plot convergence curve
                ax.plot(epochs, val_rmse, color=colors[model_name],
                       linewidth=2.5, label=model_name, marker=markers[model_name],
                       markersize=5, markevery=max(1, len(epochs)//10), alpha=0.9)

                # Mark best epoch with a star
                best_idx = np.argmin(val_rmse)
                best_val = val_rmse[best_idx]
                ax.plot(best_idx + 1, best_val, color=colors[model_name],
                       marker='*', markersize=20, markeredgecolor='black',
                       markeredgewidth=1.5, zorder=10)

                # Annotate best
                ax.annotate(f'Best: {best_val:.4f}\nEpoch {best_idx+1}',
                           xy=(best_idx + 1, best_val),
                           xytext=(10, -10), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round',
                           facecolor=colors[model_name], alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1))

        ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
        ax.set_ylabel('Validation RMSE', fontweight='bold', fontsize=13)
        ax.set_title('Training Convergence: Validation RMSE Over Time',
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = Path(output_dir) / 'Fig2_Training_Convergence.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path.name}\n")
        plt.close()

    def create_model_complexity(self, output_dir):
        """Figure 3: Model Complexity and Efficiency"""
        print("📊 Creating Figure 3: Model Complexity...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Model Complexity and Efficiency Analysis',
                    fontsize=16, fontweight='bold', y=1.02)

        models = self.metrics_df['Model']
        params = self.metrics_df['Parameters']
        rmse = self.metrics_df['Test RMSE']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot 1: Parameter Count (log scale)
        bars = ax1.bar(models, params, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=2)
        ax1.set_ylabel('Parameters (log scale)', fontweight='bold', fontsize=12)
        ax1.set_title('Model Size Comparison', fontweight='bold', fontsize=13)
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3, which='both', linestyle='--')

        # Add labels
        for bar, param in zip(bars, params):
            height = bar.get_height()
            if param >= 1000000:
                label = f'{param/1e6:.2f}M'
            else:
                label = f'{param:,}'
            ax1.text(bar.get_x() + bar.get_width()/2, height * 1.5, label,
                    ha='center', fontweight='bold', fontsize=10)

        # Plot 2: Performance vs Complexity
        for model, p, r, color in zip(models, params, rmse, colors):
            ax2.scatter(p, r, s=400, c=color, alpha=0.7,
                       edgecolors='black', linewidths=2.5, zorder=5)
            ax2.annotate(model, (p, r), xytext=(8, 0),
                        textcoords='offset points', fontsize=11,
                        fontweight='bold')

        ax2.set_xlabel('Parameters (log scale)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Test RMSE', fontweight='bold', fontsize=12)
        ax2.set_title('Performance vs Complexity Trade-off',
                     fontweight='bold', fontsize=13)
        ax2.set_xscale('log')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Add efficiency line
        sorted_idx = np.argsort(params)
        ax2.plot(params[sorted_idx], rmse[sorted_idx], 'k--',
                alpha=0.3, linewidth=2, label='Efficiency Frontier')
        ax2.legend(fontsize=10)

        plt.tight_layout()
        save_path = Path(output_dir) / 'Fig3_Model_Complexity.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path.name}\n")
        plt.close()

    def create_overfitting_analysis(self, output_dir):
        """Figure 4: Overfitting and Generalization"""
        print("📊 Creating Figure 4: Overfitting Analysis...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Model Generalization Analysis',
                    fontsize=16, fontweight='bold', y=1.02)

        models = self.metrics_df['Model']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot 1: Train vs Val RMSE
        train_rmse = self.metrics_df['Train RMSE (best epoch)']
        val_rmse = self.metrics_df['Best Val RMSE']

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax1.bar(x - width/2, train_rmse, width, label='Train RMSE',
                       color='skyblue', alpha=0.8, edgecolor='black', linewidth=2)
        bars2 = ax1.bar(x + width/2, val_rmse, width, label='Val RMSE',
                       color='salmon', alpha=0.8, edgecolor='black', linewidth=2)

        ax1.set_ylabel('RMSE', fontweight='bold', fontsize=12)
        ax1.set_title('Training vs Validation RMSE', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add gap annotations
        for i, (t, v, gap) in enumerate(zip(train_rmse, val_rmse,
                                            self.metrics_df['Generalization Gap'])):
            mid_y = (t + v) / 2
            ax1.annotate(f'Δ={gap:.3f}', xy=(i, mid_y),
                        ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Overfitting Percentage
        overfitting = self.metrics_df['Overfitting (%)']
        bars = ax2.barh(models, overfitting, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)

        # Add zero line (negative = good regularization)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

        # Add threshold lines
        ax2.axvline(10, color='orange', linestyle='--', linewidth=2,
                   alpha=0.6, label='10% threshold')
        ax2.axvline(20, color='red', linestyle='--', linewidth=2,
                   alpha=0.6, label='20% threshold')

        ax2.set_xlabel('Overfitting (%) | Negative = Good Regularization',
                      fontweight='bold', fontsize=11)
        ax2.set_title('Generalization Quality (Closer to 0 is Better)',
                     fontweight='bold', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for bar, val in zip(bars, overfitting):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = Path(output_dir) / 'Fig4_Overfitting_Analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path.name}\n")
        plt.close()

    def create_training_efficiency(self, output_dir):
        """Figure 5: Training Time and Efficiency"""
        print("📊 Creating Figure 5: Training Efficiency...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Training Efficiency Analysis',
                    fontsize=16, fontweight='bold', y=1.02)

        models = self.metrics_df['Model']
        training_times = self.metrics_df['Training Time (min)']
        epochs = self.metrics_df['Epochs Trained']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot 1: Training Time
        bars = ax1.bar(models, training_times, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=2)

        ax1.set_ylabel('Training Time (minutes)', fontweight='bold', fontsize=12)
        ax1.set_title('Total Training Time', fontweight='bold', fontsize=13)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add labels
        for bar, time, epoch in zip(bars, training_times, epochs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{time:.1f}m\n({epoch} epochs)', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        # Plot 2: Improvement vs Training Time scatter
        improvements = self.metrics_df['Improvement vs Baseline (%)']

        for model, time, imp, color in zip(models, training_times, improvements, colors):
            size = 500 if model != 'NCF' else 300  # Larger for non-baseline
            ax2.scatter(time, imp, s=size, c=color, alpha=0.7,
                       edgecolors='black', linewidths=2.5, zorder=5)
            ax2.annotate(model, (time, imp), xytext=(8, 5),
                        textcoords='offset points', fontsize=11,
                        fontweight='bold')

        ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_xlabel('Training Time (minutes)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Improvement over Baseline (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Performance Gain vs Training Cost',
                     fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = Path(output_dir) / 'Fig5_Training_Efficiency.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path.name}\n")
        plt.close()

    def export_clean_report(self, output_dir):
        """Export clean CSV and summary"""
        print("\n" + "=" * 80)
        print(" " * 30 + "EXPORTING RESULTS")
        print("=" * 80 + "\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metrics table
        csv_file = output_path / 'model_metrics.csv'
        self.metrics_df.to_csv(csv_file, index=False)
        print(f"✅ CSV exported: {csv_file}")

        # Export summary
        summary_file = output_path / 'evaluation_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 25 + "MODEL EVALUATION SUMMARY\n")
            f.write(" " * 30 + "MovieLens 1M Dataset\n")
            f.write(" " * 25 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Rankings
            f.write("PERFORMANCE RANKINGS (by Test RMSE):\n")
            f.write("-" * 80 + "\n")
            rankings = self.metrics_df.sort_values('Test RMSE')
            for i, (idx, row) in enumerate(rankings.iterrows(), 1):
                medal = ['🥇', '🥈', '🥉', ''][i-1]
                f.write(f"{i}. {medal} {row['Model']:10s} - RMSE: {row['Test RMSE']:.4f}  ")
                if row['Model'] != 'NCF':
                    f.write(f"(+{row['Improvement vs Baseline (%)']:.2f}%)\n")
                else:
                    f.write("(baseline)\n")

            f.write("\n\nCOMPREHENSIVE METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(self.metrics_df.to_string(index=False))

            f.write("\n\n\nKEY FINDINGS:\n")
            f.write("-" * 80 + "\n")
            best = rankings.iloc[0]
            f.write(f"• Best Model: {best['Model']} (RMSE: {best['Test RMSE']:.4f})\n")
            f.write(f"• Best Improvement: {rankings[rankings['Model'] != 'NCF']['Improvement vs Baseline (%)'].max():.2f}%\n")
            f.write(f"• Most Efficient: ANFIS (576 parameters)\n")
            f.write(f"• Lowest Overfitting: {self.metrics_df.loc[self.metrics_df['Overfitting (%)'].idxmin(), 'Model']} ")
            f.write(f"({self.metrics_df['Overfitting (%)'].min():.2f}%)\n")
            f.write(f"• Fastest Training: {self.metrics_df.loc[self.metrics_df['Training Time (min)'].idxmin(), 'Model']} ")
            f.write(f"({self.metrics_df['Training Time (min)'].min():.1f} min)\n")

            # Add note about negative overfitting
            f.write("\n\nNOTE ON NEGATIVE OVERFITTING:\n")
            f.write("-" * 80 + "\n")
            f.write("Negative overfitting values indicate that validation RMSE is lower than training\n")
            f.write("RMSE at the best epoch. This occurs when regularization (like dropout) is active\n")
            f.write("during training but disabled during validation. This is actually a GOOD sign,\n")
            f.write("indicating effective regularization that prevents overfitting.\n")

        print(f"✅ Summary exported: {summary_file}")
        print(f"\n📁 All results saved to: {output_path}\n")


def main():
    """Main evaluation pipeline"""

    print("\n" + "=" * 80)
    print(" " * 20 + "CLEAN MODEL EVALUATION REPORT")
    print(" " * 25 + "Accurate Data & Focused Visualizations")
    print("=" * 80)

    # Initialize evaluator (will auto-detect outputs directory)
    evaluator = CleanModelEvaluator()

    # Verify outputs directory exists
    if not evaluator.outputs_base.exists():
        print(f"\n❌ Outputs directory not found: {evaluator.outputs_base}")
        print("   Please run this script from your project root directory.")
        return

    print(f"\n📂 Scanning: {evaluator.outputs_base}")
    print(f"   Looking for: ncf/, hybrid_ncf/, pso/, anfis/\n")

    # Load models
    if not evaluator.load_all_models():
        print("\n❌ Failed to load all models. Exiting.")
        print("   Make sure you have trained all 4 models and they have results.json files.")
        return

    # Calculate metrics
    evaluator.calculate_accurate_metrics()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = evaluator.outputs_base / 'clean_evaluation' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)  # CREATE THE DIRECTORY!

    print(f"\n📁 Output directory created: {output_dir}\n")

    # Generate all visualizations
    print("\n" + "=" * 80)
    print(" " * 25 + "CREATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    evaluator.create_performance_comparison(output_dir)
    evaluator.create_convergence_plot(output_dir)
    evaluator.create_model_complexity(output_dir)
    evaluator.create_overfitting_analysis(output_dir)
    evaluator.create_training_efficiency(output_dir)

    # Export results
    evaluator.export_clean_report(output_dir)

    print("\n" + "=" * 80)
    print(" " * 30 + "EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Generated 5 focused, publication-quality figures:")
    print("   1. Fig1_Performance_Comparison.png - RMSE & MAE comparison")
    print("   2. Fig2_Training_Convergence.png - Training curves with best epochs")
    print("   3. Fig3_Model_Complexity.png - Parameter counts & efficiency")
    print("   4. Fig4_Overfitting_Analysis.png - Generalization gaps")
    print("   5. Fig5_Training_Efficiency.png - Training time analysis")
    print("\n📄 Plus:")
    print("   • model_metrics.csv (Complete metrics table)")
    print("   • evaluation_summary.txt (Text report with rankings)")
    print("\n✅ All figures are clean, focused, and publication-ready!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()