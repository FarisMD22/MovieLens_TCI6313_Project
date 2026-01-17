"""
Compare All Models: NCF, Hybrid NCF, PSO, and ANFIS
Shows comprehensive performance comparison across all models
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10


def load_model_results(model_dir):
    """Load results and history for a model"""
    history_file = model_dir / 'history.json'
    results_file = model_dir / 'results.json'

    with open(history_file, 'r') as f:
        history = json.load(f)

    with open(results_file, 'r') as f:
        results = json.load(f)

    return history, results


def load_pso_results(pso_dir):
    """Load PSO results (different structure)"""
    results_file = pso_dir / 'pso_final_results.json'
    history_file = pso_dir / 'pso_history.json'

    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(history_file, 'r') as f:
        history = json.load(f)

    return history, results


def create_comprehensive_comparison(models_data, save_dir):
    """Create comprehensive comparison visualization for all 4 models"""

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Comprehensive Model Comparison: All 4 Models',
                 fontsize=20, fontweight='bold', y=0.98)

    # Define colors for each model
    colors = {
        'NCF': '#3498db',      # Blue
        'Hybrid': '#e74c3c',   # Red
        'PSO': '#2ecc71',      # Green
        'ANFIS': '#f39c12'     # Orange
    }

    markers = {
        'NCF': 'o',
        'Hybrid': 's',
        'PSO': '^',
        'ANFIS': 'D'
    }

    # Plot 1: RMSE Comparison Over Epochs
    ax1 = fig.add_subplot(gs[0, 0])

    for name, data in models_data.items():
        if 'val_rmse' in data['history']:
            epochs = range(1, len(data['history']['val_rmse']) + 1)
            ax1.plot(epochs, data['history']['val_rmse'],
                    color=colors[name], marker=markers[name],
                    label=name, linewidth=2.5, markersize=5,
                    alpha=0.8, markevery=max(1, len(epochs)//10))

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation RMSE', fontweight='bold')
    ax1.set_title('Validation RMSE Convergence', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final Performance Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])

    model_names = list(models_data.keys())
    rmse_scores = [models_data[name]['results']['test_rmse'] for name in model_names]

    bars = ax2.bar(range(len(model_names)), rmse_scores,
                   color=[colors[name] for name in model_names],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Test RMSE', fontweight='bold')
    ax2.set_title('Final Test RMSE Comparison', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, rmse_scores)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 3: MAE Comparison
    ax3 = fig.add_subplot(gs[0, 2])

    mae_scores = [models_data[name]['results']['test_mae'] for name in model_names]

    bars = ax3.bar(range(len(model_names)), mae_scores,
                   color=[colors[name] for name in model_names],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Test MAE', fontweight='bold')
    ax3.set_title('Final Test MAE Comparison', fontweight='bold', fontsize=12)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 4: Improvement over Baseline
    ax4 = fig.add_subplot(gs[1, 0])

    baseline_rmse = models_data['NCF']['results']['test_rmse']
    improvements = {
        name: ((baseline_rmse - models_data[name]['results']['test_rmse']) / baseline_rmse) * 100
        for name in model_names[1:]  # Skip NCF baseline
    }

    bars = ax4.barh(list(improvements.keys()), list(improvements.values()),
                    color=[colors[name] for name in improvements.keys()],
                    alpha=0.7, edgecolor='black', linewidth=2)

    ax4.axvline(0, color='black', linewidth=1.5)
    ax4.set_xlabel('Improvement over Baseline (%)', fontweight='bold')
    ax4.set_title('RMSE Improvement over NCF Baseline', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, (name, value) in zip(bars, improvements.items()):
        width = bar.get_width()
        label_x = width + 0.2 if width > 0 else width - 0.2
        ha = 'left' if width > 0 else 'right'
        ax4.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{value:.2f}%', ha=ha, va='center',
                fontweight='bold', fontsize=10)

    # Plot 5: Training Time Comparison
    ax5 = fig.add_subplot(gs[1, 1])

    training_times = []
    for name in model_names:
        if 'training_time_minutes' in models_data[name]['results']:
            training_times.append(models_data[name]['results']['training_time_minutes'])
        else:
            training_times.append(0)

    bars = ax5.bar(range(len(model_names)), training_times,
                   color=[colors[name] for name in model_names],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax5.set_xlabel('Model', fontweight='bold')
    ax5.set_ylabel('Training Time (minutes)', fontweight='bold')
    ax5.set_title('Training Time Comparison', fontweight='bold', fontsize=12)
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels(model_names, rotation=0)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                f'{time_val:.1f}m', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 6: Model Complexity
    ax6 = fig.add_subplot(gs[1, 2])

    complexities = {
        'NCF': 1.387,
        'Hybrid': 1.579,
        'PSO': 1.579,  # Same as Hybrid (PSO optimizes it)
        'ANFIS': 0.000576  # 576 parameters
    }

    bars = ax6.bar(range(len(model_names)),
                   [complexities[name] for name in model_names],
                   color=[colors[name] for name in model_names],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax6.set_xlabel('Model', fontweight='bold')
    ax6.set_ylabel('Parameters (Millions)', fontweight='bold')
    ax6.set_title('Model Complexity (Parameter Count)', fontweight='bold', fontsize=12)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=0)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')  # Log scale to show ANFIS

    # Add value labels
    for i, (bar, name) in enumerate(zip(bars, model_names)):
        height = bar.get_height()
        if name == 'ANFIS':
            label = '576'
        else:
            label = f'{complexities[name]:.2f}M'
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                label, ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Plot 7: Training vs Validation RMSE (Last Epoch)
    ax7 = fig.add_subplot(gs[2, 0])

    train_rmse = []
    val_rmse = []
    for name in model_names:
        if 'train_rmse' in models_data[name]['history']:
            train_rmse.append(models_data[name]['history']['train_rmse'][-1])
            val_rmse.append(models_data[name]['history']['val_rmse'][-1])

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax7.bar(x - width/2, train_rmse, width, label='Train RMSE',
                    color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax7.bar(x + width/2, val_rmse, width, label='Val RMSE',
                    color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax7.set_xlabel('Model', fontweight='bold')
    ax7.set_ylabel('RMSE', fontweight='bold')
    ax7.set_title('Train vs Validation RMSE (Final Epoch)', fontweight='bold', fontsize=12)
    ax7.set_xticks(x)
    ax7.set_xticklabels(model_names)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Convergence Speed
    ax8 = fig.add_subplot(gs[2, 1])

    for name, data in models_data.items():
        if 'val_rmse' in data['history']:
            epochs = range(1, len(data['history']['val_rmse']) + 1)
            best_rmse = min(data['history']['val_rmse'])
            initial_rmse = data['history']['val_rmse'][0]

            # Calculate percentage of improvement achieved
            convergence = [(initial_rmse - rmse) / (initial_rmse - best_rmse) * 100
                          for rmse in data['history']['val_rmse']]

            ax8.plot(epochs, convergence,
                    color=colors[name], marker=markers[name],
                    label=name, linewidth=2.5, markersize=5,
                    alpha=0.8, markevery=max(1, len(epochs)//10))

    ax8.axhline(90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
    ax8.axhline(100, color='gold', linestyle='--', linewidth=2, alpha=0.5, label='100% (Best)')
    ax8.set_xlabel('Epoch', fontweight='bold')
    ax8.set_ylabel('% of Best Performance Achieved', fontweight='bold')
    ax8.set_title('Convergence Speed Comparison', fontweight='bold', fontsize=12)
    ax8.legend(loc='lower right', fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0, 105])

    # Plot 9: Learning Rate Evolution
    ax9 = fig.add_subplot(gs[2, 2])

    for name, data in models_data.items():
        if 'learning_rates' in data['history']:
            epochs = range(1, len(data['history']['learning_rates']) + 1)
            ax9.plot(epochs, data['history']['learning_rates'],
                    color=colors[name], marker=markers[name],
                    label=name, linewidth=2.5, markersize=4,
                    alpha=0.8, markevery=max(1, len(epochs)//10))

    ax9.set_xlabel('Epoch', fontweight='bold')
    ax9.set_ylabel('Learning Rate', fontweight='bold')
    ax9.set_title('Learning Rate Schedules', fontweight='bold', fontsize=12)
    ax9.legend(loc='upper right', fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')

    # Plot 10: Performance Summary Table
    ax10 = fig.add_subplot(gs[3, :2])
    ax10.axis('off')

    # Create ranking
    sorted_models = sorted(model_names,
                          key=lambda x: models_data[x]['results']['test_rmse'])

    medals = ['🥇', '🥈', '🥉', '  ']

    summary_text = "╔═══════════════════════════════════════════════════════════════════════════════╗\n"
    summary_text += "║                          FINAL MODEL RANKINGS                                  ║\n"
    summary_text += "╠═══════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║                                                                                ║\n"

    for i, name in enumerate(sorted_models):
        results = models_data[name]['results']
        medal = medals[i]
        improvement = ((baseline_rmse - results['test_rmse']) / baseline_rmse * 100) if name != 'NCF' else 0

        summary_text += f"║  {medal} {name:12s}  RMSE: {results['test_rmse']:.4f}  MAE: {results['test_mae']:.4f}"
        if name != 'NCF':
            summary_text += f"  (+{improvement:5.2f}%)     ║\n"
        else:
            summary_text += f"  (baseline)    ║\n"

    summary_text += "║                                                                                ║\n"
    summary_text += "╠═══════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║  KEY INSIGHTS                                                                  ║\n"
    summary_text += "╠═══════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║                                                                                ║\n"

    # Add key insights
    best_model = sorted_models[0]
    summary_text += f"║  🏆 Best Model: {best_model:12s}                                                   ║\n"
    summary_text += f"║     - Test RMSE: {models_data[best_model]['results']['test_rmse']:.4f}                                                   ║\n"
    summary_text += f"║     - Parameters: {complexities[best_model]:.3f}M                                                  ║\n"
    summary_text += "║                                                                                ║\n"

    # ANFIS efficiency
    anfis_rmse = models_data['ANFIS']['results']['test_rmse']
    hybrid_rmse = models_data['Hybrid']['results']['test_rmse']
    efficiency_ratio = complexities['Hybrid'] / complexities['ANFIS']

    summary_text += f"║  ⚡ Most Efficient: ANFIS                                                      ║\n"
    summary_text += f"║     - Only 576 parameters ({efficiency_ratio:.0f}x smaller than Hybrid NCF)                 ║\n"
    summary_text += f"║     - RMSE: {anfis_rmse:.4f} (competitive with deep models)                          ║\n"
    summary_text += "║                                                                                ║\n"

    # Overall improvement
    total_improvement = ((baseline_rmse - models_data[best_model]['results']['test_rmse']) / baseline_rmse * 100)
    summary_text += f"║  📈 Overall Progress: {total_improvement:.2f}% improvement from baseline                      ║\n"
    summary_text += "║                                                                                ║\n"
    summary_text += "╚═══════════════════════════════════════════════════════════════════════════════╝"

    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

    # Plot 11: Paradigm Classification
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    paradigm_text = """
╔═══════════════════════════════════════╗
║     COMPUTATIONAL INTELLIGENCE        ║
║          PARADIGMS COVERED            ║
╠═══════════════════════════════════════╣
║                                       ║
║  1️⃣  Neural Networks (NCF)            ║
║      • Embeddings + MLP               ║
║      • Collaborative filtering        ║
║      • 1.39M parameters               ║
║                                       ║
║  2️⃣  Deep Neural Networks (Hybrid)    ║
║      • Multi-path architecture        ║
║      • Content + Collaborative        ║
║      • 1.58M parameters               ║
║                                       ║
║  3️⃣  Evolutionary Algorithms (PSO)    ║
║      • Swarm intelligence             ║
║      • Hyperparameter optimization    ║
║      • 10 particles × 8 iterations    ║
║                                       ║
║  4️⃣  Fuzzy Systems (ANFIS)            ║
║      • Neuro-fuzzy hybrid             ║
║      • 16 fuzzy rules                 ║
║      • 576 parameters                 ║
║                                       ║
║  ✅ All paradigms successfully        ║
║     implemented and evaluated!        ║
║                                       ║
╚═══════════════════════════════════════╝
"""

    ax11.text(0.05, 0.95, paradigm_text, transform=ax11.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.2))

    plt.savefig(save_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive comparison: {save_dir / 'comprehensive_comparison.png'}")

    return fig


def main():
    """Main comparison function"""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON: ALL 4 MODELS")
    print("=" * 70)

    # Find model directories
    ncf_dir = Path('outputs/ncf')
    hybrid_dir = Path('outputs/hybrid_ncf')
    pso_dir = Path('outputs/pso')
    anfis_dir = Path('outputs/anfis')

    # Get most recent runs
    ncf_runs = sorted([d for d in ncf_dir.iterdir() if d.is_dir()])
    hybrid_runs = sorted([d for d in hybrid_dir.iterdir() if d.is_dir()])
    pso_runs = sorted([d for d in pso_dir.iterdir() if d.is_dir()])
    anfis_runs = sorted([d for d in anfis_dir.iterdir() if d.is_dir()])

    if not all([ncf_runs, hybrid_runs, pso_runs, anfis_runs]):
        print("❌ Could not find all model outputs!")
        print(f"   NCF: {len(ncf_runs)} runs")
        print(f"   Hybrid: {len(hybrid_runs)} runs")
        print(f"   PSO: {len(pso_runs)} runs")
        print(f"   ANFIS: {len(anfis_runs)} runs")
        return

    latest_ncf = ncf_runs[-1]
    latest_hybrid = hybrid_runs[-1]
    latest_pso = pso_runs[-1]
    latest_anfis = anfis_runs[-1]

    print(f"\n📊 Comparing:")
    print(f"  NCF:    {latest_ncf.name}")
    print(f"  Hybrid: {latest_hybrid.name}")
    print(f"  PSO:    {latest_pso.name}")
    print(f"  ANFIS:  {latest_anfis.name}")

    # Load results
    print("\n📥 Loading results...")
    ncf_history, ncf_results = load_model_results(latest_ncf)
    hybrid_history, hybrid_results = load_model_results(latest_hybrid)
    pso_history, pso_results = load_pso_results(latest_pso)
    anfis_history, anfis_results = load_model_results(latest_anfis)

    # Organize data
    models_data = {
        'NCF': {'history': ncf_history, 'results': ncf_results},
        'Hybrid': {'history': hybrid_history, 'results': hybrid_results},
        'PSO': {'history': pso_history, 'results': pso_results},
        'ANFIS': {'history': anfis_history, 'results': anfis_results}
    }

    # Create output directory
    comparison_dir = Path('outputs/comparison')
    comparison_dir.mkdir(exist_ok=True, parents=True)

    print("\n🎨 Creating visualizations...")

    # Create comprehensive comparison
    create_comprehensive_comparison(models_data, comparison_dir)

    print("\n" + "=" * 70)
    print("✅ COMPREHENSIVE COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Visualizations saved to: {comparison_dir}")

    # Print final rankings
    print("\n" + "=" * 70)
    print("FINAL MODEL RANKINGS")
    print("=" * 70)

    sorted_models = sorted(models_data.keys(),
                          key=lambda x: models_data[x]['results']['test_rmse'])

    medals = ['🥇', '🥈', '🥉', '  ']
    for i, name in enumerate(sorted_models):
        results = models_data[name]['results']
        print(f"\n{medals[i]} {i+1}. {name}")
        print(f"   RMSE: {results['test_rmse']:.4f}")
        print(f"   MAE:  {results['test_mae']:.4f}")

    baseline_rmse = models_data['NCF']['results']['test_rmse']
    best_rmse = models_data[sorted_models[0]]['results']['test_rmse']
    total_improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

    print(f"\n📈 Overall Improvement: {total_improvement:.2f}% from baseline")
    print("=" * 70)

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()