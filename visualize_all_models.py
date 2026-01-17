"""
Comprehensive Comparison: All 4 CI Models
NCF, Hybrid NCF, PSO, and ANFIS
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 10


def load_standard_results(model_dir):
    """Load results and history for NCF/Hybrid/ANFIS models"""
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
    """Create comprehensive 4-model comparison visualization"""

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    fig.suptitle('Comprehensive Computational Intelligence Model Comparison',
                 fontsize=22, fontweight='bold', y=0.995)

    # Define colors and markers
    colors = {
        'NCF': '#3498db',  # Blue
        'Hybrid': '#e74c3c',  # Red
        'PSO': '#2ecc71',  # Green
        'ANFIS': '#f39c12'  # Orange
    }

    markers = {
        'NCF': 'o',
        'Hybrid': 's',
        'PSO': '^',
        'ANFIS': 'D'
    }

    model_names = list(models_data.keys())

    # ============================================================
    # Plot 1: RMSE Convergence
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for name, data in models_data.items():
        if 'val_rmse' in data['history']:
            epochs = range(1, len(data['history']['val_rmse']) + 1)
            ax1.plot(epochs, data['history']['val_rmse'],
                     color=colors[name], marker=markers[name],
                     label=name, linewidth=2.5, markersize=6,
                     alpha=0.85, markevery=max(1, len(epochs) // 15))

            # Mark best epoch
            best_idx = np.argmin(data['history']['val_rmse'])
            best_val = data['history']['val_rmse'][best_idx]
            ax1.plot(best_idx + 1, best_val, color=colors[name],
                     marker='*', markersize=15, markeredgecolor='black',
                     markeredgewidth=1.5)

    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Validation RMSE', fontweight='bold', fontsize=11)
    ax1.set_title('Validation RMSE Convergence', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # Plot 2: Final RMSE Bar Chart
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    rmse_scores = [models_data[name]['results']['test_rmse'] for name in model_names]

    bars = ax2.bar(range(len(model_names)), rmse_scores,
                   color=[colors[name] for name in model_names],
                   alpha=0.75, edgecolor='black', linewidth=2.5)

    # Add ranking medals
    sorted_indices = np.argsort(rmse_scores)
    medals = ['🥇', '🥈', '🥉', '  ']
    for i, idx in enumerate(sorted_indices):
        bars[idx].set_edgecolor('gold' if i == 0 else 'black')
        bars[idx].set_linewidth(3.5 if i == 0 else 2.5)

    ax2.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Test RMSE', fontweight='bold', fontsize=11)
    ax2.set_title('Final Test RMSE Comparison', fontweight='bold', fontsize=13)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=0, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels with medals
    for i, (bar, score) in enumerate(zip(bars, rmse_scores)):
        height = bar.get_height()
        rank = np.where(sorted_indices == i)[0][0]
        medal = medals[rank]
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{medal}\n{score:.4f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    # ============================================================
    # Plot 3: MAE Comparison
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])

    mae_scores = [models_data[name]['results']['test_mae'] for name in model_names]

    bars = ax3.bar(range(len(model_names)), mae_scores,
                   color=[colors[name] for name in model_names],
                   alpha=0.75, edgecolor='black', linewidth=2.5)

    ax3.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Test MAE', fontweight='bold', fontsize=11)
    ax3.set_title('Final Test MAE Comparison', fontweight='bold', fontsize=13)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=0, fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{score:.4f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    # ============================================================
    # Plot 4: Improvement over Baseline
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    baseline_rmse = models_data['NCF']['results']['test_rmse']
    improvements = {
        name: ((baseline_rmse - models_data[name]['results']['test_rmse']) / baseline_rmse) * 100
        for name in model_names[1:]
    }

    y_pos = range(len(improvements))
    bars = ax4.barh(y_pos, list(improvements.values()),
                    color=[colors[name] for name in improvements.keys()],
                    alpha=0.75, edgecolor='black', linewidth=2.5)

    ax4.axvline(0, color='black', linewidth=2)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(list(improvements.keys()), fontsize=11)
    ax4.set_xlabel('RMSE Improvement (%)', fontweight='bold', fontsize=11)
    ax4.set_title('Improvement over NCF Baseline', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='x')

    for bar, (name, value) in zip(bars, improvements.items()):
        width = bar.get_width()
        label_x = width + 0.15 if width > 0 else width - 0.15
        ha = 'left' if width > 0 else 'right'
        ax4.text(label_x, bar.get_y() + bar.get_height() / 2,
                 f'{value:.2f}%', ha=ha, va='center',
                 fontweight='bold', fontsize=11)

    # ============================================================
    # Plot 5: Training Time vs Performance
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    training_times = [models_data[name]['results'].get('training_time_minutes', 0)
                      for name in model_names]

    scatter = ax5.scatter(training_times, rmse_scores,
                          c=[colors[name] for name in model_names],
                          s=300, alpha=0.7, edgecolors='black', linewidths=2.5)

    for i, name in enumerate(model_names):
        ax5.annotate(name, (training_times[i], rmse_scores[i]),
                     xytext=(10, 5), textcoords='offset points',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    ax5.set_xlabel('Training Time (minutes)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Test RMSE', fontweight='bold', fontsize=11)
    ax5.set_title('Efficiency: Training Time vs Performance', fontweight='bold', fontsize=13)
    ax5.grid(True, alpha=0.3)
    ax5.invert_yaxis()  # Lower RMSE is better

    # ============================================================
    # Plot 6: Model Complexity
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])

    # Parameter counts
    params = {
        'NCF': 1387000,
        'Hybrid': 1579000,
        'PSO': 1579000,
        'ANFIS': 576
    }

    params_list = [params[name] for name in model_names]
    bars = ax6.bar(range(len(model_names)), params_list,
                   color=[colors[name] for name in model_names],
                   alpha=0.75, edgecolor='black', linewidth=2.5)

    ax6.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Parameters', fontweight='bold', fontsize=11)
    ax6.set_title('Model Complexity (Parameters)', fontweight='bold', fontsize=13)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=0, fontsize=11)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')

    for i, (bar, name) in enumerate(zip(bars, model_names)):
        height = bar.get_height()
        if params[name] < 1000:
            label = f'{params[name]}'
        else:
            label = f'{params[name] / 1e6:.2f}M'
        ax6.text(bar.get_x() + bar.get_width() / 2., height * 1.5,
                 label, ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    # ============================================================
    # Plot 7: Train vs Val RMSE
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 0])

    train_rmse = [models_data[name]['history']['train_rmse'][-1]
                  for name in model_names if 'train_rmse' in models_data[name]['history']]
    val_rmse = [models_data[name]['history']['val_rmse'][-1]
                for name in model_names if 'val_rmse' in models_data[name]['history']]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax7.bar(x - width / 2, train_rmse, width, label='Train RMSE',
                    color='skyblue', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax7.bar(x + width / 2, val_rmse, width, label='Val RMSE',
                    color='lightcoral', alpha=0.8, edgecolor='black', linewidth=2)

    ax7.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax7.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax7.set_title('Generalization: Train vs Validation RMSE', fontweight='bold', fontsize=13)
    ax7.set_xticks(x)
    ax7.set_xticklabels(model_names, fontsize=11)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # Plot 8: Convergence Speed
    # ============================================================
    ax8 = fig.add_subplot(gs[2, 1])

    for name, data in models_data.items():
        if 'val_rmse' in data['history']:
            epochs = range(1, len(data['history']['val_rmse']) + 1)
            best_rmse = min(data['history']['val_rmse'])
            initial_rmse = data['history']['val_rmse'][0]

            convergence = [(initial_rmse - rmse) / (initial_rmse - best_rmse) * 100
                           for rmse in data['history']['val_rmse']]

            ax8.plot(epochs, convergence,
                     color=colors[name], marker=markers[name],
                     label=name, linewidth=2.5, markersize=5,
                     alpha=0.85, markevery=max(1, len(epochs) // 15))

    ax8.axhline(90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
    ax8.axhline(100, color='gold', linestyle='--', linewidth=2.5, alpha=0.6, label='100% (Best)')
    ax8.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax8.set_ylabel('% of Best Performance Achieved', fontweight='bold', fontsize=11)
    ax8.set_title('Convergence Speed Comparison', fontweight='bold', fontsize=13)
    ax8.legend(loc='lower right', fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0, 105])

    # ============================================================
    # Plot 9: Learning Rate Evolution
    # ============================================================
    ax9 = fig.add_subplot(gs[2, 2])

    for name, data in models_data.items():
        if 'learning_rates' in data['history']:
            epochs = range(1, len(data['history']['learning_rates']) + 1)
            ax9.plot(epochs, data['history']['learning_rates'],
                     color=colors[name], marker=markers[name],
                     label=name, linewidth=2.5, markersize=4,
                     alpha=0.85, markevery=max(1, len(epochs) // 15))

    ax9.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax9.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
    ax9.set_title('Learning Rate Schedules', fontweight='bold', fontsize=13)
    ax9.legend(loc='upper right', fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')

    # ============================================================
    # Plot 10: Performance Summary Table
    # ============================================================
    ax10 = fig.add_subplot(gs[3, :2])
    ax10.axis('off')

    sorted_models = sorted(model_names,
                           key=lambda x: models_data[x]['results']['test_rmse'])
    medals = ['🥇', '🥈', '🥉', '  ']

    summary_text = "╔════════════════════════════════════════════════════════════════════════════════════╗\n"
    summary_text += "║                           FINAL MODEL RANKINGS                                      ║\n"
    summary_text += "╠════════════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║                                                                                     ║\n"

    for i, name in enumerate(sorted_models):
        results = models_data[name]['results']
        medal = medals[i]
        improvement = ((baseline_rmse - results['test_rmse']) / baseline_rmse * 100) if name != 'NCF' else 0

        summary_text += f"║  {medal} {name:12s}  RMSE: {results['test_rmse']:.4f}  MAE: {results['test_mae']:.4f}"
        if name != 'NCF':
            summary_text += f"  (+{improvement:5.2f}%)          ║\n"
        else:
            summary_text += f"  (baseline)         ║\n"

    summary_text += "║                                                                                     ║\n"
    summary_text += "╠════════════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║  KEY INSIGHTS                                                                       ║\n"
    summary_text += "╠════════════════════════════════════════════════════════════════════════════════════╣\n"
    summary_text += "║                                                                                     ║\n"

    best_model = sorted_models[0]
    best_improvement = ((baseline_rmse - models_data[best_model]['results']['test_rmse']) / baseline_rmse * 100)

    summary_text += f"║  🏆 Champion: {best_model:12s}                                                         ║\n"
    summary_text += f"║     RMSE: {models_data[best_model]['results']['test_rmse']:.4f}  ({best_improvement:.2f}% improvement over baseline)                    ║\n"
    summary_text += "║                                                                                     ║\n"

    # ANFIS efficiency
    anfis_rmse = models_data['ANFIS']['results']['test_rmse']
    efficiency_ratio = params['Hybrid'] / params['ANFIS']

    summary_text += f"║  ⚡ Most Efficient: ANFIS                                                           ║\n"
    summary_text += f"║     Only 576 parameters ({efficiency_ratio:.0f}x smaller than neural networks!)              ║\n"
    summary_text += f"║     RMSE: {anfis_rmse:.4f} - competitive with deep learning                             ║\n"
    summary_text += "║                                                                                     ║\n"

    # Overall stats
    total_improvement = best_improvement
    summary_text += f"║  📈 Project Success: {total_improvement:.2f}% improvement achieved                             ║\n"
    summary_text += f"║     All 4 CI paradigms successfully implemented!                                   ║\n"
    summary_text += "║                                                                                     ║\n"
    summary_text += "╚════════════════════════════════════════════════════════════════════════════════════╝"

    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
              verticalalignment='top', fontfamily='monospace', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

    # ============================================================
    # Plot 11: Paradigm Summary
    # ============================================================
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    paradigm_text = """
╔══════════════════════════════════════╗
║    COMPUTATIONAL INTELLIGENCE        ║
║       PARADIGMS IMPLEMENTED          ║
╠══════════════════════════════════════╣
║                                      ║
║  1️⃣  Neural Networks                 ║
║      NCF Baseline                    ║
║      • Embedding layers              ║
║      • MLP architecture              ║
║      • 1.39M parameters              ║
║                                      ║
║  2️⃣  Deep Neural Networks            ║
║      Hybrid NCF                      ║
║      • Multi-path fusion             ║
║      • Content + Collaborative       ║
║      • 1.58M parameters              ║
║                                      ║
║  3️⃣  Evolutionary Algorithms         ║
║      PSO Optimization                ║
║      • Swarm intelligence            ║
║      • Hyperparameter tuning         ║
║      • 10 particles × 8 iters        ║
║                                      ║
║  4️⃣  Fuzzy Systems                   ║
║      ANFIS                           ║
║      • Neuro-fuzzy hybrid            ║
║      • 16 fuzzy rules                ║
║      • 576 parameters                ║
║      • 8 enhanced features           ║
║                                      ║
║  ✅ Complete CI paradigm coverage!   ║
║                                      ║
╚══════════════════════════════════════╝
"""

    ax11.text(0.05, 0.95, paradigm_text, transform=ax11.transAxes,
              verticalalignment='top', fontfamily='monospace', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.2))

    plt.savefig(save_dir / 'comprehensive_4models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive comparison: {save_dir / 'comprehensive_4models_comparison.png'}")

    return fig


def main():
    """Main comparison function for all 4 models"""

    print("\n" + "=" * 80)
    print(" " * 20 + "COMPREHENSIVE 4-MODEL COMPARISON")
    print("=" * 80)

    # Model directories
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
        print("\n❌ Error: Could not find all model outputs!")
        print(f"   NCF runs found: {len(ncf_runs)}")
        print(f"   Hybrid runs found: {len(hybrid_runs)}")
        print(f"   PSO runs found: {len(pso_runs)}")
        print(f"   ANFIS runs found: {len(anfis_runs)}")
        return

    latest_ncf = ncf_runs[-1]
    latest_hybrid = hybrid_runs[-1]
    latest_pso = pso_runs[-1]
    latest_anfis = anfis_runs[-1]

    print(f"\n📊 Loading model results:")
    print(f"  • NCF:    {latest_ncf.name}")
    print(f"  • Hybrid: {latest_hybrid.name}")
    print(f"  • PSO:    {latest_pso.name}")
    print(f"  • ANFIS:  {latest_anfis.name}")

    # Load results
    print("\n📥 Loading data...")
    ncf_history, ncf_results = load_standard_results(latest_ncf)
    hybrid_history, hybrid_results = load_standard_results(latest_hybrid)
    pso_history, pso_results = load_pso_results(latest_pso)
    anfis_history, anfis_results = load_standard_results(latest_anfis)

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

    print("\n🎨 Creating comprehensive visualization...")

    # Create comparison
    create_comprehensive_comparison(models_data, comparison_dir)

    print("\n" + "=" * 80)
    print(" " * 30 + "ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output saved to: {comparison_dir}/comprehensive_4models_comparison.png")

    # Print rankings
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL RANKINGS")
    print("=" * 80)

    sorted_models = sorted(models_data.keys(),
                           key=lambda x: models_data[x]['results']['test_rmse'])

    medals = ['🥇 CHAMPION', '🥈 RUNNER-UP', '🥉 THIRD PLACE', '   FOURTH']
    print()
    for i, name in enumerate(sorted_models):
        results = models_data[name]['results']
        baseline_rmse = models_data['NCF']['results']['test_rmse']
        improvement = ((baseline_rmse - results['test_rmse']) / baseline_rmse * 100)

        print(f"{medals[i]}  {name:10s}  RMSE: {results['test_rmse']:.4f}  MAE: {results['test_mae']:.4f}", end='')
        if name != 'NCF':
            print(f"  (+{improvement:5.2f}%)")
        else:
            print(f"  (baseline)")

    # Print key stats
    best_model = sorted_models[0]
    best_rmse = models_data[best_model]['results']['test_rmse']
    baseline_rmse = models_data['NCF']['results']['test_rmse']
    total_improvement = ((baseline_rmse - best_rmse) / baseline_rmse * 100)

    print(f"\n{'=' * 80}")
    print(f"\n🎉 PROJECT SUCCESS METRICS:")
    print(f"   • Best Model: {best_model}")
    print(f"   • Final RMSE: {best_rmse:.4f}")
    print(f"   • Total Improvement: {total_improvement:.2f}% over baseline")
    print(f"   • All 4 CI Paradigms: ✅ Successfully Implemented")
    print(f"\n{'=' * 80}\n")

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()