import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Data from the log
steps = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1099, 1199, 1299, 1399, 1499, 1599, 1699, 1799, 1899, 1999]

# Extract all metrics
eval_cost = [42.4, 39.6, 52.3, 33.4, 25.7, 29.4, 21.1, 38.1, 47.7, 38.3, 32.8, 36.6, 52.4, 39.4, 55.2, 44.9, 43.3, 42.7, 26.8, 25.9]
eval_reward = [19.94, 22.23, 22.65, 22.45, 21.98, 22.71, 19.83, 21.96, 21.68, 22.30, 21.12, 21.85, 22.40, 21.95, 20.26, 22.15, 19.96, 22.21, 20.85, 22.04]

loss_actor = [-0.908, -0.476, -0.897, -0.674, -1.025, -0.284, -0.034, -0.677, 0.179, -1.276, -0.424, -0.441, -1.082, 0.761, -0.637, -0.325, -0.466, -1.233, -1.009, -0.932]
loss_q_keep = [0.019, 0.046, 0.064, 0.011, 0.023, 0.026, 0.013, 0.014, 0.056, 0.023, 0.044, 0.047, 0.008, 0.010, 0.017, 0.008, 0.017, 0.012, 0.047, 0.044]
loss_q_forget = [0.091, 0.048, 0.064, 0.064, 0.068, 0.065, 0.055, 0.092, 0.072, 0.075, 0.092, 0.080, 0.044, 0.057, 0.076, 0.065, 0.066, 0.052, 0.054, 0.061]
loss_cq_keep = [0.232, 0.376, 0.180, 0.212, 0.287, 0.365, 0.344, 0.225, 0.435, 0.182, 0.266, 0.312, 0.142, 0.434, 0.197, 0.221, 0.161, 0.146, 0.315, 0.301]
loss_cq_forget = [1.606, 1.632, 1.379, 1.819, 2.120, 1.347, 1.887, 1.850, 1.242, 1.451, 1.449, 1.402, 1.392, 1.322, 1.303, 1.490, 1.933, 1.727, 1.844, 1.559]

Ar_gap = [1.58e-08, 7.92e-09, -1.07e-08, -2.14e-08, 3.73e-09, 5.59e-09, 9.31e-10, 1.16e-08, 2.39e-09, -6.98e-10, 2.61e-08, -9.31e-09, -5.59e-09, -1.40e-08, -2.79e-09, 1.16e-08, -1.68e-08, -6.05e-09, -9.31e-09, 6.98e-09]
Ac_gap = [1.96e-08, 0, 1.40e-08, 3.73e-09, 4.19e-09, -9.31e-09, -6.98e-09, 6.98e-10, 1.14e-08, -4.66e-09, -7.45e-09, 1.30e-08, -9.31e-10, -5.59e-09, -1.86e-09, 1.89e-08, -1.21e-08, -3.26e-09, -9.31e-10, 1.49e-08]

cost_f_original = [0.0293, 0.0313, 0.0293, 0.0352, 0.0273, 0.0293, 0.0352, 0.0352, 0.0273, 0.0273, 0.0332, 0.0391, 0.0352, 0.0176, 0.0273, 0.0332, 0.0469, 0.0293, 0.0488, 0.0449]
cost_f_relabeled = [0.0281, 0.0292, 0.0264, 0.0315, 0.0242, 0.0279, 0.0350, 0.0312, 0.0284, 0.0341, 0.0293, 0.0381, 0.0367, 0.0314, 0.0307, 0.0402, 0.0282, 0.0164, 0.0471, 0.0422]

loss_encoder = [0.535, 0.477, 0.453, 0.446, 0.444, 0.464, 0.438, 0.434, 0.450, 0.435, 0.441, 0.430, 0.429, 0.423, 0.410, 0.422, 0.425, 0.422, 0.397, 0.424]

progress = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ============================================================================
# 1. Main: Eval Cost over Training (Most Important)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(steps, eval_cost, 'o-', color='#e74c3c', linewidth=3, markersize=8, label='Eval Cost')
ax1.axhline(y=21.1, color='#27ae60', linestyle='--', linewidth=2, label='Best Cost (21.1 at step 699)')
ax1.axvline(x=699, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5, label='Convergence Point')
ax1.fill_between(steps, 0, 25, alpha=0.2, color='green', label='Target Zone (< 25)')
ax1.set_xlabel('Training Steps', fontweight='bold')
ax1.set_ylabel('Cost', fontweight='bold')
ax1.set_title('Evaluation Cost: Converges at Step 699, Then Degrades', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_ylim([0, 60])

# Annotate key points
ax1.annotate('✓ Best: 21.1', xy=(699, 21.1), xytext=(699, 10),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')
ax1.annotate('✗ Worst: 55.2', xy=(1499, 55.2), xytext=(1499, 60),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

# ============================================================================
# 2. Actor Loss (Shows instability)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
colors = ['red' if x > 0 else 'blue' for x in loss_actor]
ax2.bar(steps, loss_actor, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=699, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Steps', fontweight='bold')
ax2.set_ylabel('Actor Loss', fontweight='bold')
ax2.set_title('Actor Loss: Sign Flips After Convergence', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.text(0.95, 0.95, 'Red = Positive\nBlue = Negative', 
         transform=ax2.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# 3. Q-Function Losses (Reward Critic)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(steps, loss_q_keep, 'o-', color='#3498db', linewidth=2, markersize=6, label='Q Loss (Keep)')
ax3.plot(steps, loss_q_forget, 's-', color='#e67e22', linewidth=2, markersize=6, label='Q Loss (Forget)')
ax3.axvline(x=699, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xlabel('Steps', fontweight='bold')
ax3.set_ylabel('Loss', fontweight='bold')
ax3.set_title('Reward Critic Losses: Both Stable', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# 4. Cost Critic Losses (Shows high forget loss)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(steps, loss_cq_keep, 'o-', color='#9b59b6', linewidth=2, markersize=6, label='C Loss (Keep)')
ax4.plot(steps, loss_cq_forget, 's-', color='#e74c3c', linewidth=2, markersize=6, label='C Loss (Forget)')
ax4.axvline(x=699, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Steps', fontweight='bold')
ax4.set_ylabel('Loss', fontweight='bold')
ax4.set_title('Cost Critic Losses: Forget Loss Very High!', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.text(0.5, 0.95, '⚠ Forget loss 5-10x higher!', 
         transform=ax4.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
         fontweight='bold')

# ============================================================================
# 5. Advantage Gaps (Should go to zero)
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])
# Convert to more readable scale
Ar_gap_scaled = [x * 1e9 for x in Ar_gap]
Ac_gap_scaled = [x * 1e9 for x in Ac_gap]
ax5.plot(steps, Ar_gap_scaled, 'o-', color='#1abc9c', linewidth=2, markersize=6, label='Ar_gap (×1e-9)')
ax5.plot(steps, Ac_gap_scaled, 's-', color='#f39c12', linewidth=2, markersize=6, label='Ac_gap (×1e-9)')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.axvline(x=699, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax5.set_xlabel('Steps', fontweight='bold')
ax5.set_ylabel('Gap (×1e-9)', fontweight='bold')
ax5.set_title('Advantage Gaps: Near Zero (Good!)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================================
# 6. Relabeling Effectiveness
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(steps, cost_f_original, 'o-', color='#c0392b', linewidth=2, markersize=6, label='Original Cost')
ax6.plot(steps, cost_f_relabeled, 's-', color='#27ae60', linewidth=2, markersize=6, label='Relabeled Cost')
ax6.fill_between(steps, cost_f_original, cost_f_relabeled, alpha=0.3, color='yellow')
ax6.axvline(x=699, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax6.set_xlabel('Steps', fontweight='bold')
ax6.set_ylabel('Cost Value', fontweight='bold')
ax6.set_title('Relabeling: Changes Too Small!', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Calculate average reduction
avg_reduction = np.mean([(orig - rel) / orig * 100 for orig, rel in zip(cost_f_original, cost_f_relabeled)])
ax6.text(0.5, 0.05, f'Avg Reduction: {avg_reduction:.1f}%\n⚠ Need >30% for effectiveness', 
         transform=ax6.transAxes, ha='center', va='bottom',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5),
         fontsize=10, fontweight='bold')

# ============================================================================
# 7. Combined View: Cost vs Reward Tradeoff
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
scatter = ax7.scatter(eval_reward, eval_cost, c=steps, cmap='viridis', 
                      s=100, alpha=0.7, edgecolors='black', linewidth=1)
# Highlight best point
best_idx = eval_cost.index(21.1)
ax7.scatter(eval_reward[best_idx], eval_cost[best_idx], 
           s=500, facecolors='none', edgecolors='green', linewidths=3,
           label='Best (Step 699)')
ax7.axhline(y=25, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target Cost')
ax7.set_xlabel('Reward', fontweight='bold')
ax7.set_ylabel('Cost', fontweight='bold')
ax7.set_title('Cost-Reward Tradeoff', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Training Step', fontweight='bold')

# Add overall title
fig.suptitle('K-Means Unlearning Analysis: Converges at 35% Progress (Step 699), Then Degrades', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('kmeans_unlearning_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: kmeans_unlearning_analysis.png")
plt.show()