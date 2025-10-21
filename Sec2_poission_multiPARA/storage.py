# Visualize how alpha and beta parameters affect the Beta distribution
from scipy.stats import beta

# Define parameter combinations to explore
param_combinations = [
    # (alpha, beta, description)
    (1, 1, "Uniform (α=1, β=1)"),
    (1, 3, "Right-skewed (α=1, β=3)\nOur prior for p_zero"),
    (3, 1, "Left-skewed (α=3, β=1)"),
    (2, 2, "Symmetric (α=2, β=2)"),
    (5, 2, "Left-skewed (α=5, β=2)"),
    (2, 5, "Right-skewed (α=2, β=5)"),
    (0.5, 0.5, "U-shaped (α=0.5, β=0.5)"),
    (10, 10, "Peaked (α=10, β=10)"),
    (2, 10, "Strong right-skew (α=2, β=10)"),
]

# Create x values
x = np.linspace(0, 1, 1000)

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=200)
axes = axes.flatten()

for idx, (alpha, beta_param, description) in enumerate(param_combinations):
    ax = axes[idx]
    
    # Calculate PDF
    pdf = beta.pdf(x, alpha, beta_param)
    
    # Plot
    ax.fill_between(x, pdf, alpha=0.3, color='steelblue')
    ax.plot(x, pdf, linewidth=2.5, color='darkblue', label=f'α={alpha}, β={beta_param}')
    
    # Calculate statistics
    mean = beta.mean(alpha, beta_param)
    mode = (alpha - 1) / (alpha + beta_param - 2) if alpha > 1 and beta_param > 1 else None
    
    # Add vertical lines for mean
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.3f}')
    if mode is not None:
        ax.axvline(mode, color='green', linestyle=':', linewidth=2, label=f'Mode={mode:.3f}')
    
    # Formatting
    ax.set_xlabel('p (probability)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(description, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

plt.suptitle('Beta Distribution: Effect of α and β Parameters', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./beta_distribution_exploration.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("BETA DISTRIBUTION PROPERTIES")
print("="*80)
print(f"{'α':<6} {'β':<6} {'Mean':<10} {'Mode':<10} {'Std':<10} {'Skewness':<15}")
print("-"*80)
for alpha, beta_param, desc in param_combinations:
    mean = beta.mean(alpha, beta_param)
    std = beta.std(alpha, beta_param)
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
    else:
        mode = None
    
    # Skewness direction
    if alpha == beta_param:
        skew = "Symmetric"
    elif alpha > beta_param:
        skew = "Left-skewed"
    else:
        skew = "Right-skewed"
    
    mode_str = f"{mode:.4f}" if mode is not None else "N/A"
    print(f"{alpha:<6.1f} {beta_param:<6.1f} {mean:<10.4f} {mode_str:<10} {std:<10.4f} {skew:<15}")
print("="*80)

print("\n💡 Key Insights:")
print("  • α=1, β=3 (our prior): Right-skewed, favors smaller p_zero values")
print("  • Mean = α/(α+β), so α=1, β=3 gives mean = 0.25")
print("  • Larger α: shifts distribution to the right (higher probabilities)")
print("  • Larger β: shifts distribution to the left (lower probabilities)")
print("  • α=β: symmetric distribution around 0.5")
print("  • α,β > 1: unimodal (bell-shaped)")
print("  • α,β < 1: U-shaped (extreme values more likely)")
print("  • α,β >> 1: highly concentrated around the mean")