import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

def analyze_outliers_and_recommend_clipping(data_dir, sample_size=200):
    """
    Analyze outliers in depth data and recommend clipping thresholds
    """
    # Get all depth files
    depth_files = glob.glob(os.path.join(data_dir, "*depth.npy"))
    
    if len(depth_files) > sample_size:
        depth_files = np.random.choice(depth_files, sample_size, replace=False)
    
    print(f"Analyzing outliers in {len(depth_files)} depth files...")
    
    all_valid_depths = []
    extreme_outliers = []
    
    for depth_file in tqdm(depth_files):
        depth = np.load(depth_file)
        valid_depths = depth[depth > 0]
        
        if len(valid_depths) > 0:
            all_valid_depths.extend(valid_depths)
            
            # Find extreme outliers (beyond 99.9th percentile)
            p999 = np.percentile(valid_depths, 99.9)
            outliers = valid_depths[valid_depths > p999]
            if len(outliers) > 0:
                extreme_outliers.extend(outliers)
    
    all_valid_depths = np.array(all_valid_depths)
    extreme_outliers = np.array(extreme_outliers)
    
    # Calculate statistics
    percentiles = [90, 95, 99, 99.5, 99.9, 99.95, 99.99]
    print("\n" + "="*50)
    print("OUTLIER ANALYSIS")
    print("="*50)
    
    print(f"Total valid pixels: {len(all_valid_depths):,}")
    print(f"Extreme outliers (>99.9th percentile): {len(extreme_outliers):,}")
    print(f"Outlier ratio: {len(extreme_outliers)/len(all_valid_depths)*100:.4f}%")
    
    print(f"\nDEPTH PERCENTILES:")
    for p in percentiles:
        value = np.percentile(all_valid_depths, p)
        print(f"  {p:5.2f}th percentile: {value:8.3f}m")
    
    # IQR-based outlier detection
    q1 = np.percentile(all_valid_depths, 25)
    q3 = np.percentile(all_valid_depths, 75)
    iqr = q3 - q1
    
    # Standard outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Extreme outlier bounds
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr
    
    print(f"\nIQR-BASED OUTLIER DETECTION:")
    print(f"  Q1: {q1:.3f}m")
    print(f"  Q3: {q3:.3f}m")
    print(f"  IQR: {iqr:.3f}m")
    print(f"  Standard outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]m")
    print(f"  Extreme outlier bounds: [{extreme_lower:.3f}, {extreme_upper:.3f}]m")
    
    # Count outliers
    standard_outliers = np.sum((all_valid_depths < lower_bound) | (all_valid_depths > upper_bound))
    extreme_outliers_iqr = np.sum((all_valid_depths < extreme_lower) | (all_valid_depths > extreme_upper))
    
    print(f"  Standard outliers: {standard_outliers:,} ({standard_outliers/len(all_valid_depths)*100:.2f}%)")
    print(f"  Extreme outliers: {extreme_outliers_iqr:,} ({extreme_outliers_iqr/len(all_valid_depths)*100:.2f}%)")
    
    # Recommendations
    print(f"\n" + "="*50)
    print("CLIPPING RECOMMENDATIONS")
    print("="*50)
    
    # Based on building structure expectations
    reasonable_min = 0.5  # Minimum realistic depth for buildings
    reasonable_max_95 = np.percentile(all_valid_depths, 95)
    reasonable_max_99 = np.percentile(all_valid_depths, 99)
    
    print(f"For building structure finetuning:")
    print(f"  Conservative clipping: [{reasonable_min:.1f}m, {reasonable_max_95:.1f}m] (keeps 95% of data)")
    print(f"  Moderate clipping:     [{reasonable_min:.1f}m, {reasonable_max_99:.1f}m] (keeps 99% of data)")
    print(f"  Loose clipping:        [{reasonable_min:.1f}m, {upper_bound:.1f}m] (IQR-based)")
    
    # Visualize outliers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Full distribution
    axes[0, 0].hist(all_valid_depths, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Depth (meters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Full Depth Distribution')
    axes[0, 0].axvline(reasonable_max_99, color='red', linestyle='--', label=f'99th percentile: {reasonable_max_99:.1f}m')
    axes[0, 0].legend()
    
    # 2. Log scale
    axes[0, 1].hist(all_valid_depths, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Depth (meters)')
    axes[0, 1].set_ylabel('Frequency (log)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Full Distribution (Log Scale)')
    
    # 3. Zoomed to 99th percentile
    filtered_depths = all_valid_depths[all_valid_depths <= reasonable_max_99]
    axes[0, 2].hist(filtered_depths, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_xlabel('Depth (meters)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Distribution up to 99th Percentile\n({reasonable_max_99:.1f}m)')
    
    # 4. Box plot showing outliers
    axes[1, 0].boxplot(all_valid_depths, showfliers=True)
    axes[1, 0].set_ylabel('Depth (meters)')
    axes[1, 0].set_title('Box Plot with Outliers')
    
    # 5. Cumulative distribution
    sorted_depths = np.sort(all_valid_depths)
    cumulative = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)
    axes[1, 1].plot(sorted_depths, cumulative)
    axes[1, 1].axvline(reasonable_max_95, color='orange', linestyle='--', label=f'95th: {reasonable_max_95:.1f}m')
    axes[1, 1].axvline(reasonable_max_99, color='red', linestyle='--', label=f'99th: {reasonable_max_99:.1f}m')
    axes[1, 1].set_xlabel('Depth (meters)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Impact of different clipping strategies
    clipping_strategies = [
        ('No clipping', 0, np.inf),
        ('Conservative (95%)', reasonable_min, reasonable_max_95),
        ('Moderate (99%)', reasonable_min, reasonable_max_99),
        ('IQR-based', reasonable_min, upper_bound)
    ]
    
    retention_rates = []
    labels = []
    
    for name, min_val, max_val in clipping_strategies:
        retained = np.sum((all_valid_depths >= min_val) & (all_valid_depths <= max_val))
        retention_rate = retained / len(all_valid_depths) * 100
        retention_rates.append(retention_rate)
        labels.append(f'{name}\n{retention_rate:.1f}%')
    
    axes[1, 2].bar(range(len(retention_rates)), retention_rates, 
                   color=['gray', 'orange', 'red', 'blue'], alpha=0.7)
    axes[1, 2].set_ylabel('Data Retention (%)')
    axes[1, 2].set_title('Data Retention by Clipping Strategy')
    axes[1, 2].set_xticks(range(len(labels)))
    axes[1, 2].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return recommendations
    return {
        'conservative_range': (reasonable_min, reasonable_max_95),
        'moderate_range': (reasonable_min, reasonable_max_99),
        'iqr_range': (reasonable_min, upper_bound),
        'percentiles': {p: np.percentile(all_valid_depths, p) for p in percentiles},
        'retention_rates': dict(zip([s[0] for s in clipping_strategies], retention_rates))
    }

def create_clipped_depth_preprocessing_function(min_depth=0.5, max_depth=100.0):
    """
    Create a preprocessing function for clipping depths
    """
    def preprocess_depth(depth_array):
        """
        Clip depth values to reasonable range for building structures
        """
        # Clip valid depths
        clipped = np.clip(depth_array, min_depth, max_depth)
        
        # Preserve zero values (invalid regions)
        clipped[depth_array == 0] = 0
        
        return clipped
    
    return preprocess_depth

if __name__ == "__main__":
    data_dir = "data"
    
    print("Analyzing outliers and creating clipping recommendations...")
    recommendations = analyze_outliers_and_recommend_clipping(data_dir)
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS FOR FINETUNING")
    print("="*60)
    
    print("\nBased on the analysis, I recommend:")
    print(f"1. MODERATE CLIPPING: {recommendations['moderate_range'][0]:.1f}m to {recommendations['moderate_range'][1]:.1f}m")
    print(f"   - Retains {recommendations['retention_rates']['Moderate (99%)']:.1f}% of valid data")
    print(f"   - Removes extreme outliers that could destabilize training")
    print(f"   - Suitable for building structure finetuning")
    
    print(f"\n2. Add this to your finetuning dataset:")
    print(f"   depth_clip_fn = create_clipped_depth_preprocessing_function({recommendations['moderate_range'][0]:.1f}, {recommendations['moderate_range'][1]:.1f})")
    print(f"   depth = depth_clip_fn(depth)  # Apply before training")
    
    print(f"\n3. This will:")
    print(f"   ✓ Remove unrealistic depth values (>{recommendations['moderate_range'][1]:.1f}m)")
    print(f"   ✓ Ensure minimum depth for close surfaces (>{recommendations['moderate_range'][0]:.1f}m)")
    print(f"   ✓ Preserve zero values for sky/ground regions")
    print(f"   ✓ Improve training stability and convergence")