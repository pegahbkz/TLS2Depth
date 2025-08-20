import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

def load_normalization_data(results_folder):
    """Load all normalization parameters and compute comprehensive statistics"""
    json_files = glob.glob(os.path.join(results_folder, "*_normalization_params.json"))
    
    if not json_files:
        print(f"No normalization parameter files found in {results_folder}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['filename'] = os.path.basename(json_file).replace('_normalization_params.json', '')
            all_data.append(data)
    
    return pd.DataFrame(all_data)

def compute_research_statistics(df):
    """Compute comprehensive statistics for research paper"""
    stats_dict = {}
    
    # Basic descriptive statistics
    stats_dict['sample_size'] = len(df)
    stats_dict['mean_normalized_difference'] = df['mean_abs_diff_normalized'].mean()
    stats_dict['std_normalized_difference'] = df['mean_abs_diff_normalized'].std()
    stats_dict['median_normalized_difference'] = df['mean_abs_diff_normalized'].median()
    stats_dict['min_normalized_difference'] = df['mean_abs_diff_normalized'].min()
    stats_dict['max_normalized_difference'] = df['mean_abs_diff_normalized'].max()
    
    # Percentile analysis
    stats_dict['q25_normalized_difference'] = df['mean_abs_diff_normalized'].quantile(0.25)
    stats_dict['q75_normalized_difference'] = df['mean_abs_diff_normalized'].quantile(0.75)
    
    # Relative change statistics
    stats_dict['mean_relative_change'] = df['relative_change_percent'].mean()
    stats_dict['std_relative_change'] = df['relative_change_percent'].std()
    stats_dict['median_relative_change'] = df['relative_change_percent'].median()
    
    # Depth range analysis
    stats_dict['mean_original_range'] = (df['original_max'] - df['original_min']).mean()
    stats_dict['mean_finetuned_range'] = (df['finetuned_max'] - df['finetuned_min']).mean()
    
    # Range change analysis
    original_ranges = df['original_max'] - df['original_min']
    finetuned_ranges = df['finetuned_max'] - df['finetuned_min']
    range_ratio = finetuned_ranges / original_ranges
    
    stats_dict['mean_range_ratio'] = range_ratio.mean()
    stats_dict['std_range_ratio'] = range_ratio.std()
    
    # Statistical significance tests
    # Test if normalized differences are significantly different from zero
    t_stat, p_value = stats.ttest_1samp(df['mean_abs_diff_normalized'], 0)
    stats_dict['ttest_statistic'] = t_stat
    stats_dict['ttest_pvalue'] = p_value
    
    # Effect size (Cohen's d)
    cohens_d = df['mean_abs_diff_normalized'].mean() / df['mean_abs_diff_normalized'].std()
    stats_dict['cohens_d'] = cohens_d
    
    # Consistency analysis
    stats_dict['coefficient_of_variation'] = (df['mean_abs_diff_normalized'].std() / 
                                             df['mean_abs_diff_normalized'].mean())
    
    # Classification of changes
    small_changes = (df['relative_change_percent'] < 5).sum()
    medium_changes = ((df['relative_change_percent'] >= 5) & 
                     (df['relative_change_percent'] < 15)).sum()
    large_changes = (df['relative_change_percent'] >= 15).sum()
    
    stats_dict['small_changes_count'] = small_changes
    stats_dict['medium_changes_count'] = medium_changes
    stats_dict['large_changes_count'] = large_changes
    stats_dict['small_changes_percent'] = (small_changes / len(df)) * 100
    stats_dict['medium_changes_percent'] = (medium_changes / len(df)) * 100
    stats_dict['large_changes_percent'] = (large_changes / len(df)) * 100
    
    return stats_dict

def create_research_visualizations(df, output_folder):
    """Create publication-quality visualizations"""
    
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Distribution of normalized differences
    plt.subplot(3, 3, 1)
    plt.hist(df['mean_abs_diff_normalized'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['mean_abs_diff_normalized'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["mean_abs_diff_normalized"].mean():.3f}')
    plt.axvline(df['mean_abs_diff_normalized'].median(), color='green', linestyle='--', 
                label=f'Median: {df["mean_abs_diff_normalized"].median():.3f}')
    plt.xlabel('Mean Absolute Normalized Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Model Differences', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Box plot of relative changes
    plt.subplot(3, 3, 2)
    plt.boxplot(df['relative_change_percent'], patch_artist=True, 
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
    plt.ylabel('Relative Change (%)', fontsize=12)
    plt.title('Box Plot: Relative Changes', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Original vs Fine-tuned ranges
    plt.subplot(3, 3, 3)
    original_ranges = df['original_max'] - df['original_min']
    finetuned_ranges = df['finetuned_max'] - df['finetuned_min']
    plt.scatter(original_ranges, finetuned_ranges, alpha=0.6, s=50, color='purple')
    
    # Add diagonal line
    min_range = min(original_ranges.min(), finetuned_ranges.min())
    max_range = max(original_ranges.max(), finetuned_ranges.max())
    plt.plot([min_range, max_range], [min_range, max_range], 'k--', alpha=0.8, 
             label='No change line')
    
    plt.xlabel('Original Model Depth Range', fontsize=12)
    plt.ylabel('Fine-tuned Model Depth Range', fontsize=12)
    plt.title('Depth Range Comparison', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Change magnitude vs original range
    plt.subplot(3, 3, 4)
    plt.scatter(original_ranges, df['relative_change_percent'], alpha=0.6, s=50, color='orange')
    plt.xlabel('Original Depth Range', fontsize=12)
    plt.ylabel('Relative Change (%)', fontsize=12)
    plt.title('Change Magnitude vs Scene Depth Range', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution
    plt.subplot(3, 3, 5)
    sorted_diffs = np.sort(df['mean_abs_diff_normalized'])
    y_vals = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    plt.plot(sorted_diffs, y_vals, linewidth=2, color='darkgreen')
    plt.axvline(df['mean_abs_diff_normalized'].mean(), color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Mean Absolute Normalized Difference', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Changes', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 6. Change categories pie chart
    plt.subplot(3, 3, 6)
    small_changes = (df['relative_change_percent'] < 5).sum()
    medium_changes = ((df['relative_change_percent'] >= 5) & 
                     (df['relative_change_percent'] < 15)).sum()
    large_changes = (df['relative_change_percent'] >= 15).sum()
    
    sizes = [small_changes, medium_changes, large_changes]
    labels = ['Small (<5%)', 'Medium (5-15%)', 'Large (≥15%)']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Change Magnitudes', fontsize=12)
    
    # 7. Range ratio histogram
    plt.subplot(3, 3, 7)
    range_ratio = finetuned_ranges / original_ranges
    plt.hist(range_ratio, bins=20, alpha=0.7, color='gold', edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='No change (ratio=1)')
    plt.axvline(range_ratio.mean(), color='blue', linestyle='--', 
                label=f'Mean: {range_ratio.mean():.3f}')
    plt.xlabel('Depth Range Ratio (Fine-tuned/Original)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Depth Range Ratio Distribution', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Statistical summary text
    plt.subplot(3, 3, 8)
    plt.axis('off')
    
    stats_text = f"""STATISTICAL SUMMARY
Sample Size: {len(df)} images

NORMALIZED DIFFERENCES:
Mean ± SD: {df['mean_abs_diff_normalized'].mean():.4f} ± {df['mean_abs_diff_normalized'].std():.4f}
Median [IQR]: {df['mean_abs_diff_normalized'].median():.4f} [{df['mean_abs_diff_normalized'].quantile(0.25):.4f}, {df['mean_abs_diff_normalized'].quantile(0.75):.4f}]
Range: [{df['mean_abs_diff_normalized'].min():.4f}, {df['mean_abs_diff_normalized'].max():.4f}]

RELATIVE CHANGES:
Mean ± SD: {df['relative_change_percent'].mean():.2f}% ± {df['relative_change_percent'].std():.2f}%
Median: {df['relative_change_percent'].median():.2f}%

STATISTICAL TESTS:
Cohen's d: {(df['mean_abs_diff_normalized'].mean() / df['mean_abs_diff_normalized'].std()):.3f}
CV: {(df['mean_abs_diff_normalized'].std() / df['mean_abs_diff_normalized'].mean()):.3f}

CHANGE CATEGORIES:
Small (<5%): {small_changes} ({(small_changes/len(df)*100):.1f}%)
Medium (5-15%): {medium_changes} ({(medium_changes/len(df)*100):.1f}%)
Large (≥15%): {large_changes} ({(large_changes/len(df)*100):.1f}%)"""
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 9. Top changed images
    plt.subplot(3, 3, 9)
    top_changed = df.nlargest(10, 'relative_change_percent')
    y_pos = np.arange(len(top_changed))
    
    plt.barh(y_pos, top_changed['relative_change_percent'], color='lightcoral', alpha=0.7)
    plt.yticks(y_pos, [f.split('_')[0][:8] + "..." for f in top_changed['filename']], fontsize=8)
    plt.xlabel('Relative Change (%)', fontsize=12)
    plt.title('Top 10 Most Changed Images', fontsize=12)
    print(top_changed['filename'])
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(output_folder, 'research_statistics_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return viz_path

def generate_research_report(stats_dict, df, output_folder):
    """Generate a comprehensive research report"""
    
    report_path = os.path.join(output_folder, 'research_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("RESEARCH EVALUATION REPORT: TLS FINE-TUNED vs ORIGINAL DEPTH ESTIMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"DATASET OVERVIEW:\n")
        f.write(f"Total test images: {stats_dict['sample_size']}\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESEARCH FINDINGS:\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("1. MODEL DIFFERENCE ANALYSIS:\n")
        f.write(f"   Mean normalized difference: {stats_dict['mean_normalized_difference']:.4f} ± {stats_dict['std_normalized_difference']:.4f}\n")
        f.write(f"   Median normalized difference: {stats_dict['median_normalized_difference']:.4f}\n")
        f.write(f"   Range: [{stats_dict['min_normalized_difference']:.4f}, {stats_dict['max_normalized_difference']:.4f}]\n")
        f.write(f"   Interquartile range: [{stats_dict['q25_normalized_difference']:.4f}, {stats_dict['q75_normalized_difference']:.4f}]\n\n")
        
        f.write("2. RELATIVE CHANGE ANALYSIS:\n")
        f.write(f"   Mean relative change: {stats_dict['mean_relative_change']:.2f}% ± {stats_dict['std_relative_change']:.2f}%\n")
        f.write(f"   Median relative change: {stats_dict['median_relative_change']:.2f}%\n\n")
        
        f.write("3. STATISTICAL SIGNIFICANCE:\n")
        f.write(f"   t-statistic: {stats_dict['ttest_statistic']:.4f}\n")
        f.write(f"   p-value: {stats_dict['ttest_pvalue']:.2e}\n")
        f.write(f"   Effect size (Cohen's d): {stats_dict['cohens_d']:.4f}\n")
        
        # Interpret effect size
        if abs(stats_dict['cohens_d']) < 0.2:
            effect_interpretation = "negligible"
        elif abs(stats_dict['cohens_d']) < 0.5:
            effect_interpretation = "small"
        elif abs(stats_dict['cohens_d']) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        f.write(f"   Effect size interpretation: {effect_interpretation}\n\n")
        
        f.write("4. CONSISTENCY ANALYSIS:\n")
        f.write(f"   Coefficient of variation: {stats_dict['coefficient_of_variation']:.4f}\n")
        consistency = "high" if stats_dict['coefficient_of_variation'] < 0.3 else "moderate" if stats_dict['coefficient_of_variation'] < 0.6 else "low"
        f.write(f"   Consistency level: {consistency}\n\n")
        
        f.write("5. CHANGE MAGNITUDE DISTRIBUTION:\n")
        f.write(f"   Small changes (<5%): {stats_dict['small_changes_count']} images ({stats_dict['small_changes_percent']:.1f}%)\n")
        f.write(f"   Medium changes (5-15%): {stats_dict['medium_changes_count']} images ({stats_dict['medium_changes_percent']:.1f}%)\n")
        f.write(f"   Large changes (≥15%): {stats_dict['large_changes_count']} images ({stats_dict['large_changes_percent']:.1f}%)\n\n")
        
        f.write("6. DEPTH RANGE ANALYSIS:\n")
        f.write(f"   Mean original depth range: {stats_dict['mean_original_range']:.2f}\n")
        f.write(f"   Mean fine-tuned depth range: {stats_dict['mean_finetuned_range']:.2f}\n")
        f.write(f"   Mean range ratio: {stats_dict['mean_range_ratio']:.4f} ± {stats_dict['std_range_ratio']:.4f}\n\n")
        
        f.write("RESEARCH CONCLUSIONS:\n")
        f.write("-" * 40 + "\n\n")
        
        if stats_dict['mean_relative_change'] > 10:
            f.write(" SIGNIFICANT MODEL DIFFERENCES: The fine-tuned model shows substantial\n")
            f.write("  deviations from the original model, indicating meaningful improvements\n")
            f.write("  from TLS guidance.\n\n")
        elif stats_dict['mean_relative_change'] > 5:
            f.write(" MODERATE MODEL DIFFERENCES: The fine-tuned model shows moderate\n")
            f.write("  improvements while maintaining similarity to the original.\n\n")
        else:
            f.write(" MINIMAL MODEL DIFFERENCES: The fine-tuned model shows subtle changes\n")
            f.write("  that may require closer examination for practical significance.\n\n")
        
        if stats_dict['ttest_pvalue'] < 0.001:
            f.write(" HIGHLY SIGNIFICANT: Changes are statistically significant (p < 0.001)\n\n")
        elif stats_dict['ttest_pvalue'] < 0.05:
            f.write(" SIGNIFICANT: Changes are statistically significant (p < 0.05)\n\n")
        else:
            f.write(" NOT SIGNIFICANT: Changes are not statistically significant\n\n")
        
        if effect_interpretation in ["medium", "large"]:
            f.write(" PRACTICAL SIGNIFICANCE: Effect size indicates meaningful practical impact\n\n")
        else:
            f.write(" LIMITED PRACTICAL IMPACT: Small effect size suggests limited practical significance\n\n")

        f.write("-" * 40 + "\n")
        f.write(f"Key statistics for paper:\n")
        f.write(f"- Sample size: N = {stats_dict['sample_size']}\n")
        f.write(f"- Mean difference: {stats_dict['mean_normalized_difference']:.3f} ± {stats_dict['std_normalized_difference']:.3f}\n")
        f.write(f"- Effect size: d = {stats_dict['cohens_d']:.3f}\n")
        f.write("- Statistical significance: t = {:.2f}, p {}\n".format(
            stats_dict['ttest_statistic'],
            '<0.001' if stats_dict['ttest_pvalue'] < 0.001 else f"= {stats_dict['ttest_pvalue']:.3f}"
        ))
        f.write(f"- Images with substantial changes (≥5%): {stats_dict['medium_changes_percent'] + stats_dict['large_changes_percent']:.1f}%\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Generate Research Statistics from Normalized Inference Results')
    parser.add_argument('--results_folder', type=str, required=True,
                        help='Folder containing normalization parameter JSON files')
    parser.add_argument('--output', type=str, default='research_analysis',
                        help='Output folder for analysis results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_folder):
        print(f"Error: Results folder not found: {args.results_folder}")
        return
    
    print("=" * 70)
    print("📊 RESEARCH STATISTICS ANALYSIS FOR DEPTH ESTIMATION EVALUATION")
    print("=" * 70)
    print(f"Results folder: {args.results_folder}")
    print(f"Output folder: {args.output}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("Loading normalization data...")
    df = load_normalization_data(args.results_folder)
    
    if df is None:
        return
    
    print(f"Loaded data for {len(df)} images")
    
    # Compute statistics
    print("Computing research statistics...")
    stats_dict = compute_research_statistics(df)
    
    # Create visualizations
    print("Creating research visualizations...")
    viz_path = create_research_visualizations(df, args.output)
    
    # Generate report
    print("Generating research report...")
    report_path = generate_research_report(stats_dict, df, args.output)
    
    # Save detailed statistics as JSON
    stats_path = os.path.join(args.output, 'detailed_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2, default=str)
    
    # Save processed data as CSV
    csv_path = os.path.join(args.output, 'processed_results.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n Analysis complete!")
    print(f" Results saved to: {args.output}")
    print(f" Visualization: {os.path.basename(viz_path)}")
    print(f" Report: {os.path.basename(report_path)}")
    print(f" Statistics: detailed_statistics.json")
    print(f" Data: processed_results.csv")
    
    # Print key findings
    print(f"\n KEY FINDINGS:")
    print(f"   Mean normalized difference: {stats_dict['mean_normalized_difference']:.4f}")
    print(f"   Mean relative change: {stats_dict['mean_relative_change']:.2f}%")
    print(f"   Effect size (Cohen's d): {stats_dict['cohens_d']:.3f}")
    print(f"   Statistical significance: p = {stats_dict['ttest_pvalue']:.2e}")

if __name__ == "__main__":
    main()