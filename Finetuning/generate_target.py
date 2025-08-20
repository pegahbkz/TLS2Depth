import torch
import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from depth_anything_v2.dpt import DepthAnythingV2

def load_original_model(model_path, device):
    """Load the original DepthAnything V2 model"""
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**model_configs['vitl'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device).eval()
    return model

def create_research_tls_target(original_pred, tls_depth, tls_mask, verbose=True):
    """
    Recreate the exact target generation from research script with detailed logging
    """
    print(f"\n TARGET GENERATION PROCESS")
    print(f"=" * 50)
    
    # Check TLS coverage
    total_pixels = tls_mask.size
    valid_tls_pixels = np.sum(tls_mask > 0)
    tls_coverage_percent = (valid_tls_pixels / total_pixels) * 100
    
    print(f" TLS Coverage: {valid_tls_pixels:,}/{total_pixels:,} pixels ({tls_coverage_percent:.1f}%)")
    
    # Return original if insufficient TLS data
    if tls_coverage_percent < 1.0 or np.sum(tls_mask) < 5:
        print(f" Insufficient TLS coverage - returning original")
        return original_pred, {
            'coverage_percent': tls_coverage_percent,
            'sufficient_data': False,
            'adjustment_applied': False
        }
    
    valid_tls = tls_depth[tls_mask > 0]
    valid_original = original_pred[tls_mask > 0]

    if len(valid_tls) < 5:
        print(f" Insufficient valid TLS pixels - returning original")
        return original_pred, {
            'coverage_percent': tls_coverage_percent,
            'sufficient_data': False,
            'adjustment_applied': False
        }
    
    # Calculate correlation
    correlation = np.corrcoef(valid_tls, valid_original)[0, 1]
    print(f" TLS-Original Correlation: {correlation:.3f}")
    print(f" TLS Range: [{valid_tls.min():.2f}, {valid_tls.max():.2f}] (mean: {valid_tls.mean():.2f})")
    print(f" Original Range: [{valid_original.min():.2f}, {valid_original.max():.2f}] (mean: {valid_original.mean():.2f})")
    
    # Check correlation threshold
    if np.isnan(correlation) or abs(correlation) <= 0.03:
        print(f" Correlation too weak ({correlation:.3f}) - returning original")
        return original_pred, {
            'coverage_percent': tls_coverage_percent,
            'sufficient_data': True,
            'correlation': correlation,
            'adjustment_applied': False
        }
    
    print(f" Correlation sufficient - proceeding with target generation")
    
    # Create adjustment map
    adjustment_map = np.zeros_like(original_pred)
    
    # Normalize both to 0-1 for comparison
    tls_norm = (tls_depth - valid_tls.min()) / (valid_tls.max() - valid_tls.min() + 1e-8)
    orig_norm = (original_pred - valid_original.min()) / (valid_original.max() - valid_original.min() + 1e-8)

    print(f" Normalization complete")
    print(f"   TLS normalized range: [{tls_norm.min():.3f}, {tls_norm.max():.3f}]")
    print(f"   Original normalized range: [{orig_norm.min():.3f}, {orig_norm.max():.3f}]")
    
    # Calculate adjustment scale (35% of range)
    original_range = valid_original.max() - valid_original.min()
    max_adjustment = original_range * 0.35  # Research-grade visibility
    
    print(f"🎚 Adjustment Scale: {max_adjustment:.3f} (35% of range {original_range:.3f})")
    
    # Handle inverted correlation
    if correlation < 0:
        print(f" Negative correlation detected - inverting TLS normalization")
        tls_norm = 1.0 - tls_norm
    
    # Calculate difference and create adjustment
    diff = tls_norm - orig_norm
    adjustment_map = diff * max_adjustment
    
    # Apply 90% blending in TLS regions
    adjustment_map = adjustment_map * tls_mask * 0.9
    
    # Get adjustment statistics
    adj_in_tls = adjustment_map[tls_mask > 0]
    print(f"🎯 Adjustments in TLS regions:")
    print(f"   Range: [{adj_in_tls.min():.3f}, {adj_in_tls.max():.3f}]")
    print(f"   Mean: {adj_in_tls.mean():.3f}")
    print(f"   Magnitude: {np.mean(np.abs(adj_in_tls)):.3f}")
    
    # Apply adjustments (50% original + 50% TLS guidance)
    research_target = original_pred + adjustment_map
    
    final_diff = research_target - original_pred
    print(f"🏁 Final Target:")
    print(f"   Range: [{research_target.min():.2f}, {research_target.max():.2f}]")
    print(f"   Difference from original: [{final_diff.min():.2f}, {final_diff.max():.2f}]")
    print(f"   Mean absolute difference: {np.mean(np.abs(final_diff)):.2f}")
    
    return research_target, {
        'coverage_percent': tls_coverage_percent,
        'sufficient_data': True,
        'correlation': correlation,
        'adjustment_applied': True,
        'max_adjustment': max_adjustment,
        'mean_adjustment': adj_in_tls.mean(),
        'adjustment_magnitude': np.mean(np.abs(adj_in_tls)),
        'final_mean_diff': np.mean(np.abs(final_diff))
    }

def visualize_target_generation(rgb_path, depth_path, original_model, save_path):
    """Visualize the complete target generation process with enhanced visual clarity"""
    
    print(f"\n🔍 ANALYZING TARGET GENERATION")
    print(f"RGB: {os.path.basename(rgb_path)}")
    print(f"TLS: {os.path.basename(depth_path)}")
    
    # Load RGB image
    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Load TLS depth
    if depth_path.endswith('.npy'):
        tls_depth = np.load(depth_path).astype(np.float32)
    else:
        tls_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    if len(tls_depth.shape) == 3:
        tls_depth = tls_depth[:, :, 0]
    
    # Clean TLS depth
    tls_depth = np.clip(tls_depth, 0, 200.0)
    
    # Create TLS mask
    tls_mask = (tls_depth > 0.5).astype(np.float32)
    
    # Minimal noise filtering
    kernel = np.ones((2,2), np.uint8)
    tls_mask = cv2.morphologyEx(tls_mask, cv2.MORPH_OPEN, kernel)
    
    # Get original model prediction
    device = next(original_model.parameters()).device
    with torch.no_grad():
        rgb_resized = cv2.resize(rgb_image, (518, 518))
        rgb_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
        
        original_pred = original_model(rgb_tensor).squeeze().cpu().numpy()
        original_pred = cv2.resize(original_pred, (tls_depth.shape[1], tls_depth.shape[0]))
    
    # Generate research target with detailed logging
    research_target, stats = create_research_tls_target(original_pred, tls_depth, tls_mask)
    
    # Create comprehensive visualization with better layout
    fig = plt.figure(figsize=(24, 18))
    
    # Top row: Input data and basic processing
    # RGB Image
    plt.subplot(4, 6, 1)
    plt.imshow(rgb_image)
    plt.title('Input RGB Image', fontsize=12)
    plt.axis('off')
    
    # TLS Coverage Map with transparency
    plt.subplot(4, 6, 2)
    # Show RGB as background with TLS overlay
    plt.imshow(rgb_image, alpha=0.7)
    tls_overlay = np.zeros((*tls_mask.shape, 4))
    tls_overlay[:, :, 1] = tls_mask  # Green channel
    tls_overlay[:, :, 3] = tls_mask * 0.6  # Alpha channel
    plt.imshow(tls_overlay)
    plt.title(f'TLS Coverage\n{stats["coverage_percent"]:.1f}% coverage', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Raw TLS Depth
    plt.subplot(4, 6, 3)
    tls_masked = np.where(tls_mask > 0, tls_depth, np.nan)
    im1 = plt.imshow(tls_masked, cmap='viridis')
    plt.title(f'Raw TLS Depth\nRange: [{np.nanmin(tls_masked):.1f}, {np.nanmax(tls_masked):.1f}]', 
              fontsize=12)
    plt.axis('off')
    plt.colorbar(im1, shrink=0.6)
    
    # Original Prediction
    plt.subplot(4, 6, 4)
    im2 = plt.imshow(original_pred, cmap='viridis')
    plt.title(f'Original DepthAnything V2\nRange: [{original_pred.min():.1f}, {original_pred.max():.1f}]', 
              fontsize=12)
    plt.axis('off')
    plt.colorbar(im2, shrink=0.6)
    
    # Step-by-step process visualization
    if stats['adjustment_applied']:
        # Show normalized TLS (before potential inversion)
        valid_tls = tls_depth[tls_mask > 0]
        tls_norm_raw = (tls_depth - valid_tls.min()) / (valid_tls.max() - valid_tls.min() + 1e-8)
        
        plt.subplot(4, 6, 5)
        tls_norm_show = np.where(tls_mask > 0, tls_norm_raw, np.nan)
        im3 = plt.imshow(tls_norm_show, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'TLS Normalized [0,1]\nCorr: {stats["correlation"]:.3f}', fontsize=12)
        plt.axis('off')
        plt.colorbar(im3, shrink=0.6)
        
        # Show TLS after correlation handling (inverted if needed)
        if stats['correlation'] < 0:
            tls_norm_final = 1.0 - tls_norm_raw
            inversion_text = "TLS After Inversion\n(Correlation < 0)"
        else:
            tls_norm_final = tls_norm_raw
            inversion_text = "TLS Normalized\n(No inversion needed)"
        
        plt.subplot(4, 6, 6)
        tls_final_show = np.where(tls_mask > 0, tls_norm_final, np.nan)
        im4 = plt.imshow(tls_final_show, cmap='viridis', vmin=0, vmax=1)
        plt.title(inversion_text, fontsize=12)
        plt.axis('off')
        plt.colorbar(im4, shrink=0.6)
        
        # Row 2: Normalization and difference computation
        # Original normalized
        valid_original = original_pred[tls_mask > 0]
        orig_norm = (original_pred - valid_original.min()) / (valid_original.max() - valid_original.min() + 1e-8)
        
        plt.subplot(4, 6, 7)
        orig_norm_show = np.where(tls_mask > 0, orig_norm, np.nan)
        im5 = plt.imshow(orig_norm_show, cmap='viridis', vmin=0, vmax=1)
        plt.title('Original Normalized [0,1]\n(in TLS regions)', fontsize=12)
        plt.axis('off')
        plt.colorbar(im5, shrink=0.6)
        
        # Difference map (normalized space)
        diff_norm = tls_norm_final - orig_norm
        plt.subplot(4, 6, 8)
        diff_norm_show = np.where(tls_mask > 0, diff_norm, 0)
        max_diff = max(abs(diff_norm_show.min()), abs(diff_norm_show.max()))
        im6 = plt.imshow(diff_norm_show, cmap='viridis', vmin=-max_diff, vmax=max_diff)
        plt.title('Normalized Difference\n(TLS - Original)', fontsize=12)
        plt.axis('off')
        plt.colorbar(im6, shrink=0.6)
        
        # Scaled adjustment map
        original_range = valid_original.max() - valid_original.min()
        max_adjustment = original_range * 0.35
        adjustment_map = diff_norm * max_adjustment * tls_mask * 0.9
        
        plt.subplot(4, 6, 9)
        im7 = plt.imshow(adjustment_map, cmap='viridis')
        plt.title(f'Scaled Adjustments\n±{max_adjustment:.1f} max (35% × 90%)', fontsize=12)
        plt.axis('off')
        plt.colorbar(im7, shrink=0.6)
        
    else:
        # Show placeholders when no adjustment applied
        for i in range(5, 10):
            plt.subplot(4, 6, i)
            plt.text(0.5, 0.5, f'Step {i}\nNo adjustment\napplied', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            plt.title(f'{i}. Processing Step', fontsize=12)
            plt.axis('off')
    
    # Row 3: Final results
    # Generated Target
    plt.subplot(4, 6, 13)
    im8 = plt.imshow(research_target, cmap='viridis')
    target_title = 'Generated Target\n'
    if stats['adjustment_applied']:
        target_title += f'Mean diff: {stats["final_mean_diff"]:.2f}'
    else:
        target_title += 'No adjustment applied'
    plt.title(target_title, fontsize=12)
    plt.axis('off')
    plt.colorbar(im8, shrink=0.6)
    
    # Target - Original Difference
    plt.subplot(4, 6, 14)
    diff = research_target - original_pred
    max_abs_diff = max(abs(diff.min()), abs(diff.max())) if diff.max() != diff.min() else 1
    im9 = plt.imshow(diff, cmap='viridis', vmin=-max_abs_diff, vmax=max_abs_diff)
    plt.title(f'Final Difference\nRange: [{diff.min():.2f}, {diff.max():.2f}]', 
              fontsize=12)
    plt.axis('off')
    plt.colorbar(im9, shrink=0.6)
    
    # Adjustment Magnitude Map
    plt.subplot(4, 6, 15)
    if stats['adjustment_applied']:
        adj_magnitude = np.abs(diff) * tls_mask  # Only show adjustments in TLS regions
        im10 = plt.imshow(adj_magnitude, cmap='hot')
        plt.title(f'Adjustment Magnitude\nMax: {stats["adjustment_magnitude"]:.2f}', 
                  fontsize=12)
        plt.colorbar(im10, shrink=0.6)
    else:
        plt.imshow(np.zeros_like(original_pred), cmap='hot')
        plt.title('Adjustment Magnitude\n(None applied)', fontsize=12)
    plt.axis('off')
    
    # Row 4: Analysis plots and summaries
    # TLS vs Original Scatter (with lighter opacity)
    plt.subplot(4, 6, 16)
    if stats['adjustment_applied']:
        valid_tls = tls_depth[tls_mask > 0]
        valid_original = original_pred[tls_mask > 0]
        # Subsample for better visibility
        n_points = min(3000, len(valid_tls))
        indices = np.random.choice(len(valid_tls), n_points, replace=False)
        plt.scatter(valid_tls[indices], valid_original[indices], alpha=0.6, s=20, c='lightcoral', edgecolors='blue', linewidths=0.1)
        plt.xlabel('TLS Depth', fontsize=12)
        plt.ylabel('Original Prediction', fontsize=12)
        plt.title(f'TLS vs Original\nCorr: {stats["correlation"]:.3f}', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(valid_tls[indices], valid_original[indices], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_tls[indices].min(), valid_tls[indices].max(), 100)
        plt.plot(x_line, p(x_line), "gray", linestyle='--', alpha=0.8, linewidth=1, label=f'Trend (r={stats["correlation"]:.3f})')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Insufficient\nTLS Data', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
        plt.title('TLS vs Original', fontsize=12)
    
    # Target vs Original Scatter (with lighter opacity)
    plt.subplot(4, 6, 17)
    if stats['adjustment_applied']:
        valid_target = research_target[tls_mask > 0]
        valid_original = original_pred[tls_mask > 0]
        # Subsample for better visibility
        indices = np.random.choice(len(valid_target), n_points, replace=False)
        plt.scatter(valid_target[indices], valid_original[indices], alpha=0.6, s=20, color='lightgreen', edgecolors='darkgreen', linewidths=0.1)
        min_val = min(valid_target[indices].min(), valid_original[indices].min())
        max_val = max(valid_target[indices].max(), valid_original[indices].max())
        plt.plot([min_val, max_val], [min_val, max_val], "gray", linestyle='--', alpha=0.8, linewidth=1, label='Perfect match')
        plt.xlabel('Target Depth', fontsize=12)
        plt.ylabel('Original Prediction', fontsize=12)
        plt.title('Target vs Original\n(in TLS regions)', fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Target\nGenerated', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
        plt.title('Target vs Original', fontsize=12)
    
    # Statistics Summary
    plt.subplot(4, 6, 18)
    plt.axis('off')
    stats_text = f"""TARGET GENERATION STATS
    
TLS Coverage: {stats['coverage_percent']:.1f}%
Sufficient Data: {stats['sufficient_data']}
Adjustment Applied: {stats['adjustment_applied']}

"""
    if stats['adjustment_applied']:
        stats_text += f"""Correlation: {stats['correlation']:.3f}
Max Adjustment: {stats['max_adjustment']:.2f}
Mean Adjustment: {stats['mean_adjustment']:.2f}
Adjustment Magnitude: {stats['adjustment_magnitude']:.2f}
Final Mean Diff: {stats['final_mean_diff']:.2f}

PROCESS SUMMARY:
1. Check TLS coverage (>1%)
2. Calculate correlation
3. Normalize both to [0,1]
4. Invert TLS if corr < 0
5. Compute difference
6. Scale by 35% of range
7. Apply 90% blending
8. Generate final target"""
    else:
        stats_text += "\nREASON FOR NO ADJUSTMENT:\n"
        if stats['coverage_percent'] < 1.0:
            stats_text += "• Insufficient TLS coverage"
        elif not stats['sufficient_data']:
            stats_text += "• Too few valid TLS pixels"
        elif abs(stats.get('correlation', 0)) <= 0.03:
            stats_text += "• Correlation too weak"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    # Create a visual flow diagram
    # Process Flow Diagram - make it more visual
    plt.subplot(4, 6, (10, 12))  # Span multiple cells
    plt.axis('off')
    plt.xlim(0, 10)
    plt.ylim(0, 6)
    
    # Draw process flow boxes
    boxes = [
        (1, 5, "RGB +\nTLS"),
        (3, 5, "Create\nMask"),
        (5, 5, "Check\nCoverage"),
        (7, 5, "Calculate\nCorrelation"),
        (1, 3, "Normalize\nto [0,1]"),
        (3, 3, "Handle\nInversion"),
        (5, 3, "Compute\nDifference"),
        (7, 3, "Scale\nAdjustment"),
        (9, 3, "Apply\nBlending"),
        (5, 1, "Final\nTarget")
    ]
    
    for x, y, text in boxes:
        # Color code based on whether step was applied
        if stats['adjustment_applied']:
            color = 'lightgreen'
        else:
            color = 'lightcoral' if "Check" in text or "Calculate" in text else 'lightgray'
        
        plt.text(x, y, text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    # Draw arrows
    arrows = [
        (1.5, 5, 1, 0),   # RGB -> Mask
        (3.5, 5, 1, 0),   # Mask -> Coverage
        (5.5, 5, 1, 0),   # Coverage -> Correlation
        (7, 4.5, 0, -1),  # Correlation -> down
        (6.5, 3, -1, 0),  # -> Normalize
        (1.5, 3, 1, 0),   # Normalize -> Inversion
        (3.5, 3, 1, 0),   # Inversion -> Difference
        (5.5, 3, 1, 0),   # Difference -> Scale
        (7.5, 3, 1, 0),   # Scale -> Blending
        (8, 2.5, -2, -1), # Blending -> Target
    ]
    
    for x, y, dx, dy in arrows:
        plt.arrow(x, y, dx*0.3, dy*0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.title('VISUAL PROCESS FLOW', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 Enhanced target generation visualization saved: {save_path}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Visualize TLS Target Generation Process')
    parser.add_argument('--rgb', type=str, required=True,
                        help='Path to RGB image')
    parser.add_argument('--depth', type=str, required=True,
                        help='Path to TLS depth file (.npy or image)')
    parser.add_argument('--model', type=str, default='../checkpoints/depth_anything_v2_vitl.pth',
                        help='Path to original DepthAnything V2 model')
    parser.add_argument('--output', type=str, default='target_generation_analysis.png',
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.rgb):
        print(f"Error: RGB image not found: {args.rgb}")
        return
    
    if not os.path.exists(args.depth):
        print(f"Error: Depth file not found: {args.depth}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print("="*70)
    print("🎯 TLS TARGET GENERATION ANALYSIS")
    print("="*70)
    print(f"RGB Image: {args.rgb}")
    print(f"TLS Depth: {args.depth}")
    print(f"Original Model: {args.model}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Load original model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading original model on {device}...")
    original_model = load_original_model(args.model, device)
    
    # Analyze target generation
    stats = visualize_target_generation(args.rgb, args.depth, original_model, args.output)
    
    print(f"\n🎉 Analysis complete!")
    print(f"Target generation successful: {stats['adjustment_applied']}")
    if stats['adjustment_applied']:
        print(f"Final improvement magnitude: {stats['final_mean_diff']:.2f}")

if __name__ == "__main__":
    main()