import torch
import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add parent directory to path for depth_anything_v2 import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from depth_anything_v2.dpt import DepthAnythingV2

def load_finetuned_model(model_path, device):
    """Load the fine-tuned model"""
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Initialize model
    model = DepthAnythingV2(**model_configs['vitl'])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded fine-tuned model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            print(f"Model stats: {metrics.get('relative_deviation_percent', 0):.1f}% deviation, "
                  f"{metrics.get('tls_improvement_percent', 0):.1f}% TLS improvement")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model

def predict_depth(model, image_path, input_size=518):
    """Predict depth for a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)
    
    # Predict
    with torch.no_grad():
        depth = model(image_tensor).squeeze().cpu().numpy()
    
    # Resize back to original dimensions
    depth = cv2.resize(depth, (original_shape[1], original_shape[0]))
    
    return image_rgb, depth

def normalize_depth(depth_map):
    """Normalize depth map to [0, 1] range"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
    if depth_max == depth_min:
        return np.zeros_like(depth_map), depth_min, depth_max
    
    normalized = (depth_map - depth_min) / (depth_max - depth_min)
    return normalized, depth_min, depth_max

def get_image_files(folder_path, extensions=None):
    """Get all image files from folder"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    return sorted(image_files)

def load_original_model(model_path, device):
    """Load the original DepthAnything V2 model"""
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**model_configs['vitl'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device).eval()
    print("Loaded original DepthAnything V2 model")
    return model

def run_inference_on_folder(finetuned_model_path, original_model_path, image_folder, output_folder, num_samples=None):
    """Run inference comparing original vs fine-tuned models with normalized depths"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load both models
    print("Loading models...")
    finetuned_model = load_finetuned_model(finetuned_model_path, device)
    original_model = load_original_model(original_model_path, device)
    
    # Get image files
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    if num_samples:
        image_files = image_files[:num_samples]
    
    print(f"Processing {len(image_files)} images...")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Get predictions from both models
            rgb_image, original_depth_raw = predict_depth(original_model, image_path)
            _, finetuned_depth_raw = predict_depth(finetuned_model, image_path)
            
            # Normalize both depth maps independently
            original_depth, orig_min, orig_max = normalize_depth(original_depth_raw)
            finetuned_depth, ft_min, ft_max = normalize_depth(finetuned_depth_raw)
            
            # Calculate difference between normalized depths
            depth_diff = finetuned_depth - original_depth
            
            # Create comparison visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # RGB image
            axes[0, 0].imshow(rgb_image)
            axes[0, 0].set_title('Input Image', fontsize=12)
            axes[0, 0].axis('off')
            
            # Original depth prediction (normalized)
            im1 = axes[0, 1].imshow(original_depth, cmap='viridis', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Original DepthAnything V2 (Normalized)\nRaw range: [{orig_min:.1f}, {orig_max:.1f}]', 
                                fontsize=12)
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
            
            # Fine-tuned depth prediction (normalized)
            im2 = axes[1, 0].imshow(finetuned_depth, cmap='viridis', vmin=0, vmax=1)
            axes[1, 0].set_title(f'TLS Fine-tuned Model (Normalized)\nRaw range: [{ft_min:.1f}, {ft_max:.1f}]', 
                                fontsize=12)
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
            
            # Difference map (normalized difference)
            max_abs_diff = max(abs(depth_diff.min()), abs(depth_diff.max())) if depth_diff.max() != depth_diff.min() else 1
            im3 = axes[1, 1].imshow(depth_diff, cmap='RdGy', vmin=-max_abs_diff, vmax=max_abs_diff)
            
            # Calculate statistics on normalized depths
            mean_abs_diff = np.mean(np.abs(depth_diff))
            relative_change = mean_abs_diff * 100  # Already normalized, so this is percentage
            
            axes[1, 1].set_title(f'Normalized Difference (Fine-tuned - Original)\nMean |Δ|: {mean_abs_diff:.3f} ({relative_change:.1f}%)\nRange: [{depth_diff.min():.3f}, {depth_diff.max():.3f}]', 
                                fontsize=12)
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
            
            plt.tight_layout()
            
            # Save result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_folder, f'{base_name}_comparison_normalized.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save depth arrays (both raw and normalized)
            # Raw depths
            original_raw_path = os.path.join(output_folder, f'{base_name}_original_depth_raw.npy')
            finetuned_raw_path = os.path.join(output_folder, f'{base_name}_finetuned_depth_raw.npy')
            np.save(original_raw_path, original_depth_raw)
            np.save(finetuned_raw_path, finetuned_depth_raw)
            
            # Normalized depths
            original_norm_path = os.path.join(output_folder, f'{base_name}_original_depth_normalized.npy')
            finetuned_norm_path = os.path.join(output_folder, f'{base_name}_finetuned_depth_normalized.npy')
            np.save(original_norm_path, original_depth)
            np.save(finetuned_norm_path, finetuned_depth)
            
            # Normalized difference
            diff_path = os.path.join(output_folder, f'{base_name}_normalized_difference.npy')
            np.save(diff_path, depth_diff)
            
            # Save normalization parameters for reference
            norm_params = {
                'original_min': float(orig_min),
                'original_max': float(orig_max),
                'finetuned_min': float(ft_min),
                'finetuned_max': float(ft_max),
                'mean_abs_diff_normalized': float(mean_abs_diff),
                'relative_change_percent': float(relative_change)
            }
            
            import json
            params_path = os.path.join(output_folder, f'{base_name}_normalization_params.json')
            with open(params_path, 'w') as f:
                json.dump(norm_params, f, indent=2)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nInference complete! Results saved to: {output_folder}")
    print("Files saved:")
    print("  - *_comparison_normalized.png: 4-panel comparison with normalized depths")
    print("  - *_original_depth_raw.npy: Original model raw depth arrays")
    print("  - *_finetuned_depth_raw.npy: Fine-tuned model raw depth arrays")
    print("  - *_original_depth_normalized.npy: Original model normalized depth arrays")
    print("  - *_finetuned_depth_normalized.npy: Fine-tuned model normalized depth arrays")
    print("  - *_normalized_difference.npy: Normalized difference arrays")
    print("  - *_normalization_params.json: Normalization parameters and statistics")

def main():
    parser = argparse.ArgumentParser(description='Compare Original vs TLS Fine-tuned Model with Normalized Depths')
    parser.add_argument('--finetuned_model', type=str, required=True,
                        help='Path to fine-tuned model (.pth file)')
    parser.add_argument('--original_model', type=str, default='../checkpoints/depth_anything_v2_vitl.pth',
                        help='Path to original DepthAnything V2 model')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--output', type=str, default='inference_comparison_normalized',
                        help='Output folder for results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.finetuned_model):
        print(f"Error: Fine-tuned model file not found: {args.finetuned_model}")
        return
    
    if not os.path.exists(args.original_model):
        print(f"Error: Original model file not found: {args.original_model}")
        return
    
    if not os.path.exists(args.images):
        print(f"Error: Image folder not found: {args.images}")
        return
    
    print("="*70)
    print("🔍 NORMALIZED DEPTH COMPARISON: ORIGINAL vs TLS FINE-TUNED")
    print("="*70)
    print(f"Fine-tuned Model: {args.finetuned_model}")
    print(f"Original Model: {args.original_model}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    if args.num_samples:
        print(f"Limit: {args.num_samples} images")
    print("\nNormalization: Each depth map normalized to [0,1] independently")
    print("Difference: Computed between normalized depth maps")
    print("="*70)
    
    # Run inference
    run_inference_on_folder(args.finetuned_model, args.original_model, args.images, args.output, args.num_samples)

if __name__ == "__main__":
    main()