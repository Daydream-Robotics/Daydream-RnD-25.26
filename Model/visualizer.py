import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


def visualize_allclass_heatmaps(img, p8_map, p16_map, save_path="/home/agn/ProgramSpace/TensorFlow/Daydream-RnD-25.26/debug_allclass_grid.png"):
    """
    Visualizes all-class P8 and P16 heatmaps overlaid on the original image,
    with gridlines for both scales.

    Args:
        img: (H, W, 3) float32 tensor or NumPy array (range 0–1 or 0–255)
        p8_map: (64, 64, num_classes) heatmap tensor for stride-8 output
        p16_map: (32, 32, num_classes) heatmap tensor for stride-16 output
        save_path: path to save the resulting visualization (default = debug_allclass_grid.png)
    """

    # Ensure NumPy arrays
    img = np.array(img)
    p8_map = np.array(p8_map)
    p16_map = np.array(p16_map)

    # Combine all classes via max reduction
    p8_all = np.max(p8_map, axis=-1)
    p16_all = np.max(p16_map, axis=-1)

    # Create figure
    plt.figure(figsize=(12, 4))

    # --- Raw Image ---
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Raw Image")

    # --- P8 Overlay ---
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(p8_all, alpha=0.5, cmap="jet",
               extent=(0, 512, 512, 0), interpolation="bilinear")

    # Draw P8 gridlines (stride 8)
    for i in range(0, 512, 8):
        plt.axvline(i, color='white', lw=0.4, alpha=0.3)
    for j in range(0, 512, 8):
        plt.axhline(j, color='white', lw=0.4, alpha=0.3)
    plt.title("P8 All-Class Heatmap + Grid")

    # --- P16 Overlay ---
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(p16_all, alpha=0.5, cmap="jet",
               extent=(0, 512, 512, 0), interpolation="bilinear")

    # Draw P16 gridlines (stride 16)
    for i in range(0, 512, 16):
        plt.axvline(i, color='white', lw=0.4, alpha=0.3)
    for j in range(0, 512, 16):
        plt.axhline(j, color='white', lw=0.4, alpha=0.3)
    plt.title("P16 All-Class Heatmap + Grid")

    # Save & close
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_batch_heatmaps(dataset, num_samples=500, output_dir="/home/agn/ProgramSpace/TensorFlow/Daydream-RnD-25.26/visualizations"):
    """
    Processes multiple samples from a dataset and saves visualizations to a folder.

    Args:
        dataset: TensorFlow dataset yielding (images, labels) where labels contains p8 and p16 heatmaps
        num_samples: Number of samples to process (default: 500)
        output_dir: Directory to save all visualizations
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Saving visualizations to: {output_dir}")
    print(f"🎯 Processing {num_samples} samples...")
    
    sample_count = 0
    
    # Iterate through dataset
    for batch_idx, (imgs, labels) in enumerate(dataset):
        batch_size = imgs.shape[0]
        
        # Process each image in the batch
        for i in range(batch_size):
            if sample_count >= num_samples:
                print(f"\n✅ Completed! Generated {sample_count} visualizations.")
                return
            
            img = imgs[i]
            p8_map = labels["p8"][i]
            p16_map = labels["p16"][i]
            
            # Generate filename with zero-padding for proper sorting
            filename = f"sample_{sample_count:05d}.png"
            save_path = output_path / filename
            
            # Create visualization
            visualize_allclass_heatmaps(img, p8_map, p16_map, save_path=str(save_path))
            
            sample_count += 1
            
            # Progress update every 50 samples
            if sample_count % 50 == 0:
                print(f"  Progress: {sample_count}/{num_samples} ({100*sample_count/num_samples:.1f}%)")
    
    print(f"\n✅ Processed all available samples: {sample_count} visualizations generated.")


def visualize_single_batch(imgs, labels, output_dir="/home/agn/ProgramSpace/TensorFlow/Daydream-RnD-25.26/visualizations"):
    """
    Convenience function to visualize all images in a single batch.

    Args:
        imgs: Batch of images (B, H, W, 3)
        labels: Dict with "p8" and "p16" keys containing heatmap batches
        output_dir: Directory to save visualizations
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_size = imgs.shape[0]
    
    for i in range(batch_size):
        img = imgs[i]
        p8_map = labels["p8"][i]
        p16_map = labels["p16"][i]
        
        filename = f"batch_sample_{i:03d}.png"
        save_path = output_path / filename
        
        visualize_allclass_heatmaps(img, p8_map, p16_map, save_path=str(save_path))
    
    print(f"✅ Saved {batch_size} visualizations to {output_dir}")