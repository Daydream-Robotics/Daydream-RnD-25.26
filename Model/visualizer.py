import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
    print(f"✅ Saved visualization to {save_path}")
