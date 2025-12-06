import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import Counter

# Path to the zip file
zip_path = 'dataset/Digits Kaggle.zip'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zf:
    # Get list of all files
    file_list = zf.namelist()
    print(f"Total files in zip: {len(file_list)}")

    # Filter JPEG files
    jpeg_files = [f for f in file_list if f.endswith('.jpeg')]
    print(f"JPEG files: {len(jpeg_files)}")

    # Extract labels from folder names (digits 0-9)
    labels = []
    for f in jpeg_files:
        # Path like 'digits_jpeg/digits_jpeg/0/img001-00001.jpeg'
        parts = f.split('/')
        if len(parts) >= 3:
            try:
                label = int(parts[2])  # The digit folder
                labels.append(label)
            except ValueError:
                pass

    # Class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for digit in sorted(label_counts.keys()):
        print(f"Digit {digit}: {label_counts[digit]} samples")

    # Load a few sample images
    sample_files = jpeg_files[:10]  # First 10 images
    sample_images = []
    sample_labels = []

    for f in sample_files:
        with zf.open(f) as file:
            img = Image.open(file)
            img_array = np.array(img)
            sample_images.append(img_array)
            # Extract label
            parts = f.split('/')
            if len(parts) >= 3:
                try:
                    label = int(parts[2])
                    sample_labels.append(label)
                except ValueError:
                    sample_labels.append(-1)

    print(f"\nLoaded {len(sample_images)} sample images")

    # Analyze image properties
    if sample_images:
        shapes = [img.shape for img in sample_images]
        print(f"Image shapes: {shapes}")
        print(f"All same shape: {len(set(shapes)) == 1}")

        # Pixel value statistics
        all_pixels = np.concatenate([img.flatten() for img in sample_images])
        print("\nPixel statistics (from samples):")
        print(f"Mean: {np.mean(all_pixels):.2f}")
        print(f"Std: {np.std(all_pixels):.2f}")
        print(f"Min: {np.min(all_pixels)}")
        print(f"Max: {np.max(all_pixels)}")

        # Visualize sample images
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        for i, (ax, img, label) in enumerate(zip(axes.flat, sample_images, sample_labels)):
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('kaggle_digits_samples.png')
        plt.show()

    # Overall dataset info
    print("\nDataset Overview:")
    print(f"Total images: {len(jpeg_files)}")
    print(f"Classes: {sorted(label_counts.keys())}")
    print(f"Images per class: {dict(label_counts)}")

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    digits = sorted(label_counts.keys())
    counts = [label_counts[d] for d in digits]
    plt.bar(digits, counts)
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Class Distribution (Kaggle Digits)')
    plt.xticks(digits)
    plt.savefig('kaggle_class_distribution.png')
    plt.show()