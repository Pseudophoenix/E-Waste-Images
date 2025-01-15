import os
import shutil
import random

def split_dataset(input_dir, output_dir, split_ratio=0.8):
    """
    Splits images into train and test folders in an 8:2 ratio.

    Args:
        input_dir (str): Path to the original dataset folder.
        output_dir (str): Path to the output dataset folder.
        split_ratio (float): Proportion of images to include in the training set.
    """
    # Create train and test directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Iterate through each class folder in the dataset
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue  # Skip non-folder entries
        
        # Create corresponding class folders in train and test directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Shuffle the images
        random.shuffle(images)
        
        # Calculate the split index
        split_index = int(len(images) * split_ratio)
        
        # Split images into train and test sets
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        # Copy images to their respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))
        
        print(f"Class '{class_name}': {len(train_images)} images in train, {len(test_images)} images in test.")

# Define paths
input_dir = "dataset"  # Original dataset directory
output_dir = "output_dataset"  # Output dataset directory with train-test split

# Split dataset into 8:2 ratio
split_dataset(input_dir, output_dir, split_ratio=0.8)

print("Dataset split completed!")
