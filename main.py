# # Check GPU availability
# !nvidia-smi

# # Install required packages
# !pip install ultralytics
# !pip install opencv-python-headless  # headless version for Colab
# !pip install scikit-learn
# !pip install pillow
# !pip install pyyaml

# Import libraries
import os
import json
import yaml
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
# from google.colab import files
import zipfile
from ultralytics import YOLO

# Upload your coco.zip file


# List contents to verify
print("\\nContents extracted:")
for root, dirs, files in os.walk('coco_data'):
    level = root.replace('coco_data', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')


class COCOToYOLOConverter:
    """Convert COCO segmentation format to YOLO segmentation format"""
    
    def __init__(self, coco_json_path, images_dir, output_dir):
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
            
        self.setup_directories()
        
    def setup_directories(self):
        """Create YOLO directory structure"""
        dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def normalize_coordinates(self, segmentation, img_width, img_height):
        """Convert absolute coordinates to normalized YOLO format"""
        normalized = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / img_width
            y = segmentation[i + 1] / img_height
            normalized.extend([x, y])
        return normalized
    
    def convert_to_yolo(self, train_split=0.8):
        """Convert COCO to YOLO format with train/val split"""
        
        # Create image ID to filename mapping
        img_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Split images into train/val
        image_ids = list(annotations_by_image.keys())
        train_ids, val_ids = train_test_split(image_ids, train_size=train_split, random_state=42)
        
        print(f"Converting {len(train_ids)} training images and {len(val_ids)} validation images...")
        
        # Process training set
        self.process_split(train_ids, annotations_by_image, img_id_to_info, 'train')
        
        # Process validation set
        self.process_split(val_ids, annotations_by_image, img_id_to_info, 'val')
        
        # Create classes.txt
        categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        with open(self.output_dir / 'classes.txt', 'w') as f:
            for cat_id in sorted(categories.keys()):
                f.write(f"{categories[cat_id]}\n")
        
        # Create data.yaml
        self.create_data_yaml(categories)
        
        print("✅ Conversion completed!")
        return len(train_ids), len(val_ids)
        
    def process_split(self, image_ids, annotations_by_image, img_id_to_info, split):
        """Process images for a specific split (train/val)"""
        
        for img_id in image_ids:
            img_info = img_id_to_info[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img_path = Path(self.images_dir) / img_filename
            dst_img_path = self.output_dir / split / 'images' / img_filename
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
                
                # Create label file
                label_filename = img_filename.replace('.png', '.txt').replace('.jpg', '.txt')
                label_path = self.output_dir / split / 'labels' / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations_by_image[img_id]:
                        class_id = ann['category_id'] - 1  # YOLO uses 0-based indexing
                        
                        # Handle segmentation
                        if 'segmentation' in ann and ann['segmentation']:
                            for seg in ann['segmentation']:
                                if len(seg) >= 6:  # At least 3 points (6 coordinates)
                                    normalized_seg = self.normalize_coordinates(seg, img_width, img_height)
                                    seg_str = ' '.join([f"{coord:.6f}" for coord in normalized_seg])
                                    f.write(f"{class_id} {seg_str}\n")
    
    def create_data_yaml(self, categories):
        """Create YOLO data configuration file"""
        data_config = {
            'path': 'yolo_dataset',  # Absolute path 
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(categories),
            'names': [categories[i+1] for i in range(len(categories))]
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)

print("✅ Converter class defined!")

print("✅ Converter class defined!")

coco_json_path = 'result_coco.json'
images_dir = 'images'


print(f"📁 Found COCO file: {coco_json_path}")
print(f"📁 Images directory: {images_dir}")

# Convert COCO to YOLO
converter = COCOToYOLOConverter(
    coco_json_path=coco_json_path,
    images_dir=images_dir,
    output_dir='yolo_dataset'
)

train_count, val_count = converter.convert_to_yolo(train_split=0.8)
print(f"✅ Dataset ready: {train_count} train, {val_count} val images")

def train_yolo_segmentation():
    """Train YOLO segmentation model optimized for Colab"""



    # Model configuration
    model = YOLO('yolov8s-seg.pt')  # Pre-trained segmentation model

    # Colab-optimized training parameters
    training_args = {
        'data': 'yolo_dataset/data.yaml',
        'epochs': 200,  # Reduced for Colab time limits
        'imgsz': 640,
        'batch': 4,     # Smaller batch for Colab memory
        'patience': 20,
        'save_period': 10,
        'device': 0,    # GPU
        'workers': 2,   # Reduced workers for Colab
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        # Data augmentation for duct images
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.2,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.1,
        'copy_paste': 0.3,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'project': 'duct_segmentation',  # Project name
        'name': 'yolo_duct_v1',         # Run name
    }

    print("🚀 Starting YOLO segmentation training...")
    print(f"📊 Training parameters: {len(training_args)} settings configured")

    # Start training
    results = model.train(**training_args)

    # Validate the model
    print("📊 Validating model...")
    validation_results = model.val()

    print("✅ Training completed!")
    print(f"📁 Results saved in: /duct_segmentation/yolo_duct_v1/")

    return results, validation_results, model

print("✅ Training function defined!")

# Start the training process
print("🔥 Starting training process...")
results, validation_results, trained_model = train_yolo_segmentation()

print("🎉 Training completed!")
print("📊 Training Results:")
print(f"   - Best mAP: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
print(f"   - Training time: {results.results_dict.get('train/time', 'N/A')}")

# Test the trained model on a sample image
import matplotlib.pyplot as plt

def test_inference():
    """Test the trained model on sample images"""
    
    # Get a sample image from validation set
    val_images_dir = Path('yolo_dataset/val/images')
    sample_images = list(val_images_dir.glob('*.png'))[:3]  # Get first 3 images
    
    if not sample_images:
        sample_images = list(val_images_dir.glob('*.jpg'))[:3]
    
    if sample_images:
        print(f"🖼️ Testing on {len(sample_images)} sample images...")
        
        fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
        if len(sample_images) == 1:
            axes = [axes]
        
        for i, img_path in enumerate(sample_images):
            # Run inference
            results = trained_model(str(img_path))
            
            # Plot results
            result_img = results[0].plot()
            
            axes[i].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Detection Result {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Inference test completed!")
    else:
        print("❌ No sample images found for testing")

# Run inference test
test_inference()

# Create fixed results packaging code
fixed_results_code = '''
# Zip and download the trained model and results
import zipfile
import os
from pathlib import Path
from google.colab import files

def create_results_zip():
    """Create a zip file with training results"""
    
    # Find the actual results directory (it might have a different version number)
    base_dir = '/duct_segmentation'
    results_dir = None
    
    if os.path.exists(base_dir):
        # Find the most recent training run
        run_dirs = [d for d in os.listdir(base_dir) if d.startswith('yolo_duct_v')]
        if run_dirs:
            # Get the most recent run (highest version number)
            latest_run = sorted(run_dirs)[-1]
            results_dir = f'{base_dir}/{latest_run}'
            print(f"📁 Found results directory: {results_dir}")
        else:
            print("❌ No training results found!")
            return None
    else:
        print("❌ Training directory not found!")
        return None
    
    zip_filename = 'yolo_duct_training_results.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add weights
        weights_dir = Path(results_dir) / 'weights'
        if weights_dir.exists():
            print("📦 Adding model weights...")
            for weight_file in weights_dir.glob('*.pt'):
                zipf.write(weight_file, f'weights/{weight_file.name}')
                print(f"   ✅ Added: {weight_file.name}")
        else:
            print("⚠️ No weights directory found")
        
        # Add training plots
        print("📊 Adding training plots...")
        plot_count = 0
        for plot_file in Path(results_dir).glob('*.png'):
            zipf.write(plot_file, plot_file.name)
            print(f"   ✅ Added: {plot_file.name}")
            plot_count += 1
        
        if plot_count == 0:
            print("⚠️ No plot files found")
        
        # Add results CSV
        results_file = Path(results_dir) / 'results.csv'
        if results_file.exists():
            zipf.write(results_file, 'results.csv')
            print("   ✅ Added: results.csv")
        else:
            print("⚠️ No results.csv found")
        
        # Add configuration files
        for config_file in Path(results_dir).glob('*.yaml'):
            zipf.write(config_file, config_file.name)
            print(f"   ✅ Added: {config_file.name}")
    
    print(f"\\n✅ Results packaged in: {zip_filename}")
    return zip_filename

def download_results():
    """Package and download training results"""
    
    print("📦 Creating results package...")
    zip_file = create_results_zip()
    
    if zip_file:
        print("⬇️ Starting download...")
        files.download(zip_file)
        
        print("\\n🎉 Training complete! Results downloaded.")
        print("\\n📋 Summary:")
        print("   ✅ Data converted from COCO to YOLO format")
        print("   ✅ Model trained with optimized parameters")
        print("   ✅ Validation completed")
        print("   ✅ Sample inference tested")
        print("   ✅ Results downloaded")
        
        # Show what's in the zip
        print("\\n📦 Downloaded package contains:")
        print("   - weights/best.pt (best model)")
        print("   - weights/last.pt (last checkpoint)")
        print("   - training plots (loss curves, metrics)")
        print("   - results.csv (training metrics)")
        print("   - configuration files")
    else:
        print("❌ Failed to create results package")

# Run the download
download_results()
'''

# Save the fixed results code
with open('fixed_results_packaging.py', 'w') as f:
    f.write(fixed_results_code)

print("✅ Created fixed results packaging: fixed_results_packaging.py")
print("🔧 Key fixes:")
print("   - Added missing Path import")
print("   - Auto-detects actual results directory")
print("   - Fixed double backslashes in print statements")
print("   - Better error handling and status messages")