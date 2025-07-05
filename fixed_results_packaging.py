
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
    
    print(f"\n✅ Results packaged in: {zip_filename}")
    return zip_filename

def download_results():
    """Package and download training results"""
    
    print("📦 Creating results package...")
    zip_file = create_results_zip()
    
    if zip_file:
        print("⬇️ Starting download...")
        files.download(zip_file)
        
        print("\n🎉 Training complete! Results downloaded.")
        print("\n📋 Summary:")
        print("   ✅ Data converted from COCO to YOLO format")
        print("   ✅ Model trained with optimized parameters")
        print("   ✅ Validation completed")
        print("   ✅ Sample inference tested")
        print("   ✅ Results downloaded")
        
        # Show what's in the zip
        print("\n📦 Downloaded package contains:")
        print("   - weights/best.pt (best model)")
        print("   - weights/last.pt (last checkpoint)")
        print("   - training plots (loss curves, metrics)")
        print("   - results.csv (training metrics)")
        print("   - configuration files")
    else:
        print("❌ Failed to create results package")

# Run the download
download_results()
