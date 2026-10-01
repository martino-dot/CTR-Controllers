import os
import numpy as np

def fix_rogue_calibration_files():
    # Use the current working directory
    project_dir = os.getcwd()
    found_any = False

    print(f"Scanning local directory ({project_dir}) for rogue .npz files...")
    
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".npz"):
                full_path = os.path.join(root, file)
                try:
                    z = np.load(full_path)
                    files_in_archive = z.files
                    
                    if 'mtx' in files_in_archive:
                        print(f"\n⚠️ Found rogue file: {full_path}")
                        print(f"   Current keys: {files_in_archive}")
                        
                        # Extract the old data
                        mtx_data = z['mtx']
                        dist_data = z['dist']
                        
                        # Overwrite with the 'K' key that vision.py wants
                        np.savez(full_path, K=mtx_data, dist=dist_data)
                        
                        print(f"✅ FIXED! Renamed 'mtx' to 'K'.")
                        found_any = True
                        
                    elif 'K' in files_in_archive:
                        print(f"Skipping {file} (Already correct)")
                        
                except Exception as e:
                    print(f"Could not read {full_path}: {e}")

    if not found_any:
        print("\nNo files needed fixing in this directory!")

if __name__ == "__main__":
    fix_rogue_calibration_files()
