import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--path_to_A_folder', dest='path_to_A_folder', help='input directory for folder A', type=str, required=True)
parser.add_argument('--path_to_B_folder', dest='path_to_B_folder', help='input directory for folder B', type=str, required=True)
parser.add_argument('--path_to_AB_folder', dest='path_to_AB_folder', help='output directory', type=str, required=True)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

def process_folders(path_A, path_B, path_AB):
    subfolders_A = [f for f in os.listdir(path_A) if os.path.isdir(os.path.join(path_A, f)) and not f.startswith('.')]
    
    for subfolder in subfolders_A:
        subfolder_A = os.path.join(path_A, subfolder)
        subfolder_B = os.path.join(path_B, subfolder)
        subfolder_AB = os.path.join(path_AB, subfolder)
        
        if not os.path.exists(subfolder_B):
            print(f"Subfolder {subfolder} not found in folder B. Skipping...")
            continue
        
        if not os.path.exists(subfolder_AB):
            os.makedirs(subfolder_AB)
        
        process_subfolder(subfolder_A, subfolder_B, subfolder_AB)

def process_subfolder(subfolder_A, subfolder_B, subfolder_AB):
    files_A = [f for f in os.listdir(subfolder_A) if os.path.isfile(os.path.join(subfolder_A, f)) and not f.startswith('.')]
    
    for file_A in files_A:
        path_A = os.path.join(subfolder_A, file_A)
        path_B = os.path.join(subfolder_B, file_A)  # Assuming same filename in B
        
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = f"{os.path.splitext(file_A)[0]}_AB{os.path.splitext(file_A)[1]}"
            path_AB = os.path.join(subfolder_AB, name_AB)
            
            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
            print(f"Processed: {name_AB}")
        else:
            print(f"Skipping {file_A}: File not found in both A and B folders")

process_folders(args.path_to_A_folder, args.path_to_B_folder, args.path_to_AB_folder)