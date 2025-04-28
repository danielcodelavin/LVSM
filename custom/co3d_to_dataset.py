import os
import json
import numpy as np
from pathlib import Path
import argparse
import gzip
import pickle
from tqdm import tqdm

def co3d_to_lvsm_format(co3d_base_dir: str, output_dir: str, 
                        categories=None, max_frames_per_seq=None, 
                        frame_interval=1, min_frames=5):
    """
    Convert Co3D frame annotations to LVSM format without copying images
    
    Args:
        co3d_base_dir: Base directory of Co3D dataset
        output_dir: Output directory for JSON files only
        categories: List of categories to process. If None, process all available
        max_frames_per_seq: Maximum number of frames to use per sequence (None = use all)
        frame_interval: Sample every Nth frame (1 = use all frames)
        min_frames: Minimum frames required per sequence
    
    Returns:
        List of paths to created JSON files
    """
    # Create output directories for metadata only
    os.makedirs(output_dir, exist_ok=True)
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # If no categories specified, find all categories in the base dir
    if categories is None:
        categories = [d for d in os.listdir(co3d_base_dir) 
                     if os.path.isdir(os.path.join(co3d_base_dir, d))]
    elif isinstance(categories, str):
        # Handle case where a single category is passed as string
        categories = [categories]
    
    print(f"Processing {len(categories)} categories: {categories}")
    
    all_json_paths = []
    
    for category in categories:
        category_dir = os.path.join(co3d_base_dir, category)
        frame_annotations_path = os.path.join(category_dir, "frame_annotations.jgz")
        
        if not os.path.exists(frame_annotations_path):
            print(f"Warning: {frame_annotations_path} not found. Skipping category.")
            continue
        
        # Load frame annotations
        print(f"Loading annotations from {frame_annotations_path}")
        try:
            with gzip.open(frame_annotations_path, "rb") as f:
                frame_annotations = pickle.load(f)
        except Exception as e:
            print(f"Error loading {frame_annotations_path}: {e}")
            continue
        
        print(f"Loaded {len(frame_annotations)} frame annotations from {frame_annotations_path}")
        
        # Group frames by sequence
        sequences = {}
        for frame in frame_annotations:
            seq_name = frame.sequence_name
            if seq_name not in sequences:
                sequences[seq_name] = []
            sequences[seq_name].append(frame)
        
        print(f"Found {len(sequences)} sequences in category {category}")
        
        # Process each sequence
        category_json_paths = []
        for seq_name, frames in tqdm(sequences.items(), desc=f"Processing {category} sequences"):
            # Sort frames by frame_number for consistency
            frames.sort(key=lambda x: x.frame_number)
            
            # Sample frames if requested
            if frame_interval > 1 or max_frames_per_seq is not None:
                if frame_interval > 1:
                    frames = frames[::frame_interval]
                if max_frames_per_seq is not None and len(frames) > max_frames_per_seq:
                    # Take evenly spaced frames
                    indices = np.linspace(0, len(frames)-1, max_frames_per_seq, dtype=int)
                    frames = [frames[i] for i in indices]
            
            # Skip if too few frames
            if len(frames) < min_frames:
                print(f"Skipping sequence {seq_name} with only {len(frames)} frames (minimum {min_frames})")
                continue
            
            # Prepare data in LVSM format
            scene_data = {
                "scene_name": f"{category}_{seq_name}",
                "frames": []
            }
            
            for frame in frames:
                # Use the absolute path to the original image
                image_path = os.path.abspath(os.path.join(co3d_base_dir, frame.image.path))
                
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                
                # Compute fxfycxcy from intrinsics
                h, w = frame.image.size
                fl_x, fl_y = frame.viewpoint.focal_length
                px_x, px_y = frame.viewpoint.principal_point
                
                # Convert from NDC format to pixel format
                # In Co3D, focal_length is in NDC units (normalized by half image dimension)
                # and principal_point is in [-1, 1] range
                fx = fl_x * w / 2
                fy = fl_y * h / 2
                cx = (px_x + 1.0) * w / 2  # Convert from [-1,1] to [0,w]
                cy = (px_y + 1.0) * h / 2  # Convert from [-1,1] to [0,h]
                
                # Create world-to-camera matrix
                R = np.array(frame.viewpoint.R)
                T = np.array(frame.viewpoint.T).reshape(3, 1)
                
                # Construct 4x4 w2c matrix
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = T.flatten()
                
                frame_data = {
                    "image_path": image_path,
                    "fxfycxcy": [float(fx), float(fy), float(cx), float(cy)],
                    "w2c": w2c.tolist()
                }
                
                scene_data["frames"].append(frame_data)
            
            # Only save sequences with enough frames
            if len(scene_data["frames"]) < min_frames:
                print(f"Skipping sequence {seq_name} - not enough valid frames")
                continue
                
            # Save to JSON file
            json_path = os.path.join(metadata_dir, f"{category}_{seq_name}.json")
            with open(json_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
            
            category_json_paths.append(json_path)
            all_json_paths.append(json_path)
        
        print(f"Created {len(category_json_paths)} JSON files for category {category}")
    
    # Create a list file of all JSON paths
    list_path = os.path.join(output_dir, "full_list.txt")
    with open(list_path, 'w') as f:
        for path in all_json_paths:
            f.write(f"{os.path.abspath(path)}\n")
    
    print(f"Created dataset list with {len(all_json_paths)} entries at {list_path}")
    return all_json_paths

if __name__ == "__main__":

    
    co3d_to_lvsm_format(
        co3d_base_dir="/home/stud/lavingal/storage/group/dataset_mirrors/01_incoming/Co3D",
        output_dir="/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/co3d_index",
        categories=None,
        max_frames_per_seq=None,
        frame_interval=1,
        min_frames=7
    )