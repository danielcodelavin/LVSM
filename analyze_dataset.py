import json
import numpy as np
from tqdm import tqdm
import os

def analyze_dataset(list_file_path):
    """
    Analyzes a dataset by reading a list of scene JSON files,
    counting the frames in each, and reporting statistics.
    """
    print(f"Reading scene list from: {list_file_path}")

    if not os.path.exists(list_file_path):
        print(f"Error: The file '{list_file_path}' was not found.")
        return

    with open(list_file_path, 'r') as f:
        scene_json_paths = [line.strip() for line in f if line.strip()]

    if not scene_json_paths:
        print("Error: The list file is empty.")
        return

    print(f"Found {len(scene_json_paths)} scenes to analyze.")
    
    frame_counts = []
    
    for scene_path in tqdm(scene_json_paths, desc="Analyzing Scenes"):
        try:
            with open(scene_path, 'r') as f:
                data = json.load(f)
                if "frames" in data and isinstance(data["frames"], list):
                    frame_counts.append(len(data["frames"]))
        except Exception as e:
            # This will skip any corrupted or unreadable files but continue analysis
            print(f"\nWarning: Could not process file {scene_path}. Error: {e}")
            continue

    if not frame_counts:
        print("Could not extract any frame counts from the dataset.")
        return

    # --- Calculate and Print Statistics ---
    frame_counts_np = np.array(frame_counts)
    
    total_scenes_processed = len(frame_counts_np)
    min_frames = np.min(frame_counts_np)
    max_frames = np.max(frame_counts_np)
    mean_frames = np.mean(frame_counts_np)
    median_frames = np.median(frame_counts_np)
    
    top_5_indices = np.argsort(frame_counts_np)[-5:]
    top_5_counts = frame_counts_np[top_5_indices]
    
    print("\n--- Dataset Frame Count Analysis ---")
    print(f"Total Scenes Processed: {total_scenes_processed}")
    print(f"Min Frames in a Scene:    {min_frames}")
    print(f"Max Frames in a Scene:    {max_frames:.0f}  <-- This is the size of your heaviest scene.")
    print(f"Average Frames per Scene: {mean_frames:.2f}")
    print(f"Median Frames per Scene:  {median_frames:.0f} <-- 50% of scenes have fewer frames than this.")
    print("\n--- Top 5 Heaviest Scenes (by frame count) ---")
    for i, count in enumerate(reversed(top_5_counts)):
        print(f"#{i+1}: {count} frames")
    print("------------------------------------")


if __name__ == "__main__":
    # --- HARDCODE THE PATH TO YOUR LIST FILE HERE ---
    LIST_FILE_PATH = "/storage/slurm/lavingal/lavingal/LVSM/datasets/re10k/train/full_list.txt"
    
    analyze_dataset(LIST_FILE_PATH)