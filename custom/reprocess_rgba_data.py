import json
import copy
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm


USE_MULTIPROCESSING = True
NUM_CPUS = cpu_count()


LIST_FILE = Path("/storage/slurm/lavingal/lavingal/LVSM/datasets/GSO_rendered/full_list.txt")


GOAL_ROOT = Path("/storage/slurm/lavingal/lavingal/LVSM/datasets/gso_rendered_processed")


def process_single_image(source_path: Path, dest_path: Path) -> bool:
    try:
        img_loaded = Image.open(source_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if img_loaded.mode == 'RGBA':
            background = Image.new("RGB", img_loaded.size, (255, 255, 255))
            background.paste(img_loaded, mask=img_loaded.split()[3])
            background.save(dest_path, 'PNG')
        else:
            img_loaded.convert('RGB').save(dest_path, 'PNG')
        return True
    except FileNotFoundError:
        print(f"\n[Warning] Source file not found, skipping: {source_path}")
        return False
    except Exception as e:
        print(f"\n[Error] Image processing failed for {source_path}: {e}")
        return False


def process_scene(original_metadata_path: str) -> str | None:
    try:
        scene_id = Path(original_metadata_path).stem
        
        with open(original_metadata_path, 'r') as f:
            original_data = json.load(f)

        new_data = copy.deepcopy(original_data)

        for i, frame in enumerate(original_data.get("frames", [])):
            original_image_path = Path(frame["image_path"])
            image_filename = original_image_path.name

            dest_image_path = GOAL_ROOT / "images" / scene_id / image_filename

            if process_single_image(original_image_path, dest_image_path):
                new_data["frames"][i]["image_path"] = str(dest_image_path)

        dest_metadata_path = GOAL_ROOT / "metadata" / f"{scene_id}.json"
        dest_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_metadata_path, 'w') as f:
            json.dump(new_data, f, indent=2)

        return str(dest_metadata_path)
    except Exception as e:
        print(f"\n[Fatal Error] Scene processing failed for {original_metadata_path}: {e}")
        return None


if __name__ == "__main__":
    print("--- Starting Dataset Preprocessing ---")

    with open(LIST_FILE, 'r') as f:
        original_metadata_files = [line.strip() for line in f if line.strip()]

    new_metadata_paths = []

    if USE_MULTIPROCESSING:
        print(f"Mode: PARALLEL (using up to {NUM_CPUS} CPU cores)")
        with Pool(processes=NUM_CPUS) as pool:
            for result in tqdm(pool.imap_unordered(process_scene, original_metadata_files), total=len(original_metadata_files), desc="Processing Scenes"):
                if result:
                    new_metadata_paths.append(result)
    else:
        print("Mode: SERIAL (using a single CPU core)")
        for scene_path in tqdm(original_metadata_files, desc="Processing Scenes"):
            result = process_scene(scene_path)
            if result:
                new_metadata_paths.append(result)

    print("\nAll scenes processed.")

    if new_metadata_paths:
        new_list_filepath = GOAL_ROOT / "full_list_processed.txt"
        print(f"Writing new manifest file to: {new_list_filepath}")
        new_metadata_paths.sort()
        with open(new_list_filepath, 'w') as f:
            for path in new_metadata_paths:
                f.write(path + '\n')

    print("\n--- Preprocessing Complete! ---")
