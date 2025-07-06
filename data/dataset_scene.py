import random
import traceback
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
import signal

# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError("Scene processing took too long!")

class Dataset(Dataset):
    def __init__(self, config, load_all_views=False):
        super().__init__()
        self.config = config
        self.load_all_views = load_all_views
        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        

        self.inference = self.config.inference.get("if_inference", False)
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths


    def __len__(self):
        return len(self.all_scene_paths)


    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)
        uniform_size = self.config.training.get("uniform_crop_size", False)
        target_size = resize_h

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))
                
                if uniform_size and min_size != target_size:
                    image = image.resize((target_size, target_size), resample=Image.LANCZOS)
                    extra_scale = target_size / min_size

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
                
                if uniform_size and min_size != target_size:
                    fxfycxcy *= (extra_scale, extra_scale, extra_scale, extra_scale)
                    
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws
        

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.35):
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_pose = torch.eye(4, device=in_c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose)
        in_c2ws = avg_pose @ in_c2ws 

        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale
        in_c2ws[:, :3, 3] /= scene_scale
        return in_c2ws

    def view_selector(self, frames):
        # This function is not used when load_all_views=True (i.e., during chunking)
        # but is kept for consistency. The important check was in __getitem__.
        random_behavior= self.config.training.get("random_sample_views", False)
        if len(frames) < self.config.training.num_views:
            return None
       
        if not random_behavior:
            if len(frames) < self.config.training.num_views:
                return None
            view_selector_config = self.config.training.view_selector
            min_frame_dist = view_selector_config.get("min_frame_dist", 25)
            max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
            if max_frame_dist <= min_frame_dist:
                return None
            frame_dist = random.randint(min_frame_dist, max_frame_dist)
            if len(frames) <= frame_dist:
                return None
            start_frame = random.randint(0, len(frames) - frame_dist - 1)
            end_frame = start_frame + frame_dist
            sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
            image_indices = [start_frame, end_frame] + sampled_frames
            return image_indices
        else:  
            image_indices = random.sample(range(len(frames)), self.config.training.num_views)
            return image_indices

    def __getitem__(self, idx):
        # Set a 10-minute alarm for this function
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(600)  # 600 seconds = 10 minutes

        try:
            scene_path = self.all_scene_paths[idx].strip()
            data_json = json.load(open(scene_path, 'r'))
            frames = data_json["frames"]
            
            # --- This is the only part that needs to run for chunking ---
            # It takes all available frames from the scene.
            image_indices = list(range(len(frames)))

            # --- REMOVED THE MINIMUM VIEW CHECK AS REQUESTED ---
            # The original code had a check here that returned None if there were
            # too few views. It has been removed to process all scenes.
            # if len(image_indices) < self.config.training.num_views:
            #     signal.alarm(0)
            #     return None
            
            # Also return None if there are no frames at all, to be safe.
            if not image_indices:
                 signal.alarm(0)
                 return None
            
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
        
            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

            image_indices_tensor = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices_tensor, idx)
            indices = torch.cat([image_indices_tensor, scene_indices], dim=-1)

            result = {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": data_json["scene_name"]
            }
            signal.alarm(0) # Disable the alarm if we finish successfully
            return result

        except TimeoutError:
            print(f"[Worker PID: {os.getpid()}] TIMEOUT on scene: {self.all_scene_paths[idx].strip()}. Skipping.", flush=True)
            return None # Skip this item
        except Exception:
            print(f"[Worker PID: {os.getpid()}] ERROR processing scene: {self.all_scene_paths[idx].strip()}. Skipping.", flush=True)
            traceback.print_exc()
            return None
        finally:
            # Always ensure the alarm is disabled
            signal.alarm(0)