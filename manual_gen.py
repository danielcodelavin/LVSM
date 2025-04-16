# %% [markdown]
# # Extreme View Synthesis Explorer
# 
# Experiment with novel camera trajectories beyond standard evaluation scenarios

# %% [markdown]
# ## Setup and Configuration

# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import importlib
from pathlib import Path
from scipy.spatial.transform import Rotation

# %% [markdown]
# ### Distributed Setup (Required for Model)

# %%
# Set up dummy distributed environment for single GPU
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")

# %% [markdown]
# ### Path Configuration

# %%
CONFIG = {
    # Path to model checkpoint (single .pt file)
    "checkpoint_path": "/home/stud/lavingal/storage/slurm/lavingal/experiments/checkpoints/LVSM_scene_decoder_only/ckpt_0000000000246000.pt",
    
    # Path to dataset config file
    "dataset_config": "./configs/LVSM_scene_decoder_only.yaml",
    
    # Path to test dataset
    "test_data_path": "/path/to/datasets/re10k/test",
    
    # Output directory for visualizations
    "output_dir": "./extreme_renders"
}

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# %% [markdown]
# ## Model Initialization

# %%
def load_model(checkpoint_path, config_path):
    """Load trained model with proper distributed setup"""
    # Load base config
    config = OmegaConf.load(config_path)
    config = edict(OmegaConf.to_container(config))
    
    # Initialize model
    module, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module).__dict__[class_name]
    model = ModelClass(config).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=False)
    
    # Set to eval mode
    model.eval()
    return model

# %%
# Initialize model
model = load_model(CONFIG["checkpoint_path"], CONFIG["dataset_config"])

# %% [markdown]
# ## Scene Selection & Data Loading

# %%
def load_scene_batch(scene_path):
    """Load a scene from dataset format"""
    scene_data = {
        "image": torch.load(scene_path/"images.pt").cuda(),       # [1, V, C, H, W]
        "c2w": torch.load(scene_path/"poses.pt").cuda(),          # [1, V, 4, 4]
        "fxfycxcy": torch.load(scene_path/"intrinsics.pt").cuda() # [1, V, 4]
    }
    return edict(scene_data)

# %%
# List available scenes
scene_folders = sorted([f for f in Path(CONFIG["test_data_path"]).iterdir() if f.is_dir()])
print("Available scenes:")
for i, path in enumerate(scene_folders):
    print(f"{i}: {path.name}")

# Select scene
selected_idx = 4  # Change this to select different scenes
scene_path = scene_folders[selected_idx]
scene_data = load_scene_batch(scene_path)

# %% [markdown]
# ## Visualize Input Views

# %%
def show_views(images, titles=None):
    """Display multiple views in a grid"""
    num_views = images.shape[1]
    fig, axs = plt.subplots(1, num_views, figsize=(20, 5))
    for i in range(num_views):
        img = images[0,i].permute(1,2,0).cpu().numpy()
        axs[i].imshow(np.clip(img, 0, 1))
        axs[i].axis('off')
        if titles: axs[i].set_title(titles[i])
    plt.show()

# %%
print(f"\nInput views for scene: {scene_path.name}")
show_views(scene_data.image)

# %% [markdown]
# ## Camera Parameter Inspector

# %%
def print_camera_info(c2w, intrinsics):
    """Display camera position and orientation"""
    pos = c2w[0,0,:3,3].cpu().numpy()
    rot = c2w[0,0,:3,:3].cpu().numpy()
    fx, fy, cx, cy = intrinsics[0,0].cpu().numpy()
    
    print("Camera Position:", pos)
    print("Rotation Matrix:\n", rot)
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# %%
print("First input camera parameters:")
print_camera_info(scene_data.c2w, scene_data.fxfycxcy)

# %% [markdown]
# ## Extreme View Generator

# %%
def create_custom_camera(base_c2w, position_offset=(0,0,0), rotation_angles=(0,0,0)):
    """
    Create novel camera extrinsics
    - position_offset: (dx, dy, dz) in meters
    - rotation_angles: (pitch, yaw, roll) in degrees
    """
    new_c2w = base_c2w.clone()
    
    # Position offset
    new_c2w[0,0,:3,3] += torch.tensor(position_offset).cuda()
    
    # Rotation
    rot = Rotation.from_euler('xyz', rotation_angles, degrees=True)
    rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32).cuda()
    new_c2w[0,0,:3,:3] = rot_mat @ new_c2w[0,0,:3,:3]
    
    return new_c2w

def render_custom_view(model, scene_data, custom_c2w):
    """Render novel view with custom camera parameters"""
    # Create target data
    target = edict({
        "c2w": custom_c2w,
        "fxfycxcy": scene_data.fxfycxcy[:,:1],  # Use first camera's intrinsics
        "image_h_w": scene_data.image.shape[-2:]
    })
    
    # Compute rays
    target.ray_o, target.ray_d = model.process_data.compute_rays(
        target.c2w, 
        target.fxfycxcy,
        target.image_h_w[0],
        target.image_h_w[1],
        device="cuda"
    )
    
    # Create data batch
    data_batch = edict({
        "input": model.process_data.fetch_views(scene_data)[0],
        "target": target
    })
    
    # Render
    with torch.no_grad():
        result = model.render_video(data_batch, traj_type="custom", num_frames=1)
    
    return result.video_rendering[0,0].cpu().permute(1,2,0).numpy()

# %% [markdown]
# ## Experimentation Zone

# %%
# Get base camera parameters
base_c2w = scene_data.c2w[:,:1].clone()  # Use first input camera as base

# %% [markdown]
# ### Example 1: Extreme Position Offset

# %%
custom_c2w = create_custom_camera(
    base_c2w,
    position_offset=(2.0, -1.5, 0.5),  # Large right/down/forward movement
    rotation_angles=(15, -30, 0)
)

rendered = render_custom_view(model, scene_data, custom_c2w)
plt.imshow(np.clip(rendered, 0, 1))
plt.title("Extreme Position + Rotation")
plt.axis('off')
plt.show()

# %% [markdown]
# ### Example 2: Unusual Rotation Angles

# %%
custom_c2w = create_custom_camera(
    base_c2w,
    position_offset=(0, 0, 0),
    rotation_angles=(85, 0, 45)  # Extreme pitch + roll
)

rendered = render_custom_view(model, scene_data, custom_c2w)
plt.imshow(np.clip(rendered, 0, 1))
plt.title("Crazy Camera Angles")
plt.axis('off')
plt.show()

# %% [markdown]
# ### Example 3: Physics-Defying View

# %%
custom_c2w = create_custom_camera(
    base_c2w,
    position_offset=(0, 2.5, 0),  # Floating high above scene
    rotation_angles=(-90, 0, 0)    # Direct downward view
)

rendered = render_custom_view(model, scene_data, custom_c2w)
plt.imshow(np.clip(rendered, 0, 1))
plt.title("Bird's Eye View")
plt.axis('off')
plt.show()