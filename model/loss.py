import lpips
import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path


# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()
        
    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        
        return model.to(self.device).eval()
    
    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        
        if torch.distributed.get_rank() == 0:
            # Download weights if needed
            if not weight_file.exists():
                os.system(f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')
        torch.distributed.barrier()
        
        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        
        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                
                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        
        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())
        
        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
    
    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean
    
    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    
    def forward(self, pred_img, target_img):
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)
        
        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)
        
        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)
        
        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5
        
        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
        
        return total_loss

class LossComputer(nn.Module):
    def __init__(self, config):
        super(LossComputer, self).__init__()
        self.config = config
        
        # Existing loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Flag to determine if we're using diffusion mode.
        # Since you're in the diffusion branch, this will be set via the config.
        self.use_diffusion = config.model.get("use_diffusion", False)
        
        # Initialize LPIPS metric if available
        if lpips is not None:
            self.lpips_fn = lpips.LPIPS(net='alex')
        else:
            self.lpips_fn = None

    def calc_psnr(self, img1, img2):
        """Calculate PSNR between two images assumed to be in [0,1]."""
        mse = self.l2_loss(img1, img2)
        if mse == 0:
            return torch.tensor(100.0)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr

    def forward(self, rendering=None, target=None, predicted_noise=None, noise=None):
        """
        Calculate losses and evaluation metrics.

        For the diffusion model:
            - The training loss is MSE computed between the predicted noise and the actual noise.
            - Optionally, if 'rendering' and 'target' images are provided during evaluation,
              additional metrics (PSNR and LPIPS) are computed.

        For direct prediction (non-diffusion):
            - The loss is computed as L1 loss between the model output (rendering) and target image.
            - PSNR and LPIPS metrics are also computed.
        """
        metrics = {}
        if self.use_diffusion:
            # Training loss for diffusion: MSE on the noise prediction.
            loss = F.mse_loss(predicted_noise, noise)
            metrics['loss'] = loss
            metrics['mse_loss'] = loss.item()
            
            # Optionally compute evaluation metrics if rendering and target are available.
            if rendering is not None and target is not None:
                psnr = self.calc_psnr(rendering, target)
                metrics['psnr'] = psnr.item() if psnr is not None else None
                if self.lpips_fn is not None:
                    lpips_val = self.lpips_fn(rendering, target)
                    metrics['lpips'] = lpips_val.item()
        else:
            # Direct prediction loss (as in your original implementation)
            loss = self.l1_loss(rendering, target)
            metrics['loss'] = loss
            metrics['l1_loss'] = loss.item()
            
            psnr = self.calc_psnr(rendering, target)
            metrics['psnr'] = psnr.item() if psnr is not None else None
            if self.lpips_fn is not None:
                lpips_val = self.lpips_fn(rendering, target)
                metrics['lpips'] = lpips_val.item()
            
        return edict(metrics)