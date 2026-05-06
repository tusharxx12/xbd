"""
Research-Grade Satellite Damage Detection Model

Architecture:
- Siamese Swin Transformer Encoder (shared weights)
- Cross-Attention Fusion at bottleneck
- Diff-CNN Branch for change detection
- U-Net style decoder with skip connections
- Multi-class damage segmentation head (5 classes)

Input: pre_img, post_img, diff_img (B, 3, 512, 512)
Output: damage_mask (B, 5, 512, 512)

Classes:
    0: Background
    1: No Damage
    2: Minor Damage
    3: Major Damage
    4: Destroyed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Optional
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention module for fusing pre and post disaster features.

    Post features act as Query (Q) - "what changed?"
    Pre features act as Key (K) and Value (V) - "reference state"

    This allows the model to attend to relevant pre-disaster regions
    when analyzing post-disaster changes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Query projection for post features
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        # Key and Value projections for pre features
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Layer normalization for stability
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Feedforward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(proj_drop),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        post_features: torch.Tensor,
        pre_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            post_features: (B, N, C) - Post-disaster features (Query)
            pre_features: (B, N, C) - Pre-disaster features (Key, Value)

        Returns:
            fused_features: (B, N, C) - Cross-attended features
        """
        B, N, C = post_features.shape

        # Normalize inputs
        q_input = self.norm_q(post_features)
        kv_input = self.norm_kv(pre_features)

        # Project to Q, K, V
        Q = self.q_proj(q_input)  # (B, N, C)
        K = self.k_proj(kv_input)  # (B, N, C)
        V = self.v_proj(kv_input)  # (B, N, C)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = (attn @ V).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Output projection with residual
        out = self.proj_drop(self.out_proj(out))
        out = self.norm_out(post_features + out)

        # Feedforward with residual
        out = out + self.ffn(out)
        out = self.norm_ffn(out)

        return out


class DiffCNNBranch(nn.Module):
    """
    Lightweight 3-layer CNN for processing difference images.

    Captures explicit pixel-level changes between pre and post images.
    The output is concatenated with transformer features to provide
    complementary change detection signals.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        out_channels: int = 256,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to match bottleneck feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(None)  # Will be set dynamically

        # Channel projection to match transformer features
        self.channel_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - Difference image
            target_size: (H, W) - Target spatial size to match transformer features

        Returns:
            features: (B, C, H', W') - CNN features
        """
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.conv2(x)  # (B, 128, H/4, W/4)
        x = self.conv3(x)  # (B, 256, H/8, W/8)

        # Match target size if specified
        if target_size is not None:
            x = F.adaptive_avg_pool2d(x, target_size)

        x = self.channel_proj(x)

        return x


class DecoderBlock(nn.Module):
    """
    U-Net style decoder block with bilinear upsampling and convolutions.

    Features are upsampled, concatenated with skip connections,
    and processed through convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        # Upsampling is done via bilinear interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # First conv after concatenation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Second conv for refinement
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Input features
            skip: (B, C_skip, H*2, W*2) - Skip connection features

        Returns:
            out: (B, C_out, H*2, W*2) - Upsampled and refined features
        """
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch (can happen due to odd dimensions)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SwinEncoderWrapper(nn.Module):
    """
    Wrapper for Swin Transformer to extract multi-level features.

    Extracts features from 4 stages of Swin Transformer for
    hierarchical feature representation.
    """

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        in_chans: int = 3,
    ):
        super().__init__()

        # Load Swin Transformer backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3), 
            img_size=512# Extract from all 4 stages
        )

        # Get feature channel dimensions
        # For swin_tiny: [96, 192, 384, 768]
        self.feature_channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) - Input image

        Returns:
            features: List of 4 feature maps from each Swin stage
                - Stage 1: (B, 96, H/4, W/4)
                - Stage 2: (B, 192, H/8, W/8)
                - Stage 3: (B, 384, H/16, W/16)
                - Stage 4: (B, 768, H/32, W/32)
        """
        features = self.backbone(x)
        return features


class SatelliteDamageDetectionModel(nn.Module):
    """
    Research-Grade Satellite Damage Detection Model

    Architecture Overview:
    =====================
    1. Siamese Swin Encoder (shared weights)
       - Extracts hierarchical features from pre and post images
       - 4 stages with increasing channels: [96, 192, 384, 768]

    2. Cross-Attention Fusion (at bottleneck)
       - Post features as Query (Q)
       - Pre features as Key (K) and Value (V)
       - Learns to identify damage by comparing pre/post states

    3. Diff-CNN Branch
       - Lightweight CNN for explicit change detection
       - Complementary to transformer-based fusion

    4. U-Net Style Decoder
       - Bilinear upsampling with skip connections
       - Recovers spatial resolution progressively

    5. Segmentation Head
       - 5-class output (background + 4 damage levels)
       - Pixel-wise classification

    Input Shapes:
    - pre_img: (B, 3, 512, 512)
    - post_img: (B, 3, 512, 512)
    - diff_img: (B, 3, 512, 512)

    Output Shape:
    - damage_mask: (B, 5, 512, 512)
    """

    def __init__(
        self,
        backbone_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        num_classes: int = 5,
        cross_attn_heads: int = 8,
        cross_attn_drop: float = 0.1,
        diff_cnn_channels: int = 256,
        decoder_channels: List[int] = None,
        use_deep_supervision: bool = False,
    ):
        """
        Args:
            backbone_name: Swin Transformer variant from timm
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of damage classes (default: 5)
            cross_attn_heads: Number of attention heads
            cross_attn_drop: Dropout rate for attention
            diff_cnn_channels: Output channels for Diff-CNN branch
            decoder_channels: Channel dimensions for decoder stages
            use_deep_supervision: Enable auxiliary outputs for training
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision

        # ============================================
        # 1. Siamese Swin Transformer Encoder
        # ============================================
        self.encoder = SwinEncoderWrapper(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=3,
        )

        # Get encoder channel dimensions [96, 192, 384, 768] for swin_tiny
        self.encoder_channels = self.encoder.feature_channels
        bottleneck_channels = self.encoder_channels[-1]  # 768

        # ============================================
        # 2. Cross-Attention Fusion Module
        # ============================================
        self.cross_attention = CrossAttentionFusion(
            embed_dim=bottleneck_channels,
            num_heads=cross_attn_heads,
            attn_drop=cross_attn_drop,
            proj_drop=cross_attn_drop,
        )

        # ============================================
        # 3. Diff-CNN Branch
        # ============================================
        self.diff_cnn = DiffCNNBranch(
            in_channels=3,
            base_channels=64,
            out_channels=diff_cnn_channels,
        )

        # Fusion layer for combining cross-attention output and diff-CNN
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels + diff_cnn_channels, bottleneck_channels,
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # ============================================
        # 4. U-Net Style Decoder
        # ============================================
        if decoder_channels is None:
            # Default: [384, 192, 96, 64]
            decoder_channels = [384, 192, 96, 64]

        self.decoder_channels = decoder_channels

        # Decoder blocks (4 stages to match encoder)
        self.decoder4 = DecoderBlock(
            in_channels=bottleneck_channels,
            skip_channels=self.encoder_channels[2],  # 384
            out_channels=decoder_channels[0],        # 384
        )

        self.decoder3 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=self.encoder_channels[1],  # 192
            out_channels=decoder_channels[1],        # 192
        )

        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=self.encoder_channels[0],  # 96
            out_channels=decoder_channels[2],        # 96
        )

        self.decoder1 = DecoderBlock(
            in_channels=decoder_channels[2],
            skip_channels=0,  # No skip connection for final stage
            out_channels=decoder_channels[3],  # 64
        )

        # Final upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # ============================================
        # 5. Segmentation Head
        # ============================================
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1),
        )

        # Deep supervision auxiliary heads (optional)
        if use_deep_supervision:
            self.aux_head4 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(decoder_channels[1], num_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder and head weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        pre_img: torch.Tensor,
        post_img: torch.Tensor,
        diff_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the damage detection model.

        Args:
            pre_img: (B, 3, 512, 512) - Pre-disaster satellite image
            post_img: (B, 3, 512, 512) - Post-disaster satellite image
            diff_img: (B, 3, 512, 512) - Difference image (|post - pre|)

        Returns:
            logits: (B, 5, 512, 512) - Per-pixel damage class logits

            If use_deep_supervision=True during training:
            (logits, aux_outputs) where aux_outputs is a list of
            intermediate predictions for auxiliary loss computation
        """
        B, C, H, W = pre_img.shape

        # ============================================
        # 1. Siamese Encoder - Extract multi-scale features
        # ============================================
        # Pre-disaster features: [f1, f2, f3, f4]
        pre_features = self.encoder(pre_img)
        # Post-disaster features: [f1, f2, f3, f4]
        post_features = self.encoder(post_img)

        # Feature shapes for swin_tiny with 512x512 input:
        # Stage 1: (B, 96, 128, 128)   - H/4
        # Stage 2: (B, 192, 64, 64)    - H/8
        # Stage 3: (B, 384, 32, 32)    - H/16
        # Stage 4: (B, 768, 16, 16)    - H/32

        # ============================================
        # 2. Cross-Attention Fusion at Bottleneck
        # ============================================
        # Get bottleneck features (smallest spatial resolution)
        pre_bottleneck = pre_features[-1]   # (B, 768, 16, 16)
        post_bottleneck = post_features[-1]  # (B, 768, 16, 16)

        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        B_feat, C_feat, H_feat, W_feat = post_bottleneck.shape

        pre_flat = pre_bottleneck.flatten(2).transpose(1, 2)   # (B, 256, 768)
        post_flat = post_bottleneck.flatten(2).transpose(1, 2)  # (B, 256, 768)

        # Apply cross-attention: post queries, pre keys/values
        fused_flat = self.cross_attention(post_flat, pre_flat)  # (B, 256, 768)

        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        fused_features = fused_flat.transpose(1, 2).view(B_feat, C_feat, H_feat, W_feat)

        # ============================================
        # 3. Diff-CNN Branch
        # ============================================
        # Process difference image and match bottleneck spatial size
        diff_features = self.diff_cnn(diff_img, target_size=(H_feat, W_feat))  # (B, 256, 16, 16)

        # ============================================
        # 4. Feature Fusion
        # ============================================
        # Concatenate cross-attention output with diff-CNN features
        combined = torch.cat([fused_features, diff_features], dim=1)  # (B, 1024, 16, 16)
        combined = self.fusion_conv(combined)  # (B, 768, 16, 16)

        # ============================================
        # 5. Decoder with Skip Connections
        # ============================================
        # Combine pre and post skip connections (element-wise difference + concatenation alternative)
        # Here we use post features as primary and add change information
        skip3 = post_features[2] + (post_features[2] - pre_features[2])  # (B, 384, 32, 32)
        skip2 = post_features[1] + (post_features[1] - pre_features[1])  # (B, 192, 64, 64)
        skip1 = post_features[0] + (post_features[0] - pre_features[0])  # (B, 96, 128, 128)

        # Decoder stages
        d4 = self.decoder4(combined, skip3)   # (B, 384, 32, 32)
        d3 = self.decoder3(d4, skip2)         # (B, 192, 64, 64)
        d2 = self.decoder2(d3, skip1)         # (B, 96, 128, 128)
        d1 = self.decoder1(d2, None)          # (B, 64, 256, 256)

        # Final upsampling to input resolution
        d0 = self.final_upsample(d1)          # (B, 64, 512, 512)

        # ============================================
        # 6. Segmentation Head
        # ============================================
        logits = self.segmentation_head(d0)   # (B, 5, 512, 512)

        # Deep supervision outputs for training
        if self.use_deep_supervision and self.training:
            aux_out4 = F.interpolate(self.aux_head4(d4), size=(H, W), mode='bilinear', align_corners=False)
            aux_out3 = F.interpolate(self.aux_head3(d3), size=(H, W), mode='bilinear', align_corners=False)
            aux_out2 = F.interpolate(self.aux_head2(d2), size=(H, W), mode='bilinear', align_corners=False)
            return logits, [aux_out4, aux_out3, aux_out2]

        return logits

    def get_parameter_groups(self, lr: float = 1e-4, backbone_lr_mult: float = 0.1):
        """
        Get parameter groups with different learning rates.

        Args:
            lr: Base learning rate for decoder/head
            backbone_lr_mult: Multiplier for backbone learning rate

        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = list(self.encoder.parameters())
        other_params = [p for p in self.parameters() if not any(p is bp for bp in backbone_params)]

        return [
            {'params': backbone_params, 'lr': lr * backbone_lr_mult},
            {'params': other_params, 'lr': lr},
        ]


class DamageDetectionLoss(nn.Module):
    """
    Combined loss function for damage detection.

    Components:
    - Cross-Entropy Loss: For pixel-wise classification
    - Dice Loss: For handling class imbalance
    - Focal Loss: For hard example mining
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: torch.Tensor = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
        ignore_index: int = -1,
        deep_supervision_weights: List[float] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        self.ignore_index = ignore_index

        # Register class weights buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Deep supervision weights
        if deep_supervision_weights is None:
            self.deep_supervision_weights = [0.4, 0.2, 0.1]  # For aux outputs
        else:
            self.deep_supervision_weights = deep_supervision_weights

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
        )

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for hard example mining"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss

        if self.class_weights is not None:
            weight_map = self.class_weights[targets.clamp(0, self.num_classes - 1)]
            focal_loss = focal_loss * weight_map

        return focal_loss.mean()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss for class imbalance handling"""
        smooth = 1e-5

        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute Dice per class (skip background if needed)
        dice_per_class = []
        for c in range(1, self.num_classes):  # Skip background (class 0)
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice_c = (2.0 * intersection + smooth) / (union + smooth)
            dice_per_class.append(1 - dice_c)

        return torch.stack(dice_per_class).mean()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        aux_outputs: List[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            outputs: (B, C, H, W) - Model predictions
            targets: (B, H, W) - Ground truth labels
            aux_outputs: List of auxiliary outputs for deep supervision

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}

        # Main output loss
        if self.use_focal:
            ce_loss = self.focal_loss(outputs, targets)
        else:
            ce_loss = self.ce_loss(outputs, targets)

        dice_loss = self.dice_loss(outputs, targets)

        main_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        losses['ce_loss'] = ce_loss
        losses['dice_loss'] = dice_loss
        losses['main_loss'] = main_loss

        # Deep supervision losses
        total_loss = main_loss
        if aux_outputs is not None:
            for i, (aux_out, weight) in enumerate(zip(aux_outputs, self.deep_supervision_weights)):
                if self.use_focal:
                    aux_ce = self.focal_loss(aux_out, targets)
                else:
                    aux_ce = self.ce_loss(aux_out, targets)
                aux_dice = self.dice_loss(aux_out, targets)
                aux_loss = self.ce_weight * aux_ce + self.dice_weight * aux_dice

                losses[f'aux_loss_{i}'] = aux_loss
                total_loss = total_loss + weight * aux_loss

        losses['total_loss'] = total_loss

        return losses


def create_model(
    pretrained: bool = True,
    num_classes: int = 5,
    use_deep_supervision: bool = False,
) -> SatelliteDamageDetectionModel:
    """
    Factory function to create the damage detection model.

    Args:
        pretrained: Use ImageNet pretrained backbone
        num_classes: Number of damage classes
        use_deep_supervision: Enable auxiliary outputs

    Returns:
        Initialized model
    """
    model = SatelliteDamageDetectionModel(
        backbone_name='swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes,
        cross_attn_heads=8,
        cross_attn_drop=0.1,
        diff_cnn_channels=256,
        decoder_channels=[384, 192, 96, 64],
        use_deep_supervision=use_deep_supervision,
    )
    return model


# ============================================
# Testing and Verification
# ============================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("SATELLITE DAMAGE DETECTION MODEL TEST")
    print("=" * 60)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = create_model(pretrained=False, use_deep_supervision=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    pre_img = torch.randn(batch_size, 3, 512, 512).to(device)
    post_img = torch.randn(batch_size, 3, 512, 512).to(device)
    diff_img = torch.randn(batch_size, 3, 512, 512).to(device)

    print(f"\nInput shapes:")
    print(f"  pre_img: {pre_img.shape}")
    print(f"  post_img: {post_img.shape}")
    print(f"  diff_img: {diff_img.shape}")

    # Forward pass (training mode with deep supervision)
    model.train()
    start_time = time.time()
    outputs, aux_outputs = model(pre_img, post_img, diff_img)
    train_time = time.time() - start_time

    print(f"\nTraining mode outputs:")
    print(f"  Main output: {outputs.shape}")
    print(f"  Auxiliary outputs: {[a.shape for a in aux_outputs]}")
    print(f"  Forward time: {train_time:.3f}s")

    # Forward pass (eval mode)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(pre_img, post_img, diff_img)
        eval_time = time.time() - start_time

    print(f"\nEval mode output: {outputs.shape}")
    print(f"  Forward time: {eval_time:.3f}s")

    # Test loss computation
    loss_fn = DamageDetectionLoss(
        num_classes=5,
        class_weights=torch.tensor([0.1, 1.0, 2.0, 3.0, 4.0]).to(device),
        use_focal=True,
        deep_supervision_weights=[0.4, 0.2, 0.1],
    )

    targets = torch.randint(0, 5, (batch_size, 512, 512)).to(device)
    model.train()
    outputs, aux_outputs = model(pre_img, post_img, diff_img)
    losses = loss_fn(outputs, targets, aux_outputs)

    print(f"\nLoss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
