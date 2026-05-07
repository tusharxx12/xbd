# Video Script: Satellite Damage Detection using Siamese Swin-Transformers

## Video Overview
- **Total Duration:** ~8-10 minutes
- **Target Audience:** Researchers, ML practitioners, disaster response teams
- **Style:** Technical explanation with visual demonstrations

---

## INTRO SEQUENCE (0:00 - 0:45)

### Visual
- Aerial footage of disaster aftermath (hurricane, earthquake)
- Transition to satellite imagery comparison (before/after)
- Project title animation

### Narration
> "When natural disasters strike, rapid assessment of building damage is critical for emergency response. Traditional manual inspection takes days or weeks. But what if we could automate this using AI and satellite imagery?"
>
> "In this video, I'll walk you through our deep learning solution for automatic building damage assessment using the xView2 challenge dataset. We'll cover the architecture, training pipeline, and see real predictions."

---

## SECTION 1: THE PROBLEM (0:45 - 2:00)

### Visual
- Show xBD dataset examples
- Highlight pre/post disaster image pairs
- Display damage classification levels with color coding

### Narration
> "The xView2 challenge provides satellite imagery from multiple disasters - hurricanes, floods, wildfires, and earthquakes. Each scene contains a pre-disaster image showing intact buildings, and a post-disaster image showing the aftermath."
>
> "Our task is to classify each building pixel into 5 categories:
> - Background (black)
> - No Damage (green)
> - Minor Damage (yellow)
> - Major Damage (orange)
> - Destroyed (red)"
>
> "The challenge? Extreme class imbalance. Destroyed buildings make up less than 1% of pixels, while background dominates at 87%."

### Key Points to Show
```
Dataset Statistics:
├── Background:    87.45%
├── No Damage:      9.30%
├── Minor Damage:   1.15%
├── Major Damage:   1.45%
└── Destroyed:      0.66%
```

---

## SECTION 2: ARCHITECTURE OVERVIEW (2:00 - 4:30)

### Visual
- Animated architecture diagram
- Highlight each component as discussed
- Show tensor shapes flowing through network

### Narration

#### Part A: Siamese Encoder (2:00 - 2:45)
> "Our architecture starts with a Siamese encoder - meaning we use the SAME Swin Transformer to process both pre and post images. This weight sharing forces the network to learn consistent feature representations."
>
> "We chose Swin Transformer over CNN backbones because its hierarchical structure and shifted window attention captures both local details and global context - crucial for understanding building damage patterns."

#### Part B: Cross-Attention Fusion (2:45 - 3:30)
> "At the bottleneck, we introduce Cross-Temporal Attention. Here's the key insight: post-disaster features become Queries, asking 'what changed?' while pre-disaster features serve as Keys and Values, providing the reference state."
>
> "This attention mechanism learns to focus on regions where buildings existed before and may be damaged now. It's explicitly modeling the temporal relationship between image pairs."

#### Part C: Diff-CNN Branch (3:30 - 4:00)
> "We also compute the absolute difference between post and pre images - a simple but powerful change detection signal. A lightweight 3-layer CNN extracts features from this difference map."
>
> "This provides geometric-residual priors that complement the transformer's semantic understanding. Where transformers see 'building damage', the Diff-CNN sees 'these pixels changed significantly'."

#### Part D: Decoder (4:00 - 4:30)
> "Finally, a U-Net style decoder upsamples features back to full resolution. Skip connections bring in multi-scale information, but we enhance them with change-awareness: post features PLUS the difference from pre features."
>
> "The output is a 5-channel probability map at 512x512 resolution."

---

## SECTION 3: DATA PREPROCESSING (4:30 - 5:30)

### Visual
- Show preprocessing pipeline flowchart
- Demonstrate tiling process
- Display mask generation from polygons

### Narration
> "The xBD dataset comes with 1024x1024 images and JSON annotations containing building polygons. Our preprocessing pipeline:
>
> 1. Parses WKT polygons from JSON
> 2. Rasterizes them into segmentation masks with damage classes
> 3. Tiles images into 512x512 patches for training
> 4. Computes difference images for the Diff-CNN branch
> 5. Filters out tiles with too few buildings"
>
> "Critically, we use SCENE-WISE splitting - tiles from the same original image stay together in either train or validation. This prevents data leakage."

### Code Snippet to Show
```python
# Scene-wise split prevents data leakage
train_tiles, val_tiles = scene_wise_split(
    scenes,
    train_ratio=0.8,
    random_seed=42
)
# Result: 80% of SCENES for training, not 80% of tiles
```

---

## SECTION 4: TRAINING STRATEGY (5:30 - 6:45)

### Visual
- Training curves animation
- Loss function diagram
- Augmentation examples

### Narration

#### Part A: Loss Function (5:30 - 6:00)
> "To handle class imbalance, we combine Focal Loss and Dice Loss. Focal Loss down-weights easy background pixels, focusing learning on hard examples. Dice Loss optimizes the overlap ratio directly, helping with small damaged regions."
>
> "We also apply class weights: background gets 0.1, while destroyed buildings get 4.0 - a 40x emphasis on the rarest class."

#### Part B: Training Techniques (6:00 - 6:45)
> "Modern training techniques keep everything stable:
> - Mixed Precision (FP16) for faster training and larger batches
> - Gradient Clipping at 1.0 to prevent transformer instabilities
> - OneCycleLR scheduler with warmup for optimal convergence
> - Early Stopping on validation F1 with patience of 10 epochs"
>
> "Data augmentation includes random rotations, flips, color jittering, and scale variations - all applied consistently across pre, post, and diff images."

---

## SECTION 5: INFERENCE & TTA (6:45 - 7:45)

### Visual
- Show test image being tiled
- Animate TTA process (flip, predict, flip back, average)
- Stitching visualization

### Narration
> "For inference on 1024x1024 test images, we tile them into 512x512 patches, predict each, and stitch results back together with overlap averaging for smooth boundaries."
>
> "Test Time Augmentation boosts our F1 score: we run prediction on the original image, a horizontal flip, and a vertical flip. After reversing the transforms, we average all three probability maps."
>
> "This ensemble approach reduces prediction noise and handles edge cases better than single-pass inference."

### Code Snippet to Show
```python
# Test Time Augmentation
tta_modes = ["original", "hflip", "vflip"]

predictions = []
for mode in tta_modes:
    augmented = apply_transform(image, mode)
    pred = model(augmented)
    pred = reverse_transform(pred, mode)
    predictions.append(pred)

final = average(predictions)  # Ensemble
```

---

## SECTION 6: RESULTS DEMONSTRATION (7:45 - 9:00)

### Visual
- Side-by-side comparisons: Pre | Post | Ground Truth | Prediction
- Metrics dashboard
- Per-class performance breakdown

### Narration
> "Let's see actual predictions on validation data. Here we have a hurricane damage scene..."
>
> [Show 3-4 example predictions with commentary]
>
> "Notice how the model correctly identifies:
> - Intact buildings in green
> - Structural damage in orange
> - Completely destroyed buildings in red"
>
> "The confusion matrix reveals that background and no-damage achieve high accuracy, while the rarer damage classes present more challenge - an ongoing research direction."

### Metrics to Display
```
┌─────────────────────────────────────┐
│         VALIDATION METRICS          │
├─────────────────────────────────────┤
│  Macro F1-Score:     XX.XX%         │
│  Mean IoU (mIoU):    XX.XX%         │
│  Overall Accuracy:   XX.XX%         │
├─────────────────────────────────────┤
│  Per-Class F1:                      │
│    Background:       XX.XX%         │
│    No Damage:        XX.XX%         │
│    Minor Damage:     XX.XX%         │
│    Major Damage:     XX.XX%         │
│    Destroyed:        XX.XX%         │
└─────────────────────────────────────┘
```

---

## SECTION 7: CONCLUSION & FUTURE WORK (9:00 - 9:45)

### Visual
- Summary slide with key contributions
- Future directions diagram

### Narration
> "To summarize, our approach combines:
> - Siamese Swin-Transformers for robust temporal feature extraction
> - Cross-Attention Fusion to model 'what changed'
> - Diff-CNN branch for explicit geometric change detection
> - Hybrid Focal-Dice loss for class imbalance"
>
> "Future improvements could include:
> - Instance segmentation for building-wise damage scores
> - Multi-scale inference for varying building sizes
> - Uncertainty estimation for human-in-the-loop verification"
>
> "All code is available on GitHub. If you found this helpful, please like and subscribe!"

---

## OUTRO (9:45 - 10:00)

### Visual
- GitHub repository link
- Social media handles
- "Thanks for watching" animation

### Narration
> "Thanks for watching! Check the description for links to the code and dataset. See you in the next video!"

---

## B-ROLL SUGGESTIONS

1. **Disaster footage** - Stock footage of hurricane/earthquake damage
2. **Satellite imagery** - Google Earth timelapse of disaster areas
3. **Code scrolling** - Screen recording of key Python files
4. **Training visualization** - Loss curves, learning rate schedules
5. **Prediction examples** - Multiple disaster types (flood, fire, earthquake)

---

## GRAPHICS NEEDED

1. Architecture diagram (animated)
2. Data flow animation
3. Tiling/stitching visualization
4. TTA process animation
5. Class imbalance bar chart
6. Confusion matrix heatmap
7. Metrics dashboard

---

## MUSIC SUGGESTIONS

- Intro/Outro: Upbeat electronic (tech feel)
- Explanations: Calm ambient background
- Results: Subtle triumphant tone

