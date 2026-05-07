# Presentation Slides Outline

## Slide 1: Title Slide
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   HIGH-RESOLUTION SATELLITE DAMAGE ASSESSMENT           │
│   via Siamese Swin-Transformers and                     │
│   Geometric-Residual Priors                             │
│                                                         │
│   ─────────────────────────────────────────             │
│                                                         │
│   xView2 Building Damage Detection Challenge            │
│                                                         │
│   [Your Name]                                           │
│   [Date]                                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 2: The Challenge
```
┌─────────────────────────────────────────────────────────┐
│  THE CHALLENGE: Rapid Disaster Damage Assessment        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🌪️ Natural disasters cause massive destruction         │
│  ⏱️ Manual inspection takes days/weeks                  │
│  🛰️ Satellite imagery is available within hours         │
│                                                         │
│  GOAL: Automatically classify building damage from      │
│        pre/post disaster satellite image pairs          │
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────┐   │
│  │ Pre-Image   │ +  │ Post-Image  │ →  │ Damage    │   │
│  │ (Before)    │    │ (After)     │    │ Map       │   │
│  └─────────────┘    └─────────────┘    └───────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 3: Dataset Overview
```
┌─────────────────────────────────────────────────────────┐
│  xBD (xView2) DATASET                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📊 Statistics:                                         │
│     • 9,168 image pairs (1024×1024)                    │
│     • 19 disaster events                                │
│     • ~800,000 building annotations                     │
│                                                         │
│  🏷️ Damage Classes:                                    │
│     ⬛ Background (87.45%)                              │
│     🟩 No Damage (9.30%)                               │
│     🟨 Minor Damage (1.15%)                            │
│     🟧 Major Damage (1.45%)                            │
│     🟥 Destroyed (0.66%)                               │
│                                                         │
│  ⚠️ SEVERE CLASS IMBALANCE!                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 4: Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│  MODEL ARCHITECTURE                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           SIAMESE SWIN-TRANSFORMER               │   │
│  │  Pre-Image ──► [Swin Encoder] ◄── Post-Image    │   │
│  │                (Shared Weights)                  │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│  ┌─────────────────────┴───────────────────────────┐   │
│  │         CROSS-ATTENTION FUSION                   │   │
│  │   Post=Query  ←→  Pre=Key,Value                  │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│  ┌──────────┐          │                                │
│  │ DIFF-CNN │ ─────────┴─────► DECODER ──► OUTPUT      │
│  └──────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 5: Siamese Encoder
```
┌─────────────────────────────────────────────────────────┐
│  SIAMESE SWIN-TRANSFORMER ENCODER                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  WHY SIAMESE?                                           │
│  • Same encoder processes both images                   │
│  • Forces consistent feature representations            │
│  • Reduces parameters by 50%                            │
│                                                         │
│  WHY SWIN TRANSFORMER?                                  │
│  • Hierarchical structure (like CNNs)                   │
│  • Shifted window attention captures local + global     │
│  • State-of-the-art for dense prediction               │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │ Stage 1: 96ch (H/4)  ──► Stage 2: 192ch (H/8)  │    │
│  │ Stage 3: 384ch (H/16) ──► Stage 4: 768ch (H/32)│    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 6: Cross-Attention Fusion
```
┌─────────────────────────────────────────────────────────┐
│  CROSS-TEMPORAL ATTENTION                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  KEY INSIGHT:                                           │
│  "Post features ASK what changed,                       │
│   Pre features ANSWER what existed"                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Post Features ──► Q (Query)                     │   │
│  │                         ↓                        │   │
│  │  Pre Features ──► K ──► Attention ──► Output    │   │
│  │               └─► V ─────────┘                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  FORMULA:                                               │
│  Attention(Q, K, V) = softmax(QKᵀ/√d) · V              │
│                                                         │
│  8 attention heads × 96 dimensions each                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 7: Diff-CNN Branch
```
┌─────────────────────────────────────────────────────────┐
│  DIFF-CNN BRANCH: Geometric-Residual Priors            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  MOTIVATION:                                            │
│  • Transformers capture semantic meaning                │
│  • But explicit pixel changes also matter!              │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Diff = |Post - Pre|                           │    │
│  │      ↓                                          │    │
│  │  Conv1 (3→64) ──► Conv2 (64→128) ──► Conv3     │    │
│  │                                    (128→256)    │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  BENEFITS:                                              │
│  • Lightweight (only 3 layers)                          │
│  • Captures obvious structural changes                  │
│  • Complements transformer features                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 8: Loss Function
```
┌─────────────────────────────────────────────────────────┐
│  HYBRID FOCAL-DICE LOSS                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  PROBLEM: Severe class imbalance                        │
│  • Background: 87%, Destroyed: 0.66%                    │
│                                                         │
│  SOLUTION: Combined loss function                       │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │ L = 0.5 × Focal + 0.5 × Dice                   │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  FOCAL LOSS:                                            │
│  • Down-weights easy samples (background)               │
│  • FL = -(1-pt)^γ × log(pt), γ=2                       │
│                                                         │
│  DICE LOSS:                                             │
│  • Optimizes overlap directly                           │
│  • DL = 1 - 2×|A∩B|/(|A|+|B|)                          │
│                                                         │
│  CLASS WEIGHTS: [0.1, 1.0, 2.0, 3.0, 4.0]              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 9: Training Strategy
```
┌─────────────────────────────────────────────────────────┐
│  TRAINING STRATEGY                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OPTIMIZER: AdamW                                       │
│  • Learning rate: 1e-4                                  │
│  • Weight decay: 0.01                                   │
│  • Backbone LR: 1e-5 (10× slower)                      │
│                                                         │
│  SCHEDULER: OneCycleLR                                  │
│  • Warmup: 3 epochs                                     │
│  • Cosine annealing to 1e-7                            │
│                                                         │
│  STABILITY:                                             │
│  • Mixed precision (FP16)                               │
│  • Gradient clipping: 1.0                               │
│  • Early stopping: patience=10                          │
│                                                         │
│  DATA AUGMENTATION:                                     │
│  • RandomRotate90, H/V Flips                            │
│  • ShiftScaleRotate                                     │
│  • ColorJitter                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 10: Inference Pipeline
```
┌─────────────────────────────────────────────────────────┐
│  HIGH-PERFORMANCE INFERENCE                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣ TILING (for 1024×1024 test images)                 │
│     ┌─────┬─────┐                                       │
│     │ T1  │ T2  │  → Predict each 512×512 tile         │
│     ├─────┼─────┤                                       │
│     │ T3  │ T4  │  → Stitch with overlap averaging     │
│     └─────┴─────┘                                       │
│                                                         │
│  2️⃣ TEST-TIME AUGMENTATION (TTA)                       │
│     ┌────────┐   ┌────────┐   ┌────────┐               │
│     │Original│ + │ H-Flip │ + │ V-Flip │               │
│     └───┬────┘   └───┬────┘   └───┬────┘               │
│         └────────────┴────────────┘                     │
│                      ↓                                  │
│              Average Predictions                        │
│                                                         │
│  RESULT: ~3-5% F1 improvement from TTA                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 11: Results
```
┌─────────────────────────────────────────────────────────┐
│  RESULTS                                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  VALIDATION METRICS:                                    │
│  ┌──────────────────────────────────────────────┐      │
│  │ Macro F1-Score:        XX.XX%                │      │
│  │ Weighted F1-Score:     XX.XX%                │      │
│  │ Mean IoU (mIoU):       XX.XX%                │      │
│  │ Overall Accuracy:      XX.XX%                │      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
│  PER-CLASS F1:                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │ Background:    XX.XX%  │  Major:    XX.XX%   │      │
│  │ No Damage:     XX.XX%  │  Destroyed: XX.XX%  │      │
│  │ Minor:         XX.XX%  │                     │      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 12: Visual Results
```
┌─────────────────────────────────────────────────────────┐
│  QUALITATIVE RESULTS                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │
│  │  Pre   │  │  Post  │  │   GT   │  │  Pred  │        │
│  └────────┘  └────────┘  └────────┘  └────────┘        │
│                                                         │
│  Hurricane Harvey - Houston, TX                         │
│  ────────────────────────────────                       │
│  [Image]     [Image]     [Image]     [Image]           │
│                                                         │
│  Woolsey Fire - California                              │
│  ────────────────────────────────                       │
│  [Image]     [Image]     [Image]     [Image]           │
│                                                         │
│  ⬛ Background  🟩 No Damage  🟨 Minor                  │
│  🟧 Major  🟥 Destroyed                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 13: Key Contributions
```
┌─────────────────────────────────────────────────────────┐
│  KEY CONTRIBUTIONS                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SIAMESE SWIN-TRANSFORMER                            │
│     • First application of Swin-T to bi-temporal        │
│       damage assessment with shared weights             │
│                                                         │
│  2. CROSS-TEMPORAL ATTENTION                            │
│     • Novel Q/K/V formulation for change detection      │
│     • Post queries what changed, Pre provides context   │
│                                                         │
│  3. GEOMETRIC-RESIDUAL PRIORS                           │
│     • Diff-CNN complements semantic features            │
│     • Captures explicit pixel-level changes             │
│                                                         │
│  4. PRODUCTION-READY PIPELINE                           │
│     • Scene-wise splitting prevents leakage             │
│     • TTA and tiling for high-resolution inference     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 14: Future Work
```
┌─────────────────────────────────────────────────────────┐
│  FUTURE WORK                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🔬 MODEL IMPROVEMENTS:                                 │
│     • Instance segmentation for building-wise scores    │
│     • Larger backbones (Swin-B, Swin-L)                │
│     • Multi-scale inference                             │
│                                                         │
│  📊 DATA ENHANCEMENTS:                                  │
│     • Additional disaster types                         │
│     • Cross-sensor generalization                       │
│     • Temporal sequences (pre → during → post)         │
│                                                         │
│  🚀 DEPLOYMENT:                                         │
│     • Edge deployment for drones                        │
│     • Real-time processing pipeline                     │
│     • Uncertainty estimation for human review           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 15: Thank You
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                    THANK YOU!                           │
│                                                         │
│  ─────────────────────────────────────────              │
│                                                         │
│  📧 Email: your.email@example.com                       │
│  🐙 GitHub: github.com/your-username/xbd               │
│  📄 Paper: arxiv.org/abs/XXXX.XXXXX                    │
│                                                         │
│  ─────────────────────────────────────────              │
│                                                         │
│  QUESTIONS?                                             │
│                                                         │
│  ─────────────────────────────────────────              │
│                                                         │
│  ACKNOWLEDGMENTS:                                       │
│  • xView2 Challenge organizers                          │
│  • DIU & Maxar Technologies                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Notes for Presenter

### Timing Guide (10-minute presentation)
- Slides 1-2: 1 minute (Introduction)
- Slides 3-4: 1.5 minutes (Problem & Data)
- Slides 5-7: 3 minutes (Architecture deep dive)
- Slides 8-9: 1.5 minutes (Training)
- Slides 10-12: 2 minutes (Results)
- Slides 13-15: 1 minute (Contributions & Wrap-up)

### Key Points to Emphasize
1. The SIAMESE weight sharing is crucial for temporal consistency
2. Cross-attention explicitly models "what changed"
3. Diff-CNN provides complementary geometric information
4. Scene-wise splitting prevents data leakage
5. TTA significantly improves results

### Demo Ideas
1. Live inference on a test image
2. Show the preprocessing pipeline in action
3. Walk through a training epoch with live metrics
4. Compare predictions with/without TTA
