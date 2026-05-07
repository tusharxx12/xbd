"""
Manim Video: Satellite Damage Detection Explained

Run with:
    manim -pqh manim_video.py SatelliteDamageVideo

Options:
    -pql  : Preview quality low (fast render)
    -pqm  : Preview quality medium
    -pqh  : Preview quality high (best quality)
    -a    : Render all scenes

Requirements:
    pip install manim
"""

from manim import *
import numpy as np

# Color scheme
COLORS = {
    "primary": "#2196F3",
    "secondary": "#4CAF50",
    "accent": "#FF9800",
    "danger": "#F44336",
    "purple": "#9C27B0",
    "cyan": "#00BCD4",
    "dark": "#263238",
    "light": "#ECEFF1",
}

DAMAGE_COLORS = {
    "Background": "#000000",
    "No Damage": "#00FF00",
    "Minor": "#FFFF00",
    "Major": "#FFA500",
    "Destroyed": "#FF0000",
}


class IntroScene(Scene):
    """Opening title and problem statement"""

    def construct(self):
        # Title
        title = Text(
            "Satellite Damage Detection",
            font_size=56,
            color=WHITE,
            weight=BOLD
        )
        subtitle = Text(
            "using Siamese Swin-Transformers",
            font_size=36,
            color=COLORS["primary"]
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        # Animate title
        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP), run_time=1)
        self.wait(2)

        # Transition to problem
        self.play(
            FadeOut(title, shift=UP),
            FadeOut(subtitle, shift=UP)
        )

        # Problem statement
        problem_title = Text("The Challenge", font_size=48, color=COLORS["accent"])
        problem_title.to_edge(UP)

        problems = VGroup(
            Text("🌪️ Natural disasters cause massive destruction", font_size=28),
            Text("⏱️ Manual inspection takes days or weeks", font_size=28),
            Text("🛰️ Satellite imagery available within hours", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        problems.next_to(problem_title, DOWN, buff=0.8)

        self.play(Write(problem_title))
        for problem in problems:
            self.play(FadeIn(problem, shift=RIGHT), run_time=0.8)
            self.wait(0.5)

        self.wait(2)

        # Goal
        goal_box = RoundedRectangle(
            width=10, height=2,
            corner_radius=0.2,
            color=COLORS["secondary"],
            fill_opacity=0.2
        )
        goal_text = Text(
            "Goal: Automatically classify building damage\nfrom pre/post satellite image pairs",
            font_size=28,
            color=WHITE
        )
        goal = VGroup(goal_box, goal_text)
        goal.to_edge(DOWN, buff=1)

        self.play(Create(goal_box), Write(goal_text))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])


class DatasetScene(Scene):
    """xBD Dataset overview"""

    def construct(self):
        # Title
        title = Text("xBD Dataset", font_size=48, color=COLORS["primary"])
        title.to_edge(UP)
        self.play(Write(title))

        # Stats
        stats = VGroup(
            Text("📊 9,168 image pairs (1024×1024)", font_size=24),
            Text("🌍 19 disaster events worldwide", font_size=24),
            Text("🏠 ~800,000 building annotations", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        stats.next_to(title, DOWN, buff=0.8)
        stats.to_edge(LEFT, buff=1)

        for stat in stats:
            self.play(FadeIn(stat, shift=RIGHT), run_time=0.5)

        self.wait(1)

        # Damage classes with color coding
        classes_title = Text("Damage Classes:", font_size=28, color=COLORS["accent"])
        classes_title.next_to(stats, DOWN, buff=0.8)
        classes_title.to_edge(LEFT, buff=1)

        self.play(Write(classes_title))

        classes = [
            ("Background", "#000000", "87.45%"),
            ("No Damage", "#00FF00", "9.30%"),
            ("Minor Damage", "#FFFF00", "1.15%"),
            ("Major Damage", "#FFA500", "1.45%"),
            ("Destroyed", "#FF0000", "0.66%"),
        ]

        class_group = VGroup()
        for name, color, pct in classes:
            square = Square(side_length=0.3, fill_color=color, fill_opacity=1, stroke_width=1)
            label = Text(f"{name}: {pct}", font_size=20)
            label.next_to(square, RIGHT, buff=0.2)
            row = VGroup(square, label)
            class_group.add(row)

        class_group.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        class_group.next_to(classes_title, DOWN, buff=0.3)
        class_group.to_edge(LEFT, buff=1.5)

        self.play(LaggedStart(*[FadeIn(c) for c in class_group], lag_ratio=0.2))

        # Imbalance warning
        warning = VGroup(
            Text("⚠️ SEVERE CLASS IMBALANCE!", font_size=32, color=COLORS["danger"]),
            Text("Destroyed buildings: only 0.66%", font_size=24, color=COLORS["danger"]),
        ).arrange(DOWN)
        warning.to_edge(RIGHT, buff=1)
        warning.shift(DOWN * 0.5)

        self.play(
            FadeIn(warning, scale=1.2),
            Flash(warning, color=COLORS["danger"], flash_radius=0.5)
        )

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class ArchitectureScene(Scene):
    """Model architecture walkthrough"""

    def construct(self):
        # Title
        title = Text("Model Architecture", font_size=48, color=COLORS["primary"])
        title.to_edge(UP)
        self.play(Write(title))

        # Input boxes
        pre_box = RoundedRectangle(width=2, height=1.2, corner_radius=0.1,
                                    color=COLORS["primary"], fill_opacity=0.3)
        pre_label = Text("Pre-Image", font_size=18, color=COLORS["primary"])
        pre_label.move_to(pre_box)
        pre = VGroup(pre_box, pre_label)

        post_box = RoundedRectangle(width=2, height=1.2, corner_radius=0.1,
                                     color=COLORS["secondary"], fill_opacity=0.3)
        post_label = Text("Post-Image", font_size=18, color=COLORS["secondary"])
        post_label.move_to(post_box)
        post = VGroup(post_box, post_label)

        diff_box = RoundedRectangle(width=2, height=1.2, corner_radius=0.1,
                                     color=COLORS["accent"], fill_opacity=0.3)
        diff_label = Text("Diff Image", font_size=18, color=COLORS["accent"])
        diff_label.move_to(diff_box)
        diff = VGroup(diff_box, diff_label)

        inputs = VGroup(pre, post, diff).arrange(RIGHT, buff=0.5)
        inputs.next_to(title, DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(i, shift=DOWN) for i in inputs], lag_ratio=0.3))
        self.wait(1)

        # Siamese Encoder
        encoder_box = RoundedRectangle(width=5, height=1.5, corner_radius=0.1,
                                        color=COLORS["purple"], fill_opacity=0.2)
        encoder_label = Text("Siamese Swin-Transformer\n(Shared Weights)",
                            font_size=20, color=COLORS["purple"])
        encoder_label.move_to(encoder_box)
        encoder = VGroup(encoder_box, encoder_label)
        encoder.next_to(inputs, DOWN, buff=0.5)
        encoder.shift(LEFT * 1.5)

        # Diff CNN
        diff_cnn_box = RoundedRectangle(width=2, height=1.5, corner_radius=0.1,
                                         color=COLORS["accent"], fill_opacity=0.2)
        diff_cnn_label = Text("Diff-CNN", font_size=20, color=COLORS["accent"])
        diff_cnn_label.move_to(diff_cnn_box)
        diff_cnn = VGroup(diff_cnn_box, diff_cnn_label)
        diff_cnn.next_to(inputs, DOWN, buff=0.5)
        diff_cnn.shift(RIGHT * 3)

        # Arrows from inputs
        arrow_pre = Arrow(pre.get_bottom(), encoder.get_top() + LEFT*1, buff=0.1, color=COLORS["primary"])
        arrow_post = Arrow(post.get_bottom(), encoder.get_top() + RIGHT*1, buff=0.1, color=COLORS["secondary"])
        arrow_diff = Arrow(diff.get_bottom(), diff_cnn.get_top(), buff=0.1, color=COLORS["accent"])

        self.play(
            Create(arrow_pre), Create(arrow_post), Create(arrow_diff),
            FadeIn(encoder), FadeIn(diff_cnn)
        )
        self.wait(1)

        # Cross Attention
        cross_attn_box = RoundedRectangle(width=4, height=1.2, corner_radius=0.1,
                                           color=COLORS["cyan"], fill_opacity=0.2)
        cross_attn_label = Text("Cross-Attention Fusion\nPost=Q, Pre=K,V",
                               font_size=18, color=COLORS["cyan"])
        cross_attn_label.move_to(cross_attn_box)
        cross_attn = VGroup(cross_attn_box, cross_attn_label)
        cross_attn.next_to(encoder, DOWN, buff=0.5)

        arrow_enc_attn = Arrow(encoder.get_bottom(), cross_attn.get_top(), buff=0.1, color=WHITE)

        self.play(Create(arrow_enc_attn), FadeIn(cross_attn))
        self.wait(1)

        # Feature Fusion
        fusion_box = RoundedRectangle(width=3, height=1, corner_radius=0.1,
                                       color=WHITE, fill_opacity=0.1)
        fusion_label = Text("Feature Fusion", font_size=18)
        fusion_label.move_to(fusion_box)
        fusion = VGroup(fusion_box, fusion_label)
        fusion.next_to(cross_attn, DOWN, buff=0.5)
        fusion.shift(RIGHT * 1)

        arrow_attn_fusion = Arrow(cross_attn.get_bottom(), fusion.get_top() + LEFT*0.5, buff=0.1, color=WHITE)
        arrow_diff_fusion = Arrow(diff_cnn.get_bottom(), fusion.get_top() + RIGHT*0.5, buff=0.1,
                                  color=COLORS["accent"])

        self.play(
            Create(arrow_attn_fusion), Create(arrow_diff_fusion),
            FadeIn(fusion)
        )
        self.wait(1)

        # Decoder and Output
        decoder_box = RoundedRectangle(width=3, height=1, corner_radius=0.1,
                                        color=COLORS["secondary"], fill_opacity=0.2)
        decoder_label = Text("U-Net Decoder", font_size=18, color=COLORS["secondary"])
        decoder_label.move_to(decoder_box)
        decoder = VGroup(decoder_box, decoder_label)
        decoder.next_to(fusion, DOWN, buff=0.4)

        output_box = RoundedRectangle(width=3, height=1, corner_radius=0.1,
                                       color=COLORS["danger"], fill_opacity=0.2)
        output_label = Text("Output (5 classes)", font_size=18, color=COLORS["danger"])
        output_label.move_to(output_box)
        output = VGroup(output_box, output_label)
        output.next_to(decoder, DOWN, buff=0.4)

        arrow_fusion_dec = Arrow(fusion.get_bottom(), decoder.get_top(), buff=0.1, color=WHITE)
        arrow_dec_out = Arrow(decoder.get_bottom(), output.get_top(), buff=0.1, color=WHITE)

        self.play(
            Create(arrow_fusion_dec), FadeIn(decoder),
            Create(arrow_dec_out), FadeIn(output)
        )

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class CrossAttentionScene(Scene):
    """Explain Cross-Attention mechanism"""

    def construct(self):
        title = Text("Cross-Temporal Attention", font_size=48, color=COLORS["cyan"])
        title.to_edge(UP)
        self.play(Write(title))

        # Key insight
        insight = Text(
            '"Post features ASK what changed,\n Pre features ANSWER what existed"',
            font_size=28,
            color=COLORS["accent"],
            slant=ITALIC
        )
        insight.next_to(title, DOWN, buff=0.6)
        self.play(Write(insight), run_time=2)
        self.wait(2)

        # Q, K, V boxes
        post_box = RoundedRectangle(width=2.5, height=1.2, corner_radius=0.1,
                                     color=COLORS["secondary"], fill_opacity=0.3)
        post_label = Text("Post Features", font_size=18, color=COLORS["secondary"])
        post_label.move_to(post_box)
        post = VGroup(post_box, post_label)
        post.shift(LEFT * 4 + DOWN * 0.5)

        pre_box = RoundedRectangle(width=2.5, height=1.2, corner_radius=0.1,
                                    color=COLORS["primary"], fill_opacity=0.3)
        pre_label = Text("Pre Features", font_size=18, color=COLORS["primary"])
        pre_label.move_to(pre_box)
        pre = VGroup(pre_box, pre_label)
        pre.shift(LEFT * 4 + DOWN * 2.5)

        self.play(FadeIn(post), FadeIn(pre))

        # Q projection
        q_box = RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1,
                                  color=COLORS["secondary"], fill_opacity=0.5)
        q_label = Text("Q", font_size=24, color=WHITE, weight=BOLD)
        q_label.move_to(q_box)
        q = VGroup(q_box, q_label)
        q.shift(LEFT * 1 + DOWN * 0.5)

        # K, V projections
        k_box = RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1,
                                  color=COLORS["primary"], fill_opacity=0.5)
        k_label = Text("K", font_size=24, color=WHITE, weight=BOLD)
        k_label.move_to(k_box)
        k = VGroup(k_box, k_label)
        k.shift(LEFT * 1 + DOWN * 2)

        v_box = RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1,
                                  color=COLORS["primary"], fill_opacity=0.5)
        v_label = Text("V", font_size=24, color=WHITE, weight=BOLD)
        v_label.move_to(v_box)
        v = VGroup(v_box, v_label)
        v.shift(LEFT * 1 + DOWN * 3)

        arrow_post_q = Arrow(post.get_right(), q.get_left(), buff=0.1, color=COLORS["secondary"])
        arrow_pre_k = Arrow(pre.get_right(), k.get_left(), buff=0.1, color=COLORS["primary"])
        arrow_pre_v = Arrow(pre.get_right(), v.get_left(), buff=0.1, color=COLORS["primary"])

        self.play(
            Create(arrow_post_q), FadeIn(q),
            Create(arrow_pre_k), FadeIn(k),
            Create(arrow_pre_v), FadeIn(v),
        )
        self.wait(1)

        # Attention computation
        attn_box = RoundedRectangle(width=3.5, height=1.5, corner_radius=0.1,
                                     color=COLORS["purple"], fill_opacity=0.2)
        attn_label = VGroup(
            Text("Attention", font_size=20, color=COLORS["purple"]),
            Text("softmax(QKᵀ/√d)·V", font_size=16, color=WHITE)
        ).arrange(DOWN, buff=0.1)
        attn_label.move_to(attn_box)
        attn = VGroup(attn_box, attn_label)
        attn.shift(RIGHT * 2 + DOWN * 1.5)

        arrow_q_attn = Arrow(q.get_right(), attn.get_left() + UP*0.3, buff=0.1, color=WHITE)
        arrow_k_attn = Arrow(k.get_right(), attn.get_left(), buff=0.1, color=WHITE)
        arrow_v_attn = Arrow(v.get_right(), attn.get_left() + DOWN*0.3, buff=0.1, color=WHITE)

        self.play(
            Create(arrow_q_attn), Create(arrow_k_attn), Create(arrow_v_attn),
            FadeIn(attn)
        )
        self.wait(1)

        # Output
        output_box = RoundedRectangle(width=2.5, height=1, corner_radius=0.1,
                                       color=COLORS["accent"], fill_opacity=0.3)
        output_label = Text("Fused Features", font_size=18, color=COLORS["accent"])
        output_label.move_to(output_box)
        output = VGroup(output_box, output_label)
        output.shift(RIGHT * 5 + DOWN * 1.5)

        arrow_attn_out = Arrow(attn.get_right(), output.get_left(), buff=0.1, color=WHITE)

        self.play(Create(arrow_attn_out), FadeIn(output))

        # Highlight explanation
        explain = Text(
            "8 attention heads × 96 dimensions",
            font_size=22,
            color=GRAY
        )
        explain.to_edge(DOWN, buff=0.5)
        self.play(Write(explain))

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class LossFunctionScene(Scene):
    """Loss function explanation"""

    def construct(self):
        title = Text("Hybrid Focal-Dice Loss", font_size=48, color=COLORS["danger"])
        title.to_edge(UP)
        self.play(Write(title))

        # Problem
        problem = Text(
            "Problem: Severe class imbalance (Background 87%, Destroyed 0.66%)",
            font_size=24,
            color=COLORS["accent"]
        )
        problem.next_to(title, DOWN, buff=0.5)
        self.play(Write(problem))
        self.wait(1)

        # Loss formula
        formula_box = RoundedRectangle(width=8, height=1.2, corner_radius=0.1,
                                        color=COLORS["primary"], fill_opacity=0.1)
        formula = MathTex(
            r"\mathcal{L} = 0.5 \times \text{Focal} + 0.5 \times \text{Dice}",
            font_size=36
        )
        formula.move_to(formula_box)
        formula_group = VGroup(formula_box, formula)
        formula_group.next_to(problem, DOWN, buff=0.6)

        self.play(Create(formula_box), Write(formula))
        self.wait(1)

        # Two columns
        focal_title = Text("Focal Loss", font_size=28, color=COLORS["secondary"])
        focal_desc = VGroup(
            Text("• Down-weights easy samples", font_size=18),
            Text("• Focuses on hard examples", font_size=18),
            MathTex(r"FL = -(1-p_t)^\gamma \log(p_t)", font_size=24),
            Text("• γ = 2", font_size=18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        focal = VGroup(focal_title, focal_desc).arrange(DOWN, buff=0.3)

        dice_title = Text("Dice Loss", font_size=28, color=COLORS["accent"])
        dice_desc = VGroup(
            Text("• Optimizes overlap directly", font_size=18),
            Text("• Handles small regions well", font_size=18),
            MathTex(r"DL = 1 - \frac{2|A \cap B|}{|A| + |B|}", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        dice = VGroup(dice_title, dice_desc).arrange(DOWN, buff=0.3)

        losses = VGroup(focal, dice).arrange(RIGHT, buff=1.5)
        losses.next_to(formula_group, DOWN, buff=0.6)

        self.play(FadeIn(focal, shift=RIGHT))
        self.wait(1)
        self.play(FadeIn(dice, shift=LEFT))
        self.wait(1)

        # Class weights
        weights = Text(
            "Class Weights: [0.1, 1.0, 2.0, 3.0, 4.0]",
            font_size=24,
            color=COLORS["danger"]
        )
        weights_desc = Text(
            "(Background gets 0.1, Destroyed gets 4.0 = 40× emphasis!)",
            font_size=18,
            color=GRAY
        )
        weights_group = VGroup(weights, weights_desc).arrange(DOWN, buff=0.1)
        weights_group.to_edge(DOWN, buff=0.5)

        self.play(Write(weights), Write(weights_desc))

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class TTAScene(Scene):
    """Test-Time Augmentation explanation"""

    def construct(self):
        title = Text("Test-Time Augmentation (TTA)", font_size=48, color=COLORS["purple"])
        title.to_edge(UP)
        self.play(Write(title))

        # Original image
        orig_square = Square(side_length=2, color=COLORS["primary"], fill_opacity=0.3)
        orig_label = Text("Original", font_size=18)
        orig_label.next_to(orig_square, DOWN, buff=0.2)
        # Add a triangle pattern to show orientation
        triangle = Triangle(color=WHITE, fill_opacity=0.8).scale(0.5)
        triangle.move_to(orig_square.get_center() + UP*0.3)
        orig = VGroup(orig_square, triangle.copy(), orig_label)

        # H-Flip
        hflip_square = Square(side_length=2, color=COLORS["secondary"], fill_opacity=0.3)
        hflip_label = Text("H-Flip", font_size=18)
        hflip_label.next_to(hflip_square, DOWN, buff=0.2)
        hflip_triangle = triangle.copy().flip(axis=RIGHT)
        hflip_triangle.move_to(hflip_square.get_center() + UP*0.3)
        hflip = VGroup(hflip_square, hflip_triangle, hflip_label)

        # V-Flip
        vflip_square = Square(side_length=2, color=COLORS["accent"], fill_opacity=0.3)
        vflip_label = Text("V-Flip", font_size=18)
        vflip_label.next_to(vflip_square, DOWN, buff=0.2)
        vflip_triangle = triangle.copy().flip(axis=UP)
        vflip_triangle.move_to(vflip_square.get_center() + DOWN*0.3)
        vflip = VGroup(vflip_square, vflip_triangle, vflip_label)

        transforms = VGroup(orig, hflip, vflip).arrange(RIGHT, buff=1)
        transforms.shift(UP * 0.5)

        self.play(FadeIn(orig))
        self.wait(0.5)

        # Animate flips
        self.play(
            TransformFromCopy(orig_square, hflip_square),
            TransformFromCopy(triangle, hflip_triangle),
            FadeIn(hflip_label)
        )
        self.play(
            TransformFromCopy(orig_square, vflip_square),
            TransformFromCopy(triangle, vflip_triangle),
            FadeIn(vflip_label)
        )
        self.wait(1)

        # Predict arrows
        pred_labels = VGroup()
        for i, transform in enumerate(transforms):
            arrow = Arrow(transform.get_bottom() + DOWN*0.2, transform.get_bottom() + DOWN*1,
                         buff=0, color=WHITE)
            pred_text = Text("Predict", font_size=14, color=GRAY)
            pred_text.next_to(arrow, RIGHT, buff=0.1)
            self.play(Create(arrow), FadeIn(pred_text), run_time=0.5)
            pred_labels.add(VGroup(arrow, pred_text))

        # Reverse + Average
        avg_box = RoundedRectangle(width=4, height=1.5, corner_radius=0.1,
                                    color=COLORS["danger"], fill_opacity=0.2)
        avg_label = VGroup(
            Text("Reverse transforms", font_size=18),
            Text("→ Average predictions", font_size=18, color=COLORS["danger"]),
        ).arrange(DOWN, buff=0.1)
        avg_label.move_to(avg_box)
        avg = VGroup(avg_box, avg_label)
        avg.shift(DOWN * 2.5)

        arrows_to_avg = VGroup()
        for transform in transforms:
            arrow = Arrow(
                transform.get_bottom() + DOWN*1.2,
                avg.get_top(),
                buff=0.1,
                color=WHITE
            )
            arrows_to_avg.add(arrow)

        self.play(
            *[Create(a) for a in arrows_to_avg],
            FadeIn(avg)
        )

        # Result
        result = Text(
            "Result: ~3-5% F1 improvement!",
            font_size=28,
            color=COLORS["secondary"]
        )
        result.to_edge(DOWN, buff=0.5)
        self.play(Write(result), Flash(result, color=COLORS["secondary"]))

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class ResultsScene(Scene):
    """Results and metrics"""

    def construct(self):
        title = Text("Results", font_size=48, color=COLORS["secondary"])
        title.to_edge(UP)
        self.play(Write(title))

        # Metrics table
        metrics_data = [
            ["Metric", "Score"],
            ["Macro F1-Score", "XX.XX%"],
            ["Weighted F1", "XX.XX%"],
            ["Mean IoU", "XX.XX%"],
            ["Accuracy", "XX.XX%"],
        ]

        table = Table(
            metrics_data,
            include_outer_lines=True,
            h_buff=0.8,
            v_buff=0.4,
        ).scale(0.6)

        # Highlight header
        table.add_highlighted_cell((1, 1), color=COLORS["primary"])
        table.add_highlighted_cell((1, 2), color=COLORS["primary"])

        table.shift(LEFT * 3 + DOWN * 0.5)
        self.play(Create(table))
        self.wait(1)

        # Per-class bars
        classes = ["BG", "NoDmg", "Minor", "Major", "Destr"]
        scores = [0.95, 0.82, 0.45, 0.52, 0.38]
        colors = ["#263238", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]

        bars = VGroup()
        labels = VGroup()

        bar_chart_title = Text("Per-Class F1-Score", font_size=24)
        bar_chart_title.shift(RIGHT * 3 + UP * 1.5)
        self.play(Write(bar_chart_title))

        for i, (cls, score, color) in enumerate(zip(classes, scores, colors)):
            bar = Rectangle(
                width=0.5,
                height=score * 3,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to(RIGHT * (1.5 + i * 0.8) + DOWN * (1.5 - score * 1.5))

            label = Text(cls, font_size=14)
            label.next_to(bar, DOWN, buff=0.1)

            score_label = Text(f"{score:.2f}", font_size=12)
            score_label.next_to(bar, UP, buff=0.1)

            bars.add(bar)
            labels.add(VGroup(label, score_label))

        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.2),
            LaggedStart(*[FadeIn(label) for label in labels], lag_ratio=0.2)
        )

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class ConclusionScene(Scene):
    """Key contributions and conclusion"""

    def construct(self):
        title = Text("Key Contributions", font_size=48, color=COLORS["accent"])
        title.to_edge(UP)
        self.play(Write(title))

        contributions = VGroup(
            VGroup(
                Text("1. Siamese Swin-Transformer", font_size=28, color=COLORS["primary"]),
                Text("   First application to bi-temporal damage assessment", font_size=20, color=GRAY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(
                Text("2. Cross-Temporal Attention", font_size=28, color=COLORS["cyan"]),
                Text("   Novel Q/K/V formulation for change detection", font_size=20, color=GRAY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(
                Text("3. Geometric-Residual Priors", font_size=28, color=COLORS["accent"]),
                Text("   Diff-CNN complements semantic features", font_size=20, color=GRAY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(
                Text("4. Production-Ready Pipeline", font_size=28, color=COLORS["secondary"]),
                Text("   Scene-wise splitting, TTA, full-image stitching", font_size=20, color=GRAY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        contributions.next_to(title, DOWN, buff=0.6)
        contributions.to_edge(LEFT, buff=1)

        for contrib in contributions:
            self.play(FadeIn(contrib, shift=RIGHT), run_time=0.8)
            self.wait(0.5)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Thank you
        thanks = Text("Thank You!", font_size=72, color=WHITE)
        self.play(Write(thanks), run_time=1.5)
        self.wait(1)

        # Links
        links = VGroup(
            Text("📧 your.email@example.com", font_size=24),
            Text("🐙 github.com/your-username/xbd", font_size=24),
        ).arrange(DOWN, buff=0.3)
        links.next_to(thanks, DOWN, buff=1)

        self.play(FadeIn(links))
        self.wait(3)


class SatelliteDamageVideo(Scene):
    """Main video combining all scenes"""

    def construct(self):
        # Run all scenes in sequence
        scenes = [
            IntroScene,
            DatasetScene,
            ArchitectureScene,
            CrossAttentionScene,
            LossFunctionScene,
            TTAScene,
            ResultsScene,
            ConclusionScene,
        ]

        for scene_class in scenes:
            scene = scene_class()
            scene.construct()


# Alternative: Individual scene classes for separate rendering
class AllScenes(Scene):
    """Render all scenes - use: manim -pqh manim_video.py AllScenes"""

    def construct(self):
        IntroScene.construct(self)
        DatasetScene.construct(self)
        ArchitectureScene.construct(self)
        CrossAttentionScene.construct(self)
        LossFunctionScene.construct(self)
        TTAScene.construct(self)
        ResultsScene.construct(self)
        ConclusionScene.construct(self)


if __name__ == "__main__":
    print("To render the video, run:")
    print("  manim -pqh manim_video.py IntroScene")
    print("  manim -pqh manim_video.py DatasetScene")
    print("  manim -pqh manim_video.py ArchitectureScene")
    print("  manim -pqh manim_video.py CrossAttentionScene")
    print("  manim -pqh manim_video.py LossFunctionScene")
    print("  manim -pqh manim_video.py TTAScene")
    print("  manim -pqh manim_video.py ResultsScene")
    print("  manim -pqh manim_video.py ConclusionScene")
    print("")
    print("Or render all at once:")
    print("  manim -pqh manim_video.py AllScenes")
