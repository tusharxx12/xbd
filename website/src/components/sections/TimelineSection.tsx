'use client';

import { useRef, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const timelineSteps = [
  {
    step: '01',
    title: 'Data Preprocessing',
    description: 'Parse xBD dataset, extract building polygons from WKT annotations, rasterize damage masks, and tile images to 512×512 patches.',
    details: ['Scene-wise splitting prevents data leakage', 'MIN_BUILDING_RATIO filtering', 'Diff image computation'],
    color: '#00d4ff',
  },
  {
    step: '02',
    title: 'Feature Extraction',
    description: 'Siamese Swin-Transformer encoder processes pre and post images through shared weights, extracting hierarchical multi-scale features.',
    details: ['4-stage feature pyramid', 'Shifted window attention', 'ImageNet pretrained backbone'],
    color: '#7c3aed',
  },
  {
    step: '03',
    title: 'Cross-Attention Fusion',
    description: 'Post-disaster features query pre-disaster context through multi-head cross-attention, explicitly modeling temporal change relationships.',
    details: ['8 attention heads', 'Post=Query, Pre=Key/Value', 'FFN with residual connections'],
    color: '#f59e0b',
  },
  {
    step: '04',
    title: 'Damage Classification',
    description: 'U-Net decoder with change-aware skip connections reconstructs full-resolution segmentation mask with 5 damage classes.',
    details: ['Background, No Damage, Minor, Major, Destroyed', 'Bilinear upsampling', 'Hybrid Focal-Dice loss'],
    color: '#10b981',
  },
  {
    step: '05',
    title: 'TTA Inference',
    description: 'Test-time augmentation ensemble with horizontal/vertical flips and tile stitching for 1024×1024 full-image predictions.',
    details: ['3-5% F1 improvement', 'Overlap averaging', 'Smooth boundary predictions'],
    color: '#ef4444',
  },
];

function TimelineItem({ item, index }: { item: typeof timelineSteps[0]; index: number }) {
  const itemRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(itemRef, { once: true, margin: '-100px' });
  const isEven = index % 2 === 0;

  return (
    <motion.div
      ref={itemRef}
      initial={{ opacity: 0, x: isEven ? -50 : 50 }}
      animate={isInView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.8, delay: index * 0.15 }}
      className={`flex items-center gap-8 ${isEven ? 'md:flex-row' : 'md:flex-row-reverse'}`}
    >
      {/* Content Card */}
      <div className="flex-1 card-premium p-8 relative overflow-hidden group">
        {/* Animated border glow */}
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: `linear-gradient(135deg, ${item.color}10, transparent)`,
          }}
        />

        {/* Step number */}
        <div
          className="absolute top-4 right-4 text-6xl font-black opacity-10 select-none"
          style={{ color: item.color }}
        >
          {item.step}
        </div>

        <div className="relative z-10">
          <h3
            className="text-2xl font-bold mb-4"
            style={{ color: item.color }}
          >
            {item.title}
          </h3>
          <p className="text-gray-400 mb-6 leading-relaxed">
            {item.description}
          </p>

          {/* Details */}
          <div className="space-y-2">
            {item.details.map((detail, i) => (
              <div
                key={i}
                className="flex items-center gap-3 text-sm text-gray-500"
              >
                <div
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                {detail}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline Node */}
      <div className="hidden md:flex flex-col items-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={isInView ? { scale: 1 } : {}}
          transition={{ duration: 0.5, delay: index * 0.15 + 0.3 }}
          className="w-16 h-16 rounded-full flex items-center justify-center relative"
          style={{
            background: `linear-gradient(135deg, ${item.color}, ${item.color}80)`,
            boxShadow: `0 0 30px ${item.color}50`,
          }}
        >
          <span className="text-xl font-bold text-white">{item.step}</span>
          {/* Pulse ring */}
          <div
            className="absolute inset-0 rounded-full animate-ping opacity-30"
            style={{ backgroundColor: item.color }}
          />
        </motion.div>
      </div>

      {/* Empty space for alignment */}
      <div className="hidden md:block flex-1" />
    </motion.div>
  );
}

export default function TimelineSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate timeline line
      gsap.fromTo(
        lineRef.current,
        { scaleY: 0, transformOrigin: 'top' },
        {
          scaleY: 1,
          ease: 'none',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top center',
            end: 'bottom center',
            scrub: 1,
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="relative py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0f] via-[#12121a] to-[#0a0a0f]" />

      <div className="relative z-10 max-w-6xl mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-20">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block px-4 py-2 rounded-full glass text-purple-400 text-sm font-medium mb-6"
          >
            How It Works
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="section-title mb-6"
          >
            Processing <span className="text-gradient">Pipeline</span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="section-subtitle mx-auto"
          >
            From raw satellite imagery to pixel-precise damage classification
            in a streamlined end-to-end pipeline.
          </motion.p>
        </div>

        {/* Timeline */}
        <div className="relative">
          {/* Vertical Line */}
          <div
            ref={lineRef}
            className="hidden md:block absolute left-1/2 top-0 bottom-0 w-0.5 -translate-x-1/2"
            style={{
              background: 'linear-gradient(to bottom, #00d4ff, #7c3aed, #f59e0b, #10b981, #ef4444)',
            }}
          />

          {/* Timeline Items */}
          <div className="space-y-16">
            {timelineSteps.map((item, index) => (
              <TimelineItem key={item.step} item={item} index={index} />
            ))}
          </div>
        </div>
      </div>

      {/* Decorative elements */}
      <div className="absolute top-1/2 left-0 w-1/2 h-1/2 bg-gradient-radial from-purple-500/5 to-transparent pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-gradient-radial from-cyan-500/5 to-transparent pointer-events-none" />
    </section>
  );
}
