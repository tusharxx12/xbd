'use client';

import { useRef, useEffect, useState } from 'react';
import { motion, useInView } from 'framer-motion';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const damageClasses = [
  { name: 'Background', color: '#1a1a2e', percentage: 87.45 },
  { name: 'No Damage', color: '#22c55e', percentage: 9.30 },
  { name: 'Minor Damage', color: '#eab308', percentage: 1.15 },
  { name: 'Major Damage', color: '#f97316', percentage: 1.45 },
  { name: 'Destroyed', color: '#ef4444', percentage: 0.66 },
];

const metrics = [
  { label: 'Macro F1-Score', value: 'XX.XX', suffix: '%', color: '#00d4ff' },
  { label: 'Weighted F1', value: 'XX.XX', suffix: '%', color: '#7c3aed' },
  { label: 'Mean IoU', value: 'XX.XX', suffix: '%', color: '#10b981' },
  { label: 'Accuracy', value: 'XX.XX', suffix: '%', color: '#f59e0b' },
];

function AnimatedCounter({ value, suffix, color }: { value: string; suffix: string; color: string }) {
  const [displayValue, setDisplayValue] = useState('0.00');
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (isInView) {
      // Simulate counting animation
      const duration = 2000;
      const steps = 60;
      const stepDuration = duration / steps;
      let step = 0;

      const interval = setInterval(() => {
        step++;
        const progress = step / steps;
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = (parseFloat(value.replace('XX', '85')) * eased).toFixed(2);
        setDisplayValue(current);

        if (step >= steps) {
          clearInterval(interval);
          setDisplayValue(value);
        }
      }, stepDuration);

      return () => clearInterval(interval);
    }
  }, [isInView, value]);

  return (
    <span ref={ref} style={{ color }}>
      {displayValue}{suffix}
    </span>
  );
}

function ClassDistributionBar({ item, index }: { item: typeof damageClasses[0]; index: number }) {
  const barRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(barRef, { once: true });

  return (
    <motion.div
      ref={barRef}
      initial={{ opacity: 0, x: -30 }}
      animate={isInView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1 }}
      className="space-y-2"
    >
      <div className="flex justify-between text-sm">
        <span className="text-gray-300">{item.name}</span>
        <span style={{ color: item.color }}>{item.percentage}%</span>
      </div>
      <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={isInView ? { width: `${Math.min(item.percentage, 100)}%` } : {}}
          transition={{ duration: 1, delay: index * 0.1 + 0.3, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{
            backgroundColor: item.color,
            boxShadow: `0 0 20px ${item.color}50`,
          }}
        />
      </div>
    </motion.div>
  );
}

export default function ResultsSection() {
  const sectionRef = useRef<HTMLElement>(null);

  return (
    <section ref={sectionRef} className="relative py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0f] via-[#0f0f18] to-[#0a0a0f]" />

      <div className="relative z-10 max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-20">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block px-4 py-2 rounded-full glass text-emerald-400 text-sm font-medium mb-6"
          >
            Performance Metrics
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="section-title mb-6"
          >
            Research <span className="text-gradient">Results</span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="section-subtitle mx-auto"
          >
            Comprehensive evaluation on the xBD validation set demonstrates
            competitive performance across all damage classes.
          </motion.p>
        </div>

        {/* Main Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="card-premium p-6 text-center group"
            >
              <div
                className="text-4xl md:text-5xl font-black mb-2"
                style={{ color: metric.color }}
              >
                <AnimatedCounter value={metric.value} suffix={metric.suffix} color={metric.color} />
              </div>
              <div className="text-gray-400 text-sm">{metric.label}</div>

              {/* Glow on hover */}
              <div
                className="absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10 blur-xl"
                style={{ backgroundColor: `${metric.color}20` }}
              />
            </motion.div>
          ))}
        </div>

        {/* Two Column Layout */}
        <div className="grid md:grid-cols-2 gap-12">
          {/* Class Distribution */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="card-premium p-8"
          >
            <h3 className="text-2xl font-bold mb-2">Class Distribution</h3>
            <p className="text-gray-400 text-sm mb-8">
              xBD dataset exhibits severe class imbalance requiring specialized training strategies.
            </p>

            <div className="space-y-6">
              {damageClasses.map((item, index) => (
                <ClassDistributionBar key={item.name} item={item} index={index} />
              ))}
            </div>

            <div className="mt-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <div className="font-semibold text-red-400">Extreme Imbalance</div>
                  <div className="text-sm text-gray-400">Destroyed buildings: only 0.66% of pixels</div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Per-Class Performance */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="card-premium p-8"
          >
            <h3 className="text-2xl font-bold mb-2">Per-Class F1-Score</h3>
            <p className="text-gray-400 text-sm mb-8">
              Individual class performance showing the challenge of minority classes.
            </p>

            <div className="space-y-4">
              {damageClasses.map((item, index) => (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center gap-4 p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <div
                    className="w-4 h-4 rounded-full flex-shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="flex-1 text-gray-300">{item.name}</span>
                  <div className="text-right">
                    <div className="font-bold" style={{ color: item.color }}>XX.XX%</div>
                    <div className="text-xs text-gray-500">F1-Score</div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Note */}
            <div className="mt-8 p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/20">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <div className="font-semibold text-cyan-400">Results Pending</div>
                  <div className="text-sm text-gray-400">Metrics will be updated after full training</div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Visual Results Preview */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 p-8 rounded-3xl glass-strong"
        >
          <h3 className="text-2xl font-bold text-center mb-8">Sample Predictions</h3>

          <div className="grid grid-cols-4 gap-4 text-center text-sm mb-4">
            <div className="text-cyan-400 font-medium">Pre-Disaster</div>
            <div className="text-emerald-400 font-medium">Post-Disaster</div>
            <div className="text-purple-400 font-medium">Ground Truth</div>
            <div className="text-orange-400 font-medium">Prediction</div>
          </div>

          <div className="grid grid-cols-4 gap-4">
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="aspect-square rounded-xl bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center"
              >
                <div className="text-gray-600 text-xs">Coming Soon</div>
              </div>
            ))}
          </div>

          <p className="text-center text-gray-400 text-sm mt-6">
            Visual comparison of model predictions on validation samples
          </p>
        </motion.div>
      </div>

      {/* Decorative */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
    </section>
  );
}
