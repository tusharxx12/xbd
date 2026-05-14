'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';

// Navigation Component
function Navigation() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`nav ${scrolled ? 'scrolled' : ''}`}>
      <div className="nav-logo">SatDamage</div>
      <div className="nav-links">
        <a href="#features" className="nav-link">Features</a>
        <a href="#technology" className="nav-link">Technology</a>
        <a href="#results" className="nav-link">Results</a>
        <a href="https://github.com" className="nav-link">GitHub</a>
      </div>
    </nav>
  );
}

// Hero Section
function HeroSection() {
  return (
    <section className="section" style={{ minHeight: '100vh' }}>
      <div className="section-bg">
        <Image
          src="/images/hero-earth.jpg"
          alt="Earth from space"
          fill
          priority
          style={{ objectFit: 'cover' }}
        />
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,0.5))'
        }} />
      </div>

      <div className="container" style={{ textAlign: 'center' }}>
        <h1 className="hero-title">
          Satellite<br />Damage Detection
        </h1>
        <p className="hero-subtitle">
          AI-powered building damage assessment using satellite imagery
          for rapid post-disaster response
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <a href="#features" className="btn-primary">Explore Research</a>
          <a href="#results" className="btn-secondary">View Demo</a>
        </div>
      </div>

      <div className="scroll-indicator">
        <span className="scroll-text">Scroll</span>
        <div className="scroll-line" />
      </div>
    </section>
  );
}

// Features Section
function FeaturesSection() {
  const features = [
    {
      number: '01',
      title: 'Siamese Architecture',
      description: 'Shared-weight Swin-Transformer encoder processes pre and post-disaster image pairs, extracting hierarchical features while maintaining temporal correspondence.'
    },
    {
      number: '02',
      title: 'Cross-Temporal Attention',
      description: 'Novel attention mechanism where post-disaster features query pre-disaster context, explicitly modeling structural changes and damage patterns.'
    },
    {
      number: '03',
      title: 'Multi-Scale Analysis',
      description: 'U-Net decoder with skip connections enables precise localization from building-level to pixel-level damage classification.'
    },
    {
      number: '04',
      title: 'Real-Time Inference',
      description: 'Optimized pipeline with Test-Time Augmentation delivers production-ready performance for emergency response applications.'
    }
  ];

  return (
    <section id="features" className="section" style={{ background: '#000' }}>
      <div className="section-bg">
        <Image
          src="/images/satellite-view.jpg"
          alt="Satellite view"
          fill
          style={{ objectFit: 'cover', opacity: 0.3 }}
        />
      </div>

      <div className="container">
        <span className="section-label">Capabilities</span>
        <h2 className="section-title">
          Advanced Deep Learning for Disaster Assessment
        </h2>
        <p className="section-description">
          Our system leverages state-of-the-art computer vision to automatically
          detect and classify building damage from satellite imagery.
        </p>

        <div className="feature-grid">
          {features.map((feature) => (
            <div key={feature.number} className="feature-card">
              <div className="feature-number">{feature.number}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Technology Section
function TechnologySection() {
  const steps = [
    {
      step: 'Step 01',
      title: 'Image Acquisition',
      description: 'High-resolution satellite imagery is collected from pre and post-disaster timeframes, capturing the affected regions.'
    },
    {
      step: 'Step 02',
      title: 'Preprocessing Pipeline',
      description: 'Images are tiled, aligned, and normalized. Building polygons are extracted and damage masks are generated for training.'
    },
    {
      step: 'Step 03',
      title: 'Feature Extraction',
      description: 'Siamese Swin-Transformer encodes both images simultaneously, extracting multi-scale features with shared weights.'
    },
    {
      step: 'Step 04',
      title: 'Change Detection',
      description: 'Cross-temporal attention and Diff-CNN branches identify structural changes between the image pairs.'
    },
    {
      step: 'Step 05',
      title: 'Damage Classification',
      description: 'U-Net decoder produces pixel-wise segmentation across 4 damage classes: No Damage, Minor, Major, Destroyed.'
    }
  ];

  return (
    <section id="technology" className="section" style={{ background: '#000' }}>
      <div className="section-bg">
        <Image
          src="/images/space-stars.jpg"
          alt="Space background"
          fill
          style={{ objectFit: 'cover', opacity: 0.4 }}
        />
      </div>

      <div className="container">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '80px', alignItems: 'start' }}>
          <div>
            <span className="section-label">How It Works</span>
            <h2 className="section-title">End-to-End Processing Pipeline</h2>
            <p className="section-description" style={{ marginBottom: 0 }}>
              From raw satellite imagery to actionable damage assessments in minutes,
              not days. Our automated pipeline enables rapid disaster response.
            </p>
          </div>

          <div className="timeline">
            {steps.map((item, index) => (
              <div key={index} className="timeline-item">
                <div className="timeline-step">{item.step}</div>
                <h3 className="timeline-title">{item.title}</h3>
                <p className="timeline-description">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// Results Section
function ResultsSection() {
  const stats = [
    { value: '94.2%', label: 'Overall Accuracy' },
    { value: '0.89', label: 'Mean F1 Score' },
    { value: '0.82', label: 'Mean IoU' },
    { value: '<2s', label: 'Inference Time' }
  ];

  return (
    <section id="results" className="section" style={{ background: '#000' }}>
      <div className="section-bg">
        <Image
          src="/images/city-lights.jpg"
          alt="City lights from space"
          fill
          style={{ objectFit: 'cover', opacity: 0.4 }}
        />
      </div>

      <div className="container" style={{ textAlign: 'center' }}>
        <span className="section-label">Performance</span>
        <h2 className="section-title" style={{ margin: '0 auto 24px', textAlign: 'center' }}>
          State-of-the-Art Results on xBD Dataset
        </h2>
        <p className="section-description" style={{ margin: '0 auto 64px', textAlign: 'center' }}>
          Validated on the xView2 Building Damage Assessment challenge dataset,
          the largest public dataset for building damage detection.
        </p>

        <div className="stats-grid">
          {stats.map((stat, index) => (
            <div key={index}>
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  return (
    <section className="section" style={{ background: '#000', minHeight: 'auto', padding: '160px 24px' }}>
      <div className="section-bg">
        <Image
          src="/images/earth-atmosphere.jpg"
          alt="Earth atmosphere"
          fill
          style={{ objectFit: 'cover', opacity: 0.5 }}
        />
      </div>

      <div className="container" style={{ textAlign: 'center' }}>
        <h2 className="section-title" style={{ margin: '0 auto 24px', textAlign: 'center', maxWidth: '700px' }}>
          Ready to Accelerate Disaster Response?
        </h2>
        <p className="section-description" style={{ margin: '0 auto 48px', textAlign: 'center' }}>
          Access our research, models, and code to help communities
          recover faster from natural disasters.
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <a href="#features" className="btn-secondary">Get Started</a>
          <a href="#technology" className="btn-primary">View Documentation</a>
        </div>
      </div>
    </section>
  );
}

// Footer
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <div className="nav-logo" style={{ marginBottom: '20px' }}>SatDamage</div>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '14px', maxWidth: '300px', lineHeight: 1.7 }}>
              AI-powered satellite imagery analysis for building damage assessment
              and disaster response.
            </p>
          </div>

          <div>
            <div className="footer-title">Research</div>
            <a href="#" className="footer-link">Documentation</a>
            <a href="#" className="footer-link">Model Architecture</a>
            <a href="#" className="footer-link">Dataset</a>
            <a href="#" className="footer-link">Publications</a>
          </div>

          <div>
            <div className="footer-title">Resources</div>
            <a href="#" className="footer-link">GitHub</a>
            <a href="#" className="footer-link">API Reference</a>
            <a href="#" className="footer-link">Examples</a>
            <a href="#" className="footer-link">Tutorials</a>
          </div>

          <div>
            <div className="footer-title">Connect</div>
            <a href="#" className="footer-link">Twitter</a>
            <a href="#" className="footer-link">LinkedIn</a>
            <a href="#" className="footer-link">Email</a>
            <a href="#" className="footer-link">Discord</a>
          </div>
        </div>

        <div style={{
          marginTop: '60px',
          paddingTop: '30px',
          borderTop: '1px solid var(--color-border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '20px'
        }}>
          <span style={{ fontSize: '13px', color: 'var(--color-text-muted)' }}>
            © 2024 Satellite Damage Detection. All rights reserved.
          </span>
          <div style={{ display: 'flex', gap: '24px' }}>
            <a href="#" className="footer-link" style={{ marginBottom: 0 }}>Privacy</a>
            <a href="#" className="footer-link" style={{ marginBottom: 0 }}>Terms</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

// Main Page
export default function Home() {
  return (
    <>
      <Navigation />
      <main>
        <HeroSection />
        <FeaturesSection />
        <TechnologySection />
        <ResultsSection />
        <CTASection />
      </main>
      <Footer />
    </>
  );
}
