'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';

export default function DemoPage() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: 'Pre-Disaster Satellite Image',
      subtitle: 'Baseline imagery before the event',
      description: 'High-resolution satellite image captured before the disaster. This serves as the baseline for comparison, showing intact buildings and infrastructure from an aerial perspective.',
    },
    {
      title: 'Post-Disaster Satellite Image',
      subtitle: 'Imagery captured after the event',
      description: 'Same geographic region captured after the disaster event. The model will analyze structural changes between these two temporal snapshots.',
    },
    {
      title: 'Change Detection Analysis',
      subtitle: 'AI identifies structural changes',
      description: 'Our Siamese Swin-Transformer compares pre and post images using cross-temporal attention to detect areas where buildings have been affected.',
    },
    {
      title: 'Damage Classification Result',
      subtitle: 'Per-building damage assessment',
      description: 'Final segmentation output classifies each building into damage categories. Colors indicate severity: Green (No Damage), Yellow (Minor), Orange (Major), Red (Destroyed).',
    },
  ];

  return (
    <div style={{ minHeight: '100vh', background: '#000', color: '#fff' }}>
      {/* Navigation */}
      <nav style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        padding: '20px 40px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(0,0,0,0.9)',
        backdropFilter: 'blur(10px)'
      }}>
        <Link href="/" style={{
          fontSize: '20px',
          fontWeight: 700,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          textDecoration: 'none',
          color: '#fff'
        }}>
          SatDamage
        </Link>
        <Link href="/" className="btn-primary" style={{
          padding: '12px 24px',
          border: '2px solid #fff',
          background: 'transparent',
          color: '#fff',
          fontSize: '12px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          textDecoration: 'none',
          transition: 'all 0.3s'
        }}>
          Back to Home
        </Link>
      </nav>

      {/* Main Content */}
      <main style={{ paddingTop: '100px', padding: '120px 40px 80px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '60px' }}>
            <p style={{
              fontSize: '12px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              color: '#a0a0a0',
              marginBottom: '16px'
            }}>
              Live Demonstration
            </p>
            <h1 style={{
              fontSize: 'clamp(32px, 5vw, 56px)',
              fontWeight: 700,
              marginBottom: '24px'
            }}>
              Satellite Damage Detection
            </h1>
            <p style={{
              fontSize: '18px',
              color: '#a0a0a0',
              maxWidth: '600px',
              margin: '0 auto',
              lineHeight: 1.7
            }}>
              See how our AI model processes satellite imagery to detect and classify building damage
            </p>
          </div>

          {/* Step indicators */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '8px',
            marginBottom: '48px'
          }}>
            {steps.map((step, index) => (
              <button
                key={index}
                onClick={() => setActiveStep(index)}
                style={{
                  width: index === activeStep ? '48px' : '12px',
                  height: '4px',
                  background: index === activeStep ? '#fff' : 'rgba(255,255,255,0.3)',
                  border: 'none',
                  borderRadius: '2px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              />
            ))}
          </div>

          {/* Demo Content */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1.2fr 1fr',
            gap: '60px',
            alignItems: 'center'
          }}>
            {/* Image Display */}
            <div>
              <div style={{
                position: 'relative',
                aspectRatio: '4/3',
                background: '#111',
                borderRadius: '8px',
                overflow: 'hidden',
                border: '1px solid rgba(255,255,255,0.1)'
              }}>
                {/* Step 0: Pre-disaster */}
                {activeStep === 0 && (
                  <Image
                    src="/images/pre-disaster.jpg"
                    alt="Pre-disaster satellite image"
                    fill
                    style={{ objectFit: 'cover' }}
                  />
                )}

                {/* Step 1: Post-disaster */}
                {activeStep === 1 && (
                  <Image
                    src="/images/post-disaster.jpg"
                    alt="Post-disaster satellite image"
                    fill
                    style={{ objectFit: 'cover' }}
                  />
                )}

                {/* Step 2: Change detection */}
                {activeStep === 2 && (
                  <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                    <Image
                      src="/images/post-disaster.jpg"
                      alt="Change detection analysis"
                      fill
                      style={{ objectFit: 'cover' }}
                    />
                    {/* Animated detection boxes */}
                    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                      <rect x="15%" y="20%" width="20%" height="25%" fill="none" stroke="#ff6b6b" strokeWidth="2" strokeDasharray="8,4">
                        <animate attributeName="stroke-dashoffset" from="0" to="12" dur="0.8s" repeatCount="indefinite" />
                      </rect>
                      <rect x="45%" y="35%" width="25%" height="30%" fill="none" stroke="#ffa500" strokeWidth="2" strokeDasharray="8,4">
                        <animate attributeName="stroke-dashoffset" from="0" to="12" dur="0.8s" repeatCount="indefinite" />
                      </rect>
                      <rect x="25%" y="55%" width="18%" height="20%" fill="none" stroke="#ff6b6b" strokeWidth="2" strokeDasharray="8,4">
                        <animate attributeName="stroke-dashoffset" from="0" to="12" dur="0.8s" repeatCount="indefinite" />
                      </rect>
                      <rect x="60%" y="15%" width="22%" height="22%" fill="none" stroke="#22c55e" strokeWidth="2" strokeDasharray="8,4">
                        <animate attributeName="stroke-dashoffset" from="0" to="12" dur="0.8s" repeatCount="indefinite" />
                      </rect>
                    </svg>
                    <div style={{
                      position: 'absolute',
                      bottom: '16px',
                      left: '16px',
                      background: 'rgba(0,0,0,0.8)',
                      padding: '8px 16px',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      Analyzing changes...
                    </div>
                  </div>
                )}

                {/* Step 3: Final classification */}
                {activeStep === 3 && (
                  <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                    <Image
                      src="/images/post-disaster.jpg"
                      alt="Damage classification result"
                      fill
                      style={{ objectFit: 'cover', opacity: 0.6 }}
                    />
                    {/* Segmentation overlay */}
                    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                      {/* Destroyed - Red */}
                      <rect x="15%" y="20%" width="20%" height="25%" fill="rgba(239,68,68,0.7)" rx="2" />
                      <rect x="25%" y="55%" width="18%" height="20%" fill="rgba(239,68,68,0.7)" rx="2" />
                      {/* Major Damage - Orange */}
                      <rect x="45%" y="35%" width="25%" height="30%" fill="rgba(249,115,22,0.7)" rx="2" />
                      {/* Minor Damage - Yellow */}
                      <rect x="8%" y="50%" width="12%" height="15%" fill="rgba(234,179,8,0.7)" rx="2" />
                      <rect x="75%" y="45%" width="15%" height="18%" fill="rgba(234,179,8,0.7)" rx="2" />
                      {/* No Damage - Green */}
                      <rect x="60%" y="15%" width="22%" height="22%" fill="rgba(34,197,94,0.6)" rx="2" />
                      <rect x="78%" y="70%" width="14%" height="16%" fill="rgba(34,197,94,0.6)" rx="2" />
                    </svg>
                  </div>
                )}

                {/* Step label */}
                <div style={{
                  position: 'absolute',
                  top: '16px',
                  left: '16px',
                  background: 'rgba(0,0,0,0.85)',
                  padding: '10px 18px',
                  borderRadius: '4px',
                  fontSize: '13px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em'
                }}>
                  Step {activeStep + 1} / {steps.length}
                </div>
              </div>
            </div>

            {/* Description Panel */}
            <div>
              <h2 style={{
                fontSize: '36px',
                fontWeight: 600,
                marginBottom: '12px'
              }}>
                {steps[activeStep].title}
              </h2>
              <p style={{
                fontSize: '14px',
                color: '#a0a0a0',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                marginBottom: '28px'
              }}>
                {steps[activeStep].subtitle}
              </p>
              <p style={{
                fontSize: '16px',
                color: '#a0a0a0',
                lineHeight: 1.8,
                marginBottom: '40px'
              }}>
                {steps[activeStep].description}
              </p>

              {/* Legend for final step */}
              {activeStep === 3 && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '16px',
                  padding: '24px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.1)',
                  marginBottom: '40px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '20px', height: '20px', background: '#22c55e', borderRadius: '3px' }} />
                    <span style={{ fontSize: '14px' }}>No Damage</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '20px', height: '20px', background: '#eab308', borderRadius: '3px' }} />
                    <span style={{ fontSize: '14px' }}>Minor Damage</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '20px', height: '20px', background: '#f97316', borderRadius: '3px' }} />
                    <span style={{ fontSize: '14px' }}>Major Damage</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '20px', height: '20px', background: '#ef4444', borderRadius: '3px' }} />
                    <span style={{ fontSize: '14px' }}>Destroyed</span>
                  </div>
                </div>
              )}

              {/* Navigation buttons */}
              <div style={{ display: 'flex', gap: '16px' }}>
                <button
                  onClick={() => setActiveStep((prev) => (prev - 1 + steps.length) % steps.length)}
                  style={{
                    padding: '14px 32px',
                    background: 'transparent',
                    border: '2px solid #fff',
                    color: '#fff',
                    fontSize: '13px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  Previous
                </button>
                <button
                  onClick={() => setActiveStep((prev) => (prev + 1) % steps.length)}
                  style={{
                    padding: '14px 32px',
                    background: '#fff',
                    border: '2px solid #fff',
                    color: '#000',
                    fontSize: '13px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  Next Step
                </button>
              </div>
            </div>
          </div>

          {/* Technical Details */}
          <div style={{
            marginTop: '100px',
            paddingTop: '60px',
            borderTop: '1px solid rgba(255,255,255,0.1)'
          }}>
            <h3 style={{
              fontSize: '12px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
              color: '#a0a0a0',
              marginBottom: '32px'
            }}>
              Technical Specifications
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '40px'
            }}>
              <div>
                <div style={{ fontSize: '32px', fontWeight: 700, marginBottom: '8px' }}>512×512</div>
                <div style={{ fontSize: '13px', color: '#a0a0a0', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Input Resolution</div>
              </div>
              <div>
                <div style={{ fontSize: '32px', fontWeight: 700, marginBottom: '8px' }}>4</div>
                <div style={{ fontSize: '13px', color: '#a0a0a0', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Damage Classes</div>
              </div>
              <div>
                <div style={{ fontSize: '32px', fontWeight: 700, marginBottom: '8px' }}>&lt;2s</div>
                <div style={{ fontSize: '13px', color: '#a0a0a0', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Inference Time</div>
              </div>
              <div>
                <div style={{ fontSize: '32px', fontWeight: 700, marginBottom: '8px' }}>94.2%</div>
                <div style={{ fontSize: '13px', color: '#a0a0a0', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Accuracy</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
