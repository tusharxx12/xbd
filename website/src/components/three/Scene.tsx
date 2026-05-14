'use client';

import { Suspense, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  Float,
  Stars,
  Sphere,
  MeshDistortMaterial,
  Line,
} from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { BlendFunction } from 'postprocessing';

// Floating Satellite Component
function Satellite({ position, scale = 1, speed = 1 }: { position: [number, number, number], scale?: number, speed?: number }) {
  const meshRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.002 * speed;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.3 * speed) * 0.3;
    }
  });

  return (
    <group ref={meshRef} position={position} scale={scale}>
      {/* Main body */}
      <mesh>
        <boxGeometry args={[1, 0.5, 0.5]} />
        <meshStandardMaterial
          color="#1a1a2e"
          metalness={0.9}
          roughness={0.1}
          emissive="#00d4ff"
          emissiveIntensity={0.1}
        />
      </mesh>
      {/* Solar panels */}
      <mesh position={[1.2, 0, 0]}>
        <boxGeometry args={[1.5, 0.05, 0.8]} />
        <meshStandardMaterial
          color="#0066cc"
          metalness={0.8}
          roughness={0.2}
          emissive="#00d4ff"
          emissiveIntensity={0.3}
        />
      </mesh>
      <mesh position={[-1.2, 0, 0]}>
        <boxGeometry args={[1.5, 0.05, 0.8]} />
        <meshStandardMaterial
          color="#0066cc"
          metalness={0.8}
          roughness={0.2}
          emissive="#00d4ff"
          emissiveIntensity={0.3}
        />
      </mesh>
      {/* Antenna */}
      <mesh position={[0, 0.4, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 0.5]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      <mesh position={[0, 0.7, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial
          color="#ff3333"
          emissive="#ff0000"
          emissiveIntensity={2}
        />
      </mesh>
    </group>
  );
}

// Earth Component
function Earth({ scrollProgress }: { scrollProgress: number }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
      // Zoom in as user scrolls
      const scale = 1 + scrollProgress * 0.5;
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <mesh ref={meshRef} position={[0, -3, -5]}>
      <sphereGeometry args={[3, 64, 64]} />
      <meshStandardMaterial
        color="#1a4a6e"
        metalness={0.3}
        roughness={0.7}
        emissive="#00d4ff"
        emissiveIntensity={0.05}
      />
    </mesh>
  );
}

// Neural Network Visualization
function NeuralNetwork() {
  const groupRef = useRef<THREE.Group>(null);
  const nodes = useMemo(() => {
    const nodeArray: { position: [number, number, number]; connections: number[] }[] = [];
    const layers = [4, 8, 8, 4];
    let nodeIndex = 0;

    layers.forEach((nodeCount, layerIndex) => {
      for (let i = 0; i < nodeCount; i++) {
        const x = (layerIndex - 1.5) * 2;
        const y = (i - (nodeCount - 1) / 2) * 0.8;
        const z = Math.random() * 0.5 - 0.25;
        nodeArray.push({
          position: [x, y, z],
          connections: layerIndex < layers.length - 1 ?
            Array.from({ length: layers[layerIndex + 1] }, (_, j) => nodeIndex + nodeCount - i + j) : []
        });
        nodeIndex++;
      }
    });
    return nodeArray;
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.2;
      groupRef.current.rotation.x = Math.cos(state.clock.elapsedTime * 0.15) * 0.1;
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, -2]}>
      {nodes.map((node, i) => (
        <Float key={i} speed={2} rotationIntensity={0} floatIntensity={0.5}>
          <mesh position={node.position}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial
              color="#00d4ff"
              emissive="#00d4ff"
              emissiveIntensity={1}
            />
          </mesh>
        </Float>
      ))}
      {/* Connection lines */}
      {nodes.flatMap((node, i) =>
        node.connections.map((targetIndex, j) => {
          if (targetIndex < nodes.length) {
            const start = node.position;
            const end = nodes[targetIndex].position;
            return (
              <Line
                key={`${i}-${j}`}
                points={[start, end]}
                color="#00d4ff"
                lineWidth={1}
                transparent
                opacity={0.3}
              />
            );
          }
          return null;
        })
      )}
    </group>
  );
}

// Particle Field
function ParticleField({ count = 500 }) {
  const mesh = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 30;

      const color = new THREE.Color();
      color.setHSL(0.55 + Math.random() * 0.1, 0.8, 0.6);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [count]);

  useFrame((state) => {
    if (mesh.current) {
      mesh.current.rotation.y = state.clock.elapsedTime * 0.02;
      mesh.current.rotation.x = state.clock.elapsedTime * 0.01;
    }
  });

  return (
    <points ref={mesh} geometry={geometry}>
      <pointsMaterial
        size={0.05}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
}

// Glowing Orb
function GlowingOrb({ position, color = '#00d4ff' }: { position: [number, number, number], color?: string }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 2) * 0.1);
    }
  });

  return (
    <Float speed={3} rotationIntensity={0.5} floatIntensity={1}>
      <Sphere ref={meshRef} args={[0.5, 32, 32]} position={position}>
        <MeshDistortMaterial
          color={color}
          attach="material"
          distort={0.4}
          speed={2}
          roughness={0}
          metalness={0.5}
          emissive={color}
          emissiveIntensity={0.5}
        />
      </Sphere>
    </Float>
  );
}

// Camera Controller
function CameraController({ scrollProgress }: { scrollProgress: number }) {
  const { camera } = useThree();

  useFrame(() => {
    // Cinematic camera movement based on scroll
    camera.position.z = 10 - scrollProgress * 5;
    camera.position.y = scrollProgress * 2;
    camera.rotation.x = -scrollProgress * 0.1;
  });

  return null;
}

// Main Scene Component
interface MainSceneProps {
  scrollProgress?: number;
}

function MainScene({ scrollProgress = 0 }: MainSceneProps) {
  return (
    <>
      <CameraController scrollProgress={scrollProgress} />

      {/* Lighting */}
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#00d4ff" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#7c3aed" />
      <spotLight
        position={[0, 20, 0]}
        angle={0.3}
        penumbra={1}
        intensity={0.5}
        color="#ffffff"
      />

      {/* Background */}
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      <ParticleField />

      {/* Main Objects */}
      <Earth scrollProgress={scrollProgress} />
      <Satellite position={[3, 2, 0]} scale={0.5} speed={1.2} />
      <Satellite position={[-4, 1, -2]} scale={0.3} speed={0.8} />
      <Satellite position={[2, -1, -3]} scale={0.4} speed={1} />

      {/* Glowing orbs */}
      <GlowingOrb position={[-5, 3, -5]} color="#00d4ff" />
      <GlowingOrb position={[6, -2, -4]} color="#7c3aed" />
      <GlowingOrb position={[0, 4, -6]} color="#10b981" />

      {/* Neural Network */}
      <NeuralNetwork />

      {/* Post Processing */}
      <EffectComposer>
        <Bloom
          intensity={1.5}
          luminanceThreshold={0.1}
          luminanceSmoothing={0.9}
          mipmapBlur
        />
        <ChromaticAberration
          offset={new THREE.Vector2(0.001, 0.001)}
          blendFunction={BlendFunction.NORMAL}
        />
        <Vignette eskil={false} offset={0.1} darkness={0.5} />
      </EffectComposer>
    </>
  );
}

// Loading Fallback
function Loader() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshBasicMaterial color="#00d4ff" wireframe />
    </mesh>
  );
}

// Export Scene Component
export default function Scene({ scrollProgress = 0 }: { scrollProgress?: number }) {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas
        camera={{ position: [0, 0, 10], fov: 60 }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        }}
        dpr={[1, 2]}
      >
        <Suspense fallback={<Loader />}>
          <MainScene scrollProgress={scrollProgress} />
        </Suspense>
      </Canvas>
    </div>
  );
}
