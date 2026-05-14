import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ['3000-4cb11f95-6396-4983-9f0c-788fb3222c09.daytonaproxy01.net'],
  // Disable strict mode for Three.js compatibility
  reactStrictMode: false,
};

export default nextConfig;
