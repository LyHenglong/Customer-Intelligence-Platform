import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lean production image (docker/Dockerfile.frontend copies only
  // .next/standalone + .next/static + public, not the full node_modules).
  output: "standalone",
};

export default nextConfig;
