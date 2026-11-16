import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The phone app is served by the OpenJarvis server itself at /m, so every
// asset URL is relative to that prefix and API calls stay same-origin.
const apiTarget = process.env.OPENJARVIS_VITE_PROXY_TARGET || 'http://localhost:8787';

export default defineConfig({
  base: '/m/',
  plugins: [react()],
  build: {
    outDir: '../src/openjarvis/server/static_mobile',
    emptyOutDir: true,
    target: ['es2020', 'safari14'],
  },
  server: {
    port: 5180,
    proxy: {
      '/v1': { target: apiTarget, changeOrigin: true },
      '/health': apiTarget,
    },
  },
});
