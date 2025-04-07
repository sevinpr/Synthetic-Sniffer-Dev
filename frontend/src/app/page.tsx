'use client';

import { lazy, Suspense } from 'react';
import { CircularProgress, Box } from '@mui/material';

// Lazy load the ImageUploader component
const ImageUploader = lazy(() => import('@Sniffer/components/ImageUploader'));

export default function Home() {
  return (
    <main className="min-h-screen p-4 sm:p-8">
      <Suspense fallback={
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      }>
        <ImageUploader />
      </Suspense>
    </main>
  );
}
