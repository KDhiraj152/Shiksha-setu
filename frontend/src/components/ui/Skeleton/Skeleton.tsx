/**
 * Skeleton Loading Components
 * 
 * Smooth shimmer loading placeholders
 */

import React from 'react';
import { motion } from 'framer-motion';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  animation = 'pulse',
}) => {
  const baseClasses = 'bg-gray-200 relative overflow-hidden';
  
  const variantClasses = {
    text: 'rounded h-4',
    circular: 'rounded-full',
    rectangular: '',
    rounded: 'rounded-lg',
  };

  const style: React.CSSProperties = {
    width: width,
    height: height,
  };

  if (variant === 'text' && !height) {
    style.height = '1em';
  }

  if (variant === 'circular') {
    style.width = style.width || '40px';
    style.height = style.height || style.width;
  }

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
    >
      {animation === 'wave' && (
        <motion.div
          className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/40 to-transparent"
          animate={{ x: ['100%', '-100%'] }}
          transition={{
            repeat: Infinity,
            duration: 1.5,
            ease: 'linear',
          }}
        />
      )}
      {animation === 'pulse' && (
        <motion.div
          className="absolute inset-0 bg-gray-300"
          animate={{ opacity: [0.5, 0.8, 0.5] }}
          transition={{
            repeat: Infinity,
            duration: 1.5,
            ease: 'easeInOut',
          }}
        />
      )}
    </div>
  );
};

// Card Skeleton
export const CardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`p-6 rounded-xl border border-gray-100 bg-white space-y-4 ${className}`}>
    <div className="flex items-center gap-4">
      <Skeleton variant="circular" width={48} height={48} />
      <div className="flex-1 space-y-2">
        <Skeleton variant="text" width="60%" />
        <Skeleton variant="text" width="40%" />
      </div>
    </div>
    <Skeleton variant="rounded" height={100} />
    <div className="flex gap-2">
      <Skeleton variant="rounded" width={80} height={32} />
      <Skeleton variant="rounded" width={80} height={32} />
    </div>
  </div>
);

// Stats Card Skeleton
export const StatCardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`p-6 rounded-xl border border-gray-100 bg-white ${className}`}>
    <div className="flex items-center justify-between mb-4">
      <Skeleton variant="circular" width={40} height={40} />
      <Skeleton variant="rounded" width={60} height={24} />
    </div>
    <Skeleton variant="text" width="40%" className="mb-2" />
    <Skeleton variant="text" width="60%" height={32} />
  </div>
);

// Table Row Skeleton
export const TableRowSkeleton: React.FC<{ columns?: number }> = ({ columns = 4 }) => (
  <div className="flex items-center gap-4 p-4 border-b border-gray-100">
    {Array.from({ length: columns }).map((_, i) => (
      <Skeleton 
        key={i} 
        variant="text" 
        width={i === 0 ? '30%' : '20%'} 
      />
    ))}
  </div>
);

// Avatar Skeleton
export const AvatarSkeleton: React.FC<{ size?: number }> = ({ size = 40 }) => (
  <Skeleton variant="circular" width={size} height={size} />
);

// Button Skeleton
export const ButtonSkeleton: React.FC<{ width?: number }> = ({ width = 100 }) => (
  <Skeleton variant="rounded" width={width} height={40} />
);

// Text Block Skeleton
export const TextBlockSkeleton: React.FC<{ lines?: number }> = ({ lines = 3 }) => (
  <div className="space-y-2">
    {Array.from({ length: lines }).map((_, i) => (
      <Skeleton 
        key={i} 
        variant="text" 
        width={i === lines - 1 ? '60%' : '100%'} 
      />
    ))}
  </div>
);

// List Skeleton
export const ListSkeleton: React.FC<{ items?: number; className?: string }> = ({ 
  items = 5, 
  className = '' 
}) => (
  <div className={`space-y-3 ${className}`}>
    {Array.from({ length: items }).map((_, i) => (
      <div key={i} className="flex items-center gap-3 p-3">
        <Skeleton variant="circular" width={36} height={36} />
        <div className="flex-1 space-y-2">
          <Skeleton variant="text" width="70%" />
          <Skeleton variant="text" width="50%" />
        </div>
      </div>
    ))}
  </div>
);

// Grid Skeleton
export const GridSkeleton: React.FC<{ 
  items?: number; 
  columns?: number;
  className?: string;
}> = ({ 
  items = 6, 
  columns = 3,
  className = '' 
}) => (
  <div 
    className={`grid gap-4 ${className}`}
    style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
  >
    {Array.from({ length: items }).map((_, i) => (
      <CardSkeleton key={i} />
    ))}
  </div>
);

export default Skeleton;
