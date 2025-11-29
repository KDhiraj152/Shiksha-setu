/**
 * Animated Page Wrapper
 * 
 * Provides smooth page transitions with Framer Motion
 */

import React from 'react';
import { motion } from 'framer-motion';
import { pageVariants } from '../../../lib/animations';

interface PageTransitionProps {
  children: React.ReactNode;
  className?: string;
}

export const PageTransition: React.FC<PageTransitionProps> = ({
  children,
  className = '',
}) => {
  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="enter"
      exit="exit"
      className={className}
    >
      {children}
    </motion.div>
  );
};

export default PageTransition;
