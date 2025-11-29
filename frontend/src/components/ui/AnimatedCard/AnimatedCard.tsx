/**
 * Animated Card Component
 * 
 * Card with smooth hover and tap animations
 */

import React from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';
import { scaleIn, cardHover, tapScale } from '../../../lib/animations';

interface AnimatedCardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  className?: string;
  interactive?: boolean;
  delay?: number;
}

export const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  className = '',
  interactive = true,
  delay = 0,
  ...props
}) => {
  return (
    <motion.div
      variants={scaleIn}
      initial="initial"
      animate="animate"
      exit="exit"
      whileHover={interactive ? cardHover : undefined}
      whileTap={interactive ? tapScale : undefined}
      transition={{ delay }}
      className={`rounded-xl bg-white border border-gray-100 shadow-sm overflow-hidden ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default AnimatedCard;
