/**
 * Animated List Component
 * 
 * Staggers child animations for smooth list reveals
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { staggerContainer, staggerItem } from '../../../lib/animations';

interface AnimatedListProps {
  children: React.ReactNode;
  className?: string;
  as?: 'div' | 'ul' | 'ol' | 'section';
}

export const AnimatedList: React.FC<AnimatedListProps> = ({
  children,
  className = '',
  as = 'div',
}) => {
  const components = {
    div: motion.div,
    ul: motion.ul,
    ol: motion.ol,
    section: motion.section,
  };
  const Component = components[as];
  
  return (
    <Component
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      exit="exit"
      className={className}
    >
      <AnimatePresence mode="popLayout">
        {children}
      </AnimatePresence>
    </Component>
  );
};

interface AnimatedListItemProps {
  children: React.ReactNode;
  className?: string;
  layoutId?: string;
}

export const AnimatedListItem: React.FC<AnimatedListItemProps> = ({
  children,
  className = '',
  layoutId,
}) => {
  return (
    <motion.div
      variants={staggerItem}
      layout
      layoutId={layoutId}
      className={className}
    >
      {children}
    </motion.div>
  );
};

export default AnimatedList;
