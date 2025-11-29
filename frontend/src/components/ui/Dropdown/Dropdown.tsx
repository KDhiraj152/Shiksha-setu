/**
 * Dropdown Menu Component
 * 
 * Accessible dropdown with smooth animations
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check } from 'lucide-react';
import { mobileMenuVariants } from '../../../lib/animations';

interface DropdownItem {
  label: string;
  value: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
}

interface DropdownProps {
  trigger: React.ReactNode;
  items: DropdownItem[];
  onSelect: (value: string) => void;
  selected?: string;
  align?: 'left' | 'right';
  className?: string;
}

export const Dropdown: React.FC<DropdownProps> = ({
  trigger,
  items,
  onSelect,
  selected,
  align = 'left',
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div 
      ref={dropdownRef} 
      className={`relative inline-block ${className}`}
      onKeyDown={handleKeyDown}
    >
      <div onClick={() => setIsOpen(!isOpen)}>
        {trigger}
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            variants={mobileMenuVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className={`
              absolute z-50 mt-2 min-w-[180px] 
              bg-white rounded-xl shadow-lg border border-gray-100
              py-1 overflow-hidden
              ${align === 'right' ? 'right-0' : 'left-0'}
            `}
          >
            {items.map((item) => (
              <button
                key={item.value}
                onClick={() => {
                  if (!item.disabled) {
                    onSelect(item.value);
                    setIsOpen(false);
                  }
                }}
                disabled={item.disabled}
                className={`
                  w-full px-4 py-2.5 flex items-center gap-3
                  text-sm text-left transition-colors
                  ${item.disabled 
                    ? 'text-gray-300 cursor-not-allowed' 
                    : item.danger 
                      ? 'text-red-600 hover:bg-red-50' 
                      : 'text-gray-700 hover:bg-gray-50'
                  }
                  ${selected === item.value ? 'bg-primary-50' : ''}
                `}
              >
                {item.icon && (
                  <span className="flex-shrink-0 w-5 h-5">
                    {item.icon}
                  </span>
                )}
                <span className="flex-1">{item.label}</span>
                {selected === item.value && (
                  <Check size={16} className="text-primary-500" />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Simple dropdown button variant
interface DropdownButtonProps {
  label: string;
  items: DropdownItem[];
  onSelect: (value: string) => void;
  selected?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const DropdownButton: React.FC<DropdownButtonProps> = ({
  label,
  items,
  onSelect,
  selected,
  variant = 'secondary',
  size = 'md',
  className = '',
}) => {
  const variantClasses = {
    primary: 'bg-primary-500 text-white hover:bg-primary-600',
    secondary: 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50',
    ghost: 'text-gray-600 hover:bg-gray-100',
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base',
  };

  const selectedItem = items.find(item => item.value === selected);

  return (
    <Dropdown
      items={items}
      onSelect={onSelect}
      selected={selected}
      className={className}
      trigger={
        <button
          className={`
            inline-flex items-center gap-2 rounded-lg font-medium
            transition-colors
            ${variantClasses[variant]}
            ${sizeClasses[size]}
          `}
        >
          {selectedItem?.label || label}
          <ChevronDown size={16} />
        </button>
      }
    />
  );
};

export default Dropdown;
