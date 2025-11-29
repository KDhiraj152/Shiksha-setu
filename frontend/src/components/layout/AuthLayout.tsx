import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';

/**
 * Auth layout with beautiful gradient background and grid pattern.
 * Used for login and registration pages.
 */
export function AuthLayout() {
  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute inset-0 bg-grid-pattern opacity-20" />
      
      {/* Gradient orbs */}
      <div className="absolute top-0 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-30 animate-float" />
      <div className="absolute bottom-0 -right-40 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-30 animate-float" style={{ animationDelay: '2s' }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-20" />
      
      {/* Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center p-4 md:p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          className="w-full max-w-md"
        >
          <Outlet />
        </motion.div>
      </div>
    </div>
  );
}

export default AuthLayout;
