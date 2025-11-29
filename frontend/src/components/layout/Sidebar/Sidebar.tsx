import { NavLink, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Home, 
  Sparkles, 
  BookOpen, 
  Settings, 
  LogOut,
  Zap,
  ChevronLeft,
  ChevronRight,
  Languages,
  BookType,
  Volume2,
  MessageSquare,
  TrendingUp,
  CheckCircle,
  Shield,
  Wand2
} from 'lucide-react';
import { cn } from '../../../lib/cn';
import { Avatar } from '../../ui/Avatar/Avatar';
import { Button } from '../../ui/Button/Button';
import { useAuthStore } from '../../../store/authStore';
import toast from 'react-hot-toast';

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

const mainNavItems = [
  { name: 'AI Workspace', to: '/workspace', icon: Wand2 },
  { name: 'Overview', to: '/dashboard', icon: Home },
  { name: 'Playground', to: '/playground', icon: Sparkles },
  { name: 'Library', to: '/library', icon: BookOpen },
];

const toolsNavItems = [
  { name: 'Simplify', to: '/simplify', icon: BookType },
  { name: 'Translate', to: '/translate', icon: Languages },
  { name: 'Text to Speech', to: '/tts', icon: Volume2 },
  { name: 'Q&A', to: '/qa', icon: MessageSquare },
];

const trackingNavItems = [
  { name: 'Progress', to: '/progress', icon: TrendingUp },
  { name: 'Reviews', to: '/reviews', icon: CheckCircle },
];

const secondaryNavItems = [
  { name: 'Settings', to: '/settings', icon: Settings },
  { name: 'Admin', to: '/admin', icon: Shield },
];

/**
 * Premium sidebar navigation with collapsible state and smooth animations.
 * Features gradient active states and user profile section.
 */
export function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    toast.success('Logged out successfully');
    navigate('/login');
  };

  return (
    <aside
      className={cn(
        'flex flex-col h-full bg-[rgb(var(--color-bg-elevated))] border-r border-[rgb(var(--color-border))] transition-all duration-300 ease-out',
        collapsed ? 'w-[72px]' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className={cn(
        'flex items-center h-16 px-4 border-b border-[rgb(var(--color-border))]',
        collapsed ? 'justify-center' : 'justify-between'
      )}>
        <div className={cn('flex items-center gap-3', collapsed && 'justify-center')}>
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25">
            <Zap className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
            >
              <h1 className="text-lg font-bold text-gradient">ShikshaSetu</h1>
              <p className="text-[10px] text-[rgb(var(--color-fg-muted))] -mt-0.5">AI Education</p>
            </motion.div>
          )}
        </div>
        
        {!collapsed && onToggle && (
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            onClick={onToggle}
            className="text-[rgb(var(--color-fg-muted))]"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {/* Main */}
        <div className="mb-2">
          {!collapsed && <p className="px-3 py-1 text-xs font-semibold text-[rgb(var(--color-fg-muted))] uppercase tracking-wider">Main</p>}
        </div>
        {mainNavItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group',
              collapsed && 'justify-center px-0',
              isActive
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/30'
                : 'text-[rgb(var(--color-fg-secondary))] hover:bg-[rgb(var(--color-bg-muted))] hover:text-[rgb(var(--color-fg))]'
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn(
                  'w-5 h-5 shrink-0 transition-transform',
                  !isActive && 'group-hover:scale-110'
                )} />
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="font-medium"
                  >
                    {item.name}
                  </motion.span>
                )}
              </>
            )}
          </NavLink>
        ))}

        {/* Tools */}
        <div className="my-4 h-px bg-[rgb(var(--color-border))]" />
        <div className="mb-2">
          {!collapsed && <p className="px-3 py-1 text-xs font-semibold text-[rgb(var(--color-fg-muted))] uppercase tracking-wider">Tools</p>}
        </div>
        {toolsNavItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group',
              collapsed && 'justify-center px-0',
              isActive
                ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/30'
                : 'text-[rgb(var(--color-fg-secondary))] hover:bg-[rgb(var(--color-bg-muted))] hover:text-[rgb(var(--color-fg))]'
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn(
                  'w-5 h-5 shrink-0 transition-transform',
                  !isActive && 'group-hover:scale-110'
                )} />
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="font-medium"
                  >
                    {item.name}
                  </motion.span>
                )}
              </>
            )}
          </NavLink>
        ))}

        {/* Tracking */}
        <div className="my-4 h-px bg-[rgb(var(--color-border))]" />
        <div className="mb-2">
          {!collapsed && <p className="px-3 py-1 text-xs font-semibold text-[rgb(var(--color-fg-muted))] uppercase tracking-wider">Tracking</p>}
        </div>
        {trackingNavItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group',
              collapsed && 'justify-center px-0',
              isActive
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg shadow-green-500/30'
                : 'text-[rgb(var(--color-fg-secondary))] hover:bg-[rgb(var(--color-bg-muted))] hover:text-[rgb(var(--color-fg))]'
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn(
                  'w-5 h-5 shrink-0 transition-transform',
                  !isActive && 'group-hover:scale-110'
                )} />
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="font-medium"
                  >
                    {item.name}
                  </motion.span>
                )}
              </>
            )}
          </NavLink>
        ))}

        {/* Settings */}
        <div className="my-4 h-px bg-[rgb(var(--color-border))]" />

        {secondaryNavItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group',
              collapsed && 'justify-center px-0',
              isActive
                ? 'bg-[rgb(var(--color-bg-muted))] text-[rgb(var(--color-fg))]'
                : 'text-[rgb(var(--color-fg-secondary))] hover:bg-[rgb(var(--color-bg-muted))] hover:text-[rgb(var(--color-fg))]'
            )}
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="font-medium"
              >
                {item.name}
              </motion.span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Profile */}
      <div className={cn(
        'p-3 border-t border-[rgb(var(--color-border))]',
        collapsed && 'flex flex-col items-center'
      )}>
        {!collapsed ? (
          <div className="p-3 rounded-xl bg-[rgb(var(--color-bg-muted))] mb-2">
            <div className="flex items-center gap-3">
              <Avatar name={user?.full_name || user?.username || 'User'} size="md" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-[rgb(var(--color-fg))] truncate">
                  {user?.full_name || user?.username || 'User'}
                </p>
                <p className="text-xs text-[rgb(var(--color-fg-muted))] truncate">
                  {user?.email}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <Avatar name={user?.full_name || user?.username || 'User'} size="md" className="mb-2" />
        )}
        
        <Button
          variant="ghost"
          size={collapsed ? 'sm' : 'md'}
          fullWidth={!collapsed}
          iconOnly={collapsed}
          onClick={handleLogout}
          className="text-[rgb(var(--color-error-500))] hover:bg-[rgb(var(--color-error-500))]/10"
        >
          <LogOut className="w-4 h-4" />
          {!collapsed && <span>Logout</span>}
        </Button>
      </div>

      {/* Collapse toggle for collapsed state */}
      {collapsed && onToggle && (
        <div className="p-3 border-t border-[rgb(var(--color-border))]">
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            onClick={onToggle}
            className="w-full text-[rgb(var(--color-fg-muted))]"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      )}
    </aside>
  );
}

export default Sidebar;
