import { NavLink } from 'react-router-dom';
import { Home, Sparkles, BookOpen, MoreHorizontal } from 'lucide-react';
import { cn } from '../../lib/cn';

const navItems = [
  { name: 'Home', to: '/app/dashboard', icon: Home },
  { name: 'Playground', to: '/app/playground', icon: Sparkles },
  { name: 'Library', to: '/app/library', icon: BookOpen },
  { name: 'More', to: '/app/settings', icon: MoreHorizontal },
];

/**
 * Mobile bottom navigation bar with smooth active state animations.
 */
export function MobileNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 h-16 bg-[rgb(var(--color-bg-elevated))]/95 backdrop-blur-xl border-t border-[rgb(var(--color-border))] safe-area-inset-bottom z-30 lg:hidden">
      <div className="flex items-center justify-around h-full px-2">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) => cn(
              'flex flex-col items-center justify-center gap-1 px-3 py-2 rounded-xl transition-all min-w-[64px]',
              isActive
                ? 'text-[rgb(var(--color-primary-500))]'
                : 'text-[rgb(var(--color-fg-muted))]'
            )}
          >
            {({ isActive }) => (
              <>
                <div className={cn(
                  'p-1.5 rounded-lg transition-all',
                  isActive && 'bg-[rgb(var(--color-primary-500))]/10'
                )}>
                  <item.icon className="w-5 h-5" />
                </div>
                <span className="text-[10px] font-medium">{item.name}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

export default MobileNav;
