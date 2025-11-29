import { useLocation } from 'react-router-dom';
import { Bell, Search, Moon, Sun, Menu } from 'lucide-react';
import { Button } from '../../ui/Button/Button';
import { SearchInput } from '../../patterns/SearchInput';
import { useState } from 'react';

interface HeaderProps {
  onMenuClick?: () => void;
  showMenuButton?: boolean;
}

const pageTitles: Record<string, string> = {
  '/app/dashboard': 'Dashboard',
  '/app/playground': 'AI Playground',
  '/app/library': 'Content Library',
  '/app/settings': 'Settings',
  '/app/help': 'Help & Support',
};

/**
 * Premium header with dynamic page title, search, and actions.
 * Features glass effect and smooth transitions.
 */
export function Header({ onMenuClick, showMenuButton }: HeaderProps) {
  const location = useLocation();
  const title = pageTitles[location.pathname] || 'ShikshaSetu';
  const [isDark, setIsDark] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <header className="h-16 px-4 md:px-6 flex items-center justify-between gap-4 bg-[rgb(var(--color-bg-elevated))]/80 backdrop-blur-xl border-b border-[rgb(var(--color-border))]">
      <div className="flex items-center gap-4">
        {showMenuButton && (
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            onClick={onMenuClick}
            className="lg:hidden"
          >
            <Menu className="w-5 h-5" />
          </Button>
        )}
        
        <div>
          <h2 className="text-lg md:text-xl font-bold text-[rgb(var(--color-fg))]">
            {title}
          </h2>
        </div>
      </div>

      <div className="flex items-center gap-2 md:gap-4">
        {/* Search - Hidden on mobile */}
        <div className="hidden md:block w-64 lg:w-80">
          <SearchInput
            placeholder="Search content..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onClear={() => setSearchQuery('')}
            inputSize="sm"
          />
        </div>

        {/* Mobile search button */}
        <Button
          variant="ghost"
          size="sm"
          iconOnly
          className="md:hidden"
        >
          <Search className="w-5 h-5" />
        </Button>

        {/* Theme toggle */}
        <Button
          variant="ghost"
          size="sm"
          iconOnly
          onClick={toggleTheme}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? (
            <Sun className="w-5 h-5" />
          ) : (
            <Moon className="w-5 h-5" />
          )}
        </Button>

        {/* Notifications */}
        <Button
          variant="ghost"
          size="sm"
          iconOnly
          className="relative"
        >
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-[rgb(var(--color-error-500))] rounded-full" />
        </Button>
      </div>
    </header>
  );
}

export default Header;
