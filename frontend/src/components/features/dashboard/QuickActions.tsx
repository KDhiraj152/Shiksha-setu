import { motion } from 'framer-motion';
import { 
  ArrowRight, 
  FileText, 
  Languages, 
  Volume2, 
  Sparkles,
  Plus
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '../../ui/Button/Button';
import { cn } from '../../../lib/cn';

interface QuickActionsProps {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onUpload?: () => void;
}

const actions = [
  {
    id: 'workspace',
    title: 'AI Workspace',
    description: 'Start working with AI',
    icon: Sparkles,
    gradient: 'from-blue-500 to-purple-600',
    href: '/workspace',
    primary: true,
  },
  {
    id: 'simplify',
    title: 'Simplify Text',
    description: 'Make content easier to read',
    icon: FileText,
    gradient: 'from-purple-500 to-purple-600',
    href: '/workspace',
  },
  {
    id: 'translate',
    title: 'Translate',
    description: 'Convert to regional languages',
    icon: Languages,
    gradient: 'from-green-500 to-green-600',
    href: '/workspace',
  },
  {
    id: 'tts',
    title: 'Generate Audio',
    description: 'Create text-to-speech',
    icon: Volume2,
    gradient: 'from-amber-500 to-amber-600',
    href: '/workspace',
  },
];

export function QuickActions({ onUpload: _onUpload }: QuickActionsProps) {
  const navigate = useNavigate();

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-foreground">Quick Actions</h3>
        <Link 
          to="/workspace" 
          className="text-sm text-primary-600 hover:text-primary-700 font-medium"
        >
          Open Workspace
        </Link>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {actions.map((action, index) => (
          <motion.button
            key={action.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => navigate(action.href)}
            className={cn(
              'group relative p-4 rounded-xl border border-border/50 text-left transition-all duration-200',
              'hover:border-border hover:shadow-md hover:-translate-y-0.5',
              action.primary && 'col-span-2 bg-gradient-to-br from-primary-50 to-secondary-50 dark:from-primary-950/30 dark:to-secondary-950/30 border-primary-200 dark:border-primary-800'
            )}
          >
            {/* Icon */}
            <div
              className={cn(
                'w-10 h-10 rounded-xl flex items-center justify-center mb-3',
                `bg-gradient-to-br ${action.gradient}`,
                'shadow-lg',
                action.gradient.includes('blue') && 'shadow-blue-500/25',
                action.gradient.includes('purple') && 'shadow-purple-500/25',
                action.gradient.includes('green') && 'shadow-green-500/25',
                action.gradient.includes('amber') && 'shadow-amber-500/25'
              )}
            >
              <action.icon className="w-5 h-5 text-white" />
            </div>

            {/* Content */}
            <h4 className="font-medium text-foreground mb-0.5 group-hover:text-primary-600 transition-colors">
              {action.title}
            </h4>
            <p className="text-xs text-muted-foreground">
              {action.description}
            </p>

            {/* Arrow indicator */}
            <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
              <ArrowRight className="w-4 h-4 text-muted-foreground" />
            </div>
          </motion.button>
        ))}
      </div>

      {/* AI Assistant prompt */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-4 p-4 rounded-xl bg-gradient-to-br from-primary-500/10 to-secondary-500/10 border border-primary-200/50 dark:border-primary-800/50"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center shadow-lg shadow-primary-500/25">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <h4 className="font-medium text-foreground text-sm">Need help?</h4>
            <p className="text-xs text-muted-foreground">Ask our AI assistant to help with content</p>
          </div>
          <Button variant="ghost" size="sm">
            <Plus className="w-4 h-4 mr-1" />
            Ask AI
          </Button>
        </div>
      </motion.div>
    </div>
  );
}

export default QuickActions;
