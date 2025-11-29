import { motion } from 'framer-motion';
import { Clock, CheckCircle, XCircle, Loader2, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';
import { cn } from '../../../lib/cn';
import { Badge } from '../../ui/Badge';
import { Progress } from '../../ui/Progress';
import { formatRelativeTime } from '../../../lib/formatters';

interface Activity {
  id: string;
  type: 'process' | 'translate' | 'simplify' | 'tts';
  title: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  createdAt: Date;
  completedAt?: Date;
  language?: string;
}

interface RecentActivityProps {
  activities: Activity[];
  isLoading?: boolean;
}

const typeLabels: Record<Activity['type'], string> = {
  process: 'Processing',
  translate: 'Translation',
  simplify: 'Simplification',
  tts: 'Audio Generation',
};

const statusConfig: Record<Activity['status'], { icon: React.ElementType; color: string; label: string }> = {
  pending: { icon: Clock, color: 'text-amber-500', label: 'Pending' },
  processing: { icon: Loader2, color: 'text-blue-500', label: 'Processing' },
  completed: { icon: CheckCircle, color: 'text-green-500', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-red-500', label: 'Failed' },
};

export function RecentActivity({ activities, isLoading }: RecentActivityProps) {
  if (isLoading) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="h-6 w-32 skeleton rounded" />
          <div className="h-4 w-16 skeleton rounded" />
        </div>
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-4">
              <div className="w-10 h-10 skeleton rounded-xl" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-48 skeleton rounded" />
                <div className="h-3 w-24 skeleton rounded" />
              </div>
              <div className="h-6 w-20 skeleton rounded-full" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-foreground">Recent Activity</h3>
        <Link 
          to="/library" 
          className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center gap-1"
        >
          View all
          <ExternalLink className="w-3 h-3" />
        </Link>
      </div>

      {activities.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <Clock className="w-10 h-10 mx-auto mb-2 opacity-50" />
          <p>No recent activity</p>
          <p className="text-sm">Start processing content to see activity here</p>
        </div>
      ) : (
        <div className="space-y-4">
          {activities.map((activity, index) => {
            const StatusIcon = statusConfig[activity.status].icon;
            const isAnimating = activity.status === 'processing';

            return (
              <motion.div
                key={activity.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center gap-4 group"
              >
                {/* Icon */}
                <div
                  className={cn(
                    'w-10 h-10 rounded-xl flex items-center justify-center shrink-0',
                    activity.status === 'completed' && 'bg-green-100 dark:bg-green-900/30',
                    activity.status === 'failed' && 'bg-red-100 dark:bg-red-900/30',
                    activity.status === 'processing' && 'bg-blue-100 dark:bg-blue-900/30',
                    activity.status === 'pending' && 'bg-amber-100 dark:bg-amber-900/30'
                  )}
                >
                  <StatusIcon
                    className={cn(
                      'w-5 h-5',
                      statusConfig[activity.status].color,
                      isAnimating && 'animate-spin'
                    )}
                  />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm text-foreground truncate">
                    {activity.title}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {typeLabels[activity.type]}
                    {activity.language && ` • ${activity.language}`}
                    {' • '}
                    {formatRelativeTime(activity.createdAt)}
                  </p>
                  
                  {/* Progress bar for processing items */}
                  {activity.status === 'processing' && activity.progress !== undefined && (
                    <div className="mt-2">
                      <Progress value={activity.progress} size="sm" />
                    </div>
                  )}
                </div>

                {/* Status badge */}
                <Badge
                  variant={
                    activity.status === 'completed'
                      ? 'success'
                      : activity.status === 'failed'
                      ? 'error'
                      : activity.status === 'processing'
                      ? 'primary'
                      : 'warning'
                  }
                  className="shrink-0"
                >
                  {statusConfig[activity.status].label}
                </Badge>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default RecentActivity;
