import { motion } from 'framer-motion';
import { 
  FileText, 
  Languages, 
  Volume2, 
  Clock
} from 'lucide-react';
import { StatCard } from '../../patterns/StatCard';

interface StatsOverviewProps {
  totalContent: number;
  totalTranslations: number;
  totalAudio: number;
  processingTime: string;
  trends?: {
    content: number;
    translations: number;
    audio: number;
    time: number;
  };
  isLoading?: boolean;
}

export function StatsOverview({
  totalContent,
  totalTranslations,
  totalAudio,
  processingTime,
  trends,
  isLoading,
}: StatsOverviewProps) {
  const getTrend = (value?: number, invert = false) => {
    if (value === undefined) return undefined;
    const direction = invert 
      ? (value < 0 ? 'up' : value > 0 ? 'down' : 'neutral')
      : (value > 0 ? 'up' : value < 0 ? 'down' : 'neutral');
    return { value: Math.abs(value), direction: direction as 'up' | 'down' | 'neutral' };
  };

  const stats = [
    {
      label: 'Total Content',
      value: totalContent,
      icon: <FileText className="w-6 h-6" />,
      trend: getTrend(trends?.content),
      gradient: 'from-blue-500 to-blue-600',
      description: 'Processed documents',
    },
    {
      label: 'Translations',
      value: totalTranslations,
      icon: <Languages className="w-6 h-6" />,
      trend: getTrend(trends?.translations),
      gradient: 'from-purple-500 to-purple-600',
      description: 'Languages generated',
    },
    {
      label: 'Audio Files',
      value: totalAudio,
      icon: <Volume2 className="w-6 h-6" />,
      trend: getTrend(trends?.audio),
      gradient: 'from-green-500 to-green-600',
      description: 'TTS generated',
    },
    {
      label: 'Avg. Processing',
      value: processingTime,
      icon: <Clock className="w-6 h-6" />,
      trend: getTrend(trends?.time, true),
      gradient: 'from-amber-500 to-amber-600',
      description: 'Per document',
    },
  ];

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="card p-5 animate-pulse">
            <div className="h-10 w-10 rounded-xl bg-muted mb-4" />
            <div className="h-4 w-20 rounded bg-muted mb-2" />
            <div className="h-8 w-16 rounded bg-muted" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <StatCard
            label={stat.label}
            value={stat.value}
            description={stat.description}
            trend={stat.trend}
            icon={stat.icon}
            gradient={stat.gradient}
          />
        </motion.div>
      ))}
    </div>
  );
}

export default StatsOverview;
