import { motion } from 'framer-motion';
import { useCurrentUser, useLibrary } from '../../hooks';
import { PageHeader } from '../../components/patterns/PageHeader';
import { StatsOverview, RecentActivity, QuickActions } from '../../components/features/dashboard';
import { pageVariants, staggerContainer, staggerItem } from '../../lib/animations';

/**
 * Dashboard Page - Premium dashboard with stats, activity, and quick actions
 */
export function DashboardPage() {
  const { data: user } = useCurrentUser();
  const { data: libraryData, isLoading } = useLibrary({ limit: 10, offset: 0 });

  // Transform library data into recent activity format
  const recentActivity = libraryData?.items?.slice(0, 5).map((item) => ({
    id: item.id,
    type: 'process' as const,
    title: `${item.subject} - Grade ${item.grade_level}`,
    status: 'completed' as const, // All library items are completed
    createdAt: item.created_at ? new Date(item.created_at) : new Date(),
    language: item.language,
  })) || [];

  // Calculate stats
  const stats = {
    totalContent: libraryData?.total || 0,
    totalTranslations: libraryData?.items?.filter(i => i.translations)?.length || 0,
    totalAudio: libraryData?.items?.filter(i => i.audio_available)?.length || 0,
    processingTime: '2.3s', // This would come from actual analytics
  };

  const firstName = user?.full_name?.split(' ')[0] || 'there';
  const greeting = getGreeting();

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="enter"
      exit="exit"
      className="space-y-6"
    >
      {/* Page Header */}
      <motion.div variants={staggerItem}>
        <PageHeader
          title={`${greeting}, ${firstName}`}
          description="Here's what's happening with your content today"
        />
      </motion.div>

      {/* Stats Overview */}
      <motion.div variants={staggerItem}>
        <StatsOverview
          totalContent={stats.totalContent}
          totalTranslations={stats.totalTranslations}
          totalAudio={stats.totalAudio}
          processingTime={stats.processingTime}
          isLoading={isLoading}
        />
      </motion.div>

      {/* Main Content Grid */}
      <motion.div 
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Recent Activity - Takes 2 columns */}
        <motion.div variants={staggerItem} className="lg:col-span-2">
          <RecentActivity 
            activities={recentActivity}
            isLoading={isLoading}
          />
        </motion.div>

        {/* Quick Actions - Takes 1 column */}
        <motion.div variants={staggerItem} className="lg:col-span-1">
          <QuickActions />
        </motion.div>
      </motion.div>
    </motion.div>
  );
}

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

export default DashboardPage;
