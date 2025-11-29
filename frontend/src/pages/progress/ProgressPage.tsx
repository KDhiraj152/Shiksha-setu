/**
 * Progress Page
 * 
 * Student progress tracking with stats, achievements, and parent reports
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp,
  Trophy,
  Clock,
  Target,
  BookOpen,
  Award,
  Calendar,
  ChevronRight,
  Download,
  Flame,
  Star,
  CheckCircle2
} from 'lucide-react';
import { 
  type LearningStats, 
  type Achievement, 
  type ProgressEntry, 
  type QuizScore 
} from '../../services/progress';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { Progress } from '../../components/ui/Progress/Progress';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem, staggerContainer } from '../../lib/animations';

// Format time
const formatTime = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
};

// Stats Card
function StatCard({ 
  icon: Icon, 
  label, 
  value, 
  subtext,
  trend,
  color = 'primary'
}: { 
  icon: any; 
  label: string; 
  value: string | number; 
  subtext?: string;
  trend?: number;
  color?: 'primary' | 'green' | 'yellow' | 'purple';
}) {
  const colors = {
    primary: 'from-primary-500 to-secondary-500',
    green: 'from-green-500 to-emerald-500',
    yellow: 'from-yellow-500 to-orange-500',
    purple: 'from-purple-500 to-pink-500',
  };

  return (
    <motion.div 
      variants={staggerItem}
      className="bg-card rounded-xl border border-border p-6"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colors[color]} flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend !== undefined && (
          <Badge variant={trend >= 0 ? 'success' : 'error'}>
            {trend >= 0 ? '+' : ''}{trend}%
          </Badge>
        )}
      </div>
      <p className="text-sm text-muted-foreground mb-1">{label}</p>
      <p className="text-2xl font-bold text-foreground">{value}</p>
      {subtext && <p className="text-xs text-muted-foreground mt-1">{subtext}</p>}
    </motion.div>
  );
}

// Achievement Card
function AchievementCard({ achievement }: { achievement: Achievement }) {
  return (
    <motion.div 
      variants={staggerItem}
      className="flex items-center gap-4 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl border border-yellow-200"
    >
      <div className="w-14 h-14 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center shadow-lg shadow-yellow-500/20">
        <Trophy className="w-7 h-7 text-white" />
      </div>
      <div className="flex-1">
        <p className="font-semibold text-foreground">{achievement.name}</p>
        <p className="text-sm text-muted-foreground">{achievement.description}</p>
        <p className="text-xs text-yellow-600 mt-1">
          Earned {new Date(achievement.earned_at).toLocaleDateString()}
        </p>
      </div>
      <Star className="w-6 h-6 text-yellow-500" />
    </motion.div>
  );
}

// Progress Item
function ProgressItem({ entry }: { entry: ProgressEntry }) {
  return (
    <motion.div 
      variants={staggerItem}
      className="flex items-center gap-4 p-4 bg-card rounded-xl border border-border hover:shadow-md transition-shadow"
    >
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
        entry.completed ? 'bg-green-100 text-green-600' : 'bg-primary-100 text-primary-600'
      }`}>
        {entry.completed ? <CheckCircle2 className="w-5 h-5" /> : <BookOpen className="w-5 h-5" />}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-foreground truncate">Content #{entry.content_id.slice(0, 8)}</p>
        <p className="text-sm text-muted-foreground">
          {formatTime(entry.time_spent_seconds)} spent
        </p>
      </div>
      <div className="flex items-center gap-3">
        <div className="text-right">
          <p className="text-sm font-medium text-foreground">{entry.progress_percent}%</p>
          <Progress value={entry.progress_percent} className="w-24 h-2" />
        </div>
        <ChevronRight className="w-5 h-5 text-muted-foreground" />
      </div>
    </motion.div>
  );
}

// Quiz Score Item
function QuizScoreItem({ score }: { score: QuizScore }) {
  return (
    <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
      <div>
        <p className="font-medium text-sm text-foreground">Quiz: {score.quiz_id.slice(0, 8)}</p>
        <p className="text-xs text-muted-foreground">
          {new Date(score.submitted_at).toLocaleDateString()}
        </p>
      </div>
      <div className="text-right">
        <p className={`text-lg font-bold ${score.passed ? 'text-green-600' : 'text-red-500'}`}>
          {score.percentage}%
        </p>
        <Badge variant={score.passed ? 'success' : 'error'} className="text-xs">
          {score.passed ? 'Passed' : 'Failed'}
        </Badge>
      </div>
    </div>
  );
}

export function ProgressPage() {
  // State
  const [stats, setStats] = useState<LearningStats | null>(null);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [progressEntries, setProgressEntries] = useState<ProgressEntry[]>([]);
  const [quizScores, setQuizScores] = useState<QuizScore[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState<'week' | 'month' | 'year'>('week');

  // Mock user ID (in real app, get from auth)
  const userId = 'current-user';

  // Load data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        // In real app, these would be actual API calls
        // For now, using mock data
        setStats({
          total_time_spent: 12540, // seconds
          contents_completed: 15,
          average_score: 82,
          current_streak: 5,
          total_achievements: 8,
        });

        setAchievements([
          {
            id: 1,
            name: 'First Steps',
            description: 'Complete your first lesson',
            badge_url: '',
            earned_at: new Date().toISOString(),
            category: 'beginner',
          },
          {
            id: 2,
            name: 'Speed Reader',
            description: 'Complete 5 lessons in one day',
            badge_url: '',
            earned_at: new Date(Date.now() - 86400000).toISOString(),
            category: 'streak',
          },
          {
            id: 3,
            name: 'Quiz Master',
            description: 'Score 90%+ on 3 quizzes',
            badge_url: '',
            earned_at: new Date(Date.now() - 172800000).toISOString(),
            category: 'achievement',
          },
        ]);

        setProgressEntries([
          {
            id: 1,
            user_id: userId,
            content_id: 'abc123-content-1',
            progress_percent: 100,
            completed: true,
            time_spent_seconds: 1800,
            started_at: new Date(Date.now() - 86400000).toISOString(),
            last_accessed: new Date().toISOString(),
          },
          {
            id: 2,
            user_id: userId,
            content_id: 'def456-content-2',
            progress_percent: 65,
            completed: false,
            time_spent_seconds: 1200,
            started_at: new Date(Date.now() - 172800000).toISOString(),
            last_accessed: new Date().toISOString(),
          },
          {
            id: 3,
            user_id: userId,
            content_id: 'ghi789-content-3',
            progress_percent: 30,
            completed: false,
            time_spent_seconds: 600,
            started_at: new Date().toISOString(),
            last_accessed: new Date().toISOString(),
          },
        ]);

        setQuizScores([
          {
            id: 1,
            user_id: userId,
            content_id: 'abc123',
            quiz_id: 'quiz-001',
            score: 85,
            max_score: 100,
            percentage: 85,
            passed: true,
            submitted_at: new Date().toISOString(),
          },
          {
            id: 2,
            user_id: userId,
            content_id: 'def456',
            quiz_id: 'quiz-002',
            score: 92,
            max_score: 100,
            percentage: 92,
            passed: true,
            submitted_at: new Date(Date.now() - 86400000).toISOString(),
          },
          {
            id: 3,
            user_id: userId,
            content_id: 'ghi789',
            quiz_id: 'quiz-003',
            score: 45,
            max_score: 100,
            percentage: 45,
            passed: false,
            submitted_at: new Date(Date.now() - 172800000).toISOString(),
          },
        ]);

      } catch (e) {
        console.error('Failed to load progress data:', e);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [selectedPeriod]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Spinner size="lg" className="mb-4" />
          <p className="text-muted-foreground">Loading your progress...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      variants={pageVariants}
      initial="initial"
      animate="enter"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <TrendingUp className="w-7 h-7 text-primary-500" />
            My Progress
          </h1>
          <p className="text-muted-foreground mt-1">Track your learning journey</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as any)}
            className="px-4 py-2 rounded-lg border border-border bg-background"
          >
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="year">This Year</option>
          </select>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
      </motion.div>

      {/* Stats Grid */}
      {stats && (
        <motion.div 
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          <StatCard
            icon={Clock}
            label="Total Study Time"
            value={formatTime(stats.total_time_spent)}
            subtext="Keep it up!"
            color="primary"
          />
          <StatCard
            icon={BookOpen}
            label="Lessons Completed"
            value={stats.contents_completed}
            subtext="This period"
            trend={12}
            color="green"
          />
          <StatCard
            icon={Target}
            label="Average Score"
            value={`${stats.average_score}%`}
            subtext="Across all quizzes"
            color="yellow"
          />
          <StatCard
            icon={Flame}
            label="Current Streak"
            value={`${stats.current_streak} days`}
            subtext="Don't break it!"
            color="purple"
          />
        </motion.div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Progress List */}
        <motion.div variants={staggerItem} className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">Recent Activity</h2>
            <Button variant="ghost" size="sm">
              View All <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
          <motion.div 
            variants={staggerContainer}
            initial="initial"
            animate="animate"
            className="space-y-3"
          >
            {progressEntries.map((entry) => (
              <ProgressItem key={entry.id} entry={entry} />
            ))}
          </motion.div>

          {/* Quiz Scores */}
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-foreground mb-4">Recent Quiz Scores</h2>
            <div className="space-y-2">
              {quizScores.map((score) => (
                <QuizScoreItem key={score.id} score={score} />
              ))}
            </div>
          </div>
        </motion.div>

        {/* Achievements */}
        <motion.div variants={staggerItem} className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <Award className="w-5 h-5 text-yellow-500" />
              Achievements
            </h2>
            <Badge variant="secondary">{achievements.length} earned</Badge>
          </div>
          <motion.div 
            variants={staggerContainer}
            initial="initial"
            animate="animate"
            className="space-y-3"
          >
            {achievements.map((achievement) => (
              <AchievementCard key={achievement.id} achievement={achievement} />
            ))}
          </motion.div>

          {/* Weekly Goal */}
          <div className="bg-gradient-to-br from-primary-50 to-secondary-50 rounded-xl p-5 border border-primary-100">
            <h3 className="font-semibold text-foreground mb-2">Weekly Goal</h3>
            <p className="text-sm text-muted-foreground mb-3">Complete 5 lessons this week</p>
            <div className="flex items-center gap-3">
              <Progress value={60} className="flex-1 h-3" />
              <span className="text-sm font-medium">3/5</span>
            </div>
          </div>

          {/* Parent Report */}
          <div className="bg-card rounded-xl border border-border p-5">
            <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
              <Calendar className="w-5 h-5 text-primary-500" />
              Parent Report
            </h3>
            <p className="text-sm text-muted-foreground mb-3">
              Generate a progress report to share with parents
            </p>
            <Button variant="outline" className="w-full">
              Generate Report
            </Button>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

export default ProgressPage;
