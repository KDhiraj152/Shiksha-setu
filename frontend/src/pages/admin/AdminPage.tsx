/**
 * Admin Dashboard Page
 * 
 * System overview, user management, and content moderation
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  LayoutDashboard,
  Users,
  FileText,
  Activity,
  Server,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw,
  Search,
  Filter,
  Ban,
  Trash2,
  Eye,
  Settings
} from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { Progress } from '../../components/ui/Progress/Progress';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem, staggerContainer } from '../../lib/animations';

interface SystemStats {
  total_users: number;
  active_users: number;
  total_content: number;
  pending_reviews: number;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  requests_per_minute: number;
}

interface RecentActivity {
  id: string;
  type: 'user_signup' | 'content_upload' | 'translation' | 'review' | 'error';
  message: string;
  timestamp: string;
  user_name?: string;
}

// Stat Card
function StatCard({ 
  icon: Icon, 
  label, 
  value, 
  trend,
  color = 'primary'
}: { 
  icon: any; 
  label: string; 
  value: string | number; 
  trend?: { value: number; label: string };
  color?: string;
}) {
  const colors: Record<string, string> = {
    primary: 'from-primary-500 to-secondary-500',
    green: 'from-green-500 to-emerald-500',
    yellow: 'from-yellow-500 to-orange-500',
    red: 'from-red-500 to-rose-500',
    blue: 'from-blue-500 to-cyan-500',
    purple: 'from-purple-500 to-pink-500',
  };

  return (
    <motion.div 
      variants={staggerItem}
      className="bg-card rounded-xl border border-border p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colors[color]} flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend && (
          <Badge variant={trend.value >= 0 ? 'success' : 'error'}>
            {trend.value >= 0 ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
            {Math.abs(trend.value)}%
          </Badge>
        )}
      </div>
      <p className="text-2xl font-bold text-foreground">{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
      {trend && <p className="text-xs text-muted-foreground mt-1">{trend.label}</p>}
    </motion.div>
  );
}

// System Health Card
function SystemHealthCard({ stats }: { stats: SystemStats }) {
  const healthMetrics = [
    { label: 'CPU', value: stats.cpu_usage, icon: Cpu, color: stats.cpu_usage > 80 ? 'error' : stats.cpu_usage > 60 ? 'warning' : 'success' },
    { label: 'Memory', value: stats.memory_usage, icon: Server, color: stats.memory_usage > 80 ? 'error' : stats.memory_usage > 60 ? 'warning' : 'success' },
    { label: 'Disk', value: stats.disk_usage, icon: HardDrive, color: stats.disk_usage > 80 ? 'error' : stats.disk_usage > 60 ? 'warning' : 'success' },
  ];

  return (
    <motion.div 
      variants={staggerItem}
      className="bg-card rounded-xl border border-border p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          System Health
        </h3>
        <Badge variant="success">
          <Wifi className="w-3 h-3 mr-1" />
          Online
        </Badge>
      </div>

      <div className="space-y-4">
        {healthMetrics.map(metric => (
          <div key={metric.label}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground flex items-center gap-2">
                <metric.icon className="w-4 h-4" />
                {metric.label}
              </span>
              <span className="text-sm font-medium text-foreground">{metric.value}%</span>
            </div>
            <Progress value={metric.value} variant={metric.color as any} />
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border flex items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {stats.requests_per_minute} req/min
        </span>
        <Button variant="ghost" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>
    </motion.div>
  );
}

// Activity Item
function ActivityItem({ activity }: { activity: RecentActivity }) {
  const typeConfig = {
    user_signup: { icon: Users, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    content_upload: { icon: FileText, color: 'text-green-500', bg: 'bg-green-500/10' },
    translation: { icon: Activity, color: 'text-purple-500', bg: 'bg-purple-500/10' },
    review: { icon: CheckCircle, color: 'text-yellow-500', bg: 'bg-yellow-500/10' },
    error: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-500/10' },
  };

  const config = typeConfig[activity.type] || typeConfig.error;
  const Icon = config.icon;

  return (
    <div className="flex items-start gap-3 py-3">
      <div className={`w-8 h-8 rounded-lg ${config.bg} flex items-center justify-center`}>
        <Icon className={`w-4 h-4 ${config.color}`} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-foreground line-clamp-1">{activity.message}</p>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-muted-foreground">
            {new Date(activity.timestamp).toLocaleString()}
          </span>
          {activity.user_name && (
            <Badge variant="secondary" className="text-xs">
              {activity.user_name}
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
}

// User Table Row
function UserRow({ user, onAction }: { user: any; onAction: (action: string, userId: string) => void }) {
  return (
    <tr className="border-b border-border hover:bg-muted/50">
      <td className="py-4 px-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white font-medium">
            {user.name?.[0] || user.email?.[0]}
          </div>
          <div>
            <p className="font-medium text-foreground">{user.name || 'Anonymous'}</p>
            <p className="text-sm text-muted-foreground">{user.email}</p>
          </div>
        </div>
      </td>
      <td className="py-4 px-4">
        <Badge variant={user.role === 'admin' ? 'primary' : 'secondary'}>
          {user.role}
        </Badge>
      </td>
      <td className="py-4 px-4">
        <Badge variant={user.is_active ? 'success' : 'error'}>
          {user.is_active ? 'Active' : 'Inactive'}
        </Badge>
      </td>
      <td className="py-4 px-4 text-sm text-muted-foreground">
        {new Date(user.created_at).toLocaleDateString()}
      </td>
      <td className="py-4 px-4">
        <div className="flex items-center gap-2">
          <button 
            onClick={() => onAction('view', user.id)}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <Eye className="w-4 h-4 text-muted-foreground" />
          </button>
          <button 
            onClick={() => onAction('ban', user.id)}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <Ban className="w-4 h-4 text-muted-foreground" />
          </button>
          <button 
            onClick={() => onAction('delete', user.id)}
            className="p-2 hover:bg-error/10 rounded-lg transition-colors"
          >
            <Trash2 className="w-4 h-4 text-error" />
          </button>
        </div>
      </td>
    </tr>
  );
}

export function AdminPage() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [activities, setActivities] = useState<RecentActivity[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'content' | 'system'>('overview');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      // Mock data for demo
      setStats({
        total_users: 1234,
        active_users: 856,
        total_content: 5678,
        pending_reviews: 42,
        cpu_usage: 45,
        memory_usage: 62,
        disk_usage: 38,
        requests_per_minute: 128,
      });

      setActivities([
        { id: '1', type: 'user_signup', message: 'New user registered', timestamp: new Date().toISOString(), user_name: 'Rahul Sharma' },
        { id: '2', type: 'content_upload', message: 'New content uploaded: Science Chapter 5', timestamp: new Date(Date.now() - 3600000).toISOString() },
        { id: '3', type: 'translation', message: 'Translation completed: Hindi to Telugu', timestamp: new Date(Date.now() - 7200000).toISOString() },
        { id: '4', type: 'review', message: 'Review approved for Math content', timestamp: new Date(Date.now() - 10800000).toISOString() },
        { id: '5', type: 'error', message: 'API rate limit exceeded', timestamp: new Date(Date.now() - 14400000).toISOString() },
      ]);

      setUsers([
        { id: '1', name: 'Priya Patel', email: 'priya@example.com', role: 'admin', is_active: true, created_at: new Date().toISOString() },
        { id: '2', name: 'Amit Kumar', email: 'amit@example.com', role: 'teacher', is_active: true, created_at: new Date().toISOString() },
        { id: '3', name: 'Sneha Singh', email: 'sneha@example.com', role: 'student', is_active: true, created_at: new Date().toISOString() },
        { id: '4', name: 'Raj Malhotra', email: 'raj@example.com', role: 'student', is_active: false, created_at: new Date().toISOString() },
      ]);

    } catch (err) {
      console.error('Failed to load admin data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleUserAction = async (action: string, userId: string) => {
    console.log('Action:', action, 'User:', userId);
    // Implement action handlers
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="container py-8 max-w-7xl mx-auto"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500 to-orange-500 flex items-center justify-center">
              <LayoutDashboard className="w-6 h-6 text-white" />
            </div>
            Admin Dashboard
          </h1>
          <p className="text-muted-foreground mt-2">
            System overview and management
          </p>
        </div>
        <Button variant="outline">
          <Settings className="w-4 h-4 mr-2" />
          Settings
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-8 border-b border-border pb-4">
        {(['overview', 'users', 'content', 'system'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && stats && (
        <motion.div
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="space-y-8"
        >
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              icon={Users}
              label="Total Users"
              value={stats.total_users.toLocaleString()}
              trend={{ value: 12, label: 'vs last month' }}
              color="blue"
            />
            <StatCard
              icon={Activity}
              label="Active Users"
              value={stats.active_users.toLocaleString()}
              trend={{ value: 8, label: 'vs last month' }}
              color="green"
            />
            <StatCard
              icon={FileText}
              label="Total Content"
              value={stats.total_content.toLocaleString()}
              trend={{ value: 23, label: 'vs last month' }}
              color="purple"
            />
            <StatCard
              icon={Clock}
              label="Pending Reviews"
              value={stats.pending_reviews}
              trend={{ value: -5, label: 'vs yesterday' }}
              color="yellow"
            />
          </div>

          {/* Charts and Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* System Health */}
            <SystemHealthCard stats={stats} />

            {/* Recent Activity */}
            <motion.div 
              variants={staggerItem}
              className="lg:col-span-2 bg-card rounded-xl border border-border p-6"
            >
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                Recent Activity
              </h3>
              <div className="divide-y divide-border">
                {activities.map(activity => (
                  <ActivityItem key={activity.id} activity={activity} />
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}

      {activeTab === 'users' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {/* Search and Filter */}
          <div className="flex gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search users..."
                className="w-full pl-10 pr-4 py-3 rounded-lg border border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <Button variant="outline">
              <Filter className="w-4 h-4 mr-2" />
              Filter
            </Button>
          </div>

          {/* Users Table */}
          <div className="bg-card rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-muted">
                <tr>
                  <th className="text-left py-4 px-4 text-sm font-medium text-muted-foreground">User</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-muted-foreground">Role</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-muted-foreground">Status</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-muted-foreground">Joined</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map(user => (
                  <UserRow key={user.id} user={user} onAction={handleUserAction} />
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {activeTab === 'content' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-12"
        >
          <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
          <p className="text-lg text-muted-foreground">Content management coming soon</p>
        </motion.div>
      )}

      {activeTab === 'system' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-12"
        >
          <Server className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
          <p className="text-lg text-muted-foreground">System settings coming soon</p>
        </motion.div>
      )}
    </motion.div>
  );
}

export default AdminPage;
