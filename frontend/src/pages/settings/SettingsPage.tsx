/**
 * Settings Page
 * 
 * User preferences, account settings, and app configuration
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Settings as SettingsIcon,
  User,
  Bell,
  Shield,
  Palette,
  Volume2,
  Moon,
  Sun,
  Laptop,
  Save,
  LogOut,
  Key,
  ChevronRight,
  AlertTriangle
} from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem, staggerContainer } from '../../lib/animations';
import { useAuthStore } from '../../store/authStore';

// Theme options
type Theme = 'light' | 'dark' | 'system';

// Language options
const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी (Hindi)' },
  { code: 'bn', name: 'বাংলা (Bengali)' },
  { code: 'te', name: 'తెలుగు (Telugu)' },
  { code: 'ta', name: 'தமிழ் (Tamil)' },
  { code: 'mr', name: 'मराठी (Marathi)' },
];

// Settings Section Component
function SettingsSection({ 
  icon: Icon, 
  title, 
  description, 
  children 
}: { 
  icon: any; 
  title: string; 
  description: string; 
  children: React.ReactNode;
}) {
  return (
    <motion.div 
      variants={staggerItem}
      className="bg-card rounded-xl border border-border overflow-hidden"
    >
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <Icon className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
      </div>
      <div className="p-6 space-y-4">
        {children}
      </div>
    </motion.div>
  );
}

// Toggle Component
function Toggle({ 
  checked, 
  onChange, 
  label, 
  description 
}: { 
  checked: boolean; 
  onChange: (checked: boolean) => void; 
  label: string; 
  description?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-foreground font-medium">{label}</p>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          checked ? 'bg-primary' : 'bg-muted'
        }`}
      >
        <span
          className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            checked ? 'left-7' : 'left-1'
          }`}
        />
      </button>
    </div>
  );
}

// Select Component
function Select({ 
  value, 
  onChange, 
  options, 
  label 
}: { 
  value: string; 
  onChange: (value: string) => void; 
  options: { value: string; label: string }[]; 
  label: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-foreground mb-2">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function SettingsPage() {
  const { user, logout } = useAuthStore();
  const [saving, setSaving] = useState(false);
  const [theme, setTheme] = useState<Theme>('system');
  const [language, setLanguage] = useState('en');
  
  // Notification settings
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [weeklyDigest, setWeeklyDigest] = useState(true);
  const [progressReminders, setProgressReminders] = useState(true);
  
  // Privacy settings
  const [publicProfile, setPublicProfile] = useState(false);
  const [activityTracking, setActivityTracking] = useState(true);
  const [dataCollection, setDataCollection] = useState(true);

  // Audio settings
  const [autoplayAudio, setAutoplayAudio] = useState(false);
  const [defaultVoice, setDefaultVoice] = useState('female_1');
  const [defaultSpeed, setDefaultSpeed] = useState('1.0');

  const handleSave = async () => {
    setSaving(true);
    try {
      // API call to save settings
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Show success
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = () => {
    if (confirm('Are you sure you want to log out?')) {
      logout();
    }
  };

  const ThemeButton = ({ value, icon: Icon, label }: { value: Theme; icon: any; label: string }) => (
    <button
      onClick={() => setTheme(value)}
      className={`flex-1 flex flex-col items-center gap-2 p-4 rounded-lg border transition-all ${
        theme === value
          ? 'border-primary bg-primary/10'
          : 'border-input hover:border-primary/50'
      }`}
    >
      <Icon className={`w-6 h-6 ${theme === value ? 'text-primary' : 'text-muted-foreground'}`} />
      <span className={`text-sm font-medium ${theme === value ? 'text-primary' : 'text-foreground'}`}>
        {label}
      </span>
    </button>
  );

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="container py-8 max-w-4xl mx-auto"
    >
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-gray-500 to-gray-700 flex items-center justify-center">
            <SettingsIcon className="w-6 h-6 text-white" />
          </div>
          Settings
        </h1>
        <p className="text-muted-foreground mt-2">
          Manage your account preferences and app settings
        </p>
      </div>

      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="space-y-6"
      >
        {/* Profile Section */}
        <SettingsSection
          icon={User}
          title="Profile"
          description="Manage your personal information"
        >
          <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white text-2xl font-bold">
              {user?.full_name?.[0] || user?.email?.[0] || 'U'}
            </div>
            <div className="flex-1">
              <p className="font-semibold text-foreground">{user?.full_name || 'User'}</p>
              <p className="text-sm text-muted-foreground">{user?.email}</p>
            </div>
            <Button variant="outline" size="sm">
              Edit Profile
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Full Name</label>
              <input
                type="text"
                defaultValue={user?.full_name || ''}
                className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Email</label>
              <input
                type="email"
                defaultValue={user?.email || ''}
                disabled
                className="w-full px-4 py-3 rounded-lg border border-input bg-muted text-muted-foreground"
              />
            </div>
          </div>
        </SettingsSection>

        {/* Appearance Section */}
        <SettingsSection
          icon={Palette}
          title="Appearance"
          description="Customize how ShikshaSetu looks"
        >
          <div>
            <label className="block text-sm font-medium text-foreground mb-3">Theme</label>
            <div className="flex gap-3">
              <ThemeButton value="light" icon={Sun} label="Light" />
              <ThemeButton value="dark" icon={Moon} label="Dark" />
              <ThemeButton value="system" icon={Laptop} label="System" />
            </div>
          </div>

          <Select
            value={language}
            onChange={setLanguage}
            label="Interface Language"
            options={LANGUAGES.map(lang => ({ value: lang.code, label: lang.name }))}
          />
        </SettingsSection>

        {/* Notifications Section */}
        <SettingsSection
          icon={Bell}
          title="Notifications"
          description="Control how you receive updates"
        >
          <Toggle
            checked={emailNotifications}
            onChange={setEmailNotifications}
            label="Email Notifications"
            description="Receive updates via email"
          />
          <Toggle
            checked={pushNotifications}
            onChange={setPushNotifications}
            label="Push Notifications"
            description="Get notified in your browser"
          />
          <Toggle
            checked={weeklyDigest}
            onChange={setWeeklyDigest}
            label="Weekly Digest"
            description="Summary of your learning progress"
          />
          <Toggle
            checked={progressReminders}
            onChange={setProgressReminders}
            label="Progress Reminders"
            description="Get reminded to continue learning"
          />
        </SettingsSection>

        {/* Audio Settings Section */}
        <SettingsSection
          icon={Volume2}
          title="Audio Settings"
          description="Configure text-to-speech preferences"
        >
          <Toggle
            checked={autoplayAudio}
            onChange={setAutoplayAudio}
            label="Auto-play Audio"
            description="Automatically play audio when available"
          />
          <Select
            value={defaultVoice}
            onChange={setDefaultVoice}
            label="Default Voice"
            options={[
              { value: 'female_1', label: 'Priya (Female)' },
              { value: 'male_1', label: 'Rahul (Male)' },
              { value: 'female_2', label: 'Anita (Female)' },
              { value: 'male_2', label: 'Vijay (Male)' },
            ]}
          />
          <Select
            value={defaultSpeed}
            onChange={setDefaultSpeed}
            label="Default Speed"
            options={[
              { value: '0.5', label: '0.5x (Slow)' },
              { value: '0.75', label: '0.75x' },
              { value: '1.0', label: '1.0x (Normal)' },
              { value: '1.25', label: '1.25x' },
              { value: '1.5', label: '1.5x (Fast)' },
            ]}
          />
        </SettingsSection>

        {/* Privacy Section */}
        <SettingsSection
          icon={Shield}
          title="Privacy & Security"
          description="Control your data and privacy settings"
        >
          <Toggle
            checked={publicProfile}
            onChange={setPublicProfile}
            label="Public Profile"
            description="Allow others to see your profile"
          />
          <Toggle
            checked={activityTracking}
            onChange={setActivityTracking}
            label="Activity Tracking"
            description="Track your learning progress"
          />
          <Toggle
            checked={dataCollection}
            onChange={setDataCollection}
            label="Data Collection"
            description="Help improve ShikshaSetu with usage data"
          />

          <div className="pt-4 border-t border-border">
            <Button variant="outline" className="w-full justify-between">
              <span className="flex items-center gap-2">
                <Key className="w-4 h-4" />
                Change Password
              </span>
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          <Button variant="outline" className="w-full justify-between text-warning border-warning hover:bg-warning/10">
            <span className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Delete Account
            </span>
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingsSection>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 pt-4">
          <Button
            onClick={handleSave}
            disabled={saving}
            className="flex-1"
          >
            {saving ? (
              <>
                <Spinner size="sm" className="mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </>
            )}
          </Button>
          <Button
            variant="outline"
            onClick={handleLogout}
            className="text-error border-error hover:bg-error/10"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Log Out
          </Button>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default SettingsPage;
