import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ProtectedRoute from './components/ProtectedRoute';
import PublicRoute from './components/PublicRoute';

// Eagerly load critical routes (landing, auth)
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

// Lazy load non-critical routes for code splitting
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const TaskPage = lazy(() => import('./pages/TaskPage'));
const ContentPage = lazy(() => import('./pages/ContentPage'));
const LibraryPage = lazy(() => import('./pages/LibraryPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));
const FeaturesPage = lazy(() => import('./pages/FeaturesPage'));
const SimplifyPage = lazy(() => import('./pages/SimplifyPage'));
const TranslatePage = lazy(() => import('./pages/TranslatePage'));
const QAPage = lazy(() => import('./pages/QAPage'));

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
  </div>
);

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<PublicRoute />}>
              <Route index element={<LandingPage />} />
              <Route path="login" element={<LoginPage />} />
              <Route path="register" element={<RegisterPage />} />
              <Route path="about" element={<AboutPage />} />
            </Route>

            {/* Protected routes */}
            <Route path="/" element={<ProtectedRoute />}>
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="features" element={<FeaturesPage />} />
              <Route path="upload" element={<UploadPage />} />
              <Route path="simplify" element={<SimplifyPage />} />
              <Route path="translate" element={<TranslatePage />} />
              <Route path="qa" element={<QAPage />} />
              <Route path="library" element={<LibraryPage />} />
              <Route path="tasks/:taskId" element={<TaskPage />} />
              <Route path="content/:contentId" element={<ContentPage />} />
            </Route>
          </Routes>
        </Suspense>
      </div>
    </BrowserRouter>
  );
}

export default App;
