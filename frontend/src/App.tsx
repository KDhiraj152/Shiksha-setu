import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppProviders } from './app/providers';
import { ErrorBoundary } from './app/ErrorBoundary';
import { ProtectedRoute, PublicRoute } from './app/routes';
import { AppLayout, AuthLayout } from './components/layout';
import { ToastContainer } from './components/ui/Toast';
import { Spinner } from './components/ui/Spinner';

// Import core pages (not lazy loaded)
import { 
  LandingPage,
  LoginPage, 
  RegisterPage,
  DashboardPage,
  PlaygroundPage,
  NotFoundPage
} from './pages';

// Lazy load feature pages for better performance
const WorkspacePage = lazy(() => import('./pages/workspace/UnifiedWorkspacePage'));
const LibraryPage = lazy(() => import('./pages/library/LibraryPage'));
const ContentDetailPage = lazy(() => import('./pages/content/ContentDetailPage'));
const SimplifyPage = lazy(() => import('./pages/simplify/SimplifyPage'));
const TranslatePage = lazy(() => import('./pages/translate/TranslatePage'));
const QAPage = lazy(() => import('./pages/qa/QAPage'));
const TTSPage = lazy(() => import('./pages/tts/TTSPage'));
const ProgressPage = lazy(() => import('./pages/progress/ProgressPage'));
const ReviewsPage = lazy(() => import('./pages/reviews/ReviewsPage'));
const SettingsPage = lazy(() => import('./pages/settings/SettingsPage'));
const AdminPage = lazy(() => import('./pages/admin/AdminPage'));

// Loading fallback component
function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center">
        <Spinner size="lg" className="mb-4" />
        <p className="text-muted-foreground">Loading...</p>
      </div>
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <AppProviders>
        <BrowserRouter>
          <ToastContainer />
          <Suspense fallback={<PageLoader />}>
            <Routes>
              {/* Public Landing Page */}
              <Route path="/" element={<LandingPage />} />
              
              {/* Auth routes with AuthLayout */}
              <Route element={<AuthLayout />}>
                <Route 
                  path="login" 
                  element={
                    <PublicRoute>
                      <LoginPage />
                    </PublicRoute>
                  } 
                />
                <Route 
                  path="register" 
                  element={
                    <PublicRoute>
                      <RegisterPage />
                    </PublicRoute>
                  } 
                />
              </Route>

              {/* Protected routes with AppLayout */}
              <Route 
                element={
                  <ProtectedRoute>
                    <AppLayout />
                  </ProtectedRoute>
                }
              >
                {/* Main Dashboard */}
                <Route path="dashboard" element={<DashboardPage />} />
                
                {/* AI Workspace - Flexible AI-first interface */}
                <Route path="workspace" element={<WorkspacePage />} />
                
                {/* Content Features */}
                <Route path="playground" element={<PlaygroundPage />} />
                <Route path="library" element={<LibraryPage />} />
                <Route path="content/:contentId" element={<ContentDetailPage />} />
                
                {/* Processing Features */}
                <Route path="simplify" element={<SimplifyPage />} />
                <Route path="translate" element={<TranslatePage />} />
                <Route path="tts" element={<TTSPage />} />
                
                {/* Q&A */}
                <Route path="qa" element={<QAPage />} />
                <Route path="qa/:contentId" element={<QAPage />} />
                
                {/* Progress Tracking */}
                <Route path="progress" element={<ProgressPage />} />
                
                {/* Reviews */}
                <Route path="reviews" element={<ReviewsPage />} />
                
                {/* Settings */}
                <Route path="settings" element={<SettingsPage />} />
                
                {/* Admin (should add role check) */}
                <Route path="admin" element={<AdminPage />} />
              </Route>

              {/* 404 page */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AppProviders>
    </ErrorBoundary>
  );
}

export default App;
