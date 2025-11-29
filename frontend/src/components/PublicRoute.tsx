import { useAuthStore } from '../store/authStore';
import { Navigate, Outlet } from 'react-router-dom';

const PublicRoute = () => {
  const { isAuthenticated } = useAuthStore();

  if (isAuthenticated) {
    return <Navigate to="/workspace" replace />;
  }

  return <Outlet />;
};

export default PublicRoute;
