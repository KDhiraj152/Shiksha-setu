import { LoginForm } from '../../components/features/auth';

/**
 * Login Page - Premium authentication experience
 */
export function LoginPage() {
  return (
    <div className="card p-8 backdrop-blur-xl bg-card/80 border-border/50 shadow-2xl shadow-black/5">
      <LoginForm />
    </div>
  );
}

export default LoginPage;
