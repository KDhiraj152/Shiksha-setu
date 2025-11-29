import { RegisterForm } from '../../components/features/auth';

/**
 * Register Page - Premium registration experience
 */
export function RegisterPage() {
  return (
    <div className="card p-8 backdrop-blur-xl bg-card/80 border-border/50 shadow-2xl shadow-black/5">
      <RegisterForm />
    </div>
  );
}

export default RegisterPage;
