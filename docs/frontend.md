
# Frontend Architecture

React 19 application with TypeScript, Vite, TailwindCSS, and Zustand state management.

---

## Structure

```
frontend/src/
├── main.tsx              # Application entry
├── App.tsx               # Root component with routing
├── index.css             # Global styles
├── service-worker.ts     # PWA service worker
│
├── pages/                # Route pages
│   ├── LandingPage.tsx   # Public home
│   ├── LoginPage.tsx     # Authentication
│   ├── RegisterPage.tsx  # Registration
│   ├── DashboardPage.tsx # User dashboard
│   ├── UploadPage.tsx    # Content upload
│   ├── SimplifyPage.tsx  # Simplification UI
│   ├── TranslatePage.tsx # Translation UI
│   ├── QAPage.tsx        # Q&A interface
│   ├── LibraryPage.tsx   # Content library
│   ├── ContentPage.tsx   # Content detail
│   ├── TaskPage.tsx      # Task status
│   ├── FeaturesPage.tsx  # Feature showcase
│   └── AboutPage.tsx     # About page
│
├── components/           # Reusable components
│   ├── ui/               # Base UI components
│   ├── molecules/        # Composed components
│   ├── organisms/        # Complex components
│   ├── patterns/         # Layout patterns
│   ├── features/         # Feature-specific
│   ├── ProtectedRoute.tsx
│   └── PublicRoute.tsx
│
├── services/             # API layer
│   └── api.ts            # Axios client
│
├── store/                # State management
│   └── (Zustand stores)
│
├── hooks/                # Custom hooks
├── types/                # TypeScript types
├── utils/                # Utilities
├── lib/                  # Third-party configs
├── offline/              # PWA offline support
└── mocks/                # Test mocks
```

---

## Routing

### Route Configuration (`App.tsx`)

```tsx
<BrowserRouter>
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
</BrowserRouter>
```

### Route Guards

**ProtectedRoute**: Redirects to `/login` if not authenticated

**PublicRoute**: Allows access, shows navigation

---

## Pages

### Landing Page
Public entry point with feature highlights and call-to-action.

### Dashboard
User home showing:
- Recent content
- Processing statistics
- Quick actions

### Upload Page
File upload interface:
- Drag-and-drop zone
- File type validation
- Processing options (language, grade, subject)

### Simplify Page
Text simplification interface:
- Text input area
- Grade level selector (5-12)
- Subject selector
- Real-time preview

### Translate Page
Translation interface:
- Source text input
- Language selector (10 Indian languages)
- Translation output
- Audio generation option

### Q&A Page
Question answering interface:
- Document context
- Question input
- Answer display with sources

### Library Page
Content management:
- Filterable content list
- Search functionality
- Sort by date/subject/language

---

## API Client (`services/api.ts`)

### Configuration

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});
```

### Token Management

```typescript
// Request interceptor - attach token
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor - handle refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await refreshToken();
      return apiClient.request(error.config);
    }
    return Promise.reject(error);
  }
);
```

### API Methods

```typescript
// Authentication
api.login(email, password): Promise<TokenResponse>
api.register(data): Promise<TokenResponse>
api.refreshToken(): Promise<TokenResponse>

// Content
api.processContent(data): Promise<ProcessedContent>
api.simplifyText(data): Promise<SimplifiedText>
api.translateText(data): Promise<TranslatedText>
api.validateContent(data): Promise<ValidationResult>
api.generateTTS(data): Promise<AudioResult>

// Library
api.getContents(filters): Promise<PaginatedResponse>
api.getContent(id): Promise<ProcessedContent>
api.deleteContent(id): Promise<void>

// Q&A
api.askQuestion(data): Promise<Answer>
api.getQAHistory(): Promise<QAHistory>

// Tasks
api.getTaskStatus(taskId): Promise<TaskStatus>
```

---

## Component Architecture

### Atomic Design

```
components/
├── ui/              # Atoms: Button, Input, Card, Badge
├── molecules/       # Molecules: FormField, SearchBar
├── organisms/       # Organisms: Header, Sidebar, DataTable
├── patterns/        # Patterns: PageLayout, FormContainer
└── features/        # Features: ContentCard, TaskTracker
```

### UI Components (Radix UI)

```typescript
// Example imports
import { Dialog } from '@radix-ui/react-dialog';
import { Select } from '@radix-ui/react-select';
import { Tabs } from '@radix-ui/react-tabs';
import { Toast } from '@radix-ui/react-toast';
import { Tooltip } from '@radix-ui/react-tooltip';
```

---

## State Management

### Zustand Store Pattern

```typescript
// store/authStore.ts
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  
  login: async (email, password) => {
    const { user, tokens } = await api.login(email, password);
    localStorage.setItem('access_token', tokens.access);
    set({ user, isAuthenticated: true });
  },
  
  logout: () => {
    localStorage.removeItem('access_token');
    set({ user: null, isAuthenticated: false });
  }
}));
```

---

## Data Fetching

### React Query Integration

```typescript
import { useQuery, useMutation } from '@tanstack/react-query';

// Fetch content
const { data, isLoading, error } = useQuery({
  queryKey: ['content', contentId],
  queryFn: () => api.getContent(contentId)
});

// Process content
const mutation = useMutation({
  mutationFn: api.processContent,
  onSuccess: (data) => {
    queryClient.invalidateQueries(['library']);
  }
});
```

---

## Styling

### TailwindCSS Configuration

```javascript
// tailwind.config.js
export default {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {...},
        secondary: {...}
      }
    }
  }
}
```

### Design Tokens

```css
/* Gradient backgrounds */
.bg-gradient-to-br.from-blue-50.via-white.to-purple-50

/* Responsive containers */
.container.mx-auto.px-4

/* Card styles */
.rounded-lg.shadow-md.bg-white.p-6
```

---

## Error Handling

### Error Boundary

```tsx
class ErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <ErrorFallback />;
    }
    return this.props.children;
  }
}
```

### API Error Handling

```typescript
try {
  const result = await api.processContent(data);
} catch (error) {
  if (error.response?.status === 401) {
    // Redirect to login
  } else if (error.response?.status === 429) {
    // Show rate limit message
  } else {
    // Show generic error
  }
}
```

---

## PWA Support

### Service Worker (`service-worker.ts`)

- Offline content caching
- Background sync for uploads
- Push notifications (optional)

### Manifest

```json
{
  "name": "Shiksha Setu",
  "short_name": "ShikshaSetu",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#3B82F6"
}
```

---

## Type Definitions (`types/`)

### API Types

```typescript
interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'user' | 'educator' | 'admin';
}

interface ProcessedContent {
  id: string;
  original_text: string;
  simplified_text: string;
  translated_text: string;
  language: string;
  grade_level: number;
  subject: string;
  audio_file_path?: string;
  ncert_alignment_score: number;
  created_at: string;
}

interface TaskStatus {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: ProcessedContent;
  error?: string;
}
```

---

## Testing

### Test Setup

```typescript
// vitest.config.ts
export default {
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts'
  }
}
```

### Component Testing

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

test('login form submits correctly', async () => {
  render(<LoginPage />);
  
  await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
  await userEvent.type(screen.getByLabelText('Password'), 'password');
  await userEvent.click(screen.getByRole('button', { name: 'Login' }));
  
  expect(screen.getByText('Welcome')).toBeInTheDocument();
});
```

---

## Build & Development

### Scripts

```bash
# Development
npm run dev

# Build
npm run build

# Test
npm run test

# Lint
npm run lint
```

### Environment Variables

```bash
# .env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

---

⸻

Created by: **K Dhiraj**  
Email: kdhiraj152@gmail.com  
GitHub: [github.com/KDhiraj152](https://github.com/KDhiraj152)  
LinkedIn: [linkedin.com/in/kdhiraj152](https://linkedin.com/in/kdhiraj152)
