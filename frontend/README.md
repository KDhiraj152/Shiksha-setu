# Frontend - Shiksha Setu

React 19 + TypeScript + Vite frontend for Shiksha Setu AI education platform.

## 🚀 Quick Start

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

## 📁 Project Structure

```
frontend/src/
├── pages/               # Route pages (lazy-loaded)
│   ├── dashboard/
│   ├── workspace/       # Main unified AI workspace
│   ├── library/
│   ├── simplify/
│   ├── translate/
│   ├── tts/
│   ├── qa/
│   ├── auth/
│   └── ...
├── components/          # Reusable UI components
│   ├── ui/              # Base components (button, input, modal, etc.)
│   ├── layout/          # Layout wrappers
│   ├── features/        # Feature-specific components
│   ├── patterns/        # Common patterns (cards, headers, etc.)
│   └── molecules/       # Composite components
├── services/            # API clients & adapters
│   ├── api.ts           # Deprecated (use unifiedApi)
│   ├── unifiedApi.ts    # Main API client
│   ├── auth.ts
│   ├── content.ts
│   └── ...
├── store/               # State management (Zustand)
│   ├── authStore.ts
│   ├── pipelineStore.ts
│   └── ...
├── hooks/               # Custom React hooks
│   ├── useApi.ts
│   ├── useAuth.ts
│   ├── useContent.ts
│   └── ...
├── lib/                 # Utilities & helpers
│   ├── cn.ts            # Class name utilities
│   ├── animations.ts    # Animation presets
│   └── ...
├── types/               # TypeScript types
│   └── api.ts
├── app/                 # App-level config
│   ├── providers.tsx    # Context providers
│   ├── ErrorBoundary.tsx
│   └── routes.tsx       # Route guards
└── App.tsx              # Main app component
```

## 🔗 Backend Integration

### API Client

All API calls go through `services/unifiedApi.ts`:

```typescript
import { unifiedApi } from '@/services/unifiedApi';

// Authentication
const tokens = await unifiedApi.login({ email, password });
const newTokens = await unifiedApi.refreshToken(refreshToken);

// Content processing
const task = await unifiedApi.uploadFile(file, metadata);
const result = await unifiedApi.checkTaskStatus(taskId);

// Features
const simplified = await unifiedApi.simplify(text, gradeLevel);
const translated = await unifiedApi.translate(text, targetLanguages);
```

### State Management

Use Zustand stores for app state:

```typescript
import { useAuthStore } from '@/store/authStore';

const { user, logout, isAuthenticated } = useAuthStore();
```

## 🎨 Styling

- **TailwindCSS 4**: Utility-first CSS framework
- **Lucide Icons**: Icon library
- **Framer Motion**: Animation library

Theme configuration in `tailwind.config.js`:
- Dark mode support
- Custom color palette
- Responsive breakpoints

## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests in UI mode
npm run test:ui

# Run specific test file
npm test -- filename.test.ts

# Coverage report
npm run test:coverage
```

## 📊 Performance

- **Code Splitting**: Pages lazy-loaded with React.lazy()
- **Bundle Optimization**: ~80KB gzipped (initial load)
- **Asset Optimization**: Images minified, web fonts optimized
- **Lighthouse Target**: 95+ performance score

## 🔐 Security

- **JWT Authentication**: Access token in memory, refresh token in httpOnly cookie
- **CORS**: Configured for specific origins only
- **Input Validation**: All user inputs sanitized
- **XSS Protection**: React's built-in protection

## 📚 Documentation

- **[Architecture Reference](../docs/reference/architecture.md)** - System design
- **[API Reference](../docs/reference/api.md)** - API endpoints
- **[Frontend Reference](../docs/reference/frontend.md)** - Frontend architecture
- **[Complete Setup Guide](../docs/guides/setup.md)** - Installation steps

## 🚀 Deployment

### Build Production Bundle

```bash
npm run build
# Output: dist/

# Preview production build
npm run preview
```

### Environment Variables

Create `.env.production`:

```bash
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_APP_NAME=ShikshaSetu
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Docker Deployment

```bash
docker build -f infrastructure/docker/frontend.dockerfile -t shiksha-setu/frontend .
docker run -p 3000:3000 shiksha-setu/frontend
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5173
lsof -i :5173
kill -9 <PID>
```

### Module Not Found

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build Fails

```bash
# Check TypeScript errors
npm run build

# Clear cache
rm -rf dist .vite

# Rebuild
npm run build
```

## 📝 Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Make changes** with hot module replacement

4. **Run tests**:
   ```bash
   npm test
   ```

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

6. **Create pull request** on GitHub

## 🔗 Related Documentation

- **[Development Guide](../DEVELOPMENT.md)** - Full development guide
- **[Contributing Guide](../docs/guides/contributing.md)** - Contribution workflow
- **[Testing Guide](../docs/guides/testing.md)** - Testing best practices

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*

