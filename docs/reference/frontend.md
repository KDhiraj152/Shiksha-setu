 # Frontend — Shiksha Setu

 Overview
 - Tech: React 19, TypeScript 5.x, Vite 7, TailwindCSS 4
 - Location: `frontend/`

 Key directories
 - `frontend/src/pages/` — route pages
 - `frontend/src/components/` — reusable presentational and container components
 - `frontend/src/services/` — API clients and adapters (centralize backend calls)
 - `frontend/src/store/` — state management (Zustand/Redux pattern; prefer typed hooks)

 API integration
 - Base URL comes from `VITE_API_BASE_URL` (set in `.env` or dev env). Keep env values consistent with backend `VITE_API_BASE_URL=http://localhost:8000`.
 - All HTTP calls should be centralized in `services/api.ts` (or similar). Use a single axios instance with interceptors for auth refresh and error handling.

 Auth & token handling
 - Access token (JWT) stored in memory (preferred) or `sessionStorage` for short-lived persistence.
 - Refresh token stored in `httpOnly` cookie from backend when possible. If not using cookies, store refresh token securely and implement rotation.
 - Axios interceptor responsibilities:
   - Add `Authorization: Bearer <access_token>` to requests.
   - If 401 and refresh token present, call `/api/v1/auth/refresh`, retry original request once.
   - On refresh failure, redirect to login.

 File upload flow
 - Use `frontend/src/services/upload.ts` to manage chunked and single uploads.
 - Contract: send `multipart/form-data` for single-file upload to `/api/v1/content/upload`.
 - For chunked uploads: implement `init` -> `upload chunk N` -> `finalize` endpoints; poll `task_id` for merge result.

 UI patterns & accessibility
 - Use Tailwind utility classes; keep components small and testable.
 - Add `aria-*` attributes for dynamic content and loading indicators.

 Dev tips
 - Run dev server: `cd frontend && npm install && npm run dev`
 - Lint: `npm run lint` (if configured)
 - Tests: `npm test` or `npm run test:ui`

 Notes
 - Keep business logic out of components — move to `services/` or hooks in `hooks/`.
 - When updating endpoints, update `services/api` and run frontend tests.

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
