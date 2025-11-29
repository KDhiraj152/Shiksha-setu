# =============================================================================
# ShikshaSetu Frontend Dockerfile
# =============================================================================
# Multi-stage build: Build React app, serve with nginx

# Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --prefer-offline --no-audit

# Copy source code
COPY . .

# Build arguments for environment
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

# Build the application
RUN npm run build

# =============================================================================
# Production Stage - Nginx
# =============================================================================
FROM nginx:alpine as production

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config (this file is in infrastructure/docker/nginx.conf)
# When building, ensure the config is available in the build context
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
