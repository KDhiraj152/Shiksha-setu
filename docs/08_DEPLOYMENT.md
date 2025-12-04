# Section 8: Deployment Architecture

## Deployment Strategy
I designed Shiksha Setu to be deployed as a containerized application using **Docker**. This ensures consistency across development, testing, and production environments.

## Infrastructure Components

### 1. Docker Compose
My `docker-compose.yml` file orchestrates the services:
*   **`backend`**: The FastAPI application container.
*   **`frontend`**: The Nginx container serving the React build.
*   **`postgres`**: The database container with the `pgvector` image.
*   **`redis`**: The cache container.

### 2. Setup Scripts
*   **`setup.sh`**: I wrote this comprehensive script to:
    *   Check for system dependencies (Python 3.11, Docker).
    *   Create a Python virtual environment.
    *   Install dependencies from `requirements.txt`.
    *   Download necessary AI models to the local `data/models` directory.
*   **`start.sh`**: Starts the services. In development, it runs Uvicorn with hot-reload. In production, it runs Gunicorn.

## Environment Configuration
I manage configuration via `.env` files. Key variables include:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Deployment mode (`development`, `production`) | `development` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://...` |
| `UNIVERSAL_MODE` | Enable unrestricted topic access | `True` |
| `DEVICE` | Hardware acceleration target (`cuda`, `mps`, `cpu`) | `auto` |

## CI/CD Pipeline (Recommended)
For a production setup, I recommend implementing a CI/CD pipeline (e.g., GitHub Actions):
1.  **Linting**: Run `flake8` and `black` on commit.
2.  **Testing**: Run `pytest` suite.
3.  **Build**: Build Docker images and push to a registry.
4.  **Deploy**: SSH into the production server and run `docker-compose up -d`.

## Load Balancing & Caching
*   **Load Balancer**: I use Nginx as the reverse proxy, handling SSL termination and routing traffic to the backend and frontend containers.
*   **Caching**:
    *   **Browser Cache**: Static assets (JS/CSS) are cached by the browser.
    *   **Redis Cache**: I cache API responses for identical queries for 5 minutes to reduce load on the AI models.
