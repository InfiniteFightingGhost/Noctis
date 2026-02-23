# Noctis Project Documentation

This document provides a comprehensive overview of the Noctis ML Inference Backend, including its architecture, setup, and usage.

## 1. Project Overview

Noctis is a complete system for sleep quality monitoring, combining a custom-built hardware device with a production-grade ML inference backend. The hardware device gathers vital signs and movement data, which is then processed by the software backend to provide sleep-stage predictions and detailed analytics.

The system is designed to be multi-tenant, with robust authentication and authorization mechanisms. The backend provides a RESTful API for data ingestion, model prediction, and various MLOps-related functionalities like performance monitoring, drift detection, and model evaluation.

## 2. System Architecture

The Noctis project is composed of two primary components: the physical hardware monitor and the software backend.

### 2.1. Hardware Component: The Noctis Sleep Monitor

The data source for the system is a custom-built hardware device that monitors a person's sleep.

*   **Function:** The device uses a combination of a 60GHz millimeter-wave radar sensor and an MPU6050 accelerometer to gather comprehensive sleep data. The radar detects heart rate, respiration, and presence, while the accelerometer tracks movement and vibrations.
*   **Onboard Processing:** An ESP32 microcontroller processes the raw sensor data into 30-second epochs, each containing a 15-feature vector. This data is stored on an 8MB onboard flash memory chip.
*   **Physical Design:** The components are housed in a custom 3D-printed enclosure. The design files (`Parametric_Enclosure_v1.f3d`, `Parametric_Enclosure_v1.stl`) are located in the `Hardware/` directory.

*[INSERT IMAGE HERE: A picture of the assembled Noctis Sleep Monitor device.]*

*[INSERT IMAGE HERE: A render or picture of the 3D model for the enclosure.]*

For complete details on the hardware, including schematics, bill of materials (BOM), and assembly instructions, see `Hardware/HARDWARE.md`.

### 2.2. Software Backend

The software is a FastAPI application that ingests, stores, and analyzes the data from the hardware device. The application is structured into several layers:

-   **API Layer (FastAPI Routers):** Exposes `/v1` and `/internal` endpoints for external and internal communication. This is the entry point for all requests.
-   **Service Layer:** Contains the business logic for various domains such as evaluation, drift detection, stress testing, and monitoring.
-   **Data Access Layer:** Uses SQLAlchemy to interact with the TimescaleDB database. It handles all database operations for models like epochs, predictions, and feature statistics.
-   **Model Registry:** A file-based registry for managing different versions of machine learning models.

The application is designed to be run as a containerized service using Docker.

## 3. Installation and Setup

You can run the Noctis application locally using either Docker (recommended for production-like setup) or a Python virtual environment (for development).

### 3.1. Docker Setup (Recommended)

This method starts the API service and the TimescaleDB database.

1.  **Create Environment File:**
    Copy the example environment file. This file contains configuration variables for the application.

    ```bash
    cp .env.example .env
    ```

2.  **Build and Run Containers:**
    Use `docker-compose` to build the images and start the services.

    ```bash
    docker compose up --build
    ```

    The API will be available at `http://localhost:8000`.

### 3.2. Local Development Setup

This setup is intended for development and requires a running instance of TimescaleDB.

1.  **Create Virtual Environment:**
    Create and activate a Python virtual environment.

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Database Migrations:**
    Apply the latest database schema.

    ```bash
    alembic upgrade head
    ```

4.  **Start the Application:**
    Run the application using Uvicorn. The `--reload` flag enables hot-reloading for development.

    ```bash
    uvicorn app.main:app --reload
    ```

## 4. API Reference

The API is split into two main prefixes: `/v1` for the public API and `/internal` for internal services. All endpoints (except health checks) require JWT-based authentication.

### 4.1. Public API Endpoints (`/v1`)

-   **Authentication:** All requests require an `Authorization: Bearer <token>` header.
-   **Routers:**
    -   `devices`: Manage devices.
    -   `recordings`: Manage and retrieve recording data.
    -   `ingest`: Ingest epoch data.
    -   `predict`: Get model predictions.
    -   `models`: Manage and reload ML models.
    -   `evaluation`: Endpoints for model evaluation.
    -   `drift`: Endpoints for drift detection.
    -   `experiments`, `promotion`, `replay`: Routers for MLOps lifecycle.

### 4.2. Internal API Endpoints (`/internal`)

-   **Purpose:** These endpoints are for internal monitoring, administration, and operational tasks.
-   **Routers:**
    -   `stress`: For running load tests.
    -   `performance`: For performance monitoring.
    -   `monitoring`: For general service monitoring.
    -   `resilience`: For fault injection and resilience testing.
    -   `timescale`: For managing TimescaleDB policies.
    -   `audit`: For audit reports.

### 4.3. Health Checks

-   `GET /healthz`: Liveness probe.
-   `GET /readyz`: Readiness probe.
-   `GET /metrics`: Prometheus metrics endpoint.

## 5. Testing

The project has both unit and integration tests.

1.  **Running Unit Tests:**
    These tests do not require any external services.

    ```bash
    pytest tests/unit
    ```

2.  **Running Integration Tests:**
    These tests require a running TimescaleDB instance. You need to set the database URL as an environment variable.

    ```bash
    export INTEGRATION_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/noctis
    pytest tests/integration
    ```
