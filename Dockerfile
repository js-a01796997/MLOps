# Dockerfile para Bike Sharing Prediction API
# Usa Python 3.12 y uv para gestión de dependencias

FROM python:3.12-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración de dependencias primero (para cache de Docker)
COPY pyproject.toml uv.lock ./

# Instalar dependencias usando uv
RUN uv sync --frozen --no-dev

# Copiar solo el código de la API
COPY src/api/ ./src/api/

# Exponer el puerto 8000
EXPOSE 8000

# Variable de entorno para MLflow (puede ser sobrescrita al ejecutar el contenedor)
ENV MLFLOW_TRACKING_URI=""

# Configurar PYTHONPATH para que los imports relativos funcionen
ENV PYTHONPATH=/app/src/api

# Comando para ejecutar la aplicación
# Ejecutar desde /app pero cambiar al directorio src/api para imports
CMD ["sh", "-c", "cd /app/src/api && uv run python main.py"]

