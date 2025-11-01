# Proyecto MLOps
Repo base con Git + DVC para versionado de datos y colaboración.

## Instalación

Este proyecto usa [UV](https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales.

### Instalar UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Para más opciones de instalación, visita: https://docs.astral.sh/uv/getting-started/installation/

### Configurar el proyecto

Una vez instalado UV, sincroniza las dependencias del proyecto:

```bash
uv sync
```

Este comando:
- Crea un entorno virtual automáticamente
- Instala todas las dependencias definidas en `pyproject.toml`
- Genera/actualiza el archivo `uv.lock` para asegurar reproducibilidad

# Ejecución con dvc
`dvc.yaml` contiene las etapas de ejecución de este proyecto, para ejecutar usa el siguiente comando: `dvc repro`

## Etapas
- `preprocess`: esta etapa esta a cargo de leer los datos y limpiarlos

## MLflow Configuration

Este proyecto usa MLflow para el tracking de experimentos. La configuración se encuentra en `config/models_config.yaml`.

### Servidor MLflow Remoto

El proyecto está configurado para usar el servidor MLflow remoto en: **https://mlflow.labs.jsdevart.com/**

### Verificar Conexión

Antes de entrenar modelos, puedes verificar la conexión con MLflow ejecutando:

```bash
uv run python test_mlflow_connection.py
```

Este script:
- Verifica que la configuración de MLflow sea correcta
- Prueba la conexión con el servidor remoto
- Crea un experimento de prueba
- Muestra información útil para debugging

### Autenticación (si es necesario)

Si el servidor MLflow requiere autenticación, configura las variables de entorno:

```bash
export MLFLOW_TRACKING_USERNAME=tu_usuario
export MLFLOW_TRACKING_PASSWORD=tu_contraseña
```

O crea un archivo `.env` en la raíz del proyecto:

```
MLFLOW_TRACKING_USERNAME=tu_usuario
MLFLOW_TRACKING_PASSWORD=tu_contraseña
```

### Ver Experimentos

Después de entrenar modelos, puedes ver los experimentos en:
**https://mlflow.labs.jsdevart.com/**
