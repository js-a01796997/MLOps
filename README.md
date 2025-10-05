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
