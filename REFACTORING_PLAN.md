# Plan de Refactorización Incremental

## Análisis de Problemas Identificados

### 🔴 Problemas Críticos (Alta Prioridad)

1. **Código Duplicado Masivo**
   - `LinearRegression.py`, `LinearRegressionLasso.py`, `LinearRegressionRidge.py` tienen ~95% de código duplicado
   - Código ejecutable en nivel superior (no en funciones)
   - Duplicación de lógica de MLflow en múltiples archivos

2. **Imports Incorrectos**
   - `data_cleaning.py` usa `from clean_utils import ...` (import relativo sin punto)
   - **Contexto UV**: Los scripts se ejecutan como `python src/script.py`, lo que hace que el import funcione por casualidad
   - **Problema**: No es explícito ni escalable. Con `uv` deberíamos:
     - Opción A: Configurar `src` como paquete instalable en `pyproject.toml` (recomendado)
     - Opción B: Usar imports relativos explícitos `from .clean_utils import ...`
     - Opción C: Usar `uv run` que maneja paths automáticamente

3. **Mezcla de Responsabilidades**
   - `data_cleaning.py`: 488 líneas haciendo múltiples tareas
   - `Comparativa.py`: Código ejecutable sin función main()
   - Scripts ejecutan código directamente sin encapsulación

4. **Hardcoded Paths y Configuración**
   - Paths hardcodeados en múltiples archivos
   - Lógica de configuración MLflow duplicada
   - Sin manejo centralizado de paths

### 🟡 Problemas Moderados (Media Prioridad)

5. **Falta de Manejo de Errores**
   - No hay try/except en la mayoría de scripts
   - No valida existencia de archivos antes de leerlos
   - No maneja errores de MLflow

6. **Falta de POO donde sería útil**
   - No hay clases para encapsular lógica relacionada
   - Todo es funcional, difícil de extender
   - No hay abstracciones para diferentes tipos de modelos

7. **Funciones muy largas**
   - `data_cleaning.py` tiene código monolítico
   - Funciones con múltiples responsabilidades

8. **Configuración Duplicada**
   - `load_config()` aparece en varios archivos
   - Setup de MLflow duplicado

### 🟢 Mejoras Menores (Baja Prioridad)

9. **Nombres inconsistentes**
   - `Comparativa.py` (español) vs otros en inglés
   - Variables en español e inglés mezcladas

10. **Falta de Type Hints**
    - Algunos archivos tienen type hints, otros no
    - Inconsistencia en documentación

---

## Plan de Refactorización Incremental

### Fase 0: Preparación del Entorno UV (CRÍTICO - Hacer primero)

#### Paso 0.1: Limpiar Restos de Conflicto en pyproject.toml 🔴 CRÍTICO
**Objetivo**: Limpiar marcadores de conflicto de merge anterior que quedaron sin resolver
**Riesgo**: ALTO si no se resuelve - el archivo tiene sintaxis inválida y bloquea todo
**Archivos**: `pyproject.toml`
**Problema Detectado**:
- Marcadores `<<<<<<< HEAD`, `=======`, `>>>>>>>` en líneas 13-16
- `scikit-learn` aparece duplicado (línea 15 dentro del conflicto y línea 19 fuera)
- El archivo tiene sintaxis inválida debido a estos marcadores
**Cambios**:
- Eliminar marcadores de conflicto (líneas 13, 14, 16)
- Mantener solo una versión de `scikit-learn` (la que está en línea 19: `"scikit-learn"`)
- O usar la versión con restricción `">=1.7.2"` si se prefiere ser específico

#### Paso 0.2: Decidir Estrategia de Imports ✅ COMPLETADO (sin cambios necesarios)
**Objetivo**: Determinar cómo manejar imports entre módulos en `src/`
**Riesgo**: BAJO - Solo decisión de diseño
**Decisión**: 
- **Opción elegida**: Mantener imports simples `from clean_utils import ...`
- **Razón**: Funcionan correctamente porque los scripts se ejecutan como `python src/script.py`, lo que agrega `src/` al PYTHONPATH automáticamente
- **Alternativa considerada**: Usar build-system (hatchling/setuptools) pero no es necesario para desarrollo local
- **No se requieren cambios** en `pyproject.toml` para esta estrategia

---

### Fase 1: Correcciones Críticas (No rompen nada)

#### Paso 1.1: Arreglar Imports con UV ✅ SEGURO (después de Fase 0)
**Objetivo**: Corregir imports para usar estructura de paquete
**Riesgo**: BAJO - Solo corrige sintaxis
**Archivos**: `data_cleaning.py`
**Opción A (Recomendada - después de Paso 0.2)**:
- Cambiar `from clean_utils import ...` a `from src.clean_utils import ...`
- Ejecutar `uv pip install -e .` para instalar el paquete en modo editable

**Opción B (Si no configuramos paquete)**:
- Cambiar `from clean_utils import ...` a `from .clean_utils import ...` (import relativo)
- Requiere ejecutar como módulo: `python -m src.data_cleaning`

#### Paso 1.2: Crear Módulo de Utilidades Comunes ✅ SEGURO
**Objetivo**: Centralizar funciones comunes (config, paths, MLflow setup)
**Riesgo**: BAJO - Crea nuevo módulo sin modificar existentes
**Archivos**: Crear `src/utils/` con:
- `config_loader.py` - Carga de configuración
- `path_manager.py` - Manejo de paths
- `mlflow_setup.py` - Setup de MLflow

#### Paso 1.3: Extraer Funciones de `Comparativa.py` ✅ SEGURO
**Objetivo**: Convertir código ejecutable en funciones
**Riesgo**: BAJO - Solo encapsula código existente
**Archivos**: `Comparativa.py`
**Cambios**:
- Crear función `compare_models()`
- Crear función `main()`
- Mantener compatibilidad con ejecución directa

---

### Fase 2: Refactorización de Data Cleaning (Módulo crítico)

#### Paso 2.1: Dividir `data_cleaning.py` en funciones ✅ SEGURO
**Objetivo**: Separar lógica en funciones más pequeñas
**Riesgo**: BAJO - Solo reorganiza código existente
**Archivos**: `data_cleaning.py`
**Cambios**:
- Crear función `clean_season()`
- Crear función `clean_yr()`
- Crear función `clean_count_variables()`
- Crear función `main()` que orquesta todo

#### Paso 2.2: Crear Clase DataCleaner (Opcional POO) ⚠️ MODERADO
**Objetivo**: Encapsular lógica de limpieza en clase
**Riesgo**: MODERADO - Cambia estructura pero mantiene funcionalidad
**Archivos**: Crear `src/data/DataCleaner.py`
**Cambios**:
- Clase con métodos para cada tipo de limpieza
- Mantener funciones para compatibilidad hacia atrás

---

### Fase 3: Eliminar Código Duplicado

#### Paso 3.1: Marcar Scripts Obsoletos como Deprecated ⚠️ MODERADO
**Objetivo**: Documentar que `LinearRegression*.py` están obsoletos
**Riesgo**: BAJO - Solo agrega warnings
**Archivos**: `LinearRegression.py`, `LinearRegressionLasso.py`, `LinearRegressionRidge.py`
**Cambios**:
- Agregar `DeprecationWarning` al inicio
- Agregar comentario dirigiendo a `train_models.py`

#### Paso 3.2: Mejorar `train_models.py` ✅ SEGURO
**Objetivo**: Asegurar que `train_models.py` puede hacer todo lo que los scripts obsoletos
**Riesgo**: BAJO - Solo mejora existente
**Archivos**: `train_models.py`
**Cambios**:
- Verificar que soporta todos los casos de uso
- Mejorar manejo de errores

---

### Fase 4: Introducir POO Gradualmente

#### Paso 4.1: Crear Clase Base para Modelos ✅ SEGURO
**Objetivo**: Abstracción común para entrenamiento de modelos
**Riesgo**: BAJO - Solo agrega nueva funcionalidad
**Archivos**: Crear `src/models/BaseModelTrainer.py`
**Cambios**:
- Clase abstracta con métodos comunes
- Mantener funciones existentes funcionando

#### Paso 4.2: Crear Clases Específicas por Tipo de Modelo ⚠️ MODERADO
**Objetivo**: Estrategia Pattern para diferentes modelos
**Riesgo**: MODERADO - Requiere testing
**Archivos**: Crear `src/models/LinearModelTrainer.py`, etc.
**Cambios**:
- Implementar Strategy Pattern
- Mantener compatibilidad con funciones existentes

---

### Fase 5: Mejoras de Calidad

#### Paso 5.1: Agregar Manejo de Errores ✅ SEGURO
**Objetivo**: Validaciones y manejo de errores robusto
**Riesgo**: BAJO - Solo agrega validaciones
**Archivos**: Todos los scripts principales

#### Paso 5.2: Estandarizar Nombres ⚠️ MODERADO
**Objetivo**: Consistencia en nombres (español/inglés)
**Riesgo**: MODERADO - Puede afectar imports
**Archivos**: `Comparativa.py` → `model_comparison.py`

#### Paso 5.3: Agregar Type Hints Completos ✅ SEGURO
**Objetivo**: Mejor documentación y validación de tipos
**Riesgo**: BAJO - Solo agrega información

---

## Estrategia de Implementación

### Principios:
1. ✅ **Cada paso debe ser ejecutable y no romper código existente**
2. ✅ **Hacer commits pequeños después de cada paso**
3. ✅ **Probar cada cambio antes de continuar**
4. ✅ **Mantener compatibilidad hacia atrás cuando sea posible**

### Orden Recomendado:
1. **Fase 0 completa** (PREPARACIÓN CRÍTICA)
   - Resolver merge conflict
   - Configurar `src` como paquete con UV
2. Fase 1 completa (correcciones críticas)
3. Paso 2.1 (dividir data_cleaning)
4. Paso 3.1 (marcar como deprecated)
5. Evaluar y continuar según necesidad

### Notas Importantes sobre UV:
- **Comando para instalar en modo editable**: `uv pip install -e .`
- **Ejecutar scripts con UV**: `uv run python src/script.py` o `uv run src/script.py`
- **Ventaja de paquete instalable**: Permite imports absolutos `from src.module import ...`
- **Alternativa sin instalación**: Usar imports relativos `from .module import ...` pero requiere ejecutar como módulo

---

## Checklist de Seguridad

Antes de cada cambio:
- [ ] Hacer backup/commit del estado actual
- [ ] Verificar que tests existentes pasan (si hay)
- [ ] Probar ejecución manual del script modificado
- [ ] Verificar que el pipeline completo funciona
- [ ] Hacer commit con mensaje descriptivo

Después de cada cambio:
- [ ] Verificar que no se rompió nada
- [ ] Documentar cambios en este archivo
- [ ] Marcar paso como completado

