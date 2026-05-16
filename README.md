# 📊 Proyecto Kedro — Caso 3: Recursos Humanos
**SCY1101 Programación para la Ciencia de Datos**  
Evaluación Parcial N°1 — Encargo individual  
Framework: [Kedro](https://kedro.org/)

---

## 📁 Estructura del proyecto

```
evprogra/
├── conf/
│   └── base/
│       ├── catalog.yml          ← Definición de todos los datasets
│       └── parameters.yml       ← Parámetros configurables por pipeline
├── data/
│   ├── 01_raw/                  ← CSV originales (NO modificar)
│   │   ├── empleados.csv
│   │   ├── ausencias.csv
│   │   ├── capacitaciones.csv
│   │   └── evaluaciones.csv
│   ├── 02_intermediate/         ← Datasets limpios (output data_cleaning)
│   ├── 03_primary/              ← Dataset integrado y transformado (output data_transform)
│   └── 08_reporting/            ← Reportes de validación y diagnóstico
├── notebooks/
│   └── eda_rrhh.ipynb           ← Análisis exploratorio de datos (EDA)
├── docs/
│   └── informe_tecnico.pdf      ← Informe técnico (8-12 páginas)
├── src/
│   └── evprogra/
│       ├── pipeline_registry.py
│       └── pipelines/
│           ├── data_ingestion/  ← Carga y exploración inicial (AD 1.1)
│           ├── data_cleaning/   ← Limpieza de datos (AD 1.2)
│           ├── data_transform/  ← Transformaciones avanzadas (AD 1.3)
│           └── data_validation/ ← Verificación e integridad (AD 1.4)
├── requirements.txt
└── README.md
```

---

## ⚙️ Requisitos previos

- Python **3.10+**
- pip actualizado

---

## 🚀 Instalación y configuración

### 1. Clonar o descomprimir el proyecto

```bash
# Descomprimir el .zip entregado
unzip evprogra.zip
cd evprogra
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar los pipelines

### Ejecutar todos los pipelines (default)

```bash
kedro run
```

> Ejecuta: `data_cleaning` → `data_transform` → `data_validation` (10 nodos)

### Ejecutar un pipeline específico

```bash
kedro run --pipeline=data_ingestion   # Exploración y diagnóstico inicial
kedro run --pipeline=data_cleaning    # Limpieza de los 4 datasets
kedro run --pipeline=data_transform   # Integración y transformaciones avanzadas
kedro run --pipeline=data_validation  # Verificación de integridad
```

---

## 📊 Pipelines implementados

| Pipeline | Corresponde a | Output |
|---|---|---|
| `data_ingestion` | AD 1.1 | `data/08_reporting/reporte_ingestion.csv` |
| `data_cleaning` | AD 1.2 | `data/02_intermediate/*_clean.csv` |
| `data_transform` | AD 1.3 | `data/03_primary/rrhh_integrado.csv`, `rrhh_encoded.csv` |
| `data_validation` | AD 1.4 | `data/08_reporting/reporte_validacion.csv`, `reporte_integridad.csv` |

---

## 📓 Notebook EDA

```bash
# Con el entorno activado:
jupyter notebook notebooks/eda_rrhh.ipynb
```

El notebook realiza el análisis exploratorio inicial sobre los datos raw: estadísticas descriptivas, análisis de nulos, duplicados, distribuciones y correlaciones.

---

## 🔍 Visualizar pipelines

Con [Kedro-Viz](https://github.com/kedro-org/kedro-viz):

```bash
pip install kedro-viz
kedro viz run
```

Se abre automáticamente en `http://127.0.0.1:4141` con el diagrama interactivo de todos los pipelines.

---

## 🗂️ Datasets

| Archivo | Descripción | Filas | Columnas |
|---|---|---|---|
| `empleados.csv` | Datos maestros de empleados | 309 | 9 |
| `ausencias.csv` | Registro de ausencias laborales | 360 | 7 |
| `capacitaciones.csv` | Programas de capacitación | 412 | 8 |
| `evaluaciones.csv` | Evaluaciones de desempeño | 515 | 7 |

---

## 📦 Dependencias principales

Ver `requirements.txt`. Las principales son:

- `kedro` — orquestación del flujo de datos
- `pandas` — manipulación de datos
- `numpy` — operaciones numéricas
- `scikit-learn` — normalización y encoding
- `matplotlib` / `seaborn` — visualizaciones

---

## 📋 Resultados del pipeline (última ejecución)

```
✅ data_ingestion   → 31 columnas analizadas
✅ data_cleaning    → empleados: 286 filas | ausencias: 325 | capacitaciones: 384 | evaluaciones: 483
✅ data_transform   → rrhh_integrado: (286, 33) | rrhh_encoded: (286, 62)
✅ data_validation  → 89 checks | PASS=78 WARN=11 FAIL=0
⏱️ Tiempo total: 0.5 segundos
```
