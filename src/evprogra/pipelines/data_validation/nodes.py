"""
nodes.py — Pipeline: data_validation
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_EMPLEADOS = {
    "id_empleado":   {"dtype": "Int64",   "nullable": True,  "min": 1, "max": 999},
    "nombre":        {"dtype": "object",  "nullable": True},
    "rut":           {"dtype": "object",  "nullable": True},
    "departamento":  {"dtype": "object",  "nullable": False,
                      "allowed": ["Ventas","Operaciones","Rrhh","Ti","Legal",
                                  "Logistica","Finanzas","Marketing","Logística",
                                  "Tecnología","Tecnologia"]},
    "cargo":         {"dtype": "object",  "nullable": False,
                      "allowed": ["Analista","Jefe","Gerente","Coordinador",
                                  "Asistente","Director","Especialista"]},
    "salario":       {"dtype": "float64", "nullable": False, "min": 0, "max": 10_000_000},
    "tipo_contrato": {"dtype": "object",  "nullable": False,
                      "allowed": ["Indefinido","Plazo Fijo","Honorarios",
                                  "Práctica","Practica","Pr?ctica"]},
    "jornada":       {"dtype": "object",  "nullable": False,
                      "allowed": ["Completa","Media Jornada","Por Turnos"]},
}

SCHEMA_AUSENCIAS = {
    "id_ausencia":   {"dtype": "Int64",   "nullable": True,  "min": 1},
    "id_empleado":   {"dtype": "Int64",   "nullable": False, "min": 1},
    "tipo_ausencia": {"dtype": "object",  "nullable": False,
                      "allowed": ["Permiso","Vacaciones","Licencia Médica",
                                  "Licencia Maternal","Falta Injustificada",
                                  "Licencia Medica"]},
    "dias":          {"dtype": "float64", "nullable": False, "min": 0, "max": 31},
}

SCHEMA_CAPACITACIONES = {
    "id_capacitacion": {"dtype": "Int64",   "nullable": True,  "min": 1},
    "id_empleado":     {"dtype": "Int64",   "nullable": False, "min": 1},
    "horas":           {"dtype": "float64", "nullable": False, "min": 0, "max": 200},
    "nota_final":      {"dtype": "float64", "nullable": False, "min": 1.0, "max": 7.0},
    "estado":          {"dtype": "object",  "nullable": False,
                        "allowed": ["Completado","En Progreso","Pendiente",
                                    "En Proceso","Cancelado"]},
}

SCHEMA_EVALUACIONES = {
    "id_evaluacion":         {"dtype": "Int64",   "nullable": True,  "min": 1},
    "id_empleado":           {"dtype": "Int64",   "nullable": False, "min": 1},
    "puntaje_desempeno":     {"dtype": "float64", "nullable": False, "min": 1.0, "max": 7.0},
    "competencias_tecnicas": {"dtype": "float64", "nullable": False, "min": 1.0, "max": 7.0},
    "competencias_blandas":  {"dtype": "float64", "nullable": False, "min": 1.0, "max": 7.0},
}

SCHEMA_FINAL = {
    "id_empleado":                    {"dtype": "Int64",   "nullable": True},
    "salario":                        {"dtype": "float64", "nullable": False, "min": 0},
    "dias_ausencia_total":            {"dtype": "float64", "nullable": True,  "min": 0},
    "horas_capacitacion_total":       {"dtype": "float64", "nullable": True,  "min": 0},
    "puntaje_desempeno_promedio":     {"dtype": "float64", "nullable": True,  "min": 1, "max": 7},
    "competencias_tecnicas_promedio": {"dtype": "float64", "nullable": True,  "min": 1, "max": 7},
    "competencias_blandas_promedio":  {"dtype": "float64", "nullable": True,  "min": 1, "max": 7},
    "antiguedad_anios":               {"dtype": "float64", "nullable": True,  "min": 0},
}


def _check_schema(df: pd.DataFrame, schema: dict, dataset_name: str) -> list:
    registros = []
    for col, reglas in schema.items():
        if col not in df.columns:
            registros.append({"dataset": dataset_name, "columna": col,
                               "check": "columna_existe", "resultado": "FAIL",
                               "detalle": "Columna no encontrada"})
            continue

        dtype_real = str(df[col].dtype)
        dtype_esp  = reglas["dtype"]
        tipo_ok = (
            dtype_real == dtype_esp or
            (dtype_esp == "float64" and "float" in dtype_real) or
            (dtype_esp == "Int64"   and ("int" in dtype_real.lower() or dtype_real == "Int64")) or
            (dtype_esp == "object"  and dtype_real == "object")
        )
        registros.append({"dataset": dataset_name, "columna": col, "check": "tipo_dato",
                           "resultado": "PASS" if tipo_ok else "WARN",
                           "detalle": f"esperado={dtype_esp}, real={dtype_real}"})

        n_nulos = df[col].isnull().sum()
        pct = round(100 * n_nulos / max(len(df), 1), 2)
        nullable = reglas.get("nullable", True)
        registros.append({"dataset": dataset_name, "columna": col, "check": "nulos",
                           "resultado": "PASS" if (nullable or n_nulos == 0) else "WARN",
                           "detalle": f"{n_nulos} nulos ({pct}%)"})

        if "min" in reglas and pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            min_val = col_data.min() if len(col_data) else None
            max_val = col_data.max() if len(col_data) else None
            min_ok  = min_val is None or min_val >= reglas["min"]
            max_ok  = "max" not in reglas or max_val is None or max_val <= reglas["max"]
            registros.append({"dataset": dataset_name, "columna": col, "check": "rango",
                               "resultado": "PASS" if (min_ok and max_ok) else "FAIL",
                               "detalle": (
                                   f"min_real={round(min_val,2) if min_val is not None else 'N/A'}, "
                                   f"max_real={round(max_val,2) if max_val is not None else 'N/A'} | "
                                   f"esperado=[{reglas.get('min','')}, {reglas.get('max','')}]"
                               )})

        if "allowed" in reglas:
            no_perm = set(df[col].dropna().unique()) - set(reglas["allowed"])
            registros.append({"dataset": dataset_name, "columna": col,
                               "check": "valores_permitidos",
                               "resultado": "PASS" if not no_perm else "WARN",
                               "detalle": "OK" if not no_perm else f"Inesperados: {sorted(no_perm)[:5]}"})
    return registros


def generar_reporte_validacion(
    empleados, ausencias, capacitaciones, evaluaciones,
    empleados_clean, ausencias_clean, capacitaciones_clean, evaluaciones_clean,
) -> pd.DataFrame:
    pares = [
        ("empleados",      empleados,      empleados_clean),
        ("ausencias",      ausencias,      ausencias_clean),
        ("capacitaciones", capacitaciones, capacitaciones_clean),
        ("evaluaciones",   evaluaciones,   evaluaciones_clean),
    ]
    registros = []
    for name, raw, clean in pares:
        nulos_a = int(raw.isnull().sum().sum())
        nulos_d = int(clean.isnull().sum().sum())
        comp_a  = round(100*(1 - nulos_a / max(raw.size,   1)), 2)
        comp_d  = round(100*(1 - nulos_d / max(clean.size, 1)), 2)
        for m, v in [
            ("n_filas_antes",           len(raw)),
            ("n_filas_despues",         len(clean)),
            ("filas_eliminadas",        len(raw) - len(clean)),
            ("n_columnas_antes",        raw.shape[1]),
            ("n_columnas_despues",      clean.shape[1]),
            ("nulos_totales_antes",     nulos_a),
            ("nulos_totales_despues",   nulos_d),
            ("nulos_reducidos",         nulos_a - nulos_d),
            ("duplicados_antes",        int(raw.duplicated().sum())),
            ("duplicados_despues",      int(clean.duplicated().sum())),
            ("pct_completitud_antes",   comp_a),
            ("pct_completitud_despues", comp_d),
            ("mejora_completitud_pct",  round(comp_d - comp_a, 2)),
        ]:
            registros.append({"dataset": name, "metrica": m, "valor": v})
    reporte = pd.DataFrame(registros)
    logger.info("Reporte de validación generado: %d métricas", len(reporte))
    return reporte


def generar_reporte_integridad(
    empleados_clean, ausencias_clean, capacitaciones_clean,
    evaluaciones_clean, rrhh_encoded,
) -> pd.DataFrame:
    todos  = []
    todos += _check_schema(empleados_clean,     SCHEMA_EMPLEADOS,      "empleados_clean")
    todos += _check_schema(ausencias_clean,      SCHEMA_AUSENCIAS,      "ausencias_clean")
    todos += _check_schema(capacitaciones_clean, SCHEMA_CAPACITACIONES, "capacitaciones_clean")
    todos += _check_schema(evaluaciones_clean,   SCHEMA_EVALUACIONES,   "evaluaciones_clean")
    todos += _check_schema(rrhh_encoded,         SCHEMA_FINAL,          "rrhh_encoded")

    ids_master  = set(empleados_clean["id_empleado"].dropna().astype(int))
    ids_encoded = set(rrhh_encoded["id_empleado"].dropna().astype(int))
    huerfanos   = ids_encoded - ids_master
    todos.append({"dataset": "rrhh_encoded", "columna": "id_empleado",
                  "check": "integridad_referencial",
                  "resultado": "PASS" if not huerfanos else "FAIL",
                  "detalle": f"{len(huerfanos)} IDs sin match en empleados_clean"})
    todos.append({"dataset": "rrhh_encoded", "columna": "—",
                  "check": "shape_filas",
                  "resultado": "PASS" if len(rrhh_encoded) == len(empleados_clean) else "WARN",
                  "detalle": f"encoded={len(rrhh_encoded)}, empleados_clean={len(empleados_clean)}"})

    reporte = pd.DataFrame(todos)
    n_pass = (reporte["resultado"] == "PASS").sum()
    n_warn = (reporte["resultado"] == "WARN").sum()
    n_fail = (reporte["resultado"] == "FAIL").sum()
    logger.info("Reporte integridad: %d checks | PASS=%d WARN=%d FAIL=%d",
                len(reporte), n_pass, n_warn, n_fail)
    return reporte