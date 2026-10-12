"""Funciones y constantes auxiliares para app.py (ACA N° 1 - CUN)."""
import pandas as pd

# Orden exacto de las variables con que se entrenó el pipeline
FEATURES = ["Age", "StudyTimeWeekly", "Absences", "ParentalEducation", "ParentalSupport",
            "Ethnicity", "Gender", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering"]

GRADE_LABELS = ["A", "B", "C", "D", "F"]
GRADE_COLORS = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

ETH_MAP = {0: "Caucásico", 1: "Afroamericano", 2: "Asiático", 3: "Otro"}
PARED_MAP = {0: "Ninguna", 1: "Secundaria", 2: "Algo de universidad", 3: "Pregrado", 4: "Posgrado"}
SUPPORT_MAP = {0: "Ninguno", 1: "Bajo", 2: "Moderado", 3: "Alto", 4: "Muy alto"}

RANGOS = {"Age": (15, 18), "StudyTimeWeekly": (0, 20), "Absences": (0, 30),
          "ParentalEducation": (0, 4), "ParentalSupport": (0, 4), "Ethnicity": (0, 3),
          "Gender": (0, 1), "Tutoring": (0, 1), "Extracurricular": (0, 1),
          "Sports": (0, 1), "Music": (0, 1), "Volunteering": (0, 1)}


def nivel_riesgo(clase: int):
    """Traduce la categoría (0=A … 4=F) a un nivel de riesgo y un color."""
    if clase <= 1:
        return "Bajo", "#2a9d8f"
    if clase == 2:
        return "Medio", "#e9a23b"
    return "Alto", "#d1495b"


def recomendaciones(p: dict, clase: int) -> list[str]:
    """Reglas simples basadas en los factores más influyentes del modelo."""
    recs = []
    if p["Absences"] >= 15:
        recs.append(f"🚨 **Asistencia:** {p['Absences']} ausencias es un nivel crítico; es el factor que más pesa "
                    "en la predicción. Activar seguimiento de asistencia y contacto con la familia.")
    elif p["Absences"] >= 8:
        recs.append(f"⚠️ **Asistencia:** {p['Absences']} ausencias. Reducirlas es la palanca más efectiva para mejorar.")
    if p["StudyTimeWeekly"] < 8:
        recs.append("📚 **Estudio:** menos de 8 h semanales. Proponer un plan de estudio guiado.")
    if not p["Tutoring"] and clase >= 2:
        recs.append("🧑‍🏫 **Tutoría:** el estudiante no recibe tutoría; se asocia a mejor rendimiento.")
    if p["ParentalSupport"] <= 1:
        recs.append("👪 **Apoyo parental bajo:** involucrar a la familia en reuniones de seguimiento.")
    if not any([p["Extracurricular"], p["Sports"], p["Music"]]) and clase >= 2:
        recs.append("⚽ **Actividades:** la participación en deportes/extracurriculares se asocia a mayor GPA.")
    if not recs:
        recs.append("✅ Perfil favorable. Mantener hábitos actuales y monitoreo periódico.")
    return recs


def validar_lote(df: pd.DataFrame) -> str | None:
    """Devuelve un mensaje de error si el CSV no cumple el esquema esperado; None si es válido."""
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        return f"Faltan columnas: {', '.join(faltan)}"
    if df[FEATURES].isna().any().any():
        return "El archivo tiene valores vacíos en las columnas requeridas."
    for c, (lo, hi) in RANGOS.items():
        if not pd.api.types.is_numeric_dtype(df[c]) or not df[c].between(lo, hi).all():
            return f"La columna '{c}' tiene valores fuera del rango permitido [{lo}, {hi}]."
    return None
