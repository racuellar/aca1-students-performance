"""
ACA N° 1 - Aprendizaje Automático (CUN)
App Streamlit: Predicción del rendimiento académico (GradeClass A–F)
Dataset: Students Performance Dataset (Elkharoua, 2024)

Ejecutar localmente:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (FEATURES, GRADE_LABELS, GRADE_COLORS, ETH_MAP, PARED_MAP,
                   SUPPORT_MAP, nivel_riesgo, recomendaciones, validar_lote)

BASE = Path(__file__).parent

st.set_page_config(page_title="Predictor de Rendimiento Académico",
                   page_icon="🎓", layout="wide")


# ------------------------------------------------------------------
# Carga de artefactos (en caché para no recargar en cada interacción)
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    clf = joblib.load(BASE / "modelo" / "modelo_gradeclass.joblib")
    reg = joblib.load(BASE / "modelo" / "modelo_gpa.joblib")
    return clf, reg


@st.cache_data
def cargar_metricas():
    with open(BASE / "modelo" / "metricas.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "data" / "Student_performance_data.csv")


clf, reg = cargar_modelos()
meta = cargar_metricas()
df = cargar_datos()


def predecir(registro: pd.DataFrame):
    proba = clf.predict_proba(registro[FEATURES])
    clase = int(np.argmax(proba, axis=1)[0])
    gpa = float(np.clip(reg.predict(registro[FEATURES])[0], 0, 4))
    return clase, proba[0], gpa


# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🎓 Predictor de Rendimiento Académico")
st.caption("ACA N° 1 · Aprendizaje Automático · Especialización en Inteligencia Artificial — CUN  |  "
           f"Modelo: **{meta['modelo']}** entrenado con {meta['n_registros']:,} estudiantes".replace(",", "."))

# ------------------------------------------------------------------
# Barra lateral: perfil del estudiante
# ------------------------------------------------------------------
with st.sidebar:
    st.header("👤 Perfil del estudiante")
    st.markdown("**Hábitos y asistencia**")
    absences = st.slider("Ausencias en el año", 0, 30, 10)
    study = st.slider("Horas de estudio semanales", 0.0, 20.0, 10.0, 0.5)
    tutoring = st.toggle("Recibe tutoría", value=False)

    st.markdown("**Entorno familiar**")
    support = st.select_slider("Apoyo parental", options=list(SUPPORT_MAP),
                               value=2, format_func=SUPPORT_MAP.get)
    pared = st.selectbox("Educación de los padres", options=list(PARED_MAP),
                         index=2, format_func=PARED_MAP.get)

    st.markdown("**Actividades extracurriculares**")
    c1, c2 = st.columns(2)
    extra = c1.checkbox("Extracurricular")
    sports = c2.checkbox("Deportes")
    music = c1.checkbox("Música")
    volunt = c2.checkbox("Voluntariado")

    st.markdown("**Datos demográficos**")
    age = st.selectbox("Edad", [15, 16, 17, 18], index=1)
    gender = st.radio("Género", [0, 1], format_func={0: "Masculino", 1: "Femenino"}.get, horizontal=True)
    eth = st.selectbox("Etnia", options=list(ETH_MAP), format_func=ETH_MAP.get)

perfil = {
    "Age": age, "StudyTimeWeekly": study, "Absences": absences,
    "ParentalEducation": pared, "ParentalSupport": support, "Ethnicity": eth,
    "Gender": gender, "Tutoring": int(tutoring), "Extracurricular": int(extra),
    "Sports": int(sports), "Music": int(music), "Volunteering": int(volunt),
}
registro = pd.DataFrame([perfil])

tab_pred, tab_sim, tab_lote, tab_modelo, tab_datos = st.tabs(
    ["🔮 Predicción", "🧪 Simulador ¿Qué pasaría si…?", "📂 Predicción por lote",
     "📊 Desempeño del modelo", "🔎 Exploración de datos"])

# ------------------------------------------------------------------
# 1. Predicción individual
# ------------------------------------------------------------------
with tab_pred:
    clase, proba, gpa = predecir(registro)
    letra = GRADE_LABELS[clase]
    riesgo, color_riesgo = nivel_riesgo(clase)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Categoría predicha", letra)
    k2.metric("Confianza del modelo", f"{proba[clase]:.0%}")
    k3.metric("GPA estimado (0–4)", f"{gpa:.2f}")
    k4.markdown(f"<div style='padding:.6rem;border-radius:.5rem;background:{color_riesgo};"
                f"color:white;text-align:center'><b>Riesgo académico</b><br>"
                f"<span style='font-size:1.5rem'>{riesgo}</span></div>", unsafe_allow_html=True)

    st.subheader("Probabilidad por categoría")
    prob_df = pd.DataFrame({"Categoría": GRADE_LABELS, "Probabilidad": proba})
    chart = (alt.Chart(prob_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Categoría:N", sort=GRADE_LABELS, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Categoría:N", scale=alt.Scale(domain=GRADE_LABELS, range=GRADE_COLORS), legend=None),
        tooltip=["Categoría", alt.Tooltip("Probabilidad:Q", format=".1%")])
        .properties(height=280))
    st.altair_chart(chart, width="stretch")

    st.subheader("💡 Recomendaciones de intervención")
    for r in recomendaciones(perfil, clase):
        st.markdown(f"- {r}")
    st.caption("Escala: A (GPA ≥ 3.5) · B (3.0–3.5) · C (2.5–3.0) · D (2.0–2.5) · F (< 2.0). "
               "Herramienta de apoyo a la decisión; no reemplaza el criterio docente.")

# ------------------------------------------------------------------
# 2. Simulador what-if
# ------------------------------------------------------------------
with tab_sim:
    st.markdown("Explora cómo cambiaría la predicción si el estudiante modifica **una** variable, "
                "manteniendo el resto de su perfil fijo.")
    var = st.radio("Variable a simular", ["Absences", "StudyTimeWeekly", "ParentalSupport"],
                   format_func={"Absences": "Ausencias", "StudyTimeWeekly": "Horas de estudio",
                                "ParentalSupport": "Apoyo parental"}.get, horizontal=True)
    rangos = {"Absences": np.arange(0, 31), "StudyTimeWeekly": np.arange(0, 20.5, 0.5),
              "ParentalSupport": np.arange(0, 5)}
    sim = pd.concat([registro] * len(rangos[var]), ignore_index=True)
    sim[var] = rangos[var]
    sim["GPA estimado"] = np.clip(reg.predict(sim[FEATURES]), 0, 4)
    sim["Categoría"] = [GRADE_LABELS[i] for i in clf.predict(sim[FEATURES])]

    base = alt.Chart(sim).encode(x=alt.X(f"{var}:Q", title=var))
    linea = base.mark_line(color="#264653").encode(y=alt.Y("GPA estimado:Q", scale=alt.Scale(domain=[0, 4])))
    puntos = base.mark_circle(size=70).encode(
        y="GPA estimado:Q",
        color=alt.Color("Categoría:N", scale=alt.Scale(domain=GRADE_LABELS, range=GRADE_COLORS)),
        tooltip=[var, alt.Tooltip("GPA estimado:Q", format=".2f"), "Categoría"])
    actual = alt.Chart(pd.DataFrame({var: [perfil[var]]})).mark_rule(strokeDash=[6, 4], color="gray").encode(x=f"{var}:Q")
    st.altair_chart((linea + puntos + actual).properties(height=360), width="stretch")
    st.caption("La línea discontinua marca el valor actual del estudiante.")

# ------------------------------------------------------------------
# 3. Predicción por lote (CSV)
# ------------------------------------------------------------------
with tab_lote:
    st.markdown("Sube un CSV con las columnas del dataset original para predecir un grupo completo de estudiantes.")
    st.code(", ".join(FEATURES), language=None)
    plantilla = df[FEATURES].head(5).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar plantilla de ejemplo", plantilla, "plantilla_estudiantes.csv", "text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo)
        error = validar_lote(lote)
        if error:
            st.error(error)
        else:
            res = lote.copy()
            p = clf.predict_proba(lote[FEATURES])
            res["Categoría predicha"] = [GRADE_LABELS[i] for i in p.argmax(1)]
            res["Confianza"] = p.max(1).round(3)
            res["GPA estimado"] = np.clip(reg.predict(lote[FEATURES]), 0, 4).round(2)
            res["Riesgo"] = [nivel_riesgo(i)[0] for i in p.argmax(1)]
            c1, c2, c3 = st.columns(3)
            c1.metric("Estudiantes", len(res))
            c2.metric("En riesgo alto (D/F)", int(res["Riesgo"].eq("Alto").sum()))
            c3.metric("GPA estimado medio", f"{res['GPA estimado'].mean():.2f}")
            st.dataframe(res, width="stretch", hide_index=True)
            st.download_button("⬇️ Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                               "predicciones.csv", "text/csv")

# ------------------------------------------------------------------
# 4. Desempeño del modelo
# ------------------------------------------------------------------
with tab_modelo:
    st.markdown(f"**Modelo seleccionado:** {meta['modelo']} · hiperparámetros: `{meta['mejores_hiperparametros']}` · "
                f"evaluado en {meta['n_test']} estudiantes de prueba (20 %, estratificado).")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{meta['accuracy']:.1%}")
    m2.metric("F1-macro", f"{meta['f1_macro']:.3f}")
    m3.metric("Kappa ponderado (QWK)", f"{meta['qwk']:.3f}")
    m4.metric("ROC-AUC (OvR)", f"{meta['roc_auc_ovr']:.3f}")
    m5.metric("Exactitud ±1 categoría", f"{meta['acc_pm1']:.0%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matriz de confusión (prueba)")
        cm = pd.DataFrame(meta["matriz_confusion"], index=GRADE_LABELS, columns=GRADE_LABELS)
        cm_long = cm.reset_index().melt(id_vars="index", var_name="Predicha", value_name="n").rename(columns={"index": "Real"})
        heat = alt.Chart(cm_long).mark_rect().encode(
            x=alt.X("Predicha:N", sort=GRADE_LABELS, axis=alt.Axis(labelAngle=0)), y=alt.Y("Real:N", sort=GRADE_LABELS),
            color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = alt.Chart(cm_long).mark_text(fontSize=14).encode(
            x=alt.X("Predicha:N", sort=GRADE_LABELS, axis=alt.Axis(labelAngle=0)), y=alt.Y("Real:N", sort=GRADE_LABELS), text="n:Q",
            color=alt.condition(alt.datum.n > 100, alt.value("white"), alt.value("black")))
        st.altair_chart((heat + txt).properties(height=320), width="stretch")
    with c2:
        st.subheader("Importancia de variables (permutación)")
        imp = pd.DataFrame(list(meta["importancia_permutacion"].items()), columns=["Variable", "Δ F1-macro"])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#2a9d8f").encode(
            x="Δ F1-macro:Q", y=alt.Y("Variable:N", sort="-x")).properties(height=320), width="stretch")

    st.subheader("Comparación de algoritmos clásicos (validación cruzada, 5 pliegues)")
    st.dataframe(pd.DataFrame(meta["cv_comparacion"]).style.format(precision=3)
                 .highlight_max(subset=["Accuracy", "Bal. Accuracy", "F1-macro"], color="#d8f3dc"),
                 width="stretch", hide_index=True)
    st.subheader("Métricas por clase (prueba)")
    rep = pd.DataFrame(meta["reporte_clases"]).T.loc[GRADE_LABELS, ["precision", "recall", "f1-score", "support"]]
    st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}),
                 width="stretch")

# ------------------------------------------------------------------
# 5. Exploración de datos
# ------------------------------------------------------------------
with tab_datos:
    d = df.copy()
    d["Categoría"] = d["GPA"].apply(lambda g: "A" if g >= 3.5 else "B" if g >= 3 else "C" if g >= 2.5 else "D" if g >= 2 else "F")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estudiantes", f"{len(d):,}".replace(",", "."))
    c2.metric("Variables", d.shape[1] - 1)
    c3.metric("Etiquetas corregidas en limpieza", meta["etiquetas_corregidas"])

    eje = st.selectbox("Variable vs. GPA", ["Absences", "StudyTimeWeekly", "ParentalSupport", "Age"])
    sc = alt.Chart(d.sample(800, random_state=1)).mark_circle(size=25, opacity=.6).encode(
        x=f"{eje}:Q", y=alt.Y("GPA:Q", scale=alt.Scale(domain=[0, 4])),
        color=alt.Color("Categoría:N", scale=alt.Scale(domain=GRADE_LABELS, range=GRADE_COLORS)),
        tooltip=[eje, "GPA", "Categoría"]).properties(height=360)
    st.altair_chart(sc, width="stretch")

    dist = d["Categoría"].value_counts().reindex(GRADE_LABELS).reset_index()
    dist.columns = ["Categoría", "Estudiantes"]
    st.altair_chart(alt.Chart(dist).mark_bar().encode(
        x=alt.X("Categoría:N", sort=GRADE_LABELS, axis=alt.Axis(labelAngle=0)), y="Estudiantes:Q",
        color=alt.Color("Categoría:N", scale=alt.Scale(domain=GRADE_LABELS, range=GRADE_COLORS), legend=None))
        .properties(height=250, title="Distribución de categorías"), width="stretch")
    with st.expander("Ver datos crudos"):
        st.dataframe(df, width="stretch", hide_index=True)

st.divider()
st.caption("Fuente de datos: Elkharoua, R. (2024). *Students performance dataset* [Conjunto de datos]. Kaggle.")
