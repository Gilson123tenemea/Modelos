# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import OPTICS
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
import plotly.express as px

def cargar_datos():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    class_names = data.target_names
    return X, y, class_names

def modo_supervisado():
    st.header("Modo Supervisado - Random Forest (Clasificación)")
    X, y, class_names = cargar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")

    st.subheader("Prueba interactiva de predicción")

    user_inputs = []
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        val = st.slider(col, min_val, max_val, mean_val)
        user_inputs.append(val)

    if st.button("Predecir clase"):
        input_df = pd.DataFrame([user_inputs], columns=X.columns)
        pred_class_idx = rf.predict(input_df)[0]
        pred_label = class_names[pred_class_idx]

        st.success(f"Clase predicha: {pred_label} (índice {pred_class_idx})")

        supervised_metrics = {
            "accuracy": float(prec),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }

        current_prediction = {
            "input": user_inputs,
            "output_class": int(pred_class_idx),
            "output_label": str(pred_label)
        }

        st.session_state["supervised_metrics"] = supervised_metrics
        st.session_state["supervised_current_prediction"] = current_prediction
        st.session_state["supervised_model_pkl"] = pickle.dumps(rf)

def modo_no_supervisado():
    st.header("Modo No Supervisado - OPTICS (Clustering)")
    X, _, _ = cargar_datos()

    min_samples = st.slider("min_samples", 2, 10, 5)
    xi = st.slider("xi (tasa de cambio)", 0.01, 0.2, 0.05)

    optics = OPTICS(min_samples=min_samples, xi=xi)
    optics.fit(X)
    labels = optics.labels_

    valid = labels != -1
    num_clusters = len(set(labels[valid]))
    st.write(f"Número de clusters encontrados (sin ruido): {num_clusters}")

    if num_clusters > 1:
        sil = silhouette_score(X[valid], labels[valid])
        db = davies_bouldin_score(X[valid], labels[valid])

        c1, c2 = st.columns(2)
        c1.metric("Silhouette Score", f"{sil:.3f}")
        c2.metric("Davies-Bouldin Index", f"{db:.3f}")
    else:
        sil, db = None, None
        st.warning("No se pueden calcular métricas (menos de 2 clusters válidos).")

    st.subheader("Visualización de Clusters (2 primeras características)")
    df_plot = X.iloc[:, :2].copy()
    df_plot["cluster"] = labels.astype(str)

    fig = px.scatter(
        df_plot,
        x=df_plot.columns[0],
        y=df_plot.columns[1],
        color="cluster",
        title="Clusters encontrados por OPTICS"
    )
    st.plotly_chart(fig, width="stretch")

    unsupervised_metrics = {
        "silhouette_score": float(sil) if sil is not None else None,
        "davies_bouldin": float(db) if db is not None else None
    }

    st.session_state["unsupervised_metrics"] = unsupervised_metrics
    st.session_state["unsupervised_params"] = {"min_samples": min_samples, "xi": xi}
    st.session_state["unsupervised_labels"] = labels.tolist()
    st.session_state["unsupervised_model_pkl"] = pickle.dumps(optics)

def zona_exportacion():
    st.header("Zona de Exportación (Dev Tools)")

    # ---------- EXPORTAR RESULTADOS MODELO SUPERVISADO ----------
    st.subheader("Exportación Modelo Supervisado (Random Forest)")
    if "supervised_metrics" in st.session_state and "supervised_model_pkl" in st.session_state:

        supervised_json = {
            "model_type": "Supervised",
            "model_name": "RandomForestClassifier",
            "metrics": {
                "accuracy": st.session_state["supervised_metrics"]["accuracy"],
                "precision": st.session_state["supervised_metrics"]["precision"],
                "recall": st.session_state["supervised_metrics"]["recall"],
                "f1_score": st.session_state["supervised_metrics"]["f1_score"]
            },
            "current_prediction": st.session_state.get("supervised_current_prediction", {})
        }

        st.download_button(
            label="Descargar JSON Supervisado",
            data=json.dumps(supervised_json, indent=2),
            file_name="supervised_model_results.json",
            mime="application/json"
        )

        st.download_button(
            label="Descargar Modelo Supervisado (.pkl)",
            data=st.session_state["supervised_model_pkl"],  # ✅ bytes correctos
            file_name="random_forest_model.pkl",
            mime="application/octet-stream"
        )

    else:
        st.info("⚠️ Primero entrena el modelo en *Modo Supervisado* para habilitar la exportación.")

    st.markdown("---")

    # ---------- EXPORTAR RESULTADOS MODELO NO SUPERVISADO ----------
    st.subheader("Exportación Modelo No Supervisado (OPTICS)")
    if "unsupervised_metrics" in st.session_state and "unsupervised_model_pkl" in st.session_state:

        unsupervised_json = {
            "model_type": "Unsupervised",
            "algorithm": "OPTICS",
            "parameters": {
                "min_samples": st.session_state["unsupervised_params"]["min_samples"],
                "xi": st.session_state["unsupervised_params"]["xi"]
            },
            "metrics": {
                "silhouette_score": st.session_state["unsupervised_metrics"]["silhouette_score"],
                "davies_bouldin": st.session_state["unsupervised_metrics"]["davies_bouldin"]
            },
            "cluster_labels": st.session_state["unsupervised_labels"]
        }

        st.download_button(
            label="Descargar JSON No Supervisado",
            data=json.dumps(unsupervised_json, indent=2),
            file_name="unsupervised_model_results.json",
            mime="application/json"
        )

        st.download_button(
            label="Descargar Modelo No Supervisado (.pkl)",
            data=st.session_state["unsupervised_model_pkl"], 
            file_name="optics_model.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("⚠️ Primero entrena el modelo en *Modo No Supervisado* para habilitar la exportación.")

def main():
    st.title("Modelos Supervisado vs No Supervisado")
    st.sidebar.title("Menú")
    opcion = st.sidebar.radio(
        "Selecciona un modo:",
        ("Modo Supervisado", "Modo No Supervisado", "Zona de Exportación")
    )

    if opcion == "Modo Supervisado":
        modo_supervisado()
    elif opcion == "Modo No Supervisado":
        modo_no_supervisado()
    else:
        zona_exportacion()

if __name__ == "__main__":
    main()
