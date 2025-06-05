import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from io import BytesIO
import zipfile
import os
import tempfile

st.set_page_config(layout="wide")
st.title("Análisis de Biosensor GSR - Módulo Independiente")

# --- CARGA DE ARCHIVOS ---
st.header("1. Carga de archivos GSR")
gsr_files = st.file_uploader("Carga archivos CSV exportados de iMotions (GSR)", type="csv", accept_multiple_files=True)

if gsr_files:
    def make_unique(headers):
        from collections import Counter
        counts = Counter()
        new_headers = []
        for h in headers:
            counts[h] += 1
            if counts[h] > 1:
                new_headers.append(f"{h}_{counts[h]-1}")
            else:
                new_headers.append(h)
        return new_headers

    dfs = []
    for file in gsr_files:
        lines = file.read().decode("utf-8").splitlines()
        headers = make_unique(lines[27].split(","))
        content_data = "\n".join(lines[28:])
        df = pd.read_csv(BytesIO(content_data.encode()), names=headers)
        df["Participant"] = file.name.replace(".csv", "")
        dfs.append(df)

    df_gsr = pd.concat(dfs, ignore_index=True)
    df_gsr.to_csv("gsr_merged.csv", index=False)
    st.success("Archivos fusionados correctamente.")
    st.download_button("Descargar archivo fusionado", data=open("gsr_merged.csv", "rb"), file_name="gsr_merged.csv")

    # --- ESTADÍSTICOS POR ESTÍMULO ---
    st.header("2. Tabla de estadísticos por estímulo")
    resumen_gsr = []
    df_gsr["SourceStimuliName"] = df_gsr["SourceStimuliName"].astype(str)
    for stim, grupo in df_gsr.groupby("SourceStimuliName"):
        signal = grupo["GSR Conductance CAL"].dropna().values
        peaks, props = find_peaks(signal, height=0.02)
        amplitudes = props["peak_heights"]
        resumen_gsr.append({
            "Estímulo": stim,
            "Núm_Picos": len(peaks),
            "Amp_Media": np.mean(amplitudes) if len(amplitudes) > 0 else 0,
            "Amp_SD": np.std(amplitudes) if len(amplitudes) > 0 else 0
        })
    tabla_gsr = pd.DataFrame(resumen_gsr)
    st.dataframe(tabla_gsr)

    # Descargar tabla
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        tabla_gsr.to_excel(writer, index=False)
    st.download_button("Descargar tabla de resumen", output_excel.getvalue(), file_name="tabla_resumen_gsr.xlsx")

    # --- T-TEST POR PARES ---
    st.header("3. Comparaciones por pares (t-tests)")
    df_grafico = []
    for stim, grupo in df_gsr.groupby("SourceStimuliName"):
        signal = grupo["GSR Conductance CAL"].dropna().values
        peaks, props = find_peaks(signal, height=0.02)
        for amp in props["peak_heights"]:
            df_grafico.append({"Estímulo": stim, "Amplitud": amp})
    df_grafico = pd.DataFrame(df_grafico)

    stimuli = df_grafico["Estímulo"].unique()
    t_test_matrix = pd.DataFrame(index=stimuli, columns=stimuli)
    for i in stimuli:
        for j in stimuli:
            if i == j:
                t_test_matrix.loc[i, j] = "-"
            else:
                group1 = df_grafico[df_grafico["Estímulo"] == i]["Amplitud"]
                group2 = df_grafico[df_grafico["Estímulo"] == j]["Amplitud"]
                _, p = ttest_ind(group1, group2, equal_var=False)
                t_test_matrix.loc[i, j] = round(p, 4)

    st.dataframe(t_test_matrix)
    output_ttest = BytesIO()
    with pd.ExcelWriter(output_ttest, engine="xlsxwriter") as writer:
        t_test_matrix.to_excel(writer)
    st.download_button("Descargar matriz de t-tests", output_ttest.getvalue(), file_name="matriz_ttest_gsr.xlsx")

    # --- HEATMAP ---
    st.header("4. Gráfica Heatmap de p-values")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(t_test_matrix.replace("-", np.nan).astype(float), annot=True, cmap="coolwarm", fmt=".4f", ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # --- CARRUSEL DE GRÁFICAS ---
    st.header("5. Carrusel de gráficas por estímulo")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        for tipo, plot_func in {"boxplot": sns.boxplot, "violin": sns.violinplot}.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func(data=df_grafico, x="Estímulo", y="Amplitud", ax=ax)
            ax.set_title(f"{tipo.capitalize()} de Amplitudes por Estímulo")
            plt.xticks(rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            zipf.writestr(f"{tipo}_gsr.png", buf.getvalue())

        # Histograma combinado
        fig, ax = plt.subplots(figsize=(10, 6))
        for stim in stimuli:
            datos = df_grafico[df_grafico["Estímulo"] == stim]["Amplitud"]
            sns.histplot(datos, kde=True, label=stim, bins=30, alpha=0.6, ax=ax)
        ax.legend()
        ax.set_title("Histogramas combinados de amplitudes")
        fig.tight_layout()
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        zipf.writestr("histograma_combinado.png", buf.getvalue())

        # Histogramas individuales
        for stim in stimuli:
            fig, ax = plt.subplots(figsize=(8, 5))
            datos = df_grafico[df_grafico["Estímulo"] == stim]["Amplitud"]
            sns.histplot(datos, kde=True, bins=30, ax=ax)
            ax.set_title(f"Histograma GSR - {stim}")
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            zipf.writestr(f"histograma_{stim}.png", buf.getvalue())
            st.pyplot(fig)

    st.download_button("Descargar ZIP con gráficas", data=zip_buffer.getvalue(), file_name="graficas_gsr.zip")

    # --- ESTADÍSTICAS TXT ---
    resumen_txt = "\n".join([f"{row['Estímulo']}: N={row['Núm_Picos']}, Media={row['Amp_Media']:.4f}, SD={row['Amp_SD']:.4f}" for _, row in tabla_gsr.iterrows()])
    st.download_button("Descargar estadísticas (.txt)", data=resumen_txt, file_name="estadisticas_gsr.txt")
