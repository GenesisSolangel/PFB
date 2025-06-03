import streamlit as st
import pandas as pd
import requests
import time
import uuid
from datetime import datetime, timedelta, timezone
import plotly.express as px


st.set_page_config(page_title="Red Eléctrica", layout="centered")


BASE_URL = "https://apidatos.ree.es/es/datos/"
HEADERS = {"accept": "application/json", "content-type": "application/json"}

ENDPOINTS = {
    "demanda": ("demanda/evolucion", "hour"),
    "balance": ("balance/balance-electrico", "day"),
    "generacion": ("generacion/evolucion-renovable-no-renovable", "day"),
    "intercambios": ("intercambios/todas-fronteras-programados", "day"),
    "intercambios_baleares": ("intercambios/enlace-baleares", "day"),
}

# consultar un endpoint de la API de REE
def get_data(endpoint_name, endpoint_info, params):
    path, time_trunc = endpoint_info
    params["time_trunc"] = time_trunc
    url = BASE_URL + path

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            return []
        response_data = response.json()
    except Exception:
        return []

    data = []

    for item in response_data.get("included", []):
        attrs = item.get("attributes", {})
        category = attrs.get("title")

        if "content" in attrs:
            for sub in attrs["content"]:
                sub_attrs = sub.get("attributes", {})
                sub_cat = sub_attrs.get("title")
                for entry in sub_attrs.get("values", []):
                    entry["primary_category"] = category
                    entry["sub_category"] = sub_cat
                    data.append(entry)
        else:
            for entry in attrs.get("values", []):
                entry["primary_category"] = category
                entry["sub_category"] = None
                data.append(entry)

    return data

def get_data_for_period(start_date, end_date):
    all_dfs = []

    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, 13):
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            if month_start > end_date:
                continue
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(minutes=1)
            actual_end = min(month_end, end_date)

            for name, (path, granularity) in ENDPOINTS.items():
                params = {
                    "start_date": month_start.strftime("%Y-%m-%dT%H:%M"),
                    "end_date": actual_end.strftime("%Y-%m-%dT%H:%M"),
                    "geo_trunc": "electric_system",
                    "geo_limit": "peninsular",
                    "geo_ids": "8741"
                }

                month_data = get_data(name, (path, granularity), params)

                if month_data:
                    df = pd.DataFrame(month_data)
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                    except Exception:
                        continue

                    df['year'] = df['datetime'].dt.year
                    df['month'] = df['datetime'].dt.month
                    df['day'] = df['datetime'].dt.day
                    df['hour'] = df['datetime'].dt.hour
                    df['endpoint'] = name
                    df['extraction_timestamp'] = datetime.utcnow().replace(tzinfo=timezone.utc)
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    all_dfs.append(df)

                time.sleep(0.5)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    st.title("Análisis de la Red Eléctrica Española")

    tab1, tab2, tab3 = st.tabs(["Descripción", "Consulta de datos", "📊 Visualización"])

    # Tab 1: Descripción
    with tab1:
        st.subheader("¿Qué es esta app?")
        st.markdown("""
                Esta aplicación web interactiva se conecta con la API oficial de Red Eléctrica de España (REE) para ofrecer una visualización clara y actualizada del estado del sistema eléctrico nacional. A través de un entorno intuitivo, permite explorar datos relacionados con el balance energético, la demanda, la generación y los intercambios de electricidad, tanto en tiempo real como en períodos históricos.

                Los usuarios pueden consultar información detallada filtrando por días recientes (7, 14 o 30 días) o comparando diferentes años, facilitando así el análisis de tendencias y comportamientos del sistema eléctrico. Además, la aplicación incluye un modelo de predicción de la demanda eléctrica, desarrollado con técnicas de análisis de datos, cuyos resultados también se muestran de forma visual e interpretativa.
                """)

    # Tab 2: Consulta de datos
    with tab2:
        st.subheader("Consulta de datos")

        modo = st.radio("Tipo de consulta:", ["Últimos días", "Año específico"], horizontal=True)

        if modo == "Últimos días":
            dias = st.selectbox("¿Cuántos días atrás?", [7, 14, 30])
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=dias)

        else:

            año = st.selectbox("Selecciona el año a consultar:", [2025, 2024, 2023])

            if año == 2025:

                end_date = datetime.now(timezone.utc)

                start_date = end_date - timedelta(days=365)

            else:
                if año == 2024:
                    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
                    end_date = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
                else:
                    if año == 2023:
                        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
                        end_date = datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc)

        if st.button("Obtener datos"):
            with st.spinner("Consultando API REE..."):
                df = get_data_for_period(start_date, end_date)

            if not df.empty:
                st.session_state["ree_data"] = df
                st.session_state["start_date"] = start_date
                st.session_state["end_date"] = end_date
                st.success("Datos cargados correctamente.")
            else:
                st.warning("No se encontraron datos válidos en ese período.")

    # Tab 3: Visualización
    with tab3:
        st.subheader("Visualización")

        if "ree_data" in st.session_state:
            df = st.session_state["ree_data"]
            start_date = st.session_state.get("start_date")
            end_date = st.session_state.get("end_date")

            endpoint_sel = st.selectbox("Selecciona tipo de dato", df["endpoint"].unique())

            df_filtrado = df[
                (df["endpoint"] == endpoint_sel) &
                (df["datetime"] >= start_date) &
                (df["datetime"] <= end_date)
            ]

            fig = px.line(df_filtrado,
                          x="datetime",
                          y="value",
                          color="primary_category",
                          title=f"Evolución: {endpoint_sel}",
                          labels={"value": "Valor", "datetime": "Fecha"})
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver datos en tabla"):
                st.dataframe(df_filtrado, use_container_width=True)
        else:
            st.info("Primero consulta los datos desde la pestaña anterior.")


if __name__ == "__main__":
    main()
