import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
from supabase import create_client, Client

st.set_page_config(page_title="Red Eléctrica", layout="centered")

# (reemplaza credenciales)
SUPABASE_URL = "https://yhkeqdysmjirdrfrmvjd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inloa2VxZHlzbWppcmRyZnJtdmpkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMTA0NDQsImV4cCI6MjA2NDU4NjQ0NH0.Hn0G1jCxDuhzb4AnZyJAC3KGQGIq5Cxn8wgEc-_1fLo"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_data_from_supabase(table_name, start_date, end_date):
    # Convertir fechas a ISO string
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    
    response = (
        supabase
        .table(table_name)
        .select("*")
        .gte("datetime", start_iso)
        .lte("datetime", end_iso)
        .execute()
    )

    data = response.data
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# ----------------------------- INTERFAZ -----------------------------

def main():
    st.title("Análisis de la Red Eléctrica Española")

    tab1, tab2, tab3 = st.tabs(["Descripción", "Consulta de datos", "📊 Visualización"])

    # Tab 1: Descripción
    with tab1:
        st.subheader("¿Qué es esta app?")
        st.markdown("""
        Esta aplicación se conecta con la base de datos en Supabase que contiene datos históricos de la Red Eléctrica Española. 
        Permite consultar y visualizar datos de demanda, balance, generación e intercambios mediante filtros por fechas y categorías.
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
            elif año == 2024:
                start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
                end_date = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
            else:
                start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
                end_date = datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc)

        table = st.selectbox("Selecciona la tabla que deseas consultar:", [
            "demanda", "balance", "generacion", "intercambios", "intercambios_baleares"
        ])

        if st.button("Obtener datos"):
            with st.spinner("Consultando Supabase..."):
                df = get_data_from_supabase(table, start_date, end_date)

            if not df.empty:
                st.session_state["ree_data"] = df
                st.session_state["tabla"] = table
                st.session_state["start_date"] = start_date
                st.session_state["end_date"] = end_date
                st.success("✅ Datos cargados correctamente desde Supabase.")
            else:
                st.warning("⚠️ No se encontraron datos para ese período.")

    # Tab 3: Visualización
    with tab3:
        st.subheader("Visualización")

        if "ree_data" in st.session_state:
            df = st.session_state["ree_data"]
            start_date = st.session_state.get("start_date")
            end_date = st.session_state.get("end_date")
            tabla = st.session_state["tabla"]

            if "primary_category" in df.columns and not df["primary_category"].isnull().all():
                cat_col = "primary_category"
            else:
                cat_col = None

            st.markdown(f"**Tabla:** `{tabla}`")
            st.markdown(f"**Período:** {start_date.date()} a {end_date.date()}")

            if cat_col:
                fig = px.line(df,
                              x="datetime",
                              y="value",
                              color=cat_col,
                              title=f"Evolución: {tabla}",
                              labels={"value": "Valor", "datetime": "Fecha"})
            else:
                fig = px.line(df,
                              x="datetime",
                              y="value",
                              title=f"Evolución: {tabla}",
                              labels={"value": "Valor", "datetime": "Fecha"})

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver datos en tabla"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Primero consulta los datos desde la pestaña anterior.")

if __name__ == "__main__":
    main()
