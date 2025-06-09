import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
from supabase import create_client, Client
import schedule
import threading
import time as tiempo
import uuid
import requests
from dotenv import load_dotenv
import os
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Red Eléctrica", layout="centered")

# Constantes de configuración de la API REE
BASE_URL = "https://apidatos.ree.es/es/datos/"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

ENDPOINTS = {
    "demanda": ("demanda/evolucion", "hour"),
    "balance": ("balance/balance-electrico", "day"),
    "generacion": ("generacion/evolucion-renovable-no-renovable", "day"),
    "intercambios": ("intercambios/todas-fronteras-programados", "day"),
    "intercambios_baleares": ("intercambios/enlace-baleares", "day"),
}

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ------------------------------ UTILIDADES ------------------------------

# Función para consultar un endpoint, según los parámetros dados, de la API de REE
def get_data(endpoint_name, endpoint_info, params):
    path, time_trunc = endpoint_info
    params["time_trunc"] = time_trunc
    url = BASE_URL + path

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        # Si la búsqueda no fue bien, se devuelve una lista vacía
        if response.status_code != 200:
            return []
        response_data = response.json()
    except Exception:
        return []

    data = []

    # Verificamos si el item tiene "content" y asumimos que es una estructura compleja
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
            # Procesamos las estructuras más simples (demanda, generacion, intercambios_baleares), asumiendo que no hay subcategorías
            for entry in attrs.get("values", []):
                entry["primary_category"] = category
                entry["sub_category"] = None
                data.append(entry)

    return data

# Función para insertar cada DataFrame en Supabase
def insertar_en_supabase(nombre_tabla, df):
    df = df.copy()

    # Generamos IDs únicos
    df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Convertimos fechas a string ISO
    for col in ["datetime", "extraction_timestamp"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Reemplazamos NaN por None
    #df = df.where(pd.notnull(df), None)

    # Convertir a lista de diccionarios e insertar
    data = df.to_dict(orient="records")

    try:
        supabase.table(nombre_tabla).insert(data).execute()
        print(f"✅ Insertados en '{nombre_tabla}': {len(data)} filas")
    except Exception as e:
        print(f"❌ Error al insertar en '{nombre_tabla}': {e}")

# ------------------------------ FUNCIONES DE DESCARGA ------------------------------
# Función de extracción de datos de los últimos x años, devuelve DataFrame. Ejecutar una vez al inicio para poblar la base de datos.
def get_data_for_last_x_years(num_years=3):
    all_dfs = []
    current_date = datetime.now()
    # Calculamos el año de inicio a partir del año actual
    start_year_limit = current_date.year - num_years

    # Iteramos sobre cada año y mes
    for year in range(start_year_limit, current_date.year + 1):
        for month in range(1, 13):
            # Si el mes es mayor al mes actual y el año es el actual, lo saltamos
            month_start = datetime(year, month, 1)
            if month_start > current_date:
                continue
            # Calculamos el final del mes, asegurándonos de no exceder la fecha actual
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(minutes=1)
            end_date_for_request = min(month_end, current_date)

            monthly_data = []  # para acumular todos los dfs del mes

            # Iteramos sobre cada endpoint y sacamos los datos
            for name, (path, granularity) in ENDPOINTS.items():
                params = {
                    "start_date": month_start.strftime("%Y-%m-%dT%H:%M"),
                    "end_date": end_date_for_request.strftime("%Y-%m-%dT%H:%M"),
                    "geo_trunc": "electric_system",
                    "geo_limit": "peninsular",
                    "geo_ids": "8741"
                }

                data = get_data(name, (path, granularity), params)

                if data:
                    df = pd.DataFrame(data)
                    #Lidiamos con problemas de zona horaria en la columna "datetime"
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                    except Exception:
                        continue

                    # Obtenemos nuevas columnas y las reordenamos
                    df['year'] = df['datetime'].dt.year
                    df['month'] = df['datetime'].dt.month
                    df['day'] = df['datetime'].dt.day
                    df['hour'] = df['datetime'].dt.hour
                    df['extraction_timestamp'] = datetime.utcnow()
                    df['endpoint'] = name
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    monthly_data.append(df)
                    tiempo.sleep(1)

            # Generamos los dataframes individuales
            if monthly_data:
                df_nuevo = pd.concat(monthly_data, ignore_index=True)
                all_dfs.append(df_nuevo)

                tablas_dfs = {
                    "demanda": df_nuevo[df_nuevo["endpoint"] == "demanda"].drop(columns=["endpoint", "sub_category"], errors='ignore'),
                    "balance": df_nuevo[df_nuevo["endpoint"] == "balance"].drop(columns=["endpoint"], errors='ignore'),
                    "generacion": df_nuevo[df_nuevo["endpoint"] == "generacion"].drop(columns=["endpoint", "sub_category"], errors='ignore'),
                    "intercambios": df_nuevo[df_nuevo["endpoint"] == "intercambios"].drop(columns=["endpoint"], errors='ignore'),
                    "intercambios_baleares": df_nuevo[df_nuevo["endpoint"] == "intercambios_baleares"].drop(columns=["endpoint", "sub_category"], errors='ignore'),
                }

                for tabla, df_tabla in tablas_dfs.items():
                    if not df_tabla.empty:
                        insertar_en_supabase(tabla, df_tabla)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# Función para actualizar los datos desde la API cada 24 horas
def actualizar_datos_desde_api():
    print(f"[{datetime.now()}] ⏳ Ejecutando extracción desde API...")
    current_date = datetime.now()
    start_date = current_date - timedelta(days=1)

    all_dfs = []

    for name, (path, granularity) in ENDPOINTS.items():
        params = {
            "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
            "end_date": current_date.strftime("%Y-%m-%dT%H:%M"),
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741"
        }

        datos = get_data(name, (path, granularity), params)

        if datos:
            df = pd.DataFrame(datos)
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            except Exception:
                continue

            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
            df['extraction_timestamp'] = datetime.utcnow()
            df['endpoint'] = name
            df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

            df = df[['record_id', 'value', 'percentage', 'datetime',
                     'primary_category', 'sub_category', 'year', 'month',
                     'day', 'hour', 'endpoint', 'extraction_timestamp']]

            all_dfs.append(df)
            tiempo.sleep(1)
        else:
            print(f"⚠️ No se obtuvieron datos de '{name}'")

    if all_dfs:
        df_nuevo = pd.concat(all_dfs, ignore_index=True)

        tablas_dfs = {
            "demanda": df_nuevo[df_nuevo["endpoint"] == "demanda"].drop(columns=["endpoint", "sub_category"]),
            "balance": df_nuevo[df_nuevo["endpoint"] == "balance"].drop(columns=["endpoint"]),
            "generacion": df_nuevo[df_nuevo["endpoint"] == "generacion"].drop(columns=["endpoint", "sub_category"]),
            "intercambios": df_nuevo[df_nuevo["endpoint"] == "intercambios"].drop(columns=["endpoint"]),
            "intercambios_baleares": df_nuevo[df_nuevo["endpoint"] == "intercambios_baleares"].drop(columns=["endpoint", "sub_category"]),
        }

        for tabla, df in tablas_dfs.items():
            if not df.empty:
                insertar_en_supabase(tabla, df)

# Programador para actualizar datos desde la API cada 24 horas
def iniciar_programador_api():
    schedule.every(24).hours.do(actualizar_datos_desde_api)
    while True:
        schedule.run_pending()
        tiempo.sleep(60)

#threading.Thread(target=iniciar_programador_api, daemon=True).start()

# ------------------------------ CONSULTA SUPABASE ------------------------------

def get_data_from_supabase(table_name, start_date, end_date, page_size=1000):
    end_date += timedelta(days=1)
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    all_data = []
    offset = 0
    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .gte("datetime", start_iso)
            .lte("datetime", end_iso)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        data = response.data
        if not data:
            break
        all_data.extend(data)
        offset += page_size
        if len(data) < page_size:
            break

    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# ------------------------------ INTERFAZ ------------------------------

def main():
    st.title("Análisis de la Red Eléctrica Española")

    tab1, tab2, tab3, tab4 = st.tabs(["Descripción", "Consulta de datos", "Visualización", "Apagón"])

    with tab2:  # Mueve el contexto de la tab2 aquí para que `modo` se defina antes de usarse en session_state
        st.subheader("Consulta de datos")

        modo = st.radio("Tipo de consulta:", ["Últimos días", "Año específico", "Histórico"], horizontal=True,
                        key="query_mode_radio")
        st.session_state["modo_seleccionado"] = modo  # Guardar el modo en session_state

        tabla = st.selectbox("Selecciona la tabla:", list(ENDPOINTS.keys()), key="query_table_select")

        df = pd.DataFrame()  # Inicializamos el DataFrame para evitar errores

        if modo == "Últimos días":
            dias = st.selectbox("¿Cuántos días atrás?", [7, 14, 30], key="query_days_select")
            end_date_query = datetime.now(timezone.utc)
            start_date_query = end_date_query - timedelta(days=dias)
        elif modo == "Año específico":
            current_year = datetime.now().year
            # Se usa `key` para que el selectbox mantenga su estado
            año = st.selectbox("Selecciona el año:", [current_year - i for i in range(3)], index=0,
                               key="query_year_select")
            start_date_query = datetime(año, 1, 1, tzinfo=timezone.utc)
            end_date_query = datetime(año, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
        elif modo == "Histórico":
            # Si quieres cargar datos históricos de muchos años, ten cuidado con el rendimiento
            start_date_query = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Ejemplo de fecha inicial
            end_date_query = datetime.now(timezone.utc)

        with st.spinner("Consultando Supabase..."):
            st.session_state["tabla_seleccionada_en_tab2"] = tabla
            df = get_data_from_supabase(tabla, start_date_query, end_date_query)

        # Botón de actualización de datos desde la API
        if st.button("Actualizar datos desde la API"):
            with st.spinner("Actualizando datos desde la API (puede tardar)..."):
                get_data_for_last_x_years()
                st.success("Datos actualizados correctamente.")

        # Mostrar resultados después de la consulta de cualquier modo
        if not df.empty:
            st.session_state["ree_data"] = df
            st.session_state[
                "tabla"] = tabla  # Esto es redundante si usas tabla_seleccionada_en_tab2, pero lo mantengo por seguridad
            st.write(f"Datos recuperados: {len(df)} filas")
            st.write("Último dato:", df['datetime'].max())
            st.success("Datos cargados correctamente desde Supabase.")
        else:
            st.warning("No se encontraron datos para ese período.")

    with tab1:  # Reordeno esto para que la tab2 se cargue primero y defina el estado
        st.subheader("¿Qué es esta app?")
        st.markdown("""
        Esta aplicación se conecta con Supabase para consultar y visualizar datos de la Red Eléctrica Española
        como demanda, generación, intercambios y balance eléctrico.
        """)

    with tab3:
        st.subheader("Visualización")
        if "ree_data" in st.session_state and not st.session_state["ree_data"].empty:
            # Recuperamos el DataFrame principal de la sesión para el primer gráfico
            df = st.session_state["ree_data"]
            tabla = st.session_state["tabla_seleccionada_en_tab2"]  # Usamos la tabla seleccionada en tab2
            modo_actual = st.session_state.get("modo_seleccionado", "Últimos días")  # Obtener el modo

            if tabla == "demanda":
                fig = px.area(df, x="datetime", y="value", title="Demanda Eléctrica", labels={"value": "MW"})
                st.plotly_chart(fig, use_container_width=True)

                # --- Condición para mostrar la comparativa de años ---
                if modo_actual == "Histórico":
                    st.subheader("Comparativa de Demanda entre años")

                    # --- Condición para mostrar la comparativa de años ---
                    if modo_actual == "Histórico":
                        st.subheader("Comparativa de Demanda entre años")

                        # Definir los dos años específicos para la comparación: el año pasado y el anterior
                        current_year = datetime.now().year

                        # Los años que queremos comparar: el año anterior al actual y el año anterior a ese
                        target_years_for_comparison = [current_year - 2, current_year - 1]

                        # Obtener todos los años disponibles en el DataFrame del modo histórico
                        all_available_years_in_df = sorted(list(df['year'].unique()))

                        # Filtrar solo los años que queremos comparar y que realmente están disponibles en el df
                        years_for_comparison = [
                            year for year in target_years_for_comparison
                            if year in all_available_years_in_df
                        ]

                        if len(years_for_comparison) == 2:
                            # Asegurarse de que están en el orden deseado (ej. [2023, 2024])
                            years_for_comparison.sort()
                        elif len(years_for_comparison) == 1:
                            st.info(
                                f"Solo se encontró un año de los deseados ({years_for_comparison[0]}) en modo histórico para la comparación. Se necesitan ambos años ({target_years_for_comparison[0]} y {target_years_for_comparison[1]}).")
                            years_for_comparison = []  # Vaciar para no intentar graficar
                        else:  # len(years_for_comparison) == 0
                            st.info(
                                f"No se encontraron datos para los años {target_years_for_comparison[0]} y {target_years_for_comparison[1]} en modo histórico para la comparación.")
                            years_for_comparison = []  # Vaciar para no intentar graficar

                        if years_for_comparison:  # Solo procede si tenemos al menos dos años para comparar
                            df_comparison_demanda = df.copy()  # Usamos el df ya cargado en modo histórico

                            # Nos aseguramos de que solo tengamos los años que queremos comparar
                            df_filtered_comparison = df_comparison_demanda[
                                df_comparison_demanda['year'].isin(years_for_comparison)].copy()

                            # Convertimos la columna 'datetime' a una fecha sin el año, para comparar día a día
                            # Esta 'sort_key' se usa para el gráfico horario
                            df_filtered_comparison['sort_key'] = df_filtered_comparison['datetime'].apply(
                                lambda dt: dt.replace(year=2000)  # Usar un año base para ordenar correctamente
                            )
                            df_filtered_comparison = df_filtered_comparison.sort_values('sort_key')

                            # --- Gráfico de Demanda Horaria General Comparativa ---
                            fig_comp_hourly = px.line(
                                df_filtered_comparison,
                                x="sort_key",  # Usamos la 'sort_key' que es datetime
                                y="value",
                                color="year",
                                title="Demanda Horaria - Comparativa",
                                labels={"sort_key": "Mes y Día", "value": "Demanda (MW)", "year": "Año"},
                                hover_data={"year": True, "datetime": "|%Y-%m-%d %H:%M"}
                            )
                            fig_comp_hourly.update_xaxes(
                                tickformat="%b %d")  # Formato para mostrar Mes y Día en el eje X
                            st.plotly_chart(fig_comp_hourly, use_container_width=True)

                            # --- Gráficos de Comparación de Métricas Diarias (Media, Mediana, Mínima, Máxima) ---
                            # Agrupar por 'year' y 'month-day' para obtener las métricas diarias para cada año
                            metrics_comp = df_filtered_comparison.groupby(
                                ['year', df_filtered_comparison['datetime'].dt.strftime('%m-%d')])['value'].agg(
                                ['mean', 'median', 'min', 'max']).reset_index()
                            metrics_comp.columns = ['year', 'month_day', 'mean', 'median', 'min', 'max']

                            # La corrección para el ValueError: day is out of range for month está aquí
                            metrics_comp['sort_key'] = pd.to_datetime('2000-' + metrics_comp['month_day'],
                                                                      format='%Y-%m-%d')
                            metrics_comp = metrics_comp.sort_values('sort_key')

                            metric_names = {
                                'mean': 'Media diaria de demanda',
                                'median': 'Mediana diaria de demanda',
                                'min': 'Mínima diaria de demanda',
                                'max': 'Máxima diaria de demanda',
                            }

                            for metric in ['mean', 'median', 'min', 'max']:
                                fig = px.line(
                                    metrics_comp,
                                    x="sort_key",  # <--- CAMBIO CLAVE: Usar 'sort_key' (tipo datetime) para el eje X
                                    y=metric,
                                    color="year",
                                    title=metric_names[metric],
                                    labels={"sort_key": "Fecha (Mes-Día)", metric: "Demanda (MW)", "year": "Año"},
                                    # <--- CAMBIO EN ETIQUETA
                                )
                                fig.update_xaxes(tickformat="%b %d")  # Formato para mostrar solo Mes y Día
                                # Si las líneas siguen entrecortadas, considera añadir `connectgaps=True`
                                # fig.update_traces(connectgaps=True)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No hay suficientes datos de Demanda disponibles para la comparación.")

                        # --- Nuevo gráfico: Identificación de años outliers ---
                        st.subheader("Identificación de Años Outliers (Demanda Anual Total)")

                        # Asegurarse de que el df tiene la columna 'year'
                        if 'year' not in df.columns:
                            df['year'] = df['datetime'].dt.year

                        # Agrupar por año para obtener la demanda total anual
                        df_annual_summary = df.groupby('year')['value'].sum().reset_index()
                        df_annual_summary.rename(columns={'value': 'total_demand_MW'}, inplace=True)

                        if not df_annual_summary.empty and len(df_annual_summary) > 1:
                            # Calcular Q1, Q3 y el IQR
                            Q1 = df_annual_summary['total_demand_MW'].quantile(0.25)
                            Q3 = df_annual_summary['total_demand_MW'].quantile(0.75)
                            IQR = Q3 - Q1

                            # Calcular los límites para los outliers
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            # Identificar los años que son outliers
                            df_annual_summary['is_outlier'] = (
                                    (df_annual_summary['total_demand_MW'] < lower_bound) |
                                    (df_annual_summary['total_demand_MW'] > upper_bound)
                            )

                            # Crear el gráfico de barras
                            fig_outliers = px.bar(
                                df_annual_summary,
                                x='year',
                                y='total_demand_MW',
                                color='is_outlier',  # Colorear las barras si son outliers
                                title='Demanda Total Anual y Años Outlier',
                                labels={'total_demand_MW': 'Demanda Total Anual (MW)', 'year': 'Año',
                                        'is_outlier': 'Es Outlier'},
                                color_discrete_map={False: 'skyblue', True: 'red'}  # Definir colores
                            )

                            st.plotly_chart(fig_outliers, use_container_width=True)

                            # Mostrar los años identificados como outliers
                            outlier_years = df_annual_summary[df_annual_summary['is_outlier']]['year'].tolist()
                            if outlier_years:
                                st.warning(
                                    f"Se han identificado los siguientes años como outliers: {', '.join(map(str, outlier_years))}")
                            else:
                                st.info("No se han identificado años outliers significativos (según el método IQR).")
                        elif not df_annual_summary.empty and len(df_annual_summary) <= 1:
                            st.info("Se necesitan al menos 2 años de datos para calcular outliers de demanda anual.")
                        else:
                            st.warning("No hay datos anuales disponibles para calcular outliers.")
                    else:  # if modo_actual != "Histórico"
                        st.info(
                            "Selecciona el modo 'Histórico' en la pestaña 'Consulta de datos' para ver la comparativa de años y la identificación de outliers.")

            elif tabla == "balance":
                fig = px.bar(df, x="datetime", y="value", color="primary_category", barmode="group", title="Balance Eléctrico")
                st.plotly_chart(fig, use_container_width=True)
            elif tabla == "generacion":
                fig = px.line(df, x="datetime", y="value", color="primary_category", title="Generación")
                st.plotly_chart(fig, use_container_width=True)
            elif tabla == "intercambios":
                st.subheader("Mapa Coroplético de Intercambios Eléctricos")

                # Agrupamos y renombramos columnas
                df_map = df.groupby("primary_category")["value"].sum().reset_index()
                df_map.columns = ["pais_original", "Total"]

                # Mapeo de nombres a inglés (para coincidir con el GeoJSON)
                nombre_map = {
                    "francia": "France",
                    "portugal": "Portugal",
                    "andorra": "Andorra",
                    "marruecos": "Morocco"
                }
                df_map["Country"] = df_map["pais_original"].map(nombre_map)

                df_map = df_map.dropna(subset=["Country"])

                # Cargar el archivo GeoJSON
                with open("world_countries_with_andorra.json", "r", encoding="utf-8") as f:
                    world_geo = json.load(f)

                # Crear el mapa
                world_map = folium.Map(location=[40, -3], zoom_start=4)

                folium.Choropleth(
                    geo_data=world_geo,
                    data=df_map,
                    columns=["Country", "Total"],
                    key_on="feature.properties.name",
                    fill_color="RdBu",
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name="Saldo neto de energía (MWh)"
                ).add_to(world_map)

                # Mostrar en Streamlit
                st_folium(world_map, width=1285)

            elif tabla == "intercambios_baleares":
                fig = px.line(df, x="datetime", y="value", title="Intercambios con Baleares")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.line(df, x="datetime", y="value", title="Visualización")
                st.plotly_chart(fig, use_container_width=True)


            with st.expander("Ver datos en tabla"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Consulta primero los datos desde la pestaña anterior.")

    with tab4:
        st.subheader("Gráficos de interés sobre el apagón")
        st.markdown("(Pendiente de implementación)")

if __name__ == "__main__":
    main()
