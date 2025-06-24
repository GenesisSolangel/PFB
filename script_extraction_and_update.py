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
import os
import json
import folium
import threading
import schedule
import time as tiempo
from dotenv import load_dotenv

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
def get_data_for_last_x_years(num_years=10):
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

# Solo ejecutamos esta función una vez al principio, para obtener los datos históricos
#get_data_for_last_x_years(num_years=10)

# Función para evitar duplicar datos en la base de datos
def filtrar_registros_nuevos(nombre_tabla, df_local):
    from supabase import create_client, Client
    import os

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    datetimes = df_local['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z').tolist()

    # Para no saturar la petición, consultamos por rangos de fechas
    existing_records = []
    chunk_size = 1000

    for i in range(0, len(datetimes), chunk_size):
        chunk = datetimes[i:i + chunk_size]
        response = supabase.table(nombre_tabla).select('datetime').in_('datetime', chunk).execute()
        existing_datetimes = [item['datetime'] for item in response.data]
        existing_records.extend(existing_datetimes)

    if existing_records:
        df_local = df_local[~df_local['datetime'].astype(str).isin(existing_records)]

    return df_local

# Función para actualizar la base de datos
def actualizar_datos_desde_api():
    print(f"[{datetime.now()}] ⏳ Ejecutando extracción desde API...")
    current_date = datetime.now()
    start_date = current_date - timedelta(days=3)

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
                # ✅ Aquí filtramos duplicados antes de insertar
                df = filtrar_registros_nuevos(tabla, df)
                if not df.empty:
                    insertar_en_supabase(tabla, df)
                else:
                    print(f"✅ No hay nuevos datos para insertar en la tabla {tabla}.")


if __name__ == "__main__":
    actualizar_datos_desde_api()
