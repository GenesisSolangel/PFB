# Importamos las librerías necesarias
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import uuid

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

# Función de extracción de datos de los últimos x años, devuelve un DataFrame de Pandas
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

            # Iteramos sobre cada endpoint
            for name, (path, granularity) in ENDPOINTS.items():
                params = {
                    "start_date": month_start.strftime("%Y-%m-%dT%H:%M"),
                    "end_date": end_date_for_request.strftime("%Y-%m-%dT%H:%M"),
                    "geo_trunc": "electric_system",
                    "geo_limit": "peninsular",
                    "geo_ids": "8741"
                }

                month_data = get_data(name, (path, granularity), params)

                # Y sacamos los datos
                if month_data:
                    df = pd.DataFrame(month_data)
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
                    df['endpoint'] = name
                    df['extraction_timestamp'] = datetime.utcnow()
                    df['record_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

                    df = df[['record_id', 'value', 'percentage', 'datetime',
                             'primary_category', 'sub_category', 'year', 'month',
                             'day', 'hour', 'endpoint', 'extraction_timestamp']]

                    all_dfs.append(df)

                time.sleep(1)
                print("iter")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

# Obtenemos los datos de los últimos 3 años a partir de hoy
ree_data_df = get_data_for_last_x_years(num_years=3)

df_demanda = ree_data_df[ree_data_df["endpoint"] == "demanda"].drop(columns=["endpoint", "sub_category"])
df_balance = ree_data_df[ree_data_df["endpoint"] == "balance"].drop(columns=["endpoint"])
df_generacion = ree_data_df[ree_data_df["endpoint"] == "generacion"].drop(columns=["endpoint", "sub_category"])
df_intercambios = ree_data_df[ree_data_df["endpoint"] == "intercambios"].drop(columns=["endpoint"])
df_intercambios_baleares = ree_data_df[ree_data_df["endpoint"] == "intercambios_baleares"].drop(columns=["endpoint", "sub_category"])


############################################

# ---------------- SUPABASE: Conexión e Inserción ---------------- #
from supabase import create_client, Client

# 🔐 Configura tus credenciales de Supabase
url = "https://iuphsrhiqotzclypaqnt.supabase.co"  # Sustituye con tu URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml1cGhzcmhpcW90emNseXBhcW50Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMzUwMTUsImV4cCI6MjA2NDYxMTAxNX0.K1tJ9T4jVUfQtlHoZbMMriKtRVlzLW8AG5aE2r6c6-w"                      # Sustituye con tu anon key
supabase: Client = create_client(url, key)

# 📁 Diccionario tabla → DataFrame
tablas_dfs = {
    "demanda": df_demanda,
    "balance": df_balance,
    "generacion": df_generacion,
    "intercambios": df_intercambios,
    "intercambios_baleares": df_intercambios_baleares
}

#para insertar cada DataFrame
def insertar_en_supabase(nombre_tabla, df):
    import uuid

    df = df.copy()

    #Asegurar UUID únicos
    df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Convertir fechas a string ISO
    for col in ["datetime", "extraction_timestamp"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    #Reemplazar NaN por None
    df = df.where(pd.notnull(df), None)

    #Convertir a lista de diccionarios e insertar
    data = df.to_dict(orient="records")

    try:
        response = supabase.table(nombre_tabla).insert(data).execute()
        print(f"✅ Insertados en '{nombre_tabla}': {len(data)} filas")
    except Exception as e:
        print(f"❌ Error al insertar en '{nombre_tabla}': {e}")

# inserciones por tabla
for tabla, df in tablas_dfs.items():
    insertar_en_supabase(tabla, df)
