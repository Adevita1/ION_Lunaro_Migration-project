# Databricks notebook source
Notebook que da info de metadata y consumo con graficos

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max as spark_max, count, lit, current_timestamp, when
from pyspark.sql.types import StringType
from delta.tables import DeltaTable
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Inicializar Spark Session
spark = SparkSession.builder.appName("AnálisisTablasDelta").getOrCreate()

# Lista de rutas de archivos
file_paths = [
    "dbfs:/mnt/bronze/autoloader/dynamics_ap_creditos_cene",
    "dbfs:/mnt/silver/dynamics_ap_creditos_cene",
    "dbfs:/mnt/silver/dynamics_ap_creditos_cene_individual",
    "dbfs:/mnt/silver/dynamics_ap_cat_prelacion_pago",
    "dbfs:/mnt/silver/dynamics_ap_cat_operacion_diferencial_tasa_int",
    "dbfs:/mnt/silver/dynamics_ap_cat_revocable",
    "dbfs:/mnt/silver/dynamics_rsk_insumo_reservas_respaldo",
    "dbfs:/mnt/bronze/saf_pr_garantias",
    "dbfs:/mnt/silver/saf_pr_garantias",
    "dbfs:/mnt/bronze/saf_pr_clientes_x_credito",
    "dbfs:/mnt/silver/saf_pr_clientes_x_credito",
    "dbfs:/mnt/bronze/saf_cl_dir_clientes",
    "dbfs:/mnt/silver/saf_cl_dir_clientes",
    "dbfs:/mnt/bronze/saf_cl_id_clientes",
    "dbfs:/mnt/silver/saf_cl_id_clientes",
    "dbfs:/mnt/bronze/saf_cl_clientes",
    "dbfs:/mnt/silver/saf_cl_clientes",
    "dbfs:/mnt/bronze/saf_cl_personas_juridicas",
    "dbfs:/mnt/silver/saf_cl_personas_juridicas",
    "dbfs:/mnt/bronze/saf_cl_subact_economica",
    "dbfs:/mnt/silver/saf_cl_subact_economica",
    "dbfs:/mnt/bronze/saf_pr_creditos",
    "dbfs:/mnt/silver/saf_pr_creditos",
    "dbfs:/mnt/bronze/saf_cl_personas_fisicas",
    "dbfs:/mnt/silver/saf_cl_personas_fisicas",
    "dbfs:/mnt/bronze/saf_pr_traspaso_cartera_vencida",
    "dbfs:/mnt/silver/saf_pr_traspaso_cartera_vencida",
    "dbfs:/mnt/bronze/saf_pr_encabezado_pago",
    "dbfs:/mnt/silver/saf_pr_encabezado_pago",
    "dbfs:/mnt/bronze/saf_pr_rubro_cobro_x_credito",
    "dbfs:/mnt/silver/saf_pr_rubro_cobro_x_credito",
    "dbfs:/mnt/bronze/saf_pr_tipo_credito",
    "dbfs:/mnt/silver/saf_pr_tipo_credito",
    "dbfs:/mnt/bronze/saf_pa_tasas_interes",
    "dbfs:/mnt/silver/saf_pa_tasas_interes",
    "dbfs:/mnt/bronze/saf_pr_detalle_pago",
    "dbfs:/mnt/silver/saf_pr_detalle_pago",
    "dbfs:/mnt/bronze/saf_cf_calendarios",
    "dbfs:/mnt/silver/saf_cf_calendarios"
    # Añade aquí el resto de tus rutas
]

# Configuración del cluster según la imagen
DBU_RATE_PER_HOUR = 3.37  # DBU por hora para el cluster completo

def estimate_dbu_consumption(duration_seconds):
    duration_hours = duration_seconds / 3600
    return DBU_RATE_PER_HOUR * duration_hours

def analyze_file(df, file_path):
    start_time = time.time()
    try:
        num_rows = df.count()
        num_columns = len(df.columns)
        column_names = df.columns
        null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        data_types = {f.name: f.dataType.simpleString() for f in df.schema.fields}
        sample = df.limit(5).collect()
        end_time = time.time()
        duration = end_time - start_time
        estimated_dbu = estimate_dbu_consumption(duration)
        return {
            "file_path": file_path,
            "num_rows": num_rows,
            "num_columns": num_columns,
            "column_names": column_names,
            "null_counts": null_counts,
            "data_types": data_types,
            "duration": duration,
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            "sample": sample,
            "estimated_dbu": estimated_dbu
        }
    except Exception as e:
        print(f"Error en analyze_file para {file_path}: {str(e)}")
        return None

def process_file(file_path):
    try:
        if not DeltaTable.isDeltaTable(spark, file_path):
            print(f"Error: {file_path} no es una tabla Delta válida")
            return None

        source_df = spark.read.format("delta").load(file_path)
        return analyze_file(source_df, file_path)
    
    except Exception as e:
        print(f"Error procesando {file_path}: {str(e)}")
        return None

def plot_null_counts(result):
    null_counts = pd.DataFrame.from_dict(result['null_counts'], orient='index', columns=['null_count'])
    null_counts = null_counts.sort_values('null_count', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=null_counts.index, y='null_count', data=null_counts)
    plt.title(f"Conteo de valores nulos por columna - {result['file_path'].split('/')[-1]}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_data_types(result):
    data_types = pd.DataFrame.from_dict(result['data_types'], orient='index', columns=['data_type'])
    plt.figure(figsize=(10, 6))
    data_types['data_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f"Distribución de tipos de datos - {result['file_path'].split('/')[-1]}")
    plt.ylabel('')
    plt.show()

# Procesar cada archivo
results = []
total_dbu = 0
session_start_time = time.time()

for path in file_paths:
    result = process_file(path)
    if result:
        results.append(result)
        total_dbu += result['estimated_dbu']

session_end_time = time.time()
session_duration = session_end_time - session_start_time
total_session_dbu = estimate_dbu_consumption(session_duration)

# Imprimir resultados y generar gráficos
for result in results:
    print(f"\nArchivo: {result['file_path']}")
    print(f"Filas: {result['num_rows']}")
    print(f"Columnas: {result['num_columns']}")
    print(f"Nombres de columnas: {', '.join(result['column_names'])}")
    print(f"Datos nulos por columna: {result['null_counts']}")
    print(f"Tipos de datos: {result['data_types']}")
    print(f"Tiempo de análisis: {result['duration']:.2f} segundos")
    print(f"Inicio: {result['start_time']}")
    print(f"Fin: {result['end_time']}")
    print(f"DBU estimado: {result['estimated_dbu']:.4f}")
    print("Muestra de datos (primeras 5 filas):")
    for row in result['sample']:
        print(row)
    print("...")
    
    # Generar gráficos
    plot_null_counts(result)
    plot_data_types(result)

if not results:
    print("No se procesó ningún archivo con éxito.")
else:
    print(f"\nDuración total de la sesión: {session_duration:.2f} segundos")
    print(f"Consumo total estimado de DBU de la sesión: {total_session_dbu:.4f}")
    print(f"Suma de DBU estimado por archivo: {total_dbu:.4f}")

# Gráfico comparativo de DBU por archivo
plt.figure(figsize=(10, 6))
sns.barplot(x=[r['file_path'].split('/')[-1] for r in results], y=[r['estimated_dbu'] for r in results])
plt.title("Consumo estimado de DBU por archivo")
plt.xticks(rotation=45)
plt.ylabel("DBU estimado")
plt.tight_layout()
plt.show()
