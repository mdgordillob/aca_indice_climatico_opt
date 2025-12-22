from calcular_anomalias_temperatura import procesar_anomalias_temperatura
from calcular_anomalias_viento import procesar_anomalias_viento
from calcular_anomalias_lluvia import procesar_anomalias_lluvia
import os

def procesar_anomalias_region(region_name, shapefile_path, archivo_comparar_location, archivo_percentiles_temperatura, archivo_percentiles_wind, archivo_percentiles_lluvia, output_base_path):
    """
    Procesa las anomalías para una región específica.

    Parámetros:
        region_name (str): Nombre de la región.
        shapefile_path (str): Ruta del archivo shapefile de la región.
        archivo_comparar_location (str): Ruta de los archivos GRIB de comparación.
        archivo_percentiles_temperatura (str): Ruta del archivo de percentiles de temperatura.
        archivo_percentiles_wind (str): Ruta del archivo de percentiles de viento.
        archivo_percentiles_lluvia (str): Ruta del archivo de percentiles de lluvia.
        output_base_path (str): Ruta base para los archivos de salida.
    """
    output_csv_path_temperatura = os.path.join(output_base_path, region_name, "anomalies_temperature_combined.csv")
    output_csv_path_viento = os.path.join(output_base_path, region_name, "anomalies_wind_combined.csv")
    output_netcdf = os.path.join(output_base_path, region_name)
    salida_lluvia = "anomalies_precipitation_combined.csv"
    salida_sequia = "anomalies_drought_combined.csv"

    procesar_anomalias_temperatura(archivo_percentiles_temperatura, archivo_comparar_location, output_csv_path_temperatura, shapefile_path, output_netcdf)
    procesar_anomalias_viento(archivo_percentiles_wind, archivo_comparar_location, output_csv_path_viento, shapefile_path, output_netcdf)
    df1, df2 = procesar_anomalias_lluvia(shapefile_path, archivo_percentiles_lluvia, archivo_comparar_location, output_netcdf, salida_lluvia, salida_sequia)
    print(f'✓ Proceso finalizado para {region_name}')

if __name__ == "__main__":
    # Obtener el directorio del script y navegar a la raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Resolver todas las rutas desde la raíz del proyecto
    archivo_comparar_location = os.path.join(project_root, "data", "raw", "era5")
    archivo_percentiles_temperatura = os.path.join(project_root, "data", "processed", "era5_temperatura_percentil.nc")
    archivo_percentiles_wind = os.path.join(project_root, "data", "processed", "era5_wind_percentil.nc")
    archivo_percentiles_lluvia = os.path.join(project_root, "data", "processed")
    output_base_path = os.path.join(project_root, "data", "processed")

    regiones = [
        {"name": "anomalias_colombia", "shapefile": os.path.join(project_root, "data", "shapefiles", "colombia_4326.shp")},
        {"name": "anomalias_cundinamarca_bogota", "shapefile": os.path.join(project_root, "data", "shapefiles", "Cundinamarca_Bogota_4326.shp")},
        {"name": "anomalias_antioquia", "shapefile": os.path.join(project_root, "data", "shapefiles", "antioquia_4326.shp")},
        {"name": "anomalias_valle_cauca", "shapefile": os.path.join(project_root, "data", "shapefiles", "valle_cauca_4326.shp")},
        {"name": "anomalias_san_andres_providencia", "shapefile": os.path.join(project_root, "data", "shapefiles", "san_andres_providencia.shp")},
        {"name": "anomalias_medellin", "shapefile": os.path.join(project_root, "data", "shapefiles", "medellin_4326.shp")},
        {"name": "anomalias_cali", "shapefile": os.path.join(project_root, "data", "shapefiles", "cali_4326.shp")},
        {"name": "anomalias_bogota", "shapefile": os.path.join(project_root, "data", "shapefiles", "bogota.shp")}
    ]

    print("=" * 60)
    print("CALCULANDO ANOMALÍAS REGIONALIZADAS")
    print("=" * 60)
    print(f"Raíz del proyecto: {project_root}")
    print(f"Directorio de datos: {archivo_comparar_location}")
    print("=" * 60)

    for region in regiones:
        try:
            os.makedirs(os.path.join(output_base_path, region["name"]), exist_ok=True)
        except Exception as e:
            print(f"⚠️  No se pudo crear el directorio para {region['name']}: {e}")
            continue

        try:
            print(f"\nProcesando región: {region['name']}")
            procesar_anomalias_region(
                region_name=region["name"],
                shapefile_path=region["shapefile"],
                archivo_comparar_location=archivo_comparar_location,
                archivo_percentiles_temperatura=archivo_percentiles_temperatura,
                archivo_percentiles_wind=archivo_percentiles_wind,
                archivo_percentiles_lluvia=archivo_percentiles_lluvia,
                output_base_path=output_base_path
            )
        except Exception as e:
            print(f'❌ Error en el proceso para {region["name"]}: {e}')
            continue
    
    print("=" * 60)
    print("✓ Proceso finalizado")
    print("=" * 60)
