# Proyecto de Análisis de Datos Meteorológicos

## Instalación de Librerías

Para instalar las librerías necesarias, se recomienda crear un ambiente virtual con el siguiente comando:

```bash
python3 -m venv myenv
```

Luego, se deben instalar los paquetes requeridos ejecutando:

```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
├── data/
│   ├── raw/         # Datos crudos descargados
│   ├── processed/   # Datos procesados
│
├── notebooks/       # Jupyter notebooks para exploración de datos
│
├── src/             # Código fuente y scripts
│   ├── scripts/     # Scripts de procesamiento de datos
│   ├── excel/       # Archivos de validación en Excel
│
├── articles/        # Documentación y artículos científicos
│
├── README.md        # Instrucciones del proyecto
├── requirements.txt # Dependencias del proyecto
├── .gitignore       # Archivos a excluir en el repositorio
├── LICENSE          # Licencia del proyecto
```

---

## Flujo de Trabajo

### 1. Descargar Datos
Para descargar los datos, ejecutar:

```bash
python src/scripts/descargar_datos.py
```

Este script invoca la función `ecmwf_descarga.py` y almacena los datos en `data/raw/era5`.

### 2. Unir Datos
Para unir los archivos descargados, ejecutar:

```bash
python src/scripts/unir_archivos.py
```

Los datos combinados se guardan en `data/processed`.

### 3. Calcular Percentiles
Para calcular los percentiles de temperatura, lluvia y viento, ejecutar:

```bash
python src/scripts/calcular_percentil_temperatura.py
python src/scripts/calcular_percentil_lluvia.py
python src/scripts/calcular_percentil_viento.py
```

Los resultados se almacenan en `data/processed`.

### 4. Calcular Anomalías
Para calcular anomalías en temperatura, lluvia y viento, ejecutar:

```bash
python src/scripts/calcular_anomalias_temperatura.py
python src/scripts/calcular_anomalias_lluvia.py
python src/scripts/calcular_anomalias_viento.py
```

Los resultados se guardan en `data/processed`.

### 5. Generar Gráficas
Para visualizar los resultados, ejecutar:

```bash
python src/scripts/graficas.py
```

### 6. Análisis del Nivel del Mar
Dado que no se tenía información disponible, se utilizó `psmsl.org` y se creó un script exclusivo para su procesamiento:

```bash
python src/scripts/sealevel.py
```

### 7. Cálculo de Anomalías por Regiones
Para calcular anomalías por regiones usando shapefiles, ejecutar:

```bash
python src/scripts/calcular_anomalias_regiones.py
```

**Nota:** Las áreas analizadas deben ser grandes (mínimo a nivel departamental), ya que áreas más pequeñas pueden no ser representativas.

### 8. Pronóstico de Anomalías Climáticas Mensuales
Para generar pronósticos de anomalías climáticas mensuales usando AutoTS, ejecutar:

```bash
python src/forecast_scripts/forecast_ica_monthly.py
```

Este script realiza:
- **Carga de datos**: Lee archivos NetCDF mensuales (temperatura, viento) y CSV (precipitación, sequía)
- **Extracción regional**: Calcula la media espacial de las anomalías para la región Cundinamarca-Bogotá
- **Entrenamiento del modelo**: Utiliza AutoTS con estrategias de ensamble (simple y dist) y validación backwards
- **Generación de pronósticos**: Produce pronósticos de 12 meses con intervalos de confianza del 90%
- **Cálculo del Índice Climático**: Crea un índice composite ponderado de las anomalías principales
- **Visualización y exportación**: Genera gráficas (6 subplots), archivos CSV y reporte de resumen

**Variables de pronóstico:**
- `temperature_t_90`: Anomalía de temperatura percentil 90
- `temperature_t_10`: Anomalía de temperatura percentil 10
- `wind_anomalies_above`: Anomalías de viento
- `precipitation_Anomalia_Lluvia`: Anomalías de precipitación
- `drought_Anomalia_Sequia`: Anomalías de sequía
- `Climate_Index`: Índice composite de clima (promedio ponderado de las 4 anomalías principales)

**Salidas generadas** en `articles/graficas/forecast_cundinamarca_bogota/`:
- `forecasts_all_variables.png`: Visualización con 6 gráficos (histórico + pronóstico con intervalos de confianza)
- `forecast_cundinamarca_bogota_point.csv`: Pronósticos puntuales
- `forecast_cundinamarca_bogota_upper_90ci.csv`: Límites superiores del intervalo de confianza 90%
- `forecast_cundinamarca_bogota_lower_90ci.csv`: Límites inferiores del intervalo de confianza 90%
- `forecast_cundinamarca_bogota_combined.csv`: Datos históricos + pronóstico combinados
- `forecast_summary_report.txt`: Reporte de resumen con metadatos y recomendaciones

---

## Archivos de Validación en Excel
En la carpeta `src/excel/` se encuentran archivos de validación en Excel:

- `precipitacion_sequial.xls`
- `temperatura.xls`
- `viento.xls`

Estos archivos son un ejemplo práctico utilizado para validar el proceso y también pueden servir como referencia de la metodología aplicada.

---

## Diagrama del Proceso

![Diagrama del proceso](https://github.com/user-attachments/assets/8e9416ea-6210-4be5-a6f1-3e51bf23282f)

---

## Licencia
Este proyecto está bajo la licencia especificada en el archivo `LICENSE`.
