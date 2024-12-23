## Descripción de la Estructura

### `data/`
Almacena todos los datos relacionados al proyecto. Separa los datos crudos (`raw`) de los procesados (`processed`).

### `notebooks/`
Contiene los Jupyter notebooks para explorar datos y documentar el flujo de trabajo.

### `src/`
Código fuente organizado en scripts y utilidades reutilizables.

#### Descargar datos
Para descargar los datos, se debe ejecutar el script `ecmwf_descarga.py` ubicado en la carpeta `src/scripts`. Este script descarga los datos de la página web del ECMWF y los almacena en la carpeta `data/raw/era5`.

#### Unir los datos
Para unir los datos descargados, se debe ejecutar el script `unir_archivos.py` ubicado en la carpeta `src/scripts`. Este script une los datos descargados y los almacena en la carpeta `data/processed`.

### `docs/`
Documentación importante del proyecto, como metodología y resultados.

### `reports/`
Reportes generados para socialización, gráficos y tablas.

### `tests/`
Pruebas unitarias para verificar la funcionalidad del código.

### `articles/`
Espacio para redactar artículos científicos o reportes.

### Archivos raíz:
- **`README.md`**: Instrucciones iniciales para desarrolladores y usuarios.
- **`requirements.txt`**: Lista de dependencias del proyecto para instalación rápida.
- **`.gitignore`**: Para evitar incluir archivos innecesarios en el repositorio.
- **`LICENSE`**: Licencia que define cómo se puede usar el proyecto.
