import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANOMALIES_DIR = BASE_DIR / 'data' / 'processed' / 'anomalias_colombia'
OUTPUT_DIR = BASE_DIR / 'articles' / 'graficas' / 'anomalias_colombia' / 'recreated'
GRAFICAS_PATH = BASE_DIR / 'src' / 'scripts' / 'graficas.py'


def load_graficas_module(path: Path):
    spec = importlib.util.spec_from_file_location('graficas', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_repo_style_plots():
    graficas = load_graficas_module(GRAFICAS_PATH)

    temp_csv = ANOMALIES_DIR / 'anomalies_temperature_combined.csv'
    rain_csv = ANOMALIES_DIR / 'anomalies_precipitation_combined.csv'
    drought_csv = ANOMALIES_DIR / 'anomalies_drought_combined.csv'
    wind_csv = ANOMALIES_DIR / 'anomalies_wind_combined.csv'

    print('Generating repo-style anomaly plots in:')
    print(f'  {OUTPUT_DIR}')
    print('Using source CSVs:')
    print(f'  {temp_csv}')
    print(f'  {rain_csv}')
    print(f'  {drought_csv}')
    print(f'  {wind_csv}')

    graficas.plot_temp_anomalies(str(temp_csv), str(OUTPUT_DIR))
    graficas.plot_rainfall_anomalies(str(rain_csv), str(OUTPUT_DIR))
    graficas.plot_drought_anomalies(str(drought_csv), str(OUTPUT_DIR))
    graficas.plot_wind_anomalies(str(wind_csv), str(OUTPUT_DIR))
    graficas.plot_ICA(str(temp_csv), str(rain_csv), str(drought_csv), str(wind_csv), str(OUTPUT_DIR))


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_repo_style_plots()
