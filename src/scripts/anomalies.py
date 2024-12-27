import numpy as np

def calculate_anomalies(grid_data, percentiles):
    anomalies_above = np.zeros_like(grid_data)
    anomalies_below = np.zeros_like(grid_data)
    anomalies_above[grid_data > percentiles[90]] = 1
    anomalies_below[grid_data < percentiles[10]] = 1
    return anomalies_above, anomalies_below

def sum_anomalies(anomalies):
    return np.sum(anomalies, axis=0)

def calculate_mean(anomalies):
    return np.mean(anomalies, axis=0)

def calculate_std(anomalies):
    return np.std(anomalies, axis=0)

def normalize_anomalies(anomalies, mean, std):
    return (anomalies - mean) / std

def process_grid_data(grid_data, percentiles):
    anomalies_above, anomalies_below = calculate_anomalies(grid_data, percentiles)
    
    sum_anom_above = sum_anomalies(anomalies_above)
    mean_anom_above = calculate_mean(anomalies_above)
    std_anom_above = calculate_std(anomalies_above)
    normalized_anomalies_above = normalize_anomalies(sum_anom_above, mean_anom_above, std_anom_above)
    
    sum_anom_below = sum_anomalies(anomalies_below)
    mean_anom_below = calculate_mean(anomalies_below)
    std_anom_below = calculate_std(anomalies_below)
    normalized_anomalies_below = normalize_anomalies(sum_anom_below, mean_anom_below, std_anom_below)
    
    return normalized_anomalies_above, normalized_anomalies_below

# Example usage
if __name__ == "__main__":
    grid_data = np.random.rand(30, 10, 10)  # Example grid data for 30 days
    percentiles = {10: 0.1, 90: 0.9}  # Example percentiles
    result_above, result_below = process_grid_data(grid_data, percentiles)
    print("Normalized anomalies above 90th percentile:\n", result_above)
    print("Normalized anomalies below 10th percentile:\n", result_below)