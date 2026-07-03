"""
Error Diagnostics and Distribution Analysis for ETS+ENSO Models

This script provides comprehensive diagnostics for model residuals including:
- Normality tests (Shapiro-Wilk, Jarque-Bera, Anderson-Darling)
- Q-Q plots
- Histogram with normal overlay
- Autocorrelation analysis (ACF/PACF)
- Heteroscedasticity tests
- Time series plots of residuals

Use this to validate model assumptions and identify potential issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, jarque_bera, anderson, kstest, normaltest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelErrorDiagnostics:
    """
    Comprehensive error diagnostics for time series forecasting models.
    """
    
    def __init__(self, output_path=None):
        """
        Initialize diagnostics handler.
        
        Args:
            output_path (str): Path to save diagnostic outputs
        """
        if output_path is None:
            output_path = os.path.join(".", "articles", "graficas", "error_diagnostics")
        
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
        self.normality_results = []
        self.autocorr_results = []
        
        print(f"Error Diagnostics Output: {os.path.abspath(self.output_path)}")
    
    def calculate_residuals(self, y_true, y_pred):
        """
        Calculate various types of residuals.
        
        Args:
            y_true (pd.Series or np.ndarray): Actual values
            y_pred (pd.Series or np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary with different residual types
        """
        # Convert to arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Regular residuals
        residuals = y_true - y_pred
        
        # Standardized residuals
        residual_std = np.std(residuals)
        standardized = residuals / residual_std if residual_std > 0 else residuals
        
        # Studentized residuals (more robust)
        # Remove outliers for better std estimate
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        robust_std = iqr / 1.349  # Convert IQR to std for normal dist
        studentized = residuals / robust_std if robust_std > 0 else residuals
        
        return {
            'raw': residuals,
            'standardized': standardized,
            'studentized': studentized
        }
    
    def normality_tests(self, residuals, variable_name):
        """
        Perform comprehensive normality tests on residuals.
        
        Args:
            residuals (np.ndarray): Residuals to test
            variable_name (str): Name of the variable
            
        Returns:
            dict: Test results
        """
        results = {'variable': variable_name}
        
        # Remove NaN values
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 3:
            print(f"  [WARNING] Insufficient data for {variable_name}")
            return None
        
        # 1. Shapiro-Wilk test (most powerful for small samples)
        if len(clean_residuals) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(clean_residuals)
                results['shapiro_statistic'] = shapiro_stat
                results['shapiro_p_value'] = shapiro_p
                results['shapiro_normal'] = shapiro_p > 0.05
            except Exception as e:
                print(f"    [WARNING] Shapiro-Wilk failed: {e}")
                results['shapiro_statistic'] = np.nan
                results['shapiro_p_value'] = np.nan
                results['shapiro_normal'] = False
        else:
            # Shapiro-Wilk not reliable for very large samples
            results['shapiro_statistic'] = np.nan
            results['shapiro_p_value'] = np.nan
            results['shapiro_normal'] = None
        
        # 2. Jarque-Bera test (tests skewness and kurtosis)
        try:
            jb_stat, jb_p = jarque_bera(clean_residuals)
            results['jarque_bera_statistic'] = jb_stat
            results['jarque_bera_p_value'] = jb_p
            results['jarque_bera_normal'] = jb_p > 0.05
        except Exception as e:
            print(f"    [WARNING] Jarque-Bera failed: {e}")
            results['jarque_bera_statistic'] = np.nan
            results['jarque_bera_p_value'] = np.nan
            results['jarque_bera_normal'] = False
        
        # 3. Anderson-Darling test
        try:
            ad_result = anderson(clean_residuals, dist='norm')
            # Check against 5% significance level (index 2)
            results['anderson_statistic'] = ad_result.statistic
            results['anderson_critical_5pct'] = ad_result.critical_values[2]
            results['anderson_normal'] = ad_result.statistic < ad_result.critical_values[2]
        except Exception as e:
            print(f"    [WARNING] Anderson-Darling failed: {e}")
            results['anderson_statistic'] = np.nan
            results['anderson_critical_5pct'] = np.nan
            results['anderson_normal'] = False
        
        # 4. Kolmogorov-Smirnov test
        try:
            # Standardize first
            std_residuals = (clean_residuals - np.mean(clean_residuals)) / np.std(clean_residuals)
            ks_stat, ks_p = kstest(std_residuals, 'norm')
            results['ks_statistic'] = ks_stat
            results['ks_p_value'] = ks_p
            results['ks_normal'] = ks_p > 0.05
        except Exception as e:
            print(f"    [WARNING] K-S test failed: {e}")
            results['ks_statistic'] = np.nan
            results['ks_p_value'] = np.nan
            results['ks_normal'] = False
        
        # 5. D'Agostino-Pearson test (combines skew and kurtosis)
        try:
            dp_stat, dp_p = normaltest(clean_residuals)
            results['dagostino_statistic'] = dp_stat
            results['dagostino_p_value'] = dp_p
            results['dagostino_normal'] = dp_p > 0.05
        except Exception as e:
            print(f"    [WARNING] D'Agostino-Pearson failed: {e}")
            results['dagostino_statistic'] = np.nan
            results['dagostino_p_value'] = np.nan
            results['dagostino_normal'] = False
        
        # 6. Descriptive statistics
        results['mean'] = np.mean(clean_residuals)
        results['std'] = np.std(clean_residuals)
        results['skewness'] = stats.skew(clean_residuals)
        results['kurtosis'] = stats.kurtosis(clean_residuals)
        results['min'] = np.min(clean_residuals)
        results['max'] = np.max(clean_residuals)
        results['n_obs'] = len(clean_residuals)
        
        return results
    
    def autocorrelation_tests(self, residuals, variable_name, max_lags=24):
        """
        Test for autocorrelation in residuals (should be white noise).
        
        Args:
            residuals (np.ndarray): Residuals to test
            variable_name (str): Name of the variable
            max_lags (int): Maximum number of lags to test
            
        Returns:
            dict: Test results
        """
        results = {'variable': variable_name}
        
        # Remove NaN values
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < max_lags + 1:
            print(f"  [WARNING] Insufficient data for autocorrelation tests")
            return None
        
        # Ljung-Box test (tests for autocorrelation at multiple lags)
        try:
            lb_result = acorr_ljungbox(clean_residuals, lags=max_lags, return_df=True)
            
            # Check if any lag shows significant autocorrelation
            significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
            
            results['ljung_box_significant_lags'] = significant_lags
            results['ljung_box_min_p_value'] = lb_result['lb_pvalue'].min()
            results['ljung_box_white_noise'] = significant_lags == 0
            
            # Store first few lag p-values
            for lag in range(1, min(6, max_lags + 1)):
                results[f'lb_lag{lag}_p'] = lb_result['lb_pvalue'].iloc[lag - 1]
        
        except Exception as e:
            print(f"    [WARNING] Ljung-Box test failed: {e}")
            results['ljung_box_significant_lags'] = np.nan
            results['ljung_box_min_p_value'] = np.nan
            results['ljung_box_white_noise'] = False
        
        return results
    
    def plot_comprehensive_diagnostics(self, residuals, variable_name, 
                                      time_index=None, fitted_values=None):
        """
        Create comprehensive 6-panel diagnostic plot.
        
        Args:
            residuals (np.ndarray): Residuals to analyze
            variable_name (str): Name of the variable
            time_index (pd.DatetimeIndex): Time index for residuals
            fitted_values (np.ndarray): Fitted values (for heteroscedasticity plot)
        """
        # Clean residuals
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 10:
            print(f"  [SKIP] Insufficient data for {variable_name}")
            return
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Histogram with normal overlay (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Histogram
        n, bins, patches = ax1.hist(clean_residuals, bins=50, density=True, 
                                    alpha=0.7, color='steelblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = np.mean(clean_residuals), np.std(clean_residuals)
        x = np.linspace(clean_residuals.min(), clean_residuals.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, 
                label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax1.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean')
        ax1.axvline(mu - 2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.axvline(mu + 2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label='±2σ')
        
        ax1.set_xlabel('Residual Value', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('A) Distribution of Residuals', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        
        stats.probplot(clean_residuals, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor('steelblue')
        ax2.get_lines()[0].set_markeredgecolor('darkblue')
        ax2.get_lines()[0].set_markersize(4)
        ax2.get_lines()[1].set_color('red')
        ax2.get_lines()[1].set_linewidth(2)
        
        ax2.set_title('B) Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax2.set_ylabel('Sample Quantiles', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        
        bp = ax3.boxplot(clean_residuals, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Add horizontal line at zero
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        ax3.set_ylabel('Residual Value', fontsize=11)
        ax3.set_title('C) Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticklabels(['Residuals'])
        
        # Calculate outlier percentage
        q1, q3 = np.percentile(clean_residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((clean_residuals < lower_bound) | (clean_residuals > upper_bound)).sum()
        outlier_pct = outliers / len(clean_residuals) * 100
        ax3.text(0.5, 0.02, f'Outliers: {outliers} ({outlier_pct:.1f}%)', 
                transform=ax3.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. ACF plot (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        try:
            plot_acf(clean_residuals, lags=min(40, len(clean_residuals)//4), 
                    ax=ax4, alpha=0.05)
            ax4.set_title('D) Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Lag', fontsize=11)
            ax4.set_ylabel('ACF', fontsize=11)
        except Exception as e:
            ax4.text(0.5, 0.5, f'ACF plot failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # 5. PACF plot (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        
        try:
            plot_pacf(clean_residuals, lags=min(40, len(clean_residuals)//4), 
                     ax=ax5, alpha=0.05, method='ywm')
            ax5.set_title('E) Partial Autocorrelation (PACF)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Lag', fontsize=11)
            ax5.set_ylabel('PACF', fontsize=11)
        except Exception as e:
            ax5.text(0.5, 0.5, f'PACF plot failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Time series plot of residuals (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        
        if time_index is not None and len(time_index) == len(clean_residuals):
            ax6.plot(time_index, clean_residuals, 'o-', markersize=2, linewidth=0.5, 
                    alpha=0.6, color='steelblue')
            ax6.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            ax6.axhline(y=2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax6.axhline(y=-2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax6.set_xlabel('Time', fontsize=11)
        else:
            ax6.plot(clean_residuals, 'o-', markersize=2, linewidth=0.5, 
                    alpha=0.6, color='steelblue')
            ax6.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            ax6.axhline(y=2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax6.axhline(y=-2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax6.set_xlabel('Observation Index', fontsize=11)
        
        ax6.set_ylabel('Residual Value', fontsize=11)
        ax6.set_title('F) Residuals Over Time', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Residuals vs Fitted (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        
        if fitted_values is not None and len(fitted_values) == len(clean_residuals):
            ax7.scatter(fitted_values, clean_residuals, alpha=0.5, s=20, color='steelblue')
            ax7.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            ax7.axhline(y=2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax7.axhline(y=-2*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # Add lowess smoother to check for patterns
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(clean_residuals, fitted_values, frac=0.3)
                ax7.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, 
                        label='LOWESS smoother')
                ax7.legend(loc='best', fontsize=9)
            except:
                pass
            
            ax7.set_xlabel('Fitted Values', fontsize=11)
        else:
            ax7.scatter(range(len(clean_residuals)), clean_residuals, 
                       alpha=0.5, s=20, color='steelblue')
            ax7.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            ax7.set_xlabel('Observation Index', fontsize=11)
        
        ax7.set_ylabel('Residuals', fontsize=11)
        ax7.set_title('G) Residuals vs Fitted (Heteroscedasticity)', 
                     fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Standardized residuals (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        
        std_residuals = (clean_residuals - mu) / sigma
        ax8.plot(std_residuals, 'o-', markersize=2, linewidth=0.5, 
                alpha=0.6, color='steelblue')
        ax8.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax8.axhline(y=2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2σ')
        ax8.axhline(y=-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax8.axhline(y=3, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='±3σ')
        ax8.axhline(y=-3, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax8.set_xlabel('Observation Index', fontsize=11)
        ax8.set_ylabel('Standardized Residuals', fontsize=11)
        ax8.set_title('H) Standardized Residuals', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary statistics panel (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Perform normality tests
        norm_results = self.normality_tests(clean_residuals, variable_name)
        
        # Create summary text
        summary_text = f"""
RESIDUAL STATISTICS

Sample Size: {norm_results['n_obs']}

Descriptive:
  Mean: {norm_results['mean']:.4f}
  Std Dev: {norm_results['std']:.4f}
  Skewness: {norm_results['skewness']:.4f}
  Kurtosis: {norm_results['kurtosis']:.4f}
  Min: {norm_results['min']:.4f}
  Max: {norm_results['max']:.4f}

Normality Tests:
  Shapiro-Wilk:
    p-value: {norm_results['shapiro_p_value']:.4f}
    Normal: {'✓' if norm_results['shapiro_normal'] else '✗'}
  
  Jarque-Bera:
    p-value: {norm_results['jarque_bera_p_value']:.4f}
    Normal: {'✓' if norm_results['jarque_bera_normal'] else '✗'}
  
  Anderson-Darling:
    Stat: {norm_results['anderson_statistic']:.4f}
    Crit (5%): {norm_results['anderson_critical_5pct']:.4f}
    Normal: {'✓' if norm_results['anderson_normal'] else '✗'}

Interpretation:
  {'Residuals appear normally distributed' if norm_results['jarque_bera_normal'] else 'Residuals deviate from normality'}
        """
        
        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Overall title
        fig.suptitle(f'Comprehensive Residual Diagnostics: {variable_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save figure
        safe_name = variable_name.replace('/', '_').replace(' ', '_')
        output_file = os.path.join(self.output_path, 
                                  f'diagnostics_{safe_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved diagnostics: {output_file}")
        plt.close()
    
    def create_normality_summary_plot(self):
        """
        Create summary visualization of normality tests across all variables.
        """
        if not self.normality_results:
            print("  [WARNING] No normality results to plot")
            return
        
        df = pd.DataFrame(self.normality_results)
        
        # Sort by Jarque-Bera p-value
        df = df.sort_values('jarque_bera_p_value', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Normality test p-values
        ax1 = axes[0, 0]
        x_pos = np.arange(len(df))
        
        ax1.barh(x_pos, df['shapiro_p_value'], alpha=0.7, label='Shapiro-Wilk')
        ax1.barh(x_pos, df['jarque_bera_p_value'], alpha=0.7, label='Jarque-Bera')
        ax1.barh(x_pos, df['ks_p_value'], alpha=0.7, label='K-S')
        
        ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                   label='α = 0.05 threshold')
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(df['variable'], fontsize=8)
        ax1.set_xlabel('p-value', fontsize=11)
        ax1.set_title('Normality Test p-values by Variable', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, 1)
        
        # 2. Skewness and Kurtosis
        ax2 = axes[0, 1]
        
        colors = ['green' if (abs(s) < 0.5 and abs(k) < 3) else 'orange' 
                 for s, k in zip(df['skewness'], df['kurtosis'])]
        
        ax2.scatter(df['skewness'], df['kurtosis'], c=colors, s=100, alpha=0.7, 
                   edgecolors='black')
        
        # Add reference lines for normal distribution
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero skew')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero excess kurtosis')
        
        # Add labels for each point
        for idx, row in df.iterrows():
            ax2.annotate(row['variable'], (row['skewness'], row['kurtosis']), 
                        fontsize=7, alpha=0.7)
        
        ax2.set_xlabel('Skewness', fontsize=11)
        ax2.set_ylabel('Excess Kurtosis', fontsize=11)
        ax2.set_title('Skewness vs Kurtosis', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Proportion passing normality tests
        ax3 = axes[1, 0]
        
        test_names = ['Shapiro-Wilk', 'Jarque-Bera', 'K-S', 'Anderson-Darling', "D'Agostino"]
        pass_rates = [
            df['shapiro_normal'].sum() / len(df) * 100,
            df['jarque_bera_normal'].sum() / len(df) * 100,
            df['ks_normal'].sum() / len(df) * 100,
            df['anderson_normal'].sum() / len(df) * 100,
            df['dagostino_normal'].sum() / len(df) * 100
        ]
        
        bars = ax3.bar(test_names, pass_rates, alpha=0.7, color='steelblue', 
                      edgecolor='black')
        
        # Color bars by pass rate
        for bar, rate in zip(bars, pass_rates):
            if rate >= 80:
                bar.set_color('green')
            elif rate >= 50:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_ylabel('% Variables Passing (p > 0.05)', fontsize=11)
        ax3.set_title('Normality Test Pass Rates', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Distribution of test statistics
        ax4 = axes[1, 1]
        
        # Create violin plot of p-values
        p_value_data = [
            df['shapiro_p_value'].dropna().values,
            df['jarque_bera_p_value'].dropna().values,
            df['ks_p_value'].dropna().values,
            df['dagostino_p_value'].dropna().values
        ]
        
        parts = ax4.violinplot(p_value_data, positions=[1, 2, 3, 4], 
                              showmeans=True, showmedians=True)
        
        ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                   label='α = 0.05', alpha=0.7)
        ax4.set_xticks([1, 2, 3, 4])
        ax4.set_xticklabels(['Shapiro-Wilk', 'Jarque-Bera', 'K-S', "D'Agostino"])
        ax4.set_ylabel('p-value', fontsize=11)
        ax4.set_title('Distribution of Test p-values', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'normality_summary_all_variables.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved normality summary: {output_file}")
        plt.close()
    
    def generate_diagnostic_report(self):
        """
        Generate comprehensive text report of diagnostics.
        """
        if not self.normality_results:
            print("  [WARNING] No results to report")
            return
        
        df_norm = pd.DataFrame(self.normality_results)
        
        # Calculate summary statistics
        n_variables = len(df_norm)
        n_normal_jb = df_norm['jarque_bera_normal'].sum()
        n_normal_sw = df_norm['shapiro_normal'].sum()
        n_normal_ad = df_norm['anderson_normal'].sum()
        
        avg_skew = df_norm['skewness'].abs().mean()
        avg_kurt = df_norm['kurtosis'].abs().mean()
        
        # Identify problematic variables
        problematic = df_norm[
            (~df_norm['jarque_bera_normal']) | 
            (df_norm['skewness'].abs() > 1) | 
            (df_norm['kurtosis'].abs() > 3)
        ]['variable'].tolist()
        
        report = f"""
{'='*80}
COMPREHENSIVE ERROR DIAGNOSTICS REPORT
{'='*80}

SUMMARY STATISTICS
{'='*80}

Total Variables Analyzed: {n_variables}

NORMALITY TEST RESULTS:
  Shapiro-Wilk Test:
    Variables passing (p > 0.05): {n_normal_sw} / {n_variables} ({n_normal_sw/n_variables*100:.1f}%)
  
  Jarque-Bera Test:
    Variables passing (p > 0.05): {n_normal_jb} / {n_variables} ({n_normal_jb/n_variables*100:.1f}%)
  
  Anderson-Darling Test:
    Variables passing: {n_normal_ad} / {n_variables} ({n_normal_ad/n_variables*100:.1f}%)

DISTRIBUTION CHARACTERISTICS:
  Average |Skewness|: {avg_skew:.3f}
  Average |Excess Kurtosis|: {avg_kurt:.3f}

INTERPRETATION:
  • Skewness < 0.5: Distribution is approximately symmetric
  • Excess Kurtosis < 3: Distribution has tails similar to normal
  • p-value > 0.05: Fail to reject normality hypothesis

VARIABLES WITH NON-NORMAL RESIDUALS:
{chr(10).join([f'  • {var}' for var in problematic[:10]])}
{f'  ... and {len(problematic) - 10} more' if len(problematic) > 10 else ''}

{'='*80}
DETAILED RESULTS BY VARIABLE
{'='*80}

"""
        
        # Add detailed results for each variable
        for idx, row in df_norm.iterrows():
            status = '✓ NORMAL' if row['jarque_bera_normal'] else '✗ NON-NORMAL'
            
            report += f"""
{row['variable']}
{'-'*len(row['variable'])}
Status: {status}
Observations: {int(row['n_obs'])}

Descriptive Statistics:
  Mean: {row['mean']:.4f}
  Std Dev: {row['std']:.4f}
  Skewness: {row['skewness']:.4f} {'(symmetric)' if abs(row['skewness']) < 0.5 else '(skewed)'}
  Excess Kurtosis: {row['kurtosis']:.4f} {'(mesokurtic)' if abs(row['kurtosis']) < 3 else '(leptokurtic)' if row['kurtosis'] > 3 else '(platykurtic)'}

Normality Tests:
  Shapiro-Wilk: p = {row['shapiro_p_value']:.4f} {'✓' if row['shapiro_normal'] else '✗'}
  Jarque-Bera: p = {row['jarque_bera_p_value']:.4f} {'✓' if row['jarque_bera_normal'] else '✗'}
  Kolmogorov-Smirnov: p = {row['ks_p_value']:.4f} {'✓' if row['ks_normal'] else '✗'}
  Anderson-Darling: {'✓ Pass' if row['anderson_normal'] else '✗ Fail'}

"""
        
        report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}

1. NORMAL RESIDUALS ({n_normal_jb}/{n_variables} variables):
   • Model assumptions are satisfied
   • Confidence intervals are reliable
   • No transformation needed

2. NON-NORMAL RESIDUALS ({n_variables - n_normal_jb}/{n_variables} variables):
   Possible remedies:
   • Check for outliers and consider robust regression
   • Try Box-Cox or log transformation of response variable
   • Consider quantile regression for heavy-tailed distributions
   • Use bootstrap methods for confidence intervals
   • Investigate whether non-normality stems from model misspecification

3. HETEROSCEDASTICITY:
   • Check residuals vs fitted plots (Panel G in diagnostics)
   • If present, consider weighted least squares or GARCH models

4. AUTOCORRELATION:
   • Check ACF/PACF plots (Panels D and E in diagnostics)
   • If present, residual structure not fully captured by model
   • Consider additional AR/MA terms or different model structure

{'='*80}
GENERATED: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
        """
        
        # Save report
        report_file = os.path.join(self.output_path, 'error_diagnostics_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\n[OK] Saved report: {report_file}")
        
        # Also save detailed results as CSV
        csv_file = os.path.join(self.output_path, 'normality_test_results.csv')
        df_norm.to_csv(csv_file, index=False)
        print(f"[OK] Saved detailed results: {csv_file}")


def main():
    """
    Main function - demonstrates usage with example data.
    For actual use, integrate with your ETS+ENSO forecasting pipeline.
    """
    print("\n" + "="*80)
    print("ERROR DIAGNOSTICS MODULE - DEMONSTRATION")
    print("="*80)
    
    print("\nThis module provides comprehensive error diagnostics.")
    print("To use with your ETS+ENSO models, add the following to your pipeline:")
    print("""
# After fitting models, in your forecasting script:
from common.diagnostics import ModelErrorDiagnostics

diagnostics = ModelErrorDiagnostics(output_path='./error_diagnostics')

for variable, model_dict in self.models.items():
    # Get fitted values and calculate residuals
    y_true = self.data_wide[variable]
    y_pred = model_dict['ets'].fittedvalues  # or however you access fitted values
    
    residuals = y_true - y_pred
    
    # Run diagnostics
    norm_results = diagnostics.normality_tests(residuals.values, variable)
    diagnostics.normality_results.append(norm_results)
    
    diagnostics.plot_comprehensive_diagnostics(
        residuals.values, 
        variable,
        time_index=y_true.index,
        fitted_values=y_pred.values
    )

# Generate summary
diagnostics.create_normality_summary_plot()
diagnostics.generate_diagnostic_report()
    """)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()