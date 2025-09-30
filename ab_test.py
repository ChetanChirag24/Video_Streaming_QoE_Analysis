"""
A/B Testing Framework for Adaptive Bitrate Algorithms
Statistical testing and analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTest:
    """A/B Testing framework for streaming quality experiments"""
    
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
        self.results = {}
        
    def calculate_sample_size(self, baseline_rate, mde, alpha=None, power=None):
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Control group conversion rate
            mde: Minimum detectable effect (e.g., 0.05 for 5% improvement)
            alpha: Significance level
            power: Statistical power
        """
        alpha = alpha or self.alpha
        power = power or self.power
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        p_avg = (p1 + p2) / 2
        
        n = 2 * ((z_alpha + z_beta)**2 * p_avg * (1 - p_avg)) / ((p2 - p1)**2)
        
        logger.info(f"Required sample size per group: {int(np.ceil(n))}")
        return int(np.ceil(n))
    
    def generate_test_data(self, n_control=10000, n_treatment=10000, effect_size=0.15):
        """Generate synthetic A/B test data"""
        logger.info("Generating A/B test data...")
        
        np.random.seed(42)
        
        # Control group (current algorithm)
        control_buffering = np.random.poisson(lam=4.5, size=n_control)
        control_vst = np.random.gamma(2, 0.8, size=n_control)
        control_bitrate = np.random.normal(3000, 500, size=n_control)
        control_completion = np.random.beta(5, 2, size=n_control)
        control_quality = np.random.beta(6, 3, size=n_control)
        
        # Treatment group (new adaptive algorithm) - improved metrics
        treatment_buffering = np.random.poisson(lam=4.5 * (1 - effect_size), size=n_treatment)
        treatment_vst = np.random.gamma(2, 0.8 * (1 - effect_size), size=n_treatment)
        treatment_bitrate = np.random.normal(3000 * (1 + effect_size/2), 500, size=n_treatment)
        treatment_completion = np.random.beta(5, 2, size=n_treatment) * (1 + effect_size/3)
        treatment_quality = np.random.beta(6, 3, size=n_treatment) * (1 + effect_size/2)
        
        # Clip values
        treatment_completion = np.clip(treatment_completion, 0, 1)
        treatment_quality = np.clip(treatment_quality, 0, 1)
        
        # Create DataFrames
        control_df = pd.DataFrame({
            'group': 'control',
            'buffering_count': control_buffering,
            'video_start_time': control_vst,
            'avg_bitrate': control_bitrate,
            'completion_rate': control_completion,
            'quality_score': control_quality,
            'churned': (control_quality < 0.4).astype(int)
        })
        
        treatment_df = pd.DataFrame({
            'group': 'treatment',
            'buffering_count': treatment_buffering,
            'video_start_time': treatment_vst,
            'avg_bitrate': treatment_bitrate,
            'completion_rate': treatment_completion,
            'quality_score': treatment_quality,
            'churned': (treatment_quality < 0.4).astype(int)
        })
        
        df = pd.concat([control_df, treatment_df], ignore_index=True)
        logger.info(f"Generated {len(df)} test observations")
        
        return df
    
    def run_ttest(self, control, treatment, metric_name):
        """Run t-test for continuous metrics"""
        t_stat, p_value = ttest_ind(control, treatment)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(control)**2 + np.std(treatment)**2) / 2)
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        # Confidence interval
        ci = stats.t.interval(
            1 - self.alpha,
            len(control) + len(treatment) - 2,
            loc=np.mean(treatment) - np.mean(control),
            scale=pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
        )
        
        result = {
            'metric': metric_name,
            'control_mean': np.mean(control),
            'treatment_mean': np.mean(treatment),
            'absolute_diff': np.mean(treatment) - np.mean(control),
            'relative_diff_pct': ((np.mean(treatment) - np.mean(control)) / np.mean(control)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
        
        return result
    
    def run_proportion_test(self, control, treatment, metric_name):
        """Run proportion test for binary metrics"""
        control_success = np.sum(control)
        treatment_success = np.sum(treatment)
        control_n = len(control)
        treatment_n = len(treatment)
        
        # Two-proportion z-test
        p_control = control_success / control_n
        p_treatment = treatment_success / treatment_n
        p_pooled = (control_success + treatment_success) / (control_n + treatment_n)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_n + 1/treatment_n))
        z_stat = (p_treatment - p_control) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        result = {
            'metric': metric_name,
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'absolute_diff': p_treatment - p_control,
            'relative_diff_pct': ((p_treatment - p_control) / p_control) * 100,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        return result
    
    def analyze_test(self, df):
        """Run complete A/B test analysis"""
        logger.info("Running A/B test analysis...")
        
        control = df[df['group'] == 'control']
        treatment = df[df['group'] == 'treatment']
        
        results = []
        
        # Test continuous metrics
        continuous_metrics = [
            'buffering_count', 'video_start_time', 
            'avg_bitrate', 'completion_rate', 'quality_score'
        ]
        
        for metric in continuous_metrics:
            result = self.run_ttest(
                control[metric].values,
                treatment[metric].values,
                metric
            )
            results.append(result)
            
        # Test binary metric (churn)
        churn_result = self.run_proportion_test(
            control['churned'].values,
            treatment['churned'].values,
            'churn_rate'
        )
        results.append(churn_result)
        
        self.results = pd.DataFrame(results)
        
        # Apply Bonferroni correction for multiple testing
        self.results['bonferroni_significant'] = (
            self.results['p_value'] < self.alpha / len(results)
        )
        
        return self.results
    
    def plot_results(self, df):
        """Visualize A/B test results"""
        logger.info("Generating visualizations...")
        
        Path('results/ab_testing').mkdir(parents=True, exist_ok=True)
        
        # 1. Distribution comparisons
        metrics = ['buffering_count', 'video_start_time', 'completion_rate', 'quality_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            control_data = df[df['group'] == 'control'][metric]
            treatment_data = df[df['group'] == 'treatment'][metric]
            
            ax.hist(control_data, bins=50, alpha=0.6, label='Control', density=True)
            ax.hist(treatment_data, bins=50, alpha=0.6, label='Treatment', density=True)
            ax.axvline(control_data.mean(), color='blue', linestyle='--', linewidth=2)
            ax.axvline(treatment_data.mean(), color='orange', linestyle='--', linewidth=2)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ab_testing/distributions.png', dpi=300)
        plt.close()
        
        # 2. Summary statistics comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        summary_data = []
        for group in ['control', 'treatment']:
            group_data = df[df['group'] == group]
            summary_data.append({
                'Group': group.title(),
                'Avg Buffering': group_data['buffering_count'].mean(),
                'Avg VST (s)': group_data['video_start_time'].mean(),
                'Completion Rate': group_data['completion_rate'].mean(),
                'Quality Score': group_data['quality_score'].mean(),
                'Churn Rate': group_data['churned'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        x = np.arange(len(summary_df.columns) - 1)
        width = 0.35
        
        control_vals = summary_df[summary_df['Group'] == 'Control'].iloc[0, 1:].values
        treatment_vals = summary_df[summary_df['Group'] == 'Treatment'].iloc[0, 1:].values
        
        ax.bar(x - width/2, control_vals, width, label='Control', alpha=0.8)
        ax.bar(x + width/2, treatment_vals, width, label='Treatment', alpha=0.8)
        
        ax.set_ylabel('Value')
        ax.set_title('A/B Test: Control vs Treatment Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df.columns[1:], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ab_testing/summary_comparison.png', dpi=300)
        plt.close()
        
        # 3. Statistical significance heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = self.results[['metric', 'p_value', 'relative_diff_pct']].copy()
        plot_data['significant'] = plot_data['p_value'] < self.alpha
        plot_data['log_p_value'] = -np.log10(plot_data['p_value'])
        
        colors = ['green' if sig else 'red' for sig in plot_data['significant']]
        
        ax.barh(plot_data['metric'], plot_data['log_p_value'], color=colors, alpha=0.7)
        ax.axvline(-np.log10(self.alpha), color='black', linestyle='--', 
                  label=f'Significance Threshold (α={self.alpha})')
        ax.set_xlabel('-log10(p-value)')
        ax.set_ylabel('Metric')
        ax.set_title('Statistical Significance of A/B Test Results')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ab_testing/significance.png', dpi=300)
        plt.close()
        
        logger.info("Visualizations saved to results/ab_testing/")
    
    def generate_report(self):
        """Generate text report of A/B test results"""
        report = []
        report.append("="*70)
        report.append("A/B TEST RESULTS: ADAPTIVE BITRATE ALGORITHM")
        report.append("="*70)
        report.append("")
        
        for _, row in self.results.iterrows():
            report.append(f"\n{row['metric'].upper()}")
            report.append("-" * 50)
            
            if 'control_mean' in row:
                report.append(f"Control Mean:    {row['control_mean']:.4f}")
                report.append(f"Treatment Mean:  {row['treatment_mean']:.4f}")
                report.append(f"Absolute Diff:   {row['absolute_diff']:.4f}")
                report.append(f"Relative Diff:   {row['relative_diff_pct']:.2f}%")
                report.append(f"P-value:         {row['p_value']:.6f}")
                report.append(f"Cohen's d:       {row['cohens_d']:.4f}")
                report.append(f"95% CI:          [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
            else:
                report.append(f"Control Rate:    {row['control_rate']:.4f}")
                report.append(f"Treatment Rate:  {row['treatment_rate']:.4f}")
                report.append(f"Absolute Diff:   {row['absolute_diff']:.4f}")
                report.append(f"Relative Diff:   {row['relative_diff_pct']:.2f}%")
                report.append(f"P-value:         {row['p_value']:.6f}")
            
            significance = "✓ SIGNIFICANT" if row['significant'] else "✗ NOT SIGNIFICANT"
            report.append(f"Result:          {significance} (α={self.alpha})")
        
        report.append("\n" + "="*70)
        report.append("SUMMARY")
        report.append("="*70)
        
        sig_count = self.results['significant'].sum()
        total_count = len(self.results)
        
        report.append(f"\nSignificant results: {sig_count}/{total_count}")
        report.append(f"\nAfter Bonferroni correction:")
        bonf_sig = self.results['bonferroni_significant'].sum()
        report.append(f"Significant results: {bonf_sig}/{total_count}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        with open('results/ab_testing/test_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("Report saved to results/ab_testing/test_report.txt")
        
        return report_text


def main():
    """Run A/B testing analysis"""
    # Initialize test
    ab_test = ABTest(alpha=0.05, power=0.8)
    
    # Calculate sample size
    ab_test.calculate_sample_size(
        baseline_rate=0.25,  # 25% churn rate
        mde=0.15  # Want to detect 15% improvement
    )
    
    # Generate test data
    df = ab_test.generate_test_data(n_control=10000, n_treatment=10000, effect_size=0.15)
    
    # Save test data
    Path('data/ab_test').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/ab_test/test_data.csv', index=False)
    
    # Analyze
    results = ab_test.analyze_test(df)
    
    # Print results
    print("\nDetailed Results:")
    print(results.to_string())
    
    # Save results
    results.to_csv('results/ab_testing/test_results.csv', index=False)
    
    # Visualize
    ab_test.plot_results(df)
    
    # Generate report
    ab_test.generate_report()


if __name__ == "__main__":
    main()