import numpy as np
from typing import Dict, List, Tuple
import scipy.stats as stats

def compare_distributions(models_dict: Dict) -> Dict:
    """
    Compare stationary vs empirical distributions
    """
    comparisons = {}
    
    for region_name, results in models_dict.items():
        stationary = results["stationary"]
        empirical = results["empirical_dist"]
        
        # Calculate KL Divergence (relative entropy)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        stationary_safe = stationary + epsilon
        empirical_safe = empirical + epsilon
        
        # Normalize
        stationary_safe = stationary_safe / stationary_safe.sum()
        empirical_safe = empirical_safe / empirical_safe.sum()
        
        kl_divergence = np.sum(
            stationary_safe * np.log(stationary_safe / empirical_safe)
        )
        
        # Chi-square test
        # Scale to expected counts (assuming 1000 samples)
        expected = stationary * 1000
        observed = empirical * 1000
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        comparisons[region_name] = {
            "stationary": stationary,
            "empirical": empirical,
            "kl_divergence": kl_divergence,
            "chi2_statistic": chi2,
            "p_value": p_value,
            "mean_absolute_error": np.mean(np.abs(stationary - empirical))
        }
    
    return comparisons

def analyze_state_durations(sequences: List[List[str]], 
                           state: str = "Rainy") -> Dict:
    """
    Analyze duration of weather states (e.g., rainy streaks)
    """
    all_durations = []
    
    for seq in sequences:
        durations = []
        current_duration = 0
        
        for weather in seq:
            if weather == state:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        all_durations.extend(durations)
    
    if not all_durations:
        return {
            "mean": 0,
            "max": 0,
            "histogram": {}
        }
    
    unique, counts = np.unique(all_durations, return_counts=True)
    histogram = dict(zip(unique, counts / len(all_durations)))
    
    return {
        "mean": np.mean(all_durations),
        "std": np.std(all_durations),
        "max": np.max(all_durations),
        "histogram": histogram,
        "geometric_fit": 1 / np.mean(all_durations)  
    }

def regional_comparison_report(models_dict: Dict) -> str:
    """
    Generate a comparison report for all regions
    """
    report = []
    report.append("=" * 60)
    report.append("US REGIONAL WEATHER ANALYSIS REPORT")
    report.append("=" * 60)
    
    for region_name, results in models_dict.items():
        report.append(f"\n{region_name.upper()}:")
        report.append("-" * 40)
        
        model = results["model"]
        stationary = results["stationary"]
        
        report.append("Stationary Distribution:")
        for i, state in enumerate(model.states):
            report.append(f"  {state}: {stationary[i]:.3f}")
        
        report.append(f"\nRainy Day Statistics (30-day period):")
        report.append(f"  Average rainy days: {results['avg_rainy_days']:.1f}")
        report.append(f"  Percentage rainy: {results['rainy_percentage']*100:.1f}%")
        
        prob_7 = model.probability_rain_in_n_days(7)
        report.append(f"  Probability of rain in 7 days: {prob_7:.3f}")
    
    return "\n".join(report)