import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

def plot_transition_matrix(matrix: np.ndarray, 
                          region_name: str,
                          states: List[str] = None):
    """
    Visualize transition matrix as heatmap
    """
    if states is None:
        states = ["Sunny", "Rainy", "Cloudy"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if matrix[i, j] < 0.7 else "white")
    
    ax.set_xticks(np.arange(len(states)))
    ax.set_yticks(np.arange(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(f"Transition Matrix - {region_name}")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

def plot_stationary_vs_empirical(comparisons: Dict):
    """
    Compare stationary and empirical distributions
    """
    regions = list(comparisons.keys())
    n_regions = len(regions)
    
    fig, axes = plt.subplots(1, n_regions, figsize=(5*n_regions, 5))
    
    if n_regions == 1:
        axes = [axes]
    
    for idx, region in enumerate(regions):
        comp = comparisons[region]
        x = np.arange(3)  # 3 states
        
        width = 0.35
        axes[idx].bar(x - width/2, comp['stationary'], 
                     width, label='Stationary', alpha=0.8)
        axes[idx].bar(x + width/2, comp['empirical'], 
                     width, label='Empirical', alpha=0.8)
        
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['Sunny', 'Rainy', 'Cloudy'])
        axes[idx].set_ylabel('Probability')
        axes[idx].set_title(f'{region}\nMAE: {comp["mean_absolute_error"]:.4f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Stationary vs Empirical Distributions', fontsize=14)
    plt.tight_layout()
    return fig

def plot_rain_probability_forecast(forecasts: Dict):
    """
    Plot probability of rain over time for different regions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = sorted(list(forecasts[list(forecasts.keys())[0]].keys()))
    
    for region_name, region_forecast in forecasts.items():
        probs = [region_forecast[d]['theoretical'] for d in days]
        ax.plot(days, probs, marker='o', label=region_name, linewidth=2)
    
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Probability of Rain')
    ax.set_title('Probability of Rain Over Time - US Regions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% threshold')
    
    plt.tight_layout()
    return fig

def plot_regional_comparison(models_dict: Dict):
    """
    Create comparison plot for all regions
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    regions = list(models_dict.keys())
    avg_rainy = [models_dict[r]['avg_rainy_days'] for r in regions]
    
    ax1.barh(regions, avg_rainy, color='skyblue')
    ax1.set_xlabel('Average Rainy Days (30-day period)')
    ax1.set_title('Rainy Days by Region')
    ax1.grid(True, alpha=0.3, axis='x')
    
    rain_probs = []
    for region in regions:
        stationary = models_dict[region]['stationary']
        rain_idx = 1
        rain_probs.append(stationary[rain_idx])
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    wedges, texts, autotexts = ax2.pie(rain_probs, labels=regions, 
                                       autopct='%1.1f%%', colors=colors)
    ax2.set_title('Stationary Probability of Rain')
    
    ax3.set_title('Sample 30-Day Simulations')
    for i, region in enumerate(regions[:3]):
        seq = models_dict[region]['sequences'][0]
        numeric_seq = []
        for state in seq:
            if state == 'Sunny':
                numeric_seq.append(0)
            elif state == 'Rainy':
                numeric_seq.append(1)
            else:
                numeric_seq.append(0.5)
        
        ax3.plot(numeric_seq, label=region, alpha=0.7)
    
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Weather (0=Sunny, 0.5=Cloudy, 1=Rainy)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Convergence to Stationary Distribution')
    for region in regions[:2]:
        model = models_dict[region]['model']
        distances = []
        days = range(1, 31)
        
        for n in days:
            pi_n = model.initial @ np.linalg.matrix_power(model.P, n)
            stationary = models_dict[region]['stationary']
            distance = np.linalg.norm(pi_n - stationary)
            distances.append(distance)
        
        ax4.plot(days, distances, label=region, marker='.')
    
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Distance from Stationary')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('US Regional Weather Markov Model Analysis', fontsize=16)
    plt.tight_layout()
    return fig