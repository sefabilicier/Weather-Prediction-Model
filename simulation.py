import numpy as np
from typing import List, Dict, Tuple
from weather_model import MarkovWeatherModel

def simulate_multiple_regions(days: int = 30, 
                             simulations: int = 1000) -> Dict:
    """
    Simulate weather for all US regions
    """
    from us_regions import US_REGIONS, WEATHER_STATES
    
    results = {}
    
    for region_name, region_data in US_REGIONS.items():
        print(f"Simulating {region_name}...")
        
        # Create model
        model = MarkovWeatherModel(
            transition_matrix=region_data["transition_matrix"],
            initial_dist=region_data["initial_dist"],
            region_name=region_name
        )
        
        # Collect statistics
        region_results = {
            "model": model,
            "sequences": [],
            "rainy_counts": [],
            "stationary": model.stationary_distribution(),
            "empirical_dist": np.zeros(len(WEATHER_STATES))
        }
        
        # Run simulations
        rainy_days_total = 0
        state_counts = {state: 0 for state in WEATHER_STATES}
        
        for _ in range(simulations):
            seq = model.simulate_sequence(days)
            region_results["sequences"].append(seq)
            
            # Count rainy days
            rainy_days = seq.count("Rainy")
            region_results["rainy_counts"].append(rainy_days)
            rainy_days_total += rainy_days
            
            # Count all states
            for state in seq:
                state_counts[state] += 1
        
        # Calculate empirical distribution
        total_states = days * simulations
        for i, state in enumerate(WEATHER_STATES):
            region_results["empirical_dist"][i] = state_counts[state] / total_states
        
        # Average rainy days
        region_results["avg_rainy_days"] = rainy_days_total / simulations
        region_results["rainy_percentage"] = rainy_days_total / (days * simulations)
        
        results[region_name] = region_results
    
    return results

def forecast_probability_rain(models_dict: Dict, 
                             days_ahead: List[int] = [1, 3, 7, 14, 30]) -> Dict:
    """
    Calculate probability of rain for various days ahead
    """
    forecasts = {}
    
    for region_name, results in models_dict.items():
        model = results["model"]
        forecasts[region_name] = {}
        
        for n in days_ahead:
            # Theoretical probability
            theoretical = model.probability_rain_in_n_days(n)
            
            # Empirical from simulations
            empirical = results["rainy_percentage"]  # Simplified
            
            forecasts[region_name][n] = {
                "theoretical": theoretical,
                "empirical": empirical
            }
    
    return forecasts