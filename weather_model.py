import numpy as np
from typing import List, Tuple, Dict

class MarkovWeatherModel:
    """Markov Chain model for weather prediction"""
    
    def __init__(self, transition_matrix: np.ndarray, 
                 initial_dist: List[float],
                 region_name: str = "Generic"):
        """
        Initialize weather model
        
        Parameters:
        -----------
        transition_matrix : np.ndarray
            Square matrix where P[i][j] = probability from state i to j
        initial_dist : List[float]
            Initial probability distribution
        region_name : str
            Name of the US region
        """
        self.P = np.array(transition_matrix)
        self.initial = np.array(initial_dist)
        self.region = region_name
        self.states = ["Sunny", "Rainy", "Cloudy"]
        
        self._validate_matrix()
        
    def _validate_matrix(self):
        """Check transition matrix is valid"""
        for i in range(len(self.P)):
            if not np.isclose(self.P[i].sum(), 1.0, atol=1e-10):
                raise ValueError(f"Row {i} doesn't sum to 1: {self.P[i].sum()}")
    
    def n_step_transition(self, n: int) -> np.ndarray:
        """
        Compute n-step transition matrix: P^n
        """
        return np.linalg.matrix_power(self.P, n)
    
    def stationary_distribution(self, max_iter: int = 1000, 
                                tolerance: float = 1e-10) -> np.ndarray:
        """
        Find stationary distribution π such that πP = π
        Using power iteration method
        """
        pi = np.ones(len(self.states)) / len(self.states)
        
        for _ in range(max_iter):
            pi_next = pi @ self.P
            if np.linalg.norm(pi_next - pi) < tolerance:
                return pi_next
            pi = pi_next
        
        return pi
    
    def probability_rain_in_n_days(self, n: int, 
                                   current_state: str = None) -> float:
        """
        Calculate probability of rain in exactly n days
        
        Parameters:
        -----------
        n : int
            Number of days in future
        current_state : str or None
            If None, use initial distribution
        """
        if current_state:
            idx = self.states.index(current_state)
            start_vec = np.zeros(len(self.states))
            start_vec[idx] = 1.0
        else:
            start_vec = self.initial
        
        prob_n = start_vec @ np.linalg.matrix_power(self.P, n)
        
        rain_idx = self.states.index("Rainy")
        return prob_n[rain_idx]
    
    def expected_rainy_days(self, horizon: int, 
                           simulations: int = 1000) -> float:
        """
        Expected number of rainy days in next 'horizon' days
        """
        total_rainy = 0
        
        for _ in range(simulations):
            weather_seq = self.simulate_sequence(horizon)
            total_rainy += weather_seq.count("Rainy")
        
        return total_rainy / simulations
    
    def simulate_sequence(self, days: int, 
                          start_state: str = None) -> List[str]:
        """
        Simulate a weather sequence
        
        Parameters:
        -----------
        days : int
            Length of sequence to simulate
        start_state : str or None
            Starting weather state
        """
        sequence = []
        
        if start_state:
            current_idx = self.states.index(start_state)
        else:
            current_idx = np.random.choice(len(self.states), p=self.initial)
        
        for _ in range(days):
            sequence.append(self.states[current_idx])
            
            current_idx = np.random.choice(
                len(self.states), 
                p=self.P[current_idx]
            )
        
        return sequence