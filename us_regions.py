"""
US Regional Weather Transition Probabilities
Based on historical patterns
Probabilities: [Sunny, Rainy, Cloudy] states
"""

US_REGIONS = {
    "Southwest": {
        "states": ["AZ", "NM", "NV", "UT", "CO"],
        "transition_matrix": [
            [0.7, 0.2, 0.1],  # Sunny -> [Sunny, Rainy, Cloudy]
            [0.6, 0.3, 0.1],  # Rainy -> [Sunny, Rainy, Cloudy]
            [0.5, 0.3, 0.2]   # Cloudy -> [Sunny, Rainy, Cloudy]
        ],
        "initial_dist": [0.8, 0.1, 0.1]  # Initial probabilities
    },
    "Pacific Northwest": {
        "states": ["WA", "OR"],
        "transition_matrix": [
            [0.4, 0.4, 0.2],
            [0.3, 0.5, 0.2],
            [0.3, 0.3, 0.4]
        ],
        "initial_dist": [0.3, 0.4, 0.3]
    },
    "Northeast": {
        "states": ["NY", "MA", "CT", "NJ", "PA"],
        "transition_matrix": [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.4]
        ],
        "initial_dist": [0.4, 0.3, 0.3]
    },
    "Southeast": {
        "states": ["FL", "GA", "AL", "SC", "NC"],
        "transition_matrix": [
            [0.6, 0.3, 0.1],
            [0.5, 0.3, 0.2],
            [0.4, 0.3, 0.3]
        ],
        "initial_dist": [0.6, 0.2, 0.2]
    },
    "Midwest": {
        "states": ["IL", "IN", "OH", "MI", "WI"],
        "transition_matrix": [
            [0.5, 0.25, 0.25],
            [0.4, 0.35, 0.25],
            [0.3, 0.3, 0.4]
        ],
        "initial_dist": [0.4, 0.3, 0.3]
    }
}

WEATHER_STATES = ["Sunny", "Rainy", "Cloudy"]
STATE_INDICES = {state: i for i, state in enumerate(WEATHER_STATES)}