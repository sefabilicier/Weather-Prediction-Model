# **Weather-Prediction-Model**

A probabilistic weather prediction application using Markov Chains to simulate and analyze weather patterns across different US regions. Built with Python and Streamlit.

![Weather Markov Model](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Markov Chains](https://img.shields.io/badge/Markov_Chains-Probability_Model-8A2BE2?style=for-the-badge)

## **Overview**

<img width="1246" height="1076" alt="Image" src="https://github.com/user-attachments/assets/9e1798e7-81d8-4d9e-8e65-2cc4a75497c5" />
<img width="1386" height="1020" alt="Image" src="https://github.com/user-attachments/assets/a38db275-71a5-4988-bd78-3cbdbf2f4ca0" />

This application implements a Markov Chain model to predict weather patterns across five US regions:
- **Southwest** (AZ, NM, NV, UT, CO)
- **Pacific Northwest** (WA, OR)
- **Northeast** (NY, MA, CT, NJ, PA)
- **Southeast** (FL, GA, AL, SC, NC)
- **Midwest** (IL, IN, OH, MI, WI)

The model simulates transitions between three weather states: ☀️ **Sunny**, 🌧️ **Rainy**, and ☁️ **Cloudy**.

## ** Features**

### **Core Functionality**
- **Markov Chain Simulation**: Simulates weather transitions using probability matrices
- **Regional Analysis**: Custom transition probabilities for each US region
- **Probability Forecasting**: Predicts rain probability for 1, 3, 7, 14, and 30 days ahead
- **Statistical Validation**: Compares theoretical vs empirical distributions
- **Interactive Visualizations**: Heatmaps, probability charts, and comparison plots

### **User Interface**
- **Centered Layout**: Clean, minimalist design with black-and-white theme
- **Inter Font**: Modern typography throughout the application
- **Interactive Controls**: Adjust simulation parameters in real-time
- **Weather Badges**: Visual representation of simulated weather sequences
- **Tab-based Navigation**: Organized into three main sections

## ** Project Structure**
```text 
weather-markov-model/
├── app.py # Main Streamlit application
├── weather_model.py # Markov Chain model implementation
├── simulation.py # Weather simulation functions
├── analysis.py # Statistical analysis functions
├── visualization.py # Plotting and visualization functions
├── us_regions.py # US regional weather data and probabilities
├── requirements.txt # Python dependencies
└── README.md # This file
```


## ** Installation**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Setup**

1. **Clone or download the project files**

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

###Dependencies
```code streamlit``` - Web application framework
```code numpy``` - Numerical computations
```code matplotlib``` - Data visualization
```code seaborn``` - Statistical visualizations
```code scipy``` - Statistical functions

Running the Application
```bash
streamlit run app.py
```


## Model Details
Markov Chain Implementation

- States: Sunny, Rainy, Cloudy
- Transition Matrices: Region-specific probabilities based on historical patterns
- Stationary Distribution: Calculated using power iteration method 
- N-step Transition: Computed using matrix exponentiation

Statistical Analysis
- KL Divergence: Measures difference between theoretical and empirical distributions
- Chi-Square Test: Validates model fit against observed data
- Mean Absolute Error: Average prediction error
- Probability Calculations: Rain probability for future days

Regional Data
Each region has:
- Custom transition probability matrix
- Initial probability distribution
- State abbreviations for reference
- Based on simplified historical weather patterns

## Technical Details
### Algorithm Complexity
- Time Complexity: O(n × m × s²) where n = days, m = simulations, s = states
- Space Complexity: O(m × n) for storing simulation results
- Convergence: Stationary distribution calculated with tolerance 1e-10

### Limitations
- Simplified 3-state weather model
- Stationary climate assumptions
- Regional averages may not reflect local variations
- No seasonal variation in transition probabilities

**Enjoy exploring weather patterns with Markov Chains! ☀️🌧️☁️**