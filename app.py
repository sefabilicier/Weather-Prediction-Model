import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from us_regions import US_REGIONS, WEATHER_STATES
from weather_model import MarkovWeatherModel
import simulation
import analysis
import visualization

st.set_page_config(
    page_title="Probabilistic Weather Prediction Model",
    page_icon="☁️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* FULL PAGE WIDTH - REMOVE MAX-WIDTH CONSTRAINT */
    .stApp {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Main container for centered content */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* HIDE STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    
    /* Hide the URL location bar if it appears */
    .stApp > header {
        display: none;
    }
    
    /* Hide any other Streamlit decorative elements */
    div[class*="stAppViewBlockContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Ensure content takes full height */
    .stApp > div {
        padding: 0 !important;
    }
    
    /* Your existing styles below... */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        line-height: 1;
        padding-top: 2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666666;
        text-align: center;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 0.02em;
    }
    
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        margin: 0 auto;
    }
    
    .tab-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
        margin: 2rem 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #000000;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: #000000 !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #ffffff !important;
        color: #000000 !important;
        transform: translateY(-1px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 500;
        color: #666666;
        border: none;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000;
    }
    
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        background: transparent !important;
        border-bottom: 2px solid #000000 !important;
    }
    
    /* Select box styling */
    .stSelectbox {
        margin: 1rem 0;
    }
    
    .stSelectbox [data-baseweb="select"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #000000;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: #e0e0e0;
        margin: 2rem 0;
        width: 100%;
    }
    
    /* Info box */
    .info-box {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .info-box h3 {
        color: #000000;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Weather badges */
    .weather-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .badge-sunny {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .badge-rainy {
        background: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    .badge-cloudy {
        background: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
    }
    
    /* Data table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .data-table th {
        background: #f8f8f8;
        color: #000000;
        font-weight: 600;
        padding: 1rem;
        border-bottom: 2px solid #000000;
        text-align: left;
    }
    
    .data-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #000000;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .subsection-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #333333;
        margin: 1.5rem 0 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666666;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'simulations_run': False,
        'models_dict': None,
        'comparisons': None,
        'forecasts': None,
        'selected_region': 'Southwest',
        'days': 30,
        'simulations': 500
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display centered header"""
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Weather Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Probabilistic weather prediction using Markov Chains</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_controls():
    """Display centered control panel"""
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.session_state.selected_region = st.selectbox(
            "Region",
            list(US_REGIONS.keys()),
            index=list(US_REGIONS.keys()).index(st.session_state.selected_region)
        )
    
    with col2:
        days_options = [7, 14, 30, 60, 90]
        st.session_state.days = st.selectbox(
            "Forecast Days",
            days_options,
            index=days_options.index(st.session_state.days) if st.session_state.days in days_options else 2
        )
    
    with col3:
        sim_options = [100, 250, 500, 1000, 2000]
        st.session_state.simulations = st.selectbox(
            "Simulations",
            sim_options,
            index=sim_options.index(st.session_state.simulations) if st.session_state.simulations in sim_options else 2
        )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running weather simulations..."):
                st.session_state.models_dict = simulation.simulate_multiple_regions(
                    days=st.session_state.days,
                    simulations=st.session_state.simulations
                )
                st.session_state.comparisons = analysis.compare_distributions(
                    st.session_state.models_dict
                )
                st.session_state.forecasts = simulation.forecast_probability_rain(
                    st.session_state.models_dict,
                    days_ahead=[1, 3, 7, 14, 30]
                )
                st.session_state.simulations_run = True
            st.success("Simulation complete!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_weather_badges(sequence):
    """Display weather sequence as badges"""
    html = '<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin: 1rem 0;">'
    for i, weather in enumerate(sequence[:15], 1):  # Show first 15 days
        if weather == "Sunny":
            html += f'<div class="weather-badge badge-sunny">D{i}: ☀️</div>'
        elif weather == "Rainy":
            html += f'<div class="weather-badge badge-rainy">D{i}: 🌧️</div>'
        else:
            html += f'<div class="weather-badge badge-cloudy">D{i}: ☁️</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def display_metrics():
    """Display key metrics for all regions"""
    if not st.session_state.simulations_run:
        return
    
    st.markdown('<div class="section-header">Regional Weather Metrics</div>', unsafe_allow_html=True)
    
    regions = list(st.session_state.models_dict.keys())
    cols = st.columns(len(regions))
    
    for idx, region in enumerate(regions):
        with cols[idx]:
            if region in st.session_state.models_dict:
                data = st.session_state.models_dict[region]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{data['avg_rainy_days']:.1f}</div>
                    <div class="metric-label" style="font-size: 0.7rem; font-weight: bold;">{region}</div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem;">
                        Rainy days in {st.session_state.days}d
                    </div>
                </div>
                """, unsafe_allow_html=True)

def display_main_content():
    """Display main content tabs"""
    if not st.session_state.simulations_run:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>How to Use</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Select** a US region from the dropdown
        2. **Choose** forecast days and simulation count
        3. **Click** Run Simulation to generate predictions
        4. **Explore** results across different tabs
        
        ### Model Details
        - Uses Markov Chains for weather prediction
        - Simulates transitions between Sunny/Rainy/Cloudy states
        - Calculates stationary distributions and probabilities
        - Compares theoretical vs empirical results
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    tab1, tab2, tab3 = st.tabs([
        "📈 Regional Analysis",
        "📊 Probability Forecasts", 
        "📋 Data & Reports"
    ])
    
    with tab1:
        display_regional_analysis()
    
    with tab2:
        display_probability_forecasts()
    
    with tab3:
        display_data_reports()

def display_regional_analysis():
    """Display regional analysis content"""
    st.markdown(f'<div class="section-header">{st.session_state.selected_region} Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.selected_region not in st.session_state.models_dict:
        st.warning("Region data not available")
        return
    
    model_data = st.session_state.models_dict[st.session_state.selected_region]
    model = model_data['model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Stationary Distribution</div>', unsafe_allow_html=True)
        stationary = model.stationary_distribution()
        
        for state, prob in zip(WEATHER_STATES, stationary):
            st.metric(label=state, value=f"{prob:.3f}")
    
    with col2:
        st.markdown('<div class="subsection-header">Rain Statistics</div>', unsafe_allow_html=True)
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric(
                "Avg Rainy Days", 
                f"{model_data['avg_rainy_days']:.1f}",
                f"in {st.session_state.days}d"
            )
        
        with col2b:
            st.metric(
                "Rain Percentage",
                f"{model_data['rainy_percentage']*100:.1f}%"
            )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">Transition Matrix</div>', unsafe_allow_html=True)
    
    fig = visualization.plot_transition_matrix(model.P, st.session_state.selected_region)
    
    ax = fig.get_axes()[0]
    im = ax.images[0]
    im.set_cmap('gray')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown('<div class="subsection-header">Sample Weather Sequence</div>', unsafe_allow_html=True)
    sample_seq = model.simulate_sequence(15)
    display_weather_badges(sample_seq)

def display_probability_forecasts():
    """Display probability forecasts"""
    st.markdown('<div class="section-header">Rain Probability Forecasts</div>', unsafe_allow_html=True)
    
    forecast_data = []
    for region in US_REGIONS.keys():
        if region in st.session_state.forecasts:
            row = {"Region": region}
            for days, probs in st.session_state.forecasts[region].items():
                row[f"{days}d"] = f"{probs['theoretical']:.1%}"
            forecast_data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(forecast_data)
    st.dataframe(
        df.set_index('Region'),
        use_container_width=True,
        column_config={
            "1d": st.column_config.NumberColumn("1 Day", format="%.1f%%"),
            "3d": st.column_config.NumberColumn("3 Days", format="%.1f%%"),
            "7d": st.column_config.NumberColumn("7 Days", format="%.1f%%"),
            "14d": st.column_config.NumberColumn("14 Days", format="%.1f%%"),
            "30d": st.column_config.NumberColumn("30 Days", format="%.1f%%")
        }
    )
    
    st.markdown('<div class="subsection-header">Probability Trends</div>', unsafe_allow_html=True)
    fig = visualization.plot_rain_probability_forecast(st.session_state.forecasts)
    
    ax = fig.get_axes()[0]
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for line in ax.get_lines():
        line.set_color('black')
        line.set_linewidth(2)
    
    ax.grid(True, color='#e0e0e0', linestyle='-', linewidth=0.5)
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown('<div class="subsection-header">Regional Comparison</div>', unsafe_allow_html=True)
    fig2 = visualization.plot_regional_comparison(st.session_state.models_dict)
    
    for ax in fig2.get_axes():
        ax.set_facecolor('white')
        for line in ax.get_lines():
            line.set_color('black')
        for patch in ax.patches:
            patch.set_facecolor('#f0f0f0')
            patch.set_edgecolor('black')
    
    fig2.patch.set_facecolor('white')
    st.pyplot(fig2)
    plt.close(fig2)

def display_data_reports():
    """Display data reports and statistics"""
    st.markdown('<div class="section-header">Statistical Reports</div>', unsafe_allow_html=True)
    
    report = analysis.regional_comparison_report(st.session_state.models_dict)
    
    with st.expander("📄 View Full Analysis Report", expanded=False):
        lines = report.split('\n')
        for line in lines:
            if line.strip():
                if '=====' in line or '-----' in line:
                    st.markdown(f"**{line.replace('=', '').replace('-', '')}**")
                elif ':' in line and line.strip().endswith(':'):
                    st.markdown(f"**{line}**")
                else:
                    if any(keyword in line.lower() for keyword in ['sunny', 'rainy', 'cloudy', 'average', 'percentage', 'probability']):
                        st.markdown(f"`{line}`")
                    else:
                        st.text(line)
    
    st.markdown('<div class="subsection-header">Distribution Validation</div>', unsafe_allow_html=True)
    
    if st.session_state.comparisons:
        fig = visualization.plot_stationary_vs_empirical(st.session_state.comparisons)
        
        for ax in fig.get_axes():
            ax.set_facecolor('white')
            for patch in ax.patches:
                patch.set_color('#666666')
                patch.set_edgecolor('black')
        
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('<div class="subsection-header">Model Statistics by Region</div>', unsafe_allow_html=True)
        
        stats_data = []
        for region, comp in st.session_state.comparisons.items():
            stats_data.append({
                'Region': region,
                'KL Divergence': f"{comp['kl_divergence']:.4f}",
                'Chi-Square': f"{comp['chi2_statistic']:.2f}",
                'Mean Error': f"{comp['mean_absolute_error']:.4f}",
                'p-value': f"{comp['p_value']:.4f}"
            })
        
        import pandas as pd
        df_stats = pd.DataFrame(stats_data)
        
        st.dataframe(
            df_stats.set_index('Region'),
            use_container_width=True,
            column_config={
                "KL Divergence": st.column_config.NumberColumn("KL Divergence", format="%.4f"),
                "Chi-Square": st.column_config.NumberColumn("Chi-Square", format="%.2f"),
                "Mean Error": st.column_config.NumberColumn("Mean Error", format="%.4f"),
                "p-value": st.column_config.NumberColumn("p-value", format="%.4f")
            }
        )
        
        st.markdown("""
        <div style="background-color: #f8f8f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <small>
        <strong>Interpretation:</strong><br>
        • <strong>KL Divergence:</strong> Measures how different empirical distribution is from stationary (0 = identical)<br>
        • <strong>Chi-Square:</strong> Tests if observed frequencies match expected (lower = better fit)<br>
        • <strong>Mean Error:</strong> Average absolute difference between stationary and empirical probabilities<br>
        • <strong>p-value:</strong> Probability of observing this data if model is correct (>0.05 suggests good fit)
        </small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run simulations to see distribution comparisons.")

def display_footer():
    """Display centered footer"""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("""
    Weather Markov Model • Probability Theory Project
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    display_header()
    
    display_controls()
    
    if st.session_state.simulations_run:
        display_metrics()

    display_main_content()
    
    display_footer()

if __name__ == "__main__":
    main()