import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time

# ==========================================
# 1. PHYSIOLOGICAL PARAMETERS & LITERATURE
# ==========================================

def get_ethanol_params(sex, weight_kg, height_cm, age, meal_status, population_type):
    """
    Calculates parameters based on physiological literature.
    """
    # 1. Volume of Distribution (Vd) - Watson Formula (1980)
    # Source: Watson et al. (1980) DOI: 10.1093/ajcn/33.1.27
    # This is the gold standard for estimating Total Body Water (TBW) in alcohol research.
    if sex == "Male":
        tbw = 2.447 - (0.09156 * age) + (0.1074 * height_cm) + (0.3362 * weight_kg)
    else:
        tbw = -2.097 + (0.1069 * height_cm) + (0.2466 * weight_kg)
    
    # Ethanol distributes in Total Body Water; Vd (L) = TBW
    vd = tbw
    
    # 2. Absorption Rate (ka) - Gastric Emptying & Food Effects
    # Source: Gentry (2000) DOI: 10.1016/S0378-4274(00)00211-1
    # Food slows gastric emptying, significantly reducing peak BAC.
    ka_map = {
        "Fasted (Empty Stomach)": 3.0, # Rapid absorption
        "Light Meal": 1.2,             # Moderate delay
        "Heavy Meal": 0.6              # Significant delay
    }
    ka = ka_map.get(meal_status, 1.2)
    
    # 3. Elimination Kinetics (Michaelis-Menten)
    # Source: Jones (2010) DOI: 10.1111/j.1530-0277.2006.00201.x
    # Vmax: Rate of elimination (g/L/h); Km: Concentration at half Vmax (g/L)
    vmax_base = 0.18  # Standard mean elimination rate
    km = 0.1          # Michaelis constant for ADH/ALDH enzymes
    
    # Population Adjustments
    # Source: Lieber (2000) DOI: 10.1016/S0163-7258(02)00222-1
    # Chronic drinkers induce CYP2E1, increasing Vmax by up to 40%.
    pop_factors = {
        "Standard": 1.0,
        "Chronic Drinker (Induced Enzymes)": 1.35,
        "Genetic Sensitivity (Slower ALDH)": 0.75,
        "Elderly (Slower metabolism)": 0.85
    }
    vmax = vmax_base * pop_factors.get(population_type, 1.0)
    
    return {
        'Vd': vd,
        'ka': ka,
        'Vmax': vmax,
        'Km': km,
        'TBW': tbw
    }

def ethanol_ode(y, t, p, metab_factor):
    """
    y[0]: Amount in Gut (g)
    y[1]: Amount in Central Compartment (g)
    """
    Ag, Ac = y
    conc = Ac / p['Vd']
    
    # Absorption from Gut to Plasma
    dAg = -p['ka'] * Ag
    
    # Saturable Elimination (Michaelis-Menten)
    # Rate (g/h) = Vmax * Vd * (C / (Km + C))
    vmax_effective = p['Vmax'] * metab_factor
    elimination = (vmax_effective * conc) / (p['Km'] + conc)
    
    dAc = (p['ka'] * Ag) - (elimination * p['Vd'])
    
    # Physical constraint: concentration cannot go below zero
    if Ac <= 0 and dAc < 0:
        dAc = 0
        
    return [dAg, dAc]

# ==========================================
# 2. SIMULATION MANAGER
# ==========================================

def run_ethanol_simulation(drinks_log, params, start_dt, duration=24):
    events = []
    for d in drinks_log:
        # Pure Ethanol Mass (g) = Volume (ml) * ABV * Density (0.789 g/ml)
        grams = d['ml'] * (d['abv'] / 100.0) * 0.789
        
        delta_h = (d['dt'] - start_dt).total_seconds() / 3600.0
        if 0 <= delta_h <= duration:
            events.append((delta_h, grams))
            
    events.sort(key=lambda x: x[0])
    events.append((duration, 0))
    
    # Variability factors: 0.8 (Slow), 1.0 (Normal), 1.2 (Fast)
    factors = {'Slow': 0.8, 'Normal': 1.0, 'Fast': 1.2}
    results = {}
    common_t = None
    
    for label, f in factors.items():
        y0 = [0.0, 0.0]
        t_hist, y_hist = [], []
        curr_t = 0.0
        
        for t_next, grams in events:
            if t_next > curr_t:
                t_span = np.linspace(curr_t, t_next, int((t_next-curr_t)*60)+2)
                sol = odeint(ethanol_ode, y0, t_span, args=(params, f))
                t_hist.append(t_span[:-1]); y_hist.append(sol[:-1])
                y0 = sol[-1]
                curr_t = t_next
            y0[0] += grams
            
        full_t = np.concatenate(t_hist)
        full_y = np.concatenate(y_hist)
        
        if common_t is None: common_t = full_t
        results[label] = full_y[:, 1] / params['Vd']
        
    df = pd.DataFrame({
        'Datetime': [start_dt + timedelta(hours=t) for t in common_t],
        'BAC_Slow': results['Slow'],
        'BAC_Normal': results['Normal'],
        'BAC_Fast': results['Fast']
    })
    return df

# ==========================================
# 3. INTERFACE (STREAMLIT)
# ==========================================

st.set_page_config(page_title="Ethanol PBPK Analyst", layout="wide")
st.title("🍹 Ethanol PBPK & Driver Safety")

with st.sidebar:
    st.header("1. Biometrics")
    c1, c2 = st.columns(2)
    sex = c1.radio("Sex", ["Male", "Female"])
    age = c2.number_input("Age", 18, 100, 30)
    weight = c1.number_input("Weight (kg)", 40, 150, 75)
    height = c2.number_input("Height (cm)", 140, 220, 175)
    
    st.markdown("---")
    st.header("2. Conditions")
    meal = st.selectbox("Gastric Status", ["Fasted (Empty Stomach)", "Light Meal", "Heavy Meal"])
    pop = st.selectbox("Population Profile", [
        "Standard", 
        "Chronic Drinker (Induced Enzymes)", 
        "Genetic Sensitivity (Slower ALDH)", 
        "Elderly (Slower metabolism)"
    ])
    
    st.markdown("---")
    st.header("3. Intake Log")
    if 'alc_drinks' not in st.session_state:
        st.session_state.alc_drinks = [{'time': time(20,0), 'ml': 500, 'abv': 5.0}]
        
    for i, d in enumerate(st.session_state.alc_drinks):
        col1, col2, col3, col4 = st.columns([1, 0.8, 0.8, 0.4])
        d['time'] = col1.time_input(f"Time", d['time'], key=f"at{i}")
        d['ml'] = col2.number_input(f"ml", 10, 2000, d['ml'], key=f"am{i}")
        d['abv'] = col3.number_input(f"%Vol", 0.0, 95.0, d['abv'], key=f"av{i}")
        if col4.button("X", key=f"ax{i}"):
            st.session_state.alc_drinks.pop(i); st.rerun()
            
    if st.button("+ Add Beverage"):
        st.session_state.alc_drinks.append({'time': time(21,0), 'ml': 150, 'abv': 13.0}); st.rerun()

# --- RUN MODEL ---
min_h = min([d['time'].hour for d in st.session_state.alc_drinks])
start_dt = datetime.combine(datetime.now().date(), time(min_h, 0)) - timedelta(hours=1)

formatted_drinks = []
for d in st.session_state.alc_drinks:
    formatted_drinks.append({
        'dt': datetime.combine(datetime.now().date(), d['time']),
        'ml': d['ml'], 'abv': d['abv']
    })

P = get_ethanol_params(sex, weight, height, age, meal, pop)
df = run_simulation = run_ethanol_simulation(formatted_drinks, P, start_dt, 24)

# --- RESULTS ---
DRIVE_LIMIT = 0.5 # g/L (Italy / Europe standard)

col_metrics, col_viz = st.columns([1, 3])

with col_metrics:
    max_bac = df['BAC_Normal'].max()
    st.metric("Peak BAC (Average)", f"{max_bac:.3f} g/L")
    
    # Driver safety estimation
    # We use the 'Slow' metabolism band for a conservative safety margin
    safe_zone = df[df['BAC_Slow'] < DRIVE_LIMIT]
    
    if max_bac > DRIVE_LIMIT:
        if not safe_zone[safe_zone['Datetime'] > df.loc[df['BAC_Slow'].idxmax(), 'Datetime']].empty:
            safe_time = safe_zone[safe_zone['Datetime'] > df.loc[df['BAC_Slow'].idxmax(), 'Datetime']].iloc[0]['Datetime']
            st.success(f"Safe to Drive: **{safe_time.strftime('%H:%M')}**")
            st.caption("Conservative estimate (Slow metabolism band)")
        else:
            st.error("BAC remains above legal limit for >24h.")
    else:
        st.success("You remain under the legal limit.")
    
    st.markdown("---")
    st.write("**Model Constants:**")
    st.write(f"- TBW (Vd): {P['TBW']:.2f} L")
    st.write(f"- Clearance (Vmax): {P['Vmax']:.3f} g/L/h")

with col_viz:
    fig = go.Figure()
    
    # Uncertainty Range
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['BAC_Slow'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['BAC_Fast'], fill='tonexty', 
                             fillcolor='rgba(255, 165, 0, 0.2)', line=dict(width=0), name="Uncertainty Band"))
    
    # Normal Projection
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['BAC_Normal'], name="BAC Projection", line=dict(color='orange', width=4)))
    
    # Legal Limit Line
    fig.add_shape(type="line", x0=df['Datetime'].iloc[0], x1=df['Datetime'].iloc[-1], y0=DRIVE_LIMIT, y1=DRIVE_LIMIT,
                  line=dict(color="red", width=2, dash="dash"))
    
    fig.update_layout(template="plotly_white", hovermode="x unified", title="Ethanol Concentration Forecast",
                      xaxis_title="Time of Day", yaxis_title="BAC (g/L)")
    
    st.plotly_chart(fig, use_container_width=True)

st.warning("🚨 **WARNING:** This tool is a mathematical simulation. Actual BAC varies by genetic factors, medication, and specific health conditions. Never drive if you feel impaired, regardless of the model.")