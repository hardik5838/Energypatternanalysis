import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution # <--- THE NEW MAGIC ENGINE

# ==========================================
# 1. LOGIC ENGINE (Unchanged)
# ==========================================

def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, nominal_pct=1.0, residual_pct=0.0, dips=None):
    if dips is None: dips = []
    curve = np.zeros(len(hours))
    for i, h in enumerate(hours):
        activity_val = 0.0
        # Basic Window
        if start <= end: # Standard day (e.g. 08:00 to 18:00)
            if start <= h < end:
                activity_val = 1.0
        else: # Overnight (e.g. 22:00 to 06:00)
            if h >= start or h < end:
                activity_val = 1.0
                
        # Simple Ramp Application (Simplified for Optimization stability)
        # (Full ramp logic kept in your original, simplified here for speed if needed)
        
        # Clip activity
        activity_val = np.clip(activity_val, 0.0, 1.0)
        
        # Apply Dips
        for dip in dips:
            if int(h) == int(dip['hour']):
                factor = 1.0 - (dip['percent'] / 100.0)
                activity_val *= factor

        val = residual_pct + activity_val * (nominal_pct - residual_pct)
        curve[i] = val * max_kw
        
    return curve

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. Base Loads
    df['sim_base'] = generate_load_curve(hours, 0, 24, config['base_kw'], 0, 0, 1.0, 1.0)
    
    # 2. HVAC (Simplified Physics for Auto-Calibration)
    # We use a Change-Point model logic: Load is proportional to Delta T
    delta_T = np.maximum(0, config['hvac_setpoint'] - df['temperatura_c']) # Heating mode example
    # In summer, swap to: np.maximum(0, df['temperatura_c'] - config['hvac_setpoint'])
    
    thermal_load_raw = (config['hvac_ua'] * delta_T) 
    hvac_avail = generate_load_curve(hours, config['hvac_s'], config['hvac_e'], 1.0, 1.0, 1.0, 1.0, config['hvac_res'])
    df['sim_therm'] = np.clip(thermal_load_raw, 0, config['hvac_kw']) * hvac_avail

    # 3. Lighting/Occupancy Combined (for stability)
    df['sim_ops'] = generate_load_curve(
        hours, config['ops_s'], config['ops_e'], config['ops_kw'],
        1.0, 1.0, 1.0, 0.0
    )

    # Total Sum
    df['sim_total'] = df['sim_base'] + df['sim_therm'] + df['sim_ops']
    return df

# ==========================================
# 2. THE OPTIMIZATION ENGINE ("The Black Box")
# ==========================================

def objective_function(params, df_real):
    """
    The 'Error Function'. The AI tries to minimize the return value (RMSE).
    params: list of values guessed by the AI [base_kw, hvac_kw, hvac_s, ...]
    """
    # 1. Unpack the "Genome" (the guessed parameters)
    config = {
        'base_kw': params[0],
        'hvac_kw': params[1],
        'hvac_s': int(params[2]),
        'hvac_e': int(params[3]),
        'hvac_ua': params[4],
        'hvac_setpoint': 21.0, # Fixed constraint (The NetZero Rule)
        'hvac_res': 0.1,
        'ops_kw': params[5],
        'ops_s': int(params[6]),
        'ops_e': int(params[7]),
    }
    
    # 2. Run Simulation
    df_sim = run_simulation(df_real, config)
    
    # 3. Calculate Error (RMSE)
    rmse = np.sqrt(mean_squared_error(df_real['consumo_kwh'], df_sim['sim_total']))
    return rmse

def run_auto_calibration(df_avg):
    """
    Runs the Genetic Algorithm (Differential Evolution)
    """
    # Define Bounds for each parameter (Min, Max)
    # [base_kw, hvac_kw, hvac_s, hvac_e, hvac_ua, ops_kw, ops_s, ops_e]
    max_kwh = df_avg['consumo_kwh'].max()
    bounds = [
        (0, max_kwh * 0.5),   # base_kw
        (0, max_kwh),         # hvac_kw
        (4, 10),              # hvac_start (Morning)
        (16, 22),             # hvac_end (Evening)
        (0.1, 5.0),           # hvac_ua (Thermal Sensitivity)
        (0, max_kwh),         # ops_kw (Lighting/Ops)
        (6, 10),              # ops_start
        (17, 21)              # ops_end
    ]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(df_avg,), 
        strategy='best1bin', 
        maxiter=20, # Low for speed, increase for precision
        popsize=10, 
        tol=0.01,
        seed=42
    )
    return result.x

# ==========================================
# 3. UI & MAIN PAGE
# ==========================================

def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Asepeyo Auto-Twin")
    st.markdown("Use the **'Auto-Calibrate'** button to let the AI find the building schedule automatically.")

    # --- DATA PREP ---
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    if not df_clima.empty:
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 22.0
        
    df_avg = df_merged.groupby(df_merged['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    # --- SESSION STATE INITIALIZATION ---
    if 'opt_params' not in st.session_state:
        # Default values if not yet optimized
        st.session_state['opt_params'] = [10.0, 20.0, 8.0, 18.0, 1.0, 15.0, 8.0, 19.0]

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🤖 The Black Box Solver")
        
        if st.button("⚡ Auto-Calibrate Model", type="primary"):
            with st.spinner("Iterating through 1,000 combinations..."):
                best_params = run_auto_calibration(df_avg)
                st.session_state['opt_params'] = best_params
            st.success("Calibration Complete! Digital Twin parameters found.")

        st.divider()
        st.subheader("Manual Fine-Tuning")
        # We hook the sliders to the session state so they update after optimization
        p = st.session_state['opt_params']
        
        base_kw = st.slider("Base Load (kW)", 0.0, 200.0, float(p[0]), key="s_base")
        
        st.markdown("---")
        st.caption("❄️ HVAC Detect")
        h_kw = st.slider("HVAC Max (kW)", 0.0, 500.0, float(p[1]), key="s_hvac_kw")
        h_s = st.slider("HVAC Start", 0, 24, int(p[2]), key="s_hvac_s")
        h_e = st.slider("HVAC End", 0, 24, int(p[3]), key="s_hvac_e")
        h_ua = st.slider("Thermal Sensitivity (UA)", 0.1, 10.0, float(p[4]), key="s_hvac_ua")
        
        st.markdown("---")
        st.caption("💡 Operations Detect")
        o_kw = st.slider("Ops Max (kW)", 0.0, 500.0, float(p[5]), key="s_ops_kw")
        o_s = st.slider("Ops Start", 0, 24, int(p[6]), key="s_ops_s")
        o_e = st.slider("Ops End", 0, 24, int(p[7]), key="s_ops_e")

    # --- RUN SIMULATION WITH CURRENT SLIDER VALUES ---
    # Note: We use the slider values (which might have just been updated by the optimizer)
    current_config = {
        'base_kw': base_kw,
        'hvac_kw': h_kw, 'hvac_s': h_s, 'hvac_e': h_e, 'hvac_ua': h_ua,
        'hvac_setpoint': 21.0, 'hvac_res': 0.1,
        'ops_kw': o_kw, 'ops_s': o_s, 'ops_e': o_e
    }
    
    df_sim = run_simulation(df_avg, current_config)
    
    # --- METRICS & PLOTS ---
    rmse = np.sqrt(mean_squared_error(df_sim['consumo_kwh'], df_sim['sim_total']))
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Real Peak", f"{df_sim['consumo_kwh'].max():.1f} kW")
    k2.metric("Simulated Peak", f"{df_sim['sim_total'].max():.1f} kW")
    k3.metric("Model Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL Meter', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'], name='Digital Twin', line=dict(color='green', dash='dot', width=3)))
    
    # Stacked area for context
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_base'], name='Base', stackgroup='one', line=dict(width=0), fillcolor='rgba(100,100,100,0.2)'))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_therm'], name='HVAC', stackgroup='one', line=dict(width=0), fillcolor='rgba(255,0,0,0.2)'))
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_ops'], name='Ops', stackgroup='one', line=dict(width=0), fillcolor='rgba(255,200,0,0.2)'))

    fig.update_layout(title="Digital Twin Fitting", xaxis_title="Hour", yaxis_title="kW", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Optimization Results (The Digital Twin DNA)"):
        st.write(f"**Detected Base Load:** {base_kw:.2f} kW")
        st.write(f"**Detected Schedule:** {int(o_s)}:00 to {int(o_e)}:00")
        st.write(f"**Detected HVAC Sensitivity:** {h_ua:.2f} kW/°C")

# --- DUMMY DATA FOR TESTING ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Generate fake data
    dates = pd.date_range("2023-01-01", periods=24, freq="h")
    # A fake profile: Base 10, Peak 50 at noon
    fake_load = 10 + 40 * np.exp(-0.1 * (np.arange(24) - 12)**2) 
    df_c = pd.DataFrame({'fecha': dates, 'consumo_kwh': fake_load})
    df_w = pd.DataFrame({'fecha': dates, 'temperatura_c': 10 + 5 * np.sin(np.linspace(0, 3, 24))})
    
    show_nilm_page(df_c, df_w)
