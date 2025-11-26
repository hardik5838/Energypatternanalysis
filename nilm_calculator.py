import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. HELPER: ORGANIC SCHEDULE GENERATOR ---
def get_organic_schedule(hours, start, end, ramp_up_hours=2, ramp_down_hours=3, min_val=0.0):
    """
    Creates a smooth curve (0.0 to 1.0) instead of a square ON/OFF block.
    This simulates people arriving slowly and leaving slowly.
    """
    schedule = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        # 1. CORE WORKING HOURS (Full Power)
        if (start + ramp_up_hours) <= h < (end - ramp_down_hours):
            schedule[i] = 1.0
            
        # 2. RAMP UP (Morning)
        elif start <= h < (start + ramp_up_hours):
            # Linearly increase from min_val to 1.0
            steps = ramp_up_hours
            current_step = h - start
            schedule[i] = min_val + (1.0 - min_val) * (current_step / steps)
            
        # 3. RAMP DOWN (Evening)
        elif (end - ramp_down_hours) <= h < end:
            # Linearly decrease from 1.0 to min_val
            steps = ramp_down_hours
            current_step = h - (end - ramp_down_hours)
            schedule[i] = 1.0 - (1.0 - min_val) * (current_step / steps)
            
        # 4. OFF HOURS (Base)
        else:
            schedule[i] = min_val
            
    return schedule

# --- 2. CORE CALCULATION ENGINE ---
def calculate_organic_model(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # --- A. BASE LOADS (Constant) ---
    load_base = np.full(len(hours), config['base_kw'])
    load_misc = np.full(len(hours), config['misc_kw'])

    # --- B. LIGHTING (Occupancy Curve) ---
    # Lights don't turn on/off instantly. They follow occupancy.
    # We use a curve that stays low at night (security) and peaks mid-day.
    light_curve = get_organic_schedule(hours, config['light_start'], config['light_end'], 
                                     ramp_up_hours=1, ramp_down_hours=2, min_val=config['light_security_pct'])
    load_light = config['light_kw'] * light_curve * config['light_factor']

    # --- C. MEDICAL / PROCESS (Active Schedule) ---
    # Machinery usually ramps up faster but drops off slowly as tasks finish.
    med_curve = get_organic_schedule(hours, config['med_start'], config['med_end'], 
                                   ramp_up_hours=2, ramp_down_hours=3, min_val=0.0)
    load_med = config['med_active_kw'] * med_curve

    # --- D. HVAC: VENTILATION (Fans) ---
    # Fans usually run flat out during the day, but might have a small "tail"
    vent_curve = get_organic_schedule(hours, config['hvac_vent_start'], config['hvac_vent_end'],
                                    ramp_up_hours=1, ramp_down_hours=1, min_val=0.0)
    load_vent = config['hvac_vent_kw'] * vent_curve

    # --- E. HVAC: THERMAL (Compressors) ---
    # Physics: Delta T * Sensitivity
    delta_cool = np.maximum(0, df['temperatura_c'] - config['hvac_set_cool'])
    delta_heat = np.maximum(0, config['hvac_set_heat'] - df['temperatura_c'])
    delta_t = delta_cool + delta_heat
    
    # Capacity is constrained by the ventilation schedule (can't cool if fans are off)
    # But we allow a "linger" effect (thermal inertia)
    thermal_raw = delta_t * config['hvac_thermal_sens']
    load_thermal = thermal_raw * vent_curve 

    # --- TOTAL ---
    df['sim_base'] = load_base
    df['sim_misc'] = load_misc
    df['sim_light'] = load_light
    df['sim_med'] = load_med
    df['sim_vent'] = load_vent
    df['sim_thermal'] = load_thermal
    
    df['sim_total'] = (load_base + load_misc + load_light + 
                       load_med + load_vent + load_thermal)
    
    return df

# --- 3. UI RENDERER ---
def show_nilm_page(df_consumo, df_clima):
    st.header("🏭 Organic Load Disaggregation")
    st.markdown("Uses smooth ramp-up/ramp-down curves to simulate realistic building inertia.")

    if df_consumo.empty or df_clima.empty:
        st.error("Missing Data.")
        return

    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # FILTERS
    with st.expander("Data Settings", expanded=False):
        d_type = st.radio("Day Type", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"], horizontal=True)
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if d_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty: return

    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- INPUTS ---
    with st.sidebar:
        st.header("Parameter Controls")
        
        # 1. BASE & MISC
        with st.expander("1. Base & Misc (24/7)", expanded=True):
            base_kw = st.slider("Base Load (Server/Cryo) [kW]", 0.0, 200.0, 25.0)
            misc_kw = st.slider("Misc. (Small Power) [kW]", 0.0, 100.0, 5.0)

        # 2. HVAC
        with st.expander("2. HVAC (Fans & Compressors)", expanded=True):
            st.info("Ventilation creates the 'block', Thermal adds the 'peaks'.")
            vent_kw = st.slider("Ventilation (Fans) Max [kW]", 0.0, 200.0, 25.0)
            vent_range = st.slider("Ventilation Schedule", 0, 23, (6, 20))
            
            therm_sens = st.slider("Thermal Sensitivity (kW/°C)", 0.0, 20.0, 3.0)
            st.caption("Lower this if the mid-day peak is too high.")
            
            set_cool = st.number_input("Set Cool (°C)", value=24)
            set_heat = st.number_input("Set Heat (°C)", value=20)

        # 3. LIGHTING
        with st.expander("3. Lighting", expanded=False):
            light_kw = st.slider("Lighting Max [kW]", 0.0, 100.0, 15.0)
            light_range = st.slider("Lighting Schedule", 0, 23, (7, 21))
            light_fac = st.slider("Occupancy Factor", 0.1, 1.0, 0.8)
            light_sec = st.slider("Night Security Level", 0.0, 0.5, 0.1)

        # 4. MEDICAL/PROCESS
        with st.expander("4. Medical / Process", expanded=True):
            med_kw = st.slider("Active Equipment [kW]", 0.0, 100.0, 15.0)
            med_range = st.slider("Active Schedule", 0, 23, (8, 18))

    # --- CALCULATION ---
    config = {
        'base_kw': base_kw, 'misc_kw': misc_kw,
        'light_kw': light_kw, 'light_start': light_range[0], 'light_end': light_range[1], 
        'light_factor': light_fac, 'light_security_pct': light_sec,
        'med_active_kw': med_kw, 'med_start': med_range[0], 'med_end': med_range[1],
        'hvac_vent_kw': vent_kw, 'hvac_vent_start': vent_range[0], 'hvac_vent_end': vent_range[1],
        'hvac_thermal_sens': therm_sens, 'hvac_set_cool': set_cool, 'hvac_set_heat': set_heat
    }

    df_sim = calculate_organic_model(df_avg, config)

    # --- PLOT ---
    st.subheader(f"Load Profile Analysis ({d_type})")
    
    # ERROR METRIC
    real_sum = df_sim['consumo_kwh'].sum()
    sim_sum = df_sim['sim_total'].sum()
    diff = sim_sum - real_sum
    
    c1, c2 = st.columns(2)
    c1.metric("Real Daily Consumption", f"{real_sum:,.0f} kWh")
    c2.metric("Simulated Daily Consumption", f"{sim_sum:,.0f} kWh", f"{diff:,.0f} kWh")

    # GRAPH
    fig = go.Figure()
    x = df_sim['hora']
    
    # Stacked layers
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_base'], stackgroup='one', name='Base', line=dict(width=0, color='#95a5a6')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_misc'], stackgroup='one', name='Misc', line=dict(width=0, color='#bdc3c7')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_vent'], stackgroup='one', name='HVAC Fans', line=dict(width=0, color='#3498db')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_thermal'], stackgroup='one', name='HVAC Thermal', line=dict(width=0, color='#e74c3c')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_light'], stackgroup='one', name='Lighting', line=dict(width=0, color='#f1c40f')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_med'], stackgroup='one', name='Active Equip', line=dict(width=0, color='#e67e22')))
    
    # Real line
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REALITY', line=dict(color='black', width=3)))

    fig.update_layout(height=500, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Auto-Insight
    if sim_sum < real_sum:
        st.warning("⚠️ The model is under-estimating. Try increasing **Ventilation Max** or **Base Load**.")
        
