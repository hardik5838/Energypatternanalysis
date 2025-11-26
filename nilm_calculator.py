import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. CORE CALCULATION ENGINE ---
def calculate_granular_model(df_avg, config):
    """
    Calculates load curve based on independent schedules and separated physics.
    """
    df = df_avg.copy()
    hours = df['hora'].values
    
    # Initialize component arrays
    load_base = np.full(len(hours), config['base_kw'])
    load_misc = np.full(len(hours), config['misc_kw']) # Misc assumed constant-ish or user adjusted
    load_light = np.zeros(len(hours))
    load_med_active = np.zeros(len(hours))
    load_hvac_vent = np.zeros(len(hours))
    load_hvac_thermal = np.zeros(len(hours))

    # --- A. LIGHTING (Scheduled + Occupancy) ---
    for i, h in enumerate(hours):
        # Primary Schedule
        if config['light_start'] <= h < config['light_end']:
            # Main operation
            load_light[i] = config['light_kw'] * config['light_factor']
        else:
            # Off-hours security lighting
            load_light[i] = config['light_kw'] * config['light_security_pct']

    # --- B. MEDICAL / PROCESS ACTIVE (Scheduled) ---
    for i, h in enumerate(hours):
        if config['med_start'] <= h < config['med_end']:
            load_med_active[i] = config['med_active_kw']

    # --- C. HVAC: VENTILATION (Time-Based) ---
    # Fans/Pumps run regardless of temperature if the system is "On"
    for i, h in enumerate(hours):
        if config['hvac_vent_start'] <= h < config['hvac_vent_end']:
            load_hvac_vent[i] = config['hvac_vent_kw']

    # --- D. HVAC: THERMAL (Temperature-Based) ---
    # Compressors only run when Delta T demands it
    # AND they are within the operational window
    
    # 1. Calculate Delta T
    # If Temp > Setpoint (Cooling) or Temp < Setpoint (Heating)
    delta_cool = np.maximum(0, df['temperatura_c'] - config['hvac_set_cool'])
    delta_heat = np.maximum(0, config['hvac_set_heat'] - df['temperatura_c'])
    
    # Total Delta T needed
    delta_t = delta_cool + delta_heat
    
    # 2. Apply Capacity and Efficiency (Simplified Physics)
    # Power = (Capacity_kW * Load_Factor) / COP
    # We approximate: Power ~ Sensitivity * DeltaT
    thermal_raw = delta_t * config['hvac_thermal_sens']
    
    # 3. Apply Schedule
    for i, h in enumerate(hours):
        if config['hvac_therm_start'] <= h < config['hvac_therm_end']:
            load_hvac_thermal[i] = thermal_raw[i]
        else:
            # Night Setback (System works harder to maintain night temp? 
            # Or off? User can define "night" by setting start/end)
            # We assume off or minimal setback if outside schedule
            load_hvac_thermal[i] = 0.0

    # --- TOTAL ---
    df['sim_base'] = load_base
    df['sim_misc'] = load_misc
    df['sim_light'] = load_light
    df['sim_med'] = load_med_active
    df['sim_vent'] = load_hvac_vent
    df['sim_thermal'] = load_hvac_thermal
    
    df['sim_total'] = (load_base + load_misc + load_light + 
                       load_med_active + load_hvac_vent + load_hvac_thermal)
    
    return df

# --- 2. UI RENDERER ---
def show_nilm_page(df_consumo, df_clima):
    st.header("🏭 Advanced Load Disaggregation")
    st.markdown("Granular control over equipment schedules and capacities.")

    if df_consumo.empty or df_clima.empty:
        st.error("Missing Data.")
        return

    # --- DATA PREP ---
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    with st.expander("Step 1: Data Filters (Day Type)", expanded=False):
        d_type = st.radio("Analyze:", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"], horizontal=True)
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if d_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty: 
        st.warning("No data found for selection.")
        return

    # Average Profile
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("Building Parameters")
        
        # 1. GENERAL
        st.subheader("1. General & Base")
        area = st.number_input("Building Area (m²)", 100, 50000, 1500)
        base_kw = st.number_input("Base Load 24/7 (Server/Cryo) [kW]", 0.0, 200.0, 20.0, help="Power that NEVER turns off (Servers, MRI Cryo, Fridges).")
        
        # Auto-calc Misc
        default_misc = (area * 3.0) / 1000.0 # 3 W/m2 assumption
        misc_kw = st.number_input("Misc. Items (Computers/Small Power) [kW]", 0.0, 100.0, default_misc, help=f"Auto-estimated for {area}m². Adjust manually.")

        # 2. HVAC (Split into Vent & Thermal)
        st.markdown("---")
        st.subheader("2. HVAC System")
        st.info("Separates Ventilation (Fans) from Thermal (Cooling/Heating).")
        
        # Ventilation
        vent_kw = st.number_input("Ventilation/Fan Power [kW]", 0.0, 200.0, 15.0, help="Power of AHUs/Fans that run constantly during open hours.")
        c1, c2 = st.columns(2)
        vent_s = c1.number_input("Vent Start", 0, 23, 6)
        vent_e = c2.number_input("Vent End", 0, 23, 20)
        
        # Thermal
        therm_sens = st.number_input("Thermal Capacity Factor", 0.0, 50.0, 5.0, help="How much kW increases per degree of temp difference.")
        c3, c4 = st.columns(2)
        therm_s = c3.number_input("Compressor Start", 0, 23, 8, help="Usually starts later than fans.")
        therm_e = c4.number_input("Compressor End", 0, 23, 19)
        
        set_cool = st.slider("Cooling Setpoint (°C)", 20, 30, 24)
        set_heat = st.slider("Heating Setpoint (°C)", 15, 25, 20)

        # 3. LIGHTING
        st.markdown("---")
        st.subheader("3. Lighting")
        light_kw = st.number_input("Max Lighting Power [kW]", 0.0, 200.0, 10.0)
        c5, c6 = st.columns(2)
        light_s = c5.number_input("Light Start", 0, 23, 7)
        light_e = c6.number_input("Light End", 0, 23, 21)
        light_fac = st.slider("Diversity Factor %", 0.1, 1.0, 0.8, help="If only 80% of lights are on during the day.")
        light_sec = st.slider("Off-Hours Level %", 0.0, 0.5, 0.1, help="Security lighting at night.")

        # 4. MEDICAL / PROCESS
        st.markdown("---")
        st.subheader("4. Active Equip. (Medical/Process)")
        med_kw = st.number_input("Active Equipment Power [kW]", 0.0, 200.0, 10.0, help="X-Rays, Kitchen, Laundry active load.")
        c7, c8 = st.columns(2)
        med_s = c7.number_input("Equip Start", 0, 23, 9)
        med_e = c8.number_input("Equip End", 0, 23, 18)

    # --- CALCULATION ---
    config = {
        'base_kw': base_kw,
        'misc_kw': misc_kw,
        # Light
        'light_kw': light_kw, 'light_start': light_s, 'light_end': light_e, 
        'light_factor': light_fac, 'light_security_pct': light_sec,
        # Medical
        'med_active_kw': med_kw, 'med_start': med_s, 'med_end': med_e,
        # HVAC Vent
        'hvac_vent_kw': vent_kw, 'hvac_vent_start': vent_s, 'hvac_vent_end': vent_e,
        # HVAC Thermal
        'hvac_thermal_sens': therm_sens, 'hvac_therm_start': therm_s, 'hvac_therm_end': therm_e,
        'hvac_set_cool': set_cool, 'hvac_set_heat': set_heat
    }

    df_sim = calculate_granular_model(df_avg, config)

    # --- VISUALIZATION ---
    st.subheader(f"Detailed Load Profile ({d_type})")
    
    # KPIs
    real_sum = df_sim['consumo_kwh'].sum()
    sim_sum = df_sim['sim_total'].sum()
    diff = sim_sum - real_sum
    
    col1, col2 = st.columns(2)
    col1.metric("Real Energy (Daily)", f"{real_sum:,.0f} kWh")
    col2.metric("Simulated Energy", f"{sim_sum:,.0f} kWh", f"{diff:,.0f} kWh")

    # MAIN CHART
    fig = go.Figure()
    x = df_sim['hora']
    
    # 1. Base (Gray)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_base'], stackgroup='one', name='Base Load (24/7)', line=dict(width=0, color='#7f8c8d')))
    
    # 2. Misc (Light Gray)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_misc'], stackgroup='one', name='Misc. Items', line=dict(width=0, color='#bdc3c7')))
    
    # 3. Ventilation (Blue - Constant)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_vent'], stackgroup='one', name='HVAC: Ventilation (Fans)', line=dict(width=0, color='#2980b9')))
    
    # 4. Thermal (Red/Blue - Variable)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_thermal'], stackgroup='one', name='HVAC: Thermal (Compressors)', line=dict(width=0, color='#e74c3c')))
    
    # 5. Lighting (Yellow)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_light'], stackgroup='one', name='Lighting', line=dict(width=0, color='#f1c40f')))
    
    # 6. Medical Active (Orange)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_med'], stackgroup='one', name='Active Equip. (Med/Process)', line=dict(width=0, color='#e67e22')))
    
    # Real Line
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL CONSUMPTION', line=dict(color='black', width=3)))

    fig.update_layout(
        title="Load Disaggregation",
        xaxis_title="Hour of Day",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=550,
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLE ---
    st.markdown("### Breakdown Summary")
    summary = df_sim[['sim_base', 'sim_misc', 'sim_vent', 'sim_thermal', 'sim_light', 'sim_med']].sum().reset_index()
    summary.columns = ['Category', 'kWh/Day']
    summary['%'] = (summary['kWh/Day'] / summary['kWh/Day'].sum() * 100).round(1)
    
    st.dataframe(summary, use_container_width=True)
