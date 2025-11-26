import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. CORE LOGIC: BLOCK vs PEAK ---
def calculate_block_model(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. SETUP ARRAYS
    # Base (24/7)
    load_base = np.full(len(hours), config['base_kw'])
    
    # 2. SCHEDULED BLOCKS (Lighting & Process)
    # These are "Trapezoids" (Ramp up, Flat Top, Ramp down)
    def get_block_load(start, end, max_kw, ramp=1):
        curve = np.zeros(len(hours))
        for i, h in enumerate(hours):
            if start <= h < end:
                # Full Power Window
                curve[i] = 1.0
                # Smooth edges (Ramp)
                if h < start + ramp: # Ramp Up
                    curve[i] = (h - start + 1) / (ramp + 1)
                if h >= end - ramp: # Ramp Down
                    curve[i] = (end - h) / (ramp + 1)
        return curve * max_kw

    load_light = get_block_load(config['light_start'], config['light_end'], config['light_kw']) * config['light_factor']
    
    # Add Security Lighting (night time)
    is_night = (load_light == 0)
    load_light[is_night] = config['light_kw'] * config['light_security_pct']

    load_med = get_block_load(config['med_start'], config['med_end'], config['med_kw'])

    # 3. HVAC: VENTILATION (Constant Block)
    # Fans don't care about temperature, they just run.
    load_vent = get_block_load(config['vent_start'], config['vent_end'], config['vent_kw'])

    # 4. HVAC: THERMAL (The Tricky Part)
    # Option A: Weather Driven (Peaks)
    # Option B: Constant/Capped (Plateau)
    
    if config['hvac_mode'] == "Constant (Block)":
        # Behaves like a machine: Turns on, draws X kW, turns off.
        load_thermal = get_block_load(config['therm_start'], config['therm_end'], config['therm_kw'])
    
    else: # "Weather Driven"
        # Calculate ideal demand based on Delta T
        delta_cool = np.maximum(0, df['temperatura_c'] - config['set_cool'])
        delta_heat = np.maximum(0, config['set_heat'] - df['temperatura_c'])
        delta_t = delta_cool + delta_heat
        
        # Raw Demand = Delta T * Sensitivity
        raw_demand = delta_t * config['therm_sens']
        
        # Apply Schedule
        sched_factor = get_block_load(config['therm_start'], config['therm_end'], 1.0)
        
        # CLAMPING / CAPPING (Crucial for "Plateau" look)
        # The system cannot exceed its Installed Capacity
        actual_load = np.minimum(raw_demand, config['therm_kw']) # <--- This creates the flat top
        
        load_thermal = actual_load * sched_factor

    # --- TOTAL ---
    df['sim_base'] = load_base
    df['sim_light'] = load_light
    df['sim_med'] = load_med
    df['sim_vent'] = load_vent
    df['sim_thermal'] = load_thermal
    
    df['sim_total'] = (load_base + load_light + load_med + load_vent + load_thermal)
    
    return df

# --- 2. UI ---
def show_nilm_page(df_consumo, df_clima):
    st.header("🏭 Capacity-Based Load Model")
    st.markdown("Build your load curve using **Installed Capacities** and **Operation Modes**.")

    if df_consumo.empty or df_clima.empty:
        st.error("No data.")
        return

    # DATA PREP
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    with st.expander("Step 1: Data Filters", expanded=False):
        d_type = st.radio("Select Day Type", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"], horizontal=True)
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if d_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty: return

    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- INPUTS (SIDEBAR) ---
    with st.sidebar:
        st.header("1. Installed Capacities")
        
        # BASE
        st.subheader("Base Load (24/7)")
        base_kw = st.number_input("Base Load [kW]", 0.0, 500.0, 25.0, help="Servers, MRI Cryo, Standby.")

        # HVAC
        st.markdown("---")
        st.subheader("HVAC System")
        
        # Ventilation (Fans)
        vent_kw = st.number_input("Ventilation Fans (Installed kW)", 0.0, 500.0, 30.0)
        c1, c2 = st.columns(2)
        vent_s = c1.number_input("Vent Start", 0, 23, 6)
        vent_e = c2.number_input("Vent End", 0, 23, 20)

        # Thermal (Heating/Cooling)
        st.markdown("**Thermal Generation**")
        therm_kw = st.number_input("Thermal Unit Capacity (Installed kW)", 0.0, 500.0, 40.0, help="Max power of Compressors/Heat Pumps.")
        
        # MODE SELECTION
        hvac_mode = st.radio("Thermal Operation Mode", ["Constant (Block)", "Weather Driven"], index=0, help="'Constant' creates a flat block. 'Weather' creates peaks.")
        
        c3, c4 = st.columns(2)
        therm_s = c3.number_input("Therm Start", 0, 23, 8)
        therm_e = c4.number_input("Therm End", 0, 23, 19)
        
        therm_sens = 5.0 # Default
        set_c, set_h = 24, 20
        
        if hvac_mode == "Weather Driven":
            st.caption("Weather Settings")
            therm_sens = st.slider("Sensitivity (kW/Deg)", 0.1, 20.0, 5.0)
            set_c = st.slider("Set Cool", 18, 30, 24)
            set_h = st.slider("Set Heat", 15, 25, 20)
            st.info("💡 Note: Load will flatten out if it hits the 'Thermal Unit Capacity' limit.")

        # LIGHTING
        st.markdown("---")
        st.subheader("Lighting")
        light_kw = st.number_input("Total Lights Installed [kW]", 0.0, 200.0, 15.0)
        light_fac = st.slider("Usage Factor %", 0.1, 1.0, 0.8)
        light_sec = st.slider("Night Security %", 0.0, 0.5, 0.1)
        c5, c6 = st.columns(2)
        light_s = c5.number_input("Light Start", 0, 23, 7)
        light_e = c6.number_input("Light End", 0, 23, 21)

        # PROCESS / MEDICAL
        st.markdown("---")
        st.subheader("Process / Medical")
        med_kw = st.number_input("Process Equip. Capacity [kW]", 0.0, 200.0, 10.0, help="X-Rays, Kitchens, etc.")
        c7, c8 = st.columns(2)
        med_s = c7.number_input("Process Start", 0, 23, 8)
        med_e = c8.number_input("Process End", 0, 23, 18)

    # --- CALCULATION ---
    config = {
        'base_kw': base_kw,
        'light_kw': light_kw, 'light_start': light_s, 'light_end': light_e, 'light_factor': light_fac, 'light_security_pct': light_sec,
        'med_kw': med_kw, 'med_start': med_s, 'med_end': med_e,
        'vent_kw': vent_kw, 'vent_start': vent_s, 'vent_end': vent_e,
        'therm_kw': therm_kw, 'therm_start': therm_s, 'therm_end': therm_e, 'hvac_mode': hvac_mode,
        'therm_sens': therm_sens, 'set_cool': set_c, 'set_heat': set_h
    }

    df_sim = calculate_block_model(df_avg, config)

    # --- VISUALIZATION ---
    st.subheader(f"Capacity Analysis ({d_type})")
    
    # KPI
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    diff = sim - real
    
    k1, k2 = st.columns(2)
    k1.metric("Real Energy", f"{real:,.0f} kWh")
    k2.metric("Simulated Energy", f"{sim:,.0f} kWh", f"{diff:,.0f} kWh")

    # PLOT
    fig = go.Figure()
    x = df_sim['hora']

    # 1. BASE (Gray)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_base'], stackgroup='one', name='Base Load (24/7)', line=dict(width=0, color='gray')))
    
    # 2. VENTILATION (Blue)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_vent'], stackgroup='one', name='HVAC: Ventilation (Fans)', line=dict(width=0, color='#2980b9')))
    
    # 3. THERMAL (Red)
    therm_color = '#c0392b' if hvac_mode == "Constant (Block)" else '#e74c3c'
    therm_name = f'HVAC: Thermal ({hvac_mode})'
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_thermal'], stackgroup='one', name=therm_name, line=dict(width=0, color=therm_color)))
    
    # 4. LIGHTING (Yellow)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_light'], stackgroup='one', name='Lighting', line=dict(width=0, color='#f1c40f')))
    
    # 5. PROCESS (Orange)
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_med'], stackgroup='one', name='Process / Medical', line=dict(width=0, color='#e67e22')))

    # REAL LINE
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER', line=dict(color='black', width=3)))

    fig.update_layout(height=500, hovermode="x unified", title="Load Profile", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    
    # PIE CHART
    st.subheader("Energy Mix Estimate")
    res = df_sim[['sim_base', 'sim_vent', 'sim_thermal', 'sim_light', 'sim_med']].sum().reset_index()
    res.columns = ['Category', 'kWh']
    st.plotly_chart(go.Figure(data=[go.Pie(labels=res['Category'], values=res['kWh'])]), use_container_width=True)
