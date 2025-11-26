import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURATION DATABASE (The Logic Engine) ---
# Defines the "Energy Signature" of each equipment type based on your audit files.
EQUIPMENT_DB = {
    "HVAC_VRV": {
        "name": "VRV / VRF System (Daikin/Mitsubishi)",
        "type": "hvac",
        "profile": "seasonal", # Depends on T_ext
        "default_power": 50.0, # kW estimated for a medium center
        "base_load_pct": 0.1,  # 10% usage at night (standby/leakage)
        "description": "Common in clinics. Variable flow."
    },
    "HVAC_CHILLER": {
        "name": "Industrial Chiller (Enfriadora)",
        "type": "hvac",
        "profile": "seasonal",
        "default_power": 120.0,
        "base_load_pct": 0.05,
        "description": "Large buildings/Hospitals only."
    },
    "HVAC_SPLIT": {
        "name": "Split AC Units",
        "type": "hvac",
        "profile": "seasonal",
        "default_power": 10.0, 
        "base_load_pct": 0.0,
        "description": "Small offices/individual rooms."
    },
    "MED_MRI": {
        "name": "MRI Machine (Resonancia)",
        "type": "medical_base", # Critical 24/7 Load
        "profile": "flat_high", 
        "default_power": 45.0, # The "Black Bar" culprit (Cryogenics)
        "active_adder": 10.0, # Extra power when scanning
        "description": "High constant consumption for Cryogenics."
    },
    "MED_XRAY": {
        "name": "X-Ray / CT Scan",
        "type": "medical_active",
        "profile": "working_hours",
        "default_power": 15.0, # Average during working hours
        "description": "High peaks during operational hours."
    },
    "MED_REHAB": {
        "name": "Rehab Equipment (IR, Magneto)",
        "type": "medical_active",
        "profile": "working_hours",
        "default_power": 5.0,
        "description": "Physiotherapy machines."
    },
    "LIGHT_LED": {
        "name": "LED Lighting",
        "type": "lighting",
        "profile": "occupancy",
        "watts_per_m2": 7.0,
        "description": "Efficient modern lighting."
    },
    "LIGHT_FLUO": {
        "name": "Fluorescent Lighting",
        "type": "lighting",
        "profile": "occupancy",
        "watts_per_m2": 16.0,
        "description": "Older tubes (T5/T8)."
    },
    "BASE_SERVERS": {
        "name": "Server Rack / UPS",
        "type": "base",
        "profile": "flat",
        "default_power": 5.0,
        "description": "IT Infrastructure (24/7)."
    }
}

# --- 2. PHYSICS ENGINE ---
def calculate_inventory_model(df_avg, selected_items, params):
    """
    Builds the load curve item by item (Bottom-Up Approach).
    """
    df = df_avg.copy()
    hours = df['hora'].values
    
    # Initialize Accumulators
    df['sim_base'] = 0.0       # 24/7 Loads (Gray)
    df['sim_lighting'] = 0.0   # Lighting (Yellow)
    df['sim_medical'] = 0.0    # Medical Active (Orange)
    df['sim_hvac'] = 0.0       # Climate (Blue/Red)
    
    # A. CALCULATE SCHEDULE / OCCUPANCY
    # 0.0 = Empty, 1.0 = Full
    occupancy = np.zeros(len(hours))
    for i, h in enumerate(hours):
        if params['sched_start'] <= h < params['sched_end']:
            if h == params['sched_start']: occupancy[i] = 0.5
            elif h == params['sched_end'] - 1: occupancy[i] = 0.5
            else: occupancy[i] = 1.0
        else:
            occupancy[i] = 0.05 # Night security/cleaning

    # B. ITERATE THROUGH SELECTED EQUIPMENT
    for item_key, settings in selected_items.items():
        db_data = EQUIPMENT_DB[item_key]
        p_max = settings['power']
        
        # 1. BASE LOADS (24/7 Flat)
        if db_data['type'] == 'base' or db_data['type'] == 'medical_base':
            # Example: MRI Cryo is constant
            df['sim_base'] += p_max 
            # If it has an active component (scanning), add to medical
            if 'active_adder' in db_data:
                 df['sim_medical'] += db_data['active_adder'] * occupancy

        # 2. LIGHTING (Area based)
        elif db_data['type'] == 'lighting':
            # Watts/m2 * Area / 1000 = kW
            # Lighting doesn't turn off 100%, depends on occupancy
            light_load = (db_data['watts_per_m2'] * params['area'] / 1000)
            df['sim_lighting'] += light_load * np.maximum(occupancy, 0.1) * params['light_simultaneity']

        # 3. MEDICAL ACTIVE (Working Hours only)
        elif db_data['type'] == 'medical_active':
            df['sim_medical'] += p_max * occupancy * 0.6 # 0.6 is usage factor

        # 4. HVAC (Weather Dependent)
        elif db_data['type'] == 'hvac':
            # Base Standby (Fans moving air at night)
            df['sim_base'] += p_max * db_data['base_load_pct']
            
            # Active Load (Degree Days)
            neutral_temp = 21.0
            delta_t = (df['temperatura_c'] - neutral_temp).abs()
            delta_t = np.maximum(0, delta_t - 3.0) # Deadband
            
            # Physics: kW = Capacity * (DeltaT factor)
            # We assume p_max is the capacity at max load (deltaT=15 approx)
            hvac_load = (delta_t / 15.0) * p_max 
            
            # Schedule factor for HVAC
            hvac_sched = np.where(occupancy > 0, 1.0, 0.2) # Night setback
            
            df['sim_hvac'] += hvac_load * hvac_sched

    # TOTAL
    df['sim_total'] = df['sim_base'] + df['sim_lighting'] + df['sim_medical'] + df['sim_hvac']
    return df

# --- 3. UI ---
def show_nilm_page(df_consumo, df_clima):
    st.header("🏭 Inventory-Based Load Analysis")
    st.markdown("Build your digital twin by selecting the actual equipment installed in the building.")

    if df_consumo.empty or df_clima.empty:
        st.error("No data loaded.")
        return

    # DATA PREP (Average Day)
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    with st.expander("Filter Data (Season/Day Type)", expanded=False):
        day_type = st.radio("Day Type", ["Weekday (Mon-Fri)", "Weekend"], horizontal=True)
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if day_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        st.warning("No data for selection.")
        return

    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- INPUTS ---
    with st.sidebar:
        st.header("1. Building Params")
        area = st.number_input("Area (m²)", 100, 10000, 1500)
        c1, c2 = st.columns(2)
        s_start = c1.number_input("Open", 0, 23, 8)
        s_end = c2.number_input("Close", 0, 23, 19)
        
        st.markdown("---")
        st.header("2. Equipment Inventory")
        
        # DYNAMIC INVENTORY LIST
        selected_items = {}
        
        # PRESETS LOGIC
        preset = st.selectbox("Auto-Fill Preset", ["Custom", "Small Rehab Center", "Medium Day Hospital", "Large Hospital"])
        
        # HVAC SECTION
        st.subheader("❄️ HVAC System")
        if preset == "Small Rehab Center": def_hvac = "HVAC_SPLIT"
        elif preset == "Large Hospital": def_hvac = "HVAC_CHILLER"
        else: def_hvac = "HVAC_VRV"
        
        hvac_type = st.selectbox("Primary System", ["HVAC_VRV", "HVAC_CHILLER", "HVAC_SPLIT"], index=["HVAC_VRV", "HVAC_CHILLER", "HVAC_SPLIT"].index(def_hvac))
        hvac_cap = st.number_input("Installed Cooling Capacity (kW)", 10.0, 500.0, EQUIPMENT_DB[hvac_type]['default_power'])
        selected_items[hvac_type] = {'power': hvac_cap}

        # MEDICAL SECTION
        st.subheader("⚕️ Medical Equipment")
        
        # MRI
        has_mri = st.checkbox("MRI (Resonancia)", value=(preset=="Large Hospital"))
        if has_mri:
            mri_pow = st.number_input("MRI Cryo Base Load (kW)", 20.0, 100.0, 45.0, help="This creates the high base load")
            selected_items["MED_MRI"] = {'power': mri_pow}
            
        # XRAY
        has_xray = st.checkbox("X-Ray / CT", value=(preset!="Small Rehab Center"))
        if has_xray:
             xray_pow = st.number_input("X-Ray Max Power (kW)", 10.0, 100.0, 20.0)
             selected_items["MED_XRAY"] = {'power': xray_pow}
             
        # REHAB
        has_rehab = st.checkbox("Rehab (Physio)", value=True)
        if has_rehab:
             selected_items["MED_REHAB"] = {'power': 5.0}

        # LIGHTING / BASE
        st.subheader("💡 Lights & Base")
        l_tech = st.radio("Lighting Tech", ["LIGHT_LED", "LIGHT_FLUO"])
        selected_items[l_tech] = {'power': 0} # Power calc calculated inside based on area
        
        base_pow = st.number_input("Server Rack / UPS Base (kW)", 1.0, 50.0, 5.0)
        selected_items["BASE_SERVERS"] = {'power': base_pow}

    # --- CALCULATION ---
    params = {
        'area': area,
        'sched_start': s_start, 'sched_end': s_end,
        'light_simultaneity': 0.7
    }
    
    df_sim = calculate_inventory_model(df_avg, selected_items, params)

    # --- VISUALIZATION ---
    st.subheader("Equipment-Based Breakdown")
    
    # KPI
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    diff = sim - real
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Real Consumption", f"{real:,.0f} kWh")
    c2.metric("Inventory Simulation", f"{sim:,.0f} kWh", f"{diff:,.0f} kWh")
    
    if abs(diff) > real * 0.2:
        st.warning("⚠️ Large deviation. If Simulated < Real, you are missing equipment in the inventory (e.g., more base load).")

    # CHART
    fig = go.Figure()
    x = df_sim['hora']

    # Stacked Areas
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_base'], mode='lines', stackgroup='one', name='Base (Servers/MRI Cryo)', line=dict(width=0, color='gray')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_lighting'], mode='lines', stackgroup='one', name='Lighting', line=dict(width=0, color='#f1c40f')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_medical'], mode='lines', stackgroup='one', name='Medical Active', line=dict(width=0, color='#e67e22')))
    fig.add_trace(go.Scatter(x=x, y=df_sim['sim_hvac'], mode='lines', stackgroup='one', name='HVAC', line=dict(width=0, color='#3498db')))
    
    # Real Line
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER', line=dict(color='black', width=3)))

    fig.update_layout(height=500, hovermode="x unified", title="Detailed Load Curve")
    st.plotly_chart(fig, use_container_width=True)
    
    # Breakdown Table
    st.subheader("Estimated Energy Mix")
    res = df_sim[['sim_base', 'sim_lighting', 'sim_medical', 'sim_hvac']].sum().reset_index()
    res.columns = ['Category', 'kWh']
    res['%'] = (res['kWh'] / res['kWh'].sum() * 100).round(1)
    
    c1, c2 = st.columns(2)
    c1.dataframe(res)
    c2.plotly_chart(go.Figure(data=[go.Pie(labels=res['Category'], values=res['kWh'])]), use_container_width=True)
