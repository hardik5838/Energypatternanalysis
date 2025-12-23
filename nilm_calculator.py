## nilm_calculator.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 1. LOGIC ENGINE (Flexible Ramps for ALL)
# ==========================================
def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, dips=None):
    """
    Universal function to generate a load curve with:
    - Ramps (Up/Down)
    - Schedule (Start/End)
    - Dips (Lunch/Breaks)
    """
    if dips is None: dips = []
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        val = 0.0
        
        # 1. Active Window
        if start <= h < end:
            val = 1.0
            
            # 2. Ramp Up
            if h < (start + ramp_up):
                if ramp_up > 0: val = (h - start) / ramp_up
            
            # 3. Ramp Down
            if h >= (end - ramp_down):
                if ramp_down > 0: val = (end - h) / ramp_down
            
            # 4. Dips
            for dip in dips:
                if int(h) == int(dip['hour']):
                    val *= dip['factor']
                    
        curve[i] = np.clip(val, 0.0, 1.0) * max_kw
        
    return curve

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. BASE
    df['sim_base'] = np.full(len(hours), config['base_kw'])
    
    # 2. VENTILATION
    df['sim_vent'] = generate_load_curve(hours, config['vent_s'], config['vent_e'], config['vent_kw'], config['vent_ru'], config['vent_rd'])

    # 3. LIGHTING
    light_curve = generate_load_curve(hours, config['light_s'], config['light_e'], config['light_kw'], config['light_ru'], config['light_rd'])
    df['sim_light'] = light_curve * config['light_fac']
    
    # Security Light
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_sec']

    # 4. THERMAL (HVAC)
    if config['hvac_mode'] == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        # Weather Logic - requires temperatura_c to be numeric
        delta = (np.maximum(0, df['temperatura_c'] - config['set_c']) + 
                 np.maximum(0, config['set_h'] - df['temperatura_c']))
        raw = delta * config['therm_sens']
        sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched

    # 5. CUSTOM PROCESSES
    total_custom = np.zeros(len(hours))
    for p in config['processes']:
        p_load = generate_load_curve(hours, p['s'], p['e'], p['kw'], p['ru'], p['rd'], p['dips'])
        col_name = f"proc_{p['name']}"
        df[col_name] = p_load
        total_custom += p_load
    df['sim_proc'] = total_custom

    # TOTAL
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm'] + df['sim_proc']
    
    if 'consumo_kwh' in df.columns:
        df['diff'] = df['sim_total'] - df['consumo_kwh']
        
    return df

# ==========================================
# 2. UI LAYOUT & DATA HANDLING
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Energy Pattern Digital Twin")

    # --- DATA PREP ---
    if df_consumo.empty: 
        st.error("No Data provided to calculator"); return
    
    try:
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
        df_merged['fecha'] = pd.to_datetime(df_merged['fecha'])
    except KeyError:
        st.error("Error: Input data must have a 'fecha' column.")
        return

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        st.markdown("### 🗓️ Data Cleaning")
        available_months = df_merged['fecha'].dt.month_name().unique()
        months_to_remove = st.multiselect(
            "Exclude Months (Anomalies)", 
            options=available_months,
            placeholder="Select months to ignore..."
        )
        
        if months_to_remove:
            mask_month = ~df_merged['fecha'].dt.month_name().isin(months_to_remove)
            df_merged = df_merged[mask_month]

        st.divider()

        day_type = st.radio("Profile Type", ["Weekday", "Weekend"], horizontal=True)
        is_weekday = (day_type == "Weekday")
        
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        # Create a deep copy to safely modify types
        df_filtered = df_merged[mask_day].copy()
        
        if df_filtered.empty: 
            st.warning(f"No data found for {day_type} after filtering."); return
        
        # --- PARAMETERS ---
        with st.expander("1. Base & Vent", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0.0, 10000.0, 20.0, step=1.0)
            vent_kw = st.number_input("Vent kW", 0.0, 10000.0, 30.0, step=1.0)
            c1, c2 = st.columns(2)
            v_s, v_e = c1.slider("Vent Time", 0, 24, (6, 20))
            v_ru = c2.number_input("Ramp Up (h)", 0.0, 5.0, 0.5)
            v_rd = c2.number_input("Ramp Down (h)", 0.0, 5.0, 0.5)

        with st.expander("2. Lighting", expanded=False):
            light_kw = st.number_input("Light kW", 0.0, 10000.0, 20.0, step=1.0)
            l_fac = st.slider("Factor %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Security %", 0.0, 0.5, 0.1)
            c3, c4 = st.columns(2)
            l_s, l_e = c3.slider("Light Time", 0, 24, (7, 21))
            l_ru = c4.number_input("L-Ramp Up", 0.0, 5.0, 0.5)
            l_rd = c4.number_input("L-Ramp Down", 0.0, 5.0, 0.5)

        with st.expander("3. HVAC Thermal", expanded=False):
            therm_kw = st.number_input("Chiller kW", 0.0, 50000.0, 45.0, step=5.0)
            t_s, t_e = st.slider("Therm Time", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sens.", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Cool", 18, 30, 24)
                sh = st.number_input("Set Heat", 15, 25, 20)

        with st.expander("4. Custom Processes", expanded=False):
            if 'n_proc' not in st.session_state: st.session_state['n_proc'] = 1
            b1, b2 = st.columns(2)
            if b1.button("➕ Add"): st.session_state['n_proc'] += 1
            if b2.button("➖ Remove"): st.session_state['n_proc'] = max(0, st.session_state['n_proc'] - 1)
            
            procs = []
            for i in range(st.session_state['n_proc']):
                st.markdown(f"**Proc {i+1}**")
                pk = st.number_input(f"kW {i+1}", 0.0, 5000.0, 10.0, key=f"pk{i}")
                ps, pe = st.slider(f"Time {i+1}", 0, 24, (8, 17), key=f"pt{i}")
                pr_u = st.number_input(f"R-Up {i+1}", 0.0, 5.0, 0.5, key=f"pru{i}")
                pr_d = st.number_input(f"R-Dn {i+1}", 0.0, 5.0, 0.5, key=f"prd{i}")
                has_dip = st.checkbox(f"Lunch Dip? {i+1}", value=(i==0), key=f"pd{i}")
                dips = [{'hour': 14, 'factor': 0.5}] if has_dip else []
                procs.append({'name': f"P{i+1}", 'kw': pk, 's': ps, 'e': pe, 'ru': pr_u, 'rd': pr_d, 'dips': dips})

    # --- CALCULATION (Hardened) ---
    
    # 1. Ensure numeric types to prevent TypeError in .agg()
    for col in ['consumo_kwh', 'temperatura_c']:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # 2. Check if conversion resulted in all NaNs (common with bad CSV formatting)
    if df_filtered['consumo_kwh'].isnull().all():
        st.error("CRITICAL: The consumption column contains non-numeric data that cannot be processed.")
        return

    # 3. Aggregate safely by hour
    try:
        # Group and calculate mean
        df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
            'consumo_kwh': 'mean', 
            'temperatura_c': 'mean'
        }).reset_index()
        
        # Rename 'fecha' (which is now the hour index) to 'hora'
        df_avg = df_avg.rename(columns={'fecha': 'hora'})
        
    except Exception as e:
        st.error(f"Data Aggregation Failed: {e}")
        return

    # 4. Simulation
    config = {
        'base_kw': base_kw,
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 'light_ru': l_ru, 'light_rd': l_rd,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': procs
    }
    
    df_sim = run_simulation(df_avg, config)

    # ==========================================
    # 3. MAIN DASHBOARD AREA
    # ==========================================
    st.subheader(f"📊 Digital Twin Calibration ({day_type})")
    
    # KPI Metrics
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean())
    
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    c_kpi1.metric("Real Energy (Avg Day)", f"{real:,.1f} kWh")
    c_kpi2.metric("Simulated Energy", f"{sim:,.1f} kWh", delta=f"{sim-real:,.1f} kWh", delta_color="inverse")
    c_kpi3.metric("RMSE (Error)", f"{rmse:.2f}", help="Aim for < 5.0 for a good fit")

    # Load Profile Chart
    fig = go.Figure()
    x = df_sim['hora']
    
    layers = [
        ('sim_base', 'Base Load', 'gray'),
        ('sim_vent', 'Ventilation', '#3498db'),
        ('sim_therm', 'Thermal (HVAC)', '#e74c3c'),
        ('sim_light', 'Lighting', '#f1c40f'),
        ('sim_proc', 'Processes', '#e67e22')
    ]
    for col, name, color in layers:
        fig.add_trace(go.Scatter(x=x, y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER', line=dict(color='black', width=3)))
    
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # Split View: Pie Chart & Residuals
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        sums = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light', 'sim_proc']].sum().reset_index()
        sums.columns = ['Category', 'kWh']
        sums['Category'] = sums['Category'].str.replace('sim_', '').str.capitalize()
        fig_pie = px.pie(sums, values='kWh', names='Category', title="Energy Breakdown", hole=0.4)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        df_sim['color'] = np.where(df_sim['diff'] > 0, '#ef5350', '#42a5f5')
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=df_sim['hora'], y=df_sim['diff'], marker_color=df_sim['color']))
        fig_res.update_layout(title="Residuals (Sim - Real)", height=350, yaxis_title="kW Difference")
        st.plotly_chart(fig_res, use_container_width=True)

    with st.expander("📂 View Raw Comparison Data"):
        st.dataframe(df_sim.round(2), use_container_width=True)
