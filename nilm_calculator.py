import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 1. CORE LOGIC ENGINE
# ==========================================

def generate_flexible_load(hours, start, end, max_kw, ramp_up_hr, ramp_down_hr, dips):
    """
    Generates a load curve with:
    - Independent Ramp Up / Ramp Down slopes
    - Multiple "Dips" (reductions in load during specific hours)
    """
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        val = 0.0
        
        # 1. Determine "Active Window"
        # We extend the window slightly to account for ramps
        effective_start = start
        effective_end = end
        
        if h >= effective_start and h < effective_end:
            val = 1.0
            
            # 2. Ramp Up Logic
            # If current hour is within the ramp up period
            if h < (effective_start + ramp_up_hr):
                # Avoid division by zero
                if ramp_up_hr > 0:
                    val = (h - effective_start) / ramp_up_hr
            
            # 3. Ramp Down Logic
            # If current hour is within the ramp down period (approaching end)
            if h >= (effective_end - ramp_down_hr):
                if ramp_down_hr > 0:
                    val = (effective_end - h) / ramp_down_hr
                    
            # 4. Dip Logic (Lunch/Breaks)
            # dips is a list of dicts: [{'hour': 14, 'factor': 0.5}, ...]
            for dip in dips:
                if int(h) == int(dip['hour']):
                    val *= dip['factor'] # Reduce load by factor
                    
        curve[i] = np.clip(val, 0.0, 1.0) * max_kw
        
    return curve

def calculate_full_model(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # --- A. BASE LOAD ---
    df['sim_base'] = np.full(len(hours), config['base_kw'])
    
    # --- B. HVAC (VENT & THERMAL) ---
    # Ventilation
    df['sim_vent'] = generate_flexible_load(hours, config['vent_start'], config['vent_end'], config['vent_kw'], 0, 0, [])
    
    # Thermal
    if config['hvac_mode'] == "Constant (Block)":
        df['sim_thermal'] = generate_flexible_load(hours, config['therm_start'], config['therm_end'], config['therm_kw'], 2, 2, [])
    else: # Weather Driven
        delta_cool = np.maximum(0, df['temperatura_c'] - config['set_cool'])
        delta_heat = np.maximum(0, config['set_heat'] - df['temperatura_c'])
        raw_demand = (delta_cool + delta_heat) * config['therm_sens']
        
        # Schedule mask
        sched = generate_flexible_load(hours, config['therm_start'], config['therm_end'], 1.0, 1, 1, [])
        # Cap at capacity
        df['sim_thermal'] = np.minimum(raw_demand, config['therm_kw']) * sched

    # --- C. LIGHTING ---
    # Simple block for lighting with minor ramping
    light_raw = generate_flexible_load(hours, config['light_start'], config['light_end'], config['light_kw'], 1, 1, [])
    df['sim_light'] = light_raw * config['light_factor']
    
    # Security Lighting logic
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_security_pct']

    # --- D. CUSTOM PROCESSES (Dynamic Loop) ---
    # We sum all custom processes into one column for the chart, but keep them separate in a list for the pie chart
    total_custom = np.zeros(len(hours))
    
    # We will store individual process columns for the dataframe
    for proc in config['processes']:
        p_name = f"proc_{proc['name']}"
        p_load = generate_flexible_load(
            hours, 
            proc['start'], proc['end'], 
            proc['kw'], 
            proc['ramp_up'], proc['ramp_down'], 
            proc['dips']
        )
        df[p_name] = p_load
        total_custom += p_load

    df['sim_process'] = total_custom
    
    # --- TOTAL ---
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_thermal'] + df['sim_light'] + df['sim_process']
    
    # CALIBRATION DELTAS
    if 'consumo_kwh' in df.columns:
        df['diff'] = df['sim_total'] - df['consumo_kwh']
        
    return df

# ==========================================
# 2. UI LAYOUT
# ==========================================

def show_nilm_page(df_consumo, df_clima):
    st.title("🏭 Advanced Digital Twin Calibrator")
    st.markdown("Use the controls below to build a bottom-up model that matches your real meter data.")

    # --- 1. DATA FILTERING ---
    if df_consumo.empty or df_clima.empty:
        st.error("Missing Data")
        return

    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # Top Bar Filter
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        d_type = st.selectbox("Select Day Type", ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"])
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if d_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    # Create Average Profile
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # ==========================================
    # 3. THE "CALCULATOR" (MAIN PAGE)
    # ==========================================
    st.divider()
    st.subheader("🛠️ Calculator Variables")
    
    # We use columns to organize the inputs so it's not a long vertical list
    col1, col2, col3 = st.columns(3)

    # --- COLUMN 1: BASE & HVAC ---
    with col1:
        with st.expander("🏗️ Base & HVAC", expanded=True):
            st.markdown("**Base Load (24/7)**")
            base_kw = st.number_input("Standby Power [kW]", 0.0, 1000.0, 20.0, step=1.0)
            
            st.markdown("---")
            st.markdown("**Ventilation (Fans)**")
            vent_kw = st.number_input("Fan Power [kW]", 0.0, 500.0, 30.0)
            v_s, v_e = st.slider("Fan Schedule", 0, 24, (6, 20))
            
            st.markdown("---")
            st.markdown("**Thermal (Chillers/Boilers)**")
            therm_kw = st.number_input("Thermal Capacity [kW]", 0.0, 1000.0, 45.0)
            t_s, t_e = st.slider("Thermal Schedule", 0, 24, (8, 19))
            hvac_mode = st.selectbox("Mode", ["Constant (Block)", "Weather Driven"])
            
            therm_sens, set_c, set_h = 5.0, 24, 20
            if hvac_mode == "Weather Driven":
                therm_sens = st.slider("Sens. (kW/deg)", 0.1, 20.0, 5.0)
                set_c = st.number_input("Set Cool", 20, 30, 24)
                set_h = st.number_input("Set Heat", 15, 25, 20)

    # --- COLUMN 2: LIGHTING ---
    with col2:
        with st.expander("💡 Lighting", expanded=True):
            light_kw = st.number_input("Total Light Power [kW]", 0.0, 500.0, 20.0)
            l_s, l_e = st.slider("Light Schedule", 0, 24, (7, 21))
            light_fac = st.slider("Simultaneity (%)", 0.1, 1.0, 0.8)
            light_sec = st.slider("Night Security (%)", 0.0, 1.0, 0.1)

    # --- COLUMN 3: CUSTOM PROCESSES (THE NEW FEATURE) ---
    with col3:
        with st.expander("⚙️ Process & Equipment", expanded=True):
            st.info("Add machines/lines here.")
            
            # Session state to track number of processes
            if 'proc_count' not in st.session_state:
                st.session_state['proc_count'] = 1

            def add_proc(): st.session_state['proc_count'] += 1
            def del_proc(): st.session_state['proc_count'] = max(0, st.session_state['proc_count'] - 1)

            c_btn1, c_btn2 = st.columns(2)
            c_btn1.button("➕ Add", on_click=add_proc)
            c_btn2.button("➖ Remove", on_click=del_proc)

            processes_config = []
            
            # Dynamic Loop for Process Inputs
            for i in range(st.session_state['proc_count']):
                st.markdown(f"**Process #{i+1}**")
                name = st.text_input(f"Name #{i+1}", f"Machine {i+1}", key=f"n_{i}")
                kw = st.number_input(f"Power [kW]", 0.0, 500.0, 10.0, key=f"k_{i}")
                
                # Advanced Timing (Ramps)
                start, end = st.slider(f"Schedule", 0, 24, (8, 17), key=f"s_{i}")
                
                with st.expander("Advanced Curves (Ramp/Dip)"):
                    ru = st.slider(f"Ramp Up (Hrs)", 0.0, 4.0, 0.5, step=0.5, key=f"ru_{i}")
                    rd = st.slider(f"Ramp Down (Hrs)", 0.0, 4.0, 0.5, step=0.5, key=f"rd_{i}")
                    
                    # Multi-Dips
                    dip_hours = st.multiselect(f"Dip Hours (e.g. Lunch)", range(0, 24), default=[14] if i==0 else [], key=f"d_{i}")
                    dip_factor = st.slider(f"Dip Load %", 0.0, 1.0, 0.5, key=f"df_{i}")
                    
                    dips = [{'hour': h, 'factor': dip_factor} for h in dip_hours]

                processes_config.append({
                    'name': name, 'kw': kw, 'start': start, 'end': end,
                    'ramp_up': ru, 'ramp_down': rd, 'dips': dips
                })
                st.markdown("---")

    # --- 4. RUN CALCULATIONS ---
    config = {
        'base_kw': base_kw,
        'light_kw': light_kw, 'light_start': l_s, 'light_end': l_e, 'light_factor': light_fac, 'light_security_pct': light_sec,
        'vent_kw': vent_kw, 'vent_start': v_s, 'vent_end': v_e,
        'therm_kw': therm_kw, 'therm_start': t_s, 'therm_end': t_e, 'hvac_mode': hvac_mode,
        'therm_sens': therm_sens, 'set_cool': set_c, 'set_heat': set_h,
        'processes': processes_config
    }
    
    df_sim = calculate_full_model(df_avg, config)

    # ==========================================
    # 5. ANALYSIS DASHBOARD
    # ==========================================
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # --- METRICS ---
    real_sum = df_sim['consumo_kwh'].sum()
    sim_sum = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean())
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Real Energy (Day)", f"{real_sum:,.0f} kWh")
    m2.metric("Simulated Energy", f"{sim_sum:,.0f} kWh", f"{sim_sum - real_sum:,.0f} kWh")
    m3.metric("Match Score (RMSE)", f"{rmse:.2f}", help="Lower is better")
    m4.metric("Accuracy", f"{100 - (abs(sim_sum-real_sum)/real_sum*100):.1f}%")

    # --- CHARTS ROW 1 ---
    tab1, tab2, tab3 = st.tabs(["📈 Load Profile", "🍰 Energy Mix", "📉 Load Duration"])
    
    with tab1:
        # MAIN STACKED CHART
        fig = go.Figure()
        x = df_sim['hora']
        
        # Add Standard Layers
        fig.add_trace(go.Scatter(x=x, y=df_sim['sim_base'], stackgroup='one', name='Base Load', line=dict(width=0, color='gray')))
        fig.add_trace(go.Scatter(x=x, y=df_sim['sim_vent'], stackgroup='one', name='Ventilation', line=dict(width=0, color='#3498db')))
        fig.add_trace(go.Scatter(x=x, y=df_sim['sim_thermal'], stackgroup='one', name='Thermal', line=dict(width=0, color='#e74c3c')))
        fig.add_trace(go.Scatter(x=x, y=df_sim['sim_light'], stackgroup='one', name='Lighting', line=dict(width=0, color='#f1c40f')))
        
        # Add Process Layers Dynamically
        colors = px.colors.qualitative.Prism
        for idx, proc in enumerate(processes_config):
            col_name = f"proc_{proc['name']}"
            c = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(x=x, y=df_sim[col_name], stackgroup='one', name=proc['name'], line=dict(width=0, color=c)))

        # Add Real Line
        fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER', line=dict(color='black', width=3)))
        
        fig.update_layout(height=500, title="Load Profile Matching", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # PIE CHART
        cols_to_sum = ['sim_base', 'sim_vent', 'sim_thermal', 'sim_light'] + [f"proc_{p['name']}" for p in processes_config]
        sums = df_sim[cols_to_sum].sum().reset_index()
        sums.columns = ['Category', 'kWh']
        # Clean up names
        sums['Category'] = sums['Category'].str.replace('sim_', '').str.replace('proc_', '').str.capitalize()
        
        fig_pie = px.pie(sums, values='kWh', names='Category', title="Total Energy Breakdown (Daily %)", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        # LOAD DURATION CURVE (Engineering Tool)
        # Sorts real and sim from high to low
        real_sorted = np.sort(df_sim['consumo_kwh'])[::-1]
        sim_sorted = np.sort(df_sim['sim_total'])[::-1]
        
        fig_ldc = go.Figure()
        fig_ldc.add_trace(go.Scatter(y=real_sorted, mode='lines', name='Real Duration', line=dict(color='black')))
        fig_ldc.add_trace(go.Scatter(y=sim_sorted, mode='lines', name='Sim Duration', line=dict(color='red', dash='dash')))
        fig_ldc.update_layout(title="Load Duration Curve (Sorted Demand)", xaxis_title="Hours", yaxis_title="kW")
        st.plotly_chart(fig_ldc, use_container_width=True)

    # --- ERROR ANALYSIS & DATA ---
    st.subheader("📝 Detailed Data & Error Check")
    
    c_err1, c_err2 = st.columns([2, 1])
    
    with c_err1:
        # RESIDUAL CHART
        df_sim['color'] = np.where(df_sim['diff'] > 0, 'crimson', 'royalblue')
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=df_sim['hora'], y=df_sim['diff'], marker_color=df_sim['color']))
        fig_res.update_layout(title="Difference (Simulated - Real)", yaxis_title="Error (kW)", height=300)
        st.plotly_chart(fig_res, use_container_width=True)
    
    with c_err2:
        # RAW DATA TABLE
        st.dataframe(df_sim[['hora', 'consumo_kwh', 'sim_total', 'diff']].style.format("{:.1f}"), height=300)
        
        # CSV DOWNLOAD
        csv = df_sim.to_csv(index=False).encode('utf-8')
        st.download_button("Download Analysis CSV", csv, "calibrated_model.csv", "text/csv")

# --- DUMMY DATA FOR TESTING ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    df_consumo = pd.DataFrame({'fecha': dates, 'consumo_kwh': np.random.uniform(20, 100, 24)})
    df_clima = pd.DataFrame({'fecha': dates, 'temperatura_c': np.random.uniform(10, 25, 24)})
    show_nilm_page(df_consumo, df_clima)
