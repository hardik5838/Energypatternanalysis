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
    
    # 2. VENTILATION (Now with Ramps)
    df['sim_vent'] = generate_load_curve(hours, config['vent_s'], config['vent_e'], config['vent_kw'], config['vent_ru'], config['vent_rd'])

    # 3. LIGHTING (Now with Ramps)
    light_curve = generate_load_curve(hours, config['light_s'], config['light_e'], config['light_kw'], config['light_ru'], config['light_rd'])
    # Apply usage factor
    df['sim_light'] = light_curve * config['light_fac']
    # Security Light
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_sec']

    # 4. THERMAL (HVAC)
    if config['hvac_mode'] == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        # Weather Logic
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
# 2. UI LAYOUT
# ==========================================
def show_dashboard(df_consumo, df_clima):
    st.set_page_config(layout="wide", page_title="Load Calibrator")
    
    # --- DATA PREP ---
    if df_consumo.empty: 
        st.error("No Data"); return
    
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # A. GLOBAL FILTER
        day_type = st.radio("Profile Type", ["Weekday", "Weekend"], horizontal=True)
        is_weekday = (day_type == "Weekday")
        mask = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask].copy()
        
        if df_filtered.empty: st.warning("No data"); return
        
        # B. BASE
        with st.expander("1. Base & Vent", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0, 500, 20)
            st.markdown("**Ventilation**")
            vent_kw = st.number_input("Vent kW", 0, 500, 30)
            c1, c2 = st.columns(2)
            v_s, v_e = c1.slider("Vent Time", 0, 24, (6, 20))
            v_ru = c2.number_input("Ramp Up (h)", 0.0, 5.0, 0.5)
            v_rd = c2.number_input("Ramp Down (h)", 0.0, 5.0, 0.5, key="v_rd")

        # C. LIGHTING
        with st.expander("2. Lighting", expanded=False):
            light_kw = st.number_input("Light kW", 0, 500, 20)
            l_fac = st.slider("Factor %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Security %", 0.0, 0.5, 0.1)
            c3, c4 = st.columns(2)
            l_s, l_e = c3.slider("Light Time", 0, 24, (7, 21))
            l_ru = c4.number_input("L-Ramp Up", 0.0, 2.0, 0.5)
            l_rd = c4.number_input("L-Ramp Down", 0.0, 2.0, 0.5)

        # D. THERMAL
        with st.expander("3. HVAC Thermal", expanded=False):
            therm_kw = st.number_input("Chiller kW", 0, 500, 45)
            t_s, t_e = st.slider("Therm Time", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sens.", 1.0, 10.0, 5.0)
                sc = st.number_input("Set Cool", 20, 30, 24)
                sh = st.number_input("Set Heat", 15, 25, 20)

        # E. CUSTOM PROCESSES
        with st.expander("4. Custom Processes", expanded=False):
            if 'n_proc' not in st.session_state: st.session_state['n_proc'] = 1
            if st.button("Add Process"): st.session_state['n_proc'] += 1
            if st.button("Remove Last"): st.session_state['n_proc'] = max(0, st.session_state['n_proc'] - 1)
            
            procs = []
            for i in range(st.session_state['n_proc']):
                st.markdown(f"**Proc {i+1}**")
                pk = st.number_input(f"kW {i+1}", 0.0, 200.0, 10.0, key=f"pk{i}")
                ps, pe = st.slider(f"Time {i+1}", 0, 24, (8, 17), key=f"pt{i}")
                pr_u = st.number_input(f"R-Up {i+1}", 0.0, 5.0, 0.5, key=f"pru{i}")
                pr_d = st.number_input(f"R-Dn {i+1}", 0.0, 5.0, 0.5, key=f"prd{i}")
                
                # Simple Dip Logic
                has_dip = st.checkbox(f"Lunch Dip? {i+1}", value=(i==0), key=f"pd{i}")
                dips = [{'hour': 14, 'factor': 0.5}] if has_dip else []
                
                procs.append({'name': f"P{i+1}", 'kw': pk, 's': ps, 'e': pe, 'ru': pr_u, 'rd': pr_d, 'dips': dips})

    # --- CALCULATION ---
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

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
    
    # METRICS ROW
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean())
    
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    c_kpi1.metric("Real Energy", f"{real:,.0f} kWh")
    c_kpi2.metric("Simulated Energy", f"{sim:,.0f} kWh", delta=f"{sim-real:,.0f}")
    c_kpi3.metric("RMSE Error", f"{rmse:.2f}", help="Lower is better")

    # ROW 1: MAIN LOAD PROFILE CHART
    fig = go.Figure()
    x = df_sim['hora']
    
    # Stacked Areas
    layers = [
        ('sim_base', 'Base', 'gray'),
        ('sim_vent', 'Ventilation', '#3498db'),
        ('sim_therm', 'Thermal', '#e74c3c'),
        ('sim_light', 'Lighting', '#f1c40f'),
        ('sim_proc', 'Processes', '#e67e22')
    ]
    for col, name, color in layers:
        fig.add_trace(go.Scatter(x=x, y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        
    # Real Line
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL', line=dict(color='black', width=3)))
    
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ROW 2: SPLIT VIEW (Pie Chart | Residuals)
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Pie Chart
        # Summing specific columns
        sums = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light', 'sim_proc']].sum().reset_index()
        sums.columns = ['Category', 'kWh']
        sums['Category'] = sums['Category'].str.replace('sim_', '').str.capitalize()
        
        fig_pie = px.pie(sums, values='kWh', names='Category', title="Energy Mix", hole=0.4)
        fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        # Residuals (Bar Chart)
        df_sim['color'] = np.where(df_sim['diff'] > 0, '#ef5350', '#42a5f5')
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=df_sim['hora'], y=df_sim['diff'], marker_color=df_sim['color']))
        fig_res.update_layout(title="Deviation (Sim - Real)", height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="kW Error")
        st.plotly_chart(fig_res, use_container_width=True)

    # ROW 3: COMPACT DATA TOGGLE
    with st.expander("Show Raw Data Table"):
        st.dataframe(df_sim.round(1), use_container_width=True)

# --- EXECUTION ---
if __name__ == "__main__":
    # Dummy Data Generator
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    df_con = pd.DataFrame({'fecha': dates, 'consumo_kwh': [
        20,20,20,20,20,25,40,60,80,90,95,90,80,70,60,70,80,60,40,30,25,20,20,20
    ]})
    df_cli = pd.DataFrame({'fecha': dates, 'temperatura_c': np.linspace(15, 25, 24)})
    
    show_dashboard(df_con, df_cli)
