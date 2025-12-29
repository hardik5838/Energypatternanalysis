import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 1. LOGIC ENGINE
# ==========================================
def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, dips=None):
    if dips is None: dips = []
    curve = np.zeros(len(hours))
    for i, h in enumerate(hours):
        val = 0.0
        if start <= h < end:
            val = 1.0
            if h < (start + ramp_up) and ramp_up > 0: val = (h - start) / ramp_up
            if h >= (end - ramp_down) and ramp_down > 0: val = (end - h) / ramp_down
            for dip in dips:
                if int(h) == int(dip['hour']): val *= dip['factor']
        curve[i] = np.clip(val, 0.0, 1.0) * max_kw
    return curve

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. Base Load
    df['sim_base'] = np.full(len(hours), config['base_kw'])
    
    # 2. Ventilation
    df['sim_vent'] = generate_load_curve(hours, config['vent_s'], config['vent_e'], 
                                        config['vent_kw'], config['vent_ru'], config['vent_rd'])
    
    # 3. Lighting (Fixing the KeyError by using explicit ramp keys)
    light_curve = generate_load_curve(hours, config['light_s'], config['light_e'], 
                                     config['light_kw'], config['light_ru'], config['light_rd'])
    df['sim_light'] = light_curve * config['light_fac']
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_sec']

    # 4. HVAC
    if config['hvac_mode'] == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        delta = (np.maximum(0, df['temperatura_c'] - config['set_c']) + 
                 np.maximum(0, config['set_h'] - df['temperatura_c']))
        raw = delta * config['therm_sens']
        sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched

    # Total Sum
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm']
    
    # Fit Metrics
    if 'consumo_kwh' in df.columns:
        df['error_kw'] = df['sim_total'] - df['consumo_kwh']
        df['error_pct'] = (df['error_kw'] / df['consumo_kwh'].replace(0, 1)) * 100
    
    return df

# ==========================================
# 2. UI LAYOUT
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Energy Pattern Digital Twin")

    # Column Normalization
    df_consumo = df_consumo.copy()
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    
    if not df_clima.empty:
        df_clima = df_clima.copy()
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 20.0

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎛️ Fitting Controls")
        day_type = st.radio("Profile Type", ["Laborable", "Fin de Semana"], horizontal=True)
        is_weekday = (day_type == "Laborable")
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask_day].copy()

        with st.expander("1. Infrastructure", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0.0, 5000.0, 20.0)
            vent_kw = st.number_input("Ventilation [kW]", 0.0, 5000.0, 30.0)
            v_s, v_e = st.slider("Vent. Hours", 0, 24, (6, 20))

        with st.expander("2. Lighting", expanded=True):
            light_kw = st.number_input("Lighting Cap [kW]", 0.0, 5000.0, 15.0)
            l_fac = st.slider("Op. Factor %", 0.0, 1.0, 0.9)
            l_sec = st.slider("Security %", 0.0, 0.5, 0.05)
            l_s, l_e = st.slider("Light Hours", 0, 24, (7, 21))

        with st.expander("3. HVAC", expanded=False):
            therm_kw = st.number_input("HVAC [kW]", 0.0, 10000.0, 40.0)
            t_s, t_e = st.slider("HVAC Hours", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sensitivity", 1.0, 20.0, 5.0)
                sc = st.number_input("Cool Setpoint", 18, 30, 24)
                sh = st.number_input("Heat Setpoint", 15, 25, 20)

    # --- DATA PROCESSING ---
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    # Building the config with ALL required keys to avoid KeyError
    config = {
        'base_kw': base_kw, 
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': 0.5, 'vent_rd': 0.5,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 
        'light_ru': 0.2, 'light_rd': 0.2, # Added these keys to fix the error
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- 6 VISUALIZATION CHARTS ---
    st.subheader(f"📊 Operations Visualization: {day_type}")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        # 1. Main Load Distribution (Stacked)
        fig1 = go.Figure()
        layers = [('sim_base', 'Base', '#95a5a6'), ('sim_vent', 'Vent.', '#3498db'), 
                  ('sim_therm', 'HVAC', '#e74c3c'), ('sim_light', 'Light', '#f1c40f')]
        for col, name, color in layers:
            fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL DATA', line=dict(color='black', width=3)))
        fig1.update_layout(title="Daily Load Mix vs Real", xaxis_title="Hour", yaxis_title="kW")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Consumption Mix (Pie Chart)
        mix_data = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light']].sum()
        fig2 = px.pie(values=mix_data.values, names=mix_data.index, title="Total Energy Share (%)", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Cumulative Energy Matching
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Acc (kWh)", fill='tozeroy'))
        fig3.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Acc (kWh)"))
        fig3.update_layout(title="Volume Matching (Cumulative kWh)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        # 4. Hourly Error (Bar)
        fig4 = px.bar(df_sim, x='hora', y='error_kw', color='error_kw', 
                     color_continuous_scale='RdBu_r', title="Hourly Fitting Error (kW)")
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Accuracy Heatmap
        st.write("**Fitting Accuracy Map (%)**")
        fig5 = px.imshow(df_sim['error_pct'].values.reshape(1, -1), color_continuous_scale='Viridis', aspect="auto")
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Correlation Analysis
        fig6 = px.scatter(df_sim, x='consumo_kwh', y='sim_total', trendline="ols", title="Model Correlation (Real vs Sim)")
        st.plotly_chart(fig6, use_container_width=True)

    # --- FULL EXPORTABLE TABLE ---
    st.divider()
    st.subheader("📥 Export Load Distribution Table")
    
    # Cleaning table for export
    export_df = df_sim.rename(columns={
        'hora': 'Hour', 'sim_base': 'Base_kW', 'sim_vent': 'Vent_kW', 
        'sim_light': 'Light_kW', 'sim_therm': 'HVAC_kW', 'sim_total': 'Total_Sim_kW',
        'consumo_kwh': 'Real_kW', 'error_kw': 'Error_kW'
    })
    
    st.dataframe(export_df.style.format(precision=2), use_container_width=True)
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV for Optimization", data=csv, file_name=f"load_mix_{day_type}.csv", mime='text/csv')
