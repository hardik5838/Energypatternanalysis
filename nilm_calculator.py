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
    
    # Use .get() to prevent KeyErrors if keys are missing
    df['sim_base'] = np.full(len(hours), config.get('base_kw', 0))
    
    df['sim_vent'] = generate_load_curve(
        hours, config.get('vent_s', 6), config.get('vent_e', 20), 
        config.get('vent_kw', 0), config.get('vent_ru', 0.5), config.get('vent_rd', 0.5)
    )
    
    # Fix for the Light KeyError
    df['sim_light'] = generate_load_curve(
        hours, config.get('light_s', 7), config.get('light_e', 21), 
        config.get('light_kw', 0), config.get('light_ru', 0.5), config.get('light_rd', 0.5)
    ) * config.get('light_fac', 1.0)
    
    # Handle Night/Security Lighting
    is_off = (df['sim_light'] < (config.get('light_kw', 0) * 0.1))
    df.loc[is_off, 'sim_light'] = config.get('light_kw', 0) * config.get('light_sec', 0.1)

    # HVAC Logic
    if config.get('hvac_mode') == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config.get('therm_s', 8), config.get('therm_e', 19), config.get('therm_kw', 0), 1, 1)
    else:
        delta = (np.maximum(0, df['temperatura_c'] - config.get('set_c', 24)) + 
                 np.maximum(0, config.get('set_h', 20) - df['temperatura_c']))
        raw = delta * config.get('therm_sens', 5.0)
        sched = generate_load_curve(hours, config.get('therm_s', 8), config.get('therm_e', 19), 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config.get('therm_kw', 0)) * sched

    # Calculate Totals
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm']
    
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
        df_merged['temperatura_c'] = 22.0

    # --- SIDEBAR ---
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
            light_kw = st.number_input("Light Max [kW]", 0.0, 5000.0, 15.0)
            l_fac = st.slider("Op. Factor %", 0.0, 1.0, 0.9)
            l_s, l_e = st.slider("Light Hours", 0, 24, (7, 21))

        with st.expander("3. HVAC", expanded=False):
            therm_kw = st.number_input("HVAC Cap [kW]", 0.0, 10000.0, 40.0)
            t_s, t_e = st.slider("HVAC Hours", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sensitivity", 1.0, 20.0, 5.0)
                sc = st.number_input("Cool Set", 18, 30, 24)
                sh = st.number_input("Heat Set", 15, 25, 20)

    # --- AGGREGATION ---
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    # Full Configuration Dictionary
    config = {
        'base_kw': base_kw, 
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': 0.5, 'vent_rd': 0.5,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': 0.1, 
        'light_ru': 0.5, 'light_rd': 0.5,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- 6 CHARTS SECTION ---
    st.subheader(f"📊 Model Visualization: {day_type}")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Chart 1: Stacked Load
        fig1 = go.Figure()
        layers = [('sim_base', 'Base', '#bdc3c7'), ('sim_vent', 'Vent.', '#3498db'), 
                  ('sim_therm', 'HVAC', '#e74c3c'), ('sim_light', 'Light', '#f1c40f')]
        for col, name, color in layers:
            fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL', line=dict(color='black', width=3)))
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Consumption Mix
        mix = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light']].sum()
        st.plotly_chart(px.pie(values=mix.values, names=mix.index, title="Load Mix Share"), use_container_width=True)

        # Chart 3: Correlation
        st.plotly_chart(px.scatter(df_sim, x='consumo_kwh', y='sim_total', trendline="ols", title="Fit Quality Correlation"), use_container_width=True)

    with c2:
        # Chart 4: Hourly Error Bar
        st.plotly_chart(px.bar(df_sim, x='hora', y='error_kw', title="Hourly Error (kW)", color='error_kw', color_continuous_scale='RdBu_r'), use_container_width=True)

        # Chart 5: Error Heatmap
        st.write("**Error Percentage Heatmap**")
        fig5 = px.imshow(df_sim['error_pct'].values.reshape(1, -1), color_continuous_scale='Plasma')
        st.plotly_chart(fig5, use_container_width=True)

        # Chart 6: Cumulative
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Cumulative", fill='tozeroy'))
        fig6.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Cumulative"))
        fig6.update_layout(title="Energy Accumulation (kWh)")
        st.plotly_chart(fig6, use_container_width=True)

    # --- FULL EXPORTABLE TABLE ---
    st.divider()
    st.subheader("📋 Consumption Distribution Data")
    st.dataframe(df_sim, use_container_width=True)
    
    csv = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV Export", data=csv, file_name="load_distribution.csv", mime='text/csv')
