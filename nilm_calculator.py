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
    df['sim_base'] = np.full(len(hours), config['base_kw'])
    df['sim_vent'] = generate_load_curve(hours, config['vent_s'], config['vent_e'], config['vent_kw'], config['vent_ru'], config['vent_rd'])
    
    light_curve = generate_load_curve(hours, config['light_s'], config['light_e'], config['light_kw'], config['light_ru'], config['light_rd'])
    df['sim_light'] = light_curve * config['light_fac']
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_sec']

    if config['hvac_mode'] == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        delta = (np.maximum(0, df['temperatura_c'] - config['set_c']) + 
                 np.maximum(0, config['set_h'] - df['temperatura_c']))
        raw = delta * config['therm_sens']
        sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched

    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm']
    
    if 'consumo_kwh' in df.columns:
        df['error_abs'] = df['sim_total'] - df['consumo_kwh']
        df['error_pct'] = (df['error_abs'] / df['consumo_kwh'].replace(0, 1)) * 100
    return df

# ==========================================
# 2. UI LAYOUT
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Energy Pattern Digital Twin")

    if df_consumo.empty:
        st.error("No hay datos de consumo disponibles.")
        return

    # Normalización de nombres de columnas (Caso Insensitivo)
    df_consumo = df_consumo.copy()
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    
    if not df_clima.empty:
        df_clima = df_clima.copy()
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 20.0

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🎛️ Control Panel")
        day_type = st.radio("Tipo de Perfil", ["Laborable", "Fin de Semana"], horizontal=True)
        is_weekday = (day_type == "Laborable")
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask_day].copy()

        with st.expander("1. Base & Vent", expanded=True):
            base_kw = st.number_input("Carga Base [kW]", 0.0, 5000.0, 20.0)
            vent_kw = st.number_input("Ventilación [kW]", 0.0, 5000.0, 30.0)
            v_s, v_e = st.slider("Horario Vent.", 0, 24, (6, 20))

        with st.expander("2. Iluminación", expanded=False):
            light_kw = st.number_input("Luz Total [kW]", 0.0, 5000.0, 20.0)
            l_fac = st.slider("Factor Operación %", 0.0, 1.0, 0.8)
            l_s, l_e = st.slider("Horario Luz", 0, 24, (7, 21))

        with st.expander("3. HVAC", expanded=False):
            therm_kw = st.number_input("Capacidad HVAC [kW]", 0.0, 10000.0, 45.0)
            t_s, t_e = st.slider("Horario HVAC", 0, 24, (8, 19))
            mode = st.selectbox("Modo", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sensibilidad", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Cool", 18, 30, 24)
                sh = st.number_input("Set Heat", 15, 25, 20)

    # --- AGREGACIÓN ---
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': base_kw, 'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': 0.5, 'vent_rd': 0.5,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': 0.1,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': []
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- VISUALIZACIONES (6 CHARTS) ---
    st.subheader(f"📊 Digital Twin Analysis: {day_type}")
    
    c1, c2 = st.columns(2)
    with c1:
        # 1. Stacked Load Distribution
        fig1 = go.Figure()
        layers = [('sim_base', 'Base', '#95a5a6'), ('sim_vent', 'Vent.', '#3498db'), 
                  ('sim_therm', 'HVAC', '#e74c3c'), ('sim_light', 'Luz', '#f1c40f')]
        for col, name, color in layers:
            fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL', line=dict(color='black', width=3)))
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Error Heatmap
        st.write("Hourly Error Heatmap (%)")
        fig2 = px.imshow(df_sim['error_pct'].values.reshape(1, -1), color_continuous_scale='RdBu_r', labels=dict(color="% Error"))
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Correlation
        fig3 = px.scatter(df_sim, x='consumo_kwh', y='sim_total', trendline="ols", title="Fit Correlation")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        # 4. Hourly Error Bars
        fig4 = px.bar(df_sim, x='hora', y='error_abs', title="Hourly Error (kW)", color='error_abs', color_continuous_scale='Portland')
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Consumption Mix
        mix = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light']].sum()
        fig5 = px.pie(values=mix.values, names=mix.index, title="Load Mix Split")
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Cumulative Energy
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Acc."))
        fig6.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Acc."))
        fig6.update_layout(title="Cumulative Energy (kWh)")
        st.plotly_chart(fig6, use_container_width=True)

    # --- TABLE & EXPORT ---
    st.divider()
    st.subheader("Data Export")
    st.dataframe(df_sim, use_container_width=True)
    csv = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button("Download Fit Data (CSV)", data=csv, file_name="energy_model.csv", mime='text/csv')
