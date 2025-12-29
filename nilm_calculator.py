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
        # Lógica basada en Clima
        delta = (np.maximum(0, df['temperatura_c'] - config['set_c']) + 
                 np.maximum(0, config['set_h'] - df['temperatura_c']))
        raw = delta * config['therm_sens']
        sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched

    total_custom = np.zeros(len(hours))
    for p in config['processes']:
        p_load = generate_load_curve(hours, p['s'], p['e'], p['kw'], p['ru'], p['rd'], p['dips'])
        df[f"proc_{p['name']}"] = p_load
        total_custom += p_load
    df['sim_proc'] = total_custom
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm'] + df['sim_proc']
    
    if 'consumo_kwh' in df.columns:
        df['diff'] = df['sim_total'] - df['consumo_kwh']
    return df

# ==========================================
# 2. UI LAYOUT
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Energy Pattern Digital Twin")

    if df_consumo.empty: 
        st.error("No hay datos de consumo disponibles."); return
    if df_clima.empty:
        st.warning("⚠️ No hay datos de clima. El modo 'Weather Driven' no funcionará.");

    # --- UNIFICACIÓN DE NOMBRES DE COLUMNAS ---
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    if not df_clima.empty:
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        # Merge usando 'fecha' en minúsculas
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 20.0 # Valor por defecto si no hay clima

    if df_merged.empty:
        st.error("No se han encontrado coincidencias de fecha entre Consumo y Clima."); return

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
            v_ru = st.number_input("Ramp Up Vent", 0.0, 5.0, 0.5)
            v_rd = st.number_input("Ramp Dn Vent", 0.0, 5.0, 0.5)

        with st.expander("2. Iluminación", expanded=False):
            light_kw = st.number_input("Luz Total [kW]", 0.0, 5000.0, 20.0)
            l_fac = st.slider("Factor Operación %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Seguridad/Noche %", 0.0, 0.5, 0.1)
            l_s, l_e = st.slider("Horario Luz", 0, 24, (7, 21))
            l_ru, l_rd = 0.5, 0.5

        with st.expander("3. HVAC / Climatización", expanded=False):
            therm_kw = st.number_input("Capacidad HVAC [kW]", 0.0, 10000.0, 45.0)
            t_s, t_e = st.slider("Horario HVAC", 0, 24, (8, 19))
            mode = st.selectbox("Modo", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sensibilidad", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Cool (Verano)", 18, 30, 24)
                sh = st.number_input("Set Heat (Invierno)", 15, 25, 20)

        procs = [] # Aquí iría la lógica de Custom Processes si la necesitas

    # --- AGREGACIÓN POR HORA ---
    # Al agrupar por .dt.hour, el nombre resultante es 'fecha'
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 
        'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': base_kw, 'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 'light_ru': l_ru, 'light_rd': l_rd,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': procs
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- VISUALIZACIÓN ---
    st.subheader(f"📊 Digital Twin: {day_type}")
    
    # Gráfico de Áreas
    fig = go.Figure()
    layers = [('sim_base', 'Base', '#95a5a6'), ('sim_vent', 'Vent.', '#3498db'), 
              ('sim_therm', 'HVAC', '#e74c3c'), ('sim_light', 'Luz', '#f1c40f')]
    
    for col, name, color in layers:
        fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
    
    fig.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL', line=dict(color='black', width=3)))
    fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Métricas
    c1, c2, c3 = st.columns(3)
    real_sum = df_sim['consumo_kwh'].sum()
    sim_sum = df_sim['sim_total'].sum()
    c1.metric("Energía Real (Día Medio)", f"{real_sum:.1f} kWh")
    c2.metric("Energía Simulada", f"{sim_sum:.1f} kWh", delta=f"{sim_sum-real_sum:.1f}")
    c3.metric("Error (RMSE)", f"{np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total'])**2).mean()):.2f}")
