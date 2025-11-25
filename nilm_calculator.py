import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def calculate_average_profile(df_filtered):
    """
    Genera un perfil promedio de 24 horas basado en los datos filtrados.
    """
    # Agrupar por hora del día (0-23) y calcular medias
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean',
        'temperatura_c': 'mean',
        'humedad_relativa': 'mean'
    }).reset_index()
    
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)
    return df_avg

def simulate_load_profile(df_avg, params):
    """
    Aplica el modelo físico sobre el perfil promedio de 24h.
    """
    df = df_avg.copy()
    
    # --- 1. Carga Base (Always On) ---
    df['sim_base'] = params['base_kw']
    
    # --- 2. Iluminación (Horario solar inverso simplificado) ---
    # Asumimos que las luces se necesitan más cuando oscurece (tarde/noche) o horario fijo
    # Modelo simple: Activo entre hora inicio y fin definidos
    mask_lights = (df['hora'] >= params['light_start']) & (df['hora'] <= params['light_end'])
    df['sim_lights'] = np.where(mask_lights, params['light_kw'], params['light_kw'] * params['light_off_factor'])
    
    # --- 3. Maquinaria / Procesos (Carga Rectangular) ---
    # Para explicar consumos altos industriales o turnos
    mask_proc = (df['hora'] >= params['proc_start']) & (df['hora'] <= params['proc_end'])
    # Manejo de turno nocturno (ej: 22:00 a 06:00)
    if params['proc_end'] < params['proc_start']:
        mask_proc = (df['hora'] >= params['proc_start']) | (df['hora'] <= params['proc_end'])
        
    df['sim_process'] = np.where(mask_proc, params['proc_kw'], 0)

    # --- 4. Climatización (HVAC) basada en Temperatura Promedio Horaria ---
    # Q = Sensibilidad * DeltaT
    
    # Frío
    delta_t_cool = (df['temperatura_c'] - params['set_cool']).clip(lower=0)
    df['sim_cooling'] = (delta_t_cool * params['sens_cool']) / params['cop_cool']
    
    # Calor
    delta_t_heat = (params['set_heat'] - df['temperatura_c']).clip(lower=0)
    df['sim_heating'] = (delta_t_heat * params['sens_heat']) / params['cop_heat']
    
    # Apagar HVAC fuera de horario operativo (opcional)
    if not params['hvac_always_on']:
        mask_hvac = (df['hora'] >= params['hvac_start']) & (df['hora'] <= params['hvac_end'])
        # Turno nocturno HVAC
        if params['hvac_end'] < params['hvac_start']:
            mask_hvac = (df['hora'] >= params['hvac_start']) | (df['hora'] <= params['hvac_end'])
            
        df['sim_cooling'] = np.where(mask_hvac, df['sim_cooling'], df['sim_cooling'] * 0.1) # 10% remanente
        df['sim_heating'] = np.where(mask_hvac, df['sim_heating'], df['sim_heating'] * 0.1)

    # Suma Total
    df['sim_total'] = (df['sim_base'] + df['sim_lights'] + 
                       df['sim_process'] + df['sim_cooling'] + df['sim_heating'])
    
    return df

def show_nilm_page(df_consumo, df_clima):
    st.header("🔬 Análisis de Perfil Promedio (Calibración)")
    st.markdown("""
    Esta herramienta calcula el **"Día Promedio"** de los datos seleccionados y permite ajustar las cargas
    para entender qué está consumiendo energía. Útil para detectar cargas base altas o turnos nocturnos.
    """)

    if df_consumo.empty or df_clima.empty:
        st.error("Faltan datos.")
        return

    # Unir DataFrames
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # --- FILTROS DE SEGMENTACIÓN (Data Slicing) ---
    with st.expander("Step 1: Filtrar Datos para el Promedio", expanded=True):
        c1, c2, c3 = st.columns(3)
        
        # Filtro Meses (Estacionalidad)
        meses = df_merged['fecha'].dt.month_name().unique()
        sel_meses = c1.multiselect("Meses a analizar", meses, default=meses)
        
        # Filtro Días (Laborables vs Fin de Semana)
        tipo_dia = c2.radio("Tipo de Día", ["Todos", "Laborables (L-V)", "Fin de Semana (S-D)"], index=1)
        
        # Filtro Rango Horario (opcional para visualización, pero calculamos sobre 24h)
        # No necesario para el perfil promedio, se usa todo el día.

    # Aplicar Filtros
    mask = df_merged['fecha'].dt.month_name().isin(sel_meses)
    if tipo_dia == "Laborables (L-V)":
        mask &= df_merged['fecha'].dt.dayofweek < 5
    elif tipo_dia == "Fin de Semana (S-D)":
        mask &= df_merged['fecha'].dt.dayofweek >= 5
        
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        st.warning("No hay datos con esos filtros.")
        return

    # Calcular Perfil Real Promedio
    df_avg_real = calculate_average_profile(df_filtered)

    # --- PARAMETRIZACIÓN (Sidebar) ---
    with st.sidebar:
        st.subheader("🎛️ Configuración del Edificio")
        
        with st.expander("1. Cargas Fijas y Maquinaria", expanded=True):
            st.info("Ajusta esto primero para igualar el 'suelo' del gráfico.")
            base_kw = st.number_input("Standby / Base (kW)", 0.0, 500.0, float(df_avg_real['consumo_kwh'].min()), step=1.0)
            
            st.markdown("---")
            st.markdown("**Procesos / Maquinaria (Turnos)**")
            proc_kw = st.number_input("Potencia Proceso (kW)", 0.0, 500.0, 0.0, step=5.0)
            p_start, p_end = st.slider("Horario Proceso", 0, 23, (8, 18))
            
        with st.expander("2. Iluminación"):
            light_kw = st.number_input("Potencia Luces (kW)", 0.0, 100.0, 5.0)
            l_start, l_end = st.slider("Horario Luces", 0, 23, (7, 20))
            light_off = st.slider("% Luces fuera horario", 0.0, 1.0, 0.1)

        with st.expander("3. Climatización (HVAC)"):
            hvac_always = st.checkbox("HVAC 24h", value=False)
            h_start, h_end = st.slider("Horario HVAC", 0, 23, (8, 19), disabled=hvac_always)
            
            st.markdown("**Calefacción**")
            set_heat = st.number_input("Set Calor (°C)", 15, 25, 21)
            sens_heat = st.number_input("Pérdidas Calor (kW/°C)", 0.0, 50.0, 2.0)
            cop_heat = st.number_input("COP Calor", 1.0, 5.0, 2.5)
            
            st.markdown("**Refrigeración**")
            set_cool = st.number_input("Set Frío (°C)", 20, 30, 24)
            sens_cool = st.number_input("Ganancia Solar (kW/°C)", 0.0, 50.0, 5.0)
            cop_cool = st.number_input("EER Frío", 1.0, 5.0, 3.0)

    # Empaquetar parámetros
    params = {
        'base_kw': base_kw,
        'proc_kw': proc_kw, 'proc_start': p_start, 'proc_end': p_end,
        'light_kw': light_kw, 'light_start': l_start, 'light_end': l_end, 'light_off_factor': light_off,
        'hvac_always_on': hvac_always, 'hvac_start': h_start, 'hvac_end': h_end,
        'set_heat': set_heat, 'sens_heat': sens_heat, 'cop_heat': cop_heat,
        'set_cool': set_cool, 'sens_cool': sens_cool, 'cop_cool': cop_cool
    }

    # --- SIMULACIÓN ---
    df_sim = simulate_load_profile(df_avg_real, params)

    # --- VISUALIZACIÓN ---
    st.subheader(f"Perfil Diario Promedio: {tipo_dia}")
    
    # Métricas de Error
    real_total = df_sim['consumo_kwh'].sum()
    sim_total = df_sim['sim_total'].sum()
    diff = sim_total - real_total
    color_metric = "normal" if abs(diff) < (real_total * 0.1) else "off"
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Real (24h)", f"{real_total:,.0f} kWh")
    m2.metric("Simulado (24h)", f"{sim_total:,.0f} kWh", f"{diff:,.0f} kWh", delta_color=color_metric)
    m3.metric("Temp Promedio", f"{df_sim['temperatura_c'].mean():.1f} °C")

    # Gráfico Principal
    fig = go.Figure()

    # Áreas Apiladas (Simulación)
    x = df_sim['hora']
    
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['sim_base'], mode='lines', stackgroup='one', name='Base (Standby)',
        line=dict(width=0, color='#bdc3c7')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['sim_process'], mode='lines', stackgroup='one', name='Maquinaria/Proceso',
        line=dict(width=0, color='#9b59b6')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['sim_lights'], mode='lines', stackgroup='one', name='Iluminación',
        line=dict(width=0, color='#f1c40f')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['sim_heating'], mode='lines', stackgroup='one', name='Calefacción',
        line=dict(width=0, color='#e74c3c')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['sim_cooling'], mode='lines', stackgroup='one', name='Refrigeración',
        line=dict(width=0, color='#3498db')
    ))

    # Línea Real (Comparativa)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='Consumo REAL Promedio',
        line=dict(color='black', width=3, dash='solid')
    ))

    fig.update_layout(
        title="Desglose de Energía (Promedio Horario)",
        xaxis_title="Hora del Día",
        yaxis_title="Potencia (kW)",
        xaxis=dict(tickmode='linear', dtick=1),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Guía de ajuste:**
    1. Si el consumo nocturno real (línea negra) es alto, sube **Standby** o añade **Maquinaria** en horas nocturnas.
    2. Si hay un bloque cuadrado de consumo, usa **Maquinaria** y ajusta las horas.
    3. Usa **HVAC** solo para explicar los picos que coinciden con temperaturas extremas.
    """)
