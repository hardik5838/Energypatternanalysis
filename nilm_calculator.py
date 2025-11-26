import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. MOTOR FÍSICO (Simplificado para robustez) ---
def calculate_physics_model(df_avg, params):
    """
    Calcula el perfil energético basado en parámetros calibrados automáticamente.
    """
    df = df_avg.copy()
    hours = df['hora'].values
    
    # A. CARGA BASE (La "línea de suelo")
    # Es el consumo que existe a las 3 de la mañana.
    df['calc_base'] = params['base_kw'] 

    # B. ACTIVIDAD (Gente + Luces + Equipos)
    # Creamos una curva de ocupación basada en los horarios definidos
    occupancy = np.zeros(len(hours))
    for i, h in enumerate(hours):
        if params['sched_start'] <= h < params['sched_end']:
            # Curva trapezoidal simple (entrada -> full -> salida)
            if h == params['sched_start']: occupancy[i] = 0.5
            elif h == params['sched_end'] - 1: occupancy[i] = 0.5
            else: occupancy[i] = 1.0
    
    # Potencia Variable = (Luces + PCs + Maquinaria) * Ocupación
    df['calc_activity'] = params['activity_kw'] * occupancy

    # C. CLIMATIZACIÓN (HVAC) - El "Extra" por temperatura
    # Si hace calor fuera (verano) o frío (invierno), sumamos carga.
    
    # Delta T (Diferencia temperatura real vs ideal)
    # Usamos un "punto neutro" (ej: 21°C). Si nos alejamos, consumimos.
    delta_t = (df['temperatura_c'] - params['hvac_neutral_temp']).abs()
    
    # Eliminamos el consumo si la temperatura es "agradable" (banda muerta de +/- 3 grados)
    delta_t = np.maximum(0, delta_t - 3.0) 
    
    # Fórmula HVAC: Sensibilidad * DeltaT * Factor Horario (el clima baja de noche si no es 24h)
    hvac_factor = np.where(occupancy > 0, 1.0, 0.5) # De noche el clima trabaja al 50% (mantenimiento)
    df['calc_hvac'] = (delta_t * params['hvac_sensitivity']) * hvac_factor

    # TOTAL
    df['calc_total'] = df['calc_base'] + df['calc_activity'] + df['calc_hvac']
    
    return df

# --- 2. INTERFAZ CON AUTO-CALIBRACIÓN ---
def show_nilm_page(df_consumo, df_clima):
    st.header("🔬 Calibrador Automático de Cargas")
    
    if df_consumo.empty or df_clima.empty:
        st.error("Esperando datos...")
        return

    # 1. PREPARACIÓN DE DATOS (Día Promedio)
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # Filtro rápido (opcional, por defecto usa todo para tener mas datos)
    with st.expander("Configuración de Datos (Filtros)", expanded=False):
        dias_tipo = st.radio("Tipo de Día", ["Laborables (L-V)", "Fin de Semana"], horizontal=True)
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if dias_tipo == "Laborables (L-V)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        st.warning("No hay datos para generar la curva.")
        return

    # Generar Curva Promedio (La "Realidad" que vemos en tu foto)
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- 3. AUTO-CALIBRACIÓN (LA MAGIA) ---
    # Aquí leemos TU archivo para adivinar los valores iniciales
    
    # A. Detectar Carga Base (El mínimo de la noche, ej: 40kW)
    detected_base = df_avg['consumo_kwh'].min()
    
    # B. Detectar Pico Máximo
    detected_peak = df_avg['consumo_kwh'].max()
    
    # C. Estimar Actividad (Diferencia entre Pico y Base)
    # Asumimos que el 60% de esa subida es gente/luces y el resto clima
    detected_activity = (detected_peak - detected_base) * 0.6
    
    # D. Estimar Sensibilidad Clima (Muy burdo, pero mejor que nada)
    detected_hvac = (detected_peak - detected_base) * 0.4

    # --- 4. CONTROLES (Sliders pre-rellenos) ---
    with st.sidebar:
        st.header("🎛️ Ajuste Fino")
        st.info(f"Valores detectados: Base={detected_base:.0f}kW, Pico={detected_peak:.0f}kW")

        # BLOQUE 1: CARGA BASE (GRIS)
        st.subheader("1. Carga Base (24h)")
        base_kw = st.slider("Potencia Base (kW)", 
                            min_value=0.0, 
                            max_value=float(detected_peak * 1.2), 
                            value=float(detected_base), # <--- AUTO-AJUSTE AQUÍ
                            help="Ajusta esto para subir/bajar el SUELO gris del gráfico.")

        # BLOQUE 2: HORARIO Y ACTIVIDAD (NARANJA)
        st.subheader("2. Actividad (Horario)")
        col_t1, col_t2 = st.columns(2)
        sched_start = col_t1.number_input("Apertura", 0, 23, 7)
        sched_end = col_t2.number_input("Cierre", 0, 23, 19)
        
        activity_kw = st.slider("Carga por Actividad (kW)", 
                                min_value=0.0, 
                                max_value=float(detected_peak), 
                                value=float(detected_activity), # <--- AUTO-AJUSTE AQUÍ
                                help="Luces, PCs, Máquinas que se encienden cuando hay gente.")

        # BLOQUE 3: CLIMA (AZUL/ROJO)
        st.subheader("3. Climatización")
        hvac_sens = st.slider("Sensibilidad Clima", 0.0, 20.0, 2.0) # Valor por defecto conservador
        neutral_temp = st.number_input("Temp. Confort (°C)", 18, 26, 21)

    # --- 5. CÁLCULO ---
    params = {
        'base_kw': base_kw,
        'activity_kw': activity_kw,
        'sched_start': sched_start, 'sched_end': sched_end,
        'hvac_sensitivity': hvac_sens,
        'hvac_neutral_temp': neutral_temp
    }
    
    df_sim = calculate_physics_model(df_avg, params)

    # --- 6. VISUALIZACIÓN ---
    st.subheader("Simulación vs Realidad")
    
    # KPI de precisión
    error = df_sim['calc_total'].sum() - df_sim['consumo_kwh'].sum()
    st.caption(f"Diferencia Total Energía: {error:,.0f} kWh")

    fig = go.Figure()

    # Eje X
    x = df_sim['hora']

    # 1. Base (Gris) - Stacked
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_base'], mode='lines', stackgroup='one', name='Carga Base (Standby)',
        line=dict(width=0, color='darkgray')
    ))

    # 2. Actividad (Naranja) - Stacked
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_activity'], mode='lines', stackgroup='one', name='Actividad (Luces/Equipos)',
        line=dict(width=0, color='#F39C12')
    ))

    # 3. Clima (Azul) - Stacked
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_hvac'], mode='lines', stackgroup='one', name='Climatización Est. (HVAC)',
        line=dict(width=0, color='#3498DB')
    ))

    # 4. REALIDAD (Línea Negra)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REALIDAD',
        line=dict(color='black', width=3)
    ))

    fig.update_layout(
        xaxis_title="Hora", yaxis_title="kW",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Mensajes de ayuda dinámica
    if df_sim['calc_base'].max() < df_sim['consumo_kwh'].min() * 0.9:
        st.info("💡 **Consejo:** Tu gráfico real flota por encima del simulado. Sube la **Potencia Base** en el menú lateral.")
    elif df_sim['calc_total'].max() < df_sim['consumo_kwh'].max():
        st.info("💡 **Consejo:** Te falta altura en las horas centrales. Sube la **Carga por Actividad**.")

