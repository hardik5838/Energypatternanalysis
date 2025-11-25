import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def calcular_modelo_energetico(df_merged, area, hvac_params, light_params, baseload_params, schedule_params):
    """
    Realiza los cálculos vectorizados para estimar el desglose de energía.
    """
    df = df_merged.copy()
    
    # --- 1. Definir Horario Operativo ---
    # Asumimos horario simple: Lunes a Viernes entre horas de inicio y fin
    # 0 = Lunes, 4 = Viernes
    is_working_day = df['fecha'].dt.dayofweek <= 4 
    is_working_hour = (df['fecha'].dt.hour >= schedule_params['start']) & (df['fecha'].dt.hour <= schedule_params['end'])
    df['is_open'] = is_working_day & is_working_hour

    # --- 2. Carga Base (Always On) ---
    # Servidores, neveras, stamby, luces de seguridad
    df['est_base'] = baseload_params['power_kw']

    # --- 3. Iluminación ---
    # Potencia = Area * Densidad (W/m2) / 1000. 
    # Se aplica al 100% en horario operativo y un % reducido fuera de horario
    potencia_luces_total = (area * light_params['density_w_m2']) / 1000
    
    conditions = [df['is_open'] == True, df['is_open'] == False]
    choices = [potencia_luces_total, potencia_luces_total * light_params['off_factor']]
    
    df['est_lighting'] = np.select(conditions, choices, default=0)

    # --- 4. Climatización (HVAC) ---
    # Modelo simplificado basado en Grados-Día (Diferencia de temperatura)
    # Q = U * A * DeltaT.  Electricidad = Q / COP
    
    # Factor térmico global (simplificación de U * A)
    thermal_factor = hvac_params['thermal_sensitivity'] 
    
    # Enfriamiento (Cooling)
    delta_t_cool = (df['temperatura_c'] - hvac_params['setpoint_cool']).clip(lower=0)
    # Solo enfría si la temperatura exterior es mayor al setpoint Y está dentro del horario (o si se permite fuera de horario)
    mask_cool = (df['temperatura_c'] > hvac_params['setpoint_cool']) & df['is_open']
    df['est_cooling'] = 0.0
    df.loc[mask_cool, 'est_cooling'] = (delta_t_cool * thermal_factor) / hvac_params['cop_cool']
    
    # Calefacción (Heating)
    delta_t_heat = (hvac_params['setpoint_heat'] - df['temperatura_c']).clip(lower=0)
    mask_heat = (df['temperatura_c'] < hvac_params['setpoint_heat']) & df['is_open']
    df['est_heating'] = 0.0
    df.loc[mask_heat, 'est_heating'] = (delta_t_heat * thermal_factor) / hvac_params['cop_heat']

    # --- Total Estimado ---
    df['consumo_estimado'] = df['est_base'] + df['est_lighting'] + df['est_cooling'] + df['est_heating']
    
    return df

def show_nilm_page(df_consumo, df_clima):
    st.header("🔬 Análisis de Cargas No Intrusivo (Simulación)")
    st.markdown("""
    Esta herramienta simula el comportamiento del edificio basándose en parámetros físicos. 
    **Instrucciones:** Ajusta los deslizadores de la izquierda hasta que la línea roja (Estimado) se superponga lo mejor posible a la línea azul (Real).
    """)

    if df_consumo.empty or df_clima.empty:
        st.warning("Se necesitan datos de Consumo y Clima cargados para usar esta herramienta.")
        return

    # Unir datos para asegurar alineación temporal
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    if df_merged.empty:
        st.error("No hay fechas coincidentes entre consumo y clima.")
        return

    # --- Controles de Simulación (Sidebar Específico) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎛️ Parámetros del Edificio")
        
        # 1. Dimensiones y Horario
        area = st.number_input("Superficie Total (m²)", value=1000, step=100)
        
        c1, c2 = st.columns(2)
        sched_start = c1.number_input("Hora Inicio", 0, 23, 8)
        sched_end = c2.number_input("Hora Fin", 0, 23, 18)
        schedule_params = {'start': sched_start, 'end': sched_end}

        # 2. Carga Base
        st.markdown("**Carga Base (Standby)**")
        base_kw = st.slider("Potencia Base (kW)", 0.0, 50.0, float(df_merged['consumo_kwh'].min()), help="Consumo mínimo nocturno/fin de semana")
        baseload_params = {'power_kw': base_kw}

        # 3. Iluminación
        st.markdown("**Iluminación**")
        light_dens = st.slider("Densidad (W/m²)", 0, 20, 8, help="Oficina LED típica: 6-9 W/m²")
        light_off = st.slider("% Iluminación fuera de horario", 0.0, 1.0, 0.1)
        light_params = {'density_w_m2': light_dens, 'off_factor': light_off}

        # 4. Climatización (HVAC)
        st.markdown("**HVAC (Clima)**")
        t_sens = st.slider("Sensibilidad Térmica (Factor Aislamiento)", 0.0, 50.0, 5.0, help="Qué tanto afecta 1 grado de temperatura exterior al consumo")
        
        c3, c4 = st.columns(2)
        set_cool = c3.number_input("Setfrío (°C)", 18, 30, 24)
        cop_cool = c4.number_input("EER/COP Frío", 1.0, 5.0, 2.5)
        
        c5, c6 = st.columns(2)
        set_heat = c5.number_input("Set Calor (°C)", 15, 25, 20)
        cop_heat = c6.number_input("COP Calor", 1.0, 5.0, 2.5)
        
        hvac_params = {
            'thermal_sensitivity': t_sens,
            'setpoint_cool': set_cool, 'cop_cool': cop_cool,
            'setpoint_heat': set_heat, 'cop_heat': cop_heat
        }

    # --- Calcular Modelo ---
    df_sim = calcular_modelo_energetico(df_merged, area, hvac_params, light_params, baseload_params, schedule_params)

    # --- Visualización ---
    
    # 1. Gráfico Comparativo Principal
    st.subheader("Realidad vs Simulación")
    
    # Remuestrear a diario para ver mejor la tendencia si hay muchos datos
    df_daily = df_sim.set_index('fecha').resample('D').sum().reset_index()
    
    # Crear gráfico de áreas apiladas para el estimado
    fig = px.area(df_daily, x='fecha', y=['est_base', 'est_lighting', 'est_heating', 'est_cooling'],
                  title="Desglose Estimado de Energía (kWh diarios)",
                  labels={'value': 'kWh', 'variable': 'Uso Final'},
                  color_discrete_map={
                      'est_base': '#bdc3c7', 
                      'est_lighting': '#f1c40f',
                      'est_heating': '#e74c3c',
                      'est_cooling': '#3498db'
                  })
    
    # Añadir línea de consumo real
    fig.add_scatter(x=df_daily['fecha'], y=df_daily['consumo_kwh'], mode='lines', 
                    name='CONSUMO REAL', line=dict(color='black', width=2, dash='dot'))
    
    st.plotly_chart(fig, use_container_width=True)

    # 2. Scatter de Calibración
    st.subheader("Calidad del Ajuste")
    col_kpi1, col_kpi2 = st.columns(2)
    
    total_real = df_sim['consumo_kwh'].sum()
    total_est = df_sim['consumo_estimado'].sum()
    error_pct = ((total_est - total_real) / total_real) * 100
    
    col_kpi1.metric("Energía Total Real", f"{total_real:,.0f} kWh")
    col_kpi2.metric("Energía Total Estimada", f"{total_est:,.0f} kWh", f"{error_pct:.1f}%")
    
    with st.expander("Ver detalle horario (Debugging)"):
        st.dataframe(df_sim[['fecha', 'temperatura_c', 'consumo_kwh', 'consumo_estimado', 'est_cooling', 'est_heating']].head(100))
