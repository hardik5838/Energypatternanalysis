import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- HELPER: CURVAS DE OCUPACIÓN ---
def get_occupancy_curve(hours, start_hour, end_hour, profile_type="Oficina Estándar"):
    """
    Genera un perfil de ocupación (0.0 a 1.0) para cada hora del día.
    No es binario (0 o 1), sino gradual para simular la realidad.
    """
    occupancy = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        if start_hour <= h < end_hour:
            if profile_type == "Oficina Estándar":
                # Curva típica: Sube rápido, baja un poco a comer, baja al final
                if h == start_hour: occupancy[i] = 0.5     # Llegada
                elif h == end_hour - 1: occupancy[i] = 0.4 # Salida
                elif h == 14: occupancy[i] = 0.7           # Hora comida
                else: occupancy[i] = 1.0                   # Pleno rendimiento
            elif profile_type == "Centro Médico (Continuo)":
                # Mantiene actividad más constante
                occupancy[i] = 0.9 if h == 14 else 1.0
        else:
            # Consumo residual fuera de horario (limpieza, seguridad)
            occupancy[i] = 0.05
            
    return occupancy

def calculate_physics_model(df_avg, params):
    """
    Motor de cálculo basado en física y auditoría.
    """
    df = df_avg.copy()
    hours = df['hora'].values
    
    # --- 1. PERFIL DE OCUPACIÓN ---
    # Define qué % de gente/máquinas están activas
    occ_curve = get_occupancy_curve(hours, params['sched_start'], params['sched_end'], params['bldg_type'])
    
    # --- 2. CARGAS FIJAS (BASE / STANDBY) ---
    # Esto es el bloque GRIS: Servidores, Neveras, Standby máquinas
    # Es constante las 24h
    df['calc_base'] = params['base_kw_servers'] + params['base_kw_standby'] + params['base_kw_medical_always_on']

    # --- 3. ILUMINACIÓN (DEPENDIENTE DE OCUPACIÓN) ---
    # Potencia Total = m2 * W/m2.
    # Consumo = Potencia * (Factor Ocupación + Factor Luz Natural)
    # Si es LED, reducimos potencia base.
    tech_factor = 0.5 if params['light_tech'] == "LED" else 1.0
    total_light_power = (params['area_m2'] * params['light_density_w_m2'] / 1000) * tech_factor
    
    # La luz no se apaga linealmente con la ocupación (la gente deja luces encendidas), 
    # pero baja significativamente.
    df['calc_light'] = total_light_power * params['light_simultaneity'] * np.maximum(occ_curve, params['light_min_security'])

    # --- 4. EQUIPAMIENTO MÉDICO Y FUERZA (VARIABLE) ---
    # Equipos que dependen de que haya gente usándolos (PCs, Rayos X, Cafeteras)
    # Potencia * Ocupación
    total_equip_power = params['equip_kw_variable']
    df['calc_equip'] = total_equip_power * occ_curve * params['equip_simultaneity']
    
    # ACS (Agua Caliente) - Asumimos constante o leve pico, simplificamos a constante por ahora
    df['calc_dhw'] = params['kw_dhw']

    # --- 5. CLIMATIZACIÓN (HVAC) - MOTOR FÍSICO ---
    # Fórmula: P = (Transmittance * DeltaT) / COP
    
    # Delta T Frío (Solo si T_ext > Setpoint)
    delta_t_cool = (df['temperatura_c'] - params['hvac_set_cool']).clip(lower=0)
    
    # Delta T Calor (Solo si T_ext < Setpoint)
    delta_t_heat = (params['hvac_set_heat'] - df['temperatura_c']).clip(lower=0)
    
    # Aplicar horario al HVAC (generalmente se enciende 1h antes)
    hvac_start_pre = max(0, params['sched_start'] - 1)
    # Creamos mascara de horario HVAC
    hvac_active = np.zeros(len(hours))
    for i, h in enumerate(hours):
        if hvac_start_pre <= h < params['sched_end']:
            hvac_active[i] = 1.0
        else:
            hvac_active[i] = 0.1 # Mantenimiento nocturno reducido
            
    # Calculo Potencia Térmica requerida -> Potencia Eléctrica
    # Transmitancia (kW/°C) representa qué tan mal aislado está el edificio
    df['calc_cooling'] = (delta_t_cool * params['hvac_transmittance'] / params['hvac_cop_cool']) * hvac_active
    df['calc_heating'] = (delta_t_heat * params['hvac_transmittance'] / params['hvac_cop_heat']) * hvac_active

    # --- TOTAL ---
    df['calc_total'] = (df['calc_base'] + df['calc_light'] + df['calc_equip'] + 
                        df['calc_dhw'] + df['calc_cooling'] + df['calc_heating'])
    
    return df

def show_nilm_page(df_consumo, df_clima):
    st.header("🔬 Auditoría Energética Virtual (Modelo Físico)")
    st.markdown("""
    Este modelo desglosa el consumo utilizando la **física del edificio** y perfiles de **ocupación real**, 
    siguiendo la metodología de auditoría de Asepeyo.
    """)

    if df_consumo.empty or df_clima.empty:
        st.error("Faltan datos.")
        return

    # Preparar datos promedio (Día Tipo)
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # Filtros rápidos para definir el "Día Tipo"
    with st.expander("📅 Seleccionar Periodo para el Día Promedio", expanded=False):
        c1, c2 = st.columns(2)
        meses = df_merged['fecha'].dt.month_name().unique()
        sel_meses = c1.multiselect("Meses", meses, default=meses)
        tipo_dia = c2.radio("Días", ["Laborables (L-V)", "Fin de Semana"], index=0)
    
    mask = df_merged['fecha'].dt.month_name().isin(sel_meses)
    if tipo_dia == "Laborables (L-V)": mask &= df_merged['fecha'].dt.dayofweek < 5
    else: mask &= df_merged['fecha'].dt.dayofweek >= 5
    
    df_filtered = df_merged[mask].copy()
    if df_filtered.empty: return

    # Calcular promedios horarios reales
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- INPUTS DE AUDITORÍA (SIDEBAR) ---
    with st.sidebar:
        st.subheader("🏭 1. Configuración General")
        bldg_type = st.selectbox("Tipo Centro", ["Oficina Estándar", "Centro Médico (Continuo)"])
        area_m2 = st.number_input("Superficie (m²)", 100, 10000, 1000, step=50)
        s_start, s_end = st.slider("Horario Apertura", 0, 23, (8, 18))

        st.subheader("🔋 2. Cargas Fijas (Base - Gris)")
        st.caption("Consumo que NUNCA desaparece (24h)")
        base_servers = st.number_input("Servidores/Rack (kW)", 0.0, 50.0, 2.0, help="Rack servers: ~0.5-2kW constante")
        base_standby = st.number_input("Standby General (kW)", 0.0, 50.0, 5.0, help="~2-5W por m2 suele ser normal")
        base_mri = st.number_input("Crio-compresores (MRI) (kW)", 0.0, 100.0, 0.0, help="Si hay Resonancia Magnética, consume ~6-8kW siempre")

        st.subheader("💡 3. Iluminación (Naranja)")
        light_tech = st.radio("Tecnología", ["Fluorescente", "LED"], index=0)
        light_dens = st.number_input("Densidad (W/m²)", 5, 25, 12, help="Oficina vieja: 15-20, LED nueva: 6-8")
        light_sim = st.slider("Factor Simultaneidad Luz", 0.5, 1.0, 0.8, help="No todas las luces están encendidas a la vez")
        light_min = st.slider("% Luz Seguridad/Limpieza", 0.0, 0.5, 0.1)

        st.subheader("🖥️ 4. Equipos y Fuerza (Amarillo)")
        equip_kw = st.number_input("Potencia Instalada Equipos (kW)", 0.0, 200.0, 20.0, help="Suma de PCs (0.15kW) + Rayos X + Cafeteras")
        equip_sim = st.slider("Simultaneidad Equipos", 0.1, 1.0, 0.6)
        dhw_kw = st.number_input("Termos ACS (kW)", 0.0, 50.0, 2.0, help="Agua caliente sanitaria")

        st.subheader("❄️ 5. Climatización (Azul/Rojo)")
        st.caption("Modelo Físico: (U * A * DeltaT) / COP")
        
        col_h1, col_h2 = st.columns(2)
        hvac_trans = st.number_input("Pérdidas (kW/°C)", 0.1, 100.0, 5.0, help="Qué tan mal aislado está el edificio. Ajustar hasta coincidir con facturas.")
        
        st.markdown("**Verano (Refrigeración)**")
        set_cool = st.number_input("Set Frío (°C)", 20, 28, 24)
        cop_cool = st.number_input("EER/COP Frío", 1.0, 5.0, 3.0)
        
        st.markdown("**Invierno (Calefacción)**")
        set_heat = st.number_input("Set Calor (°C)", 18, 25, 21)
        cop_heat = st.number_input("COP Calor", 1.0, 5.0, 2.5)

    # Empaquetar parámetros
    params = {
        'bldg_type': bldg_type, 'area_m2': area_m2, 'sched_start': s_start, 'sched_end': s_end,
        'base_kw_servers': base_servers, 'base_kw_standby': base_standby, 'base_kw_medical_always_on': base_mri,
        'light_tech': light_tech, 'light_density_w_m2': light_dens, 'light_simultaneity': light_sim, 'light_min_security': light_min,
        'equip_kw_variable': equip_kw, 'equip_simultaneity': equip_sim, 'kw_dhw': dhw_kw,
        'hvac_transmittance': hvac_trans, 'hvac_set_cool': set_cool, 'hvac_cop_cool': cop_cool,
        'hvac_set_heat': set_heat, 'hvac_cop_heat': cop_heat
    }

    # --- CALCULAR MODELO ---
    df_sim = calculate_physics_model(df_avg, params)

    # --- VISUALIZACIÓN ---
    st.subheader(f"Desglose Energético: Día Promedio ({tipo_dia})")

    # KPIs de Error
    real_sum = df_sim['consumo_kwh'].sum()
    sim_sum = df_sim['calc_total'].sum()
    diff_pct = ((sim_sum - real_sum) / real_sum) * 100
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Consumo Real (24h)", f"{real_sum:,.0f} kWh")
    k2.metric("Consumo Simulado", f"{sim_sum:,.0f} kWh", f"{diff_pct:.1f}%")
    
    # Interpretación automática
    if sim_sum < real_sum:
        k3.warning("⚠️ El modelo se queda corto. Posiblemente el aislamiento (Pérdidas kW/°C) es peor o hay más cargas base.")
    else:
        k3.success("✅ El modelo cubre el consumo real.")

    # GRÁFICO STACKED (ÁREAS APILADAS)
    fig = go.Figure()
    
    x = df_sim['hora']
    
    # 1. Base (Gris)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_base'], mode='lines', stackgroup='one', name='Carga Base (Standby/Servers)',
        line=dict(width=0.5, color='darkgray'), fillcolor='lightgray'
    ))
    
    # 2. ACS (Gris oscuro)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_dhw'], mode='lines', stackgroup='one', name='ACS (Agua Caliente)',
        line=dict(width=0.5, color='gray')
    ))

    # 3. Iluminación (Naranja/Amarillo)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_light'], mode='lines', stackgroup='one', name='Iluminación',
        line=dict(width=0.5, color='#F4D03F') # Amarillo
    ))

    # 4. Equipos (Naranja Oscuro)
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_equip'], mode='lines', stackgroup='one', name='Equipos Médicos/PCs',
        line=dict(width=0.5, color='#D35400') # Naranja
    ))

    # 5. Climatización (Azul/Rojo)
    # Mostramos Calefacción o Refrigeración según domine, pero apilados
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_heating'], mode='lines', stackgroup='one', name='Calefacción',
        line=dict(width=0.5, color='#C0392B') # Rojo
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['calc_cooling'], mode='lines', stackgroup='one', name='Refrigeración',
        line=dict(width=0.5, color='#2980B9') # Azul
    ))

    # LÍNEA REAL
    fig.add_trace(go.Scatter(
        x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REALIDAD (Factura)',
        line=dict(color='black', width=3, dash='solid')
    ))

    fig.update_layout(
        title="Simulación Horaria vs Realidad",
        xaxis_title="Hora del día",
        yaxis_title="Potencia (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # TABLA DE RESULTADOS
    with st.expander("Ver Detalle Numérico"):
        st.dataframe(df_sim.round(2))
