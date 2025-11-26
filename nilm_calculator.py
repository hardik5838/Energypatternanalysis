import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. CORE LOGIC: BLOCK vs PEAK ---
def calculate_block_model(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # --- HELPER: TRAPEZOID CURVE ---
    def get_block_load(start, end, max_kw, ramp=1):
        curve = np.zeros(len(hours))
        for i, h in enumerate(hours):
            if start <= h < end:
                curve[i] = 1.0
                # Smooth edges (Ramp Up)
                if h < start + ramp:
                    curve[i] = (h - start + 1) / (ramp + 1)
                # Smooth edges (Ramp Down)
                if h >= end - ramp:
                    curve[i] = (end - h) / (ramp + 1)
        return curve * max_kw

    # 1. BASE (24/7)
    load_base = np.full(len(hours), config['base_kw'])
    
    # 2. LIGHTING
    load_light = get_block_load(config['light_start'], config['light_end'], config['light_kw']) * config['light_factor']
    
    # Security Lighting (when main lights are off)
    is_night = (load_light < (config['light_kw'] * 0.1)) # Threshold for "off"
    load_light[is_night] = config['light_kw'] * config['light_security_pct']

    # 3. PROCESS / MEDICAL
    load_med = get_block_load(config['med_start'], config['med_end'], config['med_kw'])
    
    # Lunch Dip Logic (Optional: Drops load by 40% between 14:00 and 15:00)
    if config.get('has_lunch_dip', False):
        lunch_mask = (hours == 14) # Assuming 2 PM lunch
        load_med[lunch_mask] = load_med[lunch_mask] * 0.6
        load_light[lunch_mask] = load_light[lunch_mask] * 0.8

    # 4. HVAC: VENTILATION
    load_vent = get_block_load(config['vent_start'], config['vent_end'], config['vent_kw'])

    # 5. HVAC: THERMAL
    if config['hvac_mode'] == "Constant (Block)":
        load_thermal = get_block_load(config['therm_start'], config['therm_end'], config['therm_kw'])
    
    else: # "Weather Driven"
        # Calculate Delta T
        delta_cool = np.maximum(0, df['temperatura_c'] - config['set_cool'])
        delta_heat = np.maximum(0, config['set_heat'] - df['temperatura_c'])
        delta_t = delta_cool + delta_heat
        
        # Raw Demand
        raw_demand = delta_t * config['therm_sens']
        
        # Apply Schedule
        sched_factor = get_block_load(config['therm_start'], config['therm_end'], 1.0)
        
        # Cap at Installed Capacity (Plateau effect)
        actual_load = np.minimum(raw_demand, config['therm_kw'])
        
        load_thermal = actual_load * sched_factor

    # --- AGGREGATE ---
    df['sim_base'] = load_base
    df['sim_light'] = load_light
    df['sim_med'] = load_med
    df['sim_vent'] = load_vent
    df['sim_thermal'] = load_thermal
    
    df['sim_total'] = (load_base + load_light + load_med + load_vent + load_thermal)
    
    # Calculate Residuals (Real - Sim)
    # Note: If 'consumo_kwh' exists (Real data), calculate error
    if 'consumo_kwh' in df.columns:
        df['diff_val'] = df['consumo_kwh'] - df['sim_total']
        df['diff_pct'] = (df['diff_val'] / df['consumo_kwh']).replace([np.inf, -np.inf], 0) * 100
    
    return df

# --- 2. UI FUNCTION ---
def show_nilm_page(df_consumo, df_clima):
    st.markdown("## 🎛️ Digital Twin Calibrator")
    st.markdown("Adjust the variables on the left to minimize the **Error Metrics** below.")

    if df_consumo.empty or df_clima.empty:
        st.error("No data provided.")
        return

    # --- A. DATA PREP ---
    df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    
    # Filter Controls (Top Bar)
    c_fil_1, c_fil_2 = st.columns([1, 2])
    with c_fil_1:
        d_type = st.selectbox("📅 Select Profile to Calibrate", 
                              ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"])
    
    mask = df_merged['fecha'].dt.dayofweek < 5 if d_type == "Weekday (Mon-Fri)" else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        st.warning("No data found for this selection.")
        return

    # Average Hourly Profile
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index()
    df_avg.rename(columns={'fecha': 'hora'}, inplace=True)

    # --- B. SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("1. Building Constants")
        
        with st.expander("🏗️ Base & General", expanded=True):
            base_kw = st.slider("Base Load (Standby) [kW]", 0.0, 200.0, 25.0, 0.5)
            
        with st.expander("💡 Lighting", expanded=False):
            light_kw = st.number_input("Installed Light [kW]", 0.0, 500.0, 20.0)
            light_fac = st.slider("Simultaneity Factor", 0.1, 1.0, 0.85)
            light_sec = st.slider("Security Light %", 0.0, 0.5, 0.1)
            l_s, l_e = st.slider("Light Schedule", 0, 24, (7, 21))

        with st.expander("⚙️ Process / Medical", expanded=False):
            med_kw = st.number_input("Process Capacity [kW]", 0.0, 500.0, 15.0)
            has_lunch = st.checkbox("Apply Lunch Dip (14:00)?", value=True)
            m_s, m_e = st.slider("Process Schedule", 0, 24, (8, 18))

        with st.expander("❄️ HVAC System", expanded=True):
            st.caption("Ventilation (Fans)")
            vent_kw = st.number_input("Fan Power [kW]", 0.0, 200.0, 30.0)
            v_s, v_e = st.slider("Fan Schedule", 0, 24, (6, 20))
            
            st.divider()
            st.caption("Thermal Generation")
            hvac_mode = st.radio("Mode", ["Constant (Block)", "Weather Driven"], horizontal=True)
            therm_kw = st.number_input("Chiller/Boiler Cap [kW]", 0.0, 500.0, 45.0)
            t_s, t_e = st.slider("Thermal Schedule", 0, 24, (8, 19))
            
            therm_sens, set_c, set_h = 5.0, 24, 20
            if hvac_mode == "Weather Driven":
                therm_sens = st.slider("Sensitivity (kW/°C)", 0.1, 20.0, 5.0)
                set_c = st.number_input("Set Cooling [°C]", 18, 30, 24)
                set_h = st.number_input("Set Heating [°C]", 15, 25, 20)

    # --- C. CALCULATION ---
    config = {
        'base_kw': base_kw,
        'light_kw': light_kw, 'light_start': l_s, 'light_end': l_e, 'light_factor': light_fac, 'light_security_pct': light_sec,
        'med_kw': med_kw, 'med_start': m_s, 'med_end': m_e, 'has_lunch_dip': has_lunch,
        'vent_kw': vent_kw, 'vent_start': v_s, 'vent_end': v_e,
        'therm_kw': therm_kw, 'therm_start': t_s, 'therm_end': t_e, 'hvac_mode': hvac_mode,
        'therm_sens': therm_sens, 'set_cool': set_c, 'set_heat': set_h
    }

    df_sim = calculate_block_model(df_avg, config)

    # --- D. KEY METRICS (Calibration Score) ---
    real_total = df_sim['consumo_kwh'].sum()
    sim_total = df_sim['sim_total'].sum()
    error_kwh = sim_total - real_total
    
    # RMSE Calculation (Root Mean Square Error) - The standard for "Goodness of Fit"
    mse = ((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) - careful with zeros
    mape = (np.abs((df_sim['consumo_kwh'] - df_sim['sim_total']) / df_sim['consumo_kwh'])).mean() * 100

    # Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Real Energy", f"{real_total:,.0f} kWh")
    m2.metric("Simulated Energy", f"{sim_total:,.0f} kWh", delta=f"{error_kwh:,.0f} kWh")
    m3.metric("RMSE (Fit Quality)", f"{rmse:.2f}", help="Lower is better. Ideally < 5.0")
    m4.metric("MAPE (Avg Error)", f"{mape:.1f}%", help="Average percentage error per hour")

    # --- E. TABS FOR ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Main Profile", "📉 Error Analysis", "🔢 Data Table"])

    # TAB 1: Main Stacked Plot
    with tab1:
        fig = go.Figure()
        x = df_sim['hora']
        
        # Stacked Layers
        layers = [
            ('sim_base', 'Base Load', 'gray'),
            ('sim_vent', 'Ventilation', '#2980b9'),
            ('sim_thermal', 'Thermal', '#c0392b'),
            ('sim_light', 'Lighting', '#f1c40f'),
            ('sim_med', 'Process', '#e67e22')
        ]
        
        for col, name, color in layers:
            fig.add_trace(go.Scatter(
                x=x, y=df_sim[col], stackgroup='one', name=name, 
                line=dict(width=0, color=color),
                hovertemplate='%{y:.1f} kW'
            ))

        # Real Data Line
        fig.add_trace(go.Scatter(
            x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER',
            line=dict(color='black', width=3, dash='solid')
        ))

        fig.update_layout(height=450, hovermode="x unified", title=f"Daily Load Profile ({d_type})",
                          yaxis_title="Power (kW)", xaxis_title="Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Residuals (The Diagnostic Tool)
    with tab2:
        c1, c2 = st.columns(2)
        
        # Chart 1: The Residual Bar Chart
        with c1:
            st.subheader("Where is the model wrong?")
            # Color logic: Red if Sim < Real (Underestimated), Blue if Sim > Real (Overestimated)
            df_sim['color'] = np.where(df_sim['diff_val'] > 0, '#ef5350', '#42a5f5') 
            
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=df_sim['hora'], y=df_sim['diff_val'],
                marker_color=df_sim['color'],
                name='Difference'
            ))
            fig_res.update_layout(
                title="Residuals (Real - Sim)",
                yaxis_title="Difference (kW)",
                xaxis_title="Hour",
                height=400
            )
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("Positive (Red) = Real is higher than Model (Increase variables). Negative (Blue) = Model is higher (Decrease variables).")

        # Chart 2: Correlation Scatter
        with c2:
            st.subheader("Correlation Fit")
            fig_scat = px.scatter(df_sim, x='consumo_kwh', y='sim_total', hover_data=['hora'],
                                  trendline="ols", trendline_color_override="red")
            fig_scat.add_shape(type="line", x0=0, y0=0, x1=df_sim['consumo_kwh'].max(), y1=df_sim['consumo_kwh'].max(),
                               line=dict(color="green", dash="dash"))
            fig_scat.update_layout(
                title="Real (X) vs Simulated (Y)",
                xaxis_title="Real kW", yaxis_title="Simulated kW",
                height=400
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.caption("Green dashed line is perfect match. Red line is your trend.")

    # TAB 3: Data Grid
    with tab3:
        st.dataframe(df_sim.style.background_gradient(subset=['diff_val'], cmap='coolwarm'), use_container_width=True)
        
        # Download Button
        csv = df_sim.to_csv(index=False).encode('utf-8')
        st.download_button("Download Simulation Data (CSV)", csv, "sim_model.csv", "text/csv")

# --- MAIN EXECUTION (FOR TESTING) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    # CREATE DUMMY DATA TO SHOW FUNCTIONALITY
    dates = pd.date_range(start='2024-01-01', periods=24*7, freq='H')
    
    # Fake Real Consumption: Base 20 + Bell Curve Day + Random Noise
    base = 20
    day_pattern = np.array([0,0,0,0,0,0, 5,15,35,45,50,55,50,45,40,45,40,30,15,5,0,0,0,0])
    pattern_repeated = np.tile(day_pattern, 7)
    noise = np.random.normal(0, 5, len(dates))
    consumption = base + pattern_repeated + noise
    
    # Fake Weather
    temps = np.sin(np.linspace(0, 14*np.pi, len(dates))) * 10 + 20 # varies 10 to 30
    
    df_consumo = pd.DataFrame({'fecha': dates, 'consumo_kwh': consumption})
    df_clima = pd.DataFrame({'fecha': dates, 'temperatura_c': temps})
    
    show_nilm_page(df_consumo, df_clima)
    
