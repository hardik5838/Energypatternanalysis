import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics 
import mean_squared_error

# ==========================================
# 1. LOGIC ENGINE (Helper Functions)
# ==========================================

def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, nominal_pct=1.0, residual_pct=0.0, dips=None):
    """
    Generates a load curve with ramps, hourly dips, nominal scaling, and residual consumption.
    """
    if dips is None: dips = []
    
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        activity_val = 0.0
        # Basic Window
        if start <= h < end:
            activity_val = 1.0
            
            # Ramp Up
            if ramp_up > 0 and h < (start + ramp_up):
                activity_val = (h - start) / ramp_up
            
            # Ramp Down
            if ramp_down > 0 and h >= (end - ramp_down):
                activity_val = (end - h) / ramp_down
            
            # Apply Dips (Percentage Drop relative to current activity)
            for dip in dips:
                if int(h) == int(dip['hour']):
                    factor = 1.0 - (dip['percent'] / 100.0)
                    activity_val *= factor
        
        # Clip activity between 0 and 1
        activity_val = np.clip(activity_val, 0.0, 1.0)
        
        # Apply Scaling: 
        # Low state = residual_pct
        # High state = nominal_pct
        # Value = Low + Activity * (High - Low)
        val = residual_pct + activity_val * (nominal_pct - residual_pct)
        
        curve[i] = val * max_kw
        
    return curve

def get_tariff_periods(is_weekend):
    """
    Returns list of tuples (start, end, color, name).
    Colors made less saturated (approx 20% opacity).
    """
    c_cheap = "rgba(46, 204, 113, 0.15)"   # Faint Green
    c_med = "rgba(241, 196, 15, 0.15)"     # Faint Yellow
    c_exp = "rgba(231, 76, 60, 0.15)"      # Faint Red

    if is_weekend:
        return [(0, 24, c_cheap, "")] 
    
    # Workday Schedule
    periods = [
        (0, 8, c_cheap, ""),
        (8, 9, c_med, ""),
        (9, 14, c_exp, ""),
        (14, 18, c_med, ""),
        (18, 22, c_exp, ""),
        (22, 24, c_med, "")
    ]
    return periods

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. Base Loads
    df['sim_base'] = generate_load_curve(
        hours, 0, 24, config['base_kw'], 
        config.get('base_ru', 0), config.get('base_rd', 0),
        config.get('base_nom', 1.0), config.get('base_res', 1.0)
    )
    
    # 2. Ventilation
    df['sim_vent'] = generate_load_curve(
        hours, config['vent_s'], config['vent_e'], 
        config['vent_kw'], config.get('vent_ru', 0.5), config.get('vent_rd', 0.5),
        config.get('vent_nom', 1.0), config.get('vent_res', 0.0)
    )
    
    # 3. Lighting
    df['sim_light'] = generate_load_curve(
        hours, config['light_s'], config['light_e'], 
        config['light_kw'], config.get('light_ru', 0.5), config.get('light_rd', 0.5),
        config.get('light_nom', 1.0), config.get('light_res', 0.0)
    )

    # 4. HVAC (Split System)
    delta_T_out = np.abs(df['temperatura_c'] - 20.0) 
    hvac_1_raw = (delta_T_out / 20.0) * (config['hvac1_kw'] / max(0.5, config['hvac1_cop'])) * 5.0
    hvac_1_curve = np.clip(hvac_1_raw, 0, config['hvac1_kw'])
    
    hvac_avail = generate_load_curve(hours, config['hvac_s'], config['hvac_e'], 1.0, 
                                     config['hvac_ru'], config['hvac_rd'], 
                                     config['hvac1_nom'], config['hvac1_res'])
    df['sim_hvac_1'] = hvac_1_curve * hvac_avail

    delta_T_in = np.abs(df['temperatura_c'] - config['hvac2_setpoint'])
    envelope_factor = (1.0 - (config['hvac2_eff'] / 100.0))
    hvac_2_raw = delta_T_in * envelope_factor * 2.0 
    hvac_2_curve = np.clip(hvac_2_raw, 0, config['hvac_kw_total'])
    df['sim_hvac_2'] = hvac_2_curve * hvac_avail

    df['sim_therm'] = df['sim_hvac_1'] + df['sim_hvac_2']

    # 5. Occupancy
    df['sim_occ'] = generate_load_curve(
        hours, config['occ_s'], config['occ_e'], config['occ_kw'],
        config.get('occ_ru', 1), config.get('occ_rd', 1),
        config.get('occ_nom', 1.0), config.get('occ_res', 0.0),
        config.get('occ_dips', [])
    )

    # 6. Variable Processes (1, 2, 3)
    for i in range(1, 4):
        p_key = f'proc_{i}'
        if config.get(f'{p_key}_enabled', False):
            df[f'sim_{p_key}'] = generate_load_curve(
                hours, config[f'{p_key}_s'], config[f'{p_key}_e'], config[f'{p_key}_kw'],
                config[f'{p_key}_ru'], config[f'{p_key}_rd'],
                config.get(f'{p_key}_nom', 1.0), config.get(f'{p_key}_res', 0.0),
                config.get(f'{p_key}_dips', [])
            )
        else:
            df[f'sim_{p_key}'] = 0.0

    # Total Sum
    cols_to_sum = ['sim_base', 'sim_vent', 'sim_light', 'sim_therm', 'sim_occ', 'sim_proc_1', 'sim_proc_2', 'sim_proc_3']
    df['sim_total'] = df[cols_to_sum].sum(axis=1)
    
    if 'consumo_kwh' in df.columns:
        df['error_kw'] = df['sim_total'] - df['consumo_kwh']
        
    return df

# ==========================================
# 2. UI HELPERS
# ==========================================

def render_dips_ui(key_prefix, max_dips=4):
    """Helper to render dynamic dips input in sidebar"""
    dips = []
    with st.expander(f"📉 Dips Configuration"):
        num_dips = st.number_input(f"Count", 0, max_dips, 0, key=f"n_dips_{key_prefix}")
        for i in range(num_dips):
            c1, c2 = st.columns(2)
            h = c1.number_input(f"Hour", 0, 23, 13, key=f"h_{key_prefix}_{i}")
            p = c2.number_input(f"Drop %", 0, 100, 50, key=f"p_{key_prefix}_{i}")
            dips.append({'hour': h, 'percent': p})
    return dips

def render_standard_controls(prefix, label, default_kw, default_sched):
    """
    Renders standard controls for ALL items.
    """
    st.subheader(f"{label} Settings")
    kw = st.number_input(f"{label} Max [kW]", 0.0, 10000.0, float(default_kw), key=f"{prefix}_kw")
    s, e = st.slider(f"{label} Schedule", 0, 24, default_sched, key=f"{prefix}_sched")
    
    c1, c2 = st.columns(2)
    ru = c1.number_input(f"Ramp Up (h)", 0.0, 10.0, 0.5, key=f"{prefix}_ru")
    rd = c2.number_input(f"Ramp Down (h)", 0.0, 10.0, 0.5, key=f"{prefix}_rd")
    
    nom = st.slider(f"{label} Nominal Power %", 0, 100, 100, key=f"{prefix}_nom") / 100.0
    
    res_on = st.checkbox(f"Residual Consumption?", key=f"{prefix}_res_on")
    res_val = 0.0
    if res_on:
        res_val = st.number_input(f"Residual % of Max", 0.0, 100.0, 5.0, key=f"{prefix}_res_val") / 100.0
        
    return kw, s, e, ru, rd, nom, res_val

# ==========================================
# 3. MAIN FUNCTION (Callable)
# ==========================================

def show_nilm_page(df_consumo, df_clima):
    """
    Main function to render the NILM Digital Twin page.
    Call this function from your main app.
    """
    st.title("⚡ Advanced Energy Digital Twin")
    
    # --- DATA PREP ---
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    if not df_clima.empty:
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 22.0

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("1. Global Filters")
        
        # Month Filter
        all_months = list(range(1, 13))
        selected_months = st.multiselect("Select Months", all_months, default=all_months)
        
        # Day Type
        day_type = st.radio("Profile Type", ["Workday", "Weekend"], horizontal=True)
        is_weekday = (day_type == "Workday")
        
        st.divider()
        st.header("2. Infrastructure")
        
        # Base Load
        b_kw, b_s, b_e, b_ru, b_rd, b_nom, b_res = render_standard_controls("base", "Base Load", 20.0, (0, 24))

        st.divider()
        # Ventilation
        v_kw, v_s, v_e, v_ru, v_rd, v_nom, v_res = render_standard_controls("vent", "Ventilation", 30.0, (6, 20))

        st.divider()
        # Lighting
        l_kw, l_s, l_e, l_ru, l_rd, l_nom, l_res = render_standard_controls("light", "Lighting", 15.0, (7, 21))
        
        st.divider()
        # HVAC SPLIT
        st.subheader("❄️ HVAC Configuration")
        h_s, h_e = st.slider("HVAC Operation Window", 0, 24, (8, 19))
        c1, c2 = st.columns(2)
        h_ru = c1.number_input("HVAC Ramp Up", 0.0, 5.0, 1.0)
        h_rd = c2.number_input("HVAC Ramp Down", 0.0, 5.0, 1.0)
        
        st.markdown("**HVAC 1.1: Vent/Climatization**")
        h1_kw = st.number_input("HVAC 1.1 Peak [kW]", 0.0, 5000.0, 20.0)
        h1_cop = st.slider("COP (Efficiency)", 1.0, 6.0, 3.0)
        h1_nom = st.slider("HVAC 1.1 Nominal %", 0, 100, 100) / 100.0
        
        st.markdown("**HVAC 1.2: Building Envelope**")
        h2_eff = st.slider("Envelope Efficiency %", 0, 100, 50, help="Higher means better insulation, less load")
        h2_set = st.slider("Indoor Setpoint [°C]", 16, 30, 24)
        
        h_res_on = st.checkbox("HVAC Residual?", value=False)
        h_res = (st.number_input("HVAC Residual %", 0.0, 100.0, 5.0) / 100.0) if h_res_on else 0.0

        st.divider()
        st.header("3. Variable Processes")
        
        # Occupancy
        o_kw, o_s, o_e, o_ru, o_rd, o_nom, o_res = render_standard_controls("occ", "Occupancy", 10.0, (8, 18))
        occ_dips = render_dips_ui("occ")
        
        # Generic Processes
        proc_configs = {}
        for i in range(1, 4):
            with st.expander(f"⚙️ Custom Process {i}"):
                enabled = st.checkbox(f"Enable Process {i}", value=(i==1))
                name = st.text_input(f"Name {i}", value=f"Process {i}")
                color = st.color_picker(f"Color {i}", value="#9b59b6")
                
                p_kw = st.number_input(f"Max kW {i}", 0.0, 5000.0, 50.0, key=f"p_kw_{i}")
                p_s, p_e = st.slider(f"Schedule {i}", 0, 24, (9, 17), key=f"p_sch_{i}")
                c1, c2 = st.columns(2)
                p_ru = c1.number_input(f"Ramp Up {i}", 0.0, 5.0, 1.0, key=f"p_ru_{i}")
                p_rd = c2.number_input(f"Ramp Down {i}", 0.0, 5.0, 1.0, key=f"p_rd_{i}")
                p_nom = st.slider(f"Nominal % {i}", 0, 100, 100, key=f"p_nom_{i}") / 100.0
                
                res_on_p = st.checkbox(f"Residual {i}?", key=f"p_res_on_{i}")
                p_res = (st.number_input(f"Res % {i}", 0.0, 100.0, 5.0, key=f"p_res_{i}") / 100.0) if res_on_p else 0.0
                
                p_dips = render_dips_ui(f"proc_{i}")
                
                proc_configs.update({
                    f'proc_{i}_enabled': enabled,
                    f'proc_{i}_name': name,
                    f'proc_{i}_color': color,
                    f'proc_{i}_kw': p_kw,
                    f'proc_{i}_s': p_s, f'proc_{i}_e': p_e,
                    f'proc_{i}_ru': p_ru, f'proc_{i}_rd': p_rd,
                    f'proc_{i}_nom': p_nom, f'proc_{i}_res': p_res,
                    f'proc_{i}_dips': p_dips
                })

    # --- PROCESSING ---
    mask_month = df_merged['fecha'].dt.month.isin(selected_months)
    mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask_month & mask_day].copy()

    if df_filtered.empty:
        st.warning("No data for selected filters.")
        return

    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': b_kw, 'base_ru': b_ru, 'base_rd': b_rd, 'base_nom': b_nom, 'base_res': b_res,
        'vent_kw': v_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd, 'vent_nom': v_nom, 'vent_res': v_res,
        'light_kw': l_kw, 'light_s': l_s, 'light_e': l_e, 'light_ru': l_ru, 'light_rd': l_rd, 'light_nom': l_nom, 'light_res': l_res,
        
        'hvac_s': h_s, 'hvac_e': h_e, 'hvac_ru': h_ru, 'hvac_rd': h_rd,
        'hvac_kw_total': h1_kw + 50,
        'hvac1_kw': h1_kw, 'hvac1_cop': h1_cop, 'hvac1_nom': h1_nom, 'hvac1_res': h_res,
        'hvac2_eff': h2_eff, 'hvac2_setpoint': h2_set,
        
        'occ_kw': o_kw, 'occ_s': o_s, 'occ_e': o_e, 'occ_ru': o_ru, 'occ_rd': o_rd, 'occ_nom': o_nom, 'occ_res': o_res, 'occ_dips': occ_dips,
    }
    config.update(proc_configs)

    df_sim = run_simulation(df_avg, config)

    # --- TOP METRICS ---
    st.markdown("### 📊 Key Performance Indicators")
    total_real = df_sim['consumo_kwh'].sum()
    total_sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(mean_squared_error(df_sim['consumo_kwh'], df_sim['sim_total']))
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Real Consumption", f"{total_real:,.0f} kWh")
    kpi2.metric("Total Simulated Consumption", f"{total_sim:,.0f} kWh", delta=f"{total_sim - total_real:,.0f} kWh")
    kpi3.metric("Overall Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")
    st.divider()

    # --- PLOTS ---
    st.markdown("### 📈 Main Load Profile Analysis")
    fig1 = go.Figure()
    
    tariff_periods = get_tariff_periods(not is_weekday)
    for start, end, color, name in tariff_periods:
        fig1.add_vrect(x0=start, x1=end, fillcolor=color, opacity=1, layer="below", line_width=0)

    layers = [
        ('sim_base', 'Base Load', '#7f8c8d'),
        ('sim_vent', 'Ventilation', '#3498db'),
        ('sim_light', 'Lighting', '#f1c40f'),
        ('sim_therm', 'HVAC (Total)', '#e74c3c'),
        ('sim_occ', 'Occupancy', '#e67e22')
    ]
    for i in range(1, 4):
        if config[f'proc_{i}_enabled']:
            layers.append((f'sim_proc_{i}', config[f'proc_{i}_name'], config[f'proc_{i}_color']))

    for col, name, color in layers:
        fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, mode='none', fillcolor=color))

    fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL METER', line=dict(color='black', width=4)))

    fig1.update_layout(height=600, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(title="Hour of Day", dtick=1), yaxis=dict(title="Power (kW)"), legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧩 Load Composition")
        pie_cols = [l[0] for l in layers]
        pie_names = [l[1] for l in layers]
        values = df_sim[pie_cols].sum()
        st.plotly_chart(px.pie(values=values, names=pie_names, hole=0.4), use_container_width=True)
        
        st.subheader("📉 Correlation Check")
        st.plotly_chart(px.scatter(df_sim, x='consumo_kwh', y='sim_total', trendline="ols", labels={'consumo_kwh': 'Real', 'sim_total': 'Simulated'}), use_container_width=True)

    with c2:
        st.subheader("⚠️ Hourly Error (kW)")
        st.plotly_chart(px.bar(df_sim, x='hora', y='error_kw', color='error_kw', color_continuous_scale='RdBu_r'), use_container_width=True)
        
        st.subheader("🔋 Cumulative Energy (kWh)")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Acc.", fill='tozeroy'))
        fig_cum.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Acc.", line=dict(dash='dash')))
        st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()
    with st.expander("Show Detailed Data Table"):
        st.dataframe(df_sim.style.format(precision=2), use_container_width=True)

# ==========================================
# 4. ENTRY POINT (For Testing)
# ==========================================
if __name__ == "__main__":
    # This config only runs if executed directly, NOT when imported
    st.set_page_config(page_title="NILM Digital Twin", layout="wide")
    
    # Dummy Data Generation
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="h")
    np.random.seed(42)
    base_load = 20 + np.random.normal(0, 2, len(dates))
    work_load = np.where(dates.dayofweek < 5, 50, 10) * np.where((dates.hour > 8) & (dates.hour < 18), 1, 0)
    total_load = base_load + work_load + np.random.normal(0, 5, len(dates))
    
    df_cons = pd.DataFrame({'fecha': dates, 'consumo_kwh': np.abs(total_load)})
    df_clim = pd.DataFrame({'fecha': dates, 'temperatura_c': 15 + 10 * np.sin(np.linspace(0, 3.14 * 2 * 365, len(dates)))})
    
    # Execute the function
    show_nilm_page(df_cons, df_clim)
