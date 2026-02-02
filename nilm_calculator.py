import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

# ==========================================
# 1. LOGIC ENGINE (Helper Functions)
# ==========================================

def estimate_medical_office_metrics(total_annual_kwh):
    # Benchmark for Mixed Medical/Office (kWh/m2/year)
    # Medical facilities are energy-dense; using 250 kWh/m2 as a hybrid baseline
    eui_benchmark = 200 
    estimated_area = total_annual_kwh / eui_benchmark
    ua_value = estimated_area * 1
    
    guesses = {
        "area": estimated_area,
        "light_kw": (estimated_area * 15) / 1000,   # 15 W/m2
        "hvac_therm_kw": (estimated_area * 120) / 1000, # 120 W/m2 Thermal
        "vent_kw": (estimated_area * 7) / 1000,     # 5 W/m2
        "ua": ua_value
    }
    return guesses

def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, nominal_pct=1.0, residual_pct=0.0, dips=None):
    """
    Generates a load curve with ramps, hourly dips, nominal scaling, and residual consumption.
    """
    if dips is None: dips = []
    
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        activity_val = 0.0
        # Basic Window Logic handling overnight schedules
        is_active = False
        if start <= end:
            if start <= h < end: is_active = True
        else: # Overnight
            if h >= start or h < end: is_active = True

        if is_active:
            activity_val = 1.0
            
            # Ramp Up
            # Note: Ramps are simplified for cyclic days to avoid complexity at midnight crossover in this version
            if ramp_up > 0:
                time_since_start = (h - start) if h >= start else (h + 24 - start)
                if time_since_start < ramp_up:
                    activity_val = time_since_start / ramp_up
            
            # Ramp Down
            if ramp_down > 0:
                time_until_end = (end - h) if end > h else (end + 24 - h)
                if time_until_end < ramp_down:
                    activity_val = time_until_end / ramp_down

            # Apply Dips
            for dip in dips:
                if int(h) == int(dip['hour']):
                    factor = 1.0 - (dip['percent'] / 100.0)
                    activity_val *= factor
        
        # Clip activity
        activity_val = np.clip(activity_val, 0.0, 1.0)
        
        # Apply Scaling
        val = residual_pct + activity_val * (nominal_pct - residual_pct)
        curve[i] = val * max_kw
        
    return curve

def get_tariff_periods(is_weekend):
    c_cheap = "rgba(46, 204, 113, 0.15)"
    c_med = "rgba(241, 196, 15, 0.15)"
    c_exp = "rgba(231, 76, 60, 0.15)"

    if is_weekend:
        return [(0, 24, c_cheap, "Weekend")] 
    
    return [
        (0, 8, c_cheap, "P3"), (8, 10, c_med, "P2"), (10, 14, c_exp, "P1"),
        (14, 18, c_med, "P2"), (18, 22, c_exp, "P1"), (22, 24, c_med, "P2")
    ]

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
        config['vent_kw'], config.get('vent_ru', 0), config.get('vent_rd', 0),
        config.get('vent_nom', 1.0), config.get('vent_res', 0.0)
    )
    
    # 3. Lighting
    df['sim_light'] = generate_load_curve(
        hours, config['light_s'], config['light_e'], 
        config['light_kw'], config.get('light_ru', 0.5), config.get('light_rd', 0.5),
        config.get('light_nom', 1.0), config.get('light_res', 0.0)
    )

    # 4. HVAC (Thermodynamic Model)
    # Using Heating logic (Load increases as Temp drops below setpoint)
    # For cooling dominance, this logic flips. Simplified here for universal use.
    delta_T = np.abs(config['hvac_setpoint'] - df['temperatura_c'])
    q_transmission = (config['hvac_ua'] / 1000.0) * delta_T
    total_thermal_load = q_transmission + config['hvac_q_int'] + config['hvac_q_sol'] + config['hvac_q_vent']
    
    hvac_avail = generate_load_curve(hours, config['hvac_s'], config['hvac_e'], 1.0, 
                                     config['hvac_ru'], config['hvac_rd'], 
                                     1.0, config['hvac_res'])
    
    hvac_electrical_raw = (total_thermal_load / max(0.1, config['hvac_cop']))
    df['sim_therm'] = np.clip(hvac_electrical_raw, 0, config['hvac_cap_max']) * hvac_avail

    # 5. Occupancy
    df['sim_occ'] = generate_load_curve(
        hours, config['occ_s'], config['occ_e'], config['occ_kw'],
        config.get('occ_ru', 1), config.get('occ_rd', 1),
        config.get('occ_nom', 1.0), config.get('occ_res', 0.0),
        config.get('occ_dips', [])
    )


    # Total Sum
    cols_to_sum = ['sim_base', 'sim_vent', 'sim_light', 'sim_therm', 'sim_occ']
    df['sim_total'] = df[cols_to_sum].sum(axis=1)
    
    if 'consumo_kwh' in df.columns:
        df['error_kw'] = df['sim_total'] - df['consumo_kwh']
        
    return df

# ==========================================
# 2. AUTO-CALIBRATION & UI HELPERS
# ==========================================

def run_optimizer(df_avg, m):
    max_load = df_avg['consumo_kwh'].max()
    
    # We define search bounds around the medical benchmarks (+/- 30%)
    # This guides the AI to stay within a physical reality for medical buildings
    bounds = [
        (0, max_load * 0.4),                  # 0: Base Load kW
        (m['vent_kw']*0.7, m['vent_kw']*1.3),   # 1: Vent kW
        (5, 9),                               # 2: Vent Start
        (m['light_kw']*0.7, m['light_kw']*1.3), # 3: Light kW
        (17, 22),                             # 4: Light End
        (m['hvac_therm_kw']/4, m['hvac_therm_kw']/2), # 5: HVAC Elec Max
        (4, 10),                              # 6: HVAC Start
        (16, 23),                             # 7: HVAC End
        (m['ua']*0.8, m['ua']*1.2)            # 8: UA Heat Loss
    ]
    
    def obj(p, d):
        # Maps the AI's guesses to a config dictionary
        c = {
            'base_kw': p[0], 'base_ru': 0.0, 'base_rd': 0.0, 'vent_kw': p[1], 'vent_s': int(p[2]), 'vent_e': 19, 
            'light_kw': p[3], 'light_s': 7, 'light_e': int(p[4]), 
            'hvac_cap_max': p[5], 'hvac_s': int(p[6]), 'hvac_e': int(p[7]), 
            'hvac_ua': p[8], 'hvac_setpoint': 22, 'hvac_cop': 3.0, 
            'hvac_q_int': 2.0, 'hvac_q_sol': 1.0, 'hvac_q_vent': 1.0, 
            'hvac_res': 0.05, 'hvac_ru': 1.0, 'hvac_rd': 1.0, 
            'occ_kw': 5.0, 'occ_s': 8, 'occ_e': 18
        }
        return np.sqrt(mean_squared_error(d['consumo_kwh'], run_simulation(d, c)['sim_total']))

    result = differential_evolution(obj, bounds, args=(df_avg,), maxiter=15, popsize=10, seed=42)
    return result.x

def render_standard_controls(prefix, label, default_kw, default_sched):
    st.subheader(f"{label} Settings")
    k_kw, k_sched = f"{prefix}_kw", f"{prefix}_sched"
    if k_kw not in st.session_state: st.session_state[k_kw] = float(default_kw)
    if k_sched not in st.session_state: st.session_state[k_sched] = default_sched
    kw = st.number_input(f"{label} Max [kW]", 0.0, 500000.0, key=k_kw)
    s, e = st.slider(f"{label} Schedule", 0, 24, key=k_sched)
    ru = st.number_input(f"Ramp Up (h)", 0.0, 10.0, 0.5, key=f"{prefix}_ru")
    rd = st.number_input(f"Ramp Down (h)", 0.0, 10.0, 0.5, key=f"{prefix}_rd")
    nom = st.slider(f"{label} Nominal %", 0, 100, 100, key=f"{prefix}_nom") / 100.0
    res_val = (st.number_input(f"Residual %", 0.0, 100.0, 5.0, key=f"{prefix}_res_val") / 100.0) if st.checkbox(f"Residual?", key=f"{prefix}_res_on") else 0.0
    return kw, s, e, ru, rd, nom, res_val

# ==========================================
# 3. UI HELPERS
# ==========================================

def render_dips_ui(key_prefix, max_dips=4):
    dips = []
    with st.expander(f"Dips Configuration"):
        num_dips = st.number_input(f"Count", 0, max_dips, 0, key=f"n_dips_{key_prefix}")
        for i in range(num_dips):
            c1, c2 = st.columns(2)
            h = c1.number_input(f"Hour", 0, 23, 13, key=f"h_{key_prefix}_{i}")
            p = c2.number_input(f"Drop %", 0, 100, 50, key=f"p_{key_prefix}_{i}")
            dips.append({'hour': h, 'percent': p})
    return dips

def render_standard_controls(prefix, label, default_kw, default_sched):
    st.subheader(f"{label} Settings")
    
    # We use session state for the main values to allow Auto-Calibration to overwrite them
    k_kw = f"{prefix}_kw"
    k_sched = f"{prefix}_sched"
    
    # Initialize if missing
    if k_kw not in st.session_state: st.session_state[k_kw] = float(default_kw)
    if k_sched not in st.session_state: st.session_state[k_sched] = default_sched

    # Render with key=... so Streamlit binds them to state
    kw = st.number_input(f"{label} Max [kW]", 0.0, 500000.0, key=k_kw)
    s, e = st.slider(f"{label} Schedule", 0, 24, key=k_sched)
    
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
# 4. MAIN PAGE
# ==========================================

def show_nilm_page(df_consumo, df_clima):
    st.title("Advanced Energy Digital Twin (Full Control)")
    
    # --- DATA PREP ---
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    if not df_clima.empty:
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 22.0

# --- SIDEBAR ---
    with st.sidebar:
        st.header("1. Global Filters")
        
        # Calculate Medical Intelligence Metrics
        total_annual_kwh = df_merged['consumo_kwh'].sum() * (8760 / len(df_merged))
        m = estimate_medical_office_metrics(total_annual_kwh)
        st.info(f"Est. Size: {m['area']:,.0f} m² | UA: {m['ua']:.1f} W/K")
        
        # Apply Benchmarks Button
        if st.button("Apply Medical Benchmarks", type="primary", use_container_width=True):
            st.session_state['light_kw'] = m['light_kw']
            st.session_state['vent_kw'] = m['vent_kw']
            st.session_state['hvac_ua'] = m['ua']
            st.session_state['hvac_cap_max'] = m['hvac_therm_kw'] / 3.0
            st.rerun()
            
        selected_months = st.multiselect("Select Months", list(range(1, 13)), default=list(range(1, 13)))
        day_type = st.radio("Profile Type", ["Workday", "Weekend"], horizontal=True)
        is_weekday = (day_type == "Workday")
        
        st.divider()
        
        # AI Auto-Calibration Button
        if st.button("AI Auto-Calibrate (Medical)", type="primary", use_container_width=True):
            mask_month = df_merged['fecha'].dt.month.isin(selected_months)
            mask_day = (df_merged['fecha'].dt.dayofweek < 5) if is_weekday else (df_merged['fecha'].dt.dayofweek >= 5)
            df_calib = df_merged[mask_month & mask_day].groupby(df_merged['fecha'].dt.hour).agg({
                'consumo_kwh':'mean', 'temperatura_c':'mean'
            }).reset_index().rename(columns={'fecha': 'hora'})
            
            with st.spinner("AI is fitting curves..."):
                opt = run_optimizer(df_calib, m)
                st.session_state['base_kw'] = float(opt[0])
                st.session_state['base_ru'] = 0.0
                st.session_state['vent_kw'] = float(opt[1])
                st.session_state['vent_sched'] = (int(opt[2]), 19)
                st.session_state['light_kw'] = float(opt[3])
                st.session_state['light_sched'] = (7, int(opt[4]))
                # Lighting: 30% standby
                st.session_state['light_res_on'] = True
                st.session_state['light_res_val'] = 30.0
                # Ventilation: 5% standby
                st.session_state['vent_res_on'] = True
                st.session_state['vent_res_val'] = 5.0
                # HVAC: 5% standby
                st.session_state['hvac_res_on_unique'] = True
                st.session_state['hvac_res_val_unique'] = 5.0
                st.session_state['hvac_cap_max'] = float(opt[5])
                st.session_state['hvac_win'] = (int(opt[6]), int(opt[7]))
                st.session_state['hvac_ua'] = float(opt[8])
            st.success("AI Calibration complete!")
            st.rerun()

        st.header("2. Infrastructure Controls")
        
        # IMPORTANT: Assigning function results to variables (b_kw, v_kw, etc.)
        b_kw, b_s, b_e, b_ru, b_rd, b_nom, b_res = render_standard_controls("base", "Base Load", 20.0, (0, 24))
        st.divider()
        v_kw, v_s, v_e, v_ru, v_rd, v_nom, v_res = render_standard_controls("vent", "Ventilation", 30.0, (6, 20))
        st.divider()
        l_kw, l_s, l_e, l_ru, l_rd, l_nom, l_res = render_standard_controls("light", "Lighting", 15.0, (7, 21))
        
        st.divider()
        st.subheader("HVAC Parameters")
        h_win = st.slider("Operation Window", 0, 24, value=(8, 19), key='hvac_win')
        h_s, h_e = h_win[0], h_win[1] 
        
        col1, col2 = st.columns(2)
        h_ua = col1.number_input("U × A (W/K)", 0.0, 100000.0, key='hvac_ua')
        h_cop = col2.number_input("COP", 0.5, 6.0, 3.0, key='hvac_cop')
        h_set = st.slider("Setpoint [°C]", 16, 30, 22, key='hvac_set')
        h_cap_max = st.number_input("Max Electrical Capacity [kW]", 0.0, 10000.0, key='hvac_cap_max')
        
        with st.expander("Gains & Ramps"):
            h_q_int = st.number_input("Internal Gains [kW]", 0.0, 50.0, 2.0, key='hvac_qi')
            h_q_sol = st.number_input("Solar Gains [kW]", 0.0, 50.0, 1.5, key='hvac_qs')
            h_q_vent = st.number_input("Ventilation Load [kW]", 0.0, 50.0, 1.0, key='hvac_qv')
            h_ru = st.number_input("HVAC Ramp Up", 0.0, 5.0, 1.0, key="hvac_ru")
            h_rd = st.number_input("HVAC Ramp Down", 0.0, 5.0, 1.0, key="hvac_rd")
        
        h_res_on = st.checkbox("HVAC Residual Consumption?", value=True, key="hvac_res_on_unique")
        h_res = (st.number_input("Res %", 0.0, 100.0, 5.0, key="hvac_res_val_unique") / 100.0) if h_res_on else 0.0
        
        st.divider()
        st.header("3. Variable Processes")
        o_kw, o_s, o_e, o_ru, o_rd, o_nom, o_res = render_standard_controls("occ", "Occupancy", 10.0, (8, 18))
        occ_dips = render_dips_ui("occ")
        
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
        'hvac_s': h_s, 'hvac_e': h_e,
        'hvac_ua': h_ua,
        'hvac_cop': h_cop,
        'hvac_setpoint': h_set,
        'hvac_q_int': h_q_int,
        'hvac_q_sol': h_q_sol,
        'hvac_q_vent': h_q_vent,
        'hvac_cap_max': h_cap_max,
        'hvac_ru': h_ru, 'hvac_rd': h_rd, 
        'hvac_res': h_res,
        'occ_kw': o_kw, 'occ_s': o_s, 'occ_e': o_e, 'occ_ru': o_ru, 'occ_rd': o_rd, 'occ_nom': o_nom, 'occ_res': o_res, 'occ_dips': occ_dips,
    }
    


    
    df_sim = run_simulation(df_avg, config)

    # --- METRICS & PLOTS ---
    st.markdown("###Key Performance Indicators")
    total_real = df_sim['consumo_kwh'].sum()
    total_sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(mean_squared_error(df_sim['consumo_kwh'], df_sim['sim_total']))
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Real Consumption", f"{total_real:,.0f} kWh")
    kpi2.metric("Total Simulated Consumption", f"{total_sim:,.0f} kWh", delta=f"{total_sim - total_real:,.0f} kWh")
    kpi3.metric("Overall Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")
    st.divider()

    st.markdown("###Main Load Profile Analysis")
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
    with c2:
        st.subheader("Hourly Error (kW)")
        st.plotly_chart(px.bar(df_sim, x='hora', y='error_kw', color='error_kw', color_continuous_scale='RdBu_r'), use_container_width=True)

    with st.expander("Show Detailed Data Table"):
        st.dataframe(df_sim.style.format(precision=2), use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="NILM Digital Twin", layout="wide")
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="h")
    np.random.seed(42)
    total_load = 20 + np.where(dates.dayofweek < 5, 50, 10) * np.where((dates.hour > 8) & (dates.hour < 18), 1, 0)
    df_cons = pd.DataFrame({'fecha': dates, 'consumo_kwh': np.abs(total_load + np.random.normal(0, 5, len(dates)))})
    df_clim = pd.DataFrame({'fecha': dates, 'temperatura_c': 15 + 10 * np.sin(np.linspace(0, 6, len(dates)))})
    show_nilm_page(df_cons, df_clim)
