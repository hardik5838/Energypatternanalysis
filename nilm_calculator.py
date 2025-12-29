import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import datetime

# Configuration for Page Layout
st.set_page_config(page_title="NILM Digital Twin", layout="wide")

# ==========================================
# 1. LOGIC ENGINE
# ==========================================

def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, dips=None):
    """
    Generates a load curve with ramps and specific hourly dips.
    """
    if dips is None: dips = []
    
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        val = 0.0
        # Basic Window
        if start <= h < end:
            val = 1.0
            
            # Ramp Up
            if ramp_up > 0 and h < (start + ramp_up):
                val = (h - start) / ramp_up
            
            # Ramp Down
            if ramp_down > 0 and h >= (end - ramp_down):
                val = (end - h) / ramp_down
            
            # Apply Dips (Percentage Drop)
            # dip['hour'] is the hour index, dip['percent'] is how much to drop (e.g. 0.3 for 30% drop)
            for dip in dips:
                if int(h) == int(dip['hour']):
                    # If dip is 40%, we multiply by 0.6
                    factor = 1.0 - (dip['percent'] / 100.0)
                    val *= factor
                    
        curve[i] = np.clip(val, 0.0, 1.0) * max_kw
        
    return curve

def get_tariff_periods(is_weekend):
    """
    Returns list of tuples (start, end, color, name) based on user definition.
    Cheap: 0-8 (Weekends all day)
    Medium: 8-9, 14-18, 22-24
    Expensive: 9-14, 18-22
    """
    # Colors with transparency
    c_cheap = "rgba(46, 204, 113, 0.2)"   # Green
    c_med = "rgba(241, 196, 15, 0.2)"     # Yellow
    c_exp = "rgba(231, 76, 60, 0.2)"      # Red

    if is_weekend:
        return [(0, 24, c_cheap, "Cheap")]
    
    # Workday Schedule
    periods = [
        (0, 8, c_cheap, "Cheap"),
        (8, 9, c_med, "Medium"),
        (9, 14, c_exp, "Expensive"),
        (14, 18, c_med, "Medium"),
        (18, 22, c_exp, "Expensive"),
        (22, 24, c_med, "Medium")
    ]
    return periods

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. Base Loads
    df['sim_base'] = np.full(len(hours), config.get('base_kw', 0))
    
    # 2. Ventilation
    df['sim_vent'] = generate_load_curve(
        hours, config['vent_s'], config['vent_e'], 
        config['vent_kw'], config.get('vent_ru', 0.5), config.get('vent_rd', 0.5)
    )
    
    # 3. Lighting (with smart night logic)
    raw_light = generate_load_curve(
        hours, config['light_s'], config['light_e'], 
        config['light_kw'], config.get('light_ru', 0.5), config.get('light_rd', 0.5)
    ) * config.get('light_fac', 1.0)
    
    # Minimum night lighting logic
    min_light = config['light_kw'] * config.get('light_sec', 0.1)
    df['sim_light'] = np.maximum(raw_light, min_light)

    # 4. HVAC
    if config.get('hvac_mode') == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        # Simple Degree-Day Logic
        delta = (np.maximum(0, df['temperatura_c'] - config.get('set_c', 24)) + 
                 np.maximum(0, config.get('set_h', 20) - df['temperatura_c']))
        raw = delta * config.get('therm_sens', 5.0)
        sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
        df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched

    # 5. Occupancy
    df['sim_occ'] = generate_load_curve(
        hours, config['occ_s'], config['occ_e'], config['occ_kw'],
        config.get('occ_ru', 1), config.get('occ_rd', 1), config.get('occ_dips', [])
    )

    # 6. Variable Processes (1, 2, 3)
    for i in range(1, 4):
        p_key = f'proc_{i}'
        if config.get(f'{p_key}_enabled', False):
            df[f'sim_{p_key}'] = generate_load_curve(
                hours, config[f'{p_key}_s'], config[f'{p_key}_e'], config[f'{p_key}_kw'],
                config[f'{p_key}_ru'], config[f'{p_key}_rd'], config.get(f'{p_key}_dips', [])
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
    with st.expander(f"📉 Dips Configuration ({key_prefix})"):
        num_dips = st.number_input(f"Count ({key_prefix})", 0, max_dips, 0, key=f"n_dips_{key_prefix}")
        for i in range(num_dips):
            c1, c2 = st.columns(2)
            h = c1.number_input(f"Hour", 0, 23, 13, key=f"h_{key_prefix}_{i}")
            p = c2.number_input(f"Drop %", 0, 100, 50, key=f"p_{key_prefix}_{i}")
            dips.append({'hour': h, 'percent': p})
    return dips

# ==========================================
# 3. MAIN UI
# ==========================================
def show_nilm_page(df_consumo, df_clima):
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
        
        # Base & Vent
        base_kw = st.number_input("Base Load [kW]", 0.0, 5000.0, 20.0)
        vent_kw = st.number_input("Ventilation [kW]", 0.0, 5000.0, 30.0)
        v_s, v_e = st.slider("Vent. Schedule", 0, 24, (6, 20))

        # Lighting
        st.subheader("Lighting")
        light_kw = st.number_input("Light Max [kW]", 0.0, 5000.0, 15.0)
        l_s, l_e = st.slider("Light Schedule", 0, 24, (7, 21))
        
        # HVAC
        st.subheader("HVAC")
        therm_kw = st.number_input("HVAC Cap [kW]", 0.0, 10000.0, 40.0)
        t_s, t_e = st.slider("HVAC Schedule", 0, 24, (8, 19))
        mode = st.selectbox("HVAC Mode", ["Constant", "Weather Driven"])
        
        st.divider()
        st.header("3. Variable Processes")
        
        # Occupancy Curve
        st.subheader("👥 Occupancy")
        occ_kw = st.number_input("Occupancy Max [kW]", 0.0, 5000.0, 10.0)
        occ_s, occ_e = st.slider("Occ. Schedule", 0, 24, (8, 18))
        occ_dips = render_dips_ui("occ")
        
        # 3 Generic Variable Processes
        proc_configs = {}
        for i in range(1, 4):
            with st.expander(f"⚙️ Custom Process {i}"):
                enabled = st.checkbox(f"Enable Process {i}", value=(i==1))
                name = st.text_input(f"Name {i}", value=f"Process {i}")
                color = st.color_picker(f"Color {i}", value="#9b59b6")
                p_kw = st.number_input(f"Max kW {i}", 0.0, 5000.0, 50.0)
                p_s, p_e = st.slider(f"Schedule {i}", 0, 24, (9, 17))
                c1, c2 = st.columns(2)
                ru = c1.number_input(f"Ramp Up (h) {i}", 0.0, 5.0, 1.0)
                rd = c2.number_input(f"Ramp Down (h) {i}", 0.0, 5.0, 1.0)
                
                # Dips for this process
                p_dips = render_dips_ui(f"proc_{i}")
                
                proc_configs.update({
                    f'proc_{i}_enabled': enabled,
                    f'proc_{i}_name': name,
                    f'proc_{i}_color': color,
                    f'proc_{i}_kw': p_kw,
                    f'proc_{i}_s': p_s, f'proc_{i}_e': p_e,
                    f'proc_{i}_ru': ru, f'proc_{i}_rd': rd,
                    f'proc_{i}_dips': p_dips
                })

    # --- PROCESSING ---
    
    # 1. Filter Data
    mask_month = df_merged['fecha'].dt.month.isin(selected_months)
    mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
    df_filtered = df_merged[mask_month & mask_day].copy()

    if df_filtered.empty:
        st.warning("No data for selected filters.")
        return

    # 2. Aggregate
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    # 3. Build Config
    config = {
        'base_kw': base_kw, 
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode,
        'occ_kw': occ_kw, 'occ_s': occ_s, 'occ_e': occ_e, 'occ_dips': occ_dips,
        'set_c': 24, 'set_h': 20, 'therm_sens': 5.0
    }
    config.update(proc_configs)

    # 4. Simulate
    df_sim = run_simulation(df_avg, config)

    # --- DASHBOARD ---
    
    # Chart 1: Main Full Width
    st.markdown("### 📈 Main Load Profile Analysis")
    
    fig1 = go.Figure()
    
    # Add Tariff Backgrounds
    tariff_periods = get_tariff_periods(not is_weekday)
    for start, end, color, name in tariff_periods:
        fig1.add_vrect(
            x0=start, x1=end, 
            fillcolor=color, opacity=1, 
            layer="below", line_width=0,
            annotation_text=name, annotation_position="top left"
        )

    # Add Stacked Simulation Layers
    layers = [
        ('sim_base', 'Base Load', '#7f8c8d'),
        ('sim_vent', 'Ventilation', '#3498db'),
        ('sim_light', 'Lighting', '#f1c40f'),
        ('sim_therm', 'HVAC', '#e74c3c'),
        ('sim_occ', 'Occupancy', '#e67e22')
    ]
    
    # Add Custom Processes to layers
    for i in range(1, 4):
        if config[f'proc_{i}_enabled']:
            layers.append((f'sim_proc_{i}', config[f'proc_{i}_name'], config[f'proc_{i}_color']))

    for col, name, color in layers:
        fig1.add_trace(go.Scatter(
            x=df_sim['hora'], y=df_sim[col], 
            stackgroup='one', name=name, 
            mode='none', fillcolor=color
        ))

    # Real Consumption Line
    fig1.add_trace(go.Scatter(
        x=df_sim['hora'], y=df_sim['consumo_kwh'], 
        name='REAL METER', line=dict(color='black', width=4)
    ))

    fig1.update_layout(
        height=600, 
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Hour of Day", dtick=1),
        yaxis=dict(title="Power (kW)"),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Secondary Charts Row
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🧩 Load Composition")
        # Calculate sums for pie chart
        pie_cols = [l[0] for l in layers]
        pie_names = [l[1] for l in layers]
        values = df_sim[pie_cols].sum()
        
        fig_pie = px.pie(values=values, names=pie_names, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📉 Correlation Check")
        fig_corr = px.scatter(
            df_sim, x='consumo_kwh', y='sim_total', 
            trendline="ols", labels={'consumo_kwh': 'Real', 'sim_total': 'Simulated'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with c2:
        st.subheader("⚠️ Hourly Error (kW)")
        fig_err = px.bar(
            df_sim, x='hora', y='error_kw', 
            color='error_kw', color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_err, use_container_width=True)
        
        st.subheader("🔋 Cumulative Energy (kWh)")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Acc.", fill='tozeroy'))
        fig_cum.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Acc.", line=dict(dash='dash')))
        st.plotly_chart(fig_cum, use_container_width=True)

    # Data Table
    st.divider()
    with st.expander("Show Detailed Data Table"):
        st.dataframe(df_sim.style.format(precision=2), use_container_width=True)


# ==========================================
# 4. ENTRY POINT (With Dummy Data)
# ==========================================
if __name__ == "__main__":
    # Generating dummy data for demonstration
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="h")
    
    # Create random consumption pattern
    np.random.seed(42)
    base_load = 20 + np.random.normal(0, 2, len(dates))
    work_load = np.where(dates.dayofweek < 5, 50, 10) * np.where((dates.hour > 8) & (dates.hour < 18), 1, 0)
    total_load = base_load + work_load + np.random.normal(0, 5, len(dates))
    
    # Create dataframe
    df_cons = pd.DataFrame({'fecha': dates, 'consumo_kwh': np.abs(total_load)})
    df_clim = pd.DataFrame({'fecha': dates, 'temperatura_c': 15 + 10 * np.sin(np.linspace(0, 3.14 * 2 * 365, len(dates)))})
    
    show_nilm_page(df_cons, df_clim)
