import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import BytesIO

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

    total_custom = np.zeros(len(hours))
    for p in config['processes']:
        p_load = generate_load_curve(hours, p['s'], p['e'], p['kw'], p['ru'], p['rd'], p['dips'])
        df[f"proc_{p['name']}"] = p_load
        total_custom += p_load
    
    df['sim_proc'] = total_custom
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm'] + df['sim_proc']
    
    if 'consumo_kwh' in df.columns:
        df['error_abs'] = df['sim_total'] - df['consumo_kwh']
        df['error_pct'] = (df['error_abs'] / df['consumo_kwh'].replace(0, 1)) * 100
    return df

# ==========================================
# 2. UI LAYOUT
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    st.title("⚡ Energy Digital Twin & Optimization")

    # Column Formatting
    df_consumo.columns = df_consumo.columns.str.strip().str.lower()
    if not df_clima.empty:
        df_clima.columns = df_clima.columns.str.strip().str.lower()
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 20.0

    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("🎛️ Fitting Controls")
        day_type = st.radio("Profile Type", ["Laborable", "Fin de Semana"], horizontal=True)
        is_weekday = (day_type == "Laborable")
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask_day].copy()

        with st.expander("1. Infrastructure", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0.0, 5000.0, 20.0)
            vent_kw = st.number_input("Ventilation [kW]", 0.0, 5000.0, 30.0)
            v_s, v_e = st.slider("Vent. Schedule", 0, 24, (6, 20))
            v_ru, v_rd = 0.5, 0.5

        with st.expander("2. Lighting", expanded=False):
            light_kw = st.number_input("Total Light [kW]", 0.0, 5000.0, 20.0)
            l_fac = st.slider("Op. Factor %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Security/Night %", 0.0, 0.5, 0.1)
            l_s, l_e = st.slider("Light Schedule", 0, 24, (7, 21))

        with st.expander("3. HVAC", expanded=False):
            therm_kw = st.number_input("HVAC Cap [kW]", 0.0, 10000.0, 45.0)
            t_s, t_e = st.slider("HVAC Schedule", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sens.", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Cool", 18, 30, 24)
                sh = st.number_input("Set Heat", 15, 25, 20)

    # PROCESS DATA
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': base_kw, 'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 'light_ru': 0.5, 'light_rd': 0.5,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': []
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- DASHBOARD LAYOUT ---
    
    # ROW 1: KEY METRICS
    m1, m2, m3, m4 = st.columns(4)
    real_tot = df_sim['consumo_kwh'].sum()
    sim_tot = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['error_abs'])**2).mean())
    m1.metric("Real Energy", f"{real_tot:.1f} kWh")
    m2.metric("Simulated", f"{sim_tot:.1f} kWh", delta=f"{((sim_tot/real_tot)-1)*100:.1f}%")
    m3.metric("RMSE (Fit Quality)", f"{rmse:.2f}")
    m4.metric("Peak Load", f"{df_sim['sim_total'].max():.1f} kW")

    # ROW 2: PRIMARY CHARTS
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("### 1. Load Distribution (Stack)")
        fig1 = go.Figure()
        layers = [('sim_base', 'Base', '#95a5a6'), ('sim_vent', 'Vent.', '#3498db'), 
                  ('sim_therm', 'HVAC', '#e74c3c'), ('sim_light', 'Light', '#f1c40f')]
        for col, name, color in layers:
            fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        fig1.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'], name='REAL', line=dict(color='black', width=3)))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.write("### 2. Hourly Error Analysis")
        fig2 = px.bar(df_sim, x='hora', y='error_abs', color='error_abs', 
                     color_continuous_scale='RdBu_r', title="Over/Under Estimation (kW)")
        st.plotly_chart(fig2, use_container_width=True)

    # ROW 3: OPTIMIZATION & MIX
    col_c, col_d = st.columns(2)

    with col_c:
        st.write("### 3. Consumption Mix (%)")
        mix_data = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light']].sum()
        fig3 = px.pie(values=mix_data.values, names=mix_data.index, hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.write("### 4. Correlation: Real vs Sim")
        fig4 = px.scatter(df_sim, x='consumo_kwh', y='sim_total', trendline="ols",
                         labels={'consumo_kwh': 'Real (kW)', 'sim_total': 'Sim (kW)'})
        st.plotly_chart(fig4, use_container_width=True)

    # ROW 4: ADVANCED VISUALS
    col_e, col_f = st.columns(2)
    
    with col_e:
        st.write("### 5. Cumulative Energy Fit")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['consumo_kwh'].cumsum(), name="Real Cumulative", fill='tozeroy'))
        fig5.add_trace(go.Scatter(x=df_sim['hora'], y=df_sim['sim_total'].cumsum(), name="Sim Cumulative"))
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        st.write("### 6. Error Heatmap (Percentage)")
        error_matrix = df_sim['error_pct'].values.reshape(1, -1)
        fig6 = px.imshow(error_matrix, labels=dict(x="Hour of Day", y="Error", color="% Error"),
                        x=list(range(24)), color_continuous_scale='Viridis')
        st.plotly_chart(fig6, use_container_width=True)

    # --- EXPORTABLE TABLE ---
    st.divider()
    st.subheader("📋 Export Distribution Data")
    
    # Calculate percentages for the table
    export_df = df_sim.copy()
    for c in ['sim_base', 'sim_vent', 'sim_therm', 'sim_light']:
        export_df[f'{c}_pct'] = (export_df[c] / export_df['sim_total'] * 100).round(1)
    
    st.dataframe(export_df.style.background_gradient(subset=['error_pct'], cmap='RdYlGn_r'), use_container_width=True)
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Best Fit Results (CSV)", data=csv, file_name=f"energy_fit_{day_type}.csv", mime='text/csv')

    # --- OPTIMIZATION SECTION ---
    st.divider()
    st.subheader("🚀 Energy Optimization Simulator")
    o1, o2 = st.columns(2)
    
    with o1:
        peak_limit = st.slider("Set Peak Shaving Limit (kW)", float(df_sim['sim_total'].min()), float(df_sim['sim_total'].max()), float(df_sim['sim_total'].max()))
        optimized_total = np.where(df_sim['sim_total'] > peak_limit, peak_limit, df_sim['sim_total'])
        savings = df_sim['sim_total'].sum() - optimized_total.sum()
        st.info(f"Potential Savings by Shaving: {savings:.2f} kWh/day")

    with o2:
        eff_gain = st.slider("Efficiency Improvement (%)", 0, 50, 10) / 100
        new_total = optimized_total * (1 - eff_gain)
        st.success(f"Total Daily Energy After Optimization: {new_total.sum():.2f} kWh")
