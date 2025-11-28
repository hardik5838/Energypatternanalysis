import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ... (Keep your existing Logic Engine functions: generate_load_curve and run_simulation) ...

# ==========================================
# 2. UI LAYOUT (UPDATED)
# ==========================================
def show_nilm_page(df_consumo, df_clima):
    # --- DATA PREP ---
    if df_consumo.empty: 
        st.error("No Data provided to calculator"); return
    
    # Merge and handle missing data safely
    try:
        df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
        # Ensure fecha is datetime
        df_merged['fecha'] = pd.to_datetime(df_merged['fecha'])
    except KeyError:
        st.error("Error: Input data must have a 'fecha' column.")
        return

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # --- NEW: MONTH FILTERING ---
        st.markdown("### 🗓️ Data Cleaning")
        # Get unique months from data (e.g., 'January', 'February')
        available_months = df_merged['fecha'].dt.month_name().unique()
        
        # User selects months to REMOVE
        months_to_remove = st.multiselect(
            "Exclude Months (Anomalies)", 
            options=available_months,
            placeholder="Select months to ignore...",
            help="Remove months with holidays or shutdowns to clean the average curve."
        )
        
        # Apply Month Filter
        if months_to_remove:
            mask_month = ~df_merged['fecha'].dt.month_name().isin(months_to_remove)
            df_merged = df_merged[mask_month]
            st.caption(f"Filtered out {len(months_to_remove)} months.")

        st.divider()

        # A. GLOBAL FILTER (Weekday/Weekend)
        day_type = st.radio("Profile Type", ["Weekday", "Weekend"], horizontal=True)
        is_weekday = (day_type == "Weekday")
        
        # Apply Day Type Filter
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask_day].copy()
        
        if df_filtered.empty: 
            st.warning(f"No data found for {day_type} after filtering."); return
        
        # B. BASE
        with st.expander("1. Base & Vent", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0.0, 10000.0, 20.0, step=1.0)
            st.markdown("**Ventilation**")
            vent_kw = st.number_input("Vent kW", 0.0, 10000.0, 30.0, step=1.0)
            c1, c2 = st.columns(2)
            v_s, v_e = c1.slider("Vent Time", 0, 24, (6, 20))
            v_ru = c2.number_input("Ramp Up (h)", 0.0, 5.0, 0.5)
            v_rd = c2.number_input("Ramp Down (h)", 0.0, 5.0, 0.5, key="v_rd")

        # C. LIGHTING
        with st.expander("2. Lighting", expanded=False):
            light_kw = st.number_input("Light kW", 0.0, 10000.0, 20.0, step=1.0)
            l_fac = st.slider("Factor %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Security %", 0.0, 0.5, 0.1)
            c3, c4 = st.columns(2)
            l_s, l_e = c3.slider("Light Time", 0, 24, (7, 21))
            l_ru = c4.number_input("L-Ramp Up", 0.0, 5.0, 0.5)
            l_rd = c4.number_input("L-Ramp Down", 0.0, 5.0, 0.5)

        # D. THERMAL
        with st.expander("3. HVAC Thermal", expanded=False):
            therm_kw = st.number_input("Chiller kW", 0.0, 50000.0, 45.0, step=5.0)
            t_s, t_e = st.slider("Therm Time", 0, 24, (8, 19))
            mode = st.selectbox("Mode", ["Constant", "Weather Driven"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Weather Driven":
                sens = st.slider("Sens.", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Cool", 18, 30, 24)
                sh = st.number_input("Set Heat", 15, 25, 20)

        # E. CUSTOM PROCESSES
        with st.expander("4. Custom Processes", expanded=False):
            if 'n_proc' not in st.session_state: st.session_state['n_proc'] = 1
            
            # Process Management Buttons
            b1, b2 = st.columns(2)
            if b1.button("➕ Add"): st.session_state['n_proc'] += 1
            if b2.button("➖ Remove"): st.session_state['n_proc'] = max(0, st.session_state['n_proc'] - 1)
            
            procs = []
            for i in range(st.session_state['n_proc']):
                st.markdown(f"**Proc {i+1}**")
                pk = st.number_input(f"kW {i+1}", 0.0, 5000.0, 10.0, key=f"pk{i}")
                ps, pe = st.slider(f"Time {i+1}", 0, 24, (8, 17), key=f"pt{i}")
                pr_u = st.number_input(f"R-Up {i+1}", 0.0, 5.0, 0.5, key=f"pru{i}")
                pr_d = st.number_input(f"R-Dn {i+1}", 0.0, 5.0, 0.5, key=f"prd{i}")
                
                # Simple Dip Logic
                has_dip = st.checkbox(f"Lunch Dip? {i+1}", value=(i==0), key=f"pd{i}")
                dips = [{'hour': 14, 'factor': 0.5}] if has_dip else []
                
                procs.append({'name': f"P{i+1}", 'kw': pk, 's': ps, 'e': pe, 'ru': pr_u, 'rd': pr_d, 'dips': dips})
                st.divider()

    # --- CALCULATION ---
    # We calculate the average based on the filtered data (minus excluded months)
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': base_kw,
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 'light_ru': l_ru, 'light_rd': l_rd,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': procs
    }
    
    df_sim = run_simulation(df_avg, config)

    # ==========================================
    # 3. MAIN DASHBOARD AREA
    # ==========================================
    st.subheader(f"📊 Digital Twin Calibration ({day_type})")
    if months_to_remove:
        st.caption(f"⚠️ Excluding data from: {', '.join(months_to_remove)}")
    
    # METRICS ROW
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean())
    
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    c_kpi1.metric("Real Energy (Avg Day)", f"{real:,.0f} kWh")
    c_kpi2.metric("Simulated Energy", f"{sim:,.0f} kWh", delta=f"{sim-real:,.0f} kWh")
    c_kpi3.metric("RMSE (Error)", f"{rmse:.2f}", help="Root Mean Square Error. Aim for < 5.0")

    # ROW 1: MAIN LOAD PROFILE CHART
    fig = go.Figure()
    x = df_sim['hora']
    
    # Stacked Areas
    layers = [
        ('sim_base', 'Base Load', 'gray'),
        ('sim_vent', 'Ventilation', '#3498db'),
        ('sim_therm', 'Thermal (HVAC)', '#e74c3c'),
        ('sim_light', 'Lighting', '#f1c40f'),
        ('sim_proc', 'Processes', '#e67e22')
    ]
    for col, name, color in layers:
        fig.add_trace(go.Scatter(x=x, y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        
    # Real Line
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL METER', line=dict(color='black', width=3)))
    
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ROW 2: SPLIT VIEW (Pie Chart | Residuals)
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Pie Chart
        sums = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light', 'sim_proc']].sum().reset_index()
        sums.columns = ['Category', 'kWh']
        sums['Category'] = sums['Category'].str.replace('sim_', '').str.capitalize()
        
        fig_pie = px.pie(sums, values='kWh', names='Category', title="Energy Breakdown Estimate", hole=0.4)
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        # Residuals (Bar Chart)
        df_sim['color'] = np.where(df_sim['diff'] > 0, '#ef5350', '#42a5f5')
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=df_sim['hora'], y=df_sim['diff'], marker_color=df_sim['color']))
        fig_res.update_layout(title="Error Analysis (Real - Sim)", height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="kW Difference")
        st.plotly_chart(fig_res, use_container_width=True)

    # ROW 3: COMPACT DATA TOGGLE
    with st.expander("📂 Show Raw Simulation Data"):
        st.dataframe(df_sim.round(1), use_container_width=True)
        csv = df_sim.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "sim_model.csv", "text/csv")


