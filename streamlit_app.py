import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import os
from urllib.parse import quote
import nilm_calculator  # <--- IMPORTANTE: Asegúrate de tener nilm_calculator.py en la misma carpeta

# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard Energético Asepeyo", page_icon="⚡", layout="wide")

# --- Funciones de Carga de Datos ---

@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético local o remoto."""
    try:
        if file_path.startswith('http'):
            df = pd.read_csv(file_path, sep=',', decimal='.', skipinitialspace=True)
        else:
            df = pd.read_csv(file_path, sep=',', decimal='.')
            
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo debe contener las columnas 'Fecha' y 'Energía activa (kWh)'. Columnas encontradas: {list(df.columns)}")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo: {e}")
        return pd.DataFrame()

@st.cache_data
def load_nasa_weather_data(file_path):
    """Carga y procesa el archivo de clima histórico de NASA POWER."""
    try:
        if file_path.startswith('http'):
            response = requests.get(file_path)
            response.raise_for_status()
            lines = response.text.splitlines()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        start_row = 0
        for i, line in enumerate(lines):
            if "YEAR,MO,DY,HR" in line:
                start_row = i
                break
        
        if file_path.startswith('http'):
            from io import StringIO
            df = pd.read_csv(StringIO('\n'.join(lines[start_row:])))
        else:
            df = pd.read_csv(file_path, skiprows=start_row)
        
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M', 'RH2M']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"El archivo de clima no tiene el formato esperado (Faltan columnas: {expected_cols})")
            return pd.DataFrame()

        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        df.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
        
        for col in ['temperatura_c', 'humedad_relativa']:
            df[col] = df[col].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c', 'humedad_relativa']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima: {e}")
        return pd.DataFrame()

# --- Barra Lateral ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    # NAVEGACIÓN ENTRE PÁGINAS
    page = st.selectbox("Seleccionar Herramienta", ["Dashboard General", "Simulación NILM (Avanzado)"])
    st.markdown("---")
    
    source_type = st.radio("Fuente de datos", ["Cargar Archivos", "Desde GitHub"])
    
    df_consumo = pd.DataFrame()
    df_clima = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy = st.file_uploader("Archivo Consumo (CSV)", type="csv")
        uploaded_weather = st.file_uploader("Archivo Clima (CSV)", type="csv")
        
        if uploaded_energy:
            df_consumo = load_asepeyo_energy_data(uploaded_energy)
        if uploaded_weather:
            import io
            content = uploaded_weather.getvalue().decode("utf-8")
            lines = content.splitlines()
            start = 0
            for i, l in enumerate(lines):
                if "YEAR,MO,DY,HR" in l:
                    start = i
                    break
            df_clima_raw = pd.read_csv(io.StringIO("\n".join(lines[start:])))
            df_clima_raw['fecha'] = pd.to_datetime(df_clima_raw[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
            df_clima_raw.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
            df_clima = df_clima_raw[['fecha', 'temperatura_c', 'humedad_relativa']]

    else:
        # Carga desde GitHub
        base_url = "https://raw.githubusercontent.com/hardik5838/EnergyPatternAnalysis/main/data/"
        file_energy = "251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"
        file_weather = "weather_Barcelona_2023.4-25.8.csv"
        
        st.info("Conectando con GitHub...")
        url_energy = base_url + quote(file_energy)
        url_weather = base_url + quote(file_weather)
        
        df_consumo = load_asepeyo_energy_data(url_energy)
        df_clima = load_nasa_weather_data(url_weather)
        
        if df_consumo.empty:
             st.error("❌ Fallo al cargar datos de Consumo.")
        if df_clima.empty:
             st.warning("⚠️ Fallo al cargar datos de Clima.")

    # --- FILTROS ESPECÍFICOS DEL DASHBOARD GENERAL ---
    # Solo mostramos estos filtros si estamos en la página principal y hay datos
    if page == "Dashboard General" and not df_consumo.empty:
        st.markdown("---")
        st.header("Filtros")
        
        min_date = df_consumo['fecha'].min().date()
        max_date = df_consumo['fecha'].max().date()
        date_range = st.date_input("Rango de Fechas", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        sel_dias = st.multiselect("Días de la semana", list(dias.keys()), format_func=lambda x: dias[x], default=list(dias.keys()))
        
        sel_horas = st.slider("Horas del día", 0, 23, (0, 23))
        
        st.markdown("---")
        st.header("Opciones Avanzadas")
        remove_base = st.checkbox("Eliminar Consumo Base")
        val_base = float(df_consumo['consumo_kwh'].quantile(0.1)) if not df_consumo.empty else 0.0
        umbral_base = st.number_input("Umbral Base (kWh)", value=val_base)
        
        remove_peak = st.checkbox("Eliminar Picos")
        umbral_pico = st.number_input("Percentil Picos", value=99.0, min_value=90.0, max_value=100.0)

# --- LÓGICA PRINCIPAL (CONTROL DE PÁGINAS) ---

if page == "Dashboard General":
    st.title("Dashboard Energético")

    if not df_consumo.empty:
        # Validar rango de fechas
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            # Aplicar filtros
            mask = (df_consumo['fecha'].dt.date >= date_range[0]) & (df_consumo['fecha'].dt.date <= date_range[1])
            mask &= df_consumo['fecha'].dt.dayofweek.isin(sel_dias)
            mask &= (df_consumo['fecha'].dt.hour >= sel_horas[0]) & (df_consumo['fecha'].dt.hour <= sel_horas[1])
            
            df_filtered = df_consumo[mask].copy()
            
            if remove_base:
                df_filtered = df_filtered[df_filtered['consumo_kwh'] > umbral_base]
            if remove_peak:
                limit = df_filtered['consumo_kwh'].quantile(umbral_pico/100)
                df_filtered = df_filtered[df_filtered['consumo_kwh'] < limit]

            if df_filtered.empty:
                st.warning("Los filtros seleccionados no han devuelto datos.")
            else:
                # --- Gráficos ---
                st.subheader("Patrones de Consumo")
                st.plotly_chart(px.line(df_filtered, x='fecha', y='consumo_kwh', title="Evolución Temporal"), use_container_width=True)
                
                col1, col2 = st.columns(2)
                # Perfil Diario
                perfil_horario = df_filtered.groupby(df_filtered['fecha'].dt.hour)['consumo_kwh'].mean().reset_index()
                col1.plotly_chart(px.bar(perfil_horario, x='fecha', y='consumo_kwh', title="Perfil Diario Promedio", labels={'fecha': 'Hora'}), use_container_width=True)
                
                # Perfil Semanal corregido
                perfil_semanal = df_filtered.groupby(df_filtered['fecha'].dt.dayofweek)['consumo_kwh'].mean().reset_index()
                if not perfil_semanal.empty:
                    perfil_semanal.columns = ['dia_num', 'consumo_kwh']
                    perfil_semanal['dia_nombre'] = perfil_semanal['dia_num'].map(dias)
                    perfil_semanal = perfil_semanal.sort_values('dia_num')
                    fig_semanal = px.bar(perfil_semanal, x='dia_nombre', y='consumo_kwh', title="Perfil Semanal Promedio",
                                       labels={'dia_nombre': 'Día', 'consumo_kwh': 'Consumo Promedio (kWh)'})
                    col2.plotly_chart(fig_semanal, use_container_width=True)

                # --- Correlación Clima ---
                if not df_clima.empty:
                    st.markdown("---")
                    st.subheader("Correlación con Clima")
                    df_merged = pd.merge(df_filtered, df_clima, on='fecha', how='inner')
                    
                    if not df_merged.empty:
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(px.scatter(df_merged, x='temperatura_c', y='consumo_kwh', title="Consumo vs Temperatura", 
                                                 trendline="ols", trendline_color_override="red"), use_container_width=True)
                        c2.plotly_chart(px.scatter(df_merged, x='humedad_relativa', y='consumo_kwh', title="Consumo vs Humedad", 
                                                 trendline="ols", trendline_color_override="red"), use_container_width=True)
                    else:
                        st.info("No hay coincidencia de fechas entre consumo y clima.")
        else:
            st.info("Seleccione un rango de fecha completo (Inicio y Fin).")
    else:
        st.info("Carga archivos para comenzar.")

elif page == "Simulación NILM (Avanzado)":
    # Llamada al módulo externo
    nilm_calculator.show_nilm_page(df_consumo, df_clima)
