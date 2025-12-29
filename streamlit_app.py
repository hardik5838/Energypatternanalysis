import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import os
import io
from urllib.parse import quote
import nilm_calculator  # Asegúrate de que este archivo esté en la misma carpeta



# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard Energético Asepeyo", page_icon="⚡", layout="wide")

# --- Funciones de Carga de Datos ---

@st.cache_data
def load_asepeyo_energy_data(file_input):
    """Carga y procesa el archivo de consumo energético con limpieza profunda."""
    try:
        if isinstance(file_input, str):
            if file_input.startswith('http'):
                df = pd.read_csv(file_input, sep=',', skipinitialspace=True)
            else:
                df = pd.read_csv(file_input, sep=',')
        else:
            file_input.seek(0)
            df = pd.read_csv(file_input, sep=',')
            
        # 1. Limpiar espacios en nombres de columnas
        df.columns = df.columns.str.strip()
        
        # 2. Identificar columnas necesarias
        col_Fecha = 'Fecha'
        col_energia = 'Energía activa (kWh)'
        
        if col_Fecha not in df.columns or col_energia not in df.columns:
            st.error(f"Columnas requeridas no encontradas. Disponibles: {list(df.columns)}")
            return pd.DataFrame()
            
        # 3. Renombrar
        df = df.rename(columns={col_Fecha: 'Fecha', col_energia: 'consumo_kwh'})
        
        # 4. CONVERSIÓN CRÍTICA: Forzar a numérico (corrige el error de quantile)
        # Si hay comas en lugar de puntos, las reemplazamos primero
        if df['consumo_kwh'].dtype == 'object':
            df['consumo_kwh'] = df['consumo_kwh'].astype(str).str.replace(',', '.')
            
        df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
        
        # 5. Procesar Fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        # 6. Limpiar filas nulas creadas por errores de formato
        df = df.dropna(subset=['Fecha', 'consumo_kwh'])
        
        return df.sort_values('Fecha')

    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo: {e}")
        return pd.DataFrame()

@st.cache_data
def load_nasa_weather_data(file_path):
    """Carga y procesa el archivo de clima histórico."""
    try:
        if isinstance(file_path, str) and file_path.startswith('http'):
            response = requests.get(file_path)
            response.raise_for_status()
            content = response.text
        elif hasattr(file_path, 'getvalue'): # Es un objeto de Streamlit
            content = file_path.getvalue().decode("utf-8")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        lines = content.splitlines()
        start_row = 0
        for i, line in enumerate(lines):
            if "YEAR,MO,DY,HR" in line:
                start_row = i
                break
        
        df = pd.read_csv(io.StringIO('\n'.join(lines[start_row:])))
        
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M', 'RH2M']
        if not all(col in df.columns for col in expected_cols):
            return pd.DataFrame()

        df['Fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        df.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
        
        for col in ['temperatura_c', 'humedad_relativa']:
            df[col] = df[col].replace(-999, np.nan).ffill()
        
        return df[['Fecha', 'temperatura_c', 'humedad_relativa']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima: {e}")
        return pd.DataFrame()

# --- Barra Lateral ---
with st.sidebar:
    st.title('⚡ Panel de Control')
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
            df_clima = load_nasa_weather_data(uploaded_weather)
    else:
        base_url = "https://raw.githubusercontent.com/hardik5838/EnergyPatternAnalysis/main/data/"
        file_energy = "251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"
        file_weather = "weather.csv"
        
        url_energy = base_url + quote(file_energy)
        url_weather = base_url + quote(file_weather)
        
        df_consumo = load_asepeyo_energy_data(url_energy)
        df_clima = load_nasa_weather_data(url_weather)

    # Filtros
    if page == "Dashboard General" and not df_consumo.empty:
        st.markdown("---")
        st.header("Filtros")
        
        min_date = df_consumo['Fecha'].min().date()
        max_date = df_consumo['Fecha'].max().date()
        date_range = st.date_input("Rango de Fechas", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        sel_dias = st.multiselect("Días de la semana", list(dias.keys()), format_func=lambda x: dias[x], default=list(dias.keys()))
        sel_horas = st.slider("Horas del día", 0, 23, (0, 23))
        
        st.markdown("---")
        st.header("Opciones Avanzadas")
        remove_base = st.checkbox("Eliminar Consumo Base")
        
        # Cálculo seguro del cuantil
        val_base_init = float(df_consumo['consumo_kwh'].quantile(0.1)) if not df_consumo.empty else 0.0
        umbral_base = st.number_input("Umbral Base (kWh)", value=val_base_init)
        
        remove_peak = st.checkbox("Eliminar Picos")
        umbral_pico = st.number_input("Percentil Picos", value=99.0, min_value=90.0, max_value=100.0)

# --- LÓGICA PRINCIPAL ---

if page == "Dashboard General":
    st.title("Dashboard Energético")

    if not df_consumo.empty:
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            # Aplicar filtros
            mask = (df_consumo['Fecha'].dt.date >= date_range[0]) & (df_consumo['Fecha'].dt.date <= date_range[1])
            mask &= df_consumo['Fecha'].dt.dayofweek.isin(sel_dias)
            mask &= (df_consumo['Fecha'].dt.hour >= sel_horas[0]) & (df_consumo['Fecha'].dt.hour <= sel_horas[1])
            
            df_filtered = df_consumo[mask].copy()
            
            if remove_base:
                df_filtered = df_filtered[df_filtered['consumo_kwh'] > umbral_base]
            if remove_peak and not df_filtered.empty:
                limit = df_filtered['consumo_kwh'].quantile(umbral_pico/100)
                df_filtered = df_filtered[df_filtered['consumo_kwh'] < limit]

            if df_filtered.empty:
                st.warning("No hay datos para los filtros seleccionados.")
            else:
                st.subheader("Patrones de Consumo")
                st.plotly_chart(px.line(df_filtered, x='Fecha', y='consumo_kwh', title="Evolución Temporal"), use_container_width=True)
                
                col1, col2 = st.columns(2)
                perfil_horario = df_filtered.groupby(df_filtered['Fecha'].dt.hour)['consumo_kwh'].mean().reset_index()
                col1.plotly_chart(px.bar(perfil_horario, x='Fecha', y='consumo_kwh', title="Perfil Diario (Media)", labels={'Fecha': 'Hora'}), use_container_width=True)
                
                perfil_semanal = df_filtered.groupby(df_filtered['Fecha'].dt.dayofweek)['consumo_kwh'].mean().reset_index()
                perfil_semanal['dia_nombre'] = perfil_semanal['Fecha'].map(dias)
                col2.plotly_chart(px.bar(perfil_semanal, x='dia_nombre', y='consumo_kwh', title="Perfil Semanal (Media)"), use_container_width=True)

                if not df_clima.empty:
                    st.markdown("---")
                    st.subheader("Correlación con Clima")
                    df_merged = pd.merge(df_filtered, df_clima, on='Fecha', how='inner')
                    if not df_merged.empty:
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(px.scatter(df_merged, x='temperatura_c', y='consumo_kwh', title="Consumo vs Temperatura", trendline="ols"), use_container_width=True)
                        c2.plotly_chart(px.scatter(df_merged, x='humedad_relativa', y='consumo_kwh', title="Consumo vs Humedad", trendline="ols"), use_container_width=True)
        else:
            st.info("Por favor, selecciona un rango de Fechas válido en la barra lateral.")
    else:
        st.info("Carga archivos CSV para comenzar el análisis.")

elif page == "Simulación NILM (Avanzado)":
    nilm_calculator.show_nilm_page(df_consumo, df_clima)
    
