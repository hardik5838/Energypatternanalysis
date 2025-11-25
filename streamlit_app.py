import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import os
from urllib.parse import quote  # Importación necesaria para corregir URLs con espacios

# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard Energético Asepeyo", page_icon="⚡", layout="wide")

# --- Funciones de Carga de Datos ---

@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético local o remoto."""
    try:
        # Si es una URL
        if file_path.startswith('http'):
            df = pd.read_csv(file_path, sep=',', decimal='.', skipinitialspace=True)
        else:
            df = pd.read_csv(file_path, sep=',', decimal='.')
            
        # Limpieza de nombres de columnas
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        # Verificar columnas requeridas
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
        # Leer archivo omitiendo cabecera variable
        if file_path.startswith('http'):
            # Para URL, necesitamos encontrar dónde empiezan los datos
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

        # Crear fecha y limpiar
        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        df.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
        
        # Reemplazar -999 con NaN y rellenar
        for col in ['temperatura_c', 'humedad_relativa']:
            df[col] = df[col].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c', 'humedad_relativa']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima: {e}")
        return pd.DataFrame()

# --- Barra Lateral ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    source_type = st.radio("Fuente de datos", ["Cargar Archivos", "Desde GitHub"])
    
    df_consumo = pd.DataFrame()
    df_clima = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy = st.file_uploader("Archivo Consumo (CSV)", type="csv")
        uploaded_weather = st.file_uploader("Archivo Clima (CSV)", type="csv")
        
        if uploaded_energy:
            # Guardar temporalmente para usar la función que pide 'file_path' o pasar objeto
            df_consumo = load_asepeyo_energy_data(uploaded_energy)
        if uploaded_weather:
            # Lógica para uploader directo de clima
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
        # --- SECCIÓN CORREGIDA PARA GITHUB ---
        # 1. URL base corregida a minúsculas (/data/)
        base_url = "https://raw.githubusercontent.com/hardik5838/EnergyPatternAnalysis/main/data/"
        
        file_energy = "251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"
        file_weather = "POWER_Point_Hourly_20230401_20250831_041d38N_002d18E_LST.csv"
        
        st.info("Conectando con GitHub...")
        
        # 2. Codificación de URL para manejar espacios (%20)
        url_energy = base_url + quote(file_energy)
        url_weather = base_url + quote(file_weather)
        
        df_consumo = load_asepeyo_energy_data(url_energy)
        df_clima = load_nasa_weather_data(url_weather)
        
        # 3. Debugging en caso de fallo
        if df_consumo.empty:
             st.error("❌ Fallo al cargar datos de Consumo.")
             with st.expander("Ver URL de Consumo generada"):
                 st.code(url_energy, language='text')
                 
        if df_clima.empty:
             st.warning("⚠️ Fallo al cargar datos de Clima.")
             with st.expander("Ver URL de Clima generada"):
                 st.code(url_weather, language='text')

    # --- Filtros Globales ---
    if not df_consumo.empty:
        st.markdown("---")
        st.header("Filtros")
        
        # Fecha
        min_date = df_consumo['fecha'].min().date()
        max_date = df_consumo['fecha'].max().date()
        date_range = st.date_input("Rango de Fechas", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # Días de la semana
        dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        sel_dias = st.multiselect("Días de la semana", list(dias.keys()), format_func=lambda x: dias[x], default=list(dias.keys()))
        
        # Horas
        sel_horas = st.slider("Horas del día", 0, 23, (0, 23))
        
        # Opciones Avanzadas
        st.markdown("---")
        st.header("Opciones Avanzadas")
        remove_base = st.checkbox("Eliminar Consumo Base (Outliers bajos)")
        # Manejo seguro del quantile si hay pocos datos
        if not df_consumo.empty:
            val_base = float(df_consumo['consumo_kwh'].quantile(0.1))
        else:
            val_base = 0.0
        umbral_base = st.number_input("Umbral Base (kWh)", value=val_base)
        
        remove_peak = st.checkbox("Eliminar Picos (Outliers altos)")
        umbral_pico = st.number_input("Percentil Picos", value=99.0, min_value=90.0, max_value=100.0)

# --- Lógica Principal ---
st.title("Dashboard Energético")

if not df_consumo.empty:
    # Aplicar filtros
    # Asegurar que date_range tenga 2 valores (inicio y fin)
    if len(date_range) == 2:
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
            
            # Evolución
            st.plotly_chart(px.line(df_filtered, x='fecha', y='consumo_kwh', title="Evolución Temporal"), use_container_width=True)
            
            col1, col2 = st.columns(2)
            # Perfil Diario
            perfil_horario = df_filtered.groupby(df_filtered['fecha'].dt.hour)['consumo_kwh'].mean().reset_index()
            col1.plotly_chart(px.bar(perfil_horario, x='fecha', y='consumo_kwh', title="Perfil Diario Promedio", labels={'fecha': 'Hora'}), use_container_width=True)
            
            # Perfil Semanal (Solo si hay datos)
            perfil_semanal = df_filtered.groupby(df_filtered['fecha'].dt.dayofweek)['consumo_kwh'].mean()
            if not perfil_semanal.empty:
                perfil_semanal.index = perfil_semanal.index.map(dias)
                # Reordenar correctamente
                perfil_semanal = perfil_semanal.reindex([dias[i] for i in sorted(dias.keys()) if i in perfil_semanal.index])
                col2.plotly_chart(px.bar(perfil_semanal, title="Perfil Semanal Promedio"), use_container_width=True)

            # --- Correlación Clima ---
            if not df_clima.empty:
                st.markdown("---")
                st.subheader("Correlación con Clima")
                
                # Unir datos
                df_merged = pd.merge(df_filtered, df_clima, on='fecha', how='inner')
                
                if not df_merged.empty:
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(px.scatter(df_merged, x='temperatura_c', y='consumo_kwh', title="Consumo vs Temperatura", trendline="ols"), use_container_width=True)
                    c2.plotly_chart(px.scatter(df_merged, x='humedad_relativa', y='consumo_kwh', title="Consumo vs Humedad", trendline="ols"), use_container_width=True)
                else:
                    st.info("No hay coincidencia de fechas entre consumo y clima para los filtros seleccionados.")
    else:
        st.info("Seleccione un rango de fecha completo (Inicio y Fin).")
else:
    st.info("Carga archivos para comenzar o selecciona 'Desde GitHub'.")
