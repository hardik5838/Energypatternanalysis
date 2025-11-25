# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard Energético Avanzado de Asepeyo",
    page_icon="⚡",
    layout="wide",
)

# --- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_csv_from_source(source):
    """Carga datos desde un archivo subido o una URL de GitHub."""
    if source is None:
        return pd.DataFrame()
    try:
        # Si es una URL, usa requests para manejarla
        if isinstance(source, str) and source.startswith('http'):
            response = requests.get(source)
            if response.status_code != 200:
                st.error(f"Error al descargar el archivo: {response.status_code}")
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
        # Si es un archivo subido
        return pd.read_csv(source, skipinitialspace=True)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    st.header("1. Fuente de Datos")
    source_type = st.radio("Seleccionar origen", ["Cargar Archivos", "Desde GitHub"], key="source_type")

    df_energy_raw = pd.DataFrame()
    df_weather_raw = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy_file = st.file_uploader("Archivo de Consumo (CSV)", type="csv")
        uploaded_weather_file = st.file_uploader("Archivo de Clima (CSV)", type="csv")
        if uploaded_energy_file:
            df_energy_raw = load_csv_from_source(uploaded_energy_file)
        if uploaded_weather_file:
            df_weather_raw = load_csv_from_source(uploaded_weather_file)
            
    else: # Desde GitHub (Método Directo para evitar Rate Limits)
        github_repo = st.text_input("Repositorio GitHub (usuario/repo)", "hardik5838/EnergyPatternAnalysis")
        
        # Lista de archivos hardcoded para evitar llamadas a la API que causan error 403
        # Puedes añadir más nombres de archivo a esta lista según los subas a tu repo
        known_files = [
            "251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv",
            "POWER_Point_Hourly_20230401_20250831_041d38N_002d18E_LST.csv",
            "POWER_Point_Hourly_20230401_20250831_041d40N_002d15E_LST.csv"
        ]
        
        base_url = f"https://raw.githubusercontent.com/{github_repo}/main/Data/"
        
        selected_energy_file = st.selectbox("Selecciona archivo de consumo", [""] + known_files)
        selected_weather_file = st.selectbox("Selecciona archivo de clima", [""] + known_files)

        if selected_energy_file:
            df_energy_raw = load_csv_from_source(base_url + selected_energy_file)
        if selected_weather_file:
            df_weather_raw = load_csv_from_source(base_url + selected_weather_file)

    # --- Procesamiento y Filtros ---
    df_energy = pd.DataFrame()
    if not df_energy_raw.empty:
        try:
            df_energy = df_energy_raw.copy()
            # Renombrado flexible de columnas
            col_map = {
                'Fecha': 'datetime', 
                'Energía activa (kWh)': 'Consumption_kWh',
                'Date & Time': 'datetime',
                'Consumption(kWh)': 'Consumption_kWh'
            }
            df_energy.rename(columns=col_map, inplace=True)
            
            if 'datetime' in df_energy.columns:
                df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True, errors='coerce')
                df_energy.dropna(subset=['datetime'], inplace=True)
                df_energy.set_index('datetime', inplace=True)
                st.sidebar.success("Datos de consumo cargados.")

                st.sidebar.markdown("---")
                st.sidebar.header("2. Filtros de Datos")
                
                dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                selected_days = st.sidebar.multiselect("Días de la semana", options=list(dias_semana.keys()), format_func=lambda x: dias_semana[x], default=list(dias_semana.keys()))
                selected_hours = st.sidebar.slider("Horas del día", 0, 23, (0, 23))
                
                if not df_energy.empty:
                    min_date = df_energy.index.min().date()
                    max_date = df_energy.index.max().date()
                    date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                else:
                     date_range = []

                st.sidebar.markdown("---")
                st.sidebar.header("3. Ajustes de Análisis")
                
                remove_baseline = st.sidebar.checkbox("Eliminar consumo base")
                baseline_val = float(df_energy['Consumption_kWh'].quantile(0.1)) if not df_energy.empty else 0.0
                baseline_threshold = st.sidebar.number_input("Umbral base (kWh)", value=baseline_val, disabled=not remove_baseline)
                
                remove_anomalies = st.sidebar.checkbox("Eliminar anomalías (picos)")
                anomaly_percentile = st.sidebar.number_input("Percentil para anomalías", value=99.0, min_value=90.0, max_value=100.0, disabled=not remove_anomalies)
                
                st.sidebar.markdown("---")
                st.sidebar.header("4. Constantes Matemáticas (HVAC)")
                base_temp_heating = st.sidebar.number_input("Temp. base calefacción (°C)", value=18.0, step=0.5)
                base_temp_cooling = st.sidebar.number_input("Temp. base refrigeración (°C)", value=21.0, step=0.5)
            else:
                 st.sidebar.error("No se encontró una columna de fecha válida en el archivo.")
                 df_energy = pd.DataFrame()

        except Exception as e:
            st.sidebar.error(f"Error procesando datos de energía: {e}")
            df_energy = pd.DataFrame()

# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")

if not df_energy.empty:
    # --- Aplicar filtros ---
    df_filtered = df_energy.copy()
    
    # Filtro Fecha
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        # Ajustar end_date para incluir todo el día
        end_date = end_date + pd.Timedelta(hours=23, minutes=59)
        df_filtered = df_filtered.loc[start_date:end_date]
    
    # Filtro Días
    if selected_days:
        df_filtered = df_filtered[df_filtered.index.dayofweek.isin(selected_days)]
    
    # Filtro Horas
    df_filtered = df_filtered[(df_filtered.index.hour >= selected_hours[0]) & (df_filtered.index.hour <= selected_hours[1])]
    
    # Filtro Baseline/Anomalías
    if remove_baseline:
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] > baseline_threshold]
    if remove_anomalies and not df_filtered.empty:
        upper_bound = df_filtered['Consumption_kWh'].quantile(anomaly_percentile / 100.0)
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] < upper_bound]

    # --- VERIFICACIÓN CRÍTICA: ¿Quedan datos? ---
    if df_filtered.empty:
        st.warning("⚠️ Los filtros seleccionados han excluido todos los datos. Por favor, ajusta el rango de fechas, días u horas.")
    else:
        st.markdown(f"Mostrando **{len(df_filtered):,}** registros tras aplicar filtros.")
        st.markdown("---")
        
        st.header("Análisis de Patrones Temporales")
        st.plotly_chart(px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Evolución del Consumo Energético'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patrón Diario Promedio")
            # Agrupar solo si hay datos
            df_hourly = df_filtered.groupby(df_filtered.index.hour)['Consumption_kWh'].mean().reset_index()
            df_hourly.rename(columns={'datetime': 'Hora'}, inplace=True) # Index name is typically passed as column name if not reset properly or named
            # Fix column name issue from groupby index
            df_hourly.columns = ['Hora', 'Consumption_kWh']
            st.plotly_chart(px.bar(df_hourly, x='Hora', y='Consumption_kWh', title='Consumo Promedio por Hora'), use_container_width=True)
            
        with col2:
            st.subheader("Patrón Semanal Promedio")
            # Agrupar por día de la semana
            df_weekly = df_filtered.groupby(df_filtered.index.dayofweek)['Consumption_kWh'].mean()
            
            # Mapear nombres solo para los días que existen en los datos filtrados
            df_weekly.index = df_weekly.index.map(dias_semana)
            
            # Crear gráfico solo si hay datos
            if not df_weekly.empty:
                st.plotly_chart(px.bar(df_weekly.reset_index(), x='datetime', y='Consumption_kWh', title='Consumo Promedio por Día', labels={'datetime': 'Día'}), use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar el patrón semanal.")

        # --- Análisis Climático ---
        if not df_weather_raw.empty:
            st.markdown("---")
            st.header("Correlación del Consumo con el Clima")
            try:
                df_weather = df_weather_raw.copy()
                
                # Procesar fecha del archivo de clima (Formato NASA POWER)
                if 'YEAR' in df_weather.columns:
                    df_weather['datetime'] = pd.to_datetime(df_weather[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
                    df_weather = df_weather[['datetime', 'T2M', 'RH2M']].set_index('datetime')
                
                # Unir con datos filtrados
                df_merged = df_filtered.join(df_weather, how='inner').dropna()

                if not df_merged.empty:
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Consumo vs. Temperatura")
                        st.plotly_chart(px.scatter(df_merged, x='T2M', y='Consumption_kWh', labels={'T2M': 'Temperatura (°C)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
                    with col4:
                        st.subheader("Consumo vs. Humedad")
                        st.plotly_chart(px.scatter(df_merged, x='RH2M', y='Consumption_kWh', labels={'RH2M': 'Humedad (%)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
                else:
                    st.warning("No hay coincidencia de fechas entre los datos de consumo filtrados y el archivo de clima.")
            except Exception as e:
                st.error(f"Error al procesar los datos de clima: {e}")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona uno desde GitHub en la barra lateral.")
