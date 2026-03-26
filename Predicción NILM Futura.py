import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

### --------------------------------------------------------------------------
### |  1. CREACIÓN DE FEATURES (REZAGOS, TEMPERATURA Y CALENDARIO)           |
### --------------------------------------------------------------------------
def crear_features_prediccion(df):
    """
    Crea columnas de features diarias basadas en la fecha, rezagos (lags) y temperatura.
    """
    # Features temporales basadas en el código existente
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int) [2]
    
    # Variables de rezago (Lag variables) solicitadas
    df['consumo_lag_1'] = df['consumo_kwh'].shift(1)
    df['consumo_lag_7'] = df['consumo_kwh'].shift(7)
    
    # Eliminamos los valores nulos generados por los shifts
    df = df.dropna().reset_index(drop=True)
    return df

### --------------------------------------------------------------------------
### |  2. MODELO DE REGRESIÓN DE CARGA A FUTURO                              |
### --------------------------------------------------------------------------
def entrenar_modelo_y_predecir(df_historico, df_futuro):
    """
    Entrena un Random Forest Regressor usando variables de temperatura y lags,
    y predice el consumo total futuro.
    """
    # Preparar datos de entrenamiento
    features = ['temp_avg_c', 'dia_semana', 'dia_mes', 'mes', 'es_finde', 'consumo_lag_1', 'consumo_lag_7'] [2, 3]
    X = df_historico[features]
    y = df_historico['consumo_kwh']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) [4]
    
    # Entrenar modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42) [4]
    modelo.fit(X_train, y_train)
    
    # Predicción futura
    predicciones = modelo.predict(df_futuro[features])
    df_futuro['consumo_total_predicho'] = predicciones
    
    return df_futuro, modelo

### --------------------------------------------------------------------------
### |  3. APLICACIÓN DE NILM A PREDICCIONES FUTURAS                          |
### --------------------------------------------------------------------------
def aplicar_nilm_futuro(df_prediccion):
    """
    Aplica el análisis NILM perfeccionado en datos pasados para desagregar 
    el consumo total predicho en HVAC e Iluminación.
    """
    # Esta lógica simula la desagregación física basada en parámetros de calibración previos
    # Asumiendo ratios derivados del 'run_optimizer' o configuraciones estándar [5, 6]
    
    # Ejemplo de distribución basada en la estrategia física:
    # Asumimos que HVAC depende fuertemente de la temperatura y la iluminación del horario/día
    df_prediccion['hvac_predicho'] = np.where(
        df_prediccion['temp_avg_c'] > 22, 
        df_prediccion['consumo_total_predicho'] * 0.45, # Mayor uso de clima si hace calor
        df_prediccion['consumo_total_predicho'] * 0.20  # Consumo residual / ventilación
    )
    
    df_prediccion['iluminacion_predicho'] = np.where(
        df_prediccion['es_finde'] == 0,
        df_prediccion['consumo_total_predicho'] * 0.35, # Uso nominal en días de semana
        df_prediccion['consumo_total_predicho'] * 0.10  # Uso reducido (seguridad) fines de semana
    )
    
    # Resto de equipos
    df_prediccion['otros_predicho'] = df_prediccion['consumo_total_predicho'] - (df_prediccion['hvac_predicho'] + df_prediccion['iluminacion_predicho'])
    
    return df_prediccion

### --------------------------------------------------------------------------
### |  4. PÁGINA PRINCIPAL DE PREDICCIÓN NILM                                |
### --------------------------------------------------------------------------
def show_forecast_nilm_page(df_consumo_hist, df_clima_hist, df_clima_futuro):
    st.title("Predicción de Consumo y NILM a Futuro") [7]
    st.subheader("Pronóstico de Carga (Lags & Temperatura) + Desagregación (HVAC/Iluminación)") [8]
    st.markdown("---")
    
    with st.spinner('Procesando datos históricos y entrenando el modelo de regresión...'): [8]
        # 1. Unir consumo histórico y clima para crear el dataset de entrenamiento
        df_hist = pd.merge(df_consumo_hist, df_clima_hist, on='fecha', how='inner')
        df_hist = crear_features_prediccion(df_hist)
        
        # 2. Preparar los datos futuros (requiere simular los lags de los últimos días conocidos)
        df_futuro = crear_features_prediccion(df_clima_futuro.copy())
        
        # 3. Entrenar y predecir
        df_predicho, modelo = entrenar_modelo_y_predecir(df_hist, df_futuro)
        
        # 4. Aplicar desagregación NILM a los datos predichos
        df_resultado = aplicar_nilm_futuro(df_predicho)
    
    # 5. Visualización UI [6, 9]
    st.write("### Resultados del Pronóstico Desagregado")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_resultado['fecha'], y=df_resultado['hvac_predicho'], name='HVAC Predicho', marker_color='blue'))
    fig.add_trace(go.Bar(x=df_resultado['fecha'], y=df_resultado['iluminacion_predicho'], name='Iluminación Predicha', marker_color='yellow'))
    fig.add_trace(go.Bar(x=df_resultado['fecha'], y=df_resultado['otros_predicho'], name='Otros Equipos', marker_color='gray'))
    
    fig.update_layout(barmode='stack', title="Proyección de Carga NILM (Próximos Días)", xaxis_title="Fecha", yaxis_title="Consumo (kWh)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_resultado[['fecha', 'temp_avg_c', 'consumo_total_predicho', 'hvac_predicho', 'iluminacion_predicho']])
