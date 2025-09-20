import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar módulos
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Gasolina",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("⛽ Predictor de Precios de Gasolina en México")
st.markdown("---")

# Descripción de la aplicación
st.markdown("""
### Descripción
Aplicacion para predecir el costo de la gasolina en México según la entidad, mes y año.

El modelo fue entrenado con datos históricos de precios de gasolina de diferentes entidades federativas de México.
""")

@st.cache_data
def cargar_datos():
    """Carga los datos expandidos de gasolina"""
    try:
        # Intentar cargar desde el directorio padre
        archivo_datos = parent_dir / "Gasolina_expandido.csv"
        if archivo_datos.exists():
            df = pd.read_csv(archivo_datos)
            return df
        else:
            st.error(f"No se encontró el archivo de datos en: {archivo_datos}")
            return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y los encoders"""
    try:
        # Intentar cargar desde el directorio padre
        archivo_modelo = parent_dir / "modelo_gasolina.pkl"
        if archivo_modelo.exists():
            with open(archivo_modelo, 'rb') as f:
                modelo_data = pickle.load(f)
            return modelo_data
        else:
            st.error(f"No se encontró el archivo del modelo en: {archivo_modelo}")
            return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def predecir_precio(modelo_data, entidad, mes, año):
    """Realiza la predicción del precio"""
    try:
        modelo = modelo_data['modelo']
        encoder_entidad = modelo_data['encoder_entidad']
        encoder_mes = modelo_data['encoder_mes']
        
        # Verificar que la entidad existe
        if entidad not in encoder_entidad.classes_:
            return None, f"Error: La entidad '{entidad}' no existe en el dataset"
        
        # Verificar que el mes existe
        if mes not in encoder_mes.classes_:
            return None, f"Error: El mes '{mes}' no existe en el dataset"
        
        # Codificar las variables
        entidad_encoded = encoder_entidad.transform([entidad])[0]
        mes_encoded = encoder_mes.transform([mes])[0]
        
        # Crear el vector de características
        X_pred = np.array([[entidad_encoded, mes_encoded, año]])
        
        # Realizar la predicción
        precio_predicho = modelo.predict(X_pred)[0]
        
        return precio_predicho, None
        
    except Exception as e:
        return None, f"Error en la predicción: {str(e)}"

# Cargar datos y modelo
df_gasolina = cargar_datos()
modelo_data = cargar_modelo()

if df_gasolina is not None and modelo_data is not None:
    
    # Obtener opciones únicas
    entidades = sorted(df_gasolina['Entidad'].unique())
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    años = sorted(df_gasolina['Año'].unique())
    
    # Widgets de selección
    entidad_seleccionada = st.selectbox(
        "Selecciona el Estado/Entidad:",
        entidades,
        index=entidades.index('Nacional') if 'Nacional' in entidades else 0
    )
    
    mes_seleccionado = st.selectbox(
        "Selecciona el Mes:",
        meses,
        index=0
    )
    
    año_seleccionado = st.selectbox(
        "Selecciona el Año:",
        list(range(min(años), max(años) + 5)),  # Permitir predicciones futuras
        index=len(años) - 1 if años else 0
    )
    
    # Botón de predicción
    if st.button("🔮 Realizar Predicción", type="primary"):
        precio_predicho, error = predecir_precio(
            modelo_data, entidad_seleccionada, mes_seleccionado, año_seleccionado
        )
        
        if error:
            st.error(error)
        else:
                        # Información adicional
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Entidad Seleccionada",
                    value=entidad_seleccionada
                )
            
            with col2:
                st.metric(
                    label="Período",
                    value=f"{mes_seleccionado} {año_seleccionado}"
                )
            
            with col3:
                # Calcular precio promedio histórico para comparación
                precio_historico = df_gasolina[
                    (df_gasolina['Entidad'] == entidad_seleccionada) &
                    (df_gasolina['Mes'] == mes_seleccionado)
                ]['Precio'].mean()
                
                if not pd.isna(precio_historico):
                    diferencia = precio_predicho - precio_historico
                    st.metric(
                        label="vs Promedio Histórico",
                        value=f"${precio_predicho:.2f}",
                        delta=f"{diferencia:+.2f}"
                    )
                else:
                    st.metric(
                        label="Precio Predicho",
                        value=f"${precio_predicho:.2f}"
                    )
            st.write("--")
            # Mostrar resultado principal
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### Resultado de la Predicción")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #fdcddc 0%, #fa82a7 100%);
                    padding: 2rem;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                ">
                    <h2 style="margin: 0; color: white;">💰 ${precio_predicho:.2f} MXN</h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        {entidad_seleccionada} • {mes_seleccionado} {año_seleccionado}
                    </p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("""
    ❌ **Error de Configuración**
    
    No se pudieron cargar los datos o el modelo necesarios. 
    
    **Pasos para solucionar:**
    1. Asegúrate de que el archivo `Gasolina_expandido.csv` existe en el directorio padre
    2. Asegúrate de que el archivo `modelo_gasolina.pkl` existe en el directorio padre
    3. Ejecuta primero el notebook `analisis_gasolina.ipynb` para generar estos archivos
    """)
