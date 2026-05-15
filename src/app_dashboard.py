import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página
st.set_page_config(
    page_title="Sistema Credit Scoring",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Cargar modelo y preprocessors"""
    model = joblib.load('../models/best_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    label_encoders = joblib.load('../models/label_encoders.pkl')
    metadata = pd.read_json('../models/metadata.json')
    return model, scaler, label_encoders, metadata

@st.cache_data
def load_data():
    """Cargar datos"""
    return pd.read_csv('../data/credit_data.csv')

def main():
    st.title("🏦 Sistema de Credit Scoring")
    st.markdown("### Predicción de Riesgo Crediticio con Machine Learning")

    # Cargar modelo y datos
    model, scaler, label_encoders, metadata = load_model()
    df = load_data()

    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Seleccione una página:",
        ["📊 Overview", "🔮 Predicción Individual", "📈 Análisis de Portfolio", "🎯 Análisis de Features"]
    )

    if page == "📊 Overview":
        show_overview(df, metadata)
    elif page == "🔮 Predicción Individual":
        show_prediction(model, scaler, label_encoders, df)
    elif page == "📈 Análisis de Portfolio":
        show_portfolio_analysis(df, model, scaler)
    elif page == "🎯 Análisis de Features":
        show_feature_analysis(df)


def show_overview(df, metadata):
    """Página de overview"""
    st.header("📊 Resumen del Sistema")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Clientes", f"{len(df):,}")

    with col2:
        default_rate = df['default'].mean() * 100
        st.metric("Tasa de Default", f"{default_rate:.2f}%")

    with col3:
        avg_score = df['credit_score'].mean()
        st.metric("Credit Score Promedio", f"{avg_score:.0f}")

    with col4:
        model_auc = metadata['roc_auc'].iloc[0]
        st.metric("ROC-AUC del Modelo", f"{model_auc:.4f}")

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de Credit Score")
        fig = px.histogram(
            df,
            x='credit_score',
            nbins=50,
            color='default',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'default': 'Default', 'credit_score': 'Credit Score'},
            title="Distribución por Estado"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribución de Ingresos")
        fig = px.box(
            df,
            x='default',
            y='ingreso_mensual',
            color='default',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'default': 'Default', 'ingreso_mensual': 'Ingreso Mensual'},
            title="Ingresos por Estado"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Más análisis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default por Nivel de Educación")
        default_by_edu = df.groupby('nivel_educacion')['default'].agg(['mean', 'count']).reset_index()
        default_by_edu['mean'] = default_by_edu['mean'] * 100

        fig = px.bar(
            default_by_edu,
            x='nivel_educacion',
            y='mean',
            text='count',
            labels={'mean': 'Tasa Default (%)', 'nivel_educacion': 'Nivel Educación'},
            title="Tasa de Default por Educación"
        )
        fig.update_traces(texttemplate='%{text} clientes', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Default por Propósito del Préstamo")
        default_by_purpose = df.groupby('proposito')['default'].agg(['mean', 'count']).reset_index()
        default_by_purpose['mean'] = default_by_purpose['mean'] * 100

        fig = px.bar(
            default_by_purpose,
            x='proposito',
            y='mean',
            text='count',
            labels={'mean': 'Tasa Default (%)', 'proposito': 'Propósito'},
            title="Tasa de Default por Propósito"
        )
        fig.update_traces(texttemplate='%{text} clientes', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def show_prediction(model, scaler, label_encoders, df):
    """Página de predicción individual"""
    st.header("🔮 Predicción de Riesgo Individual")

    st.markdown("### Ingrese los datos del solicitante:")

    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=80, value=35)
        genero = st.selectbox("Género", ['M', 'F'])
        estado_civil = st.selectbox("Estado Civil", ['Soltero', 'Casado', 'Divorciado', 'Viudo'])
        nivel_educacion = st.selectbox("Nivel Educación", ['Secundaria', 'Técnico', 'Universitario', 'Posgrado'])

    with col2:
        ingreso_mensual = st.number_input("Ingreso Mensual ($)", min_value=800, max_value=50000, value=5000)
        tipo_empleo = st.selectbox("Tipo de Empleo", ['Empleado', 'Independiente', 'Empresario', 'Desempleado'])
        antiguedad_laboral = st.number_input("Antigüedad Laboral (meses)", min_value=0, max_value=360, value=36)
        tiene_propiedad = st.checkbox("Tiene Propiedad")
        valor_propiedad = st.number_input("Valor Propiedad ($)", min_value=0, max_value=500000, value=0) if tiene_propiedad else 0
        tiene_vehiculo = st.checkbox("Tiene Vehículo")

    with col3:
        antiguedad_historial = st.number_input("Antigüedad Historial Crediticio (meses)", min_value=0, max_value=360, value=60)
        num_cuentas = st.number_input("Número de Cuentas de Crédito", min_value=0, max_value=15, value=3)
        num_prestamos = st.number_input("Número de Préstamos Activos", min_value=0, max_value=8, value=1)
        deuda_total = st.number_input("Deuda Total ($)", min_value=0, max_value=200000, value=10000)
        utilizacion_credito = st.slider("Utilización de Crédito (%)", 0, 100, 30)
        num_pagos_atrasados = st.number_input("Pagos Atrasados (últimos 12 meses)", min_value=0, max_value=12, value=0)
        dias_max_atraso = st.selectbox("Días Máximo Atraso", [0, 30, 60, 90, 120])
        num_consultas = st.number_input("Consultas de Crédito (últimos 6 meses)", min_value=0, max_value=20, value=2)

    st.markdown("### Datos del Préstamo Solicitado:")
    col1, col2, col3 = st.columns(3)

    with col1:
        monto_solicitado = st.number_input("Monto Solicitado ($)", min_value=1000, max_value=100000, value=15000)

    with col2:
        plazo_meses = st.selectbox("Plazo (meses)", [12, 24, 36, 48, 60, 72])

    with col3:
        proposito = st.selectbox("Propósito", ['Consumo', 'Vivienda', 'Vehiculo', 'Educacion', 'Negocio', 'Consolidacion'])

    if st.button("🔍 Predecir Riesgo", type="primary"):
        # Calcular variables derivadas
        ratio_deuda_ingreso = (deuda_total / (ingreso_mensual * 12)) * 100
        ratio_cuota_ingreso = ((monto_solicitado / plazo_meses) / ingreso_mensual) * 100

        # Calcular credit score simulado
        prob_base = (
            -0.3 * (edad < 25) +
            -0.5 * (tipo_empleo == 'Desempleado') +
            -0.01 * antiguedad_laboral +
            -0.005 * antiguedad_historial +
            0.3 * (num_pagos_atrasados > 2) +
            0.4 * (dias_max_atraso >= 90) +
            0.02 * (utilizacion_credito - 50) +
            0.01 * (ratio_deuda_ingreso - 40) +
            0.02 * (ratio_cuota_ingreso - 30) +
            -0.3 * tiene_propiedad +
            -0.2 * tiene_vehiculo
        )
        prob_default = 1 / (1 + np.exp(-prob_base))
        credit_score_calc = int(np.clip(300 + 550 * (1 - prob_default), 300, 850))

        # Preparar features
        features = {
            'edad': edad,
            'ingreso_mensual': ingreso_mensual,
            'antiguedad_laboral_meses': antiguedad_laboral,
            'antiguedad_historial_meses': antiguedad_historial,
            'num_cuentas_credito': num_cuentas,
            'num_prestamos_activos': num_prestamos,
            'deuda_total': deuda_total,
            'utilizacion_credito': utilizacion_credito,
            'num_pagos_atrasados_12m': num_pagos_atrasados,
            'dias_max_atraso': dias_max_atraso,
            'num_consultas_6m': num_consultas,
            'tiene_propiedad': int(tiene_propiedad),
            'valor_propiedad': valor_propiedad,
            'tiene_vehiculo': int(tiene_vehiculo),
            'monto_solicitado': monto_solicitado,
            'plazo_meses': plazo_meses,
            'ratio_deuda_ingreso': ratio_deuda_ingreso,
            'ratio_cuota_ingreso': ratio_cuota_ingreso,
            'credit_score': credit_score_calc,
            'genero_encoded': label_encoders['genero'].transform([genero])[0],
            'estado_civil_encoded': label_encoders['estado_civil'].transform([estado_civil])[0],
            'nivel_educacion_encoded': label_encoders['nivel_educacion'].transform([nivel_educacion])[0],
            'tipo_empleo_encoded': label_encoders['tipo_empleo'].transform([tipo_empleo])[0],
            'proposito_encoded': label_encoders['proposito'].transform([proposito])[0]
        }

        X = pd.DataFrame([features])
        X_scaled = scaler.transform(X)

        # Predicción
        probabilidad = model.predict_proba(X_scaled)[0, 1]
        prediccion = "ALTO RIESGO ⚠️" if probabilidad > 0.5 else "BAJO RIESGO ✅"

        st.markdown("---")
        st.markdown("## 🎯 Resultado de la Evaluación")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Decisión", prediccion)

        with col2:
            st.metric("Probabilidad de Default", f"{probabilidad*100:.2f}%")

        with col3:
            st.metric("Credit Score Estimado", credit_score_calc)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilidad * 100,
            title={'text': "Riesgo de Default (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probabilidad > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Recomendaciones
        st.markdown("### 📋 Análisis de Factores")

        if probabilidad > 0.5:
            st.error("**⚠️ Factores de Alto Riesgo Detectados:**")
            factores_riesgo = []

            if ratio_deuda_ingreso > 40:
                factores_riesgo.append(f"- Ratio deuda/ingreso alto: {ratio_deuda_ingreso:.1f}% (ideal < 40%)")
            if utilizacion_credito > 70:
                factores_riesgo.append(f"- Utilización de crédito alta: {utilizacion_credito}% (ideal < 30%)")
            if num_pagos_atrasados > 0:
                factores_riesgo.append(f"- Historial de pagos atrasados: {num_pagos_atrasados} en últimos 12 meses")
            if dias_max_atraso >= 60:
                factores_riesgo.append(f"- Atrasos significativos: hasta {dias_max_atraso} días")
            if tipo_empleo == 'Desempleado':
                factores_riesgo.append("- Situación laboral: Desempleado")
            if antiguedad_laboral < 12:
                factores_riesgo.append(f"- Antigüedad laboral baja: {antiguedad_laboral} meses")

            for factor in factores_riesgo:
                st.markdown(factor)

        else:
            st.success("**✅ Perfil Crediticio Favorable:**")
            factores_positivos = []

            if ratio_deuda_ingreso <= 40:
                factores_positivos.append(f"- Ratio deuda/ingreso saludable: {ratio_deuda_ingreso:.1f}%")
            if utilizacion_credito <= 30:
                factores_positivos.append(f"- Baja utilización de crédito: {utilizacion_credito}%")
            if num_pagos_atrasados == 0:
                factores_positivos.append("- Sin pagos atrasados recientes")
            if tiene_propiedad:
                factores_positivos.append(f"- Propietario de vivienda (${valor_propiedad:,.0f})")
            if antiguedad_laboral >= 24:
                factores_positivos.append(f"- Estabilidad laboral: {antiguedad_laboral} meses")

            for factor in factores_positivos:
                st.markdown(factor)


def show_portfolio_analysis(df, model, scaler):
    """Análisis de portfolio"""
    st.header("📈 Análisis de Portfolio")

    # Filtros
    st.sidebar.markdown("### Filtros")
    min_score = st.sidebar.slider("Credit Score Mínimo", 300, 850, 300)
    max_score = st.sidebar.slider("Credit Score Máximo", 300, 850, 850)

    df_filtered = df[(df['credit_score'] >= min_score) & (df['credit_score'] <= max_score)]

    # Métricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clientes Filtrados", f"{len(df_filtered):,}")

    with col2:
        default_rate = df_filtered['default'].mean() * 100
        st.metric("Tasa Default", f"{default_rate:.2f}%")

    with col3:
        monto_total = df_filtered['monto_solicitado'].sum()
        st.metric("Monto Total Solicitado", f"${monto_total/1e6:.1f}M")

    with col4:
        exposicion_riesgo = df_filtered[df_filtered['default'] == 1]['monto_solicitado'].sum()
        st.metric("Exposición en Riesgo", f"${exposicion_riesgo/1e6:.1f}M")

    # Matriz de riesgo
    st.subheader("Matriz de Riesgo por Segmento")

    df_filtered['segmento_score'] = pd.cut(
        df_filtered['credit_score'],
        bins=[300, 580, 670, 740, 850],
        labels=['Muy Bajo', 'Bajo', 'Bueno', 'Excelente']
    )

    df_filtered['segmento_ingreso'] = pd.cut(
        df_filtered['ingreso_mensual'],
        bins=[0, 2000, 5000, 10000, 100000],
        labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
    )

    matriz = df_filtered.groupby(['segmento_score', 'segmento_ingreso']).agg({
        'default': 'mean',
        'monto_solicitado': 'sum'
    }).reset_index()

    matriz['default_pct'] = matriz['default'] * 100

    fig = px.density_heatmap(
        matriz,
        x='segmento_ingreso',
        y='segmento_score',
        z='default_pct',
        title="Tasa de Default por Segmento (%)",
        color_continuous_scale='RdYlGn_r',
        labels={'default_pct': 'Default %'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_feature_analysis(df):
    """Análisis de features"""
    st.header("🎯 Análisis de Features")

    # Correlaciones
    st.subheader("Correlación con Default")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('default')

    correlations = df[numeric_cols + ['default']].corr()['default'].drop('default').sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=correlations.values[:15],
        y=correlations.index[:15],
        orientation='h',
        marker_color=['red' if x > 0 else 'green' for x in correlations.values[:15]]
    ))
    fig.update_layout(
        title="Top 15 Features - Correlación con Default",
        xaxis_title="Correlación",
        yaxis_title="Feature",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribuciones
    st.subheader("Distribuciones de Variables Clave")

    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox("Seleccione variable:", numeric_cols)

    with col2:
        log_scale = st.checkbox("Escala logarítmica")

    fig = px.histogram(
        df,
        x=feature,
        color='default',
        nbins=50,
        log_y=log_scale,
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'default': 'Default'},
        title=f"Distribución de {feature}"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
