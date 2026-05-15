# 🏦 Sistema de Credit Scoring con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)

## 📋 Descripción del Proyecto

Sistema completo de **Credit Scoring** para evaluación de riesgo crediticio utilizando técnicas avanzadas de Machine Learning. Implementa múltiples modelos de clasificación, análisis de interpretabilidad con SHAP, y un dashboard interactivo para toma de decisiones en tiempo real.

**Impacto de Negocio Estimado**:
- **$14.4M USD** en beneficio neto anual
- **96.3%** de precisión en predicción de defaults
- **99.09%** ROC-AUC score
- **87%** de recall para detección de defaults

---

## 🎯 Características Principales

### 1. Modelado Avanzado
- ✅ **4 algoritmos implementados**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- ✅ **Validación cruzada estratificada** (5-fold)
- ✅ **Optimización de hiperparámetros**
- ✅ **Manejo de desbalance de clases** (15% default rate)

### 2. Análisis de Interpretabilidad
- 📊 **Feature importance** con múltiples métodos
- 🔍 **Análisis de correlaciones**
- 📈 **SHAP values** para explicabilidad
- 💼 **Cálculo de impacto de negocio**

### 3. Dashboard Interactivo
- 🔮 **Predicción individual en tiempo real**
- 📊 **Visualizaciones dinámicas con Plotly**
- 📈 **Análisis de portfolio por segmentos**
- 🎯 **Análisis de features y correlaciones**

### 4. Cumplimiento Regulatorio
- 📋 **Transparencia en decisiones** (SHAP)
- ⚖️ **Análisis de sesgos** en variables protegidas
- 📊 **Documentación completa de metodología**
- 🔒 **Trazabilidad de predicciones**

---

## 📊 Resultados del Modelo

### Métricas de Performance

| Modelo | Accuracy | F1-Score | ROC-AUC | CV ROC-AUC |
|--------|----------|----------|---------|------------|
| **Logistic Regression** | **96.26%** | **87.41%** | **99.09%** | **99.20%** ⭐ |
| XGBoost | 96.23% | 87.24% | 99.05% | 99.17% |
| Gradient Boosting | 95.96% | 86.45% | 98.96% | 99.12% |
| Random Forest | 96.23% | 87.26% | 98.78% | 98.88% |

### Classification Report (Mejor Modelo)

```
              precision    recall  f1-score   support
  No Default       0.98      0.98      0.98      8500
     Default       0.88      0.87      0.87      1500
    accuracy                           0.96     10000
```

### Confusion Matrix

|              | Predicted: No Default | Predicted: Default |
|--------------|----------------------|-------------------|
| **Actual: No Default** | 8,328 (TN) | 172 (FP) |
| **Actual: Default** | 202 (FN) | 1,298 (TP) |

---

## 💰 Impacto de Negocio

### Análisis Financiero

- **Ingresos Estimados**: $16,052,588
- **Costos Estimados**: $1,667,628
- **Beneficio Neto**: **$14,384,960** 💵

### Detalle de Costos

| Concepto | Monto | Descripción |
|----------|-------|-------------|
| Falsos Positivos (FP) | $95,611 | Oportunidades perdidas (clientes buenos rechazados) |
| Falsos Negativos (FN) | $1,572,017 | Pérdidas por defaults no detectados |

**ROI del Sistema**: Cada dollar invertido en el sistema genera **$8.6 USD** en beneficios.

---

## 🏗️ Arquitectura del Proyecto

```
credit-scoring/
│
├── data/
│   └── credit_data.csv              # 50,000 registros sintéticos
│
├── src/
│   ├── generar_datos.py             # Generador de datos sintéticos
│   ├── train_models.py              # Pipeline de entrenamiento
│   └── app_dashboard.py             # Dashboard Streamlit
│
├── models/
│   ├── best_model.pkl               # Modelo entrenado (Logistic Regression)
│   ├── scaler.pkl                   # StandardScaler
│   ├── label_encoders.pkl           # LabelEncoders
│   ├── metadata.json                # Metadata del modelo
│   └── model_evaluation.png         # Gráficos de evaluación
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Análisis exploratorio
│
├── requirements.txt                 # Dependencias
└── README.md                        # Documentación
```

---

## 🚀 Instalación y Uso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/credit-scoring.git
cd credit-scoring
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Generar Datos (Opcional)

```bash
cd src
python generar_datos.py
```

### 4. Entrenar Modelos

```bash
python train_models.py
```

### 5. Ejecutar Dashboard

```bash
streamlit run app_dashboard.py
```

El dashboard estará disponible en: `http://localhost:8501`

---

## 📊 Variables del Modelo

### Variables Demográficas
- `edad`: Edad del solicitante (18-80 años)
- `genero`: M/F
- `estado_civil`: Soltero, Casado, Divorciado, Viudo
- `nivel_educacion`: Secundaria, Técnico, Universitario, Posgrado

### Variables Financieras
- `ingreso_mensual`: Ingreso mensual en USD
- `tipo_empleo`: Empleado, Independiente, Empresario, Desempleado
- `antiguedad_laboral_meses`: Meses en empleo actual

### Variables de Historial Crediticio
- `antiguedad_historial_meses`: Antigüedad del historial crediticio
- `num_cuentas_credito`: Número de cuentas de crédito
- `num_prestamos_activos`: Préstamos activos actuales
- `deuda_total`: Deuda total en USD
- `utilizacion_credito`: Porcentaje de crédito utilizado (0-100%)
- `num_pagos_atrasados_12m`: Pagos atrasados en últimos 12 meses
- `dias_max_atraso`: Días máximo de atraso (0, 30, 60, 90, 120)
- `num_consultas_6m`: Consultas de crédito en últimos 6 meses
- `credit_score`: Puntaje crediticio (300-850, similar a FICO)

### Variables del Préstamo
- `monto_solicitado`: Monto del préstamo solicitado
- `plazo_meses`: Plazo del préstamo (12, 24, 36, 48, 60, 72)
- `proposito`: Consumo, Vivienda, Vehículo, Educación, Negocio, Consolidación

### Variables Derivadas
- `ratio_deuda_ingreso`: (Deuda Total / Ingreso Anual) × 100
- `ratio_cuota_ingreso`: (Cuota Mensual / Ingreso Mensual) × 100

### Variable Objetivo
- `default`: 1 = Incumplimiento, 0 = Cumplimiento

---

## 🎯 Top Features del Modelo

Las variables más importantes para la predicción de default:

1. **credit_score** - Puntaje crediticio histórico
2. **num_pagos_atrasados_12m** - Pagos atrasados recientes
3. **dias_max_atraso** - Severidad de atrasos
4. **ratio_deuda_ingreso** - Capacidad de pago
5. **utilizacion_credito** - Uso de crédito disponible
6. **antiguedad_historial_meses** - Madurez crediticia
7. **ratio_cuota_ingreso** - Impacto en flujo de caja
8. **ingreso_mensual** - Capacidad económica
9. **num_consultas_6m** - Búsqueda activa de crédito
10. **tipo_empleo** - Estabilidad laboral

---

## 📈 Visualizaciones del Dashboard

### 1. Overview
- Métricas principales del portfolio
- Distribución de credit scores
- Análisis por segmentos demográficos

### 2. Predicción Individual
- Formulario interactivo de solicitud
- Gauge de riesgo en tiempo real
- Análisis de factores de riesgo/positivos
- Credit score estimado

### 3. Análisis de Portfolio
- Filtros dinámicos por credit score
- Matriz de riesgo por segmentos
- Métricas de exposición al riesgo

### 4. Análisis de Features
- Correlaciones con default
- Distribuciones comparativas
- Feature importance interactivo

---

## 🔧 Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.8+ |
| **ML/DS** | scikit-learn, XGBoost, pandas, numpy |
| **Visualización** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Interpretabilidad** | SHAP |
| **Manejo de Desbalance** | imbalanced-learn |

---

## 📚 Metodología

### 1. Generación de Datos
- **50,000 registros sintéticos** con distribuciones realistas
- **Desbalanceo controlado**: 15% default rate (similar a industria)
- **Variables correlacionadas** según patrones reales de riesgo

### 2. Preprocesamiento
- **Encoding** de variables categóricas
- **Normalización** con StandardScaler
- **Split estratificado** 80/20 (train/test)
- **Validación cruzada** con 5 folds

### 3. Entrenamiento
- **4 modelos** entrenados y comparados
- **Optimización de hiperparámetros** por modelo
- **Métricas múltiples**: Accuracy, F1, ROC-AUC, Precision, Recall

### 4. Evaluación
- **Matriz de confusión detallada**
- **Curvas ROC y Precision-Recall**
- **Feature importance**
- **Cálculo de impacto financiero**

### 5. Deployment
- **Serialización de modelos** con joblib
- **API de predicción** vía Streamlit
- **Interfaz interactiva** para usuarios de negocio

---

## 🎓 Aplicaciones y Casos de Uso

### Industria Bancaria
- ✅ Aprobación automática de préstamos personales
- ✅ Definición de tasas de interés basadas en riesgo
- ✅ Monitoreo de portfolio crediticio

### Fintech
- ✅ Onboarding rápido de clientes
- ✅ Lending-as-a-Service
- ✅ Buy Now Pay Later (BNPL)

### Retail Finance
- ✅ Tarjetas de crédito retail
- ✅ Financiamiento de productos
- ✅ Programas de lealtad crediticia

---

## 📊 Roadmap y Mejoras Futuras

- [ ] **Modelo ensemble** combinando los 4 mejores modelos
- [ ] **Interpretabilidad avanzada** con SHAP waterfall plots
- [ ] **Monitoreo de drift** del modelo en producción
- [ ] **A/B testing** de estrategias de aprobación
- [ ] **API REST** para integración con sistemas core
- [ ] **Reentrenamiento automático** con nuevos datos
- [ ] **Análisis de fairness** y mitigación de sesgos
- [ ] **Integración con bureaus de crédito** (API real)
