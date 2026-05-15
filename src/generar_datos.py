import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generar_datos_credit_scoring(n_samples=50000):
    """
    Genera dataset sintético realista para credit scoring
    """

    # Variables demográficas
    edad = np.random.normal(40, 15, n_samples).clip(18, 80).astype(int)

    genero = np.random.choice(['M', 'F'], n_samples)

    estado_civil = np.random.choice(
        ['Soltero', 'Casado', 'Divorciado', 'Viudo'],
        n_samples,
        p=[0.35, 0.45, 0.15, 0.05]
    )

    nivel_educacion = np.random.choice(
        ['Secundaria', 'Técnico', 'Universitario', 'Posgrado'],
        n_samples,
        p=[0.25, 0.30, 0.35, 0.10]
    )

    # Variables financieras
    ingreso_mensual = np.random.lognormal(8.5, 0.8, n_samples).clip(800, 50000)

    # Historial crediticio
    antiguedad_historial_meses = np.random.exponential(60, n_samples).clip(0, 360).astype(int)

    num_cuentas_credito = np.random.poisson(3, n_samples).clip(0, 15)

    num_prestamos_activos = np.random.poisson(1.5, n_samples).clip(0, 8)

    # Deuda total (relacionada con ingresos)
    ratio_deuda_base = np.random.beta(2, 5, n_samples)
    deuda_total = ingreso_mensual * ratio_deuda_base * np.random.uniform(5, 20, n_samples)

    # Utilización de crédito (0-100%)
    utilizacion_credito = np.random.beta(2, 5, n_samples) * 100

    # Pagos atrasados
    num_pagos_atrasados_12m = np.random.poisson(0.8, n_samples).clip(0, 12)

    dias_max_atraso = np.where(
        num_pagos_atrasados_12m > 0,
        np.random.choice([30, 60, 90, 120], n_samples),
        0
    )

    # Consultas de crédito recientes
    num_consultas_6m = np.random.poisson(2, n_samples).clip(0, 20)

    # Tipo de empleo
    tipo_empleo = np.random.choice(
        ['Empleado', 'Independiente', 'Empresario', 'Desempleado'],
        n_samples,
        p=[0.65, 0.20, 0.10, 0.05]
    )

    # Antigüedad laboral
    antiguedad_laboral_meses = np.where(
        tipo_empleo != 'Desempleado',
        np.random.exponential(48, n_samples).clip(0, 360),
        0
    ).astype(int)

    # Valor de propiedad (si tiene)
    tiene_propiedad = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    valor_propiedad = np.where(
        tiene_propiedad == 1,
        np.random.lognormal(11.5, 0.6, n_samples).clip(30000, 500000),
        0
    )

    # Tiene vehículo
    tiene_vehiculo = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])

    # Monto del préstamo solicitado
    monto_solicitado = np.random.lognormal(9, 0.8, n_samples).clip(1000, 100000)

    # Plazo del préstamo (meses)
    plazo_meses = np.random.choice([12, 24, 36, 48, 60, 72], n_samples)

    # Propósito del préstamo
    proposito = np.random.choice(
        ['Consumo', 'Vivienda', 'Vehiculo', 'Educacion', 'Negocio', 'Consolidacion'],
        n_samples,
        p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10]
    )

    # Crear DataFrame
    df = pd.DataFrame({
        'edad': edad,
        'genero': genero,
        'estado_civil': estado_civil,
        'nivel_educacion': nivel_educacion,
        'ingreso_mensual': ingreso_mensual.round(2),
        'tipo_empleo': tipo_empleo,
        'antiguedad_laboral_meses': antiguedad_laboral_meses,
        'antiguedad_historial_meses': antiguedad_historial_meses,
        'num_cuentas_credito': num_cuentas_credito,
        'num_prestamos_activos': num_prestamos_activos,
        'deuda_total': deuda_total.round(2),
        'utilizacion_credito': utilizacion_credito.round(2),
        'num_pagos_atrasados_12m': num_pagos_atrasados_12m,
        'dias_max_atraso': dias_max_atraso,
        'num_consultas_6m': num_consultas_6m,
        'tiene_propiedad': tiene_propiedad,
        'valor_propiedad': valor_propiedad.round(2),
        'tiene_vehiculo': tiene_vehiculo,
        'monto_solicitado': monto_solicitado.round(2),
        'plazo_meses': plazo_meses,
        'proposito': proposito
    })

    # Calcular variables derivadas
    df['ratio_deuda_ingreso'] = (df['deuda_total'] / (df['ingreso_mensual'] * 12) * 100).round(2)
    df['ratio_cuota_ingreso'] = ((df['monto_solicitado'] / df['plazo_meses']) / df['ingreso_mensual'] * 100).round(2)

    # GENERAR VARIABLE OBJETIVO: default (1 = incumplimiento, 0 = cumplimiento)
    # Modelo de scoring basado en factores de riesgo
    score_riesgo = (
        -0.3 * (df['edad'] < 25).astype(int)
        - 0.5 * (df['tipo_empleo'] == 'Desempleado').astype(int)
        - 0.4 * (df['tipo_empleo'] == 'Independiente').astype(int)
        - 0.01 * df['antiguedad_laboral_meses']
        - 0.005 * df['antiguedad_historial_meses']
        + 0.3 * (df['num_pagos_atrasados_12m'] > 2).astype(int)
        + 0.4 * (df['dias_max_atraso'] >= 90).astype(int)
        + 0.02 * (df['utilizacion_credito'] - 50)
        + 0.01 * (df['ratio_deuda_ingreso'] - 40)
        + 0.02 * (df['ratio_cuota_ingreso'] - 30)
        + 0.2 * (df['num_consultas_6m'] > 5).astype(int)
        - 0.3 * df['tiene_propiedad']
        - 0.2 * df['tiene_vehiculo']
        + np.random.normal(0, 0.5, n_samples)  # Ruido aleatorio
    )

    # Convertir score a probabilidad
    prob_default = 1 / (1 + np.exp(-score_riesgo))

    # Generar target con desbalanceo realista (15% default)
    df['default'] = (prob_default > np.percentile(prob_default, 85)).astype(int)

    # Credit score (300-850, similar a FICO)
    df['credit_score'] = (
        300 +
        550 * (1 - prob_default) +
        np.random.normal(0, 20, n_samples)
    ).clip(300, 850).round(0).astype(int)

    return df

if __name__ == "__main__":
    print("Generando dataset de Credit Scoring...")
    df = generar_datos_credit_scoring(50000)

    # Guardar datos
    df.to_csv('../data/credit_data.csv', index=False)

    print(f"\n[OK] Dataset generado: {len(df)} registros")
    print(f"\nDistribucion de la variable objetivo:")
    print(df['default'].value_counts())
    print(f"\nTasa de default: {df['default'].mean()*100:.2f}%")

    print(f"\nEstadisticas descriptivas:")
    print(df.describe())

    print(f"\nDatos guardados en: data/credit_data.csv")
