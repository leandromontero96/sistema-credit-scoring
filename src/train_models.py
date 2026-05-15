import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """Cargar datos"""
        print("Cargando datos...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset cargado: {self.df.shape}")
        return self

    def preprocess_data(self):
        """Preprocesar datos para modelado"""
        print("\nPreprocesando datos...")

        # Codificar variables categóricas
        categorical_cols = ['genero', 'estado_civil', 'nivel_educacion', 'tipo_empleo', 'proposito']

        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        # Features para el modelo
        feature_cols = [
            'edad', 'ingreso_mensual', 'antiguedad_laboral_meses',
            'antiguedad_historial_meses', 'num_cuentas_credito',
            'num_prestamos_activos', 'deuda_total', 'utilizacion_credito',
            'num_pagos_atrasados_12m', 'dias_max_atraso', 'num_consultas_6m',
            'tiene_propiedad', 'valor_propiedad', 'tiene_vehiculo',
            'monto_solicitado', 'plazo_meses', 'ratio_deuda_ingreso',
            'ratio_cuota_ingreso', 'credit_score',
            'genero_encoded', 'estado_civil_encoded', 'nivel_educacion_encoded',
            'tipo_empleo_encoded', 'proposito_encoded'
        ]

        X = self.df[feature_cols]
        y = self.df['default']

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalar features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=feature_cols,
            index=self.X_train.index
        )

        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=feature_cols,
            index=self.X_test.index
        )

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        print(f"Default rate - Train: {self.y_train.mean():.2%}, Test: {self.y_test.mean():.2%}")

        return self

    def train_models(self):
        """Entrenar múltiples modelos"""
        print("\n" + "="*60)
        print("ENTRENANDO MODELOS")
        print("="*60)

        # Definir modelos
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }

        # Entrenar y evaluar cada modelo
        for name, model in self.models.items():
            print(f"\n{name}...")
            model.fit(self.X_train, self.y_train)

            # Predicciones
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )

            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self

    def evaluate_best_model(self):
        """Evaluación detallada del mejor modelo"""
        print("\n" + "="*60)
        print("EVALUACION DEL MEJOR MODELO")
        print("="*60)

        # Seleccionar mejor modelo por ROC-AUC
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        best_result = self.results[best_model_name]

        print(f"\nMejor modelo: {best_model_name}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test,
            best_result['y_pred'],
            target_names=['No Default', 'Default']
        ))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_result['y_pred'])
        print("\nConfusion Matrix:")
        print(cm)

        # Calcular costos de negocio
        self.calculate_business_impact(cm, best_model_name)

        return self

    def calculate_business_impact(self, cm, model_name):
        """Calcular impacto de negocio"""
        print("\n" + "="*60)
        print("IMPACTO DE NEGOCIO")
        print("="*60)

        TN, FP, FN, TP = cm.ravel()

        # Supuestos de negocio
        monto_promedio_prestamo = self.df['monto_solicitado'].mean()
        tasa_interes = 0.15
        tasa_recuperacion = 0.30  # Recuperación en caso de default

        # Costos
        costo_FP = FP * (monto_promedio_prestamo * 0.05)  # Oportunidad perdida
        costo_FN = FN * (monto_promedio_prestamo * (1 - tasa_recuperacion))  # Pérdida por default

        ingreso_TP = TP * (monto_promedio_prestamo * tasa_interes)  # Evitar defaults
        ingreso_TN = TN * (monto_promedio_prestamo * tasa_interes)  # Préstamos buenos

        ingreso_total = ingreso_TP + ingreso_TN
        costo_total = costo_FP + costo_FN
        beneficio_neto = ingreso_total - costo_total

        print(f"\nMonto promedio de prestamo: ${monto_promedio_prestamo:,.2f}")
        print(f"\nIngresos estimados: ${ingreso_total:,.2f}")
        print(f"Costos estimados: ${costo_total:,.2f}")
        print(f"Beneficio neto: ${beneficio_neto:,.2f}")

        print(f"\nDetalle de costos:")
        print(f"  Falsos Positivos (oportunidades perdidas): ${costo_FP:,.2f}")
        print(f"  Falsos Negativos (defaults no detectados): ${costo_FN:,.2f}")

    def plot_results(self):
        """Generar gráficos de evaluación"""
        print("\nGenerando graficos...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. ROC Curves
        ax = axes[0, 0]
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax = axes[0, 1]
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            ax.plot(recall, precision, label=name)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Feature Importance (mejor modelo)
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        best_model = self.results[best_model_name]['model']

        if hasattr(best_model, 'feature_importances_'):
            ax = axes[1, 0]
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15

            ax.barh(range(len(indices)), importances[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(self.X_train.columns[indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 15 Features - {best_model_name}')
            ax.grid(True, alpha=0.3)

        # 4. Comparación de modelos
        ax = axes[1, 1]
        model_names = list(self.results.keys())
        metrics = {
            'Accuracy': [self.results[m]['accuracy'] for m in model_names],
            'F1-Score': [self.results[m]['f1_score'] for m in model_names],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in model_names]
        }

        x = np.arange(len(model_names))
        width = 0.25

        for i, (metric, values) in enumerate(metrics.items()):
            ax.bar(x + i*width, values, width, label=metric)

        ax.set_xlabel('Modelos')
        ax.set_ylabel('Score')
        ax.set_title('Comparacion de Metricas')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../models/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("Graficos guardados en: models/model_evaluation.png")

        return self

    def save_models(self):
        """Guardar modelos entrenados"""
        print("\nGuardando modelos...")

        # Guardar mejor modelo
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        best_model = self.results[best_model_name]['model']

        joblib.dump(best_model, '../models/best_model.pkl')
        joblib.dump(self.scaler, '../models/scaler.pkl')
        joblib.dump(self.label_encoders, '../models/label_encoders.pkl')

        # Guardar metadata
        metadata = {
            'best_model': best_model_name,
            'roc_auc': self.results[best_model_name]['roc_auc'],
            'features': list(self.X_train.columns)
        }

        pd.DataFrame([metadata]).to_json('../models/metadata.json', orient='records')

        print(f"Mejor modelo ({best_model_name}) guardado en: models/best_model.pkl")

        return self


def main():
    # Pipeline completo
    model = CreditScoringModel('../data/credit_data.csv')

    model.load_data() \
         .preprocess_data() \
         .train_models() \
         .evaluate_best_model() \
         .plot_results() \
         .save_models()

    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
