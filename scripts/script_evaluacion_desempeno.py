import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

LABELS_MAP = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusión (F)"}

class EvaluacionDesempeno:
    def __init__(self, y_true, y_pred, nombre_experimento='modelo_hibrido', output_dir='resultados'):
        self.nombre_experimento = nombre_experimento
        self.output_dir = output_dir
        self.metricas = {}
        # Asegurar directorio
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Manejo robusto de etiquetas (one-hot o índice)
        y_true_arr = np.array(y_true)
        self.y_true = np.argmax(y_true_arr, axis=1) if y_true_arr.ndim > 1 and y_true_arr.shape[1] > 1 else y_true_arr
        self.y_pred = np.array(y_pred)
        print("="*80 + "\nSCRIPT 3: EVALUACIÓN DE DESEMPEÑO DEL MODELO\n" + "="*80)

    def calcular_metricas(self):
        print("\n[CALCULANDO MÉTRICAS DE DESEMPEÑO]")
        report = classification_report(self.y_true, self.y_pred, target_names=LABELS_MAP.values(), output_dict=True, zero_division=0)
        cm = confusion_matrix(self.y_true, self.y_pred, labels=list(LABELS_MAP.keys()))
        
        metricas_clase = []
        for i, nombre in LABELS_MAP.items():
            TP, FP, FN, TN = int(cm[i, i]), int(cm[:, i].sum() - cm[i, i]), int(cm[i, :].sum() - cm[i, i]), int(cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i]))
            # La línea mágica que combina todo, incluyendo el reporte de sklearn
            metricas_clase.append({'clase': nombre, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, **report.get(nombre, {})})

        self.metricas = {
            'fecha_registro': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'experimento': self.nombre_experimento,
            'indicadores': {
                'C1_precision_weighted_porcentaje': round(report['weighted avg']['precision'] * 100, 2),
                'C2_exactitud_porcentaje': round(accuracy_score(self.y_true, self.y_pred) * 100, 2),
                'C3_recall_weighted_porcentaje': round(report['weighted avg']['recall'] * 100, 2),
                'C4_f1_score_weighted_porcentaje': round(report['weighted avg']['f1-score'] * 100, 2)
            },
            'metricas_por_clase': metricas_clase, 'matriz_confusion': cm.tolist()
        }
        print("\n[CÁLCULO DE MÉTRICAS COMPLETADO]")
        return self.metricas

    def generar_ficha_desempeno(self):
        ruta = os.path.join(self.output_dir, 'ficha_desempeno.json')
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(self.metricas, f, indent=4, ensure_ascii=False)
        print(f"\n[OK] Ficha de desempeño guardada en: {ruta}")

    def generar_tabla_excel(self):
        ruta = os.path.join(self.output_dir, 'tabla_desempeno.xlsx')
        indicadores = self.metricas['indicadores']
        datos = {
            'Dimensión': ['C) Desempeño del modelo'] * 4,
            'Indicador': ['C.1) Precisión', 'C.2) Exactitud', 'C.3) Recall', 'C.4) F1-score'],
            'Valor (%)': [indicadores['C1_precision_weighted_porcentaje'], indicadores['C2_exactitud_porcentaje'], indicadores['C3_recall_weighted_porcentaje'], indicadores['C4_f1_score_weighted_porcentaje']]
        }
        with pd.ExcelWriter(ruta, engine='openpyxl') as writer:
            pd.DataFrame(datos).to_excel(writer, sheet_name='Indicadores Globales', index=False)
            pd.DataFrame(self.metricas['metricas_por_clase']).to_excel(writer, sheet_name='Metricas Por Clase', index=False)
        print(f"[OK] Tabla Excel de desempeño guardada en: {ruta}")

    def generar_graficos(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        
        cm = np.array(self.metricas['matriz_confusion'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_MAP.values(), yticklabels=LABELS_MAP.values(), ax=ax)
        ax.set_title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicción', fontsize=12); ax.set_ylabel('Valor Real', fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'matriz_confusion.png'), dpi=300); plt.close(fig)
        print(f"[OK] Gráfico de Matriz de Confusión guardado.")
        
        df = pd.DataFrame(self.metricas['metricas_por_clase'])
        df_melted = df.melt(id_vars='clase', value_vars=['precision', 'recall', 'f1-score'], var_name='Métrica', value_name='Valor')
        df_melted['Valor'] *= 100
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_melted, x='clase', y='Valor', hue='Métrica', palette='viridis', ax=ax)
        ax.set_title('Métricas de Desempeño por Clase', fontsize=16, fontweight='bold')
        ax.set_xlabel('Clase de Arritmia', fontsize=12); ax.set_ylabel('Porcentaje (%)', fontsize=12)
        ax.set_ylim(0, 105); ax.legend(title='Métrica')
        for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'metricas_por_clase.png'), dpi=300); plt.close(fig)
        print(f"[OK] Gráfico de Métricas por Clase guardado.")

    def imprimir_resumen(self):
        print("\n" + "="*80 + "\nRESUMEN DE DESEMPEÑO DEL MODELO\n" + "="*80)
        print(f"Experimento: {self.metricas['experimento']}\nFecha: {self.metricas['fecha_registro']}")
        print("\n[INDICADORES GLOBALES (PONDERADOS)]")
        for k, v in self.metricas['indicadores'].items(): print(f"  {k[:3]}) {k[4:-11].replace('_',' ').capitalize()}:  {v:.2f}%")
        print("\n[DESEMPEÑO DETALLADO POR CLASE]")
        df = pd.DataFrame(self.metricas['metricas_por_clase'])
        for col in ['precision', 'recall', 'f1-score']: 
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
        print(df[['clase', 'precision', 'recall', 'f1-score', 'support']].to_string(index=False) + "\n" + "="*80)

    def run(self):
        self.calcular_metricas()
        self.generar_ficha_desempeno()
        self.generar_tabla_excel()
        self.generar_graficos()
        return self.metricas