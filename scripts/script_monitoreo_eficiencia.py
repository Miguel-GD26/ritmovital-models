import time
import json
import pandas as pd
from datetime import datetime
import psutil
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import os

try:
    import GPUtil
    GPU_DISPONIBLE = True
except ImportError:
    GPU_DISPONIBLE = False

class MonitorEficienciaCallback(Callback):
    def __init__(self, nombre_experimento='modelo_hibrido', output_dir='resultados'):
        super().__init__()
        self.nombre_experimento = nombre_experimento
        self.output_dir = output_dir
        self.tiempo_inicio_entrenamiento = 0
        self.historial_epocas = []
        self.metricas_finales = {}
        # Asegurar directorio
        os.makedirs(self.output_dir, exist_ok=True)
        print("="*80)
        print("SCRIPT 2: MONITOR DE EFICIENCIA INICIALIZADO")
        print("="*80)

    def on_train_begin(self, logs=None):
        self.tiempo_inicio_entrenamiento = time.time()
        print(f"\n[MONITOR] Entrenamiento iniciado en {datetime.now().strftime('%H:%M:%S')}. Monitoreando eficiencia (CPU/GPU)...")

    def on_epoch_end(self, epoch, logs=None):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        gpu_load, gpu_memory = 0.0, 0.0
        if GPU_DISPONIBLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_load = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUtil * 100
            except Exception:
                pass
        
        epoca_info = {'epoca': epoch + 1, 'cpu_percent': cpu_percent, 'gpu_load_percent': gpu_load, 'gpu_memory_percent': gpu_memory}
        if logs:
            logs_cleaned = {k: round(v, 4) for k, v in logs.items()}
            epoca_info.update(logs_cleaned)
        
        self.historial_epocas.append(epoca_info)
        print(f"Época {epoch + 1}: CPU: {cpu_percent:.1f}% | GPU Load: {gpu_load:.1f}% | loss: {logs.get('loss', 0):.4f} | val_loss: {logs.get('val_loss', 0):.4f}")

    def on_train_end(self, logs=None):
        tiempo_fin_entrenamiento = time.time()
        tiempo_total_seg = tiempo_fin_entrenamiento - self.tiempo_inicio_entrenamiento
        
        df_historial = pd.DataFrame(self.historial_epocas)
        cpu_promedio = df_historial['cpu_percent'].mean() if not df_historial.empty else 0
        gpu_load_promedio = df_historial['gpu_load_percent'].mean() if GPU_DISPONIBLE and not df_historial.empty else 0.0
        gpu_memory_promedio = df_historial['gpu_memory_percent'].mean() if GPU_DISPONIBLE and not df_historial.empty else 0.0

        print("\n" + "="*80)
        print("MONITOREO DE EFICIENCIA COMPLETADO")
        print("="*80)
        
        self.metricas_finales = {
            'fecha_registro': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experimento': self.nombre_experimento,
            'indicadores': {
                'B1_tiempo_entrenamiento_segundos': round(tiempo_total_seg, 2),
                'B1_tiempo_entrenamiento_minutos': round(tiempo_total_seg / 60, 2),
                'B2_cpu_promedio_porcentaje': round(cpu_promedio, 2),
                'B3_gpu_load_promedio_porcentaje': round(gpu_load_promedio, 2),
                'B3_gpu_memory_promedio_porcentaje': round(gpu_memory_promedio, 2)
            },
            'num_epocas': len(self.historial_epocas),
            'historial_epocas': self.historial_epocas
        }
        self.generar_ficha_eficiencia()
        self.generar_tabla_excel()
        self.generar_graficos()
        self.imprimir_resumen()

    def generar_ficha_eficiencia(self):
        ruta_salida = os.path.join(self.output_dir, 'ficha_eficiencia.json')
        metricas_serializables = json.loads(pd.Series(self.metricas_finales).to_json())
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(metricas_serializables, f, indent=4, ensure_ascii=False)
        print(f"\n[OK] Ficha de eficiencia guardada en: {ruta_salida}")

    def generar_tabla_excel(self):
        ruta_salida = os.path.join(self.output_dir, 'tabla_eficiencia.xlsx')
        datos_indicadores = {
            'Variable': ['VI: Método basado en aprendizaje profundo'] * 3,
            'Dimensión': ['B) Eficiencia'] * 3,
            'Indicador': ['B.1) Tiempo de entrenamiento (min)', 'B.2) Uso promedio de CPU (%)', 'B.3) Uso promedio de GPU (%)'],
            'Valor': [f"{self.metricas_finales['indicadores']['B1_tiempo_entrenamiento_minutos']:.2f}",
                      f"{self.metricas_finales['indicadores']['B2_cpu_promedio_porcentaje']:.2f}%", 
                      f"{self.metricas_finales['indicadores']['B3_gpu_load_promedio_porcentaje']:.2f}%"],
            'Instrumento': ['Callback de Keras'] * 3,
            'Fecha': [self.metricas_finales['fecha_registro']] * 3
        }
        df_indicadores = pd.DataFrame(datos_indicadores)
        df_historial = pd.DataFrame(self.metricas_finales['historial_epocas'])
        with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
            df_indicadores.to_excel(writer, sheet_name='Indicadores', index=False)
            df_historial.to_excel(writer, sheet_name='Historial_Por_Epoca', index=False)
        print(f"[OK] Tabla Excel de eficiencia guardada en: {ruta_salida}")

    def generar_graficos(self):
        ruta_salida = os.path.join(self.output_dir, 'monitoreo_cpu_gpu.png')
        df = pd.DataFrame(self.metricas_finales['historial_epocas'])
        if df.empty:
            print("[ADVERTENCIA] No hay datos de historial para generar gráficos.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        ax1.plot(df['epoca'], df['cpu_percent'], marker='o', linestyle='-', color='#3498db', label='Uso CPU')
        ax1.axhline(y=self.metricas_finales['indicadores']['B2_cpu_promedio_porcentaje'], color='red', linestyle='--', linewidth=2, label=f"Promedio: {self.metricas_finales['indicadores']['B2_cpu_promedio_porcentaje']:.1f}%")
        ax1.set_title('Uso de CPU durante Entrenamiento (Indicador B.2)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('CPU (%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        if GPU_DISPONIBLE and 'gpu_load_percent' in df.columns and df['gpu_load_percent'].max() > 0:
            ax2.plot(df['epoca'], df['gpu_load_percent'], marker='s', linestyle='-', color='#e74c3c', label='Uso GPU')
            ax2.axhline(y=self.metricas_finales['indicadores']['B3_gpu_load_promedio_porcentaje'], color='red', linestyle='--', linewidth=2, label=f"Promedio: {self.metricas_finales['indicadores']['B3_gpu_load_promedio_porcentaje']:.1f}%")
            ax2.set_title('Uso de GPU durante Entrenamiento (Indicador B.3)', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'GPU no disponible o no utilizada', ha='center', va='center', fontsize=14, color='gray')
            ax2.set_title('Uso de GPU', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('GPU Load (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        if 'gpu_load_percent' in df.columns and df['gpu_load_percent'].max() > 0:
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=300)
        plt.close(fig)
        print(f"[OK] Gráfico de monitoreo guardado en: {ruta_salida}")

    def imprimir_resumen(self):
        print("\n[INDICADORES TABLA I - DIMENSIÓN B]")
        print(f"  B.1) Tiempo de entrenamiento: {self.metricas_finales['indicadores']['B1_tiempo_entrenamiento_minutos']:.2f} minutos")
        print(f"  B.2) CPU promedio: {self.metricas_finales['indicadores']['B2_cpu_promedio_porcentaje']:.2f}%")
        print(f"  B.3) GPU Load promedio: {self.metricas_finales['indicadores']['B3_gpu_load_promedio_porcentaje']:.2f}%")
        print("="*80)