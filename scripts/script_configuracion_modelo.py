import json
import pandas as pd
from datetime import datetime
import tensorflow as tf
import os

class ConfiguracionModelo:
    def __init__(self, modelo, output_dir='resultados'):
        self.modelo = modelo
        self.output_dir = output_dir
        self.config = {}
        # Asegurar que el directorio existe
        os.makedirs(self.output_dir, exist_ok=True)

    def extraer_configuracion(self):
        print("="*80)
        print("SCRIPT 1: EXTRACCIÓN DE CONFIGURACIÓN DEL MODELO")
        print("="*80)
        print("Variable Independiente (VI): Método basado en aprendizaje profundo")
        print("Dimensión: Configuración del modelo")
        print("="*80)

        num_capas_total = len(self.modelo.layers)
        print(f"\n[INDICADOR A.1] Número total de capas: {num_capas_total}")

        capas_por_tipo = {}
        for layer in self.modelo.layers:
            tipo = layer.__class__.__name__
            capas_por_tipo[tipo] = capas_por_tipo.get(tipo, 0) + 1

        print("\nDesglose por tipo de capa:")
        for tipo, cantidad in sorted(capas_por_tipo.items()):
            print(f"  - {tipo}: {cantidad}")

        params_totales = self.modelo.count_params()
        params_entrenables = sum([tf.size(w).numpy() for w in self.modelo.trainable_weights])
        params_no_entrenables = params_totales - params_entrenables

        print(f"\n[INDICADOR A.2] Número total de parámetros: {params_totales:,}")
        print(f"  - Entrenables: {params_entrenables:,}")
        print(f"  - No entrenables: {params_no_entrenables:,}")

        componentes = self._identificar_componentes()

        self.config = {
            'fecha_registro': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nombre_modelo': self.modelo.name,
            'indicadores': {
                'A1_numero_capas_total': num_capas_total,
                'A2_numero_parametros_total': int(params_totales),
                'A2_parametros_entrenables': int(params_entrenables),
                'A2_parametros_no_entrenables': int(params_no_entrenables)
            },
            'desglose_capas': capas_por_tipo,
            'componentes': componentes,
            'arquitectura': {
                'input_shape': str(self.modelo.input_shape),
                'output_shape': str(self.modelo.output_shape)
            }
        }
        return self.config

    def _identificar_componentes(self):
        componentes = {
            'CNN': {'capas': [], 'parametros': 0},
            'LSTM': {'capas': [], 'parametros': 0},
            'Atencion': {'capas': [], 'parametros': 0},
            'Clasificacion': {'capas': [], 'parametros': 0}
        }

        for layer in self.modelo.layers:
            layer_name = layer.name
            layer_params = layer.count_params()

            if any(x in layer_name for x in ['conv', 'pool', 'flatten', 'bn']):
                componentes['CNN']['capas'].append(layer_name)
                componentes['CNN']['parametros'] += layer_params
            elif any(x in layer_name for x in ['lstm', 'bidirectional', 'reshape']):
                componentes['LSTM']['capas'].append(layer_name)
                componentes['LSTM']['parametros'] += layer_params
            elif any(x in layer_name for x in ['attention', 'global_avg_pool', 'layer_norm']):
                componentes['Atencion']['capas'].append(layer_name)
                componentes['Atencion']['parametros'] += layer_params
            elif 'dense' in layer_name or 'output' in layer_name or 'dropout' in layer_name:
                componentes['Clasificacion']['capas'].append(layer_name)
                componentes['Clasificacion']['parametros'] += layer_params
        return componentes

    def generar_ficha_observacion(self):
        ruta_salida = os.path.join(self.output_dir, 'ficha_configuracion.json')
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Ficha de observación guardada en: {ruta_salida}")

    def generar_tabla_excel(self):
        ruta_salida = os.path.join(self.output_dir, 'tabla_configuracion.xlsx')
        datos = {
            'Variable': ['VI: Método basado en aprendizaje profundo'] * 2,
            'Dimensión': ['A) Configuración del modelo'] * 2,
            'Indicador': ['A.1) Número total de capas', 'A.2) Número total de parámetros'],
            'Valor': [self.config['indicadores']['A1_numero_capas_total'], f"{self.config['indicadores']['A2_numero_parametros_total']:,}"],
            'Instrumento': ['Ficha de observación'] * 2,
            'Fecha': [self.config['fecha_registro']] * 2
        }
        df = pd.DataFrame(datos)
        with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Indicadores', index=False)
        print(f"[OK] Tabla Excel de configuración guardada en: {ruta_salida}")

    def imprimir_resumen(self):
        print("\n" + "="*80)
        print("RESUMEN DE CONFIGURACIÓN DEL MODELO")
        print("="*80)
        print(f"Modelo: {self.config['nombre_modelo']}")
        print(f"\n[INDICADORES TABLA I - DIMENSIÓN A]")
        print(f"  A.1) Número total de capas: {self.config['indicadores']['A1_numero_capas_total']}")
        print(f"  A.2) Número total de parámetros: {self.config['indicadores']['A2_numero_parametros_total']:,}")
        print("="*80)

    def run(self):
        self.extraer_configuracion()
        self.generar_ficha_observacion()
        self.generar_tabla_excel()
        return self.config