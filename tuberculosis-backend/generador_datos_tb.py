import random
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple

class GeneradorDatosTB:
    def __init__(self):
        # Síntomas principales de TB con múltiples variaciones
        self.sintomas_tb = {
            'tos': [
                "tengo tos persistente", "he estado tosiendo mucho", "llevo semanas con tos",
                "tengo ataques de tos frecuentes", "no puedo dejar de toser", "tengo tos seca",
                "tengo tos con flema", "toso sangre ocasionalmente", "la tos me despierta por la noche"
            ],
            'fiebre': [
                "tengo fiebre intermitente", "me sube la fiebre por las tardes", "tengo escalofríos y fiebre",
                "he tenido fiebre baja constante", "me da fiebre por la noche", "tengo sudores febriles"
            ],
            'perdida_peso': [
                "he perdido peso sin explicación", "estoy adelgazando sin cambiar mi dieta",
                "he bajado varios kilos sin motivo", "mi ropa me queda muy holgada", 
                "he perdido el apetito y estoy perdiendo peso"
            ],
            'sudores_nocturnos': [
                "sudo mucho por la noche", "me despierto empapado en sudor", 
                "tengo sudores nocturnos intensos", "las sábanas están mojadas por el sudor"
            ],
            'fatiga': [
                "me siento extremadamente cansado", "no tengo energía para nada",
                "me fatigo con actividades simples", "me cuesta levantarme por el cansancio",
                "me siento agotado todo el tiempo"
            ],
            'dolor_pecho': [
                "me duele el pecho al respirar", "siento punzadas en el pecho",
                "tengo dolor en el pecho al toser", "siento presión en el tórax"
            ],
            'dificultad_respirar': [
                "me falta el aire", "tengo dificultad para respirar",
                "me cuesta recuperar el aliento", "siento que no puedo respirar profundamente",
                "me canso al subir escaleras"
            ]
        }
        
        # Síntomas no relacionados con TB (negativos)
        self.sintomas_no_tb = {
            'alergia': [
                "tengo alergia estacional", "estornudo mucho por alergias",
                "tengo picazón en los ojos por alergia", "la alergia me da congestión nasal"
            ],
            'gripe': [
                "tengo síntomas de gripe", "me duele el cuerpo por la gripe",
                "tengo malestar general por gripe", "la gripe me da fiebre leve"
            ],
            'resfriado': [
                "estoy resfriado", "tengo congestión nasal por resfriado",
                "el resfriado me da dolor de garganta", "tengo mocos por resfriado"
            ],
            'ansiedad': [
                "tengo ansiedad generalizada", "siento palpitaciones por ansiedad",
                "la ansiedad me da dificultad para respirar", "tengo ataques de pánico"
            ],
            'reflujo': [
                "tengo acidez estomacal", "el reflujo me da tos",
                "siento ardor en el pecho por reflujo", "tengo regurgitación ácida"
            ]
        }
        
        # Síntomas menos específicos (pueden ser ambos)
        self.sintomas_ambiguos = [
            "tengo dolor de cabeza frecuente", "me siento mareado a veces",
            "tengo náuseas ocasionales", "me duele el cuerpo",
            "tengo pérdida de apetito", "me siento débil"
        ]
        
        # Patrones mejorados con manejo de casos de 1 o 2 síntomas
        self.patrones_frases = [
            "{sintoma1}", 
            "{sintoma1} y {sintoma2}", 
            "{sintoma1}, además {sintoma2}", 
            "desde hace días {sintoma1} y {sintoma2}",
            "últimamente {sintoma1}, también {sintoma2}",
            "{sintoma1}, a veces {sintoma2}"
        ]
        
        # Configuración de distribución de clases
        self.target_distribution = {'tb': 0.3, 'no_tb': 0.7}  # 30% TB, 70% no TB

    def seleccionar_sintomas(self, categoria: str, num_sintomas: int = 2) -> List[str]:
        """Selecciona síntomas asegurando diversidad"""
        sintomas_pool = self.sintomas_tb if categoria == 'tb' else self.sintomas_no_tb
        categorias = random.sample(list(sintomas_pool.keys()), min(num_sintomas, len(sintomas_pool)))
        return [random.choice(sintomas_pool[categoria]) for categoria in categorias]

    def generar_frase_tb(self) -> str:
        """Genera una frase coherente para casos de TB"""
        num_sintomas = random.choices([1, 2, 3], weights=[0.2, 0.6, 0.2])[0]
        sintomas = self.seleccionar_sintomas('tb', num_sintomas)
        
        # 30% de probabilidad de añadir síntoma ambiguo
        if random.random() < 0.3 and num_sintomas < 3:
            sintomas.append(random.choice(self.sintomas_ambiguos))
        
        return self._construir_frase(sintomas)

    def generar_frase_no_tb(self) -> str:
        """Genera una frase coherente para casos no TB"""
        num_sintomas = random.choices([1, 2], weights=[0.4, 0.6])[0]
        sintomas = self.seleccionar_sintomas('no_tb', num_sintomas)
        
        # 40% de probabilidad de añadir síntoma ambiguo
        if random.random() < 0.4 and num_sintomas < 2:
            sintomas.append(random.choice(self.sintomas_ambiguos))
        
        return self._construir_frase(sintomas)

    def _construir_frase(self, sintomas: List[str]) -> str:
        """Construye la frase final según patrones"""
        if len(sintomas) == 1:
            patron = "{sintoma1}"
        else:
            patron = random.choice([p for p in self.patrones_frases if 'sintoma2' in p])
        
        try:
            if len(sintomas) == 1:
                frase = patron.format(sintoma1=sintomas[0])
            else:
                frase = patron.format(
                    sintoma1=sintomas[0],
                    sintoma2=sintomas[1] if len(sintomas) > 1 else ""
                )
        except (KeyError, IndexError):
            frase = " ".join(sintomas)
        
        return frase[0].upper() + frase[1:]

    def determinar_etiqueta(self, frase: str) -> int:
        """Determina la etiqueta basada en síntomas específicos"""
        sintomas_fuertes_tb = ['tos persistente', 'tosiendo mucho', 'semanas con tos', 'toso sangre', 
                              'fiebre intermitente', 'sudor nocturno', 'sudores nocturnos', 'empapado en sudor',
                              'perdido peso', 'adelgazando sin', 'bajado varios kilos']

        sintomas_fuertes_no_tb = ['alergia estacional', 'estornudo mucho', 'congestión nasal', 
                                 'síntomas de gripe', 'dolor de garganta', 'acidez estomacal',
                                 'ataques de pánico', 'ansiedad generalizada']
        
        count_tb = sum(1 for sintoma in sintomas_fuertes_tb if sintoma in frase.lower())
        count_no_tb = sum(1 for sintoma in sintomas_fuertes_no_tb if sintoma in frase.lower())
        
        # Lógica de etiquetado mejorada
        if count_tb >= 2 or (count_tb >= 1 and 'sangre' in frase.lower()):
            return 1  # TB
        elif count_no_tb >= 2:
            return 0  # No TB
        else:
            # Caso ambiguo, decidir basado en distribución objetivo
            return 1 if random.random() < self.target_distribution['tb'] else 0

    def generar_dataset_balanceado(self, num_frases: int) -> Tuple[List[str], List[int]]:
        """Genera dataset balanceado según distribución objetivo"""
        textos = []
        etiquetas = []
        
        n_tb = int(num_frases * self.target_distribution['tb'])
        n_no_tb = num_frases - n_tb
        
        # Generar casos TB
        for _ in range(n_tb):
            frase = self.generar_frase_tb()
            textos.append(frase)
            etiquetas.append(1)
        
        # Generar casos no TB
        for _ in range(n_no_tb):
            frase = self.generar_frase_no_tb()
            textos.append(frase)
            etiquetas.append(0)
        
        # Mezclar el dataset
        combined = list(zip(textos, etiquetas))
        random.shuffle(combined)
        textos, etiquetas = zip(*combined)
        
        return list(textos), list(etiquetas)

    def generar_archivos(self, sintomas_file: str, etiquetas_file: str, num_frases: int = 10000):
        """Genera los archivos de entrenamiento con metadatos"""
        try:
            textos, etiquetas = self.generar_dataset_balanceado(num_frases)
            
            with open(sintomas_file, "w", encoding="utf-8") as f_sintomas, \
                 open(etiquetas_file, "w", encoding="utf-8") as f_etiquetas:
                
                for texto, etiqueta in zip(textos, etiquetas):
                    f_sintomas.write(texto + "\n")
                    f_etiquetas.write(str(etiqueta) + "\n")
            
            # Guardar metadatos
            metadatos = {
                "fecha_generacion": datetime.now().isoformat(),
                "total_instancias": num_frases,
                "distribucion_clases": {
                    "tb": sum(etiquetas),
                    "no_tb": len(etiquetas) - sum(etiquetas)
                },
                "configuracion": {
                    "target_distribution": self.target_distribution,
                    "version": "1.1"
                }
            }
            
            with open("metadatos_dataset.json", "w", encoding="utf-8") as f_meta:
                json.dump(metadatos, f_meta, indent=2)
            
            print(f"✅ Archivos generados exitosamente. Distribución: {sum(etiquetas)} TB, {len(etiquetas)-sum(etiquetas)} no TB")
            
        except Exception as e:
            print(f"❌ Error al generar archivos: {e}")
            raise

if __name__ == '__main__':
    try:
        generador = GeneradorDatosTB()
        generador.generar_archivos(
            sintomas_file="sintomas_mejorado.txt",
            etiquetas_file="etiquetas_mejorado.txt",
            num_frases=10000
        )
    except Exception as e:
        print(f"❌ Error en ejecución: {e}")