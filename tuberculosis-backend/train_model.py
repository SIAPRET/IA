from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import re
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

class TextPreprocessor:
    def __init__(self):
        # Lista extendida de stopwords en español
        self.stop_words = {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se',
            'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es'
        }
    
    def clean_text(self, text):
        """Limpieza básica y segura del texto"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\sáéíóúñ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text):
        """Tokenización que devuelve string unido"""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return ' '.join(filtered_words)

def load_data(symptoms_file, labels_file):
    """Carga los datos desde archivos de texto con validación mejorada"""
    try:
        with open(symptoms_file, "r", encoding="utf-8") as f:
            textos = [line.strip() for line in f if line.strip()]
        
        with open(labels_file, "r", encoding="utf-8") as f:
            etiquetas = [int(line.strip()) for line in f if line.strip()]
        
        if len(textos) != len(etiquetas):
            raise ValueError("Número de síntomas y etiquetas no coincide")
            
        return textos, np.array(etiquetas)
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise

def build_model(params=None):
    """Construye el pipeline del modelo con hiperparámetros optimizables"""
    default_params = {
        'vectorizer__ngram_range': (1, 2),
        'vectorizer__min_df': 2,
        'vectorizer__max_df': 0.8,
        'classifier__alpha': 0.1
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', ComplementNB())
    ])
    
    return pipeline.set_params(**default_params)

def optimize_hyperparameters(X, y):
    """Optimización de hiperparámetros usando RandomizedSearchCV"""
    param_dist = {
        'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vectorizer__min_df': [1, 2, 3, 4, 5],
        'vectorizer__max_df': [0.5, 0.6, 0.7, 0.8, 0.9],
        'classifier__alpha': np.logspace(-3, 0, 10)
    }
    
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=50,
        cv=cv,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X, y)
    return search.best_params_

def evaluate_model(model, X, y, X_test, y_test):
    """Evaluación completa del modelo con validación cruzada y test"""
    print("\n📊 Evaluación con Validación Cruzada (5 folds):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    print(f"F1-score promedio: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    print("\n📈 Evaluación en Conjunto de Test:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No TB', 'TB']))
    
    # Calcular pesos de clases para desbalanceo
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print(f"\n⚖️ Pesos de clases calculados: {class_weights}")

def main():
    try:
        print("⚙️ Cargando datos...")
        textos, etiquetas = load_data("sintomas_mejorado.txt", "etiquetas_mejorado.txt")
        
        print("🔧 Preprocesando texto...")
        preprocessor = TextPreprocessor()
        textos_procesados = [preprocessor.tokenize_text(preprocessor.clean_text(text)) for text in textos]
        
        # Dividir datos (conservar distribución de clases)
        X_train, X_test, y_train, y_test = train_test_split(
            textos_procesados, etiquetas, test_size=0.2, 
            random_state=42, stratify=etiquetas
        )
        
        print("\n🔎 Optimizando hiperparámetros...")
        best_params = optimize_hyperparameters(X_train, y_train)
        print(f"Mejores parámetros encontrados: {best_params}")
        
        print("\n🏋️ Entrenando modelo final...")
        final_model = build_model(best_params)
        final_model.fit(X_train, y_train)
        
        # Evaluación completa
        evaluate_model(final_model, X_train, y_train, X_test, y_test)
        
        print("\n💾 Guardando modelo...")
        joblib.dump({
            'pipeline': final_model,
            'stop_words': preprocessor.stop_words,
            'best_params': best_params,
            'class_distribution': np.bincount(etiquetas)
        }, "modelo_tb.pkl")
        
        print("\n✅ Modelo entrenado y guardado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()