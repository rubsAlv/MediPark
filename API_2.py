from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib
from collections import Counter

# === CONFIGURACIÓN INICIAL ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === CARGA DE MODELOS Y ESCALADORES ===
modelo_1 = joblib.load('modelo_por_voluntad.pkl')         # voluntario/involuntario
escalador_1 = joblib.load('escalador_voluntad.pkl')
modelo_2 = joblib.load('modelo_por_nivel.pkl')            # bajo/medio/brusco
escalador_2 = joblib.load('escalador_nivel.pkl')
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# === FUNCIÓN AUXILIAR ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === ENDPOINT DE PREDICCIÓN ===
@app.route('/predecir', methods=['POST'])
def predecir():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if not set(features).issubset(df.columns):
                return jsonify({'error': 'Faltan columnas necesarias'}), 400

            X = df[features]

            # === MODELO 1: Voluntario / Involuntario ===
            X_scaled_1 = escalador_1.transform(X)
            pred_1 = modelo_1.predict(X_scaled_1)
            resumen_1 = dict(Counter(pred_1))
            pred_dominante = pd.Series(pred_1).mode()[0]

            if pred_dominante == 'voluntario':
                return jsonify({
                    'resultado': 'voluntario',
                    'resumen_modelo_1': resumen_1
                })

            # === MODELO 2: Nivel de temblor ===
            X_scaled_2 = escalador_2.transform(X)
            pred_2 = modelo_2.predict(X_scaled_2)
            resumen_2 = dict(Counter(pred_2))
            pred_nivel_dominante = pd.Series(pred_2).mode()[0]

            return jsonify({
                'resultado': 'involuntario',
                'resumen_modelo_1': resumen_1,
                'nivel_temblor': pred_nivel_dominante,
                'resumen_modelo_2': resumen_2
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Archivo no permitido'}), 400

# === INICIAR SERVIDOR ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
