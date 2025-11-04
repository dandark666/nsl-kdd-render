from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import os

app = Flask(__name__)
plt.switch_backend('Agg')

# Carpeta donde se guardan las gráficas
SAVE_DIR = os.path.join('static', 'plots')
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_to_base64():
    """Convierte el gráfico actual a base64"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    # === 1. LECTURA DEL DATASET ===
    try:
        dataset_files = [f for f in os.listdir('.') if f.endswith(('.txt', '.arff', '.TXT')) and 'KDD' in f.upper()]
        
        if dataset_files:
            dataset_path = dataset_files[0]
            print(f"📁 Cargando dataset: {dataset_path}")
            
            if dataset_path.endswith('.arff'):
                from scipy.io import arff
                data, meta = arff.loadarff(dataset_path)
                df_orig = pd.DataFrame(data)
                for col in df_orig.columns:
                    if df_orig[col].dtype == object:
                        df_orig[col] = df_orig[col].str.decode('utf-8')
            else:
                df_orig = pd.read_csv(dataset_path, header=None)
        else:
            raise FileNotFoundError("No se encontró archivo del dataset")
            
        df = df_orig.copy()
        
        # Nombres de columnas
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
        ]
        for i, col in enumerate(df.columns):
            if i < len(column_names):
                df = df.rename(columns={col: column_names[i]})
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        np.random.seed(42)
        n_samples = 500
        sample_data = {
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'dns'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'dst_bytes': np.random.randint(0, 5000, n_samples),
            'duration': np.random.randint(0, 100, n_samples),
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            'dst_host_srv_count': np.random.randint(0, 100, n_samples),
            'dst_host_same_srv_rate': np.random.uniform(0, 1, n_samples),
            'class': np.random.choice(['normal', 'anomaly'], n_samples, p=[0.7, 0.3])
        }
        df_orig = pd.DataFrame(sample_data)
        df = df_orig.copy()
    
    # === 2. PREPROCESAMIENTO ===
    labelencoder = LabelEncoder()
    categorical_columns = []
    for col in ['class', 'protocol_type', 'service', 'flag']:
        if col in df.columns:
            df[col] = labelencoder.fit_transform(df[col].astype(str))
            categorical_columns.append(col)
    
    # === 3. VISUALIZACIONES ===
    plots = {}

    # Gráfica 1: Distribución de Tipos de Protocolo
    plt.figure(figsize=(10, 6))
    if 'protocol_type' in df_orig.columns:
        protocol_counts = df_orig['protocol_type'].value_counts()
        plt.bar(protocol_counts.index, protocol_counts.values, 
                color=['#3498db', '#9b59b6', '#e67e22', '#2ecc71'])
        plt.title('Distribución de Tipos de Protocolo', fontsize=14, fontweight='bold')
        plt.xlabel('Tipo de Protocolo')
        plt.ylabel('Frecuencia')
        plt.grid(axis='y', alpha=0.3)
    plots['protocol_hist'] = plot_to_base64()
    plt.close()
    
    # Gráfica 2: Distribución de Variables Numéricas (ya guardada)
    hist_path = os.path.join(SAVE_DIR, 'distribucion_numericas.png')
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            plots['multiple_hist'] = base64.b64encode(f.read()).decode()
    else:
        plots['multiple_hist'] = None

    # Gráfica 3: Matriz de Scatter (ya guardada)
    scatter_path = os.path.join(SAVE_DIR, 'matriz_scatter.png')
    if os.path.exists(scatter_path):
        with open(scatter_path, "rb") as f:
            plots['scatter_matrix'] = base64.b64encode(f.read()).decode()
    else:
        plots['scatter_matrix'] = None

    # Gráfica 4: Heatmap de Correlaciones
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        important_cols = [col for col in numeric_df.columns if any(keyword in col for keyword in 
                        ['rate', 'count', 'bytes', 'duration', 'class'])]
        if len(important_cols) < 2:
            important_cols = numeric_df.columns[:10]
        correlation_matrix = numeric_df[important_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8},
                    annot_kws={"size": 10})
        plt.title('Mapa de Calor - Correlaciones entre Variables', 
                fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plots['correlation_heatmap'] = plot_to_base64()
    plt.close()

    # Gráfica 5: Distribución de Clases
    plt.figure(figsize=(10, 6))
    if 'class' in df_orig.columns:
        class_distribution = df_orig['class'].value_counts()
        colors = ['#2ecc71' if 'normal' in str(x).lower() else '#e74c3c' for x in class_distribution.index]
        plt.bar(range(len(class_distribution)), class_distribution.values, color=colors)
        plt.title('Distribución de Clases (Normal vs Ataques)', fontsize=14, fontweight='bold')
        plt.xlabel('Tipo de Conexión')
        plt.ylabel('Cantidad')
        plt.xticks(range(len(class_distribution)), 
                  [str(x) for x in class_distribution.index], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate(class_distribution.values):
            plt.text(i, v + max(class_distribution.values)*0.01, str(v), 
                     ha='center', va='bottom', fontweight='bold')
    plots['class_distribution'] = plot_to_base64()
    plt.close()
    
    # === 4. TABLAS Y ESTADÍSTICAS ===
    table_head = df.head(15).to_dict('records')
    dtype_info = []
    for col in df.columns:
        dtype_info.append({
            'columna': col,
            'tipo': str(df[col].dtype),
            'no_nulos': df[col].notnull().sum(),
            'nulos': df[col].isnull().sum(),
            'unicos': df[col].nunique()
        })
    
    stats_data = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(categorical_columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'dataset_source': dataset_path if 'dataset_path' in locals() else 'Datos de ejemplo',
        'null_values': df.isnull().sum().sum()
    }
    numeric_stats = df.describe().round(3).to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    
    return render_template('index.html', 
                         plots=plots,
                         stats=stats_data,
                         table_head=table_head,
                         dtype_info=dtype_info,
                         numeric_stats=numeric_stats,
                         columns=df.columns.tolist())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Iniciando NSL-KDD Dashboard en puerto {port}...")
    print("📊 Usando datos de ejemplo del NSL-KDD")
    print(f"🌐 Accede en: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
