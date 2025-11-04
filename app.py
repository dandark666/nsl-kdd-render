from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
plt.switch_backend('Agg')

# Carpeta donde se guardan las gráficas
SAVE_DIR = os.path.join('static', 'plots')
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_to_base64():
    """Convertir plot a base64 para HTML"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def image_to_base64(image_path):
    """Convertir imagen estática a base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"⚠️  Advertencia: No se encontró la imagen {image_path}")
        return None

def create_sample_data():
    """Generar datos de ejemplo realistas del NSL-KDD"""
    np.random.seed(42)
    n_samples = 1500
    sample_data = {
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'dns', 'ssh', 'telnet'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'SH'], n_samples, p=[0.6,0.15,0.1,0.1,0.05]),
        'src_bytes': np.random.exponential(1000, n_samples).astype(int),
        'dst_bytes': np.random.exponential(500, n_samples).astype(int),
        'duration': np.random.exponential(10, n_samples).astype(int),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'same_srv_rate': np.random.beta(2,5, n_samples),
        'diff_srv_rate': np.random.beta(5,2, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(8, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5,2, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(2,5, n_samples),
        'class': np.random.choice(['normal','neptune','portsweep','satan','ipsweep','smurf'], n_samples,
                                  p=[0.7,0.1,0.05,0.05,0.05,0.05])
    }
    return pd.DataFrame(sample_data)

@app.route('/')
def index():
    df_orig = create_sample_data()
    df = df_orig.copy()

    # Preprocesamiento de categorías
    labelencoder = LabelEncoder()
    categorical_columns = []
    for col in ['class','protocol_type','service','flag']:
        if col in df.columns:
            df[col] = labelencoder.fit_transform(df[col].astype(str))
            categorical_columns.append(col)

    plots = {}

    # 1️⃣ Distribución de clases
    plt.figure(figsize=(12,6))
    class_counts = df_orig['class'].value_counts()
    colors = ['#2ecc71' if 'normal' in str(x).lower() else '#e74c3c' for x in class_counts.index]
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.8)
    plt.title('Distribución de Clases - NSL-KDD', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tipo de Conexión', fontweight='bold')
    plt.ylabel('Cantidad de Registros', fontweight='bold')
    plt.xticks(range(len(class_counts)), [str(x) for x in class_counts.index], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, v + max(class_counts.values)*0.01, f'{v}', ha='center', va='bottom', fontweight='bold')
    plots['class_distribution'] = plot_to_base64()
    plt.close()

    # 2️⃣ Distribución de protocolos
    plt.figure(figsize=(10,6))
    proto_counts = df_orig['protocol_type'].value_counts()
    colors = ['#3498db', '#9b59b6', '#e67e22']
    bars = plt.bar(proto_counts.index, proto_counts.values, color=colors, alpha=0.8)
    plt.title('Distribución de Protocolos de Red', fontsize=14, fontweight='bold')
    plt.xlabel('Protocolo', fontweight='bold')
    plt.ylabel('Frecuencia', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, proto_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, v + max(proto_counts.values)*0.01, f'{v}', ha='center', va='bottom', fontweight='bold')
    plots['protocol_hist'] = plot_to_base64()
    plt.close()

    # 3️⃣ Heatmap de correlaciones
    plt.figure(figsize=(12,10))
    numeric_cols = ['src_bytes','dst_bytes','duration','count','srv_count','dst_host_count','dst_host_srv_count']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if len(numeric_cols)>=3:
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                    cbar_kws={"shrink":.8}, annot_kws={"size":10})
        plt.title('Mapa de Calor - Correlaciones entre Variables', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plots['correlation_heatmap'] = plot_to_base64()
    plt.close()

    # 4️⃣ Histogramas de bytes
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.hist(df['src_bytes'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.title('Bytes de Origen', fontweight='bold')
    plt.xlabel('Bytes')
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.hist(df['dst_bytes'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    plt.title('Bytes de Destino', fontweight='bold')
    plt.xlabel('Bytes')
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.3)
    plt.suptitle('Distribución de Tráfico de Red', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plots['bytes_hist'] = plot_to_base64()
    plt.close()

    # 5️⃣ Servicios más usados
    plt.figure(figsize=(12,6))
    service_counts = df_orig['service'].value_counts().head(8)
    colors = plt.cm.Set3(np.linspace(0,1,len(service_counts)))
    bars = plt.bar(service_counts.index, service_counts.values, color=colors, alpha=0.8)
    plt.title('Top 8 Servicios Más Utilizados', fontsize=14, fontweight='bold')
    plt.xlabel('Servicio', fontweight='bold')
    plt.ylabel('Frecuencia', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, service_counts.values):
        plt.text(bar.get_x()+bar.get_width()/2, v + max(service_counts.values)*0.01, f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    plots['services'] = plot_to_base64()
    plt.close()

    # 6️⃣ Distribución de Variables Numéricas (desde imagen estática)
    static_distribution = image_to_base64("static/plots/distribucion_numericas.png")
    if static_distribution:
        plots['numeric_distribution'] = static_distribution
    else:
        # Si no existe la imagen, usar un placeholder
        plots['numeric_distribution'] = None

    # 7️⃣ Matriz de Scatter (desde imagen estática)
    static_scatter = image_to_base64("static/plots/matriz_scatter.png")
    if static_scatter:
        plots['scatter_matrix'] = static_scatter
    else:
        # Si no existe la imagen, usar un placeholder
        plots['scatter_matrix'] = None

    # Estadísticas y tablas
    stats_data = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(categorical_columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB",
        'dataset_source': 'NSL-KDD Dataset Simulado',
        'null_values': df.isnull().sum().sum(),
        'attack_percentage': f"{(len(df_orig[df_orig['class']!='normal'])/len(df_orig)*100):.1f}%"
    }
    table_head = df_orig.head(12).to_dict('records')
    columns = df_orig.columns.tolist()
    dtype_info = [{'columna': c, 'tipo': str(df_orig[c].dtype),
                   'no_nulos': df_orig[c].notnull().sum(),
                   'nulos': df_orig[c].isnull().sum(),
                   'unicos': df_orig[c].nunique()} for c in df_orig.columns[:10]]

    return render_template('index.html',
                           plots=plots,
                           stats=stats_data,
                           table_head=table_head,
                           dtype_info=dtype_info,
                           columns=columns)

@app.route('/health')
def health():
    return {'status':'healthy', 'message':'NSL-KDD Dashboard funcionando'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5001))
    print(f"🚀 Iniciando NSL-KDD Dashboard en puerto {port}...")
    print(f"🌐 Accede en: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
