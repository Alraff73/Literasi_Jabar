import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template
import plotly.graph_objs as go

app = Flask(__name__)

# Fungsi untuk memproses data dan membuat visualisasi
def process_data():
    # Membaca data dari file Excel
    data = pd.read_excel("dataset_project2_new.xlsx")
    data = data.dropna()  # Menghapus nilai NaN jika ada

    # Menyiapkan data untuk clustering
    X = data[['indeks_pembangunan_literasi_masyarakat', 'indeks_pendidikan', 'indeks_masyarakat_digital_indonesia']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menjalankan KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Mendapatkan centroid
    centroids = kmeans.cluster_centers_

    # Membuat visualisasi Plotly
    trace = go.Scatter3d(
        x=data['indeks_pembangunan_literasi_masyarakat'],
        y=data['indeks_pendidikan'],
        z=data['indeks_masyarakat_digital_indonesia'],
        mode='markers',
        marker=dict(color=data['Cluster'], colorscale='Viridis', size=5),
        name='Data Points',
        hovertext=data.apply(
            lambda row: f"Kabupaten/Kota: {row['nama_kabupaten_kota']}<br>Kluster: {row['Cluster']}", axis=1
        ),
        hoverinfo='x+y+z+text'
    )

    centroids_data = go.Scatter3d(
        x=centroids[:, 0] * scaler.scale_[0] + scaler.mean_[0],
        y=centroids[:, 1] * scaler.scale_[1] + scaler.mean_[1],
        z=centroids[:, 2] * scaler.scale_[2] + scaler.mean_[2],
        mode='markers+text',
        marker=dict(color='black', size=10, symbol='diamond'),
        name='Centroids',
        text=['Sedang', 'Tinggi', 'Rendah'],
        textposition='top center'
    )

    data_plotly = [trace, centroids_data]
    layout = go.Layout(
        title='3D KMeans Clustering with Centroids (Plotly)',
        scene=dict(
            xaxis=dict(title='Indeks Pembangunan Literasi Masyarakat'),
            yaxis=dict(title='Indeks Pendidikan'),
            zaxis=dict(title='Indeks Masyarakat Digital Indonesia')
        )
    )

    fig = go.Figure(data=data_plotly, layout=layout)

    # Konversi visualisasi Plotly ke HTML
    plot_html = fig.to_html(full_html=False)
    return plot_html

@app.route('/')
def index():
    plot_html = process_data()  # Proses data dan buat plot
    return render_template('index.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
