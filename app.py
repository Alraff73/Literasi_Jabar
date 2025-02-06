import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Membaca data dari file Excel
@st.cache_data
def load_data():
    data = pd.read_excel("dataset_project2_new.xlsx")
    return data.dropna()  # Hapus nilai NaN jika ada

# Fungsi untuk memproses data dan membuat visualisasi
def process_data():
    data = load_data()

    # Menyiapkan data untuk clustering
    X = data[['indeks_pembangunan_literasi_masyarakat', 'indeks_pendidikan', 'indeks_masyarakat_digital_indonesia']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menjalankan KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
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
    return fig

# Tampilan Streamlit
st.title("3D KMeans Clustering with Plotly & Streamlit")

# Tampilkan scatterplot 3D
st.plotly_chart(process_data())

# Menambahkan iframe Tableau menggunakan markdown
st.markdown("""
    <iframe src="https://public.tableau.com/app/profile/nita.sawalia/viz/project_17385103843730/Dashboard1" 
            width="100%" height="800px" frameborder="0"></iframe>
""", unsafe_allow_html=True)