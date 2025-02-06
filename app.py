import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
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
    return data, fig

# Tampilan Streamlit
st.title("3D KMeans Clustering with Plotly & Streamlit")

# Load data dan tampilkan scatterplot 3D
data, fig = process_data()
st.plotly_chart(fig)

# Big Numbers untuk Insight
st.header("Summary Data")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="large")
with col1:
    st.metric("Total Wilayah", f"{data['nama_kabupaten_kota'].nunique()} Kabupaten/Kota")
with col2:
    st.metric("Rata-Rata Indeks Literasi", f"{data['indeks_pembangunan_literasi_masyarakat'].mean():.2f}")
with col3:
    st.metric("Rata-Rata Indeks Pendidikan", f"{data['indeks_pendidikan'].mean():.2f}")
with col4:
    st.metric("Rata-Rata Indeks Digital", f"{data['indeks_masyarakat_digital_indonesia'].mean():.2f}")

# Visualisasi Persebaran Kota (Map Scatter Plot)
st.header("Persebaran Kabupaten/Kota Berdasarkan Kluster")
fig_map = px.scatter_mapbox(
    data, lat="latitude", lon="longitude", color="Cluster", hover_name="nama_kabupaten_kota",
    zoom=7, mapbox_style="carto-positron",
    title="Persebaran Kabupaten/Kota Berdasarkan Cluster"
)
st.plotly_chart(fig_map)

# Bar Chart Jumlah Kabupaten/Kota per Kluster
st.header("Jumlah Kabupaten/Kota pada Tiap Kluster")
cluster_counts = data['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Jumlah']
bar_chart = px.bar(
    cluster_counts, x='Cluster', y='Jumlah',
    title="Jumlah Kabupaten/Kota per Kluster",
    color='Cluster'
)
st.plotly_chart(bar_chart)

# Pie Chart Persentase per Kluster
st.header("Persentase Kabupaten/Kota per Kluster")
pie_chart = px.pie(
    cluster_counts, values='Jumlah', names='Cluster',
    title="Persentase Kabupaten/Kota per Kluster"
)
st.plotly_chart(pie_chart)
