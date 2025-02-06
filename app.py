import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

# Load dataset dan geojson
@st.cache_data
def load_data():
    data = pd.read_excel("dataset_project2_new.xlsx")
    return data.dropna()

@st.cache_data
def load_geojson():
    return gpd.read_file("ADMINISTRASI_KABKOT_AR_BPS.geojson")

# Clustering function
def cluster_data(data):
    X = data[['indeks_pembangunan_literasi_masyarakat', 'indeks_pendidikan', 'indeks_masyarakat_digital_indonesia']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map cluster categories
    centroids = kmeans.cluster_centers_
    sorted_indices = centroids[:, 0].argsort()
    cluster_map = {sorted_indices[0]: 'Rendah', sorted_indices[1]: 'Sedang', sorted_indices[2]: 'Tinggi'}
    data['Kategori'] = data['Cluster'].map(cluster_map)
    
    remap_cluster = {0: 1, 1: 2, 2: 0}
    data['Cluster'] = data['Cluster'].map(remap_cluster)

    return data

# Load data
st.title("Visualisasi K-Means Clustering dengan Persebaran Kota di Jawa Barat")
data = load_data()
geojson_data = load_geojson()
data = cluster_data(data)

# Visualisasi Big Numbers
st.subheader("Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Wilayah", f"{data['nama_kabupaten_kota'].nunique()}")
col2.metric("Rata-Rata Literasi", f"{data['indeks_pembangunan_literasi_masyarakat'].mean():.2f}")
col3.metric("Rata-Rata Pendidikan", f"{data['indeks_pendidikan'].mean():.2f}")
col4.metric("Rata-Rata Digital", f"{data['indeks_masyarakat_digital_indonesia'].mean():.2f}")

# Plotly 3D Scatter Plot
trace = go.Scatter3d(
    x=data['indeks_pembangunan_literasi_masyarakat'],
    y=data['indeks_pendidikan'],
    z=data['indeks_masyarakat_digital_indonesia'],
    mode='markers',
    marker=dict(color=data['Cluster'], colorscale='Viridis', size=5),
    name='Data Points',
    hovertext=data.apply(
        lambda row: f"{row['nama_kabupaten_kota']}<br>Kategori: {row['Kategori']}", axis=1
    ),
)
layout = go.Layout(
    title='3D KMeans Clustering Jawa Barat',
    scene=dict(
        xaxis=dict(title='Indeks Literasi'),
        yaxis=dict(title='Indeks Pendidikan'),
        zaxis=dict(title='Indeks Digital')
    )
)
fig = go.Figure(data=[trace], layout=layout)
st.plotly_chart(fig)

# Mapping using Folium
st.subheader("Persebaran Klaster Kota/Kabupaten Jawa Barat")
m = folium.Map(location=[-6.9, 107.6], zoom_start=8)
marker_cluster = MarkerCluster().add_to(m)

for _, row in data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"<b>{row['nama_kabupaten_kota']}</b><br>Kategori: {row['Kategori']}"),
        icon=folium.Icon(color='blue' if row['Kategori'] == 'Tinggi' else 'green' if row['Kategori'] == 'Sedang' else 'red')
    ).add_to(marker_cluster)

# Tampilkan map
st_data = st_folium(m, width=700)
