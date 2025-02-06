import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point

# Fungsi untuk load dataset
@st.cache_data
def load_data():
    data = pd.read_excel("dataset_project2_new.xlsx")
    return data.dropna()

# Fungsi untuk load geojson
@st.cache_data
def load_geojson():
    geojson_data = gpd.read_file("kota.geojson")

    # Pastikan kolom geometry memiliki data centroid
    geojson_data['geometry'] = geojson_data['geometry'].apply(lambda x: x.centroid if not isinstance(x, Point) else x)

    # Ekstraksi latitude dan longitude
    geojson_data['latitude'] = geojson_data.geometry.y
    geojson_data['longitude'] = geojson_data.geometry.x
    return geojson_data

# Fungsi clustering
def cluster_data(data):
    X = data[['indeks_pembangunan_literasi_masyarakat', 'indeks_pendidikan', 'indeks_masyarakat_digital_indonesia']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Mapping kategori berdasarkan urutan nilai centroid
    centroids = kmeans.cluster_centers_
    sorted_indices = centroids[:, 0].argsort()
    cluster_map = {sorted_indices[0]: 'Rendah', sorted_indices[1]: 'Sedang', sorted_indices[2]: 'Tinggi'}
    data['Kategori'] = data['Cluster'].map(cluster_map)
    remap_cluster = {0: 1, 1: 2, 2: 0}
    data['Cluster'] = data['Cluster'].map(remap_cluster)
    return data, kmeans, scaler

# Load data
st.title("Visualisasi K-Means Clustering dengan Persebaran Kota di Jawa Barat")
data = load_data()
geojson_data = load_geojson()

# Gabungkan data utama dengan geojson
data = pd.merge(data, geojson_data[['bps_nama', 'latitude', 'longitude']], 
                 left_on='bps_nama', right_on='bps_nama', how='left')

data, kmeans, scaler = cluster_data(data)

# Visualisasi Big Numbers
st.subheader("Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Wilayah", f"{data['bps_nama'].nunique()}")
col2.metric("Rata-Rata Literasi", f"{data['indeks_pembangunan_literasi_masyarakat'].mean():.2f}")
col3.metric("Rata-Rata Pendidikan", f"{data['indeks_pendidikan'].mean():.2f}")
col4.metric("Rata-Rata Digital", f"{data['indeks_masyarakat_digital_indonesia'].mean():.2f}")

# Plotly 3D Scatter Plot
data_plot_trace = go.Scatter3d(
    x=data['indeks_pembangunan_literasi_masyarakat'],
    y=data['indeks_pendidikan'],
    z=data['indeks_masyarakat_digital_indonesia'],
    mode='markers',
    marker=dict(color=data['Cluster'], colorscale='Viridis', size=5),
    name='Data Points',
    hovertext=data.apply(lambda row: f"Kabupaten/Kota: {row['bps_nama']}<br>Kategori: {row['Kategori']}", axis=1),
    hoverinfo='x+y+z+text'
)

centroids_denormalized = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_trace = go.Scatter3d(
    x=centroids_denormalized[:, 0],
    y=centroids_denormalized[:, 1],
    z=centroids_denormalized[:, 2],
    mode='markers+text',
    marker=dict(color='black', size=10, symbol='diamond'),
    name='Centroids',
    text=['Sedang', 'Tinggi', 'Rendah'],
    textposition='top center'
)

fig_3d = go.Figure(data=[data_plot_trace, centroid_trace])
fig_3d.update_layout(
    title='3D KMeans Clustering dengan Centroid',
    scene=dict(
        xaxis=dict(title='Indeks Literasi'),
        yaxis=dict(title='Indeks Pendidikan'),
        zaxis=dict(title='Indeks Digital')
    )
)
st.plotly_chart(fig_3d)

# Mapping menggunakan Folium
st.subheader("Persebaran Klaster Kota/Kabupaten Jawa Barat")
m = folium.Map(location=[-6.9, 107.6], zoom_start=8)
marker_cluster = MarkerCluster().add_to(m)

for _, row in data.dropna(subset=['latitude', 'longitude']).iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"<b>{row['bps_nama']}</b><br>Kategori: {row['Kategori']}"),
        icon=folium.Icon(color='blue' if row['Kategori'] == 'Tinggi' else 'green' if row['Kategori'] == 'Sedang' else 'red')
    ).add_to(marker_cluster)

st_data = st_folium(m, width=700)

# Visualisasi Bar dan Pie Chart
import plotly.express as px

cluster_counts = data.groupby('Cluster')['bps_nama'].count().reset_index()
fig_bar = px.bar(
    cluster_counts, 
    x='Cluster', 
    y='bps_nama', 
    title='Jumlah Kabupaten/Kota per Cluster', 
    labels={'Cluster': 'Cluster', 'bps_nama': 'Jumlah Kabupaten/Kota'}
)
st.plotly_chart(fig_bar)

fig_pie = px.pie(
    cluster_counts, 
    values='bps_nama', 
    names='Cluster', 
    title='Persentase Kabupaten/Kota per Cluster'
)
st.plotly_chart(fig_pie)
