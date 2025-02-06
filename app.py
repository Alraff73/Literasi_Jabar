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

    # Mapping cluster ke kategori berdasarkan indeks literasi
    centroids = kmeans.cluster_centers_
    sorted_indices = centroids[:, 0].argsort()  # Urutkan berdasarkan indeks literasi
    cluster_map = {sorted_indices[0]: 'Rendah', sorted_indices[1]: 'Sedang', sorted_indices[2]: 'Tinggi'}
    
    # Terapkan mapping kategori yang benar
    data['Kategori'] = data['Cluster'].map(cluster_map)

    return data, kmeans, scaler

# Load data
st.title("Visualisasi K-Means Clustering dengan Persebaran Kota di Jawa Barat")
data = load_data()
geojson_data = load_geojson()
data, kmeans, scaler = cluster_data(data)

# Visualisasi Big Numbers
st.subheader("Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Wilayah", f"{data['nama_kabupaten_kota'].nunique()}")
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
    hovertext=data.apply(
        lambda row: f"Kabupaten/Kota: {row['nama_kabupaten_kota']}<br>Kategori: {row['Kategori']}", axis=1
    ),
    hoverinfo='x+y+z+text'
)

# Denormalisasi centroid untuk skala asli
centroids_denormalized = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_trace = go.Scatter3d(
    x=centroids_denormalized[:, 0],
    y=centroids_denormalized[:, 1],
    z=centroids_denormalized[:, 2],
    mode='markers+text',
    marker=dict(color='black', size=10, symbol='diamond'),
    name='Centroids',
    text=['Rendah', 'Sedang', 'Tinggi'],
    textposition='top center'
)

# Gabungkan plot data dan centroid
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

for _, row in data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=(f"<b>{row['nama_kabupaten_kota']}</b><br>Kategori: {row['Kategori']}"),
        icon=folium.Icon(color='blue' if row['Kategori'] == 'Tinggi' else 'green' if row['Kategori'] == 'Sedang' else 'red')
    ).add_to(marker_cluster)

st_data = st_folium(m, width=700)

# Visualisasi Bar dan Pie Chart
import plotly.express as px

cluster_counts = data.groupby('Kategori')['nama_kabupaten_kota'].count().reset_index()
fig_bar = px.bar(
    cluster_counts, 
    x='Kategori', 
    y='nama_kabupaten_kota', 
    title='Jumlah Kabupaten/Kota per Kategori', 
    labels={'Kategori': 'Kategori', 'nama_kabupaten_kota': 'Jumlah Kabupaten/Kota'}
)
st.plotly_chart(fig_bar)

fig_pie = px.pie(
    cluster_counts, 
    values='nama_kabupaten_kota', 
    names='Kategori', 
    title='Persentase Kabupaten/Kota per Kategori'
)
st.plotly_chart(fig_pie)
