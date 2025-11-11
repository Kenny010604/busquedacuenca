# app_dashboard.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import heapq
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional

# ---------------- NODOS ----------------
CUENCA_NODES = {
    "Catedral Nueva": {"lat": -2.8975, "lon": -79.005, "descripcion": "Centro histórico", "tipo": "Otro"},
    "Parque Calderón": {"lat": -2.89741, "lon": -79.00438, "descripcion": "Corazón de Cuenca", "tipo": "Parque"},
    "Puente Roto": {"lat": -2.90423, "lon": -79.00142, "descripcion": "Monumento histórico", "tipo": "Otro"},
    "Museo Pumapungo": {"lat": -2.90607, "lon": -78.99681, "descripcion": "Museo de antropología", "tipo": "Museo"},
    "Terminal Terrestre": {"lat": -2.89222, "lon": -78.99277, "descripcion": "Terminal de buses", "tipo": "Otro"},
    "Mirador de Turi": {"lat": -2.92583, "lon": -79.0040, "descripcion": "Mirador panorámico", "tipo": "Parque"},
    "Parque de la Madre": {"lat": -2.902, "lon": -79.006, "descripcion": "Parque popular", "tipo": "Parque"},
    "El Vergel": {"lat": -2.915, "lon": -79.008, "descripcion": "Zona comercial", "tipo": "Otro"},
    "Parque Infantil": {"lat": -2.898, "lon": -79.003, "descripcion": "Parque para niños", "tipo": "Parque"},
    "Plaza Abdon Calderón": {"lat": -2.8965, "lon": -79.0035, "descripcion": "Plaza principal", "tipo": "Otro"},
    "Puente de Otorongo": {"lat": -2.903, "lon": -79.002, "descripcion": "Puente histórico", "tipo": "Otro"},
    "Museo del Banco Central": {"lat": -2.905, "lon": -78.995, "descripcion": "Museo cultural", "tipo": "Museo"},
    "Universidad de Cuenca": {"lat": -2.897, "lon": -79.006, "descripcion": "Institución educativa", "tipo": "Universidad"},
    "Hospital Vicente Corral Moscoso": {"lat": -2.898, "lon": -78.991, "descripcion": "Hospital público", "tipo": "Hospital"},
    "Parque Miraflores": {"lat": -2.907, "lon": -79.002, "descripcion": "Zona verde", "tipo": "Parque"},
    "Calle Larga": {"lat": -2.8985, "lon": -79.002, "descripcion": "Zona comercial", "tipo": "Otro"},
    "Iglesia del Carmen": {"lat": -2.899, "lon": -79.0045, "descripcion": "Iglesia histórica", "tipo": "Iglesia"},
    "Biblioteca Municipal": {"lat": -2.8978, "lon": -79.003, "descripcion": "Biblioteca pública", "tipo": "Otro"},
    "Estadio Alejandro Serrano": {"lat": -2.910, "lon": -79.003, "descripcion": "Estadio deportivo", "tipo": "Otro"},
    "Plaza San Francisco": {"lat": -2.9035, "lon": -79.004, "descripcion": "Plaza histórica", "tipo": "Otro"},
}

GRAPH_EDGES = {
    "Catedral Nueva": ["Parque Calderón", "Museo Pumapungo", "Parque Infantil", "Plaza Abdon Calderón", "Parque de la Madre"],
    "Parque Calderón": ["Catedral Nueva", "Terminal Terrestre", "Puente Roto", "Plaza Abdon Calderón"],
    "Puente Roto": ["Catedral Nueva", "Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "El Vergel", "Plaza San Francisco"],
    "Museo Pumapungo": ["Catedral Nueva", "Puente Roto", "Terminal Terrestre", "Museo del Banco Central"],
    "Terminal Terrestre": ["Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Hospital Vicente Corral Moscoso"],
    "Mirador de Turi": ["Puente Roto", "Terminal Terrestre", "El Vergel"],
    "Parque de la Madre": ["Catedral Nueva", "Parque Infantil"],
    "El Vergel": ["Puente Roto", "Mirador de Turi", "Calle Larga"],
    "Parque Infantil": ["Catedral Nueva", "Parque de la Madre", "Plaza Abdon Calderón"],
    "Plaza Abdon Calderón": ["Catedral Nueva", "Parque Infantil", "Parque Calderón", "Iglesia del Carmen"],
    "Puente de Otorongo": ["Plaza San Francisco", "Parque Miraflores"],
    "Museo del Banco Central": ["Museo Pumapungo", "Biblioteca Municipal"],
    "Universidad de Cuenca": ["Calle Larga", "Biblioteca Municipal"],
    "Hospital Vicente Corral Moscoso": ["Terminal Terrestre", "Calle Larga"],
    "Parque Miraflores": ["Puente de Otorongo", "Estadio Alejandro Serrano"],
    "Calle Larga": ["El Vergel", "Universidad de Cuenca", "Hospital Vicente Corral Moscoso"],
    "Iglesia del Carmen": ["Plaza Abdon Calderón", "Biblioteca Municipal"],
    "Biblioteca Municipal": ["Museo del Banco Central", "Universidad de Cuenca", "Iglesia del Carmen"],
    "Estadio Alejandro Serrano": ["Parque Miraflores", "Plaza San Francisco"],
    "Plaza San Francisco": ["Puente Roto", "Puente de Otorongo", "Estadio Alejandro Serrano"],
}

# ---------------- FUNCIONES ----------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R * c

class AStarPathFinder:
    def __init__(self, nodes: Dict, edges: Dict):
        self.nodes = nodes
        self.edges = edges

    def heuristic(self, node: str, goal: str) -> float:
        n, g = self.nodes[node], self.nodes[goal]
        return haversine_distance(n["lat"], n["lon"], g["lat"], g["lon"])

    def get_distance(self, node1: str, node2: str) -> float:
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    def find_path(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, List[str]]:
        frontier = []
        heapq.heappush(frontier, (0, start, [start], 0))
        visited = set()
        explored_nodes = set()
        while frontier:
            f, current, path, g_score = heapq.heappop(frontier)
            explored_nodes.add(current)
            if current == goal:
                return path, g_score, list(explored_nodes)
            visited.add(current)
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    g = g_score + self.get_distance(current, neighbor)
                    h = self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (g + h, neighbor, path + [neighbor], g))
        return None, float("inf"), list(explored_nodes)

pathfinder = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES)

# ---------------- SESSION STATE ----------------
if "ruta" not in st.session_state: st.session_state.ruta = []
if "distancia" not in st.session_state: st.session_state.distancia = 0.0
if "explorados" not in st.session_state: st.session_state.explorados = []
if "tiempo" not in st.session_state: st.session_state.tiempo = 0.0

# ---------------- SIDEBAR ----------------
st.sidebar.title("🗺️ Navegación")
inicio = st.sidebar.selectbox("📍 Punto de inicio", list(CUENCA_NODES.keys()))
destino = st.sidebar.selectbox("🏁 Punto de destino", list(CUENCA_NODES.keys()))
mostrar_no_vistos = st.sidebar.checkbox("👀 Mostrar nodos no explorados")
calcular_btn = st.sidebar.button("🔍 Buscar ruta óptima")
limpiar_btn = st.sidebar.button("🧹 Limpiar ruta")
puntos_interes_btn = st.sidebar.button("⭐ Puntos de interés")

st.sidebar.markdown("---")
st.sidebar.header("📌 Datos de Cuenca")
st.sidebar.markdown("""
- **Población:** 400,000 aprox.  
- **Clima:** Templado, 12-25°C  
- **Monumentos:** Catedral Nueva, Puente Roto, Museo Pumapungo  
- **Turismo:** Mirador de Turi, Parque Calderón  
- **Universidades:** Universidad de Cuenca, Universidad del Azuay  
""")

# ---------------- BOTONES ----------------
if limpiar_btn:
    st.session_state.ruta = []
    st.session_state.distancia = 0.0
    st.session_state.explorados = []
    st.session_state.tiempo = 0.0

if calcular_btn:
    start_time = time.time()
    ruta, distancia, explorados = pathfinder.find_path(inicio, destino)
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time

    if ruta:
        st.session_state.ruta = ruta
        st.session_state.distancia = distancia
        st.session_state.explorados = explorados
        st.session_state.tiempo = tiempo_ejecucion
    else:
        st.error("No se encontró ruta")
        st.session_state.ruta = []
        st.session_state.distancia = 0.0
        st.session_state.explorados = []
        st.session_state.tiempo = 0.0

# ---------------- INFORMACION ----------------
st.subheader("📝 Información de la Ruta")
if st.session_state.ruta:
    st.markdown(f"**Ruta encontrada:** {' -> '.join(st.session_state.ruta)}")
    st.markdown(f"**Distancia total:** {st.session_state.distancia:.2f} km")
    st.markdown(f"**Tiempo de ejecución:** {st.session_state.tiempo:.4f} segundos")
    st.markdown(f"**Nodos explorados:** {', '.join(st.session_state.explorados)}")

    # CSV con detalles y resumen
    df = pd.DataFrame({
        "Nodo": st.session_state.ruta,
        "Latitud": [CUENCA_NODES[n]["lat"] for n in st.session_state.ruta],
        "Longitud": [CUENCA_NODES[n]["lon"] for n in st.session_state.ruta],
        "Descripcion": [CUENCA_NODES[n]["descripcion"] for n in st.session_state.ruta]
    })
    df_resumen = pd.DataFrame({
        "Tiempo_ejecucion_segundos": [st.session_state.tiempo],
        "Distancia_total_km": [st.session_state.distancia]
    })
    df_final = pd.concat([df, pd.DataFrame([["Resumen","", "", ""]], columns=df.columns)], ignore_index=True)
    df_final = pd.concat([df_final, df_resumen], axis=1, ignore_index=False)
    st.download_button("⬇️ Descargar detalles en CSV", df_final.to_csv(index=False), "ruta_cuenca.csv", "text/csv")
else:
    st.markdown("Seleccione un punto de inicio y fin, luego haga clic en **Buscar ruta óptima**.")

# ---------------- MAPA ----------------
st.subheader("🗺️ Mapa de Cuenca")
mapa = folium.Map(location=[-2.8975, -79.005], zoom_start=14)
tipo_color = {
    "Museo": "purple",
    "Parque": "green",
    "Iglesia": "darkred",
    "Hospital": "red",
    "Universidad": "blue",
    "Otro": "gray"
}

for nodo, info in CUENCA_NODES.items():
    color = tipo_color.get(info["tipo"], "gray")
    if mostrar_no_vistos and nodo not in st.session_state.explorados:
        color = "orange"
    if nodo in st.session_state.ruta:
        color = "blue"
    folium.Marker(
        [info["lat"], info["lon"]],
        popup=f"📌 {nodo}: {info['descripcion']} ({info['tipo']})",
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(mapa)

if st.session_state.ruta:
    coords = [[CUENCA_NODES[n]["lat"], CUENCA_NODES[n]["lon"]] for n in st.session_state.ruta]
    folium.PolyLine(coords, color="blue", weight=5).add_to(mapa)
    folium.Marker([CUENCA_NODES[st.session_state.ruta[0]]["lat"],
                   CUENCA_NODES[st.session_state.ruta[0]]["lon"]],
                  popup=f"🏁 Inicio: {st.session_state.ruta[0]}",
                  icon=folium.Icon(color="green")).add_to(mapa)
    folium.Marker([CUENCA_NODES[st.session_state.ruta[-1]]["lat"],
                   CUENCA_NODES[st.session_state.ruta[-1]]["lon"]],
                  popup=f"🏁 Destino: {st.session_state.ruta[-1]}",
                  icon=folium.Icon(color="red")).add_to(mapa)

st_folium(mapa, width=900, height=650)
