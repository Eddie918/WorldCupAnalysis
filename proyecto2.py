import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from collections import Counter
import json
import matplotlib.image as mpimg

# Configuración de estilo para las gráficas
sns.set(style="whitegrid", palette="muted", color_codes=True)
plt.rcParams['figure.figsize'] = (12, 8)

# Función para cargar datos desde un archivo JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return pd.DataFrame(json.load(file))

# Función para preparar datos de eventos
def prepare_events_data(events_data):
    events_data['x'] = events_data['positions'].apply(lambda pos: pos[0]['x'] if pos else None)
    events_data['y'] = events_data['positions'].apply(lambda pos: pos[0]['y'] if pos else None)
    return events_data

# Función para filtrar eventos relevantes
def filter_relevant_events(events_data, event_types):
    return events_data[events_data['eventName'].isin(event_types)]

# Cargar datos
events_data = load_json("events_World_Cup.json")
matches_data = load_json("matches_World_Cup.json")
teams_data = load_json("teams.json")
coaches_data = load_json("coaches.json")

# Preparar datos de eventos
events_data = prepare_events_data(events_data)

# Filtrar eventos relevantes
relevant_events = ['Pass', 'Shot', 'Foul', 'Duel', 'Goal']
filtered_events = filter_relevant_events(events_data, relevant_events)

# Verificar si existen eventos de goles
print(filtered_events['eventName'].value_counts())

# Extraer coachId de teamsData
def extract_coach_id(teamsData):
    if isinstance(teamsData, dict):
        for team_data in teamsData.values():
            if 'coachId' in team_data:
                return team_data['coachId']
    return None

matches_data['coachId'] = matches_data['teamsData'].apply(extract_coach_id)

# Inspección de columnas para verificar las claves de unión
print(matches_data.columns)
print(matches_data.head())
print(coaches_data.columns)
print(coaches_data.head())

# Intentar la combinación nuevamente asegurándonos de que las claves son correctas
matches_coaches = matches_data.merge(coaches_data, left_on='coachId', right_on='wyId', suffixes=('_match', '_coach'))
events_coaches = filtered_events.merge(matches_coaches, left_on='matchId', right_on='wyId_match')

# Análisis de influencia de entrenadores
coach_style = events_coaches.groupby(['shortName', 'eventName']).size().unstack().fillna(0)
top_coaches = coach_style.sum(axis=1).sort_values(ascending=False).head(5).index

# Visualización de la influencia de los 5 entrenadores principales
top_coach_style = coach_style.loc[top_coaches]

plt.figure(figsize=(14, 10))
sns.heatmap(top_coach_style, annot=True, cmap='coolwarm', cbar_kws={'label': 'Number of Events'})
plt.title('Influence of Top 5 Coaches on Event Types', fontsize=16)
plt.ylabel('Coach', fontsize=12)
plt.xlabel('Event Type', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Tablas de influencia de los 5 entrenadores principales
print("Influencia de los 5 entrenadores principales:")
print(top_coach_style)

# Análisis de la distribución de tipos de eventos
plt.figure(figsize=(12, 8))
sns.countplot(x='eventName', data=filtered_events, palette='muted')
plt.title('Distribution of Event Types in World Cup Matches', fontsize=16)
plt.xlabel('Event Type', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Análisis de clustering para identificar patrones en la ubicación de eventos
X = filtered_events[['x', 'y']].dropna().to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método del codo para determinar el número óptimo de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.show()

# Ajustar el modelo de clustering con el número óptimo de clusters
optimal_clusters = 4  # Determinar el número óptimo basado en el gráfico del codo
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Visualización de los clusters de eventos
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('Clustering of Event Positions in World Cup Matches', fontsize=16)
plt.xlabel('Standardized X', fontsize=12)
plt.ylabel('Standardized Y', fontsize=12)
plt.colorbar(scatter, label='Cluster ID')
plt.show()

# Calcular métricas de evaluación
silhouette_avg = silhouette_score(X_scaled, clusters)
db_score = davies_bouldin_score(X_scaled, clusters)
ch_score = calinski_harabasz_score(X_scaled, clusters)

# Resultados de las métricas de evaluación
evaluation_metrics = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'],
    'Score': [silhouette_avg, db_score, ch_score]
})
print("Métricas de evaluación del clustering:")
print(evaluation_metrics)

# Mejorar las métricas de clustering con GMM
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Aplicar Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=4, random_state=0)
gmm_clusters = gmm.fit_predict(X_pca)

# Visualización de los clusters de eventos utilizando GMM
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_clusters, cmap='viridis')
plt.title('GMM Clustering of Event Positions in World Cup Matches', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.colorbar(scatter, label='Cluster ID')
plt.show()

# Calcular métricas de evaluación para GMM
silhouette_avg_gmm = silhouette_score(X_pca, gmm_clusters)
db_score_gmm = davies_bouldin_score(X_pca, gmm_clusters)
ch_score_gmm = calinski_harabasz_score(X_pca, gmm_clusters)

# Resultados de las métricas de evaluación para GMM
evaluation_metrics_gmm = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'],
    'Score': [silhouette_avg_gmm, db_score_gmm, ch_score_gmm]
})
print("Métricas de evaluación del clustering utilizando GMM:")
print(evaluation_metrics_gmm)

# Secuencias de eventos por equipo
def event_sequences(events):
    sequences = events.groupby('matchId')['eventName'].apply(list)
    return sequences.apply(lambda seq: [seq[i:i+2] for i in range(len(seq)-1)])

team_sequences = filtered_events.groupby('teamId').apply(event_sequences)
team_sequences = team_sequences.reset_index(level=1, drop=True)

# Calcular las frecuencias de secuencias de eventos
team_sequence_freqs = team_sequences.apply(lambda seqs: Counter([tuple(seq) for seq in seqs if len(seq) == 2]))
team_sequence_freqs = team_sequence_freqs.reset_index()

# Visualizar las secuencias de eventos más comunes
for team_id, seq_freq in team_sequence_freqs.values:
    team_name = teams_data[teams_data['wyId'] == team_id]['name'].values[0]
    common_seqs = pd.DataFrame(seq_freq.most_common(10), columns=['Sequence', 'Count'])
    common_seqs[['Event 1', 'Event 2']] = pd.DataFrame(common_seqs['Sequence'].tolist(), index=common_seqs.index)
    common_seqs.drop(columns='Sequence', inplace=True)
    print(f"\nCommon Event Sequences for {team_name}")
    print(common_seqs)

# Crear heatmaps para la ubicación de eventos por equipo
plt.figure(figsize=(14, 8))
sns.kdeplot(data=filtered_events, x='x', y='y', fill=True, cmap='viridis', thresh=0.05)
plt.title('Heatmap of Event Locations', fontsize=16)
plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)
plt.show()

# Crear heatmaps para la ubicación de eventos por equipo
team_names = ['France', 'Argentina', 'Uruguay', 'Portugal', 'Spain']
plt.figure(figsize=(20, 12))
for i, team in enumerate(team_names):
    plt.subplot(2, 3, i + 1)
    team_events = filtered_events[filtered_events['teamId'] == teams_data[teams_data['name'] == team]['wyId'].values[0]]
    sns.kdeplot(data=team_events, x='x', y='y', fill=True, cmap='viridis', thresh=0.05)
    plt.title(f'Heatmap of {team}', fontsize=16)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
plt.tight_layout()
plt.show()

# Comparar el equipo ganador (Francia) con equipos que no pasaron de octavos de final

# Filtrar equipos
winner_team_id = teams_data[teams_data['name'] == 'France']['wyId'].values[0]
round_of_16_teams = ['Argentina', 'Uruguay', 'Portugal', 'Spain', 'Denmark', 'Mexico', 'Japan', 'Switzerland']
round_of_16_team_ids = teams_data[teams_data['name'].isin(round_of_16_teams)]['wyId'].values

# Filtrar eventos por equipo
winner_events = filtered_events[filtered_events['teamId'] == winner_team_id]
round_of_16_events = filtered_events[filtered_events['teamId'].isin(round_of_16_team_ids)]

# Análisis de eventos para cada grupo de equipos
teams_events = {
    'France (Winner)': winner_events,
    'Round of 16 Teams': round_of_16_events
}

# Mejorar la visualización utilizando histogramas
plt.figure(figsize=(14, 10))
for team, events in teams_events.items():
    sns.histplot(events['eventSec'], bins=50, label=team, kde=True)
plt.title('Event Frequency over Match Time', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.legend(title='Team')
plt.show()

# Calcular el promedio de eventos por equipo
average_events_by_team = filtered_events.groupby('teamId')['eventName'].value_counts(normalize=True).unstack().fillna(0)

# Solución para evitar el error
existing_team_ids = teams_data['wyId'][teams_data['wyId'].isin(average_events_by_team.index)]
average_events_by_team = average_events_by_team.loc[existing_team_ids.values]
average_events_by_team.index = teams_data.set_index('wyId').loc[existing_team_ids]['name']

# Identificar los equipos más consistentes (con menor desviación estándar en la distribución de eventos)
consistency_scores = average_events_by_team.std(axis=1).sort_values()
top_consistent_teams = consistency_scores.head(5).index

# Imprimir los resultados de los equipos más consistentes
print("Equipos más consistentes en términos de distribución de eventos:")
print(average_events_by_team.loc[top_consistent_teams])

# Imprimir si los equipos más consistentes llegaron a semifinales o cuartos
semifinal_teams = ['France', 'Belgium', 'England', 'Croatia']
quarterfinal_teams = semifinal_teams + ['Brazil', 'Uruguay', 'Russia', 'Sweden']

print("\nEquipos que llegaron a semifinales o cuartos:")
print(top_consistent_teams[top_consistent_teams.isin(quarterfinal_teams)])

# Visualización mejorada con cancha de fútbol
field_img = mpimg.imread('soccer_field_adjusted.png')  # Reemplaza con la ruta a tu imagen de la cancha

# Crear heatmaps con fondo de cancha de fútbol
plt.figure(figsize=(20, 12))
for i, team in enumerate(team_names):
    plt.subplot(2, 3, i + 1)
    team_events = filtered_events[filtered_events['teamId'] == teams_data[teams_data['name'] == team]['wyId'].values[0]]
    plt.imshow(field_img, extent=[0, 100, 0, 100], aspect='auto')
    sns.kdeplot(data=team_events, x='x', y='y', fill=True, cmap='viridis', thresh=0.05, alpha=0.6)
    plt.title(f'Heatmap of {team}', fontsize=16)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
plt.tight_layout()
plt.show()

# Añadimos plt.close('all') para evitar gráficos residuales
plt.close('all')

# Análisis de oportunidades de gol y goles marcados
# Extraer goles desde los datos de partidos
def extract_goals(team_data):
    goals = 0
    for team_id, team_info in team_data.items():
        goals += team_info.get('score', 0)
    return goals

matches_data['total_goals'] = matches_data['teamsData'].apply(extract_goals)

# Total de goles por equipo
def goals_by_team(team_data):
    goals_dict = {}
    for team_id, team_info in team_data.items():
        team_name = teams_data.loc[teams_data['wyId'] == int(team_id), 'name'].values[0]
        if team_name not in goals_dict:
            goals_dict[team_name] = 0
        goals_dict[team_name] += team_info.get('score', 0)
    return goals_dict

team_goals = matches_data['teamsData'].apply(goals_by_team)

# Combinar goles por equipo en un solo DataFrame
goals_df = pd.DataFrame(list(team_goals)).sum().reset_index()
goals_df.columns = ['team', 'total_goals']

# Filtrar equipos semifinalistas y cuartofinalistas
semifinal_and_quarterfinal_teams = ['France', 'Belgium', 'England', 'Croatia', 'Brazil', 'Uruguay', 'Russia', 'Sweden']
goals_df = goals_df[goals_df['team'].isin(semifinal_and_quarterfinal_teams)]

# Convertir team name a wyId para la combinación posterior
goals_df['teamId'] = goals_df['team'].apply(lambda x: teams_data.loc[teams_data['name'] == x, 'wyId'].values[0])

# Filtrar eventos de disparos para equipos semifinalistas y cuartofinalistas
shot_events = filtered_events[filtered_events['eventName'] == 'Shot']
shot_events_teams = shot_events[shot_events['teamId'].isin(goals_df['teamId'].values)]

# Contar disparos por equipo
shots_per_team = shot_events_teams.groupby('teamId').size()

# Convertir a DataFrame
shots_goals_df = pd.DataFrame({'Shots': shots_per_team, 'Goals': goals_df.set_index('teamId')['total_goals']}).fillna(0)

# Convertir teamId a team name para la visualización
shots_goals_df.index = shots_goals_df.index.map(lambda x: teams_data.loc[teams_data['wyId'] == x, 'name'].values[0])

# Mostrar los 5 mejores equipos con más oportunidades de gol y goles marcados
top_5_teams = shots_goals_df.sort_values(by=['Goals', 'Shots'], ascending=False).head(5)
print("Top 5 equipos con más oportunidades de gol y goles marcados:")
print(top_5_teams)

# Graficar los 10 mejores equipos con más oportunidades de gol y goles marcados
top_10_teams = shots_goals_df.sort_values(by=['Goals', 'Shots'], ascending=False).head(10)

plt.figure(figsize=(12, 8))
plt.bar(top_10_teams.index, top_10_teams['Shots'], label='Shots')
plt.bar(top_10_teams.index, top_10_teams['Goals'], label='Goals')
plt.title('Top 10 Teams with Most Shots and Goals', fontsize=16)
plt.xlabel('Team', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.show()