import pandas as pd
import json

# Función para cargar datos desde un archivo JSON
def load_json(file_path):
    with open(file_path, 'r') as file:
        return pd.DataFrame(json.load(file))

# Cargar datos de equipos
teams_data = load_json("teams.json")

# Cargar datos de partidos
matches_data = load_json("matches_World_Cup.json")

# Filtrar goles de los datos de partidos
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
        goals_dict[team_name] = team_info.get('score', 0)
    return goals_dict

team_goals = matches_data['teamsData'].apply(goals_by_team)

# Combinar goles por equipo en un solo DataFrame
goals_df = pd.DataFrame(list(team_goals)).sum().reset_index()
goals_df.columns = ['team', 'total_goals']

# Filtrar equipos semifinalistas y cuartofinalistas
semifinal_and_quarterfinal_teams = ['France', 'Belgium', 'England', 'Croatia', 'Brazil', 'Uruguay', 'Russia', 'Sweden']
goals_df = goals_df[goals_df['team'].isin(semifinal_and_quarterfinal_teams)]

# Mostrar los 5 equipos con más goles
top_teams_goals = goals_df.sort_values(by='total_goals', ascending=False).head(5)
print("Top 5 equipos con más goles:")
print(top_teams_goals)
