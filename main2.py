import nflreadpy as nfl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

years = list(range(2012,2026))

df = nfl.load_team_stats(years, 'week') # for now just 2023 - update later for more years, more data.

team_stats = df.to_pandas()

#team_stats.to_excel('Test.xlsx')

# Removing 'win' column from predictors
team_stats_nowins = [c for c in team_stats.columns if c != 'win'] 

TEAM_A = "ATL"
TEAM_B = "TB"

mask = (
    ((team_stats["team"] == TEAM_A) | (team_stats["opponent_team"] == TEAM_B)) |
    ((team_stats["team"] == TEAM_B) | (team_stats["opponent_team"] == TEAM_A))
)

# Creates a new table with only those games
matchups_df = team_stats.loc[mask, team_stats_nowins].copy()

print(matchups_df)

games =  nfl.load_schedules(seasons=years).to_pandas()

# Filter to only matchups between TEAM_A and TEAM_B
sched_mask = (
    ((games["home_team"] == TEAM_A) | (games["away_team"] == TEAM_B)) |
    ((games["home_team"] == TEAM_B) | (games["away_team"] == TEAM_A))
)
games = games.loc[sched_mask].copy()

# Add team_a_win column: 1 if TEAM_A won, 0 if TEAM_A lost
games["team_a_win"] = games.apply(
    lambda row: 1 if (row["home_team"] == TEAM_A and row["home_score"] > row["away_score"]) or 
                     (row["away_team"] == TEAM_A and row["away_score"] > row["home_score"]) else 0,
    axis=1
)
#preview of list, check no errors appear
print(games[["game_id", "home_team", "away_team", "home_score", "away_score", "team_a_win"]])
games = games.fillna(0) #prevent future bugs, taking NaN as wrong datatype

# Use only features that exist in both games and team_stats
all_features = [i for i in games.columns if i not in ["team_a_win"] and games[i].dtype != 'object']
features = [f for f in all_features if f in team_stats.columns]

y = games["team_a_win"]
x = games[features]

print(x)
print(y)

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

model = LogisticRegression(max_iter=1000)
model.fit(scaled_x,y)

t10 = pd.Series(model.coef_[0], index=features).sort_values(ascending=False).head(10)

print("Top 10 coefficients")
print(t10)

df_2025 = df.filter(df["season"] == 2025)

print (df_2025)

# Select only features that exist in team_stats
A = df_2025.filter(df_2025["team"] == TEAM_A).select(features).mean()
B = df_2025.filter(df_2025["team"] == TEAM_B).select(features).mean()

A_prob = model.predict_proba(scaler.transform([A.to_numpy().flatten().tolist()]))[0][1]
B_prob = model.predict_proba(scaler.transform([B.to_numpy().flatten().tolist()]))[0][1]

print("Falcons win probability:", A_prob)
print("Buccaneers win probability:", B_prob)
