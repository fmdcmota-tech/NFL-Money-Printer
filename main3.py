import nflreadpy as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

years = list(range(2012,2026))

team_stats_polars = nfl.load_team_stats(seasons=years, summary_level="week")
team_stats = team_stats_polars.to_pandas()

TEAM_A = "ARI"
TEAM_B = "DAL"

team_stats = team_stats[team_stats["team"].isin([TEAM_A, TEAM_B])]

sched_polars = nfl.load_schedules(seasons=years)
games = sched_polars.to_pandas()

records = []
for _, r in games.iterrows():
    hs = r["home_score"]
    ascore = r["away_score"]
    if pd.notna(hs) and pd.notna(ascore):
        if r["home_team"] in (TEAM_A, TEAM_B):
            records.append([r["season"], r["week"], r["home_team"], int(hs > ascore)])
        if r["away_team"] in (TEAM_A, TEAM_B):
            records.append([r["season"], r["week"], r["away_team"], int(ascore > hs)])

win_df = pd.DataFrame(records, columns=["season","week","team","win"])

df = pd.merge(
    team_stats,
    win_df,
    on=["season","week","team"],
    how="inner"
)

drop_cols = ["season_type","opponent_team","fg_made_list","fg_missed_list","fg_blocked_list"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])


df = df.fillna(0)

features = [c for c in df.columns if c not in ["win"] and df[c].dtype != "object"]
X = df[features]
y = df["win"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

coef_series = pd.Series(model.coef_[0], index=features)
top10 = coef_series.abs().sort_values(ascending=False).head(10)
print("Top 10 Features:")
print(top10)

df_2025 = df[df["season"] == 2024]

A = df_2025[df_2025["team"] == TEAM_A][features].mean().fillna(0)
B = df_2025[df_2025["team"] == TEAM_B][features].mean().fillna(0)

A_prob = model.predict_proba(scaler.transform([A]))[0][1]
B_prob = model.predict_proba(scaler.transform([B]))[0][1]

print("TEAM_A win probability:", A_prob)
print("TEAM_B win probability:", B_prob)
