import nflreadpy as nfl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

years = 2022 #list(range(2012,2025))



print(years)

#df = nfl.load_player_stats(seasons=years, summary_level="week") #load in stats

df = nfl.load_team_stats(years, 'week') # for now just 2023 - update later for more years, more data.

df = df.to_pandas()
# schedules: one row per game
games = nfl.load_schedules(years).to_pandas()

# 2) Home team rows
home = games[['season', 'week',
              'home_team', 'away_team',
              'home_score', 'away_score']].rename(columns={
    'home_team':  'team',
    'away_team':  'opponent_team',
    'home_score': 'points_for',
    'away_score': 'points_against',
})
home['win'] = (home['points_for'] > home['points_against']).astype(int)

# 3) Away team rows
away = games[['season', 'week',
              'home_team', 'away_team',
              'home_score', 'away_score']].rename(columns={
    'away_team':  'team',
    'home_team':  'opponent_team',
    'away_score': 'points_for',
    'home_score': 'points_against',
})
away['win'] = (away['points_for'] > away['points_against']).astype(int)

# 4) One row per (season, week, team, opponent_team)
schedule_long = pd.concat([home, away], ignore_index=True)[
    ['season', 'week', 'team', 'opponent_team', 'win']
]

# 5) Merge win flag into your team stats
df = df.merge(
    schedule_long,
    on=['season', 'week', 'team', 'opponent_team'],
    how='left'
)

# IMPORTANT: don't use 'win' itself as a predictor column
cols = [c for c in df.columns.tolist() if c != 'win']

mse_list = []
for col in cols:
    Category = col
    print(Category)
    # select the entire column (as a DataFrame) by name, header is the column name, not part of the data
    x = df[[col]]
    y = df['win']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    lreg = LogisticRegression()
    lreg.fit(x_train, y_train)

    y_pred = lreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    mse_list.append(mse)