import nflreadpy as nfl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

years = 2022 #list(range(2012,2025))

df = nfl.load_team_stats(years, 'week') # for now just 2023 - update later for more years, more data.

team_stats = df.to_pandas()

team_stats.to_excel('Test.xlsx')

# schedules: one row per game
team_stats["win"] = (team_stats["points_for"] > df["points_against"]).astype(int)

# Removing 'win' column from predictors
cols = [c for c in df.columns.tolist() if c != 'win'] #

print('cols:' + cols)

# We must now adjust the df so we can filter by team

team1 = "KC"    # kansas city
team2 = "BUF" # Buffalo

df_filtered = df[team_stats['team'].isin([team1, team2])].copy()

print(f"Total rows: {len(df_filtered)}")
print(f"Teams present: {df_filtered['team'].unique()}")


games = nfl.load_schedules(years).to_pandas()
games.to_excel('Test2.xlsx')


print(df_filtered)
mse_list = []
for col in cols:
    Category = col
    print(Category)
    # select the entire column (as a DataFrame) by name, header is the column name, not part of the data
    x = df[[col]]
    y = df['win']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    lreg = LogisticRegression()
    lreg.fit(x_train, y_train)

    y_pred = lreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    mse_list.append(mse)