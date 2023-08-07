import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

data_players = pd.read_excel('NHL2.xlsx')
data_players = data_players[data_players['Season'] == 2021]
age_data = {
    'Age': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'Peak': [1.6, 1.65, 1.85, 1.95, 2.23, 2.39, 2.5, 2.49, 2.46, 2.34, 2.34, 2.31, 2.23, 2.12]
}
age_data = pd.DataFrame(age_data)
data = pd.merge(data_players, age_data, on='Age', how='left')
team_ranking_data = {
    'Team_ me': ['Florida Panthers', 'Colorado Avalanche', 'Carolina Hurricanes', 'Toronto Maple Leafs', 'Minnesota Wild', 'Calgary Flames', 'New York Rangers', 'Tampa Bay Lightning', 'St. Louis Blues', 'Boston Bruins', 'Edmonton Oilers', 'Pittsburgh Penguins', 'Washington Capitals', 'Los Angeles Kings', 'Dallas Stars', 'Nashville Predators', 'Vegas Golden Knights', 'Vancouver Canucks', 'Winnipeg Jets', 'New York Islanders', 'Columbus Blue Jackets', 'San Jose Sharks', 'Anaheim Ducks', 'Buffalo Sabres', 'Detroit Red Wings', 'Ottawa Senators', 'Chicago Blackhawks', 'New Jersey Devils', 'Philadelphia Flyers', 'Seattle Kraken', 'Arizona Coyotes', 'Montreal Canadiens'],
    'Team_PTS': [122, 119, 116, 115, 113, 111, 110, 110, 109, 107, 104, 103, 100, 99, 98, 97, 94, 92, 89, 84, 81, 77, 76, 75, 74, 73, 68, 63, 61, 60, 57, 55]
}
team_ranking_data = pd.DataFrame(team_ranking_data)
data['Team_ me'] = data['Team_ me'].replace('A heim Ducks', 'Anaheim Ducks')
data = pd.merge(data, team_ranking_data, on='Team_ me', how='left')
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        data[column] = data[column].fillna(data[column].mean())

data['FOW_ratio'] = data['FOW'] / (data['FOW'] + data['FOL'])
features = ['Age', 'GP', 'G', 'A', 'PTS', 'PlusMinus', 'PIM', 'S', 'HIT', 'FOW', 'FOL', 'Peak', 'PTS', 'TOI', 'FOW_ratio']
X = data[features]
y = data[['G', 'PTS', 'A', 'PlusMinus']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = np.nan_to_num(X_train_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                   n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_scaled, y_train)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
importances = best_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for i in range(X.shape[1]):
    print(f"{features[sorted_indices[i]]}: {importances[sorted_indices[i]]}")

top_features = [features[i] for i in sorted_indices[:10]]
X_train_scaled = X_train_scaled[:, sorted_indices[:10]]
X_test_scaled = X_test_scaled[:, sorted_indices[:10]]
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
predictions_df = pd.DataFrame(y_pred, columns=['G_pred', 'PTS_pred', 'A_pred', 'PlusMinus_pred'])
predictions_df['Player_ me'] = data.loc[X_test.index, 'Player_ me'].values
idx_max = predictions_df.groupby('Player_ me')['PTS_pred'].idxmax()
predictions_df_max = predictions_df.loc[idx_max]
predictions_df_max = predictions_df_max[['Player_ me', 'G_pred', 'PTS_pred', 'A_pred', 'PlusMinus_pred']]
predictions_df_max.to_excel('predictions_NHL2022.v9.xlsx', index=False)
