import pandas as pd
df=pd.read_csv("speed.csv")
df = df.apply(lambda col: col.map(
    lambda x: x[2:-1] if isinstance(x, str) and x.startswith("b'") else x
))
base_cols = [
    'gender', 'age', 'age_o', 'd_age',
    'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
    'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner',
    'sports', 'exercise', 'dining', 'movies', 'music', 'reading', 'gaming',
    'expected_num_matches', 'expected_num_interested_in_me',
    'like', 'guess_prob_liked',
    'match','race','samerace','importance_same_race', 'importance_same_religion'
]
df = df[base_cols]
df = df.drop(columns=['expected_num_interested_in_me'])
df = df.dropna(subset=['age', 'age_o'])
df.fillna(df.median(numeric_only=True), inplace=True)
df['match'] = df['match'].astype(int)
y = df['match']
X = df.drop(columns=['match'])
X = X.drop(columns=['like', 'guess_prob_liked'])
X = pd.get_dummies(X, drop_first=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# train model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# predict
y_probs = rf.predict_proba(X_test)[:, 1]

y_pred_rf = (y_probs > 0.3).astype(int)
# evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
importance = pd.Series(rf.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))
