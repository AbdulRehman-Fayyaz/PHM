import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("bearing_features.csv")
X = df.drop(columns="label")      # the features
y = df["label"]                   # the right answers

# 70% to learn from, 30% hidden for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Test A: only the simple features
# Test B: all 6 features
for name, cols in [("Only RMS + kurtosis", ["rms", "kurtosis"]),
                   ("All 6 features", list(X.columns))]:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])
    print(f"{name}: accuracy = {accuracy_score(y_test, pred):.2f}")

# Details for the full model
labels = ["healthy", "outer", "inner"]
print("\nConfusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels),
                   index=labels, columns=labels))

print("\nFeature importance:")
print(pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).round(3))
