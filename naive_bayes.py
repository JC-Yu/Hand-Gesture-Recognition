import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import joblib

# ========================== Parameters ==========================
FEATURE_PATH = "./gesture_features.csv"
MODEL_PATH = "./gesture_model_NB.pkl"
# ================================================================

if __name__ == "__main__":
    # TODO 1: Load Features
    df = pd.read_csv(FEATURE_PATH)
    X = df.iloc[:, :14].values  # 14 dimension features
    y = df.iloc[:, 14].values   # gesture labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # TODO 2: Train
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)

    # TODO 3: Validation
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"==> Training Accuracy = {train_accuracy * 100 :.2f} %,"
          f" Validation Accuracy = {test_accuracy * 100 :.2f} %")

    # TODO 4: Save Model
    joblib.dump(clf, MODEL_PATH)
    print(f"==> Model had been saved in {MODEL_PATH}")
