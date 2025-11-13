import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    balanced_accuracy_score
)

print("Loading dataset...")
data = pd.read_csv("student_data_large.csv")
print("Dataset loaded successfully.\n")

X = data[['regMid', 'regEnd', 'Final', 'attendance']]
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)
print("Model trained successfully.\n")

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%\n")

cv_scores = cross_val_score(model, X, y, cv=5, scoring='balanced_accuracy')
print(f"5-Fold Cross-Validation Balanced Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=model.classes_, zero_division=0))

plt.figure(figsize=(6, 4))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.barh(importances.index, importances.values, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(cmap='Blues', xticks_rotation=90, colorbar=True, ax=ax)
plt.title("Confusion Matrix — Random Forest Model", fontsize=13, pad=15)
plt.xlabel("Predicted label", fontsize=11)
plt.ylabel("True label", fontsize=11)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.subplots_adjust(bottom=0.35, top=0.9)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='status', data=data, palette='tab20')
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.axis('off')
plt.text(
    0.05, 0.6,
    "Diagonal values represent correct predictions.\n"
    "Off-diagonal cells indicate misclassifications.\n"
    "A strong diagonal pattern means the model is stable.",
    fontsize=11, va='center'
)
plt.title("Interpretation Guide")
plt.show()

joblib.dump(model, "rf_model.pkl")
print("Model saved to rf_model.pkl\n")

print("Label distribution in dataset:")
print(y.value_counts())

f1_macro = f1_score(y_test, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\nAdditional metrics:")
print(f"F1 (macro average): {f1_macro:.4f}")
print(f"Balanced Accuracy:  {balanced_acc:.4f}")
