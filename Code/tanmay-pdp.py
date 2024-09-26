import optuna
import warnings
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_excel("../Data/dataproject2024.xlsx")

# Define the feature matrix X and the target variable y
X = df.drop(columns=["Default (y)", "Pred_default (y_hat)", "PD", "Group", "ID"])
y = df["Default (y)"]

# Display the first few rows of X and y to verify
print(X.head())
print(y.head())


# Monthly payment to income ratio
X["Monthly payment to income ratio"] = X["Monthly payment"] / (
    X["Funding amount"] + 1e-9
)  # Adding a small value to avoid division by zero

# Loan to car price ratio
X["Loan to car price ratio"] = X["Funding amount"] / (
    X["Car price"] + 1e-9
)  # Adding a small value to avoid division by zero


# Interaction terms between job tenure, age, and monthly payment
X["Job tenure * Age"] = X["Job tenure"] * X["Age"]
X["Job tenure * Monthly payment"] = X["Job tenure"] * X["Monthly payment"]
X["Age * Monthly payment"] = X["Age"] * X["Monthly payment"]


# Display the first few rows of the updated DataFrame to verify
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display the shapes of the resulting datasets to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define the objective function for Optuna
def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Split the data
    X_train_2, X_valid, y_train_2, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    class_weights = {0: 1, 1: len(y_train) / sum(y_train)}
    # Train the model
    model = xgb.XGBClassifier(
        **param, use_label_encoder=False, class_weights=class_weights
    )
    model.fit(X_train_2, y_train_2)

    # Predict on the validation set
    y_pred = model.predict(X_valid)

    # Calculate the balanced accuracy
    balanced_acc = balanced_accuracy_score(y_valid, y_pred)
    return balanced_acc
    
# Ignore all warnings
warnings.filterwarnings("ignore")

# Create a study object and optimize the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the best parameters
print("Best parameters:", study.best_params)
print("Best balanced accuracy:", study.best_value)

# Train the best model using the best parameters from the Optuna study with class imbalance weights
best_params = study.best_params

# Calculate class weights
class_weights = {0: 1, 1: len(y_train) / sum(y_train)}
best_model = xgb.XGBClassifier(
    **best_params, class_weights=class_weights, eval_metric="logloss"
)

# Train the model
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best = best_model.predict(X_test)

# Calculate the balanced accuracy
balanced_acc_best = balanced_accuracy_score(y_test, y_pred_best)
print("Balanced Accuracy (Best Model):", balanced_acc_best)

# Confusion Matrix
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Best Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(
    "Classification Report (Best Model):\n", classification_report(y_test, y_pred_best)
)

# ROC Curve and AUC
fpr_best, tpr_best, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc_best = auc(fpr_best, tpr_best)
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_best,
    tpr_best,
    color="darkorange",
    lw=2,
    label="ROC curve (area = %0.2f)" % roc_auc_best,
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (Best Model)")
plt.legend(loc="lower right")
plt.show()