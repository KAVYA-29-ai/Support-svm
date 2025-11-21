!pip install kagglehub

import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

%matplotlib inline



# DOWNLOAD DATASET

path = kagglehub.dataset_download("ranajawadriaz/uci-adult-income-dataset")
print("Dataset downloaded to:", path)

import os
files = [f for f in os.listdir(path) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(path, files[0]))
print("CSV file loaded:", files[0])




# EDA

print("Shape:", df.shape)
print("\n--- INFO ---")
df.info()

print("\n--- DESCRIPTION ---")
print(df.describe(include='all').T)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Income distribution
sns.countplot(data=df, x="income")
plt.title("Income Class Distribution")
plt.show()

print(df["income"].value_counts(normalize=True) * 100)


# Numerical distributions
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols].hist(figsize=(12, 10), bins=30)
plt.show()


# Categorical value distribution
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()


# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Boxplots for outliers
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot – {col}")
    plt.show()


# Income vs Numerical Columns
for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=df, x="income", y=col)
    plt.title(f"{col} vs Income")
    plt.show()


# Income vs Categorical Columns
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, hue="income")
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Income")
    plt.show()




# DATA CLEANING & PREPROCESSING
# Replace '?' with NaN
df = df.replace("?", np.nan)

# Drop missing values
df = df.dropna()

print("\nAfter dropping missing values:", df.shape)



# ENCODING CATEGORICAL COLUMNS

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCategorical columns encoded.")




# SEPARATE FEATURES & TARGET

X = df.drop("income", axis=1)
y = df["income"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)




# SCALING NUMERICAL COLUMNS
# ---- Option 1: StandardScaler (Recommended for ML models) ----
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ---- Option 2: MinMaxScaler (Uncomment if needed) ----
# scaler = MinMaxScaler()
# X[num_cols] = scaler.fit_transform(X[num_cols])

print("\nFeature scaling completed.")




# FINAL OUTPUT

print("\n--- Final Cleaned + Encoded + Scaled Dataset Head ---")
print(df.head())

print("\nREADY FOR MODEL TRAINING ✔\n")




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.svm import SVC

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------
# 1️⃣ TRADITIONAL BATCH SVM (FULL DATA TRAINING)
# -----------------------------------------------------
batch_model = SVC(kernel="rbf", probability=True, random_state=42)
batch_model.fit(X_train, y_train)

batch_pred = batch_model.predict(X_test)
batch_proba = batch_model.predict_proba(X_test)[:, 1]

print("\n====== Traditional SVM Metrics ======")
print("Accuracy:", accuracy_score(y_test, batch_pred))
print("Precision:", precision_score(y_test, batch_pred))
print("Recall:", recall_score(y_test, batch_pred))
print("F1 Score:", f1_score(y_test, batch_pred))
print("ROC-AUC:", roc_auc_score(y_test, batch_proba))

print("\nClassification Report:\n", classification_report(y_test, batch_pred))


# -----------------------------------------------------
# 2️⃣ STREAMING SVM (BATCH-WISE)
# -----------------------------------------------------

batch_size = 5000
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)

num_batches = len(X_train_np) // batch_size + 1

# Initialize first model
stream_model = SVC(kernel="rbf", probability=True, random_state=42)

# GLOBAL SUPPORT VECTORS
global_sv_X = None
global_sv_y = None

for i in range(num_batches):

    start = i * batch_size
    end = min((i + 1) * batch_size, len(X_train_np))

    if start >= end:
        break

    X_batch = X_train_np[start:end]
    y_batch = y_train_np[start:end]

    print(f"\nTraining on Batch {i+1}/{num_batches} → Size: {len(X_batch)}")

    # Train local SVM
    local_model = SVC(kernel="rbf", random_state=42)
    local_model.fit(X_batch, y_batch)

    # Extract local SVs
    sv_idx = local_model.support_
    local_sv_X = X_batch[sv_idx]
    local_sv_y = y_batch[sv_idx]

    # Merge global SVs
    if global_sv_X is None:
        global_sv_X = local_sv_X
        global_sv_y = local_sv_y
    else:
        global_sv_X = np.vstack((global_sv_X, local_sv_X))
        global_sv_y = np.hstack((global_sv_y, local_sv_y))

    # Retrain global SVM on merged SVs
    stream_model.fit(global_sv_X, global_sv_y)


# final prediction
stream_pred = stream_model.predict(X_test)
stream_proba = stream_model.predict_proba(X_test)[:, 1]

print("\n====== Streaming SVM Metrics ======")
print("Accuracy:", accuracy_score(y_test, stream_pred))
print("Precision:", precision_score(y_test, stream_pred))
print("Recall:", recall_score(y_test, stream_pred))
print("F1 Score:", f1_score(y_test, stream_pred))
print("ROC-AUC:", roc_auc_score(y_test, stream_proba))

print("\nClassification Report:\n", classification_report(y_test, stream_pred))


# -----------------------------------------------------
# 3️⃣ FINAL COMPARISON
# -----------------------------------------------------

print("\n================ Final Comparison ================")

print("\nTraditional SVM Results:")
print({
    "Accuracy": accuracy_score(y_test, batch_pred),
    "Precision": precision_score(y_test, batch_pred),
    "Recall": recall_score(y_test, batch_pred),
    "F1": f1_score(y_test, batch_pred),
    "ROC-AUC": roc_auc_score(y_test, batch_proba)
})

print("\nStreaming SVM Results:")
print({
    "Accuracy": accuracy_score(y_test, stream_pred),
    "Precision": precision_score(y_test, stream_pred),
    "Recall": recall_score(y_test, stream_pred),
    "F1": f1_score(y_test, stream_pred),
    "ROC-AUC": roc_auc_score(y_test, stream_proba)
})
