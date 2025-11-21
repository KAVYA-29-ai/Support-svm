import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

# Load data
df = pd.read_csv('/content/UCI_Adult_Income_Dataset.csv')
print(df.head())

# EDA - Exploratory Data Analysis
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Dataset Shape ===")
print(f"Shape: {df.shape}")

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Data Types ===")
print(df.dtypes)

# Visualizations
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Missing Values Heatmap')
plt.show()

# Distribution plots for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Count plots for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()

# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n=== DATA PREPROCESSING ===")

# Handle Missing Values
print("\n--- Handling Missing Values ---")
print(f"Missing values before handling:\n{df.isnull().sum()}")

# Drop rows with missing values or fill them
df = df.dropna()  # Drop rows with any missing values
# Alternative: Fill missing values with median/mode
# df.fillna(df.median(numeric_only=True), inplace=True)

print(f"Missing values after handling:\n{df.isnull().sum()}")

# Separate features and target variable
X = df.drop(columns=[df.columns[-1]])  # Drop last column (usually target)
y = df.iloc[:, -1]  # Last column as target

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode Categorical Variables
print("\n--- Encoding Categorical Variables ---")
categorical_features = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Encode target variable if it's categorical
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.astype(str))
    print(f"Target variable encoded: {len(le_target.classes_)} classes")

print(f"\nFeatures after encoding:\n{X.head()}")

# Standardize Features
print("\n--- Standardizing Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Mean of scaled features (should be ~0): \n{X.mean()}")
print(f"Std of scaled features (should be ~1): \n{X.std()}")

# Save preprocessed data
print("\n--- Data Preprocessing Complete ---")
print(f"Preprocessed data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================
# SVM MODEL TRAINING
# ============================================================

print("\n=== SVM MODEL TRAINING ===")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================
# 1. TRADITIONAL BATCH SVM
# ============================================================

print("\n--- Traditional Batch SVM ---")
kernel_type = 'rbf'  # Options: 'linear', 'rbf', 'poly', 'sigmoid'
print(f"Kernel type: {kernel_type}")

svm_batch = SVC(kernel=kernel_type, C=1.0, random_state=42, verbose=1)
print("Training SVM on full batch...")
svm_batch.fit(X_train, y_train)

# Evaluate Batch SVM
y_pred_batch = svm_batch.predict(X_test)
accuracy_batch = accuracy_score(y_test, y_pred_batch)
print(f"\nBatch SVM Accuracy: {accuracy_batch:.4f}")
print("\nClassification Report (Batch SVM):")
print(classification_report(y_test, y_pred_batch))

# Store support vectors from batch model
support_vectors_batch = svm_batch.support_vectors_
print(f"Number of support vectors (Batch): {len(support_vectors_batch)}")

# ============================================================
# 2. STREAMING BATCH SVM
# ============================================================

print("\n--- Streaming Batch SVM ---")

# Split remaining training data into streaming batches
n_batches = 5
batch_size = len(X_train) // n_batches
print(f"Number of batches: {n_batches}")
print(f"Batch size: {batch_size}")

all_support_vectors = []
streaming_models = []

# Train SVM on each batch
for batch_num in range(n_batches):
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size if batch_num < n_batches - 1 else len(X_train)
    
    X_batch = X_train.iloc[start_idx:end_idx]
    y_batch = y_train.iloc[start_idx:end_idx]
    
    print(f"\nBatch {batch_num + 1}: Training on {len(X_batch)} samples")
    
    # Train SVM on batch
    svm_stream = SVC(kernel=kernel_type, C=1.0, random_state=42)
    svm_stream.fit(X_batch, y_batch)
    
    # Extract and store support vectors
    support_vectors = svm_stream.support_vectors_
    support_vector_indices = svm_stream.support_
    
    print(f"  Support vectors: {len(support_vectors)}")
    all_support_vectors.extend(support_vectors)
    streaming_models.append(svm_stream)

# Convert all support vectors to DataFrame
all_support_vectors_array = pd.DataFrame(all_support_vectors, columns=X.columns)
print(f"\nTotal support vectors collected: {len(all_support_vectors_array)}")

# ============================================================
# 3. RETRAIN SVM ON MERGED SUPPORT VECTORS
# ============================================================

print("\n--- Retraining on Merged Support Vectors ---")

# Use support vectors as new training set
if len(all_support_vectors_array) > 0:
    # Get corresponding labels for support vectors
    X_sv_train = all_support_vectors_array.iloc[:min(len(all_support_vectors_array), len(X_train))]
    
    # Map support vectors to their labels (approximate by finding closest points)
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    distances, indices = nbrs.kneighbors(X_sv_train)
    y_sv_train = y_train.iloc[indices.flatten()].values
    
    print(f"Training merged SVM on {len(X_sv_train)} support vectors")
    
    svm_merged = SVC(kernel=kernel_type, C=1.0, random_state=42)
    svm_merged.fit(X_sv_train, y_sv_train)
    
    # Evaluate Merged SVM
    y_pred_merged = svm_merged.predict(X_test)
    accuracy_merged = accuracy_score(y_test, y_pred_merged)
    print(f"\nMerged SVM Accuracy: {accuracy_merged:.4f}")
    print(f"Number of support vectors (Merged): {len(svm_merged.support_vectors_)}")

# ============================================================
# 4. COMPARISON AND EVALUATION
# ============================================================

print("\n=== MODEL COMPARISON ===")

print(f"\nBatch SVM:")
print(f"  Accuracy: {accuracy_batch:.4f}")
print(f"  Support Vectors: {len(support_vectors_batch)}")
print(f"  Model Size: Relatively large")

if 'accuracy_merged' in locals():
    print(f"\nMerged SVM (from streaming batches):")
    print(f"  Accuracy: {accuracy_merged:.4f}")
    print(f"  Support Vectors: {len(svm_merged.support_vectors_)}")
    print(f"  Accuracy Difference: {abs(accuracy_batch - accuracy_merged):.4f}")

# Confusion matrices
from sklearn.metrics import confusion_matrix
cm_batch = confusion_matrix(y_test, y_pred_batch)
if 'y_pred_merged' in locals():
    cm_merged = confusion_matrix(y_test, y_pred_merged)

print("\nConfusion Matrix (Batch SVM):")
print(cm_batch)

if 'cm_merged' in locals():
    print("\nConfusion Matrix (Merged SVM):")
    print(cm_merged)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_batch, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Batch SVM Confusion Matrix\nAccuracy: {accuracy_batch:.4f}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

if 'cm_merged' in locals():
    sns.heatmap(cm_merged, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Merged SVM Confusion Matrix\nAccuracy: {accuracy_merged:.4f}')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# ============================================================
# 5. EXPORT MODELS AND ARTIFACTS
# ============================================================

print("\n--- Exporting Models ---")

joblib.dump(svm_batch, '/workspaces/Support-svm/svm_batch_model.pkl')
print("✓ Batch SVM model exported: svm_batch_model.pkl")

if 'svm_merged' in locals():
    joblib.dump(svm_merged, '/workspaces/Support-svm/svm_merged_model.pkl')
    print("✓ Merged SVM model exported: svm_merged_model.pkl")

joblib.dump(scaler, '/workspacesMountPath/Support-svm/scaler.pkl')
print("✓ Scaler exported: scaler.pkl")

joblib.dump(label_encoders, '/workspaces/Support-svm/label_encoders.pkl')
print("✓ Label encoders exported: label_encoders.pkl")

# Save support vectors
pd.DataFrame(support_vectors_batch, columns=X.columns).to_csv('/workspaces/Support-svm/support_vectors_batch.csv', index=False)
print("✓ Support vectors (Batch) exported: support_vectors_batch.csv")

all_support_vectors_array.to_csv('/workspaces/Support-svm/support_vectors_merged.csv', index=False)
print("✓ Support vectors (Merged) exported: support_vectors_merged.csv")

print("\n=== Training Complete ===")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

print("\n=== GENERATING VISUALIZATIONS ===")

import time
import numpy as np
from sklearn.decomposition import PCA

# For visualization, reduce dimensions to 2D using PCA
print("\nReducing dimensions to 2D for visualization...")
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# ============================================================
# 7. PLOT 1: Support Vectors Over Iterations
# ============================================================

print("\n--- Plot 1: Support Vectors Over Iterations ---")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Support Vectors Evolution Across Streaming Batches', fontsize=16, fontweight='bold')

for batch_num in range(min(n_batches, 6)):
    ax = axes[batch_num // 3, batch_num % 3]
    
    # Get batch data in 2D
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size if batch_num < n_batches - 1 else len(X_train)
    
    X_batch_2d = X_train_2d[start_idx:end_idx]
    y_batch = y_train.iloc[start_idx:end_idx]
    
    # Get support vectors from this batch's model
    sv_indices = streaming_models[batch_num].support_
    sv_2d = X_batch_2d[sv_indices]
    
    # Plot all points in batch
    scatter = ax.scatter(X_batch_2d[:, 0], X_batch_2d[:, 1], 
                         c=y_batch, cmap='coolwarm', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Highlight support vectors
    ax.scatter(sv_2d[:, 0], sv_2d[:, 1], 
               marker='*', s=400, c='gold', edgecolors='red', linewidth=2, 
               label=f'Support Vectors ({len(sv_2d)})')
    
    ax.set_title(f'Batch {batch_num + 1}\n({len(X_batch_2d)} samples, {len(sv_2d)} SVs)', fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 8. PLOT 2: Decision Boundaries
# ============================================================

print("\n--- Plot 2: Decision Boundaries ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Decision Boundaries: Batch SVM vs Merged SVM', fontsize=16, fontweight='bold')

# Helper function to plot decision boundary
def plot_decision_boundary(ax, model, X_2d, y, title, pca_transformer):
    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Transform mesh back to original space and predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca_transformer.inverse_transform(mesh_points)
    
    Z = model.decision_function(mesh_points_original)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    
    # Plot training points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', 
                         s=50, edgecolors='k', linewidth=0.5, alpha=0.7)
    
    # Plot support vectors
    sv_indices = model.support_
    ax.scatter(X_2d[sv_indices, 0], X_2d[sv_indices, 1], 
               marker='*', s=400, c='gold', edgecolors='red', linewidth=2,
               label=f'Support Vectors ({len(sv_indices)})')
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return scatter

plot_decision_boundary(axes[0], svm_batch, X_train_2d, y_train.values, 
                       f'Batch SVM Decision Boundary\nAccuracy: {accuracy_batch:.4f}', pca)

if 'svm_merged' in locals():
    plot_decision_boundary(axes[1], svm_merged, X_train_2d, y_train.values, 
                           f'Merged SVM Decision Boundary\nAccuracy: {accuracy_merged:.4f}', pca)

plt.tight_layout()
plt.show()

# ============================================================
# 9. PLOT 3: Accuracy vs Time Tradeoff
# ============================================================

print("\n--- Plot 3: Accuracy vs Time Tradeoff ---")

# Measure training time for different scenarios
training_times = []
accuracies = []
sv_counts = []
labels_scenarios = []

# Scenario 1: Batch SVM
print("Recording Batch SVM metrics...")
training_times.append(0)  # Already trained
accuracies.append(accuracy_batch)
sv_counts.append(len(support_vectors_batch))
labels_scenarios.append('Batch SVM')

# Scenario 2: Streaming batches (cumulative time)
cumulative_time = 0
print("Recording Streaming Batch metrics...")
for batch_num in range(n_batches):
    start_time = time.time()
    # Use pre-trained models
    cumulative_time += 0.001  # Simulated time (already trained)
    
    training_times.append(cumulative_time)
    
    # Evaluate streaming model on test set
    y_pred_streaming = streaming_models[batch_num].predict(X_test)
    acc_streaming = accuracy_score(y_test, y_pred_streaming)
    accuracies.append(acc_streaming)
    sv_counts.append(len(streaming_models[batch_num].support_vectors_))
    labels_scenarios.append(f'Batch {batch_num + 1}')

# Scenario 3: Merged SVM
if 'accuracy_merged' in locals():
    print("Recording Merged SVM metrics...")
    training_times.append(cumulative_time + 0.002)
    accuracies.append(accuracy_merged)
    sv_counts.append(len(svm_merged.support_vectors_))
    labels_scenarios.append('Merged SVM')

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Accuracy vs Iteration
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(range(len(accuracies)), accuracies, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
ax1.fill_between(range(len(accuracies)), accuracies, alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Scenario', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy Across Scenarios', fontweight='bold', fontsize=12)
ax1.set_xticks(range(len(labels_scenarios)))
ax1.set_xticklabels(labels_scenarios, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
for i, acc in enumerate(accuracies):
    ax1.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Support Vectors vs Iteration
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(range(len(sv_counts)), sv_counts, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Scenario', fontweight='bold', fontsize=11)
ax2.set_ylabel('Number of Support Vectors', fontweight='bold', fontsize=11)
ax2.set_title('Support Vectors Count', fontweight='bold', fontsize=12)
ax2.set_xticks(range(len(labels_scenarios)))
ax2.set_xticklabels(labels_scenarios, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for i, (bar, sv) in enumerate(zip(bars, sv_counts)):
    ax2.text(bar.get_x() + bar.get_width()/2, sv + 5, str(sv), 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 3: Accuracy vs Support Vectors (Tradeoff)
ax3 = fig.add_subplot(gs[1, 0])
scatter = ax3.scatter(sv_counts, accuracies, s=200, c=range(len(accuracies)), 
                      cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
for i, label in enumerate(labels_scenarios):
    ax3.annotate(label, (sv_counts[i], accuracies[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
ax3.set_xlabel('Number of Support Vectors', fontweight='bold', fontsize=11)
ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax3.set_title('Accuracy vs Support Vectors Tradeoff', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary Statistics
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = f"""
╔════════════════════════════════════════╗
║     MODEL PERFORMANCE SUMMARY          ║
╠════════════════════════════════════════╣
║ Batch SVM:                             ║
║   • Accuracy: {accuracy_batch:.4f}              ║
║   • Support Vectors: {len(support_vectors_batch)}          ║
║   • Training: Full Dataset             ║
║                                        ║
║ Streaming Batches:                     ║
║   • Number of Batches: {n_batches}                ║
║   • Batch Size: {batch_size}              ║
║   • Total SVs Collected: {len(all_support_vectors_array)}        ║
║                                        ║
║ Merged SVM:                            ║
║   • Accuracy: {accuracy_merged:.4f}              ║
║   • Support Vectors: {len(svm_merged.support_vectors_)}          ║
║   • Training: {len(X_sv_train)} Merged SVs         ║
║                                        ║
║ Accuracy Difference:                   ║
║   • Batch vs Merged: {abs(accuracy_batch - accuracy_merged):.6f}  ║
║                                        ║
║ Compression Ratio:                     ║
║   • Original SVs / Merged SVs:         ║
║     {len(all_support_vectors_array)} / {len(svm_merged.support_vectors_)} = {len(all_support_vectors_array)/max(len(svm_merged.support_vectors_), 1):.2f}x        ║
╚════════════════════════════════════════╝
"""

ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('SVM Performance Analysis: Batch vs Streaming vs Merged', 
             fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ============================================================
# 10. PLOT 4: Detailed Comparison Table
# ============================================================

print("\n--- Plot 4: Detailed Comparison ---")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

comparison_data = [
    ['Metric', 'Batch SVM', 'Merged SVM', 'Difference'],
    ['Accuracy', f'{accuracy_batch:.4f}', f'{accuracy_merged:.4f}', f'{abs(accuracy_batch - accuracy_merged):.6f}'],
    ['Support Vectors', f'{len(support_vectors_batch)}', f'{len(svm_merged.support_vectors_)}', f'{len(support_vectors_batch) - len(svm_merged.support_vectors_)}'],
    ['Model Complexity', 'High', 'Low', '-'],
    ['Training Data Size', f'{len(X_train)}', f'{len(X_sv_train)}', f'{len(X_train) - len(X_sv_train)}'],
    ['Kernel Type', f'{kernel_type}', f'{kernel_type}', 'Same'],
]

table = ax.table(cellText=comparison_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(comparison_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F4F8')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('Batch SVM vs Merged SVM: Detailed Comparison', fontweight='bold', fontsize=14, pad=20)
plt.show()

print("\n=== VISUALIZATIONS COMPLETE ===")

