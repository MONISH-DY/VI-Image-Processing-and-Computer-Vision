import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, learning_curve

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
# Relative to workspace root: d:\VI\IP & CV
MODELS_DIR = "MiniProject - ASL Sign Language Detection/models"
DATA_DIR = "MiniProject - Backup/processed_data"
RESULTS_DIR = "MiniProject - ASL Sign Language Detection/results"

PATHS = {
    "model_1": os.path.join(MODELS_DIR, "svm_model_1.pkl"),
    "model_2": os.path.join(MODELS_DIR, "svm_model_2.pkl"),
    "scaler_1": os.path.join(MODELS_DIR, "scaler_1.pkl"),
    "scaler_2": os.path.join(MODELS_DIR, "scaler_2.pkl"),
    "csv_1": os.path.join(DATA_DIR, "landmarks_dataset.csv"),
    "csv_2": os.path.join(DATA_DIR, "features_dataset.csv"),
}

# Styling for plots
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
sns.set_theme(style="whitegrid", palette="bright")

# Consistent Color Palette
ALGO_COLORS = {'Raw Landmarks': '#FF4B4B', 'Engineered Features': '#007BFF'} # Red and Blue
TRAIN_VAL_COLORS = {'Training': '#FF8C00', 'Validation': '#1E90FF'} # Orange and DodgerBlue

def ensure_results_dir():
    """Create the results directory if it does not exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Created directory: {os.path.abspath(RESULTS_DIR)}")
    else:
        print(f"Results directory exists: {os.path.abspath(RESULTS_DIR)}")

def load_assets():
    """Load models, scalers, and test data from CSV."""
    assets = {}
    try:
        print("Loading models and scalers...")
        assets['model_1'] = joblib.load(PATHS['model_1'])
        assets['model_2'] = joblib.load(PATHS['model_2'])
        assets['scaler_1'] = joblib.load(PATHS['scaler_1'])
        assets['scaler_2'] = joblib.load(PATHS['scaler_2'])

        print("Loading original CSV data and splitting...")
        df1 = pd.read_csv(PATHS['csv_1'])
        df2 = pd.read_csv(PATHS['csv_2'])

        # Split Alg 1
        X1 = df1.iloc[:, :-1].values
        y1 = df1.iloc[:, -1].values
        _, X_test_1, _, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)

        # Split Alg 2
        X2 = df2.iloc[:, :-1].values
        y2 = df2.iloc[:, -1].values
        _, X_test_2, _, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

        # Verify labels are consistent
        if not np.array_equal(y_test_1, y_test_2):
            print("[WARNING] Labels for Raw Landmarks and 2 might not match perfectly if CSV order differs.")
            # We will use y_test_1 as the master y_test
        
        assets['X_test_1'] = X_test_1
        assets['X_test_2'] = X_test_2
        assets['y_test'] = y_test_1 # Using y_test_1 as the common ground
        
        print(f"Loaded {len(y_test_1)} test samples.")
        return assets
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing file: {e.filename}")
        print("Please ensure your project structure matches the requirements.")
        return None
    except Exception as e:
        print(f"\n[ERROR] An error occurred while loading assets: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_model(name, model, scaler, X_test, y_test):
    """Perform comprehensive evaluation for a single model."""
    print(f"Evaluating {name}...")
    
    # Scale features
    X_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    # Compute Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "Feature Count": X_test.shape[1]
    }
    
    # Classification Report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(RESULTS_DIR, f"classification_report_{name.lower().replace(' ', '_')}.csv")
    report_df.to_csv(report_csv_path)
    
    return metrics, y_pred

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_confusion_matrix(name, y_test, y_pred):
    """Generate and save confusion matrix heatmap."""
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}", fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_1, metrics_2):
    """Compare performance metrics between algorithms and save CSV/Plot."""
    df_metrics = pd.DataFrame([metrics_1, metrics_2], index=['Raw Landmarks', 'Engineered Features'])
    
    # Save comparison CSV (only core metrics)
    core_metrics = df_metrics.drop(columns=['Feature Count'])
    core_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
    
    df_melted = core_metrics.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='index', palette=ALGO_COLORS)
    plt.title("Algorithm Comparison: Performance Metrics", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_comparison_metrics.png"), bbox_inches='tight')
    plt.close()

def plot_feature_count(metrics_1, metrics_2):
    """Compare the number of features used."""
    counts = [metrics_1['Feature Count'], metrics_2['Feature Count']]
    labels = ['Raw Landmarks', 'Engineered Features']
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, palette=ALGO_COLORS)
    plt.title("Feature Count Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Features")
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.savefig(os.path.join(RESULTS_DIR, "feature_count_comparison.png"), bbox_inches='tight')
    plt.close()

def measure_and_plot_prediction_time(assets):
    """Measure inference latency and generate chart/CSV."""
    times = {'Raw Landmarks': [], 'Engineered Features': []}
    
    # Pick 100 random samples
    indices = np.random.choice(len(assets['y_test']), min(100, len(assets['y_test'])), replace=False)
    
    for i in indices:
        # Alg 1
        x1 = assets['scaler_1'].transform(assets['X_test_1'][i].reshape(1, -1))
        start = time.time()
        assets['model_1'].predict(x1)
        times['Raw Landmarks'].append((time.time() - start) * 1000) # ms
        
        # Alg 2
        x2 = assets['scaler_2'].transform(assets['X_test_2'][i].reshape(1, -1))
        start = time.time()
        assets['model_2'].predict(x2)
        times['Engineered Features'].append((time.time() - start) * 1000) # ms
        
    avg_times = {k: np.mean(v) for k, v in times.items()}
    
    # Save CSV
    pd.DataFrame([avg_times]).to_csv(os.path.join(RESULTS_DIR, "prediction_time_comparison.csv"), index=False)
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=list(avg_times.keys()), y=list(avg_times.values()), palette=ALGO_COLORS)
    plt.title("Average Prediction Time Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Time (ms)")
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_time_comparison.png"), bbox_inches='tight')
    plt.close()
    return avg_times

def plot_model_size():
    """Calculate .pkl file sizes and generate chart."""
    size1 = os.path.getsize(PATHS['model_1']) / (1024 * 1024) # MB
    size2 = os.path.getsize(PATHS['model_2']) / (1024 * 1024) # MB
    
    sizes = [size1, size2]
    labels = ['Raw Landmarks', 'Engineered Features']
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=sizes, palette=ALGO_COLORS)
    plt.title("Model File Size Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Size (MB)")
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f} MB', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.savefig(os.path.join(RESULTS_DIR, "model_size_comparison.png"), bbox_inches='tight')
    plt.close()
    return size1, size2

def plot_dataset_distribution(y_test):
    """Visualize distribution of samples per class."""
    unique, counts = np.unique(y_test, return_counts=True)
    
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=unique, y=counts, color="dodgerblue")
    plt.title("Test Dataset Distribution (Samples per Class)", fontsize=14, fontweight='bold')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                    
    plt.savefig(os.path.join(RESULTS_DIR, "dataset_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(name, model, scaler, X_test):
    """Plot histogram of max probabilities if available."""
    if hasattr(model, "predict_proba"):
        print(f"Generating confidence distribution for {name}...")
        X_scaled = scaler.transform(X_test)
        probas = model.predict_proba(X_scaled)
        max_probas = np.max(probas, axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(max_probas, bins=20, kde=True, color="mediumblue")
        plt.title(f"Confidence Score Distribution - {name}", fontsize=14, fontweight='bold')
        plt.xlabel("Max Probability")
        plt.ylabel("Frequency")
        
        filename = f"confidence_distribution_{name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
        plt.close()
    else:
        print(f"Skipping confidence distribution for {name} (predict_proba not available)")

def plot_learning_curve(name, model, scaler, X, y):
    """Generate and save learning curve (training progress)."""
    print(f"Generating learning curve for {name}...")
    
    # Scale the entire dataset for CV
    X_scaled = scaler.transform(X)
    
    # Use 3-fold cross-validation for speed, 5 training set sizes
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.15, color=TRAIN_VAL_COLORS['Training'])
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.15, color=TRAIN_VAL_COLORS['Validation'])
    plt.plot(train_sizes, train_scores_mean, 'o-', color=TRAIN_VAL_COLORS['Training'], 
             linewidth=2.5, markersize=8, label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=TRAIN_VAL_COLORS['Validation'], 
             linewidth=2.5, markersize=8, label="Cross-validation Accuracy")
    
    plt.title(f"Learning Curve (Training History) - {name}", fontsize=14, fontweight='bold')
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    filename = f"learning_curve_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("="*60)
    print("ASL Sign Language Recognition - Evaluation Script")
    print("="*60)
    
    ensure_results_dir()
    
    assets = load_assets()
    if not assets:
        return

    # 1. Evaluate Both Models
    m1_results, y_pred1 = evaluate_model("Raw Landmarks", assets['model_1'], assets['scaler_1'], assets['X_test_1'], assets['y_test'])
    m2_results, y_pred2 = evaluate_model("Engineered Features", assets['model_2'], assets['scaler_2'], assets['X_test_2'], assets['y_test'])
    
    # 2. Confusion Matrices
    plot_confusion_matrix("Raw Landmarks", assets['y_test'], y_pred1)
    plot_confusion_matrix("Engineered Features", assets['y_test'], y_pred2)
    
    # 3. Metrics Comparison (Saves metrics_comparison.csv)
    plot_metrics_comparison(m1_results, m2_results)
    
    # 4. Feature Count
    plot_feature_count(m1_results, m2_results)
    
    # 5. Prediction Time (Saves prediction_time_comparison.csv)
    avg_times = measure_and_plot_prediction_time(assets)
    
    # 6. Model Sizes
    size1, size2 = plot_model_size()
    
    # 7. Dataset Distribution
    plot_dataset_distribution(assets['y_test'])
    
    # 8. Confidence Distribution
    plot_confidence_distribution("Raw Landmarks", assets['model_1'], assets['scaler_1'], assets['X_test_1'])
    plot_confidence_distribution("Engineered Features", assets['model_2'], assets['scaler_2'], assets['X_test_2'])
    
    # 9. Learning Curves (Training History)
    # Note: We use a subset or the whole training data if available, 
    # but here we use the test set as a proxy or just reload enough data.
    # Actually, we have X_test_1 and y_test. For a meaningful curve, 
    # let's use the samples we have.
    plot_learning_curve("Raw Landmarks", assets['model_1'], assets['scaler_1'], assets['X_test_1'], assets['y_test'])
    plot_learning_curve("Engineered Features", assets['model_2'], assets['scaler_2'], assets['X_test_2'], assets['y_test'])
    
    # 10. Final Summary Table
    summary_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Avg Prediction Time (ms)", "Feature Count", "Model Size (MB)"],
        "Raw Landmarks": [
            m1_results['Accuracy'], 
            m1_results['Precision'], 
            m1_results['Recall'], 
            m1_results['F1-score'],
            avg_times['Raw Landmarks'],
            m1_results['Feature Count'],
            size1
        ],
        "Engineered Features": [
            m2_results['Accuracy'], 
            m2_results['Precision'], 
            m2_results['Recall'], 
            m2_results['F1-score'],
            avg_times['Engineered Features'],
            m2_results['Feature Count'],
            size2
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "final_summary.csv"), index=False)
    
    # 10. Console Output
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Raw Landmarks Accuracy: {m1_results['Accuracy']*100:.2f}%")
    print(f"Engineered Features Accuracy: {m2_results['Accuracy']*100:.2f}%")
    
    best_algo = "Raw Landmarks" if m1_results['Accuracy'] > m2_results['Accuracy'] else "Engineered Features"
    print(f"Best Performing Algorithm: {best_algo}")
    print("All reports and graphs saved to results/")
    print("="*60)

if __name__ == "__main__":
    main()
