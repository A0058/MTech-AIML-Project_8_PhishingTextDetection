from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import load_dataset, split_dataset, make_modified_test_set
from model import (
    build_logistic_regression, build_linear_svm,
    train_embedding_classifier, predict_embeddings
)

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Phishing Precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Phishing Recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Macro-F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def calculate_performance_drop(results_df):
    rows = []
    for name in results_df["Model"].unique():
        part = results_df[results_df["Model"] == name]
        original = part[part["Test Type"] == "Original"]
        modified = part[part["Test Type"] == "Modified"]
        if original.empty or modified.empty:
            continue
        original_f1 = original["Macro-F1"].iloc[0]
        modified_f1 = modified["Macro-F1"].iloc[0]
        drop = ((original_f1 - modified_f1) / original_f1 * 100
                if original_f1 != 0 else float("nan"))
        rows.append({
            "Model": name,
            "Original Macro-F1": original_f1,
            "Modified Macro-F1": modified_f1,
            "Performance Drop (%)": drop
        })
    return pd.DataFrame(rows)

def run_experiment(data_path=None, max_test_samples=4000):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "phishing_email.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Put phishing_email.csv in data/."
        )

    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    df = load_dataset(data_path)
    print("Dataset shape:", df.shape)
    print("Label distribution:\\n", df["label"].value_counts().sort_index())
    print("ASSUMPTION: label 1 = phishing, label 0 = legitimate.")
    print("Verify this mapping using the dataset's documentation.")

    X_train, X_test, y_train, y_test = split_dataset(
        df, random_state=42, max_test_samples=max_test_samples
    )
    X_modified = make_modified_test_set(X_test, random_state=42)

    print(f"Training messages: {len(X_train)}")
    print(f"Test messages: {len(X_test)}")
    print("\\nOriginal example:", X_test.iloc[0][:300])
    print("Modified example:", X_modified.iloc[0][:300])

    rows, predictions = [], {}
    models = [
        ("TF-IDF + Logistic Regression", build_logistic_regression()),
        ("TF-IDF + Linear SVM", build_linear_svm())
    ]

    for name, model in models:
        print("\\nTraining", name)
        model.fit(X_train, y_train)
        for test_type, texts in [
            ("Original", X_test), ("Modified", X_modified)
        ]:
            pred = model.predict(texts)
            predictions[(name, test_type)] = pred
            rows.append({
                "Model": name, "Test Type": test_type,
                **calculate_metrics(y_test, pred)
            })
            print(f"\\n{name} / {test_type}")
            print(classification_report(
                y_test, pred, labels=[0, 1],
                target_names=["Legitimate", "Phishing"],
                zero_division=0
            ))

    try:
        print("\\nTraining sentence-embedding classifier...")
        encoder, classifier = train_embedding_classifier(X_train, y_train)
        name = "Sentence Embeddings + Logistic Regression"
        for test_type, texts in [
            ("Original", X_test), ("Modified", X_modified)
        ]:
            pred = predict_embeddings(encoder, classifier, texts)
            predictions[(name, test_type)] = pred
            rows.append({
                "Model": name, "Test Type": test_type,
                **calculate_metrics(y_test, pred)
            })
    except Exception as exc:
        print("Embedding model skipped:", str(exc)[:700])

    results = pd.DataFrame(rows)
    comparison = calculate_performance_drop(results)
    results.to_csv(results_dir / "results.csv", index=False)
    comparison.to_csv(results_dir / "comparison_table.csv", index=False)

    prediction_table = pd.DataFrame({
        "Original Message": X_test,
        "Modified Message": X_modified,
        "Actual Label": y_test
    })
    for (name, test_type), pred in predictions.items():
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        prediction_table[f"{safe_name}_{test_type}_Prediction"] = pred
    prediction_table.to_csv(results_dir / "predictions.csv", index=False)

    for metric in [
        "Accuracy", "Phishing Precision", "Phishing Recall", "Macro-F1", "MCC"
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results, x="Model", y=metric, hue="Test Type")
        plt.title(f"{metric}: Original vs Modified Messages")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric.lower().replace(' ', '_')}.png")
        plt.close()

    for (name, test_type), pred in predictions.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"]
        )
        plt.title(f"{name} - {test_type}")
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.tight_layout()
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        plt.savefig(figures_dir / f"confusion_{safe_name}_{test_type.lower()}.png")
        plt.close()

    print("\\nRESULTS\\n", results.round(4).to_string(index=False))
    print("\\nCOMPARISON TABLE\\n", comparison.round(4).to_string(index=False))
    print("\\nSaved outputs to results/ and figures/.")
    return results, comparison

if __name__ == "__main__":
    run_experiment()
