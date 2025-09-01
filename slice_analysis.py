import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

def generate_error_analysis_report(
    df, y_true, y_prob, y_pred, subgroup_features, output_path="error_analysis.md"
):
    """
    Generate an error analysis report:
    - Slice metrics across subgroups
    - Identify likely error causes
    - Suggest improvements
    - Save report as a Markdown file
    
    Parameters
    ----------
    df : pd.DataFrame
        Validation dataframe (must contain subgroup features).
    y_true : array-like
        True labels (binary).
    y_prob : array-like
        Predicted probabilities.
    y_pred : array-like
        Predicted classes (after threshold).
    subgroup_features : list
        List of categorical features to slice by.
    output_path : str
        Filepath to save the Markdown report.
    
    Returns
    -------
    report : str
        Markdown-style error analysis report.
    slice_df : pd.DataFrame
        Slice performance metrics table.
    """
    
    # --- Slice analysis ---
    slice_metrics = []
    for feature in subgroup_features:
        for val in df[feature].unique():
            idx = df[feature] == val
            if idx.sum() < 50:  # skip very small subgroups
                continue
            try:
                roc = roc_auc_score(y_true[idx], y_prob[idx])
            except ValueError:
                roc = np.nan
            f1 = f1_score(y_true[idx], y_pred[idx])
            slice_metrics.append({
                "feature": feature,
                "value": val,
                "n_samples": idx.sum(),
                "roc_auc": roc,
                "f1_score": f1
            })
    slice_df = pd.DataFrame(slice_metrics)
    
    # --- Identify error causes ---
    likely_causes = []
    
    # Subgroup performance gaps
    if not slice_df.empty:
        gap = slice_df.groupby("feature")["roc_auc"].max() - slice_df.groupby("feature")["roc_auc"].min()
        for feat, g in gap.items():
            if g > 0.05:  # >5% performance gap
                likely_causes.append(f"Performance varies across `{feat}` (gap in ROC-AUC ~{g:.2f}), suggesting subgroup imbalance.")
    
    # Label uncertainty heuristic
    if (y_prob[(y_prob > 0.4) & (y_prob < 0.6)].size / len(y_prob)) > 0.2:
        likely_causes.append("Many borderline predictions (~0.5 probability) → possible label uncertainty or noisy ground truth.")
    
    # Class imbalance
    pos_rate = y_true.mean()
    if pos_rate < 0.1:
        likely_causes.append(f"Low prevalence of positive class ({pos_rate:.1%}) → class imbalance likely contributes to errors.")
    
    # --- Suggested improvements ---
    improvements = [
        "Apply subgroup-aware resampling or class weights in XGBoost.",
        "Engineer temporal or interaction features to capture more predictive patterns.",
        "Tune decision thresholds per subgroup to balance sensitivity and specificity.",
        "Consider probabilistic labeling if diagnosis/event dates are uncertain."
    ]
    
    # --- Assemble Markdown report ---
    report = "# 🔎 Training-Related Error Analysis\n\n"
    report += "## Slice Analysis (Performance by Subgroup)\n\n"
    if not slice_df.empty:
        report += slice_df.to_markdown(index=False) + "\n\n"
    else:
        report += "_No subgroup metrics available_\n\n"
    
    report += "## Likely Causes of Errors\n"
    if likely_causes:
        for cause in likely_causes:
            report += f"- {cause}\n"
    else:
        report += "- No strong error causes detected from available slices.\n"
    
    report += "\n## Proposed Improvements\n"
    for imp in improvements:
        report += f"- {imp}\n"
    
    # --- Save report ---
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Error analysis report saved to {output_path}")
    
    return report, slice_df


# Load validation dataset
X_val = pd.read_csv("X_val.csv.gz")
y_val = pd.read_csv("y_val.csv.gz").astype(int)

# Load preprocessor and trained model
preprocessor = joblib.load("artifacts/preprocessor.joblib")
model = joblib.load("artifacts/model.joblib")
threshold = 0.25  # Example threshold from training

# Preprocess validation features
X_val_trans = preprocessor.transform(X_val)
y_prob = model.predict_proba(X_val_trans)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

roc_auc = roc_auc_score(y_val, y_prob)
prec, rec, _ = precision_recall_curve(y_val, y_prob)
pr_auc = np.trapz(rec, prec)
f1 = f1_score(y_val, y_pred)


categorical_features = ["sex", "race", "has_diabetes"]
slice_metrics = []

for feature in categorical_features:
    for val in X_val[feature].unique():
        idx = X_val[feature] == val
        y_true = y_val[idx]
        y_p = y_prob[idx]
        y_pr = y_pred[idx]
        slice_metrics.append({
            "feature": feature,
            "value": val,
            "n_samples": idx.sum(),
            "roc_auc": roc_auc_score(y_true, y_p),
            "f1_score": f1_score(y_true, y_pr)
        })

slice_df = pd.DataFrame(slice_metrics)
slice_df

report, slice_table = generate_error_analysis_report(
    df=X_val,
    y_true=y_val,
    y_prob=y_prob,
    y_pred=y_pred,
    subgroup_features=["sex", "race", "has_diabetes"],
    output_path="error_analysis.md"
)
