import sys
from pathlib import Path

# ============================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import pandas as pd
import time

from models.cnn_model import SimpleCNN

from xai.gradcam import GradCAM
from xai.shap_explainer import ShapExplainer
from xai.lime_explainer import LimeExplainer

import xai.combine as cmb

from evaluation.fidelity import fidelity
from evaluation.stability import stability
from evaluation.clustering import clustering
from evaluation.interpretability import interpretability
from evaluation.scoring import final_score

from visualization.plots import (
    show_failures_grid,
    show_best_vs_all,
    plot_pca,
    show_all_methods_full
)

from recommendation.adaptive_recommend import (
    generate_adaptive_recommendations
)

# ============================================
# DEVICE
# ============================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# RESULTS DIRECTORY
# ============================================
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================
# LOAD MODEL
# ============================================
model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load(
        ROOT_DIR / "models" / "mnist_model.pth"
    )
)

model.eval()

# ============================================
# LOAD FAILURES
# ============================================
failures = torch.load(
    ROOT_DIR / "data" / "failures.pt"
)

show_failures_grid(failures, n=8)

# ============================================
# XAI METHODS
# ============================================
gradcam = GradCAM(model)

background = torch.stack(
    [f[0] for f in failures[:50]]
).to(device)

if background.dim() == 5:
    background = background.squeeze(1)

if background.dim() == 3:
    background = background.unsqueeze(1)

shap_exp = ShapExplainer(
    model,
    background
)

# ============================================
# PREDICT FUNCTION FOR LIME
# ============================================
def predict_fn(x):

    x = torch.tensor(x).permute(
        0, 3, 1, 2
    ).float().to(device)

    # RGB -> GRAYSCALE
    x = x.mean(
        dim=1,
        keepdim=True
    )

    return model(x).detach().cpu().numpy()

lime_exp = LimeExplainer(predict_fn)

# ============================================
# STORAGE
# ============================================
methods = {
    "gradcam": [],
    "shap": [],
    "lime": [],
    "g+s": [],
    "g+l": [],
    "s+l": [],
    "all": []
}

metrics = {
    k: {
        "f": [],
        "s": [],
        "t": [],
        "i": []
    }
    for k in methods
}

all_explanations = {
    "gradcam": [],
    "shap": [],
    "lime": [],
    "g+s": [],
    "g+l": [],
    "s+l": [],
    "all": []
}

# ============================================
# MAIN LOOP
# ============================================
for img, lbl, pred in failures[:100]:

    # ========================================
    # PREPARE IMAGE
    # ========================================
    img = (
        img.to(device)
        .squeeze()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # ========================================
    # BASE EXPLANATIONS
    # ========================================
    g = gradcam.generate(img, pred)

    s = shap_exp.generate(img)

    img_np = img.cpu().numpy()[0]

    img_np = img_np.transpose(1, 2, 0)

    img_rgb = np.repeat(
        img_np,
        3,
        axis=2
    )

    l = lime_exp.generate(img_rgb)

    # ========================================
    # SHOW VISUALS ONCE
    # ========================================
    if len(methods["gradcam"]) == 0:

        show_all_methods_full(
            img.cpu(),
            g,
            s,
            l,
            cmb
        )

        show_best_vs_all(
            img.cpu(),
            g,
            s,
            l,
            cmb
        )

    # ========================================
    # COMBINED METHODS
    # ========================================
    outputs = {

        "gradcam": cmb.gradcam_only(g),

        "shap": cmb.shap_only(s),

        "lime": cmb.lime_only(l),

        "g+s": cmb.g_s(g, s),

        "g+l": cmb.g_l(g, l),

        "s+l": cmb.s_l(s, l),

        "all": cmb.all_three(g, s, l)
    }

    # ========================================
    # SAVE EXPLANATIONS
    # ========================================
    for method_name, exp in outputs.items():

        all_explanations[
            method_name
        ].append(exp)

    # ========================================
    # NOISY INPUT
    # ========================================
    noisy = img + (
        0.01 * torch.randn_like(img)
    )

    g_noisy = gradcam.generate(
        noisy,
        pred
    )

    s_noisy = shap_exp.generate(img)

    noisy_np = noisy.cpu().numpy()[0]

    noisy_np = noisy_np.transpose(
        1,
        2,
        0
    )

    noisy_rgb = np.repeat(
        noisy_np,
        3,
        axis=2
    )

    l_noisy = lime_exp.generate(
        noisy_rgb
    )

    noisy_outputs = {

        "gradcam": cmb.gradcam_only(
            g_noisy
        ),

        "shap": cmb.shap_only(
            s_noisy
        ),

        "lime": cmb.lime_only(
            l_noisy
        ),

        "g+s": cmb.g_s(
            g_noisy,
            s_noisy
        ),

        "g+l": cmb.g_l(
            g_noisy,
            l_noisy
        ),

        "s+l": cmb.s_l(
            s_noisy,
            l_noisy
        ),

        "all": cmb.all_three(
            g_noisy,
            s_noisy,
            l_noisy
        )
    }

    # ========================================
    # METRICS
    # ========================================
    for k, exp in outputs.items():

        start = time.time()

        f = fidelity(
            model,
            img,
            exp,
            pred
        )

        i = interpretability(exp)

        noisy_exp = noisy_outputs[k]

        s_score = stability(
            exp,
            noisy_exp
        )

        t = time.time() - start

        methods[k].append(
            exp.flatten()
        )

        metrics[k]["f"].append(f)

        metrics[k]["s"].append(
            s_score
        )

        metrics[k]["t"].append(t)

        metrics[k]["i"].append(i)

# ============================================
# FINAL RESULTS TABLE
# ============================================
rows = []

for k in methods:

    sil, db = clustering(
        methods[k]
    )

    f = np.mean(metrics[k]["f"])

    s_val = np.mean(metrics[k]["s"])

    t_val = np.mean(metrics[k]["t"])

    i_val = np.mean(metrics[k]["i"])

    score = final_score(
        f,
        s_val,
        sil,
        db,
        i_val
    )

    rows.append({

        "Method": k,

        "Fidelity": f,

        "Stability": s_val,

        "Silhouette": sil,

        "DB": db,

        "Interpretability": i_val,

        "Time": t_val,

        "Score": score
    })

# ============================================
# DATAFRAME
# ============================================
df = pd.DataFrame(rows).sort_values(
    "Score",
    ascending=False
)

# ============================================
# SAVE CSV
# ============================================
csv_path = RESULTS_DIR / "final_results.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\nCSV SAVED:")
print(csv_path)

print("\n=========== FINAL SCORES ===========\n")

print(
    df[
        [
            "Method",
            "Fidelity",
            "Stability",
            "Silhouette",
            "DB",
            "Interpretability",
            "Score"
        ]
    ]
)

print("\n====================================\n")

# ============================================
# SELECT BEST METHOD
# ============================================
best_method = df.iloc[0]["Method"]

print(
    "\nBest explanation method:",
    best_method
)

selected_explanations = (
    all_explanations[best_method]
)

# ============================================
# GENERATE RECOMMENDATIONS
# ============================================
recs, feature_summary = (
    generate_adaptive_recommendations(
        selected_explanations,
        best_method
    )
)

feature_df = pd.DataFrame(
    [feature_summary]
)

feature_df.to_csv(
    RESULTS_DIR /
    "recommendation_features.csv",
    index=False
)

# ============================================
# PRINT RECOMMENDATIONS
# ============================================
print(
    "\n===== XAIFA RECOMMENDATIONS ====="
)

for r in recs:
    print("-", r)

# ============================================
# SAVE RECOMMENDATIONS
# ============================================
with open(
    RESULTS_DIR /
    "recommendations.txt",
    "w"
) as f:

    f.write(
        "===== XAIFA RECOMMENDATIONS =====\n\n"
    )

    f.write(
        f"Best explanation method: "
        f"{best_method}\n\n"
    )

    for i, r in enumerate(recs, 1):

        f.write(f"{i}. {r}\n")

# ============================================
# PCA VISUALIZATION
# ============================================
plot_pca(
    methods[best_method],
    f"Best_{best_method}"
)

plot_pca(
    methods["all"],
    "Combined"
)

# ============================================
# FINAL LOGS
# ============================================
print("\n==============================")
print("FINAL RESULTS CSV GENERATED")
print("==============================")

print(df)

print("\nCSV LOCATION:")
print(csv_path)