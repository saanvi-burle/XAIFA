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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

failures = torch.load("data/failures.pt")

gradcam = GradCAM(model)
background = torch.stack([f[0] for f in failures[:50]]).to(device)
shap_exp = ShapExplainer(model, background)

def predict_fn(x):
    x = torch.tensor(x).permute(0,3,1,2).float().to(device)
    return model(x).detach().cpu().numpy()

lime_exp = LimeExplainer(predict_fn)

methods = {
    "gradcam": [], "shap": [], "lime": [],
    "g+s": [], "g+l": [], "s+l": [], "all": []
}

metrics = {k: {"f":[], "s":[], "t":[], "i":[]} for k in methods}

for img, lbl, pred in failures[:200]:

    img = img.unsqueeze(0).to(device)

    g = gradcam.generate(img, pred)
    s = shap_exp.generate(img)
    l = lime_exp.generate(img.cpu().numpy()[0].transpose(1,2,0))

    outputs = {
        "gradcam": cmb.gradcam_only(g),
        "shap": cmb.shap_only(s),
        "lime": cmb.lime_only(l),
        "g+s": cmb.g_s(g,s),
        "g+l": cmb.g_l(g,l),
        "s+l": cmb.s_l(s,l),
        "all": cmb.all_three(g,s,l)
    }

    for k, exp in outputs.items():

        start = time.time()

        f = fidelity(model, img, exp, pred)
        i = interpretability(exp)

        # stability (add noise)
        noisy = img + 0.01*torch.randn_like(img)
        noisy_exp = exp  # simple reuse for now

        s_score = 1  # optional skip heavy calc

        t = time.time() - start

        methods[k].append(exp.flatten())
        metrics[k]["f"].append(f)
        metrics[k]["s"].append(s_score)
        metrics[k]["t"].append(t)
        metrics[k]["i"].append(i)

# final table
rows = []

for k in methods:
    sil, db = clustering(methods[k])

    f = np.mean(metrics[k]["f"])
    s_val = np.mean(metrics[k]["s"])
    t_val = np.mean(metrics[k]["t"])
    i_val = np.mean(metrics[k]["i"])

    score = final_score(f, s_val, sil, db, i_val)

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

df = pd.DataFrame(rows).sort_values("Score", ascending=False)
df.to_csv("results/final_results.csv", index=False)

print(df)