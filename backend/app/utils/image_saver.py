from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.core.config import settings


def save_heatmap(
    heatmap: np.ndarray,
    method: str,
    failure_id: str,
):
    """
    Save heatmap image to storage folder.
    """

    save_dir = (
        settings.outputs_dir
        / "heatmaps"
        / method
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_dir / f"{failure_id}.png"

    plt.figure(figsize=(4, 4))

    plt.imshow(heatmap, cmap="jet")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    print(f"✅ Saved {method}: {save_path}")