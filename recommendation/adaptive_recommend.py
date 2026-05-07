# ============================================
# METHOD-AWARE ADAPTIVE XAIFA RECOMMENDATION
# ============================================

import numpy as np
import cv2


# ============================================
# NORMALIZE
# ============================================

def normalize(exp):

    exp = exp.astype(np.float32)

    if exp.max() != exp.min():
        exp = (exp - exp.min()) / (exp.max() - exp.min())

    return exp


# ============================================
# FEATURE EXTRACTION
# ============================================

def entropy(exp):

    exp = exp / (exp.sum() + 1e-8)

    return -np.sum(exp * np.log(exp + 1e-8))


def concentration(exp):

    exp = exp / (exp.sum() + 1e-8)

    return np.max(exp)


def sparsity(exp):

    thresh = 0.5 * np.max(exp)

    return np.sum(exp > thresh) / exp.size


def edge_attention(exp):

    h, w = exp.shape

    border = max(2, int(min(h, w) * 0.1))

    mask = np.zeros_like(exp)

    mask[:border, :] = 1
    mask[-border:, :] = 1
    mask[:, :border] = 1
    mask[:, -border:] = 1

    return np.sum(exp * mask) / exp.size


def fragmentation(exp):

    binary = (exp > 0.5).astype(np.uint8)

    num_labels, _ = cv2.connectedComponents(binary)

    return num_labels


def sharpness(exp):

    lap = cv2.Laplacian(exp.astype(np.float32), cv2.CV_32F)

    return lap.var()


# ============================================
# MAIN RECOMMENDATION ENGINE
# ============================================

def generate_adaptive_recommendations(
    explanations,
    best_method
):

    explanations = [normalize(e) for e in explanations]

    ent = np.mean([entropy(e) for e in explanations])
    conc = np.mean([concentration(e) for e in explanations])
    sparse = np.mean([sparsity(e) for e in explanations])
    edge = np.mean([edge_attention(e) for e in explanations])
    frag = np.mean([fragmentation(e) for e in explanations])
    sharp = np.mean([sharpness(e) for e in explanations])

    recs = []

    # ========================================
    # GRADCAM
    # ========================================

    if best_method == "gradcam":

        if edge > 0.15:
            recs.append(
                "GradCAM localization frequently activates border regions.\n"
                "Recommendation: apply centering normalization and cropping augmentation.\n"
            )

        if conc < 0.03:
            recs.append(
                "GradCAM attention is weakly concentrated.\n"
                "Recommendation: improve discriminative localization using harder samples.\n"
            )

        if sharp < 0.01:
            recs.append(
                "GradCAM maps are blurry and spatially weak.\n"
                "Recommendation: strengthen feature extraction using deeper convolutional layers.\n"
            )

    # ========================================
    # SHAP
    # ========================================

    elif best_method == "shap":

        if sparse < 0.03:
            recs.append(
                "SHAP attributions are highly sparse and unstable.\n"
                "Recommendation: improve robustness using noise augmentation and dropout.\n"
            )

        if ent > 5:
            recs.append(
                "SHAP explanations exhibit high attribution entropy.\n"
                "Recommendation: improve feature separability using regularization.\n"
            )

        if frag > 12:
            recs.append(
                "SHAP explanations are fragmented across disconnected regions.\n"
                "Recommendation: improve spatial continuity through smoother feature learning.\n"
            )

    # ========================================
    # LIME
    # ========================================

    elif best_method == "lime":

        if frag > 10:
            recs.append(
                "LIME explanations show unstable segmented regions.\n"
                "Recommendation: improve perturbation robustness using stronger augmentations.\n"
            )

        if ent > 5:
            recs.append(
                "LIME explanations are spatially inconsistent.\n"
                "Recommendation: increase local feature consistency during training.\n"
            )

        if edge > 0.12:
            recs.append(
                "LIME frequently emphasizes background regions.\n"
                "Recommendation: improve foreground-background separation.\n"
            )

    # ========================================
    # G+S
    # ========================================

    elif best_method == "g+s":

        if conc < 0.04:
            recs.append(
                "Hybrid GradCAM+SHAP explanations show reduced concentration.\n"
                "Recommendation: improve feature discrimination using curriculum learning.\n"
            )

        if ent > 4.5:
            recs.append(
                "Hybrid explanations remain moderately diffused.\n"
                "Recommendation: apply attention-guided regularization.\n"
            )

        if sharp < 0.02:
            recs.append(
                "Hybrid explanations have weak structural sharpness.\n"
                "Recommendation: enhance edge-aware preprocessing.\n"
            )

    # ========================================
    # G+L
    # ========================================

    elif best_method == "g+l":

        if frag > 10:
            recs.append(
                "GradCAM+LIME explanations are fragmented.\n"
                "Recommendation: improve regional consistency using smoother augmentations.\n"
            )

        if edge > 0.12:
            recs.append(
                "Hybrid attention over-focuses on peripheral regions.\n"
                "Recommendation: apply spatial normalization.\n"
            )

    # ========================================
    # S+L
    # ========================================

    elif best_method == "s+l":

        if sparse < 0.04:
            recs.append(
                "SHAP+LIME explanations are unstable and sparse.\n"
                "Recommendation: improve robustness using adversarial augmentation.\n"
            )

        if frag > 10:
            recs.append(
                "SHAP+LIME explanations show excessive segmentation fragmentation.\n"
                "Recommendation: improve local continuity in feature learning.\n"
            )

    # ========================================
    # ALL
    # ========================================

    elif best_method == "all":

        if ent > 5:
            recs.append(
                "Fully fused explanations exhibit excessive diffusion.\n"
                "Recommendation: reduce redundancy through weighted fusion optimization.\n"
            )

        if sparse < 0.03:
            recs.append(
                "Combined explanations contain unstable attribution regions.\n"
                "Recommendation: improve ensemble balancing between explanation methods.\n"
            )

        if frag > 12:
            recs.append(
                "Combined explanations are spatially fragmented.\n"
                "Recommendation: apply adaptive explanation weighting.\n"
            )

    # ========================================
    # FALLBACK
    # ========================================

    if len(recs) == 0:

        recs.append(
            f"{best_method} explanations are stable and well-localized.\n"
            "Minor optimization and augmentation tuning are recommended.\n"
        )

    return recs