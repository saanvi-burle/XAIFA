def interpretability(exp):
    sparsity = (exp > 0.5).sum() / exp.size
    return 1 - sparsity