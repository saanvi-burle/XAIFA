from sklearn.metrics.pairwise import cosine_similarity

def stability(exp1, exp2):
    return cosine_similarity(
        exp1.flatten().reshape(1,-1),
        exp2.flatten().reshape(1,-1)
    )[0][0]