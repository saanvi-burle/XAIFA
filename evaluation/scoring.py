def final_score(f, s, sil, db, interp):
    return (
        0.3*f +
        0.2*s +
        0.3*sil -
        0.1*db +
        0.1*interp
    )