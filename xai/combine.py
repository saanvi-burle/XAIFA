from xai.fusion import fuse_maps

# SINGLE
def gradcam_only(g): return fuse_maps([g])
def shap_only(s): return fuse_maps([s])
def lime_only(l): return fuse_maps([l])

# PAIRWISE
def g_s(g,s): return fuse_maps([g,s])
def g_l(g,l): return fuse_maps([g,l])
def s_l(s,l): return fuse_maps([s,l])

# TRIPLE
def all_three(g,s,l): return fuse_maps([g,s,l])