#!/usr/bin/env python3

def run(h5ad, genes, min_n_counts_adjusted:float=5., figdir:str="", n_pcs:int=0, use_leiden:bool=True, ica:bool=False):
    import os
    import pandas as pd
    import anndata
    import scanpy as sc
    sc.settings.figdir = figdir
    sc.settings.verbosity = 2
    sc.set_figure_params(dpi_save=600)
    adata = anndata.read_h5ad(h5ad, backed="r")
    adata = adata[adata.obs["n_counts_adjusted"] >= min_n_counts_adjusted, :].to_memory()
    if n_pcs <= 0:
        n_pcs = adata.obsm["X_pca"].shape[1]
    if ica:
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=n_pcs)
        X_ica = ica.fit_transform(adata.obsm["X_pca"])
        #ICs = adata.varm["PCs"] @ ica.components_.T ### components_ is (n_components, n_features)
        flip = np.abs(X_ica.min(0)) >= X_ica.max(0)
        X_ica[:, flip] *= -1.
        ica.components_[flip, :] *= -1.      
        adata.obsm["X_ica"] = X_ica
        if "PCs" in adata.varm.keys():
            adata.varm["ICs"] = adata.varm["PCs"] @ ica.components_.T
        for i in range(n_pcs):
            adata.obs["IC%d" % (i + 1)] = adata.obsm["X_ica"][:, i]
            sc.pl.spatial(adata, color="IC_%d" % (i + 1), save="_IC%d.png" % (i + 1), color_map="Greens", vmin=0)
        sc.pp.neighbors(adata, use_rep="X_ica")
    else:
        sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=0.3)
    for g in genes:
        sc.pl.umap(adata, color=g, save=f"_{g}.png", color_map="Reds", frameon=False)
    obs = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["U0", "U1"])
    obs["spatial_0"] = adata.obsm["spatial"][:, 0]
    obs["spatial_1"] = adata.obsm["spatial"][:, 1]
    if use_leiden:
        sc.tl.leiden(adata)
        sc.pl.umap(adata, color="leiden", save="_leiden_ondata.png", legend_loc="on data", legend_fontsize=4)
        sc.pl.umap(adata, color="leiden", save="_leiden_beside.png")
        for library_id in adata.uns["spatial"].keys():
            sc.pl.spatial(adata, color="leiden", save=f"_{library_id}_leiden.png", library_id=library_id)
        obs['leiden'] = adata.obs["leiden"]
    obs.to_csv(os.path.join(sc.settings.figdir, "umap.tsv.gz"), sep="\t")
    return h5ad

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="h5ad")
    ap.add_argument("-f", "--figdir", default="")
    ap.add_argument("--min-n-counts-adjusted", type=float, default=25.)
    ap.add_argument("-g", "--genes", nargs="+")
    ap.add_argument("-p", "--n-pcs", type=int, default=0)
    ap.add_argument("--use-leiden", dest="use_leiden", action="store_true")
    ap.add_argument("--no-leiden", dest="use_leiden", action="store_false")
    ap.add_argument("--ica", dest="ica", action="store_true")
    ap.add_argument("--no-ica", dest="ica", action="store_false")
    ap.set_defaults(use_leiden=True, ica=False)
    args = vars(ap.parse_args())
    if args["figdir"] == "":
        args["figdir"] = args["h5ad"].replace(".h5ad", "")
    run(h5ad=args["h5ad"], figdir=args["figdir"], genes=args["genes"], min_n_counts_adjusted=args["min_n_counts_adjusted"],
        n_pcs=args["n_pcs"], use_leiden=args["use_leiden"], ica=args["ica"])

