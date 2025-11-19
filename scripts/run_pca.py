#!/usr/bin/env python3
def run(h5ad_list, output, quantile, min_total_counts:float=0., n_components:int=50, lamb:float=0.1,
        normalization:str="log1p", weight:str="n_counts_adjusted",
        mu:float=2, rho:float=1.5, batch_size:int=50000,
        sk_adjust:bool=True,
        hvg:int=0, batch_key:str="slices",
        use_laplacian_of_gaussian:bool=True, use_residuals:bool=False,
        keep_pc_loadings:bool=True,
        use_lag:bool=True, kernel_use_diag:bool=True):
    import numpy as np
    from scipy.sparse import diags
    import anndata
    import scanpy as sc
    from banksy_hd import FitPCA
    out = []
    stbl = {}
    for h5ad in h5ad_list:
        adata = anndata.read_h5ad(h5ad, backed="r")
        del adata.layers
        adata = adata[adata.obs["n_counts"] >= min_total_counts, :].to_memory()
        stbl.update(adata.uns["spatial"])
        out.append(adata)
    adata = anndata.concat(out, merge="same", uns_merge="same", pairwise=True)
    adata.uns["spatial"] = stbl
    if normalization == "log1p":
        if "n_counts_sk_adjusted" in adata.obs.columns:
            print("Using S/K adjustment")
            size_factor = diags(adata.obs["n_counts_sk_adjusted"] / adata.obs["n_counts"])
        elif "n_counts_adjusted" in adata.obs.columns:
            print("Using bin2cell adjustment")
            size_factor = diags(adata.obs["n_counts_adjusted"] / adata.obs["n_counts"])
        else:
            size_factor = diags(np.ones_like(adata.obs["n_counts"]))
    else:
        size_factor = diags(10000./adata.obs["n_counts"].values)
    if normalization == "tf-idf":
        adata.var["idf"] = np.log1p(adata.shape[0]/np.ravel(adata.X.sum(0) + 1e-300))
    adata.X = size_factor @ adata.X
    sc.pp.log1p(adata)
    if normalization == "tf-idf":
        adata.X = adata.X @ diags(adata.var["idf"].values)
    adata.X = adata.X.astype(np.float32)
    if hvg > 0:
        if batch_key in adata.obs.columns:
            sc.pp.highly_variable_genes(adata, batch_key=batch_key, n_top_genes=hvg)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        mask_var="highly_variable"
    else:
        mask_var=""
    FitPCA(adata, quantiles=np.asarray(quantile),
           weight=weight, mask_var=mask_var,
           n_components=n_components, lamb=lamb, mu=mu, rho=rho,
           batch_size=batch_size, use_lag=use_lag, use_laplacian_of_gaussian=use_laplacian_of_gaussian,
           keep_pc_loadings=keep_pc_loadings,
           use_residuals=use_residuals, kernel_use_diag=kernel_use_diag)
    adata.write_h5ad(output, compression="gzip")
    
### lambda: weighted vs nonweighted
### mu: neighbor vs LoG
### rho: residual vs rest 
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="h5ad_list", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-q", "--quantile", nargs="+", type=float)
    ap.add_argument("-l", "--lambda", dest="lamb", type=float, default=0.2)
    ap.add_argument("--no-laplacian-of-gaussian", dest="use_laplacian_of_gaussian", action="store_false")
    ap.add_argument("--no-lag", dest="use_lag", action="store_false")
    ap.add_argument("--no-residual", dest="use_residuals", action="store_false")
    ap.add_argument("--use-residual", dest="use_residuals", action="store_true")
    ap.add_argument("-m", "--mu", type=float, default=1.5, help="Decimal odds between lagged and difference of lags feature matrices")
    ap.add_argument("-r", "--rho", type=float, default=5., help="Decimal odds between lagged residuals and the lagged + difference of lags matrices")
    ap.add_argument("-b", "--batch-size", type=int, default=50000)
    ap.add_argument("--hvg", type=int, default=0)
    ap.add_argument("--batch-key", type=str, default="slices")
    ap.add_argument("-p", "--n-pcs", type=int, default=100)
    ap.add_argument("--min-total-counts", type=float, default=0.)
    ap.add_argument("--kernel-use-diag", dest="kernel_use_diag", action="store_true")
    ap.add_argument("--kernel-no-use-diag", dest="kernel_use_diag", action="store_false")
    ap.add_argument("--keep-extra-pc-loadings", dest="keep_pc_loadings", action="store_true")
    ap.add_argument("--no-keep-extra-loadings", dest="keep_pc_loadings", action="store_false")    
    ap.add_argument("-w", "--weight", default="")
    ap.add_argument("-n", "--normalization", default="log1p", choices=["log1p", "logCP10k", "tf-idf"])
    ap.set_defaults(use_laplacian_of_gaussian=True, use_lag=True, use_residuals=False, kernel_use_diag=False, keep_pc_loadings=True)
    args = vars(ap.parse_args())
    if args["quantile"] is None:
        print("Quantile(s) not specified, just using median")
        args["quantile"] = [0.5]
    print(args)
    run(h5ad_list=args["h5ad_list"],
        output=args["output"],
        normalization=args["normalization"],
        quantile=args["quantile"],
        min_total_counts=args["min_total_counts"],
        n_components=args["n_pcs"],
        lamb=args["lamb"],
        mu=args["mu"],
        weight=args["weight"],
        batch_size=args["batch_size"],
        use_laplacian_of_gaussian=args["use_laplacian_of_gaussian"],
        use_lag=args["use_lag"],
        use_residuals=args["use_residuals"],
        hvg=args["hvg"],
        batch_key=args["batch_key"],
        kernel_use_diag=args["kernel_use_diag"])
        
