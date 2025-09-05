#!/usr/bin/env python3
from typing import List, Union

def summarize_probes(adata, filter_probes:bool=False):
    import numpy as np
    import pandas as pd
    import anndata
    import scanpy as sc
    from scipy.sparse import csr_matrix
    adata.var["spliced"] = adata.var["probe_region"] == "spliced"
    sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars=["spliced", "filtered_probes"],
                               percent_top=None, log1p=False)
    ### summarize 
    ug = pd.Index(pd.unique(adata.var["gene_ids"]))
    col_nuniq = adata.var.groupby("gene_ids").agg(**{k: (k, "nunique") for k in adata.var.columns}).max(0)
    gf = adata.var.loc[:, col_nuniq[col_nuniq==1].index.values].drop_duplicates("gene_ids")
    gf.index = gf["gene_ids"].values
    gf = gf.loc[ug.values, :]
    gf.index = gf["gene_name"].values
    ### filtering
    if filter_probes:
        adata = adata[:, adata.var["filtered_probes"]].copy()
        ### Recalculate % spliced
        sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars=["spliced"],
                                   percent_top=None, log1p=False)
    vs = adata.var.loc[adata.var["spliced"], :].copy()
    SS = csr_matrix((np.ones(vs.shape[0]),
                    (adata.var_names.get_indexer(vs.index.values),
                     ug.get_indexer(vs["gene_ids"].values))),
                   dtype=int, shape=(adata.shape[1], len(ug)))
    vu = adata.var.loc[~adata.var["spliced"], :].copy()
    SU = csr_matrix((np.ones(vu.shape[0]),
                    (adata.var_names.get_indexer(vu.index.values),
                     ug.get_indexer(vu["gene_ids"].values))),
                   dtype=int, shape=(adata.shape[1], len(ug)))
    adata = anndata.AnnData(X=adata.X.astype(int) @ (SS + SU), obs=adata.obs,
                            obsp=adata.obsp, obsm=adata.obsm,
                            layers={"spliced": adata.X.astype(int) @ SS, "unspliced": adata.X.astype(int) @ SU},
                            uns=adata.uns, var=gf)
    adata.var_names_make_unique()
    return adata

def process_h5ad(adata, output:str="output.h5ad", euclidean:bool=True,
                 radius:int=4, filter_probes:bool=True):
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import squidpy as sq
    from scipy.sparse import diags
    if "probe_region" in adata.var.columns:
        adata = summarize_probes(adata, filter_probes=filter_probes)
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL", "Rps", "Rpl"))
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars=["ribo", "mt"], percent_top=None)
    library_key = "Sample"
    if "slice" in adata.obs.columns:
        adata.obs["slice"] = pd.Categorical(adata.obs["slice"].values)
        print(adata.obs["slice"].value_counts())
        library_key = "slice"
    ### Find neighbors
    sample = adata.obs["Sample"].values[0]
    if euclidean:
        bin_radius = radius * adata.uns["spatial"][sample]["scalefactors"]["spot_diameter_fullres"]
        sq.gr.spatial_neighbors(adata, library_key=library_key, coord_type="generic", radius=bin_radius)
    else:
        sq.gr.spatial_neighbors(adata, library_key=library_key, coord_type="grid", n_rings=radius, n_neighs=4)
    dmax = adata.X.data.max()
    if dmax < 32768:
        adata.X = adata.X.astype(np.int16)
    else:
        adata.X = adata.X.astype(np.int32)
    adata.write_h5ad(output, compression="gzip")

def run_bin2cell(adata, mpp:float=0.5, sample:str="mysample"):
    import os
    import bin2cell as b2c
    X = adata.X.copy()
    b2c.destripe(adata, adjust_counts=True)
    ### STARDIST
    os.makedirs("stardist", exist_ok=True)
    b2c.scaled_he_image(adata, mpp=mpp, save_path=f"stardist/{sample}.tiff")
    b2c.stardist(image_path=f"stardist/{sample}.tiff",
                 labels_npz_path=f"stardist/{sample}.npz",
                 stardist_model="2D_versatile_he",
                 prob_thresh=0.01)
    b2c.insert_labels(adata, labels_npz_path=f"stardist/{sample}.npz",
                      basis="spatial",
                      mpp=mpp,
                      labels_key="labels_he")
    b2c.expand_labels(adata, labels_key="labels_he", expanded_labels_key="labels_he_expanded")
    adata.X = X
    if "spatial_cropped" in adata.obsm.keys():
        del adata.obsm["spatial_cropped"]
    return adata


def build_h5ad(in_h5, source_image_path:str="myimage.tiff",
               output:str="output.h5ad",
               sample:str="MySample",
               radius:int=4, mpp:float=0.5,
               slices:str="MySlices.tsv.gz",
               bad_bins_csv:Union[List[str],None]=None,
               use_bin2cell:bool=True,
               filter_probes:bool=True):
    import os, gc
    import numpy as np
    import pandas as pd
    import bin2cell as b2c
    binned_outputs = os.path.dirname(in_h5)
    count_file = os.path.basename(in_h5)
    sr_image_path = os.path.join(binned_outputs, "spatial")
    adata = b2c.read_visium(binned_outputs, count_file=count_file, library_id=sample,
                            source_image_path=source_image_path, spaceranger_image_path=sr_image_path)
    adata.obs["Sample"] = sample
    adata.obs_names = [f"{sample}#{bc}" for bc in adata.obs_names]
    adata.var_names_make_unique()
    adata.X = adata.X.astype(np.int32)
    adata.obs["n_counts"] = np.round(np.ravel(adata.X.sum(1))).astype(int)
    if use_bin2cell:
        adata = run_bin2cell(adata, mpp=mpp, sample=sample)
    else:
        b2c.destripe(adata, adjust_counts=False)
    ### Split by slice
    if bad_bins_csv is None:
        bad_bins_csv = []
    for bad_bin_csv in list(bad_bins_csv):
        cf = pd.read_csv(bad_bin_csv, index_col=0, sep="\t")
        print(bad_bin_csv, " % of bad bins: ", 100 * adata.obs_names.isin(cf.index.values).mean())
        adata = adata[~adata.obs_names.isin(cf.index.values), :].copy()
        gc.collect()
    if os.path.exists(slices):
        sf = pd.read_csv(slices, sep="\t", index_col=0)
        adata = adata[adata.obs_names.isin(sf.index.values), :].copy()
        adata.obs["slice"] = sf["slice"]
        print(adata.obs.groupby("slice", observed=True).agg(total_counts=("n_counts", "sum")))
    gc.collect()
    process_h5ad(adata, radius=radius, output=output)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="in_h5")
    ap.add_argument("--image", required=True, dest="source_image_path")
    ap.add_argument("-o", "--output", required=True, dest="output")
    ap.add_argument("-s", "--sample-name", required=True, dest="sample")
    ap.add_argument("-r", "--radius", type=int, default=4)
    ap.add_argument("--mpp", type=float, default=0.5)
    ap.add_argument("--slices", default="")
    ap.add_argument("--filter-probes", action="store_true", dest="filter_probes")
    ap.add_argument("--keep-all-probes", action="store_false", dest="filter_probes")
    ap.add_argument("--bin2cell", action="store_true", dest="use_bin2cell")
    ap.add_argument("--no-bin2cell", action="store_false", dest="use_bin2cell")
    ap.add_argument("--remove-bins-by-barcodes", nargs="+")
    ap.set_defaults(filter_probes=True, use_bin2cell=True)
    args = vars(ap.parse_args())
    build_h5ad(in_h5=args["in_h5"], output=args["output"],
               source_image_path=args["source_image_path"], mpp=args["mpp"],
               sample=args["sample"], radius=args["radius"],
               use_bin2cell=args["use_bin2cell"],
               bad_bins_csv=args["remove_bins_by_barcodes"],
               slices=args["slices"])
