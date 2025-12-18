# ShapeUpLMD

## Description

ShapeUpLMD is a collection of python functions which allows the generation of laser-microdissection (LMD) shapes from larger areas of interest (AOI). For example - you have a tissue sample on a slide, and you want to collect a region you've either drawn by hand, received from a pathologist, or generated using some computer vision software program. The regions are most likely not LMD friendly:
- the regions are too large, small, or complex, causing tissue peeling and collection of undesired regions
- the regions are defined with too much precision, causing the scope to waste time calculating unproductive movements

ShapeUpLMD converts regions into LMD-friendly shapes and outputs XML files ready to load onto your LMD system. It supports two kinds of shape creation:
- targeted, spatially resolved ROIs
- unbiased, whole tissue ROIs

![Tumor and non-tumor cell populations are annotated in digital histopathology images by expert pathology review and inform annotation of tumor and adjacent non-tumor ROIs on adjacent polyethylene naphthalate (PEN) slide images that include fiducials enabling feature alignment. Annotated ROIs are used to train a tumor-specific classifier. Tumor ROIs are then merged with an annotation layer including fiducial coordinates to undergo shape optimization using ShapeUpLMD. Optimized tumor ROIs and fiducial coordinates were imported onto the laser microscope for automated sample collection.](images/Figure1A.png)

## Installation and Setup
Code has been tested with Python 3.10.18.

Install the requirements using pip:
```bash
python -m pip install -r requirements.txt
```
If running into errors installing `stringzilla` on Windows (required for `tiatoolbox`), either install the C++ build tools as described in the error, or try installing pre-compiled stringzilla using a package manager such as anaconda:
```bash
conda install -c conda-forge stringzilla
python -m pip install -r requirements.txt
```

The `.svs` files can be downloaded from `https://lmdomics.org/ShapeUpLMD/`. Please complete the download for one file before downloading the second.

## Project Layout

`ShapeUpLMD_demo.ipynb` is the main walkthrough
`halo_shape_functions.py` holds the helper functions
`data` contains your slide and annotation files (`.svs` and `.annotations`)
`XML_files` and `unbiased_XML_files` will appear automatically when you run the notebook

Example expected structure

```
ShapeUpLMD_demo.ipynb
halo_shape_functions.py
data/
    wsi_analysis_demo.svs
    wsi_analysis_demo.annotations
    wsi_unbiased_analysis_demo.annotations
    wsi_unbiased_analysis_demo.svs
XML_files
unbiased_XML_files
```

Set the notebook directory as your working directory so all paths are resolved automatically. The default parameters work fine for the demo but feel free to adjust the micrometer limits (`minimum_micrometers_between_shapes`, `minimum_micrometers_squared`, `maximum_micrometers_squared`) and slice counts (`n_slices_to_check`) in `make_shapes_lmdable_xml` to fit your tissue and workflow.

## Targeted, Spatially Resolved ROI Workflow

The targeted, spatially resolved ROI workflow returns XML exports for a custom number of LMD compatible clusters in the target ROI. If you’d like to modify the number of clusters and the shape area (in mm2), modify the `num_clusters` and `area_mm2` parameters. Note that if `num_clusters` and `area_mm2` are incompatible, an error will be thrown.

The first section of `ShapeUpLMD_demo.ipynb` goes through the following steps:  
1. Load tumor tissue and calibration layers from HALO annotation files and convert them into shapely MultiPolygons.
2. Display tumor and tissue masks so you can sanity check the inputs.
3. Create LMD compatible shapes with `make_shapes_lmdable_xml` which slices large polygons into smaller pieces that fit typical LMD limits.
4. Plot these shapes on the tissue so you can see how they will be cut.
5. Split tumor regions into spatial clusters using `spatial_segmentation_brute_force` controlled by your number of clusters and target area.
6. Overlay the spatial map on a WSI crop for quick visual validation.
7. Export XML files for each spatial region with `create_scope_xml2`.

## Unbiased Whole Tissue Workflow

The unbiased whole tissue workflow returns XML exports for a user-defined number of LMD compatible shapes across the whole tissue. If you'd like to modify the number of shapes, modify the `cluster_count` parameter in `rebalance_multipolygons_by_area`.

The second section of `ShapeUpLMD_demo.ipynb` goes through the following steps:
1. Load a full slide and parse tissue calibration layers.
2. Generate LMD compatible shapes for the entire tissue area.
3. Rebalance everything into a chosen number of uniform area clusters.
4. Plot the complete spatial map and export all clusters as XML in organized chunks using `partition_collections_to_xml`.

## Cite ShapeUpLMD
