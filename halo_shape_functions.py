__version__ = "1.0.0"

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

from PIL import Image
from shapely.geometry import MultiPolygon, Point
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pathlib import Path

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.affinity import translate
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances
import random


import pandas as pd
import math

import json
from collections import defaultdict

import cv2

import matplotlib as mpl
from matplotlib.collections import PatchCollection
from shapely.geometry import LinearRing

import shapely
from shapely import MultiPolygon, Polygon, make_valid
from shapely.ops import unary_union, nearest_points

from tiatoolbox.wsicore.wsireader import WSIReader

from tqdm import tqdm
import itertools


def convert_to_polygon_list(multipolygon, verbose: bool = False):
    if type(multipolygon) == shapely.MultiPolygon: ret = list(multipolygon.geoms)
    elif type(multipolygon) == shapely.Polygon:
        if multipolygon.is_valid:
            ret = [multipolygon]
        else:
            valid_shape = shapely.make_valid(multipolygon)

            ret = convert_to_polygon_list(valid_shape)
    elif type(multipolygon) == shapely.GeometryCollection: 
        ret = []
        for polygon in multipolygon.geoms:
            ret.extend(convert_to_polygon_list(shapely.make_valid(polygon), verbose=verbose))
    else:
        if verbose:
            print(f'Object is not a Polygon, MultiPolygon, or GeometryCollection! Returning empty list...\n\t`{type(multipolygon)=}`')
        ret = []
    return ret

def list_of_shapes_to_polygons(list_of_shapes, minimum_shape_area=0):
    ret_shapes = []
    for polygon in list_of_shapes:
        ret_shapes.extend(convert_to_polygon_list(polygon))
    # remove small shapes
    for polygon in ret_shapes.copy():
        if polygon.area < minimum_shape_area:
            ret_shapes.remove(polygon)
    return ret_shapes


def convert_HALO_regions_to_multipolygon(regions):
    """Converts HALO regions dictionary to a multipolygon.
    Requires that there is 'isExclusionRegion' in the result.
    Requires that the result is GEO_JSON.

    NOTE: the last make_valid in this function seems to make
    any polygons on top of larger polygons holes in the larger
    polygon. this isn't a problem for most of our use cases
    """
    cutouts = []
    polys = []
    for region in regions:
        pre_polygon = np.array(json.loads(region['geometry'])['coordinates'])
        # get shape type
        shape_type = region['shapeType']
        # shapes that I'm going to take care of 
        # POLYGON
        # RECTANGLE
        # ELLIPSE
        # throw an error if it's not any of these shapes... probably
        ##### commenting out rectangle, as it seems they have changed how rectangles are handled ..?
        #if shape_type == 'RECTANGLE':
        #    # first point is bottom left, second it top right
        #    pre_polygon = np.vstack([
        #        pre_polygon[0],
        #        [pre_polygon[1,0], pre_polygon[0,1]],
        #        pre_polygon[1],
        #        [pre_polygon[0,0], pre_polygon[1,1]]
        #    ])
        #    pre_polygon = np.vstack([pre_polygon, pre_polygon[0]]) # duplicate first point as the last point, making sure it's a closed object
        if shape_type == 'POLYGON' or shape_type == 'RECTANGLE':
            pre_polygon = np.vstack([pre_polygon, pre_polygon[0]]) # duplicate first point as the last point, making sure it's a closed object # not sure if this is needed. doesn't hurt? pretty sure
        elif shape_type == 'ELLIPSE':
            # if the ellipse is rotated it's turned into a polygon, so don't worry about that c:
            # get the x parts of the two points
            xs = [pre_polygon[0][0], pre_polygon[1][0]]
            ys = [pre_polygon[0][1], pre_polygon[1][1]] 
            x_len = xs[1] - xs[0]
            y_len = ys[1] - ys[0]
            x_pos = xs[1] - (x_len/2)
            y_pos = ys[1] - (y_len/2)
            circle = shapely.Point(x_pos, y_pos).buffer(1)
            ellipse = shapely.affinity.scale(circle, x_len/2, y_len/2)
            pre_polygon = np.array([x for x in ellipse.exterior.coords])
            # no need to duplicate first point, already done by shapely
        else:
            # raise an error
            raise NotImplementedError(f"Converting this shape type ({shape_type}) into a polygon/multipolygon isn't implemented.")
        polygon = Polygon(pre_polygon)
        # make the polygon valid
        polygon = make_valid(polygon)
        if region['isExclusionRegion']:
            cutouts.append(polygon)
        else:
            polys.append(polygon)
    
    # make sure there are no multipolygons in polys
    polys = list_of_shapes_to_polygons(polys)

    polygons = [] # this part can definitely be sped up, probably easiest with geopandas?
    for poly in polys:
        for cutout in cutouts:
            if poly.contains(cutout):
                poly = poly.difference(cutout)
        polygons.append(poly)

    # make sure there are no multipolygons in polys
    polygons = list_of_shapes_to_polygons(polygons)

    # pretty sure the make_valid here cuts holes into larger polygons if smaller ones are on top, not necessarily a problem atm?
    return shapely.make_valid(shapely.MultiPolygon(polygons))



def get_metrics(input_mask, output_mask, metric_area_mask = None):
    """ This function takes in two masks, previously aligned, and gets 
    metrics for determining if a classifier is accurate and stuff.

    If there are certain regions of the image that you don't want to
    include in the metric calculations, provide a metric_area_mask, 
    with 1s being where you want metrics, and 0 being where you
    don't want metrics.

    Currently gets (by pixel):
    - true positives (i20 - o1 == 19)
    - true negatives (i10 - o0 == 10)
    - false positives (i10 - o1 == 9)
    - false negatives (i20 - o0 == 20)
    - accuracy ((tp + tn) / (everything (tp + tn + fp + fn)))
    - precision (tp / (tp + fp))
    - recall (tp / (tp + fn))
    - f1 (2 * precision * recall / (precision + recall))
    - jaccard's index (tp / (tp + fn + fp))
    """
    # make sure input/output mask shapes are the same
    assert input_mask.shape == output_mask.shape, f"Mask shapes not the same!! ({input_mask.shape=} {output_mask.shape=})"

    # true is input
    # change output matrix
    input_mask += 1
    input_mask *= 10
    
    # subtract
    ## this makes:
    ## 19: true positive
    ## 20: false negative
    ## 10: true negative
    ## 9: false positive
    diff = input_mask - output_mask
    
    # adjust for metric area mask
    # if there is a metric area mask, make values that equal
    # 0 in the metric area mask 0 in the diff (removing them
    # from calculation)
    if type(metric_area_mask) != type(None):
        # make sure the diff mask has the same shape as the metric_area_mask
        assert metric_area_mask.shape == diff.shape, f"Metric area mask different than input/output mask shapes ({metric_area_mask.shape=} {diff.shape=})"
        diff = np.where(metric_area_mask == 0, 0, diff)

    # takes ~10 seconds each?
    tp = (diff == 19).sum(dtype=np.int64)
    tn = (diff == 10).sum(dtype=np.int64)
    fp = (diff == 9).sum(dtype=np.int64)
    fn = (diff == 20).sum(dtype=np.int64)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn) # same as tpr

    f1 = 2 * precision * recall / (precision + recall)

    jaccard = tp / (tp + fn + fp)

    fpr = fp/(fp+tn)

    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "jaccard":jaccard, "fpr":fpr}

def simplify_and_smooth(polygon, mitre_distance=10, thinness_distance=25, first_simplify=5, second_simplify=3):
    """Simplifies, smooths, and then checks for tight regions on the shape(s) and separates them further.
    Can result in a MultiPolygon!"""
    original_polygon = polygon
    polygon = polygon.simplify(first_simplify, preserve_topology=True)
    # can results in multiple polygons
    polygon = polygon.buffer(-mitre_distance, join_style="mitre").buffer(mitre_distance, join_style="mitre")
    #if type(polygon) == shapely.MultiPolygon:
    #    polygons = []
    #    for geom in list(polygon.geoms):
    #        if geom.area > 10:
    #            polygons.append(chaikin_smooth(geom).simplify(second_simplify, preserve_topology=True))
    #        else:
    #            continue
    #    polygon = shapely.MultiPolygon(polygons)
    #else:
    #    if polygon.area > 10:
    #        polygon = chaikin_smooth(polygon).simplify(second_simplify, preserve_topology=True)
    #    else:
    #        # return an empty polygon
    #        return Polygon()
    
    polygon = make_valid(polygon)
    # check for tight regions, if the polygon splits into a multipolygon, keep the split polygon
    # if not, keep the original (because this step can mess with the the original shapes)
    first_buffer_step = make_valid(polygon.buffer(-thinness_distance, join_style="mitre"))
    second_buffer_step = first_buffer_step.buffer(thinness_distance, join_style="mitre")
    split_poly = make_valid(polygon.intersection(make_valid(second_buffer_step)))
    if type(split_poly) == shapely.MultiPolygon:
        polygon = split_poly
    

    fix_multigeo = []
    if type(polygon) == shapely.GeometryCollection:
        for shape in polygon.geoms:
            if type(shape) == shapely.Polygon:
                fix_multigeo.append(shape)
        polygon = shapely.MultiPolygon(fix_multigeo)

    polygon = original_polygon.intersection(polygon)

    return polygon

# finding the slices could use a lot of fixing up
# 1. vertical and horizontal slice functions, or at least their shared lines of code, should be combined
# 2. object type checking should be handled more smoothly
# 3. better documentation
def find_optimal_vertical_slice(polygon, cut_size, tests, minimum_shape_area, maximum_shape_area, pbar=False, **kwargs):
    if polygon.area < maximum_shape_area:
        return {'polygon':polygon, 'stats':pd.Series({
            'area':polygon.area,
            'n_shapes':1,
            'all_shapes_smaller_than_max':True,
            'mean_diffs_from_perfect':0
        })}
    exterior_np_array = np.array(polygon.exterior.coords)
    maxs = exterior_np_array.max(axis=0)
    mins = exterior_np_array.min(axis=0)
    top_right = (maxs[0], maxs[1])
    bottom_left = (mins[0], mins[1])
    min_x = mins[0] + cut_size*2
    max_x = maxs[0] - cut_size*2
    min_y = mins[1]
    max_y = maxs[1]
    step = (max_x - min_x) / tests

    test_cut_polygons = []
    #unary_union([shapely.Point(top_right).buffer(50), polygon, shapely.Point(bottom_left).buffer(60)])
    for test in tqdm(range(tests), disable=not pbar):
        cut_shape = shapely.LineString([(min_x + step*test, min_y), (min_x + step*test, max_y)]).buffer(cut_size) # line string from the top to the bottom of the shape, with the width of the cut_size
        cut_polygon = polygon.difference(cut_shape)
        # simplify and smooth the cut shapes
        cut_polygon = simplify_and_smooth(cut_polygon, **kwargs)
        test_cut_polygons.append(cut_polygon)

    # check each test cut to see which is the 'best' by certain parameters
    # first, we get a dataframe with the statistics for the shapes
    test_cut_shape_stats = {}
    for i, test_cut_multipolygon in tqdm(enumerate(test_cut_polygons), total=tests, disable=not pbar):
        current_cut_shape_stats = defaultdict(int)
        current_cut_shape_stats['all_shapes_smaller_than_max'] = True
        test_cut_geometrycollection = []
        if type(test_cut_multipolygon) == shapely.GeometryCollection:
            # remove anything that isn't a polygon
            for cut_shape in list(test_cut_multipolygon.geoms):
                if type(cut_shape) == shapely.Polygon:
                    test_cut_geometrycollection.append(cut_shape)
            test_cut_multipolygon = shapely.MultiPolygon(test_cut_geometrycollection)
        elif type(test_cut_multipolygon) != shapely.MultiPolygon: # everything else
            test_cut_multipolygon = convert_to_polygon_list(test_cut_multipolygon)
            test_cut_multipolygon = shapely.MultiPolygon(test_cut_multipolygon)

        
        n_shapes = len(list(test_cut_multipolygon.geoms))
        perfect_cut_area = test_cut_multipolygon.area/n_shapes
        diffs_from_perfect = []
        for cut_shape in list(test_cut_multipolygon.geoms):
            # skip shapes that are too small to cut/an unreasonable size # i think this was causing problems, we can deal with small shapes later
            #if cut_shape.area < minimum_shape_area:
            #    continue
            # the area of the shapes created (of reasonable size)
            current_cut_shape_stats['area'] = current_cut_shape_stats['area'] + cut_shape.area
            # number of shapes created (again, of resonable size)
            current_cut_shape_stats['n_shapes'] = current_cut_shape_stats['n_shapes'] + 1
            # check if shape's area is smaller than the max allowed
            diffs_from_perfect.append(np.abs(perfect_cut_area - cut_shape.area))
            if cut_shape.area > maximum_shape_area:
                current_cut_shape_stats['all_shapes_smaller_than_max'] = False
        current_cut_shape_stats['mean_diffs_from_perfect'] = np.mean(diffs_from_perfect)
        test_cut_shape_stats[i] = current_cut_shape_stats.copy()

    stat_df = pd.DataFrame(test_cut_shape_stats).T

    cuts_still_being_considered = stat_df
    # remove rows that have n_shapes < 2
    cuts_still_being_considered = cuts_still_being_considered[cuts_still_being_considered['n_shapes'] > 1]
    best_cut_polygon = {}
    # if there are cuts where the remaining shapes are all smaller than the max allowed shape size, just look at those
    if cuts_still_being_considered["all_shapes_smaller_than_max"].sum() > 1:
        cuts_still_being_considered = cuts_still_being_considered[cuts_still_being_considered["all_shapes_smaller_than_max"] == True]
        # return the highest area remaining post cut
        best_cut_polygon['polygon'] = test_cut_polygons[cuts_still_being_considered.sort_values('area', ascending=False).index[0]]
        best_cut_polygon['stats'] = cuts_still_being_considered.sort_values('area', ascending=False).iloc[0, :]
        return best_cut_polygon
    cuts_still_being_considered = cuts_still_being_considered.sort_values("mean_diffs_from_perfect", ascending=True)


    # currently chooses a combination of the highest area and somewhere in the middle of the shape...
    thingy = pd.DataFrame(cuts_still_being_considered['area'].rank(ascending=False))
    thingy['mean_diffs_from_perfect'] = cuts_still_being_considered['mean_diffs_from_perfect'].rank()
    try:
        to_keep = thingy.sum(axis=1).sort_values().index[0] # this causes an indexerror if there are no shapes, probably because the shape is so small to start with.
    except IndexError: # need to make an empty polygon
        to_keep = 0
        test_cut_polygons.append(Polygon())
        cuts_still_being_considered = pd.DataFrame({0:{'area':0}}).T    
    best_cut_polygon['polygon'] = test_cut_polygons[to_keep]
    best_cut_polygon['stats'] = cuts_still_being_considered.loc[to_keep, ]
    return best_cut_polygon

def find_optimal_horizontal_slice(polygon, cut_size, tests, minimum_shape_area, maximum_shape_area, pbar=False, **kwargs):
    if polygon.area < maximum_shape_area:
        return {'polygon':polygon, 'stats':pd.Series({
            'area':polygon.area,
            'n_shapes':1,
            'all_shapes_smaller_than_max':True,
            'mean_diffs_from_perfect':0
        })}
    exterior_np_array = np.array(polygon.exterior.coords)
    maxs = exterior_np_array.max(axis=0)
    mins = exterior_np_array.min(axis=0)
    top_right = (maxs[0], maxs[1])
    bottom_left = (mins[0], mins[1])
    min_x = mins[0]
    max_x = maxs[0]
    min_y = mins[1] + cut_size*2
    max_y = maxs[1] - cut_size*2
    step = (max_x - min_x) / tests

    test_cut_polygons = []
    #unary_union([shapely.Point(top_right).buffer(50), polygon, shapely.Point(bottom_left).buffer(60)])
    for test in tqdm(range(tests), disable=not pbar):
        cut_shape = shapely.LineString([(min_x, min_y + step*test), (max_x, min_y + step*test)]).buffer(cut_size) # line string from the top to the bottom of the shape, with the width of the cut_size
        cut_polygon = polygon.difference(cut_shape)
        # simplify and smooth the cut shapes
        cut_polygon = simplify_and_smooth(cut_polygon, **kwargs)
        test_cut_polygons.append(cut_polygon)

    # check each test cut to see which is the 'best' by certain parameters
    # first, we get a dataframe with the statistics for the shapes
    test_cut_shape_stats = {}
    for i, test_cut_multipolygon in tqdm(enumerate(test_cut_polygons), total=tests, disable=not pbar):
        current_cut_shape_stats = defaultdict(int)
        current_cut_shape_stats['all_shapes_smaller_than_max'] = True
        test_cut_geometrycollection = []
        if type(test_cut_multipolygon) == shapely.GeometryCollection:
            # remove anything that isn't a polygon
            for cut_shape in list(test_cut_multipolygon.geoms):
                if type(cut_shape) == shapely.Polygon:
                    test_cut_geometrycollection.append(cut_shape)
            test_cut_multipolygon = shapely.MultiPolygon(test_cut_geometrycollection)
        elif type(test_cut_multipolygon) != shapely.MultiPolygon: # everything else
            test_cut_multipolygon = convert_to_polygon_list(test_cut_multipolygon)
            test_cut_multipolygon = shapely.MultiPolygon(test_cut_multipolygon)

        
        n_shapes = len(list(test_cut_multipolygon.geoms))
        perfect_cut_area = test_cut_multipolygon.area/n_shapes
        diffs_from_perfect = []
        for cut_shape in list(test_cut_multipolygon.geoms):
            # skip shapes that are too small to cut/an unreasonable size # i think this was causing problems, we can deal with small shapes later
            #if cut_shape.area < minimum_shape_area:
            #    continue
            # the area of the shapes created (of reasonable size)
            current_cut_shape_stats['area'] = current_cut_shape_stats['area'] + cut_shape.area
            # number of shapes created (again, of resonable size)
            current_cut_shape_stats['n_shapes'] = current_cut_shape_stats['n_shapes'] + 1
            # check if shape's area is smaller than the max allowed
            diffs_from_perfect.append(np.abs(perfect_cut_area - cut_shape.area))
            if cut_shape.area > maximum_shape_area:
                current_cut_shape_stats['all_shapes_smaller_than_max'] = False
        current_cut_shape_stats['mean_diffs_from_perfect'] = np.mean(diffs_from_perfect)
        test_cut_shape_stats[i] = current_cut_shape_stats.copy()

    stat_df = pd.DataFrame(test_cut_shape_stats).T

    cuts_still_being_considered = stat_df
    # remove rows that have n_shapes < 2
    cuts_still_being_considered = cuts_still_being_considered[cuts_still_being_considered['n_shapes'] > 1]
    best_cut_polygon = {}
    # if there are cuts where the remaining shapes are all smaller than the max allowed shape size, just look at those
    if cuts_still_being_considered["all_shapes_smaller_than_max"].sum() > 1:
        cuts_still_being_considered = cuts_still_being_considered[cuts_still_being_considered["all_shapes_smaller_than_max"] == True]
        # return the highest area remaining post cut
        best_cut_polygon['polygon'] = test_cut_polygons[cuts_still_being_considered.sort_values('area', ascending=False).index[0]]
        best_cut_polygon['stats'] = cuts_still_being_considered.sort_values('area', ascending=False).iloc[0, :]
        return best_cut_polygon
    cuts_still_being_considered = cuts_still_being_considered.sort_values("mean_diffs_from_perfect", ascending=True)


    # currently chooses a combination of the highest area and somewhere in the middle of the shape...
    thingy = pd.DataFrame(cuts_still_being_considered['area'].rank(ascending=False))
    thingy['mean_diffs_from_perfect'] = cuts_still_being_considered['mean_diffs_from_perfect'].rank()
    try:
        to_keep = thingy.sum(axis=1).sort_values().index[0] # this causes an indexerror if there are no shapes, probably because the shape is so small to start with.
    except IndexError: # need to make an empty polygon
        to_keep = 0
        test_cut_polygons.append(Polygon())
        cuts_still_being_considered = pd.DataFrame({0:{'area':0}}).T
    best_cut_polygon['polygon'] = test_cut_polygons[to_keep]
    best_cut_polygon['stats'] = cuts_still_being_considered.loc[to_keep, ]
    return best_cut_polygon

def find_best_slice(polygon, cut_size, tests, minimum_shape_area, maximum_shape_area, pbar=False, **kwargs):
    if polygon.area < maximum_shape_area:
        return polygon
    else:
        horiz = find_optimal_horizontal_slice(polygon, cut_size=cut_size, tests=tests, minimum_shape_area=minimum_shape_area, maximum_shape_area=maximum_shape_area, pbar=pbar, **kwargs)
        vert = find_optimal_vertical_slice(polygon, cut_size=cut_size, tests=tests, minimum_shape_area=minimum_shape_area, maximum_shape_area=maximum_shape_area, pbar=pbar, **kwargs)

        # remove anything that isnt a polygon from horiz and vert
        horiz_polygon = []
        for shape in horiz['polygon'].geoms:
            if type(shape) == shapely.Polygon:
                horiz_polygon.append(shape)
        horiz_polygon = shapely.MultiPolygon(horiz_polygon)

        vert_polygon = []
        for shape in vert['polygon'].geoms:
            if type(shape) == shapely.Polygon:
                vert_polygon.append(shape)
        vert_polygon = shapely.MultiPolygon(vert_polygon)

        
        if vert['stats']['area'] > horiz['stats']['area']:
            return vert_polygon
        else:
            return horiz_polygon

# thank you @StefanBrand_EOX (https://gis.stackexchange.com/questions/333709/inserting-n-points-equally-distributed-into-many-polygons-in-qgis/333822#333822)
def _get_voronoi_starting_points(polygon: Polygon, point_count: int) -> shapely.geometry.MultiPoint:
    perimeter = polygon.exterior
    segment_length = perimeter.length / point_count
    segment_starts = itertools.islice(itertools.count(0, segment_length), 0, point_count)

    return shapely.geometry.MultiPoint([shapely.ops.substring(perimeter, start, start) for start in segment_starts])


def spatial_segmentation_brute_force(multipolygon: shapely.geometry.MultiPolygon,
                                     tissue: shapely.geometry.MultiPolygon | shapely.geometry.Polygon,
                                     n: int,
                                     s: float,
                                     mpp: float,
                                     verbose: bool = False):
    """ Using a tissue shape and a bunch of shapes on the tissue shape, segment
    the shapes into n segments of size s (mm^2) with micrometers per pixel mpp.

    Returns a geopandas dataframe..?
    """
    # convert s from mm^2 to pixels
    s_pixels_squared = s * 1000000 * (1/mpp**2)

    # get the centroids of all the geometries
    centroids = []
    multipolygon_polygons = []
    for geom in multipolygon.geoms:
        centroids.append(geom.centroid)
        multipolygon_polygons.append(geom)

    # check that it is feasible
    real_area = multipolygon.area / 1000000 * mpp**2
    if (s * n) > real_area: raise Exception(f"There isn't enough area to choose from. {(s * n)=}mm², {real_area=:.2f}mm²")

    voronoi = shapely.ops.voronoi_diagram(_get_voronoi_starting_points(tissue, n))
    equally_distributed_points = [part.intersection(tissue).representative_point() for part in voronoi.geoms]

    working_segments = {x:shapely.MultiPolygon() for x in range(n)}
    final_segments = {}
    current_largest_segment_area = 0.000000000001 # current largest segment, to grow other segments to
    while len(final_segments) != n and len(centroids) > 0:
        for segment_n in working_segments:
            if segment_n not in final_segments:
                current_segment_size = working_segments[segment_n].area
                # check if we need more for the largest segment
                while current_segment_size < current_largest_segment_area and len(centroids) > 0: # i think this has a possiblity to run forever...
                    # find the closest centroid
                    closest_centroid = shapely.ops.nearest_points(equally_distributed_points[segment_n], shapely.geometry.MultiPoint(centroids))[1]
                    # get the index of the centroid in our centroids list
                    closest_centroid_index = centroids.index(closest_centroid)
                    # get the polygon that created that centroid
                    closest_polygon = multipolygon_polygons[closest_centroid_index]
                    # add the polygon to our current segment
                    ## I believe this is the line that takes the longest, especially when the working segment already has a lot of shapes
                    working_segments[segment_n] = shapely.unary_union([closest_polygon, working_segments[segment_n]]) # update the current polygon
                    # get the new area
                    current_segment_size = working_segments[segment_n].area
                    current_largest_segment_area = max(current_segment_size, current_largest_segment_area, s)
                    # remove the centroid and the polygon from their respective lists
                    centroids.remove(closest_centroid)
                    multipolygon_polygons.remove(closest_polygon)

                    # check if the current size is equal to 
                    if current_segment_size >= s_pixels_squared:
                        final_segments[segment_n] = working_segments[segment_n]
                        break
    if len(centroids) == 0:
        if verbose:
            print('Ran out of shapes!')
        return working_segments
    else:
        if verbose:
            print('Completed!')
        return final_segments

def get_all_intersections(geoseries: gpd.GeoSeries, verbose: bool = False):
    """ Gets all the intersections between shapes in a
    gpd.GeoSeries object.

    Returns a list of polygons.
    """
    intersections = []
    for geometry in geoseries.values:
        intersection = convert_to_polygon_list(geoseries.intersection(geometry).intersection_all(), verbose=verbose)
        for geom in intersection:
            intersections.append(geom)
    return intersections

def remove_thin_regions(shape: shapely.Polygon, cut_size: int | float, verbose: bool = False):
    """ Takes a polygon and removes regions from shapes that are too
    thin (comparative to cut_size).

    Returns either a polygon or multipolygon.
    """
    negative_buffer = shape.buffer(-cut_size)
    positive_buffer = negative_buffer.buffer(cut_size + (cut_size*0.01))
    # get intersection with original shape
    intersection = positive_buffer.intersection(shape)
    # convert into a polygon list
    polygon_list = convert_to_polygon_list(intersection, verbose=verbose)
    ## remove non polygons (linestrings, etc)
    polygon_list = [polygon for polygon in polygon_list if type(polygon) == shapely.Polygon]
    final_shapes = []
    # remove internal pins
    for pg in polygon_list:
        # positive buffer
        buf = pg.buffer(cut_size)
        # negative buffer
        neg = buf.buffer(-cut_size)
        # intersection with original
        final_shapes.append(shape.intersection(neg))
    return shapely.unary_union(final_shapes)

def remove_close_calls(shapes: shapely.MultiPolygon, cut_size: int | float):
    """ Takes a multipolygon and slices out regions where objects are too close
    to one another within the multipolygon.

    Returns a multipolygon (if provided a multipolygon..?).
    """
    shapes_gpd = gpd.GeoSeries(shapes.geoms)
    intersections = gpd.GeoSeries(get_all_intersections(shapes_gpd.buffer(cut_size)))
    bs = shapes_gpd.buffer(cut_size)

    intersections = []
    for geom in bs.values:
        bs_without_geom = bs[bs != geom]
        thing = bs_without_geom.intersection(geom)
        thing = thing.polygonize()
        for pg in thing.values:
            intersections.append(pg)
    thin_regions_removed = shapes.difference(shapely.union_all(intersections))
    return thin_regions_removed

def scrub_cycle(shape: shapely.Polygon, cut_size: int | float, final_shapes: list, verbose: bool = False):
    """ Takes a polygon and removes regions that are too thin/pins into the shape
    recursively.

    Adds all final shapes to the provided list as shapely.Polygon objects.
    """
    # remove thin regions from the shape
    thin_regions_removed = remove_thin_regions(shape, cut_size=cut_size)
    thin_region_polygons_list = convert_to_polygon_list(thin_regions_removed, verbose=verbose)
    # check for multipolygon
    if len(thin_region_polygons_list) > 1:
        close_calls_removed = remove_close_calls(shapely.MultiPolygon(thin_region_polygons_list), cut_size=cut_size)
        close_calls_removed_polygon_list = convert_to_polygon_list(close_calls_removed, verbose=verbose)
        if len(close_calls_removed_polygon_list) > 1:
            for close_call in close_calls_removed_polygon_list:
                scrub_cycle(close_call, cut_size=cut_size, final_shapes=final_shapes)
        else:
            scrub_cycle(close_calls_removed_polygon_list[0], cut_size=cut_size, final_shapes=final_shapes)
    elif len(thin_region_polygons_list) == 1:
        final_shapes.append(thin_region_polygons_list[0])
        return 1
    else:
        # the shape(s) has/have disappeared, probably too small?
        return 0

# TODO - split shapes up first before making connectors to negative regions,
# as we might just split on the negative region anyways, would probably save on yield
def make_shapes_lmdable(image_location: str | Path,
                        collection_regions: MultiPolygon,
                        minimum_micrometers_squared: int = 10000,
                        maximum_micrometers_squared: int = 600*600,
                        minimum_micrometers_between_shapes: int = 60,
                        n_slices_to_check: int = 5,
                        verbose: int = True):
    """ Converts HALO regions into shapely shapes that can be exported to XML to cut on an LMD scope.

    Args:
        image_location (:class:`str` | :class:`pathlib.Path`):
            The image file location (specifically, the same file that HALO is using)
        collection_regions (:class:`MultiPolygon`):
            The regions to be prepared for LMD collection
        minimum_micrometers_squared (:class:`int`):
            Minimum size of an object to cut out, smaller sized objects are removed
        maximum_micrometers_squared (:class:`int`):
            Maximum size of an object to cut out, larger sized objects are sliced 
            into smaller ones
        minimum_micrometers_between_shapes (:class:`int`):
            The minimum distance between shapes. Anything too close will be shaved,
            and larger object slices will be separated with this distance
        n_slices_to_check (:class:`int`):
            The number of horizontal and vertical slice positions to check when
            deciding the most optimal
        verbose (:class:`int`):
            Whether or not to create progress bars and print status updates
    
    Returns:
        :class:`shapely.MultiPolygon`:
            MultiPolygon filled with shapes that can be converted to XML for
            cutting on an LMD scope
    """
    if type(image_location) == str:
        image_location = Path(image_location)
    assert image_location.exists(), f"Path {image_location=} does not exist!"
    # get mpp from image
    if verbose:
        print("reading in image with WSIReader (getting mpp)")
    wsi = WSIReader.open(image_location)
    mpp = wsi.info.as_dict()['mpp']
    assert mpp[0] == mpp[1], "x y mpp different, didn't account for this..."
    mpp = mpp[0]

    # get minimum shape areas in pixels
    minimum_shape_area = minimum_micrometers_squared/(mpp**2)
    maximum_shape_area = maximum_micrometers_squared/(mpp**2)

    # get slice/cut size
    cut_size = minimum_micrometers_between_shapes/2/mpp

    # get all the objects into a mask
    polygons_all = list(collection_regions.geoms)
    polygons = []
    cutouts = []
    
    for poly in polygons_all:
        # Exterior polygon
        polygons.append(Polygon(poly.exterior))
    
        # Interior (hole) polygons
        for interior in poly.interiors:
            # Make sure the hole is a valid polygon (must be closed ring with area)
            ring = LinearRing(interior)
            if ring.is_valid and not ring.is_empty:
                cutouts.append(Polygon(ring))

    # make connectors to cutouts
    polys = []
    for polygon in tqdm(polygons, desc="connectors", disable=not verbose):
        for cutout in cutouts:
            if polygon != cutout and polygon.intersects(cutout):
                # fix polygon (cutting out region, but also connecting it to non-shape)
                # figure out of polygon is a multipolygon
                if type(polygon) != shapely.Polygon:
                    poly_list = []
                    for actual_polygon in tqdm(list(polygon.geoms), desc="connectors inner", disable=not verbose):
                        if type(actual_polygon) == shapely.LineString or type(actual_polygon) == shapely.Point:
                            # skip these
                            continue
                        # add a connector to the polygon
                        connector = shapely.LineString([[x for x in nearest_points(actual_polygon.exterior, cutout.centroid)][0], [x for x in cutout.centroid.coords][0]]).buffer(cut_size, cap_style='square', join_style='round')
                        actual_polygon = actual_polygon.difference(unary_union([cutout, connector, cutout.centroid.buffer(cut_size)]))
                        poly_list.append(actual_polygon)
                    polygon = unary_union(poly_list)
                else:
                    # add a connector to the polygon
                    connector = shapely.LineString([[x for x in nearest_points(polygon.exterior, cutout.centroid)][0], [x for x in cutout.centroid.coords][0]]).buffer(cut_size, cap_style='square', join_style='round')
                    polygon = polygon.difference(unary_union([cutout, connector, cutout.centroid.buffer(cut_size)]))
        polys.append(polygon)        
        
    # fix multipolygons (in case there is multipolygon inception?)
    if verbose:
        print('fixing multipolygons')
    polys = list_of_shapes_to_polygons(polys, minimum_shape_area=minimum_shape_area)

    # simplify and smooth objects, splitting up objects that have tight regions between them
    simplified_polys = []
    for poly in tqdm(polys, desc="simplifying", disable=not verbose):
        #new_poly = hsf.simplify_and_smooth(shapely.make_valid(poly), thinness_distance=cut_size)
        new_poly = shapely.simplify(poly, 5)
        if type(new_poly) == shapely.MultiPolygon:
            for polyi in list(new_poly.geoms):
                simplified_polys.append(polyi)
        else:
            simplified_polys.append(new_poly)

    # fix multipolygons (in case there is multipolygon inception?)
    if verbose:
        print('fixing multipolygons')
    simplified_polys = list_of_shapes_to_polygons(simplified_polys, minimum_shape_area=minimum_shape_area)

    # slice up polygons that are too large
    if verbose:
        print('slicing')
    # kwargs: mitre_distance=10, thinness_distance=25, first_simplify=5, second_simplify=3
    n_too_large = 0
    for simplified_poly in simplified_polys:
        if simplified_poly.area > maximum_shape_area:
            n_too_large = n_too_large + 1
    while n_too_large > 0:
        n_too_large = 0
        for simplified_poly in tqdm(simplified_polys.copy(), disable=not verbose):
            if simplified_poly.area > maximum_shape_area:
                n_too_large = n_too_large + 1
                simplified_polys.remove(simplified_poly)
                simplified_poly = make_valid(simplified_poly) # make the polygon valid
                # double checking for multipolygons/geometrycollections
                temp_polygon_list = convert_to_polygon_list(simplified_poly)
                sliced_polys = []
                for simplified_poly_temp in temp_polygon_list:
                    sliced_poly = find_best_slice(simplified_poly_temp, cut_size=cut_size, tests=n_slices_to_check, minimum_shape_area=minimum_shape_area, maximum_shape_area=maximum_shape_area, 
                                                thinness_distance=cut_size, mitre_distance=0, pbar=verbose)
                    sliced_polys.append(make_valid(sliced_poly))
                # add polygons back to simplified_polys
                simplified_polys.extend(sliced_polys)
        if verbose:
            print(f"{n_too_large=} remaining")


    # fix multipolygons
    if verbose:
        print('fixing multipolygons')
    simplified_polys = list_of_shapes_to_polygons(simplified_polys, minimum_shape_area=minimum_shape_area)


    # removing close calls
    if verbose:
        print('removing `close calls`')
    simplified_polys = remove_close_calls(shapely.make_valid(shapely.MultiPolygon(simplified_polys)), cut_size=cut_size)
    simplified_polys = convert_to_polygon_list(simplified_polys, verbose=verbose)

    # scrub cycle
    if verbose:
        print('shape scrub cycle')
    scrubbed_polys = []
    for poly in tqdm(simplified_polys, disable=not verbose):
        intermediately_scrubbed = []
        scrub_cycle(poly, cut_size=cut_size, final_shapes=intermediately_scrubbed)
        scrubbed_polys.extend(intermediately_scrubbed.copy())


    simplified_polys = scrubbed_polys.copy()

    # remove shapes that have been added that are too small
    for simplified_poly in tqdm(simplified_polys.copy(), desc="removing small shapes", disable=not verbose):
        if simplified_poly.area < minimum_shape_area:
            simplified_polys.remove(simplified_poly)

    # simplify and smooth objects, splitting up objects that have tight regions between them
    simplified_polys_again = []
    for poly in tqdm(simplified_polys, desc="simplifying", disable=not verbose):
        #new_poly = hsf.simplify_and_smooth(shapely.make_valid(poly), thinness_distance=cut_size)
        new_poly = shapely.simplify(poly, 5)
        if type(new_poly) == shapely.MultiPolygon:
            for polyi in list(new_poly.geoms):
                simplified_polys_again.append(polyi)
        else:
            simplified_polys_again.append(new_poly)

    # fix multipolygons (in case there is multipolygon inception?)
    if verbose:
        print('fixing multipolygons')
    simplified_polys_again = list_of_shapes_to_polygons(simplified_polys_again, minimum_shape_area=minimum_shape_area)

    # make a valid multipolygon
    final_shapely_multipolygon = shapely.make_valid(shapely.MultiPolygon(simplified_polys_again))
    
    return final_shapely_multipolygon

def plot_shapes_onto_tissue(shapes: dict[int|str, shapely.geometry.MultiPolygon | shapely.geometry.Polygon] | shapely.geometry.MultiPolygon | shapely.geometry.Polygon,
                            image_location: str | Path,
                            framing_shape: shapely.geometry.MultiPolygon | shapely.geometry.Polygon = None,
                            cmap: str = 'tab10',
                            buffer_size: int = 700,
                            mpp: float = 10.0,
                            plot: bool = True,
                            ax = None,
                            draw_convex_hulls = False,
                            draw_text = False,
                            border_kwargs: dict = dict(
                                linewidth=1,
                                color='black'
                            ),
                            shape_kwargs: dict = dict(
                                alpha=0.7
                            ),
                            text_kwargs: dict = dict(
                                fontsize=12, 
                                fontweight='bold', 
                                backgroundcolor='white', 
                                ha='center',
                                va='center'
                            ),
                            line_kwargs: dict = dict(
                                linewidth=2, 
                                color='black'
                            )):
    """Plot shapes onto a section of tissue.

    Args:
        shapes (:class:`dict[int|str, shapely.geometry.MultiPolygon | shapely.geometry.Polygon]`):
            Dictionary of multipolygons/polygons, each key/item pair being a different color in the plot
        image_location (:class:`str | Path`):
            Path to image
        framing_shape (:class:`shapely.geometry.MultiPolygon | shapely.geometry.Polygon = None`):
            Either a polygon or a multipolygon that is used to gather the boundary of the image/figure to plot
        buffer_size (:class:`int = 700`):
            Additional buffer around the framing shape (in pixels)
        new_mpp (:class:`float = 10.0`):
            The new resolution in micrometers per pixel (mpp) to have the image section be in. Larger values means lower resolution, with 0.5 usually being around the highest resolution 
        plot (:class:`bool = True`):
            Whether to actually plot the figure
        ax (`None`):
            Matplotlib ax to plot on
        shape_kwargs (:class:`dict = dict(alpha=0.7)`):
            kwargs to pass to the shape plotting function
        border_kwargs (:class:`dict = dict(linewidth=1, color='black')`):
            kwargs to pass to the border plotting function

    Returns:
        :class:`dict`:
            Transformed shapes (key `plotted_shapes`) and image section (key `section_image`) in a dictionary
    """
    # check shape types
    if type(shapes) == dict:
        for key in shapes:
            shape = shapes[key]
            assert (type(shape) == shapely.geometry.MultiPolygon) or (type(shape) == shapely.geometry.Polygon), "Make sure `shapes` contains only Polygons or MultiPolygons from shapely"
    else:
        assert (type(shapes) == shapely.geometry.MultiPolygon) or (type(shapes) == shapely.geometry.Polygon), "Make sure `shapes` is a Polygon or MultiPolygon from shapely"
    # check image location exists
    if type(image_location) == str:
        image_location = Path(image_location)
    assert image_location.exists(), f"`image_location` doesn't exist ({image_location=})"
    # check framing type
    if framing_shape != None:
        assert (type(framing_shape) == shapely.geometry.MultiPolygon) or (type(framing_shape) == shapely.geometry.Polygon), "Make sure `framing_shape` is a Polygon or MultiPolygon from shapely"
    # check that colormap exists
    assert cmap in mpl.colormaps.keys(), f"{cmap} is not a colormap (try `[print(x) for x in matplotlib.colormaps.keys()]`)"

    # make dictionary to return
    ret_dict = {}

    # create geopandas dataframe with shapes
    df = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapes))

    # get bounds
    if framing_shape != None:
        bounds_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries(framing_shape))
    else:
        bounds_df = df.copy()

    min_x = bounds_df.bounds.min()['minx'] - buffer_size
    min_y = bounds_df.bounds.min()['miny'] - buffer_size
    max_x = bounds_df.bounds.max()['maxx'] + buffer_size
    max_y = bounds_df.bounds.max()['maxy'] + buffer_size
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    loc = (int(min_x), int(min_y))

    # load image
    image_wsi = WSIReader.open(image_location)
    original_mpp = image_wsi.info.as_dict()['mpp'][0]

    # get mpp ratio for shape translation
    mpp_ratio = original_mpp / mpp # get the ratio of the original mpp to the new mpp
    rect = image_wsi.read_rect(location=loc, size=(int(width*mpp_ratio), int(height*mpp_ratio)), resolution=mpp, units="mpp")

    # translate and scale the shapes
    translated_and_scaled = df.reset_index(names=['spatial_segment'])
    translated_and_scaled['geometry'] = translated_and_scaled.translate(-min_x, -min_y).scale(mpp_ratio, mpp_ratio, origin=(0, 0))
    #color = ["#2ca02c", "#9467bd", "#e377c2", "#bcbd22","#17becf", "#1f77b4"]
    #mpl.colors.ListedColormap(color)
    if plot:
        if ax == None:
            _, ax = plt.subplots(figsize=(15,15))
        ax.imshow(rect)
        translated_and_scaled.plot(column='spatial_segment', cmap=cmap, ax=ax, **shape_kwargs)
        translated_and_scaled.boundary.plot(ax=ax, **border_kwargs)

        # draw convex hull lines
        if draw_convex_hulls:
            for group_of_shapes in translated_and_scaled.iterrows():
                c_hull = shapely.convex_hull(group_of_shapes[1]['geometry'])
                hull_coords = np.array([[x, y] for x, y in c_hull.boundary.coords])
                hull_x, hull_y = hull_coords[:, 0], hull_coords[:, 1]
                plt.plot(hull_x, hull_y, 
                        **line_kwargs)

        # add the text on top of the groups
        if draw_text:
            for group_of_shapes in translated_and_scaled.iterrows():
                c_hull = shapely.convex_hull(group_of_shapes[1]['geometry'])
                num_pos_x, num_pos_y = [(x, y) for x, y in c_hull.centroid.coords][0]
                plt.text(num_pos_x, num_pos_y, s=group_of_shapes[0], 
                        **text_kwargs)

    ret_dict['plotted_shapes'] = translated_and_scaled.copy()
    ret_dict['section_image'] = rect

    return ret_dict


def read_imagescope_xml_annotations(xml_path: Path | str):
    """ Function that reads in annotations from an exported ImageScope annotation xml file.

    Returns a dictionary of dictionaries of shapes and areas.
    First set of keys are the annotation layer names. Second set of keys are for the shapes and areas. 
    """
    # convert string path into path path
    if type(xml_path) == str:
        xml_path = Path(xml_path)

    total_data_dict = dict() # dictionary that holds all the shapes, first set of keys are the annotation file names, second set of keys are the shorthand annotation layer names, 
    # final object is a dataframe holding the shapes, 'my' areas (area retrieved from the shapes, converted with mpp), and imagescope areas (straight from the xml file) 
    root = ET.parse(xml_path).getroot() # xml stuff
    mpp = float(root.get('MicronsPerPixel'))
    annotations = root.findall('Annotation')
    for annotation_layer in annotations: # for each annotation layer in the file
        annotation_layer_name = annotation_layer.get('Name')
        # get regions from annotation layer
        regions = annotation_layer.find('Regions').findall('Region')
        shape_dict = {} # holds geometries
        my_area_dict = {} # holds areas generated from the shapes and mpp
        imagescope_area_dict = {} # holds areas from imagescope
        for i, region in enumerate(regions):
            vertices = []
            for vertex in region.find('Vertices').findall('Vertex'):
                x, y = int(vertex.get('X')), int(vertex.get('Y'))
                vertices.append((x,y))
            shape = shapely.Polygon(np.array(vertices))
            shape_dict[i] = shape
            my_area_dict[i] = mpp**2 * shape.area # converting areas from pixels squared to micrometers squared
            imagescope_area_dict[i] = region.get('AreaMicrons')
        regions_df = pd.DataFrame([shape_dict, my_area_dict, imagescope_area_dict], index=['shape', 'python_area(µm²)', 'imagescope_area(µm²)']).T
        regions_df = dict(regions_df)
        total_data_dict[annotation_layer_name] = regions_df
    return total_data_dict


## commenting out for now
# def make_shapes_lmdable_xml(image_location: str | Path,
#                         collection_regions: MultiPolygon,
#                         minimum_micrometers_squared: int = 10000,
#                         maximum_micrometers_squared: int = 600*600,
#                         minimum_micrometers_between_shapes: int = 60,
#                         n_slices_to_check: int = 5,
#                         verbose: int = True):
#     """ Converts HALO regions into shapely shapes that can be exported to XML to cut on an LMD scope.

#     Args:
#         image_location (:class:`str` | :class:`pathlib.Path`):
#             The image file location (specifically, the same file that HALO is using)
#         collection_regions (:class:`MultiPolygon`):
#             The regions to be prepared for LMD collection
#         minimum_micrometers_squared (:class:`int`):
#             Minimum size of an object to cut out, smaller sized objects are removed
#         maximum_micrometers_squared (:class:`int`):
#             Maximum size of an object to cut out, larger sized objects are sliced 
#             into smaller ones
#         minimum_micrometers_between_shapes (:class:`int`):
#             The minimum distance between shapes. Anything too close will be shaved,
#             and larger object slices will be separated with this distance
#         n_slices_to_check (:class:`int`):
#             The number of horizontal and vertical slice positions to check when
#             deciding the most optimal
#         verbose (:class:`int`):
#             Whether or not to create progress bars and print status updates
    
#     Returns:
#         :class:`shapely.MultiPolygon`:
#             MultiPolygon filled with shapes that can be converted to XML for
#             cutting on an LMD scope
#     """
#     if type(image_location) == str:
#         image_location = Path(image_location)
#     assert image_location.exists(), f"Path {image_location=} does not exist!"
#     # get mpp from image
#     if verbose:
#         print("reading in image with WSIReader (getting mpp)")
#     wsi = WSIReader.open(image_location)
#     mpp = wsi.info.as_dict()['mpp']
#     assert mpp[0] == mpp[1], "x y mpp different, didn't account for this..."
#     mpp = mpp[0]

#     # get minimum shape areas in pixels
#     minimum_shape_area = minimum_micrometers_squared/(mpp**2)
#     maximum_shape_area = maximum_micrometers_squared/(mpp**2)

#     # get slice/cut size
#     cut_size = minimum_micrometers_between_shapes/2/mpp

#     # get all the objects into a mask
#     #polygons = []
#     #cutouts = []
#     # loop over polygons
#     #for polygon_info in collection_regions:
#         # pre_polygon = np.array(json.loads(polygon_info['geometry'])['coordinates'])
#         # pre_polygon = np.vstack([pre_polygon, pre_polygon[0]]) # duplicate first point as the last point, making sure it's a closed object

#         # polygon = shapely.Polygon(pre_polygon)
#         # if not polygon_info['isExclusionRegion']:
#         #     polygons.append(polygon)
#         # else:
#         #     cutouts.append(polygon)


#     polygons_all = list(collection_regions.geoms)
#     polygons = []
#     cutouts = []
    
#     for poly in polygons_all:
#         # Exterior polygon
#         polygons.append(Polygon(poly.exterior))
    
#         # Interior (hole) polygons
#         for interior in poly.interiors:
#             # Make sure the hole is a valid polygon (must be closed ring with area)
#             ring = LinearRing(interior)
#             if ring.is_valid and not ring.is_empty:
#                 cutouts.append(Polygon(ring))
    
#     # make connectors to cutouts
#     polys = []
#     for polygon in tqdm(polygons, desc="connectors", disable=not verbose):
#         for cutout in cutouts:
#             if polygon != cutout and polygon.intersects(cutout):
#                 # fix polygon (cutting out region, but also connecting it to non-shape)
#                 # figure out of polygon is a multipolygon
#                 if type(polygon) != shapely.Polygon:
#                     poly_list = []
#                     for actual_polygon in tqdm(list(polygon.geoms), desc="connectors inner", disable=not verbose):
#                         if type(actual_polygon) == shapely.LineString or type(actual_polygon) == shapely.Point:
#                             # skip these
#                             continue
#                         # add a connector to the polygon
#                         connector = shapely.LineString([[x for x in nearest_points(actual_polygon.exterior, cutout.centroid)][0], [x for x in cutout.centroid.coords][0]]).buffer(60/mpp, cap_style=3, join_style=3)
#                         actual_polygon = actual_polygon.difference(unary_union([cutout, connector, nearest_points(actual_polygon.exterior, cutout.centroid)[0].buffer(75/mpp), cutout.centroid.buffer(75/mpp)]))
#                         poly_list.append(actual_polygon)
#                     polygon = unary_union(poly_list)
#                 else:
#                     # add a connector to the polygon
#                     connector = shapely.LineString([[x for x in nearest_points(polygon.exterior, cutout.centroid)][0], [x for x in cutout.centroid.coords][0]]).buffer(60/mpp, cap_style=3, join_style=3)
#                     polygon = polygon.difference(unary_union([cutout, connector, nearest_points(polygon.exterior, cutout.centroid)[0].buffer(75/mpp), cutout.centroid.buffer(75/mpp)]))
#         polys.append(polygon)        
        
#     # fix multipolygons (in case there is multipolygon inception?)
#     if verbose:
#         print('fixing multipolygons')
#     polys = list_of_shapes_to_polygons(polys, minimum_shape_area=minimum_shape_area)

#     # simplify and smooth objects, splitting up objects that have tight regions between them
#     simplified_polys = []
#     for poly in tqdm(polys, desc="simplifying", disable=not verbose):
#         #new_poly = hsf.simplify_and_smooth(shapely.make_valid(poly), thinness_distance=cut_size)
#         new_poly = shapely.simplify(poly, 5)
#         if type(new_poly) == shapely.MultiPolygon:
#             for polyi in list(new_poly.geoms):
#                 simplified_polys.append(polyi)
#         else:
#             simplified_polys.append(new_poly)

#     # fix multipolygons (in case there is multipolygon inception?)
#     if verbose:
#         print('fixing multipolygons')
#     simplified_polys = list_of_shapes_to_polygons(simplified_polys, minimum_shape_area=minimum_shape_area)

#     # slice up polygons that are too large
#     if verbose:
#         print('slicing')
#     # kwargs: mitre_distance=10, thinness_distance=25, first_simplify=5, second_simplify=3
#     n_too_large = 0
#     for simplified_poly in simplified_polys:
#         if simplified_poly.area > maximum_shape_area:
#             n_too_large = n_too_large + 1
#     while n_too_large > 0:
#         n_too_large = 0
#         for simplified_poly in tqdm(simplified_polys.copy(), disable=not verbose):
#             if simplified_poly.area > maximum_shape_area:
#                 n_too_large = n_too_large + 1
#                 simplified_polys.remove(simplified_poly)
#                 simplified_poly = make_valid(simplified_poly) # make the polygon valid
#                 # double checking for multipolygons/geometrycollections
#                 temp_polygon_list = convert_to_polygon_list(simplified_poly)
#                 sliced_polys = []
#                 for simplified_poly_temp in temp_polygon_list:
#                     sliced_poly = find_best_slice(simplified_poly_temp, cut_size=cut_size, tests=n_slices_to_check, minimum_shape_area=minimum_shape_area, maximum_shape_area=maximum_shape_area, 
#                                                 thinness_distance=cut_size, mitre_distance=0, pbar=verbose)
#                     sliced_polys.append(make_valid(sliced_poly))
#                 # add polygons back to simplified_polys
#                 simplified_polys.extend(sliced_polys)
#         if verbose:
#             print(f"{n_too_large=} remaining")


#     # fix multipolygons
#     if verbose:
#         print('fixing multipolygons')
#     simplified_polys = list_of_shapes_to_polygons(simplified_polys, minimum_shape_area=minimum_shape_area)


#     # removing close calls
#     if verbose:
#         print('removing `close calls`')
#     simplified_polys = remove_close_calls(shapely.make_valid(shapely.MultiPolygon(simplified_polys)), cut_size=cut_size)
#     simplified_polys = convert_to_polygon_list(simplified_polys, verbose=verbose)

#     # scrub cycle
#     if verbose:
#         print('shape scrub cycle')
#     scrubbed_polys = []
#     for poly in tqdm(simplified_polys, disable=not verbose):
#         intermediately_scrubbed = []
#         scrub_cycle(poly, cut_size=cut_size, final_shapes=intermediately_scrubbed)
#         scrubbed_polys.extend(intermediately_scrubbed.copy())


#     simplified_polys = scrubbed_polys.copy()

#     # remove shapes that have been added that are too small
#     for simplified_poly in tqdm(simplified_polys.copy(), desc="removing small shapes", disable=not verbose):
#         if simplified_poly.area < minimum_shape_area:
#             simplified_polys.remove(simplified_poly)

#     # simplify and smooth objects, splitting up objects that have tight regions between them
#     simplified_polys_again = []
#     for poly in tqdm(simplified_polys, desc="simplifying", disable=not verbose):
#         #new_poly = hsf.simplify_and_smooth(shapely.make_valid(poly), thinness_distance=cut_size)
#         new_poly = shapely.simplify(poly, 5)
#         if type(new_poly) == shapely.MultiPolygon:
#             for polyi in list(new_poly.geoms):
#                 simplified_polys_again.append(polyi)
#         else:
#             simplified_polys_again.append(new_poly)

#     # fix multipolygons (in case there is multipolygon inception?)
#     if verbose:
#         print('fixing multipolygons')
#     simplified_polys_again = list_of_shapes_to_polygons(simplified_polys_again, minimum_shape_area=minimum_shape_area)

#     # make a valid multipolygon
#     final_shapely_multipolygon = shapely.make_valid(shapely.MultiPolygon(simplified_polys_again))
    
#     return final_shapely_multipolygon


def parse_annotation_file(layer, file):    
    # Parse the XML
    tree = ET.parse(file)  # Replace with your file
    root = tree.getroot()
    
    # Storage for coordinate lists
    tumor_roa_0_coords = []
    tumor_roa_1_coords = []
    
    # Extract coordinates based on Annotation Name and NegativeROA
    for annotation in root.findall(".//Annotation[@Name='{}']".format(layer)):
        for region in annotation.findall(".//Region"):
            roa = region.attrib.get('NegativeROA')
            vertices = region.find('Vertices')
            if vertices is not None:
                coords = [(int(v.attrib['X']), int(v.attrib['Y'])) for v in vertices.findall('V')]
    
                if len(coords) >= 3:  # Polygon requires at least 3 points
                    if roa == "0":
                        tumor_roa_0_coords.append(coords)
                    elif roa == "1":
                        tumor_roa_1_coords.append(coords)
    
    # Convert to MultiPolygons
    tumor_roa_0_multipolygon = MultiPolygon([Polygon(coords) for coords in tumor_roa_0_coords])
    tumor_roa_1_multipolygon = MultiPolygon([Polygon(coords) for coords in tumor_roa_1_coords])
    
    shapes = tumor_roa_0_multipolygon.difference(tumor_roa_1_multipolygon)
    if shapes.area == 0: # this could be done earlier/better
        # print(f"No annotation layer found named '{layer}'")
        raise FileNotFoundError(f"No annotation layer found named '{layer}' in the file '{file}'")
    if isinstance(shapes, MultiPolygon):
        return shapes
    elif isinstance(shapes, Polygon):
        return MultiPolygon([shapes])


def create_scope_xml(xml_multipolygon: shapely.MultiPolygon | gpd.GeoSeries, 
                     calib_layer: shapely.MultiPolygon, 
                     xml_file_name: str, 
                     verbose: bool = False, 
                     rows_cols: tuple = None):
    """ Generates an XML file to be imported onto the
    LMD scope for cutting.
    """

    #CAPTURE COORDINATES FOR CALIBRATION MARKS
    calib_marks = []
    for x in calib_layer.geoms:
        dic ={"x" : x.exterior.coords[0][0], "y": x.exterior.coords[0][1]}
        calib_marks.append(dic)
        
        
    # sort the calibration marks
    ## first is the top left, then top right, then bottom right
    calib_mark_df = pd.DataFrame(calib_marks)
    assert not calib_mark_df.empty, "No calibration marks in annotation layer..."
    assert calib_mark_df.shape == (3, 2), f"Unexpected number of calibration marks (expects 3, got {calib_mark_df.shape[0]})"
    # top left
    top_left = calib_mark_df.sort_values('x', ascending=True).iloc[0, :].to_dict()
    top_left = (top_left['x'], top_left['y'])
    # top right
    ## drop top left
    calib_mark_df_dropped = calib_mark_df.sort_values('x', ascending=True).drop([0], axis=0).reset_index(drop=True)
    calib_mark_df_dropped = calib_mark_df_dropped.sort_values('y', ascending=True)
    ## 0 should be top right, 1 should be bottom right
    top_right = calib_mark_df_dropped.iloc[0, :].to_dict()
    top_right = (top_right['x'], top_right['y'])
    # bottom right
    bottom_right = calib_mark_df_dropped.iloc[1, :].to_dict()
    bottom_right = (bottom_right['x'], bottom_right['y'])

    # OPEN XML FILE AND ADD CALIBRATION COORDINATES TO THE TOP OF THE PAGE
    output = open(xml_file_name, 'w')
    output.write("<ImageData>")
    output.write("\n")
    output.write("<GlobalCoordinates>1</GlobalCoordinates>")
    output.write("\n")
    output.write("<X_CalibrationPoint_1>{}</X_CalibrationPoint_1>".format(int(top_left[0])))
    output.write("\n")
    output.write("<Y_CalibrationPoint_1>{}</Y_CalibrationPoint_1>".format(int(top_left[1])))
    output.write("\n")
    output.write("<X_CalibrationPoint_2>{}</X_CalibrationPoint_2>".format(int(top_right[0])))
    output.write("\n")
    output.write("<Y_CalibrationPoint_2>{}</Y_CalibrationPoint_2>".format(int(top_right[1])))
    output.write("\n")
    output.write("<X_CalibrationPoint_3>{}</X_CalibrationPoint_3>".format(int(bottom_right[0])))
    output.write("\n")
    output.write("<Y_CalibrationPoint_3>{}</Y_CalibrationPoint_3>".format(int(bottom_right[1])))
    output.write("\n")

    if type(xml_multipolygon) == gpd.GeoSeries:
        assert rows_cols != None, 'If providing a GeoSeries (spatially resolved regions), make sure to provide the number of rows and columns that can be collected in.'
        n_rows, n_cols = rows_cols
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if n_rows > len(alphabet):
            raise NotImplementedError('More rows than letters of the alphabet. Someone needs to implement this (check what scope requires)')
        collection_tube_locations = []
        for n_row in range(n_rows):
            for n_col in range(1, n_cols+1):
                index = f"{alphabet[n_row]}{n_col}"
                collection_tube_locations.append(index)
        assert len(collection_tube_locations) >= xml_multipolygon.shape[0], f"More spatial regions ({xml_multipolygon.shape[0]}) than tubes to collect in ({len(collection_tube_locations)})!"
        idx_to_collection_loc = {}
        for i, idx in enumerate(xml_multipolygon.index):
            idx_to_collection_loc[idx] = collection_tube_locations[i]

        # get the number of shapes (shape count)
        scount = xml_multipolygon.count_geometries().sum()

        #ADD OVERALL SHAPE COUNT TO XML IMPORT
        output.write("<ShapeCount>{}</ShapeCount>".format(scount))
        output.write("\n")

        current_shape = 0
        for idx in xml_multipolygon.index:
            multipoly = xml_multipolygon[idx]
            # convert multipolygon to polygon list
            polygons = convert_to_polygon_list(multipoly, verbose=verbose)

            # CALCULATE POINT COUNTS FOR EACH SHAPE
            for s_num, shape in enumerate(polygons):
                current_shape += 1 # shape number has 1-based indexing, so doing this at the front
                pre_polygon = shape.exterior.coords # updated to exterior instead of boundary... boundary was providing multilinestrings for some reason
                
                p_count = len(pre_polygon)
                
                # INSERT SHAPE NUMBER AND POINT COUNTS INTO XML FILE
                output.write(f"<Shape_{current_shape}>")
                output.write("\n")
                output.write(f"<PointCount>{int(p_count)}</PointCount>")
                output.write("\n")
                # CapID for where to collect the sample
                output.write(f"<CapID>{idx_to_collection_loc[idx]}</CapID>")
                output.write("\n")
                
                # ENTER POINT NUMBER AND X / Y COORDINATES FOR EACH POINT INTO XML FILE
                for i, coord in enumerate(pre_polygon):
                    i = i + 1 # 1-based indexing
                    output.write(f"<X_{i}>{int(coord[0])}</X_{i}>")
                    output.write("\n")
                    output.write(f"<Y_{i}>{int(coord[1])}</Y_{i}>")
                    output.write("\n")    
                
                #CLOSE SHAPE 
                output.write(f"</Shape_{current_shape}>")
                output.write("\n")
        #CLOSE COORDINATE ENTRY 
        output.write("</ImageData>")

        #CLOSE XML DOCUMENT
        output.close()
        return idx_to_collection_loc

    else:
        # convert multipolygon to polygon list
        polygons = convert_to_polygon_list(xml_multipolygon, verbose=verbose)

        # get the number of shapes (shape count)
        scount = len(polygons)

        #ADD OVERALL SHAPE COUNT TO XML IMPORT
        output.write("<ShapeCount>{}</ShapeCount>".format(scount))
        output.write("\n")


        # CALCULATE POINT COUNTS FOR EACH SHAPE
        for s_num, shape in enumerate(polygons):
            num = s_num + 1 # shape number has 1-based indexing
            pre_polygon = shape.exterior.coords # updated to exterior instead of boundary... boundary was providing multilinestrings for some reason
            
            p_count = len(pre_polygon)
            
            # INSERT SHAPE NUMBER AND POINT COUNTS INTO XML FILE
            output.write(f"<Shape_{num}>")
            output.write("\n")
            output.write(f"<PointCount>{int(p_count)}</PointCount>")
            output.write("\n")
            # CapID for where to collect the sample
            
            # ENTER POINT NUMBER AND X / Y COORDINATES FOR EACH POINT INTO XML FILE
            for i, coord in enumerate(pre_polygon):
                i = i + 1 # 1-based indexing
                output.write(f"<X_{i}>{int(coord[0])}</X_{i}>")
                output.write("\n")
                output.write(f"<Y_{i}>{int(coord[1])}</Y_{i}>")
                output.write("\n")    
            
            #CLOSE SHAPE 
            output.write(f"</Shape_{num}>")
            output.write("\n")
        #CLOSE COORDINATE ENTRY 
        output.write("</ImageData>")

        #CLOSE XML DOCUMENT
        output.close()
        return 1   


# getting errors in the notebook when trying to use this function - TODO
def plot_multipolygons_on_wsi(multipolygon_dict, tissue_shape: Polygon, wsi_path: Path):
    
    # Open the whole slide image
    slide = WSIReader.open(str(wsi_path))

    # Get bounds of tissue shape
    minx, miny, maxx, maxy = map(int, tissue_shape.bounds)
    width = maxx - minx
    height = maxy - miny

    # Read region from WSI
    region = slide.read_region((minx, miny), 0, (width, height))
    
    # Convert to PIL Image if necessary
    if isinstance(region, np.ndarray):
        cropped_image = Image.fromarray(region)
    else:
        cropped_image = region

    # Ensure RGB
    if hasattr(cropped_image, "convert"):
        cropped_image = cropped_image.convert("RGB")

    # Plotting setup
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.imshow(cropped_image, extent=(0, width, height, 0))

    # Get sorted list of keys
    sorted_keys = sorted(multipolygon_dict.keys())

    # Calculate centroids of each multipolygon to use for sorting
    centroids = {key: multipolygon_dict[key].centroid for key in sorted_keys}
    centroid_coords = np.array([[c.x, c.y] for c in centroids.values()])

    # Compute distance matrix between centroids
    distances = pairwise_distances(centroid_coords)
    np.fill_diagonal(distances, np.inf)  # so self-distance isn't zero

    # Assign colors trying to maximize distance between neighbors
    num_keys = len(sorted_keys)
    base_cmap = cm.get_cmap('gist_ncar', num_keys)
    color_list = [base_cmap(i / num_keys) for i in range(num_keys)]
    random.shuffle(color_list)  # Shuffle to reduce adjacent similarity

    # Assign shuffled colors to keys
    color_map = {key: color_list[i] for i, key in enumerate(sorted_keys)}

    # Track label positions to avoid overlap
    placed_boxes = []

    def is_overlapping(new_box, existing_boxes, buffer=30):
        for box in existing_boxes:
            if (abs(new_box[0] - box[0]) < buffer) and (abs(new_box[1] - box[1]) < buffer):
                return True
        return False

    for key in sorted_keys:
        multipolygon = multipolygon_dict[key]

        if not isinstance(multipolygon, MultiPolygon):
            raise ValueError(f"Value at key {key} is not a MultiPolygon")

        patches = []
        for poly in multipolygon.geoms:
            shifted_poly = translate(poly, xoff=-minx, yoff=-miny)
            coords = list(shifted_poly.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))

        color = color_map[key]

        patch_collection = PatchCollection(
            patches,
            facecolor=color,
            edgecolor='black',
            alpha=0.8,
            linewidth=2
        )
        ax.add_collection(patch_collection)

        # Centroid and offset for label
        centroid = multipolygon.centroid
        centroid_x = centroid.x - minx
        centroid_y = centroid.y - miny
        offset = 80
        label_x = centroid_x + offset
        label_y = centroid_y - offset

        # Avoid label overlap
        attempts = 0
        while is_overlapping((label_x, label_y), placed_boxes) and attempts < 20:
            label_x += 40
            label_y += 40
            attempts += 1
        placed_boxes.append((label_x, label_y))

        ax.annotate(
            str(key),
            xy=(centroid_x, centroid_y),
            xytext=(label_x, label_y),
            textcoords='data',
            fontsize=21,
            fontweight='bold',
            color='white',
            ha='center', va='center',
            bbox=dict(facecolor=color, alpha=0.5, edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color=color, linewidth=2)
        )

    ax.set_title("Spatial Clusters Overlay on Cropped WSI", fontsize=18)
    ax.set_aspect('equal')
    ax.axis('off')

    # Save high-resolution image
    plt.savefig("unbiased_XML_files/output_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def rebalance_multipolygons_by_area(
    multipolygon: MultiPolygon,
    cluster_count: int,
    tolerance: float = 0.05,
    max_iterations: int = 1000):
    """ Creates groups of shapes (n = cluster_count) of similar sizes, determining a group for every shape provided.
    """
    # Extract individual polygons
    polygons = list(multipolygon.geoms)

    if len(polygons) < cluster_count:
        raise ValueError("Number of polygons is less than number of clusters.")

    gdf = gpd.GeoSeries(polygons)
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf])

    # Initial clustering
    clustering = AgglomerativeClustering(n_clusters=cluster_count, linkage='ward')
    initial_labels = clustering.fit_predict(centroids)

    # Organize polygons into clusters
    cluster_groups = {i: [] for i in range(cluster_count)}
    labels = {}

    for idx, label in enumerate(initial_labels):
        cluster_groups[label].append(idx)
        labels[idx] = label

    def cluster_area(indices):
        return sum(gdf[i].area for i in indices)

    def cluster_centroid(indices):
        """Compute weighted centroid of a cluster."""
        if not indices:
            return Point(0, 0)
        weighted_x = sum(gdf[i].centroid.x * gdf[i].area for i in indices)
        weighted_y = sum(gdf[i].centroid.y * gdf[i].area for i in indices)
        total_area = sum(gdf[i].area for i in indices)
        return Point(weighted_x / total_area, weighted_y / total_area)

    # -- KEEP EXISTING HELPERS --
    def find_neighboring_clusters():
        neighbors = {i: set() for i in range(cluster_count)}
        spatial_index = gdf.sindex

        poly_to_cluster = {
            idx: cluster_id
            for cluster_id, poly_indices in cluster_groups.items()
            for idx in poly_indices
        }

        for cluster_id, poly_indices in cluster_groups.items():
            for idx in poly_indices:
                poly = gdf.geometry.iloc[idx]
                buffered_bounds = poly.buffer(500).bounds
                candidate_idxs = list(spatial_index.intersection(buffered_bounds))
                for n_idx in candidate_idxs:
                    if n_idx == idx or n_idx not in poly_to_cluster:
                        continue
                    neighbor_cluster = poly_to_cluster[n_idx]
                    if neighbor_cluster == cluster_id:
                        continue
                    neighbor_poly = gdf.geometry.iloc[n_idx]
                    dist = poly.distance(neighbor_poly)
                    if poly.touches(neighbor_poly) or dist < 500:
                        neighbors[cluster_id].add(neighbor_cluster)
        return neighbors

    def get_boundary_polygons(cluster_id):
        spatial_index = gdf.sindex
        boundary_polys = []
        for idx in cluster_groups[cluster_id]:
            poly = gdf.iloc[idx]
            buffered_bounds = poly.buffer(500).bounds
            for n_idx in spatial_index.intersection(buffered_bounds):
                if n_idx == idx:
                    continue
                neighbor_poly = gdf.iloc[n_idx]
                neighbor_cluster = labels[n_idx]
                if neighbor_cluster != cluster_id:
                    if poly.touches(neighbor_poly) or poly.distance(neighbor_poly) < 500:
                        boundary_polys.append(idx)
                        break
        return boundary_polys

    def polygon_neighbors(poly_idx, target_cluster, gdf, cluster_groups, buffer_tolerance=100.0):
        poly = gdf.loc[poly_idx]
        if poly is None or poly.is_empty:
            return []
        buffered_poly = poly.buffer(buffer_tolerance)
        target_indices = cluster_groups.get(target_cluster, [])
        if not target_indices:
            return []
        target_polys = gdf.loc[target_indices]
        sindex = target_polys.sindex
        candidate_idxs = list(sindex.intersection(buffered_poly.bounds))
        possible_matches = target_polys.iloc[candidate_idxs]
        neighbors = [idx for idx, geom in possible_matches.items() if buffered_poly.intersects(geom)]
        return neighbors

    # ---- MAIN BALANCING LOOP ----
    for iteration in range(max_iterations):
        changed = False
        cluster_areas = {cid: cluster_area(indices) for cid, indices in cluster_groups.items()}
        cluster_centroids = {cid: cluster_centroid(indices) for cid, indices in cluster_groups.items()}
        avg_area = np.mean(list(cluster_areas.values()))

        # Refresh labels
        for cid, indices in cluster_groups.items():
            for idx in indices:
                labels[idx] = cid

        neighbors = find_neighboring_clusters()

        # Process largest clusters first
        sorted_clusters = sorted(cluster_areas.items(), key=lambda x: x[1], reverse=True)

        for cid, area in sorted_clusters:
            if area <= avg_area * (1 + tolerance):
                continue

            boundary_polys = get_boundary_polygons(cid)
            if not boundary_polys:
                continue

            cluster_center = cluster_centroids[cid]

            # Rank boundary polygons by distance from cluster center (farther first)
            sorted_boundary = sorted(
                boundary_polys,
                key=lambda i: gdf[i].centroid.distance(cluster_center),
                reverse=True
            )

            for poly_idx in sorted_boundary:
                poly_center = gdf[poly_idx].centroid
                # Try to move polygon to the nearest underweight neighboring cluster
                candidate_neighbors = [
                    (neigh_cid, cluster_centroids[neigh_cid].distance(poly_center))
                    for neigh_cid in neighbors[cid]
                    if cluster_areas[neigh_cid] < avg_area * (1 - tolerance)
                ]

                if not candidate_neighbors:
                    continue

                # Prefer neighbor cluster that is closest spatially to the polygon
                candidate_neighbors.sort(key=lambda x: x[1])
                best_neighbor = candidate_neighbors[0][0]

                if polygon_neighbors(poly_idx, best_neighbor, gdf, cluster_groups):
                    # Transfer polygon
                    cluster_groups[cid].remove(poly_idx)
                    cluster_groups[best_neighbor].append(poly_idx)
                    labels[poly_idx] = best_neighbor
                    changed = True
                    break  # Stop after one successful transfer

            if changed:
                break

        if not changed:
            break

    # Compile result
    result = {
        cid: MultiPolygon([gdf[i] for i in indices])
        for cid, indices in cluster_groups.items()
    }

    return result
    
def partition_collections_to_xml(MultiPolygon_Dict: dict, calib: shapely.MultiPolygon, rows: int = 2, columns: int = 7):
    """ Generates XML files set up for multiple collections iteratively with the largest 'chunk' size.

    Small wrapper for 'create_scope_xml'
    """
    
    # Sort the keys
    sorted_keys = sorted(MultiPolygon_Dict.keys())
    # Iterate through keys in chunks
    chunk_size = rows * columns # chunk size so multiple files can be generated with the max number of collected samples per file
    for i in range(0, len(sorted_keys), chunk_size):
        chunk_keys = sorted_keys[i:i + chunk_size]
        chunk = {k: MultiPolygon_Dict[k] for k in chunk_keys}
        geoseries_mapper = create_scope_xml(
                                        xml_multipolygon = gpd.GeoSeries(chunk), 
                                        xml_file_name = Path('unbiased_XML_files/Whole_Tissue_Spatial_{}.xml'.format(i)),
                                        calib_layer = calib,
                                        rows_cols = (rows,columns))