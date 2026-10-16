import os
import pandas as pd
import pickle
from shapely.geometry import box
import numpy as np


def convert_to_bottomleft(XMin, XMax, YMin, YMax):
    x, y, w, h = XMin, YMin, XMax - XMin, YMax - YMin
    return (x, y, w, h)


def convert_to_boundary(x, y, w, h):
    XMin, XMax, YMin, YMax = x, x + w, y, y + h
    return (XMin, XMax, YMin, YMax)


# Assign each pixel to a tile
def find_tile_with_centroid(px, py, tile_idx):
    candidates = list(tile_idx.intersection((px, py, px, py)))  # Query point
    return int(candidates[0]) if candidates else None


def find_tile_with_bbox(x, y, h, w, tile_idx, coverage=0.5):
    XMin, XMax, YMin, YMax = convert_to_boundary(x, y, w, h)
    cell = box(XMin, YMin, XMax, YMax)
    # print(f"cell area = {cell.area}")
    intersections = []
    candidates = list(tile_idx.intersection((XMin, YMin, XMax, YMax), objects=True))
    if candidates:
        for candidate in candidates:
            # print(f"    candidate id = {candidate.id}, bbox = {candidate.bbox}")
            tile = box(
                candidate.bbox[0],
                candidate.bbox[1],
                candidate.bbox[2],
                candidate.bbox[3],
            )
            overlap = cell.intersection(tile)
            if not overlap.is_empty:
                # print(f"        overlap area = {overlap.area}")
                intersections.append(overlap.area / cell.area)
                # print(f"        intersection = {overlap.area / cell.area}")
        # print(f"    all intersections = {intersections}")
        id = [
            i
            for i, intersection in enumerate(intersections)
            if intersection >= coverage and intersection == max(intersections)
        ]
        return int(candidates[id[0]].id) if id else None
    else:
        return None


def make_infodocs(dst, name, level_from, param_tiles):
    infodict = {}
    infos = []

    for i, para in enumerate(param_tiles):
        infos.append(
            {
                "ID": i,
                "x": para[0],
                "y": para[1],
                "w": para[2],
                "h": para[3],
                "level": level_from,
            }
        )
        infodict[i] = {
            "x": para[0],
            "y": para[1],
            "w": para[2],
            "h": para[3],
            "level": level_from,
        }
    df = pd.DataFrame(infos)
    # Saving
    df.to_csv(
        os.path.join(dst, name + "_infos.csv"),
        index=False,
    )
    with open(os.path.join(dst, name + "_infodict.pickle"), "wb") as f:
        pickle.dump(infodict, f)


def hard_vote(group):
    return group.dropna().mode().iloc[0] if not group.dropna().empty else np.nan


def safe_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan
