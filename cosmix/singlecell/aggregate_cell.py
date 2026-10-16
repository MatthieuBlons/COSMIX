from typing import Optional
import os
import pandas as pd
from rtree import index
from slide.tile import read_h5_coords
import numpy as np
from singlecell.aggregate.utils import (
    find_tile_with_bbox,
    find_tile_with_centroid,
    safe_mode,
)
from singlecell.aggregate.keys import KEYS_ORION
from slide.draw import map_cell_feat, visualise_tile_feat
import matplotlib.pyplot as plt
import yaml


class GroupCellFeat:
    def __init__(
        self,
        slide,
        coords_path: str,
        cells_path: str,
        method: str = "centroid",
        coverage: Optional[float] = 0.9,
        keys: dict = KEYS_ORION,
        dst: Optional[str] = None,
    ) -> None:
        self.slide = slide
        self.cells_path = cells_path
        self.group_keys = keys
        self.cell_table = self.read_cell_data()
        self.cell_keys = self.cell_table.keys().to_list()
        self.coords_path = coords_path
        self.coords_meta, self.coords = self.read_coords()
        # check if target mag and cell analyse mag match
        # raise error if not
        # make a lazy grouping
        # group cells and
        self.group_cells(how=method, coverage=coverage)
        self.group_table = self.make_group_table()

        # save cells table and group table if dst is provided
        if dst:
            self.cell_dst, self.group_dst = self.save_tables(dst)

    def read_cell_data(self):
        _, ext = os.path.splitext(os.path.basename(self.cells_path))
        if ext == ".csv":
            return pd.read_csv(self.cells_path, index_col=0)

    def read_coords(self):
        _, ext = os.path.splitext(os.path.basename(self.coords_path))
        if ext == ".h5":
            attrs, coords = read_h5_coords(self.coords_path)
            return attrs, coords

    def group_cells(self, how="centroid", coverage=1):
        # insert coords path
        # as basename only
        # just after wsi_name
        self.cell_table.insert(
            loc=self.cell_table.shape[1],
            column="coords_path",
            value=os.path.basename(self.coords_path),
        )
        # Create an R-tree index for group coords
        idx_group = index.Index()
        for i, (x, y, w, h) in enumerate(self.coords):
            idx_group.insert(i, (x, y, x + w, y + h))
        # Attribute each cell to a group
        if how == "bbox":
            self.cell_table.insert(
                loc=self.cell_table.shape[1],
                column="idx_tile",
                value=self.cell_table.apply(
                    lambda row: find_tile_with_bbox(
                        row["x"],
                        row["y"],
                        row["w"],
                        row["h"],
                        idx_group,
                        coverage=coverage,
                    ),
                    axis=1,
                ),
            )
        elif how == "centroid":
            self.cell_table.insert(
                loc=self.cell_table.shape[1],
                column="idx_tile",
                value=self.cell_table.apply(
                    lambda row: find_tile_with_centroid(
                        row["x"],
                        row["y"],
                        idx_group,
                    ),
                    axis=1,
                ),
            )
            loc = self.cell_table.columns.get_loc("y")
            self.cell_table.insert(loc=loc + 1, column="w", value=0)
            self.cell_table.insert(loc=loc + 2, column="h", value=0)

    def make_group_table(self):
        group_table = self.cell_table.copy()
        cell_pos_to_drop = [
            "cell_id",
            "panel_id",
            "cx",
            "cy",
            "x",
            "y",
            "w",
            "h",
            "in_tile",
        ]
        group_table.drop(
            labels=cell_pos_to_drop,
            axis=1,
            errors="ignore",
            inplace=True,
        )
        aggregation = {
            **{
                k: "sum"
                for k in [
                    k
                    for k in self.group_keys["one-hot"]
                    if k in list(group_table.columns.values)
                ]
            },
            **{
                k: "mean"
                for k in [
                    k
                    for k in self.group_keys["intensity"]
                    if k in list(group_table.columns.values)
                ]
            },
            **{
                k: safe_mode
                for k in [
                    k
                    for k in list(group_table.columns.values)
                    if k
                    not in self.group_keys["intensity"] + self.group_keys["one-hot"]
                ]
            },
        }

        group_table = (
            group_table.groupby("idx_tile", dropna=True).agg(aggregation)
        ).set_index("idx_tile")

        tile_pos_to_add = pd.DataFrame(
            self.coords, columns=["x", "y", "w", "h"]
        ).rename_axis("idx_tile")

        group_table = pd.merge(
            group_table, tile_pos_to_add, on="idx_tile", how="left"
        ).sort_index()

        nb_feat = len(
            [k for k in self.group_keys["one-hot"] + self.group_keys["intensity"]]
        )
        # filter nan
        group_table.dropna(
            thresh=group_table.shape[1] - nb_feat + 1,
            axis=0,
            inplace=True,
            ignore_index=False,
        )
        # filter zero?

        cells_counts = self.cell_table["idx_tile"].value_counts()
        group_table.insert(loc=group_table.shape[1], column="cells", value=cells_counts)

        # convert index to int
        group_table.index.astype(int)

        # order columns
        group_table = group_table[
            [
                k
                for k in list(group_table.columns.values)
                if k not in self.group_keys["intensity"] + self.group_keys["one-hot"]
            ]
            + self.group_keys["intensity"]
            + self.group_keys["one-hot"]
        ]

        return group_table

    def draw_cell(
        self,
        feat: str,
        size: tuple = (1024, 1024),
        discrete: bool = True,
        ax=None,
        show=False,
        save: str | None = None,
    ):

        thumbnail = self.get_thumbnail(size, numpy=True)
        thumbnail_height, thumbnail_width, _ = thumbnail.shape
        downsample_factor = max(
            self.slide.level_dimensions[self.coords_meta["level"]][0] / thumbnail_width,
            self.slide.level_dimensions[self.coords_meta["level"]][1]
            / thumbnail_height,
        )

        level_to_view, _, _ = self.slide.get_best_level_for_downsample(
            downsample_factor
        )

        map_cell_feat(
            slide=self.slide,
            df=self.cell_table,
            feat=feat,
            analyse_level=self.coords_meta["level"],
            level_to_view=level_to_view,
            discrete=discrete,
            ax=ax,
            title=feat,
            show=show,
        )

        if save:
            os.makedirs(save, exist_ok=True)
            path = os.path.join(save, f"{self.slide.name}_cell_{feat}.jpg")
            plt.savefig(path)

    def paint_group(
        self,
        feat: str,
        size: tuple = (1024, 1024),
        ax=None,
        show=False,
        save: str = None,
    ):
        thumbnail = self.get_thumbnail(size, numpy=True)
        thumbnail_height, thumbnail_width, _ = thumbnail.shape
        downsample_factor = max(
            self.slide.level_dimensions[self.coords_meta["level"]][0] / thumbnail_width,
            self.slide.level_dimensions[self.coords_meta["level"]][1]
            / thumbnail_height,
        )
        level_to_view, _, _ = self.slide.get_best_level_for_downsample(
            downsample_factor
        )

        visualise_tile_feat(
            self.slide,
            self.group_table,
            feat,
            analyse_level=self.coords_meta["level"],
            level_to_view=level_to_view,
            ax=ax,
            title=feat,
            show=show,
        )

        if save:
            os.makedirs(save, exist_ok=True)
            path = os.path.join(save, f"{self.slide.name}_group_{feat}.jpg")
            plt.savefig(path)

    def get_thumbnail(self, size, numpy: bool = False):
        thumbnail = self.slide.get_thumbnail(size)
        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]
        return thumbnail

    def save_tables(self, dst: str | None = None):
        cell_dst = os.path.join(dst, "cell")
        os.makedirs(cell_dst, exist_ok=True)
        cell_path = os.path.join(cell_dst, str(self.slide.name) + "_cell.csv")
        self.cell_table.to_csv(cell_path)

        group_dst = os.path.join(dst, "group")
        os.makedirs(group_dst, exist_ok=True)
        group_path = os.path.join(group_dst, str(self.slide.name) + "_group.csv")
        self.group_table.to_csv(group_path)

        # write keys in yaml
        with open(
            os.path.join(group_dst, str(self.slide.name) + "_feats.yaml"), "w"
        ) as outfile:
            yaml.dump(
                self.group_keys, outfile, default_flow_style=False, sort_keys=False
            )

        return cell_dst, group_path
