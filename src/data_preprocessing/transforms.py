import datetime
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd


def loc_by_expansion(mibi_features: pd.DataFrame,
                     expansion_type: str = "mask_only",
                     average: bool = True,
                     data_type="Data",
                     columns_to_keep: list = None,
                     columns_to_drop: list = None):
    """
    Loc MIBI dataframe by expansion
    """

    assert expansion_type in ["mask_only", "expansion_only", "mask_and_expansion"], "Unexpected expansion type"

    if expansion_type == "mask_only":
        transformed_features = mibi_features.loc[pd.IndexSlice[:,
                                                 :,
                                                 :-1,
                                                 data_type], :]
    elif expansion_type == "mask_and_expansion":
        transformed_features = mibi_features.loc[pd.IndexSlice[:,
                                                 :,
                                                 :,
                                                 data_type], :]
    elif expansion_type == "expansion_only":
        transformed_features = mibi_features.loc[pd.IndexSlice[:,
                                                 :,
                                                 0:,
                                                 data_type], :]

    if average:
        if columns_to_keep is not None:
            transformed_features = keep_columns_in_df(transformed_features, columns_to_keep)
        elif columns_to_drop is not None:
            transformed_features = drop_columns_in_df(transformed_features, columns_to_drop)

        transformed_features.reset_index(level=['Point', 'Vessel'], inplace=True)

        transformed_features = transformed_features.groupby(['Point', 'Vessel']).mean()

    return transformed_features


def melt_markers(mibi_features: pd.DataFrame,
                 non_id_vars: list = None,
                 id_vars: list = None,
                 reset_index: list = None,
                 add_marker_group: bool = False,
                 marker_groups: dict = None):
    """
    Melt MIBI dataframe by markers
    """

    if add_marker_group:
        assert marker_groups is not None, "No marker groups specified!"

    if non_id_vars is not None:
        id_vars = np.setdiff1d(mibi_features.columns, non_id_vars)

    transformed_features = pd.melt(mibi_features,
                                   id_vars=id_vars,
                                   ignore_index=False)

    transformed_features = transformed_features.rename(columns={'variable': 'Marker',
                                                                'value': 'Expression'})

    if reset_index is not None:
        transformed_features.reset_index(level=reset_index, inplace=True)

    if add_marker_group:
        for key in marker_groups.keys():
            for marker, marker_name in enumerate(marker_groups[key]):
                transformed_features.loc[transformed_features["Marker"] == marker_name, "Marker Group"] = key

    return transformed_features


def keep_columns_in_df(mibi_features: pd.DataFrame,
                       columns_to_keep: list):
    """
    Get data from MIBI dataframe
    """

    return mibi_features[columns_to_keep]


def drop_columns_in_df(mibi_features: pd.DataFrame,
                       columns_to_drop: list):
    """
    Drop data from MIBI dataframe
    """
    return mibi_features.drop(columns_to_drop, axis=1, errors='ignore')
