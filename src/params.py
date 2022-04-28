"""
This file contains code for computing state transition parameters
"""
import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.util import update_config


class TransitionParams:
    """
    Class for computing state transition parameters using linear regression
    """

    def __init__(self, data_file_path: str, idle_threshold: str) -> None:
        """
        Initialize a new instance

        Args:
            data_file_path (str): File path for HVAC power consumption and temperature
                change dataset in csv format.
            idle_threshold (str): The preset temperature threshold for idle state
        """
        # Assume the input csv follows the format of df_all in cell 8
        self.df_all = pd.read_csv(data_file_path)
        self.idle_threshold = idle_threshold
        self.data_preprocessing()

    def data_preprocessing(self) -> None:
        """
        Preprocess the dataframe for linear regression
        """
        # Start from cell 98
        df_temp = self.df_all.copy()
        df_temp["time_round_1"] = df_temp.index + datetime.timedelta(minutes=1)

        df_all_xy = df_temp.reset_index().merge(
            self.df_all.reset_index(),
            how="inner",
            left_on="time_round_1",
            right_on="time_round_hvac",
            validate="one_to_one",
        )
        df_all_xy["outdoor_temp_x_gap"] = (
            df_all_xy["outdoor_temp_x"] - df_all_xy["indoor_temp_x"]
        )
        df_all_xy["supply_temp_x_gap"] = (
            df_all_xy["supply_temp_x"] - df_all_xy["indoor_temp_x"]
        )
        df_all_xy["indoor_temp_delta"] = (
            df_all_xy["indoor_temp_y"] - df_all_xy["indoor_temp_x"]
        )
        self.df_all_xy = df_all_xy

    def get_queries(
        self, component_name: str, relation: str, idle_relation: str = None
    ) -> list:
        """
        Returns the list of queries that filters out component data

        Args:
            component_name (str): HVAC component name
            relation (str): Greater or less than outdoor temp
            idle_relation (str, optional): For idle state only, greater or less than
                preset threshold. Defaults to None.

        Returns:
            list: List of queries
        """
        query_list = []

        if component_name == "idle":
            query_list.append(
                "compressor_status_x == compressor_status_y == freecool_status_x == \
                    freecool_status_y == heater_status_x == heater_status_y == 0"
            )
            query_list.append(f"indoor_temp_x {idle_relation} {self.idle_threshold}")
        else:
            query_list.append(
                f"{component_name}_status_x == {component_name}_status_y == 1"
            )

        if relation:
            query_list.append(f"outdoor_temp_x_gap {relation} 0")

        return query_list

    def linear_regression(self, query_list: list) -> tuple[float, float]:
        """
        Linear regression to compute the state transition parameters for a component under
        queried conditions

        Args:
            query_list (list): List of queries that filters out component data

        Returns:
            tuple[float, float]: State transition intercept and coefficient
        """
        df_all_xy_clean_component = self.df_all_xy.copy()
        for query in query_list:
            df_all_xy_clean_component = df_all_xy_clean_component.query(query)

        lm_all = LinearRegression()
        lm_all.fit(
            df_all_xy_clean_component[["outdoor_temp_x_gap"]],
            df_all_xy_clean_component[["indoor_temp_delta"]],
        )

        # intercept, coef
        return np.transpose(lm_all.intercept_.tolist(), lm_all.coef_[0, :].tolist())

    def compute_component(
        self,
        component: str,
        relation_key_val: tuple,
        idle_relation_key_val: tuple = (None, None),
    ) -> None:
        """
        Compute all state transition parameters for a component

        Args:
            component (str): Component name
            relation_key_val (tuple): Relation key value pair. Key is relation name in
                config file, value is the relation used in query
            idle_relation_key_val (tuple, optional): Idle relation key value pair.
                Key is idle relation name in config file, value is the idle relation
                used in query. Defaults to None.
        """
        relation_name, relation = relation_key_val
        idle_relation_name, idle_relation = idle_relation_key_val

        queries = self.get_queries(component, relation, idle_relation)
        intercept, coef = self.linear_regression(queries)
        intercept_name = f"temp_step_{component}_intercept_{relation_name}"
        coef_name = f"temp_step_{component}_coef_{relation_name}"

        if idle_relation:
            intercept_name += f"_{idle_relation_name}"
            coef_name += f"_{idle_relation_name}"

        update_config("environment_params.cfg", "transition", intercept_name, intercept)
        update_config("environment_params.cfg", "transition", coef_name, coef)

    def compute_all(self):
        """
        Compute all componenets state transition parameters
        """
        relations = {"all": None, "neg_diff": "<=", "pos_diff": ">"}
        idle_relations = {"above_th": ">", "below_th": "<="}
        components = ["compressor", "freecool", "heater"]
        for comp in components:
            for key_val in relations.items():
                self.compute_component(comp, key_val)
        for idle_key_val in idle_relations.items():
            for key_val in relations:
                self.compute_component("idle", key_val, idle_key_val)
