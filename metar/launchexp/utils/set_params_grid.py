from itertools import combinations_with_replacement

from launchexp.utils.set_estimator import NOMINAL_CLASSIFIERS, ORDINAL_CLASSIFIERS


def set_params_grid(estimator_name, random_state):
    estimator_name = estimator_name.casefold()

    if estimator_name in NOMINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticregressor":

                base_grid = {
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "solver": "saga",
                }

                param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "lightgbmclassifier":

                base_grid = {
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "verbose": -1,
                    "n_jobs": 1,
                }

                param_grid = {
                    "n_estimators": [100, 500, 1000],
                    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "num_leaves": [10, 20, 30, 40, 50],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                }

            case "lightgbmclassifier_fast":

                base_grid = {
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "verbose": -1,
                    "n_jobs": 1,
                }

                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, -1],
                    "num_leaves": [20, 31, 50],
                    "learning_rate": [0.05, 0.1, 0.2, 0.3],
                }

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in NOMINAL_CLASSIFIERS "
                    + "but not implemented in set_params_grid function."
                )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticat":

                base_grid = {"class_weight": "balanced"}
                # base_grid = {"class_weight": {0: 30, 1: 78.5, 2: 11, 3: 1.1}}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "logisticit":

                base_grid = {"class_weight": "balanced"}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "nnop":

                base_grid = {"class_weight": "balanced"}

                param_grid = {
                    "n_hidden": [10, 50],
                    "max_iter": [250, 500],
                    "lambda_value": [0.001, 0.01, 0.1, 1],
                }

            case "nnpom":

                base_grid = {"class_weight": "balanced"}

                param_grid = {
                    "n_hidden": [10, 50],
                    "max_iter": [250, 500],
                    "lambda_value": [0.001, 0.01, 0.1, 1],
                }

            case "ordinaldecomposition":

                base_grid = {
                    "base_classifier": "LogisticRegression",
                    "parameters": {"class_weight": "balanced", "max_iter": 5000}
                }

                param_grid = [
                    {
                        "dtype": ["ordered_partitions"],
                        "decision_method": ["frank_hall", "exponential_loss", "hinge_loss", "logarithmic_loss"],
                    },
                    {
                        "dtype": ["one_vs_next", "one_vs_followers", "one_vs_previous"],
                        "decision_method": ["exponential_loss", "hinge_loss", "logarithmic_loss"],
                    }
                ]

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                    + "but not implemented in set_param_grid function."
                )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")

    return base_grid, param_grid
