from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from launchexp.metrics import amae

NOMINAL_CLASSIFIERS = [
    "logisticregressor",
    "lightgbmclassifier",
    "lightgbmclassifier_fast",
]

ORDINAL_CLASSIFIERS = [
    "logisticat",
    "logisticit",
    "nnop",
    "nnpom",
    "ordinaldecomposition",
]


def set_estimator(
    estimator_name,
    base_grid,
    param_grid,
    random_state,
    interactive,
    n_jobs=-1,
    **kwargs,
):
    estimator_name = estimator_name.casefold()

    if estimator_name in NOMINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticregressor":
                from sklearn.linear_model import LogisticRegression

                estimator = LogisticRegression(**base_grid)

            case "lightgbmclassifier" | "lightgbmclassifier_fast":
                from lightgbm import LGBMClassifier

                estimator = LGBMClassifier(**base_grid)

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in NOMINAL_CLASSIFIERS "
                    + "but not implemented in set_estimator function."
                )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticat":
                from launchexp.ordinal_classification import LogisticAT

                estimator = LogisticAT(**base_grid)

            case "logisticit":
                from launchexp.ordinal_classification import LogisticIT

                estimator = LogisticIT(**base_grid)

            case "nnop":
                from launchexp.ordinal_classification import NNOP
                
                estimator = NNOP(**base_grid)

            case "nnpom":
                from launchexp.ordinal_classification import NNPOM
                
                estimator = NNPOM(**base_grid)

            case "ordinaldecomposition":
                from launchexp.ordinal_classification import OrdinalDecomposition

                estimator = OrdinalDecomposition(**base_grid)

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                    + "but not implemented in set_estimator function."
                )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=make_scorer(amae, greater_is_better=False),  # neg_median_absolute_error
        n_iter=30 if not interactive else 5,
        n_jobs=n_jobs,
        cv=3,
        error_score="raise",
        random_state=random_state,
        verbose=0,
        **kwargs,
    )
