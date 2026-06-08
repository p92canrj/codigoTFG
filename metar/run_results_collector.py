import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import parallel_backend
from remayn.report import create_excel_columns_report, create_excel_summary_report
from remayn.result_set import ResultFolder

from launchexp.utils import compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run results collector")
    parser.add_argument("appendix", type=str, help="Appendix for the output files")
    parser.add_argument("--skip-zip", action="store_true", help="Skip zipping the results")
    args = parser.parse_args()
    appendix = args.appendix
    
    path = Path("./results_debug")
    results = ResultFolder(path)
    print("Results loaded")
    
    # Fields from the experiment config that will be included in the dataframe as columns
    config_columns_to_include = [
        "dataset",
        "estimator_name",
        "random_state",
        "execution_time",
    ]
    best_params_columns_to_include = []
    filter_methods = [
        "logisticat",
        "logisticregressor",
        "lightgbmclassifier_fast",
        "nnop",
        "ordinaldecomposition",
        "nnpom"
    ]
    filter_datasets = [
        # "LEVX_sensors_autoreg",
        # "LEST_sensors_autoreg",
        # "LEST_sensors_areg_only",
        # "LEVX_sensors_areg_only",
        # "LEST_sensors_areg_era5",
        # "LEVX_sensors_areg_era5",
        # "LEST_era5",
        # "LEVX_era5",
    ]
    
    seeds = range(30)
    
    
    def filter_fn(result):
        if (
            len(filter_methods) > 0
            and result.config["estimator_name"] not in filter_methods
        ):
            return False
    
        if len(filter_datasets) > 0 and result.config["dataset"] not in filter_datasets:
            return False
    
        if result.config["random_state"] not in seeds:
            return False
    
        return True
    
    
    with parallel_backend("multiprocessing"):
        df = results.create_dataframe(
            config_columns=config_columns_to_include,
            best_params_columns=best_params_columns_to_include,
            metrics_fn=compute_metrics,
            filter_fn=filter_fn,
            include_train=True,
            include_val=False,
            config_columns_prefix="",
        )
    
    
    df.sort_values(by=["dataset", "estimator_name", "random_state"], inplace=True)
    
    group_columns = ["dataset", "estimator_name"]
    
    print(df)
    
    output_path_wo_ext = (
        f'prepared_results_def/{datetime.now().strftime(r"%Y%m%d_%H%M%S")}_{appendix}'
    )
    
    metrics_example = compute_metrics([1, 2, 3, 4], [2, 1, 3, 4])
    metric_columns = list(metrics_example.keys())
    
    Path(output_path_wo_ext).parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(f"{output_path_wo_ext}.xlsx", mode="w") as writer:
        create_excel_summary_report(df, "", group_columns, excel_writer=writer)
        create_excel_columns_report(
            df,
            "",
            metric_columns=metric_columns,
            pivot_index="random_state",
            pivot_columns=["estimator_name", "dataset"],
            excel_writer=writer,
        )
    
    if not args.skip_zip:
        from shutil import make_archive
    
        make_archive(output_path_wo_ext, "zip", str(path))
