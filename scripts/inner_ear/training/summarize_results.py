import numpy as np
import pandas as pd


def summarize_source_domain():
    result_path = "./results/train_domain_postprocessed.csv"
    results = pd.read_csv(result_path)

    ribbon_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }
    ribbon_mean = np.round(results["ribbon"].mean() * 100, 2)
    ribbon_std = np.round(results["ribbon"].std() * 100, 2)
    ribbon_results["dataset"].append("source")
    ribbon_results["source_model"].append(f"{ribbon_mean} +- {ribbon_std}")
    ribbon_results["target_model"].append("")
    ribbon_results = pd.DataFrame(ribbon_results)

    PD_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }
    PD_mean = np.round(results["PD"].mean() * 100, 2)
    PD_std = np.round(results["PD"].std() * 100, 2)
    PD_results["dataset"].append("source")
    PD_results["source_model"].append(f"{PD_mean} +- {PD_std}")
    PD_results["target_model"].append("")
    PD_results = pd.DataFrame(PD_results)

    return ribbon_results, PD_results


def summarize_rat():
    ribbon_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }
    PD_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }

    result_paths = {
        "source_model": "results/rat_Src.csv",
        "target_model": "results/rat_Adapted.csv",
    }

    ribbon_results["dataset"].append("source")
    PD_results["dataset"].append("source")

    for model, result_path in result_paths.items():
        results = pd.read_csv(result_path)
        ribbon_mean = np.round(results["ribbon"].mean() * 100, 2)
        ribbon_std = np.round(results["ribbon"].std() * 100, 2)
        ribbon_results[model].append(f"{ribbon_mean} +- {ribbon_std}")

        PD_mean = np.round(results["PD"].mean() * 100, 2)
        PD_std = np.round(results["PD"].std() * 100, 2)
        PD_results[model].append(f"{PD_mean} +- {PD_std}")

    ribbon_results = pd.DataFrame(ribbon_results)
    PD_results = pd.DataFrame(PD_results)
    return ribbon_results, PD_results


def summarize_ves_pool():
    ribbon_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }
    PD_results = {
        "dataset": [],
        "source_model": [],
        "target_model": [],
    }

    result_paths = {
        "source_model": "results/vesicle_pools_Src.csv",
        "target_model": "results/vesicle_pools_Adapted.csv",
    }

    ribbon_results["dataset"].append("source")
    PD_results["dataset"].append("source")

    for model, result_path in result_paths.items():
        results = pd.read_csv(result_path)
        ribbon_mean = np.round(results["ribbon"].mean() * 100, 2)
        ribbon_std = np.round(results["ribbon"].std() * 100, 2)
        ribbon_results[model].append(f"{ribbon_mean} +- {ribbon_std}")

        PD_mean = np.round(results["PD"].mean() * 100, 2)
        PD_std = np.round(results["PD"].std() * 100, 2)
        PD_results[model].append(f"{PD_mean} +- {PD_std}")

    ribbon_results = pd.DataFrame(ribbon_results)
    PD_results = pd.DataFrame(PD_results)
    return ribbon_results, PD_results


def main():
    ribbon_results, PD_results = summarize_source_domain()
    # ribbon_results, PD_results = summarize_ves_pool()
    print("Ribbon")
    print(ribbon_results)
    print("PD")
    print(PD_results)


if __name__ == "__main__":
    main()
