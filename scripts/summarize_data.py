import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO inner ear train data and mito training data are missing
az_train = pd.read_excel("data_summary/active_zone_training_data.xlsx")
compartment_train = pd.read_excel("data_summary/compartment_training_data.xlsx")
vesicle_train = pd.read_excel("data_summary/vesicle_training_data.xlsx")
vesicle_da = pd.read_excel("data_summary/vesicle_domain_adaptation_data.xlsx", sheet_name="cryo")


def training_resolutions():
    res_az = np.round(az_train["resolution"].mean(), 2)
    res_compartment = np.round(compartment_train["resolution"].mean(), 2)
    res_cryo = np.round(vesicle_da["resolution"].mean(), 2)
    res_vesicles = np.round(vesicle_train["resolution"].mean(), 2)

    print("Training resolutions for models:")
    print("active_zone:", res_az)
    print("compartments:", res_compartment)
    # TODO
    print("mitochondria:", 1.0)
    print("vesicles_2d:", res_vesicles)
    print("vesicles_3d:", res_vesicles)
    print("vesicles_cryo:", res_cryo)
    # TODO inner ear


def pie_chart(data, count_col, title):
    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        data[count_col],
        labels=data["Condition"],
        autopct="%1.1f%%",  # Display percentages
        startangle=90,      # Start at the top
        colors=plt.cm.Paired.colors[:len(data)],  # Optional: Custom color palette
        textprops={"fontsize": 14}
    )

    for autot in autotexts:
        autot.set_fontsize(18)

    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def summarize_vesicle_train_data():
    condition_summary = {
        "Condition": [],
        "Tomograms": [],
        "Vesicles": [],
    }

    conditions = pd.unique(vesicle_train.condition)
    for condition in conditions:
        ctab = vesicle_train[vesicle_train.condition == condition]
        n_tomos = len(ctab)
        n_vesicles_all = ctab["vesicle_count_all"].sum()
        n_vesicles_imod = ctab["vesicle_count_imod"].sum()
        print(condition)
        print("Tomograms:", n_tomos)
        print("All-Vesicles:", n_vesicles_all)
        print("Vesicles-From-Manual:", n_vesicles_imod)
        print()
        condition_summary["Condition"].append(condition)
        condition_summary["Tomograms"].append(n_tomos)
        condition_summary["Vesicles"].append(n_vesicles_all)
    condition_summary = pd.DataFrame(condition_summary)
    print()
    print()

    print("Total:")
    print("Tomograms:", len(vesicle_train))
    print("All-Vesicles:", vesicle_train["vesicle_count_all"].sum())
    print("Vesicles-From-Manual:", vesicle_train["vesicle_count_imod"].sum())
    print()

    train_tomos = vesicle_train[vesicle_train.used_for == "train/val"]
    print("Training:")
    print("Tomograms:", len(train_tomos))
    print("All-Vesicles:", train_tomos["vesicle_count_all"].sum())
    print("Vesicles-From-Manual:", train_tomos["vesicle_count_imod"].sum())
    print()

    test_tomos = vesicle_train[vesicle_train.used_for == "test"]
    print("Test:")
    print("Tomograms:", len(test_tomos))
    print("All-Vesicles:", test_tomos["vesicle_count_all"].sum())
    print("Vesicles-From-Manual:", test_tomos["vesicle_count_imod"].sum())

    pie_chart(condition_summary, "Tomograms", "Tomograms per Condition")
    pie_chart(condition_summary, "Vesicles", "Vesicles per Condition")


def summarize_vesicle_da():
    for name in ("inner_ear", "endbulb", "cryo", "frog", "maus_2d"):
        tab = pd.read_excel("data_summary/vesicle_domain_adaptation_data.xlsx", sheet_name=name)
        print(name)
        print("N-tomograms:", len(tab))
        print("N-test:", (tab["used_for"] == "test").sum())
        print("N-vesicles:", tab["vesicle_count"].sum())
        print()


def summarize_az_train():
    conditions = pd.unique(az_train.condition)
    print(conditions)

    print("Total:")
    print("Tomograms:", len(az_train))
    print("Active Zones:", az_train["az_count"].sum())
    print()

    train_tomos = az_train[az_train.used_for == "train/val"]
    print("Training:")
    print("Tomograms:", len(train_tomos))
    print("Active Zones:", train_tomos["az_count"].sum())
    print()

    test_tomos = az_train[az_train.used_for == "test"]
    print("Test:")
    print("Tomograms:", len(test_tomos))
    print("Active Zones:", test_tomos["az_count"].sum())


def summarize_compartment_train():
    conditions = pd.unique(compartment_train.condition)
    print(conditions)

    print("Total:")
    print("Tomograms:", len(compartment_train))
    print("Compartments:", compartment_train["compartment_count"].sum())
    print()

    train_tomos = compartment_train[compartment_train.used_for == "train/val"]
    print("Training:")
    print("Tomograms:", len(train_tomos))
    print("Compartments:", train_tomos["compartment_count"].sum())
    print()

    test_tomos = compartment_train[compartment_train.used_for == "test"]
    print("Test:")
    print("Tomograms:", len(test_tomos))
    print("Compartments:", test_tomos["compartment_count"].sum())


def summarize_inner_ear_data():
    # NOTE: this is not all trainig data, but the data on which we run the analysis
    # New tomograms from Sophia.
    n_tomos_sophia_tot = 87
    n_tomos_sophia_manual = 33  # noqa
    # This is the training data
    n_tomos_sohphia_train = ""  # TODO  # noqa

    # Published tomograms
    n_tomos_rat = 19
    n_tomos_tether = 3
    n_tomos_ves_pool = 6

    # 28
    print("Total published:", n_tomos_rat + n_tomos_tether + n_tomos_ves_pool)
    # 115
    print("Total:", n_tomos_rat + n_tomos_tether + n_tomos_ves_pool + n_tomos_sophia_tot)


# training_resolutions()
# summarize_vesicle_train_data()
# summarize_vesicle_da()
# summarize_az_train()
# summarize_compartment_train()
# summarize_inner_ear_data()
summarize_inner_ear_data()
