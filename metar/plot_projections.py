# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from remayn.result_set import ResultFolder
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(data_dir, dataset, random_state, interactive):

    inputs = dataset.split("_")

    if inputs[1] == "sensors":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        # 1h prediction task. X is the current time, y is the next time
        X = df.values[:-1, 5:10]  # wind_dir, wind_speed, temp, dewpt, press
        print(X.shape)

    elif inputs[1] == "era5":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)  # era5
        X = df.values[:-1, 5:]

    elif inputs[1] == "concat":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)  # sensors
        X = df.values[:-1, 5:10]
        df_2 = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)  # era5
        X = np.column_stack((X, df_2.values[:-1, 5:]))

    df_output = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
    y = df_output.values[1:, -1]
    if inputs[2] == "autoreg":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        X = np.column_stack((X, df.values[:-1, -1]))
        print(X)

    # discretise the target in 4 classes
    y = pd.cut(y, bins=[-np.inf, 500, 1000, 5000, np.inf], labels=[0, 1, 2, 3])

    print(
        "Full (100%) ->",
        np.unique(y, return_counts=True)[1],
        np.round(np.unique(y, return_counts=True)[1] / len(y) * 100, 2),
    )

    # split the data into train and test
    test_starts = int(len(y) * 0.75)

    X_train, y_train = X[:test_starts], y[:test_starts]
    X_test, y_test = X[test_starts:], y[test_starts:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    print(
        "Train (75%) ->",
        np.unique(y_train, return_counts=True)[1],
        np.round(np.unique(y_train, return_counts=True)[1] / len(y_train) * 100, 2),
    )

    print(
        "Test  (25%) ->",
        np.unique(y_test, return_counts=True)[1],
        np.round(np.unique(y_test, return_counts=True)[1] / len(y_test) * 100, 2),
    )

    if interactive:
        return X_train, y_train, X_test, y_test
        return X_train[:5000], y_train[:5000], X_test[:1000], y_test[:1000]
    else:
        return X_train, y_train, X_test, y_test


datasets = ["LEST_sensors_autoreg", "LEVX_sensors_autoreg"]
method = "logisticat"

data_dir = "./data/"
random_state = 0
interactive = False

class_names = [
    r"$\mathcal{C}_0$",
    r"$\mathcal{C}_1$",
    r"$\mathcal{C}_2$",
    r"$\mathcal{C}_3$",
]


sns.set_palette(sns.color_palette("pastel")[1:2] + sns.color_palette("pastel")[0:1])

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

etiquetas = [r"(a) LEST", r"(b) LEVX"]

for idx, data in enumerate(datasets):

    X_train, y_train, X_test, y_test = load_data(
        data_dir, data, random_state, interactive
    )

    def filter_fn(result):
        if (
            result.config["estimator_name"] == method
            and result.config["dataset"] == data
        ):
            return True

        return False

    path = Path("./results_debug")
    results = ResultFolder(path)
    print("Results loaded")

    results = results.filter(filter_fn)

    resultado_metodo = list(results.results_.values())[0].get_data()

    w = resultado_metodo.best_model.coef_
    print(w)
    theta = resultado_metodo.best_model.theta_
    print(theta)

    proyecciones = np.asarray(X_test.dot(w))[-744:]

    tmp = theta[:, None] - proyecciones
    predicciones = np.sum(tmp < 0, axis=0).astype(int)
    observed = resultado_metodo.targets[-744:]
    # if idx == 0:
    #     for jj in range(len(observed)):
    #         if observed[jj] == 1 and predicciones[jj] == 2:
    #             print(jj)
    #             patron = X_test[jj]
    # else:
    #     valorpatron = patron.dot(w)
    #     print(valorpatron)
    mask = predicciones == observed
    proyecciones_true = proyecciones[mask]
    proyecciones_false = proyecciones[~mask]
    observed_true = observed[mask]
    observed_false = observed[~mask]

    # plot seaborn scatterplot with marker depending on proyecciones_true
    a = sns.scatterplot(
        x=proyecciones_true,
        y=observed_true,
        ax=axs[idx],
        hue=observed_true,
        marker="o",
        palette="pastel",
        s=60,
    )

    # plot seaborn scatterplot with marker depending on proyecciones_true
    b = sns.scatterplot(
        x=proyecciones_false,
        y=observed_false,
        ax=axs[idx],
        hue=observed_false,
        marker="x",
        palette="pastel",
        s=60,
    )

    # combine legend markers
    handles, labels = axs[idx].get_legend_handles_labels()
    # get handles 0 and 4 for the legend in a var rre

    from matplotlib.legend_handler import HandlerTuple

    axs[idx].legend(
        [tuple([handles[i], handles[i + 4]]) for i in range(4)],
        class_names,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=14,
        loc="lower right",
    )

    # plot theta vertical lines between y=0 and y=3
    for idx2, t in enumerate(theta):
        axs[idx].axvline(t, color="black", ymin=0.1, linestyle="--", alpha=0.2)

    # include theta labels below the avxline
    for idx2, t in enumerate(theta):
        axs[idx].text(t, -0.4, rf"$\theta_{idx2+1}$", fontsize=14, ha="center")

    # include a title for each interval defined by theta
    axs[idx].text(
        theta[0] - 0.7,
        3.6,
        r"$\mathcal{C}_0$",
        fontsize=16,
        ha="center",
        alpha=0.35,
    )
    axs[idx].annotate(
        "",
        xy=(theta[0], 3.5),
        xycoords="data",
        xytext=(-6.2, 3.5),
        textcoords="data",
        arrowprops=dict(arrowstyle="<|-|>", ls="--", alpha=0.2, color="black"),
    )

    # annotate the intervals defined by theta

    for idx2, t in enumerate(theta):
        if idx2 == len(theta) - 1:
            axs[idx].text(
                (t + 3) / 2,
                3.6,
                r"$\mathcal{C}_" f"{idx2+1}$",
                fontsize=16,
                ha="center",
                alpha=0.35,
            )
            axs[idx].annotate(
                "",
                xy=(theta[idx2] + (3.9 if idx == 0 else 4.3), 3.5),
                xycoords="data",
                xytext=(theta[idx2], 3.5),
                textcoords="data",
                arrowprops=dict(arrowstyle="<|-|>", ls="--", alpha=0.2, color="black"),
            )
        else:
            axs[idx].text(
                (t + theta[idx2 + 1]) / 2,
                3.6,
                r"$\mathcal{C}_" f"{idx2+1}$",
                fontsize=16,
                ha="center",
                alpha=0.35,
            )
            axs[idx].annotate(
                "",
                xy=(theta[idx2 + 1], 3.5),
                xycoords="data",
                xytext=(theta[idx2], 3.5),
                textcoords="data",
                arrowprops=dict(arrowstyle="<|-|>", ls="--", alpha=0.2, color="black"),
            )

    # include infinite symbol in (-6.2, 3.5) and in (3, 3.5)
    axs[idx].text(-6.05, 3.3, r"$-\infty$", fontsize=14, ha="center", alpha=0.35)
    axs[idx].text(-5.7, 3.3, r"$(\theta_0)$", fontsize=11, ha="center", alpha=0.35)
    axs[idx].text(2.93, 3.3, r"$\infty$", fontsize=14, ha="center", alpha=0.35)
    axs[idx].text(3.2, 3.3, r"$(\theta_4)$", fontsize=11, ha="center", alpha=0.35)

    # set y limits
    axs[idx].set_ylim(-0.5, 3.9)
    axs[idx].tick_params(axis="x", labelsize=12)

    # remove yaxis
    axs[idx].yaxis.set_visible(False)

    # set title per subfigure
    axs[idx].set_title(etiquetas[idx], fontsize=16)

    # add x axis label
    axs[idx].set_xlabel("Projections", fontsize=16)

# Aspecto relevante 1
axs[0].add_patch(
    plt.Rectangle(
        (-5.95, 2.85),
        1.05,
        0.3,
        color="purple",
        fill=False,
        linewidth=1,
    )
)
axs[0].text(-5.4, 2.6, "1", fontsize=14, ha="center", color="purple")

# Aspecto relevante 2
axs[1].add_patch(
    plt.Rectangle(
        (-5.7, -0.15),
        1.5,
        0.3,
        color="purple",
        fill=False,
        linewidth=1,
    )
)
axs[1].text(-4.95, 0.27, "2", fontsize=14, ha="center", color="purple")

# Aspecto relevante 3
axs[1].add_patch(
    plt.Rectangle(
        (-1, -0.15),
        0.5,
        1.3,
        color="purple",
        fill=False,
        linewidth=1,
    )
)
axs[1].text(-0.75, 1.27, "3", fontsize=14, ha="center", color="purple")

plt.show()

fig.savefig("projections.pdf", bbox_inches="tight")

# %%
