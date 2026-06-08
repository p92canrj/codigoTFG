# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from remayn.result_set import ResultFolder

datasets = ["LEST_sensors_autoreg", "LEVX_sensors_autoreg"]
method = "logisticat"

sns.set_palette(sns.color_palette("pastel")[1:2] + sns.color_palette("pastel")[0:1])

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

etiquetas = [r"(a) LEST", r"(b) LEVX"]

for idx, data in enumerate(datasets):

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

    observed = resultado_metodo.targets[-31 * 24 :]  # noqa
    sns.lineplot(data=observed + 5, ax=axs[idx], label="Observed")

    preds = resultado_metodo.predictions[-31 * 24 :]  # noqa
    sns.lineplot(data=preds, ax=axs[idx], label="Predicted")

    axs[idx].set_yticks(
        ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        labels=[
            r"$\mathcal{C}_0$",
            r"$\mathcal{C}_1$",
            r"$\mathcal{C}_2$",
            r"$\mathcal{C}_3$",
            "",
            r"$\mathcal{C}_0$",
            r"$\mathcal{C}_1$",
            r"$\mathcal{C}_2$",
            r"$\mathcal{C}_3$",
        ],
    )
    yticks = axs[idx].yaxis.get_major_ticks()
    yticks[4].set_visible(False)

    axs[idx].set_xticks(
        ticks=[0, 185, 372, 557, 720],
        labels=[
            "2023-12-01",
            "2023-12-08",
            "2023-12-15",
            "2023-12-23",
            "2023-12-31",
        ],
    )

    axs[idx].tick_params(axis="x", labelsize=12)
    axs[idx].tick_params(axis="y", labelsize=14)

    # set title per subfigure
    axs[idx].set_title(etiquetas[idx], fontsize=16)

plt.legend()
plt.show()

fig.savefig("predictions.pdf", bbox_inches="tight")


# %%
