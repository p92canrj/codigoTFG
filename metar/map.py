# %%
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_map():
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Map of Spain
    ax[0].set_title("Map of Spain")
    m1 = Basemap(
        projection="merc",
        llcrnrlon=-10.0,
        llcrnrlat=35.0,
        urcrnrlon=5.0,
        urcrnrlat=45.5,
        resolution="i",
        ax=ax[0],
    )
    m1.drawcoastlines()
    m1.drawcountries()
    m1.fillcontinents(color="mediumseagreen", lake_color="lightblue")
    m1.drawmapboundary(fill_color="lightblue")

    # Highlight Galicia
    galicia_lon = [-9.3, -7.0]
    galicia_lat = [41.9, 43.8]
    x1, y1 = m1(galicia_lon[0], galicia_lat[0])
    x2, y2 = m1(galicia_lon[1], galicia_lat[1])
    ax[0].add_patch(
        plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor="darkred", linewidth=2, fill=False
        )
    )

    # Zoomed Map of Galicia
    ax[1].set_title("Zoomed Map of Galicia")
    m2 = Basemap(
        projection="merc",
        llcrnrlon=galicia_lon[0],
        llcrnrlat=galicia_lat[0],
        urcrnrlon=galicia_lon[1],
        urcrnrlat=galicia_lat[1],
        resolution="i",
        ax=ax[1],
    )
    m2.drawcoastlines()
    m2.drawcountries()
    m2.drawstates()  # Draw province boundaries
    m2.fillcontinents(color="mediumseagreen", lake_color="lightblue")
    m2.drawmapboundary(fill_color="lightblue")

    # Highlight specified locations
    locations = [
        {"lat": 42.23180, "lon": -8.62677, "label": "LEVX"},
        {"lat": 42.89630, "lon": -8.41514, "label": "LEST"},
    ]

    for loc in locations:
        x, y = m2(loc["lon"], loc["lat"])
        m2.plot(x, y, "o", markersize=8, color="darkred", label=loc["label"])
        x1, y1 = m1(loc["lon"], loc["lat"])
        m1.plot(x1, y1, "o", markersize=2, color="darkred", label=loc["label"])
        plt.text(
            x + 20000,
            y - 5000,
            loc["label"],
            fontsize=20,
            fontweight="bold",
            color="darkred",
        )

    plt.tight_layout()
    plt.show()

    # save figure in pdf
    fig.savefig("map.pdf", bbox_inches="tight")


plot_map()

# %%
