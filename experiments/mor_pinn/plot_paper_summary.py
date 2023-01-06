# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Times New Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)

# %%
cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])

x = [1, 2, 3, 4, 5, 6, 7]
y = [
    1.4649852444708864e-06,
    0.00621321865381177,
    0.00019787361905420703,
    3.847280264760879e-06,
    1.0,
    1.0,
    1.0,
]

labels = [
    "FOM",
    "ROM rb=4",
    "ROM rb=8",
    "ROM rb=12",
    "PINN",
    "PINN+MOR",
    "PINN+MOR error sens",
]


ax.set_yscale("log")
ax.plot(x, y, "*")
# ax.xticks(x, labels, rotation="vertical")
ax.set_xticks(x)
ax.set_xticklabels(labels)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.show()
# %%
