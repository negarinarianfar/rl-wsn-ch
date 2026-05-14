import os
import matplotlib.pyplot as plt

methods = ["Random-MR", "LEACH-MR", "HEED-MR", "RL-MR"]

FND = [891, 290, 515, 887]
Lifetime = [932, 1242, 1103, 933]

plt.figure(figsize=(7, 5))

plt.scatter(FND, Lifetime, s=80)

# label offsets
offsets = {
    "Random-MR": (10, -20),
    "LEACH-MR": (10, 10),
    "HEED-MR": (10, 10),
    "RL-MR": (10, 10)
}

for i, label in enumerate(methods):
    dx, dy = offsets[label]
    plt.text(FND[i] + dx, Lifetime[i] + dy, label)

plt.xlabel("FND (First Node Dies)")
plt.ylabel("Network Lifetime")
plt.title("FND–Lifetime Trade-off Across Multi-Role Clustering Methods")

plt.grid(True)
plt.tight_layout()

os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/fnd_lifetime_tradeoff.png", dpi=300)

plt.show()