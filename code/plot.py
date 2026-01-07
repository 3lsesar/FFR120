
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ast


# Ensure figures are saved in /home/oai/share with correct names
save_dir = './plots/'
read_dir = './data/'

read_dir = './data/'
read_file = read_dir + "simulation_data_.txt"

# Leer todo el archivo
with open(read_file, "r") as f:
    content = f.read().strip()

# Quitar 'results =' si existe
if content.startswith("results ="):
    content = content.split("=", 1)[1].strip()

# Convertir a objeto Python
results = ast.literal_eval(content)

print(f"Loaded {len(results)} simulations")

# --- HELPER FUNCTION ---
def mean_jam_and_exited(results_list):
    """Calcula la media de jammed y exited de una lista de resultados."""
    jam_mean = np.mean([1 if r['jammed'] else 0 for r in results_list])
    exited_mean = np.mean([r['exited'] for r in results_list])
    return jam_mean, exited_mean

# --- 1) Jam probability vs D (mu fixed at 0.6) ---
D_values = sorted(list(set([r['D'] for r in results if r['mu'] == 0.6])))
jam_prob_D = []
removed_D = []
for D in D_values:
    group = [r for r in results if r['D'] == D and r['mu'] == 0.6]
    jam, rem = mean_jam_and_exited(group)
    jam_prob_D.append(jam)
    removed_D.append(rem)

# --- 2) Jam probability vs mu (D fixed at 0.06) ---
mu_values = sorted(list(set([r['mu'] for r in results if r['D'] == 0.06])))
jam_prob_mu = []
removed_mu = []
for mu in mu_values:
    group = [r for r in results if r['D'] == 0.06 and r['mu'] == mu]
    jam, rem = mean_jam_and_exited(group)
    jam_prob_mu.append(jam)
    removed_mu.append(rem)

# --- 3) Radius noise sweep (D subset con ruido, si se tiene info) ---
D_noise = D_values
jam_prob_noise = []
removed_noise = []
for D in D_noise:
    group = [r for r in results if r['D'] == D and r['mu'] == 0.6]  # mu=0.6
    if group:
        jam, rem = mean_jam_and_exited(group)
    else:
        jam, rem = 0, 0
    jam_prob_noise.append(jam)
    removed_noise.append(rem)

# --- 4) Multi-mu per D ---
selected_D = sorted(list(set([r['D'] for r in results])))
mu_vals = sorted(list(set([r['mu'] for r in results])))
jam_prob_mu_multi = {}
removed_mu_multi = {}
for D in selected_D:
    jam_prob_mu_multi[D] = {'prob': []}
    removed_mu_multi[D] = {'removed': []}
    for mu in mu_vals:
        group = [r for r in results if r['D'] == D and r['mu'] == mu]
        if group:
            jam, rem = mean_jam_and_exited(group)
        else:
            jam, rem = 0, 0
        jam_prob_mu_multi[D]['prob'].append(jam)
        removed_mu_multi[D]['removed'].append(rem)

# --- 5) Remaining discs vs mu (para un D fijo, ejemplo primer D) ---
D_fixed = D_values[2]
mu_range = sorted(list(set([r['mu'] for r in results if r['D']==D_fixed])))
remaining = []
for mu in mu_range:
    group = [r for r in results if r['D']==D_fixed and r['mu']==mu]
    if group:
        _, rem = mean_jam_and_exited(group)
        remaining.append(group[0]['N'] - rem)  # suponemos mismo N
    else:
        remaining.append(0)

### DATA DICTIONARY ###
data = {
    'D_values': D_values,
    'jam_prob_D': jam_prob_D,
    'removed_D': removed_D,
    'mu_values': mu_values,
    'jam_prob_mu': jam_prob_mu,
    'removed_mu': removed_mu,
    'D_noise': D_noise,
    'jam_prob_noise': jam_prob_noise,
    'removed_noise': removed_noise,
    'selected_D': selected_D,
    'mu_vals': mu_vals,
    'jam_prob_mu_multi': jam_prob_mu_multi,
    'removed_mu_multi': removed_mu_multi,
    'mu_range': mu_range,
    'remaining': remaining
}
###-------------------###

### PLOTS ###

def plot_jamD(D_values, jamD_prob, jamD_removed):
    # Jam probability vs D plot
    plt.figure()
    plt.plot(D_values, [p*100 for p in jamD_prob], marker='o')
    plt.xlabel('Orifice width D (mm)', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Jam probability (%)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_jam_vs_D_friction_updated.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

    # Removed discs vs D plot
    plt.figure()
    plt.plot(D_values, jamD_removed, marker='o')
    plt.xlabel('Orifice width D (mm)', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Mean discs removed (#)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_removed_vs_D_friction_updated.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

def plot_jamMu(mu_values, jamMu_prob, jamMu_removed):
    plt.figure()
    plt.plot(mu_values, [p*100 for p in jamMu_prob], marker='o')
    plt.xlabel('Friction coefficient mu', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Jam probability (%)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_jam_vs_mu.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

    plt.figure()
    plt.plot(mu_values, jamMu_removed, marker='o')
    plt.xlabel('Friction coefficient mu', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Mean discs removed (#)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_removed_vs_mu.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

def plot_jamDnoise(D_values, jamDnoisy_prob, jamDnoisy_removed):
    plt.figure()
    plt.plot(D_values, [p*100 for p in jamDnoisy_prob], marker='o')
    plt.xlabel('Orifice width D (mm)', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Jam probability (%)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_jam_vs_D_noise.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

    plt.figure()
    plt.plot(D_values, jamDnoisy_removed, marker='o')
    plt.xlabel('Orifice width D (mm)', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Mean discs removed (#)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_removed_vs_D_noise.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

def plot_multipleD(selected_D, mu_values, jamMu_data, removed_data):
    plt.figure()
    for D in selected_D:
        plt.plot(mu_values, [p*100 for p in jamMu_data[D]['prob']], marker='o', label=f'D={D:.2f}')
    plt.xlabel('Friction coefficient mu', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Jam probability (%)', fontsize=17)  # tamaño letra eje Y
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_jam_vs_mu_multiple_D_updated.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

    plt.figure()
    for D in selected_D:
        plt.plot(mu_values, removed_data[D]['removed'], marker='o', label=f'D={D:.2f}')
    plt.xlabel('Friction coefficient mu', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Mean discs removed (#)', fontsize=17)  # tamaño letra eje Y
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_removed_vs_mu_multiple_D_updated.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

def plot_remainingDiscs(mu_values, remaining_data):
    plt.figure()
    plt.plot(mu_values, remaining_data, marker='o')
    plt.xlabel('Friction coefficient mu', fontsize=17)  # tamaño letra eje X
    plt.ylabel('Discs remaining in silo (#)', fontsize=17)  # tamaño letra eje Y
    plt.grid(True)
    plt.savefig(f"{save_dir}/md_friction_vs_remaining.png")
    plt.xticks(fontsize=12)  # tamaño letra números eje X
    plt.yticks(fontsize=12)  # tamaño letra números eje Y
    plt.show()

if __name__ == "__main__":
    plot_jamD(data['D_values'], data['jam_prob_D'], data['removed_D'])
    plot_jamMu(data['mu_values'], data['jam_prob_mu'], data['removed_mu'])
    plot_jamDnoise(data['D_noise'], data['jam_prob_noise'], data['removed_noise'])
    plot_multipleD(data['selected_D'], data['mu_vals'], data['jam_prob_mu_multi'], data['removed_mu_multi'])
    plot_remainingDiscs(data['mu_range'], data['remaining'])