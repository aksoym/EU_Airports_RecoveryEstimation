import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



pickle_path = 'Schiphol_recovery_rates_dict_w_7masks.pickle'
apt_name = pickle_path.split('_')[0]

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)

mask_names = [key for key in recovery_rate_dict.keys() if recovery_rate_dict[key]]
alpha = 0.4
marker_list = ['x', 'o', 's', '^', '.', '*', 'X']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=200)
for idx, name in enumerate(mask_names):
    values = list(recovery_rate_dict[name].values())
    values_np_array = np.concatenate(values)
    ax.scatter([name]*len(values_np_array), values_np_array, label=name, marker='.', linewidth=0.75, alpha=0.8)

    ax.boxplot(values_np_array, positions=[idx], labels=[name], sym="", boxprops={'alpha':alpha},
               whiskerprops={'alpha':alpha}, capprops={'alpha':alpha})

plt.title(f'Recovery Rates for Different Regulation Causes ({apt_name} Airport)')
plt.ylabel('Recovery Rate')
plt.xlabel('Regulation Type')
plt.legend()
plt.savefig(f'{apt_name}_recoveryRates_{len(mask_names)}masks.png')
plt.show()
