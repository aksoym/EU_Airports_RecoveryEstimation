import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt




pickle_path = 'recoveryRate_pickles/Schiphol_allTime_recoveryRates.pickle'
#apt_name = pickle_path.split('_')[0]
apt_name = 'Schiphol'

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
plt.savefig(f'{apt_name}_recoveryRates_allTime.png')

plt.show()
print(len(recovery_rate_dict), recovery_rate_dict.keys(), len(recovery_rate_dict['Unmasked']))
