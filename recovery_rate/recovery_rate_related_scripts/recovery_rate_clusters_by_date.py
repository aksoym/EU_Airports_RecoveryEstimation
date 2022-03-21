import numpy as np
import pickle
import matplotlib.pyplot as plt


def rolling_avg(x, interval=5):
    if type(x) != np.ndarray:
        x = np.array(x)

    q, remainder = divmod(len(x), interval)

    y = x[:(q-1)*interval].reshape(-1, interval).mean(axis=1).tolist()

    z = [x[(q-1)*interval:].mean()]

    return y + z

pickle_path = 'recoveryRate_pickles/Schiphol_allTime_recoveryRates.pickle'
#apt_name = pickle_path.split('_')[0]
apt_name = 'Schiphol'

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)



datetime_object_list = np.array([key[0] for key in recovery_rate_dict['Unmasked'].keys()])
date_strings_list = np.datetime_as_string(datetime_object_list)




recovery_rate_values = []
mean_recovery_rate_values = []
for recovery_rate_array in  recovery_rate_dict['Unmasked'].values():
    mean_recovery_rate_values.append(np.nanmean(recovery_rate_array))
    np_array_with_nans_removed = recovery_rate_array[~np.isnan(recovery_rate_array)]
    recovery_rate_list = np_array_with_nans_removed.tolist()
    recovery_rate_values.append(recovery_rate_list)




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), dpi=300)
ax.tick_params(axis='x', labelsize=8, labelrotation=55)

for idx, recovery_list in enumerate(recovery_rate_values):
    ax.scatter([date_strings_list[idx]]*len(recovery_list), recovery_list, s=15, color='c')

ax.plot(date_strings_list[:180:5], rolling_avg(mean_recovery_rate_values, 5))
fig_name = 'schiphol_rr_w_date'
plt.savefig('RR_plots_with_date/' + fig_name + '.png')

