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


mask_names = ['CB', 'Fog', 'Snow', 'Rain', 'Thunder', 'Wind', 'Capacity']


pickle_path = 'recoveryRate_pickles/Munich_allTime_recoveryRates.pickle'
#apt_name = pickle_path.split('_')[0]
apt_name = 'Munich'

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)



datetime_object_list = np.array([key[0] for key in recovery_rate_dict['Unmasked'].keys()])
date_strings_list = np.datetime_as_string(datetime_object_list)

regulation_date_index = dict()
for name in mask_names:
    regulation_dates_list = [key[0] for key in recovery_rate_dict[name].keys()]
    regulation_date_index[name] = regulation_dates_list





recovery_rate_values = []
mean_recovery_rate_values = []
for recovery_rate_array in  recovery_rate_dict['Unmasked'].values():
    mean_recovery_rate_values.append(np.nanmean(recovery_rate_array))
    np_array_with_nans_removed = recovery_rate_array[~np.isnan(recovery_rate_array)]
    recovery_rate_list = np_array_with_nans_removed.tolist()
    recovery_rate_values.append(recovery_rate_list)




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5), dpi=600)
ax.tick_params(axis='x', labelsize=4, labelrotation=55)
color_code = {'CB':'black',
              'Fog': 'r',
              'Snow': 'g',
              'Rain': 'fuchsia',
              'Thunder': 'orange',
              'Wind': 'blueviolet',
              'Capacity': 'saddlebrown'}


for idx, recovery_list in enumerate(recovery_rate_values):
    date = date_strings_list[idx]

    for mask_name, date_list in regulation_date_index.items():
        if date in date_list:
            color = color_code[mask_name]
            label = mask_name
            size = 15
            break
        else:
            color = 'tab:blue'
            label = 'No Regulation'
            size=12


    ax.scatter([date_strings_list[idx]]*len(recovery_list), recovery_list, s=size, color=color, label=label)

ax.plot(date_strings_list[:180:5], rolling_avg(mean_recovery_rate_values, 5), linestyle='dashed',
        color='olive', label='Trendline', alpha=0.8)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size':8})
ax.set_xlabel('Date')
ax.set_ylabel('Recovery Rate')
ax.set_title(f'Recovery Rates of {apt_name} Over 6 Months')
plt.gcf().subplots_adjust(bottom=0.15)
fig_name = f'{apt_name}_rr_w_date'
plt.savefig('RR_plots_with_date/' + fig_name + '.png', dpi=600)

