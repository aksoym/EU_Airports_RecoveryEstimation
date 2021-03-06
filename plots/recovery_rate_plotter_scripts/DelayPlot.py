import pandas as pd
import numpy as np
from functions import *

from scipy import interpolate
import matplotlib.pyplot as plt

file = "../csv/201805"
date = 20180529
tw = 61
aiport_plot_name = 'Schiphol'
single_airport_code = ['EHAM']
apt_code = single_airport_code[0]
regulation = 'CB Ext.'
upper_limit = 2.0
legend_location = 'upper left'

date_str = str(date)

df_flights = pd.read_csv(file + "/" + str(date) + ".csv")
apt_df_filtered = pd.read_csv("../misc_data/airportFiltered.csv", index_col=0)

differential_timestep = 1/2

plot_date = f'{date_str[-2:]}-{date_str[-4:-2]}-{date_str[:4]}'

df_subflights, df_subsubflights = dfFlights_twFilter(tw, df_flights)
df_flow, airportList = flightFlow(apt_df_filtered, df_subsubflights)

central_apt_list = get_apt_centrality_list(df_flow, n_largest=10)


central_apt_delays = get_delays_around_tw(df_flights, central_apt_list, tw)
all_apt_delays = get_delays_around_tw(df_flights, airportList, tw)
single_delay = get_delays_around_tw(df_flights, single_airport_code, tw)

infection_rates, recovery_rates = get_rates_around_tw(df_flights, airportList, tw, apt_df_filtered)


delay_probs = get_diff_probs(infection_rates, recovery_rates, airportList, differential_timestep,
                             tw, df_flights)

central_apt_delay_probs = delay_probs.loc[central_apt_list, :].mean(axis=0).values
all_apt_delay_probs = delay_probs.mean(axis=0).values
single_apt_delay_prob = delay_probs.loc[single_airport_code, :].mean(axis=0).values
print(single_apt_delay_prob)

x_input = np.arange(-3, 6)
x_input_for_probs = np.linspace(-3, 5, int(3*(1/differential_timestep + 1)))

fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4.5), dpi=200)

ax.plot(x_input, central_apt_delays, label='Central Airports (top 10)', c='m',
        linewidth=2)
ax.plot(x_input, single_delay, label=f'{aiport_plot_name} Airport', c='g',
        linewidth=2)
ax2.plot(x_input_for_probs, central_apt_delay_probs, label='Central Airports (top 10)', c='m',
         linewidth=2)
ax2.plot(x_input_for_probs, single_apt_delay_prob, label=f'{aiport_plot_name} Airport', c='g',
         linewidth=2)

ax.plot(x_input, all_apt_delays, label='All Airports', c='c', linewidth=2)
ax2.plot(x_input_for_probs, all_apt_delay_probs, label='All Airports', c='c', linewidth=2)

ax.set_xlabel('Relative Time (h)')
ax2.set_xlabel('Relative Time (h)')

ax.set_ylabel('Norm. Avg. Delay per Flight (min/60min)')
ax2.set_ylabel('Avg. Delay per Flight as Probability')


arrow = {'arrowstyle':'simple'}
ax.annotate('Regulation t_0', xy=(0, 0.02), xytext=(-2, 0.02), arrowprops=arrow)
ax.annotate(f"{np.round(recovery_rates[0][apt_code], 3)}", xy=(-1.8, 1.4))
ax.annotate(f"{np.round(recovery_rates[1][apt_code], 3)}", xy=(1.2, 1.4))
ax.annotate(f"{np.round(recovery_rates[2][apt_code], 3)}", xy=(4.2, 1.4))
ax.axvline(x=0, c='r', linestyle='--')
ax2.axvline(x=0, c='r', linestyle='--')

xmax = 0.10
xmin = 0.45
alpha = 0.25
#ax.axhline(y=central_apt_delays_from_spline[0], xmax=xmax, c='m', linestyle='--', alpha=alpha)
#ax.axhline(y=np.mean(central_apt_delays_from_spline[20:35]), xmin=xmin, c='m', linestyle='--', alpha=alpha)

ax2.axhline(y=central_apt_delay_probs[0], xmax=xmax, c='m', linestyle='--', alpha=alpha)
ax2.axhline(y=np.mean(central_apt_delay_probs[20:35]), xmin=xmin, c='m', linestyle='--', alpha=alpha)

#ax.axhline(y=all_apt_delays_from_spline[0], xmax=xmax, c='c', linestyle='--', alpha=alpha)
#ax.axhline(y=np.mean(all_apt_delays_from_spline[20:35]), xmin=xmin,c='c', linestyle='--', alpha=alpha)

ax2.axhline(y=all_apt_delay_probs[0], xmax=xmax, c='c', linestyle='--', alpha=alpha)
ax2.axhline(y=np.mean(all_apt_delay_probs[20:35]), xmin=xmin,c='c', linestyle='--', alpha=alpha)


ax.set_xticks(np.arange(-3, 6, 1.0))
ax2.set_xticks(np.arange(-3, 6, 1.0))
ax.set_title(f'{regulation} reg. on {aiport_plot_name} Airport ({plot_date}, tw:{str(tw)})')
ax2.set_title(f'Epidemic Model Probability Output ({plot_date}, tw:{str(tw)})')
ax.set_ylim(0, upper_limit)
ax2.set_ylim(0, upper_limit)
ax.legend(loc=legend_location)
ax2.legend(loc=legend_location)
ax.tick_params(right=True, labelright=True)
ax2.tick_params(right=True, labelright=True)
fig.suptitle('Comparison of Actual Avg. Delay and Inferred Probability')
#plt.savefig(f'prob_plots_w_recovery_rates/{aiport_plot_name}_{plot_date}_{tw}_w_recovery_rates.png')
plt.show()

