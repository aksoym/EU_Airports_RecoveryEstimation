from functions import *



def capacity_change(airport_code, tw, df_flights):

    first_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 0)
                                         | (df_flights['ftfmArr_tw'] == tw + 1)
                                         | (df_flights['ftfmArr_tw'] == tw + 2)
                                         | (df_flights['ftfmArr_tw'] == tw + 3))].__len__()


    third_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 8)
                                         | (df_flights['ftfmArr_tw'] == tw + 9)
                                         | (df_flights['ftfmArr_tw'] == tw + 10)
                                         | (df_flights['ftfmArr_tw'] == tw + 11))].__len__()





    return third_hours_capacity - first_hours_capacity

def capacity(airport_code, tw, df_flights):
    """

    :param airport_code:
    :param tw:
    :param df_flights:
    :return: returns tuple (avg_capacity, total_capacity)
    """
    first_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 0)
                                         | (df_flights['ftfmArr_tw'] == tw + 1)
                                         | (df_flights['ftfmArr_tw'] == tw + 2)
                                         | (df_flights['ftfmArr_tw'] == tw + 3))].__len__()

    second_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 4)
                                         | (df_flights['ftfmArr_tw'] == tw + 5)
                                         | (df_flights['ftfmArr_tw'] == tw + 6)
                                         | (df_flights['ftfmArr_tw'] == tw + 7))].__len__()

    third_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 8)
                                         | (df_flights['ftfmArr_tw'] == tw + 9)
                                         | (df_flights['ftfmArr_tw'] == tw + 10)
                                         | (df_flights['ftfmArr_tw'] == tw + 11))].__len__()

    total_capacity = first_hours_capacity + second_hours_capacity + third_hours_capacity
    avg_capacity = total_capacity / 3

    return (avg_capacity, total_capacity)

def demand(airport_code, tw, df_flights, apt_df_filtered):
    """

    :param airport_code:
    :param tw:
    :param df_flights:
    :return: returns tuple (avg_demand, total_demand)
    """

    demand = 0
    for inner_tw in range(tw, tw+12):
        df_subflights, _ = dfFlights_twFilter(tw, df_flights)
        flow_matrix_df, _ = flightFlow(apt_df_filtered, df_subflights)
        demand += flow_matrix_df.loc[airport_code, :].sum()

    avg_demand = demand / 12
    total_demand = demand

    return (avg_demand, total_demand)