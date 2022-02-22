from functions import *


def capacity(airport_code, tw, df_flights):
    """

    :param airport_code:
    :param tw:
    :param df_flights:
    :return: returns tuple (avg_capacity, total_capacity)
    """
    capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ctfmArr_tw'] == tw + 0)
                                         | (df_flights['ctfmArr_tw'] == tw + 1)
                                         | (df_flights['ctfmArr_tw'] == tw + 2)
                                         | (df_flights['ctfmArr_tw'] == tw + 3))].__len__()
    return capacity


def get_airport_specific_demand(airport_code, df_subflights, apt_df_filtered):
    airportList = list(apt_df_filtered.index)
    airportList.append("XXXX")

    df_subflights_aggApt = df_subflights.copy(deep=True)
    df_subflights_aggApt.loc[~df_subflights["dep"].isin(airportList)] = df_subflights_aggApt.loc[
        ~df_subflights["dep"].isin(airportList)].assign(dep="XXXX")
    df_subflights_aggApt.loc[~df_subflights["arr"].isin(airportList)] = df_subflights_aggApt.loc[
        ~df_subflights["arr"].isin(airportList)].assign(arr="XXXX")


    return len(df_subflights_aggApt.loc[df_subflights_aggApt['arr'] == airport_code])




##DEPRECATED UNTIL FURTHER NOTICE.
# def demand(airport_code, tw, df_flights, apt_df_filtered):
#     """
#
#     :param airport_code:
#     :param tw:
#     :param df_flights:
#     :return: returns tuple (avg_demand, total_demand)
#     """
#
#     df_subflights, _ = dfFlights_twFilter(tw, df_flights)
#
#     #Get demand for 3 hours.
#     demand = get_airport_specific_demand(airport_code, df_subflights, apt_df_filtered)
#
#     #Divide by 3 to get the hourly demand.
#     avg_demand = demand / 3
#
#
#     return avg_demand


def demand(airport_code, tw, df_flights):
    """

    :param airport_code:
    :param tw:
    :param df_flights:
    :return: returns demand
    """
    demand = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 0)
                                         | (df_flights['ftfmArr_tw'] == tw + 1)
                                         | (df_flights['ftfmArr_tw'] == tw + 2)
                                         | (df_flights['ftfmArr_tw'] == tw + 3))].__len__()
    return demand


def outflow(airport_code, tw, df_flights):
    """

    :param airport_code:
    :param tw:
    :param df_flights:
    :return: outflow
    """

    outflow = df_flights[(df_flights['arr'] == airport_code)
                        & ((df_flights['ctfmDep_tw'] == tw + 0)
                           | (df_flights['ctfmDep_tw'] == tw + 1)
                           | (df_flights['ctfmDep_tw'] == tw + 2)
                           | (df_flights['ctfmDep_tw'] == tw + 3))].__len__()

    return outflow
