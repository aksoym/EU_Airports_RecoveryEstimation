from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def get_apt_centrality_list(df_flow, n_largest=10):
    flow_graph = nx.from_pandas_adjacency(df_flow.iloc[0:-1, 0:-1], create_using=nx.DiGraph)
    centrality = nx.eigenvector_centrality_numpy(flow_graph)
    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
    central_apt_list = centrality_df.nlargest(n_largest, 0).index.values.tolist()
    return central_apt_list

def get_rates_around_tw(df_flights, airportList, tw, apt_df_filtered):
    prev_section_flights, _ = dfFlights_twFilter(tw - 12, df_flights)
    init_section_flights, _ = dfFlights_twFilter(tw, df_flights)
    post_section_flights, _ = dfFlights_twFilter(tw + 12, df_flights)

    prev_section_flow, _ = flightFlow(apt_df_filtered, prev_section_flights)
    init_section_flow, _ = flightFlow(apt_df_filtered, init_section_flights)
    post_section_flow, _ = flightFlow(apt_df_filtered, post_section_flights)

    prev_inf_rates = generateInfectionRates(prev_section_flow, airportList)
    init_inf_rates = generateInfectionRates(init_section_flow, airportList)
    post_inf_rates = generateInfectionRates(post_section_flow, airportList)

    prev_recovery_rates = generateRecoveryRates_Delay(prev_inf_rates,
                                                      calculateAirportDelays(df_flights,
                                                                             airportList,
                                                                             tw - 12))
    init_recovery_rates = generateRecoveryRates_Delay(init_inf_rates,
                                                      calculateAirportDelays(df_flights,
                                                                             airportList,
                                                                             tw))

    post_recovery_rates = generateRecoveryRates_Delay(post_inf_rates,
                                                      calculateAirportDelays(df_flights,
                                                                             airportList,
                                                                             tw + 12))

    return (prev_inf_rates, init_inf_rates, post_inf_rates), \
           (prev_recovery_rates, init_recovery_rates, post_recovery_rates)

def get_delays_around_tw(df_flights, airportList, tw):
    args = (df_flights, airportList)
    prev_section = calculateAirportDelays(*args, tw-12)
    initial_section = calculateAirportDelays(*args, tw)
    post_section = calculateAirportDelays(*args, tw+12)

    sections = [prev_section, initial_section, post_section]

    delay_values = np.concatenate([
        each_section.loc[:, ['d_0_avg15', 'd_M_avg15', 'd_L_avg15']].mean().values for each_section in sections
    ])

    return delay_values

def dfFlights_twFilter(tw, df_flights):
    ## For filtering the data according to the selected time windows

    df_subflights = df_flights.loc[(df_flights["ftfmDep_tw"] == (tw)) | (df_flights["ftfmArr_tw"] == (tw)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 1)) | (df_flights["ftfmArr_tw"] == (tw + 1)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 2)) | (df_flights["ftfmArr_tw"] == (tw + 2)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 3)) | (df_flights["ftfmArr_tw"] == (tw + 3)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 8)) | (df_flights["ftfmArr_tw"] == (tw + 8)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 9)) | (df_flights["ftfmArr_tw"] == (tw + 9)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 10)) | (df_flights["ftfmArr_tw"] == (tw + 10)) |
                                   (df_flights["ftfmDep_tw"] == (tw + 11)) | (df_flights["ftfmArr_tw"] == (tw + 11))]

    df_subsubflights = df_subflights.loc[(df_flights["ftfmDep_tw"] == (tw)) |
                                         (df_flights["ftfmDep_tw"] == (tw + 1)) |
                                         (df_flights["ftfmDep_tw"] == (tw + 2)) |
                                         (df_flights["ftfmDep_tw"] == (tw + 3))]

    return df_subflights, df_subsubflights


def get_diff_probs(infection_rates, recovery_rates, airportList,
                   differential_timestep, tw, df_flights):
    tw_list = [tw - 12, tw, tw + 12]
    diff_delays_list = []
    for i, tw in enumerate(tw_list):
        airportDelays = calculateAirportDelays(df_flights, airportList, tw)
        diff_delays = diffSISModel_Delay(recovery_rates[i], infection_rates[i],
                                         airportDelays, airportList, differential_timestep)
        diff_delays_list.append(diff_delays)

    return pd.concat(diff_delays_list, axis=1)


def flightFlow(apt_df_filtered, df_subflights):
    ## For finding the number of each OD pairs and obtaining the flow matrix

    airportList = list(apt_df_filtered.index)
    airportList.append("XXXX")

    df_subflights_aggApt = df_subflights.copy(deep=True)
    df_subflights_aggApt.loc[~df_subflights["dep"].isin(airportList)] = df_subflights_aggApt.loc[
        ~df_subflights["dep"].isin(airportList)].assign(dep="XXXX")
    df_subflights_aggApt.loc[~df_subflights["arr"].isin(airportList)] = df_subflights_aggApt.loc[
        ~df_subflights["arr"].isin(airportList)].assign(arr="XXXX")

    #     df_flow = pd.DataFrame(index = airportList, columns = airportList)

    #     for dep in airportList:
    #         for arr in airportList:
    #             df_flow[dep][arr] = len(df_subflights_aggApt.loc[(df_subflights_aggApt["dep"] == dep) & (df_subflights_aggApt["arr"] == arr)])

    ## For paralllelization
    def flowParallel(dep, arr):
        return len(
            df_subflights_aggApt.loc[(df_subflights_aggApt["dep"] == dep) & (df_subflights_aggApt["arr"] == arr)])

    countList = Parallel(n_jobs=-1)(delayed(flowParallel)(dep, arr) for arr in airportList for dep in airportList)
    countData = np.array(countList).reshape((len(airportList), len(airportList)))
    df_flow = pd.DataFrame(data=countData, index=airportList, columns=airportList)

    return df_flow, airportList


def generateRecoveryRates_Delay(infectionRates, df_aptDelayVal):
    ## For generating recovery rates

    df_aptDelayVal["d_diff_avg15"] = df_aptDelayVal["d_L_avg15"] - df_aptDelayVal["d_0_avg15"]
    weightedInfectionMat = infectionRates.dot(df_aptDelayVal["d_0_avg15"])
    recoveryRates = weightedInfectionMat - weightedInfectionMat * df_aptDelayVal["d_0_avg15"] - df_aptDelayVal[
        "d_diff_avg15"]

    recoveryRates[df_aptDelayVal["d_0_avg15"] == 0] = None
    recoveryRates[df_aptDelayVal["d_0_avg15"] != 0] = recoveryRates[df_aptDelayVal["d_0_avg15"] != 0] / \
                                                      df_aptDelayVal["d_0_avg15"][df_aptDelayVal["d_0_avg15"] != 0]

    return recoveryRates


def calculateAirportDelays(df_flights, airportList, tw):
    ## For extracting the total/average delays at airports

    def parallelDelayCount(apt):
        # P 0 (starting point)
        df_subflights_twApt_dep1 = df_flights.loc[((df_flights["dep"] == apt)) &
                                                  ((df_flights["ftfmDep_tw"] == (tw)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 1)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 2)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 3)))]

        df_subflights_twApt_arr1 = df_flights.loc[((df_flights["arr"] == apt)) &
                                                  ((df_flights["ftfmArr_tw"] == (tw)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 1)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 2)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 3)))]

        df_subflights_twApt_dep1.loc[df_subflights_twApt_dep1["delayDep"] < 0] = 0
        df_subflights_twApt_arr1.loc[df_subflights_twApt_arr1["delayArr"] < 0] = 0

        df_subflights_twAptDelay_dep1 = df_subflights_twApt_dep1["delayDep"]
        df_subflights_twAptDelay_arr1 = df_subflights_twApt_arr1["delayArr"]

        df_subflights_twAptDelay_1 = df_subflights_twAptDelay_dep1.sum() + df_subflights_twAptDelay_arr1.sum()
        df_subflights_twApt_1 = (len(df_subflights_twApt_dep1) + len(df_subflights_twApt_arr1))

        df_subflights_twApt_1_15 = (df_subflights_twApt_1 * 60)

        if (df_subflights_twApt_1 == 0) or (df_subflights_twAptDelay_1 < 15):
            df_aptDelayVal_d0 = 0
            df_aptDelayVal_d0avg = 0
            df_aptDelayVal_d0avg15 = 0
        else:
            df_aptDelayVal_d0 = df_subflights_twAptDelay_1
            df_aptDelayVal_d0avg = df_subflights_twAptDelay_1 / df_subflights_twApt_1
            df_aptDelayVal_d0avg15 = df_subflights_twAptDelay_1 / df_subflights_twApt_1_15

        # P last (2 hour - simulation time)
        df_subflights_twApt_dep2 = df_flights.loc[((df_flights["dep"] == apt)) &
                                                  ((df_flights["ftfmDep_tw"] == (tw + 8)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 9)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 10)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 11)))]

        df_subflights_twApt_arr2 = df_flights.loc[((df_flights["arr"] == apt)) &
                                                  ((df_flights["ftfmArr_tw"] == (tw + 8)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 9)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 10)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 11)))]

        df_subflights_twApt_dep2.loc[df_subflights_twApt_dep2["delayDep"] < 0] = 0
        df_subflights_twApt_arr2.loc[df_subflights_twApt_arr2["delayArr"] < 0] = 0

        df_subflights_twAptDelay_dep2 = df_subflights_twApt_dep2["delayDep"]
        df_subflights_twAptDelay_arr2 = df_subflights_twApt_arr2["delayArr"]

        df_subflights_twAptDelay_2 = df_subflights_twAptDelay_dep2.sum() + df_subflights_twAptDelay_arr2.sum()
        df_subflights_twApt_2 = (len(df_subflights_twApt_dep2) + len(df_subflights_twApt_arr2))
        df_subflights_twApt_2_15 = (df_subflights_twApt_2 * 60)

        if (df_subflights_twApt_2 == 0) or (df_subflights_twAptDelay_2 < 15):
            df_aptDelayVal_dl = 0
            df_aptDelayVal_dlavg = 0
            df_aptDelayVal_dlavg15 = 0
        else:
            df_aptDelayVal_dl = df_subflights_twAptDelay_2
            df_aptDelayVal_dlavg = df_subflights_twAptDelay_2 / df_subflights_twApt_2
            df_aptDelayVal_dlavg15 = df_subflights_twAptDelay_2 / df_subflights_twApt_2_15

        # P middle (1 hour)
        df_subflights_twApt_depm = df_flights.loc[((df_flights["dep"] == apt)) &
                                                  ((df_flights["ftfmDep_tw"] == (tw + 4)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 5)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 6)) |
                                                   (df_flights["ftfmDep_tw"] == (tw + 7)))]

        df_subflights_twApt_arrm = df_flights.loc[((df_flights["arr"] == apt)) &
                                                  ((df_flights["ftfmArr_tw"] == (tw + 4)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 5)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 6)) |
                                                   (df_flights["ftfmArr_tw"] == (tw + 7)))]

        df_subflights_twApt_depm.loc[df_subflights_twApt_depm["delayDep"] < 0] = 0
        df_subflights_twApt_arrm.loc[df_subflights_twApt_arrm["delayArr"] < 0] = 0

        df_subflights_twAptDelay_depm = df_subflights_twApt_depm["delayDep"]
        df_subflights_twAptDelay_arrm = df_subflights_twApt_arrm["delayArr"]

        df_subflights_twAptDelay_m = df_subflights_twAptDelay_depm.sum() + df_subflights_twAptDelay_arrm.sum()
        df_subflights_twApt_m = (len(df_subflights_twApt_depm) + len(df_subflights_twApt_arrm))
        df_subflights_twApt_m_15 = (df_subflights_twApt_m * 60)

        if (df_subflights_twApt_m == 0) or (df_subflights_twAptDelay_m < 15):
            df_aptDelayVal_dm = 0
            df_aptDelayVal_dmavg = 0
            df_aptDelayVal_dmavg15 = 0
        else:
            df_aptDelayVal_dm = df_subflights_twAptDelay_m
            df_aptDelayVal_dmavg = df_subflights_twAptDelay_m / df_subflights_twApt_m
            df_aptDelayVal_dmavg15 = df_subflights_twAptDelay_m / df_subflights_twApt_m_15

        df_aptDelayVal = [df_aptDelayVal_d0, df_aptDelayVal_dl, df_aptDelayVal_dm, df_aptDelayVal_d0avg,
                          df_aptDelayVal_dlavg, df_aptDelayVal_dmavg, df_aptDelayVal_d0avg15, df_aptDelayVal_dlavg15,
                          df_aptDelayVal_dmavg15]

        return df_aptDelayVal

    df_aptDelayVal = Parallel(n_jobs=-1)(delayed(parallelDelayCount)(apt) for apt in airportList)

    df_aptDelayVal = pd.DataFrame(data=df_aptDelayVal, index=airportList,
                                  columns=["d_0", "d_L", "d_M", "d_0_avg", "d_L_avg", "d_M_avg", "d_0_avg15",
                                           "d_L_avg15", "d_M_avg15"])

    #     df_aptDelayVal.loc["XXXX"] = 0

    return df_aptDelayVal


def diffSISModel_Delay(recoveryRates, infectionRates, df_aptDelayVal, airportList, timeStep=1):
    ## For solving the differential equation of the SIS model

    #     timeStep = 1
    timeStepRev = 1 / timeStep

    df_probs = pd.DataFrame(0, index=airportList, columns=range(int(timeStepRev) + 1))
    df_probs[0] = df_aptDelayVal["d_0_avg15"]

    df_recoveryRates = pd.DataFrame(0, index=recoveryRates.index, columns=recoveryRates.index,
                                    dtype=recoveryRates.dtype)
    np.fill_diagonal(df_recoveryRates.values, recoveryRates)
    df_infectionRates = infectionRates

    p = df_aptDelayVal["d_0_avg15"]
    for i in range(1, int(timeStepRev) + 1):
        dot_p = (df_infectionRates - df_recoveryRates).dot(p) - (df_infectionRates.dot(p)) * p
        p = p + dot_p * timeStep
        p = p.fillna(0)
        p[p > 1] = 1
        p[p < 0] = 0
        df_probs[i] = p

    return df_probs


def generateInfectionRates(df_flow, airportList):
    ## For obtaining infection rates for the epidemic spreading model

    # Normalize by arrivals
    df_flowNorm = pd.DataFrame(index=airportList, columns=airportList)
    for arr in airportList:
        arr_sum = df_flow.sum()[arr]
        if arr_sum == 0:
            df_flowNorm[arr] = 0
        else:
            df_flowNorm[arr] = df_flow[arr] / arr_sum
        df_flowNorm.loc[arr]["XXXX"] = 0
    df_flowNorm = df_flowNorm.assign(XXXX=0)
    infectionRates = df_flowNorm

    return df_flowNorm


def plotEpidemic(df_probs_Delay, df_aptDelayVal, numApt):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Delay-based Model", fontsize=16)
    axs[0].title.set_text("Evaluated Fractions")
    axs[1].title.set_text("Real Fractions")
    axs[0].set_xlabel("Hours")
    axs[0].set_ylabel("Probability")
    axs[1].set_xlabel("Hours")
    axs[1].set_ylabel("Probability")
    axs[0].set_xlim([1, 3])
    axs[0].set_ylim([0, 0.8])
    axs[1].set_xlim([1, 3])
    axs[1].set_ylim([0, 0.8])
    for i in df_probs_Delay.index[:numApt]:
        axs[0].plot(np.linspace(1, 3, len(df_probs_Delay.columns)), df_probs_Delay.loc[i].values, label=i)
        axs[1].plot([1, 3], df_aptDelayVal[["d_0_avg15", "d_L_avg15"]].loc[i].values, label=i)
    #         axs[1].plot([1, 2, 3], df_aptDelayVal[["d_0_avg15", "d_M_avg15", "d_L_avg15"]].loc[i].values, label = i)

    axs[0].legend(loc="lower center", ncol=int(len(df_probs_Delay.index[:10]) / 2), bbox_to_anchor=(1, -.2))
    #     fig.savefig("Delay-based_results.png", dpi=900, bbox_inches="tight")
    plt.show()

def recoveryRatePipeline(df_flights, apt_df_filtered, tw):

    subflights, subsubflights = dfFlights_twFilter(tw, df_flights)
    flight_flow, airportList = flightFlow(apt_df_filtered, subflights)
    apt_delay_values = calculateAirportDelays(df_flights, airportList, tw)
    infectionRates = generateInfectionRates(flight_flow, airportList)
    recoveryRates = generateRecoveryRates_Delay(infectionRates, apt_delay_values)

    return recoveryRates
