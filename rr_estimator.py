# Keras Libraries
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
#from keras.optimizers import Adam, SGD
from keras import backend as k
from keras import regularizers
#from keras.utils import multi_gpu_model
import tensorflow as tf
import random as rn
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#sess = tf.Session(graph=tf.get_default_graph())
#k.set_session(sess)



combined_csv = pd.read_csv("recoveryRate_data/15minDelayLimit/combined_data/combined_csv_withReg.csv")

combined_csv.fillna(0, inplace = True)
combined_csv_filter = combined_csv[combined_csv["airport"] != "XXXX"]
combined_csv_filter0 = combined_csv[(combined_csv["recovery_rates"] != 0) & (combined_csv["airport"] != "XXXX")]


featureList = ["avgDelayPerFlight_0", "avgDelayPerFlight_L_event", "total_b*d0/60", 'time_window', 'outflow', 'inflow', 'totalDelay_0',
               'EGLL', 'LFPG', 'EHAM', 'LTBA', 'EDDF', 'LEMD', 'EDDM', 'LEBL', 'LIRF', 'LSZH', 'EGKK', 'EKCH', 'ENGM', 'ESSA', 'LFPO', 'LTFJ', 'LPPT', 'EBBR', 'EIDW', 'LOWW', 'EFHK', 'LSGG', 'EGSS', 'EDDL', 'LIMC', 'EGCC', 'EPWA', 'UUEE', 'LGAV', 'GCLP', 'LLBG', 'EDDT', 'OMDB', 'EDDK', 'EDDH', 'LTAC', 'LKPR', 'EGGW', 'EGPH', 'LIML', 'LROP', 'LHBP', 'OTHH', 'LFMN', 'LFLL', 'GMMN', 'EDDS', 'EGBB', 'LEMG', 'LFBO', 'EDDB', 'KJFK', 'LIME', 'LPPR', 'LFML', 'UKBB', 'LEPA', 'EDDP', 'EGPF', 'GCTS', 'LTBJ', 'EGLC', 'ENBR', 'EVRA', 'ELLX', 'ESGG', 'LFSB', 'LEAL', 'LIPZ', 'LIPE', 'GCXO', 'LTAI', 'EGNX', 'EGPD', 'GCRR', 'UUDD', 'LIRN', 'LEVC', 'EGGD', 'LYBE', 'LICC', 'ENZV', 'UBBB', 'OLBA', 'LBSF', 'LFBD', 'GCFV', 'EDDV', 'ENVA', 'LIRA', 'LFPB', 'EPKK', 'LFRS', 'EBCI', 'ESSB', 'DAAG', 'UUWW', 'LEZL', 'ENTC', 'EGAA', 'KEWR', 'EDDN', 'EBLG', 'GMMX', 'LIMF', 'LMML', 'OMAA', 'EYVI', 'HECA', 'EPGD', 'VHHH', 'DTTA', 'LICJ', 'LCLK', 'ENBO', 'EKBI', 'EETN', 'LEBB', 'ULLI', 'ZSPD', 'BIKF', 'EGHI', 'OEJN', 'EGNT', 'LTAF', 'VTBS', 'LDZA', 'EGGP', 'EHEH', 'LGTS', 'ZBAA', 'EGAC', 'UGTB',
               'EGLL_d0', 'LFPG_d0', 'EHAM_d0', 'LTBA_d0', 'EDDF_d0', 'LEMD_d0', 'EDDM_d0', 'LEBL_d0', 'LIRF_d0', 'LSZH_d0', 'EGKK_d0', 'EKCH_d0', 'ENGM_d0', 'ESSA_d0', 'LFPO_d0', 'LTFJ_d0', 'LPPT_d0', 'EBBR_d0', 'EIDW_d0', 'LOWW_d0', 'EFHK_d0', 'LSGG_d0', 'EGSS_d0', 'EDDL_d0', 'LIMC_d0', 'EGCC_d0', 'EPWA_d0', 'UUEE_d0', 'LGAV_d0', 'GCLP_d0', 'LLBG_d0', 'EDDT_d0', 'OMDB_d0', 'EDDK_d0', 'EDDH_d0', 'LTAC_d0', 'LKPR_d0', 'EGGW_d0', 'EGPH_d0', 'LIML_d0', 'LROP_d0', 'LHBP_d0', 'OTHH_d0', 'LFMN_d0', 'LFLL_d0', 'GMMN_d0', 'EDDS_d0', 'EGBB_d0', 'LEMG_d0', 'LFBO_d0', 'EDDB_d0', 'KJFK_d0', 'LIME_d0', 'LPPR_d0', 'LFML_d0', 'UKBB_d0', 'LEPA_d0', 'EDDP_d0', 'EGPF_d0', 'GCTS_d0', 'LTBJ_d0', 'EGLC_d0', 'ENBR_d0', 'EVRA_d0', 'ELLX_d0', 'ESGG_d0', 'LFSB_d0', 'LEAL_d0', 'LIPZ_d0', 'LIPE_d0', 'GCXO_d0', 'LTAI_d0', 'EGNX_d0', 'EGPD_d0', 'GCRR_d0', 'UUDD_d0', 'LIRN_d0', 'LEVC_d0', 'EGGD_d0', 'LYBE_d0', 'LICC_d0', 'ENZV_d0', 'UBBB_d0', 'OLBA_d0', 'LBSF_d0', 'LFBD_d0', 'GCFV_d0', 'EDDV_d0', 'ENVA_d0', 'LIRA_d0', 'LFPB_d0', 'EPKK_d0', 'LFRS_d0', 'EBCI_d0', 'ESSB_d0', 'DAAG_d0', 'UUWW_d0', 'LEZL_d0', 'ENTC_d0', 'EGAA_d0', 'KEWR_d0', 'EDDN_d0', 'EBLG_d0', 'GMMX_d0', 'LIMF_d0', 'LMML_d0', 'OMAA_d0', 'EYVI_d0', 'HECA_d0', 'EPGD_d0', 'VHHH_d0', 'DTTA_d0', 'LICJ_d0', 'LCLK_d0', 'ENBO_d0', 'EKBI_d0', 'EETN_d0', 'LEBB_d0', 'ULLI_d0', 'ZSPD_d0', 'BIKF_d0', 'EGHI_d0', 'OEJN_d0', 'EGNT_d0', 'LTAF_d0', 'VTBS_d0', 'LDZA_d0', 'EGGP_d0', 'EHEH_d0', 'LGTS_d0', 'ZBAA_d0', 'EGAC_d0', 'UGTB_d0']


x_origin = np.column_stack(([combined_csv_filter0[i] for i in featureList]))
y_origin = np.transpose(np.column_stack((combined_csv_filter0['recovery_rates'])))


x_train, x_test, y_train, y_test = train_test_split(x_origin, y_origin, test_size=0.0001, random_state=27, shuffle=True)


scaler1=StandardScaler()
scaler2=StandardScaler()

sc1 = scaler1.fit( x_train )
x_train_sc = sc1.transform( x_train )
x_test_sc = sc1.transform( x_test )

sc2 = scaler2.fit( y_train )
y_train_sc = sc2.transform( y_train )
y_test_sc = sc2.transform( y_test )





apt_df_filtered = pd.read_csv("misc_data/airportFiltered.csv", index_col=0)
airportList = apt_df_filtered.index
airportList = list(airportList)
airportList.append("XXXX")

date_ = 20180311
tw = 12
# used = ["avgDelayPerFlight_0", "avgDelayPerFlight_event", "total_b*d0/60"]
used = ["avgDelayPerFlight_0", "avgDelayPerFlight_L_event", "total_b*d0/60",
        'time_window', 'outflow', 'inflow', 'totalDelay_0',
        'EGLL', 'LFPG', 'EHAM', 'LTBA', 'EDDF', 'LEMD', 'EDDM', 'LEBL', 'LIRF', 'LSZH', 'EGKK', 'EKCH', 'ENGM', 'ESSA', 'LFPO', 'LTFJ', 'LPPT', 'EBBR', 'EIDW', 'LOWW', 'EFHK', 'LSGG', 'EGSS', 'EDDL', 'LIMC', 'EGCC', 'EPWA', 'UUEE', 'LGAV', 'GCLP', 'LLBG', 'EDDT', 'OMDB', 'EDDK', 'EDDH', 'LTAC', 'LKPR', 'EGGW', 'EGPH', 'LIML', 'LROP', 'LHBP', 'OTHH', 'LFMN', 'LFLL', 'GMMN', 'EDDS', 'EGBB', 'LEMG', 'LFBO', 'EDDB', 'KJFK', 'LIME', 'LPPR', 'LFML', 'UKBB', 'LEPA', 'EDDP', 'EGPF', 'GCTS', 'LTBJ', 'EGLC', 'ENBR', 'EVRA', 'ELLX', 'ESGG', 'LFSB', 'LEAL', 'LIPZ', 'LIPE', 'GCXO', 'LTAI', 'EGNX', 'EGPD', 'GCRR', 'UUDD', 'LIRN', 'LEVC', 'EGGD', 'LYBE', 'LICC', 'ENZV', 'UBBB', 'OLBA', 'LBSF', 'LFBD', 'GCFV', 'EDDV', 'ENVA', 'LIRA', 'LFPB', 'EPKK', 'LFRS', 'EBCI', 'ESSB', 'DAAG', 'UUWW', 'LEZL', 'ENTC', 'EGAA', 'KEWR', 'EDDN', 'EBLG', 'GMMX', 'LIMF', 'LMML', 'OMAA', 'EYVI', 'HECA', 'EPGD', 'VHHH', 'DTTA', 'LICJ', 'LCLK', 'ENBO', 'EKBI', 'EETN', 'LEBB', 'ULLI', 'ZSPD', 'BIKF', 'EGHI', 'OEJN', 'EGNT', 'LTAF', 'VTBS', 'LDZA', 'EGGP', 'EHEH', 'LGTS', 'ZBAA', 'EGAC', 'UGTB',
        'EGLL_d0', 'LFPG_d0', 'EHAM_d0', 'LTBA_d0', 'EDDF_d0', 'LEMD_d0', 'EDDM_d0', 'LEBL_d0', 'LIRF_d0', 'LSZH_d0', 'EGKK_d0', 'EKCH_d0', 'ENGM_d0', 'ESSA_d0', 'LFPO_d0', 'LTFJ_d0', 'LPPT_d0', 'EBBR_d0', 'EIDW_d0', 'LOWW_d0', 'EFHK_d0', 'LSGG_d0', 'EGSS_d0', 'EDDL_d0', 'LIMC_d0', 'EGCC_d0', 'EPWA_d0', 'UUEE_d0', 'LGAV_d0', 'GCLP_d0', 'LLBG_d0', 'EDDT_d0', 'OMDB_d0', 'EDDK_d0', 'EDDH_d0', 'LTAC_d0', 'LKPR_d0', 'EGGW_d0', 'EGPH_d0', 'LIML_d0', 'LROP_d0', 'LHBP_d0', 'OTHH_d0', 'LFMN_d0', 'LFLL_d0', 'GMMN_d0', 'EDDS_d0', 'EGBB_d0', 'LEMG_d0', 'LFBO_d0', 'EDDB_d0', 'KJFK_d0', 'LIME_d0', 'LPPR_d0', 'LFML_d0', 'UKBB_d0', 'LEPA_d0', 'EDDP_d0', 'EGPF_d0', 'GCTS_d0', 'LTBJ_d0', 'EGLC_d0', 'ENBR_d0', 'EVRA_d0', 'ELLX_d0', 'ESGG_d0', 'LFSB_d0', 'LEAL_d0', 'LIPZ_d0', 'LIPE_d0', 'GCXO_d0', 'LTAI_d0', 'EGNX_d0', 'EGPD_d0', 'GCRR_d0', 'UUDD_d0', 'LIRN_d0', 'LEVC_d0', 'EGGD_d0', 'LYBE_d0', 'LICC_d0', 'ENZV_d0', 'UBBB_d0', 'OLBA_d0', 'LBSF_d0', 'LFBD_d0', 'GCFV_d0', 'EDDV_d0', 'ENVA_d0', 'LIRA_d0', 'LFPB_d0', 'EPKK_d0', 'LFRS_d0', 'EBCI_d0', 'ESSB_d0', 'DAAG_d0', 'UUWW_d0', 'LEZL_d0', 'ENTC_d0', 'EGAA_d0', 'KEWR_d0', 'EDDN_d0', 'EBLG_d0', 'GMMX_d0', 'LIMF_d0', 'LMML_d0', 'OMAA_d0', 'EYVI_d0', 'HECA_d0', 'EPGD_d0', 'VHHH_d0', 'DTTA_d0', 'LICJ_d0', 'LCLK_d0', 'ENBO_d0', 'EKBI_d0', 'EETN_d0', 'LEBB_d0', 'ULLI_d0', 'ZSPD_d0', 'BIKF_d0', 'EGHI_d0', 'OEJN_d0', 'EGNT_d0', 'LTAF_d0', 'VTBS_d0', 'LDZA_d0', 'EGGP_d0', 'EHEH_d0', 'LGTS_d0', 'ZBAA_d0', 'EGAC_d0', 'UGTB_d0']



# Load the model with best results
model_name = 'test05_NeuralNetwork.hdf5'
path = 'neuralNetwork_models/' + model_name

model = load_model(path)

# Make the prediction
x_test_apt = np.array(combined_csv_filter0[(combined_csv_filter0["date"] == float(date_)) & (combined_csv_filter0["time_window"] == tw)]["airport"])
x_test = np.array(combined_csv_filter0[(combined_csv_filter0["date"] == float(date_)) & (combined_csv_filter0["time_window"] == tw)][used])
y_test = np.array(combined_csv_filter0[(combined_csv_filter0["date"] == float(date_)) & (combined_csv_filter0["time_window"] == tw)]["recovery_rates"])


y_predicted = model.predict(x_test)

print(y_predicted)
print(airportList)