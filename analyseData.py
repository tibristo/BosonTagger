import numpy as np
from sklearn.externals import joblib
import os, sys, time
from sklearn.preprocessing import StandardScaler
import WTaggingPerf as ww
from create_folds import scalerNN
import client as apy  #the agilepack client
import yaml
import bz2
import math
import argparse
import numpy.lib.recfunctions as nf
import pickle
import root_numpy as rn
import ROOT as rt


def analyse(bdt_models, bdt_taggers, dnn_models, dnn_taggers, dnn_scaler, data_files):
    # using bdt_model
    # bdt_model.predict_proba(data)

    # need to scale the data to use the dnn
    # for i, v in enumerate(scaler.variables):
    #     data[v] = (data[v] - scaler.means[i]) / scaler.std[i]
    # This can be done on an event by event basis too
    # event.variable = event.variable - means[i]/ std[i]
    
    # using dnn_model
    # dnn_model.predict(data)[0]

    # The variables have different names in data, so we need to map the variable names to something the
    # models can use/ have the same names.
    bdt_var_map = {'Aplanarity':'jetTrimX_aplanarity', 'EEC_C2_1':'jetTrimX_c2beta1', 'EEC_D2_1':'jetTrimX_d2beta1', 'Sphericity':'jetTrimX_sphericity', 'SPLIT12': 'jetTrimX_groosplit12', 'TauWTA1':'jetTrimX_grootau1','TauWTA2':'jetTrimX_grootau2', 'TauWTA2TauWTA1':'jetTrimX_grootau21', 'Mu12':'jetTrimX_mufilt', 'yfilt': 'jetTrimX_ysfilt', 'y':'jetTrimX_y', 'nTracks':'jetTrimX_ungrngtrk', 'PlanarFlow': 'jetTrimX_planarflow'}
    # gotta get the order right for the variables
    bdt_vars_1 = []
    bdt_vars_2 = []
    for b in bdt_taggers:
        bdt_vars_1.append(bdt_var_map[b].replace('X','1'))
        bdt_vars_2.append(bdt_var_map[b].replace('X','2'))
    bdt_ivd = {}
    for k, v in bdt_var_map.items():
        bdt_ivd[v.replace('X','1')] = k
        bdt_ivd[v.replace('X','2')] = k

    
    dnn_var_map = {'aplanarity':'jetTrimX_aplanarity', 'eec_c2_1':'jetTrimX_c2beta1','eec_d2_1':'jetTrimX_d2beta1', 'sphericity':'jetTrimX_sphericity', 'split12': 'jetTrimX_groosplit12', 'tauwta1':'jetTrimX_grootau1','tauwta2':'jetTrimX_grootau2', 'tauwta2tauwta1':'jetTrimX_grootau21', 'mu12':'jetTrimX_mufilt', 'yfilt': 'jetTrimX_ysfilt', 'y':'jetTrimX_y', 'ntracks':'jetTrimX_ungrngtrk', 'planarflow':'jetTrimX_planarflow'}
    # gotta get the order right for the variables
    dnn_vars_1 = []
    dnn_vars_2 = []
    print 'dnn_taggers'
    print dnn_taggers
    for b in dnn_taggers:
        dnn_vars_1.append(dnn_var_map[b].replace('X','1'))
        dnn_vars_2.append(dnn_var_map[b].replace('X','2'))

    dnn_ivd = {}
    for k, v in dnn_var_map.items():
        dnn_ivd[v.replace('X','1')] = k
        dnn_ivd[v.replace('X','2')] = k

        
    # read in the data so we can analyse it!
    # gotta do it one at a time for each data file :(
    for data_file in data_files:
        # benchmark and try a few options
        # first read in data using root2array
        print data_file
        data_arr = rn.root2rec(data_file)

        # unfortunately, some of the values in the ntuples are NaNs!!!
        # keep track of which ones these are...
        nan_idx_tmp = np.empty([0])
        no_values_tmp = np.empty([0])
        for d in data_arr.dtype.names:
            if d not in dnn_vars_1 and d not in dnn_vars_2:
                continue

            if np.any(np.isnan(data_arr[d])) or not np.all(np.isfinite(data_arr[d])):
                if len(nan_idx_tmp) == 0:
                    nan_idx_tmp = np.asarray(np.where(np.isnan(data_arr[d])))[0]
                else:
                    nan_idx_tmp = np.concatenate((nan_idx_tmp, np.asarray(np.where(np.isnan(data_arr[d])))[0]))
                data_arr[d] = np.nan_to_num(data_arr[d])
                
            if len(no_values_tmp) == 0:
                no_values_tmp = np.asarray(np.where(data_arr[d]<-98))[0]
            else:
                no_values_tmp = np.concatenate((no_values_tmp, np.asarray(np.where(data_arr[d]<-98))[0]))
        
        # concatenate
        nan_idx = np.unique(nan_idx_tmp)
        no_values = np.unique(no_values_tmp)

        data_arr['jetTrim1_groosplit12'] = data_arr['jetTrim1_groosplit12']/1000.
        data_arr['jetTrim2_groosplit12'] = data_arr['jetTrim2_groosplit12']/1000.
        # get the columns for classifying for jet1
        bdt_data = data_arr[bdt_vars_1]
        dnn_data = data_arr[dnn_vars_1]
        # get the columns for classifying for jet2
        bdt_data_2 = data_arr[bdt_vars_2]
        dnn_data_2 = data_arr[dnn_vars_2]

        # do we have to rename the data fields?
        # easy enough to rename if the fields
        recs = []

        bdt_data.dtype.names = [bdt_ivd[d] for d in bdt_data.dtype.names]
        bdt_data_2.dtype.names = [bdt_ivd[d] for d in bdt_data_2.dtype.names]

        for d in bdt_data.dtype.names:
            if d != 'nTracks' and d.find('trk') == -1:
                recs.append((d, 'float'))
            else:
                recs.append((d, 'int'))
            #print np.any(np.isnan(bdt_data[d]))
            #print np.all(np.isfinite(bdt_data[d]))
            #print np.where(np.isnan(bdt_data[d]))


        bdt_data_arr = bdt_data.view(np.float32).reshape(bdt_data.shape + (-1,))
        bdt_data_arr_2 = bdt_data_2.view(np.float32).reshape(bdt_data_2.shape + (-1,))

        bdt_proba = None
        bdt_proba_2 = None
        for m in bdt_models:
            if bdt_proba is not None:
                bdt_proba += m.predict_proba(bdt_data_arr)[:,1]
                bdt_proba_2 += m.predict_proba(bdt_data_arr_2)[:,1]
            else:
                bdt_proba = m.predict_proba(bdt_data_arr)[:,1]
                bdt_proba_2 = m.predict_proba(bdt_data_arr_2)[:,1]
            # scale data
        bdt_proba/=float(len(bdt_models))
        bdt_proba_2/=float(len(bdt_models))
        
        
        for i, v in enumerate(dnn_scaler.variables):
            # reverse lookup for v
            #print v
            if v in dnn_var_map.keys():
                #print 'found v in dnn_var_map'
                #print dnn_var_map[v]
                if dnn_var_map[v].replace('X','1') in dnn_data.dtype.names:
                    '''
                    print v
                    print 'means and std'
                    print dnn_scaler.means[i]
                    print dnn_scaler.std[i]
                    print np.mean(dnn_data[dnn_var_map[v].replace('X','1')])
                    '''
                    dnn_data[dnn_var_map[v].replace('X','1')] = (dnn_data[dnn_var_map[v].replace('X','1')] - dnn_scaler.means[i]) / dnn_scaler.std[i]
                    dnn_data_2[dnn_var_map[v].replace('X','2')] = (dnn_data_2[dnn_var_map[v].replace('X','2')] - dnn_scaler.means[i]) / dnn_scaler.std[i]

        dnn_data.dtype.names = [dnn_ivd[d] for d in dnn_data.dtype.names]
        dnn_data_2.dtype.names = [dnn_ivd[d] for d in dnn_data_2.dtype.names]
            
        # do we have to rename the data fields to get this to work?
        dnn_predictions = None
        for m in dnn_models:
            if dnn_predictions is not None:
                dnn_predict1 = m.predict(dnn_data)[0]
                dnn_predict1.dtype.names = ['jetTrim1_dnn']
                dnn_predict2 = m.predict(dnn_data_2)[0]
                dnn_predict2.dtype.names = ['jetTrim2_dnn']
                for n in xrange(len(dnn_predict1['jetTrim1_dnn'])):
                    dnn_predictions['jetTrim1_dnn'][n] += dnn_predict1['jetTrim1_dnn'][n]
                    dnn_predictions_2['jetTrim2_dnn'][n] += dnn_predict2['jetTrim2_dnn'][n]
            else:
                dnn_predictions = m.predict(dnn_data)[0]
                dnn_predictions.dtype.names = ['jetTrim1_dnn']
                dnn_predictions_2 = m.predict(dnn_data_2)[0]
                dnn_predictions_2.dtype.names = ['jetTrim2_dnn']

        for n in xrange(len(dnn_predictions['jetTrim1_dnn'])):
            dnn_predictions['jetTrim1_dnn'][n] /= float(len(dnn_models))
            dnn_predictions_2['jetTrim2_dnn'][n] /= float(len(dnn_models))
        # set all of the nan index ones to zero
        for n in np.unique(np.concatenate((no_values, nan_idx))):#nan_idx:
            dnn_predictions['jetTrim1_dnn'][n] = 0
            dnn_predictions_2['jetTrim2_dnn'][n] = 0
            bdt_proba[n] = 0
            bdt_proba_2[n] = 0
        data_arr['jetTrim1_groosplit12'] = data_arr['jetTrim1_groosplit12']*1000.
        data_arr['jetTrim2_groosplit12']*=1000.
        # have to do this annoying .copy() to be able to add the dtype.names to any
        # arrays that come from a slice.
        #bdt_ = X.copy().view(dtype=[(n, np.float64) for n in variables]).reshape(len(X))
        # now add them to the data file and write it out
        #data_scored = nf.append_fields(data_arr, names=['jetTrim1_bdt','jetTrim2_bdt','jetTrim1_dnn','jetTrim2_dnn'], data=[bdt_proba, bdt_proba_2, dnn_predictions, dnn_predictions_2], usemask = False)
        data_scored = nf.append_fields(data_arr, names=['jetTrim1_bdt','jetTrim2_bdt','jetTrim1_dnn','jetTrim2_dnn'], data=[bdt_proba, bdt_proba_2, dnn_predictions, dnn_predictions_2], usemask = False)
        rn.array2root(data_scored, data_file.replace('.root','_scored_nonTrk_avg_vTest.root'),'dibjet','recreate')
            
            

def main(args):
    
    parser = argparse.ArgumentParser(description='Score data with MVA tagger.')
    parser.add_argument('--dnn-file', dest = 'dnn_file', default='None', help='Yaml config file for the dnn.', required=True)
    parser.add_argument('--dnn-scaler', dest = 'dnn_scaler', default = 'None', help = 'Scaler file to standardise data.', required=True)
    parser.add_argument('--bdt-file', dest = 'bdt_file', default = 'None', help = 'Config file or model file for the tagger.', required=True)
    parser.add_argument('--data-input-file', dest = 'data_input_file', default = 'data.txt', help = 'Input file with list of all data files to analyse', required=True)

    args = parser.parse_args()
    
    bdtFile =  open(args.bdt_file,'r')
    bdt_models = []
    bdt_taggers = None        
    # get the bdt model
    for l in bdtFile.readlines():
        with bz2.BZ2File(l.strip(),'r') as p:
            modelObject = pickle.load(p)
        bdt_models.append(modelObject.model)
        # get the variables we use for classification
        bdt_taggers = modelObject.taggers
    print bdt_taggers
    bdtFile.close()
    # load the dnn schema
    
    dnnFile = open(args.dnn_file,'r')
    dnn_taggers = None
    dnn_models = []
    dnn_scalers = []
    for l in dnnFile.readlines():
        dnn_file_obj = open(l.strip(),'r')
        schema = yaml.load(dnn_file_obj)
        dnn_model = apy.NeuralNet()
        print schema
        print type(schema)
        #for dnn_tagger_file, specifications in schema['taggers'].iteritems():
        if len(schema['taggers'].keys()) == 0:
            print 'no taggers specified in the input schema! Check input dnn-file'
            return
        dnn_tagger_file=schema['taggers'].keys()[0]
        print dnn_tagger_file
        dnn_model.load(dnn_tagger_file)
        dnn_models.append(dnn_model)
        print 'read'
        #break
        # now we can use dnn_tagger.predict(data)[0]    
        dnn_taggers = dnn_model.inputs
        # load scaler
    print args.dnn_scaler
    with open(args.dnn_scaler, 'r') as p:
        dnn_scaler = pickle.load(p)
    #dnn_scalers.append(dnn_scaler_tmp)
    with open(args.data_input_file, 'r') as f:
        input_file_list = f.read().splitlines()

    dnnFile.close()
    analyse(bdt_models, bdt_taggers, dnn_models, dnn_taggers, dnn_scaler, input_file_list)
    
    # get the bdt model
    '''
    with bz2.BZ2File(args.bdt_file,'r') as p:
        modelObject = pickle.load(p)
    bdt_model = modelObject.model
    # get the variables we use for classification
    bdt_taggers = modelObject.taggers
    print bdt_taggers
    # load the dnn schema
    dnn_file_obj = open(args.dnn_file,'r')
    schema = yaml.load(dnn_file_obj)
    dnn_model = apy.NeuralNet()
    print schema
    print type(schema)
    #for dnn_tagger_file, specifications in schema['taggers'].iteritems():
    if len(schema['taggers'].keys()) == 0:
        print 'no taggers specified in the input schema! Check input dnn-file'
        return
    dnn_tagger_file=schema['taggers'].keys()[0]
    print dnn_tagger_file
    dnn_model.load(dnn_tagger_file)
    print 'read'
    #break
    # now we can use dnn_tagger.predict(data)[0]    
    dnn_taggers = dnn_model.inputs
    # load scaler
    print args.dnn_scaler
    with open(args.dnn_scaler, 'r') as p:
        dnn_scaler = pickle.load(p)
    with open(args.data_input_file, 'r') as f:
        input_file_list = f.read().splitlines()
    analyse(bdt_model, bdt_taggers, dnn_model, dnn_taggers, dnn_scaler, input_file_list)
    '''
    #rootAnalyse(bdt_model, bdt_taggers, dnn_model, dnn_taggers, dnn_scaler, input_file_list)

if __name__ == '__main__':
    main(sys.argv)
