import sys
import argparse
import modelEvaluation as ev
import os
import pickle
import bz2
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import operator
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import floor, log10
import itertools
columns = ['test_id','cv_id','max_depth','n_estimators','learning_rate','f1_train','f1_test','recall_train','recall_test','precision_train','precision_test','accuracy_train','accuracy_test','bkg_rej_train','bkg_rej_test','filename']


def matchAll(test, filters=[]):
    '''
    Check if every string in filters is found in the test string, in order.
    '''
    prev = 0
    for f in filters:
        pos = test.find(f, prev)
        if pos != -1:
            prev = pos
        else:
            return False
    return True


def getTaggerScores(files):
    all_taggers_scores = {}
    all_taggers_positions = {}
    scores = {}
    max_score = 0
    max_id = ""
    for f in files:
        print 'plotting file: ' + f
        if f.endswith('pickle'):
            with open('evaluationObjects/'+f,'r') as p:
                model = pickle.load(p)
        else:
            with bz2.BZ2File('evaluationObjects/'+f,'r') as p:
                model = pickle.load(p)
        taggers = model.taggers
        
        for t in taggers:
            if t not in all_taggers_scores.keys():
                all_taggers_scores[t] = []
                all_taggers_positions[t] = []
                
        feature_importances = 100.0 * (model.feature_importances / model.feature_importances.max())


        sorted_idx = np.argsort(feature_importances)[::-1]
        for x in range(len(sorted_idx)):
            # add the feature importance score and the position
            all_taggers_scores[taggers[sorted_idx[x]]].append(feature_importances[sorted_idx[x]])
            all_taggers_positions[taggers[sorted_idx[x]]].append(x)
        scores[model.job_id] = model.ROC_rej_power_05
        if model.score > max_score:
            max_score = model.ROC_rej_power_05
            max_id = model.job_id

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    print sorted_scores
    print 'press a key to get tagger scores'
    # write to file
    f = open('scores'+key+'.txt','w')
    for s in sorted_scores:
        f.write(s[0] + ': ' + str(s[1]) +'\n')
    f.close()

    # find median of each
    medians = {}
    for t in all_taggers_scores.keys():
        medians[t] = np.median(all_taggers_scores[t])

    sorted_medians = sorted(medians.items(), key=operator.itemgetter(1))
    print sorted_medians


def recreateFull(job_id, full_dataset, suffix = 'v2', compress=True, transform_valid_weights = False, weight_validation = False):
    
    # load the model from the cv model
    if job_id.endswith('pickle'):
        with open('evaluationObjects/'+job_id,'r') as p:
            model = pickle.load(p)
    else:
        import bz2
        with bz2.BZ2File('evaluationObjects/'+job_id,'r') as p:
            model = pickle.load(p)
        
    file_full = full_dataset
    # check that this file exists
    if not os.path.isfile(file_full):
        print 'could not locate the full dataset'
        return roc_bkg_rej
    print file_full
    X_full, y_full, w_full, efficiencies = joblib.load(file_full, mmap_mode='c')


    if transform_valid_weights:# and weight_validation:
        for idx in xrange(0, w_full.shape[0]):
            if y_full[idx] == 1:
                w_full[idx] = 1.0
            else:
                w_full[idx] = np.arctan(1./w_full[idx])

    if weight_validation:
        w_tmp = w_full
        #for idx in xrange(0, w_full.shape[0]):
        #    if y_full[idx] == 1:
        #        w_full[idx] = 1.0
        #w_tmp[sig
    else:
        w_tmp = None
    print efficiencies
    print w_tmp
    full_score = model.model.score(X_full, y_full, sample_weight=w_tmp)
    prob_predict_full = model.model.predict_proba(X_full)[:,1]
    fpr_full, tpr_full, thresh_full = roc_curve(y_full, prob_predict_full, sample_weight = w_tmp)
    # need to set the maximum efficiencies for signal and bkg
    m_full = ev.modelEvaluation(fpr_full, tpr_full, thresh_full, model.model, model.params, job_id.replace('.pbz2','').replace('.pickle','')+suffix, model.taggers, model.Algorithm, full_score, file_full,feature_importances=model.feature_importances, decision_function=model.decision_function, decision_function_sig = model.df_sig_idx, decision_function_bkg = model.df_bkg_idx)
    m_full.setSigEff(efficiencies[0])
    m_full.setBkgEff(efficiencies[1])
    # get the indices in the full sample
    sig_full_idx = y_full == 1
    bkg_full_idx = y_full == 0
    # set the probabilities and the true indices of the signal and background
    m_full.setProbas(prob_predict_full, sig_full_idx, bkg_full_idx, w_tmp)
    # set the different scoresx
    y_pred_full = model.model.predict(X_full)
    m_full.setScores('test',accuracy=accuracy_score(y_pred_full, y_full, sample_weight = w_tmp), precision=precision_score(y_pred_full, y_full, sample_weight = w_tmp), recall=recall_score(y_pred_full, y_full, sample_weight = w_tmp), f1=f1_score(y_pred_full, y_full, sample_weight = w_tmp))
    print accuracy_score(y_pred_full, y_full, sample_weight = w_tmp)
    # need to get the train scores!
    m_full.setScores('train',accuracy=model.train_accuracy, precision=model.train_precision, recall=model.train_recall, f1=model.train_f1)
    # write this into a root file
    m_full.toROOT()
    # save the train score
    m_full.setTrainRejection(model.ROC_rej_power_05)
    if not compress:
        f_name_full = 'evaluationObjects/'+job_id.replace('.pickle','')+'_'+suffix+'.pickle'
        try:
            with open(f_name_full,'w') as d2:
                pickle.dump(m_full, d2)
            d2.close()
            print 'pickled ' + f_name_full
    
        except:
            msg = 'unable to dump '+job_id+ '_'+suffix+' object'
            with open(f_name_full,'w') as d2:
                pickle.dump(msg, d2)
            d2.close()
            print 'unable to dump '+job_id+ '_full object:', sys.exc_info()[0]            
    else:
        if job_id.endswith('pickle'):
            f_name_full = 'evaluationObjects/'+job_id.replace('.pickle','')+'_'+suffix+'.pbz2'
        else:
            f_name_full = 'evaluationObjects/'+job_id.replace('.pbz2','')+'_'+suffix+'.pbz2'
        try:
            with bz2.BZ2File(f_name_full,'w') as d2:
                pickle.dump(m_full, d2)
            d2.close()
            print 'pickled ' + f_name_full
    
        except:
            msg = 'unable to dump '+job_id+ '_'+suffix+' object'
            with bz2.BZ2File(f_name_full,'w') as d2:
                pickle.dump(msg, d2)
            d2.close()
            print 'unable to dump '+job_id+ '_full object:', sys.exc_info()[0]

            

def createDataframe(key, files):
    '''
    create a dataframe containing the info for all of the evaluation objects contained in the files list.
    '''
    data = []
    num_files = str(len(files))
    
    for i, f in enumerate(files):
        print 'plotting file: ' + f + ' file number: ' + str(i) + ' out of ' + num_files
        if f.endswith('pickle'):
            with open('evaluationObjects/'+f,'r') as p:
                model = pickle.load(p)
        else:
            
            with bz2.BZ2File('evaluationObjects/'+f,'r') as p:
                model = pickle.load(p)
        # the filename contains the iteration number and the cv number which together form a unique id.
        #  we have a string like paramID_iternum_key_cvnum[_full].pickle
        # first number in the filename is the iternum
        numbers = re.findall(r'\d+', f)
        iter_num = numbers[0]
        # last number is the cv number, unless we have _full_v2.pickle
        if f.find('full_v') != -1:
            cv_num = numbers[-2]
        else:
            cv_num = numbers[-1]
        # now we want to get max_depth parameter
        params = model.model.get_params()
        max_depth = -1
        learning_rate = -1.0
        n_estimators = -1
        for k in params.keys():
            if k.find('max_depth') != -1:
                max_depth = int(params[k])
            elif k.find('learning_rate') != -1:
                learning_rate = float(params[k])
            elif k.find('n_estimators') != -1:
                n_estimators = int(params[k])

        # get the f1 scores
        f1_train = model.train_f1
        f1_test = model.test_f1
        # get the recall scores
        recall_train = model.train_recall
        recall_test = model.test_recall
        # get the accuracy scores
        accuracy_train = model.train_accuracy
        accuracy_test = model.test_accuracy
        # get the precision scores
        precision_train = model.train_precision
        precision_test = model.test_precision
        # get the bkg rejection power scores
        bkg_rej_train = model.ROC_rej_power_05_train
        bkg_rej_test = model.ROC_rej_power_05
        thedict  = {'test_id':int(iter_num), 'cv_id': int(cv_num), 'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate,'f1_train':f1_train,'f1_test':f1_test,'recall_train':recall_train,'recall_test':recall_test,'precision_train':precision_train,'precision_test':precision_test,'accuracy_train':accuracy_train,'accuracy_test':accuracy_test,'bkg_rej_train':bkg_rej_train,'bkg_rej_test':bkg_rej_test,'filename': f}
        data.append(thedict)
        
    return data


def plotValidationCurve(key, parameters, param_abbrev, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std, param_range,  val_variable = '$Max depth$', file_id = ''):
    '''
    Plot a validation curve. For the moment this does a validation curve for the bdt. A bdt normally depends on the tree depth
    for over and under-fitting.

    Keyword args:
    key -- file identifier
    parameters --- the values for the training parameters: dictionary with learning_rate: lr, n_est: n, max_depth: md
    parameters --- the abbreviations for the training parameters to be used in the file name, example: _lr_xyz_n_xyz etc
    mean and std for training and testing scores
    param_range --- the range of the max_depth variable
    val_variable --- the variable being validationed
    file_id  --- suffix for the filename
    '''
    xlabel = val_variable
    parameter_string = ''
    for x in parameters.keys():
        parameter_string += '_'+param_abbrev[x]+'_' + str(parameters[x])
            # set up the file name
    #fname = "validation_curves/"+key+'_lr_'+str(parameters['learning_rate'])+'_n_'+str(parameters['n_estimators'])+file_id+'.png'
    fname = "validation_curves/"+key+parameter_string+file_id+'.png'
    plt.clf()
    plt.title("Validation Curve with BDT")
    plt.xlabel(xlabel.replace('_',' '))
    plt.ylabel("Accuracy")
    # find out what the y limits should be
    y_up = max(np.amax(train_scores_mean), np.amax(test_scores_mean))
    y_do = min(np.amin(train_scores_mean), np.amin(test_scores_mean))
    plt.ylim(y_do*0.95, y_up*1.05)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                                      train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                              color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                                      test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.savefig(fname)



    

def getDataFrame(recreate_csv=False, keys=['features_l_2_10_v6'], file_id = 'validation', compress_id ='pbz2', fullset = True):
    key = keys[0]
    if fullset:
        files = [f for f in os.listdir('evaluationObjects/') if matchAll(f,keys) and f.endswith(compress_id) and f.find('full') != -1]
    else:
        files = [f for f in os.listdir('evaluationObjects/') if matchAll(f,keys) and f.endswith(compress_id) and f.find('full') == -1]
    print 'total number of objects: ' + str(len(files))

    # check that the file exists if we have decided not to recreate the csv file
    if not recreate_csv:
        filefound = False
        fname = 'data_'+key+file_id+'.csv'
        while not filefound:
            if not os.path.isfile(fname):
                print "file "+fname+" doesn't exist! Enter another name to try or hit enter to create a new csv"
                fname = input()
                if fname == '':
                    recreate_csv = True
                    filefound = True
                continue
            filefound = True
    
    if recreate_csv:
        #get a dict of the data
        data_dict = createDataframe(key, files)
        # now create a dataframe out of the data
        df = pd.DataFrame(data_dict)
        # save it as a csv so that we don't have to do this all over again every time.
        #if save_csv:
        csv_file = open('data_'+key+file_id+'.csv','w')
        # write the column names to the csv file
        csv_file.write('test_id,cv_id,max_depth,n_estimators,learning_rate,f1_train,f1_test,recall_train,recall_test,precision_train,precision_test,accuracy_train,accuracy_test,bkg_rej_train,bkg_rej_test,filename\n')
        for i,d in enumerate(data_dict):
            # write the keys to the file if it is the first one
            num_keys = len(d.keys())
            #if i != 0:
            for j,k in enumerate(columns):
                csv_file.write(str(d[k]))
                if j != num_keys-1:
                    csv_file.write(',')
                else:
                    csv_file.write('\n')
        csv_file.close()

    else:
        # we can read it in from the csv file
        df = pd.read_csv(fname)

    return df


def evaluateVariableCombined(df, variable, key, file_id):
    # because the indexing is so hard and complicated...
    # you can't do lr_mean.loc[0.1], whereas you can do that if your loc[] thing is an int
    # my hack is to convert all of the learning rates to ints and use it like that
    scale_factor = 10
    if df[variable].dtype == np.float64:
        df[variable] = df[variable]*scale_factor
        df[variable] = df[variable].astype(int)
    
    train_leg = mpatches.Patch(color='blue', label='Train accuracy')
    val_leg = mpatches.Patch(color='red', label='Validation accuracy')

    plt.clf()
    plt.title("Accuracy as a function of " + variable)#max depth")
    plt.xlabel("$"+variable+"$")
    plt.ylabel("Accuracy")
    plt.legend(handles=[train_leg,val_leg])
    grouped_md = df.groupby([variable,'test_id'])
    md_mean = grouped_md.mean()
    md = md_mean.index.levels[0].values # get all of the unique values for max_depth
    # goign to store these stats
    train_mean = np.zeros(len(md))
    test_mean = np.zeros(len(md))
    train_std = np.zeros(len(md))
    test_std = np.zeros(len(md))
    # loop through all of the validation variable values
    for i, m in enumerate(md):
        md_m = md_mean.loc[m]
        test_mean[i] = np.mean(md_m['accuracy_test'])
        test_std[i] = np.std(md_m['accuracy_test'])
        train_mean[i] = np.mean(md_m['accuracy_train'])
        train_std[i] = np.std(md_m['accuracy_train'])
        plt.scatter(np.repeat([m],len(md_m)),md_m['accuracy_test'], label=str(m), color='r')#, label='Validation')
        plt.scatter(np.repeat([m],len(md_m)),md_m['accuracy_train'], label=str(m), color='b')#, label='Train')
        
    plt.savefig('validation_curves/combined_'+variable+key+file_id+'.png')
    # create a validation curve, rather than a scatter plot
    plotValidationCurve(key, {'combined':variable} , {'combined':variable}, train_mean, test_mean, train_std, test_std, md, '$'+variable+'$',file_id) 


def evaluateVariable(df, key, file_id = '', grouping = ['learning_rate','n_estimators','max_depth']):

    # some preset param_abbreviations
    preset_abbrev = {'learning_rate':'lr','n_estimators':'n','max_depth':'md'}
    
    # now we want to be able to the dataframe sort on a number of criteria and create meaningful stats
    # easiest to run in interactive mode first.
    grouped = df.groupby(grouping)
    # get the mean scores
    gmean = grouped.mean()
    gstd = grouped.std()
    # get the index
    idx = gmean.index
    # get the levels - this stores the keys for the different groupby objects
    lvls = idx.levels
    # lvls is a FrozenList with each element being a Frozen64Index
    # get all of the values for the different parameters
    level_values = []
    for i, x in enumerate(grouping):
        level_values.append(lvls[i].values)

    # this is the list of parameters that gets passed to plotValidationCurve
    grouping_variables = {}
    param_abbrev = {}
    for g in grouping[:-1]:
        grouping_variables[g] = 0.0
        if g in preset_abbrev.keys():
            param_abbrev[g] = preset_abbrev[g]
        else: # a default value - first letter
            param_abbrev[g] = g[0]
            
    # we are interested in the scores for the LAST variable in the grouping!
    validation_variable = level_values[-1]
    # name
    val_var_name = grouping[-1]
    
    training_points = np.zeros(len(validation_variable))
    training_std = np.zeros(len(validation_variable))
    testing_points = np.zeros(len(validation_variable))
    testing_std = np.zeros(len(validation_variable))
    # now we can get the scores for the different values of the validation variable
    # create a list with a combination of all variables
    # exclude the validation variable as we will loop over this on it's own
    for v in validation_variable:
        variable_list = [list(x) for x in level_values]
        #combinations = itertools.product(*variable_list)
    indices = pd.MultiIndex.from_product(variable_list, names = grouping)
    plot_values = gmean.loc[indices]
    plot_std = gstd.loc[indices]
    rows = len(plot_values)
    init_val = variable_list[-1][0]
    counter = 0
    # loop through all of the combinations now
    for i in xrange(rows):
        # if we have looped through a full set of the last variable, start over and plot the val curve
        if plot_values.iloc[i].name[-1] == init_val:
            if counter != 0:
                # create the parameter dictionary
                for j, g in enumerate(grouping[:-1]): # get all of the values
                    grouping_variables[g] = plot_values.iloc[i].name[j]
                plotValidationCurve(key, grouping_variables, param_abbrev, training_points, testing_points, training_std, testing_std, validation_variable, val_var_name, file_id)
            counter = 0

        else:
            counter+=1
        # record the values for each value of the validation variable
        training_points[counter] = plot_values.iloc[i]['accuracy_train']
        training_std[counter] = plot_std.iloc[i]['accuracy_train']
        testing_points[counter] = plot_values.iloc[i]['accuracy_test']
        testing_std[counter] = plot_std.iloc[i]['accuracy_test']



def main(args):
    parser = argparse.ArgumentParser(description='Plot the evaluation objects and get some meaningful stats from the output of the BDTs.')
    parser.add_argument('--key', default='features_l_2_10_v6', help = 'Key to use for reading in the evaluation objects. This gets added to the output filenames for the stats files. If a wildcard is encountered, such as *, then the first part of the key is used for naming, the full key for matching.')
    parser.add_argument('--fileid', default='legit_full', help = 'File id to add to the output file name (works together with the key).')
    parser.add_argument('--fulldataset', default='persist/data_features_nc_2_10_v5_100.pkl', help = 'Name of the full dataset.')
    parser.add_argument('--createcsv', dest='createcsv',  action='store_true', help = 'Whether or not to recreate the dataframe from the csv file or to read in all of the evaluation objects to create the dataframe.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help = 'If the bdt is getting tested on some new data, using the trained models. Default is false.')
    parser.add_argument('--weight-validation', dest='weight_validation', action='store_true', help = 'If the validation samples should be weighted.')
    parser.add_argument('--transform-weight-validation', dest='tx_weight_validation', action='store_true', help = 'If the validation sample weights should be transformed.')
    parser.add_argument('--new-fileid', dest = 'fullid', help = 'File identifier if the bdt is getting tested on some new data.')
    
    parser.set_defaults(createcsv=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(tx_weight_validation=False)
    parser.set_defaults(weight_validation=False)
    parser.set_defaults(fullid='new_eval')
    
    args = parser.parse_args()
    
    recreate_csv = args.createcsv
    # wildcards
    ksplit = args.key.split("*")
    
    key = ksplit[0]
    
    file_id = args.fileid
    full_dataset = args.fulldataset#'persist/data_features_nc_2_10_v5_100.pkl'
    # are we wanting to use bz2?
    compress_id = 'pbz2' # or pickle
    if args.evaluate:

                    
        jobids = [f for f in os.listdir('evaluationObjects/') if matchAll(f,ksplit) and f.find('_full.pickle')==-1 and f.endswith('full.'+compress_id)]
        print 'total number of objects: ' + str(len(jobids))
        total = len(jobids)

        for i, j in enumerate(jobids):
            print 'progress: ' + str(float(100.0*i/total))
            recreateFull(j,full_dataset, '_'+args.fullid, compress=True, transform_valid_weights = args.tx_weight_validation, weight_validation = args.weight_validation)#    '_bkg_training_12_16', compress=True)    
        print 'finished evaluation'
        sys.exit(0)

    # get the dataframe. either from a df written to a csv file or from the evaluation objects
    df = getDataFrame(recreate_csv, ksplit, file_id = file_id, compress_id = compress_id, fullset = True)

    grouped_id = df.groupby(['test_id']) # this combines all of the cv folds
    # get the mean scores
    id_mean = grouped_id.mean()
    id_std = grouped_id.std()

    # sort the groups based on bkg rej power at 50%
    srt_bkg = id_mean.sort('bkg_rej_test',ascending=False)

    # before writing to csv, format it to only use 4 sig digits
    to_round = ['learning_rate','accuracy_test','bkg_rej_test']
    for r in to_round:
        srt_bkg[r] = srt_bkg[r].map(lambda x: round(x, -int(floor(log10(x)))+3) )
    # make sure the max depth and n_estimators are ints
    srt_bkg[['max_depth','n_estimators']] = srt_bkg[['max_depth','n_estimators']].astype(int)
    
    srt_bkg.to_csv('data_'+key+file_id+'_sortedresults.csv',columns=['max_depth','learning_rate','n_estimators','accuracy_test','bkg_rej_test'])

    # we also want to know how all of the max_depth values did, for all parameters.

    evaluateVariable(df, key, file_id, ['learning_rate','n_estimators','max_depth'])
    evaluateVariable(df, key, file_id, ['n_estimators','max_depth','learning_rate'])
    evaluateVariable(df, key, file_id, ['max_depth','learning_rate', 'n_estimators'])

    evaluateVariableCombined(df, 'max_depth', key, file_id)
    evaluateVariableCombined(df, 'learning_rate', key, file_id)
    evaluateVariableCombined(df, 'n_estimators', key, file_id)


if __name__ == '__main__':
    main(sys.argv)
