# assuming the csv files are created with Tagger.py (scan_13tev_blah_csv.sh)



# set key to whatever the name is in the csv creation
cd csv
for x in `ls *key*sig.csv` ; do source merge.sh $x ; done
# create the root files too
for x in `ls *key*.csv` ; do python csv2root.py $x ; done
cd ..


# have a look at the variables, their correlations and a simple bdt with all available variables. Then can look at the feature_importances and make a decision about which to keep.
# correlation matrix is saved as corr_matrices/corr_matrix_full_allvars_mc15.pdf
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw --plotCorrMatrix=True --fileid=full_allvars_mc15 --key=mc15_v1_2_10_v6 --createFoldsOnly=True --folds=1 --runTestCase=True --allVars=True

# now get the feature importances from the test sample.  This can just write it to a temp file like test_features.txt
python featureImportances.py
cat test_features.txt

# now after choosing the variables to keep, create the proper samples to use (with the reduced variables)
# create the full dataset from the NOT CLEANED sample (the one where no contraints apart from mass window have been applied to the sample)
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_notcleaned_v1_200_1000_mw --key=mc15_nc_v1_2_10_v1 --createFoldsOnly=True --onlyFull=True

# create the folds for the CLEANED sample (set the fulldataset to that obtained above
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw --key=mc15_v1_2_10 --createFoldsOnly=True --fulldataset=persist/data_mc15_nc_v1_2_10_v1_100.pkl

# run a test to check that it works and that the efficiencies look legit (68% for signal) (note this overwrites the output from the test above, but not the test_features.txt file)
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw --key=mc15_v1_2_10 --runTestCase=True --fulldataset=persist/data_mc15_nc_v1_2_10_v1_100.pkl

# now run the entire mva
startcluster8
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw --key=mc15_v1_2_10 --fulldataset=persist/data_mc15_nc_v1_2_10_v1_100.pkl --runMVA=True

# can now get some info out of the cv splits if we want
python mva_tools.py --algorithm=AntiKt10LCTopoTrimmedPtFrac5SmallR20_13tev_mc15_v1_200_1000_mw --key=mc15_v1_2_10 --fulldataset=persist/data_mc15_nc_v1_2_10_v1_100.pkl --plotCV=True

# show the top bdts
cat bests/bestsmc15_v1_2_10.txt

# now run some plots to create the validation plots and some more score stats about the different bdts
python plotEvaluationObjects.py --key=mc15_jz5_v1_8_12_v4 --fileid=mc15_jz5_v4_full --createcsv=True


# if we want to test it on some other data:
python plotEvaluationObjects.py --key=mc15_v2.3_4_16 --fulldataset=persist/data_mc15_jz5_nc_v2_8_12_v1_100.pkl --evaluate --weight-validation --new-fileid=tx_weightvalidation_jz5 --transform-weight-validation
