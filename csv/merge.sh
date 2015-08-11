sigfile=${1}
# replace sig with bkg in the signal file name
bkgfile="${sigfile/sig/bkg}"
# replace sig with merged
m="${sigfile/sig/merged}"
# check the name is correct
echo $m

# pipe signal to merged file
cat $sigfile > $m

# pipe background (except the top two lines which we have from the signal file already)
tail -n+2 $bkgfile >> $m
