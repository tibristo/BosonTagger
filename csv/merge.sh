sigfile=${1}
bkgfile="${sigfile/sig/bkg}"
m="${sigfile/sig/merged}"
echo $m

cat $sigfile > $m

tail -n+2 $bkgfile >> $m