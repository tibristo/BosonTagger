cd Websites/t/tibristo/

cd ${1}
mput index.html

# if 13 tev
if [[ ${1} == *"13tev"* ]]
then
cd CamKt10LCTopoPrunedCaRCutFactor50Zcut15/

mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_200_350/index.html
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_350_500/index.html
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_500_1000/index.html
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_1000_1500/index.html
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_1500_2000/index.html
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_2000_3000/index.html
else
# if 8 tev
cd CamKt10LCTopoPrunedCaRcutFactor50Zcut15/

mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_200_350/index.html
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_350_500/index.html
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_500_1000/index.html
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_1000_1500/index.html
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_1500_2000/index.html
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_2000_3000/index.html
fi

cd ../AntiKt10LCTopoTrimmedPtFrac5SmallR20/
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_200_350/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_350_500/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_500_1000/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_1000_1500/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_1500_2000/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_2000_3000/index.html

cd ../AntiKt10LCTopoTrimmedPtFrac5SmallR30/
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_200_350/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_350_500/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_500_1000/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_1000_1500/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_1500_2000/index.html
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_2000_3000/index.html

cd ../CamKt12LCTopoSplitFilteredMu100SmallR30YCut15/
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_200_350/index.html
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_350_500/index.html
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_500_1000/index.html
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_1000_1500/index.html
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_1500_2000/index.html
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_2000_3000/index.html
