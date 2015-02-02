cd Websites/t/tibristo/${1}

mput matrixinv_${1}*.png

# if 13 tev
if [[ ${1} == *"13tev"* ]]
then
cd CamKt10LCTopoPrunedCaRCutFactor50Zcut15/

mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}*ROCPlot.png

mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_200_350/*
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_350_500/*
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_500_1000/*
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_1000_1500/*
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_1500_2000/*
mput CamKt10LCTopoPrunedCaRCutFactor50Zcut15_${1}_2000_3000/*
#if 8 tev
else
cd CamKt10LCTopoPrunedCaRcutFactor50Zcut15/
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}*ROCPlot.png

mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_200_350/*
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_350_500/*
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_500_1000/*
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_1000_1500/*
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_1500_2000/*
mput CamKt10LCTopoPrunedCaRcutFactor50Zcut15_${1}_2000_3000/*
fi

cd ../AntiKt10LCTopoTrimmedPtFrac5SmallR20/

mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}*ROCPlot.png

mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_200_350/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_350_500/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_500_1000/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_1000_1500/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_1500_2000/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR20_${1}_2000_3000/*

cd ../AntiKt10LCTopoTrimmedPtFrac5SmallR30/
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}*ROCPlot.png

mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_200_350/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_350_500/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_500_1000/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_1000_1500/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_1500_2000/*
mput AntiKt10LCTopoTrimmedPtFrac5SmallR30_${1}_2000_3000/*

cd ../CamKt12LCTopoSplitFilteredMu100SmallR30YCut15/

mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}*ROCPlot.png

mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_200_350/*
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_350_500/*
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_500_1000/*
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_1000_1500/*
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_1500_2000/*
mput CamKt12LCTopoSplitFilteredMu100SmallR30YCut15_${1}_2000_3000/*
