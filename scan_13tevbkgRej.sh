. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_200_350 -t outputTree -m true --ptlow=200 --pthigh=350
. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_500_1000 -t outputTree -m true --ptlow=500 --pthigh=1000
. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_1000_1500 -t outputTree -m true --ptlow=1000 --pthigh=1500
. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_1500_2000 -t outputTree -m true --ptlow=1500 --pthigh=2000
. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_2000_3000 -t outputTree -m true --ptlow=2000 --pthigh=3000
. startcluster.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/13tev_v4/ -f 13tev_v4_files.txt -c 13tev_v4_configs.txt -v 13tev_v4_350_500 -t outputTree -m true --ptlow=350 --pthigh=500
. stopcluster.sh
