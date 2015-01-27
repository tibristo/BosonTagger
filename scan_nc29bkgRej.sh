. startcluster_all.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_200_350 -t physics -m true --ptlow=200 --pthigh=350
. startcluster_all.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_500_1000 -t physics -m true --ptlow=500 --pthigh=1000
. startcluster_all.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_1000_1500 -t physics -m true --ptlow=1000 --pthigh=1500
. startcluster_all.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_1500_2000 -t physics -m true --ptlow=1500 --pthigh=2000
#. startcluster_all.sh
#sleep 10
#python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_1500_2000 -t physics -m true --ptlow=2000 --pthigh=3000
. startcluster_all.sh
sleep 10
python scanBkgRej.py -b /media/win/BoostedBosonFiles/nc29_fast_v3/ -f nc29_fast_v3_files.txt -c nc29_fast_v3_configs.txt -v nc29_fast_v3_350_500 -t physics -m true --ptlow=350 --pthigh=500
. stopcluster.sh
