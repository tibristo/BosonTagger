startdir=$PWD
for x in `cat plotsToUpload.txt`;
do
cp create_subindex.py $x/
cd $x/
python create_subindex.py
cd $startdir
done