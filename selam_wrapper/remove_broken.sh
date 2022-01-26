for file in *
do
lines=`cat $file | wc -l`
if [ $lines -lt 1000 ]
then
rm $file
fi
done
