FOLDER=training_output/twostage_weighted_10
mkdir $FOLDER
cp prediction/config.py $FOLDER
srun -p csc490-compute -c 6 -N 1 --gres gpu -o $(pwd)/$FOLDER/outfile python -m prediction.main train --data_root=/u/csc490h/dataset --output_root=$FOLDER/outputs
