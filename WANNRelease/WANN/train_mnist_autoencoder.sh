today=`date +%Y-%m-%d.%H-%M-%S`
while :
do
	python wann_train.py -p p/mnist_autoencoder.json -o mnist_autoencoder -n 4 | tee log/training_log_autoencoder_$today.txt
done
