#!/bin/bash

# k nearest neighbors
k=5
# soft thresholding parameters
phi=4

datafile='iris.txt'
weightfile=weights.ph$phi.k$k
n=`cat $datafile|wc -l`
dim=`head -n1 $datafile|wc -w`

bindir=../../bin

$bindir/distance $n $dim verbose $datafile 0 < $datafile | $bindir/knn_weights $n $dim $phi $k > $weightfile

cat config.iris.template | sed "s/WEIGHTFILE/$weightfile/" |sed "s/SUBJECTS/$n/" |sed "s/DIMENSION/$dim/" | sed "s/DATAFILE/$datafile/" > config.iris
rm -fr iris_clusters
mkdir -p iris_clusters
$bindir/proxmap cluster config.iris 2>iris.stderr
