#!/bin/bash

# configure these settings as appropriate

# k nearest neighbors
k=5
# soft thresholding parameters
phi=4
# executable directory
# uncomment below for statically linked binaries
bindir=.
# uncomment below for dynamically linked binaries
#bindir=../../bin
# the base directory of your project
project='iris'
# any optional settings can be edited here
configtemplate='config.template'
# the data matrix file is specified here
datafile='iris.txt'
# the name of the weights file that will be generated
weightfile='weights.txt'

# the pipeline begins here

current=$PWD
cd $project

n=`cat $datafile|wc -l`
dim=`head -n1 $datafile|wc -w`
../$bindir/distance $n $dim verbose $datafile 0 < $datafile | ../$bindir/knn_weights $n $dim $phi $k > $weightfile

cat $configtemplate | sed "s/WEIGHTFILE/$weightfile/" |sed "s/SUBJECTS/$n/" |sed "s/DIMENSION/$dim/" | sed "s/DATAFILE/$datafile/" |sed "s/OUT_PATH/clusters/" > config.txt
rm -fr clusters
mkdir -p clusters
../$bindir/convexcluster config.txt

cd $current
