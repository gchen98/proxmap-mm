#!/bin/bash

#project directory here
project='iris'
# specify the original data matrix file here
datafile='iris.txt'
# specify a single column file here, where the values denote the labels for
# each row of the datafile
labelfile='iris.labels'
# the name of the PDF to be generated
pdffile='iris_projection.pdf'

# pipeline begins here

current=$PWD
cd $project
clusterfile='clusters.txt'
ls clusters/*clusters.txt > $clusterfile
cat ../plot_path.r | sed "s/DATAFILE/$datafile/" | sed "s/LABELFILE/$labelfile/" | sed "s/CLUSTERFILE/$clusterfile/" | sed "s/OUTPUTPDF/$pdffile/" | R --vanilla  1>plot.debug
cd $current
