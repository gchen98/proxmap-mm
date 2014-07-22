#!/bin/bash

datafile='iris.txt'
labelfile='iris.labels'
clusterfile='iris.clusters'
pdffile='iris_projection.pdf'

ls iris_clusters/*clusters.txt > $clusterfile
cat plot_path.r | sed "s/DATAFILE/$datafile/" | sed "s/LABELFILE/$labelfile/" | sed "s/CLUSTERFILE/$clusterfile/" | sed "s/OUTPUTPDF/$pdffile/" | R --vanilla 
