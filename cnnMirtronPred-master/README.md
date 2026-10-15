This software are Convolutional Neural Network(cnn)-based miRNA classifier. We only use the nucleotide/base sequence of pre-miRNA to predict the classification. Based on the mechanism of generation, miRNA can be classified into mirtron and classical miRNA.

We trained our models on the partiton of human pre-miRNAs dataset and the performance was evaluated on the test dataset.

Using the well-trained model, the pre-miRNA sequence can be easily classified.

This implementation is written in python3 and numpy and tensorflow is required.

Python Requirements

Python 3.5.2

pip install numpy 
pip install tensorflow

USAGE: python isMirtron.py -s pre-miRNA sequnece
    or python isMirtron.py --sequence  pre-miRNA sequnece
for example: python isMirtron.py -s GTAAGTCTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG

USAGE HELP: python isMirtron.py -h or python isMirtron --help

# Using with CSV input

In this FORK the classifier was adapted to accept CSV entries in the format:

```txt
id,seq
seq1,ACGTACGTACGTACGTACGTACGT
seq2,CGTACGTACGTACGTACGTACGTA
seq3,GTACGTACGTACGTACGTACGTAC
```

## Install

You need to create a new conda environment

```bash
cd CNNMIRTRONPRED_DIR
conda create -n cnnmirtronpred -y python=3.5
conda install -n cnnmirtronpred -c anaconda -y numpy
conda install -n cnnmirtronpred -c anaconda -y tensorflow
```

Or use `tensorflow-gpu` if you have a GPU.

```bash
cd CNNMIRTRONPRED_DIR
conda create -n cnnmirtronpred -y python=3.5
conda install -n cnnmirtronpred -c anaconda -y numpy
conda install -n cnnmirtronpred -c anaconda -y tensorflow-gpu
```

## Run - Method n 01

1. You need to activate the environment

```bash
conda activate cnnmirtronpred
```

2. Classify your CSV with

```bash
python isMirtronCsv.py --csv data/samples.csv
```

## Run - Method n 02

Run directly in the environment.

```bash
cd CNNMIRTRONPRED_DIR
conda run -n cnnmirtronpred python isMirtronCsv.py --csv samples.csv
```

## Utility: Convert your FASTA file to CSV

```bash
echo 'id,seq' > samples.csv
awk -f fasta2csv.awk samples.fa >> samples.csv
```

## Utility: Convert sequence list to CSV

```bash
echo 'id,seq' > samples.csv
awk 'BEGIN{OFS=","} { print "seq" sprintf("%05d", NR), $0 }' samples.txt >> samples.csv
```

## Utility: Convert sequence list to CNN input file

```bash
echo 'id,seq,class' > datac/ds_mirtrons.csv
awk 'BEGIN{OFS=","} { print "seq" sprintf("%05d", NR), $0,"TRUE" }' datac/mirtrons_gan.txt | head -n 707 >> datac/ds_mirtrons.csv

echo 'id,seq,class' > datac/ds_canonical.csv
awk 'BEGIN{FS=","; OFS=","} { if ($3 == "FALSE") { print $0 } }' data/miRBase_set.csv >> datac/ds_canonical.csv
```

## Treinamento do modelo

O trabalho original usou 707 canonical miRNAs and 417 mirtrons, ou seja, uma proporção de 1 miRNA para cada 0.58 mirtrons.
