#!/bin/bash
./dataset/clean.sh

cd dataset
wget http://mattmahoney.net/dc/text8.zip 
mv text8.zip text8
cd text8
unzip text8.zip
cd ../ptb
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/test.txt
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/train.txt
wget https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/valid.txt
cd ../..

th dataset/process_text8.lua -batch_size 128
th dataset/process_ptb.lua -batch_size 32
