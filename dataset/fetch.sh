#!/bin/bash
./dataset/clean.sh

cd dataset
wget http://mattmahoney.net/dc/text8.zip 
unzip text8.zip
cd ..

th dataset/process_text8.lua -batch_size 128
