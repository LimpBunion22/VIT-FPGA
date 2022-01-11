#!/bin/bash

source $HOME/Mustang/QuartusPro_v18.1.0.222_Linux64/OpenInit.sh
make
cp ./include/netFPGA.h $HOME/workspace_development/include/
# for file in "$include"/*
# do
#     cp "$include"/"$file" $HOME/workspace_development/include/
# done