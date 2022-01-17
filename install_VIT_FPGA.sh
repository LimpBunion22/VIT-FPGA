#!/bin/bash

export AOCL_BOARD_PACKAGE_ROOT=$HOME/Mustang/QuartusPro_v18.1.0.222_Linux64/a10_1150_sg1
source $HOME/intelFPGA_pro/18.1/hld/init_opencl.sh
export PATH=$HOME/intelFPGA_pro/18.1/quartus/bin:$PATH #$HOME/Mustang/QuartusPro_v18.1.0.222_Linux64/OpenInit.sh

make
cp ./include/netFPGA.h $HOME/workspace_development/include/
# for file in "$include"/*
# do
#     cp "$include"/"$file" $HOME/workspace_development/include/
# done