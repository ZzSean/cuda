#!/bin/bash
nvcc -arch=sm_70 add_vec.cu -I /usr/local/cuda/include -o add_vec
