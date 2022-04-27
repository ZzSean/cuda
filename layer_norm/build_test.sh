#!/bin/bash
nvcc -arch=sm_70 --ptxas-options=-v test.cu -o test
