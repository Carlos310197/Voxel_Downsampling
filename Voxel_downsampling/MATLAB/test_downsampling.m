%%
%Autor: Carlos Huapaya
%%
clear all;close all;clc
%%
%Voxel grid and downsampling

table = readtable("dataXYZ1.csv");
point_cloud = table2array(table);

leaf_size = [300 300 300];
tic
downsampled_cloud = voxel_dowsampling(point_cloud, leaf_size);
fprintf("Elapsed time: %f ms\n",toc*1000);

writematrix(downsampled_cloud, "matlab_downsampling.csv");