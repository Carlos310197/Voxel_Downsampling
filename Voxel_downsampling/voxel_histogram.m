clear all
close all
clc

table = readtable("log_voxels.csv");
array = table2array(table);

figure(1)
% scatter(voxels,n_points)
% histogram(voxels, n_points)
binranges = 0:1585170;
histogram(array,1585170)