%%Estos son los datos mejorados
clear;clc;close all
load('Data_mina')
temp=spheric_point_cloud_without_overlapped;
figure(1);scatter3(temp(:,1),temp(:,2),temp(:,3))

temp=point_cloud_without_overlapped;
figure(2);scatter3(temp(:,1),temp(:,2),temp(:,3))

T=Triangle_mesh;
figure(3);trimesh(T,temp(:,1),temp(:,2),temp(:,3))

% temp=spheric_point_cloud_without_overlapped;
% figure(4);trimesh(T,temp(:,1),temp(:,2),temp(:,3))
