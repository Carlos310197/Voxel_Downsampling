%%
% Author: Carlos Huapaya
% Surface Reconstruction based on voxel downsampling
%%
close all;clear all;clc
%%
n_donuts = 6;% number of donuts to generate
% sphere = generate_sphere(n_donuts);% get the 1-radius sphere
% 
% % plot the sphere with the given number of donuts
% figure
% hold on
% for i = 1:n_donuts
%    scatter3(sphere(:,1,i),sphere(:,2,i),sphere(:,3,i))
% end
% axis vis3d
% hold off

% load point cloud data from mat file
% load('mining_data/esfera_ideal.mat');
% load('mining_data/nube_puntos_sin_trasl.mat');
load('mining_data/Data_mina.mat')
sphere = point_cloud_without_overlapped;
scatter3(sphere(:,1),sphere(:,2),sphere(:,3))
axis vis3d

% Compute the downsampled cloud and plot
% leaf_size = [0.1,0.1,0.1];
leaf_size = [800,800,800];
point_cloud = permute(sphere,[1 3 2]);
point_cloud = reshape(point_cloud,[],size(sphere,2),1);
tic
[downsampled_cloud, idx_points, pos_out, repeat] = voxel_downsampling(point_cloud,leaf_size);
fprintf("Downsampling elapsed time: %.3f ms\n",toc*1000);
figure
scatter3(downsampled_cloud(:,1),downsampled_cloud(:,2),downsampled_cloud(:,3))
axis vis3d

% load surface 3d data from mat file
% load('mining_data/malla_triang_ideal.mat');
% load('mining_data/malla_triangular_sin_trasl.mat');
surface = Triangle_mesh;

% Generate the surface
surface = surface_reconstruction(surface, point_cloud,downsampled_cloud, n_donuts, idx_points, pos_out, repeat);

%% DXF generation
s=surface;
fname='mina';
fullname=sprintf('%s.dxf',fname);
fid=fopen(fullname,'w');
fprintf(fid,'0\nSECTION\n2\nENTITIES\n0\n');
% minZ=abs(min(min(A1(:,3))));
% maxZ=max(max(A1(:,3)))+minZ;
% C=255*((A1(:,3)+minZ)/maxZ);
for i=1:size(s,1)
    fprintf(fid,'3DFACE\n8\n1\n');
    %create new 3DFACE element
    % fprintf(fid,' 62\n %1.0f\n',C(s(i,1)));
    % fprintf(fid,' 62\n 1\n');
    fprintf(fid,' 62\n %1.0f\n',255*i/size(s,1));
    %corresponding color of the autocad pallete
    fprintf(fid,'10\n %.4f\n 20\n %.4f\n 30\n %.4f\n',point_cloud(s(i,1),1),point_cloud(s(i,1),2),point_cloud(s(i,1),3));
    %vertex 1
    fprintf(fid,'11\n %.4f\n 21\n %.4f\n 31\n %.4f\n',point_cloud(s(i,2),1),point_cloud(s(i,2),2),point_cloud(s(i,2),3));
    %vertex 2
    fprintf(fid,'12\n %.4f\n 22\n %.4f\n 32\n %.4f\n',point_cloud(s(i,3),1),point_cloud(s(i,3),2),point_cloud(s(i,3),3));
    %vertex 3
    fprintf(fid,'13\n %.4f\n 23\n %.4f\n 33\n %.4f\n',point_cloud(s(i,3),1),point_cloud(s(i,3),2),point_cloud(s(i,3),3));
    %vertex 4      
    fprintf(fid,' 0\n');
end
fprintf(fid,'ENDSEC\n 0\nEOF\n');
fclose(fid);