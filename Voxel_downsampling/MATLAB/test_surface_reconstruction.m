%%
% Author: Carlos Huapaya
% Surface Reconstruction based on voxel downsampling
%%
close all;clear all;clc
%%
n_donuts = 2;% number of donuts to generate
sphere = generate_sphere(n_donuts);% get the 1-radius sphere

% plot the sphere with the given number of donuts
figure
hold on
for i = 1:n_donuts
   scatter3(sphere(:,1,i),sphere(:,2,i),sphere(:,3,i))
end
hold off

% Compute the downsampled cloud and plot
leaf_size = [0.05,0.05,0.05];
point_cloud = permute(sphere,[1 3 2]);
point_cloud = reshape(point_cloud,[],size(sphere,2),1);
donwsampled_cloud = voxel_downsampling(point_cloud,leaf_size);
figure
scatter3(donwsampled_cloud(:,1),donwsampled_cloud(:,2),donwsampled_cloud(:,3))


surface = surface_reconstruction(sphere);