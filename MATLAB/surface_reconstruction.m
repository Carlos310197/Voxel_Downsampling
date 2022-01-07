%%
% Author: Carlos Huapaya
function surface = surface_reconstruction(surface,point_cloud, downsampled_cloud, n_donuts, idx_points, pos_out, repeat)
    
    tic
    Np = 16*1024;% 16384 points
%     s = [];% conections. Size: 2*(16-1)*(1024-1)
% 
%     % Compute the conections only one time
%     for y=0:Np/16-1-1 % -1 for index, -1 because last azimuth is special
%         s=[s;16*y+0+1,16*y+1+1,16*(y+1)+0+1]; %1st point right
%         for z=1:14
%             s=[s;16*y+z+0+1,16*(y+1)+z+0+1,16*(y+1)+z-1+1]; %upper-right
%             s=[s;16*y+z+0+1,16*(y+1)+z+0+1,16*y+z+1+1]; %upper-left
%         end
%         s=[s;16*y+15+0+1,16*(y+1)+15+0+1,16*(y+1)+15-1+1]; %upper-right
%     end
%     
%     surface = [];
%     for i = 1:n_donuts
%         surface = [surface;s+Np*(i-1)];
%     end

    % Plot it with TRISURF
%     figure
%     trisurf(surface, point_cloud(:,1), point_cloud(:,2), point_cloud(:,3));
%     axis vis3d
    
    % Surface reconstruction using the downsampled cloud
    for i = 1:size(downsampled_cloud,1)
        if repeat(i)>1
            % get all points in a voxel
            temp = idx_points(pos_out(i):pos_out(i)+repeat(i)-1);
            
            % get the first point in the voxel
            idx = idx_points(pos_out(i));
            
            % replace all points that fall in same voxel
            % in the surface conection array
            for j = 1:repeat(i)
                surface(surface==temp(j)) = idx;
                point_cloud(idx,:) = downsampled_cloud(i,:);
            end
        end
    end
    
    surface_new = [];
    % Delete all points that repeat 2 or 3 times in the surface conection
    for i = 1:size(surface,1)
        triangle = surface(i,:);
        if (triangle(1)~=triangle(2) && triangle(2)~=triangle(3) && triangle(1)~=triangle(3)) 
            surface_new = [surface_new;triangle];
        end
    end
    fprintf("Surface reconstruction elapsed time: %.3f s\n",toc);
    
    % Plot it with TRISURF
    figure
%     scatter3(downsampled_cloud(:,1),downsampled_cloud(:,2),downsampled_cloud(:,3),'.','red');
%     hold on
    trisurf(surface_new, point_cloud(:,1), point_cloud(:,2), point_cloud(:,3));
%     axis vis3d
end