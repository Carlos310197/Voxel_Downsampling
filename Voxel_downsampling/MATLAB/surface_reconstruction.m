function surface = surface_reconstruction(point_cloud)
    
    A1 = point_cloud(:,:,1); % Load the data
    Np = size(A1,1);
    s=[]; % surface

    for y=0:Np/16-1-1 % -1 for index, -1 because last azimuth is special
        s=[s;16*y+0+1,16*y+1+1,16*(y+1)+0+1]; %1st point right
        for z=1:14
            s=[s;16*y+z+0+1,16*(y+1)+z+0+1,16*(y+1)+z-1+1]; %upper-right
            s=[s;16*y+z+0+1,16*(y+1)+z+0+1,16*y+z+1+1]; %upper-left
        end
        s=[s;16*y+15+0+1,16*(y+1)+15+0+1,16*(y+1)+15-1+1]; %upper-right
    end

    % Plot it with TRISURF
    figure
    trisurf(s, A1(:,1), A1(:,2), A1(:,3));
    axis vis3d
end