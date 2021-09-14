%%
%Autor: Carlos Huapaya

function donwsampled_cloud = voxel_dowsampling(point_cloud, leaf_size)

    %------------------------------
    %----Set the main variables----
    %------------------------------
    
    % Inverse of the leaf size (for performance)
    inv_leaf_size = 1./leaf_size;
    
    % Minimum and maximum dimensions
    min_p = min(point_cloud,[],1);
    max_p = max(point_cloud,[],1);
    
    % Compute the minimum and maximum bounding box values
    min_b = floor(min_p.*inv_leaf_size);
    max_b = floor(max_p.*inv_leaf_size);
    
    % Compute the number of division along the axis
    div_b = max_b - min_b + [1 1 1];
    
    % Set up the division multiplier
    div_b_mul = [1 div_b(1) div_b(1)*div_b(2)];
    n_voxels = div_b(1)*div_b(2)*div_b(3);
    
    %------------------------------
    %----------First pass----------
    %------------------------------
    % Go over all points and insert them into the index vector with calculated indices
	% Points with the same index value (#voxel) will contribute to the same points of resulting cloud point
    ijk = floor(point_cloud.*repmat(inv_leaf_size,size(point_cloud,1),1))-repmat(min_b,size(point_cloud,1),1);
    idx = ijk*(div_b_mul)';
    
    %------------------------------
    %----------Second pass---------
    %------------------------------
    % Sort the index vector using value representing target cell as index (according to the #voxel)
	% In effect, all points belonging to the same output will be next to each other
    [idx_voxels, idx_points] = sort(idx);
    
    %------------------------------
    %----------Third pass----------
    %------------------------------
    % Count output cells. We need to skip all the same adjacent idx values
    out_size = size(idx_voxels,1);
    pos_out = zeros(out_size,1); repeat = zeros(out_size,1);
    counter = 1;
    pos_out(counter) = 1;
    repeat(counter) = repeat(counter) + 1;
    for i = 2:size(point_cloud, 1)
        if (idx_voxels(i)~=idx_voxels(i-1))
            counter = counter + 1;
            pos_out(counter) = i;
        end
        repeat(counter) = repeat(counter) + 1;
    end
    out_size = counter;%update the number of output points
    fprintf("Number of filled voxels: %d.\n", counter);
    
    %------------------------------
    %----------Fourth pass---------
    %------------------------------
    % Compute the centroids and insert them into their final position
    donwsampled_cloud = zeros(counter,3);
    for i = 1:out_size
        rep = repeat(i);
        index = idx_points(pos_out(i):pos_out(i)+rep-1);
        donwsampled_cloud(i,:) = sum(point_cloud(index,:),1)/rep;
    end
end