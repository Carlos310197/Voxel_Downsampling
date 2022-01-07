function point_cloud = generate_sphere(n_donuts)
    %characteristic of LIDAR OS1-16
        %Angle elevation range; [-16.677; 16,545]
            %In this particular device we have 16 laser beams/points
            %Also, every azimuth block has an angle offset (-1.24°)
            %AZBLK=[15.379 13.236 11.128 9.03 6.941 4.878 2.788 0.705 -1.454 -3.448 -5.518 -7.601 -9.697 -11.789 -13.914 -16.062]
        %About encode range: [0,90111]
        %In mode 1023 azimuth. Every azimuth increments in 88 ticks
            %1 tick = 360/90112= 0.003995°.     Aprox -> 4°
            %88 tick = 360*88/90112 = 0.3516°
        %In total, we obtain 1024*16 = 16384 points in every LIDAR scan (donut)

    %Set Local variables:
    AZBLK=[15.379 13.236 11.128 9.03 6.941 4.878 2.788 0.705 -1.454 -3.448 -5.518 -7.601 -9.697 -11.789 -13.914 -16.062];
    azimuth_offset=-1.24*(pi/180); 
    n_AZBLK=1024;
    ticks_per_azimuth=90112/n_AZBLK;
    ang_diff=2*pi/n_AZBLK;

    %Interception with synthetic spheric room sphere with radius=1
    n_rotation=n_donuts;
    temp=zeros(n_AZBLK*length(AZBLK),3);
    point_cloud=zeros(n_AZBLK*length(AZBLK),3,n_donuts);

    %Rotation -> 31°
    rot_angle=-31*pi/180;
    rot_matrix=[cos(rot_angle) -sin(rot_angle)  0;
                sin(rot_angle) cos(rot_angle)   0;
                0               0               1];
    R=1;%radius sphere
    for i=1:n_rotation
        for j=1:n_AZBLK
            x=sin(ang_diff*(j-1)+azimuth_offset);
            z=-cos(ang_diff*(j-1)+azimuth_offset);
            for k=1:length(AZBLK)
                y=tan(AZBLK(k)*pi/180);
                new_xyz=rot_matrix^(i-1)*[x;y;z];
                %sphere
                [thetha,phi,~]=cart2sph(new_xyz(1),new_xyz(2),new_xyz(3));
                [temp_x,temp_y,temp_z]=sph2cart(thetha,phi,R);           
                temp(length(AZBLK)*(j-1)+k,:)=[temp_x,temp_y,temp_z];
            end
        end
        point_cloud(:,:,i)=temp;
    end
end