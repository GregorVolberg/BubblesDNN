function [bubble_center, bubble_dims, face_coords] = prepare_alpha_ml(patch, face, b_centers, t_area)
% t_area ist .facedims
% generate random locations for bubbles (center)
if any(t_area > size(face))
    error('impossible target indices'); end
offset = round((size(face) - t_area)/2);

if numel(b_centers) > 0


    for bubble = 1:size(b_centers,1)
    % location in face where bubble is to be centered
    %[trow, tcol]  = ind2sub(t_area, target_center(bubble));
    trow = b_centers(bubble, 1);
    tcol = b_centers(bubble, 2);
     
    % center patch on target location within face and compute intersection
    patch_dims   = RectOfMatrix(patch);
    face_dims    = RectOfMatrix(face);
    t_coords = CenterRectOnPoint(patch_dims, tcol, trow); % left top right bottom, in xy-coordinates!
    t_fit    = ClipRect(face_dims, t_coords);    

    % cut patch to fit into intersection
    start_patch   = (t_fit(1:2) - t_coords(1:2)) + patch_dims(1:2) + 1;
    stop_patch    = (t_fit(3:4) - t_coords(3:4)) + patch_dims(3:4);

    % return values
    bubble_center(bubble,:)  = [trow, tcol];
    bubble_dims(bubble, :)   = [start_patch(2), stop_patch(2), start_patch(1), stop_patch(1)];
    face_coords(bubble, :)   = [t_fit(2) + 1, t_fit(4), t_fit(1) + 1, t_fit(3)];
    end
else
    bubble_center = [];
    bubble_dims   = [];
    face_coords   = [];
end
end