
function [alpha_plane] = add_alphaplane(patch, face, bubble_center, bubble_dims, face_coords);
    alpha_plane = zeros(size(face));
    if ~isempty(bubble_center)
        for bubble = 1:size(bubble_center,1)
          resized_patch = patch(bubble_dims(bubble, 1):bubble_dims(bubble, 2),...
                                bubble_dims(bubble, 3):bubble_dims(bubble, 4));
          tmp = zeros(size(face));
              tmp(face_coords(bubble, 1): face_coords(bubble, 2),...
                  face_coords(bubble, 3):face_coords(bubble, 4)) = resized_patch;    

        alpha_plane = alpha_plane + tmp;
        end
    end
end
