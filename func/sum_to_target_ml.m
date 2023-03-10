function [t_stim, t_planes] = sum_to_target_ml(a_planes, npic)
t_stim = zeros(size(npic{1}));
for m = 1:numel(a_planes)
t_planes{m} = npic{m}.*a_planes{m};
t_stim = t_stim + t_planes{m};
end
%if use_lowpassresidual == 0
    t_stim = t_stim + 127;
%elseif  use_lowpassresidual == 1
%    t_stim = t_stim + npic{length(npic)};
%elseif use_lowpassresidual == 2
%    t_stim = t_stim + lp_avg;
%     tdims = size(common_lp);
%     dims  = size(t_stim);
%     margins = round(tdims/2 - dims/2);
%     common_lp(margins(1)+1 : (margins(1)+dims(1)), margins(2)+1 : (margins(2)+dims(2))) = ...
%         common_lp(margins(1)+1 : (margins(1)+dims(1)), margins(2)+1 : (margins(2)+dims(2))) + t_stim;
%     t_stim = common_lp;
%end
end