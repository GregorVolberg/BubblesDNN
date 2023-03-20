function [] = get_composite_ml(rawData, rppath, vp_selection, patches, npic, outFileName, conCodes)
 
n           = 0 ;                     % Stored or entered by user.
fid = fopen([rppath, outFileName], 'w'); 
% select participants
tmpcell = {rawData(:).vpcode}';
if all(ismember(vp_selection, 'all'))
    vpnums = 1:numel(rawData);
else
    vpnums = find(ismember(tmpcell, vp_selection));
end
% loop
max_alpha = max(patches{1}(:));

for vpnr = 1:numel(vpnums)
vp = vpnums(vpnr);

fprintf(1, 'Verarbeite Proband %s von %s\n', num2str(vpnr), num2str(numel(vpnums)));
%rawData(vp).b_centers{1,1} % 

for trial = 1:numel(rawData(vp).b_centers)
for scale = 1:5

%bdims = rawData(vp).b_dims{trial}{scale};
%bdims(:,2) = size(patches{scale},1);
%bdims(:,4) = size(patches{scale},1);
[~, bubble_dims, face_coords] = prepare_alpha_ml(patches{scale}, npic{scale}, rawData(vp).b_centers{trial}{scale}, rawData(vp).facedims);
rawData(vp).b_dims{trial}{scale} = bubble_dims;
rawData(vp).f_coords{trial}{scale} = face_coords;
rp = add_alphaplane(patches{scale}, npic{1}, rawData(vp).b_centers{trial}{scale}, rawData(vp).b_dims{trial}{scale}, rawData(vp).f_coords{trial}{scale}); %one patch size at a time
rp(rp > max_alpha) = max_alpha;
a_planes{scale}=rp;
  %[a_planes, b_centers, b_dims, f_coords] = get_alphaplanes(patches, stim.npic(indx(trial),:), n_bubbles, stim.facedims);
  %      [t_stim, ~] = sum_to_target(a_planes, stim.npic(indx(trial),:), use_lowpassresidual, stim.lp_avg);
end

if rawData(vp).outmat(trial,17) > 4 % condition 'sad'
picNumber = rawData(vp).outmat(trial,3) + 60;
else
picNumber = rawData(vp).outmat(trial,3); % condition 'happy'
end

n = n + 1;
nPadded = sprintf( '%07d', n);
[t_stim, ~] = sum_to_target_ml(a_planes, npic(picNumber,:));
imwrite(t_stim/255, [rppath, conCodes{rawData(vp).outmat(trial,17)}, 'comp', nPadded, '.png'], 'PNG'); clear t_stim

picText = ['f', num2str(picNumber + 1000)];
% add 1000, 2000 etc for 
fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ['comp', nPadded], rawData(vp).vpcode, num2str(scale), ...
            num2str(rawData(vp).outmat(trial,17)), rawData(vp).group, picText, num2str(trial), ...
            num2str(rawData(vp).outmat(trial,7)), num2str(rawData(vp).outmat(trial,5)));
end
end
fclose(fid);
end


