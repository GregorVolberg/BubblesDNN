% ==================
% 2023-03-20
% must first compute the response plane
% per scale or overall? Test with S14
% 
% G. Volberg
% ==================

% paths etc
addpath(genpath('../func'));
rppath = './allzimages/';
if ~exist('allzimages', 'dir')
    mkdir(rppath);
end
outFileName = [rppath, 'BubblesProtocolzImages.txt'];

% load files
load ('../../BKH_Bubbles/raw/BubblesRawData.mat', 'rawData');
picfilename = ['../../BKH_Bubbles/', rawData(1).stmfile]; % 'p5_struct_npic_470x349.mat'
[~, ~, npic, mids] = load_stimuli(picfilename);
fm = load('../../BKH_Bubbles/face_mask.mat'); % fm.fmask, logical

% global variables
patches = get_patches_ml(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss, 4);
max_alpha = max(patches{1}(:)); % for re-scaling
zcutoff = 1.96;
cons = [1, 2; % NE-HA happy,   correct = 1, incorrect = 2 
        3, 4; % NE-HA neutral, correct = 1, incorrect = 2
        5, 6; % NE-SA sad,     correct = 1, incorrect = 2
        7, 8];% NE-SA neutral, correct = 1, incorrect = 2

 

% prepare loop
vp_selection = {rawData.vpcode}; 
fid = fopen(outFileName, 'w'); 
n           = 0 ;                     
zThreshold = 1.96; % one-sided p<05 

for vp = 1:numel(vp_selection)
    fprintf(1, 'Verarbeite Proband %s von %s\n', num2str(vp), num2str(numel(vp_selection)));
for conditions = 1:size(cons, 1)
corr = find(rawData(vp).outmat(:,17) == cons(conditions, 1));
inc = find(rawData(vp).outmat(:,17)  == cons(conditions, 2));
NonFace_perTrial  = zeros([numel(inc), rawData(1).picdims, 5]);
    for trl = 1:length(inc)
        trial = inc(trl);
        a_planes = cell(1,5);
        for scale = 1:5
            [~, bubble_dims, face_coords] = prepare_alpha_ml(patches{scale}, npic{scale}, rawData(vp).b_centers{trial}{scale}, rawData(vp).facedims);
            rawData(vp).b_dims{trial}{scale} = bubble_dims;
            rawData(vp).f_coords{trial}{scale} = face_coords;
            rp = add_alphaplane(patches{scale}, npic{1}, rawData(vp).b_centers{trial}{scale}, rawData(vp).b_dims{trial}{scale}, rawData(vp).f_coords{trial}{scale}); %one patch size at a time
            rp(rp > max_alpha) = max_alpha;
            a_planes{scale}=rp;
        end
    NonFace_perTrial(trl, :,:,:) = reshape(cell2mat(a_planes), [470, 349, 5]);
    clear a_planes 
    end
sdi      = squeeze(std(NonFace_perTrial)); % per Pixel
meanInc  = squeeze(mean(NonFace_perTrial, 1)); % per Pixel

% now the correct trials
for trl = 1:length(corr)
    trial = corr(trl);
    a_planes = cell(1,5);
    for scale = 1:5
        [~, bubble_dims, face_coords] = prepare_alpha_ml(patches{scale}, npic{scale}, rawData(vp).b_centers{trial}{scale}, rawData(vp).facedims);
        rawData(vp).b_dims{trial}{scale} = bubble_dims;
        rawData(vp).f_coords{trial}{scale} = face_coords;
        rp = add_alphaplane(patches{scale}, npic{1}, rawData(vp).b_centers{trial}{scale}, rawData(vp).b_dims{trial}{scale}, rawData(vp).f_coords{trial}{scale}); %one patch size at a time
        rp(rp > max_alpha) = max_alpha;
        a_planes{scale}=rp;
    end
    imgPerTrial = reshape(cell2mat(a_planes), [470, 349, 5]);
    diffimg = imgPerTrial - meanInc;
    zDiff   = diffimg ./ sdi;
    zStat   = double(zDiff > zThreshold);
    for nscale = 1:5
    %zalphaPlane{nscale} = imgaussfilt(zStat(:,:,nscale), rawData(1).sd_Gauss*2);
    zalphaPlane{nscale} = imfilter(zStat(:,:,nscale), patches{nscale}/sum(patches{nscale}(:)), 'same');
    end
    
    if rawData(vp).outmat(trial,17) > 4 % condition 'sad'
    picNumber = rawData(vp).outmat(trial,3) + 60;
    else
    picNumber = rawData(vp).outmat(trial,3); % condition 'happy'
    end
    n = n + 1;
    nPadded = sprintf( '%07d', n);

    [t_stim, ~] = sum_to_target_ml(zalphaPlane, npic(picNumber,:));
    imwrite(t_stim/255, [rppath, 'comp', nPadded, '.png'], 'PNG'); clear t_stim
    picText = ['f', num2str(picNumber + 1000)];
    fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ['comp', nPadded], rawData(vp).vpcode, num2str(scale), ...
            num2str(rawData(vp).outmat(trial,17)), rawData(vp).group, picText, num2str(trial), ...
            num2str(rawData(vp).outmat(trial,7)), num2str(rawData(vp).outmat(trial,5)));
end
end
end
fclose(fid);