% machine learning cooperation with Jens, Philipp, Alex

addpath(genpath('../func'));
outFileName = 'BubblesProtocolComposite2.txt';
rppath = './dsetComposite/';
conCodes = {'happyCorrect/', 'happyIncorrect/', 'happyNeutralCorrect/', 'happyNeutralIncorrect/', ...
            'sadCorrect/', 'sadIncorrect/', 'sadNeutralCorrect/', 'sadNeutralIncorrect/'};

tfolders = dir(rppath);
if ~all(ismember(conCodes, strcat({tfolders.name}, '/')))
for k = 1:numel(conCodes)
    mkdir([rppath ,conCodes{k}]);    
end
end

load ('../../BKH_Bubbles/raw/BubblesRawData.mat', 'rawData');
picfilename = ['../../BKH_Bubbles/', rawData(1).stmfile]; % 'p5_struct_npic_470x349.mat'
[~, ~, npic, mids] = load_stimuli(picfilename);
%patches  = get_patches(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss);
vp_selection = {rawData.vpcode}; %
patches = get_patches_ml(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss, 4);

get_composite_ml2(rawData, rppath, vp_selection, patches, npic, outFileName, conCodes); % write to disk


% 1: NE-HA happy + correct 
% 2: NE-HA happy + incorrect
% 3: NE-HA neutral + correct
% 4: NE-HA neutral + incorrect
% 5: NE-SA sad + correct 
% 6: NE-SA sad + incorrect
% 7: NE-SA neutral + correct
% 8: NE-SA neutral + incorrect

