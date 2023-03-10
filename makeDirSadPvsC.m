% machine learning cooperation with Jens, Philipp, Alex

targetDirs = {'sadPvsC/Controls'; 'sadPvsC/Patients'};
fileName = 'BubblesProtocolComposite.txt';
rppath = './dsetComposite/';
conCodes = {'happyCorrect/', 'happyIncorrect/', 'happyNeutralCorrect/', 'happyNeutralIncorrect/', ...
            'sadCorrect/', 'sadIncorrect/', 'sadNeutralCorrect/', 'sadNeutralIncorrect/'};


if ~exist(targetDirs{1}) 
    for n = 1:length(targetDirs)
    mkdir(targetDirs{n});
    end
end

[img, vp, ~, condition, group, ~] = textread([rppath, fileName], '%s%s%u%u%s%s', 'delimiter', '\t');

sourceImages = strcat(rppath,  conCodes(condition)', img, '.png');
destImages   = strcat(targetDirs(ismember(group, 'experimental')+1), '/', img, '.png');

sourceImages = sourceImages(condition == 8); 
destImages   = destImages(condition == 8); 
group        = group(condition == 8);

for k = 1:length(sourceImages)
    copyfile(sourceImages{k}, destImages{k});        
end

% 1: NE-HA happy + correct 
% 2: NE-HA happy + incorrect
% 3: NE-HA neutral + correct
% 4: NE-HA neutral + incorrect
% 5: NE-SA sad + correct 
% 6: NE-SA sad + incorrect
% 7: NE-SA neutral + correct
% 8: NE-SA neutral + incorrect

