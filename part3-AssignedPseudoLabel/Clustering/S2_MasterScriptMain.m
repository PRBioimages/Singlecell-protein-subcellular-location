function MasterScriptMain
for i=0:18
%% add paths 
addpath(genpath('./functions'))
addpath('./')
addpath(['./IFCellDataset/normal/', int2str(i), '/'])

%addpath('../test/')
%% initial
if i < 18
    infolder = ['./IFCellDataset/normal/', int2str(i), '/'];
else
    infolder = './IFCellDataset/scv/';
end

%infolder = '../test';
if i <18
    outfolder = './SLFs/results';
else
    outfolder = './SLFs/results';
end

grext = '_green';
blueext = '_blue';
yellowext = '_yellow';
redext = '_red';
extensions = {blueext,grext,redext,yellowext};
color = [];
resolution = 0.08;
seg_channels = {'er','mt'};
dirpatterns = {};
imgpattern = '';
mstype = 'confocal';
imgtype = '.png';

submitstruct.indir = infolder;
submitstruct.outdir = outfolder;
submitstruct.resolution = resolution;
submitstruct.color = color;
submitstruct.extensions = extensions;
submitstruct.pattern = imgpattern;
submitstruct.mstype = mstype;
submitstruct.seg_channels = seg_channels;

extensions = strcat(submitstruct.extensions,imgtype);
%% process images
[~,exit_code] = process_img(submitstruct.indir,submitstruct.outdir,...
    submitstruct.resolution,submitstruct.color,extensions,...
    submitstruct.pattern,submitstruct.mstype,submitstruct.seg_channels,[],1);

end
end