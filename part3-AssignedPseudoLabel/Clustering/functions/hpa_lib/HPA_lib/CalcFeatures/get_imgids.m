function [imgids] = get_imgids(image_path,imgpathlist)
imgids = cell(size(imgpathlist));
for i = 1:length(imgpathlist)
    imgid = strrep(imgpathlist{i},'_green.png','');
    imgids{i} =  strrep(imgid,image_path,'');
end
imgids = imgids';
end

