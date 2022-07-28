function [new_feats] = featprocessing(old_feats)
new_feats = []
for i=1:size(old_feats, 2)
    old_feat = old_feats(:,i);
    NANind = find(ismissing(old_feat)==1);
    if length(NANind)>= length(old_feat)/2
        old_feat(NANind) = [];
    else
        
    end
    new_feats = [new_feats,old_feat];
end
end

