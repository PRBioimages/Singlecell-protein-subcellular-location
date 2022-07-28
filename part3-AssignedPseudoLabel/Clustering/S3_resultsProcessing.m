function resultsProcessing
%% initial paths
for i=0:18
    
if i < 18
    result_dir = ['./SLFs/results/', int2str(i), '/'];
else
    result_dir = './SLFs/results/scv/';
end

testid_path = [result_dir 'imageids.mat'];

if i < 18
    testfeat_path = [result_dir, int2str(i), '_features.csv'];
else
    testfeat_path = [result_dir, 'scv_features.csv'];
end  

if i <18
    dnn_test = ['./SLFs/', 'normal','/', int2str(i), '_features.csv'];
else
    dnn_test = ['./SLFs/', 'scv','/', 'SCV_all_features.csv'];
end

load(testid_path)
[feats] = csvread(testfeat_path);
feats(isnan(feats))=0;
fid = fopen(dnn_test, 'w');
feat_N = size(feats,1);
for i = 1:feat_N
    feat = change_cell(feats(i,:));
    featstr = strjoin(feat,',');
    imageid = imageids{i};
    csvstr = [imageid ',' featstr char(10)];
    fwrite(fid, csvstr);
end
fclose(fid);
end

end

function [feat_cell] = change_cell(feat_mat)
feat_cell = cell(size(feat_mat));
for i = 1:length(feat_mat)
    feat_cell{i} = num2str(feat_mat(i));
end
end