% raw Age
rawfeaturesT = [T(:,9), T(:,11:22)];
rawtargetsT = T(:,4:7);
rawfeatures = table2array(rawfeaturesT)';
rawtargets = table2array(rawtargetsT)';

% Age Mean Normalize
mfeaturesT = [T(:,42), T(:,11:22)];
mtargetsT = T(:,43:46);
mfeatures = table2array(mfeaturesT)';
mtargets = table2array(mtargetsT)';

% Age MaxMin Normalize
mmfeaturesT = [T(:,47), T(:,11:22)];
mmtargetsT = T(:,48:51);
mmfeatures = table2array(mmfeaturesT)';
mmtargets = table2array(mmtargetsT)';

% Age across Normalizer
afeaturesT = [T(:,52), T(:,11:22)];
atargetsT = T(:,53:56);
afeatures = table2array(afeaturesT)';
atargets = table2array(atargetsT)';

% Age Normalizer
nfeaturesT = [T(:,57), T(:,11:22)];
ntargetsT = T(:,58:61);
nfeatures = table2array(nfeaturesT)';
ntargets = table2array(ntargetsT)';

nntool