T = ovehicleemission1;
% raw Age
a = 14;
b = 31;
c = 61;
t = 7;
rawfeaturesT = [T(:,a), T(:,b:c)];
rawtargetsT = T(:,t:t+3);
rawfeatures = table2array(rawfeaturesT)';
rawtargets = table2array(rawtargetsT)';

% Age Mean Normalize
a = 62;
t = a+1;
mfeaturesT = [T(:,a), T(:,b:c)];
mtargetsT = T(:,t:t+3);
mfeatures = table2array(mfeaturesT)';
mtargets = table2array(mtargetsT)';

% Age MaxMin Normalize
a = a+5;
t = a+1;
mmfeaturesT = [T(:,a), T(:,b:c)];
mmtargetsT = T(:,t:t+3);
mmfeatures = table2array(mmfeaturesT)';
mmtargets = table2array(mmtargetsT)';

% Age across Normalizer
a = a+5;
t = a+1;
afeaturesT = [T(:,a), T(:,b:c)];
atargetsT = T(:,t:t+3);
afeatures = table2array(afeaturesT)';
atargets = table2array(atargetsT)';

% Age Normalizer
a = a+5;
t = a+1;
nfeaturesT = [T(:,a), T(:,b:c)];
ntargetsT = T(:,t:t+3);
nfeatures = table2array(nfeaturesT)';
ntargets = table2array(ntargetsT)';

nntool