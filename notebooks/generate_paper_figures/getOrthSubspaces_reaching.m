%%% Identifies orthogonal sets of dimensions that maximize variance in
%%% prep, move, and posture epochs 

% running this analysis in matlab because Gamal didn't write it in python 
function getOrthSubspaces_reaching()

% add the directory containing Gamal's code to the current path
addpath('/Users/sherryan/glaserlab/sca_analysis_parent/sca_analysis/gamal/');
% % 
% % where can we find the monkeys' data?
% load_folder = '/Users/sherryan/glaserlab/sca_analysis_parent/sca_analysis/';
% 
% % where do we want to save the loadings and projections?
% save_folder = '/Users/sherryan/glaserlab/sca_analysis_parent/sca_analysis/';

% where can we find the monkeys' data?
load_folder = '/Users/sherryan/glaserlab/sca_analysis_parent/sca_analysis/bootstraps/';

% where do we want to save the loadings and projections?
save_folder = '/Users/sherryan/glaserlab/sca_analysis_parent/sca_analysis/bootstraps/';


%% IMPORTANT TIME POINTS

% target on: 20
% outward movement: 77
% return movement: 200 

%% define prep, move, and posture epochs
% note that I'm using both outward and return epochs for prep and move
% (because the movements themselves are slightly different) 

prepWindow = [32:67 165:190];
% moveWindow = [72:102 195:225];
% postWindow = 132:162;
mixWindow = [72:162 195:225];
% mixWindow = [72:102 132:162 195:225];

%% number of dimensions to find
% note that this is per-epoch
numDims = 12;

%% Balboa first

% define monkey name
% monkName = 'Balboa';
monkName = 'Alex';

% load data
% 'data' are trial-averaged, filtered firing rates (C x N x T) 
% load([load_folder monkName '_proc_neural_data.mat']);
load([load_folder monkName '_bootstraps_100_dim8.mat']);

%% 

numConds = 8;
% numN = 124;
numN = 130;
timeL = 301;
% 
prepDims_list = zeros(numN,numDims,size(bs_neuron_proc_activity, 3));
mixDims_list = zeros(numN,numDims,size(bs_neuron_proc_activity, 3));
prepProj_list = zeros(timeL*numConds,numDims,size(bs_neuron_proc_activity, 3));
mixProj_list = zeros(timeL*numConds,numDims,size(bs_neuron_proc_activity,3));

% prepDims_list = zeros(numN,numDims,size(neuron_proc_activity, 3));
% mixDims_list = zeros(numN,numDims,size(neuron_proc_activity, 3));
% prepProj_list = zeros(timeL*numConds,numDims,size(neuron_proc_activity, 3));
% mixProj_list = zeros(timeL*numConds,numDims,size(neuron_proc_activity,3));

% prepDims_list = zeros(numN,numDims,size(neuron_proc_activity, 3));
% moveDims_list = zeros(numN,numDims,size(neuron_proc_activity, 3));
% postDims_list = zeros(numN,numDims,size(neuron_proc_activity, 3));
% prepProj_list = zeros(timeL*numConds,numDims,size(neuron_proc_activity, 3));
% moveProj_list = zeros(timeL*numConds,numDims,size(neuron_proc_activity, 3));
% postProj_list = zeros(timeL*numConds,numDims,size(neuron_proc_activity, 3));

for i = 1:size(bs_neuron_proc_activity, 3)
% for i = 1:size(neuron_proc_activity, 3)
    time = repmat([1:timeL]',numConds,1);
    data_norm = bs_neuron_proc_activity(:,:,i);
    prepA = data_norm(ismember(time,prepWindow),:);
    mixA = data_norm(ismember(time,mixWindow),:);
    % moveA = data_norm(ismember(time,moveWindow),:);
    % postA = data_norm(ismember(time,postWindow),:);
    % put data into a structure
    DataStruct(1).A = prepA;
    DataStruct(1).dim = numDims;
    
    DataStruct(2).A = mixA;
    DataStruct(2).dim = numDims;
    % 
    % DataStruct(2).A = moveA;
    % DataStruct(2).dim = numDims;
    % 
    % DataStruct(3).A = postA;
    % DataStruct(3).dim = numDims;
    % 
    
    % run gamal's method
    QSubspaces = getSubspaces(DataStruct);
    prepDims = QSubspaces(1).Q;
    mixDims = QSubspaces(2).Q;
    % moveDims = QSubspaces(2).Q;
    % postDims = QSubspaces(3).Q;

    % pull out our dimensions
    prepDims_list(:,:,i) = prepDims;
    mixDims_list(:,:,i) = mixDims;
    % moveDims_list(:,:,i) = moveDims;
    % postDims_list(:,:,i) = postDims;
    
    prepProj = data_norm * prepDims;
    mixProj = data_norm * mixDims;
    % moveProj = data_norm * moveDims;
    % postProj = data_norm * postDims;

    % % project into these spaces
    prepProj_list(:,:,i) = prepProj;
    mixProj_list(:,:,i) = mixProj;
    % moveProj_list(:,:,i) = moveProj;
    % postProj_list(:,:,i) = postProj;

    % proj = cat(2,prepProj(:,1:3),moveProj(:, 1:3),postProj(:,1:3));
    proj = cat(2,prepProj(:,1:3),mixProj(:, 1:3));

    % define some plotting colors
    % blue: prep; red: move; green: posture
    colors = [19 159 255 162 20 47 17 176 5]./255;
    % colors = reshape(repmat(colors,3,1),[],3);
    colors = reshape(repmat(colors,3,1),[],3);
    if i < 6
        % plot
        figure;
        % for D = 1:9
        % 
        %     subplot(9,1,D);hold on;
        for D = 1:6
        
            subplot(6,1,D);hold on;            
            plot(reshape(proj(:,D),[],numConds),'linewidth',1,'color',colors(D,:));
            
            % set consistent y axis
            ylim([-2 2]);
        end
        saveas(gcf,[save_folder monkName '_bs' num2str(i) '_MIX2_gamalLoadings.png']);
        close();
    end

end

save([save_folder monkName '_bs_100_MIX2_gamalLoadings.mat'],'prepDims_list','mixDims_list',...
    'prepProj_list','mixProj_list','prepWindow','mixWindow');
% save([save_folder monkName '_MIX2_gamalLoadings.mat'],'prepDims_list','mixDims_list',...
%     'prepProj_list','mixProj_list','prepWindow','mixWindow');
% save([save_folder monkName '_bs_100_gamalLoadings.mat'],'prepDims_list','moveDims_list','postDims_list',...
%     'prepProj_list','moveProj_list','postProj_list','prepWindow','moveWindow','postWindow');
% save([save_folder monkName '_gamalLoadings.mat'],'prepDims_list','moveDims_list','postDims_list',...
%     'prepProj_list','moveProj_list','postProj_list','prepWindow','moveWindow','postWindow');

%% 

% % downsample the data by a factor a 10 (to match ssa analyses)
% data = data(:,:,1:10:end);
% 
% % subtract cross-condition mean
% data_dm = data - mean(data);

% some useful numbers
% [numConds, numN,timeL] = size(data); 

% % reshape data to be size CT x N 
% data_dm = reshape(permute(data_dm,[3 1 2]),[],numN);
% 
% % normalize data (fr range + 5);
% data_norm = data_dm ./ (range(data_dm) + 5);

% define a time mask
% time = repmat([1:timeL]',numConds,1);



% pull out our prep, move, and posture data
% prepA = data_norm(ismember(time,prepWindow),:);
% moveA = data_norm(ismember(time,moveWindow),:);
% postA = data_norm(ismember(time,postWindow),:);

% % put data into a structure
% DataStruct(1).A = prepA;
% DataStruct(1).dim = numDims;
% 
% DataStruct(2).A = moveA;
% DataStruct(2).dim = numDims;
% 
% DataStruct(3).A = postA;
% DataStruct(3).dim = numDims;
% 
% 
% % run gamal's method
% QSubspaces = getSubspaces(DataStruct);
% 
% % pull out our dimensions
% prepDims = QSubspaces(1).Q;
% moveDims = QSubspaces(2).Q;
% postDims = QSubspaces(3).Q;
% 
% % project into these spaces
% prepProj = data_dm * prepDims;
% moveProj = data_dm * moveDims;
% postProj = data_dm * postDims;



% plot activity in the top 3 dimensions of each space just as a sanity
% check

% % concatenate for simplicity
% proj = cat(2,prepProj(:,1:3),moveProj(:,1:3),postProj(:,1:3));
% 
% % define some plotting colors
% % blue: prep; red: move; green: posture
% colors = [19 159 255 162 20 47 17 176 5]./255;
% colors = reshape(repmat(colors,3,1),[],3);
% 
% % plot
% figure;
% for D = 1:9
% 
%     subplot(9,1,D);hold on; 
%     plot(reshape(proj(:,D),[],numConds),'linewidth',1,'color',colors(D,:));
%     
%     % set consistent y axis
%     ylim([-120 140]);
% end
% 
% % save projections and loadings
% save([save_folder monkName '_gamalLoadings.mat'],'prepDims','moveDims','postDims',...
%     'prepProj','moveProj','postProj','prepWindow','moveWindow','postWindow');








