%% settings
includeBSD = true;
folder = '~/workspace/Data/SR/Train';
size_input = 41;
stride = 41;


%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
filepathsBSD = dir(fullfile(strcat(folder,'/BSD300'),'*.jpg'));

for scale = 2:4
    
%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_input, size_input, 1, 1);
count = 0;

[count, data, label] = extractPatch(count, data, label, scale, stride, size_input, filepaths, folder);

if includeBSD == true
   [count, data, label] = extractPatch(count, data, label, scale, stride, size_input, filepathsBSD, strcat(folder,'/BSD300'));
end


order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 


fpX = fopen(strcat('~/workspace/Data/SR/trainX_',num2str(scale),'.bin'),'w');
fpY = fopen(strcat('~/workspace/Data/SR/trainY_',num2str(scale),'.bin'),'w');

fwrite(fpX,permute(data,[2,1,3,4]),'uint8');
fwrite(fpY,permute(label,[2,1,3,4]),'uint8');

fclose(fpX);
fclose(fpY);

end



