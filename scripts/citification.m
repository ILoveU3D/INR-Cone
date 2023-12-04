clc;clear;
citi = @(x, y) ssim(x,y);
x = open_file("/media/wyk/wyk/Data/raws/label.raw");
y = open_file("/media/wyk/wyk/Data/raws/output.raw");
x = x ./ max(y);
y = y ./ max(y);
citi(x, y)

function file = open_file(path)
fileID = fopen(path, "r"); % Open the current proj file
file = fread(fileID,'float');
fclose(fileID);
end