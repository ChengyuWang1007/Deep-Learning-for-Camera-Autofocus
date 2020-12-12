%% Model Calibration 
%
% Chengyu Wang, Duke University
% chengyu.wang@duke.edu
%
% Code to calibrate the defocus model as described in 
% "Deep Learning for Camera Autofocus" by Chengyu Wang,
% Qian Huang, Ming Chengyu, Zhan Ma and David J. Brady.

%% Focused image
Focused = double(rgb2gray(imread('data_1400_1400.jpg')))/255; 
Focused = Focused.^2.4;

%% Defocused image
Defocused = double(rgb2gray(imread('data_1400_1700.jpg')))/255;
Defocused = Defocused.^2.4;

%% Calibration
Score = 0;
R = 0;
Alpha = 1;

for r = 26:1:28
    h = fspecial('disk',r);
    Modeled = conv2(Focused,h);
    Modeled = Modeled(floor(size(Modeled,1)/2)-600:floor(size(Modeled,1)/2)+600,...
        floor(size(Modeled,2)/2)-1000:floor(size(Modeled,1)/2)+1000);
    for alpha = 1.017:0.001:1.019
        Modeled_resize = imresize(Modeled,alpha);
        c = normxcorr2(Modeled_resize,Defocused);
        if max(c(:)) > Score
            Score = max(c(:));
            R = r;
            Alpha = alpha;
        end
    end
end

fprintf('For Z0 = 1400, Zi = 1700, r = %f, alpha = %f.\n', R, Alpha);

