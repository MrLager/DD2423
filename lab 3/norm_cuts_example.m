colour_bandwidth = 10.0; % color bandwidth, controls the similarities calculation
radius = 9;              % maximum neighbourhood distance
ncuts_thresh = 0.08;      % cutting threshold. ncuts_thresh controls the maximum allowed value for Ncut(A, B) for acut to take place
min_area = 200;          % minimum area of segment
max_depth = 30;           % maximum splitting depth
scale_factor = 0.4;      % image downscale factor
image_sigma = 2.0;       % image preblurring scale'

colour_bandwidth = 18.0
radius = 2
ncuts_thresh = 0.06
min_area = 100
max_depth = 7
scale_factor = 0.4
image_sigma = 2.0

% colour_bandwidth = 16.0; % color bandwidth, controls the similarities calculation
% radius = 7;              % maximum neighbourhood distance
% ncuts_thresh = 0.040;      % cutting threshold. ncuts_thresh controls the maximum allowed value for Ncut(A, B) for a cut to take place
% min_area = 105;          % minimum area of segment
% max_depth = 10;           % maximum splitting depth
% scale_factor = 0.4;      % image downscale factor
% image_sigma = 2.1;       % image preblurring scale

% colour_bandwidth = 18.0
% radius = 8
% ncuts_thresh = 0.06
% min_area = 100
% max_depth = 7
% scale_factor = 0.4
% image_sigma = 2.0
% colour_bandwidth = 12.0 
% radius = 6 
% ncuts_thresh = 1
% min_area = 10
% max_depth = 8           
% scale_factor = 0.4      
% image_sigma = 2.0

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(121)
imshow(I)
subplot(122)
imshow(Inew)
imwrite(Inew,'result/normcuts1.png')
imwrite(I,'result/normcuts2.png')

