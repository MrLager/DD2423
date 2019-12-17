K = 12;               % number of clusters used
L = 100;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale

% I = imread('orange.jpg');
I = imread('tiger1.jpg');
% I = imread('tiger2.jpg');
% I = imread('tiger3.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
Iold = I;
for K = 2:9
    tic
    [ segm, centers ] = kmeans_segm(I, K, L, seed);
    toc

    Inew = mean_segments(Iback, segm);
    Io = overlay_bounds(Iback, segm);
    % if isequal(Inew, Iold)
    %     break;
    % end
%     Iold = Inew;
    subplot(3,3,K)
    imshow(Io)
end
% L
imwrite(Inew,'result/kmeans1.png')
imwrite(Io,'result/kmeans2.png')

function [segmentation, centers] = kmeans_segm(image, K, L, seed);
    % get the size of the image
    [width, height, ncolors] = size(image);

    % change the image representaion from 3d -> 2d.
    image2d = reshape(double(image), width*height, 3);

    % add a column to store the cluster that each pixel belongs to
    pixelsBellongingToCenters = zeros(width*height,3,K);

    % feed the random number generator with a seed
    rng(seed);

    % initiate K random selected clusters
    % centers = datasample(image2d(:,:), K)
    centers = zeros(K, 3);
    centers(1, :) = image(randi(width*height));
    for k = 2:K
        if k == 2
            D2 = pdist2(centers(1, :), image2d, 'squaredeuclidean');
        else
            D2 = min(pdist2(centers(1:k-1, :), image2d, 'squaredeuclidean'));
        end

        probs = D2 / sum(D2);
        cumprobs = cumsum(probs);
        t = rand();
        [~, I] = max(cumprobs >= t); % yields index of first value larger than t

        centers(k, :) = image2d(I, :);
    end
    
    % interate L times
    for l = 1:L
        [distances, indexes] = min(pdist2(centers, image2d));
        for Kindex = 1:K
            centers(Kindex, :) = mean(image2d(indexes(:)==Kindex, :));
        end
    end
    
   %  
   segmentation = reshape(uint8(indexes), [width, height]);
end
