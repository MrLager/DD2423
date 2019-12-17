clear
image = imread('test.jpg');

K = 10;
L = 15;
seed = 210;

% get the size of the image
[width, height, ncolors] = size(image);

% change the image representaion from 3d -> 2d.
image2d = reshape(double(image), width*height, 3);

% add a column to store the cluster that each pixel belongs to
pixelsBellongingToCenters = zeros(width*height,3,K);

% feed the random number generator with a seed
rng(seed);

% initiate K random selected clusters
centers = zeros(K, 3);
centers(1, :) = image(randi(width*height));
for k = 2:K
    if k == 2
        D2 = pdist2(centers(1, :), image2d, 'squaredeuclidean');
    else
        pdist2(centers(1:k-1, :), image2d, 'squaredeuclidean')
        D2 = min(pdist2(centers(1:k-1, :), image2d, 'squaredeuclidean'))
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

segmentation = uint8(centers(indexes(:), :));
segmentation = reshape(segmentation, [width, height, 3]);

imshow(segmentation)



