
function [prob] = mixture_prob (img, K, L, mask)
%Use mask to identify pixels from img, used to estimate a mix of K Gaussian components


% get the size of the image
[width, height, ncolors] = size(img);

%1. Let I be a set of pixels and V a set of K Guassian comp in 3D (R,G,B)
I_vec = im2double(reshape(img, width*height, 3));
M_vec = reshape(mask, width*height, 1);

I_msk = I_vec(M_vec == 1, :);


%2. Randomly initialize the K components using masked pixels
[segmentation, centers] = kmeans_segm(I_msk, K, L, 3456);
cov = cell(K, 1);
cov(:) = {rand * eye(3)};

wk = zeros(1, K);
for i = 1 : K
    wk(i) = sum(segmentation == i) / size(segmentation, 1);
end


wg = zeros(length(I_msk), K);

%3. Iterate L times - run Expectation-Maximation for each loop
for i = 1 : L
    %4. Expectation: Compute probabilities P_ik using masked pixel
    for k = 1:K
        wg(:, k) = wk(k) * mvnpdf(I_msk, centers(k, :), cov{k});
    end
  
 
    prob = wg ./ sum(wg, 2);
    
    %5. Maximization: Update weights, means and covariances using masked pixels
    wk = sum(prob) / length(I_msk);
    for k = 1:K
        norm = sum(prob(:, k));
        centers(k, :) = prob(:, k)' * I_msk / norm;

        diff = I_msk - centers(k, :);
        top = diff' *(diff .* repmat(prob(:,k),[1 3]));
        cov{k} = top / norm;
    end
end

%6. Compute probabilities p(c_i) in Eq.(3) for all pixels I
wg = zeros(length(I_vec), K);
for k = 1:K
    wg(:, k) = wk(k) * mvnpdf(I_vec, centers(k, :), cov{k});
end
prob = sum(wg, 2);


prob = reshape(prob, height, width, 1);

end

% this is the same k-means as in kmeans_segm.m but without the 
% reshaping function since the data is already reshaped in our
% mixturepob.
function [segmentation, centers] = kmeans_segm(image, K, L, seed);
    % get the size of the image
    [width, height, ncolors] = size(image);
    n = length(image);

    % change the image representaion from 3d -> 2d.
    image2d = double(image);

    % add a column to store the cluster that each pixel belongs to
    pixelsBellongingToCenters = zeros(n,3,K);

    % feed the random number generator with a seed
    rng(seed);

    % initiate K random selected clusters
    % centers = datasample(image2d(:,:), K)
    centers = zeros(K, 3);
    centers(1, :) = image(randi(n));
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
   segmentation = indexes;
end