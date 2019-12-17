% convenient funktions:
%----------------------
% imread; -read image
% imwrite; -write image
% imresize; -change size of image
% I = imread(’filename.jpg’); -example of reading image
% Ivec = reshape(I, width*height, 3); - reshape from 3d to 2d
% imshow;



orange = imread('tiger1.jpg');
imshow(orange)
waitforbuttonpress;
orange_small = imresize(orange, 0.5);
[segmentation, centers] = kmeans_segm(orange, 5, 10, 100);
imshow(segmentation)


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
    centers = datasample(image2d(:,:), K)

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
end



