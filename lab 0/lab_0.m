% reansar workspace
clear 
load canoe256
% 
% phone = phonecalc256;
% image(phone)
% waitforbuttonpress;
% 
% % read a matlab variable
% 
% % read an immage defined as a function
% nallo = nallo256;
% 
% % read images in tiff and jpg format
% hund1 = double(imread('dog.jpg'));
% hund2 = double(imread('dog.tiff'));
% 
% % get the size of hund1, we can also use "whos"
% size(hund1);
% 
% % get the largest and smallest value of an image
% %max(max(Canoe));
% %max(Canoe(:));
% 
% % set the axis to be equal
% image(Canoe)
% waitforbuttonpress;
% 
% %rescale to 256 colors
% %colormap = colormap(gray(256));
% image(Canoe)
% waitforbuttonpress;
% 
% axis equal;
% image(Canoe)
% 
% %pause(5);
% %image(Canoe)
% %help showgrey
% 
% waitforbuttonpress;
% showgrey(Canoe)
% waitforbuttonpress;
% showgrey(Canoe, 2)
% waitforbuttonpress;
% showgray(Canoe, 256)
% waitforbuttonpress;
% 
% % read phone picture
% % phone = phonecalc256;
% image(phone)
% waitforbuttonpress;
% 
% for c = [64,32,16,8,4,2]
%     showgray(phone, c)
%     waitforbuttonpress;
% end
% 
% vad = whatisthis256;
% showgray(vad)
% waitforbuttonpress;
% 
% zmax = max(vad(:));
% zmin = min(vad(:));
% showgray(vad, 256, 1, 60)
% we only show the part of the image where there is alot of information
% in the image.
% 
% colormap(gray(256));
% image(nallo256)
% waitforbuttonpress;
% colormap(cool);
% image(nallo256)
% waitforbuttonpress;
% colormap(hot);
% image(nallo256)
% waitforbuttonpress;

% ????? den hotta då den är ljusare och färjerna passar med kontext

% function pixels = rewsubsample(inpic)
%     [m, n] = size(inpic);
%     pixels = inpic(1:2:m, 1:2:n);

%ninepic = indexpic9;
% ninepic = indexpic9;
% for i = 1:4
%     ninepic = rawsubsample(ninepic);
%     image(ninepic)
%     waitforbuttonpress;
% end    

% we decrease the resolution
% function pixels = binsubsample(inpic)
% raw just dumps 50% of the pixels
% bin 
% colormap = colormap(gray(256));
% phone1 = phonecalc256;
% while size(phone1) > 1
%     phone1 = binsubsample(phone1);
%     image(phone1)
%     waitforbuttonpress;
% end 
% 
% phone2 = phonecalc256;
% while size(phone2) > 1
%     phone2 = rawsubsample(phone2);
%     image(phone2)
%     waitforbuttonpress;
% end
showgray(Canoe)
waitforbuttonpress;

neg1 = - Canoe;
%showgray(neg1)
waitforbuttonpress;
neg2 = 255 - Canoe;
%showgray(neg2)
waitforbuttonpress;

nallo = nallo256;
%showgray(nallo.^(1/3))
waitforbuttonpress;
%showgray(cos(nallo/10))
waitforbuttonpress;

hist(neg1(:))
waitforbuttonpress;
hist(neg2(:))
waitforbuttonpress;

% inpic = phonecalc256;
% outpic = compose(lookuptable, inpic);
% negtransf = ( 255 : -1 : 0)';
% neg3 = compose(negtransf, Canoe + 1);

phone = phonecalc256;
image(phone)
waitforbuttonpress;

freqphone = fft2(phone);
image(abs(freqphone))
waitforbuttonpress;
phone = ifft2(freqphone);
image(abs(phone))



