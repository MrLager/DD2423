clear
% step 1 create image
Fhat = zeros(128, 128);
p = 5;
q = 9;
Fhat(p, q) = 1;
showgrey(Fhat);

% step 2 
% F = ifft2(Fhat);
% Fabsmax = max(abs(F(:)));
% subplot(221)
% showgrey(real(F), 64, -Fabsmax, Fabsmax) % real part
% waitforbuttonpress;
% subplot(222)
% showgrey(imag(F), 64, -Fabsmax, Fabsmax) % imaginary part
% waitforbuttonpress;
% subplot(223)
% abs_var = abs(F);
% showgrey(abs(F), 64, -Fabsmax, Fabsmax) % magnitude
% waitforbuttonpress;
% subplot(224)
% showgrey(angle(F), 64, -pi, pi) % phase
% A=[1 4 7 10; 2 5 8 11; 3 6 9 12];

%for point = [5 9; 9 5; 17 9; 17 121]
for point = [1 2 128 2 5 9 17 17 120; 1 1 128 2 9 5 9 17 120]
    fftwavetemplate(point(1), point(2))
    waitforbuttonpress;
end

% questions 7, 8, 9
F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F + 2 * G;

Fhat = fft2(F);
Ghat = fft2(G);
Hhat = fft2(H);
sz =128;
%Chat = conv2(Ghat, Fhat);

% subplot(111)
% showgrey(F)
% subplot(322)
% showgrey(log(1 + abs(Fhat)))
% waitforbuttonpress;
% 
% subplot(323)
% showgrey(G)
% subplot(324)
% showgrey(log(1+ abs(Ghat)))
% waitforbuttonpress;
% 
% subplot(325)
% showgrey(H)
% subplot(326)
% showgrey(log(1+ abs(Hhat)))
% waitforbuttonpress;
% 
% showgrey(log(1+ abs(fftshift(Hhat))))
% waitforbuttonpress;
% showgrey(log(1 + abs(Chat)))
% waitforbuttonpress;
% 
% phone = phonecalc256;
% subplot(231)
% showgrey(phone/40, 64, 0, 6)
% title(sprintf('original phone'))
% subplot(234)
% hist(phone/40)
% subplot(232)
% showgrey(log(1 + phone), 64, 0, 6)
% title(sprintf('phone with log and alpha=1'))
% subplot(235)
% hist(log(1 + phone))
% subplot(233)
% showgrey(log(100 + phone), 64, 0, 6)
% title(sprintf('phone with log and alpha=100'))
% subplot(236)
% hist(log(100 + phone))


% showgray(F.*G)
% waitforbuttonpress;
% showgray(conv2(F,G), 64,0,100)
% showgrey(log(1+ abs(fftshift(fft2(F.*G)))))

% waitforbuttonpress;
% showfs(fft2(F.*G))
% showgray(conv2(F, G))
% waitforbuttonpress;
% showfs(fft2(F)* fft2(G))
% waitforbuttonpress;
% showgray(abs(fftshift(fft2(F)* fft2(G))))


% showgrey(F.*G)
% subplot(131)
% showfs(fft2(F.*G))
% subplot(132)
% % ?
% subplot(133)
% c = conv2(Fhat, Ghat)/sz.^2;  % normalize by multiplying 1/sz.^2
% showfs(c(1:150, 1:150));
% showgray(fftshift(log(1+ abs(c(1:128, 1:128)))));


% scaling
% F2 = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* ...
% [zeros(128, 48) ones(128, 32) zeros(128, 48)];
% 
% subplot(231)
% showgrey(F)
% subplot(232)
% showgrey(F2)
% subplot(234)
% showfs(fft2(F))
% subplot(235)
% showfs(fft2(F2))
% subplot(233)
% showgrey(F.*G)
% subplot(236)
% showfs(fft2(F.*G))
% waitforbuttonpress;
% 
% 
% 
% % rotation
% count = 1;
% for alpha = [0,60,80,90]
%     G = rot(F, alpha );
%     subplot(420+count)
%     showgrey(G)
%     axis on
%     Ghat = fft2(G);
%     Hhat = rot(fftshift(Ghat), -alpha );
%     subplot(420+count+1)
%     showgrey(log(1 + abs(Hhat)))
%     waitforbuttonpress;
%     count = count + 2;
% end
% 
% % question 13 
% a = 10^-10;
% i = 0;
% titles = {'phonelcalc128' 'few128' 'nallo128'};
% images = {phonecalc128 few128 nallo128};
% for m = 1:length(images)
%     i = i+1;
%     subplot(length(images),3,i)
%     showgrey(images{m})
%     title(titles{m})
%     
%     i = i+1;
%     subplot(length(images),3,i)
%     showgrey(pow2image(images{m},a))
%     title('pow2image()')
%     
%     i = i+1;
%     subplot(length(images),3,i)
%     showgrey(randphaseimage(images{m}))
%     title('randphaseimage()')
% end



% pic = phonecalc256;
% filtered_pic = gaussfft(pic, 100);
% subplot(131);
% showgrey(abs(filtered_pic))
% subplot(132);
% showgrey(pic)
% subplot(133);
% showgrey(discgaussfft(pic, 10))
% waitforbuttonpress;
% 
% for t = [0.1 0.3 1.0 10.0 100.0]
%     psf = gaussfft(deltafcn(128,128),t);
%     subplot(121);
%     plot(psf);
%     subplot(122);
%     showgray(psf)
%     waitforbuttonpress;
%     t
%     variance(psf) 
%     sprintf('----------')
% end
% 
% % task 16
% phone = phonecalc256;
% office = office256;
% suburb = suburb256;
% 
% for t = [1.0 4.0 16.0 64.0 256]
%     subplot(131);
%     showgray(gaussfft(phone, t))
%     subplot(132);
%     showgray(gaussfft(office, t))
%     subplot(133);
%     showgray(gaussfft(suburb, t))
%     waitforbuttonpress;
% end


% cnvoluion is bitwise multiplication
% Question 10 
% figure(1)
% subplot(121)
% showgrey(F.*G)
% title('F.*G')
% subplot(122)
% showfs(fft2(F.*G))
% title('fft2(F.*G)')
% 
% figure(2)
% subplot(121)
% c = conv2(Fhat, Ghat)/sz.^2;
% showfs(c(1:sz, 1:sz));
% title('conv2(Fhat, Ghat)/sz.^2')

% smoothning
office = office256;
% create noisy image
add = gaussnoise(office, 16); %
sap = sapnoise(office, 0.1, 255);% salt and pepper

% original images
subplot(231)
showgray(office)
subplot(232)
showgray(add)
subplot(233)
showgray(sap)

% restored versions
subplot(235)
showgray(gaussfft(add, 2))
subplot(236)
showgray(medfilt(sap, 5))
waitforbuttonpress;

img = add;
% Ideal low-pass filter
i = 1;
ncol = 3;
nrow = 2;
subplot(nrow,ncol,i)
showgrey(img)
for cutoffFreq = [1 2 3 4 5]
    i = i+1;
    subplot(nrow,ncol,i)
    showgrey(gaussfft(img,cutoffFreq))
    title(sprintf('%.2f',cutoffFreq))
end

% smoothning and subsampling
img = phonecalc256;
smoothimggauss = img;
smoothimgideal = img;
N=5;
for i=1:N
    if i>1 % generate subsampled versions
        img = rawsubsample(img);
        smoothimggauss = gaussfft(smoothimggauss, 0.9);% <call_your_filter_here>(smoothimg, <params>);
        smoothimgideal = ideal(smoothimgideal, 0.3);% <call_your_filter_here>(smoothimg, <params>);
        smoothimggauss = rawsubsample(smoothimggauss);
        smoothimgideal = rawsubsample(smoothimgideal);
    end
    subplot(3, N, i)
    showgrey(img)
    subplot(3, N, i+N)
    showgrey(smoothimggauss)
    subplot(3, N, i+N+N)
    showgrey(smoothimgideal)
end

function ans = gauss2(x,y,t)
    ans = (1/(2*pi*t))*exp(-(x.^2 + y.^2)/2*t);
end

function filtered = gaussfft(pic, t)
    [numRows,numCols] = size(pic);
    [X,Y] = meshgrid(-numCols/2:(numCols/2)-1);    
    G = (1/(2*pi*t))*exp(-(X.^2 + Y.^2)/(2*t));
    filtered = fftshift((ifft2(fft2(pic).*fft2(G))));
end






