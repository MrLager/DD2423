


x = 1:30;
 
p1 = subplot(131)
plot(x,costs(x,1,1))
p2 = subplot(132)
plot(x,costs(x,100,1))
p3 = subplot(133)
plot(x,costs(x,1,100))



function ret = costs(diff, alpha, sigma)
    ret = (alpha*sigma)./(diff + sigma);
end 