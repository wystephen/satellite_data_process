%normalized
img = X;
flag_set  = Y;
Xl = zeros(size(img));
Yl = zeros(size(flag_set));

x_mean = mean(img);
x_std = std(img);

for i = 1:size(img,1)
    Xl(i,:) = (img(i,:)-x_mean)./x_std;
end

Yl(find(flag_set>1)) = 1;
Yl(find(flag_set<1)) = 0;
