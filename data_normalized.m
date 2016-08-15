%normalized

X = zeros(size(img));
Y = zeros(size(flag_set));

x_mean = mean(img);
x_std = std(img);

for i = 1:size(img,1)
    X(i,:) = (img(i,:)-x_mean)./x_std;
end

Y(find(flag_set>1)) = 1;
