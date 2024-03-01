function value = psnr(x, y)
%calculate PSNR(x,y)
mse = sum((x-y).^2) / (length(x));
value = 10 * log10(norm(y, 'inf')^2/mse);
end

