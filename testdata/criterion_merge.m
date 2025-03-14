% 3 channels  criterion
clc;clear;
list_psnr = zeros(10,31);
list_ssim= zeros(10,31);
for i=1:10
    load(strcat(int2str(i), '/psnr_fullcolor.mat') );
    psnr = (b+g+r)/3;
    list_psnr(i,:) = psnr(1,:);
    load(strcat(int2str(i), '/ssim_fullcolor.mat') );
    ssim = (b+g+r)/3;
    list_ssim(i,:) = ssim(1,:);

end

save('psnr.mat','list_psnr')
save('ssim.mat','list_ssim')
