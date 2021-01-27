load('fits','v_theta');
isi = v_theta(:,1)*1.5;
nsuj = length(isi);
cutoff = 4; % seconds
I = isi<cutoff;
media = mean(isi(I));
stde = nanstd(isi(I))/sqrt(sum(I));