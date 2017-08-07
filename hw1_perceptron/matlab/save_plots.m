filename = 'cs545_hw1_rev2_eta0p1.mat';
load(filename);
plot(x_a, trn_a, 'LineWidth',2)
hold on
grid on 
plot(x_a, test_a, 'LineWidth',2)
axis([0 50 60 100])
title('Perceptron Learning Algorithm: eta = 0.1')

%Parameters for saved images
ImageDPI=500;
ImageSizeX=6;
ImageSizeY=4;
ImageFontSize=9;
FileLabel='WithFormatting___3';
FontName='Garamond';
AxisFontName='CMU Serif';

set(gca,'FontName',AxisFontName,'FontSize',ImageFontSize)
%Legend entries
%IMPORTANT NOTE
%Matlab has problems drawing the box around the legend if the font has been
%changed. The only way that I have found to get around this is to force
%blank spaces at the end of the longest legend entry. In this case
%the 'Another Signal{ }' has three blank spaces after it.
m2=legend('Training Data', 'Test Data','NorthWest');

xlabel('Epochs','fontsize',ImageFontSize)
ylabel('Percent Accuracy','fontsize',ImageFontSize)

%====================
%Set the image size and save to FileLabel.png where FileLabel is set at line 9.
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 ImageSizeX ImageSizeY])
print('-dpng', strcat(FileLabel, '.png') , strcat('-r',num2str(ImageDPI)))