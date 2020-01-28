resimdir = fullfile('C:\Program Files\MATLAB\R2019a\bin\resimler');
resimset = imageDatastore(resimdir);
montage(resimset.Files)
resimler = readimage(resimset, 1);
griresim = rgb2gray(resimler);
noktalar = detectSURFFeatures(griresim);
[ozellikler, noktalar] = extractFeatures(griresim, noktalar);
numImages = numel(resimset.Files);
donustur(numImages) = projective2d(eye(3));
imageSize = zeros(numImages,2);
for n = 2:numImages
    oncekinoktalar = noktalar;
    oncekiozellikler = ozellikler;
    resimler = readimage(resimset, n);
    griresim = rgb2gray(resimler);
    imageSize(n,:) = size(griresim);
    noktalar = detectSURFFeatures(griresim);
    [ozellikler, noktalar] = extractFeatures(griresim, noktalar);
    indexPairs = matchFeatures(ozellikler, oncekiozellikler, 'Unique', true);
    eslesennoktalar = noktalar(indexPairs(:,1), :);
    oncekieslesennoktalar = oncekinoktalar(indexPairs(:,2), :);
    donustur(n) = estimateGeometricTransform(eslesennoktalar, oncekieslesennoktalar,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    donustur(n).T = donustur(n).T * donustur(n-1).T;
end

for i = 1:numel(donustur)
    [xlim(i,:), ylim(i,:)] = outputLimits(donustur(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end
ortalamaXLim = mean(xlim, 2);
[~, idx] = sort(ortalamaXLim);
merkezIdx = floor((numel(donustur)+1)/2);
Idx = idx(merkezIdx);
Tinv = invert(donustur(centerImageIdx));
for i = 1:numel(donustur)
    donustur(i).T = donustur(i).T * Tinv.T;
end
for i = 1:numel(donustur)
    [xlim(i,:), ylim(i,:)] = outputLimits(donustur(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end
maxImageSize = max(imageSize);
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
genislik  = round(xMax - xMin);
yukseklik = round(yMax - yMin);
panorama = zeros([yukseklik genislik 3], 'like', resimler);
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([yukseklik genislik], xLimits, yLimits);
for i = 1:numImages
     resimler = readimage(resimset, i);
     warplanmisresim = imwarp(resimler, donustur(i), 'OutputView', panoramaView);
     maske = imwarp(true(size(resimler,1),size(resimler,2)), donustur(i), 'OutputView', panoramaView);
     panorama = step(blender, panorama, warplanmisresim, maske);
end
figure
imshow(panorama)