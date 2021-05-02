N = 50; %5000
exy = eye_centres(N);
% (2,N)
size(exy)  % 56 x 2
options = {"Qbb"}
voronoi (exy(:,1), exy(:,2), options);

errrr

% K = 10;  % Try comparing K=1 (overfitting) vs. K=10 (more generalized)
% knn = cv.KNearest();
% knn.DefaultK = K;
% tic
% knn.train(data, labels);
% toc

x = exy(1, :);
y = exy(2, :);
% h = convhull (x, y);
[vx, vy] = voronoi (x, y);
vx


% https://octave.sourceforge.io/octave/function/voronoi.html
