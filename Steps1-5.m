%% ESE 2180 Project 2 — Steps 1–5

clear; clc; close all;

%% (1) Load first 5000 training images & labels 
num_train = 5000;
[imgs_tr, labels_tr] = readMNIST('train-images-idx3-ubyte', ...
                                 'train-labels-idx1-ubyte', ...
                                 num_train, 0);     % imgs_tr: 20x20xN

[h, w, Ntr] = size(imgs_tr);   

% Encode labels: +1 for digit 0, -1 otherwise
y_tr = 2*double(labels_tr == 0) - 1;   % Ntr x 1

%% (2) Feature selection: pixels nonzero in >= 600 images 
% Here we're flattening each image to a column, then count nonzeros per pixel over the dataset
Xflat_tr = reshape(imgs_tr, h*w, Ntr);              % (400 x Ntr)
pixel_hits = sum(Xflat_tr > 0, 2);                  % (400 x 1)
feat_mask = (pixel_hits >= 600);                    % logical (400 x 1)
M0 = nnz(feat_mask);
fprintf('Selected %d features (pixels).\n', M0);

%% (3) Build A, y; solve LS; plot theta on pixel grid 
% Designing matrix A: each row = one image, columns = selected pixels
A_tr = Xflat_tr(feat_mask, :).';                    % (Ntr x M0)

% Solve least squares robustly (QR/backslash). This is A^+ y.
theta = A_tr \ y_tr;                                % (M0 x 1)

% Map theta back to a 20x20 image for visualization
theta_img = zeros(h*w, 1);
theta_img(feat_mask) = theta;
theta_img = reshape(theta_img, h, w);

figure;
imagesc(theta_img); axis image off; colorbar;
title('\theta coefficients mapped to pixel locations');

%% (4) Evaluate on first 5000 test images 
num_test = 5000;
[imgs_te, labels_te] = readMNIST('t10k-images-idx3-ubyte', ...
                                 't10k-labels-idx1-ubyte', ...
                                  num_test, 0);

Nte = size(imgs_te, 3);
y_te = 2*double(labels_te == 0) - 1;                % +1 for 0, -1 otherwise

Xflat_te = reshape(imgs_te, h*w, Nte);              % (400 x Nte)
A_te = Xflat_te(feat_mask, :).';                    % (Nte x M0)

scores = A_te * theta;                              % (Nte x 1)
y_hat = sign(scores);                               
y_hat(y_hat == 0) = -1;                             % tie → -1

% Metrics
err_rate = mean(y_hat ~= y_te);

neg_idx = (y_te == -1); pos_idx = (y_te == 1);
FPR = mean(y_hat(neg_idx) == 1);                    % predicted 1 but true -1
FNR = mean(y_hat(pos_idx) == -1);                   % predicted -1 but true 1

fprintf('Test Error Rate: %.4f\n', err_rate);
fprintf('False Positive Rate: %.4f\n', FPR);
fprintf('False Negative Rate: %.4f\n', FNR);

%% (5) Repeat with only the first 100 training images (same features)
num_train_small = 100;
idx_small = 1:num_train_small;

% Building A_small and y_small using the SAME feat_mask from Step 2
A_tr_small = Xflat_tr(feat_mask, idx_small).';      % (num_train_small x M0)
y_tr_small = y_tr(idx_small);                        % (num_train_small x 1)

% Solving least squares again (backslash/QR)
theta_small = A_tr_small \ y_tr_small;               % (M0 x 1)

% Mapping theta_small back to 20x20 for visualization
theta_img_small = zeros(h*w, 1);
theta_img_small(feat_mask) = theta_small;
theta_img_small = reshape(theta_img_small, h, w);

figure;
imagesc(theta_img_small); axis image off; colorbar;
title('\theta (trained on first 100 images) mapped to pixel locations');

% Evaluating on the same test set (first 5000 test images)
scores_small = A_te * theta_small;                   % (Nte x 1)
y_hat_small = sign(scores_small);
y_hat_small(y_hat_small == 0) = -1;                  % tie → -1

% Metrics for small-training run
err_rate_small = mean(y_hat_small ~= y_te);
FPR_small = mean(y_hat_small(neg_idx) == 1);         % predicted 1 but true -1
FNR_small = mean(y_hat_small(pos_idx) == -1);        % predicted -1 but true 1

fprintf('\n=== Step 5: Train on only 100 images (same features) ===\n');
fprintf('Test Error Rate (100 train): %.4f\n', err_rate_small);
fprintf('False Positive Rate (100 train): %.4f\n', FPR_small);
fprintf('False Negative Rate (100 train): %.4f\n', FNR_small);


