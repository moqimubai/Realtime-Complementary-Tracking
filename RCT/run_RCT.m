function results = run_RCT(seq, res_path, bSaveImage)
% Entry point for the Wu - CVPR2013 benchmark

    close all;
    s_frames = seq.s_frames;

    target_sz = floor([seq.init_rect(1,4), seq.init_rect(1,3)]);
    pos = floor([seq.init_rect(1,2), seq.init_rect(1,1)]+target_sz/2);

    addpath(genpath('/matconvnet-1.0-beta20'));
    addpath('2016-08-17.net.mat');
    vl_setupnn;
    
    thresh_lambda = 0.95;
    
    grayscale_sequence = false;
    hog_cell_size = 4;
    fixed_area = 150^2;           % standard area to which we resize the target
    n_bins = 2^5;                            % number of bins for the color histograms (bg and fg models)
    learning_rate_pwp = 0.04;           % bg and fg color models learning rate 
    feature_type = 'fhog';
    %         params.feature_type = 'gray';
    inner_padding = 0.2;             % defines inner area used to sample colors from the foreground
    output_sigma_factor = 1/16 ;             % standard deviation for the desired translation filter output
    lambda = 1e-3;                                   % regularization weight
    learning_rate_cf = 0.01;            % HOG model learning rate
    merge_factor = 0.3;              % fixed interpolation factor - how to linearly combine the two responses
    merge_method = 'const_factor';
    den_per_channel = false;

    %% scale related
    hog_scale_cell_size = 4;                % Default DSST=4
    learning_rate_scale = 0.025;
    scale_sigma_factor = 1/4;
    num_scales = 33;
    scale_model_factor = 1.0;
    scale_step = 1.02;
    scale_model_max_area = 32*16;
    
    p.scaleStep = 1.0255;
    p.scalePenalty = 0.962;
    p.scaleLR = 0.34; % damping factor for scale update
    p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
    p.windowing = 'cosine'; % to penalize large displacements
    p.wInfluence = 0.168; % windowing influence (in convex sum)
    p.net = '2016-08-17.net.mat';
    p.bbox_output = false;
    p.fout = -1;
        %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
        %% SiamFC prefix and ids
    p.prefix_z = 'a_'; % used to identify the layers of the exemplar
    p.prefix_x = 'b_'; % used to identify the layers of the instance
    p.prefix_join = 'xcorr';
    p.prefix_adj = 'adjust';
    p.id_feat_z = 'a_feat';
    p.id_score = 'score';
    
    p.net_base_path = 'path/to/nets/';
    p.stats_path = 'path\to/ILSVRC15-VID/stats.mat'; % (optional)
    
    if exist('stats.mat','file')
        stats = load('stats.mat');
    else
        warning('No stats found at %s', 'stats.mat');
        stats = [];
    end
     net_z = load_pretrained('2016-08-17.net.mat', 1);
     net_x = load_pretrained('2016-08-17.net.mat', []);%这里为什么不是1
     
      % exemplar branch (used only once per video) computes features for the target
     remove_layers_from_prefix(net_z, p.prefix_x);
     remove_layers_from_prefix(net_z, p.prefix_join);
     remove_layers_from_prefix(net_z, p.prefix_adj);
     % instance branch computes features for search region x and cross-correlates with z features
     remove_layers_from_prefix(net_x, p.prefix_z);
     zFeatId = net_z.getVarIndex(p.id_feat_z);
     scoreId = net_x.getVarIndex(p.id_score);
     
     mode = 1;%0 for Staple and 1 for Siamese
     time = 0;  

     rect_position = zeros(numel(s_frames), 4);
    for frame = 1:numel(s_frames),
        im = imread(s_frames{frame});
        if(size(im,3)==1)
           grayscale_sequence = true;
        end
        
        if frame == 1
            avg_dim = sum(target_sz)/2;
            bg_area = round(target_sz + avg_dim);
            fg_area = round(target_sz - avg_dim * inner_padding);
            if(bg_area(2)>size(im,2)), bg_area(2)=size(im,2)-1; end
            if(bg_area(1)>size(im,1)), bg_area(1)=size(im,1)-1; end
            bg_area = bg_area - mod(bg_area - target_sz, 2);
            fg_area = fg_area + mod(bg_area - fg_area, 2);
            area_resize_factor = sqrt(fixed_area/prod(bg_area));
            norm_bg_area = round(bg_area * area_resize_factor);
            cf_response_size = floor(norm_bg_area / hog_cell_size);
            norm_target_sz_w = 0.75*norm_bg_area(2) - 0.25*norm_bg_area(1);
            norm_target_sz_h = 0.75*norm_bg_area(1) - 0.25*norm_bg_area(2);
            norm_target_sz = round([norm_target_sz_h norm_target_sz_w]);
            norm_pad = floor((norm_bg_area - norm_target_sz) / 2);
            radius = min(norm_pad);
            norm_delta_area = (2*radius+1) * [1, 1];
            norm_pwp_search_area = norm_target_sz + norm_delta_area - 1;
            
            
            patch_padded = getSubwindow(im, pos, norm_bg_area, bg_area);
            new_pwp_model = true;
            [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, norm_bg_area, n_bins, grayscale_sequence);
            new_pwp_model = false;
            if isToolboxAvailable('Signal Processing Toolbox')
               hann_window = single(hann(cf_response_size(1)) * hann(cf_response_size(2))');
            else
               hann_window = single(myHann(cf_response_size(1)) * myHann(cf_response_size(2))');
            end
            output_sigma = sqrt(prod(norm_target_sz)) * output_sigma_factor / hog_cell_size;
            y = gaussianResponse(cf_response_size, output_sigma);
            yf = fft2(y);


            scale_factor = 1;
            base_target_sz = target_sz;
            scale_sigma = sqrt(num_scales) * scale_sigma_factor;
            ss = (1:num_scales) - ceil(num_scales/2);
            ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
            ysf = single(fft(ys));
            if mod(num_scales,2) == 0
                scale_window = single(hann(num_scales+1));
                scale_window = scale_window(2:end);
            else
                scale_window = single(hann(num_scales));
            end;

            ss = 1:num_scales;
            scale_factors = scale_step.^(ceil(num_scales/2) - ss);
            if scale_model_factor^2 * prod(norm_target_sz) > scale_model_max_area
                scale_model_factor = sqrt(scale_model_max_area/prod(norm_target_sz));
            end
            scale_model_sz = floor(norm_target_sz * scale_model_factor);
            min_scale_factor = scale_step ^ ceil(log(max(5 ./ bg_area)) / log(scale_step));
            max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(scale_step));
            
            im_patch_bg = getSubwindow(im, pos, norm_bg_area, bg_area);
            xt = getFeatureMap(im_patch_bg, feature_type, cf_response_size, hog_cell_size);
            xt = bsxfun(@times, hann_window, xt);
            xtf = fft2(xt);
            hf_num = bsxfun(@times, conj(yf), xtf) / prod(cf_response_size);
            hf_den = (conj(xtf) .* xtf) / prod(cf_response_size);    

            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            sf_num = bsxfun(@times, ysf, conj(xsf));
            sf_den = sum(xsf .* conj(xsf), 1);
            
            img = gpuArray(single(im));
            if(size(img, 3)==1)
               img = repmat(img, [1 1 3]);
            end
            avgChans = gather([mean(mean(img(:,:,1))) mean(mean(img(:,:,2))) mean(mean(img(:,:,3)))]);
            wc_z = target_sz(2) + p.contextAmount*sum(target_sz);
            hc_z = target_sz(1) + p.contextAmount*sum(target_sz);
            s_z = sqrt(wc_z*hc_z);
            scale_z = p.exemplarSize / s_z;
        
            [z_crop, ~] = get_subwindow_tracking(img, pos, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
            if p.subMean
                z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
            end
            d_search = (p.instanceSize - p.exemplarSize)/2;
            pad = d_search/scale_z;
            s_x = s_z + 2*pad;
            % arbitrary scale saturation

            switch p.windowing
                case 'cosine'
                    window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
                case 'uniform'
                    window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
            end
            % make the window sum 1
            window = window / sum(window(:));
            % evaluate the offline-trained network for exemplar z features
            net_z.eval({'exemplar', z_crop});
            z_features = net_z.vars(zFeatId).value;
            
           
            rect_position(frame,:) = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; 
            
        else
            tic;
            if mode == 1
                img = gpuArray(single(im));
                if(size(img, 3)==1)
                    img = repmat(img, [1 1 3]);
                end
                
                [x_crops, ~] = get_subwindow_tracking(img, pos, [p.instanceSize p.instanceSize], [round(s_x) round(s_x)], avgChans);
                net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});
                responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize]);
           %     responseMap = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp)));
                responseMap = imresize(responseMaps, p.responseUp, 'bicubic');
                responseMap = responseMap - min(responseMap(:));
                responseMap = responseMap / sum(responseMap(:));
                responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
                [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
                if isempty(r_max)
                     r_max = ceil(params.scoreSize/2);
                end
                if isempty(c_max)
                     c_max = ceil(params.scoreSize/2);
                end
                p_corr = [r_max, c_max];
                disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
                disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
                disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
                pos = pos + gather(disp_instanceFrame);
            end
                
            im_patch_cf = getSubwindow(im, pos, norm_bg_area, bg_area);
            pwp_search_area = round(norm_pwp_search_area / area_resize_factor);
            im_patch_pwp = getSubwindow(im, pos, norm_pwp_search_area, pwp_search_area);
            xt = getFeatureMap(im_patch_cf, feature_type, cf_response_size, hog_cell_size);
            xt_windowed = bsxfun(@times, hann_window, xt);
            xtf = fft2(xt_windowed);
            if den_per_channel
                hf = hf_num ./ (hf_den + lambda);
            else
                hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+lambda);
            end
            response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));
            response_cf = cropFilterResponse(response_cf,floor_odd(norm_delta_area / hog_cell_size));
            if hog_cell_size > 1
                 response_cf = mexResize(response_cf, norm_delta_area,'auto');
            end
            [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, n_bins, grayscale_sequence);
            likelihood_map(isnan(likelihood_map)) = 0;
            response_pwp = getCenterLikelihood(likelihood_map, norm_target_sz);
            response = mergeResponses(response_cf, response_pwp, merge_factor, merge_method);
            figure(2);imagesc(response);
            [row,col] = find(response==max(response(:)),1);
            center = (1+norm_delta_area) / 2;
            pos = pos + ([row, col] - center) / area_resize_factor;
            
            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda) ));
            recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
            scale_factor = scale_factor * scale_factors(recovered_scale);
            if scale_factor < min_scale_factor
                scale_factor = min_scale_factor;
            elseif scale_factor > max_scale_factor
                scale_factor = max_scale_factor;
            end
            s_x = s_x * scale_factors(recovered_scale);
            target_sz = round(base_target_sz * scale_factor);
            rect_position(frame,:) = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            
            avg_dim = sum(target_sz)/2;
            bg_area = round(target_sz + avg_dim);
            if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
            if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end
            bg_area = bg_area - mod(bg_area - target_sz, 2);
            fg_area = round(target_sz - avg_dim * inner_padding);
            fg_area = fg_area + mod(bg_area - fg_area, 2);
            area_resize_factor = sqrt(fixed_area/prod(bg_area));

            im_patch_bg = getSubwindow(im, pos, norm_bg_area, bg_area);
            xt = getFeatureMap(im_patch_bg, feature_type, cf_response_size, hog_cell_size);
            xt = bsxfun(@times, hann_window, xt);
            xtf = fft2(xt);
            new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(cf_response_size);
            new_hf_den = (conj(xtf) .* xtf) / prod(cf_response_size);  
            hf_den = (1 - learning_rate_cf) * hf_den + learning_rate_cf * new_hf_den;
            hf_num = (1 - learning_rate_cf) * hf_num + learning_rate_cf * new_hf_num;
            [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, norm_bg_area, n_bins, grayscale_sequence, bg_hist, fg_hist, learning_rate_pwp);  

            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            new_sf_num = bsxfun(@times, ysf, conj(xsf));
            new_sf_den = sum(xsf .* conj(xsf), 1);
            sf_den = (1 - learning_rate_scale) * sf_den + learning_rate_scale * new_sf_den;
            sf_num = (1 - learning_rate_scale) * sf_num + learning_rate_scale * new_sf_num;
            
            measure = getMeasure(response,row,col);
            
            if frame == 2
                ada_measure = measure;
            else
                ada_measure = thresh_lambda*ada_measure +(1-thresh_lambda)*measure;
            end
            
            threshold = ada_measure *1.1;
            if (measure<threshold)
                mode = 0;
            else
                mode = 1;
            end
            time = time + toc();
        end
    
        if bSaveImage
            figure(1), imshow(uint8(im));
            if mode == 0
                figure(1), rectangle('Position', rect_position(frame,:), 'LineWidth', 4, 'EdgeColor', 'g');
            else
                figure(1), rectangle('Position', rect_position(frame,:), 'LineWidth', 4, 'EdgeColor', 'r');
            end
            drawnow
        end
                
             
%         if bSaveImage
%             if frame == 1,  %first frame, create GUI
%                 figure(1);
%     %             figure('Number','off', 'Name',['Tracker - ' video_path]);
%                 im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
%                 hold on;
%                 rect_handle = rectangle('Position',rect_position(frame,:), 'EdgeColor','g');
%                 text_handle = text(10, 10, int2str(frame));
%                 set(text_handle, 'color', [0 1 1]);
%             else
%                 try  %subsequent frames, update GUI
%                     if mode ==0
%                     set(im_handle, 'CData', im)
%                     set(rect_handle, 'Position', rect_position(frame,:))
%                     set(text_handle, 'string', int2str(frame));
%                     else
%                     set(im_handle, 'CData', im)
%                     rectangle('Position',rect_position(frame,:), 'EdgeColor','r');
%                     set(text_handle, 'string', int2str(frame));
%                     end
%                 catch
%                     return
%                 end
%                 hold off;
%             end
%             drawnow
%        end
    end
    
    fps = numel(s_frames)/time;
    disp(['fps: ' num2str(fps)])

    results.type = 'rect';
    results.res = rect_position;%each row is a rectangle
    results.fps = fps;
end

function H = myHann(X)
    H = .5*(1 - cos(2*pi*(0:X-1)'/(X-1)));
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end

    

