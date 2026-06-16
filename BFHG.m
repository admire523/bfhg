function response = run_BFHG(img, params)

    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img_double = double(img);
    
    img_min = min(img_double(:));
    img_max = max(img_double(:));
    img_n = (img_double - img_min) / (img_max - img_min + eps);
    [R, C] = size(img_n);


    scale_configs =[
        3,  7,  3;   
        5,  9,  4;   
        7, 11,  5   
    ];
    num_scales = size(scale_configs, 1);

    if nargin < 2
        alpha_p = 5;           
        beta_n  = 8;           
        sigma_h = 0.25;         
        gamma_f = 10.0;          
    else
        alpha_p = params.alpha_p;
        beta_n  = params.beta_n;
        sigma_h = params.sigma_h;
        gamma_f = params.gamma_f;
    end

    MS_Fusion_Maps = zeros(R, C, num_scales);

 
    
    for s = 1:num_scales
        in_w   = scale_configs(s, 1);
        out_w  = scale_configs(s, 2);
        step_d = scale_configs(s, 3);
        
        mask_out = ones(out_w, out_w);
        mask_in  = zeros(out_w, out_w);
        c_center = ceil(out_w/2); half_in = floor(in_w/2);
        mask_in(c_center-half_in : c_center+half_in, c_center-half_in : c_center+half_in) = 1;
        
        mask_ring = mask_out - mask_in;
        mask_ring = mask_ring / sum(mask_ring(:)); 
        

        I_c = ordfilt2(img_n, in_w^2, ones(in_w, in_w), 'symmetric');
        I_bg = imfilter(img_n, mask_ring, 'replicate');
        
        diff_img = I_c - I_bg;
        diff_img(diff_img < 0) = 0; 
        mu_p = 1 - exp(-alpha_p * diff_img);
        
        local_std = stdfilt(img_n, ones(in_w, in_w));
        
  
        mu_n = exp(-beta_n * local_std) - 1; 


        shifts =[-step_d, -step_d; -step_d, 0; -step_d, step_d; ...
                    0, -step_d;               0, step_d; ...
                    step_d, -step_d;  step_d, 0;  step_d, step_d];
               
        pad_mu_p = padarray(mu_p,[step_d, step_d], 'replicate');
        pad_mu_n = padarray(mu_n,[step_d, step_d], 'replicate');
        
        mu_p_shifts = cell(1, 8); 
        mu_n_shifts = cell(1, 8);
        for i = 1:8
            dy = shifts(i, 1); dx = shifts(i, 2);
            mu_p_shifts{i} = pad_mu_p(1+dy+step_d : R+dy+step_d, 1+dx+step_d : C+dx+step_d);
            mu_n_shifts{i} = pad_mu_n(1+dy+step_d : R+dy+step_d, 1+dx+step_d : C+dx+step_d);
        end
        
        sector_indices = {[1,8], [2,7], [3,6], [4,5]}; 
        Hyperedge_Weights = zeros(R, C, 4);
        
        for k = 1:4
            n1 = sector_indices{k}(1); 
            n2 = sector_indices{k}(2);
          
            d_sq_1 = 0.5 * (mu_p - mu_p_shifts{n1}).^2 + 0.5 * (mu_n - mu_n_shifts{n1}).^2;
            d_sq_2 = 0.5 * (mu_p - mu_p_shifts{n2}).^2 + 0.5 * (mu_n - mu_n_shifts{n2}).^2;
        
            max_dist_sq = max(d_sq_1, d_sq_2);
            Hyperedge_Weights(:,:,k) = exp(- max_dist_sq / (2 * sigma_h^2));
        end
      
        Soft_Degree = sum(Hyperedge_Weights, 3) / 4.0; 


        MS_Fusion_Maps(:,:,s) = mu_p .* ( (1 - Soft_Degree) .^ gamma_f );
    end

    final_fusion_map = max(MS_Fusion_Maps,[], 3);

    out = final_fusion_map .* img_double;
    out(out < 0) = 0; 
    
    response = out;

end
