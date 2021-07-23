#include "polyline_gpu.cuh"

__global__ void cuda_gradient_operator_kernel(unsigned char * const input_img, 
                                              unsigned char * const output_img,
                                              unsigned int nrows,
                                              unsigned int ncols,
                                              int proximity_term,
                                              GradientDirection mode){
    
    // Sample pixel based on thread dimensions
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;  
    const unsigned int pixel_loc = j * ncols + i;
    
    // Declare value of the gradient calculation 
    float dx = 0;
    float dy = 0;
    int gradient;
    
    // Limit derivative to non edges to perform convolution
    if(i > 0 && j > 0 && i < ncols-1 && j < nrows-1) {
        if(mode == HORIZONTAL || mode == COMPOSITE){
            dx = input_img[(j-1)*ncols + (i-1)] + proximity_term*input_img[j*ncols+(i-1)] + input_img[(j+1)*ncols+(i-1)] -
                 input_img[(j-1)*ncols + (i+1)] - proximity_term*input_img[j*ncols+(i+1)] - input_img[(j+1)*ncols+(i+1)];
        }
        
        if(mode == VERTICAL || mode == COMPOSITE){
            dy = input_img[(j-1)*ncols + (i-1)] + proximity_term*input_img[(j-1)*ncols+i] + input_img[(j-1)*ncols+(i+1)] -
                 input_img[(j+1)*ncols + (i-1)] - proximity_term*input_img[(j+1)*ncols+i] - input_img[(j+1)*ncols+(i+1)];
        }
        
        gradient = sqrt(dx*dx+dy*dy);

        if(gradient > 255){
            gradient = 255;
        }
        output_img[pixel_loc] = gradient;
    }
}

__global__ void cuda_hist_kernel(unsigned char* const src_img, 
                                 unsigned int* const hist,
                                 unsigned int nrows,
                                 unsigned int ncols){

    // Declare shared memory
    // A histogram will only have 256 slots 
    // since that's the range of pixel values for
    // a greyscale image
    __shared__ unsigned int hist_shared[256];

    // Sample pixel based on thread dimensions
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;  

    // Calculate index that this thread writes to the shared memory histogram
    const unsigned char hist_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread needs to clear the 
    // shared memory before we start the calculation
    hist_shared[hist_idx%256] = 0; 

    // Sync threads after shared memory has been properly cleaned
    __syncthreads();

    // If valid location, add this pixel to 
    // the shared histogram
    if(i < ncols && j < nrows){
        const unsigned long pixel_loc = ncols * j + i;
        const unsigned char value = src_img[pixel_loc];
        atomicAdd(&(hist_shared[value]), 1);
    }

    __syncthreads();

    // 32 by 32 means that there are 32 threads by 32 threads
    // The first 256 threads are defined by the range
    // 0 to 7*32+31 = 255. Since we only want to update each index
    // of the global histogram once per block, we should 
    // only use up to the eighth thread block in the y dimension
    if(threadIdx.y < 256/blockDim.y){
        atomicAdd(&(hist[hist_idx]), hist_shared[hist_idx]);
    }

}

 __global__ void cuda_WGE_kernel(unsigned char *const grad_img,
                                 unsigned char* output_img,
                                 unsigned int nrows,
                                 unsigned int ncols,
                                 unsigned int threshold){

    // Sample pixel based on thread dimensions
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;  
    const unsigned int pixel_loc = j * ncols + i;

    // Limit derivative to non edges to perform convolution
    if(i <= ncols-1 && j <= nrows-1) {
        if(grad_img[pixel_loc] < threshold){
            output_img[pixel_loc] = 0;
        }
    }
}

__device__
float min_eigenvalue(float a, float b, float c, float d)
{
	float ev_1 = (a + d)/2 + pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	float ev_2 = (a + d)/2 - pow(((a + d) * (a + d))/4 - (a * d - b * c), 0.5);
	if (ev_1 >= ev_2){
		return ev_2;
	}
	else{
		return ev_1;
	}
}

__global__ void cuda_shi_tomasi_kernel(unsigned char* const hgrad,
                                        unsigned char* const vgrad,
                                        eigenvalue_data_t* eigenvalues,
                                        unsigned int nrows,
                                        unsigned int ncols){

    // Sample pixel based on thread dimensions
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;  
    const unsigned int pixel_loc = j * ncols + i;
    unsigned int loc;

    // Limit Shi Tomasi corners to non edges
    if(i > 0 && j > 0 && i < ncols-1 && j < nrows-1) {
        float Sxx = 0, Syy = 0, Sxy = 0;
        float h_val, v_val;
        for (int di = -1; di < 2; di++) {
            for (int dj = -1; dj < 2; dj++) {
                loc = (j + dj) * ncols + (i + di);
                
                h_val = hgrad[loc];
                v_val = vgrad[loc];
                
                Sxx += h_val*h_val;
                Syy += v_val*v_val;
                Sxy += h_val*v_val;
            }
        }
        eigenvalues[pixel_loc].i = i;
        eigenvalues[pixel_loc].j = j;
        eigenvalues[pixel_loc].ev = min_eigenvalue(Sxx, Sxy, Sxy, Syy);
    }
}

cv::Mat cuda_call_gradient_kernel(cv::Mat src_img,
                                  int proximity_term,
                                  GradientDirection mode,
                                  unsigned int thread_dim){

    const unsigned int nrows = src_img.rows;
    const unsigned int ncols = src_img.cols;

    // Declare threads per block
    // Must be a square for shared memory indexing
    dim3 threads_per_block(thread_dim, thread_dim);
    // Break up image into squares. Each block should handle
    // a single square of the image.
    dim3 blocks_per_grid((ncols + threads_per_block.x - 1)/threads_per_block.x, 
                        (nrows + threads_per_block.y - 1)/threads_per_block.y);

    // Declare gpu data for src and dst image
    unsigned char *gpu_src_img;
    unsigned char *gpu_dst_img;
    
    // Declare new Mat object to hold destination image
    cv::Mat dst_img = src_img.clone();

    // Allocate memory for src and dst image
    cudaMalloc((void**)&gpu_src_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc((void**)&gpu_dst_img, ncols*nrows*sizeof(unsigned char));

    // Copy grey image to gpu
    cudaMemcpy(gpu_src_img, src_img.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set dst values to zero
    cudaMemset(gpu_dst_img, 0, nrows*ncols*sizeof(unsigned char));
    
    // Run GPU histogram
    cuda_gradient_operator_kernel<<<blocks_per_grid, threads_per_block>>>(gpu_src_img, 
                                                                          gpu_dst_img, 
                                                                          nrows, 
                                                                          ncols,
                                                                          proximity_term,
                                                                          mode);

    // Copy values from the gpu histogram to the host histogram
    cudaMemcpy(dst_img.data, gpu_dst_img, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    return dst_img;

}

unsigned int* cuda_call_hist_kernel(cv::Mat src_img, 
                                    unsigned int thread_dim){

    // Get dimensions
    const uint nrows = src_img.rows;
    const uint ncols = src_img.cols;

    // Declare threads per block
    // Must be a square for shared memory indexing
    dim3 threads_per_block(thread_dim, thread_dim);
    // Break up image into squares. Each block should handle
    // a single square of the image.
    dim3 blocks_per_grid((ncols + threads_per_block.x - 1)/threads_per_block.x, 
                        (nrows + threads_per_block.y - 1)/threads_per_block.y);
    
    // Declare gpu data for image and histogram
    unsigned char *gpu_src_img;
    unsigned int *gpu_hist;

    // Create new histogram to write to from gpu
    unsigned int* hist = (unsigned int*) malloc(256*sizeof(unsigned int));
    
    // Allocate memory space for the src image and histogram
    cudaMalloc((void**)&gpu_src_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc((void**)&gpu_hist, 256*(sizeof(unsigned int)));
    
    // Copy grey image to gpu
    cudaMemcpy(gpu_src_img, src_img.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set histogram values to zero
    cudaMemset(gpu_hist, 0, 256*sizeof(unsigned int));

    // Run GPU histogram
    cuda_hist_kernel<<<blocks_per_grid, threads_per_block>>>(gpu_src_img, 
                                                             gpu_hist, 
                                                             nrows, 
                                                             ncols);

    // Copy values from the gpu histogram to the host histogram
    cudaMemcpy(hist, gpu_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free CUDA malloced memory 
    cudaFree(gpu_src_img);
    cudaFree(gpu_hist);

    return hist;
}


cv::Mat cuda_call_WGE_kernel(cv::Mat src_img,
    int proximity_term,
    float percentile,
    unsigned int thread_dim){

    const unsigned int nrows = src_img.rows;
    const unsigned int ncols = src_img.cols;

    // Declare threads per block
    // Must be a square for shared memory indexing
    dim3 threads_per_block(thread_dim, thread_dim);
    // Break up image into squares. Each block should handle
    // a single square of the image.
    dim3 blocks_per_grid((ncols + threads_per_block.x - 1)/threads_per_block.x, 
    (nrows + threads_per_block.y - 1)/threads_per_block.y);

    // Calculate gradient image
    cv::Mat grad_img = cuda_call_gradient_kernel(src_img,
                                                proximity_term,
                                                COMPOSITE,
                                                thread_dim);

    // Calculate histogram
    unsigned int* hist;
    hist = cuda_call_hist_kernel(grad_img, thread_dim);
    
    // Calculate threshold from histogram
    int threshold = 0;
    int pixel_count = 0;
    int pixel_threshold = percentile*nrows*ncols;

    while(pixel_count < pixel_threshold){
        pixel_count += hist[threshold];
        threshold += 1;
    }

    // Declare gpu data for src and dst image
    unsigned char *gpu_grad_img;
    unsigned char *gpu_dst_img;

    // Declare new Mat object to hold destination image
    cv::Mat dst_img = src_img.clone();

    // Allocate memory for src and dst image
    cudaMalloc((void**)&gpu_grad_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc((void**)&gpu_dst_img, ncols*nrows*sizeof(unsigned char));

    // Copy grey image to gpu
    cudaMemcpy(gpu_grad_img, grad_img.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dst_img, dst_img.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Call kernel
    cuda_WGE_kernel<<<blocks_per_grid, threads_per_block>>>(gpu_grad_img, 
                                    gpu_dst_img, 
                                    nrows, 
                                    ncols,
                                    threshold);

    // Copy values from the gpu histogram to the host histogram
    cudaMemcpy(dst_img.data, gpu_dst_img, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    return dst_img;
}

struct sort_by_eig_value_desc {
    __host__ __device__
    bool operator()(const eigenvalue_data_t& a, const eigenvalue_data_t& b)
    {
        return a.ev > b.ev;
    }
};

std::vector<cv::Point2f> cuda_call_ST_kernel(cv::Mat src_img,
                                             float eig_threshold,
                                             float dist_threshold,
                                             unsigned int max_features,
                                             int proximity_term,
                                             unsigned int thread_dim){

    const uint nrows = src_img.rows;
    const uint ncols = src_img.cols;
    
    // Declare threads per block
    // Must be a square for shared memory indexing
    dim3 threads_per_block(thread_dim, thread_dim);
    // Break up image into squares. Each block should handle
    // a single square of the image.
    dim3 blocks_per_grid((ncols + threads_per_block.x - 1)/threads_per_block.x, 
                        (nrows + threads_per_block.y - 1)/threads_per_block.y);
    
    // If the gradient image is NULL, then run gradient kernel
    // Calculate horizontal and vertical gradients
    cv::Mat hgrad = cuda_call_gradient_kernel(src_img,
                                              proximity_term,
                                              HORIZONTAL,
                                              thread_dim);
    
    
    cv::Mat vgrad = cuda_call_gradient_kernel(src_img,
                                              proximity_term,
                                              VERTICAL,
                                              thread_dim);

    // Declare new Mat object to hold destination image
    eigenvalue_data_t* eigenvalues = (eigenvalue_data_t*) malloc(ncols*nrows*sizeof(eigenvalue_data_t));

    // Declare gpu data for src and dst image
    unsigned char *gpu_hgrad_img;
    unsigned char *gpu_vgrad_img;
    eigenvalue_data_t *gpu_eigenvalues;

    // Allocate memory for src and dst image
    cudaMalloc((void**)&gpu_hgrad_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc((void**)&gpu_vgrad_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc((void**)&gpu_eigenvalues, ncols*nrows*sizeof(eigenvalue_data_t));

    // Copy grey image to gpu
    cudaMemcpy(gpu_hgrad_img, hgrad.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_vgrad_img, vgrad.data, nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set candidate pixel values to zero
    cudaMemset(gpu_eigenvalues, 0, nrows*ncols*sizeof(eigenvalue_data_t));
    
    // Run GPU histogram
    cuda_shi_tomasi_kernel<<<blocks_per_grid, threads_per_block>>>(gpu_hgrad_img, 
                                                                   gpu_vgrad_img,
                                                                   gpu_eigenvalues, 
                                                                   nrows, 
                                                                   ncols);

    // Copy values from the gpu histogram to the host histogram
    cudaMemcpy(eigenvalues, gpu_eigenvalues, nrows*ncols*sizeof(eigenvalue_data_t), cudaMemcpyDeviceToHost);
    
    // Free cuda malloced memory
    cudaFree(gpu_hgrad_img);
    cudaFree(gpu_vgrad_img);

    thrust::device_ptr<eigenvalue_data_t> thrust_eigenvalues(gpu_eigenvalues);
	thrust::sort(thrust_eigenvalues, thrust_eigenvalues + nrows*ncols, sort_by_eig_value_desc());
    gpu_eigenvalues = thrust::raw_pointer_cast(thrust_eigenvalues);

    // Host buffer for the eigenvalues.
    eigenvalue_data_t *sorted_eigenvalues = (eigenvalue_data_t*)malloc(sizeof(eigenvalue_data_t)*nrows*ncols);

    // Copy the sorted eigenvalues back to host memory so we can proceed.

    cudaMemcpy(sorted_eigenvalues, gpu_eigenvalues, nrows*ncols, cudaMemcpyDeviceToHost);


    std::vector<cv::Point2f> ST_points;
    unsigned int candidate_ptr = 0;
    float dist;
    unsigned int valid;

    while(candidate_ptr < nrows*ncols && ST_points.size() <= max_features){
        eigenvalue_data_t candidate_pixel = sorted_eigenvalues[candidate_ptr];
        
        valid = 1;
        for (auto corner : ST_points) {
            dist = (corner.x-candidate_pixel.i)*(corner.x-candidate_pixel.i) + 
                   (corner.y-candidate_pixel.j)*(corner.y-candidate_pixel.j);

            if(dist < pow(dist_threshold, 2)){
                valid = 0;
                break;
            }
        }

        if(valid){
            cv::Point2f new_feature = cv::Point2f(candidate_pixel.i, candidate_pixel.j);
            ST_points.push_back(new_feature);
        }
        candidate_ptr += 1;
    }
    
    return ST_points;
}