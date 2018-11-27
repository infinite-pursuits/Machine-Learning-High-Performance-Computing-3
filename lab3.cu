#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#define bsunpadded 8
#define bs 8

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


double find_checksum(double * output)
{
    double checksum = 0.0;
    int K=10, H=4096, W=4096;
    for (int k=0; k<K; k++){
        for(int row=0; row<H; row++){
            for(int col=0; col<W; col++){
                checksum+=output[(k*H*W)+(row*W)+col];
            }
        }
    }
    return checksum;
}

__global__ void C1(int W, int H, int C, int FW, int FH, int K, double* input, double* filter, double* output) {

    int image_col = blockDim.x * blockIdx.x + threadIdx.x;
    int image_row = blockDim.y * blockIdx.y + threadIdx.y;
    int fil_num = blockIdx.z;

    if (image_row < H && image_col < W && fil_num < K) {

        int p1col = image_col - FW / 2;
        int p1row = image_row - FH / 2;
        double conv_val = 0.0, img_val = 0.0, fil_val = 0.0;
        int ncol = 0, nrow = 0;

        for (int c = 0; c < C; c++) {
            for (int row = 0; row < FH; row++) {
                for (int col = 0; col < FW; col++) {
                    img_val = 0.0;
                    ncol = p1col + col;
                    nrow = p1row + row;

                    fil_val = filter[(fil_num * C * FW * FH) + (c * FW * FH) + ((FH - row - 1) * FW) + (FW - 1 - col)];
                    if (0 <= ncol && ncol < W)
                        if (0 <= nrow && nrow < H)
                            img_val = input[(c * H * W) + (nrow) * W + ncol];

                    conv_val = conv_val + (img_val * fil_val);
                }
            }
        }
        output[(fil_num * H * W) + (image_row * W) + image_col] = conv_val;
    }
}

__global__ void C2(int W, int H, int C, int FW, int FH, int K, double* input, double* filter, double* output){

    int image_col = threadIdx.x + blockDim.x * blockIdx.x;
    int image_row = threadIdx.y + blockDim.y * blockIdx.y;
    int fil_num = threadIdx.z;

    __shared__ double tile[3][10][10];

    if(image_row<H && image_col<W && fil_num<K){

        int actual_tr = threadIdx.y, actual_tc = threadIdx.x;

        for(int c=0; c<C; c++){
            tile[c][actual_tr+1][actual_tc+1]=input[(c*W*H)+(image_row*W)+image_col];
        }

        if(actual_tr == bsunpadded - 1){
            for (int c = 0; c < C; c++){
                if (image_row < H - 1)
                    tile[c][actual_tr + 2][actual_tc + 1] = input[(c * H * W) + ((image_row + 1) * W) + image_col];
                else
                    tile[c][actual_tr + 2][actual_tc + 1] = 0.0;
            }


            if (actual_tc == bsunpadded - 1){
                for (int c = 0; c < C; c++) {
                    if (image_col < W - 1 && image_row < H - 1)
                        tile[c][actual_tr + 2][actual_tc + 2] = input[(c * H * W) + ((image_row + 1) * W) +
                                                                      image_col + 1];
                    else
                        tile[c][actual_tr + 2][actual_tc + 2] = 0.0;
                    }
                }
            }

        if(actual_tc == 0){
            for(int c=0;c<C;c++){
               if(image_col>=1)
                    tile[c][actual_tr+1][actual_tc] = input[(c*H*W)+(image_row*W)+image_col-1] ;
                else
                    tile[c][actual_tr+1][actual_tc] = 0.0 ;
            }

            if(actual_tr == bsunpadded - 1){
                    for(int c=0;c<C;c++) {
                        if (image_col >= 1 && image_row < H - 1)
                            tile[c][actual_tr + 2][actual_tc] = input[(c * H * W) + ((image_row + 1) * W) + image_col - 1];
                        else
                            tile[c][actual_tr + 2][actual_tc] = 0.0;
                    }
                }
            }


        if(actual_tr == 0){
            for(int c=0;c<C;c++){
                if(image_row>=1)
                    tile[c][actual_tr][actual_tc+1] = input[(c*H*W)+((image_row-1)*W)+image_col] ;
                else
                    tile[c][actual_tr][actual_tc+1] = 0.0;
            }

            if(actual_tc==0){
                for(int c = 0; c < C; c++){
                    if (image_col >= 1 && image_row >= 1)
                        tile[c][actual_tr][actual_tc] = input[(c * H * W) + ((image_row - 1) * W) + image_col - 1];
                    else
                        tile[c][actual_tr][actual_tc] = 0.0;

                }
            }
        }

        if(actual_tc == bsunpadded - 1){
            for(int c=0;c<C;c++) {
                if (image_col < W - 1)
                    tile[c][actual_tr + 1][actual_tc + 2] = input[(c * H * W) + (image_row * W) + image_col + 1];
                else
                    tile[c][actual_tr + 1][actual_tc + 2] = 0.0;
            }

            if(actual_tr==0){
                for (int c = 0; c < C; c++){
                    if (image_col < W - 1 && image_row >= 1)
                        tile[c][actual_tr][actual_tc + 2] = input[(c * H * W) + ((image_row - 1) * W) + image_col + 1];
                    else
                        tile[c][actual_tr][actual_tc + 2] = 0.0;

                }
            }
        }

        __syncthreads();

            int nrow=0, ncol=0;
           double conv_val = 0.0;
            for (int c = 0; c < C; c++) {
                for (int row = 0; row < FH; row++) {
                    for (int col = 0; col < FW; col++) {
                        double tile_val = 0.0;
                        ncol = actual_tc + col;
                        nrow = actual_tr + row;
                        double fil_val = filter[(fil_num * C * FW * FH) + (c*FW*FH)+ ((FH - row - 1) * FW) + (FW - 1 - col)];
                        tile_val = tile[c][nrow][ncol];
                        conv_val = conv_val + (fil_val * tile_val);
                    }
                }
            }

        output[(fil_num*W*H)+(image_row*W)+image_col] = conv_val;
    }
}

void C3(int W, int H, int C, int FW, int FH, int K, double* d_input, double* d_filter, double* d_output, double* h_output){
    struct timespec start, end;

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t i_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&i_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(i_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

    cudnnTensorDescriptor_t o_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&o_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(o_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

    cudnnFilterDescriptor_t f_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&f_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    cudnnConvolutionFwdAlgo_t conv_algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, i_desc, f_desc, conv_desc, o_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));

    size_t wssize = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, i_desc, f_desc, conv_desc, o_desc, conv_algo, &wssize));
    double *d_workspace;
    cudaMalloc(&d_workspace, wssize);

    size_t outsize = K*W*H* sizeof(double);
    double alpha = 1.0, beta = 0.0;

    clock_gettime(CLOCK_MONOTONIC,&start);
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, i_desc, d_input, f_desc, d_filter, conv_desc, conv_algo, d_workspace, wssize, &beta, o_desc, d_output));
    clock_gettime(CLOCK_MONOTONIC,&end);

    cudaMemcpy(h_output, d_output, outsize, cudaMemcpyDeviceToHost);

    double checksum = find_checksum(h_output);
    double time = (end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1E9)*1000.0;
    printf("%f,%4.3lf\n", checksum, time);

    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(i_desc);
    cudnnDestroyTensorDescriptor(o_desc);
    cudnnDestroyFilterDescriptor(f_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    cudnnDestroy(cudnn);

}

int main(){
    int W = 4096, H = 4096, C = 3, FW = 3, FH = 3, K = 10;
    struct timespec start1, end1;
    double *d_input, *d_filter, *d_output;
    double checksum=0.0, time1 = 0.0;

    int  inputsize = H*W*C* sizeof(double);
    int filtersize = FH*FW*C*K* sizeof(double);
    int outputsize = H*W*K* sizeof(double);

    cudaMalloc((void **)&d_input, inputsize);
    cudaMalloc((void **)&d_filter, filtersize);
    cudaMalloc((void **)&d_output, outputsize);

    double* h_input = (double*) malloc(inputsize);
    double* h_filter = (double*) malloc(filtersize);
    double* h_output = (double*) malloc(outputsize);


    for(int channel=0;channel<C;channel++){
        for(int height=0; height<H; height++){
            for(int width=0; width<W; width++){
                h_input[(channel*W*H)+(height*W)+width]= channel * (width+height);
            }
        }
    }

    for(int k=0;k<K;k++){
        for(int channel=0;channel<C;channel++){
            for(int height=0; height<FH; height++){
                for(int width=0; width<FW; width++){
                    h_filter[(k*C*FW*FH)+(channel*FW*FH)+(height*FW)+width] = (channel+k)*(width+height);
                }
            }
        }
    }

    cudaMemcpy(d_input, h_input, inputsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filtersize, cudaMemcpyHostToDevice);

    dim3 threads1(bs, bs);
    dim3 blocks1(ceil(W/bs), ceil(H/bs), K);
    clock_gettime(CLOCK_MONOTONIC,&start1);
    C1<<<blocks1, threads1>>>(W, H, C, FW, FH, K, d_input, d_filter, d_output);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&end1);
    cudaMemcpy(h_output, d_output, outputsize, cudaMemcpyDeviceToHost);
    time1 = (end1.tv_sec - start1.tv_sec + (end1.tv_nsec - start1.tv_nsec)/1E9)*1000.0;
    checksum = find_checksum(h_output);
    printf("%f,%4.3lf\n", checksum, time1);

    cudaFree(d_output);
    cudaMalloc((void **)&d_output, outputsize);

    dim3 threads2(bs, bs, K);
    dim3 blocks2(ceil(W/bs), ceil(H/bs), 1);
    clock_gettime(CLOCK_MONOTONIC,&start1);
    C2<<<blocks2, threads2>>>(W, H, C, FW, FH, K, d_input, d_filter, d_output);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&end1);
    cudaMemcpy(h_output, d_output, outputsize, cudaMemcpyDeviceToHost);
    time1 = (end1.tv_sec - start1.tv_sec + (end1.tv_nsec - start1.tv_nsec)/1E9)*1000.0;
    checksum = find_checksum(h_output);
    printf("%f,%4.3lf\n", checksum, time1);

    cudaFree(d_output);
    cudaMalloc((void **)&d_output, outputsize);

    C3(W, H, C, FW, FH, K, d_input, d_filter, d_output, h_output);


    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    free(h_output);
    free(h_input);
    free(h_filter);

    return 0;

}
