  hipdnnHandle_t cudnn;
  checkCUDNN(hipdnnCreate(&cudnn));

  hipdnnTensorDescriptor_t input_descriptor;
  checkCUDNN(hipdnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(hipdnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/HIPDNN_TENSOR_NCHW,
                                        /*dataType=*/HIPDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/Hin,
                                        /*image_width=*/Win));

  hipdnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(hipdnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(hipdnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/HIPDNN_DATA_FLOAT,
                                        /*format=*/HIPDNN_TENSOR_NCHW,
                                        /*out_channels=*/M,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/K,
                                        /*kernel_width=*/K));

  hipdnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(hipdnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(hipdnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/0,
                                             /*pad_width=*/0,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/HIPDNN_CROSS_CORRELATION,
                                             /*computeType=*/HIPDNN_DATA_FLOAT));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(hipdnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  #ifdef DEBUG
  std::cerr << "Output Image(NxHxWxC): " << batch_size << " x "
            << height << " x " << width << " x " << channels << std::endl;
  #endif

  hipdnnTensorDescriptor_t output_descriptor;
  checkCUDNN(hipdnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(hipdnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/HIPDNN_TENSOR_NCHW,
                                        /*dataType=*/HIPDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/M,
                                        /*image_height=*/Hout,
                                        /*image_width=*/Wout));

  int requestedAlgoCount = HIPDNN_CONVOLUTION_FWD_ALGO_COUNT;
  int returnedAlgoCount = -1;
  hipdnnConvolutionFwdAlgoPerf_t results[HIPDNN_CONVOLUTION_FWD_ALGO_COUNT];

  checkCUDNN(hipdnnFindConvolutionForwardAlgorithm(cudnn,
                                                  input_descriptor,
                                                  kernel_descriptor,
                                                  convolution_descriptor,
                                                  output_descriptor,
                                                  requestedAlgoCount,
                                                  &returnedAlgoCount,
                                                  results));

  #ifdef DEBUG
  std::cout << "Testing hipdnnFindConvolutionForwardAlgorithm ...\n";
  for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n",
           hipdnnGetErrorString(results[algoIndex].status),
           results[algoIndex].algo, results[algoIndex].time,
           (unsigned long long)results[algoIndex].memory);
  }
  #endif

  hipdnnConvolutionFwdAlgo_t convolution_algorithm = results[0].algo;

  size_t workspace_bytes{0};
  checkCUDNN(hipdnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  void* d_workspace{nullptr};
  if (workspace_bytes > 0) {
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    hipMalloc(&d_workspace, workspace_bytes);
  }

  const float alpha = 1.0f, beta = 0.0f;

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkCUDNN(hipdnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       dX, //d_input,
                                       kernel_descriptor,
                                       dW, //d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       dY )); //d_output
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  hipdnnDestroyTensorDescriptor(input_descriptor);
  hipdnnDestroyTensorDescriptor(output_descriptor);
  hipdnnDestroyFilterDescriptor(kernel_descriptor);
  hipdnnDestroyConvolutionDescriptor(convolution_descriptor);
  hipdnnDestroy(cudnn);
