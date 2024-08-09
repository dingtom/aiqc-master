/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <stdint.h>
#include <iostream>

// 计算每个网格的检测结果
__global__ void gpuYoloLayer_nc(
    const float *input, int *num_detections, float *detection_boxes, float *detection_scores, int *detection_classes,
    const float scoreThreshold, const uint netWidth, const uint netHeight, const uint gridSizeX, const uint gridSizeY,
    const uint numOutputClasses, const uint numBBoxes, const float scaleXY, const float *anchors)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的x坐标，对应的是网格的x坐标
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程的y坐标，对应的是网格的y坐标
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z; // 当前线程的z坐标，对应的是anchor的索引

    if (x_id >= gridSizeX || y_id >= gridSizeY || z_id >= numBBoxes)
        return;

    const int numGridCells = gridSizeX * gridSizeY; // 一个特征图的网格数量：80*80，40*40，20*20
    const int bbindex = y_id * gridSizeX + x_id;    // 当前网格的索引

    const float objectness = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]; // 当前网格的检测结果的置信度

    if (objectness < scoreThreshold) // 如果置信度小于阈值0.25，直接返回，不进行后续的处理
        return;

    // 有检测结果
    int count = (int)atomicAdd(num_detections, 1); // 计算当前检测到的目标数量（atomicAdd具有原子性和同步性，可以确保多个线程对同一个变量进行操作时不会出现数据不一致的情况）

    const float alpha = scaleXY;             // 2
    const float beta = -0.5 * (scaleXY - 1); // -0.5*(2-1) = -0.5

    float x = (input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)] * alpha + beta + x_id) * netWidth / gridSizeX;  // 当前检测结果的中心点坐标：tx*2-0.5+y_id， *netWidth/gridSizeX相当于640/80或者640/40或者640/20，即每个网格的宽度，这样可以将中心点坐标转换为相对于整个图像的坐标
    float y = (input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)] * alpha + beta + y_id) * netHeight / gridSizeY; // 同理，y方向的坐标
    float w = __powf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)] * 2, 2) * anchors[z_id * 2];              // 当前检测结果的宽度: （tw*2）^2 * anchor_w
    float h = __powf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)] * 2, 2) * anchors[z_id * 2 + 1];          // 当前检测结果的高度：（th*2）^2 * anchor_h

    float maxProb = 0.0f;
    int maxIndex = -1;

    // 寻找最大的概率列别的索引
    for (uint i = 0; i < numOutputClasses; ++i)
    {
        float prob = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]; // 对应各个类别（与训练集一致）的概率

        if (prob > maxProb) // 找到最大的概率
        {
            maxProb = prob;
            maxIndex = i;
        }
    }

    detection_boxes[count * 4 + 0] = x - 0.5 * w;   // x_min
    detection_boxes[count * 4 + 1] = y - 0.5 * h;   // y_min
    detection_boxes[count * 4 + 2] = x + 0.5 * w;   // x_max
    detection_boxes[count * 4 + 3] = y + 0.5 * h;   // y_max
    detection_scores[count] = objectness * maxProb; // 将置信度和概率相乘作为最终的得分
    detection_classes[count] = maxIndex;            // 最大概率的类别索引
}

cudaError_t cudaYoloLayer_nc(
    const void *input, void *num_detections, void *detection_boxes, void *detection_scores, void *detection_classes,
    const uint &batchSize, uint64_t &inputSize, uint64_t &outputSize, const float &scoreThreshold, const uint &netWidth,
    const uint &netHeight, const uint &gridSizeX, const uint &gridSizeY, const uint &numOutputClasses, const uint &numBBoxes,
    const float &scaleXY, const void *anchors, cudaStream_t stream);

cudaError_t cudaYoloLayer_nc(
    const void *input, void *num_detections, void *detection_boxes, void *detection_scores, void *detection_classes,
    const uint &batchSize, uint64_t &inputSize, uint64_t &outputSize, const float &scoreThreshold, const uint &netWidth,
    const uint &netHeight, const uint &gridSizeX, const uint &gridSizeY, const uint &numOutputClasses, const uint &numBBoxes,
    const float &scaleXY, const void *anchors, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4); // 每个block的线程数量
    // 分别计算x,y,z方向上的block数量
    // x方向：根据网格的宽度计算需要多少个block，对应的是网格的x坐标
    // y方向：根据网格的高度计算需要多少个block，对应的是网格的y坐标
    // z方向：根据anchor的数量计算需要多少个block（这里是3），对应的是anchor的索引
    dim3 number_of_blocks((gridSizeX / threads_per_block.x) + 1,
                          (gridSizeY / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);

    for (unsigned int batch = 0; batch < batchSize; ++batch) // 遍历batch
    {
        gpuYoloLayer_nc<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float *>(input) + (batch * inputSize),
            reinterpret_cast<int *>(num_detections) + (batch),
            reinterpret_cast<float *>(detection_boxes) + (batch * 4 * outputSize),
            reinterpret_cast<float *>(detection_scores) + (batch * outputSize),
            reinterpret_cast<int *>(detection_classes) + (batch * outputSize),
            scoreThreshold, netWidth, netHeight, gridSizeX, gridSizeY, numOutputClasses, numBBoxes, scaleXY,
            reinterpret_cast<const float *>(anchors));
    }
    return cudaGetLastError();
}
