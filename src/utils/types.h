#pragma once

#include "config.h"

struct YoloKernel {
  int width;
  int height;
  float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {  // 使用 alignas(float) 来指定结构体成员的对齐方式
  float bbox[4];  // 一个包含四个 float 的数组，表示 bounding box 的坐标 // xmin ymin xmax ymax
  float conf;  // 
  float class_id;
};