#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include "NvInfer.h"
#include <opencv2/core.hpp>
#include <vector>
#include <queue>
#include <set>
#include <thread>
#include <mutex>
#include <iostream>
#include <string>
#include <spdlog/spdlog.h>

using namespace nvinfer1;

/**
 * @brief detection result
 */
struct Object
{
    cv::Rect2f bbox;    ///< bounding box
    int category;       ///< object class
    float confidence;   ///< confidence score
};

/**
 * @brief anchor points information
 */
struct GridAndStride
{
    int grid_x;         ///< grid position x
    int grid_y;         ///< grid position y
    int stride;         ///< stride level
};

// tensorrt logger class
class Logger : public ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

/**
 * @brief detector class. 
 * Each detector runs on one gpu and with its own thread.
 */
class Detector
{
public:
    Detector(const char* model_path, int gpu_id, std::string config_file, std::shared_ptr<spdlog::logger> logger);
    ~Detector();
    void process(std::vector<cv::Mat>& images, std::vector<std::vector<Object>>& results);
private:
    void generateGridStrides(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    void postprocess(float* outpub_blob, float scale, int img_w, int img_h, std::vector<Object>& objects);

    int gpu_id;
    float conf_thresh;
    std::set<int> ignore_types;

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    int inputIndex;
    int outputIndex;
    void* dev_buf[2];
    float* host_input_buf;
    float* host_output_buf;

    std::vector<int> strides;
    std::vector<GridAndStride> grid_strides;
    int input_w;
    int input_h;
    int num_classes;
    int input_size;
    int output_size;
    const char* input_blob_name = "images";
    const char* output_blob_name = "output";

    std::shared_ptr<spdlog::logger> logger;
};

#endif