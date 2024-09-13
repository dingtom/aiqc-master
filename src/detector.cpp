#include "detector.h"
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <yaml-cpp/yaml.h>

/* constant definition */
static const int MAX_BATCH = 16;
static const float NMS_THRESH = 0.1;

/* global variable */
static Logger gLogger;

static const unsigned char key[32] = {0x5a, 0xae, 0x58, 0x7d, 0x49, 0xaf, 0x5c, 0x5a,
                                      0xf1, 0xc4, 0x27, 0xc4, 0x99, 0x20, 0x72, 0x98,
                                      0x9c, 0x5a, 0xbf, 0x7d, 0x0a, 0xed, 0x48, 0x0e,
                                      0x19, 0x40, 0xdc, 0xe3, 0x9a, 0x1b, 0x9f, 0xc4};
// static const unsigned char iv[16] = {0x16, 0x37, 0xe9, 0xff, 0x71, 0x7f, 0xe1, 0x24,
//                             0xac, 0x11, 0x01, 0x37, 0xa4, 0x70, 0xd3, 0x9f};


/* function definition */
static void resizeKeepRatio(cv::Mat &src, cv::Mat &dst, int dst_w, int dst_h, float &scale) {
    int src_w = src.cols;
    int src_h = src.rows;
    scale = std::max(1.0 * src_w / dst_w, 1.0 * src_h / dst_h);
    int rs_w = src_w / scale;
    int rs_h = src_h / scale;

    // resize while keeping aspect ratio
    cv::Mat rs;
    cv::resize(src, rs, cv::Size(rs_w, rs_h));
    // pad image to target size
    cv::copyMakeBorder(rs, dst, 0, dst_h - rs_h, 0, dst_w - rs_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}


static void nms(std::vector <Object> &objects, float nms_threshold) {
    // 首先按照对象的置信度从高到低排序整个对象列表
    std::sort(objects.begin(), objects.end(), [](Object a, Object b) { return a.confidence > b.confidence; });
    // 初始化一个与对象数量相同的浮点数向量vArea，用于存储每个边界框的面积
    std::vector<float> vArea(objects.size());
    for (int i = 0; i < int(objects.size()); ++i) {
        // 计算每个对象的边界框面积
        vArea[i] = objects[i].bbox.width * objects[i].bbox.height;
    }
    // 双层循环遍历所有对象对，检查是否有重叠的边界框
    for (int i = 0; i < int(objects.size()); ++i) {
        for (int j = i + 1; j < int(objects.size());) {
            // 计算两个边界框的左上角和右下角坐标
            float xx1 = (std::max)(objects[i].bbox.x, objects[j].bbox.x);
            float yy1 = (std::max)(objects[i].bbox.y, objects[j].bbox.y);
            float xx2 = (std::min)(objects[i].bbox.x + objects[i].bbox.width,
                                   objects[j].bbox.x + objects[j].bbox.width);
            float yy2 = (std::min)(objects[i].bbox.y + objects[i].bbox.height,
                                   objects[j].bbox.y + objects[j].bbox.height);
            // 计算两个边界框相交区域的宽度和高度
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            // 计算两个边界框的交集面积
            float inter = w * h;
            // 计算交并比（IoU），即交集面积除以两者并集面积
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_threshold) {
                // 移除置信度较低的对象（此处假设j指向的是较靠后且置信度更低的对象）
                objects.erase(objects.begin() + j);
                // 同时更新vArea数组，移除对应已删除对象的面积
                vArea.erase(vArea.begin() + j);
            } else {
                // 若交并比小于等于阈值，则保留两个框并移动到下一个待比较对象
                j++;
            }
        }
    }
}

static void handleErrors(void) {  // 定义一个静态函数 handleErrors，用于处理错误
    ERR_print_errors_fp(stderr);  // 打印错误信息到标准错误输出（stderr）
    abort();                      // 终止程序执行
}

// 模型解密
static void decrypt(
        unsigned char *in_buf,
        unsigned int in_size,        // 输入缓冲区的大小（字节数）
        unsigned char *out_buf,
        unsigned long &out_size,     // 输出缓冲区的大小（字节数），将被更新为实际解密数据的大小
        const unsigned char *key,    // 密钥，用于解密
        const unsigned char *iv      // 初始向量，用于 CBC 模式的解密
) {
    EVP_CIPHER_CTX *ctx;         // 创建一个 EVP_CIPHER_CTX 对象，用于进行解密操作
    int out_len;                     // 用于存储每次解密操作后产生的数据长度

    if (!(ctx = EVP_CIPHER_CTX_new())) {  // 创建一个新的 EVP_CIPHER_CTX 对象
        handleErrors();
    }

    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {  // 初始化解密上下文
        handleErrors();
    }

    if (1 != EVP_DecryptUpdate(ctx, out_buf, &out_len, in_buf, in_size)) {  // 解密输入缓冲区中的大部分数据
        handleErrors();
    }
    out_size = out_len;                         // 更新输出缓冲区的大小

    if (1 != EVP_DecryptFinal_ex(ctx, out_buf + out_len, &out_len)) {  // 解密剩余的数据，并完成解密操作
        handleErrors();
    }
    out_size += out_len;                     // 更新输出缓冲区的大小，包括最后解密的数据
}


Detector::Detector(const char *model_path, int gpu_id, std::string config_file,
                   std::shared_ptr <spdlog::logger> logger) {
    this->logger = logger;
    YAML::Node config = YAML::LoadFile(config_file);
    try {
        YAML::Node defects = config["defects"];
        num_classes = 0;
        for (int i = 0; i < defects.size(); i++) {
            int idx = defects[i]["id"].as<int>();
            if (!defects[i]["enable"].as<bool>()) {
                ignore_types.insert(idx);
                logger->info("disable detecting {}", defects[i]["name"].as<std::string>());
            }
            num_classes += 1;
        }
        conf_thresh = config["confidence_threshold"].as<float>();
        logger->info("detection confidence threshold = {:.2f}", conf_thresh);
    } catch (std::exception &e) {
        logger->error("parse config file error: {}", e.what());
    }


    input_w = 1920;
    input_h = 1216;
    strides = {8, 16, 32};
    generateGridStrides(strides, grid_strides);
    input_size = 3 * input_w * input_h;
    output_size = grid_strides.size() * (num_classes + 5);

    // GPU 信息
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);
    logger->info("run on {}", props.name);

    // create infer runtime
    this->gpu_id = gpu_id;
    cudaSetDevice(gpu_id);
    runtime = createInferRuntime(gLogger);

    // read engine and deserialize
    unsigned char *decoded_model = NULL;
    size_t model_size;
    unsigned char *encoded_model = NULL;
    size_t encoded_size;
    std::ifstream file(model_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        encoded_size = file.tellg();
        file.seekg(0, file.beg);
        encoded_model = new unsigned char[encoded_size];
        file.read((char *) encoded_model, encoded_size);
        file.close();

        decoded_model = new unsigned char[encoded_size];
        decrypt(encoded_model, encoded_size, decoded_model, model_size, key, (unsigned char *) props.uuid.bytes);
        if (encoded_model) {
            delete[] encoded_model;
        }
        logger->debug("encoded model size = {:d}, decoded model size = {:d}", encoded_size, model_size);
    } else {
        logger->error("load model file {} fail", model_path);
    }
    engine = runtime->deserializeCudaEngine(decoded_model, model_size);
    if (engine)
    if (decoded_model) {
        delete[] decoded_model;
    }
    if (engine == nullptr) {
        logger->error("deserializeCudaEngine {} fail", model_path);
    } else {
        logger->info("load model success");
    }

    // create execution context
    context = engine->createExecutionContext();
    inputIndex = engine->getBindingIndex(input_blob_name);
    outputIndex = engine->getBindingIndex(output_blob_name);
    cudaMalloc(&dev_buf[inputIndex], input_size * sizeof(float) * MAX_BATCH);
    cudaMalloc(&dev_buf[outputIndex], output_size * sizeof(float) * MAX_BATCH);

    host_input_buf = (float *) malloc(input_size * sizeof(float));
    host_output_buf = (float *) malloc(output_size * sizeof(float) * MAX_BATCH);
}

Detector::~Detector() {
    cudaFree(dev_buf[inputIndex]);
    cudaFree(dev_buf[outputIndex]);
    if (host_input_buf) {
        free(host_input_buf);
    }
    if (host_output_buf) {
        free(host_output_buf);
    }
    if (context) {
        context->destroy();
    }
    if (engine) {
        engine->destroy();
    }
    if (runtime) {
        runtime->destroy();
    }
}

void Detector::process(std::vector <cv::Mat> &images, std::vector <std::vector<Object>> &results) {
    // 设置当前使用的GPU设备ID
    cudaSetDevice(gpu_id);
    // 创建一个新的CUDA流，用于异步数据传输和计算
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 初始化已处理图像的数量
    int processedNum = 0;
    while (processedNum < images.size()) {
        // 计算当前批次处理的图像数量，不超过最大批次大小MAX_BATCH
        int curBatchSize = (processedNum + MAX_BATCH < images.size()) ? MAX_BATCH : (images.size() - processedNum);
        // 初始化图像缩放比例向量，初始化为1.0
        std::vector<float> scales(curBatchSize, 1.);
        // preprocess
        for (int i = 0; i < curBatchSize; i++) {
            // 获取当前批次的图像
            cv::Mat img = images[processedNum + i];
            // 如果图像为空，则创建一个全黑图像并设置默认缩放比例.
            // 否则，调整图像大小保持长宽比，并记录实际缩放比例
            cv::Mat rgb;
            if (img.empty()) {
                rgb = cv::Mat(input_h, input_w, CV_8UC3, cv::Scalar(0, 0, 0));
                scales[i] = 1.0;
            } else {
                resizeKeepRatio(img, rgb, input_w, input_h, scales[i]);
            }
            // 图像转为RGB格式并分割通道
            std::vector <cv::Mat> channels;
            cv::split(rgb, channels);
            // 将每个通道的数据转换为float类型并复制到host_input_buf中
            for (int j = 0; j < 3; j++) {
                cv::Mat curChannel;
                channels[j].convertTo(curChannel, CV_32FC1);
                memcpy(&host_input_buf[j * input_h * input_w], curChannel.data, input_h * input_w * sizeof(float));
            }
            // 将处理好的图像数据从CPU拷贝到GPU内存中，准备推理
            cudaMemcpyAsync(dev_buf[inputIndex] + i * input_size * sizeof(float), host_input_buf,
                            input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        // 设置优化配置、输入尺寸，执行模型推理
        context->setOptimizationProfile(0);
        context->setBindingDimensions(0, Dims4(curBatchSize, 3, input_h, input_w));
        context->enqueueV2(dev_buf, stream, nullptr);
        // 将推理结果从GPU拷贝回CPU内存
        cudaMemcpyAsync(host_output_buf, dev_buf[outputIndex], output_size * sizeof(float) * MAX_BATCH,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        // 后处理阶段
        for (int i = 0; i < curBatchSize; i++) {
            std::vector <Object> objects;
            // 对每张图像的推理结果进行解码，得到目标对象信息
            postprocess(host_output_buf + i * output_size, scales[i], images[i].cols, images[i].rows, objects);
            // 将解析出的对象信息加入到最终结果容器中
            results.push_back(objects);
        }
        // 更新已处理图像的数量
        processedNum += curBatchSize;
    }
    // 清理CUDA流资源
    cudaStreamDestroy(stream);
}

void Detector::generateGridStrides(std::vector<int> &strides, std::vector <GridAndStride> &grid_strides) {
    for (auto stride: strides) {
        // 根据输入图像的高度（input_h）和当前遍历到的步长值（stride），计算网格在垂直方向（Y轴）的数量（num_grid_y）
        int num_grid_y = input_h / stride;
        // 根据输入图像的宽度（input_w）和当前步长值，计算网格在水平方向（X轴）的数量（num_grid_x）
        int num_grid_x = input_w / stride;
        for (int y = 0; y < num_grid_y; y++) {
            for (int x = 0; x < num_grid_x; x++) {
                // 创建一个新的GridAndStride结构体实例，包含当前的x、y坐标和当前步长值
                // 将这个结构体实例添加到grid_strides向量中，记录下所有的网格位置和对应的步长
                grid_strides.push_back((GridAndStride) {x, y, stride});
            }
        }
    }
}

void Detector::postprocess(float *output_blob, float scale, int img_w, int img_h, std::vector <Object> &objects) {
    // 遍历网格步长信息集合，grid_strides 中的每个元素包含用于定位预测框的网格坐标和步长信息。
    for (int i = 0; i < grid_strides.size(); i++) {
        // 获取当前网格单元对应的横纵坐标
        int grid_x = grid_strides[i].grid_x;
        int grid_y = grid_strides[i].grid_y;
        // 获取当前网格单元的步长
        int stride = grid_strides[i].stride;
        // 根据索引计算指向当前网格单元第一个预测框属性的指针
        float *output = &output_blob[(num_classes + 5) * i];
        // 解码中心坐标 (cx, cy)，将其与网格单元中心结合，并乘以步长得到预测框的绝对中心坐标
        float cx = (output[0] + grid_x) * stride;
        float cy = (output[1] + grid_y) * stride;
        // 解码预测框的宽和高，利用指数函数计算真实宽度和高度，再乘以步长得到绝对尺寸
        float w = exp(output[2]) * stride;
        float h = exp(output[3]) * stride;
        // 获取预测框所属物体的概率（也称为目标性）
        float objectness = output[4];
        // 若目标性的概率小于预设阈值 conf_thresh，则跳过此预测框
        if (objectness < conf_thresh) {
            continue;
        }
        // 初始化最大类别置信度为0，以及最大类别的索引
        float max_conf = 0.;
        int max_class = -1;
        // 遍历所有类别，寻找最高置信度类别
        for (int j = 0; j < num_classes; j++) {
            if (output[5 + j] > max_conf) {
                max_conf = output[5 + j];
                max_class = j;
            }
        }
        // 计算最终类别置信度，即最大类别置信度与目标性概率的乘积
        float conf = max_conf * objectness;
        // 如果最终类别置信度大于预设阈值且当前类别不在忽略类型列表中，则保留此预测框
        if (conf > conf_thresh && ignore_types.find(max_class) == ignore_types.end()) {
            Object obj;
            // 计算预测框的绝对坐标，并确保其位于图像范围内（使用MIN/MAX函数进行边界约束）
            float x1 = (cx - w / 2) * scale;
            float y1 = (cy - h / 2) * scale;
            float x2 = (cx + w / 2) * scale;
            float y2 = (cy + h / 2) * scale;
            x1 = MIN(MAX(0, x1), img_w - 1);
            x2 = MIN(MAX(0, x2), img_w - 1);
            y1 = MIN(MAX(0, y1), img_h - 1);
            y2 = MIN(MAX(0, y2), img_h - 1);
            float bw = x2 - x1;
            float bh = y2 - y1;
            obj.bbox = cv::Rect2f(x1, y1, bw, bh);
            obj.category = max_class;
            obj.confidence = conf;
            objects.push_back(obj);
        }
    }
    // 对检测到的对象列表进行非极大值抑制（NMS），去除冗余的检测框
    nms(objects, NMS_THRESH);
}