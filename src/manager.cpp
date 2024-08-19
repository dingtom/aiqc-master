#include "manager.h"
#include "json.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <algorithm>
#include <string>
#include <cctype>
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>
#include <libbase64.h>
#include <opencv2/imgcodecs.hpp>
#include <turbojpeg.h>
#include <cuda_runtime_api.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <yaml-cpp/yaml.h>

using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using namespace nlohmann;

Manager *Manager::m_instance = NULL;  // 在类定义外部初始化静态成员变量m_instance

static const int NUM_DEC_WORKER = 4;
static const int MAX_BATCH = 8;

#ifndef PASS_BINARY
static const int DEC_BASE64_OUT_BUF_SIZE = 25000000;
//static 关键字表明这是一个静态变量，意味着它的作用域仅限于定义它的文件，并且它在程序的生命周期中只初始化一次。
//const 表示这是一个常量，其值在定义后不可更改。
//DEC_BASE64_OUT_BUF_SIZE 是一个常量整数，值为25000000，表示解码Base64后输出缓冲区的建议大小，
// 单位通常是字节。这个值用来预先分配足够大的空间来存放解码后的数据。

static void decode_base64(char *src_buf, size_t src_size, char *out_buf, size_t *out_size) {
    base64_decode(src_buf, src_size, out_buf, out_size, 0);
    // printf("src: %d, out: %d\n", src_size, *out_size);
}

#endif

Manager::Manager() {}

// 这是一个静态成员函数，用于获取Manager类的唯一实例
Manager *Manager::getManager() {
    if (!m_instance) {
        m_instance = new Manager(); // 如果不存在，则创建一个新的Manager对象，并将其地址赋给m_instance
    }
    return m_instance; // 返回Manager类的唯一实例的指针
}

void Manager::startService(int port, std::string config_file) {
    Manager *manager = Manager::getManager();
    manager->init(port, config_file);
    manager->run();
}

void Manager::stopService() {
    Manager *manager = Manager::getManager();
    manager->stop();
}


// 初始化Manager类的成员函数，设置WebSocket服务器配置和读取配置文件
void Manager::init(int port, std::string config_file) {
    // 配置置WebSocket服务器错误日志级别，仅记录错误级别日志
    m_server.clear_error_channels(websocketpp::log::elevel::all);
    m_server.set_error_channels(websocketpp::log::elevel::rerror);
    // 配置置WebSocket服务器访问日志级别，记录核心访问日志
    m_server.clear_access_channels(websocketpp::log::alevel::all);
    m_server.set_access_channels(websocketpp::log::alevel::access_core);

    // 设置消息处理回调函数，当接收到消息时调用Manager类的msgHandler方法处理
    m_server.set_message_handler(bind(&Manager::msgHandler, this, _1, _2));
//    bind: 这是C++标准库中的一个函数，用于生成可调用的对象，这个对象能够在调用时调用某个函数，并且预填充部分或全部的参数。
//    这里是用来绑定成员函数Manager::msgHandler和特定的this指针，以及两个占位符_1和_2。
//    &Manager::msgHandler: 这是指向Manager类中名为msgHandler的成员函数的指针。
//    成员函数需要类对象的上下文（即this指针）才能被调用，所以通过bind将this绑定进来。

//    this: 是C++中的一个关键字，代表当前对象的指针。在这里，它指向Manager类的实例，
//    确保msgHandler函数被正确地作为成员函数调用，可以访问到Manager类的成员变量和其他方法。

//    _1 和 _2: 这些是std::placeholders库中的占位符，用于在bind表达式中表示未确定的参数。
//    _1代表第一个未被显式指定的参数， _2代表第二个。
//    在这个上下文中，它们分别代表了当消息实际到来时，服务器接收到的消息内容和相关的会话（或客户端连接）信息。

    // 初始化Asio库，用于异步IO操作
    m_server.init_asio();
    // 设置单条消息的最大大小限制，这里是250MB
    m_server.set_max_message_size(2500 * 1024 * 1024);

    // 设置类成员变量
    this->port = port;
    this->config_file = config_file;
    // 初始化日志器，分别用于记录Manager和WebSocket相关的日志信息
    this->manager_logger = spdlog::stdout_color_st("manager");
    this->websocket_logger = spdlog::stdout_color_mt("websocket");

    // 从YAML配置文件加载配置
    YAML::Node config = YAML::LoadFile(config_file);
    // 读取缺陷类别配置

    YAML::Node defects = config["defects"];
    for (int i = 0; i < defects.size(); i++) {
        // 将缺陷名称添加到class_names向量中
        class_names.push_back(defects[i]["name"].as<std::string>());
    }

    // 读取模型目录配置
    this->model_dir = config["model_dir"].as<std::string>();
    // 记录日志，显示模型目录
    spdlog::info("model_dir: {}", model_dir);

    // 读取GPU配置
    YAML::Node visible_gpus = config["gpu"];
    for (int i = 0; i < visible_gpus.size(); i++) {
        // 将指定的GPU编号插入到use_gpus集合中
        use_gpus.insert(visible_gpus[i].as<int>());
    }

    // 读取版本信息
//    std::ifstream ifs("version");
//    if (ifs.is_open()) {
//        // 如果文件打开成功，读取第一行作为版本信息
//        std::getline(ifs, this->version);
//    } else {
//        // 否则版本信息默认为"NA"
//        this->version = "NA";
//    }
    this->version = model_dir;
    // 记录日志，显示正在运行的Fabric缺陷算法服务器版本信息
    manager_logger->info("running fabric defect algorithm server, version: {}", this->version);
}

void Manager::run() {
    // 设置运行标志为true，表明Manager开始运行
    running = true;

    // 创建指定数量的解码工作线程（NUM_DEC_WORKER定义了解码线程的数量）
    for (int i = 0; i < NUM_DEC_WORKER; i++) {
        // 使用new创建一个新的线程，执行Manager类的decode方法，并传入this（当前对象）和线程编号i作为参数
        std::thread *decode_worker = new std::thread(&Manager::decode, this, i);
        // 将创建的线程指针添加到decode_workers容器中保存
        decode_workers.push_back(decode_worker);
    }

    // 获取当前系统中可用的GPU数量
    cudaGetDeviceCount(&num_gpus);

    // 如果没有可用的GPU
    if (num_gpus == 0) {
        // 日志记录错误信息
        manager_logger->error("no available gpus");
    } else {
        // 否则，记录可用GPU的数量
        manager_logger->info("{} available gpu(s)", num_gpus);
    }

    // 根据配置使用特定的GPU启动推理工作线程
    for (int i = 0; i < num_gpus; i++) {
        // 如果当前GPU索引在使用列表中
        if (use_gpus.find(i) != use_gpus.end()) {
            // 记录将使用的GPU信息
            manager_logger->info("use gpu {}", i);
            // 创建一个新的线程执行Manager类的inference方法，并传入GPU索引i
            std::thread *inference_worker = new std::thread(&Manager::inference, this, i);
            // 保存推理工作线程的指针
            inference_workers.push_back(inference_worker);
        }
    }

    // 创建一个单独的工作线程用于发送结果
    reply_worker = new std::thread(&Manager::sendResult, this);

    // 尝试启动WebSocket服务器
    try {
        // 监听指定端口
        m_server.listen(port);
        // 开始接受连接
        m_server.start_accept();
        // 记录日志，表示开始监听端口
        manager_logger->info("start to listen port {}", port);
        // 运行服务器，进入事件循环处理连接和消息
        m_server.run();
    }
        // 捕获并处理WebSocket相关的异常
    catch (websocketpp::exception const &e) {
        // 记录启动失败的日志，包括异常信息
        manager_logger->error("start error: {}", e.what());
    }
}

void Manager::stop() {
    // 设置运行标志为false，指示Manager应该停止运行
    running = false;

    // 停止WebSocket服务器的所有活动
    m_server.stop();
    // 停止监听新的连接请求
    m_server.stop_listening();

    // 记录日志，表明不再监听指定端口
    manager_logger->info("stop to listen port {}", port);

    // 等待并停止所有的解码工作线程
    for (int i = 0; i < NUM_DEC_WORKER; i++) {
        // 检查线程指针是否有效
        if (decode_workers[i]) {
            // 等待线程完成（结束）
            decode_workers[i]->join();
            // 释放线程资源
            delete decode_workers[i];
        }
    }

    // 同样地，停止所有推理工作线程
    for (int i = 0; i < num_gpus; i++) {
        if (inference_workers[i]) {
            inference_workers[i]->join();
            delete inference_workers[i];
        }
    }

    // 停止并清理回复工作线程
    if (reply_worker) {
        reply_worker->join();
        delete reply_worker;
    }

    // 记录日志，表明所有工作线程已停止，服务器即将退出
    manager_logger->info("all workers stopped, server exit");
}

void Manager::msgHandler(websocketpp::connection_hdl hdl, server::message_ptr msg) {
    auto con = m_server.get_con_from_hdl(hdl);
    std::string subpath = con->get_resource();
    websocket_logger->info("received msg from {}", subpath);
    // deal with flaw command
    if (subpath == "/flaw") {
        if (msg->get_opcode() == websocketpp::frame::opcode::TEXT) {
            std::string payload = msg->get_payload();
            websocket_logger->debug("received text data {:.2f}KB", payload.size() / 1024.0);
            try {
                json data = json::parse(payload);
                for (auto s_data: data) {
                    std::string batch_no = s_data.at("no").get<std::string>();
                    std::string base64_image = s_data.at("image").get<std::string>();
                    websocket_logger->info("received task {}, {:.2f}KB", batch_no, base64_image.size() / 1024.0);

                    // auto t1 = std::chrono::steady_clock::now();
                    // std::vector<json> results;
                    // json jresult;
                    // jresult["no"] = batch_no;
                    // std::vector<json> defects;
                    // jresult["defects"] = defects;
                    // results.push_back(jresult);
                    // json reply = results;
                    // try
                    // {
                    //     m_server.send(hdl, reply.dump(), websocketpp::frame::opcode::TEXT);
                    // }
                    // catch (websocketpp::exception const & e) 
                    // {
                    //     websocket_logger->error("send error: {}", e.what());
                    // }
                    // auto t2 = std::chrono::steady_clock::now();
                    // auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
                    // websocket_logger->debug("send result, runtime {}ms", runtime.count());


                    DecTask decode_task;
                    decode_task.no = batch_no;
#ifdef USE_PATH
                    decode_task.image_path = base64_image;
#else
                    decode_task.data_size = base64_image.size();
                    decode_task.image_data.reset(new char[decode_task.data_size]);
                    memcpy(decode_task.image_data.get(), base64_image.c_str(), decode_task.data_size);
                    // decode_task.base64_image = "/data/" + batch_no + ".jpg";
#endif
                    decode_task.handle = hdl;
                    decode_lock.lock();
                    decode_tasks.push(decode_task);
                    decode_lock.unlock();
                }
            }
            catch (std::exception const &e) {
                websocket_logger->error("handle message error: {}", e.what());
            }
        } else {
            std::string payload = msg->get_payload();
            websocket_logger->debug("received binary data {:.2f}KB", payload.size() / 1024.0);
#ifdef PASS_BINARY
            const char* data = payload.c_str();
            size_t total_size = payload.size();
            size_t cur_size = 0;
            const char* cur_data = data;
            while (cur_size < total_size)
            {
                DecTask decode_task;
                uint16_t no_size;
                char no[100];
                uint32_t data_size;
                memcpy(&no_size, cur_data, 2);
                memcpy(&data_size, cur_data+2, 4);
                if (cur_size + no_size + data_size + 6 > total_size)
                {
                    websocket_logger->warn("buffer overflow, throw away data");
                    break;
                }
                memcpy(no, cur_data+6, no_size);
                no[no_size] = 0;
                websocket_logger->info("received task {}, {:.2f}KB", no, data_size/1024.0);

                    // auto t1 = std::chrono::steady_clock::now();
                    // std::vector<json> results;
                    // json jresult;
                    // jresult["no"] = no;
                    // std::vector<json> defects;
                    // jresult["defects"] = defects;
                    // results.push_back(jresult);
                    // json reply = results;
                    // try
                    // {
                    //     m_server.send(hdl, reply.dump(), websocketpp::frame::opcode::TEXT);
                    // }
                    // catch (websocketpp::exception const & e) 
                    // {
                    //     websocket_logger->error("send error: {}", e.what());
                    // }
                    // auto t2 = std::chrono::steady_clock::now();
                    // auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
                    // websocket_logger->debug("send result, runtime {}ms", runtime.count());
                decode_task.no = no;
                decode_task.handle = hdl;
                decode_task.data_size = data_size;
                decode_task.image_data.reset(new char[data_size]);
                memcpy(decode_task.image_data.get(), cur_data+6+no_size, data_size);
                decode_lock.lock();
                decode_tasks.push(decode_task);
                decode_lock.unlock();
                cur_size += (6 + no_size + data_size);
                cur_data = data + cur_size;
            }
#endif
        }
    }
        // deal with system command
    else if (subpath == "/system") {
        if (msg->get_opcode() == websocketpp::frame::opcode::TEXT) {
            std::string payload = msg->get_payload();
            try {
                json data = json::parse(payload);
                std::string cmd = data.at("command").get<std::string>();
                std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](unsigned char c) { return std::tolower(c); });
                websocket_logger->info("received command: {}", cmd);
                if (cmd == "shutdown") {
                    this->stop();
                    manager_logger->flush();
                    websocket_logger->flush();
                    system("echo shutdown > /shutdown_signal");
                    // system("echo o > /sysrq");
                } else if (cmd == "restart") {
                    this->stop();
                    manager_logger->flush();
                    websocket_logger->flush();
                    system("echo restart > /shutdown_signal");
                    // system("echo b > /sysrq");
                } else if (cmd == "version") {
                    json reply;
                    reply["version"] = version;
                    try {
                        m_server.send(hdl, reply.dump(), websocketpp::frame::opcode::TEXT);
                        websocket_logger->info("return msg: {}", reply.dump());
                    }
                    catch (websocketpp::exception const &e) {
                        websocket_logger->error("send error: {}", e.what());
                    }
                }
            }
            catch (std::exception const &e) {
                websocket_logger->error("handle message error: {}", e.what());
            }
        }
    }
}

#ifdef USE_PATH
void Manager::decode(int id)
{
    char logger_name[64];
    sprintf(logger_name, "decode worker %d", id);
    auto logger = spdlog::stdout_color_st(logger_name);

    tjhandle tj_handle = tjInitDecompress();

    while (running)
    {
        DecTask decode_task;
        decode_lock.lock();
        if (!decode_tasks.empty())
        {
            decode_task = decode_tasks.front();
            decode_tasks.pop();
        }
        else
        {
            decode_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        decode_lock.unlock();

        auto t1 = std::chrono::steady_clock::now();
        InfTask decode_result;
        FILE* fp = fopen(decode_task.image_path.c_str(), "rb");
        unsigned long src_size = 0;
        unsigned char* src_buf = NULL;
        if (fp)
        {
            fseek(fp, 0, SEEK_END);
            src_size = ftell(fp);
            src_buf = new unsigned char[src_size];
            fseek(fp, 0, SEEK_SET);
            fread(src_buf, src_size, 1, fp);
            fclose(fp);
        }
        int subsamp, cs;
        int width, height;
        int ret = tjDecompressHeader3(tj_handle, src_buf, src_size, &width, &height, &subsamp, &cs);
        if (ret != 0)
        {
            logger->error("decompress jpeg to rgb fail");
        }
        cv::Mat img(height, width, CV_8UC3);
        ret = tjDecompress2(tj_handle, src_buf, src_size, img.data, width, width * 3, height, TJPF_BGR, TJFLAG_NOREALLOC);
        if (ret != 0)
        {
            logger->error("decompress jpeg to rgb fail");
        }
        decode_result.no = decode_task.no;
        decode_result.handle = decode_task.handle;
        decode_result.image = img;
        inference_lock.lock();
        inference_tasks.push(decode_result);
        inference_lock.unlock();
        auto t2 = std::chrono::steady_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
        logger->debug("task {} runtime {}ms", decode_task.no, runtime.count());
        if (src_buf)
        {
            delete [] src_buf;
        }
    }

    tjDestroy(tj_handle);
    logger->info("stop");
}
#else

// 定义Manager类的decode成员函数，负责解码图像数据
void Manager::decode(int id) {
    // 初始化日志器名称，包含解码工作线程的ID
    char logger_name[64];
    sprintf(logger_name, "decode worker %d", id);
    // 创建并配置日志器，用于记录当前解码工作线程的日志
    auto logger = spdlog::stdout_color_st(logger_name);

    // 初始化TurboJPEG解码器句柄
    tjhandle tj_handle = tjInitDecompress();

    // 根据是否定义PASS_BINARY宏决定是否分配解码缓冲区
#ifndef PASS_BINARY
    char *out_buf = new char[DEC_BASE64_OUT_BUF_SIZE]; // 分配解码缓冲区
#endif

    // 当解码线程运行标志为真时，持续执行
    while (running) {
        // 定义解码任务变量
        DecTask decode_task;

        // 上锁，安全访问解码任务队列
        decode_lock.lock();
        // 如果解码任务队列不为空，取出首个任务
        if (!decode_tasks.empty()) {
            decode_task = decode_tasks.front();
            decode_tasks.pop(); // 移除已取出的任务
        } else {
            // 若队列为空，解锁并稍作等待后继续循环
            decode_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 等待50毫秒
            continue; // 继续下一次循环
        }
        decode_lock.unlock(); // 解锁

        // 记录开始时间
        auto t1 = std::chrono::steady_clock::now();

        // 初始化推理任务变量
        InfTask decode_result;
        size_t src_size;

        // 根据是否定义PASS_BINARY宏选择解码方式
#ifndef PASS_BINARY
        // 调用base64解码函数
        decode_base64(decode_task.image_data.get(), decode_task.data_size, out_buf, &src_size);
#endif

        // 获取JPEG图像的宽、高、子采样和色彩空间信息
        int subsamp, cs;
        int width, height;

        // 根据是否定义PASS_BINARY宏选择解压头信息的方式
#ifdef PASS_BINARY
        int ret = tjDecompressHeader3(tj_handle, (const unsigned char*)decode_task.image_data.get(), decode_task.data_size, &width, &height, &subsamp, &cs);
#else
        int ret = tjDecompressHeader3(tj_handle, (const unsigned char *) out_buf, src_size, &width, &height, &subsamp, &cs);
#endif

        // 错误处理
        if (ret != 0) {
            logger->error("decompress jpeg to rgb fail");
        }

        // 使用OpenCV创建图像矩阵准备存储解码后的图像
        cv::Mat img(height, width, CV_8UC3);

        // 根据是否定义PASS_BINARY宏选择解码到RGB的方式
#ifdef PASS_BINARY
        ret = tjDecompress2(tj_handle, (const unsigned char*)decode_task.image_data.get(), decode_task.data_size, img.data, width, width * 3, height, TJPF_BGR, TJFLAG_NOREALLOC);
#else
        ret = tjDecompress2(tj_handle, (const unsigned char *) out_buf, src_size, img.data, width, width * 3, height, TJPF_BGR, TJFLAG_NOREALLOC);
#endif

        // 错误处理
        if (ret != 0) {
            logger->error("decompress jpeg to rgb fail");
        }

        // 封装解码结果
        decode_result.no = decode_task.no;
        decode_result.handle = decode_task.handle;
        decode_result.image = img;

        // 上锁，准备加入推理任务队列
        inference_lock.lock();
        inference_tasks.push(decode_result); // 加入推理队列
        inference_lock.unlock(); // 解锁

        // 计算并记录本次任务耗时
        auto t2 = std::chrono::steady_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        logger->debug("task {} runtime {}ms", decode_task.no, runtime.count());
    }

    // 根据是否定义PASS_BINARY宏释放解码缓冲区
#ifndef PASS_BINARY
    if (out_buf) {
        delete[] out_buf; // 释放内存
    }
#endif

    // 释放TurboJPEG解码器句柄
    tjDestroy(tj_handle);

    // 解码工作结束日志
    logger->info("stop");
}

#endif

void Manager::inference(int gpu_id) {
    // 为当前推理工作线程创建一个日志器，名称包含GPU ID
    char logger_name[64];
    sprintf(logger_name, "inference worker %d", gpu_id);
    auto logger = spdlog::stdout_color_st(logger_name);

    // 构建模型路径，包含GPU ID
    char model_path[256];
    sprintf(model_path, "%s/fabric_gpu%d.model", model_dir.c_str(), gpu_id);

    // 记录模型路径
    spdlog::info("model path: {}", model_path);

    // 创建Detector实例，用于执行推理，传入模型路径、GPU ID、配置文件和日志器
    Detector *detector = new Detector(model_path, gpu_id, config_file, logger);

    // 主循环，持续执行直到Manager的运行标志为false
    while (running) {
        // 获取推理任务的锁
        inference_lock.lock();

        // 如果当前没有任务，则解锁并短暂休眠
        if (inference_tasks.empty()) {
            inference_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } else {
            // 初始化批次处理相关变量
            int batch_size = 0;
            std::vector<cv::Mat> images;
            std::vector<RepTask> inference_results;

            // 收集一批推理任务，直到达到最大批次或队列为空
            while (!inference_tasks.empty() && batch_size < MAX_BATCH) {
                InfTask inference_task = inference_tasks.front(); // 取队首任务
                inference_tasks.pop();                         // 从队列移除任务
                batch_size++;                               // 增加批次计数
                images.push_back(inference_task.image);         // 收集图像
                RepTask inference_result;                    // 准备响应任务结构
                inference_result.no = inference_task.no;       // 设置任务编号
                inference_result.handle = inference_task.handle; // 设置句柄
                inference_results.push_back(inference_result);  // 添加到响应任务列表
            }

            // 释放锁，准备执行推理
            inference_lock.unlock();

            // 日志记录当前批次大小
            logger->debug("infer batch size {}", images.size());

            // 对每个任务进行日志记录
            for (int i = 0; i < inference_results.size(); i++) {
                logger->debug("task {}", inference_results[i].no);
            }

            // 记录推理开始时间
            auto t1 = std::chrono::steady_clock::now();

            // 执行模型推理
            std::vector<std::vector<Object>> objects;
            detector->process(images, objects);

            // 记录推理结束时间，计算并打印推理耗时
            auto t2 = std::chrono::steady_clock::now();
            auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            logger->debug("runtime {}ms", runtime.count());

            // 分配推理结果到每个响应任务，并加入到回复队列
            for (int i = 0; i < inference_results.size(); i++) {
                inference_results[i].results = objects[i];

                // 锁定回复任务队列
                reply_lock.lock();
                reply_tasks.push(inference_results[i]); // 加入回复队列
                reply_lock.unlock();                  // 解锁回复任务队列
            }
        }
    }

    // 清理Detector实例
    delete detector;

    // 记录线程停止信息
    logger->info("stop");
}

// Manager类的sendResult成员函数，负责发送检测结果至客户端
void Manager::sendResult() {
    // 初始化一个共享指针，指向控制台颜色输出的日志记录器，用于记录结果处理线程日志
    std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_st("result worker");

    // 当运行标志为真时，持续执行
    while (running) {
        // 初始化一个回复任务的向量，用于收集所有待发送的回复
        std::vector<RepTask> replies;

        // 上锁，以安全访问回复任务队列
        reply_lock.lock();
        // 当回复任务队列不为空时，依次取出所有任务加入到replies中
        while (!reply_tasks.empty()) {
            RepTask reply_task = reply_tasks.front();
            replies.push_back(reply_task);
            reply_tasks.pop(); // 移除已处理的任务
        }
        reply_lock.unlock(); // 解锁

        // 如果没有待处理的回复，短暂休眠后继续下一轮循环
        if (replies.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 休眠100毫秒
            continue; // 继续下一轮循环
        }

        // 记录开始处理时间
        auto t1 = std::chrono::steady_clock::now();

        // 初始化一个json对象的向量，用于存储所有结果
        std::vector<json> results;

        // 遍历回复任务，构建json数据
        for (int i = 0; i < replies.size(); i++) {
            json jresult; // 初始化一个json对象
            jresult["no"] = replies[i].no; // 添加任务编号

            // 构建缺陷检测结果部分
            std::vector<json> defects;
            for (int j = 0; j < replies[i].results.size(); j++) {
                Object obj = replies[i].results[j]; // 获取单个检测结果
                json obj_result; // 初始化单个缺陷的json对象
                obj_result["cls"] = class_names[obj.category]; // 添加类别
                obj_result["x"] = (int) obj.bbox.x; // 添加左上角X坐标
                obj_result["y"] = (int) obj.bbox.y; // 添加左上角Y坐标
                obj_result["w"] = (int) obj.bbox.width; // 添加宽度
                obj_result["h"] = (int) obj.bbox.height; // 添加高度
                obj_result["r"] = static_cast<int>(obj.confidence * 100.0f); // 添加置信度百分比
                defects.push_back(obj_result); // 将该缺陷结果添加到缺陷列表
            }

            // 将缺陷列表添加到当前任务的json对象中
            jresult["defects"] = defects;

            // 记录任务日志
            logger->info("task {} with {} defects", replies[i].no, replies[i].results.size());
            logger->info("{}", jresult.dump()); // 输出完整的json数据

            // 将处理好的json对象加入到结果列表
            results.push_back(jresult);
        }

        // 将所有结果合并成单一的json对象
        json reply = results;

        // 尝试通过WebSocket发送结果
        try {
            m_server.send(replies[0].handle, reply.dump(), websocketpp::frame::opcode::TEXT); // 发送json字符串到第一个回复任务的连接
        }
        catch (websocketpp::exception const &e) { // 捕获异常
            logger->error("send error: {}", e.what()); // 记录发送错误日志
        }

        // 计算并记录处理耗时
        auto t2 = std::chrono::steady_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        logger->debug("send {} results, runtime {}ms", replies.size(), runtime.count());

        // 休眠一段时间后继续下一轮
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 结束时记录日志
    logger->info("stop");
}