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

Manager *Manager::m_instance = NULL;

static const int NUM_DEC_WORKER = 4;
static const int MAX_BATCH = 8;

#ifndef PASS_BINARY
static const int DEC_BASE64_OUT_BUF_SIZE = 25000000;

static void decode_base64(char *src_buf, size_t src_size, char *out_buf, size_t *out_size) {
    base64_decode(src_buf, src_size, out_buf, out_size, 0);
    // printf("src: %d, out: %d\n", src_size, *out_size);
}

#endif

Manager::Manager() {}

Manager *Manager::getManager() {
    if (!m_instance) {
        m_instance = new Manager();
    }
    return m_instance;
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

void Manager::init(int port, std::string config_file) {
    m_server.clear_error_channels(websocketpp::log::elevel::all);
    m_server.set_error_channels(websocketpp::log::elevel::rerror);
    m_server.clear_access_channels(websocketpp::log::alevel::all);
    m_server.set_access_channels(websocketpp::log::alevel::access_core);

    m_server.set_message_handler(bind(&Manager::msgHandler, this, _1, _2));
    m_server.init_asio();
    m_server.set_max_message_size(2500 * 1024 * 1024);       // set single message size limit, unit=byte

    this->port = port;
    this->config_file = config_file;
    this->manager_logger = spdlog::stdout_color_st("manager");
    this->websocket_logger = spdlog::stdout_color_mt("websocket");

    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node defects = config["defects"];
    YAML::Node model_dir = config["model_dir"];
    model_dir = model_dir.as<std::char>();

    for (int i = 0; i < defects.size(); i++) {
        class_names.push_back(defects[i]["name"].as<std::string>());
    }
    YAML::Node visible_gpus = config["gpu"];
    for (int i = 0; i < visible_gpus.size(); i++) {
        use_gpus.insert(visible_gpus[i].as<int>());
    }

    std::ifstream ifs("version");
    if (ifs.is_open()) {
        std::getline(ifs, this->version);
    } else {
        this->version = "NA";
    }
    manager_logger->info("running fabric defect algorithm server, version: {}", this->version);
}

void Manager::run() {
    running = true;

    for (int i = 0; i < NUM_DEC_WORKER; i++) {
        std::thread *decode_worker = new std::thread(&Manager::decode, this, i);
        decode_workers.push_back(decode_worker);
    }

    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        manager_logger->error("no available gpus");
    } else {
        manager_logger->info("{} available gpu(s)", num_gpus);
    }
    for (int i = 0; i < num_gpus; i++) {
        if (use_gpus.find(i) != use_gpus.end()) {
            manager_logger->info("use gpu {}", i);
            std::thread *inference_worker = new std::thread(&Manager::inference, this, i);
            inference_workers.push_back(inference_worker);
        }
    }
    reply_worker = new std::thread(&Manager::sendResult, this);

    try {
        m_server.listen(port);
        m_server.start_accept();
        manager_logger->info("start to listen port {}", port);
        m_server.run();
    }
    catch (websocketpp::exception const &e) {
        manager_logger->error("start error: {}", e.what());
    }
}

void Manager::stop() {
    running = false;
    m_server.stop();
    m_server.stop_listening();
    manager_logger->info("stop to listen port {}", port);

    for (int i = 0; i < NUM_DEC_WORKER; i++) {
        if (decode_workers[i]) {
            decode_workers[i]->join();
            delete decode_workers[i];
        }
    }
    for (int i = 0; i < num_gpus; i++) {
        if (inference_workers[i]) {
            inference_workers[i]->join();
            delete inference_workers[i];
        }
    }
    if (reply_worker) {
        reply_worker->join();
        delete reply_worker;
    }

    manager_logger->info("all workers stopped, server exit");
}

void Manager::msgHandler(websocketpp::connection_hdl hdl, server::message_ptr msg) {
    auto con = m_server.get_con_from_hdl(hdl);
    std::string subpath = con->get_resource();
    websocket_logger->info("recv msg from {}", subpath);
    // deal with flaw command
    if (subpath == "/flaw") {
        if (msg->get_opcode() == websocketpp::frame::opcode::TEXT) {
            std::string str = msg->get_payload();
            websocket_logger->debug("recv text data {:.2f}KB", str.size() / 1024.0);
            try {
                json data = json::parse(str);
                for (auto s_data: data) {
                    std::string batch_no = s_data.at("no").get<std::string>();
                    std::string base64_image = s_data.at("image").get<std::string>();
                    websocket_logger->info("recv task {}, {:.2f}KB", batch_no, base64_image.size() / 1024.0);

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
            std::string str = msg->get_payload();
            websocket_logger->debug("recv binary data {:.2f}KB", str.size() / 1024.0);
#ifdef PASS_BINARY
            const char* data = str.c_str();
            size_t total_size = str.size();
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
                websocket_logger->info("recv task {}, {:.2f}KB", no, data_size/1024.0);

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
            std::string str = msg->get_payload();
            try {
                json data = json::parse(str);
                std::string cmd = data.at("command").get<std::string>();
                std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](unsigned char c) { return std::tolower(c); });
                websocket_logger->info("recv command: {}", cmd);
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

void Manager::decode(int id) {
    char logger_name[64];
    sprintf(logger_name, "decode worker %d", id);
    auto logger = spdlog::stdout_color_st(logger_name);

    tjhandle tj_handle = tjInitDecompress();
#ifndef PASS_BINARY
    char *out_buf = new char[DEC_BASE64_OUT_BUF_SIZE];
#endif

    while (running) {
        DecTask decode_task;
        decode_lock.lock();
        if (!decode_tasks.empty()) {
            decode_task = decode_tasks.front();
            decode_tasks.pop();
        } else {
            decode_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        decode_lock.unlock();

        auto t1 = std::chrono::steady_clock::now();
        InfTask decode_result;
        size_t src_size;
#ifndef PASS_BINARY
        decode_base64(decode_task.image_data.get(), decode_task.data_size, out_buf, &src_size);
#endif
        // auto tt = std::chrono::steady_clock::now();
        int subsamp, cs;
        int width, height;
#ifdef PASS_BINARY
        int ret = tjDecompressHeader3(tj_handle, (const unsigned char*)decode_task.image_data.get(), decode_task.data_size, &width, &height, &subsamp, &cs);
#else
        int ret = tjDecompressHeader3(tj_handle, (const unsigned char *) out_buf, src_size, &width, &height, &subsamp,
                                      &cs);
#endif
        if (ret != 0) {
            logger->error("decompress jpeg to rgb fail");
        }
        cv::Mat img(height, width, CV_8UC3);
#ifdef PASS_BINARY
        ret = tjDecompress2(tj_handle, (const unsigned char*)decode_task.image_data.get(), decode_task.data_size, img.data, width, width * 3, height, TJPF_BGR, TJFLAG_NOREALLOC);
#else
        ret = tjDecompress2(tj_handle, (const unsigned char *) out_buf, src_size, img.data, width, width * 3, height,
                            TJPF_BGR, TJFLAG_NOREALLOC);
#endif
        if (ret != 0) {
            logger->error("decompress jpeg to rgb fail");
        }
        decode_result.no = decode_task.no;
        decode_result.handle = decode_task.handle;
        decode_result.image = img;
        inference_lock.lock();
        inference_tasks.push(decode_result);
        inference_lock.unlock();
        auto t2 = std::chrono::steady_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tt-t1).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-tt).count() << "\n";
        logger->debug("task {} runtime {}ms", decode_task.no, runtime.count());
    }

#ifndef PASS_BINARY
    if (out_buf) {
        delete[] out_buf;
    }
#endif
    tjDestroy(tj_handle);
    logger->info("stop");
}

#endif

void Manager::inference(int gpu_id) {
    char logger_name[64];
    sprintf(logger_name, "inference worker %d", gpu_id);
    auto logger = spdlog::stdout_color_st(logger_name);

    char model_path[256];
    sprintf(model_path, "%s/fabric_gpu%d.model", MODEL_DIR, gpu_id);
    Detector *detector = new Detector(model_path, gpu_id, config_file, logger);
    while (running) {
        inference_lock.lock();
        if (inference_tasks.empty()) {
            inference_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } else {
            int batch_size = 0;
            std::vector<cv::Mat> images;
            std::vector<RepTask> inference_results;
            while (!inference_tasks.empty() && batch_size < MAX_BATCH) {
                InfTask inference_task = inference_tasks.front();
                inference_tasks.pop();
                batch_size++;
                images.push_back(inference_task.image);
                RepTask inference_result;
                inference_result.no = inference_task.no;
                inference_result.handle = inference_task.handle;
                inference_results.push_back(inference_result);
            }
            inference_lock.unlock();

            logger->debug("infer batch size {}", images.size());
            for (int i = 0; i < inference_results.size(); i++) {
                logger->debug("task {}", inference_results[i].no);
            }
            auto t1 = std::chrono::steady_clock::now();
            std::vector<std::vector<Object>> objects;
            detector->process(images, objects);
            auto t2 = std::chrono::steady_clock::now();
            auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            logger->debug("runtime {}ms", runtime.count());
            // fflush(stdout);

            for (int i = 0; i < inference_results.size(); i++) {
                inference_results[i].results = objects[i];
                reply_lock.lock();
                reply_tasks.push(inference_results[i]);
                reply_lock.unlock();
            }
        }
    }

    delete detector;
    logger->info("stop");
}

void Manager::sendResult() {
    std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_st("result worker");

    while (running) {
        std::vector<RepTask> replies;
        reply_lock.lock();
        while (!reply_tasks.empty()) {
            RepTask reply_task = reply_tasks.front();
            replies.push_back(reply_task);
            reply_tasks.pop();
        }
        reply_lock.unlock();

        if (replies.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto t1 = std::chrono::steady_clock::now();
        std::vector<json> results;
        for (int i = 0; i < replies.size(); i++) {
            json jresult;
            jresult["no"] = replies[i].no;
            std::vector<json> defects;
            for (int j = 0; j < replies[i].results.size(); j++) {
                Object obj = replies[i].results[j];
                json obj_result;
                obj_result["cls"] = class_names[obj.category];
                obj_result["x"] = (int) obj.bbox.x;
                obj_result["y"] = (int) obj.bbox.y;
                obj_result["w"] = (int) obj.bbox.width;
                obj_result["h"] = (int) obj.bbox.height;
                defects.push_back(obj_result);
            }
            jresult["defects"] = defects;
            logger->info("task {} with {} defects", replies[i].no, replies[i].results.size());
            logger->info("{}", jresult.dump());
            results.push_back(jresult);
        }
        json reply = results;
        try {
            m_server.send(replies[0].handle, reply.dump(), websocketpp::frame::opcode::TEXT);
        }
        catch (websocketpp::exception const &e) {
            logger->error("send error: {}", e.what());
        }
        auto t2 = std::chrono::steady_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        logger->debug("send {} results, runtime {}ms", replies.size(), runtime.count());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    logger->info("stop");
}