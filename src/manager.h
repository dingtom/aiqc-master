#ifndef _MANAGER_H_
#define _MANAGER_H_

#include "detector.h"
#include <queue>
#include <vector>
#include <set>
#include <string>
#include <thread>
#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <spdlog/spdlog.h>

typedef websocketpp::server<websocketpp::config::asio> server;

struct DecTask
{
    std::string no;
    websocketpp::connection_hdl handle;
#ifdef USE_PATH
    std::string image_path;
#else
    std::shared_ptr<char[]> image_data;
    uint32_t data_size;
#endif
};

struct InfTask
{
    std::string no;
    cv::Mat image;
    websocketpp::connection_hdl handle;
};

struct RepTask
{
    std::string no;
    std::vector<Object> results;
    websocketpp::connection_hdl handle;
};

class Manager
{
public:
    static void startService(int port, std::string config_file);
    static void stopService();

private:
    Manager();
    static Manager* getManager();
    static Manager* m_instance;

    void init(int port, std::string config_file);
    void run();
    void stop();

    void msgHandler(websocketpp::connection_hdl hdl, server::message_ptr msg);
    void decode(int id);
    void inference(int gpu_id);
    void sendResult();

    server m_server;
    int port;
    bool running;
    int num_gpus;
    std::set<int> use_gpus;
    std::string config_file;
    std::string version;
    std::char model_dir;

    std::vector<std::thread*> decode_workers;
    std::vector<std::thread*> inference_workers;
    std::thread* reply_worker;
    
    std::queue<DecTask> decode_tasks;
    std::mutex decode_lock;
    std::queue<InfTask> inference_tasks;
    std::mutex inference_lock;
    std::queue<RepTask> reply_tasks;
    std::mutex reply_lock;

    std::shared_ptr<spdlog::logger> websocket_logger;
    std::shared_ptr<spdlog::logger> manager_logger;

    std::vector<std::string> class_names;
};

#endif