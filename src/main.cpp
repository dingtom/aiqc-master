#include "manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

void stop_server(int signum) {
    spdlog::info("received stop msg");
    Manager::stopService();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s [port] [config-file]\n", argv[0]);
        return -1;
    }
    int port = atoi(argv[1]);
    char *config_file = argv[2];
    try
    {
        YAML::Node config = YAML::LoadFile(config_file);
        int log_level = config["log_level"].as<int>();
        spdlog::set_level(spdlog::level::level_enum(log_level));
        signal(SIGINT, stop_server);

        Manager::startService(port, config_file);
        return 0;

    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to load config file: {}", e.what());
        return -1;
    }
}