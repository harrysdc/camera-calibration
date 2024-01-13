#include "zhangMethod.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "usage: ./zhangMethod [xxx.yaml]" << std::endl;
        exit(1);
    }

    zhangMethod zm;
    zm.open_yaml(argv[1]);
    zm.findCameraPosition();

    return 0;
}