#include <iostream>

#include "quadriga_lib.hpp"

// Simple code for parsing the command line arguments
char *getCmdOption(char **begin, char **end, const std::string &option)
{
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
        return *itr;
    return NULL;
}
bool cmdOptionExists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char **argv)
{

    // Input BIN file name
    std::string fn;
    if (cmdOptionExists(argv, argv + argc, "-i") && getCmdOption(argv, argv + argc, "-i") != NULL)
        fn = (std::string)getCmdOption(argv, argv + argc, "-i");
    else
    {
        std::cout << "QRT-File not given. Use: afl_qdant_read -i [QRT_FILE]" << std::endl;
        return 0;
    }

    // Read the file
    try
    {
        auto ant1 = quadriga_lib::qdant_read<float>(fn);
    }
    catch (const std::invalid_argument &ex)
    {
        std::cout << "ERROR: " <<  ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "ERROR: " <<  "Unknown failure occurred. Possible memory corruption!" << std::endl;
    }
}
