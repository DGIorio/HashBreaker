#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <limits>

// --- Compile-Time Limit ---
// Max *TOTAL* string length the compiled code can handle in buffers.
constexpr int MAX_TOTAL_LEN_COMPILE_LIMIT = 80;
constexpr int MAX_VARIABLE_LEN_COMPILE_LIMIT = 14;


// --- Default Configuration Values ---
// These are used if the config file doesn't provide a value.
const std::string DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_012";
constexpr int DEFAULT_MAX_VARIABLE_LEN = 8;
constexpr int DEFAULT_MAX_RESULTS_BUFFER = 4096;
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;
constexpr int DEFAULT_PREFERRED_BLOCKS = 8192;
constexpr int DEFAULT_START_STRING_LEN = 1;
const std::string DEFAULT_OUTPUT_FILENAME = "found_crc_matches.txt";
const std::string DEFAULT_PREFIX = "";
const std::string DEFAULT_SUFFIX = "";
const std::string DEFAULT_CONTAINS = "";

// Struct to hold loaded configuration
struct ConfigData {
    std::string charset = DEFAULT_CHARSET;
    int max_results_buffer = DEFAULT_MAX_RESULTS_BUFFER;
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    int preferred_blocks = DEFAULT_PREFERRED_BLOCKS;
    std::string output_filename = DEFAULT_OUTPUT_FILENAME;
    std::string prefix = DEFAULT_PREFIX;
    std::string suffix = DEFAULT_SUFFIX;
    std::string contains = DEFAULT_CONTAINS;
    int start_string_len = DEFAULT_START_STRING_LEN;
    int max_variable_len = DEFAULT_MAX_VARIABLE_LEN;
    // Derived values
    unsigned long long strings_per_launch = 0;
    int prefix_len = 0;
    int suffix_len = 0;
    int contains_len = 0;
};


// Struct to hold results
struct FoundMatch {
    uint32_t target_hash;
    // Using the TOTAL compile-time limit for safety
    char found_string[MAX_TOTAL_LEN_COMPILE_LIMIT + 1];
};

// Helper macro for CUDA error checking
#define CHECK_CUDA_ERROR(call)                                          \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        cudaDeviceReset();                                              \
        throw std::runtime_error(cudaGetErrorString(err));              \
    }                                                                   \
} while (0)


// --- Device Code ---

// Constant memory for CRC table
__constant__ uint32_t CRC32_TABLE_DEVICE[256] = {
    0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B, 0x1A864DB2, 0x1E475005,
    0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61, 0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
    0x4C11DB70, 0x48D0C6C7, 0x4593E01E, 0x4152FDA9, 0x5F15ADAC, 0x5BD4B01B, 0x569796C2, 0x52568B75,
    0x6A1936C8, 0x6ED82B7F, 0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3, 0x709F7B7A, 0x745E66CD,
    0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039, 0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5,
    0xBE2B5B58, 0xBAEA46EF, 0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033, 0xA4AD16EA, 0xA06C0B5D,
    0xD4326D90, 0xD0F37027, 0xDDB056FE, 0xD9714B49, 0xC7361B4C, 0xC3F706FB, 0xCEB42022, 0xCA753D95,
    0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1, 0xE13EF6F4, 0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D,
    0x34867077, 0x30476DC0, 0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5, 0x2AC12072,
    0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16, 0x018AEB13, 0x054BF6A4, 0x0808D07D, 0x0CC9CDCA,
    0x7897AB07, 0x7C56B6B0, 0x71159069, 0x75D48DDE, 0x6B93DDDB, 0x6F52C06C, 0x6211E6B5, 0x66D0FB02,
    0x5E9F46BF, 0x5A5E5B08, 0x571D7DD1, 0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
    0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B, 0xBB60ADFC, 0xB6238B25, 0xB2E29692,
    0x8AAD2B2F, 0x8E6C3698, 0x832F1041, 0x87EE0DF6, 0x99A95DF3, 0x9D684044, 0x902B669D, 0x94EA7B2A,
    0xE0B41DE7, 0xE4750050, 0xE9362689, 0xEDF73B3E, 0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2,
    0xC6BCF05F, 0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34, 0xDC3ABDED, 0xD8FBA05A,
    0x690CE0EE, 0x6DCDFD59, 0x608EDB80, 0x644FC637, 0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB,
    0x4F040D56, 0x4BC510E1, 0x46863638, 0x42472B8F, 0x5C007B8A, 0x58C1663D, 0x558240E4, 0x51435D53,
    0x251D3B9E, 0x21DC2629, 0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5, 0x3F9B762C, 0x3B5A6B9B,
    0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF, 0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623,
    0xF12F560E, 0xF5EE4BB9, 0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65, 0xEBA91BBC, 0xEF68060B,
    0xD727BBB6, 0xD3E6A601, 0xDEA580D8, 0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD, 0xCDA1F604, 0xC960EBB3,
    0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7, 0xAE3AFBA2, 0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B,
    0x9B3660C6, 0x9FF77D71, 0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74, 0x857130C3,
    0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640, 0x4E8EE645, 0x4A4FFBF2, 0x470CDD2B, 0x43CDC09C,
    0x7B827D21, 0x7F436096, 0x7200464F, 0x76C15BF8, 0x68860BFD, 0x6C47164A, 0x61043093, 0x65C52D24,
    0x119B4BE9, 0x155A565E, 0x18197087, 0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
    0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D, 0x2056CD3A, 0x2D15EBE3, 0x29D4F654,
    0xC5A92679, 0xC1683BCE, 0xCC2B1D17, 0xC8EA00A0, 0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB, 0xDBEE767C,
    0xE3A1CBC1, 0xE760D676, 0xEA23F0AF, 0xEEE2ED18, 0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4,
    0x89B8FD09, 0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662, 0x933EB0BB, 0x97FFAD0C,
    0xAFB010B1, 0xAB710D06, 0xA6322BDF, 0xA2F33668, 0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4
};

// Device function for CRC32 calculation
__device__ uint32_t calculate_custom_crc32(const char* data, int length) {
    if (length <= 0) { return 0; }
    uint32_t result = 0;
    if (length < 4) {
        result = 0xFFFFFFFF;
        for (int i = 0; i < length; ++i) {
            uint8_t index = (result ^ static_cast<uint8_t>(data[i])) & 0xFF;
            result = (result >> 8) ^ CRC32_TABLE_DEVICE[index];
        }
        result = ~result; // Final XOR
    }
    else {
        uint32_t initial_value = (static_cast<uint32_t>(static_cast<uint8_t>(data[0])) << 24) |
            (static_cast<uint32_t>(static_cast<uint8_t>(data[1])) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(data[2])) << 8) |
            static_cast<uint32_t>(static_cast<uint8_t>(data[3]));
        result = ~initial_value; // Initialize with NOT of first 4 bytes
        for (int i = 4; i < length; ++i) {
            result = ((result << 8) | static_cast<uint32_t>(static_cast<uint8_t>(data[i]))) ^ CRC32_TABLE_DEVICE[(result >> 24) & 0xFF];
        }
        result = ~result; // Apply final NOT
    }
    return result & 0xFFFFFFFF; // Mask
}

// Device-side string search
__device__ const char* dev_strstr(const char* haystack, int haystack_len, const char* needle, int needle_len) {
    if (!needle || needle_len <= 0) return haystack;
    if (!haystack || haystack_len < needle_len) return nullptr;

    for (int i = 0; i <= haystack_len - needle_len; ++i) {
        int j = 0;
        volatile const char* h_ptr = haystack + i;
        volatile const char* n_ptr = needle;
        while (j < needle_len && h_ptr[j] == n_ptr[j]) {
            j++;
        }
        if (j == needle_len) {
            return haystack + i; // Found
        }
    }
    return nullptr; // Not found
}

// CUDA Kernel (uses compile-time limit for arrays)
__global__ void crack_multi_crc32_kernel(
    // Target Hashes
    const uint32_t* d_target_hashes, int num_targets,
    // Charset
    const char* d_charset, int charset_len,
    // Current Test Length & Index
    int string_len, unsigned long long start_index,
    // Constraints
    const char* d_prefix, int prefix_len,
    const char* d_suffix, int suffix_len,
    const char* d_contains, int contains_len,
    // Results
    FoundMatch* d_results, int* d_result_count, int max_results)
{
    // --- Calculate Variable Part Length ---
    int variable_len = string_len - prefix_len - suffix_len;
    // If constraints make this length impossible, exit thread
    if (variable_len < 0) {
        return;
    }

    // Calculate global thread ID relative to the variable part's search space
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long current_variable_index = start_index + tid;


    // Candidate buffer sized by TOTAL compile-time limit
    char candidate[MAX_TOTAL_LEN_COMPILE_LIMIT + 1];

    // --- Construct Candidate String with Constraints ---
    // 1. Copy Prefix
    for (int i = 0; i < prefix_len; ++i) {
        candidate[i] = d_prefix[i];
    }

    // 2. Generate Variable Part based on index
    unsigned long long temp_index = current_variable_index;
    bool index_overflow = false; // Check if index is too large for this variable_len

    // Handle case where variable_len is 0 (string is just prefix + suffix)
    if (variable_len == 0) {
        if (current_variable_index > 0) { // Only index 0 is valid if variable_len is 0
            index_overflow = true;
        }
    }
    else {
        // Generate variable part chars from right to left
        for (int i = variable_len - 1; i >= 0; --i) {
            if (charset_len == 0) {
                index_overflow = true;
                break;
            }
            int char_idx = temp_index % charset_len;
            candidate[prefix_len + i] = d_charset[char_idx];
            temp_index /= charset_len;
        }
        // If temp_index is not 0, the original current_variable_index was too large
        if (temp_index != 0) {
            index_overflow = true;
        }
    }

    // If the index was out of bounds for the variable part, stop.
    if (index_overflow) {
        return;
    }

    // 3. Copy Suffix
    for (int i = 0; i < suffix_len; ++i) {
        candidate[prefix_len + variable_len + i] = d_suffix[i];
    }

    // 4. Null Terminate at the correct string_len
    candidate[string_len] = '\0';

    // --- Filter for CONTAINS constraint ---
    // Maybe it would be faster to ignore this check, and do it later in the results
    if (contains_len > 0) {
        if (dev_strstr(candidate, string_len, d_contains, contains_len) == nullptr) {
            return; // Does not contain the required substring, exit thread
        }
    }

    // --- CRC Calculation & Check (If all constraints passed) ---
    uint32_t calculated_crc = calculate_custom_crc32(candidate, string_len);

    // Check against every target hash (same as before)
    for (int j = 0; j < num_targets; ++j) {
        if (calculated_crc == d_target_hashes[j]) {
            int result_idx = atomicAdd(d_result_count, 1);
            if (result_idx < max_results) {
                d_results[result_idx].target_hash = d_target_hashes[j];
                // Copy the generated candidate string (up to string_len)
                for (int k = 0; k < string_len; ++k) {
                    d_results[result_idx].found_string[k] = candidate[k];
                }
                d_results[result_idx].found_string[string_len] = '\0';
            }
        }
    }
}


// --- Host Code ---

// Helper to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

// Function to load configuration from file
bool loadConfig(const std::string& filename, ConfigData& config) {
    std::ifstream cFile(filename);
    if (!cFile.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << std::endl;
        std::cerr << "Using default configuration values." << std::endl;
        config.prefix_len = config.prefix.length();
        config.suffix_len = config.suffix.length();
        config.contains_len = config.contains.length();
        config.strings_per_launch = (unsigned long long)config.threads_per_block * config.preferred_blocks;
        return true; // Prints a warning but allow proceeding with defaults
    }

    std::string line;
    int lineNum = 0;
    std::cout << "Loading configuration from: " << filename << std::endl;
    while (std::getline(cFile, line)) {
        lineNum++;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }

        std::size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            std::cerr << "Warning: Invalid line format in config (missing '=') on line " << lineNum << ": \"" << line << "\". Skipping." << std::endl;
            continue;
        }

        std::string key = trim(line.substr(0, eqPos));
        std::string value = trim(line.substr(eqPos + 1));

        try {
            if (key == "CHARSET") {
                if (!value.empty()) config.charset = value;
                else std::cerr << "Warning: Empty value for CHARSET on line " << lineNum << ". Using default." << std::endl;
            }
            else if (key == "START_STRING_LEN") {
                int val = std::stoi(value);
                config.start_string_len = (val >= 1) ? val : DEFAULT_START_STRING_LEN;
                if (config.start_string_len != val) std::cerr << "Warning: Invalid START_STRING_LEN (" << val << ") on line " << lineNum << ". Using default/minimum (" << config.start_string_len << ")." << std::endl;
            }
            else if (key == "MAX_VARIABLE_LEN") {
                //int val = std::stoi(value);
                //config.max_variable_len = (val >= 0) ? val : DEFAULT_MAX_VARIABLE_LEN;
                //if (config.max_variable_len != val) std::cerr << "Warning: Invalid MAX_VARIABLE_LEN (" << val << ") on line " << lineNum << ". Using default/minimum (0)." << std::endl;

                int val = std::stoi(value);
                if (val > 0 && val <= MAX_VARIABLE_LEN_COMPILE_LIMIT) {
                    config.max_variable_len = val;
                }
                else if (val > MAX_VARIABLE_LEN_COMPILE_LIMIT) {
                    std::cerr << "Warning: MAX_VARIABLE_LEN (" << val << ") in config exceeds compile limit ("
                        << MAX_VARIABLE_LEN_COMPILE_LIMIT << "). Using compile limit." << std::endl;
                    config.max_variable_len = MAX_VARIABLE_LEN_COMPILE_LIMIT;
                }
                else {
                    std::cerr << "Warning: Invalid value for MAX_VARIABLE_LEN (" << val << ") on line " << lineNum << ". Using default (" << config.max_variable_len << ")." << std::endl;
                }
            }
            else if (key == "MAX_RESULTS_BUFFER") {
                int val = std::stoi(value);
                config.max_results_buffer = (val > 0) ? val : DEFAULT_MAX_RESULTS_BUFFER;
            }
            else if (key == "THREADS_PER_BLOCK") {
                int val = std::stoi(value);
                config.threads_per_block = (val > 0 && (val & (val - 1)) == 0 && val <= 1024) ? val : DEFAULT_THREADS_PER_BLOCK; // Basic power-of-2 check up to 1024
                if (config.threads_per_block != val) std::cerr << "Warning: Invalid THREADS_PER_BLOCK on line " << lineNum << ". Using default (" << config.threads_per_block << ")." << std::endl;
            }
            else if (key == "PREFERRED_BLOCKS") {
                int val = std::stoi(value);
                config.preferred_blocks = (val > 0) ? val : DEFAULT_PREFERRED_BLOCKS;
            }
            else if (key == "OUTPUT_FILENAME") {
                if (!value.empty()) config.output_filename = value;
                else std::cerr << "Warning: Empty value for OUTPUT_FILENAME on line " << lineNum << ". Using default." << std::endl;
            }
            else if (key == "PREFIX") {
                config.prefix = value;
            }
            else if (key == "SUFFIX") {
                config.suffix = value;
            }
            else if (key == "CONTAINS") {
                config.contains = value;
            }
            else {
                std::cerr << "Warning: Unknown configuration key on line " << lineNum << ": \"" << key << "\". Ignoring." << std::endl;
            }
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid numeric value format on line " << lineNum << " for key '" << key << "': \"" << value << "\". Skipping." << std::endl;
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Warning: Numeric value out of range on line " << lineNum << " for key '" << key << "': \"" << value << "\". Skipping." << std::endl;
        }
    }
    cFile.close();

    // Store lengths and calculate derived values
    config.prefix_len = config.prefix.length();
    config.suffix_len = config.suffix.length();
    config.contains_len = config.contains.length();
    config.strings_per_launch = (unsigned long long)config.threads_per_block * config.preferred_blocks;

    // --- Constraint Validation ---
    bool config_ok = true;
    int max_possible_total_len = config.prefix_len + config.suffix_len + config.max_variable_len;
    if (max_possible_total_len > MAX_TOTAL_LEN_COMPILE_LIMIT) {
        std::cerr << "Error: The maximum possible total string length calculated from\n"
            << "       Prefix (" << config.prefix_len << ") + Suffix (" << config.suffix_len
            << ") + Max Variable (" << config.max_variable_len << ") = " << max_possible_total_len << " characters\n"
            << "       exceeds the program's compiled buffer limit of "
            << MAX_TOTAL_LEN_COMPILE_LIMIT << " characters." << std::endl;
        std::cerr << "Solution: Shorten the Prefix, Suffix, or MAX_VARIABLE_LEN in the config file,\n"
            << "          OR recompile the program with a larger MAX_TOTAL_LEN_COMPILE_LIMIT value." << std::endl;
        std::cerr << "=======================================================================" << std::endl;
        config_ok = false;
    }

    // Validate start length against max possible length
    if (config.start_string_len > max_possible_total_len && config_ok) { // Only warn if other checks passed
        std::cerr << "\nWarning: START_STRING_LEN (" << config.start_string_len
            << ") is greater than the maximum possible total length (" << max_possible_total_len
            << ") allowed by constraints and MAX_VARIABLE_LEN. No strings will be tested." << std::endl;
    }
    // Validate start length against minimum possible length
    int min_possible_len = config.prefix_len + config.suffix_len;
    if (config.start_string_len < min_possible_len) {
        std::cerr << "\nWarning: START_STRING_LEN (" << config.start_string_len
            << ") is less than the minimum possible length (" << min_possible_len
            << ") required by Prefix/Suffix. Will start testing from length " << min_possible_len << "." << std::endl;
        // Adjusting the start later in main, this is just a warning
    }

    std::cout << "Configuration loaded." << (config_ok ? "" : " (With Errors!)") << std::endl;

    return config_ok;
}

// Function to read hashes from file
bool readHashesFromFile(const std::string& filename, std::vector<uint32_t>& targetHashes) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open hash file: " << filename << std::endl;
        return false;
    }
    std::string line;
    int lineNum = 0;
    while (std::getline(infile, line)) {
        lineNum++;
        line = trim(line); // Use trim helper
        if (line.empty() || line[0] == '#') { continue; }
        try {
            size_t processed = 0;
            unsigned long long val = std::stoull(line, &processed, 16);
            if (processed != line.length() || val > UINT32_MAX) {
                std::cerr << "Warning: Invalid hash format or out of range on line " << lineNum << ": \"" << line << "\". Skipping." << std::endl;
                continue;
            }
            targetHashes.push_back(static_cast<uint32_t>(val));
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid hash format on line " << lineNum << ": \"" << line << "\". Skipping." << std::endl;
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Warning: Hash out of range on line " << lineNum << ": \"" << line << "\". Skipping." << std::endl;
        }
    }
    infile.close();
    return !targetHashes.empty();
}

// Function to write results to file
bool writeResultsToFile(const std::string& filename, const std::map<uint32_t,
    std::vector<std::string>>&results, const ConfigData& config, std::chrono::milliseconds duration,
    const char* gpuName) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file for writing: " << filename << std::endl;
        return false;
    }

    outfile << "# CUDA CRC32 Cracker Results" << std::endl;
    outfile << "#----------------------------------------" << std::endl;
    outfile << "# Configuration Used:" << std::endl;
    outfile << "#   Charset: \"" << config.charset << "\"" << std::endl;
    outfile << "#   Start Total Length Tested: " << config.start_string_len << std::endl;
    outfile << "#   Max Variable Length Tested: " << config.max_variable_len << std::endl;
    outfile << "#   Compiled Total Length Buffer Limit: " << MAX_TOTAL_LEN_COMPILE_LIMIT << std::endl;
    outfile << "#   Prefix: \"" << config.prefix << "\"" << std::endl;
    outfile << "#   Suffix: \"" << config.suffix << "\"" << std::endl;
    outfile << "#   Contains: \"" << config.contains << "\"" << std::endl;
    outfile << "#   GPU Used: " << gpuName << std::endl;
    outfile << "#   Threads/Block: " << config.threads_per_block << std::endl;
    outfile << "#   Preferred Blocks/Launch: " << config.preferred_blocks << std::endl;
    outfile << "#----------------------------------------" << std::endl;
    outfile << "# Execution Time: " << std::fixed << std::setprecision(3) << duration.count() / 1000.0 << " seconds" << std::endl;
    outfile << "#----------------------------------------" << std::endl << std::endl;

    bool anyResults = false;
    for (const auto& pair : results) {
        uint32_t hash = pair.first;
        const auto& strings = pair.second;
        if (!strings.empty()) {
            anyResults = true;
            outfile << "Hash: 0x" << std::hex << std::setw(8) << std::setfill('0') << hash << std::dec << std::endl;
            for (const std::string& s : strings) {
                outfile << "  -> \"" << s << "\"" << std::endl;
            }
            outfile << std::endl;
        }
    }

    if (!anyResults) {
        outfile << "# No matches found for the provided hashes within the tested limits." << std::endl;
    }

    outfile.close();
    std::cout << "Results successfully written to: " << filename << std::endl;
    return true;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config.ini> <hash_file.txt>" << std::endl;
        return 1;
    }
    std::string config_filename = argv[1];
    std::string hash_filename = argv[2];

    ConfigData config;
    std::vector<uint32_t> h_target_hashes;
    std::map<uint32_t, std::vector<std::string>> all_found_matches;

    // --- Load Config ---
    if (!loadConfig(config_filename, config)) {
        std::cerr << "Configuration loading failed or constraints invalid. Exiting." << std::endl;
        return 1; // Exit if config validation failed
    }

    // --- Read Hashes ---
    std::cout << "\nReading target hashes from: " << hash_filename << std::endl;
    if (!readHashesFromFile(hash_filename, h_target_hashes)) {
        std::cerr << "Error reading hashes or file is empty/invalid." << std::endl;
        return 1;
    }
    int num_targets = static_cast<int>(h_target_hashes.size());
    std::cout << "Successfully read " << num_targets << " target hashes." << std::endl;
    for (uint32_t hash : h_target_hashes) {
        all_found_matches[hash] = std::vector<std::string>(); // Initialize map entries
    }


    try {
        std::cout << "\n--- Effective Configuration ---" << std::endl;
        std::cout << "Charset: \"" << config.charset << "\" (Length: " << config.charset.length() << ")" << std::endl;
        std::cout << "Start Total String Length: " << config.start_string_len << std::endl;
        std::cout << "Max Variable Length: " << config.max_variable_len << std::endl;
        std::cout << "Compiled Total Length Buffer Limit: " << MAX_TOTAL_LEN_COMPILE_LIMIT << std::endl;
        std::cout << "Prefix Constraint: \"" << config.prefix << "\"" << std::endl;
        std::cout << "Suffix Constraint: \"" << config.suffix << "\"" << std::endl;
        std::cout << "Contains Constraint: \"" << config.contains << "\"" << std::endl;
        std::cout << "GPU Result Buffer Size: " << config.max_results_buffer << std::endl;
        std::cout << "Threads/Block: " << config.threads_per_block << std::endl;
        std::cout << "Preferred Blocks/Launch: " << config.preferred_blocks << std::endl;
        std::cout << "Target strings/launch: " << config.strings_per_launch << std::endl;
        std::cout << "Output File: " << config.output_filename << std::endl;
        std::cout << "-----------------------------" << std::endl;


        // --- GPU setup ---
        int deviceId;
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceId));
        std::cout << "Using GPU: " << deviceProp.name << std::endl;
        if (deviceProp.major < 3) {
            std::cerr << "Warning: GPU Compute Capability might be too low (" << deviceProp.major << "." << deviceProp.minor << "). Requires >= 3.0 for atomicAdd." << std::endl;
        }


        // --- Device Memory Allocation (using config values) ---
        char* d_charset = nullptr;
        uint32_t* d_target_hashes = nullptr;
        FoundMatch* d_results = nullptr;
        int* d_result_count = nullptr;
        char* d_prefix = nullptr;
        char* d_suffix = nullptr;
        char* d_contains = nullptr;

        int charset_len = config.charset.length();

        CHECK_CUDA_ERROR(cudaMalloc(&d_charset, charset_len * sizeof(char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_target_hashes, num_targets * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_results, config.max_results_buffer * sizeof(FoundMatch)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_result_count, sizeof(int)));
        // Allocate for constraints (even if empty, allocate minimally to avoid null pointers)
        CHECK_CUDA_ERROR(cudaMalloc(&d_prefix, (config.prefix_len + 1) * sizeof(char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_suffix, (config.suffix_len + 1) * sizeof(char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_contains, (config.contains_len + 1) * sizeof(char)));


        // --- Host Memory for Results Batch ---
        std::vector<FoundMatch> h_results_batch(config.max_results_buffer);
        int h_total_processed_results = 0;


        // --- Data Transfers ---
        CHECK_CUDA_ERROR(cudaMemcpy(d_charset, config.charset.c_str(), charset_len * sizeof(char), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_target_hashes, h_target_hashes.data(), num_targets * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_result_count, 0, sizeof(int)));
        // Copy constraints (including null terminator for safety, though kernel uses length)
        CHECK_CUDA_ERROR(cudaMemcpy(d_prefix, config.prefix.c_str(), (config.prefix_len + 1) * sizeof(char), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_suffix, config.suffix.c_str(), (config.suffix_len + 1) * sizeof(char), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_contains, config.contains.c_str(), (config.contains_len + 1) * sizeof(char), cudaMemcpyHostToDevice));


        // --- Main Cracking Loop ---
        auto start_time = std::chrono::high_resolution_clock::now();
        bool buffer_overflow_warning = false;
        int min_possible_total_len = config.prefix_len + config.suffix_len; // Min length due to constraints

        std::cout << "\nStarting tests for variable lengths 0 to " << config.max_variable_len << "." << std::endl;
        if (config.start_string_len > min_possible_total_len) {
            std::cout << "(Will skip iterations until TOTAL length reaches " << config.start_string_len << ")" << std::endl;
        }

        // Loop through possible variable lengths
        for (int current_variable_len = 0; current_variable_len <= config.max_variable_len; ++current_variable_len) {
            // Calculate the total string length for this iteration
            int current_total_len = config.prefix_len + config.suffix_len + current_variable_len;

            if (current_total_len < config.start_string_len) {
                continue; // Skip this variable length, total length too short
            }

            // Check if total length exceeds compile limit (should be caught by loadConfig)
            if (current_total_len > MAX_TOTAL_LEN_COMPILE_LIMIT) {
                std::cerr << "\nInternal Error: Calculated total length " << current_total_len
                    << " exceeds compile limit " << MAX_TOTAL_LEN_COMPILE_LIMIT << ". Stopping." << std::endl;
                break; // Stop processing
            }

            std::cout << "\nTesting variable length: " << current_variable_len << " (Total length: " << current_total_len << ")" << std::endl;


            // Calculate total combinations for the VARIABLE part
            unsigned long long total_variable_combinations = 1;
            bool overflow_detected = false;
            if (current_variable_len > 0) { // Only calculate if there's a variable part
                unsigned long long check_val = 1;
                for (int i = 0; i < current_variable_len; ++i) {
                    if (charset_len > 0 && ULLONG_MAX / charset_len < check_val) {
                        total_variable_combinations = ULLONG_MAX;
                        overflow_detected = true;
                        break;
                    }
                    // Handle charset_len == 0
                    if (charset_len == 0) {
                        total_variable_combinations = 0; // No combinations possible
                        break;
                    }
                    check_val *= charset_len;
                }
                if (!overflow_detected && total_variable_combinations != 0) {
                    total_variable_combinations = check_val;
                }
            }
            else {
                total_variable_combinations = 1; // Only 1 combination if variable_len is 0
            }


            if (overflow_detected) {
                std::cout << "  Search space for length " << current_variable_len << " is extremely large." << std::endl;
            }
            else {
                std::cout << "  Variable combinations: " << total_variable_combinations << std::endl;
            }
            if (total_variable_combinations == 0 && current_variable_len > 0) {
                std::cout << "  Skipping variable length " << current_variable_len << " due to zero combinations (empty charset?)." << std::endl;
                continue;
            }


            // Batch loop
            unsigned long long current_start_index = 0;
            unsigned long long variable_strings_processed = 0;
            int h_result_count_device = 0;

            // Loop while index is valid for variable combinations
            while (current_start_index < total_variable_combinations || overflow_detected) {

                // Reset device counter if it's full
                if (h_result_count_device >= config.max_results_buffer) {
                    if (!buffer_overflow_warning) {
                        std::cerr << "\nWarning: GPU results buffer filled. Subsequent matches in this/prev batch might be missed until buffer cleared." << std::endl;
                        buffer_overflow_warning = true;
                    }
                    CHECK_CUDA_ERROR(cudaMemset(d_result_count, 0, sizeof(int)));
                    h_total_processed_results += h_result_count_device;
                    h_result_count_device = 0;
                }

                // Determine grid size based on variable combinations
                unsigned long long strings_in_this_launch = config.strings_per_launch;
                if (!overflow_detected && current_start_index + strings_in_this_launch > total_variable_combinations) {
                    strings_in_this_launch = total_variable_combinations - current_start_index;
                }
                if (strings_in_this_launch == 0) break; // Should only happen if total_variable_combinations is 0 initially


                dim3 threadsPerBlock(config.threads_per_block);
                dim3 numBlocks((unsigned int)ceil((double)strings_in_this_launch / config.threads_per_block));

                // --- Kernel Launch (pass constraint pointers/lengths) ---
                crack_multi_crc32_kernel << <numBlocks, threadsPerBlock >> > (
                    d_target_hashes, num_targets,
                    d_charset, charset_len,
                    current_total_len,
                    current_start_index,
                    d_prefix, config.prefix_len,
                    d_suffix, config.suffix_len,
                    d_contains, config.contains_len,
                    d_results, d_result_count, config.max_results_buffer
                    );
                CHECK_CUDA_ERROR(cudaGetLastError());

                variable_strings_processed += strings_in_this_launch;

                // Check for Results (Periodically)
                bool check_now = (variable_strings_processed % (config.strings_per_launch * 10) == 0) ||
                    (!overflow_detected && current_start_index + strings_in_this_launch >= total_variable_combinations) ||
                    (overflow_detected && variable_strings_processed % (config.strings_per_launch * 10) == 0);

                if (check_now) {
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_count_device, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));

                    int new_results_count = h_result_count_device;
                    if (new_results_count > 0) {
                        buffer_overflow_warning = false;
                        CHECK_CUDA_ERROR(cudaMemcpy(h_results_batch.data(), d_results, new_results_count * sizeof(FoundMatch), cudaMemcpyDeviceToHost));
                        for (int i = 0; i < new_results_count; ++i) {
                            all_found_matches[h_results_batch[i].target_hash].push_back(h_results_batch[i].found_string);
                        }
                        CHECK_CUDA_ERROR(cudaMemset(d_result_count, 0, sizeof(int)));
                        h_total_processed_results += new_results_count;
                        h_result_count_device = 0;
                        //printf("\n[+] Found %d new matches. Total found so far: %d\n", new_results_count, h_total_processed_results);
                    }
                }

                // Progress Indicator
                if (!overflow_detected) {
                    double percent = (total_variable_combinations > 0) ? (double)variable_strings_processed / total_variable_combinations * 100.0 : 100.0;
                    printf("  Progress: %.2f%% (%llu / %llu) | Found: %d \r", percent, variable_strings_processed, total_variable_combinations, h_total_processed_results);
                }
                else {
                    printf("  Processed: %llu strings... | Found: %d \r", variable_strings_processed, h_total_processed_results);
                }
                fflush(stdout);

                // Update start index
                if (!overflow_detected) {
                    current_start_index += strings_in_this_launch;
                }
                else {
                    current_start_index += strings_in_this_launch;
                }
            } // End batch loop

            printf("\n");

            // Final check for results at the end of each length
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_result_count_device, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_result_count_device > 0) {
                buffer_overflow_warning = false;
                CHECK_CUDA_ERROR(cudaMemcpy(h_results_batch.data(), d_results, h_result_count_device * sizeof(FoundMatch), cudaMemcpyDeviceToHost));
                for (int i = 0; i < h_result_count_device; ++i) {
                    all_found_matches[h_results_batch[i].target_hash].push_back(h_results_batch[i].found_string);
                }
                CHECK_CUDA_ERROR(cudaMemset(d_result_count, 0, sizeof(int)));
                h_total_processed_results += h_result_count_device;
                //printf("[+] Found %d final matches for length %d. Total found so far: %d\n", h_result_count_device, current_len, h_total_processed_results);
                h_result_count_device = 0;
            }

        } // End length loop


        // --- Final Summary & Write Results ---
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Cracking Complete!" << std::endl;
        std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Total matches found across all targets: " << h_total_processed_results << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        writeResultsToFile(config.output_filename, all_found_matches, config, duration, deviceProp.name);


        // --- Cleanup ---
        std::cout << "\nCleaning up CUDA resources..." << std::endl;
        if (d_charset) cudaFree(d_charset);
        if (d_target_hashes) cudaFree(d_target_hashes);
        if (d_results) cudaFree(d_results);
        if (d_result_count) cudaFree(d_result_count);
        if (d_prefix) cudaFree(d_prefix);
        if (d_suffix) cudaFree(d_suffix);
        if (d_contains) cudaFree(d_contains);
        CHECK_CUDA_ERROR(cudaDeviceReset());

    }
    catch (const std::exception& e) {
        std::cerr << "\nRuntime Error: " << e.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }
    catch (...) {
        std::cerr << "\nAn unknown error occurred." << std::endl;
        cudaDeviceReset();
        return 1;
    }

    return 0;
}