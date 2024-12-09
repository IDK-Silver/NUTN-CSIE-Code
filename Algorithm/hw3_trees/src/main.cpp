#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include "trees/splay.hpp"
#include "trees/bst.hpp"
#include "trees/treap.hpp"
#include "trees/two_three.hpp"
#include "token/token.hpp"

std::vector<Token> read_tokens_from_file(const std::string &filename)
{
    std::ifstream infile(filename);
    std::vector<Token> tokens;
    std::string word;
    while (infile >> word)
    {
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        if (!word.empty())
        {
            tokens.emplace_back(Token(word));
        }
    }
    return tokens;
}

struct Argument
{
    std::string input_file;
    std::string output_file;
    std::string test_file;
    std::string target;
};

bool decode_argument(Argument &args, int argc, char *argv[])
{
    auto print_help = []() {

    };
    if (argc == 1)
    {
        // std::cerr << "錯誤：未提供任何參數\n"
        //           << "使用 --help 查看使用說明\n";
        // return false;
        return true;
    }

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            print_help();
        }

        if (i + 1 >= argc)
        {
            std::cerr << "錯誤：參數 " << arg << " 需要一個值\n"
                      << "使用 --help 查看使用說明\n";
            return false;
        }

        if (arg == "--input")
        {
            args.input_file = argv[++i];
        }
        else if (arg == "--output")
        {
            args.output_file = argv[++i];
        }
        else if (arg == "--test")
        {
            args.test_file = argv[++i];
        }
        else if (arg == "--target") {
            args.target = argv[++i];
        }
        else
        {
            std::cerr << "未知的參數: " << arg << "\n"
                      << "使用 --help 查看使用說明\n";
            return false;
        }
    }
    return true;
}

template <typename T>
void test_tree(
    T tree,
    const std::vector<Token> & input_tokens,
    const std::vector<Token> & test_tokens,
    const std::string tree_name,
    const std::string output_file,
    const std::string target
) 
{
    std::cout << "\nstart test " << tree_name << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &token : input_tokens)
    {
        tree.insert_node(token);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "insert cost-time : " << insert_duration << " µs.\n";

    size_t success_count = 0;
    size_t failed_count = 0;

    start = std::chrono::high_resolution_clock::now();
    for (const auto & token : test_tokens) {
        if (tree.search_node(token))
            success_count++;
        else
            failed_count++;
    }
    end = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "search cost-time : " << search_duration << " µs.\n";
    const auto acc = static_cast<double>(success_count) / (success_count + failed_count);
    std::cout << "success : " << success_count << "\t failed : " << failed_count << "\t"
              << "acc : " << acc << "\n\n";

    // Write results to CSV file
    std::ofstream file(output_file, std::ios::app);  // Open file in append mode
    if (file.is_open()) {
        file << target << ","
             << tree_name << ","
             << insert_duration << ","
             << search_duration << ","
             << success_count << ","
             << failed_count << ","
             << acc <<  ", "
             << input_tokens.size() << ", "
             << test_tokens.size() <<"\n";
        file.close();
    } else {
        std::cerr << "Unable to open file: " << output_file << std::endl;
    }
}

int main(int argc, char *argv[])
{

    Argument args;

    args.input_file = "in.in";
    args.test_file = "test.in";
    args.output_file = "output.csv";    
    args.target = "default";

    if (!decode_argument(args, argc, argv))
        return 1;

    TwoThreeTree<Token> two_three_tree;
    SplayTree<Token> splay_tree;
    BinarySearchTree<Token> bst;
    Treap<Token> treap;

    // 讀取建構字典的檔案
    std::vector<Token> input_tokens = read_tokens_from_file(
        args.input_file
    );

    std::vector<Token> test_tokens = read_tokens_from_file(
        args.test_file
    );
    
    std::ifstream in_file(args.output_file);
    if (!std::filesystem::exists(args.output_file)) {
        std::ofstream out_file(args.output_file);
            out_file << "Taraget,Tree Name,Insert Time (µs),Search Time (µs),Success Count,Failed Count,Accuracy, Input Count, Test Count\n";
            out_file.close();
    }

    if (in_file.is_open()) {
        std::string title;
        in_file >> title;
        if (title.empty()) {
            in_file.close();  
            std::ofstream out_file(args.output_file);
            out_file << "Target,Tree Name,Insert Time (µs),Search Time (µs),Success Count,Failed Count,Accuracy, Input Count, Test Count\n";
            out_file.close();
        }
        else{
            in_file.close();
        }
    }

    test_tree(
        bst, input_tokens, test_tokens, "BinarySearchTree", args.output_file, args.target
    );

    test_tree(
        splay_tree, input_tokens, test_tokens, "SplayTree", args.output_file, args.target
    );

    test_tree(
        treap, input_tokens, test_tokens, "Treap", args.output_file, args.target
    );

    test_tree(
        two_three_tree, input_tokens, test_tokens, "2-3 Tree", args.output_file, args.target
    );


    return 0;
}
