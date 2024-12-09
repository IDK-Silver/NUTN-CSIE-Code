#include "token.hpp"
#include <sstream>

// Implementation of the from_string method
std::vector<Token> Token::from_string(const std::string& str, char delimiter) {
    std::vector<Token> tokens;
    std::stringstream ss(str);
    std::string temp;

    while (std::getline(ss, temp, delimiter)) {
        if (!temp.empty()) { // Avoid adding empty tokens
            tokens.emplace_back(Token(temp));
        }
    }

    return tokens;
}
