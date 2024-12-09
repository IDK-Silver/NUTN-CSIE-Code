#ifndef TOKEN_H
#define TOKEN_H

#include <string>
#include <vector>
#include <ostream>

class Token {
private:
    std::string value; // The actual token value

public:
    // Constructors
    Token() : value("") {}
    Token(const std::string& val) : value(val) {}

    // Getter for the token value
    std::string get_value() const {
        return value;
    }

    // Comparison operators to allow BST to organize tokens
    bool operator<(const Token& other) const {
        return value < other.value;
    }

    bool operator<=(const Token& other) const {
        return value <= other.value;
    }

    bool operator>(const Token& other) const {
        return value > other.value;
    }

    bool operator>=(const Token& other) const {
        return value >= other.value;
    }

    bool operator==(const Token& other) const {
        return value == other.value;
    }

    

    // Static method to parse a string and return a vector of Tokens
    static std::vector<Token> from_string(const std::string& str, char delimiter = ' ');
};

// Overload operator<< for Token (Inline Implementation)
inline std::ostream& operator<<(std::ostream& os, const Token& token) {
    os << token.get_value();
    return os;
}

#endif // TOKEN_H