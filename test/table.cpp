#include "velta/tool/table.hpp"

namespace velta::tool
{
    table::table(const std::string& path, char delim, bool has_header) :
        table_(path),
        delim_(delim),
        has_header_(has_header),
        rows_(0),
        cols_(0)
    {
        std::string dummy;
        while (std::getline(table_, dummy)) { ++rows_; }
        rows_ -= has_header_;

        reset();
    }

    template<>
    int table::convert(const std::string& s)
    {
        return std::stoi(s);
    }

    template<>
    double table::convert(const std::string& s)
    {
        return std::stod(s);
    }

    template<typename T>
    bool table::read_tuple(std::istream& in, T& value) const
    {
        std::string temp;
        std::getline(in, temp, this->delim_);
        value = convert<T>(temp);
        return true;
    }

    template<typename T>
    bool table::read_line(std::vector<T>& v)
    {
        v.clear();
        std::string line;
        if (std::getline(this->table_, line))
        {
            std::istringstream iss(line);
            std::string temp;
            while (std::getline(iss, temp, this->delim_))
            {
                v.push_back(convert<T>(temp));
            }
            return true;
        }
        return false;
    }

    void table::reset()
    {
        this->table_.clear();
        this->table_.seekg(0, std::ios::beg);

        std::string dummy;
        std::getline(table_, dummy);
        cols_ = 1 + std::count_if(dummy.begin(), dummy.end(), [this](char c) { return c == delim_; });

        if (!has_header_)
        {
            this->table_.clear();
            this->table_.seekg(0, std::ios::beg);
        }
    }

    std::size_t table::rows() const
    {
        return rows_;
    }

    std::size_t table::cols() const
    {
        return cols_;
    }

#define EXPORT_TABLE(T) template TOOL_API bool table::read_tuple<T>(std::istream& in, T& value) const;\
                        template TOOL_API bool table::read_line<T>(std::vector<T>& v)

    EXPORT_TABLE(int);
    EXPORT_TABLE(double);

#undef EXPORT_TABLE
}