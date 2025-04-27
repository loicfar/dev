#pragma once

#include "velta/tool/tool_api.hpp"

#include <fstream>

namespace velta::tool
{
    class TOOL_API table
    {
    public:
        template<typename ...Args>
        class iterator
        {
        private:
            std::tuple<Args...> t_;

        public:
            template<std::size_t I>
            std::tuple_element<I, std::tuple<Args...>>::type& get()
            {
                return std::get<I>(t_);
            }

            template<std::size_t I>
            const std::tuple_element<I, std::tuple<Args...>>::type& get() const
            {
                return std::get<I>(t_);
            }
        };

    private:
        std::ifstream table_;
        char delim_;
        bool has_header_;

        std::size_t rows_;
        std::size_t cols_;

    public:
        table(const std::string& path, char delim = ',', bool has_header = true);

    private:
        template<typename T>
        static T convert(const std::string& s);

        template<typename T>
        bool read_tuple(std::istream& in, T& value) const;

        template<typename iterator_t, std::size_t... I>
        void read_tuple(std::istream& in, iterator_t& it, std::index_sequence<I...>) const
        {
            std::initializer_list<bool>({ this->read_tuple(in, it.template get<I>())... });
        }

    public:
        void reset();
        std::size_t rows() const;
        std::size_t cols() const;

        template<typename ...Args>
        bool read_line(iterator<Args...>& it)
        {
            std::string line;
            if (std::getline(this->table_, line))
            {
                std::istringstream iss(line);
                this->read_tuple(iss, it, std::make_index_sequence<sizeof...(Args)>{});
                return true;
            }
            return false;
        }

        template<typename T>
        bool read_line(std::vector<T>& v);
    };
}