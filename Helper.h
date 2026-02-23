#pragma once
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <emmintrin.h>

namespace present80_internal
{

    using namespace std;

    enum class rotationType
    {
        None,
        Rotate_Left_2_Bytes,
    };

    static inline __m128i getRotationMask(rotationType kind) // functon that will return the kind of needed for the current operation
    {
        switch (kind)
        {


        case rotationType::Rotate_Left_2_Bytes:
            return _mm_set_epi8(
                1, 0, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6,
                5, 4, 3, 2);

        default:
            return _mm_set_epi8(
                11, 7, 2, 6,
                1, 12, 8, 3,
                14, 13, 9, 4,
                15, 10, 5, 0);
        }
    }

    static inline void print_m128i(const std::string &label, __m128i v)
    {
        alignas(16) uint8_t b[16];
        _mm_store_si128(reinterpret_cast<__m128i *>(b), v);

        std::cout << label << "\n";
        std::cout << "Chunk | Bytes        | Bit positions  | Hex values  | Bits\n";
        std::cout << "------+--------------+----------------+-------------+-------------------------\n";

        for (int chunk = 3; chunk >= 0; chunk--)
        {
            int start = chunk * 4;
            int end = start + 3;

            // Print chunk index
            std::cout << "  " << chunk << "   |  ";

            // Print byte indices
            std::cout << std::setw(2) << end << " "
                      << std::setw(2) << end - 1 << " "
                      << std::setw(2) << end - 2 << " "
                      << std::setw(2) << end - 3 << " |   ";

            // Print bit ranges
            int bit_hi = (end + 1) * 8 - 1;
            int bit_lo = start * 8;
            std::cout << std::setw(3) << bit_hi << " .. "
                      << std::setw(3) << bit_lo << "   | ";

            // Print hex values
            for (int i = end; i >= start; i--)
            {
                std::cout << std::hex << std::uppercase
                          << std::setw(2) << std::setfill('0') << (int)b[i] << " ";
            }
            std::cout << std::dec << std::setfill(' ') << "| ";

            // Print bits for each byte
            for (int i = end; i >= start; i--)
            {
                for (int bit = 7; bit >= 0; bit--)
                {
                    std::cout << ((b[i] >> bit) & 1);
                }
                std::cout << " ";
            }

            std::cout << "\n";
        }

        std::cout << "\n";
    }

}