#pragma once
#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
//#include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "PRESENT80.h" 
#include "Helper.h"

using namespace std; 

int main(){

    uint8_t key[16] = {
                    0xFF, 0xFF, 0xFF, 0xFF, 
                    0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0x00, 0x00,     
                    0x00, 0x00, 0x00, 0x00                                        
                };
 
    __m128i Key = _mm_loadu_si128((__m128i*)key);
//    
//    // Step 1: rotate right by 2 bytes / 16 bits
//    const __m128i mask = _mm_set_epi8(
//        1, 0, 15, 14, 13, 12,
//        11, 10, 9, 8, 7, 6,
//        5, 4, 3, 2
//    );
//    __m128i x = _mm_shuffle_epi8(Key, mask);
//
//    // Step 2: rotate right 3 bits
//    __m128i lo = _mm_srli_epi64(x, 3);
//    __m128i hi = _mm_slli_epi64(x, 61);
//    __m128i rotated = _mm_or_si128(lo, hi);
//
//    // Step 3: keep only 80 bits
//    const __m128i keep80 = _mm_set_epi8(
//        0,0,0,0,0,0,
//        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
//    );
//
//    __m128i rotated_left_by_61_bis = _mm_and_si128(rotated, keep80);
    
    PRESENT_80_CORE Cypher(Key);
    

return 0;

}