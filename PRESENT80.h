#pragma once
#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
//#include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
//#include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "Helper.h"

using namespace std;

class PRESENT_80_CORE{

    private:
    __m128i RoundKeys[31];
    static const uint8_t PRESENT80_SBOX[16];  // S-Box for the cypher
    static const __m128i RC[32]; // an array that holds round constants for al 32 rounds
    __m128i shiftBytesInsideBlock(__m128i block, present80_internal::rotationType mode); // helper function that'll help with the rotation

    __m128i rotateLeftBy61Bits(__m128i block); // The traditional way to shift 61 bits to the left
    __m128i rotateLeftBy61Bits_SIMD_ONLY(__m128i block); // The SIMD approach to shify 61 bits to the left by only right shifts
 
    void expandRoundKeys(__m128i key, uint8_t Round);
    //uint8_t applySBox(uint8_t x); 
    __m128i sBoxLayer(__m128i state);
    __m128i pLayer(__m128i state);   // conventional(scalar) approach to implement p layer
    __m128i pLayer_using_SIMD_Only(__m128i state); //SIMD approach to permtate the p-layer. not tested yet. Some correction needed
    
    
    public:
    PRESENT_80_CORE(__m128i key);
    __m128i encrypt64(uint64_t  block);
};
