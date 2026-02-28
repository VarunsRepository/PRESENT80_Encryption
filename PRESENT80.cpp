#pragma once

#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
// #include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "PRESENT80.h"
#include "Helper.h"
#include <emmintrin.h>   // exposes _mm_cvtsi128_si64 on MSVC
#include <intrin.h>
#include <smmintrin.h>   // SSE4.1

// to force the compiler to use the 64 bit version compiler 
#if !defined(_M_X64) && !defined(__x86_64__)
#error "Listen mate, this needs a 64-bit compiler! Switch to x64." // 
#endif


using namespace std;

const uint8_t PRESENT_80_CORE::PRESENT80_SBOX[16] = {0xC, 0x5, 0x6, 0xB,
                                                     0x9, 0x0, 0xA, 0xD,
                                                     0x3, 0xE, 0xF, 0x8,
                                                     0x4, 0x7, 0x1, 0x2};

const __m128i SIMD_S_BOX = _mm_set_epi8(  //SIMD-freidnly version S Box that'll be used to substitution Layer.
        0x02, 0x01, 0x07, 0x04,           // it contains the same S_box as above but in reverse order. 
        0x08, 0x0F, 0x0E, 0x03,           // The first byte in the rght hand side constant is the value at 0xF positon in the S-box
        0x0D, 0x0A, 0x00, 0x09,           // and the last is the byte at the first position in the above S-box
        0x0B, 0x06, 0x05, 0x0C
    );

const __m128i PRESENT_80_CORE::RC[32] =
    {
        _mm_set_epi64x(0, 0x00000000), // round 0 (unused)
        _mm_set_epi64x(0, 0x00008000), // round 1
        _mm_set_epi64x(0, 0x00010000), // round 2
        _mm_set_epi64x(0, 0x00018000), // round 3
        _mm_set_epi64x(0, 0x00020000), // round 4
        _mm_set_epi64x(0, 0x00028000), // round 5
        _mm_set_epi64x(0, 0x00030000), // round 6
        _mm_set_epi64x(0, 0x00038000), // round 7
        _mm_set_epi64x(0, 0x00040000), // round 8
        _mm_set_epi64x(0, 0x00048000), // round 9
        _mm_set_epi64x(0, 0x00050000), // round 10
        _mm_set_epi64x(0, 0x00058000), // round 11
        _mm_set_epi64x(0, 0x00060000), // round 12
        _mm_set_epi64x(0, 0x00068000), // round 13
        _mm_set_epi64x(0, 0x00070000), // round 14
        _mm_set_epi64x(0, 0x00078000), // round 15
        _mm_set_epi64x(0, 0x00080000), // round 16
        _mm_set_epi64x(0, 0x00088000), // round 17
        _mm_set_epi64x(0, 0x00090000), // round 18
        _mm_set_epi64x(0, 0x00098000), // round 19
        _mm_set_epi64x(0, 0x000A0000), // round 20
        _mm_set_epi64x(0, 0x000A8000), // round 21
        _mm_set_epi64x(0, 0x000B0000), // round 22
        _mm_set_epi64x(0, 0x000B8000), // round 23
        _mm_set_epi64x(0, 0x000C0000), // round 24
        _mm_set_epi64x(0, 0x000C8000), // round 25
        _mm_set_epi64x(0, 0x000D0000), // round 26
        _mm_set_epi64x(0, 0x000D8000), // round 27
        _mm_set_epi64x(0, 0x000E0000), // round 28
        _mm_set_epi64x(0, 0x000E8000), // round 29
        _mm_set_epi64x(0, 0x000F0000), // round 30
        _mm_set_epi64x(0, 0x000F8000)  // round 31
};

PRESENT_80_CORE::PRESENT_80_CORE(__m128i key)
{
    RoundKeys[0] = key;
    // cout << "~~~~~~~~~~~~~~~~~~~~ Round 1 ~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    // expandRoundKeys(RoundKeys[0], 0); 

     for(int    round=1; round<32; ++round){
         cout << "#### ROUND NUMBER : " << round <<  " #### " <<endl;
         expandRoundKeys(RoundKeys[round-1], round);
         present80_internal::print_m128i("Roundkey ", RoundKeys[round]);
     }
}


// functiont that will perform the rotation inside the SIMD register
__m128i PRESENT_80_CORE::shiftBytesInsideBlock(__m128i block, present80_internal::rotationType mode)
{
    __m128i mask = getRotationMask(mode);
    return _mm_shuffle_epi8(block, mask);
}

// This function rotates the key left by 61 bits using the traditional way 
// 1. Extracting bytes into a local array
// 2. Reconstructing the 80‑bit key
// 3. The rotation math
// 4. Packing back into bytes
// 5. Returning the block
__m128i PRESENT_80_CORE::rotateLeftBy61Bits(__m128i block)
{
    // Step 1: Extract bytes 0-9 (80 bits)
    alignas(16) uint8_t bytes[16];
    _mm_store_si128((__m128i*)bytes, block);

    // Step 2: Combine 80-bit key into 64-bit + 16-bit parts
    uint64_t hi = 0; // bits 79..16
    uint16_t lo = 0; // bits 15..0

    // hi = bytes[0..7] (MSB first)
    for (int i = 0; i < 8; ++i) {
        hi <<= 8;
        hi |= bytes[i];
    }

    // lo = bytes[8..9]
    lo = (bytes[8] << 8) | bytes[9];

    // Step 3: Perform 80-bit rotate left by 61
    // Formula: newKey = (key << 61) | (key >> 19)
    uint64_t new_hi = ((hi << 61) | ((uint64_t)lo << 61 >> 64) | (hi >> 19));
    uint16_t new_lo = ((hi << 61) >> 64) | (lo >> 3);

    // Step 4: Pack back into bytes
    for (int i = 7; i >= 0; --i) {
        bytes[i] = new_hi & 0xFF;
        new_hi >>= 8;
    }
    bytes[8] = (new_lo >> 8) & 0xFF;
    bytes[9] = new_lo & 0xFF;

    // Zero upper bytes
    for (int i = 10; i < 16; ++i)
        bytes[i] = 0;

    // Step 5: Load back into __m128i
    return _mm_load_si128((__m128i*)bytes);
}

// This function performs the 61 bit left shit in the 80 bit word but using SIMD intrinsics thus exploting the CPU's 
// Vectorised instructions(SIMD) to get things done faster and easier. Here's the algorithm 
// 1. Lane Extraction for the Lower Bits
// 2. Right‑shift by 19
// 3. Masking the Lower Contribution
// 4. Packing back into bytes
// 5. Returning the block
// 4. Shuffle to Align the Lower Bits
// 5. Upper Lanes Extraction
// 6. Right‑shift by 3
// 7. Masking the Upper Contribution
// 8. Shuffle to Align the Upper Bits
// 9. Merge the 2 parts
__m128i PRESENT_80_CORE::rotateLeftBy61Bits_SIMD_ONLY(__m128i block)
{
    //load lanes 2, 1, 1, 0 from the original key into a 128 bit block
    __m128i LowerLanesRearrangement = _mm_shuffle_epi32(block, _MM_SHUFFLE(2, 1, 1, 0));
//    present80_internal::print_m128i("LowerLanesRearrangement with lanes 2,1,1,0 from orignal key", LowerLanesRearrangement);

    // shift right by 19 bits, this way you'll get the bits 31-0 of the lower lane ready for the final output 
    // you'll also get bits 61-32 of the upper lane reader for the final output 
    LowerLanesRearrangement = _mm_srli_epi64(LowerLanesRearrangement, 19);
//    present80_internal::print_m128i("LowerLanesRearrangement >> 19", LowerLanesRearrangement);

    // creating a mask to remove the bits that we don't need
    // create a mask   [
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   1F FF FF FF | 00011111 11111111 11111111 11111111
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   FF FF FF FF | 11111111 11111111 11111111 11111111
    //                ]
    __m128i maskLower = _mm_set_epi32(0x00,
                                      0x1FFFFFFF,
                                      0x00,
                                      0xFFFFFFFF);
//    present80_internal::print_m128i("LowerLanesRearrangement mask", maskLower);

    // LowerLanesRearrangement AND Mask - to remove the unwanted bits 
    LowerLanesRearrangement = _mm_and_si128(LowerLanesRearrangement, maskLower);
//    present80_internal::print_m128i("LowerLanesRearrangement AND mask", LowerLanesRearrangement);

    //rearranging get the bits 31-0 and bits 61-32 in the right bit postions for the final output
    // this is the LSB part of left rotated 61-bit word  
    LowerLanesRearrangement = _mm_shuffle_epi32(LowerLanesRearrangement, _MM_SHUFFLE(1, 1, 2, 0));
//    present80_internal::print_m128i("LowerLanesRearrangement shuffled with the lanes 1,1,2,0 to get the LSBs aligned", LowerLanesRearrangement);

    //loading lane 0 accross all the 4 lanes in the 128 bit SIMD register.
    // These lower bits from the original key will occupy the MSB in after left rotation 
    __m128i UpperLanesRearrangement = _mm_shuffle_epi32(block, _MM_SHUFFLE(0, 0, 0, 0));
//    present80_internal::print_m128i("UpperLanesRearrangement with lanes 0,0,0,0 from the original key", UpperLanesRearrangement);

    //rotate right by 3 bits.
    // if you recall this mask from above :
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   1F FF FF FF | 00011111 11111111 11111111 11111111
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   FF FF FF FF | 11111111 11111111 11111111 11111111
    // you'll notice that lane 2(1F FF FF FF | 00011111 11111111 11111111 11111111) has 000 at its MSBs.
    // after rotating-right the 32 LSBs of the original key by 3 bits, those 3 LSBs will fall into these positons and will compelte the key 
    UpperLanesRearrangement = _mm_srli_epi64(UpperLanesRearrangement, 3);
//    present80_internal::print_m128i("UpperLanesRearrangement >> 3", UpperLanesRearrangement);
    // instrincs for :  create a mask
    //                       [
    //                         00 00 FF FF | 00000000 00000000 11111111 11111111
    //                         E0 00 00 00 | 11100000 00000000 00000000 00000000
    //                         00 00 00 00 | 00000000 00000000 00000000 00000000
    //                         00 00 00 00 | 00000000 00000000 00000000 00000000
    //                       ]
    //  Mask for UpperLanes [0, 0, 0x0000FFFF, 0x00000000]
    //  We mask to keep only the relevant moved bits.
    __m128i maskHigher = _mm_set_epi32(0x0000FFFF,
                                       0xE0000000,
                                       0x00,
                                       0x00);
//    present80_internal::print_m128i("UpperLanesRearrangement mask", maskHigher);

    // intrnsic for UpperLanesRearrangement = UpperLanesRearrangement AND Mask to get the required bits and zero-ing the rest
    UpperLanesRearrangement = _mm_and_si128(UpperLanesRearrangement, maskHigher);
//    present80_internal::print_m128i("UpperLanesRearrangement AND mask", UpperLanesRearrangement);

    //shurffling the lanes so that the final orde aligns with the 80 bit output  
    UpperLanesRearrangement = _mm_shuffle_epi32(UpperLanesRearrangement, _MM_SHUFFLE(0, 3, 2, 0));
//    present80_internal::print_m128i("UpperLanesRearrangement shuffled with the lanes 0,3,2,0 to get the MSBs aligned", UpperLanesRearrangement);

    // We use OR to merge the bits into the final 80-bit state
    // instrincs for result = pperLanesRearrangement AND LowerLanesRearrangement
    __m128i result = _mm_or_si128(UpperLanesRearrangement, LowerLanesRearrangement);
//    present80_internal::print_m128i("UpperLanesRearrangement OR  LowerLanesRearrangement", result);

    return result;
}

void PRESENT_80_CORE::expandRoundKeys(__m128i key, uint8_t Round)
{
    present80_internal::print_m128i("Original Key", key);

    key = rotateLeftBy61Bits_SIMD_ONLY(key);
    present80_internal::print_m128i("rotated_left_by_61_bis", key);

    uint8_t topByte = _mm_extract_epi8(key, 9);          // extracting the top byte from the rotated key
    cout << "topByte = 0x" << setw(2) << setfill('0') << hex << (int)topByte << dec << endl;

    uint8_t upperNibble = topByte >> 4;                  // right shft the top byte by 4 bits to get the higher nible 
    cout << "upperNibble = 0x" << setw(2) << setfill('0') << hex << (int)upperNibble << dec << endl;

    uint8_t upperNibbleSubstitution  = PRESENT80_SBOX[upperNibble]; // get the S-box value at the nibble
    cout << "upperNibbleSubstitution = 0x" << setw(2) << setfill('0') << hex << (int)upperNibbleSubstitution << dec << endl;

   // cout << "SBox substitution for " <<hex << upperNibble <<" = " <<upperNibbleSubstitution << dec << endl;  
    topByte = (topByte & 0x0F) | (upperNibbleSubstitution << 4);    // make space inside the upper nibble  from the key as that we can insert the substituted value back
    cout << "upperNibble reinsertion into top byte = 0x" << setw(2) << setfill('0') << hex << (int)topByte << dec << endl;

    //cout << "Top Byte = " <<hex << topByte <<dec <<endl;
    key = _mm_insert_epi8(key, topByte, 9);              // reinsertt the byte into the key
    present80_internal::print_m128i("Substituted Top Nibble back into the key", key);

//    uint64_t rc = (Round & 0x1F);   // generating the 5-bit round constant
//    uint64_t rc_shifted = rc << 15; // move into bit positions 19..15

    present80_internal::print_m128i("round Constant", RC[Round]); // XORing the round key with the round constant
    // XOR into lane0 (bits 19..15)
    key = _mm_xor_si128(key, RC[Round]);
    present80_internal::print_m128i("Final RoundKey : After XOR-ing with the round Constant", key);

    // adding the roundkey to the roundKeys  array
    RoundKeys[Round] = key;

    // code to test the S box Substituton separately
    //__m128i testData = _mm_set_epi64x(0x000000000000C000, 0x00000000FF00FF00); 
//
    //uint8_t TB = _mm_extract_epi8(testData, 9);           // extracting the top byte from the rotated key
    //cout << "topByte = 0x" << setw(2) << setfill('0') << hex << (int)TB << dec << endl;
    //
    //uint8_t Up_Nib = TB >> 4 ;
    //                // right shft the top byte by 4 bits to get the higher nible 
    //cout << "Up_Nib = 0x" << setw(2) << setfill('0') << hex << (int)Up_Nib << dec << endl;
    //
    //uint8_t S_BOX_index = (int)Up_Nib;
    //cout << "S_BOX_index = " << S_BOX_index << endl;
//
    //uint8_t S_BOX_VAL  = PRESENT80_SBOX[S_BOX_index]; 
    //cout << "S_BOX_VAL = 0x" << setw(2) << setfill('0') << hex << (int)S_BOX_VAL << dec << endl;
    //
    //TB = (TB & 0x0F) | (Up_Nib << 4);
    //cout << "topByte = 0x" << setw(2) << setfill('0') << hex << (int)TB << dec << endl;
 
}

// Applies the PRESENT S‑box to all 16 nibbles of the 64‑bit state.
// 
// The state is stored in the lower 64 bits of a 128‑bit SIMD register.
// Each byte contains two nibbles: the high nibble (bits 7–4) and the low nibble (bits 3–0).
//
// The algorithmic idea:
//   1. Extract all high nibbles into one SIMD register.
//   2. Extract all low nibbles into another SIMD register.
//   3. Use PSHUFB with the SIMD_S_BOX lookup table to substitute all 16 nibbles in parallel.
//   4. Recombine the substituted high and low nibbles back into bytes.
//
// This leverages the fact that PSHUFB performs 16 parallel 4‑bit table lookups
// when the input values are in the range 0–15.
__m128i PRESENT_80_CORE::sBoxLayer(__m128i state)
{
    // Extract high nibbles: shift right by 4, then mask to keep only 0x0–0xF.
    __m128i high = _mm_and_si128(_mm_srli_epi16(state, 4), _mm_set1_epi8(0x0F));

    // Extract low nibbles: mask directly.
    __m128i low  = _mm_and_si128(state, _mm_set1_epi8(0x0F));

    // Substitute both nibble streams using the SIMD S‑box lookup table.
    __m128i low_s  = _mm_shuffle_epi8(SIMD_S_BOX, low);
    __m128i high_s = _mm_shuffle_epi8(SIMD_S_BOX, high);

    //merging the upper and lower nibbles
    __m128i finalStateafterSBOxing = _mm_or_si128(_mm_slli_epi16(high_s, 4), low_s); 
    
    // removign the top 8 bytes from the result as they contain no data
    finalStateafterSBOxing = _mm_and_si128(finalStateafterSBOxing,
                                       _mm_set_epi64x(0, -1));


    // Recombine substituted nibbles: high nibble << 4 | low nibble.
    return finalStateafterSBOxing;
}

 

__m128i PRESENT_80_CORE::pLayer(__m128i state)
{
    // Extract low 64 bits via store (no _mm_cvtsi128_si64x needed)
    uint64_t x = 0;
    
    state = _mm_and_si128(state, _mm_set_epi64x(0, -1));
    
    present80_internal::print_m128i("state after ignoring the top 8 bytes", state);

    _mm_storel_epi64(reinterpret_cast<__m128i*>(&x), state);
     present80_internal::print_m128i("state after reinterpret cast ", state);

     cout << "Original bytes are :  "  << hex << x << dec << endl;
    // Scalar PRESENT pLayer on the 64-bit block
    uint64_t y = 0;

    for (uint32_t i = 0; i < 63; ++i) {
        uint64_t bit = (x >> i) & 1ULL;
        uint32_t j   = (16u * i) % 63u;
        y |= (bit << j);
    }
     cout << "Permutated bytes are : " << hex << y << dec << endl;
    // Bit 63 stays in place
    y |= (x & (1ULL << 63));
     cout << "y |= (x & (1ULL << 63));(filling the Last bit(bit[63])) : " << hex << y << dec << endl;
    
//    __m128i finalState = _mm_set_epi64x( 0x0000000000000000ULL, y); //strange behavoir of __m128 it,
//    if you put zero in the first lane, it will take full the entire register with zeroes, 
//    but if you put a non-zero value int the first lane, it starts to work again.

    // to overcome the above behavoir, we are loading out data in  upper word and zeroes in lower word
    __m128i finalState = _mm_set_epi64x( y, 0x0000000000000000);    
    finalState = _mm_shuffle_epi32(finalState, _MM_SHUFFLE(1,0,3,2)); //shuffling to get the actual word in lower lane
    
     return finalState;
}
 
__m128i PRESENT_80_CORE:: pLayer_using_SIMD_Only(__m128i state)
{
    const __m128i mask1  = _mm_set1_epi64x(0x5555555555555555ULL);
    const __m128i mask2  = _mm_set1_epi64x(0x3333333333333333ULL);
    const __m128i mask4  = _mm_set1_epi64x(0x0F0F0F0F0F0F0F0FULL);
    const __m128i mask8  = _mm_set1_epi64x(0x00FF00FF00FF00FFULL);
    const __m128i mask16 = _mm_set1_epi64x(0x0000FFFF0000FFFFULL);
    const __m128i mask32 = _mm_set1_epi64x(0x00000000FFFFFFFFULL);

    __m128i x = state;

    // stage 1: swap adjacent bits
    __m128i t = _mm_and_si128(x, mask1);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 1),
            _mm_and_si128(_mm_srli_epi64(x, 1), mask1)
        );

    // stage 2: swap bit pairs
    t = _mm_and_si128(x, mask2);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 2),
            _mm_and_si128(_mm_srli_epi64(x, 2), mask2)
        );

    // stage 3: swap nibbles
    t = _mm_and_si128(x, mask4);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 4),
            _mm_and_si128(_mm_srli_epi64(x, 4), mask4)
        );

    // stage 4: swap bytes
    t = _mm_and_si128(x, mask8);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 8),
            _mm_and_si128(_mm_srli_epi64(x, 8), mask8)
        );

    // stage 5: swap 16-bit words
    t = _mm_and_si128(x, mask16);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 16),
            _mm_and_si128(_mm_srli_epi64(x, 16), mask16)
        );

    // stage 6: swap 32-bit halves
    t = _mm_and_si128(x, mask32);
    x = _mm_or_si128(
            _mm_slli_epi64(t, 32),
            _mm_and_si128(_mm_srli_epi64(x, 32), mask32)
        );

    return x;
}

 
// This layer performs the bit permutations in the block.
// Some scalar operations were used and this layer did not entirely use SIMD intrinsics
// I am stil working out to rotate the bits isndie the 64 bit-block using the vectorised instructions
// there seems to be an issue with the Microsoft visual C++ compiler.
// I throws an error during compile time, complaing that if does not have the  followinf instrincisc

// 1. _mm_cvtsi128_si64 / _mm_cvtsi128_si64x
//   Purpose: Extract the low 64 bits of a __m128i into a uint64_t.
//   Problem: MSVC does not expose _mm_cvtsi128_si64x unless compiling for x64 with SSE4.1+.
//   On some configurations, the intrinsic is simply not defined, causing a compile‑time error.

// 2. _mm_extract_epi64
// Purpose: Extract a 64‑bit lane from a __m128i.
 
// 3. _mm_insert_epi64
// Purpose: Insert a 64‑bit integer into a specific lane of a __m128i.
// Problem:  SSE4.1‑only. Not available in the MSVC
// 4. _mm_blend_epi16, _mm_blend_epi32, _mm_blend_epi64
// Purpose: Blend lanes or sub‑lanes of SIMD registers.
// Problem: All require SSE4.1.
// Not available in your build.
// Impact:  We could not use blending to assemble the pLayer output. 
// if these intrinsics were availble, we could  have done the p-layer in full hardware earily
 

__m128i PRESENT_80_CORE::encrypt64(uint64_t block)
{   
        int round=1 ;
        __m128i state = _mm_set_epi64x(0, block);
//        cout << "~~~~~~~~~~~~~~ Round : " << round << " ~~~~~~~~~~~~~~"   <<endl;
//        // aligning the key in the lower bytes of the SIMD register so as we can XOR it with the state or data directy
//        RoundKeys[round] = shiftBytesInsideBlock(RoundKeys[round],present80_internal::rotationType::alignKeyForEncryption);
//        present80_internal::print_m128i("roundKey is", RoundKeys[round]);
//
// //       __m128i state = _mm_cvtsi64x_si128(block); //does not compile cause Microsoft's C++ compiler has some plan of its own
//        __m128i state = _mm_set_epi64x(0, block);
//        present80_internal::print_m128i("state is", state);
//
//        state = _mm_xor_si128(RoundKeys[round], state); //pre-whitening or XOR-ing the key with the input  
//        present80_internal::print_m128i("State after roundkey Whitening is", state);
//
//        state = sBoxLayer(state); //Substituting all the 16 bytes from the S-Box one-by-one  
//        present80_internal::print_m128i("State S-Box Substitution is", state);
//
//        state = pLayer(state); //permutating the bits in the register   
//        present80_internal::print_m128i("permutated the bits in the register", state);

 
        for (round = 1; round<32; round=round+1 ){
        cout << "~~~~~~~~~~~~~~ Encryption Round : " << round << " ~~~~~~~~~~~~~~"   <<endl;
        // aligning the key in the lower bytes of the SIMD register so as we can XOR it with the state or data directy
        RoundKeys[round - 1] = shiftBytesInsideBlock(RoundKeys[round - 1],present80_internal::rotationType::alignKeyForEncryption);
        present80_internal::print_m128i("roundKey is", RoundKeys[round - 1]);

 //       __m128i state = _mm_cvtsi64x_si128(block); //does not compile cause Microsoft's C++ compiler has some plan of its own
        //state = _mm_set_epi64x(0, block);
        present80_internal::print_m128i("state is", state);

        state = _mm_xor_si128(RoundKeys[round - 1], state); //pre-whitening or XOR-ing the key with the input  
        present80_internal::print_m128i("State after roundkey Whitening is", state);

        state = sBoxLayer(state); //Substituting all the 16 bytes from the S-Box one-by-one  
        present80_internal::print_m128i("State S-Box Substitution is", state);

        //state = pLayer_using_SIMD_Only(state); //permutating the bits in the register   
        state = pLayer(state); //permutating the bits in the register   
        present80_internal::print_m128i("permutated the bits in the register", state);
 
            }

        cout << "~~~~~~~~~~~~~~ Encryption Round : " << round << " ~~~~~~~~~~~~~~"   <<endl;       
        // aligning the key in the lower bytes of the SIMD register so as we can XOR it with the state or data directy
        RoundKeys[round - 1] = shiftBytesInsideBlock(RoundKeys[round - 1],present80_internal::rotationType::alignKeyForEncryption);
        present80_internal::print_m128i("roundKey is", RoundKeys[round - 1]);

 //       __m128i state = _mm_cvtsi64x_si128(block); //does not compile cause Microsoft's C++ compiler has some plan of its own
        //state = _mm_set_epi64x(0, block);
        present80_internal::print_m128i("state is", state);

        state = _mm_xor_si128(RoundKeys[round - 1], state); //pre-whitening or XOR-ing the key with the input  
        present80_internal::print_m128i("State after roundkey Whitening is", state);


    return state;
}
