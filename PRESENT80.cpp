#pragma once
#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
// #include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
// #include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "PRESENT80.h"
#include "Helper.h"

using namespace std;

const uint8_t PRESENT_80_CORE::PRESENT80_SBOX[16] = {    0xC, 0x5, 0x6, 0xB, 
                                        0x9, 0x0, 0xA, 0xD, 
                                        0x3, 0xE, 0xF, 0x8, 
                                        0x4, 0x7, 0x1, 0x2
                                    };

PRESENT_80_CORE::PRESENT_80_CORE(__m128i key)
{
    expandRoundKeys(key, 1);
}

// functiont that will perform the rotation inside the SIMD register
__m128i PRESENT_80_CORE::shiftBytesInsideBlock(__m128i block, present80_internal::rotationType mode)
{
    __m128i mask = getRotationMask(mode);
    return _mm_shuffle_epi8(block, mask);
}

//This function rotates the key left by 61 bits
// It does that in 4 parts 
// since we're implementing the Cypher usign SIMD instrinsics, we will make use of __m12i data type which mimics a CPU register 
// 1. Load the rotation mask so that we can move the  bytes easily accross the block >> rotate the block by 2 bytes to the the right 
// 2. break down the block into 2 parts. Hi and Lo. Initially copy the whole block in 2 128 bt words - hi and lo 
// 3. Lo block to be shifted right by 3 bytes
//    Hi block to be shifted left by 61 bytes 
// 4. now the high and the low bytes are in the right order, just OR the 2 blocks, 
// there would be zeroes in the lower bits of the high block whcih would be filled by teh lower block
// fianlly in the original 
__m128i PRESENT_80_CORE::rotateLeftBy61Bits(__m128i block)
{
    
    // Step 1: rotate right by 2 bytes / 16 bits
    const __m128i mask = _mm_set_epi8(
1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2
    );

    block = _mm_shuffle_epi8(block, mask);
    present80_internal::print_m128i("Original - After shuffling", block);

    // Step 2: copying the key into a low-block and high-block of 64 -bits ( 8 bytes)
    __m128i low = block;
    __m128i high = block;

    // Step 3: rotate right 3 bits     
    low = _mm_srli_epi64(low, 3);
    high = _mm_slli_epi64(high, 61);
    
    present80_internal::print_m128i("Low Byte After right shift by 3", low);
    present80_internal::print_m128i("High Byte  After left shift by 61", high);

    // Step 4: Combine the high and the low block using ligical OR and recrete the _m128i block    
    __m128i rotated = _mm_or_si128(low, high);
    present80_internal::print_m128i("Low and High Merged", rotated);

    // create a mask to discard the higher bytes from the block i.e. byte 10 - 15, 
    // and to keep bytes 0 - 9  
    __m128i keep80 = _mm_set_epi8(
        0,0,0,0,0,0,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    );
     present80_internal::print_m128i("Mask to retain only lower 10 bytes in the Register ", keep80);
    // apply the mask to the block and we're done
    
    __m128i rotated_left_by_61_bits = _mm_and_si128(rotated, keep80);
    present80_internal::print_m128i("After final rotation ", rotated_left_by_61_bits);

    return rotated_left_by_61_bits;
}


__m128i PRESENT_80_CORE::rotateLeftBy61Bits_SIMD_ONLY(__m128i block){

    __m128i LowerLanesRearrangement = _mm_shuffle_epi32(block, _MM_SHUFFLE(2,1,1,0));
    present80_internal::print_m128i("LowerLanesRearrangement with lanes 2,1,1,0 from orignal key", LowerLanesRearrangement);

    LowerLanesRearrangement = _mm_srli_epi64(LowerLanesRearrangement, 19);
    present80_internal::print_m128i("LowerLanesRearrangement >> 19", LowerLanesRearrangement);

    // create a mask   [
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   1F FF FF FF | 00011111 11111111 11111111 11111111
    //                   00 00 00 00 | 00000000 00000000 00000000 00000000
    //                   FF FF FF FF | 11111111 11111111 11111111 11111111
    //                ]
    __m128i maskLower = _mm_set_epi32(        0x00, 
                                        0x1FFFFFFF, 
                                              0x00, 
                                        0xFFFFFFFF);
 

                                        present80_internal::print_m128i("LowerLanesRearrangement mask", maskLower);

    // intrnsic for LowerLanesRearrangement = LowerLanesRearrangement AND Mask
    LowerLanesRearrangement = _mm_and_si128(LowerLanesRearrangement, maskLower); 
    present80_internal::print_m128i("LowerLanesRearrangement AND mask", LowerLanesRearrangement);

    LowerLanesRearrangement = _mm_shuffle_epi32(LowerLanesRearrangement, _MM_SHUFFLE(1,1,2,0));
    present80_internal::print_m128i("LowerLanesRearrangement shuffled with the lanes 1,1,2,0 to get the LSBs aligned", LowerLanesRearrangement);



    __m128i UpperLanesRearrangement = _mm_shuffle_epi32(block, _MM_SHUFFLE(0,0,0,0));
    present80_internal::print_m128i("UpperLanesRearrangement with lanes 0,0,0,0 from the original key", UpperLanesRearrangement);

    UpperLanesRearrangement = _mm_srli_epi64(UpperLanesRearrangement, 3);
     present80_internal::print_m128i("UpperLanesRearrangement >> 3", UpperLanesRearrangement); 
    //instrincs for :  create a mask  
    //                      [ 
    //                        00 00 FF FF | 00000000 00000000 11111111 11111111
    //                        E0 00 00 00 | 11100000 00000000 00000000 00000000
    //                        00 00 00 00 | 00000000 00000000 00000000 00000000
    //                        00 00 00 00 | 00000000 00000000 00000000 00000000   
    //                      ]
    // Mask for UpperLanes [0, 0, 0x0000FFFF, 0x00000000] 
    // Note: PRESENT-80 uses bits 0-79. 
    // We mask to keep only the relevant moved bits.

__m128i maskHigher = _mm_set_epi32(       0x0000FFFF, 
                                          0xE0000000, 
                                                0x00, 
                                                0x00);

present80_internal::print_m128i("UpperLanesRearrangement mask", maskHigher);
       
    // intrnsic for UpperLanesRearrangement = UpperLanesRearrangement AND Mask
    UpperLanesRearrangement = _mm_and_si128(UpperLanesRearrangement, maskHigher);
    present80_internal::print_m128i("UpperLanesRearrangement AND mask", UpperLanesRearrangement);

    UpperLanesRearrangement = _mm_shuffle_epi32(UpperLanesRearrangement, _MM_SHUFFLE(0,3,2,0));
    present80_internal::print_m128i("UpperLanesRearrangement shuffled with the lanes 0,3,2,0 to get the MSBs aligned", UpperLanesRearrangement);



    // We use OR to merge the bits into the final 80-bit state
    // instrincs for result = pperLanesRearrangement AND LowerLanesRearrangement
    __m128i result = _mm_or_si128(UpperLanesRearrangement, LowerLanesRearrangement);
    present80_internal::print_m128i("UpperLanesRearrangement OR  LowerLanesRearrangement", result);

    return result;

}
 
 
void PRESENT_80_CORE::expandRoundKeys(__m128i key, uint8_t Round)
{
    present80_internal::print_m128i("Original Key", key);

    key = rotateLeftBy61Bits_SIMD_ONLY(key);
    present80_internal::print_m128i("rotated_left_by_61_bis", key);

    uint8_t topByte = _mm_extract_epi8(key, 9);  // extracting the top byte from the rotated key
    uint8_t upper_nibble = PRESENT80_SBOX[topByte >> 4];// right shft the top byte by 4 bits to get the higher nible and get the S-box value at the nibble
    topByte = (topByte & 0x0F) | (upper_nibble << 4);   // make space inside the upper nibble  from the key as that we can insert the substituted value back
    key = _mm_insert_epi8(key, topByte, 9);             // reinsertt the byte into the key 
    present80_internal::print_m128i("Substituted Top Nibble", key);
 
    uint64_t rc = (Round & 0x1F); // generating the 5-bit round constant
    uint64_t rc_shifted = rc << 15; // move into bit positions 19..15
    cout << hex <<  "RC:" <<rc << "RC Shfited by 15 bits left:"<< rc_shifted <<endl <<dec;

    __m128i rc_vec = _mm_set_epi64x(0, rc_shifted); //create a 128 bit block for XOR-ing the RC with the key
    present80_internal::print_m128i("round Constant", rc_vec);

    // XOR into lane0 (bits 19..15) 
    key = _mm_xor_si128(key, rc_vec);
     present80_internal::print_m128i("XOR-ing with the round Cosntant", key);


}
 