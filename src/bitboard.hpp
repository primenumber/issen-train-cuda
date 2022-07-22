#pragma once
#include <cstdint>
#include <x86intrin.h>

namespace bitboard {

inline uint64_t parallel_extract(uint64_t src, uint64_t mask) {
#ifdef __BMI2__
  return _pext_u64(src, mask);
#else
  uint64_t result = 0;
  for (uint64_t bb = 1; mask; bb += bb) {
    if (src & mask & -mask) result |= bb;
    mask &= mask - 1;
  }
  return result;
#endif
}

inline int pop_count(uint64_t src) {
#ifdef __POPCNT__
  return _popcnt64(src);
#else
  int result = 0;
  for (; src; src &= src - 1) {
    ++result;
  }
  return result;
#endif
}

inline uint64_t delta_swap(uint64_t src, uint64_t mask, int delta) {
  uint64_t t = ((src >> delta) ^ src) & mask;
  return t ^ (t << delta) ^ src;
}

inline uint64_t flip_vertical(uint64_t src) {
  return _bswap64(src);
}

inline uint64_t flip_diag_A8H1(uint64_t src) {
  src = delta_swap(src, 0x0000'0000'F0F0'F0F0, 28);
  src = delta_swap(src, 0x0000'CCCC'0000'CCCC, 14);
  return delta_swap(src, 0x00AA'00AA'00AA'00AA, 7);
}

inline uint64_t rot90(uint64_t src) {
  return flip_vertical(flip_diag_A8H1(src));
}

// codes from: https://github.com/okuhara/edax-reversi-AVX/blob/31ff745bd136f798a236012e35e966a26f7eb9c9/src/board_sse.c#L194-L221
inline uint64_t get_moves(const uint64_t P, const uint64_t O) {
	__m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
	__m128i	M;
	const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
	const __m256i mflipH = _mm256_set_epi64x(0x7e7e7e7e7e7e7e7e, 0x7e7e7e7e7e7e7e7e, -1, 0x7e7e7e7e7e7e7e7e);

	PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(P));
	mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(O)), mflipH);

	flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
	flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
	pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
	pre_r = _mm256_srlv_epi64(pre_l, shift1897);
	shift2 = _mm256_add_epi64(shift1897, shift1897);
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
	flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
	MM = _mm256_sllv_epi64(flip_l, shift1897);
	MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));

	M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
	M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
	return _mm_cvtsi128_si64(M) & ~(P|O);	// mask with empties
}

}  // namespace bitboard
