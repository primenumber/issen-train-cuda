#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

#include "bitboard.hpp"

struct State {
  uint64_t player, opponent;
  int32_t score;
  size_t stone_count() const {
    return bitboard::pop_count(player) + bitboard::pop_count(opponent);
  }
};

using DataSet = std::vector<State>;

struct PatternIndexer {
 public:
  explicit PatternIndexer(const std::vector<uint64_t>& masks);
  size_t get_index(size_t id, uint64_t player, uint64_t opponent) const {
    uint64_t p_bit = bitboard::parallel_extract(player, masks.at(id));
    uint64_t o_bit = bitboard::parallel_extract(opponent, masks.at(id));
    size_t p_index = base_3.at(p_bit);
    size_t o_index = base_3.at(o_bit);
    return offsets.at(id) + p_index + 2 * o_index;
  }
  size_t mask_size() const { return std::size(masks); }
  size_t pattern_size() const { return offsets.back(); }
  std::vector<uint64_t> masks;
  std::vector<size_t> offsets;
  std::vector<size_t> base_3;
 private:
};

struct CSRMat {
  std::vector<double> weights;
  std::vector<int> cols;
  std::vector<int> row_starts;
  size_t col_size_;
  size_t col_size() const { return col_size_; }
  size_t row_size() const { return row_starts.size() - 1; }
  size_t nnz() const { return weights.size(); }
};

CSRMat generate_matrix(const DataSet&, const PatternIndexer&);
CSRMat transpose(const CSRMat&);
