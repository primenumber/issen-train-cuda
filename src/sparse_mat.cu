#include "sparse_mat.hpp"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>

PatternIndexer::PatternIndexer(const std::vector<uint64_t>& masks) : masks(masks), offsets(std::size(masks) + 1), base_3() {
  size_t largest_mask = 0;
  for (auto&& mask : masks) {
    largest_mask = std::max<size_t>(largest_mask, bitboard::pop_count(mask));
  }
  std::vector<size_t> pow_3(largest_mask + 1);
  pow_3[0] = 1;
  for (size_t i = 1; i <= largest_mask; ++i) {
    pow_3[i] = 3 * pow_3[i-1];
  }
  offsets[0] = 0;
  for (size_t i = 0; i < std::size(masks); ++i) {
    offsets[i+1] = offsets[i] + pow_3[bitboard::pop_count(masks[i])];
  }
  size_t largest_pow2 = 1 << largest_mask;
  base_3.resize(largest_pow2);
  for (uint64_t i = 0; i < largest_pow2; ++i) {
    size_t index = 0;
    for (size_t j = 0; j < largest_mask; ++j) {
      if ((i >> j) & 1) {
        index += pow_3[j];
      }
    }
    base_3[i] = index;
  }
}

std::tuple<std::vector<double>, std::vector<int>, double> generate_row(const State& state, const PatternIndexer& indexer) {
  std::vector<int> indices;
  indices.reserve(indexer.mask_size() * 8);
  for (size_t id = 0; id < indexer.mask_size(); ++id) {
    auto player = state.player;
    auto opponent = state.opponent;
    for (size_t i = 0; i < 4; ++i) {
      const auto player_v = bitboard::flip_vertical(player);
      const auto opponent_v = bitboard::flip_vertical(opponent);
      auto index = indexer.get_index(id, player, opponent);
      auto index_v = indexer.get_index(id, player_v, opponent_v);
      indices.push_back(index);
      indices.push_back(index_v);
      player = bitboard::rot90(player);
      opponent = bitboard::rot90(opponent);
    }
  }
  std::sort(std::begin(indices), std::end(indices));
  std::vector<double> weights;
  std::vector<int> cols;
  weights.reserve(indexer.mask_size() * 8);
  cols.reserve(indexer.mask_size() * 8);
  for (auto&& index : indices) {
    if (cols.empty()) {
      cols.emplace_back(index);
      weights.emplace_back(1.0);
    } else if (cols.back() == index) {
      weights.back() += 1.0;
    } else {
      cols.emplace_back(index);
      weights.emplace_back(1.0);
    }
  }
  auto player = state.player;
  auto opponent = state.opponent;
  // global
  weights.push_back(1.0 * bitboard::pop_count(bitboard::get_moves(player, opponent)));
  cols.push_back(indexer.pattern_size() + 0);
  weights.push_back(1.0 * bitboard::pop_count(bitboard::get_moves(opponent, player)));
  cols.push_back(indexer.pattern_size() + 1);
  weights.push_back(1.0 * bitboard::pop_count(~(player | opponent)));
  cols.push_back(indexer.pattern_size() + 2);
  // constant
  weights.push_back(1.0);
  cols.push_back(indexer.pattern_size() + 3);
  return {std::move(weights), std::move(cols), state.score};
}

std::pair<CSRMat, std::vector<double>> generate_matrix(const DataSet& data_set, const PatternIndexer& indexer) {
  const size_t col_size = indexer.pattern_size() + 3 + 1;  // pattern, global (3), constant (1)
  const size_t row_size = std::size(data_set) + col_size; // dataset, l2 norm (col_size)
  std::vector<size_t> row_nnz(row_size);
#pragma omp parallel for
  for (size_t i = 0; i < std::size(data_set); ++i) {
    const auto [weights, cols, score] = generate_row(data_set[i], indexer);
    row_nnz[i] = std::size(weights);
  }
  for (size_t i = 0; i < col_size; ++i) {
    row_nnz[i + std::size(data_set)] = 1;
  }
  CSRMat mat;
  mat.row_starts.resize(row_size + 1);
  mat.col_size_ = col_size;
  std::partial_sum(std::begin(row_nnz), std::end(row_nnz), std::begin(mat.row_starts) + 1);
  const size_t nnz = mat.row_starts.back();
  mat.cols.resize(nnz);
  mat.weights.resize(nnz);
  std::vector<double> vec(row_size);
#pragma omp parallel for
  for (size_t i = 0; i < std::size(data_set); ++i) {
    const auto [weights, cols, score] = generate_row(data_set[i], indexer);
    size_t from = mat.row_starts[i];
    std::copy(std::begin(weights), std::end(weights), std::begin(mat.weights) + from);
    std::copy(std::begin(cols), std::end(cols), std::begin(mat.cols) + from);
    vec[i] = score;
  }
  // L2 normalization
  const size_t from = mat.row_starts[std::size(data_set)];
  std::fill(std::begin(mat.weights) + from, std::end(mat.weights), 1.0);
  std::iota(std::begin(mat.cols) + from, std::end(mat.cols), 0);
  std::fill(std::begin(vec) + std::size(data_set), std::end(vec), 0.0);
  return {std::move(mat), std::move(vec)};
}
