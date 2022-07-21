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
  std::vector<std::pair<int, double>> elements;
  elements.reserve(indexer.mask_size() * 8);
  for (size_t id = 0; id < indexer.mask_size(); ++id) {
    auto player = state.player;
    auto opponent = state.opponent;
    for (size_t i = 0; i < 4; ++i) {
      const auto player_v = bitboard::flip_vertical(player);
      const auto opponent_v = bitboard::flip_vertical(opponent);
      auto index = indexer.get_index(id, player, opponent);
      auto index_v = indexer.get_index(id, player_v, opponent_v);
      elements.push_back({index, 1.0});
      elements.push_back({index_v, 1.0});
      player = bitboard::rot90(player);
      opponent = bitboard::rot90(opponent);
    }
  }
  std::sort(std::begin(elements), std::end(elements));
  std::vector<std::pair<int, double>> compressed;
  compressed.reserve(indexer.mask_size() * 8);
  for (auto&& [index, weight] : elements) {
    if (compressed.empty()) compressed.emplace_back(index, weight);
    else if (compressed.back().first == index) {
      compressed.back().second += weight;
    } else {
      compressed.emplace_back(index, weight);
    }
  }
  std::vector<double> weights;
  std::vector<int> cols;
  for (auto&& [index, weight] : compressed) {
    weights.push_back(weight);
    cols.push_back(index);
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
  return {weights, cols, state.score};
}

std::pair<CSRMat, std::vector<double>> generate_matrix(const DataSet& data_set, const PatternIndexer& indexer) {
  CSRMat mat;
  std::vector<double> vec;
  mat.col_size_ = indexer.pattern_size() + 3 + 1;
  mat.row_starts.reserve(std::size(data_set) + 2);
  mat.cols.reserve((indexer.mask_size() * 8 + 4) * std::size(data_set) + mat.col_size_);
  mat.weights.reserve((indexer.mask_size() * 8 + 4) * std::size(data_set) + mat.col_size_);
  mat.row_starts.push_back(0);
  std::mutex mtx;
#pragma omp parallel for
  for (size_t i = 0; i < std::size(data_set); ++i) {
    const auto [weights, cols, score] = generate_row(data_set[i], indexer);
    {
      std::lock_guard lock(mtx);
      mat.weights.insert(std::end(mat.weights), std::begin(weights), std::end(weights));
      mat.cols.insert(std::end(mat.cols), std::begin(cols), std::end(cols));
      mat.row_starts.push_back(std::size(mat.weights));
      vec.push_back(score);
    }
  }
  // L2 normalization
  for (size_t i = 0; i < mat.col_size_; ++i) {
    mat.weights.push_back(1.0);
    mat.cols.push_back(i);
  }
  mat.row_starts.push_back(std::size(mat.weights));
  vec.push_back(0);
  return {mat, vec};
}

CSRMat transpose(const CSRMat& mat) {
  std::vector<int> count(mat.col_size());
  for (auto&& col : mat.cols) {
    ++count.at(col);
  }
  std::vector<int> row_starts(mat.col_size() + 1);
  std::partial_sum(std::begin(count), std::end(count), std::begin(row_starts) + 1);
  std::vector<int> offsets(mat.col_size());
  for (size_t i = 0; i < mat.col_size(); ++i) {
    offsets[i] = row_starts[i];
  }
  std::vector<double> weights(mat.nnz());
  std::vector<int> cols(mat.nnz());
  for (size_t row = 0; row < mat.row_size(); ++row) {
    for (int i = mat.row_starts[row]; i < mat.row_starts[row + 1]; ++i) {
      const size_t col = mat.cols[i];
      weights[offsets[col]] = mat.weights[i];
      cols[offsets[col]] = row;
      ++offsets[col];
    }
  }
  return {
    weights,
    cols,
    row_starts,
    mat.row_size(),
  };
}
