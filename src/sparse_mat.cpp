#include "sparse_mat.hpp"
#include <numeric>
#include <iostream>

CSRMat transpose(const CSRMat& mat) {
  std::vector<int> count(mat.col_size());
  for (auto&& col : mat.cols) {
    ++count.at(col);
  }
  std::vector<int> row_starts(mat.col_size() + 1);
  std::partial_sum(std::begin(count), std::end(count), std::begin(row_starts) + 1);
  std::cerr << row_starts.back() << std::endl;
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
