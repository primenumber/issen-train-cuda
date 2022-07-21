#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

struct CSRMat {
  std::vector<double> weights;
  std::vector<int> cols;
  std::vector<int> row_starts;
  size_t col_size_;
  size_t col_size() const { return col_size_; }
  size_t row_size() const { return row_starts.size() - 1; }
  size_t nnz() const { return weights.size(); }
};

CSRMat transpose(const CSRMat&);
