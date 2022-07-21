#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "bitboard.hpp"
#include "sparse_mat.hpp"

struct Param {
  std::string input_path, config_path, output_path;
  size_t from, to, width;
};

Param parse_options(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " from to width config_path input_path output_path" << std::endl;
    exit(EXIT_FAILURE);
  }
  const size_t from = std::stoi(argv[1]);
  const size_t to = std::stoi(argv[2]);
  const size_t width = std::stoi(argv[3]);
  const std::string config_path = argv[4];
  const std::string input_path = argv[5];
  const std::string output_path = argv[6];
  return {
    input_path, config_path, output_path,
    from, to, width,
  };
}

struct Config {
  size_t stones_from, stones_to;
  std::vector<uint64_t> masks;
};

Config load_config(const std::string& config_path) {
  std::ifstream ifs(config_path);
  size_t stones_from, stones_to;
  ifs >> stones_from >> stones_to;
  size_t mask_count;
  ifs >> mask_count;
  std::string mask_str;
  std::vector<uint64_t> masks;
  for (size_t i = 0; i < mask_count; ++i) {
    ifs >> mask_str;
    uint64_t mask = 0;
    for (size_t j = 0; j < 64; ++j) {
      if (mask_str[j] == '1') {
        mask |= 1;
      }
      mask <<= 1;
    }
    masks.push_back(mask);
  }
  return {
    stones_from, stones_to,
    masks
  };
}

struct State {
  uint64_t player, opponent;
  int32_t score;
  size_t stone_count() const {
    return bitboard::pop_count(player) + bitboard::pop_count(opponent);
  }
};

using DataSet = std::vector<State>;

DataSet load_dataset(const std::string& input_path) {
  std::ifstream ifs(input_path);
  size_t length;
  ifs >> length;
  DataSet result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    uint64_t player, opponent;
    int32_t score;
    uint32_t best_pos;
    ifs >> std::hex >> player >> opponent >> std::dec >> score >> best_pos;
    result.push_back({player, opponent, score});
  }
  return result;
}

struct PatternIndexer {
 public:
  explicit PatternIndexer(const std::vector<uint64_t>& masks) : masks(masks), offsets(std::size(masks) + 1), base_3() {
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

CSRMat generate_matrix(const DataSet& data_set, const PatternIndexer& indexer) {
  CSRMat mat;
  mat.col_size_ = indexer.pattern_size() + 3 + 1;
  mat.row_starts.push_back(0);
  for (auto&& state : data_set) {
    std::vector<std::pair<int, double>> elements;
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
    for (auto&& [index, weight] : elements) {
      if (compressed.empty()) compressed.emplace_back(index, weight);
      else if (compressed.back().first == index) {
        compressed.back().second += weight;
      } else {
        compressed.emplace_back(index, weight);
      }
    }
    for (auto&& [index, weight] : compressed) {
      mat.weights.push_back(weight);
      mat.cols.push_back(index);
    }
    auto player = state.player;
    auto opponent = state.opponent;
    // global
    mat.weights.push_back(1.0 * bitboard::pop_count(bitboard::get_moves(player, opponent)));
    mat.cols.push_back(indexer.pattern_size() + 0);
    mat.weights.push_back(1.0 * bitboard::pop_count(bitboard::get_moves(opponent, player)));
    mat.cols.push_back(indexer.pattern_size() + 1);
    mat.weights.push_back(1.0 * bitboard::pop_count(~(player | opponent)));
    mat.cols.push_back(indexer.pattern_size() + 2);
    // constant
    mat.weights.push_back(1.0);
    mat.cols.push_back(indexer.pattern_size() + 3);
    // end row
    mat.row_starts.push_back(std::size(mat.weights));
  }
  return mat;
}

class MatDescr {
 public:
  MatDescr() {
    cusparseCreateMatDescr(&descr);
  }
  ~MatDescr() {
    cusparseDestroyMatDescr(descr);
  }
  const cusparseMatDescr_t& get() const { return descr; }
  cusparseMatDescr_t& get() { return descr; }
 private:
  cusparseMatDescr_t descr;
};

struct BSRMatDev {
  MatDescr descr;
  size_t mb, nb, nnzb, block_size;
  double* weights;
  int* cols;
  int* row_starts;
};

#define CHECK_CUDA(expr) \
{ \
  cudaError_t status = (expr); \
  if (status != cudaSuccess) { \
    std::cerr << "CUDA API failed at line: " << __LINE__ << " with error: " << cudaGetErrorString(status) << " (" << status << ")" << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
}

#define CHECK_CUSPARSE(expr) \
{ \
  cusparseStatus_t status = (expr); \
  if (status != CUSPARSE_STATUS_SUCCESS) { \
    std::cerr << "CUSPARSE API failed at line: " << __LINE__ << " with error: " << cusparseGetErrorString(status) << "-" << cusparseGetErrorName(status) << " (" << status << ")" << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
}

class Handle {
 public:
  Handle() {
    CHECK_CUSPARSE(cusparseCreate(&handle))
  }
  ~Handle() { cusparseDestroy(handle); }
  const cusparseHandle_t& get() const { return handle; }
  cusparseHandle_t& get() { return handle; }
 private:
  cusparseHandle_t handle;
};

class Stream {
 public:
  Stream() {
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  }
  ~Stream() { cudaStreamDestroy(stream); }
  const cudaStream_t& get() const { return stream; }
  cudaStream_t& get() { return stream; }
 private:
  cudaStream_t stream;
};

struct Context {
  Handle handle;
  Stream stream;
};

constexpr int block_dim = 2;

BSRMatDev convert_to_dev(const CSRMat& mat, const Context& context) {
  MatDescr csr_descr;
  CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr.get(), CUSPARSE_INDEX_BASE_ZERO))
  CHECK_CUSPARSE(cusparseSetMatType(csr_descr.get(), CUSPARSE_MATRIX_TYPE_GENERAL))
  double *csr_weights;
  int *csr_cols;
  int *csr_row_starts;
  CHECK_CUDA(cudaMalloc((void**)&csr_weights, mat.weights.size() * sizeof(double)))
  CHECK_CUDA(cudaMalloc((void**)&csr_cols, mat.cols.size() * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void**)&csr_row_starts, mat.row_starts.size() * sizeof(int)))
  CHECK_CUDA(cudaMemcpy(csr_weights, mat.weights.data(), mat.weights.size() * sizeof(double), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(csr_cols, mat.cols.data(), mat.cols.size() * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(csr_row_starts, mat.row_starts.data(), mat.row_starts.size() * sizeof(int), cudaMemcpyHostToDevice))
  int base, nnzb;
  int mb = (mat.row_size() + block_dim - 1) / block_dim;
  int *nnzTotalDevHostPtr = &nnzb;
  BSRMatDev result;
  CHECK_CUSPARSE(cusparseSetMatIndexBase(result.descr.get(), CUSPARSE_INDEX_BASE_ZERO))
  CHECK_CUSPARSE(cusparseSetMatType(result.descr.get(), CUSPARSE_MATRIX_TYPE_GENERAL))
  CHECK_CUDA(cudaMalloc((void**)&result.row_starts, (mb + 1) * sizeof(int)))
  CHECK_CUSPARSE(cusparseXcsr2bsrNnz(context.handle.get(), CUSPARSE_DIRECTION_ROW, mat.row_size(), mat.col_size(),
      csr_descr.get(), csr_row_starts, csr_cols, block_dim,
      result.descr.get(), result.row_starts, nnzTotalDevHostPtr))
      
  if (nnzTotalDevHostPtr != nullptr) {
    nnzb = *nnzTotalDevHostPtr;
  } else {
    CHECK_CUDA(cudaMemcpy(&nnzb, result.row_starts + mb, sizeof(int), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(&base, result.row_starts, sizeof(int), cudaMemcpyDeviceToHost))
    nnzb -= base;
  }
  CHECK_CUDA(cudaMalloc((void**)&result.weights, nnzb * block_dim * block_dim * sizeof(double)))
  CHECK_CUDA(cudaMalloc((void**)&result.cols, nnzb *  sizeof(int)))
  CHECK_CUSPARSE(cusparseDcsr2bsr(context.handle.get(), CUSPARSE_DIRECTION_ROW, mat.row_size(), mat.col_size(),
      csr_descr.get(), csr_weights, csr_row_starts, csr_cols, block_dim,
      result.descr.get(), result.weights, result.row_starts, result.cols))
  CHECK_CUDA(cudaStreamSynchronize(context.stream.get()))

  result.mb = mb;
  result.nb = (mat.col_size() + block_dim - 1) / block_dim;
  result.nnzb = nnzb;
  result.block_size = block_dim;
  CHECK_CUDA(cudaFree(csr_weights));
  CHECK_CUDA(cudaFree(csr_cols));
  CHECK_CUDA(cudaFree(csr_row_starts));
  return result;
}

__global__ void sub(double* y, const double* b, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = index; i < n; i += stride) {
    y[i] -= b[i];
  }
}

void solve(const CSRMat& mat, const std::vector<double>& b, std::vector<double>& x, const Context& context) {
  const auto dev_mat = convert_to_dev(mat, context);
  const auto mat_tr = transpose(mat);
  const auto dev_mat_tr = convert_to_dev(mat_tr, context);
  double* dev_x;
  CHECK_CUDA(cudaMalloc((void**)&dev_x, x.size() * sizeof(double)))
  CHECK_CUDA(cudaMemcpy(dev_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice))
  double* dev_b;
  CHECK_CUDA(cudaMalloc((void**)&dev_b, b.size() * sizeof(double)))
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice))
  double* dev_y;
  CHECK_CUDA(cudaMalloc((void**)&dev_y, b.size() * sizeof(double)))
  std::vector<double> y(b.size());

  double alpha_forward = 1.0;
  double beta_forward = 0.0;
  double alpha_backward = -3e-7;
  double beta_backward = 1.0;
  for (size_t i = 0; i < 100000; ++i) {
    CHECK_CUSPARSE(cusparseDbsrmv(context.handle.get(),
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        dev_mat.mb,
        dev_mat.nb,
        dev_mat.nnzb,
        &alpha_forward,
        dev_mat.descr.get(),
        dev_mat.weights,
        dev_mat.row_starts,
        dev_mat.cols,
        dev_mat.block_size,
        dev_x,
        &beta_forward,
        dev_y
    ))
    sub<<<128, 128, 0, context.stream.get()>>>(dev_y, dev_b, b.size());
    CHECK_CUDA(cudaMemcpyAsync(y.data(), dev_y, b.size() * sizeof(double), cudaMemcpyDeviceToHost, context.stream.get()));
    CHECK_CUSPARSE(cusparseDbsrmv(context.handle.get(),
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        dev_mat_tr.mb,
        dev_mat_tr.nb,
        dev_mat_tr.nnzb,
        &alpha_backward,
        dev_mat_tr.descr.get(),
        dev_mat_tr.weights,
        dev_mat_tr.row_starts,
        dev_mat_tr.cols,
        dev_mat_tr.block_size,
        dev_y,
        &beta_backward,
        dev_x
    ))
    CHECK_CUDA(cudaStreamSynchronize(context.stream.get()))
    double l1 = 0.0;
    for (auto&& e : y) {
      l1 += abs(e);
    }
    l1 /= y.size();
    if (i % 100 == 0) std::cerr << i << " " << l1 << std::endl;
  }
}

int main(int argc, char** argv) {
  const auto param = parse_options(argc, argv);
  const auto config = load_config(param.config_path);
  const auto data_set = load_dataset(param.input_path);

  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  Context context;
  CHECK_CUSPARSE(cusparseSetStream(context.handle.get(), context.stream.get()))

  PatternIndexer indexer(config.masks);
  size_t vec_len = indexer.pattern_size() + 3 + 1; // pattern, global(3), constant(1)
  while (vec_len % block_dim != 0) ++vec_len;
  std::vector<double> vec(vec_len);
  for (size_t mid = param.from; mid <= param.to; ++mid) {
    std::cerr << mid << std::endl;
    const size_t lower = mid - param.width + 1;
    const size_t upper = mid + param.width - 1;
    DataSet filtered;
    for (auto&& state : data_set) {
      size_t stone_count = state.stone_count();
      if (lower <= stone_count && stone_count <= upper) {
        filtered.push_back(state);
      }
    }
    const auto mat = generate_matrix(filtered, indexer);
    std::vector<double> scores;
    for (auto&& state : filtered) {
      scores.push_back(state.score);
    }
    while (scores.size() % block_dim != 0) scores.push_back(0.0);
    solve(mat, scores, vec, context);
    //std::ofstream ofs(param.output_path + "/weight_" + std::to_string(mid));
    //for (auto&& w : vec) {
    //  ofs << w << "\n";
    //}
  }
  return 0;
}
