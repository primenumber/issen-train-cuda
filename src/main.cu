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

struct CSRMatDev {
  explicit CSRMatDev(const CSRMat& mat) {
    CHECK_CUDA(cudaMalloc((void**)&weights, mat.weights.size() * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&cols, mat.cols.size() * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&row_starts, mat.row_starts.size() * sizeof(int)))
    CHECK_CUDA(cudaMemcpy(weights, mat.weights.data(), mat.weights.size() * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(cols, mat.cols.data(), mat.cols.size() * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(row_starts, mat.row_starts.data(), mat.row_starts.size() * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUSPARSE(cusparseCreateCsr(&descr, mat.row_size(), mat.col_size(), mat.nnz(),
          row_starts, cols, weights,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  }
  ~CSRMatDev() {
    CHECK_CUSPARSE(cusparseDestroySpMat(descr))
    CHECK_CUDA(cudaFree(weights))
    CHECK_CUDA(cudaFree(cols))
    CHECK_CUDA(cudaFree(row_starts))
  }
  double *weights;
  int *cols;
  int *row_starts;
  cusparseSpMatDescr_t descr;
};

struct DnVec {
  explicit DnVec(const std::vector<double>& v) : length(v.size()) {
    CHECK_CUDA(cudaMalloc((void**)&data, length * sizeof(double)))
    CHECK_CUDA(cudaMemcpy(data, v.data(), length * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUSPARSE(cusparseCreateDnVec(&descr, length, data, CUDA_R_64F))
  }
  explicit DnVec(size_t length) : length(length) {
    CHECK_CUDA(cudaMalloc((void**)&data, length * sizeof(double)))
    CHECK_CUDA(cudaMemset(data, 0, length * sizeof(double)))
    CHECK_CUSPARSE(cusparseCreateDnVec(&descr, length, data, CUDA_R_64F))
  }
  ~DnVec() {
    CHECK_CUSPARSE(cusparseDestroyDnVec(descr))
    CHECK_CUDA(cudaFree(data))
  }
  size_t size() const { return length; }
  size_t length;
  double *data;
  cusparseDnVecDescr_t descr;
};

__global__ void sub(const double *src1, const double *src2, double *dst, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = index; i < n; i += stride) {
    dst[i] = src1[i] - src2[i];
  }
}

__global__ void fma_sc(const double src1, const double *src2, double *dst, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = index; i < n; i += stride) {
    dst[i] += src1 * src2[i];
  }
}

__global__ void fma4_sc(const double src1, const double *src2, const double *acc, double *dst, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = index; i < n; i += stride) {
    dst[i] = src1 * src2[i] + acc[i];
  }
}

__global__ void accum_l2(double* buf_acc, const double* v, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  double result = 0.0;
  for (size_t i = index; i < n; i += stride) {
    result += v[i] * v[i];
  }
  buf_acc[index] = result;
}

__global__ void accum_l1(double* l1_acc, const double* y, size_t n) {
  const size_t stride = blockDim.x * gridDim.x;
  const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  double result = 0.0;
  for (size_t i = index; i < n; i += stride) {
    result += abs(y[i]);
  }
  l1_acc[index] = result;
}

void solve_impl(const Context& context, CSRMatDev& mat, const CSRMatDev& mat_tr, const DnVec& a, const DnVec& b) {
  double alpha = 1.0;
  double beta = 0.0;
  auto external_buffer = [&] {
    size_t bufsize_forward = 0, bufsize_backward = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(context.handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha,
          mat.descr,
          a.descr,
          &beta,
          b.descr,
          CUDA_R_64F,
          CUSPARSE_SPMV_CSR_ALG1,
          &bufsize_forward))
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(context.handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha,
          mat_tr.descr,
          b.descr,
          &beta,
          a.descr,
          CUDA_R_64F,
          CUSPARSE_SPMV_CSR_ALG1,
          &bufsize_backward))
    const size_t bufsize = std::max(bufsize_forward, bufsize_backward);
    void *buf;
    CHECK_CUDA(cudaMalloc(&buf, bufsize))
    return buf;
  }();
  auto spmv_non_trans = [&] (auto&& v_in, auto&& v_out) {
    CHECK_CUSPARSE(cusparseSpMV(context.handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha,
          mat.descr,
          v_in.descr,
          &beta,
          v_out.descr,
          CUDA_R_64F,
          CUSPARSE_SPMV_CSR_ALG1,
          external_buffer))
  };
  auto spmv_trans = [&] (auto&& v_in, auto&& v_out) {
    CHECK_CUSPARSE(cusparseSpMV(context.handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha,
          mat_tr.descr,
          v_in.descr,
          &beta,
          v_out.descr,
          CUDA_R_64F,
          CUSPARSE_SPMV_CSR_ALG1,
          external_buffer))
  };
  auto sub_vec = [&] (auto&& src1, auto&& src2, auto&& dst) {
    sub<<<1024, 256, 0, context.stream.get()>>>(src1.data, src2.data, dst.data, dst.size());
  };
  auto fma_sc_vec = [&] (double src1, auto&& src2, auto&& dst) {
    fma_sc<<<1024, 256, 0, context.stream.get()>>>(src1, src2.data, dst.data, dst.size());
  };
  auto fma4_sc_vec = [&] (double src1, auto&& src2, auto&& acc, auto&& dst) {
    fma4_sc<<<1024, 256, 0, context.stream.get()>>>(src1, src2.data, acc.data, dst.data, dst.size());
  };
  double* buf_acc;
  CHECK_CUDA(cudaMallocManaged((void**)&buf_acc, 1024 * sizeof(double)))
  auto l1_norm = [&] (auto&& v) {
    accum_l1<<<8, 128, 0, context.stream.get()>>>(buf_acc, v.data, v.size()); 
    CHECK_CUDA(cudaStreamSynchronize(context.stream.get()))
    double sum = 0.0;
    for (size_t i = 0; i < 1024; ++i) {
      sum += buf_acc[i];
    }
    return sum;
  };
  auto l2_norm = [&] (auto&& v) {
    accum_l2<<<8, 128, 0, context.stream.get()>>>(buf_acc, v.data, v.size()); 
    CHECK_CUDA(cudaStreamSynchronize(context.stream.get()))
    double sum = 0.0;
    for (size_t i = 0; i < 1024; ++i) {
      sum += buf_acc[i];
    }
    return sum;
  };

  DnVec pa(b.size());
  spmv_non_trans(a, pa);
  DnVec r(b.size());
  sub_vec(b, pa, r);
  DnVec p(a.size());
  spmv_trans(r, p);
  DnVec s(a.size());
  CHECK_CUDA(cudaMemcpyAsync(s.data, p.data, s.size() * sizeof(double), cudaMemcpyDeviceToDevice, context.stream.get()))
  double old_s_norm = l2_norm(s);
  DnVec q(b.size());

  for (size_t i = 0; i < 300; ++i) {
    spmv_non_trans(p, q);
    const auto alpha = old_s_norm / l2_norm(q);
    fma_sc_vec(alpha, p, a);
    fma_sc_vec(-alpha, q, r);
    spmv_trans(r, s);
    const auto new_s_norm = l2_norm(s);
    if (i % 10 == 0) {
      double l1 = l1_norm(r) / r.size();
      std::cerr << i << " " << l1 << std::endl;
    }
    if (new_s_norm < 1.0) {
      break;
    }
    const auto beta = new_s_norm / old_s_norm;
    fma4_sc_vec(beta, p, s, p);
    old_s_norm = new_s_norm;
  }
  CHECK_CUDA(cudaStreamSynchronize(context.stream.get()))
  cudaFree(external_buffer);
  cudaFree(buf_acc);
}

void solve(const Context& context, const CSRMat& mat, std::vector<double>& a, const std::vector<double>& b) {
  CSRMatDev dev_mat(mat);
  const auto dev_mat_tr = [&] {
    const auto mat_tr = transpose(mat);
    return CSRMatDev(mat_tr);
  }();
  DnVec dev_a(a);
  DnVec dev_b(b);
  solve_impl(context, dev_mat, dev_mat_tr, dev_a, dev_b);
  CHECK_CUDA(cudaMemcpyAsync(a.data(), dev_a.data, a.size() * sizeof(double), cudaMemcpyDeviceToHost, context.stream.get()))
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
    scores.push_back(0.0); // L2 normalization
    solve(context, mat, vec, scores);
    std::ofstream ofs(param.output_path + "/weight_" + std::to_string(mid));
    for (auto&& w : vec) {
      ofs << w << "\n";
    }
  }
  return 0;
}
