// orchard_ops/mps/flash_attn.mm
#import <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <atomic>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <fstream>
#include <torch/script.h>
// orchard_ops/mps/flash_attn.mm
#include <tuple>
#include <vector>
#ifdef __APPLE__
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSUtils.h>
#import <Metal/Metal.h>
#endif

// --- Global kernel call counter (for debug logging) ---
static std::atomic<int> flashattn_call_count(0);

// --- FORWARD: uses PyTorch's attention then applies explicit dropout mask ---
std::tuple<at::Tensor, at::Tensor>
orchard_flash_attn_fwd(const at::Tensor &q, const at::Tensor &k,
                       const at::Tensor &v, double scale, double dropout_p,
                       bool causal) {
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] FWD call=" << flashattn_call_count << '\n';
  }
  auto attn = at::native::scaled_dot_product_attention(
      q, k, v, /*attn_mask=*/{}, /*dropout_p=*/0.0, causal,
      static_cast<float>(scale));
  at::Tensor mask = at::bernoulli(at::ones_like(attn), 1.0 - dropout_p);
  at::Tensor out = mask.mul(attn).div(1.0 - dropout_p);
  return std::make_tuple(out, mask);
}

// --- BACKWARD: fused Metal kernel applying dropout mask and scale ---
namespace {
#ifdef __APPLE__
static void launch_flash_attn_bwd(const at::Tensor &grad_out,
                                  const at::Tensor &q, const at::Tensor &k,
                                  const at::Tensor &v, const at::Tensor &mask,
                                  at::Tensor &grad_q, at::Tensor &grad_k,
                                  at::Tensor &grad_v, uint32_t n, float scale,
                                  float dropout_p, bool causal) {
  auto stream = at::mps::getCurrentMPSStream();
  @autoreleasepool {
    NSError *err = nil;
    id<MTLDevice> device = stream.device();
    static id<MTLComputePipelineState> pso = nil;
    if (!pso) {
      id<MTLLibrary> lib = at::mps::getMTLLibrary("flash_attn_backward");
      id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_bwd"];
      pso = [device newComputePipelineStateWithFunction:fn error:&err];
      TORCH_CHECK(!err, "Failed to create pipeline state for flash_attn_bwd");
      [fn release];
    }
    id<MTLCommandBuffer> cb = stream.commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_out.storage().data_ptr()
            offset:grad_out.storage_offset() * sizeof(float)
           atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q.storage().data_ptr()
            offset:q.storage_offset() * sizeof(float)
           atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)k.storage().data_ptr()
            offset:k.storage_offset() * sizeof(float)
           atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v.storage().data_ptr()
            offset:v.storage_offset() * sizeof(float)
           atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask.storage().data_ptr()
            offset:mask.storage_offset() * sizeof(float)
           atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_q.storage().data_ptr()
            offset:grad_q.storage_offset() * sizeof(float)
           atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_k.storage().data_ptr()
            offset:grad_k.storage_offset() * sizeof(float)
           atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_v.storage().data_ptr()
            offset:grad_v.storage_offset() * sizeof(float)
           atIndex:7];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&scale length:sizeof(float) atIndex:9];
    [enc setBytes:&dropout_p length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(bool) atIndex:11];
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    stream.enqueue(cb);
  }
}
#endif

#ifdef __APPLE__
static void launch_flash_attn_bwd_dropout(
    const at::Tensor &grad_out, const at::Tensor &q, const at::Tensor &k,
    const at::Tensor &v, const at::Tensor &mask, at::Tensor &grad_q,
    at::Tensor &grad_k, at::Tensor &grad_v, uint32_t n, float scale,
    float dropout_p, bool causal) {
  auto stream = at::mps::getCurrentMPSStream();
  @autoreleasepool {
    NSError *err = nil;
    id<MTLDevice> device = stream.device();
    static id<MTLComputePipelineState> pso = nil;
    if (!pso) {
      id<MTLLibrary> lib =
          at::mps::getMTLLibrary("flash_attn_backward_dropout");
      id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_bwd_dropout"];
      pso = [device newComputePipelineStateWithFunction:fn error:&err];
      TORCH_CHECK(!err,
                  "Failed to create pipeline state for flash_attn_bwd_dropout");
      [fn release];
    }
    id<MTLCommandBuffer> cb = stream.commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_out.storage().data_ptr()
            offset:grad_out.storage_offset() * sizeof(float)
           atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q.storage().data_ptr()
            offset:q.storage_offset() * sizeof(float)
           atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)k.storage().data_ptr()
            offset:k.storage_offset() * sizeof(float)
           atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v.storage().data_ptr()
            offset:v.storage_offset() * sizeof(float)
           atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask.storage().data_ptr()
            offset:mask.storage_offset() * sizeof(float)
           atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_q.storage().data_ptr()
            offset:grad_q.storage_offset() * sizeof(float)
           atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_k.storage().data_ptr()
            offset:grad_k.storage_offset() * sizeof(float)
           atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad_v.storage().data_ptr()
            offset:grad_v.storage_offset() * sizeof(float)
           atIndex:7];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&scale length:sizeof(float) atIndex:9];
    [enc setBytes:&dropout_p length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(bool) atIndex:11];
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    stream.enqueue(cb);
  }
}
#endif
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
orchard_flash_attn_bwd(const at::Tensor &grad_out, const at::Tensor &q,
                       const at::Tensor &k, const at::Tensor &v,
                       const at::Tensor &dropout_mask, double scale,
                       double dropout_p, bool causal) {
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] BWD call=" << flashattn_call_count << '\n';
  }
  at::Tensor grad_q = at::empty_like(q);
  at::Tensor grad_k = at::empty_like(k);
  at::Tensor grad_v = at::empty_like(v);
#ifdef __APPLE__
  auto n = static_cast<uint32_t>(grad_out.numel());
  launch_flash_attn_bwd(grad_out, q, k, v, dropout_mask, grad_q, grad_k, grad_v,
                        n, static_cast<float>(scale),
                        static_cast<float>(dropout_p), causal);
#else
  at::Tensor grad_in = grad_out.mul(dropout_mask).div(1.0 - dropout_p);
  grad_q.copy_(grad_in.mul(scale));
  grad_k.copy_(grad_in.mul(scale));
  grad_v.copy_(grad_in);
#endif
  return std::make_tuple(grad_q, grad_k, grad_v);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
orchard_flash_attn_bwd_dropout(const at::Tensor &grad_out, const at::Tensor &q,
                              const at::Tensor &k, const at::Tensor &v,
                              const at::Tensor &dropout_mask, double scale,
                              double dropout_p, bool causal) {
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] BWD call=" << flashattn_call_count << '\n';
  }
  at::Tensor grad_q = at::empty_like(q);
  at::Tensor grad_k = at::empty_like(k);
  at::Tensor grad_v = at::empty_like(v);
#ifdef __APPLE__
  auto n = static_cast<uint32_t>(grad_out.numel());
  launch_flash_attn_bwd_dropout(grad_out, q, k, v, dropout_mask, grad_q,
                                grad_k, grad_v, n, static_cast<float>(scale),
                                static_cast<float>(dropout_p), causal);
#else
  at::Tensor grad_in = grad_out.mul(dropout_mask).div(1.0 - dropout_p);
  grad_q.copy_(grad_in.mul(scale));
  grad_k.copy_(grad_in.mul(scale));
  grad_v.copy_(grad_in);
#endif
  return std::make_tuple(grad_q, grad_k, grad_v);
}

// --- Register with Torch dispatcher under correct schema ---
static auto fwd_schema =
    "flash_attn_mps::_flash_attn_fwd(Tensor q, Tensor k, Tensor v, float "
    "scale, float dropout_p, bool causal) -> (Tensor, Tensor)";
static auto bwd_schema =
    "flash_attn_mps::_flash_attn_bwd(Tensor grad_out, Tensor q, Tensor k, "
    "Tensor v, Tensor dropout_mask, float scale, float dropout_p, bool causal) "
    "-> (Tensor, Tensor, Tensor)";
static auto bwd_dropout_schema =
    "flash_attn_mps::_flash_attn_bwd_dropout(Tensor grad_out, Tensor q, Tensor k, "
    "Tensor v, Tensor dropout_mask, float scale, float dropout_p, bool causal) -> "
    "(Tensor, Tensor, Tensor)";

TORCH_LIBRARY(flash_attn_mps, m) {
  m.def(fwd_schema, orchard_flash_attn_fwd);
  m.def(bwd_schema, orchard_flash_attn_bwd);
  m.def(bwd_dropout_schema, orchard_flash_attn_bwd_dropout);
}
