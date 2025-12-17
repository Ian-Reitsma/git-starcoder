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
#import <Foundation/Foundation.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#import <Metal/Metal.h>

// Runtime-compiled Metal sources (avoid relying on internal ATen/mps/MPSUtils.h
// which is not shipped in pip wheels).
#include "flash_attn_backward_source.h"
#include "flash_attn_backward_dropout_source.h"
#endif
#ifdef __APPLE__
static id<MTLLibrary> orchard_compile_metal_library(id<MTLDevice> device, const char* src, NSError** errOut) {
  @autoreleasepool {
    if (!device || !src) {
      return nil;
    }
    NSString* source = [NSString stringWithUTF8String:src];
    if (!source) {
      return nil;
    }
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    // Keep options default for maximum compatibility.
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:errOut];
    [opts release];
    return lib;
  }
}

// Convert an MPS tensor storage into an MTLBuffer via the public IMPSAllocator
// interface (works with pip wheels; avoids internal MPSUtils.h).
static id<MTLBuffer> orchard_mtlbuffer_from_tensor_storage(
    const at::Tensor& t,
    id<MTLDevice> device,
    NSUInteger* out_offset_bytes) {
  TORCH_CHECK(t.is_mps(), "orchard_mtlbuffer_from_tensor_storage: expected MPS tensor");
  TORCH_CHECK(t.storage().data_ptr().get() != nullptr, "orchard_mtlbuffer_from_tensor_storage: null storage");

  void* storage_ptr = t.storage().data_ptr().get();
  auto* alloc = at::mps::getIMPSAllocator(/*sharedAllocator=*/false);
  TORCH_CHECK(alloc, "orchard_mtlbuffer_from_tensor_storage: no IMPSAllocator");
  TORCH_CHECK(alloc->isSharedStorageSupported(), "orchard_mtlbuffer_from_tensor_storage: shared storage not supported");

  // Force shared buffers only; without internal headers we cannot retrieve a
  // private MTLBuffer handle from a pointer.
  TORCH_CHECK(alloc->isSharedBuffer(storage_ptr), "orchard: tensor storage is not shared; cannot get MTLBuffer handle");

  auto shared = alloc->getSharedBufferPtr(storage_ptr);
  const void* shared_base = shared.first;
  uint32_t shared_base_offset = shared.second;

  ssize_t unaligned_size = alloc->getUnalignedBufferSize(storage_ptr);
  TORCH_CHECK(unaligned_size > 0, "orchard: invalid shared buffer size");

  // Wrap shared memory into a Metal buffer without copying.
  id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:(void*)shared_base
                                               length:(NSUInteger)unaligned_size
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
  TORCH_CHECK(buf != nil, "orchard: failed to create MTLBuffer wrapper");

  // Offset = allocator-provided base offset + tensor view offset.
  uint64_t view_off = (uint64_t)t.storage_offset() * (uint64_t)t.element_size();
  uint64_t off = (uint64_t)shared_base_offset + view_off;
  *out_offset_bytes = (NSUInteger)off;
  return buf;
}
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
  TORCH_CHECK(stream != nullptr, "orchard: getCurrentMPSStream returned null");
  @autoreleasepool {
    NSError *err = nil;
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    static id<MTLComputePipelineState> pso = nil;
    if (!pso) {
      static id<MTLLibrary> lib = nil;
      if (!lib) {
        lib = orchard_compile_metal_library(device, kFlashAttnBackwardMetalSrc, &err);
        TORCH_CHECK(!err && lib, "Failed to compile Metal library for flash_attn_bwd");
      }
      id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_bwd"];
      pso = [device newComputePipelineStateWithFunction:fn error:&err];
      TORCH_CHECK(!err, "Failed to create pipeline state for flash_attn_bwd");
      [fn release];
    }
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)stream->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    NSUInteger off0=0, off1=0, off2=0, off3=0, off4=0, off5=0, off6=0, off7=0;
    id<MTLBuffer> b0 = orchard_mtlbuffer_from_tensor_storage(grad_out, device, &off0);
    id<MTLBuffer> b1 = orchard_mtlbuffer_from_tensor_storage(q, device, &off1);
    id<MTLBuffer> b2 = orchard_mtlbuffer_from_tensor_storage(k, device, &off2);
    id<MTLBuffer> b3 = orchard_mtlbuffer_from_tensor_storage(v, device, &off3);
    id<MTLBuffer> b4 = orchard_mtlbuffer_from_tensor_storage(mask, device, &off4);
    id<MTLBuffer> b5 = orchard_mtlbuffer_from_tensor_storage(grad_q, device, &off5);
    id<MTLBuffer> b6 = orchard_mtlbuffer_from_tensor_storage(grad_k, device, &off6);
    id<MTLBuffer> b7 = orchard_mtlbuffer_from_tensor_storage(grad_v, device, &off7);

    [enc setBuffer:b0 offset:off0 atIndex:0];
    [enc setBuffer:b1 offset:off1 atIndex:1];
    [enc setBuffer:b2 offset:off2 atIndex:2];
    [enc setBuffer:b3 offset:off3 atIndex:3];
    [enc setBuffer:b4 offset:off4 atIndex:4];
    [enc setBuffer:b5 offset:off5 atIndex:5];
    [enc setBuffer:b6 offset:off6 atIndex:6];
    [enc setBuffer:b7 offset:off7 atIndex:7];

    [enc setBytes:&n length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&scale length:sizeof(float) atIndex:9];
    [enc setBytes:&dropout_p length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(bool) atIndex:11];
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];

    // Commit but do not block.
    stream->synchronize(at::mps::SyncType::COMMIT);
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
  TORCH_CHECK(stream != nullptr, "orchard: getCurrentMPSStream returned null");
  @autoreleasepool {
    NSError *err = nil;
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    static id<MTLComputePipelineState> pso = nil;
    if (!pso) {
      static id<MTLLibrary> lib = nil;
      if (!lib) {
        lib = orchard_compile_metal_library(device, kFlashAttnBackwardDropoutMetalSrc, &err);
        TORCH_CHECK(!err && lib, "Failed to compile Metal library for flash_attn_bwd_dropout");
      }
      id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_bwd_dropout"];
      pso = [device newComputePipelineStateWithFunction:fn error:&err];
      TORCH_CHECK(!err,
                  "Failed to create pipeline state for flash_attn_bwd_dropout");
      [fn release];
    }
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)stream->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    NSUInteger off0=0, off1=0, off2=0, off3=0, off4=0, off5=0, off6=0, off7=0;
    id<MTLBuffer> b0 = orchard_mtlbuffer_from_tensor_storage(grad_out, device, &off0);
    id<MTLBuffer> b1 = orchard_mtlbuffer_from_tensor_storage(q, device, &off1);
    id<MTLBuffer> b2 = orchard_mtlbuffer_from_tensor_storage(k, device, &off2);
    id<MTLBuffer> b3 = orchard_mtlbuffer_from_tensor_storage(v, device, &off3);
    id<MTLBuffer> b4 = orchard_mtlbuffer_from_tensor_storage(mask, device, &off4);
    id<MTLBuffer> b5 = orchard_mtlbuffer_from_tensor_storage(grad_q, device, &off5);
    id<MTLBuffer> b6 = orchard_mtlbuffer_from_tensor_storage(grad_k, device, &off6);
    id<MTLBuffer> b7 = orchard_mtlbuffer_from_tensor_storage(grad_v, device, &off7);

    [enc setBuffer:b0 offset:off0 atIndex:0];
    [enc setBuffer:b1 offset:off1 atIndex:1];
    [enc setBuffer:b2 offset:off2 atIndex:2];
    [enc setBuffer:b3 offset:off3 atIndex:3];
    [enc setBuffer:b4 offset:off4 atIndex:4];
    [enc setBuffer:b5 offset:off5 atIndex:5];
    [enc setBuffer:b6 offset:off6 atIndex:6];
    [enc setBuffer:b7 offset:off7 atIndex:7];

    [enc setBytes:&n length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&scale length:sizeof(float) atIndex:9];
    [enc setBytes:&dropout_p length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(bool) atIndex:11];
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];

    // Commit but do not block.
    stream->synchronize(at::mps::SyncType::COMMIT);
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
