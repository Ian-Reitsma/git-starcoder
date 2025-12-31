/*
 * Python bindings for custom FlashAttention Turing kernel
 *
 * COMPLETE IMPLEMENTATION - Both Forward and Backward passes
 */

#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declare CUDA functions
extern "C" void flash_attn_forward_cuda(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    float* softmax_lse,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream
);

extern "C" void flash_attn_backward_cuda(
    const half* dout,
    const half* q,
    const half* k,
    const half* v,
    const half* out,
    const float* softmax_lse,
    half* dq,
    half* dk,
    half* dv,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream
);

// Python wrapper for forward pass
std::tuple<torch::Tensor, torch::Tensor> flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal
) {
    // Allocate output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kHalf)
        .device(q.device());
    auto lse_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q.device());

    torch::Tensor out = torch::empty({batch, heads, seq_len, head_dim}, options);
    torch::Tensor softmax_lse = torch::empty({batch, heads, seq_len}, lse_options);

    // Get CUDA stream
    cudaStream_t stream = 0;

    // Call CUDA kernel
    flash_attn_forward_cuda(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        softmax_lse.data_ptr<float>(),
        batch,
        heads,
        seq_len,
        head_dim,
        scale,
        is_causal,
        stream
    );

    return std::make_tuple(out, softmax_lse);
}

// Python wrapper for backward pass
void flash_attn_backward(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    torch::Tensor dq,
    torch::Tensor dk,
    torch::Tensor dv,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal
) {
    // Get CUDA stream (use default stream for simplicity)
    cudaStream_t stream = 0;  // Default CUDA stream

    // Call CUDA kernel
    flash_attn_backward_cuda(
        reinterpret_cast<const half*>(dout.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(out.data_ptr<at::Half>()),
        softmax_lse.data_ptr<float>(),
        reinterpret_cast<half*>(dq.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dk.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dv.data_ptr<at::Half>()),
        batch,
        heads,
        seq_len,
        head_dim,
        scale,
        is_causal,
        stream
    );
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward_cuda", &flash_attn_forward, "FlashAttention Turing forward (CUDA)");
    m.def("flash_attn_backward_cuda", &flash_attn_backward, "FlashAttention Turing backward (CUDA)");
}
