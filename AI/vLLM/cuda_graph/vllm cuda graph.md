# CUDA Graph 简介

CUDA Graph（CUDA 图）是一种 NVIDIA CUDA 技术，用于优化 GPU 操作的执行效率。

## **核心概念**

CUDA Graph 是一种将一系列 CUDA 操作（kernels）预先捕获（capture）并保存为图结构的技术，然后可以重复执行这个图。这避免了每次执行时的 CPU 开销和 kernel 启动延迟。

**主要优势：**
1. **降低 CPU 开销**：减少了 CPU-GPU 通信成本
2. **减少 kernel 启动延迟**：GPU kernels 可以连续执行，无需每次都从 CPU 发起
3. **提升吞吐量**：特别适合重复执行相同计算模式的场景

## **工作原理**

传统方式 vs CUDA Graph：

**传统方式**：
```
每次推理 → CPU 发起 kernel 1 → GPU 执行 → CPU 发起 kernel 2 → GPU 执行 → ...
```

**CUDA Graph 方式**：
```
第一次：捕获所有操作到图中
之后：直接重放(replay)整个图，一次性执行所有操作
```

## **适用场景**

✅ **适合使用 CUDA Graph 的场景：**
- 推理服务（固定的模型结构）
- Decode 阶段（token-by-token 生成）
- 重复执行相同形状的计算

❌ **不适合的场景：**
- 动态控制流（if-else 分支）
- 可变的计算图
- 需要频繁与 CPU 交互的操作

# **在 vLLM 中的应用**

vLLM 对 CUDA Graph 进行了深度优化，提供了多种模式：

## **CUDA Graph 模式**

```python
# 0.11.1版本
from vllm.config import CUDAGraphMode

# 可用模式：
# 1. NONE - 关闭 CUDA Graph
# 2. PIECEWISE - 分段捕获（除了 attention 等不兼容操作）
# 3. FULL - 完整捕获（包括 attention）
# 4. FULL_DECODE_ONLY - 仅对 decode 阶段使用完整捕获
# 5. FULL_AND_PIECEWISE - 混合模式（默认，性能最佳）
```

## attention的捕获
vLLM原本默认的CUDA Graph捕获模式为PIECEWISE，即跳过attention层的cuda graph捕获。足以说明attention层的捕获多让人头大。
以下是CUDA Graph的捕获难点：
1. **动态 Kernel 选择**
   - 部分attention backend的 Prefill 阶段（处理输入 prompt）和 Decode 阶段（逐 token 生成）需要不同的 kernel
   - 有些实现会根据 `query_len` 动态选择优化的 kernel
   - CUDA Graph 要求图捕获时的执行路径和重放时完全一致

2. **动态形状和内存访问模式**
   - Batch 中每个序列的长度可能不同
   - Paged Attention 的内存访问依赖于动态的 block table
   - 这些动态性可能导致不同的 GPU 操作序列

3. **CPU-GPU 交互**
   - 某些 attention backend 需要在执行前进行调度计算
   - 如果涉及 CPU 侧的决策，就无法完全捕获到 CUDA Graph

### 不同 Attention Backend 的 CUDA Graph 支持级别

vLLM 定义了四个支持级别：

```python
class AttentionCGSupport(enum.Enum):
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""
```

### **各 Backend 的实际支持情况**

| Attention Backend | 支持级别 | 原因 |
|:---|:---|:---|
| **FlashAttention v3** | `ALWAYS` ✅ | 使用统一的 kernel 路径处理所有情况 |
| **FlashAttention v2** | `UNIFORM_BATCH` | 对 `max_query_len=1` 有特殊的 packed-GQA 处理 |
| **Triton Attention** | `ALWAYS` | 完全支持，但不同场景性能差异大 |
| **FlashInfer** | `UNIFORM_SINGLE_TOKEN_DECODE` | 只在纯 decode (query_len=1) 时支持 |
| **FlashMLA** | `UNIFORM_BATCH` | 支持统一 query 长度的 batch |
| **Cascade Attention** | `NEVER` ❌ | 使用动态控制流，无法捕获 |

像flashinfer使用了 wrapper 机制来管理不同的执行路径：
- `BatchPrefillWithPagedKVCacheWrapper` - prefill 专用
- `BatchDecodeWithPagedKVCacheWrapper` - decode 专用  
- `MultiLevelCascadeAttentionWrapper` - cascade attention
这种随着batch内场景变化导致的kernel变化，就注定了无法使用一个统一的CUDA Graph来捕获。

### attention能够捕获CUDA Graph的前提

1. **固定的输入张量地址**：
   - `K_Buffer` / `k_cache` 的地址固定
   - `Req_to_tokens` / `block_table` 的地址固定
   - **只有张量内容变化，地址不变**

2. **GPU 侧全部完成**：
   - 所有的查表、地址计算、内存访问都在 GPU kernel 内完成
   - 没有 CPU-GPU 交互
   - 没有动态 kernel 选择

3. **Kernel 序列固定**：
   - 相同 batch size 下，调用的 kernel 完全相同
   - 虽然每次访问的数据不同（通过 block_table 间接寻址），但执行路径相同

一开始可能会觉得vllm使用pagedattention导致每次kv cache存储、寻址都不同，无法捕获。
但实际上，在每次execute model前，model runner都会在prepare_inputs时提前把像kv cache的存储地址(sloting_mapping)、每个请求kv块的映射(block_table)等提前计算好，并存储在固定的张量中。这些带有调度信息的张量传递给算子后，看上去动态、可变的过程实际上都在算子里完成了。从数据流、算子执行的角度看，整个过程是固定的。

相对来讲，纯decode的过程更能满足上述要求，因为kernel固定，每个请求计算的长度固定(1或者1+num_spec_tokens)，像mask什么的都固定了。如果包含了prefill，kernel，mask什么的都不是很固定，捕获难度相比于纯decode要大很多。
因此，很久之前vllm默认的CUDA Graph捕获模式为PIECEWISE，即跳过attention层的cuda graph捕获，一刀切以节省开发成本。
后面优化后，vllm也支持了FULL_AND_PIECEWISE模式，即在纯decode时使用FULL模式，在包含prefill时使用PIECEWISE模式，以提升纯decode的性能，因为纯decode的场景在vllm中占比很大。

### 📊 **数据流示意图**

```
CUDA Graph 捕获时：
固定地址的 block_table[0, 1, 2] → GPU Kernel 读取 → 访问 K_cache 物理块 [0, 1, 2]

CUDA Graph 重放时：
同一地址的 block_table[5, 3, 7] → 相同 GPU Kernel → 访问 K_cache 物理块 [5, 3, 7]
     ↑                                  ↑
  内容变了                          执行路径相同 ✅
  地址不变 ✅                        可以捕获！
```