# Qwen3 Embedding

项目地址：https://github.com/QwenLM/Qwen3-Embedding

## 简介

Qwen3 Embedding 系列模型专为文本表征、检索与排序任务设计，基于 Qwen3 基础模型进行训练，充分继承了 Qwen3 在多语言文本理解能力方面的优势。

## 主要特点

### 卓越的泛化性
Qwen3 Embedding 系列在多个下游任务评估中达到行业领先水平：
- 8B 参数规模的Embedding模型在MTEB多语言Leaderboard榜单中位列第一（截至 2025 年 6 月 5 日，得分 70.58）
- 性能超越众多商业 API 服务
- 排序模型在各类文本检索场景中表现出色，显著提升了搜索结果的相关性

### 灵活的模型架构
Qwen3 Embedding 系列提供从 0.6B 到 8B 参数规模的 3 种模型配置，以满足不同场景下的性能与效率需求：

- 开发者可以灵活组合表征与排序模块，实现功能扩展
- 支持以下定制化特性：
  1. 表征维度自定义：允许用户根据实际需求调整表征维度，有效降低应用成本
  2. 指令适配优化：支持用户自定义指令模板，以提升特定任务、语言或场景下的性能表现

### 全面的多语言支持
Qwen3 Embedding 系列支持超过 100 种语言，涵盖主流自然语言及多种编程语言。该系列模型具备：
- 强大的多语言检索能力
- 跨语言检索能力
- 代码检索能力
- 能够有效应对多语言场景下的数据处理需求

## 模型总览
| Model Type | Models | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------|---------|------|---------|-----------------|-------------------|-------------|------------------|
| Text Embedding | Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | Yes | Yes |
| Text Embedding | Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | Yes | Yes |
| Text Embedding | Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | Yes | Yes |
| Text Reranking | Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | - | - | Yes |
| Text Reranking | Qwen3-Reranker-4B | 4B | 36 | 32K | - | - | Yes |
| Text Reranking | Qwen3-Reranker-8B | 8B | 36 | 32K | - | - | Yes |
注：MRL Support 表示 Embedding 模型是否支持最终向量的自定义维度。Instruction Aware 表示 Embedding 或 Reranking 模型是否支持根据不同任务定制输入指令。

## 模型架构
Qwen3 Embedding 系列模型基于 Qwen3 基础模型开发，采用两种不同的架构设计：

### Embedding 模型架构
- 采用双塔结构设计
- 通过 LoRA 微调保留基础模型的文本理解能力
- 输入：单段文本
- 输出：使用最后一层 [EOS] 标记的隐藏状态向量作为文本的语义表示
```python
# 0.6B 参数规模的 Embedding 模型架构
Qwen3Model(
  (embed_tokens): Embedding(151669, 1024)
  (layers): ModuleList(
    (0-27): 28 x Qwen3DecoderLayer(
      (self_attn): Qwen3Attention(
        (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
        (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
        (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
      )
      (mlp): Qwen3MLP(
        (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
    )
  )
  (norm): Qwen3RMSNorm((1024,), eps=1e-06)
  (rotary_emb): Qwen3RotaryEmbedding()
)
```

### Reranking 模型架构
- 采用单塔结构设计
- 输入：文本对（如用户查询与候选文档）
- 输出：两个文本之间的相关性得分
```python
# 0.6B 参数规模的 Reranking 模型架构
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151669, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151669, bias=False)
)
```


这种架构设计使得模型能够高效地完成文本表征和相关性排序任务。