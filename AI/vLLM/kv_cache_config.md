# KV Cache Config

随着模型结构越来越复杂，kv cache也变得越来越不一样。从最常见的full Attenion到后面的sliding window attention、mla、mamba，一个模型可能只有一种层级结构，也可能是多种层级结构混合，如qwen next。此外，在并行推理场景下，如流水线并行，不同worker因为分配层数不同，导致模型空间占用、kv cache形状也不同。综上，vLLM需要一个灵活的kv cache配置系统，以满足不同模型结构、不同并行场景下的需求。
注：本篇只专注于数据类，不关心具体方法实现

## KVCacheSpec——层级的kv表征
首先，针对不同的层级结构，kv cache需要有不同的表征方式。vllm里用KVCacheSpec来描述和kv cache相关的特征，基类如下：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    # vllm是以块为单位进行kv cache的存储
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same.")
        return copy.deepcopy(specs[0])
```

这里需要关心两个方法：
page_size_bytes：计算每个block需要占用空间大小
merge：判断不同层的KVCacheSpec是否可以合并成同一个表示

这里列举一个常见的多头注意力的实现：

```python
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)
```

除此以外还有其他实现：
- **SlidingWindowSpec**：滑动窗口注意力层
- **ChunkedLocalAttentionSpec**：分块局部注意力层
- **CrossAttentionSpec**：交叉注意力层（编码器-解码器模型）
- **EncoderOnlyAttentionSpec**：仅编码器的注意力层
- **MambaSpec**：Mamba 模型的特殊缓存规格

EngineCore初始KV Cache时，调用每个worker上的的ModelRunner的**get_kv_cache_spec**方法。此时model已经实例化，这里会根据每层layer的attention实现类，去实例化对应的KVCacheSpec，最后返回一个{layer_name: KVCacheSpec}的字典，这里举个最简单的例子，所有层都是一样的FullAttention：

```python
{
    'model.layers.0.self_attn.attn': FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16), 
    'model.layers.1.self_attn.attn': FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16), 
    'model.layers.2.self_attn.attn': FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16), 
    'model.layers.3.self_attn.attn': FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
}
```

最后EngineCore会把所有worker上的结果采集成一个List[{layer_name: KVCacheSpec}]的数据结构

## KVCacheConfig——Worker级的kv表征

从worker角度看，它需要KVCacheConfig记录些什么呢？
1. worker上能分多少块，这个涉及后面调度
2. 有多少layer需要分配kv cache，又分别需要多大空间，这涉及后面kv cache的分配和映射
3. 需要分配kv cache的layer们都是什么样的KVCacheSpec，这涉及了对应layer kv cache的shape

从上述角度，就可以很好的理解KVCacheConfig的结构了：
```python
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """
    """The number of KV cache blocks"""
    num_blocks: int
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_tensors: list[KVCacheTensor]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """
    kv_cache_groups: list[KVCacheGroupSpec]


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """
    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
# 一个KVCacheTensor就对应着需要在设备上划分一块size长度的空间作为kv cache供shared_by里面的layers使用


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
# 一个KVCacheGroupSpec就对应着layer_names里面的layers可以使用同一个KVCacheSpec表征，说白了就是结构一样的
```
### 计算num_blocks

在采集完每个worker上的{layer_name: KVCacheSpec}后，EngineCore会调用**determine_available_memory**方法，去获取每个worker上可用于kv cache的空间。
有了{layer_name: KVCacheSpec}，我们能够计算出一个block size下所有layer的KVCache需要的空间，用worker上的可用空间除一下就能得到num_blocks了

### 计算kv_cache_tensors

有了num_blocks，根据每个layer的KVCacheSpec，我们能够计算出一个block size下每个layer的KVCache需要的空间，乘一下就可以得到计算kv_cache_tensors的size

### 计算kv_cache_groups

通过KVCacheSpec的merge功能就可以将同构的layers合并成一个KVCacheGroupSpec

