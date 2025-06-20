# 基于Embedding语义拆分文档

## 📖 参考资料
> **来源**: [微信文章 - 基于embedding语义拆分文档](https://mp.weixin.qq.com/s/saEr5vNLw-gu9xRUwJLqsg)

## 🎯 核心思路

基于embedding的语义拆分主要包含以下步骤：

1. **细粒度拆分文档**
   - 以句子或很短的切分长度为单位进行拆分

2. **计算embedding向量**
   - 为每个片段计算对应的embedding向量

3. **基于相似度重组**
   - 根据各个片段的embedding相似度进行重新组合

## 🛠️ 实现方法

### 方法一：带位置奖励的层次聚类

对embedding向量进行层次聚类，同时引入位置奖励机制。

**核心问题**：
- 特别短的句子单独切分可能导致语义改变
- 例如："But because I chose to split on sentences, there was an issue with small short sentences after a long one. You know?"
- 其中"You know?"这样的短句单独切开计算相似度显然不合适

**解决方案**：
- 引入**位置奖励**：对可能挨在一起的句子在切分时给予奖励
- 让语义相关且位置相近的句子更容易聚合在一起

**缺点**：
- ⚠️ 调参过程缓慢
- ⚠️ 不是最优解决方案

### 方法二：在连续句子中寻找切分点 ⭐（推荐）

通过计算相邻句子间的语义距离来确定切分点。

**核心算法**：
1. 计算相邻句子的距离：`距离 = 1 - 余弦相似度`
2. 当距离突破特定**阈值**时，在此处添加切分点
3. 采用**滑动窗口**方式进行平滑处理（窗口大小通常为3）

**示例**：
```
sentence_1 ←─┐
sentence_2   ├─ 距离较近，保持在同一切片
sentence_3 ←─┘
            │ 距离突破阈值，在此处切分
sentence_4 ←─── 开始新的切片
```

**最终结果**：
- sentence_1、sentence_2、sentence_3 → 切片A
- sentence_4... → 切片B

**优势**：
- ✅ 保持语义连贯性
- ✅ 自动适应内容变化
- ✅ 计算效率高
- ✅ 参数调优相对简单

## 已有实现
1、langchain-experimental的SemanticChunker
2、参考langchain SemanticChunker实现的https://github.com/rango-ramesh/advanced-chunker/tree/main，仅有聚类实现，没有位置奖励
