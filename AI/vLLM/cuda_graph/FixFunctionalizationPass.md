## FixFunctionalizationPass 的作用

### 核心目的
**去函数化（Defunctionalization）某些节点，以避免冗余的张量拷贝。**

### 背景知识

在 PyTorch 的编译过程中，为了保持函数式编程的纯粹性，某些会修改输入张量的操作（in-place operations）会被自动"函数化"（functionalized）。这意味着：

1. **原始操作**：直接修改输入张量（in-place）
2. **函数化后**：创建新的张量副本，不修改原始输入

虽然函数化保证了函数的纯粹性，但会带来额外的内存拷贝开销。

### 这个 Pass 做什么

`FixFunctionalizationPass` 会识别图中被 `auto_functionalized` 包装的特定操作，然后将它们"去函数化"，恢复为原始的 in-place 操作，从而：

- ✅ **减少内存拷贝**
- ✅ **提高性能**
- ⚠️ **代价**：这些节点可能看起来像"死代码"（因为它们直接修改输入），所以之后不能运行 DCE（死代码消除）

### 处理的具体操作

从代码中可以看到，这个 pass 针对以下 vLLM 自定义算子进行去函数化：

```49:84:vllm/compilation/fix_functionalization.py
if at_target == torch.ops._C.rotary_embedding.default:
                query = kwargs['query']
                mm_node = query.args[0].args[0]

                # rotary_embedding is a special case: the two mutating inputs
                # are query and key, which are slices of mm_node.
                # While functionalized, results at[1] and at[2] are scattered
                # back into mm_node. After de-functionalization, we can just
                # use mm_node directly.
                for idx, user in self.getitem_users(node).items():
                    for user_of_getitem in user.users:
                        if is_func(user_of_getitem,
                                   torch.ops.aten.slice_scatter.default):
                            user_of_getitem.replace_all_uses_with(mm_node)
                            self._remove(user_of_getitem)
                    self._remove(user)

                self.insert_defunctionalized(graph, node)
                self._remove(node)

            # rms_norm replacements avoid the most copies for LLaMa.
            elif at_target == torch.ops._C.fused_add_rms_norm.default:
                mutated_args = {1: 'input', 2: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.fused_add_rms_norm_static_fp8_quant.default:  # noqa: E501
                mutated_args = {1: 'result', 2: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.rms_norm_dynamic_per_token_quant.default:  # noqa: E501
                mutated_args = {1: 'result', 2: 'scale', 3: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target in [
                    torch.ops._C.rms_norm.default,
                    torch.ops._C.rms_norm_static_fp8_quant.default,
            ]:
                mutated_args = {1: 'result'}
                self.defunctionalize(graph, node, mutated_args)
```

主要包括：
- **rotary_embedding**（旋转位置编码）
- **rms_norm** 系列（RMS 归一化，对 LLaMA 模型特别重要）
- **silu_and_mul** 系列（激活函数）

### 工作流程

1. **遍历图节点**：找到所有 `auto_functionalized` 包装的节点
2. **识别目标操作**：检查是否是需要去函数化的特定操作
3. **替换使用者**：将使用函数化输出的节点替换为直接使用被修改的输入
4. **插入去函数化节点**：创建新的直接调用原始函数的节点
5. **删除旧节点**：移除函数化节点和相关的 getitem 节点

### 示例

假设有一个函数化的 `rms_norm` 操作：

**函数化前的概念**：
```python
result = torch.empty(...)
rms_norm(input, result)  # in-place 修改 result
```

**函数化后**：
```python
auto_functionalized_node = auto_functionalized(rms_norm, input=input, result=result)
result_copy = auto_functionalized_node[1]  # 返回的是副本
```

**去函数化后**：
```python
rms_norm(input, result)  # 恢复 in-place 操作
# 直接使用 result，无需拷贝
```

### 重要注意事项

```21:27:vllm/compilation/fix_functionalization.py
"""
    This pass defunctionalizes certain nodes to avoid redundant tensor copies.
    After this pass, DCE (dead-code elimination) should never be run,
    as de-functionalized nodes may appear as dead code.

    To add new nodes to defunctionalize, add to the if-elif chain in __call__.
    """
```

- ⚠️ 运行此 pass 后，**不能再运行 DCE**（死代码消除）
- 📝 要添加新的去函数化操作，只需在 `__call__` 方法的 if-elif 链中添加

### 性能影响

注释中特别提到：
```python
# rms_norm replacements avoid the most copies for LLaMa.
```

这个 pass 对 LLaMA 等模型特别重要，因为 RMS 归一化在这些模型中被频繁使用，避免拷贝可以显著提升性能。

---

## 详细解释：从 PyTorch 层面理解去函数化的替换过程

### 一、PyTorch 的 auto_functionalize 机制

首先理解 PyTorch 为什么要做函数化：

#### 1. **原始的 in-place 操作**
```python
# 假设有一个自定义算子 rms_norm
torch.ops._C.rms_norm(input, result)  # result 被原地修改
```

这个操作会**直接修改** `result` 张量的内容。

#### 2. **函数化后的图结构**

当 PyTorch 编译器遇到这种 in-place 操作时，会自动将其包装成 `auto_functionalized` 节点：

```python
# 伪代码表示函数化后的图
node = auto_functionalized(
    torch.ops._C.rms_norm,  # 原始函数
    input=input_tensor,
    result=result_tensor    # 会被克隆
)

# auto_functionalized 返回一个元组
# node[0] = 原始返回值（如果有的话）
# node[1] = 修改后的 result 的副本（新张量！）
```

**关键点**：`auto_functionalized` 内部会：
1. **克隆** `result` 张量（创建副本）
2. 在副本上调用原始函数
3. 返回修改后的副本

这样保证了函数式编程的纯粹性，但代价是**额外的内存拷贝**。

### 二、FX Graph 中的实际结构

让我用一个具体例子说明。假设有如下代码：

```python
# 原始代码
result = torch.empty(shape)
torch.ops._C.rms_norm(input, result)
use_result(result)
```

#### **函数化后的 FX Graph**：

```
节点1: result = torch.empty(shape)
节点2: af_node = call_function[auto_functionalized](
           args=(torch.ops._C.rms_norm,),
           kwargs={'input': input, 'result': result}
       )
节点3: getitem_0 = af_node[0]  # 原始返回值（通常是 None）
节点4: getitem_1 = af_node[1]  # 修改后的 result 副本
节点5: use_result(getitem_1)   # 使用副本
```

**问题**：`getitem_1` 是 `result` 的副本，这意味着：
- 内存开销：需要额外的内存存储副本
- 拷贝开销：需要时间复制数据

### 三、FixFunctionalizationPass 的替换过程

现在来看 `FixFunctionalizationPass` 如何优化这个问题：

#### **步骤 1：识别 auto_functionalized 节点**

```python
for node in graph.nodes:
    if not is_func(node, auto_functionalized):
        continue  # 跳过非函数化节点
    
    kwargs = node.kwargs
    at_target = node.args[0]  # 获取原始函数
```

#### **步骤 2：找到 getitem 使用者**

```python
def getitem_users(self, node: torch.fx.Node) -> dict[int, torch.fx.Node]:
    """
    返回所有通过 operator.getitem 访问 auto_functionalized 节点输出的节点
    """
    users = {}
    for user in node.users:
        if is_func(user, operator.getitem):
            idx = user.args[1]  # getitem 的索引
            users[idx] = user
    return users
```

对于上面的例子：
- `users[0]` = 节点3 (getitem_0)
- `users[1]` = 节点4 (getitem_1)

#### **步骤 3：替换 getitem 使用者**

这是**核心步骤**！

```python
def replace_users_with_mutated_args(self, node, mutated_args):
    """
    mutated_args = {1: 'result'}  # 索引1对应的输出应该用 'result' 参数替换
    """
    for idx, user in self.getitem_users(node).items():
        arg = mutated_args[idx]
        arg = node.kwargs[arg] if isinstance(arg, str) else arg
        # 关键：将 getitem_1 的所有使用替换为原始的 result
        user.replace_all_uses_with(arg)
        self._remove(user)
```

**具体到我们的例子**：
```python
mutated_args = {1: 'result'}
# idx=1, user=节点4 (getitem_1)
# arg = node.kwargs['result'] = 原始的 result 张量
# 将节点4的所有使用替换为原始 result
节点4.replace_all_uses_with(result)
```

**替换后的图**：
```
节点1: result = torch.empty(shape)
节点2: af_node = call_function[auto_functionalized](...)
节点3: getitem_0 = af_node[0]
节点4: getitem_1 = af_node[1]  # 已无使用者
节点5: use_result(result)      # 现在直接使用原始 result！
```

#### **步骤 4：插入去函数化节点**

```python
def insert_defunctionalized(self, graph, node, args=None):
    with graph.inserting_before(node):
        function = node.args[0]  # torch.ops._C.rms_norm
        if args is None:
            # 直接调用原始函数，使用原始参数
            graph.call_function(function, kwargs=node.kwargs)
```

**插入后的图**：
```
节点1: result = torch.empty(shape)
节点1.5: call_function[torch.ops._C.rms_norm](
            kwargs={'input': input, 'result': result}
        )  # 新插入的去函数化调用！
节点2: af_node = call_function[auto_functionalized](...)  # 将被删除
节点3: getitem_0 = af_node[0]  # 将被删除
节点4: getitem_1 = af_node[1]  # 将被删除
节点5: use_result(result)      # 使用原始 result
```

#### **步骤 5：删除旧节点**

```python
self._remove(node)  # 删除 af_node
# 之前已经删除了 getitem 节点
```

**最终的图**：
```
节点1: result = torch.empty(shape)
节点1.5: call_function[torch.ops._C.rms_norm](
            kwargs={'input': input, 'result': result}
        )
节点5: use_result(result)
```

### 四、特殊案例：rotary_embedding

`rotary_embedding` 更复杂，因为它涉及到 **slice** 和 **slice_scatter** 操作：

#### **函数化后的图**：
```python
# 原始：query 和 key 是 mm_node 的切片
query = mm_node[:, :seq_len, :]
key = mm_node[:, seq_len:, :]
rotary_embedding(query, key, ...)  # in-place 修改 query 和 key
```

**函数化后**：
```
1. mm_node = matmul(...)
2. query = slice(mm_node, ...)
3. key = slice(mm_node, ...)
4. af_node = auto_functionalized(rotary_embedding, query=query, key=key, ...)
5. new_query = af_node[1]  # 修改后的 query 副本
6. new_key = af_node[2]    # 修改后的 key 副本
7. mm_node_1 = slice_scatter(mm_node, new_query, ...)   # 将 new_query 散射回去
8. mm_node_2 = slice_scatter(mm_node_1, new_key, ...)   # 将 new_key 散射回去
9. use(mm_node_2)
```

**问题**：
- `new_query` 和 `new_key` 是副本
- 需要两次 `slice_scatter` 操作将副本散射回原张量
- 大量的内存拷贝！

**去函数化后**：
```python
for idx, user in self.getitem_users(node).items():
    for user_of_getitem in user.users:
        if is_func(user_of_getitem, torch.ops.aten.slice_scatter.default):
            # 直接用原始的 mm_node 替换 slice_scatter 的结果
            user_of_getitem.replace_all_uses_with(mm_node)
```

**最终的图**：
```
1. mm_node = matmul(...)
2. query = slice(mm_node, ...)
3. key = slice(mm_node, ...)
4. rotary_embedding(query, key, ...)  # 直接 in-place 修改！
5. use(mm_node)  # 直接使用原始 mm_node，它已经被修改了
```

### 五、为什么这样做是安全的？

**关键理解**：在 vLLM 的使用场景中，这些被修改的张量：

1. **不会被后续读取旧值**：去函数化后，原始张量被修改，但代码逻辑保证不会再读取旧值
2. **没有别名问题**：这些张量没有其他引用会观察到修改
3. **性能关键路径**：这些操作在推理的热路径上，避免拷贝能显著提升性能

**但是**：
```python
# After this pass, DCE (dead-code elimination) should never be run,
# as de-functionalized nodes may appear as dead code.
```

去函数化的节点看起来没有返回值被使用（因为是 in-place），DCE 可能会错误地删除它们！

### 六、总结对比

| 方面 | 函数化 | 去函数化 |
|------|--------|----------|
| **内存** | 需要副本 | 无需副本 |
| **拷贝次数** | 每次调用都拷贝 | 0 次拷贝 |
| **函数纯粹性** | ✅ 纯函数 | ❌ 有副作用 |
| **安全性** | ✅ 总是安全 | ⚠️ 需要保证正确性 |
| **性能** | 较慢 | 更快 |
| **DCE 兼容** | ✅ 兼容 | ❌ 不兼容 |

这就是为什么 `FixFunctionalizationPass` 对 vLLM 如此重要：在推理场景中，性能至关重要，而这个 pass 通过消除不必要的张量拷贝，显著提升了性能。

---

