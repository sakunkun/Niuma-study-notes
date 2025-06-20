import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen3Reranker:
    """
    基于Qwen3模型的文档重排序器
    用于评估文档与查询的相关性，并给出相关性分数
    """
    def __init__(
        self,
        model_name_or_path: str,  # 模型路径或名称
        max_length: int = 4096,   # 最大序列长度
        instruction=None,         # 自定义指令
        attn_type='causal',       # 注意力类型
    ) -> None:
        self.max_length=max_length
        # 初始化tokenizer，设置padding在左侧
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side='left')
        # 加载预训练模型，使用float16精度
        self.lm = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16).eval()
        # 获取"yes"和"no"对应的token ID
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # 设置提示模板
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # 将提示模板转换为token序列
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.instruction = instruction
        if self.instruction is None:
            self.instruction = "Given the user query, retrieval the relevant passages"

    def format_instruction(self, instruction, query, doc):
        """
        格式化输入指令、查询和文档
        返回格式化的字符串
        """
        if instruction is None:
            instruction = self.instruction
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        """
        处理输入数据
        将输入文本转换为模型所需的token格式
        """
        out = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        # 为每个输入添加前缀和后缀token
        for i, ele in enumerate(out['input_ids']):
            out['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        # 进行padding并转换为tensor
        out = self.tokenizer.pad(out, padding=True, return_tensors="pt", max_length=self.max_length)
        # 将数据移动到模型所在的设备
        for key in out:
            out[key] = out[key].to(self.lm.device)
        return out

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        """
        计算模型输出的logits
        返回相关性分数
        """
        # 获取模型输出的logits
        batch_scores = self.lm(**inputs).logits[:, -1, :]
        # 提取"yes"和"no"对应的logits
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        # 将两个logits堆叠在一起
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        # 计算log_softmax
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        # 计算最终的相关性分数
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_scores(
        self,
        pairs,  # 查询-文档对列表
        instruction=None,  # 自定义指令
        **kwargs
    ):
        """
        计算查询-文档对的相关性分数
        返回每个对的相关性分数列表
        """
        # 格式化输入
        pairs = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        # 处理输入数据
        inputs = self.process_inputs(pairs)
        # 计算分数
        scores = self.compute_logits(inputs)
        return scores

if __name__ == '__main__':
    # 测试代码
    model = Qwen3Reranker(model_name_or_path='/root/code/Qwen3-Reranker-0.6B', instruction="Retrieval document that can answer user's query", max_length=2048)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    pairs = list(zip(queries, documents))
    instruction="Given the user query, retrieval the relevant passages"
    new_scores = model.compute_scores(pairs, instruction)
    print('scores', new_scores)