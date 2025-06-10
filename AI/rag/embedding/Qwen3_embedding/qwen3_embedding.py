#coding:utf8
from typing import List, Union
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.utils import is_flash_attn_2_available


class Qwen3Embedding():
    """
    Qwen3Embedding类：用于生成文本嵌入向量的类
    使用Qwen3模型将文本转换为向量表示，支持查询和文档的编码
    """
    def __init__(self, model_name_or_path, instruction=None,  use_fp16: bool = True, use_cuda: bool = False, max_length=8192):
        """
        初始化Qwen3Embedding模型
        
        参数:
            model_name_or_path: 模型路径或名称
            instruction: 任务指令，用于指导模型如何处理查询
            use_fp16: 是否使用半精度浮点数
            use_cuda: 是否使用GPU
            max_length: 输入序列的最大长度
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        # 根据是否支持flash attention和是否使用GPU来初始化模型
        if is_flash_attn_2_available() and use_cuda:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
        if use_cuda:
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side='left')
        self.max_length=max_length
    
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        """
        对最后一层的隐藏状态进行池化操作
        
        参数:
            last_hidden_states: 最后一层的隐藏状态
            attention_mask: 注意力掩码
            
        返回:
            池化后的张量
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        生成详细的指令文本
        
        参数:
            task_description: 任务描述
            query: 查询文本
            
        返回:
            格式化后的指令文本
        """
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction=None, dim: int = -1):
        """
        将文本编码为向量表示
        
        参数:
            sentences: 输入文本或文本列表
            is_query: 是否为查询文本
            instruction: 自定义指令
            dim: 输出向量的维度，-1表示使用模型默认维度
            
        返回:
            归一化后的文本向量表示
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs.to(self.model.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
            output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
            if dim != -1:
                output = output[:, :dim]
            output  = F.normalize(output, p=2, dim=1)
        return output

if __name__ == "__main__":
    # 示例代码：展示如何使用Qwen3Embedding类
    model_path = "/root/code/Qwen3-Embedding-0.6B"
    model = Qwen3Embedding(model_path)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    dim = 1024
    # 编码查询和文档
    query_outputs = model.encode(queries, is_query=True, dim=dim)
    doc_outputs = model.encode(documents, dim=dim)
    print('query outputs', query_outputs)
    print('doc outputs', doc_outputs)
    # 计算相似度分数
    scores = (query_outputs @ doc_outputs.T) * 100
    print(scores.tolist())