from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from transformers.models.bert import BertModel
from transformers.models.roberta import RobertaModel
from transformers.models.electra import ElectraModel

from .modeling_nezha import NezhaModel


class PretrainedLanguageModelEmbedding(nn.Module):
    """
    加载预训练模型

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training. If this is `False`, the
        transformer weights are not updated during training.
    eval_mode: `bool`, optional (default = `False`)
        If this is `True`, the model is always set to evaluation mode (e.g., the dropout is disabled and the
        batch normalization layer statistics are not updated). If this is `False`, such dropout and batch
        normalization layers are only set to evaluation mode when when the model is evaluating on development
        or test data.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        max_length: int = 0,
        train_parameters: bool = True,
        projection_dim: Optional[int] = None,
        post_layer_norm_eps: Optional[float] = None,
        eval_mode: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if "nezha" in model_name:
            Model = NezhaModel
        elif "roberta" in model_name:
            Model = RobertaModel
        elif "electra" in model_name:
            Model = ElectraModel
        elif "bert" in model_name:  # bert, structbert, macbert
            Model = BertModel
        else:
            raise Exception("not supported model: ", model_name)

        self.transformer_model = Model.from_pretrained(model_name, **(transformer_kwargs or {}))
        if 'pooler' in self.transformer_model._modules:
            self.transformer_model.pooler = None

        if max_length > 512 and isinstance(self.transformer_model, NezhaModel):
            self.transformer_model.reset_relative_positions_embeddings(max_length)

        if projection_dim:
            self.projection = nn.Linear(
                self.transformer_model.config.hidden_size, projection_dim, False
            )  # the bias of linear makes the normalization ineffective and can lead to unstable training
            self._output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self._output_dim = self.transformer_model.config.hidden_size

        if post_layer_norm_eps:
            self.layer_norm = nn.LayerNorm(self._output_dim, post_layer_norm_eps)
        else:
            self.layer_norm = nn.Identity()

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "transformer_model":
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self):
        return self._output_dim

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:  # type: ignore
        transformer_output = self.transformer_model(
            token_ids, attention_mask=mask, token_type_ids=type_ids
        )  # return_dict
        outputs = self.layer_norm(self.projection(transformer_output.last_hidden_state))
        return outputs
