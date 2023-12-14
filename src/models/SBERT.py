import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer


class SBERT(nn.Module):
    def __init__(self, model_name: str, pooling_type: str = "mean", device="cpu"):
        """The SBERT model adapted to support multiple pooling options."""
        super(SBERT, self).__init__()
        self.pooling_type = pooling_type

        if self.pooling_type not in ["cls", "max", "mean"]:
            raise Exception(f"Unsupported pooling type: {self.pooling_type}")

        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Generates the document embedding.
        Args:
            text (str): The text to be embedded.
        """

        encodings = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # get query embeddings
        embeds = self.model(**encodings)

        if self.pooling_type == "cls":
            pooling_func = cls_pooling
        elif self.pooling_type == "max":
            pooling_func = max_pooling
        elif self.pooling_type == "mean":
            pooling_func = mean_pooling

        embeds = pooling_func(embeds, encodings["attention_mask"])
        # normalize the vector before calculating
        embeds = f.normalize(embeds, p=2, dim=1)
        return embeds.cpu()


# ===============================================
# Helper Functions
# ===============================================


def cls_pooling(model_output, attn_mask):
    return model_output[0][:, 0]


def max_pooling(model_output, attn_mask):
    token_embeds = model_output[0]
    input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    token_embeds[input_mask_expanded == 0] = -1e9  # padding tokens to large negatives
    return torch.max(token_embeds, 1)[0]


def mean_pooling(model_output, attn_mask):
    token_embeds = model_output[0]
    input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    sum_embeddings = torch.sum(token_embeds * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
