import torch
import torch.nn as nn
from span_marker import SpanMarkerModel


class MNER(nn.Module):
    def __init__(self, model_name: str, device="cpu"):
        """The MNER model using the SpanMarker model.
        Link: https://tomaarsen.github.io/SpanMarkerNER/
        """
        super(MNER, self).__init__()

        self.device = torch.device(device)
        self.model = SpanMarkerModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Generates the document embedding.
        Args:
            text (str): The text to be embedded.
        """

        entities = self.model.predict(text)

        return [(e["span"], e["label"]) for e in entities]
