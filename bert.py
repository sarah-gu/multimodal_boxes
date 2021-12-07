import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.models
from transformers import BertModel


class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
          ### New layers:
          self.linear1 = nn.Linear(768, 256)
          self.linear2 = nn.Linear(256, 3) ## 3 is the number of classes in this example

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids,
               attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

          linear2_output = self.linear2(linear2_output)

          return linear2_output

