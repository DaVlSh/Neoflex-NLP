import torch
class BERT_Arch(torch.nn.Module):

   def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = torch.nn.Dropout(0.1)
      self.relu = torch.nn.ReLU()
      self.fc1 = torch.nn.Linear(768, 512)
      self.fc2 = torch.nn.Linear(512, 2)
      self.softmax = torch.nn.LogSoftmax(dim=1)

   def forward(self, sent_id, mask):
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x