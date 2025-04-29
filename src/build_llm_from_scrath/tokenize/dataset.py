import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import gen_tokens

class GPTDataset(Dataset):
   
   def __init__(self, text, tokenizer, max_length, stride):
      self.input_ids = []
      self.target_ids = []
      
      token_ids = tokenizer.encode(text)
      for i in range(0, len(token_ids) - max_length, stride):
         input_trunk = token_ids[i: i + max_length]
         target_trunk = token_ids[i + 1: i + 1 +max_length]
         self.input_ids.append(torch.Tensor(input_trunk))
         self.target_ids.append(torch.Tensor(target_trunk))
   
   def __len__(self):
      return len(self.input_ids)
   
   def __getitem__(self, idx):
      return self.input_ids[idx], self.target_ids[idx]

if __name__ == "__main__":
   tokenizer = tiktoken.get_encoding("gpt2")
   with open(gen_tokens.download_file(), "r", encoding="utf-8") as f:
      raw_text = f.read()
   dataset = GPTDataset(raw_text, tokenizer, 4, 1)
   print(len(dataset))
   
   dataloader = DataLoader(dataset=dataset,
                           batch_size=4,
                           shuffle=True,
                           num_workers=2)
   data_iter = iter(dataloader)
   first_batch = next(data_iter)
   print(first_batch[0])
   
   