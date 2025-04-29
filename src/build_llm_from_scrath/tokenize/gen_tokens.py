import re
import urllib.request
import os


def download_file():
   url = ("https://raw.githubusercontent.com/rasbt/"
          "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
          "the-verdict.txt")
   
   file_path = "../data/the-verdict.txt"
   if not os.path.exists(file_path):
      urllib.request.urlretrieve(url, file_path)
   
   return file_path


def build_vocab(text):
   result = re.split(r'([,.;?_!"()\']|--|\s)', text)
   result = [item.strip() for item in result if item.strip()]
   all_words = sorted(set(result))
   all_words.extend("<unk>", "<|endoftext|>")
   vocab = {token: i for i, token in enumerate(all_words)}
   return vocab


class SimpleTokenizer:
   def __init__(self, vocab):
      self.str_to_int = vocab
      self.int_to_str = {i:s for s, i in vocab.items()}
   
   def encode(self, text):
      proceed = re.split(r'([,.;?_!"()\']|--|\s)', text)
      proceed = [
         item.strip() for item in proceed if item.strip()
      ]
      proceed = [i if i in self.str_to_int else "<unk>" for i in proceed]
      ids = [self.str_to_int[s] for s in proceed]
      return ids

   def decode(self, ids):
      text = " ".join(self.int_to_str[i] for i in ids)
      text = re.sub(r'[,.;?_!"()\']', r'\1', text)
      return text


if __name__ == '__main__':
   
   # 1. download file
   with open(download_file(), "r", encoding="utf-8") as f:
      raw_text = f.read()
   
   # print("The total number of character:", len(raw_text))
   # print(raw_text[:99])
   
   # 2. build vocabulary
   index = 0
   vocab = build_vocab(raw_text)
   for i in vocab.keys():
      print(i, ":", vocab[i])
      if index < 10:
         index += 1
      else:
         break
   
   #  构建分词器
   simpleTokenizer = SimpleTokenizer(vocab)
   