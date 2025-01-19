import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from joblib import load

import torch
from transformers import BertTokenizer

# Загрузка модели
model = load('model.joblib')

# Загрузка данных сотрудников из CSV
input_file = 'input.csv'
output_file = 'output.csv'

def prediction(input_file):
   input = pd.read_csv(input_file, index_col=[0]).iloc[:, 0]

   tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

   tokens_test = tokenizer.batch_encode_plus(
      input,
      max_length=15,
      padding='max_length',
      truncation=True
   )

   test_mask = torch.tensor(tokens_test['attention_mask'])
   test_seq = torch.tensor(tokens_test['input_ids'])

   list_seq = np.array_split(test_seq, 50)
   list_mask = np.array_split(test_mask, 50)

   predictions = []
   for num, elem in enumerate(list_seq):
      with torch.no_grad():
         preds = model(elem, list_mask[num])
         predictions.append(preds.detach().cpu().numpy())

   all_predictions = np.concatenate(predictions, axis=0)
   predicted_classes = np.argmax(all_predictions, axis=1)

   return predicted_classes

def process_data(input_file, output_file, predicted_classes):
   # Чтение данных
   df = pd.read_csv(input_file)

   # Предсказания
   df['satisfaction'] = predicted_classes

   # Конвертируем предсказания в понятные метки
   df['satisfaction'] = df['satisfaction'].map({1: 'ответы, требующие внимания', 0: 'ответы, не требующие внимания'})

   # Сохранение размеченного файла
   df.to_csv(output_file, index=False)


def load_file():
   input_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
   if not input_file:
      return

   predicted_classes = prediction(input_file)

   output_file = "output.csv"  # Имя выходного файла
   try:
      process_data(input_file, output_file, predicted_classes)
      messagebox.showinfo("Успех", f"Результаты сохранены в {output_file}")
   except Exception as e:
      messagebox.showerror("Ошибка", str(e))


root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = 400
window_height = 300

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title("Приложение для анализа удовлетворенности сотрудников их задачами")

load_button = tk.Button(root, text="Загрузить CSV", command=load_file, width=20, height=5,)
load_button.pack(pady=20)
load_button.place(relx=0.5, rely=0.5, anchor='center')

root.mainloop()