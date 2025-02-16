import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import torch
import threading
import time
from transformers import BertTokenizer
from joblib import load

class SatisfactionModelApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Система предсказания удовлетворенности сотрудников")

        self.center_window(400, 200)

        self.label = ttk.Label(master, text="Загрузите CSV файл с ответами сотрудников:")
        self.label.pack(pady=10)

        self.load_button = ttk.Button(master, text="Загрузить CSV файл", command=self.load_file)
        self.load_button.pack(pady=10)

        self.progress = None
        self.model = None
        self.cancelled = False

    def center_window(self, width, height):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.master.geometry(f"{width}x{height}+{x}+{y}")

    def load_model(self):
        self.model = load('model.joblib')
        self.model.eval()

    def prediction(self, input_file):
        input_data = pd.read_csv(input_file, index_col=[0]).iloc[:, 0]

        tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

        tokens_test = tokenizer.batch_encode_plus(
            input_data.tolist(),
            max_length=15,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        test_mask = tokens_test['attention_mask']
        test_seq = tokens_test['input_ids']

        with torch.no_grad():
            preds = self.model(test_seq, test_mask)
            predicted_classes = torch.argmax(preds, dim=1).cpu().numpy()

        return predicted_classes

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.show_loading_screen()
            self.cancelled = False
            threading.Thread(target=self.process_file, args=(file_path,)).start()

    def show_loading_screen(self):
        self.loading_window = tk.Toplevel(self.master)
        self.loading_window.title("Загрузка модели")

        self.center_window_loading(300, 200)

        loading_label = ttk.Label(self.loading_window, text="Загрузка модели, пожалуйста подождите...")
        loading_label.pack(pady=20)
        self.progress = ttk.Progressbar(self.loading_window, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()

        self.cancel_button = ttk.Button(self.loading_window, text="Отмена", command=self.cancel_loading)
        self.cancel_button.pack(pady=10)

    def center_window_loading(self, width, height):
        screen_width = self.loading_window.winfo_screenwidth()
        screen_height = self.loading_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.loading_window.geometry(f"{width}x{height}+{x}+{y}")

    def cancel_loading(self):
        self.cancelled = True
        self.loading_window.destroy()
        messagebox.showinfo("Отмена", "Загрузка отменена.")

    def process_file(self, file_path):
        self.load_model()
        time.sleep(1)

        if not self.cancelled:
            predictions = self.prediction(file_path)
            data = pd.read_csv(file_path)
            data['Satisfaction'] = predictions

            output_file = "output_predictions.csv"
            data.to_csv(output_file, index=False)

            self.loading_window.destroy()
            messagebox.showinfo("Готово", f"Результаты сохранены в файле: {output_file}")
        else:
            self.loading_window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SatisfactionModelApp(root)
    root.mainloop()