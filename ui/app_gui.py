# -*- coding: utf-8 -*-

"""Main GUI application for NLP co-occurrence exploration."""


import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
import ttkbootstrap as tb
from core.data_manager import DataManager


class NLPGuiApp:
    def __init__(self, master):
        self.master = master
        self.style = tb.Style('darkly')
        master.title('NLP Co-occurrence Explorer')
        master.geometry('980x760')

        self.dm = DataManager()
        self._build_ui()

    def _build_ui(self):
        pad = 8
        frame_top = tb.Frame(self.master)
        frame_top.pack(fill='x', padx=12, pady=8)

        tb.Label(frame_top, text='⚙️ تنظیمات', font=('Tahoma', 14, 'bold')).grid(row=0, column=0,
                                                                                 sticky='w')

        self.limit_var = tk.IntVar(value=2000)
        self.vocab_var = tk.IntVar(value=2000)
        self.window_var = tk.IntVar(value=1)
        self.target_var = tk.StringVar(value='دادگاه')
        self.topn_var = tk.IntVar(value=5)


        tb.Label(frame_top, text='تعداد جملات (limit):').grid(row=1, column=0, sticky='w')
        tb.Entry(frame_top, textvariable=self.limit_var, width=10).grid(row=1, column=1)

        tb.Label(frame_top, text='حداکثر واژگان:').grid(row=1, column=2, sticky='w')
        tb.Entry(frame_top, textvariable=self.vocab_var, width=10).grid(row=1, column=3)

        tb.Label(frame_top, text='طول پنجره:').grid(row=1, column=4, sticky='w')
        tb.Entry(frame_top, textvariable=self.window_var, width=8).grid(row=1, column=5)

        self.build_btn = tb.Button(frame_top, text='📦 ساخت ماتریس', bootstyle='success-outline',
                                   command=self.start_build)
        self.build_btn.grid(row=2, column=0, pady=10)

        self.progress = tb.Progressbar(frame_top, bootstyle='info-striped', orient='horizontal',
                                       length=400, mode='indeterminate')
        self.progress.grid(row=2, column=1, columnspan=4, padx=8)

        frm_search = tb.Labelframe(self.master, text='🔎 جستجو و نمایش نتایج', padding=12)
        frm_search.pack(fill='x', padx=12, pady=6)

        tb.Label(frm_search, text='کلمه هدف:').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        tb.Entry(frm_search, textvariable=self.target_var, width=20).grid(row=0, column=1, padx=6)

        tb.Label(frm_search, text='تعداد مشابه:').grid(row=0, column=2, padx=6)
        tb.Entry(frm_search, textvariable=self.topn_var, width=6).grid(row=0, column=3, padx=6)

        tb.Button(frm_search, text='🔎 پیدا کن', bootstyle='primary',
                  command=self.find_similar).grid(row=0, column=4, padx=8)
        tb.Button(frm_search, text='📊 نمایش PCA', bootstyle='info',
                  command=self.start_show_pca).grid(row=0, column=5, padx=8)

        self.result_box = scrolledtext.ScrolledText(self.master, height=18, font=('Tahoma', 11))
        self.result_box.pack(fill='both', padx=12, pady=8, expand=True)

        frame_bottom = tb.Frame(self.master)
        frame_bottom.pack(fill='x', padx=12, pady=6)

        tb.Button(frame_bottom, text='پاک کن', bootstyle='warning',
                  command=self.clear_results).pack(side='left', padx=6)
        tb.Button(frame_bottom, text='خروج', bootstyle='danger', command=self.master.quit).pack(
            side='right', padx=6)

        self.log('اپلیکیشن آماده است. برای شروع روی "ساخت ماتریس" کلیک کنید.')

    def log(self, txt: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.result_box.insert(tk.END, f'[{ts}] {txt}\n')
        self.result_box.see(tk.END)

    def clear_results(self):
        self.result_box.delete('1.0', tk.END)


    def start_build(self):
        self.build_btn.config(state='disabled')
        self.progress.start(10)
        t = threading.Thread(target=self._build_thread, daemon=True)
        t.start()

    def _build_thread(self):
        try:
            limit = int(self.limit_var.get())
            maxv = int(self.vocab_var.get())
            window = int(self.window_var.get())
            self.log('در حال بارگذاری دیتاست...')
            texts = self.dm.load_law_dataset(limit=limit)
            self.log(f'تعداد جملات: {len(texts)}')
            self.log('در حال ساخت واژگان...')
            vocab, w2i = self.dm.build_vocabulary(max_vocab_size=maxv)
            self.log(f'واژگان ساخته شد: {len(vocab)}')
            self.log('در حال ساخت ماتریس هم‌رخدادی...')
            mat = self.dm.build_cooccurrence_matrix(window_size=window)
            self.log('ماتریس ساخته شد (sparse csr).')
        except Exception as e:
            messagebox.showerror('خطا', f'خطا: {e}')
            self.log(f'خطا: {e}')
        finally:
            self.progress.stop()
            self.build_btn.config(state='normal')

    def find_similar(self):
        if self.dm.matrix is None:
            messagebox.showwarning('هشدار', 'ابتدا ماتریس را بسازید.')
            return
        target = self.target_var.get().strip()
        topn = int(self.topn_var.get())
        self.log(f'پیدا کردن {topn} مشابه برای: {target}')
        res = self.dm.most_similar(target, topn)
        if not res:
            self.log('کلمه در واژگان یافت نشد.')
            return
        self.clear_results()
        self.log(f'کلمات شبیه به «{target}»:')
        for w, s in res:
            self.log(f"{w:30} -> {s:.4f}")

    def start_show_pca(self):
        if self.dm.matrix is None:
            messagebox.showwarning('هشدار', 'ابتدا ماتریس را بسازید.')
            return
        t = threading.Thread(target=self._pca_thread, daemon=True)
        t.start()

    def _pca_thread(self):
        if self.dm.matrix is None:
            messagebox.showwarning('هشدار', 'ابتدا ماتریس را بسازید.')
            return
        target = self.target_var.get().strip()
        topn = int(self.topn_var.get())
        try:
            self.log(f'در حال تولید PCA برای "{target}" با {topn} کلمه مشابه...')
            self.dm.plot_pca_for_target(target_word=target, top_n=topn)
            self.log('نمودار PCA نمایش داده شد.')
        except Exception as e:
            self.log(f'خطا در PCA: {e}')
            messagebox.showerror('خطا', f'خطا در PCA: {e}')
