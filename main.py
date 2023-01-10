import collections

import matplotlib
import numpy as np

import dataset
import recorder
from mel import get_mel, mel_to_frames

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

import tkinter as tk
from tkinter import ttk
from model import nn, model_to_transfer_learning_model
import pyaudio

classes = []

with open('dataset/class_names') as f:
    for line in f:
        classes.append(line.strip())
p = pyaudio.PyAudio()
X_train, X_test, y_train, y_test, train_mean, train_std, y_train_id, y_test_id = None, None, None, None, None, None, None, None

train_mean, train_std = -58.38601, 20.77519
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 22050
interval = 100
chunk = 2048
n_mels = 128
mel_buffer = [[]] * n_mels
sec = 5
i_animate = 0
n_fft = 512
animation_on = False
hop_length = n_fft // 2
chunk_mel = int(np.ceil(chunk / hop_length))
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=False,
                frames_per_buffer=chunk)
record_cache = None
style.use("ggplot")

f = Figure(figsize=(5, 5), dpi=100)
a = f.add_subplot(211)
b = f.add_subplot(212)
ilist = np.zeros((n_mels, int(sample_rate / chunk * chunk_mel * sec)))

wlist = np.zeros(sample_rate * sec)

gui = {"label_acc": None,
       "LARGE_FONT": ("Verdana", 12),
       "SMALL_FONT": ("Verdana", 7)
       }
model = nn(pretrained=True)


def animate(i):
    if not animation_on:
        return
    global gui, wlist, ilist, mel_buffer, i_animate
    i_animate += 1
    data = stream.read(chunk)
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    wlist = np.append(wlist[chunk:], arr)
    a.clear()
    a.set_ylim([-15000, 15000])
    a.axis("off")
    a.plot(wlist)
    mel = get_mel(arr, sample_rate)
    ilist = np.append(ilist[:, mel.shape[1]:], mel[::1, :], axis=1)

    mel_buffer = np.concatenate([mel_buffer, mel], axis=1)

    if (i_animate + 1) % 6 == 0:

        mel_frames = mel_to_frames(mel_buffer)
        mel_frames = (mel_frames - train_mean) / train_std
        y = model.predict(mel_frames, verbose=0)
        pred = y.argmax(axis=1)
        print(pred.max(),len(classes))

        pred = pred[pred != 0]
        if len(pred) != 0:
            speaker = collections.Counter(pred).most_common()[0][0]
            print([classes[p] for p in pred])
        else:
            speaker = 0  # not speaking
        if gui["label_acc"]:
            gui["label_acc"].config(text="prediction: " + classes[speaker])
        mel_buffer = [[]] * n_mels

    b.clear()
    b.imshow(ilist, cmap='magma', vmin=-80, vmax=0)
    b.axis("off")


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Voice Recognition")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (PageHome, PageDataset, PageAddSpeaker, PageApp):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(PageHome)

    def show_frame(self, cont, animation=False):
        global animation_on
        if animation:
            animation_on = True
        else:
            animation_on = False
        frame = self.frames[cont]
        frame.tkraise()

    def download(self):
        print("First time download and unzip should take few minutes")
        dataset.download()
        dataset.unzip()
        print("Download done")

    def load_Xy(self):
        global X_train, X_test, y_train, y_test, train_mean, train_std
        print(
            "First time preprocessing should take few minutes, it will transfer the audio data into mel spectrogram image data")
        X_train, X_test, y_train, y_test, train_mean, train_std = dataset.load_and_preprocess_dataset()
        print("preprocessing done")

    def record_and_merge_to_trainset(self):
        global record_cache, X_train, y_train, train_mean, train_std, y_train_id, y_test_id
        if X_train is None:
            print("Click button \"Load and preprocess dataset\" first ")
            return
        r = recorder.record()
        print("Merging new record to dataset")
        mel = get_mel(r, sample_rate)
        record_cache = mel_to_frames(mel)
        record_cache = (record_cache - train_mean) / train_std
        X_train = np.concatenate([X_train, record_cache], axis=0)
        y = ["You"] * len(record_cache)
        y_train = np.concatenate([y_train, y], axis=0)
        ids = np.random.permutation(len(X_train))
        X_train, y_train = X_train[ids], y_train[ids]
        classes.append("You")
        label_dict = {}
        for i in range(len(classes)):
            label_dict[classes[i]] = i
        y_train_id = np.array([label_dict[y] for y in y_train])
        y_test_id = np.array([label_dict[y] for y in y_test])
        print("Merging new record done")

    def train(self):
        global model
        if y_train_id is None:
            print("Click button \"Record\" first ")
        else:
            if model.layers[-1].output_shape[1] == 185:
                model = model_to_transfer_learning_model(model)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            ids = np.random.permutation(int(len(X_train) * 0.05))
            ids2 = np.random.permutation(int(len(X_test) * 0.05))
            model.fit(X_train[ids], y_train_id[ids], epochs=1, batch_size=128, validation_data=(X_test[ids2], y_test_id[ids2]))


class PageHome(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Voice Recognition App", font=gui["LARGE_FONT"])
        label.pack(pady=10, padx=10)
        button = ttk.Button(self, text="Dataset Utils", command=lambda: controller.show_frame(PageDataset))
        button2 = ttk.Button(self, text="Add new Speaker", command=lambda: controller.show_frame(PageAddSpeaker))
        button3 = ttk.Button(self, text="App", command=lambda: controller.show_frame(PageApp, animation=True))
        button.pack()
        button2.pack()
        button3.pack()
        text = tk.Text(self, width=60, height=4)
        text.insert('end',
                    "Dataset Utils is for adding new speaker, no need to use Dataset Utils if you only use the original app.\n"
                    "Please keep looking at the console for progress.")
        text.configure(state='disabled')
        text.pack()


class PageDataset(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Dataset Utils", font=gui["LARGE_FONT"])
        label.pack(pady=10, padx=10)
        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(PageHome))
        button2 = ttk.Button(self, text="Download Dataset", command=lambda: controller.download())
        button3 = ttk.Button(self, text="Load and Preprocess Dataset", command=lambda: controller.load_Xy())
        button1.pack()
        button2.pack()
        button3.pack()


class PageAddSpeaker(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Add New Speaker", font=gui["LARGE_FONT"])
        label.pack(pady=10, padx=10)
        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(PageHome))
        button2 = ttk.Button(self, text="Record/re-record for 10 second",
                             command=lambda: controller.record_and_merge_to_trainset())
        button3 = ttk.Button(self, text="Train recorded audio for 1 epoch", command=lambda: controller.train())
        button1.pack()
        button2.pack()
        button3.pack()


class PageApp(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        gui["label_acc"] = tk.Label(self, text="", font=gui["LARGE_FONT"])
        gui["label_acc"].pack(pady=10, padx=10)
        label = tk.Label(self, text="Not speaking can be inaccuracy due to insufficient noise dataset",
                         font=gui["SMALL_FONT"])
        label.pack(pady=20, padx=10)
        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(PageHome))
        button1.pack()

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = App()
ani = animation.FuncAnimation(f, animate, interval=10)
app.mainloop()
