#pip install opencv-python pydub tqdm numpy scipy matplotlib 

import cv2
import pydub
import math
import os
from tqdm import tqdm
import time
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.io.wavfile import read


#SPECTROGRAPH MAKER
class Spectrogram():
    def __init__(self, width, height):
        self.spectrogram = None
        self.frames = None
        self.sample = None
        self.anim = None
        self.audio = None
        self.fr = None
        self.width = width
        self.height = height

    def __call__(self, title, fps):
        if title.endswith('.mp3'):
            song = pydub.AudioSegment.from_mp3(title)            
        else:
            song = pydub.AudioSegment.from_wav(title)

        self.fr = song.frame_rate
        self.audio = np.array(song.get_array_of_samples())

        if song.channels == 2:
            self.audio = self.audio.reshape((-1, 2))

        #Display
        plt.style.use('dark_background')
        self.sample = math.floor(self.fr / fps)
        self.frames = math.floor(self.audio.shape[0] / self.sample)
        interval = int((1/fps)*1000)
        # color = sys.argv[1]
        
        dpi = 240
        figure = plt.figure(figsize=(width/dpi, height/dpi))
        axes = plt.axes(xlim=(0, self.fr/2), ylim=(0, 1e7))
        axes.axis('off')
        self.spectrogram,  = axes.plot([], [])

        anim = animation.FuncAnimation(figure, self.animate, init_func=self.init, frames=self.frames, interval=interval, blit=True)
        anim.save('temp.mp4', fps=fps, dpi=dpi, extra_args=['-vcodec', 'libx264'])

        # plt.show()

    def init(self):
        self.spectrogram.set_data([], [])
        return self.spectrogram, 

    def animate(self, i):
        if i > self.frames:
            self.anim.event_source.stop()
        start, end = i*self.sample, (i*self.sample)+self.sample
        f = np.abs(fft(self.audio[start:end, 0]))
        #f = normalize(f)
        w = np.linspace(0, self.fr, len(f))
        f = f[0:len(f)//2]
        w = w[0:len(w)//2]

        self.spectrogram.set_data(w, f)
        return self.spectrogram, 


meta = open('meta.txt', 'r')
author, title = meta.readline().split(' - ')
meta.close()

video_name = [n for n in os.listdir() if n.endswith('.mp4')]
cap = cv2.VideoCapture(video_name[0])
fps = cap.get(cv2.CAP_PROP_FPS)
clip_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

overlay = cv2.imread('./logo.png', cv2.IMREAD_UNCHANGED)
logo_h, logo_w, _ = overlay.shape
b,g,r,a = cv2.split(overlay)
a = a/255
beta = 1 - a
overlay = cv2.merge((b*a, g*a, r*a)).astype(np.uint16)

width = int(cap.get(3))
height = int(cap.get(4))
padding = 40

song_name = [n for n in os.listdir() if (n.endswith('.mp3') or n.endswith('.wav'))]
recording = cv2.VideoWriter('./output.mp4', fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=fps, frameSize=(width, height), apiPreference=0)

print('Please wait as the spectrum generation might take some time...\n')
if [z for z in os.listdir() if z == 'temp.mp4'] == []:
    o = Spectrogram(width, height)
    o(song_name[0], fps)
print('Spectrum generated successfully !\n')

print('Video rendering in process...\n')
spec = cv2.VideoCapture('./temp.mp4')
spec_w = int(spec.get(3))
spec_h = int(spec.get(4))

cap.release()


cap_fade = None
cap = cv2.VideoCapture(video_name[0])
pace = 1 / (fps)
fade = 0

spec_len = int(spec.get(cv2.CAP_PROP_FRAME_COUNT))

for i in tqdm(range(spec_len)):
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= (clip_len - 2*fps) or cap_fade is not None: 
        if cap_fade is None:
            cap_fade = cv2.VideoCapture(video_name[0])
            cap_fade.set(cv2.CAP_PROP_POS_FRAMES, (clip_len - 2*fps))
            cap.release()
            cap = cv2.VideoCapture(video_name[0])

        ret, fore = cap_fade.read()       
        if not ret:
            cap_fade.release()
            cap_fade = None
            _, frame = cap.read()
            fade = 0
        else:
            _, back = cap.read()
            if fade > 1:
                fade = 1
            frame = cv2.addWeighted(fore, 1-fade, back, fade, 0)
            fade += pace
    else:
        _, frame = cap.read()

    region = frame[padding:padding+logo_h, width-logo_w-padding:width-padding]
    b, g, r = cv2.split(region)
    region = cv2.merge((b*beta, g*beta, r*beta))    
    frame[padding:padding+logo_h, width-logo_w-padding:width-padding] = cv2.add(region, overlay, dtype=0) 
    
    frame = cv2.putText(frame, author, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA, False)
    frame = cv2.putText(frame, title, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA, False)

    #Write Specter
    ret, graph = spec.read()
    if ret:
        region2 = frame[height-spec_h:height, 0:spec_w]
        frame[height-spec_h:height, 0:spec_w] = cv2.add(region2, graph, dtype=0)
    
    recording.write(frame)

cap.release()
recording.release()
spec.release()

#Merge the audio and video
import moviepy.editor as mpe
my_clip = mpe.VideoFileClip('./output.mp4')
audio_background = mpe.AudioFileClip(song_name[0])
final_clip = my_clip.set_audio(audio_background)
final_clip.write_videofile("FINAL.mp4")

os.remove('./output.mp4')
os.remove('./temp.mp4')

print('DONE ! Check the FINAL.mp4 file \n')