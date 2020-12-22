import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow

x, sampling_rate = librosa.load('win_xp_shutdown.wav')

T = x.size / sampling_rate
print(
    f'x[k] has {x.size} samples',
    f'the sampling rate is {sampling_rate * 1e-3}kHz',
    f'x(t) is {T:.1f}s long'
    , sep='\n')

plt.style.use(['dark_background', 'bmh'])
plt.rc('axes', facecolor='k')
plt.rc('figure', facecolor='k')
plt.rc('figure', figsize=(16, 8), dpi=100)

dt = 1 / sampling_rate
t = np.r_[0:T:dt]

X = librosa.stft(x)
X_dB = librosa.amplitude_to_db(np.abs(X))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.xlim([0, T])
plt.ylabel('amplitude [/]')
plt.title('Audio signal x(t) and its spectrogram X(t)')
plt.setp(plt.gca().get_xticklabels(), visible=False)
plt.subplot(2, 1, 2)
specshow(X_dB, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.ylim(top=2000)
plt.grid(True)
plt.show()

plt.figure()
convs = list()
# Manual recontruction of the melody:
# pick the melody frequencies/notes from the spectrogram above
Ab6 = 1661  # Hz
Eb6 = 1244  # Hz
Ab5 = 830  # Hz
Bb5 = 932  # Hz
TT = .1  # s
tt = np.r_[0:TT:dt]
n = 1
A = {
    '1661': np.sin(2 * np.pi * Ab6 * tt),
    '1244': np.sin(2 * np.pi * Eb6 * tt),
    '830': np.sin(2 * np.pi * Ab5 * tt),
    '932': np.sin(2 * np.pi * Bb5 * tt),
}
for a in A.items():
    plt.subplot(4, 1, n)
    plt.title(rf'$x(t) \star a^{{({a[0]})}}(t)$', backgroundcolor='black', verticalalignment='top', size=17)
    n += 1
    convs.append(np.convolve(x, a[1], mode='same'))
    plt.plot(t, convs[-1])
    if n < 5:
        plt.setp(plt.gca().get_xticklabels(), visible=False)
plt.ylabel('amplitude [/]')
plt.xlabel('time [s]')
plt.show()
