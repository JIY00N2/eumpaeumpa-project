from __future__ import division

import os
import logging

import numpy
from scipy.signal import lfilter
from copy import deepcopy

from .SpeechProc import speech_wave, sflux, pitchblockdetect, snre_highenergy, snre_vad

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, 2019.
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection."
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# 2017-12-02, Achintya Kumar Sarkar and Zheng-Hua Tan

# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel


class VoiceActivityDecection(object):
    def __init__(self, ftThres=0.5, vadThres=0.4, opts=1, winlen=0.025, ovrlen=0.01, pre_coef=0.97, nfilter=20, nfft=512):
        self.path = os.getcwd()
        self.savedir = os.path.join(self.path, "result")

        self.winlen = winlen
        self.ovrlen = ovrlen
        self.pre_coef = pre_coef
        self.nfilter = nfilter
        self.nftt = nfft

        self.ftThres = ftThres
        self.vadThres = vadThres
        self.opts = opts

    def __call__(self, finwav, fvad="vad_result.txt"):
        self.createDirectory(self.savedir)
        fvad = os.path.join(self.savedir, fvad)

        fs, data = speech_wave(finwav)
        ft, flen, fsh10, nfr10 = sflux(data, fs, self.winlen, self.ovrlen, self.nftt)

        # --spectral flatness --
        pv01 = numpy.zeros(nfr10)

        pv01[numpy.less_equal(ft, self.ftThres)] = 1
        pitch = deepcopy(ft)

        pvblk = pitchblockdetect(pv01, pitch, nfr10, self.opts)

        # --filtering--
        ENERGYFLOOR = numpy.exp(-50)
        b = numpy.array([0.9770, -0.9770])
        a = numpy.array([1.0000, -0.9540])
        fdata = lfilter(b, a, data, axis=0)

        # --pass 1--
        noise_samp, noise_seg, n_noise_samp = snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

        # sets noisy segments to zero
        for j in range(n_noise_samp):
            fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

        vad_seg = snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, self.vadThres)
        self.res_save(vad_seg, fvad)

        return vad_seg.astype(int)

    def res_save(self, vad_seg, fvad):
        numpy.savetxt(fvad, vad_seg.astype(int), fmt='%i')


    def createDirectory(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create the directory.")

