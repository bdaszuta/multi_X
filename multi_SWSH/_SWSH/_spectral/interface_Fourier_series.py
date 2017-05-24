"""
 ,-*
(_) Created on Fri Oct 23 14:28:36 2015

@author: Boris Daszuta
@function: Here we provide a simplified interface for Fourier series methods.
In particular for making use of FFTW via the pyFFTW bindings.

TODO:
Check wisdom and plan accumulation

In a newer version of pyfftw fix the ifft such that conjugation trick is not
required.
"""

import multiprocessing as _mp        # purely for thread-count estimation
import numpy as _np
import pyfftw as _pfft


class FourierSeries:
    '''
    Small class for making use of FFTW via pyfftw bindings.
    '''
    def __init__(self, use_cache=True,
                 fftw_plan_effort='FFTW_MEASURE',
                 fftw_threads=_mp.cpu_count()):

        # ESTIMATE,MEASURE,EXHAUSTIVE
        self.fftw_plan_effort = fftw_plan_effort
        self.fftw_threads = fftw_threads

        # Allocate memory once together with optimized plan - then reuse
        # Without killing it (use_cache=True behaviour)
        self.use_cache = use_cache
        self.fft_cache = {}

        self.irfft_cache = {}
        self.rfft_cache = {}

    def _have_fft_cache(self, data_shape, type_xform):
        '''
        Check whether we have a fft obj and memory cached
        '''
        if type_xform in self.fft_cache.keys():
            if data_shape in self.fft_cache[type_xform].keys():
                return True
        else:
            return False
        return False

    def _put_fft_cache(self, fft_obj, mem_alloc, type_xform, data_shape):
        '''
        Put an fft obj and memory in cache
        '''
        # Ensure we have the type
        if not (type_xform in self.fft_cache.keys()):
            self.fft_cache[type_xform] = {}
        # Ensure we have the shape
        if not (data_shape in self.fft_cache[type_xform].keys()):
            self.fft_cache[type_xform][data_shape] = {}

        # Put plan and memory
        self.fft_cache[type_xform][data_shape] = \
            {'fft_obj': fft_obj,
             'mem_alloc': mem_alloc}

    def _get_fft_cache(self, data_shape, type_xform):
        '''
        Get and fft obj and memory from cache
        '''
        return \
            self.fft_cache[type_xform][data_shape]['fft_obj'], \
            self.fft_cache[type_xform][data_shape]['mem_alloc']

    def _have_rfft_cache(self, data_shape, type_xform):
        '''
        Check whether we have a fft obj and memory cached
        '''
        if type_xform in self.rfft_cache.keys():
            if data_shape in self.rfft_cache[type_xform].keys():
                return True
        else:
            return False
        return False

    def _put_rfft_cache(self, fft_obj, mem_alloc, type_xform, data_shape):
        '''
        Put an fft obj and memory in cache
        '''
        # Ensure we have the type
        if not (type_xform in self.rfft_cache.keys()):
            self.rfft_cache[type_xform] = {}
        # Ensure we have the shape
        if not (data_shape in self.rfft_cache[type_xform].keys()):
            self.rfft_cache[type_xform][data_shape] = {}

        # Put plan and memory
        self.rfft_cache[type_xform][data_shape] = \
            {'fft_obj': fft_obj,
             'mem_alloc': mem_alloc}

    def _get_rfft_cache(self, data_shape, type_xform):
        '''
        Get and fft obj and memory from cache
        '''
        return \
            self.rfft_cache[type_xform][data_shape]['fft_obj'], \
            self.rfft_cache[type_xform][data_shape]['mem_alloc']

    def _have_irfft_cache(self, data_shape, type_xform):
        '''
        Check whether we have a fft obj and memory cached
        '''
        if type_xform in self.irfft_cache.keys():
            if data_shape in self.irfft_cache[type_xform].keys():
                return True
        else:
            return False
        return False

    def _put_irfft_cache(self, fft_obj, mem_alloc, type_xform, data_shape):
        '''
        Put an fft obj and memory in cache
        '''
        # Ensure we have the type
        if not (type_xform in self.irfft_cache.keys()):
            self.irfft_cache[type_xform] = {}
        # Ensure we have the shape
        if not (data_shape in self.irfft_cache[type_xform].keys()):
            self.irfft_cache[type_xform][data_shape] = {}

        # Put plan and memory
        self.irfft_cache[type_xform][data_shape] = \
            {'fft_obj': fft_obj,
             'mem_alloc': mem_alloc}

    def _get_irfft_cache(self, data_shape, type_xform):
        '''
        Get and fft obj and memory from cache
        '''
        return \
            self.irfft_cache[type_xform][data_shape]['fft_obj'], \
            self.irfft_cache[type_xform][data_shape]['mem_alloc']

    def rfft(self, data, specified_type=False, normalized=True):
        '''
        Forward (Hermitian symmetry) transform
        '''
        # Get relevant data type
        type_xform = data.dtype if not specified_type else specified_type

        # Allocate aligned memory and plan if not extant
        if self.use_cache and self._have_rfft_cache(data.shape, type_xform):
            # Found a cache - use it
            rfft, mem = self._get_rfft_cache(data.shape, type_xform)

            # Perform transform
            mem[:] = data[:]
            out = rfft()
        else:

            # Setup
            n = _pfft.simd_alignment
            mem = _pfft.n_byte_align_empty(data.shape, n, dtype=type_xform)
            rfft = _pfft.builders.rfft(mem,
                                       planner_effort=self.fftw_plan_effort,
                                       threads=self.fftw_threads)

            # Perform transform
            mem[:] = data[:]
            out = rfft()

            # Cache transform memory and plan
            if self.use_cache:
                self._put_rfft_cache(rfft, mem, type_xform, data.shape)

        # Optional normalization
        if normalized:
            out = out / out.size
        return out

    def irfft(self, data, specified_type=False, normalized=True):
        '''
        Backward (Hermitian symmetry) transform
        '''

        # Get relevant data type
        type_xform = data.dtype if not specified_type else specified_type

        # Allocate aligned memory and plan if not extant
        if self.use_cache and self._have_rfft_cache(data.shape, type_xform):
            # Found a cache - use it
            rfft, mem = self._get_rfft_cache(data.shape, type_xform)

            # Perform transform
            mem[:] = data[:]
            out = rfft()
        else:

            # Setup
            n = _pfft.simd_alignment
            mem = _pfft.n_byte_align_empty(data.shape, n, dtype=type_xform)
            irfft = _pfft.builders.irfft(mem,
                                         planner_effort=self.fftw_plan_effort,
                                         threads=self.fftw_threads)

            # Perform transform
            mem[:] = data[:]
            out = irfft()

            # Cache transform memory and plan
            if self.use_cache:
                self._put_irfft_cache(irfft, mem, type_xform, data.shape)

        # Optional normalization
        if normalized:
            out = out / out.size
        return out

    def fft(self, data, specified_type=False, normalized=True):
        '''
        Forward transform:

        FFTW computes:

        Given 'N' samples (X_0, X_1, ..., X_{N-1})
        Y_k = Sum[X_j exp(-2 pi j*k*sqrt(-1)/N), {j,0,N-1}]

        Where the output is scaled and rearranged such that:
            N even:
            X -> N (Y_0, Y_1, ..., Y_{N/2-1}; Y_{-N/2}, ..., Y_{-2}, Y_{-1} )
            N odd:
            X -> N (Y_0, ..., Y_{(N-1)/2}; Y_{-(N-1)/2}, ..., Y_{-2}, Y_{-1} )
        '''

        # Get relevant data type
        type_xform = data.dtype if not specified_type else specified_type

        # Allocate aligned memory and plan if not extant
        if self.use_cache and self._have_fft_cache(data.shape, type_xform):
            # Found a cache - use it
            fft, mem = self._get_fft_cache(data.shape, type_xform)

            # Perform transform
            mem[:] = data[:]
            out = _np.copy(fft())
        else:
            n = _pfft.simd_alignment
            mem = _pfft.n_byte_align_empty(data.shape, n, dtype=type_xform)

            # Form a plan
            fft = _pfft.builders.fftn(mem,
                                      planner_effort=self.fftw_plan_effort,
                                      threads=self.fftw_threads)

            # Perform transform
            mem[:] = data[:]
            out = _np.copy(fft())

            # Cache transform memory and plan
            if self.use_cache:
                self._put_fft_cache(fft, mem, type_xform, data.shape)

        # Optional normalization
        if normalized:
            out = out / _np.prod(out.shape)
        return out

    def ifft(self, data, specified_type=False, normalized=True):
        '''
        Backward transform:

        FFTW computes:

        Given 'N' samples (X_0, X_1, ..., X_{N-1})
        Y_k = Sum[X_j exp(2 pi j*k*sqrt(-1)/N), {j,0,N-1}]

        See fft for scalings.
        '''
        #####
        # OLD
        #####

        # # get relevant data type
        # type_xform = data.dtype if not specified_type else specified_type

        # # allocate aligned memory
        # n = _pfft.simd_alignment
        # mem = _pfft.n_byte_align_empty(data.shape, n, dtype=type_xform)

        # # form a plan
        # ifft = _pfft.builders.ifftn(mem, \
        #     planner_effort=self.fftw_plan_effort, \
        #     threads=self.fftw_threads)

        # # perform transform
        # mem[:] = data[:]
        # out = ifft()

        # # optional normalization
        # if normalized:
        #     out = out*_np.prod(out.shape)
        # return out

        ##############
        # NEW (Faster)
        ##############

        # pyfftw has an internal bug that casts to single precision in some
        # versions -- we can instead just the the fft which is fine via
        # a conjucation trick:
        # ifft(x) = conj(fft(conj(x)))/N
        out = _np.conj(self.fft(_np.conj(data), specified_type=specified_type,
                                normalized=False))
        out = out / _np.prod(out.shape)

        # Optional normalization
        if normalized:
            out = out * _np.prod(out.shape)
        return out

    def fftshift(self, data, ax=None):
        '''
        Shift zero frequency
        [Uses numpy shift]
        '''
        return _np.fft.fftshift(data, axes=ax)

    def ifftshift(self, data, ax=None):
        '''
        Inverse of fftshift
        [Uses numpy shift]
        '''
        return _np.fft.ifftshift(data, axes=ax)


def _main():
    dat = _np.array(_np.random.rand(1000, 1000), dtype=_np.complex128)
    FS = lambda use_cache: FourierSeries(use_cache=use_cache)
    return FS(False)(dat), FS(True)(dat)

#
# :D
#
