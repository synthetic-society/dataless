import pathlib
from typing import Tuple

import matlab
import matlab.engine
import numpy.typing as npt

from .model import PYP, multiplicities_from_freqs


try:
    eng = matlab.engine.start_matlab()
    
except matlab.engine.EngineError:
    eng = matlab.engine.connect_matlab()

current_path = pathlib.Path(__file__).resolve().parent
pymentropy_path = (current_path / "../PYMentropy/src").absolute()
eng.addpath(str(pymentropy_path))


def pyp_from_freqs(freqs: npt.ArrayLike) -> PYP:
    """ Estimate a PYP from a frequency array, using the PYMentropy Matlab code. """
    mm, icts = multiplicities_from_freqs(freqs)
    p, _, _ = pyp_from_multiplicities(mm, icts)
    return p


def pyp_from_multiplicities(mm, icts) -> Tuple[PYP, float, float]:
    mm_matlab = matlab.double([int(m) for m in mm])
    icts_matlab = matlab.double([int(i) for i in icts])
    h_mean, h_var, params = eng.computeH_PYM(
        mm_matlab,
        icts_matlab,
        nargout=3
    )

    α, d = params['llog_a'], params['llog_d']

    return PYP(α=α, d=d), h_mean, h_var
