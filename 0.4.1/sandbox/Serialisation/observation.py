# from jax.typing import Array, ArrayLike
import jax
import zodiax as zdx
import jax.numpy as np
from abc import abstractmethod
from jax import vmap
from jax.tree_util import tree_map
from equinox import tree_at
from dLux.core import Instrument
from typing import Any
import dLux
Array = jax.numpy.ndarray


class AbstractObservation(zdx.ExtendedBase):
    name : str

    def __init__(self, name='AbstractObservation'):
        self.name = str(name)
    
    @abstractmethod
    def observe(self, instrument) -> Any:
        pass


class Dither(AbstractObservation):
    dithers : Array


    def __init__(self, dithers, name='Observation'):
        super().__init__(name)
        self.dithers = np.asarray(dithers, float)


    def dither_position(self, instrument : Instrument, dither : Array) -> Instrument:
        """
        Dithers the position of the source objects by dither.

        Parameters
        ----------
        dither : Array, radians
            The (x, y) dither to apply to the source positions.

        Returns
        -------
        instrument : Instrument
            The instrument with the sources dithered.
        """
        assert dither.shape == (2,), ("dither must have shape (2,) ie (x, y)")

        # Define the dither function
        dither_fn = lambda source: source.add('position', dither)

        # Map the dithers across the sources
        dithered_sources = tree_map(dither_fn, instrument.scene.sources, \
                is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

        # Apply updates
        return tree_at(lambda instrument: instrument.scene.sources, instrument, \
                       dithered_sources)


    def dither_and_model(self,
                         instrument : Instrument,
                         dithers : Array,
                         **kwargs) -> Any:
        """
        Applies a series of dithers to the instrument sources and calls the
        .model() method after applying each dither.

        Parameters
        ----------
        dithers : Array, radians
            The array of dithers to apply to the source positions.

        Returns
        -------
        psfs : Array
            The psfs generated after applying the dithers to the source
            positions.
        """
        dith_fn = lambda dither: self.dither_position(instrument, dither).model(**kwargs)
        return vmap(dith_fn, 0)(dithers)
    

    def observe(self, instrument):
        return self.dither_and_model(instrument, self.dithers)
    
    def _construct():
        return Dither(np.array([]))