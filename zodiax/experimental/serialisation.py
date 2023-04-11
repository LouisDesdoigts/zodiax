from __future__ import annotations
import sys
import pickle
import zodiax
from typing import Any
from jax import Array
# from jax.typing import ArrayLike
import numpy as onp
import jax.numpy as jnp
from equinox import tree_serialise_leaves, tree_deserialise_leaves
from pathlib import Path


__all__ = ['serialise', 'deserialise', 'load_structure', 'build_structure']


#####################
### Serialisation ###
#####################
def _check_node(obj      : Any, 
                self_key : str  = None, 
                depth    : int  = 0., 
                _print   : bool = False) -> bool:
    """
    Checks if the input object is a container or a leaf node. If the object is
    a leaf False is returned else True.

    Container types:
        list, tuple, dict, zodiax.Base
    
    Leaf types:
        jax.ArrayLike, numpy.ndarray, bool, complex, float, int, str, None
    
    All other types will raise an error as they are not supported, but likely
    could be supported in future.

    Parameters
    ----------
    obj : Any
        The object to check.
    depth : int = 0
        The depth of the object in the tree. Use to print the tree structure
        for debugging.
    self_key : str = None
        The key of the object in the parent container. Use to print the tree
        structure for debugging.
    _print : bool = False
        If True, print the tree structure for debugging.
    
    Returns
    -------
    is_container : bool
        True if the object is a container, False if it is a leaf.
    """
    t = '  ' * depth
    conatiner_types = (list, tuple, dict, zodiax.Base)
    leaf_types = (Array, onp.ndarray, onp.bool_, onp.number, bool, int, float,
        complex, str)

    # Contianer node
    if isinstance(obj, conatiner_types):
        if _print: print(f"{t}Node '{self_key}' of type: {type(obj)}")
        return True

    # NOTE: Checking None types in a tuple of types has some off behaviour. 
    # There have been changes from python 3.7 - 3.8 and then 3.9-3.10.
    # This seems to be the most robust way to check for None types.

    # Leaf node
    elif isinstance(obj, leaf_types) or isinstance(obj, type(None)):
        if _print: print(f"{t}Leaf '{self_key}' of type: {type(obj)}")
        return False

    # Unknown - Raise error
    else:
        raise ValueError(f"Unknown node '{self_key}' of type: {type(obj)}. "
            "Please raise an issue on GitHub to enable support for this type.")
    

def _get_accessor(obj : Any) -> tuple:
    """
    Returns the keys and accessor for the input object.

    Parameters
    ----------
    obj : Any
        The object to get the keys and accessor for.

    Returns
    -------
    (keys, accessor) : tuple
        keys : list
            The keys of the object.
        accessor : function
            The accessor function for the object.
    """
    accessor = lambda object, x: object[x]

    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        keys = list(range(len(obj)))
        return keys, accessor

    # Zodaix object types
    elif isinstance(obj, zodiax.Base):
        accessor = lambda object, x: getattr(object, x)
        keys = obj.__dataclass_fields__.keys()
        return keys, accessor

    # Dictionary types
    else:
        return obj.keys(), accessor


def _format_type(obj : Any) -> str:
    """
    Returns a string of the object type with the extra '< class ' and '>' 
    removed.

    Parameters
    ----------
    obj : Any
        The object to get the type of.
    
    Returns
    -------
    type_str : str
        The string of the object type.
    """
    class_str = str(type(obj))
    return class_str[class_str.find("'")+1:class_str.rfind("'")]


def _format_dtype(dtype : onp.dtype) -> str:
    """
    Formats the array dtype into a string. Supports boolean, integer, float 
    and complex types.

    Parameters
    ----------
    dtype : numpy.dtype
        The array dtype.
    
    Returns
    -------
    dtype_str : str
    """
    kind_map = {'b': 'bool', 'f': 'float', 'i': 'int', 'c': 'complex'}
    return f"{kind_map[dtype.kind]}{8*dtype.itemsize}"


def _build_node(obj    : Any, 
               key    : str  = None, 
               depth  : int  = 0, 
               _print : bool = False) -> dict:
    """
    Builds a node for the input object.

    Parameters
    ----------
    obj : Any
        The object to get the leaves of.
    key : str
        The key of the object in the parent container. Use to print the tree
        structure for debugging.
    depth : int
        The depth of the object in the tree. Use to print the tree structure
        for debugging.
    _print : bool
        If True, print the tree structure for debugging.
    
    Returns
    -------
    node_dict : dict
        The dictionary detailing the type and metadata of the node.
    """
    inner_structure = build_structure(obj, key, depth+1, _print)

    # Container node
    if isinstance(inner_structure, dict):
        return _build_conatiner_node(obj, inner_structure)

    # Leaf node
    else:
        return _build_leaf_node(obj)
    

def _build_leaf_node(obj : Any) -> dict:
    """
    Builds a leaf node for the input object. Explicity handles string and array
    types. Strings are stored directly and serialised in the structure 
    dictionary and arrays have their shape and dtypes serialised for 
    reconstruction using the `equinox.deserialise_tree_leaves`.

    Parameters
    ----------
    obj : Any
        The object to get the leaves of.
    
    Returns
    -------
    node_dict : dict
        The dictionary detailing the type and metadata of the leaf.
    """
    # Basic info
    node_dict = {'node_type': 'leaf', 
                 'type': _format_type(obj)}
    
    # Handle string case
    if isinstance(obj, str):
        node_dict['value'] = obj
    
    # Append meta-data for arrays
    elif isinstance(obj, (Array, onp.ndarray)):
        node_dict['shape'] = obj.shape
        node_dict['dtype'] = _format_dtype(obj.dtype)
    
    return node_dict


def _build_conatiner_node(obj : Any, inner_structure : dict) -> dict:
    """
    Builds a container node for the input object.

    Parameters
    ----------
    obj : Any
        The object to build the container node for.
    inner_structure : dict
        The dictionary detailing the structure of the object.

    Returns
    -------
    node_dict : dict
        The dictionary detailing the type and metadata of the container.
    """
    return {'node_type': 'container', 
            'type': _format_type(obj), 
            'node': inner_structure}


def build_structure(obj      : Any, 
                    self_key : str  = None, 
                    depth    : int  = 0,
                    _print   : bool = False):
    """
    Recursively iterates over the input object in order to return a dictionary 
    detailing the strucutre of the of the object. Each node can be either a
    conainter node or leaf node. Each node is a dictionary with the following
    structure:

    {'node_type': 'container' or 'leaf',
     'type': str,
     'node': {
        param1 : {'node_type' : 'container', ...}, -> If container
        param2 : {'node_type' : 'leaf',
                  '...' : ...}, -> If leaf conatining any leaf metadata
        }
    
    Specific leaf metadata:
        Strings:
            String values are stored in the 'value' key and serialised via the
            returned structure dictionary.
        Jax/Numpy Arrays:
            Both the array shape and dtype are stored in the 'shape' and
            'dtype' keys respectively. 
    
    This method can be developed further to support more leaf types, since each
    individual leaf type can be made to store any arbitrarity metadata, as long
    as it can be serialised by json and used to deserialise it later.
    
    This dictionary can then be serialised using pickle and then later
    used to deserialise the object in conjunction with equinox leaf 
    serialise/deserialise methods.

    NOTE: This method is not equipped to handle `equinox.static_field()` 
    parameters, as they can be arbitrary data types but do not get serialised
    by the  `equinox.serialise_tree_leaves()` methods and hence require custom 
    serialisation via this method. Therefore this method currently does not
    handle this case correctly. This is not checked for currently so will
    silently break or result in unexpected behaviour.

    TODO: Serialise package versions in order to raise warnings when 
    deserialising about inconsistent versions.

    Parameters
    ----------
    obj : Any
        The object to get the leaves of.
    self_key : str = None
        The key of the object in the parent container. Use to print the tree
        structure for debugging.
    depth : int = 0
        The depth of the object in the tree. Use to print the tree structure
        for debugging.
    _print : bool = False
        If True, print the tree structure for debugging.
    
    Returns
    -------
    structure : dict
        The dictionary detailing the structure of the object.
    """
    structure = {}
    is_container = _check_node(obj, self_key, depth, _print=_print)

    # Recursive case
    if is_container:
        keys, accessor = _get_accessor(obj)

        # Iterate over parameters
        for key in keys:
            sub_obj = accessor(obj, key)
            structure[key] = _build_node(sub_obj, key, depth, _print)
                    
        # Deal with outermost container
        if depth == 0:
            return _build_conatiner_node(obj, structure)
        else:
            return structure

    # Base case    
    else:
        return obj


def serialise(path : str, obj : Any) -> None:
    """
    Serialises the input zodiax pytree to the input path. This method works by
    creating a dictionary detailing the structure of the object to be
    serialised. This dictionary is then serialised using `pickle` and the 
    pytree leaves are serialised using `equinox.serialise_tree_leaves()`. This
    object can then be deserialised using the `deserialise()` method. 

    This method is currently considered experimental for a number of reasons:
     - Some objects can not be gaurenteed to be deserialised correctly. 
     - User-defined classes _can_ be serialised but it is up to the user to 
     import the class into the global namespace when deserialising. 
     - User defined functions can not be gaurenteed to be deserialised
     correctly.
     - Different versions of packages can cause issues when deserialising. This
    metadata is planned to be serialised in the future and have warnings raised
    when deserialising.
     - static_field() parameters are not handled correctly. Since array types
     can be set as static_field() parameters, they are not serialised by
     `equinox.serialise_tree_leaves()` and hence require custom serialisation
     via this method. This is not checked for currently so will silently break.
     This can be fixed with some pre-filtering and type checking using the
     `.tree_flatten()` method.

    Parameters
    ----------
    path : str
        The path to serialise the object to.
    obj : Any
        The object to serialise.
    """
    # Check path type
    if not isinstance(path, (str, Path)):
        raise TypeError(f'path must be a string or Path, not {type(path)}')
    else:
        # Convert to string in case of Path for adding .zdx extension
        path = str(path)
    
    # Add default .zdx extension
    if len(path.split('.')) == 1:
        path += '.zdx'
    
    # Serialise
    structure = build_structure(obj)
    with open(path, 'wb') as f:
        pickle.dump(structure, f)
        tree_serialise_leaves(f, obj)
    

#######################
### Deserialisation ###
#######################
def _construct_class(modules_str : str) -> object:
    """
    Constructs an empty instance of some class from a string of the form
    'module.sub_module.class'. 

    Explicitly handled types: Nones, bool, int, float, str, complex, list, 
    tuple, dict, jax arrays, numpy arrays.

    Other types are imported and attempted to be instantiated.


    Parameters
    ----------
    modules_str : str
        The string of the form 'module.sub_module.class' to construct

    Returns
    -------
    object : object
        The instantiated class
    """
    # None case
    if modules_str == 'NoneType':
        return None

    # Regular python types
    elif modules_str in ['bool', 'int', 'float', 'str', 'complex', 'list', 
                         'tuple', 'dict']:
        return eval(f"{modules_str}()")

    # Array types
    elif modules_str == 'jaxlib.xla_extension.Array':
        import jax.numpy as jnp
        return jnp.array([])
    elif modules_str == 'numpy.ndarray':
        import numpy as onp
        return onp.array([])

    # Modules that existed in the global namespace - Search for it
    elif modules_str.startswith('__main__'):
        modules = modules_str.split('.')[1:]
        try:
            module = getattr(sys.modules['__main__'], modules[0])
            if len(modules) == 1:
                return module.__new__(module)
            else:
                for sub_module in modules:
                    module = getattr(module, sub_module)
                return module.__new__(module)
        except AttributeError:
            raise AttributeError(f"The module/class '{'.'.join(modules)}' "
                "originally existed in the global namespace, but does not "
                "exist in this script. It must be imported in order to be "
                "initialised.")
    
    # Finally attempt import of module
    else:
        from importlib import import_module as im
        modules = modules_str.split('.')

        if len(modules) == 1:
            raise ValueError(f"Unhandled case: {modules}")
        elif len(modules) == 2:
            cls = getattr(im(f'{modules[0]}'), modules[-1])
        else:
            sub_modules = '.'.join(modules[1:-1])
            cls = getattr(im(f'{modules[0]}.{sub_modules}'), modules[-1])
        return cls.__new__(cls)


def _load_container(obj : object, key : str, value : object) -> object:
    """
    Updates and return the object with the supplied key and value pair.

    TODO: Check the setting of different types of containers, ie list
     -> What about tuples which are immutable?

    Parameters
    ----------
    obj : object
        The object to update.
    key : str
        The key at which to update.
    value : object
        The corresponding value to update with.
    
    Returns
    -------
    obj : object
        The updated object.
    """
    # Recursively load the sub node
    sub_node = load_structure(value)

    # Object case
    if isinstance(obj, zodiax.Base):
        object.__setattr__(obj, key, sub_node)
    
    # List/tuple case
    # TODO: Check this preserves order
    # TODO: Check original input type and re-cast to tuple if necessary
    elif isinstance(obj, list):
        obj.append(sub_node)

    # Dict case
    else:
        obj[key] = sub_node
    
    return obj


def _load_leaf(obj : object, structure : dict) -> object:
    """
    Returns the leaf value, handing the special cases of strings and arrays.
    Can be expanded to deal with mode complex cases.

    Parameters
    ----------
    obj : object
        The object to update.
    structure : dict
        The structure to load.

    Returns
    -------
    obj : object
        The updated object.
    """
    # String case
    if isinstance(obj, str):
        obj = structure['value']

    # Jax Array casees
    elif isinstance(obj, Array):
        dtype = getattr(jnp, structure['dtype'])
        obj = jnp.zeros(structure['shape'], dtype=dtype)

    # Numpy Array case
    elif isinstance(obj, onp.ndarray):
        dtype = getattr(onp, structure['dtype'])
        obj = onp.zeros(structure['shape'], dtype=dtype)
    
    # All others
    else:
        pass
    
    return obj


def load_structure(structure : dict) -> object:
    """
    Load a structure from a dictionary to later be used in conjuction with
    `eqx.tree_deserialise_leaves()`.


    Custom leaf node desrialisation is handled by the `_load_leaf` function.
    
    Parameters
    ----------
    structure : dict
        The structure to load.

    Returns
    -------
    obj : object
        The loaded structure.
    """
    # Construct the object
    obj = _construct_class(structure['type'])

    # Container Node
    if structure['node_type'] == 'container': 
        
        # Iterarte over all parameters and update the object
        for key, value in structure['node'].items():
            obj = _load_container(obj, key, value)
        return obj
    
    # Leaf Node
    else: 
        return _load_leaf(obj, structure)


def deserialise(path : str):
    """
    Deserialises the input object at the input path.

    Parameters
    ----------
    path : str
        The path to serialise the object to.
    
    Returns
    -------
    obj : Any
        The deserialised object.
    """
    with open(path, 'rb') as f:
        structure = pickle.load(f)
        like = load_structure(structure)
        obj = tree_deserialise_leaves(f, like)
    return obj