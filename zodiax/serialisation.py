import sys
import pickle
import zodiax
from typing import Any
from jax import Array
from jax.typing import ArrayLike
import numpy as onp
import jax.numpy as jnp
from equinox import tree_serialise_leaves, tree_deserialise_leaves


__all__ = ['serialise', 'deserialise', 'load_structure', 'build_structure']


#####################
### Serialisation ###
#####################
def check_node(obj      : Any, 
               self_key : str  = None, 
               depth    : int  = 0., 
               _print   : bool = False) -> bool:
    """
    Checks if the input object is a container or a leaf node. If the object is
    a leaf a tuple of (False, obj) is returned. If the object is a container
    a tuple of (True, (obj, keys, accessor)) is returned. The accessor is 
    require to handle the different getter methods for dictionaries, lists, 
    objects etc. Also checks that object containter types have the required
    `_construct()` method required for deserilisation. It does not check that
    the `_construct()` method is valid, this is required to be done by the
    user.

    Container types:
        list, tuple, dict, zodiax.Base
    
    Leaf types:
        ArrayLike, bool, complex, float, int, numpy.ndarray, str, None

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
    (is_container, node) : tuple
        is_container : bool
            True if the object is a container, False if it is a leaf.
        node : tuple
            If is_container is True, the node is a tuple of (obj, keys, 
            accessor). If is_container is False, the node is the object itself.
    """
    t = '  ' * depth
    conatiner_types = (list, tuple, dict, zodiax.Base)
    leaf_types = (ArrayLike, bool, complex, float, int, str, type(None))

    # Contianer node
    if isinstance(obj, conatiner_types):
        if _print: print(f"{t}Node '{self_key}' of type: {type(obj)}")
        return True

    # Leaf node
    elif isinstance(obj, leaf_types):
        if _print: print(f"{t}Leaf '{self_key}' of type: {type(obj)}")
        return False

    # Unknown - Raise error
    else:
        raise ValueError(f"Unknown node '{self_key}' of type: {type(obj)}")
    

def get_accessor(obj : Any) -> tuple:
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
        if not hasattr(obj, '_construct'):
            raise ValueError(f"Object '{obj}' must have a `_construct()` "
                "method that instanitates the class and takes no inputs.")
        
        accessor = lambda object, x: getattr(object, x)
        keys = obj.__dataclass_fields__.keys()
        return keys, accessor

    # Dictionary types
    else:
        return obj.keys(), accessor


def format_type(obj : Any) -> str:
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


def format_dtype(dtype : onp.dtype) -> str:
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


def build_node(obj    : Any, 
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
        return build_conatiner_node(obj, inner_structure)

    # Leaf node
    else:
        return build_leaf_node(obj)
    

def build_leaf_node(obj : Any) -> dict:
    """
    Builds a leaf node for the input object.

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
                 'type': format_type(obj)}
    
    # Handle string case
    if isinstance(obj, str):
        node_dict['value'] = obj
    
    # Append meta-data for arrays
    elif isinstance(obj, (Array, onp.ndarray)):
        node_dict['shape'] = obj.shape
        node_dict['dtype'] = format_dtype(obj.dtype)
    
    return node_dict


def build_conatiner_node(obj : Any, inner_structure : dict) -> dict:
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
            'type': format_type(obj), 
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
                  '...' : ...}, -> If leaf where '...' conatins any leaf metadata
        }
    
    Specific leaf metadata:
        Strings:
            String values are stored in the 'value' key and serialised via the
            json
        Jax/Numpy Arrays:
            Both the array shape and dtype are stored in the 'shape' and
            'dtype' keys respectively. 
    
    This method can be developed further to support more leaf types, since each
    individual leaf type can be made to store any arbitrarity metadata, as long
    as it can be serialised by json and used to deserialise it later.
    
    This dictionary can then be serialised using json and then later
    used to deserialise the object in conjunction with equinox leaf 
    serialise/deserialise methods.

    NOTE: In order for the deseriaslisation proccess to be automated, all 
    class instances must have a `_construct()` method which instantiates an
    'empty' instance of th class, which can then be populated with the
    deserialised parameters.

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
    if depth == 0 and not isinstance(obj, zodiax.Base):
        raise ValueError("Input object must be a `zodiax` object.")

    structure = {}
    is_container = check_node(obj, self_key, depth, _print=_print)

    # Recursive case
    if is_container:
        keys, accessor = get_accessor(obj)

        # Iterate over parameters
        for key in keys:
            sub_obj = accessor(obj, key)
            structure[key] = build_node(sub_obj, key, depth, _print)
                    
        # Deal with outermost container
        if depth == 0:
            return build_conatiner_node(obj, structure)
        else:
            return structure

    # Base case    
    else:
        return obj


def serialise(path : str, obj : Any) -> None:
    """
    Serialises the input object to the input path.

    Parameters
    ----------
    path : str
        The path to serialise the object to.
    obj : Any
        The object to serialise.
    """
    structure = build_structure(obj)
    with open(path, 'wb') as f:
        pickle.dump(structure, f)
        tree_serialise_leaves(f, obj)
    

#######################
### Deserialisation ###
#######################
def instantiate_class(cls : object) -> object:
    """
    Constructs an 'empty' instance of the object using the class's _construct
    method if it exists, then tries to construct it with an empty call of the 
    class, otherwise returns the class itself.
    
    Parameters
    ----------
    cls : object
        The class to instantiate
    
    Returns
    -------
    object : object
        The instantiated class
    """
    if hasattr(cls, '_construct'):
        return cls._construct()
    else:
        try:
            return cls()
        except BaseException as e:
            print(f"Error: {e}")
            return cls


def construct_class(modules_str : str) -> object:
    """
    Constructs an empty instance of some class from a string of the form
    'module.sub_module.class'. 

    Parameters
    ----------
    modules_str : str
        The string of the form 'module.sub_module.class' to construct

    Returns
    -------
    object : object
        The instantiated class
    """
    # Regular Python types
    try:
        return eval(f"{modules_str}()")
    except:
        pass

    # None case
    if modules_str == 'NoneType':
        return None

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
                return instantiate_class(module)
            else:
                for sub_module in modules:
                    module = getattr(module, sub_module)
                return instantiate_class(module)
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
        return instantiate_class(cls)


def load_container(obj : object, key : str, value : object) -> object:
    """
    Updates and return the object with the supplied key and value pair.

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
        
        # Handle annoying None edge case - zodiax issue #2
        if sub_node is None:
            key, sub_node = [key], [sub_node]

        obj = obj.set(key, sub_node)
    
    # Dict case
    else:
        obj[key] = sub_node
    
    return obj


def load_leaf(obj : object, structure : dict) -> object:
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


    Custom leaf node desrialisation is handled by the `load_leaf` function.
    
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
    obj = construct_class(structure['type'])

    # Container Node
    if structure['node_type'] == 'container': 
        
        # Iterarte over all parameters and update the object
        for key, value in structure['node'].items():
            obj = load_container(obj, key, value)
        return obj
    
    # Leaf Node
    else: 
        return load_leaf(obj, structure)


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