RCCF's VEX U Competition Team UCF7 - Daydream's RnD Repository for the 2025-2026 academic year.
Main Focus: AI Systems & Computer Vision

Using Python Version (default in debian 12 bookworm): 3.11.2  
Package versions in [requirements.txt](requirements.txt)  
To install packages use `pip install -r requirements.txt`  

### `init_map()` dependencies
Use of std::optional requires at least c++17.\
If this is not possible on pros use of an `is_valid` flag inside each\
struct must be used to check if the object was detected in a frame.\
<br>
>[!WARNING]
>The index of an array returned from `get_obj()` must always be checked to exist using `yourarrayname[ENUM_ALIAS_INDEX].has_value()`\
>If the `has_value()` std::optional member function returns `true` the element exists, and the `value()` std::optional member function must be used to acess the structure.\
>If the element at the index does not exist and a statement attempts to preform an operation on the non-existing element\
>a `const std::bad_optional_access` exception will be thrown\
>using the `.has_value()` std::optional member function bypasses using `try` `except` statements when accessing elements.

### `init_map()` return
returns an array of `GamePieceArray` type containing std::optoinal wrapper class elements that wrap a `GamePieceData` structure\
the `GamePeiceData` structure is definded in `objectHandler.h`\
<br>
# `get_obj()` return
returns an array of `GamePieceArray` type containing std::optoinal wrapper class elements that wrap a `GamePieceData` structure


Warning Note:
Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.