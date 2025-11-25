# `init_map()` dependencies\
>[!WARNING]
>Use of std::optional requires at least c++17.\
>If this is not possible on pros use of an is_valid flag inside each\
>struct must be used to check if the object was detected in a frame.\
>The index of an array returned from `get_obj()` must always be checked to exist using `yourarrayname[ENUM_ALIAS_INDEX].has_value()`\
>If the `has_value()` std::optional member function returns `true` the element exists, and the `value()` std::optional member function must be used to acess the structure.\
>If the element at the index does not exist and a statement attempts to preform an operation on the non-existing element\
>a `const std::bad_optional_access` exception will be thrown

# `init_map()` return\
returns an array of `GamePieceArray` type containing std::optoinal wrapper class elements that wrap a `GamePieceData` structure\
the `GamePeiceData` structure is definded in `objectHandler.h`\
\
# `get_obj()` return\
returns an array of `GamePieceArray` type containing std::optoinal wrapper class elements that wrap a `GamePieceData` structure
