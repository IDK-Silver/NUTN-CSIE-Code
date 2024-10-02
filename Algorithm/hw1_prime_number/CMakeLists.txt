cmake_minimum_required(VERSION 3.5)

project(Unit_Fermat)

set(
    Unit_Fermat_SourceCode
    fermat/src/main.cpp
)

add_subdirectory(lib/big_int big_int_lib_build)


add_executable(Unit_Fermat ${Unit_Fermat_SourceCode})

project(Unit_MillerRabin)
set(
    Unit_MillerRabin_SourceCode
    miller_rabin/src/main.cpp
)
add_executable(Unit_MillerRabin ${Unit_MillerRabin_SourceCode})
target_link_libraries(Unit_MillerRabin big_int_lib)

project(Unit_SieveOfEratosthenes)
set(
    Unit_SieveOfEratosthenes_SourceCode
    sieve_of_eratosthenes/src/main.cpp
)
add_executable(Unit_SieveOfEratosthenes ${Unit_SieveOfEratosthenes_SourceCode})

