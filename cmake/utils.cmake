function(check_path_and_inform LIB PATH NAME)
    if(NOT EXISTS ${PATH})
        message(FATAL_ERROR "${PATH} does not exist, specify the path to a working\
    ${LIB} installation by -D${NAME}=...")
    else()
        message(STATUS "Found ${LIB}: ${PATH}")
    endif()
endfunction()
