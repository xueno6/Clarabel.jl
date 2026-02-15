add_executable(clarabel_cpp_tests
    test_main.cpp
)

target_link_libraries(clarabel_cpp_tests PRIVATE clarabel_cpp)

add_test(NAME clarabel_cpp_tests COMMAND clarabel_cpp_tests)
