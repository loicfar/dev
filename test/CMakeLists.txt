set(BASE_EXPORT_HEADER_FILES
    export/velta/ai/ai_api.hpp
    export/velta/ai/factory.hpp
    export/velta/ai/i_dataset.hpp
    export/velta/ai/i_nn.hpp
    export/velta/ai/options.hpp
)

set(BASE_HEADER_FILES
    hpp/velta/ai/autoencoder.hpp
    hpp/velta/ai/dataset.hpp
    hpp/velta/ai/dcnn.hpp
    hpp/velta/ai/dqn.hpp
    hpp/velta/ai/io_dataset.hpp
    hpp/velta/ai/mlp.hpp
    hpp/velta/ai/nn.hpp
    hpp/velta/ai/utils.hpp
)

set(BASE_SOURCE_FILES
    cpp/autoencoder.cpp
    cpp/dataset.cpp
    cpp/dcnn.cpp
    cpp/dqn.cpp
    cpp/factory.cpp
    cpp/io_dataset.cpp
    cpp/i_dataset.cpp
    cpp/i_nn.cpp
    cpp/mlp.cpp
    cpp/nn.cpp
    cpp/options.cpp
    cpp/utils.cpp
)

set(EXPORT_HEADER_FILES
    ${BASE_EXPORT_HEADER_FILES}
)

set(HEADER_FILES
    ${BASE_HEADER_FILES}
)

set(SOURCE_FILES
    ${BASE_SOURCE_FILES}
)

set(PCH_FILES
    export/velta/ai/pch/stl.hpp
    export/velta/ai/pch/torch.hpp
)

source_group("ai/cpp" FILES ${BASE_SOURCE_FILES})
source_group("ai/hpp" FILES ${BASE_HEADER_FILES})
source_group("ai/export" FILES ${BASE_EXPORT_HEADER_FILES})
source_group("ai/pch" FILES ${PCH_FILES})

add_library(ai SHARED ${EXPORT_HEADER_FILES} ${HEADER_FILES} ${SOURCE_FILES} ${PCH_FILES})

add_lib_prefix(ai)

target_link_libraries(ai
    PRIVATE
        protobuf::libprotobuf
        ${TORCH_LIBRARIES}
        math
        tool
)

set_torch(ai)

velta_target_include_directories(ai hpp)

target_precompile_headers(ai PRIVATE ${PCH_FILES})

install(FILES ${BASE_EXPORT_HEADER_FILES} DESTINATION include/velta/ai)

add_compile_definitions(AI_LIBRARY)