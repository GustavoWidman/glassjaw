# host simulator build. pulls the real freertos kernel with its posix port so
# queue/semaphore/task semantics match the esp32 build exactly.
set(FREERTOS ${CMAKE_SOURCE_DIR}/third_party/FreeRTOS-Kernel)
add_library(freertos_posix STATIC
  ${FREERTOS}/tasks.c
  ${FREERTOS}/queue.c
  ${FREERTOS}/list.c
  ${FREERTOS}/timers.c
  ${FREERTOS}/event_groups.c
  ${FREERTOS}/stream_buffer.c
  ${FREERTOS}/portable/ThirdParty/GCC/Posix/port.c
  ${FREERTOS}/portable/ThirdParty/GCC/Posix/utils/wait_for_event.c
  ${FREERTOS}/portable/MemMang/heap_4.c
)
target_include_directories(freertos_posix PUBLIC
  ${FREERTOS}/include
  ${FREERTOS}/portable/ThirdParty/GCC/Posix
  ${CMAKE_SOURCE_DIR}/sim
)
target_compile_options(freertos_posix PRIVATE -Wno-unused-parameter)
target_link_libraries(freertos_posix PUBLIC pthread)

target_sources(gj_core PRIVATE ${CMAKE_SOURCE_DIR}/src/app/pipeline.cpp)
target_include_directories(gj_core PUBLIC
  ${FREERTOS}/include
  ${FREERTOS}/portable/ThirdParty/GCC/Posix
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/sim)

add_executable(gj_sim main.cpp)
target_link_libraries(gj_sim PRIVATE gj_core freertos_posix)
target_compile_options(gj_sim PRIVATE -Wall -Wextra -Werror)
