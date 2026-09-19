// c++ entry point for the vendored freertos kernel. the kernel headers are
// plain c with no extern "c" guards, so everything gets wrapped here once
// (this is exactly what esp-idf's freertos component does for you).
#pragma once
extern "C" {
#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <semphr.h>
#include <timers.h>
}
