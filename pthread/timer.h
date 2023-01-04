#ifndef _TIMER_H_
#define _TIMER_H_

#include <time.h>

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
   clock_t t = clock(); \
   now = (double)(t) / CLOCKS_PER_SEC; \
}

#endif
