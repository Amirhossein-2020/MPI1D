#MPI_PROCS=1

# Editable build settings. Uncomment or add assignments below.
# Standard options: CC, CPPFLAGS, CFLAGS, LDFLAGS, LDLIBS.
# Use += to extend defaults, or = to replace them.
CACHE_LINE = $(shell getconf LEVEL1_DCACHE_LINESIZE)
CFLAGS +=	-O3	-march=sapphirerapids	-DCACHE_LINE_SIZE=$(CACHE_LINE)
#	-march=znver5
# CPPFLAGS += -DMY_OPTION=1
# LDFLAGS += -L/path/to/libraries
# LDLIBS += -lmy_library
# Command-line assignments take precedence, e.g. make CFLAGS="-O2 -Wall".
