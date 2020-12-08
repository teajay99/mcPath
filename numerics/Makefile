TARGET ?= main
SRC_DIRS ?= ./src
LIB_DIRS ?= ./lib
INC_DIRS ?= ./include
SYS_LIBS ?= -lncurses -lutil
CXXFLAGS+= -pthread --std=c++17 -g -D_GLIBCXX_DEBUG -O3

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
LIBS := $(shell find $(LIB_DIRS) -name *.a)
DEPS := $(OBJS:.o=.d)


#all:
#	@echo $(value LIBS)

INC_DIRS += $(shell find $(SRC_DIRS) -type d) 

INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) $(LIBS) $(SYS_LIBS) -o $@ $(LOADLIBES) $(LDLIBS)
	$(RM) $(OBJS) $(DEPS)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)

#-include $(DEPS)
