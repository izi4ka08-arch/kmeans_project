CC ?= cc

BUILD_DIR := build
SRC_DIR := src
DEMO_DIR := demo

TARGET := $(BUILD_DIR)/kmeans_demo

SRCS := \
	$(DEMO_DIR)/kmeans_demo.c \
	$(SRC_DIR)/kmeans.c \
	$(SRC_DIR)/csv.c

OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

CFLAGS ?=
CFLAGS += -O2 -std=c11 -Wall -Wextra -I$(SRC_DIR)

LDFLAGS ?=
LDLIBS ?=

.PHONY: all clean run dirs

all: $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)/$(SRC_DIR) $(BUILD_DIR)/$(DEMO_DIR)

$(TARGET): dirs $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	$(TARGET) --in data.csv --k 3 --out pred.csv --seed 42

clean:
	@rm -rf $(BUILD_DIR)

-include $(DEPS)
