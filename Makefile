CC ?= cc

BUILD_DIR := build
SRC_DIR := src
DEMO_DIR := demo

TARGET := $(BUILD_DIR)/kmeans_demo
ELBOW_TARGET := $(BUILD_DIR)/elbow_demo
CLUSTERING_TARGET := $(BUILD_DIR)/clustering_demo

SRCS := \
	$(DEMO_DIR)/kmeans_demo.c \
	$(SRC_DIR)/kmeans.c \
	$(SRC_DIR)/csv.c

ELBOW_SRCS := \
	$(DEMO_DIR)/elbow_demo.c \
	$(SRC_DIR)/kmeans.c \
	$(SRC_DIR)/csv.c

CLUSTERING_SRCS := \
	$(DEMO_DIR)/clustering_demo.c \
	$(SRC_DIR)/kmeans.c \
	$(SRC_DIR)/dbscan.c \
	$(SRC_DIR)/csv.c

OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)
ELBOW_OBJS := $(ELBOW_SRCS:%.c=$(BUILD_DIR)/%.o)
CLUSTERING_OBJS := $(CLUSTERING_SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ELBOW_DEPS := $(ELBOW_OBJS:.o=.d)
CLUSTERING_DEPS := $(CLUSTERING_OBJS:.o=.d)

CFLAGS ?=
CFLAGS += -O2 -std=c11 -Wall -Wextra -I$(SRC_DIR)

LDFLAGS ?=
LDLIBS ?= -lm

.PHONY: all clean run dirs elbow clustering

all: $(TARGET) $(ELBOW_TARGET) $(CLUSTERING_TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)/$(SRC_DIR) $(BUILD_DIR)/$(DEMO_DIR)

$(TARGET): dirs $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(ELBOW_TARGET): dirs $(ELBOW_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(ELBOW_OBJS) $(LDLIBS)

$(CLUSTERING_TARGET): dirs $(CLUSTERING_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(CLUSTERING_OBJS) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	$(TARGET) --in data.csv --k 3 --out pred.csv --seed 42

elbow: $(ELBOW_TARGET)
	$(ELBOW_TARGET) --in data.csv --k-min 2 --k-max 10 --seed 42

clustering: $(CLUSTERING_TARGET)
	@echo "Использование clustering_demo:"
	@echo "  K-means: $(CLUSTERING_TARGET) data.csv 3 kmeans"
	@echo "  DBSCAN:  $(CLUSTERING_TARGET) data.csv 0.5 4 dbscan"

clean:
	@rm -rf $(BUILD_DIR)

-include $(DEPS)
-include $(ELBOW_DEPS)
-include $(CLUSTERING_DEPS)
