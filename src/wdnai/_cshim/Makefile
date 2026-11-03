# ---------------------------------------------------------------------------
# wdnai cshim —— Qwen-Image-2.1 出图运行时（C 实现）
#
#   make            # 编译成 libqi_qwen.so（放在本目录）
#   make clean
#   make CC=clang
#
# ⚠️ 依赖：**只有系统自带的 librknn3_api.so**（门面会 dlopen rkcp 后端）
#    ⇒ 发布包里不需要带任何 Rockchip 的 .so / .bin。
#    （板端：/usr/lib/librknn3_api.so；实测与模型包里那份**逐字节相同**。）
# ---------------------------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter -fPIC
CPPFLAGS += -Iinclude -Iqi
LDFLAGS  ?=
LDLIBS   := -lrknn3_api -lpthread -ldl -lm

SRCS := qi/api.cpp qi/util.cpp qi/te_encoder.cpp qi/np_noise.cpp \
        qi/rknn3_runner.cpp qi/resident_chain.cpp qi/vae_chain.cpp
OBJS := $(patsubst qi/%.cpp,build/%.o,$(SRCS))
TARGET := libqi_qwen.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)
	@echo "✅ $@"

build/%.o: qi/%.cpp
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf build $(TARGET)

-include $(wildcard build/*.d)
