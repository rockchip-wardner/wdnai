# ---------------------------------------------------------------------------
# wdnai cshim —— Qwen-Image-2.1 出图运行时（C 实现）
#
#   make            # 编译成 libqi_qwen.so（放在本目录）
#   make clean
#   make CC=clang
#
# ⚠️ 依赖：
#   - **只有系统自带的 librknn3_api.so**（门面会 dlopen rkcp 后端）
#     ⇒ 发布包里不需要带任何 Rockchip 的 .so / .bin。
#   - **rknn2（librknnrt.so）不是链接依赖**，而是 **dlopen** 的
#     ⇒ 没装 rknn2 runtime 的板子照样能编译 / 运行（VAE 自动退回 rknn3 链）。
# ---------------------------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra -Wno-unused-parameter -fPIC
CPPFLAGS += -Iinclude -Iqi
LDFLAGS  ?=
LDLIBS   := -lrknn3_api -lpthread -ldl -lm

SRCS := qi/api.cpp qi/util.cpp qi/te_encoder.cpp qi/np_noise.cpp \
        qi/rknn3_runner.cpp qi/resident_chain.cpp qi/vae_chain.cpp \
        qi/rknn2_runner.cpp qi/vae_rknn2.cpp
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
