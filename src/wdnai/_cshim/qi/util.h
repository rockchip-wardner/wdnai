// ---------------------------------------------------------------------------
// wdnai cshim —— 通用小工具（路径 / 文件读写）
//
// 与参考实现（deploy/src/main.cpp 里的同名静态函数）**逐字一致**，
// 只是从 main.cpp 抽出来给 C API 用。
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

namespace qi {

// ⚠️ read_file / write_file 已经在 rknn3_runner.h 里声明并实现（同一个 qi 命名空间）
//    ⇒ 这里**不要**重复定义，否则链接期 multiple definition。
//    join_path / file_exists / read_whole_file 是本文件新增的。

std::string join_path(const std::string& a, const std::string& b);
bool file_exists(const std::string& p);
// 读整个文件（失败返回 false）
bool read_whole_file(const std::string& path, std::vector<unsigned char>* out);

}  // namespace qi
