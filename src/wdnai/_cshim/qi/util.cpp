#include "util.h"

#include <cstdio>
#include <sys/stat.h>

#include "rknn3_runner.h"   // qi::read_file

namespace qi {

std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a[a.size() - 1] == '/') return a + b;
  return a + "/" + b;
}

bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

bool read_whole_file(const std::string& path, std::vector<unsigned char>* out) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  out->resize((size_t)st.st_size);
  if (st.st_size == 0) return true;
  return read_file(path, out->data(), out->size());
}

}  // namespace qi
