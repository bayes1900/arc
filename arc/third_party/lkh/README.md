# LKH-3

本目录包含 LKH-3 的 C 源码与 ARC 的调用接口。

## 来源

- 原始项目：LKH-3，Keld Helsgaun
- 复制自 RACER：
  `swarm_exploration/utils/lkh_mtsp_solver/`

## 用途

- `src/*.c`：LKH-3 求解器
- `include/*.h`：LKH-3 头文件
- `lkh_mtsp_interface.cpp`：`solveMTSPWithLKH3()`
- ARC 通过 `arc_lkh` 静态库链接这些源文件

## 编译注意事项

1. **`-fcommon`**
   LKH 的头文件在多个 `.c` 文件中定义全局变量。GCC 10+ 默认 `-fno-common` 会报 multiple definition，必须加 `-fcommon`。

2. **`-w`**
   LKH 是 C 代码，旧式写法会产生大量警告，编译时关闭警告。

3. **全局状态**
   LKH 使用全局变量，不能并发调用。ARC 的 `LkhSolver` 用互斥锁串行化。

4. **文件路径**
   `.par` 中的 `PROBLEM_FILE` 和 `TOUR_FILE` 使用绝对路径。

## 许可

LKH-3 的许可条款以原始发布为准。本目录仅用于学习与复现 RACER。