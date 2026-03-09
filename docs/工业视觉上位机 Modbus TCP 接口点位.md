# 工业视觉上位机 Modbus TCP 接口点位（v3 草案）

## 1. 通信与角色

- 物理链路：以太网（同一局域网）
- 协议：Modbus TCP
- 角色：
  - 上位机：Modbus TCP Server（从站语义）
  - PLC：Modbus TCP Client（主站语义）
- 端口：以配置文件为准（现场常用 `1502`，也可 `502`）

## 2. 数据区与偏移

- 本接口使用以下区块：
  - `0x Coils`：触发命令（PLC 写）
  - `1x Discrete Inputs`：快速状态/结果（上位机写）
  - `3x Input Registers`：详细结果、计数、配方回执（上位机写）
  - `4x Holding Registers`：配方切换请求参数（PLC 写）
- 偏移量（均为 PDU 0-based 偏移）：
  - `coil_offset`
  - `di_offset`
  - `ir_offset`
  - `hr_offset`
- 计算规则：
  - `PDU_effective = PDU_table + offset`
  - `ENG_effective = ENG_table + offset`

## 3. 功能码建议

- 触发：`FC05`（Write Single Coil）
- 读 DI：`FC02`（Read Discrete Inputs）
- 读 IR：`FC04`（Read Input Registers）
- 写 HR：`FC06`/`FC16`（Write Single/Multiple Registers）

## 4. 点位表

### 4.1 0x Coils（PLC 写）

| PDU | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 00001 | CMD_TRIG_TOGGLE | 1bit | 触发请求。PLC 每次触发翻转一次（0→1 或 1→0） |

### 4.2 1x Discrete Inputs（上位机写，PLC 读）

| PDU | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 10001 | ST_HEARTBEAT_TOGGLE | 1bit | 心跳，1s 翻转一次 |
| 1 | 10002 | ST_ACCEPT_TOGGLE | 1bit | 触发成功入队后翻转 |
| 2 | 10003 | ST_RESULT_TOGGLE | 1bit | 新结果提交后翻转 |
| 3 | 10004 | ST_RESULT_OK | 1bit | 快速 OK |
| 4 | 10005 | ST_RESULT_NG | 1bit | 快速 NG（若 ERR=1，则 NG 也为 1） |
| 5 | 10006 | ST_RESULT_ERR | 1bit | 快速 ERR |

### 4.3 4x Holding Registers（PLC 写，上位机读）

| PDU | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 40001 | RECIPE_SLOT | uint16 | 待切换配方槽位（1..N） |
| 1 | 40002 | RECIPE_SEQ | uint16 | 命令序号。每次新请求必须递增（同槽位重载也要递增） |

### 4.4 3x Input Registers（上位机写，PLC 读）

| PDU | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 30001 | ST_TRIG_YEAR_UTC | uint16 | 触发时间 UTC 年 |
| 1 | 30002 | ST_TRIG_MONTH_UTC | uint16 | 触发时间 UTC 月 |
| 2 | 30003 | ST_TRIG_DAY_UTC | uint16 | 触发时间 UTC 日 |
| 3 | 30004 | ST_TRIG_HOUR_UTC | uint16 | 触发时间 UTC 时 |
| 4 | 30005 | ST_TRIG_MIN_UTC | uint16 | 触发时间 UTC 分 |
| 5 | 30006 | ST_TRIG_SEC_UTC | uint16 | 触发时间 UTC 秒 |
| 6 | 30007 | ST_RESULT_SEQ | uint16 | 结果序号（循环计数） |
| 7 | 30008 | ST_RESULT_CODE | uint16 | 结果结论码（见 5.1） |
| 8 | 30009 | ST_ERROR_CODE | uint16 | 错误原因码（见 5.2） |
| 9 | 30010 | ST_CYCLE_MS | uint16 | 处理耗时 ms（饱和） |
| 10 | 30011 | ST_COUNT_TOTAL | uint16 | 累计总数（饱和） |
| 11 | 30012 | ST_COUNT_OK | uint16 | 累计 OK（饱和） |
| 12 | 30013 | ST_COUNT_NG | uint16 | 累计 NG（饱和） |
| 13 | 30014 | ST_COUNT_ERR | uint16 | 累计 ERR（饱和） |
| 14 | 30015 | ST_RECIPE_ACK_SEQ | uint16 | 最近完成的配方命令序号 |
| 15 | 30016 | ST_RECIPE_ACK_STATUS | uint16 | 最近配方命令状态（见 5.3） |

## 5. 码表

### 5.1 ST_RESULT_CODE

- `1` = OK
- `2` = NG
- `3` = ERROR

### 5.2 ST_ERROR_CODE（仅当 RESULT=ERROR 时有效）

- `0` = NONE
- `1` = TIMEOUT
- `2` = DETECT_EXCEPTION
- `3` = CAMERA_ERROR
- `4` = QUEUE_OVERFLOW

约束：

- `ST_RESULT_CODE != 3` 时，`ST_ERROR_CODE` 必须为 `0`
- `ST_RESULT_CODE == 3` 时，`ST_ERROR_CODE` 必须为非 `0`

### 5.3 ST_RECIPE_ACK_STATUS

- `0` = IDLE（启动初始态）
- `1` = RUNNING（正在执行配方切换）
- `2` = OK（执行成功）
- `3` = ERR（执行失败）

## 6. 关键语义

### 6.1 结果提交顺序

每次产出结果时：

1. 先写 IR 结果块（`30001..30010`）
2. 再写 DI 快速结果（`OK/NG/ERR`）
3. 最后翻转 `ST_RESULT_TOGGLE`

### 6.2 配方切换命令流程（HR + IR）

PLC 发起：

1. 写 `RECIPE_SLOT`
2. 写 `RECIPE_SEQ`（相对上一条命令递增）

上位机处理：

1. 检测到 `RECIPE_SEQ` 新值后，置 `ST_RECIPE_ACK_STATUS=RUNNING`
2. 执行切换
3. 完成时写：
   - `ST_RECIPE_ACK_SEQ = RECIPE_SEQ`
   - `ST_RECIPE_ACK_STATUS = OK` 或 `ERR`

PLC 判定命令完成：

- `ST_RECIPE_ACK_SEQ == RECIPE_SEQ` 即本次命令已完成
- 再看 `ST_RECIPE_ACK_STATUS` 判断成功或失败

说明：

- `ST_RECIPE_ACK_STATUS` 不强制清零，保持最近一次命令状态即可。

### 6.3 同槽位重载语义（替代 CMD_RESET）

当 `RECIPE_SLOT` 等于当前活动槽位时，仍然执行一次完整软复位流程：

- 停止接收新触发并清空在途队列
- 重建并替换检测器
- 清空统计与历史缓存（TOTAL/OK/NG/ERR 重新计数）
- 清空 Modbus 结果输出区（DI/IR 结果位与结果寄存器恢复默认态）

即：本版本通过“同槽位切换”实现原先的复位需求，不再定义独立 `CMD_RESET`。

### 6.4 饱和策略

- `ST_CYCLE_MS`、`ST_COUNT_TOTAL/OK/NG/ERR` 均为 `uint16` 饱和显示（超过 `65535` 写 `65535`）。

## 7. 轮询建议

- DI 心跳：`200~500ms` 轮询一次；连续 `3s` 无翻转判离线
- DI 结果：`100~200ms` 轮询 `ST_RESULT_TOGGLE/OK/NG/ERR`
- IR 结果：检测到 `ST_RESULT_TOGGLE` 变化后读取 `30001..30016`
- 配方命令：PLC 写完 `RECIPE_SLOT/RECIPE_SEQ` 后轮询 `ST_RECIPE_ACK_SEQ/STATUS` 直到完成
