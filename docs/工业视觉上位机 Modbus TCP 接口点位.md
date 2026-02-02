# 工业视觉上位机 Modbus TCP 接口点位

## 1. 通信与角色

- 物理链路：以太网（同一局域网）
- 协议：Modbus TCP（TCP 之上）
- 角色：
  - 上位机：Modbus TCP Server（从站语义，提供点表与数据块）
  - PLC：Modbus TCP Client（主站语义，周期性读取状态与结果，按需写入触发/复位命令）
- 推荐端口：
  - 标准端口：`502/tcp`
  - 工程建议：
    - 若运行环境允许绑定特权端口，使用 `502`
    - 若不便绑定特权端口（常见于非 root 进程），建议使用 `1502` 并在现场统一约定（PLC 侧直接连 `1502`）

## 2. 数据区使用约定

- 本接口仅使用以下数据区：
  - 0x Coils：命令区（PLC 写 / 上位机读）
  - 1x Discrete Inputs：状态与快速结果（上位机写 / PLC 读）
  - 3x Input Registers：详细结果块（上位机写 / PLC 读）

- 地址说明：
  - 表内同时给出 PDU 地址（0-based）与常见工程地址（1-based）
  - PLC 工程以工程地址配置为准；若设备存在 0/1 基址差异，以联调时 PLC 实际读写结果校准

### 2.1 区块地址偏移（新增要求）

PLC 要求：**各个区块的起始地址需要带有可配置的偏移量**。

- 每个区块（0x/1x/3x）均应支持独立的偏移量：  
  - `coil_offset`（0x Coils）  
  - `di_offset`（1x Discrete Inputs）  
  - `ir_offset`（3x Input Registers）
- 偏移量以 **PDU(0-based) 地址** 为基准计算
- 计算规则（表内地址为“未偏移”基准值）：
  - `PDU_effective = PDU_table + {coil_offset | di_offset | ir_offset}`
  - `ENG_effective = ENG_table + {coil_offset | di_offset | ir_offset}`
- 示例：
  - 若 1x 区块配置 `di_offset = 10`，则 `ST_HEARTBEAT_TOGGLE` 的 PDU 变为 `10`，工程地址变为 `10011`

> 注：偏移只改变区块内相对起始地址，不改变点位语义。

## 3. 功能码建议

- 写单线圈（触发/复位）：FC05 Write Single Coil
- 读离散输入（状态/快速结果）：FC02 Read Discrete Inputs
- 读输入寄存器（详细结果块）：FC04 Read Input Registers

## 4. 点位表

### 4.1 0x Coils（命令区，PLC 写 / 上位机读）

| PDU(0-based) | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 00001 | CMD_TRIG_TOGGLE | 1bit | 触发请求：PLC 每次触发翻转一次（0→1 或 1→0 均表示新触发） |
| 1 | 00002 | CMD_RESET | 1bit | 初始化/复位：上位机收到后执行初始化（清空队列与计数） |

CMD_RESET 语义：用于“初始化程序”。执行后 ST_RESULT_SEQ 从 1 重新开始，结果输出恢复为默认状态（见 6.2）。

### 4.2 1x Discrete Inputs（状态/快速结果，上位机写 / PLC 读）

| PDU(0-based) | 工程地址 | 点位名 | 类型 | 语义 |
| --- | --- | --- | --- | --- |
| 0 | 10001 | ST_HEARTBEAT_TOGGLE | 1bit | 心跳：上位机每 1s 翻转一次 |
| 1 | 10002 | ST_ACCEPT_TOGGLE | 1bit | 触发任务成功入队后翻转一次 |
| 2 | 10003 | ST_RESULT_TOGGLE | 1bit | 上位机提交新结果时翻转一次 |
| 3 | 10004 | ST_RESULT_OK | 1bit | 快速 OK |
| 4 | 10005 | ST_RESULT_NG | 1bit | 快速 NG（若 ERR=1，则 NG 也为 1） |
| 5 | 10006 | ST_RESULT_ERR | 1bit | 快速 ERR |

### 4.3 3x Input Registers（详细结果块，上位机写 / PLC 按需读）

| PDU(0-based) | 工程地址 | 点位名 | 类型 | 语义 |
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
| 9 | 30010 | ST_CYCLE_MS | uint16 | 处理耗时(ms)，饱和显示 |

## 5. 码表（默认建议，最终以实际需求冻结为准）

### 5.1 ST_RESULT_CODE（只给结论）

- `1` = OK
- `2` = NG
- `3` = ERROR（异常/不可判定/设备错误导致流程失败或结果不可用）

### 5.2 ST_ERROR_CODE（只给原因，仅当 RESULT=ERROR 时有效）

- `0` = NONE
- `1` = TIMEOUT
- `2` = DETECT_EXCEPTION（算法异常/运行时异常）
- `3` = CAMERA_ERROR
- `4` = QUEUE_OVERFLOW

约束：

- 当 `ST_RESULT_CODE != ERROR(3)` 时，`ST_ERROR_CODE` 必须为 `0`
- 当 `ST_RESULT_CODE == ERROR(3)` 时，`ST_ERROR_CODE` 必须为非 0
  - 若无法分类，使用预留的通用错误码（当前表未定义则可扩展）

## 6. 关键语义约束

### 6.1 结果提交一致性（上位机写入顺序）

每次生成结果时，上位机按以下顺序更新：

1. 写 3x 详细寄存器（时间、SEQ、RESULT_CODE、ERROR_CODE、CYCLE_MS）
2. 写 1x 快速位（OK/NG/ERR）
3. 最后翻转 `ST_RESULT_TOGGLE`（作为提交边界）

PLC 以 `ST_RESULT_TOGGLE` 的变化判定“新结果到达”，再读取快慢两层数据。

### 6.2 CMD_RESET 初始化行为

上位机执行初始化时至少满足：

- 清空触发队列、清空内部计数
- `ST_RESULT_SEQ` 重置为从 `1` 重新开始
- 清除结果输出到默认状态
  - 建议：`OK=0`，`NG=0`，`ERR=0`，`RESULT_CODE=0`
  - 或保留为上次值但不再翻转 `RESULT_TOGGLE`（具体取决于实现；如需冻结，可在 ICD 进一步明确默认值）

### 6.3 ST_CYCLE_MS 饱和显示

真实耗时超过 `uint16` 最大表示范围时，`ST_CYCLE_MS` 写入最大值（饱和）。

## 7. 轮询建议（便于联调）

- 心跳 DI：200–500 ms 轮询一次；连续 3 s 无翻转判离线
- 结果 DI：100–200 ms 轮询 `ST_RESULT_TOGGLE / OK / NG / ERR`
- 详细寄存器：仅在检测到 `ST_RESULT_TOGGLE` 变化后按需读取 `30001–30010` 全块
