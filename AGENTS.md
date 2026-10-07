# AGENTS.md

微信悬浮窗助手（macOS）：OCR 读微信窗口 → 本地模型判意图/风险 → LLM 生成候选回复 → 悬浮窗展示/一键填入。纯只读、零封号风险是**核心原则**，任何改动不得破坏。

## 目录与命令

- `src/perception.py` 抓图+OCR+抽消息（含自己发出消息的识别）；`src/judge.py` 判断层接口 + 主后端（decider-2b）；`src/judge_laya.py` 本地判断兜底（laya-coreml，Core ML，无 key 无网络）；`src/judge_jev.py` 云端判断（TypeSafe Jev，仅剩手动 CLI，应用不再调用）；`src/relationship.py` 关系信号规则层（字面线索 + 最近上下文，可解释、不调模型）；`src/generate.py` 候选生成（OpenAI/Anthropic 兼容 API）；`src/hud.py` 悬浮窗+轮询主循环；`src/power.py` 面板上的免 root 功耗读数（IOReport 能量轨 + AppleSmartBattery）；`src/fill.py` 辅助功能写入（AX 优先）；`src/visual_fill.py`+`src/input_region.py` 显式点击触发的视觉填入后备；`src/settings.py`+`src/settings_config.py` 原生模型设置窗（写回 env）；`src/builtin.py` 随包内置默认凭据；`src/styles.py` 话术；`src/userconfig.py` 配置加载
- 启动：`./start.command`（用户平时的方式；`Ctrl+C` 退出）。无 CI，自测靠贡献者跑离线回归套件 + 分层手工自测：

  ```bash
  uv run python -B -m unittest discover -s tests   # 离线回归：发出消息识别/填入安全/视觉后备/设置持久化（合成 OCR，不读屏、不调 API、不碰真凭据）
  uv run python src/perception.py                  # 感知层（读屏，见下方 CLI 验证陷阱）
  uv run python src/judge.py "明天早上能帮我带份早饭不"      # 主后端 decider-2b（首次跑会下载模型包）
  uv run python src/judge_laya.py "明天早上能帮我带份早饭不"  # 兜底 laya-coreml
  uv run python src/judge_zh_test.py               # 54 条意图回归（orig/ab/confirm 分组报分）——改判断层 prompt 后必须重跑
  uv run python src/generate.py --check            # 生成层凭据解析
  uv run python src/power.py 5                     # 功耗读数（免 root；真机跑，不进回归套件）
  uv run python -B probe/settings_smoke.py         # 设置窗冒烟（临时配置+本地 HTTP，渲染 PNG 到 /tmp）
  uv run python -B probe/hud_smoke.py              # 悬浮窗面板冒烟（合成 payload 离屏渲染，不读屏、不调模型）
  uv run python -B probe/judge_prompt_ab.py        # 判断层 prompt A/B（shipped vs 候选，54 条用例分组报分）
  HF_HUB_OFFLINE=1 uv run python -B probe/couple_prompt_ab.py --variants baseline,shipped --confirm # 情侣/暧昧意图与学生对照（需已缓存模型）
  ```

- `probe/` 探针与回归：`settings_smoke`（设置窗冒烟）、`hud_smoke`（悬浮窗面板冒烟，合成 payload 离屏渲染）、`judge_prompt_ab`（判断层 prompt A/B，走 judge 模块全局量，测的就是线上代码路径）、`perception_regression`（感知层离线回归）、`bootstrap_regression`（.app 启动 shell 回归）、`capability_probe`/`capture_probe`（真机验证 AX/抓图/OCR 通路）。

- 日志：`~/Library/Logs/jev-jarvis.log`，分阶段耗时（读屏/判断/生成/排序/端到端）。**刻意不含消息正文与候选文字**（用户可放心贴 issue），只在事件发生时打、不在每跳打；首次调用标注「首次」。
- 发版：版本号只有 `pyproject.toml` 一处；`./packaging/release.sh --publish` 从**干净 worktree** 构建（zip 解压回验+SHA256+gh release，显式 `--latest` + 发布后 Latest 指针自检，资产名固定 `jev-chat-jarvis-macos.zip` 保稳定直链）；无 Apple 公证，首次打开要教右键。
- 认领协议：动任何 issue 的代码前，先按 [CONTRIBUTING.md](CONTRIBUTING.md) 完成认领三步自检 + 评论认领 + 设 assignee——多人多 AI 并行扫 issue，不认领必撞车。

## 架构与硬约束

- **纯只读**：不注入、不 hook、不解密微信数据。「填入」是唯一写动作：**AX 辅助功能写入优先**；微信不暴露 AX 输入控件时，显式点击「填入」可走视觉后备（`visual_fill`：点输入区+键盘事件，**不发回车、不用剪贴板、不覆盖草稿、读回确认**，只在这一触发条件下用）。**别改回剪贴板+模拟 Cmd+V**（切前台不可靠、覆盖剪贴板、失败会贴进别的应用，见 `src/fill.py` 顶部注释）。
- **轮询**：定时器 0.25s 触发，`_next_read_ts` 门控分三档——静止（指纹相同）跳过 OCR、0.25s 一跳；**变化后先 0.45s×3 跳**（burst 下一条尽快被发现），持续再动才回 1s。**未变化帧仍要跑停稳判定**（复用 `_last_full` 缓存），否则分析永远不触发。停稳 `SETTLE_S=1.2` 是防刷屏**上限不能删**；连续 `STABLE_READS` 跳安静最早 `EARLY_SETTLE_S` 可提前开闸。最小分析间隔 `MIN_GAP_S=2.0` 不能删（预判命中路径本就免冷却）。
- **预判+生成都早跑**（`_prejudge_loop` / `_pregen_loop`，同款 latest-wins 槽位）：消息一出现两个半边同时起跑，停稳门只消费「文本仍是最新」的结果；生成结果还要话术匹配（`_take_pregen`），迟到/过期结果由 `applyCandidates_` 的话术守卫挡掉。候选**先上屏再排序**（prob=None 显示「排序中」，`_rank_payload` 完成后原位重排）。
- **分析在独立线程**（`_run_analysis` + `_analyzing` 防重入），别塞回 tick 线程——那会重新造成分析期间轮询停摆。
- **关系信号只做规则层**：`src/relationship.py` 不新增模型前向、不改 `judge.INTENTS`，只把字面线索和最近上下文变成可解释提示；任何“她一定喜欢你”式的确定结论都违反该层设计。改动用户可见措辞要同步 README，并先跑 `tests/test_relationship.py`。
- **YOLO 检测框**（`_build_overlay`/`applyBoxes_`，`JEV_BOXES=1` 启动即开、菜单栏可切、默认关）：透明点击穿透窗把最近一次 OCR 的消息画成检测框，纯视觉层——窗口 ID 抓图看不见它、不参与任何管线逻辑；坐标映射依赖 1x nominal 采集尺寸=窗口点尺寸（`capture_image(nominal=True)` 成立）。`Message` 的 x/w 是框几何，折行时在 `extract_messages` 里维护。
- **本地推理用 float16**：MPS 对 bfloat16 算子覆盖不全会走慢路径（实测 ~1.4s vs ~0.75s，准确率不变）。
- **OCR 用 Vision**：语言只留 `zh-Hans`（多加 en-US 逐块一致却慢 30%）、Accurate 档（Fast 漏字）、语言校正开着、别缩 ROI（丢上下文）。**采集分辨率降到 1x**（`kCGWindowImageNominalResolution`）是实测过的例外：合成中文 6 行 2x ~140ms → 1x ~100ms、逐字一致；布局常量全是归一化的，不受影响。
- **HTTP 走 keep-alive 池**（`generate.py` 的 `http_post_json`/`post_stream`，judge_jev 共用）：每次 urllib.urlopen 新建 DNS+TCP+TLS 白付 ~0.1–0.3 s。一次性与流式（SSE）请求都从池里拿连接（流式响应读完必须 `release()` 归还）；网络异常换新连接重试一次；>=300 按 `urllib.error.HTTPError` 形状抛（调用方 `e.read()` 拿正文），不跟随重定向。改池的行为先跑 `tests/test_generate_stream.py`（本地 SSE 端点，断言连接复用）。
- **配置只有 env 一种格式**（无 config.json）：`~/.config/jev-jarvis/env` 等，**凭据解析以 key 为准**——提供 key 的来源同时决定端点和模型。不提供第二种配置文件格式是有意为之。
- **设置窗只是 env 的编辑器**（`settings.py`/`settings_config.py`）：写回同一份 env（保留注释、0600 权限），**不改运行中凭据、保存后需重启生效**。用户一个 key 都没配时回落到 `src/builtin.py` 内置专用 token（限额+模型白名单+可过期，轮换只改这一个文件；`API_KEY` 留空即退回「必须自配」老行为）。
- 两种启动方式（`start.command` / `.app`）必须同 Python 3.12（包跟 `.python-version` 走）；`.app` 是「启动器包」（不冻结 torch，首次启动 uv 建 venv）。
- **模型缓存跟着启动方式走**：`userconfig` 在 import 时把 `HF_HOME`/`LAYA_COREML_CACHE` 指向 `<repo>/.models/`（已设则不覆盖，`get()` 会连 env 文件一起看，所以 CLI 直跑也认），源码跑因此不碰用户级缓存；`.app` 包内路径含 `.app/Contents/` 时**故意不设**——包内那份会被每次升级冲掉。`.models/` 已 gitignore（~8 GB），`build_app.sh` 只按清单拷文件、`release.sh` 只 zip 构建出的 `.app`，都不会带上它。改动这块先看 `tests/test_model_cache.py`。
- **功耗读数免 root，只信能读到的通道**（`src/power.py`，面板右上角、tick 里调）：IOReport 的 `Energy Model` 组里**只有名字以 ` Energy` 结尾的聚合能量轨**才是能量计数器——本机实测 `GPU Energy`（nJ）可用、`CPU Energy`（mJ）**恒为 0，即使 8 核满载**，而那部分正是 `powermetrics` 用 root 换来的；`GPU0`/`PCPU1DTL2a`/`ANE0`/`DRAM0` 是同一组里的 per-block 计数器，**加进去会凭空放大瓦数**。所以面板显示 `电池 x W`（放电≈整机）/`充电 x W`/`GPU x W`/`功耗 —`，**绝不编 CPU 或整机数字**。电池功率 = `Voltage(mV)×InstantAmperage(mA)`，插电且 |电流| < 20 mA 时电池说明不了整机功耗，必须退回 GPU 轨。采样节流 1 Hz、单次 ~0.4 ms，放在 `tick_` 的暂停/忙碌门**之前**（暂停或收起时读数照常走，它不读屏）；能量轨要两个采样点，插电时头 1 秒是「功耗 —」（电池瓦数不需要基线，首次就有）。改动先跑 `tests/test_power.py` + `probe/hud_smoke.py`（后者断言读数不压聊天名、不压齿轮、折叠后仍在）。

## 已知的坑

- **CLI 进程里验不了感知层**：独立 shell 进程里 `CGWindowListCreateImage` 会被拒（静默退子进程路径、无指纹）。验证要么用合成 CGImage 测纯函数，要么起真应用看日志。
- 坐标系：本模块布局常量（`CHAT_PANE_X_MIN` 等）是**底部原点**（Vision 口径）；`CGImageCreateWithImageInRect` 是**左上原点**，换算别搞反。
- 生成层**不能用 thinking 模型**（思考吃光 `max_tokens`，候选 0 条，面板只报「生成失败」误导用户）。
- README 实测数字皆有口径：历史意图 98.1%（decider-2b）来自**无上下文、54 条学生用例、仅意图单槽探针**；2026-10-03 将 `judge_zh_test.py` 改为调用应用实际的意图+风险路径，通用模式为 49/54（90.7%）、laya 为 20/54（37.0%），不能混用两个口径。改判断层 prompt 后必须重跑；判断耗时引用应用内实测（~1s），不是回归脚本均值。`JEV_CHAT_SCENE=relationship` 是显式选择的情侣/暧昧问句，主判断与兜底共享 `judge.intent_question(scene)`，不改标签描述或风险刻度；情侣合成调词集 29/36→31/36、冻结确认集 18/20→19/20，但学生集 48/54，因此不得自动取代通用场景。确认集只做最终确认，不能用来继续调词。
- **判断层 prompt 别凭感觉改，也别顺手重写标签描述**：两轮压缩措辞（保语义锚点）在职场口径上实测 81.8% / 77.3%，低于原文 86.4%——批评/要解释 的边界对措辞极敏感。2026-09 学生化那轮又量了一遍，结论一样：重写 7 条标签描述（保长度、换学生场景）在 40 条调词集上是 35/40，和原问句持平而没赚，而且给 `问进度` 描述塞「报告」会把「实验报告能发我参考下不」从 帮忙 拽走。最后生效的是**问句**（学生场景 + 两条判定规则），标签描述一字未动：54 条 46/54 → 53/54。改措辞前先跑 `probe/judge_prompt_ab.py` 拿前后数字，`confirm` 组只能用于最后确认，用它调词后就等于没了确认集。
- **风险分没有回归**：`RISK_LEVELS`/`RISK_QUESTION` 只有 `probe/judge_prompt_ab.py` 的 16 条 sanity 查单调性（mae/rho），没有金标、也没查刻度。实测 decider-2b 把高风险消息压在 3–5 分：「你把我们组的实验数据弄丢了？」4.7、「这格式全不对，重做一版」2.5——面板上的风险等级偏保守就是这个原因。要动刻度得先立一套带金标的风险回归，别只跑 sanity 就改。
- `.gitignore` 忽略全部 png 只放行 `docs/**`；新图片必须进 docs/。
- AppKit 控件宽度要渲染成 PNG 实测，`cellSize()` 会谎报。
- 判断模型冷启动 10–20s 是已知问题（见 issue #1 预热方案；指主后端 decider-2b 的 torch 加载，启动预热已兜住）；启动后第一条慢是正常现象，别误判成回归。预热**卡死**另有防护（`judge.LOAD_WAIT_S=30s`）：真实消息在加载锁上最多等 30s，超时抛 `JudgeNotReady`，`FallbackJudge` 对它做**单条** laya 兜底、不永久切换（日志实测过 547s 卡死加载把首条消息顶了 542s）；普通异常仍是首次失败永久切换，语义没变。相关单测 `tests/test_fallback_judge.py`。
- **自己发出的消息不能触发分析**（`perception.py` 识别发出侧，#15）：改抽消息/指纹逻辑后必跑 `tests/test_outgoing_messages.py`，否则会把回复当新消息造成自我循环。
- 改动用户可见行为要同步 README；待办与已定方案看 GitHub issues 和 README「下一步」。
