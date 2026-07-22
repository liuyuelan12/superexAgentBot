# 帮助中心抓取与冲突治理

对应任务书《SuperEx.com 全站爬取、知识库整理与 Wiki 迁移》。

## 为什么不做「全站爬取」

任务书要求「尽可能多抓取」,但实测数据推翻了这个做法:

| 事实 | 数据 | 影响 |
|---|---|---|
| `www.superex.com` 是 SPA | curl 只拿到空壳 HTML;`robots.txt` / `sitemap.xml` 都被 catch-all 路由返回同一个页面 | 官网页面只能靠浏览器渲染逐页取,无法批量爬 |
| 帮助中心 95% 是公告 | 英文站 2878 篇中 2744 篇是上币/下币/维护/活动公告 | 全量入库会把规则类内容淹没 |
| 全语种总量 | 11 语种约 13,000 篇 | 按实测嵌入速度(bge-m3 / MPS,约 0.5 块/秒)需约 30 小时 |

因此按「只要规则类、全 11 语种」执行:**1,279 篇**规则/教程文章,约 2,882 个 chunk。

## 数据来源

`support.superex.com` 提供公开的 Zendesk Help Center API,比爬 SPA 可靠得多:

```
/api/v2/help_center/locales.json
/api/v2/help_center/{locale}/categories.json
/api/v2/help_center/{locale}/sections.json
/api/v2/help_center/{locale}/articles.json
```

公告按 **category id** 过滤(`kb/zendesk.py:ANNOUNCEMENT_CATEGORY_IDS`)。
分类 id 在 11 个语种间完全一致,已验证 —— 所以按 id 过滤是语言安全的,按名称过滤则不是。

## 管线

```
scripts/crawl_help_center.py        # 抓取 → raw/帮助中心/<locale>/<category>/*.md
        ↓                           #        + output/superex-help-center-sitemap.{json,md}
kb/loader.py:_load_help_center      # frontmatter 的 url / updated_at 进入 chunk metadata
        ↓
scripts/rebuild_index.py --force    # 重建向量 + BM25 索引
        ↓
scripts/detect_conflicts.py         # → output/superex-wiki-conflict-report.md
```

### 常用命令

```bash
python -m scripts.crawl_help_center                    # 全部 11 语种
python -m scripts.crawl_help_center --locale zh-hk     # 单语种
python -m scripts.crawl_help_center --dry-run          # 只数数,不写盘
python -m scripts.detect_conflicts --no-llm            # 只出候选,零 LLM 成本
python -m scripts.preview_answer "vip6 手续费"          # 离线看检索+答案+HTML
```

## 多语言检索

同一事实在 11 个语种里是 **11 篇独立文章**(Zendesk 各 locale 的 article id 不共享),
文本不同所以去重折叠不了它们。若不处理,中文提问会被韩文/日文文章占满 TOP_K。

`config.py:LANG_MISMATCH_PENALTY` 对**非用户语言的帮助中心文档**降权:
既然用户的语言在已抓取的 11 种之内,同语言版本必然存在,没必要让译文占位。
土耳其语、阿拉伯语等**不在**抓取范围的语言不受此惩罚 —— 对这些用户,跨语言检索是唯一能拿到答案的途径。

## 冲突治理

处理原则(运营方明确):**以 SuperEx 官网 / 最新链接为准**。落地在两处:

1. **离线**:`scripts/detect_conflicts.py` 产出冲突报告供人工确认,**不自动改写**任何既有文件。
2. **在线**:`bot/handlers.py:_format_hits` 给每个 chunk 打 `[OFFICIAL, updated YYYY-MM-DD]`
   或 `[internal note]` 标签,`SYSTEM_ANSWER` 第 15 条据此裁决 —— 官方压内部,同为官方取更新的。
   所以即使旧内容还留在库里,回答也会采用官方值。

### 检测方法与其取舍

先做确定性预筛,只把幸存者交给 LLM。**四道门槛全部是踩了误报的坑之后加的**,
每一道都对应一类真实发生过的错误判定:

| # | 门槛 | 起因 |
|---|---|---|
| 1 | 只比对**同一单位的数值分歧** | 换个说法不算冲突,`0.2%` vs `0.1%` 才算 |
| 2 | 主题须在**声明句级**一致,不能只在文档级 | 早期把「手续费 0.02%」和「保证金率≤100%」凑成一对(都含 `%`),LLM 顺势编造出一条不存在的冲突,还给 1.0 置信度 |
| 3 | 产品归属**只看元数据**(路径 / 标题 / Zendesk 板块),不看正文 | Index Futures 官方规则是保证金率 ≤100%,U 本位永续是 ≤0%,**两者都对**。而一篇长教程正文里会顺带提到别的产品,导致产品集合互相有交集而放行 |
| 4 | **算例整块排除** | 「案例二:体验金 100 → 保证金 300」与「案例三:體驗金 20 → 保證金 220」各自算术都正确,是不同算例而非参数变更 |

误报收敛过程:`75 → 19 → 13 → 9 → 0` 候选确认数 `8 → 5 → 3 → 1 → 0`。

**结论是 0 条真实冲突** —— 剩余候选全部是不同返佣项目之间的比较
(KOL 代理商 30-60% / 邀请好友 40% / Free Market 副本分润 80% / AMM 100% 给 LP),
LLM 正确判否。

> 每一条误报都会诱导人去"修正"本来正确的内容,所以这里宁可牺牲召回换精确率。
> 如果以后要放宽,请先补真实语料上的回归用例 —— `tests/test_conflicts.py` 里
> 每道门槛都有对应的 regression test。

LLM prompt 明令禁止推断文本里没写的数值,并点名「邀请返佣 / KOL 返佣 / 副本分润是不同项目,不得比较」。

只比对 zh-hk / en-001:既有 wiki 和客服资料是中英文写的,拿波斯语译文去比会产出噪音而不是发现。
