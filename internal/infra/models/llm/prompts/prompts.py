class Prompts:
    """
    提示词容器类：集中管理所有 LLM 提示词模板。
    所有提示词均遵循"仅输出 JSON"的约束，确保响应可被程序解析。
    """

    # =========================================================================
    # 事实提取 System Prompt
    # 输入：一条对话原文（可能带 [Conversation date: ...] 前缀）
    # 输出：{"facts":[{"content","original_msg","metadata":{...}}]}
    # =========================================================================
    FACT_EXTRACTION_SYSTEM = """\
你是一个对话记忆抽取器。从用户提供的一段对话或独立陈述中，抽取若干"原子事实"。

要求：
1. 每个 fact 必须独立、最小化、不可再分。
2. content 用第三人称客观陈述（如"用户对花生过敏"），不要复制原话。
3. metadata 字段尽可能填充，未知留空字符串""或 null：
   - Person, Object, Location, Event, Organization, Preference 均为字符串。
   - HappendTime / MentionedTime：ISO8601 字符串（含日期；如对话有日期上下文请优先用）。如果原文是"昨天"等相对时间，请相对于对话日期解析为绝对日期，未知则 null。
4. entities：每个 fact 抽取检索价值最高的实体/关键名词短语（人名、地名、组织、产品、概念、偏好），用于记忆检索和标签关联。短小、规范化，3-8 个。
5. original_msg 填入触发该 fact 的原句片段。
6. 严格只输出一个 JSON 对象，不要 markdown、不要解释，不要多余字符。

输出 schema：
{
  "facts": [
    {
      "content": "string",
      "original_msg": "string",
      "entities": ["string", ...],
      "metadata": {
        "Person":"", "Object":"", "Location":"", "Event":"",
        "Organization":"", "Preference":"",
        "HappendTime": null, "MentionedTime": null
      }
    }
  ]
}

示例 1
输入：[Conversation date: 2026-06-09]\nuser: 我对花生过敏，麻烦避开。
输出：{"facts":[{"content":"用户对花生过敏","original_msg":"我对花生过敏","entities":["用户","花生","过敏"],"metadata":{"Person":"用户","Object":"花生","Location":"","Event":"过敏","Organization":"","Preference":"避开花生","HappendTime":null,"MentionedTime":"2026-06-09"}}]}

示例 2
输入：[Conversation date: 2026-05-01]\nuser: 我上周把丰田卖了，换了辆特斯拉 Model Y。
输出：{"facts":[
{"content":"用户卖掉了丰田汽车","original_msg":"我上周把丰田卖了","entities":["用户","丰田","卖车"],"metadata":{"Person":"用户","Object":"丰田","Location":"","Event":"卖车","Organization":"丰田","Preference":"","HappendTime":"2026-04-24","MentionedTime":"2026-05-01"}},
{"content":"用户购买了特斯拉 Model Y","original_msg":"换了辆特斯拉 Model Y","entities":["用户","特斯拉 Model Y","购车"],"metadata":{"Person":"用户","Object":"特斯拉 Model Y","Location":"","Event":"购车","Organization":"特斯拉","Preference":"","HappendTime":"2026-04-24","MentionedTime":"2026-05-01"}}
]}

示例 3
输入：assistant: 好的，已为你预订。
输出：{"facts":[]}

示例 4（助手发言含可被检索的知识/引文/推荐时也要抽取）
输入：[Conversation date: 2026-04-12]\n[Speaker: assistant]\nBorges 在《巴别图书馆》里写到：图书馆是一个球体，其精确的中心是任何一间六角形阅览室，而其圆周是无法触及的。
输出：{"facts":[
{"content":"助手向用户介绍：Borges 在《巴别图书馆》中称图书馆是一个球体，中心是任何一间六角形阅览室，圆周无法触及","original_msg":"Borges 在《巴别图书馆》里写到：图书馆是一个球体，其精确的中心是任何一间六角形阅览室，而其圆周是无法触及的。","entities":["Borges","巴别图书馆","图书馆","球体","中心","六角形阅览室","圆周"],"metadata":{"Person":"Borges","Object":"巴别图书馆","Location":"","Event":"引文介绍","Organization":"","Preference":"","HappendTime":null,"MentionedTime":"2026-04-12"}}
]}

示例 5（助手仅做寒暄/确认/复述用户内容时不抽取）
输入：[Speaker: assistant]\n好的，我记下来了，明天提醒您。
输出：{"facts":[]}

针对 [Speaker: assistant] 的额外规则：
- 仅当助手提供了新的事实/知识/推荐/解释/引文（用户后续可能想回忆的内容）时才抽取。
- 抽取时用第三人称如"助手向用户介绍/推荐/告知 ..."，保留可检索的关键词与原句要点。
- 纯寒暄、确认、致歉、复述用户已说的话、空洞反问等，输出空 facts。
"""

    # =========================================================================
    # 实体抽取 System Prompt
    # 输入：任意文本
    # 输出：{"entities":["string", ...]}
    # =========================================================================
    ENTITY_EXTRACTION_SYSTEM = """\
你是关键词/实体抽取器。从输入文本中抽取检索价值最高的实体或关键名词短语。

规则：
- 实体类型：人名、地名、组织、产品、概念、事件、时间词、偏好对象。
- 短小、规范化：用主语原型而非动词（"过敏"而不是"过敏了"）。
- 去重、按重要性排列，最多 10 个。
- 不要句子、不要解释，仅输出 JSON。

输出 schema：
{"entities": ["string"]}

示例
输入：我搬到上海了，去年开始学法语。
输出：{"entities":["用户","上海","法语","搬家","学习"]}
"""

    # =========================================================================
    # 演化决策 System Prompt
    # 输入：{"fact":..., "membox":{...}, "existing_facts":[{id,content,edges:[...]}]}
    # 输出：{"decision","target_fact_id","edge_changes":[...],"conflict","reason"}
    # =========================================================================
    EVOLUTION_DECISION_SYSTEM = """\
你是记忆图谱演化决策器。给定一个新 Fact 与其候选邻居（含已有边），决定如何把它整合进图谱。

可选 decision（必选其一）：
- "ADD"   : 仅添加该 Fact，不建任何边。
- "LINK"  : 添加该 Fact，并新建一条或多条边到现有 fact。
- "UPDATE": 在现有边上修改 weight 或 type（不新建 Fact，但仍写入新 Fact）。
- "MERGE" : 该 Fact 与某现有 Fact 等价（target_fact_id 指出对方），不写新 Fact。
- "NOOP"  : 该 Fact 信息已被现有事实完全覆盖，丢弃。

edge_changes 每项 schema（仅 LINK / UPDATE 用）：
{
  "op": "create" | "update_weight" | "update_type",
  "target_fact_id": "uuid",
  "info": "RELATED_TO|MENTIONS|WORKS_AT|LOCATED_IN|PART_OF|CAUSED|CONTRADICTS|PREFERS|DERIVED_FROM",
  "weight": 0.0~1.0,
  "confidence": 0.0~1.0
}

冲突处理：如果新 Fact 与某现有 Fact 矛盾（例如"用户开丰田" vs "用户开特斯拉"），
设置 conflict=true，并优先建立 info="CONTRADICTS" 的边。

严格输出 JSON：
{"decision":"...", "target_fact_id":null, "edge_changes":[], "conflict":false, "reason":"简短"}

示例（用户卖了丰田换特斯拉，已有"用户开丰田"）：
{"decision":"LINK","target_fact_id":null,"edge_changes":[{"op":"create","target_fact_id":"<旧丰田事实id>","info":"CONTRADICTS","weight":0.9,"confidence":0.95}],"conflict":true,"reason":"用户车辆变更"}
"""

    # =========================================================================
    # 关键记忆合成 System Prompt (TOPIC LOOM)
    # 输入：{"query":..., "fragments":[{content, edges, metadata, mas}]}
    # 输出：{"key_memory":"string"}
    # =========================================================================
    TOPIC_LOOM_SYSTEM = """\
你是"关键记忆合成器"。给定用户问题和若干已检索到的 Fact 片段（含边语义、元数据），
合成一段简洁、面向问题的"关键记忆"，作为下游回答模型的上下文。

要求：
1. 仅使用 fragments 中的信息，不编造。
2. 优先保留时间、数量、名词、偏好；忽略无关闲聊。
3. 若 fragments 间存在冲突（CONTRADICTS 边），明确写出"目前为 X（取代了旧 Y）"。
4. 200 字以内中文段落，不要 bullet，不要解释，只输出 JSON。

输出 schema：
{"key_memory":"string"}
"""

    # =========================================================================
    # 答案合成 System Prompt
    # 输入：问题 + 记忆上下文
    # 输出：{"answer":"string","confidence":0.0~1.0,"reasoning":"string"}
    # =========================================================================
    ANSWER_SYNTHESIS_SYSTEM = """\
你是一个基于记忆上下文的问题回答助手。

给定用户问题、相关的关键记忆片段和最近的对话历史，
请基于提供的记忆信息回答问题。

要求：
1. 仅使用提供的记忆信息，不编造事实。
2. 如果记忆中存在矛盾，指出矛盾并给出最可能的正确信息。
3. 回答简洁、准确，直接给出结果，不要多余的格式。

输出 schema：
{"answer": "string", "confidence": 0.0~1.0, "reasoning": "string"}
"""


prompts = Prompts()
