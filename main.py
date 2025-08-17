import json
import time
import datetime
from typing import Dict
from dataclasses import dataclass

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    confidence: float = 0.0
    overall_score: float = 0.0
    related_messages: list = None

    def __post_init__(self):
        if self.related_messages is None:
            self.related_messages = []


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0



class HeartflowPlugin(star.Star):

    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config

        # 判断模型配置
        self.judge_provider_name = self.config.get("judge_provider_name", "")

        # 心流参数配置
        self.reply_threshold = self.config.get("reply_threshold", 0.6)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.1)
        self.energy_recovery_rate = self.config.get("energy_recovery_rate", 0.02)
        self.context_messages_count = self.config.get("context_messages_count", 5)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}
        
        # 系统提示词缓存：{conversation_id: {"original": str, "summarized": str, "persona_id": str}}
        self.system_prompt_cache: Dict[str, Dict[str, str]] = {}

        # 判断权重配置
        self.weights = {
            "relevance": 0.25,
            "willingness": 0.2,
            "social": 0.2,
            "timing": 0.15,
            "continuity": 0.2  # 新增：与上次回复的连贯性
        }

        logger.info("心流插件已初始化")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前会话的人格系统提示词"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return self.context.provider_manager.selected_default_persona.get("prompt", "")

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if conversation and conversation.persona_id:
                if conversation.persona_id == "[%None]":
                    return "" # 用户显式取消了人格
                
                # 在所有可用的人格中查找
                for persona in self.context.provider_manager.personas:
                    if persona.id == conversation.persona_id:
                        return persona.prompt
            
            # 如果没有找到特定的人格，返回默认人格
            return self.context.provider_manager.selected_default_persona.get("prompt", "")
        except Exception as e:
            logger.error(f"获取人格系统提示词失败: {e}")
            return ""

    async def _get_or_create_summarized_system_prompt(self, event: AstrMessageEvent, original_prompt: str) -> str:
        """获取或创建精简版系统提示词"""
        try:
            # 获取当前会话ID
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return original_prompt
            
            # 获取当前人格ID作为缓存键的一部分
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            persona_id = conversation.persona_id if conversation else "default"
            
            # 构建缓存键
            cache_key = f"{curr_cid}_{persona_id}"
            
            # 检查缓存
            if cache_key in self.system_prompt_cache:
                cached = self.system_prompt_cache[cache_key]
                # 如果原始提示词没有变化，返回缓存的总结
                if cached.get("original") == original_prompt:
                    logger.debug(f"使用缓存的精简系统提示词: {cache_key}")
                    return cached.get("summarized", original_prompt)
            
            # 如果没有缓存或原始提示词发生变化，进行总结
            if not original_prompt or len(original_prompt.strip()) < 50:
                # 如果原始提示词太短，直接返回
                return original_prompt
            
            summarized_prompt = await self._summarize_system_prompt(original_prompt)
            
            # 更新缓存
            self.system_prompt_cache[cache_key] = {
                "original": original_prompt,
                "summarized": summarized_prompt,
                "persona_id": persona_id
            }
            
            logger.info(f"创建新的精简系统提示词: {cache_key} | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}")
            return summarized_prompt
            
        except Exception as e:
            logger.error(f"获取精简系统提示词失败: {e}")
            return original_prompt
    
    async def _summarize_system_prompt(self, original_prompt: str) -> str:
        """使用小模型对系统提示词进行总结"""
        try:
            if not self.judge_provider_name:
                return original_prompt
            
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                return original_prompt
            
            summarize_prompt = f"""请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。
总结后的内容应该在100-200字以内，突出最重要的角色特点。

原始角色设定：
{original_prompt}

请以JSON格式回复：
{{
    "summarized_persona": "精简后的角色设定，保留核心特征和行为方式"
}}

**重要：你的回复必须是完整的JSON对象，不要包含任何其他内容！**"""

            llm_response = await judge_provider.text_chat(
                prompt=summarize_prompt,
                contexts=[]  # 不需要上下文
            )

            content = llm_response.completion_text.strip()
            
            # 尝试提取JSON
            try:
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                result_data = json.loads(content)
                summarized = result_data.get("summarized_persona", "")
                
                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return original_prompt
                    
            except json.JSONDecodeError:
                logger.error(f"小模型总结系统提示词返回非有效JSON: {content}")
                return original_prompt
                
        except Exception as e:
            logger.error(f"总结系统提示词异常: {e}")
            return original_prompt

    async def judge_with_tiny_model(self, event: AstrMessageEvent) -> JudgeResult:
        """使用小模型进行智能判断"""

        if not self.judge_provider_name:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取指定的 provider
        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(should_reply=False, reasoning=f"提供商不存在: {self.judge_provider_name}")
        except Exception as e:
            logger.error(f"获取提供商失败: {e}")
            return JudgeResult(should_reply=False, reasoning=f"获取提供商失败: {str(e)}")

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        original_persona_prompt = await self._get_persona_system_prompt(event)
        logger.debug(f"小参数模型获取原始人格提示词: {'有' if original_persona_prompt else '无'} | 长度: {len(original_persona_prompt) if original_persona_prompt else 0}")
        
        # 获取或创建精简版系统提示词
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(event, original_persona_prompt)
        logger.debug(f"小参数模型使用精简人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        # 构建判断上下文
        chat_context = await self._build_chat_context(event)
        recent_messages = await self._get_recent_messages(event)
        last_bot_reply = await self._get_last_bot_reply(event)  # 新增：获取上次bot回复

        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

## 当前群聊情况
- 群聊ID: {event.unified_msg_origin}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前

## 群聊基本信息
{chat_context}

## 最近{self.context_messages_count}条对话历史
{recent_messages}

## 上次机器人回复
{last_bot_reply if last_bot_reply else "暂无上次回复记录"}

## 待判断消息
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime('%H:%M:%S')}

## 评估要求
请从以下5个维度评估（0-10分），**重要提醒：基于上述机器人角色设定来判断是否适合回复**：

1. **内容相关度**(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - **结合机器人角色特点，判断是否符合角色定位**

2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - **基于机器人角色设定，判断是否应该主动参与此话题**

3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - **考虑机器人角色在群中的定位和表现方式**

4. **时机恰当性**(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)

**关联消息筛选要求**：
- 从上面的对话历史中找出与当前消息内容相关的消息
- 直接复制相关消息的完整内容，保持原有格式
- 如果没有相关消息，返回空数组

**重要！！！请严格按照以下JSON格式回复，不要添加任何其他内容：**

请以JSON格式回复：
{{
    "relevance": 分数,
    "willingness": 分数,
    "social": 分数,
    "timing": 分数,
    "continuity": 分数,
    "reasoning": "详细分析原因，说明为什么应该或不应该回复，需要结合机器人角色特点进行分析，特别说明与上次回复的关联性",
    "should_reply": true/false,
    "confidence": 0.0-1.0,
    "related_messages": ["从上面对话历史中筛选出与当前消息可能有关联的消息，直接复制完整内容保持原格式，如果没有关联消息则为空数组"]
}}

**注意：你的回复必须是完整的JSON对象，不要包含任何解释性文字或其他内容！**
"""

        try:
            # 构建完整的判断提示词，将系统提示直接整合到prompt中
            complete_judge_prompt = "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
            if persona_system_prompt:
                complete_judge_prompt += f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
            complete_judge_prompt += "\n\n**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**\n\n"
            complete_judge_prompt += judge_prompt

            llm_response = await judge_provider.text_chat(
                prompt=complete_judge_prompt,
                contexts=[]  # [FIX 1] 关键修复：判断模型不应继承主对话上下文，防止被上下文污染而回复非JSON内容
            )

            content = llm_response.completion_text.strip()

            # 尝试提取JSON
            try:
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                judge_data = json.loads(content)

                # 计算综合评分
                overall_score = (
                    judge_data.get("relevance", 0) * self.weights["relevance"] +
                    judge_data.get("willingness", 0) * self.weights["willingness"] +
                    judge_data.get("social", 0) * self.weights["social"] +
                    judge_data.get("timing", 0) * self.weights["timing"] +
                    judge_data.get("continuity", 0) * self.weights["continuity"]
                ) / 10.0

                return JudgeResult(
                    relevance=judge_data.get("relevance", 0),
                    willingness=judge_data.get("willingness", 0),
                    social=judge_data.get("social", 0),
                    timing=judge_data.get("timing", 0),
                    continuity=judge_data.get("continuity", 0),
                    reasoning=judge_data.get("reasoning", ""),
                    should_reply=judge_data.get("should_reply", False) and overall_score >= self.reply_threshold,
                    confidence=judge_data.get("confidence", 0.0),
                    overall_score=overall_score,
                    related_messages=judge_data.get("related_messages", [])
                )
            except json.JSONDecodeError:
                logger.error(f"小参数模型返回非有效JSON: {content}")
                return JudgeResult(should_reply=False, reasoning=f"JSON解析失败")

        except Exception as e:
            logger.error(f"小参数模型判断异常: {e}")
            return JudgeResult(should_reply=False, reasoning=f"异常: {str(e)}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口"""

        # 检查基本条件
        if not self._should_process_message(event):
            return

        try:
            # 小参数模型判断是否需要回复
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

                # 设置唤醒标志为真，调用LLM
                event.is_at_or_wake_command = True
                
                # 更新主动回复状态
                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                
                # 不需要yield任何内容，让核心系统处理
                return
            else:
                # 记录被动状态
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新被动（不回复）状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        chat_state.total_messages += 1

        # 精力恢复
        chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""

        # 检查插件是否启用
        if not self.config.get("enable_heartflow", False):
            return False

        # 跳过已经被其他插件或系统标记为唤醒的消息
        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        # 检查白名单
        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        # 检查日期重置
        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            # 每日重置时恢复一些精力
            state.energy = min(1.0, state.energy + 0.2)

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    async def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """获取最近的对话上下文（用于传递给小参数模型）
        
        注意：此方法会过滤掉函数调用相关内容，只保留纯文本消息，
        以避免小参数模型因不支持函数调用而报错。
        """
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return []

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return []

            context = json.loads(conversation.history)

            # 获取最近的 context_messages_count 条消息
            recent_context = context[-self.context_messages_count:] if len(context) > self.context_messages_count else context

            # 过滤掉函数调用相关内容，避免小参数模型报错
            filtered_context = []
            for msg in recent_context:
                # 只保留纯文本的用户和助手消息
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role in ["user", "assistant"] and content and isinstance(content, str):
                    # 创建一个干净的消息副本，只包含文本内容
                    clean_msg = {
                        "role": role,
                        "content": content
                    }
                    filtered_context.append(clean_msg)

            return filtered_context

        except Exception as e:
            logger.debug(f"获取对话上下文失败: {e}")
            return []

    async def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文"""
        chat_state = self._get_chat_state(event.unified_msg_origin)
        total_messages = chat_state.total_messages
        # 避免除零错误
        if total_messages == 0:
            reply_rate = 0.0
        else:
            reply_rate = (chat_state.total_replies / total_messages * 100)

        context_info = f"""最近活跃度: {'高' if total_messages > 100 else '中' if total_messages > 20 else '低'}
历史回复率: {reply_rate:.1f}%
当前时间: {datetime.datetime.now().strftime('%H:%M')}"""
        return context_info

    async def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """获取最近的消息历史（用于小参数模型判断）"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return "暂无对话历史"

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return "暂无对话历史"

            context = json.loads(conversation.history)

            # 获取最近的 context_messages_count 条消息
            recent_context = context[-self.context_messages_count:] if len(context) > self.context_messages_count else context
            
            # 将上下文格式化为字符串
            messages_text = []
            for msg in recent_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user" and content and isinstance(content, str):
                    sender_name = msg.get("metadata", {}).get("sender_name", "用户")
                    messages_text.append(f"{sender_name}: {content}")
                elif role == "assistant" and content and isinstance(content, str):
                    messages_text.append(f"机器人: {content}")
            
            return "\n".join(messages_text) if messages_text else "暂无对话历史"

        except Exception as e:
            logger.debug(f"获取消息历史失败: {e}")
            return "暂无对话历史"

    async def _get_last_bot_reply(self, event: AstrMessageEvent) -> str:
        """获取上次机器人的回复消息"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return None

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return None

            context = json.loads(conversation.history)

            # 从后往前查找最后一条assistant消息
            for msg in reversed(context):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant" and content and isinstance(content, str):
                    return content

            return None

        except Exception as e:
            logger.debug(f"获取上次bot回复失败: {e}")
            return None

    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新回复相关状态
        chat_state.last_reply_time = time.time()
        chat_state.total_replies += 1
        chat_state.total_messages += 1

        # 精力消耗（回复后精力下降）
        # [FIX 2] 修复了变量名错误，之前是 self.energy，应该是 self.energy_decay_rate
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)
