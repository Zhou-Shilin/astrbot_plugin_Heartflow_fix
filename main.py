import json
import time
import datetime
from typing import Dict, Optional
from dataclasses import dataclass

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import Personality
from astrbot.api import logger


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0
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

    def __init__(self, context: star.Context, config: star.AstrBotConfig):
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
        
        # 系统提示词缓存：{cache_key: {"original": str, "summarized": str}}
        self.system_prompt_cache: Dict[str, Dict[str, str]] = {}

        # 判断权重配置
        self.weights = {
            "relevance": 0.25,
            "willingness": 0.2,
            "social": 0.2,
            "timing": 0.15,
            "continuity": 0.2
        }

        logger.info("心流插件已初始化")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前会话生效的人格系统提示词"""
        try:
            uid = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(uid)
            if not curr_cid:
                # 如果没有当前对话，则使用默认人格
                default_persona_name = self.context.provider_manager.selected_default_persona.get("name")
                if not default_persona_name:
                    return ""
                
                personas: list[Personality] = self.context.provider_manager.personas
                for p in personas:
                    if p.name == default_persona_name:
                        return p.prompt
                return ""

            conversation = await self.context.conversation_manager.get_conversation(uid, curr_cid)
            if not conversation:
                return ""

            persona_id = conversation.persona_id
            
            # 显式取消人格
            if persona_id == "[%None]":
                return ""
            
            # 使用默认人格
            if not persona_id:
                default_persona_name = self.context.provider_manager.selected_default_persona.get("name")
                if not default_persona_name:
                    return ""
                
                personas: list[Personality] = self.context.provider_manager.personas
                for p in personas:
                    if p.name == default_persona_name:
                        return p.prompt
                return ""

            # 使用指定人格
            personas: list[Personality] = self.context.provider_manager.personas
            for p in personas:
                if p.name == persona_id:
                    return p.prompt
            
            return ""
        except Exception as e:
            logger.error(f"获取人格系统提示词失败: {e}")
            return ""

    async def _get_or_create_summarized_system_prompt(self, event: AstrMessageEvent, original_prompt: str) -> str:
        """获取或创建精简版系统提示词"""
        if not original_prompt or len(original_prompt.strip()) < 50:
            return original_prompt
            
        try:
            # 使用人格提示词的哈希值作为缓存键，确保相同的人格只总结一次
            import hashlib
            prompt_hash = hashlib.md5(original_prompt.encode('utf-8')).hexdigest()
            cache_key = f"persona_summary_{prompt_hash}"
            
            # 检查缓存
            if cache_key in self.system_prompt_cache:
                cached = self.system_prompt_cache[cache_key]
                logger.debug(f"使用缓存的精简系统提示词: {cache_key}")
                return cached.get("summarized", original_prompt)
            
            # 如果没有缓存，进行总结
            summarized_prompt = await self._summarize_system_prompt(original_prompt)
            
            # 更新缓存
            self.system_prompt_cache[cache_key] = {
                "original": original_prompt,
                "summarized": summarized_prompt,
            }
            
            logger.info(f"创建新的精简系统提示词 | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}")
            return summarized_prompt
            
        except Exception as e:
            logger.error(f"获取或创建精简系统提示词失败: {e}")
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
                contexts=[]
            )

            content = llm_response.completion_text.strip()
            
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

        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(should_reply=False, reasoning=f"提供商不存在: {self.judge_provider_name}")
        except Exception as e:
            logger.error(f"获取提供商失败: {e}")
            return JudgeResult(should_reply=False, reasoning=f"获取提供商失败: {str(e)}")

        chat_state = self._get_chat_state(event.unified_msg_origin)
        original_persona_prompt = await self._get_persona_system_prompt(event)
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(event, original_persona_prompt)

        chat_context = await self._build_chat_context(event)
        recent_messages = await self._get_recent_messages(event)
        last_bot_reply = await self._get_last_bot_reply(event)

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
2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
4. **时机恰当性**(0-10)：回复时机是否恰当
5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)

**关联消息筛选要求**：
- 从上面的对话历史中找出与当前消息内容相关的消息。如果没有相关消息，返回空数组。

**重要！！！请严格按照以下JSON格式回复，不要添加任何其他内容：**
{{
    "relevance": <分数>,
    "willingness": <分数>,
    "social": <分数>,
    "timing": <分数>,
    "continuity": <分数>,
    "reasoning": "<详细分析原因，说明为什么应该或不应该回复>",
    "should_reply": <true/false>,
    "confidence": <0.0-1.0>,
    "related_messages": ["<从对话历史中筛选出的关联消息>"]
}}

**注意：你的回复必须是完整的JSON对象，不要包含任何解释性文字或其他内容！**
"""

        try:
            llm_response = await judge_provider.text_chat(
                prompt=judge_prompt,
                contexts=[]  # [核心修复] 判断模型不应被对话历史干扰，仅依赖prompt中的信息
            )

            content = llm_response.completion_text.strip()

            try:
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                judge_data = json.loads(content)
                
                # 确保所有评分键都存在
                for key in self.weights.keys():
                    if key not in judge_data:
                        judge_data[key] = 0

                overall_score = (
                    judge_data["relevance"] * self.weights["relevance"] +
                    judge_data["willingness"] * self.weights["willingness"] +
                    judge_data["social"] * self.weights["social"] +
                    judge_data["timing"] * self.weights["timing"] +
                    judge_data["continuity"] * self.weights["continuity"]
                ) / 10.0

                return JudgeResult(
                    relevance=judge_data["relevance"],
                    willingness=judge_data["willingness"],
                    social=judge_data["social"],
                    timing=judge_data["timing"],
                    continuity=judge_data["continuity"],
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

        if not self._should_process_message(event):
            return

        try:
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")
                event.is_at_or_wake_command = True
                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
            else:
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}", exc_info=True)

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""
        if not self.config.get("enable_heartflow", False):
            return False

        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        if self.whitelist_enabled:
            if not self.chat_whitelist:
                return False
            if event.unified_msg_origin not in self.chat_whitelist:
                return False

        if event.get_sender_id() == event.get_self_id():
            return False

        if not event.message_str or not event.message_str.strip():
            return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取或创建群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            state.energy = min(1.0, state.energy + 0.2)
            logger.info(f"每日重置群聊状态: {chat_id}")

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)
        if chat_state.last_reply_time == 0:
            return 999
        return int((time.time() - chat_state.last_reply_time) / 60)

    async def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """获取最近的对话上下文（用于传递给大参数模型）"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return []
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return []
            return json.loads(conversation.history)
        except Exception as e:
            logger.debug(f"获取对话上下文失败: {e}")
            return []

    async def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文"""
        chat_state = self._get_chat_state(event.unified_msg_origin)
        return f"""最近活跃度: {'高' if chat_state.total_messages > 100 else '中' if chat_state.total_messages > 20 else '低'}
历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%
当前时间: {datetime.datetime.now().strftime('%H:%M')}"""

    async def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """获取最近的消息历史（用于小参数模型判断）"""
        try:
            context = await self._get_recent_contexts(event)
            recent_context = context[-self.context_messages_count:] if len(context) > self.context_messages_count else context

            messages_text = []
            for msg in recent_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user" and content:
                    messages_text.append(f"用户: {content}")
                elif role == "assistant" and content:
                    messages_text.append(f"机器人: {content}")
            
            return "\n".join(messages_text) if messages_text else "暂无对话历史"
        except Exception as e:
            logger.debug(f"获取消息历史失败: {e}")
            return "暂无对话历史"

    async def _get_last_bot_reply(self, event: AstrMessageEvent) -> Optional[str]:
        """获取上次机器人的回复消息"""
        try:
            context = await self._get_recent_contexts(event)
            for msg in reversed(context):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and isinstance(content, str) and content.strip():
                    return content
            return None
        except Exception as e:
            logger.debug(f"获取上次bot回复失败: {e}")
            return None

    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)
        chat_state.last_reply_time = time.time()
        chat_state.total_replies += 1
        chat_state.total_messages += 1
        # [BUG修复] self.energy 应该是 self.energy_decay_rate
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

    def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """[新增] 更新被动状态（不回复）"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)
        chat_state.total_messages += 1
        # 精力恢复
        chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

    @filter.command("heartflow_reset")
    async def reset_chat_state(self, event: AstrMessageEvent):
        """重置当前群聊的心流状态"""
        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            self.chat_states[chat_id] = ChatState()
            logger.info(f"已重置群聊 {chat_id} 的心流状态。")
            yield event.plain_result("当前群聊的心流状态已重置。")
        else:
            yield event.plain_result("当前群聊没有心流状态记录。")
