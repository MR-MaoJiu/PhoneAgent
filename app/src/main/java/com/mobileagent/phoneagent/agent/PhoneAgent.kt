/**
 * Phone Agent - 核心智能体类
 * 
 * 项目地址: https://github.com/MR-MaoJiu/PhoneAgent
 * 
 * 负责协调 AI 模型、截图、操作执行等
 */
package com.mobileagent.phoneagent.agent

import android.content.Context
import android.graphics.Bitmap
import android.media.projection.MediaProjection
import android.util.Log
import com.mobileagent.phoneagent.action.ActionHandler
import com.mobileagent.phoneagent.action.ActionResult
import com.mobileagent.phoneagent.model.ContentItem
import com.mobileagent.phoneagent.model.ImageUrl
import com.mobileagent.phoneagent.model.Message
import com.mobileagent.phoneagent.model.ModelClient
import com.mobileagent.phoneagent.service.PhoneAgentAccessibilityService
import com.mobileagent.phoneagent.utils.ScreenshotManager
import com.mobileagent.phoneagent.utils.ScreenshotUtils
import kotlinx.coroutines.*
import org.json.JSONObject

/**
 * 运行模式枚举
 */
enum class Mode {
    VISION,          // 视觉模式：通过截图上传图片
    ACCESSIBILITY,   // 无障碍模式：通过无障碍服务获取屏幕内容
    HYBRID          // 混合模式：结合视觉模式和无障碍模式
}

/**
 * Phone Agent - 核心智能体类
 * 负责协调 AI 模型、截图、操作执行等
 */
class PhoneAgent(
    private val context: Context,
    private val modelClient: ModelClient,
    private val accessibilityService: PhoneAgentAccessibilityService,
    private val mediaProjection: MediaProjection?,
    private val screenWidth: Int,
    private val screenHeight: Int,
    private val maxSteps: Int = Int.MAX_VALUE, // 移除最大步数限制，只有任务完成才停止
    private val systemPrompt: String,
    private val mode: Mode = Mode.VISION, // 运行模式，默认视觉模式
    private val onStepCallback: ((StepResult) -> Unit)? = null,
    private val onUserInterventionCallback: ((String) -> Unit)? = null
) {
    private val TAG = "PhoneAgent"
    private val actionHandler = ActionHandler(accessibilityService)
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    private var contextMessages = mutableListOf<Message>()
    private var stepCount = 0
    private var isRunning = false
    private var screenshotManager: ScreenshotManager? = null
    private var currentTask: String? = null // 保存当前任务描述
    
    // 上下文管理：智能压缩
    private val contextSizeThreshold = 4_000_000 // 上下文大小阈值（字符数，约4MB）
    private val minMessagesToKeep = 4 // 压缩时至少保留的消息数（系统消息 + 最近1-2对对话）
    private val maxRetries = 3 // 最大重试次数
    private var consecutiveFailures = 0 // 连续失败次数
    private var lastFailedAction: String? = null // 上次失败的操作
    private var compressedHistory: String? = null // 压缩后的历史摘要
    private var waitingForUserIntervention = false // 是否正在等待用户介入

    /**
     * 运行任务
     */
    fun run(task: String, onComplete: (String) -> Unit) {
        if (isRunning) {
            Log.w(TAG, "⚠️ Agent 已在运行中，忽略重复请求")
            return
        }

        Log.d(TAG, "========================================")
        Log.d(TAG, "🚀 开始执行任务")
        Log.d(TAG, "任务: $task")
        Log.d(TAG, "屏幕尺寸: ${screenWidth}x${screenHeight}")
        Log.d(TAG, "========================================")

        isRunning = true
        contextMessages.clear()
        stepCount = 0
        consecutiveFailures = 0
        lastFailedAction = null
        compressedHistory = null
        currentTask = task
        waitingForUserIntervention = false

        scope.launch {
            try {
                // 根据模式初始化 ScreenshotManager（视觉模式和混合模式需要）
                if (mode == Mode.VISION || mode == Mode.HYBRID) {
                    if (mediaProjection != null) {
                        try {
                            withContext(Dispatchers.Main) {
                                val density = context.resources.displayMetrics.densityDpi
                                screenshotManager = ScreenshotManager(mediaProjection!!, screenWidth, screenHeight, density)
                                screenshotManager?.initialize()
                                Log.d(TAG, "✅ ScreenshotManager 已初始化（模式: $mode）")
                            }
                        } catch (e: SecurityException) {
                            Log.e(TAG, "❌ MediaProjection 已过期或无效", e)
                            isRunning = false
                            onComplete("MediaProjection 已过期，请重新授权屏幕录制权限后重试")
                            return@launch
                        } catch (e: Exception) {
                            Log.e(TAG, "❌ 初始化 ScreenshotManager 失败", e)
                            isRunning = false
                            onComplete("初始化截图管理器失败: ${e.message}")
                            return@launch
                        }
                    } else {
                        Log.e(TAG, "❌ MediaProjection 为 null，无法初始化 ScreenshotManager（模式: $mode）")
                        isRunning = false
                        onComplete("MediaProjection 未初始化，请先授权屏幕录制权限")
                        return@launch
                    }
                } else {
                    Log.d(TAG, "✅ 无障碍模式，无需初始化 ScreenshotManager")
                }
                
                // 初始化系统提示词
                contextMessages.add(Message("system", systemPrompt))
                Log.d(TAG, "系统提示词已添加")

                // 执行第一步
                Log.d(TAG, "执行第一步...")
                val firstResult = executeStep(task, isFirst = true)
                if (firstResult.finished) {
                    Log.d(TAG, "第一步即完成任务")
                    isRunning = false
                    onComplete(firstResult.message ?: "任务完成")
                    return@launch
                }
                while (isRunning) {
                    Log.d(TAG, "========================================")
                    Log.d(TAG, "🔄 循环执行 - 当前步数: $stepCount")
                    Log.d(TAG, "任务目标: $currentTask")
                    Log.d(TAG, "连续失败次数: $consecutiveFailures")
                    Log.d(TAG, "等待用户介入: $waitingForUserIntervention")
                    Log.d(TAG, "========================================")
                    
                    // 如果连续失败次数过多，添加提示让AI重新规划
                    if (consecutiveFailures >= 2) {
                        Log.w(TAG, "⚠️ 连续失败 $consecutiveFailures 次，添加重新规划提示")
                        val planningHint = Message("user", 
                            "** ⚠️ 重要提示：需要重新规划策略 **\n\n" +
                            "你已经连续失败了 $consecutiveFailures 次。当前方法不可行，请立即尝试完全不同的方法：\n\n" +
                            "1. **重新分析任务**：任务目标是 '$currentTask'，请确认是否理解正确\n" +
                            "2. **尝试不同操作**：\n" +
                            "   - 如果点击失败，尝试调整坐标位置、尝试滑动、或使用 Launch 启动应用\n" +
                            "   - 如果找不到元素，尝试滑动屏幕查看更多内容、返回上一页、或重新搜索\n" +
                            "   - 如果操作无效，尝试返回后重新进入、等待页面加载、或使用不同的操作方式\n" +
                            "3. **检查当前状态**：仔细分析屏幕截图，确认当前处于什么页面、什么状态\n" +
                            "4. **不要放弃**：继续尝试不同的方法，直到任务完成\n" +
                            "5. **如果确实无法完成**：使用 finish(message=\"无法完成的原因\") 说明情况\n\n" +
                            "请立即重新分析屏幕，制定新的操作计划，并继续执行。"
                        )
                        contextMessages.add(planningHint)
                        consecutiveFailures = 0 // 重置计数，给AI一次机会
                    }
                    
                    // 如果连续失败次数过多（超过5次），考虑是否需要用户介入
                    if (consecutiveFailures >= 5) {
                        Log.w(TAG, "⚠️ 连续失败 $consecutiveFailures 次，可能需要用户介入")
                        val userInterventionHint = Message("user",
                            "** ⚠️ 需要用户介入 **\n\n" +
                            "已经连续失败 $consecutiveFailures 次，当前方法似乎无法完成任务。\n" +
                            "如果遇到以下情况，请使用 Take_over 请求用户介入：\n" +
                            "1. 需要输入验证码、密码等安全信息\n" +
                            "2. 需要用户手动选择或确认\n" +
                            "3. 遇到无法自动处理的情况\n\n" +
                            "否则，请继续尝试不同的方法完成任务。"
                        )
                        contextMessages.add(userInterventionHint)
                    }
                    
                    val result = executeStep(isFirst = false)
                    
                    Log.d(TAG, "========================================")
                    Log.d(TAG, "📊 步骤执行结果")
                    Log.d(TAG, "成功: ${result.success}")
                    Log.d(TAG, "完成: ${result.finished}")
                    Log.d(TAG, "思考: ${result.thinking.take(100)}...")
                    Log.d(TAG, "操作: ${result.action.take(100)}...")
                    if (result.message != null) {
                        Log.d(TAG, "消息: ${result.message}")
                    }
                    Log.d(TAG, "连续失败次数: $consecutiveFailures")
                    Log.d(TAG, "========================================")
                    
                    // 只有明确使用 finish() 时才停止，否则继续执行
                    if (result.finished) {
                        // 检查是否是真正的完成（finish action）还是因为错误而停止
                        val isRealFinish = result.action.contains("\"_metadata\":\"finish\"") || 
                                         result.action.contains("finish(") ||
                                         (result.message != null && result.message.contains("任务完成"))
                        
                        if (isRealFinish) {
                            Log.d(TAG, "✅ 任务在步骤 $stepCount 完成（AI明确使用finish）")
                        isRunning = false
                        onComplete(result.message ?: "任务完成")
                        return@launch
                        } else {
                            Log.w(TAG, "⚠️ 步骤标记为完成，但可能是错误导致的，继续执行")
                            // 继续执行，不停止
                        }
                    }
                    
                    // 即使操作失败，也继续执行（让AI尝试不同方法）
                    // 只有在达到最大步数或明确完成时才停止
                    
                    // 短暂延迟，避免过快执行，给UI和系统一些时间
                    kotlinx.coroutines.delay(800)
                }

                // 移除最大步数限制检查，只有任务完成才停止
            } catch (e: Exception) {
                Log.e(TAG, "❌ 执行任务失败", e)
                e.printStackTrace()
                isRunning = false
                onComplete("任务执行失败: ${e.message}")
            } finally {
                // 清理资源
                screenshotManager?.cleanup()
                screenshotManager = null
                Log.d(TAG, "✅ 资源已清理")
            }
        }
    }

    /**
     * 停止任务
     */
    fun stop() {
        isRunning = false
        screenshotManager?.cleanup()
        screenshotManager = null
        scope.cancel()
        Log.d(TAG, "任务已停止，资源已清理")
    }

    /**
     * 更新任务（允许在任务执行中更新）
     */
    fun updateTask(newTask: String) {
        if (isRunning) {
            Log.d(TAG, "更新任务: $newTask")
            currentTask = newTask
            // 添加任务更新消息到上下文
            val updateMessage = Message("user",
                "** 📝 任务已更新 **\n\n" +
                "原任务目标: ${currentTask}\n" +
                "新任务目标: $newTask\n\n" +
                "请根据新的任务目标继续执行。如果新任务与当前状态不符，请先返回或重新开始。"
            )
            contextMessages.add(updateMessage)
        }
    }

    /**
     * 检查任务是否正在运行（公开属性）
     */
    fun isTaskRunning(): Boolean = isRunning

    /**
     * 执行单步
     */
    private suspend fun executeStep(
        userPrompt: String? = null,
        isFirst: Boolean = false
    ): StepResult = withContext(Dispatchers.IO) {
        stepCount++
        Log.d(TAG, "========================================")
        Log.d(TAG, "📸 步骤 $stepCount: 开始执行")
        if (isFirst) {
            Log.d(TAG, "任务: $userPrompt")
        }

        // 根据模式获取屏幕数据
        Log.d(TAG, "当前模式: $mode")
        val contentItems = mutableListOf<ContentItem>()
        
        // 视觉模式或混合模式：获取截图
        if (mode == Mode.VISION || mode == Mode.HYBRID) {
            Log.d(TAG, "正在截图...")
            val screenshot = captureScreenshot()
            if (screenshot == null) {
                Log.e(TAG, "❌ 截图失败")
                if (mode == Mode.VISION) {
                    // 视觉模式必须要有截图
                    return@withContext StepResult(
                        success = false,
                        finished = true,
                        thinking = "截图失败",
                        action = "",
                        message = "截图失败"
                    )
                } else {
                    // 混合模式截图失败可以继续使用无障碍内容
                    Log.w(TAG, "⚠️ 混合模式截图失败，继续使用无障碍内容")
                }
            } else {
                Log.d(TAG, "✅ 截图成功: ${screenshot.width}x${screenshot.height}")
                val imageBase64 = com.mobileagent.phoneagent.utils.ScreenshotUtils.bitmapToBase64(screenshot)
                val imageUrl = "data:image/png;base64,$imageBase64"
                Log.d(TAG, "图片 Base64 长度: ${imageBase64.length}")
                contentItems.add(ContentItem(
                    type = "image_url",
                    imageUrl = ImageUrl(url = imageUrl)
                ))
            }
        }
        
        // 无障碍模式或混合模式：获取屏幕内容
        if (mode == Mode.ACCESSIBILITY || mode == Mode.HYBRID) {
            Log.d(TAG, "正在获取无障碍屏幕内容...")
            val screenContent = accessibilityService.getScreenContent()
            Log.d(TAG, "✅ 获取屏幕内容成功，长度: ${screenContent.length}")
            contentItems.add(ContentItem(
                type = "text",
                text = screenContent
            ))
        }
        
        // 获取当前应用名称（而不是包名）
        val currentApp = accessibilityService.getCurrentAppName()
        Log.d(TAG, "当前应用: $currentApp")
        Log.d(TAG, "任务描述: $userPrompt")
        Log.d(TAG, "已执行步骤: $stepCount")
        
        if (contentItems.isEmpty()) {
            Log.e(TAG, "❌ 未获取到任何屏幕数据")
            return@withContext StepResult(
                success = false,
                finished = true,
                thinking = "未获取到屏幕数据",
                action = "",
                message = "未获取到屏幕数据"
            )
        }

        contextMessages.add(Message("user", contentItems))
        
        // 智能压缩上下文（如果超过阈值）
        val messagesToSend = compressContextIfNeeded(contextMessages)
        val contextSize = calculateContextSize(messagesToSend)
        Log.d(TAG, "消息已添加到上下文，总消息数: ${contextMessages.size}，发送消息数: ${messagesToSend.size}")
        Log.d(TAG, "上下文大小: ${contextSize / 1024}KB (阈值: ${contextSizeThreshold / 1024}KB)")

        // 调用模型
        Log.d(TAG, "🤖 调用 AI 模型...")
        val modelResponse = try {
            modelClient.request(messagesToSend)
        } catch (e: Exception) {
            Log.e(TAG, "❌ 模型请求失败", e)
            Log.e(TAG, "错误详情: ${e.message}")
            e.printStackTrace()
            return@withContext StepResult(
                success = false,
                finished = true,
                thinking = "模型请求失败: ${e.message}",
                action = "",
                message = "模型请求失败: ${e.message}"
            )
        }
        Log.d(TAG, "✅ 模型响应接收成功")
        Log.d(TAG, "💭 AI思考过程（完整）:")
        Log.d(TAG, modelResponse.thinking)
        Log.d(TAG, "🎯 AI操作指令（完整）:")
        Log.d(TAG, modelResponse.action)

        // 通过回调输出完整的思考过程到UI
        onStepCallback?.invoke(StepResult(
            success = false,
            finished = false,
            thinking = modelResponse.thinking,
            action = "分析中...",
            message = "正在分析屏幕，制定操作计划..."
        ))

        // 解析操作
        Log.d(TAG, "解析操作指令...")
        Log.d(TAG, "原始响应（完整）: ${modelResponse.action}")
        Log.d(TAG, "原始响应长度: ${modelResponse.action.length}")
        val actionJson = try {
            val parsed = parseActionFromResponse(modelResponse.action)
            Log.d(TAG, "✅ 操作解析成功: $parsed")
            // 验证解析结果是否包含必要字段
            val parsedJson = org.json.JSONObject(parsed)
            if (parsedJson.optString("action") == "Tap" && !parsedJson.has("element")) {
                Log.e(TAG, "❌ 警告：Tap 操作缺少 element 坐标！原始响应: ${modelResponse.action.take(500)}")
            }
            parsed
        } catch (e: Exception) {
            Log.e(TAG, "❌ 解析操作失败", e)
            Log.e(TAG, "原始操作文本: ${modelResponse.action}")
            e.printStackTrace()
            return@withContext StepResult(
                success = false,
                finished = false,
                thinking = modelResponse.thinking,
                action = modelResponse.action,
                message = "解析操作失败"
            )
        }

        // 执行操作（带重试机制）
        Log.d(TAG, "========================================")
        Log.d(TAG, "🎯 开始执行操作")
        Log.d(TAG, "操作指令: $actionJson")
        Log.d(TAG, "========================================")
        
        // 通过回调输出操作指令到UI
        onStepCallback?.invoke(StepResult(
            success = false,
            finished = false,
            thinking = modelResponse.thinking,
            action = actionJson,
            message = "正在执行操作..."
        ))
        
        val actionResult = executeActionWithRetry(actionJson, screenWidth, screenHeight)
        Log.d(TAG, "========================================")
        Log.d(TAG, "📊 操作执行结果")
        Log.d(TAG, "成功: ${actionResult.success}")
        Log.d(TAG, "完成: ${actionResult.shouldFinish}")
        if (actionResult.message != null) {
            Log.d(TAG, "消息: ${actionResult.message}")
        }
        Log.d(TAG, "========================================")
        
        // 更新失败计数
        if (!actionResult.success) {
            consecutiveFailures++
            lastFailedAction = actionJson
            Log.w(TAG, "⚠️ 操作失败，连续失败次数: $consecutiveFailures")
        } else {
            consecutiveFailures = 0
            lastFailedAction = null
        }
        if (actionResult.message != null) {
            Log.d(TAG, "操作消息: ${actionResult.message}")
        }

        // 检查是否需要用户介入
        if (actionResult.requiresTakeover && actionResult.message != null) {
            Log.w(TAG, "========================================")
            Log.w(TAG, "⚠️ 需要用户介入")
            Log.w(TAG, "原因: ${actionResult.message}")
            Log.w(TAG, "========================================")
            waitingForUserIntervention = true
            onUserInterventionCallback?.invoke(actionResult.message!!)
            
            // 添加用户介入信息到上下文，等待用户完成后继续
            val interventionMessage = Message("user", 
                "** ⚠️ 用户介入提示 **\n" +
                "${actionResult.message}\n\n" +
                "用户已完成介入操作，请继续执行任务。分析当前屏幕状态，继续下一步操作。"
            )
            contextMessages.add(interventionMessage)
            
            // 等待用户介入完成（给用户更多时间完成操作）
            Log.d(TAG, "⏳ 等待用户介入完成（5秒）...")
            onStepCallback?.invoke(StepResult(
                success = true,
                finished = false,
                thinking = "等待用户完成操作: ${actionResult.message}",
                action = "Take_over",
                message = "等待用户操作完成，将在5秒后继续..."
            ))
            kotlinx.coroutines.delay(5000) // 增加等待时间到5秒
            waitingForUserIntervention = false
            Log.d(TAG, "✅ 用户介入完成，继续执行任务")
            
            // 用户介入后，重新截图并继续执行
            Log.d(TAG, "用户介入后，重新分析屏幕状态...")
            // 不返回，继续执行下一步
        }

        // 在操作执行后，移除最后一条用户消息（包含图片的那条）的图片
        removeImageFromLastUserMessage()

        // 添加助手回复到上下文
        contextMessages.add(Message("assistant", modelResponse.rawContent))
        
        // 如果操作失败，添加失败信息到上下文，帮助AI下次尝试不同方法
        if (!actionResult.success && actionResult.message != null) {
            val failureMessage = Message("user", 
                "⚠️ 上次操作失败: ${actionResult.message}\n" +
                "请分析失败原因，并尝试完全不同的方法。不要重复相同的操作。"
            )
            contextMessages.add(failureMessage)
            Log.d(TAG, "已添加失败信息到上下文，帮助AI重新规划")
        }

        val stepResult = StepResult(
            success = actionResult.success,
            finished = actionResult.shouldFinish,
            thinking = modelResponse.thinking,
            action = actionJson,
            message = actionResult.message
        )

        Log.d(TAG, "步骤 $stepCount 完成: success=${stepResult.success}, finished=${stepResult.finished}")
        if (stepResult.finished) {
            Log.d(TAG, "✅ 任务完成: ${stepResult.message}")
        }
        Log.d(TAG, "========================================")

        onStepCallback?.invoke(stepResult)
        stepResult
    }

    /**
     * 计算上下文大小（字符数，包括图片base64）
     */
    private fun calculateContextSize(messages: List<Message>): Int {
        return messages.sumOf { message ->
            when (val content = message.content) {
                is String -> content.length
                is List<*> -> {
                    content.filterIsInstance<ContentItem>().sumOf { item ->
                        when (item.type) {
                            "text" -> item.text?.length ?: 0
                            "image_url" -> {
                                // 图片URL包含base64数据，需要计算大小
                                item.imageUrl?.url?.length ?: 0
                            }
                            else -> 0
                        }
                    }
                }
                else -> 0
            }
        }
    }
    
    /**
     * 智能压缩上下文（如果超过阈值）
     * 策略：当上下文超过阈值时，让AI压缩历史消息，保留关键信息
     */
    private suspend fun compressContextIfNeeded(messages: List<Message>): List<Message> {
        val currentSize = calculateContextSize(messages)
        
        // 如果未超过阈值，直接返回
        if (currentSize <= contextSizeThreshold) {
            return messages
        }
        
        Log.w(TAG, "⚠️ 上下文超过阈值 (${currentSize / 1024}KB > ${contextSizeThreshold / 1024}KB)，开始智能压缩...")
        
        // 分离系统消息、历史消息和最新消息
        val systemMessage = messages.firstOrNull { it.role == "system" }
        val otherMessages = messages.filter { it.role != "system" }
        
        // 保留最新的消息（至少保留最近2对对话，确保有足够的上下文）
        val keepCount = (minMessagesToKeep * 2).coerceAtMost(otherMessages.size) // 保留2对对话 = 4条消息
        val recentMessages = otherMessages.takeLast(keepCount)
        val oldMessages = otherMessages.dropLast(keepCount)
        
        if (oldMessages.isEmpty()) {
            Log.d(TAG, "没有需要压缩的历史消息，但上下文仍然很大，可能是单条消息太大")
            // 如果单条消息就很大（比如图片），只能返回原消息
            return messages
        }
        
        Log.d(TAG, "准备压缩 ${oldMessages.size} 条历史消息，保留 ${recentMessages.size} 条最新消息")
        
        // 压缩历史消息
        val compressedSummary = compressHistoryMessages(oldMessages)
        
        // 构建压缩后的上下文
        val compressedMessages = mutableListOf<Message>()
        systemMessage?.let { compressedMessages.add(it) }
        
        // 如果有压缩摘要，添加摘要消息
        if (compressedSummary.isNotEmpty()) {
            compressedMessages.add(Message("user", 
                "** 📋 历史操作摘要 **\n" +
                compressedSummary +
                "\n\n---\n" +
                "以上是之前的操作历史摘要。请基于此摘要和当前屏幕状态继续执行任务。"
            ))
        }
        
        // 添加最新的消息
        compressedMessages.addAll(recentMessages)
        
        val newSize = calculateContextSize(compressedMessages)
        Log.d(TAG, "✅ 上下文压缩完成: ${currentSize / 1024}KB -> ${newSize / 1024}KB (减少 ${(currentSize - newSize) / 1024}KB)")
        Log.d(TAG, "压缩前消息数: ${messages.size}，压缩后消息数: ${compressedMessages.size}")
        
        // 验证压缩后的上下文是否仍然超过阈值（如果超过，说明单条消息太大，无法进一步压缩）
        if (newSize > contextSizeThreshold) {
            Log.w(TAG, "⚠️ 压缩后仍然超过阈值，可能是当前截图太大")
        }
        
        return compressedMessages
    }
    
    /**
     * 压缩历史消息，让AI总结关键信息
     */
    private suspend fun compressHistoryMessages(oldMessages: List<Message>): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "开始压缩 ${oldMessages.size} 条历史消息...")
            
            // 提取历史消息的文本内容（不包括图片，只提取文本部分）
            val historyText = oldMessages.mapNotNull { message ->
                val content = when (val msgContent = message.content) {
                    is String -> msgContent
                    is List<*> -> {
                        // 只提取文本内容，忽略图片
                        val textParts = msgContent.filterIsInstance<ContentItem>()
                            .filter { it.type == "text" }
                            .mapNotNull { it.text }
                        if (textParts.isNotEmpty()) {
                            textParts.joinToString("\n")
                        } else {
                            null // 如果只有图片没有文本，跳过这条消息
                        }
                    }
                    else -> null
                }
                
                if (content != null && content.isNotBlank()) {
                    when (message.role) {
                        "user" -> "用户: $content"
                        "assistant" -> "助手: $content"
                        else -> "${message.role}: $content"
                    }
                } else {
                    null
                }
            }.joinToString("\n\n")
            
            if (historyText.isBlank()) {
                Log.w(TAG, "历史消息中没有文本内容，使用简单摘要")
                return@withContext "已执行 ${oldMessages.size / 2} 步操作，继续执行任务。"
            }
            
            // 构建压缩请求
            val compressPrompt = """
                请总结以下对话历史，提取关键信息：
                1. 任务目标是什么
                2. 已执行了哪些主要操作（列出关键步骤）
                3. 遇到了什么困难，如何解决的
                4. 当前处于什么状态
                
                请用简洁的中文总结，保留重要信息，忽略细节和图片描述。
                总结格式：
                - 任务目标：[目标]
                - 已执行操作：[操作列表]
                - 遇到问题：[问题及解决方案]
                - 当前状态：[状态]
            """.trimIndent()
            
            val compressMessages = listOf(
                Message("system", "你是一个对话历史总结专家，能够提取关键信息并压缩对话内容。请用简洁的中文总结。"),
                Message("user", "$compressPrompt\n\n对话历史：\n$historyText")
            )
            
            // 调用模型压缩（使用较少的token）
            val response = modelClient.request(compressMessages)
            
            val summary = response.rawContent.trim()
            if (summary.isBlank()) {
                throw Exception("压缩结果为空")
            }
            
            Log.d(TAG, "✅ 历史消息压缩完成，摘要长度: ${summary.length}")
            Log.d(TAG, "压缩摘要: ${summary.take(300)}...")
            
            // 保存压缩摘要
            compressedHistory = summary
            
            summary
        } catch (e: Exception) {
            Log.e(TAG, "❌ 压缩历史消息失败", e)
            e.printStackTrace()
            // 如果压缩失败，返回简单的摘要
            val stepCount = oldMessages.count { it.role == "user" }
            "已执行约 $stepCount 步操作，继续执行任务。如果遇到问题，请尝试不同的方法。"
        }
    }
    
    /**
     * 在操作执行后，立即移除当前用户消息的图片，只保留文本内容
     * 在视觉模式和混合模式下，移除图片以节省上下文空间
     */
    private fun removeImageFromLastUserMessage() {
        if (contextMessages.isEmpty()) {
            return
        }
        
        // 只在视觉模式和混合模式下移除图片
        if (mode == Mode.ACCESSIBILITY) {
            return // 无障碍模式没有图片，无需处理
        }
        
        // 找到最后一条用户消息（应该包含图片）
        for (i in contextMessages.size - 1 downTo 0) {
            val message = contextMessages[i]
            if (message.role == "user") {
                when (val content = message.content) {
                    is List<*> -> {
                        val items = content.filterIsInstance<ContentItem>()
                        val hasImage = items.any { it.type == "image_url" }
                        if (hasImage) {
                            // 只保留文本内容，移除图片
                            val textItems = items.filter { it.type == "text" }
                            if (textItems.isNotEmpty()) {
                                contextMessages[i] = Message(message.role, textItems)
                                Log.d(TAG, "✅ 已移除最后一条用户消息中的图片，节省上下文空间（模式: $mode）")
                            } else if (mode == Mode.HYBRID) {
                                // 混合模式：如果移除图片后还有无障碍文本，保留文本
                                // 如果只有图片，保留一个占位符文本
                                contextMessages[i] = Message(message.role, 
                                    "屏幕内容已通过无障碍服务获取（图片已移除以节省空间）")
                                Log.d(TAG, "✅ 混合模式：已移除图片，保留文本占位符")
                            }
                        }
                        break // 只处理最后一条用户消息
                    }
                }
            }
        }
    }
    
    /**
     * 带重试机制的操作执行
     * 如果操作失败，会根据失败类型尝试不同的策略
     */
    private suspend fun executeActionWithRetry(
        actionJson: String,
        screenWidth: Int,
        screenHeight: Int
    ): ActionResult {
        var lastResult: ActionResult? = null
        
        // 如果连续失败次数过多，尝试让AI重新规划
        if (consecutiveFailures >= 2 && lastFailedAction != null) {
            Log.w(TAG, "⚠️ 连续失败 ${consecutiveFailures} 次，可能需要重新规划策略")
            // 添加失败提示到上下文，让AI知道需要换方法
            val failureHint = Message("user", "上次操作失败，请尝试不同的方法。失败的操作: $lastFailedAction")
            contextMessages.add(failureHint)
        }
        
        // 执行操作
        return try {
            actionHandler.execute(actionJson, screenWidth, screenHeight)
        } catch (e: Exception) {
            Log.e(TAG, "❌ 操作执行失败", e)
            e.printStackTrace()
            ActionResult(
                success = false,
                shouldFinish = false,
                message = "操作执行失败: ${e.message}"
            )
        }
    }

    /**
     * 截图（使用 ScreenshotManager）
     * 仅在视觉模式或混合模式下使用
     */
    private suspend fun captureScreenshot(): Bitmap? = withContext(Dispatchers.IO) {
        val manager = screenshotManager
        if (manager == null) {
            Log.w(TAG, "ScreenshotManager 未初始化，无法截图（当前模式: $mode）")
            return@withContext null
        }

        manager.captureScreen()
    }

    /**
     * 从响应中解析操作 JSON
     * 支持完整的 JSON 对象，包括嵌套的数组
     */
    private fun parseActionFromResponse(response: String): String {
        Log.d(TAG, "解析操作响应: ${response.take(200)}...")
        
        // 方法1: 尝试提取完整的 JSON 对象（支持嵌套）
        // 查找第一个 { 到最后一个匹配的 }
        var braceCount = 0
        var startIndex = -1
        var endIndex = -1
        
        for (i in response.indices) {
            when (response[i]) {
                '{' -> {
                    if (startIndex == -1) startIndex = i
                    braceCount++
                }
                '}' -> {
                    braceCount--
                    if (braceCount == 0 && startIndex != -1) {
                        endIndex = i
                        break
                    }
                }
            }
        }
        
        if (startIndex != -1 && endIndex != -1) {
            val jsonStr = response.substring(startIndex, endIndex + 1)
            try {
                // 验证是否是有效的 JSON
                org.json.JSONObject(jsonStr)
                Log.d(TAG, "✅ 提取到完整 JSON: $jsonStr")
                return jsonStr
            } catch (e: Exception) {
                Log.w(TAG, "提取的 JSON 无效，尝试其他方法", e)
            }
        }
        
        // 方法2: 如果不是 JSON 格式，尝试解析 do() 或 finish() 调用
        return parseActionFromCode(response)
    }

    /**
     * 从代码格式解析操作
     */
    private fun parseActionFromCode(code: String): String {
        val json = JSONObject()
        
        when {
            code.contains("finish(") -> {
                json.put("_metadata", "finish")
                val messageMatch = """message=["']([^"']+)["']""".toRegex().find(code)
                if (messageMatch != null) {
                    json.put("message", messageMatch.groupValues[1])
                }
            }
            code.contains("do(") -> {
                json.put("_metadata", "do")
                
                // 解析 action
                val actionMatch = """action=["']([^"']+)["']""".toRegex().find(code)
                if (actionMatch != null) {
                    val action = actionMatch.groupValues[1]
                    json.put("action", action)
                    
                    // 根据不同的 action 解析参数
                    when (action) {
                        "Tap", "Click" -> {
                            // 支持多种格式：
                            // 1. element=[x,y] - 2个数字（中心点坐标）
                            // 2. element=[x1,y1,x2,y2] - 4个数字（边界框，需要计算中心点）
                            // 3. element="x, y" - 字符串格式
                            
                            // 先尝试匹配4个数字的边界框格式 [xmin, ymin, xmax, ymax]
                            var elementMatch4 = """element=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]""".toRegex().find(code)
                            if (elementMatch4 != null) {
                                // 边界框格式：计算中心点
                                val x1 = elementMatch4.groupValues[1].toInt()
                                val y1 = elementMatch4.groupValues[2].toInt()
                                val x2 = elementMatch4.groupValues[3].toInt()
                                val y2 = elementMatch4.groupValues[4].toInt()
                                val centerX = (x1 + x2) / 2
                                val centerY = (y1 + y2) / 2
                                Log.d(TAG, "检测到边界框格式: [$x1,$y1,$x2,$y2] -> 中心点: ($centerX, $centerY)")
                                json.put("element", org.json.JSONArray().apply {
                                    put(centerX)
                                    put(centerY)
                                })
                            } else {
                                // 尝试匹配2个数字的中心点格式 [x, y]
                                var elementMatch = """element=\[(\d+),\s*(\d+)\]""".toRegex().find(code)
                                // 支持字符串格式: element="x, y"
                                if (elementMatch == null) {
                                    elementMatch = """element=["'](\d+),\s*(\d+)["']""".toRegex().find(code)
                                }
                                if (elementMatch != null) {
                                    json.put("element", org.json.JSONArray().apply {
                                        put(elementMatch.groupValues[1].toInt())
                                        put(elementMatch.groupValues[2].toInt())
                                    })
                                } else {
                                    Log.w(TAG, "⚠️ 未找到 element 坐标，原始代码: ${code.take(200)}")
                                }
                            }
                            // 统一使用 "Tap"
                            json.put("action", "Tap")
                        }
                        "Type", "Type_Name" -> {
                            val textMatch = """text=["']([^"']+)["']""".toRegex().find(code)
                            if (textMatch != null) {
                                json.put("text", textMatch.groupValues[1])
                            }
                        }
                        "Swipe" -> {
                            val startMatch = """start=\[(\d+),(\d+)\]""".toRegex().find(code)
                            val endMatch = """end=\[(\d+),(\d+)\]""".toRegex().find(code)
                            if (startMatch != null && endMatch != null) {
                                json.put("start", org.json.JSONArray().apply {
                                    put(startMatch.groupValues[1].toInt())
                                    put(startMatch.groupValues[2].toInt())
                                })
                                json.put("end", org.json.JSONArray().apply {
                                    put(endMatch.groupValues[1].toInt())
                                    put(endMatch.groupValues[2].toInt())
                                })
                            }
                        }
                        "Launch" -> {
                            val appMatch = """app=["']([^"']+)["']""".toRegex().find(code)
                            if (appMatch != null) {
                                json.put("app", appMatch.groupValues[1])
                            }
                        }
                    }
                }
            }
            else -> {
                json.put("_metadata", "finish")
                json.put("message", code)
            }
        }
        
        return json.toString()
    }
}

/**
 * 步骤结果
 */
data class StepResult(
    val success: Boolean,
    val finished: Boolean,
    val thinking: String,
    val action: String,
    val message: String? = null
)

