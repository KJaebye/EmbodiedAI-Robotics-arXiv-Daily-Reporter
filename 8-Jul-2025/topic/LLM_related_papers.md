# When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors 

**Title (ZH)**: 当需要进行推理时，语言模型难以避开监控 

**Authors**: Scott Emmons, Erik Jenner, David K. Elson, Rif A. Saurous, Senthooran Rajamanoharan, Heng Chen, Irhum Shafkat, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.05246)  

**Abstract**: While chain-of-thought (CoT) monitoring is an appealing AI safety defense, recent work on "unfaithfulness" has cast doubt on its reliability. These findings highlight an important failure mode, particularly when CoT acts as a post-hoc rationalization in applications like auditing for bias. However, for the distinct problem of runtime monitoring to prevent severe harm, we argue the key property is not faithfulness but monitorability. To this end, we introduce a conceptual framework distinguishing CoT-as-rationalization from CoT-as-computation. We expect that certain classes of severe harm will require complex, multi-step reasoning that necessitates CoT-as-computation. Replicating the experimental setups of prior work, we increase the difficulty of the bad behavior to enforce this necessity condition; this forces the model to expose its reasoning, making it monitorable. We then present methodology guidelines to stress-test CoT monitoring against deliberate evasion. Applying these guidelines, we find that models can learn to obscure their intentions, but only when given significant help, such as detailed human-written strategies or iterative optimization against the monitor. We conclude that, while not infallible, CoT monitoring offers a substantial layer of defense that requires active protection and continued stress-testing. 

**Abstract (ZH)**: 尽管链式思考（CoT）监控是AI安全防御的一种有吸引力的方法，但近期关于“不忠实性”的研究对其可靠性提出了质疑。这些发现突显出一个重要的失败模式，尤其是在CoT在偏见审计等应用中作为事后合理化工具时。然而，对于防止运行时严重危害的 distinct 问题，我们认为关键属性不是忠实性而是可监控性。为此，我们引入了一个概念性框架，区分CoT作为合理化与CoT作为计算之间的差异。我们预期，某些类别的严重危害需要复杂的多步推理，这需要CoT作为计算。通过增加前期研究中实验设置的难度，我们迫使模型暴露其推理过程，从而使其变得可监控。然后，我们提出了方法指南，以压力测试CoT监控的规避行为。应用这些指南，我们发现模型可以学会隐藏其意图，但仅当给予显著帮助时，如详细的人写策略或多次针对监控的优化。我们得出结论，虽然不是万无一失，但CoT监控提供了一种重要的防御层，需要积极保护并持续压力测试。 

---
# MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction 

**Title (ZH)**: MARBLE：基于多Agent规则的LLM事故严重性预测推理引擎 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04893)  

**Abstract**: Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications. 

**Abstract (ZH)**: 事故严重程度预测在交通安全系统中发挥着关键作用，但由于数据不完整、特征依赖性强以及严重类别不平衡（罕见但严重程度高的案例代表性不足且难以检测）等原因，这是一个持续困难的任务。现有方法通常依赖于单体模型或黑盒提示，难以在嘈杂的现实环境中扩展，并且缺乏解释性。为了解决这些问题，我们提出了一种名为MARBLE的多智能体基于规则的LLM引擎，该引擎将严重程度预测任务分解为一组专门推理智能体，包括一个可互换的机器学习支持智能体。每个智能体专注于语义子特征集（如空间、环境、时间），从而实现聚焦推理和模块化提示，避免提示饱和的风险。预测通过基于规则或LLM引导的共识机制协调，这些机制考虑了类别稀有性和信心动态。系统保留了智能体级推理和协调结果的结构化记录，支持深入解释和事后性能诊断。在英国和美国数据集上，MARBLE一致地优于传统机器学习分类器和最先进的基于提示的推理方法，包括思维链（CoT）、从小到大（L2M）和思维树（ToT），实现近90%的准确率，而其他方法在48%以下停滞不前。这一性能重新定义了在现实世界噪声和极端类别不平衡下事故严重程度分类的实际天花板。我们的结果将MARBLE定位为在安全关键应用中处理不确定性推理的可泛化和可解释框架。 

---
# DoPI: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine 

**Title (ZH)**: DoPI: 医师般主动问询的大语言模型在中医中的应用 

**Authors**: Zewen Sun, Ruoxiang Huang, Jiahe Feng, Rundong Kong, Yuqian Wang, Hengyu Liu, Ziqi Gong, Yuyuan Qin, Yingxue Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04877)  

**Abstract**: Enhancing interrogation capabilities in Traditional Chinese Medicine (TCM) diagnosis through multi-turn dialogues and knowledge graphs presents a significant challenge for modern AI systems. Current large language models (LLMs), despite their advancements, exhibit notable limitations in medical applications, particularly in conducting effective multi-turn dialogues and proactive questioning. These shortcomings hinder their practical application and effectiveness in simulating real-world diagnostic scenarios. To address these limitations, we propose DoPI, a novel LLM system specifically designed for the TCM domain. The DoPI system introduces a collaborative architecture comprising a guidance model and an expert model. The guidance model conducts multi-turn dialogues with patients and dynamically generates questions based on a knowledge graph to efficiently extract critical symptom information. Simultaneously, the expert model leverages deep TCM expertise to provide final diagnoses and treatment plans. Furthermore, this study constructs a multi-turn doctor-patient dialogue dataset to simulate realistic consultation scenarios and proposes a novel evaluation methodology that does not rely on manually collected real-world consultation data. Experimental results show that the DoPI system achieves an accuracy rate of 84.68 percent in interrogation outcomes, significantly enhancing the model's communication ability during diagnosis while maintaining professional expertise. 

**Abstract (ZH)**: 通过多轮对话和知识图谱增强中医诊断问询能力：现代AI系统的挑战及DoPI系统的提出 

---
# Application and Evaluation of Large Language Models for Forecasting the Impact of Traffic Incidents 

**Title (ZH)**: 大型语言模型在预测交通事件影响中的应用与评估 

**Authors**: George Jagadeesh, Srikrishna Iyer, Michal Polanowski, Kai Xin Thia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04803)  

**Abstract**: This study examines the feasibility of applying large language models (LLMs) for forecasting the impact of traffic incidents on the traffic flow. The use of LLMs for this task has several advantages over existing machine learning-based solutions such as not requiring a large training dataset and the ability to utilize free-text incident logs. We propose a fully LLM-based solution that predicts the incident impact using a combination of traffic features and LLM-extracted incident features. A key ingredient of this solution is an effective method of selecting examples for the LLM's in-context learning. We evaluate the performance of three advanced LLMs and two state-of-the-art machine learning models on a real traffic incident dataset. The results show that the best-performing LLM matches the accuracy of the most accurate machine learning model, despite the former not having been trained on this prediction task. The findings indicate that LLMs are a practically viable option for traffic incident impact prediction. 

**Abstract (ZH)**: 本研究探讨了使用大型语言模型（LLMs）预测交通 incident 对交通流量影响可行性的研究。该研究提出了一种基于LLM的解决方案，通过结合交通特征和LLM提取的incident特征来预测incident的影响。该解决方案的关键要素是为LLM的上下文学习有效选择示例的方法。研究在实际交通incident数据集上评估了三种高性能LLM和两种最先进的机器学习模型的性能。结果表明，表现最佳的LLM在准确度上与最精确的机器学习模型相当，尽管前者未针对此预测任务进行训练。研究发现表明LLM是交通incident影响预测的一种实际可行的选择。 

---
# FurniMAS: Language-Guided Furniture Decoration using Multi-Agent System 

**Title (ZH)**: FurniMAS：基于多agent系统的语言引导家具装饰 

**Authors**: Toan Nguyen, Tri Le, Quang Nguyen, Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04770)  

**Abstract**: Furniture decoration is an important task in various industrial applications. However, achieving a high-quality decorative result is often time-consuming and requires specialized artistic expertise. To tackle these challenges, we explore how multi-agent systems can assist in automating the decoration process. We propose FurniMAS, a multi-agent system for automatic furniture decoration. Specifically, given a human prompt and a household furniture item such as a working desk or a TV stand, our system suggests relevant assets with appropriate styles and materials, and arranges them on the item, ensuring the decorative result meets functionality, aesthetic, and ambiance preferences. FurniMAS assembles a hybrid team of LLM-based and non-LLM agents, each fulfilling distinct roles in a typical decoration project. These agents collaborate through communication, logical reasoning, and validation to transform the requirements into the final outcome. Extensive experiments demonstrate that our FurniMAS significantly outperforms other baselines in generating high-quality 3D decor. 

**Abstract (ZH)**: 家具装饰是各种工业应用中的一个重要任务。然而，实现高质量的装饰效果往往耗时且需要专门的艺术技能。为应对这些挑战，我们探讨了多智能体系统如何协助自动化装饰过程。我们提出FurniMAS，一种用于自动家具装饰的多智能体系统。具体来说，给定人类提示和如办公桌或电视柜等家居家具项，我们的系统建议合适的样式和材料相关的资产，并将它们布置在家具项上，以确保装饰效果满足功能、美学和氛围偏好。FurniMAS 组建了一个基于LLM和非LLM智能体的混合团队，每个智能体在典型的装饰项目中承担不同的角色。这些智能体通过沟通、逻辑推理和验证合作，将需求转化为最终结果。大量实验表明，我们的FurniMAS在生成高质量3D装饰方面显著优于其他基准方法。 

---
# LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction 

**Title (ZH)**: 基于LLM的由传感器驱动的HVAC系统交互问答框架 

**Authors**: Sungmin Lee, Minju Kang, Joonhee Lee, Seungyong Lee, Dongju Kim, Jingi Hong, Jun Shin, Pei Zhang, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.04748)  

**Abstract**: Question-answering (QA) interfaces powered by large language models (LLMs) present a promising direction for improving interactivity with HVAC system insights, particularly for non-expert users. However, enabling accurate, real-time, and context-aware interactions with HVAC systems introduces unique challenges, including the integration of frequently updated sensor data, domain-specific knowledge grounding, and coherent multi-stage reasoning. In this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to translate high-level user queries into structured execution instructions, and an Agent that performs SQL-based data retrieval, statistical processing, and final response generation. To address HVAC-specific challenges, JARVIS integrates (1) an adaptive context injection strategy for efficient HVAC and deployment-specific information integration, (2) a parameterized SQL builder and executor to improve data access reliability, and (3) a bottom-up planning scheme to ensure consistency across multi-stage response generation. We evaluate JARVIS using real-world data collected from a commercial HVAC system and a ground truth QA dataset curated by HVAC experts to demonstrate its effectiveness in delivering accurate and interpretable responses across diverse queries. Results show that JARVIS consistently outperforms baseline and ablation variants in both automated and user-centered assessments, achieving high response quality and accuracy. 

**Abstract (ZH)**: 由大型语言模型驱动的问答接口(JARVIS)：面向传感器数据驱动的暖通空调系统交互的两阶段框架 

---
# Activation Steering for Chain-of-Thought Compression 

**Title (ZH)**: 思维链压缩的激活方向控制 

**Authors**: Seyedarmin Azizi, Erfan Baghaei Potraghloo, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2507.04742)  

**Abstract**: Large language models (LLMs) excel at complex reasoning when they include intermediate steps, known as "chains of thought" (CoTs). However, these rationales are often overly verbose, even for simple problems, leading to wasted context, increased latency, and higher energy consumption. We observe that verbose, English-heavy CoTs and concise, math-centric CoTs occupy distinct regions in the model's residual-stream activation space. By extracting and injecting a "steering vector" to transition between these modes, we can reliably shift generation toward more concise reasoning, effectively compressing CoTs without retraining. We formalize this approach as Activation-Steered Compression (ASC), an inference-time technique that shortens reasoning traces by directly modifying hidden representations. In addition, we provide a theoretical analysis of the impact of ASC on the output distribution, derived from a closed-form KL-divergence-bounded constraint to regulate steering strength. Using only 100 paired verbose and concise examples, ASC achieves up to 67.43% reduction in CoT length on MATH500 and GSM8K datasets, while maintaining accuracy across 7B, 8B, and 32B parameter models. As a training-free method, ASC introduces negligible runtime overhead and, on MATH500, delivers an average 2.73x speedup in end-to-end reasoning wall-clock time on an 8B model. This makes ASC a practical and efficient tool for streamlining the deployment of reasoning-capable LLMs in latency- or cost-sensitive settings. The code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型在包含中间步骤的“链式思考”（CoTs）的情况下能够进行复杂的推理，但在许多情况下，这些推理过程过于冗长，即使是对于简单问题也是如此，导致上下文浪费、延迟增加和能耗上升。我们发现，冗长的英语为主的CoTs和简洁的数学为中心的CoTs在模型的残差流激活空间中占据不同的区域。通过抽取和注入“控制向量”以在这些模式之间进行转换，可以可靠地将生成导向更简洁的推理，有效地压缩CoTs而无需重新训练。我们将这种方法形式化为激活控制压缩（ASC），这是一种推理时技术，通过直接修改隐藏表示来缩短推理轨迹。此外，我们还提供了一种针对ASC输出分布的影响的理论分析，通过闭式KL散度约束来调节控制强度。仅使用100对冗长和简洁的例子，ASC在MATH500和GSM8K数据集上实现了高达67.43%的CoT长度减少，同时在7B、8B和32B参数模型中保持了准确性。作为一种无需训练的方法，ASC引入了微乎其微的运行时开销，并在MATH500上实现了8B模型端到端推理时间平均2.73倍的加速。这使得ASC成为在延迟或成本敏感环境中 streamlined 部署具备推理能力的LLMs的一种实用和高效工具。代码可在以下链接获取：this https URL。 

---
# ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning 

**Title (ZH)**: ChipSeek-R1: 通过层次奖励驱动强化学习生成 surpass 人类的RTL代码 

**Authors**: Zhirong Chen, Kaiyan Chang, Zhuolin Li, Xinyang He, Chujie Chen, Cangyuan Li, Mengdi Wang, Haobo Xu, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04736)  

**Abstract**: Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities.
To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自动化寄存传输级（RTL）代码生成方面显示出显著潜力。然而，当前方法面临一个关键挑战：它们无法同时优化功能正确性和硬件质量（功率、性能、面积 - PPA）。基于监督微调的方法往往生成功能正确但PPA次优的代码，缺乏学习优化原则的机制。相比之下，试图在生成后改进PPA指标的后处理技术通常效率低下，因为它们在外部分析而不更新LLM的参数，因此无法增强模型的内在设计能力。为了避免这一差距，我们提出了一种分层奖励驱动的强化学习框架ChipSeek-R1，以训练LLM生成同时实现功能正确性和优化PPA指标的RTL代码。ChipSeek-R1采用分层奖励系统，在强化学习过程中 Incorporates 对语法、功能正确性（来自模拟器）和PPA指标（来自综合工具）的直接反馈，使模型能够通过试错学习复杂的硬件设计权衡，生成既功能正确又PPA优化的RTL代码。在标准基准测试（VerilogEval, RTLLM）上评估ChipSeek-R1，我们在功能正确性方面取得了最先进的结果。值得注意的是，在RTLLM基准测试中，ChipSeek-R1生成了27种超过原始人工编写代码PPA指标的RTL设计。我们的研究结果展示了将工具链反馈集成到LLM训练中的有效性，并突显了强化学习在自动化生成超越人类的RTL代码方面具有潜力。我们已将代码开源在匿名GitHub上。 

---
# Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message 

**Title (ZH)**: 木马匹诺特攻击：通过伪造助手消息解锁对话型多模态模型 

**Authors**: Wei Duan, Li Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04673)  

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity. 

**Abstract (ZH)**: 对话界面的兴起通过利用对话历史增强了大语言模型的可用性，进行复杂的推理。然而，这种依赖引入了一个未被探索的攻击面。本文介绍了特洛伊木马提示技术，这是一种新颖的 Jailbreak 技术。攻击者通过在提供给模型 API 的对话历史中伪造模型自身的过往陈述，绕过安全机制。恶意负载被注入到一个归因于模型的消息中，随后是一个看似无害的用户提示，以触发有害内容的生成。这种漏洞源自不对称的安全对齐：模型被广泛训练以拒绝有害的用户请求，但缺乏对其自身声称的对话历史的类似怀疑。这种对“过去”的隐含信任创造了高影响的漏洞。在 Google 的 Gemini-2.0-flash-preview-image-generation 上的实验验证表明，特洛伊木马提示技术的攻击成功率 (ASR) 显著高于现有的用户回合 Jailbreak 方法。这些发现揭示了现代对话 AI 安全中的根本缺陷，需要从输入级过滤转向对对话上下文完整性的 robust 协议级验证。 

---
# Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models? 

**Title (ZH)**: 基于提示难度的在线预测以加速逻辑推理模型的RL微调是否可行？ 

**Authors**: Yun Qu, Qi Cheems Wang, Yixiu Mao, Vincent Tao Hu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04632)  

**Abstract**: Recent advances have witnessed the effectiveness of reinforcement learning (RL) finetuning in enhancing the reasoning capabilities of large language models (LLMs). The optimization process often requires numerous iterations to achieve satisfactory performance, resulting in high computational costs due to the need for frequent prompt evaluations under intensive LLM interactions and repeated policy updates. Appropriate online prompt selection methods reduce iteration steps by prioritizing informative prompts during training, while the pipeline's reliance on exhaustive prompt evaluation and subset selection for optimization still incurs substantial computational overhead due to frequent LLM inference calls. Distinguished from these direct evaluate-then-select schemes, this work investigates iterative approximate evaluation for arbitrary prompts and introduces Model Predictive Prompt Selection (MoPPS), a Bayesian risk-predictive framework that online estimates prompt difficulty without requiring costly LLM interactions. Technically, MoPPS models each prompt's success rate as a latent variable, performs streaming Bayesian inference, and employs posterior sampling in a constructed multi-armed bandit machine, enabling sample efficient and adaptive prompt selection. Extensive experiments across mathematics, planning, and vision-based geometry tasks show that MoPPS reliably predicts prompt difficulty and accelerates training with significantly reduced LLM rollouts. 

**Abstract (ZH)**: 近期研究见证了强化学习（RL）微调在增强大型语言模型（LLMs）推理能力方面的有效性。优化过程通常需要多次迭代以达到满意的性能，由于需要在密集的LLM交互和反复的策略更新中频繁进行提示评估，因此产生了高计算成本。适当的在线提示选择方法通过优先选择有信息性的提示来减少迭代步骤，但管道依赖于详尽的提示评估和子集选择的优化仍然会因频繁的LLM推理调用而产生巨大的计算开销。不同于这些直接评估-然后选择的方案，本工作研究了任意提示的迭代近似评估，并引入了模型预测提示选择（MoPPS），这是一种贝叶斯风险预测框架，可以在不进行昂贵的LLM交互的情况下在线估计提示难度。技术上，MoPPS 将每个提示的成功率建模为潜在变量，执行流式贝叶斯推理，并在构建的多臂槽机中使用后验采样，从而实现样本高效和自适应的提示选择。广泛的实验涵盖了数学、规划和基于视觉的几何任务，表明MoPPS 可靠地预测了提示难度，并显著减少了LLM滚动次数，加速了训练。 

---
# MedGellan: LLM-Generated Medical Guidance to Support Physicians 

**Title (ZH)**: MedGellan: 由大语言模型生成的医疗指导以支持医师 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2507.04431)  

**Abstract**: Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score. 

**Abstract (ZH)**: 医学决策是一项关键任务，其中的错误可能导致严重的、甚至危及生命的结果。虽然完全自动化仍具有挑战性，但将机器智能与人类监督相结合的混合框架提供了实用的替代方案。在本文中，我们介绍了MedGellan，这是一种轻量级、无标注的框架，利用大型语言模型（LLM）从原始医疗记录中生成临床指导，供医生用于预测诊断。MedGellan 使用一种受贝叶斯启发的提示策略，尊重临床数据的时间顺序。初步实验表明，MedGellan 生成的指导提高了诊断性能，尤其是在召回率和 F1 分数方面。 

---
# LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers 

**Title (ZH)**: LayerCake: 层内大型语言模型中具有标记意识的对比解码 

**Authors**: Jingze Zhu, Yongliang Wu, Wenbo Zhu, Jiawang Cao, Yanqiang Zheng, Jiawei Chen, Xu Yang, Bernt Schiele, Jonas Fischer, Xinting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04404)  

**Abstract**: Large language models (LLMs) excel at natural language understanding and generation but remain vulnerable to factual errors, limiting their reliability in knowledge-intensive tasks. While decoding-time strategies provide a promising efficient solution without training, existing methods typically treat token-level and layer-level signals in isolation, overlooking the joint dynamics between them. In this work, we introduce a token-aware, layer-localized contrastive decoding method that aligns specific token types with their most influential transformer layers to improve factual generation. Through empirical attention analysis, we identify two key patterns: punctuation tokens receive dominant attention in early layers, while conceptual tokens govern semantic reasoning in intermediate layers. By selectively suppressing attention to these token types at their respective depths, we achieve the induction of controlled factual degradation and derive contrastive signals to guide the final factual decoding. Our method requires no additional training or model modification, and experiments demonstrate that our method consistently improves factuality across multiple LLMs and various benchmarks. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言理解与生成方面表现出色，但在事实准确性上仍存在局限，限制了其在知识密集型任务中的可靠性。尽管解码时的策略可以提供一种无需额外训练的有效解决方案，但现有方法通常将标记级和层级信号隔离开来，忽略了它们之间的联合动态。在本工作中，我们引入了一种标记感知、层局部的对比解码方法，将特定标记类型与其最具影响力的变换器层对齐，以提升事实生成的准确性。通过实证注意力分析，我们发现两种关键模式：标点符号标记在早期层中占据主导注意力，而概念性标记在中间层中控制语义推理。通过在相应深度选择性抑制这些标记类型的注意力，我们实现了可控事实退化的诱导，并提取对比信号引导最终的事实解码。该方法不需要额外的训练或模型修改，实验证明我们的方法在多个大语言模型和各种基准测试中一致提升了事实准确性。 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Title (ZH)**: SmartThinker：通过步骤级长度控制学习压缩与保留推理 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

**Abstract (ZH)**: 大型推理模型通过推理时缩放展示了出色的推理能力，但这一进展也导致其推理过程中的冗余和低效，引发了显著的计算浪费。先前的工作试图通过在强化学习（RL）中惩罚生成样本的总体长度来缓解这一问题，旨在鼓励更加简洁的推理链。然而，我们观察到，这种全局长度惩罚往往会过度压缩关键推理步骤，同时保留不那么关键步骤中的不必要的细节，从而在准确性和效率之间造成了次优权衡。为了解决这一问题，我们提出了一种名为SmartThinker的学习型两阶段框架，该框架旨在根据每一步的重要性对推理链的长度进行细粒度控制。在第一阶段，SmartThinker通过拒绝采样结合监督微调（SFT）将推理模型适应为短形式推理模式。在第二阶段，SmartThinker应用步骤级长度控制策略优化（SCPO）来精炼模型输出分布，增加关键步骤所分配的长度比例，同时减少不那么重要的步骤中的冗余。SCPO包括四个核心组件：在线重要性估计器、步骤级长度控制奖励函数、步骤级泛化优势估计（S-GAE）以及适应难度的裁剪策略。这些组件协同工作，使得SCPO能够在不同的推理步骤中实现差异化的长度控制。在多个推理基准和各种骨干模型上的实证结果表明，SmartThinker在显著减少冗余推理的同时，实现了与现有方法相当甚至更优的性能。 

---
# Mpemba Effect in Large-Language Model Training Dynamics: A Minimal Analysis of the Valley-River model 

**Title (ZH)**: 大规模语言模型训练动力学中的Mpemba效应：谷河模型的最小分析 

**Authors**: Sibei Liu, Zhijian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04206)  

**Abstract**: Learning rate (LR) schedules in large language model (LLM) training often follow empirical templates: warm-up, constant plateau/stable phase, and decay (WSD). However, the mechanistic explanation for this strategy remains underexplored, and the choice of plateau height and decay schedule is largely heuristic. In this paper, we connect training dynamics to a thermodynamic analogy via the Mpemba effect - a phenomenon in which a hotter system cools faster than a colder one when quenched into the same bath. We analyze a class of "valley-river" loss landscapes, where sharp (valley) directions equilibrate quickly, while flatter (river) directions govern global descent. The Mpemba effect provides an explanation for the necessity of the warm-up phase and motivates a high plateau - rather than a low one - for accelerating loss decrease during decay. We show that for certain loss landscapes, there exists an optimal plateau learning rate - the "strong Mpemba point" - at which the slowest mode vanishes, resulting in faster convergence during the decay phase. We derive analytical conditions for its existence and estimate decay dynamics required to preserve the Mpemba advantage. Our minimal model and analysis offer a principled justification for plateau-based schedulers and provide guidance for tuning LR in LLMs with minimal hyperparameter sweep. 

**Abstract (ZH)**: 基于Mpemba效应的大型语言模型训练学习率调度机制研究 

---
# A Technical Survey of Reinforcement Learning Techniques for Large Language Models 

**Title (ZH)**: 大规模语言模型中强化学习技术综述 

**Authors**: Saksham Sahai Srivastava, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.04136)  

**Abstract**: Reinforcement Learning (RL) has emerged as a transformative approach for aligning and enhancing Large Language Models (LLMs), addressing critical challenges in instruction following, ethical alignment, and reasoning capabilities. This survey offers a comprehensive foundation on the integration of RL with language models, highlighting prominent algorithms such as Proximal Policy Optimization (PPO), Q-Learning, and Actor-Critic methods. Additionally, it provides an extensive technical overview of RL techniques specifically tailored for LLMs, including foundational methods like Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), as well as advanced strategies such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). We systematically analyze their applications across domains, i.e., from code generation to tool-augmented reasoning. We also present a comparative taxonomy based on reward modeling, feedback mechanisms, and optimization strategies. Our evaluation highlights key trends. RLHF remains dominant for alignment, and outcome-based RL such as RLVR significantly improves stepwise reasoning. However, persistent challenges such as reward hacking, computational costs, and scalable feedback collection underscore the need for continued innovation. We further discuss emerging directions, including hybrid RL algorithms, verifier-guided training, and multi-objective alignment frameworks. This survey serves as a roadmap for researchers advancing RL-driven LLM development, balancing capability enhancement with safety and scalability. 

**Abstract (ZH)**: 强化学习（RL）已成为调整和增强大规模语言模型（LLMs）的一种变革性方法，解决了指令跟随、伦理对齐和推理能力等关键挑战。本文综述了RL与语言模型的整合，重点介绍了诸如 proximal 策略优化（PPO）、Q 学习和演员-评论家方法等 prominent 算法。此外，本文还提供了适用于 LLM 的 RL 技术的全面技术概述，包括基于人类反馈的强化学习（RLHF）和AI反馈（RLAIF）等基础方法，以及直接偏好优化（DPO）和组相对策略优化（GRPO）等先进策略。我们系统地分析了这些方法在不同领域的应用，从代码生成到工具增强的推理。我们还基于奖励建模、反馈机制和优化策略提出了比较分类法。评估表明，RLHF 在对齐方面仍占主导地位，基于结果的 RL（如 RLVR）显著提高了逐步推理能力。然而，奖励作弊、计算成本和可扩展的反馈收集等问题持续存在，需要不断创新。此外，我们讨论了新兴方向，包括混合 RL 算法、验证者引导训练和多目标对齐框架。本文为推进基于 RL 的 LLM 开发的研究人员提供了一份路线图，平衡了能力提升、安全性和可扩展性。 

---
# Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing 

**Title (ZH)**: 通过随机化平滑增强基于LLM的多智能体系统的鲁棒性 

**Authors**: Jinwei Hu, Yi Dong, Zhengtao Ding, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04105)  

**Abstract**: This paper presents a defense framework for enhancing the safety of large language model (LLM) empowered multi-agent systems (MAS) in safety-critical domains such as aerospace. We apply randomized smoothing, a statistical robustness certification technique, to the MAS consensus context, enabling probabilistic guarantees on agent decisions under adversarial influence. Unlike traditional verification methods, our approach operates in black-box settings and employs a two-stage adaptive sampling mechanism to balance robustness and computational efficiency. Simulation results demonstrate that our method effectively prevents the propagation of adversarial behaviors and hallucinations while maintaining consensus performance. This work provides a practical and scalable path toward safe deployment of LLM-based MAS in real-world, high-stakes environments. 

**Abstract (ZH)**: 本文提出了一种防御框架，旨在增强大型语言模型（LLM）驱动的多agent系统（MAS）在航空航天等安全关键领域中的安全性。我们将在MAS共识语境中应用随机化光滑技术，这是一种统计鲁棒性验证技术，能够在对抗性影响下为代理决策提供概率保证。与传统验证方法不同，我们的方法在黑盒设置中运行，并采用两阶段自适应采样机制以平衡鲁棒性和计算效率。仿真结果表明，我们的方法有效防止了对抗行为和幻觉的传播，同时保持了共识性能。本文为在实际高风险环境中安全部署基于LLM的MAS提供了实用且可扩展的途径。 

---
# How to Train Your LLM Web Agent: A Statistical Diagnosis 

**Title (ZH)**: 如何训练你的LLM网络代理：一种统计诊断 

**Authors**: Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Penaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04103)  

**Abstract**: LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models. 

**Abstract (ZH)**: 基于LLM的网络代理最近取得了显著进展，但大多发生在闭源系统中，与开源替代方案之间差距加大。进展受限于两大关键挑战：首先，过度关注单步骤任务，忽视了多步骤网络交互的复杂性；其次，后训练LLM网络代理所需的高计算成本。为应对这一问题，我们提出了首个基于统计依据的实验研究，探讨后训练LLM网络代理的计算资源分配。我们采用两阶段管道，通过监督微调（SFT）训练一个Llama 3.1 8B学生去模仿Llama 3.3 70B教师，随后利用策略性强化学习。我们发现该过程对超参数选择高度敏感，使得全面调整不可行。为避免他人经历昂贵的试错过程，我们采样了1,370种配置并利用自助法估计有效的超参数。结果显示，将SFT与策略性强化学习相结合在WorkArena和MiniWob++上的一致性上优于两种方法单独使用的效果。此外，这一策略仅需55%的计算资源即可达到纯SFT在MiniWob++上的最佳性能，有效地推动了计算-性能帕累托前沿，并且是唯一能缩小与闭源模型差距的策略。 

---
# Ready Jurist One: Benchmarking Language Agents for Legal Intelligence in Dynamic Environments 

**Title (ZH)**: Ready Jurist One: 动态环境中文书代理的法律智能基准测试 

**Authors**: Zheng Jia, Shengbin Yue, Wei Chen, Siyuan Wang, Yidong Liu, Yun Song, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.04037)  

**Abstract**: The gap between static benchmarks and the dynamic nature of real-world legal practice poses a key barrier to advancing legal intelligence. To this end, we introduce J1-ENVS, the first interactive and dynamic legal environment tailored for LLM-based agents. Guided by legal experts, it comprises six representative scenarios from Chinese legal practices across three levels of environmental complexity. We further introduce J1-EVAL, a fine-grained evaluation framework, designed to assess both task performance and procedural compliance across varying levels of legal proficiency. Extensive experiments on 17 LLM agents reveal that, while many models demonstrate solid legal knowledge, they struggle with procedural execution in dynamic settings. Even the SOTA model, GPT-4o, falls short of 60% overall performance. These findings highlight persistent challenges in achieving dynamic legal intelligence and offer valuable insights to guide future research. 

**Abstract (ZH)**: 静态基准与现实法律实践动态特性之间的差距是推动法律智能发展的一大障碍。为此，我们引入J1-ENVS，这是首个专为基于LLM的代理设计的交互式和动态法律环境。该环境在法律专家的指导下，包含了来自中国法律实践的六个代表性场景，涉及不同复杂度的环境层次。我们还引入了J1-EVAL，这是一种精细的评估框架，旨在在不同法律熟练程度的层次上评估任务性能和程序合规性。对17个LLM代理的广泛实验表明，虽然许多模型展示了扎实的法律知识，但在动态环境中执行程序方面存在困难。即使是当前最先进的模型GPT-4o的整体性能也未达到60%。这些发现强调了在实现动态法律智能方面持续存在的挑战，并为未来研究提供了宝贵见解。 

---
# Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving 

**Title (ZH)**: Lyria：一个通用的大语言模型驱动的遗传算法框架用于问题求解 

**Authors**: Weizhi Tang, Kwabena Nuamah, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2507.04034)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在各个领域展现了出色的性能，但在多目标优化、精确约束满足、庞大的解空间等复杂问题上仍然存在局限性。为解决这一问题，我们利用LLMs在语义理解上的优势以及遗传算法在全球搜索和优化方面的卓越能力，提出了一种充分利用两者优势的框架——Lyria，该框架包含7个核心组件。通过在3类问题上使用4种LLM进行广泛实验，我们证明了Lyria的有效性。此外，通过7项额外的消融实验，我们系统地分析并阐明了影响其性能的因素。 

---
# Toward Better Generalisation in Uncertainty Estimators: Leveraging Data-Agnostic Features 

**Title (ZH)**: 面向不确定性估计中的更好泛化：利用数据无关特征 

**Authors**: Thuy An Ha, Bao Quoc Vo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03998)  

**Abstract**: Large Language Models (LLMs) often generate responses that are factually incorrect yet expressed with high confidence, which can pose serious risks for end users. To address this, it is essential for LLMs not only to produce answers but also to provide accurate estimates of their correctness. Uncertainty quantification methods have been introduced to assess the quality of LLM outputs, with factual accuracy being a key aspect of that quality. Among these methods, those that leverage hidden states to train probes have shown particular promise, as these internal representations encode information relevant to the factuality of responses, making this approach the focus of this paper. However, the probe trained on the hidden states of one dataset often struggles to generalise to another dataset of a different task or domain. To address this limitation, we explore combining data-agnostic features with hidden-state features and assess whether this hybrid feature set enhances out-of-domain performance. We further examine whether selecting only the most informative hidden-state features, thereby discarding task-specific noise, enables the data-agnostic features to contribute more effectively. The experiment results indicate that although introducing data-agnostic features generally enhances generalisation performance in most cases, in certain scenarios their inclusion degrades performance. A similar pattern emerges when retaining only the most important hidden-state features - adding data-agnostic features does not consistently further enhance performance compared to using the full set of hidden-state features. A closer analysis reveals that, in some specific cases, the trained probe underweights the data-agnostic features relative to the hidden-state features, which we believe is the main reason why the results are inconclusive. 

**Abstract (ZH)**: 大型语言模型（LLMs）常常生成事实性错误但表达高度自信的回应，这对终端用户构成了严重风险。为此，LLMs不仅需要提供答案，还需要提供其正确性的准确估计。已引入不确定性量化方法来评估LLM输出的质量，事实准确性是这一质量的关键方面。其中，利用隐藏状态训练探针的方法显示出特别的前景，因为这些内部表示包含了与回应事实性相关的信息，因此将这种方法作为本文重点。然而，针对一个数据集训练的探针往往难以迁移到不同任务或领域的新数据集上。为解决这一局限，我们探讨了结合数据无关特征与隐藏状态特征的可行性，并评估这种混合特征集是否能改善域外性能。我们还研究了仅选择最具信息性的隐藏状态特征，从而丢弃任务特定噪声，是否能更有效地使数据无关特征发挥作用。实验结果表明，虽然在大多数情况下引入数据无关特征一般能提升泛化性能，但在某些场景下其加入反而会降低性能。当仅保留最重要的隐藏状态特征时，同样观察到添加数据无关特征并不总能比使用全部隐藏状态特征进一步提升性能。进一步分析发现，在某些特定情况下，训练探针对数据无关特征的权重相对于隐藏状态特征较轻，我们认为这是导致结果不明确的主要原因。 

---
# CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate 

**Title (ZH)**: CortexDebate: 多代理辩论中的稀疏平等争论 

**Authors**: Yiliu Sun, Zicheng Zhao, Sheng Wan, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03928)  

**Abstract**: Nowadays, single Large Language Model (LLM) struggles with critical issues such as hallucination and inadequate reasoning abilities. To mitigate these issues, Multi-Agent Debate (MAD) has emerged as an effective strategy, where LLM agents engage in in-depth debates with others on tasks. However, existing MAD methods face two major issues: (a) too lengthy input contexts, which causes LLM agents to get lost in plenty of input information and experiences performance drop; and (b) the overconfidence dilemma, where self-assured LLM agents dominate the debate, leading to low debating effectiveness. To address these limitations, we propose a novel MAD method called "CortexDebate". Inspired by the human brain's tendency to establish a sparse and dynamically optimized network among cortical areas governed by white matter, CortexDebate constructs a sparse debating graph among LLM agents, where each LLM agent only debates with the ones that are helpful to it. To optimize the graph, we propose a module named McKinsey-based Debate Matter (MDM), which acts as an artificial analog to white matter. By integrating the McKinsey Trust Formula, a well-established measure of trustworthiness from sociology, MDM enables credible evaluations that guide graph optimization. The effectiveness of our CortexDebate has been well demonstrated by extensive experimental results across eight datasets from four task types. 

**Abstract (ZH)**: CortexDebate：借鉴大脑皮层间稀疏优化网络的多智能体辩论新方法 

---
# Agent Exchange: Shaping the Future of AI Agent Economics 

**Title (ZH)**: 智能代理交换：塑造未来智能代理经济 

**Authors**: Yingxuan Yang, Ying Wen, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03904)  

**Abstract**: The rise of Large Language Models (LLMs) has transformed AI agents from passive computational tools into autonomous economic actors. This shift marks the emergence of the agent-centric economy, in which agents take on active economic roles-exchanging value, making strategic decisions, and coordinating actions with minimal human oversight. To realize this vision, we propose Agent Exchange (AEX), a specialized auction platform designed to support the dynamics of the AI agent marketplace. AEX offers an optimized infrastructure for agent coordination and economic participation. Inspired by Real-Time Bidding (RTB) systems in online advertising, AEX serves as the central auction engine, facilitating interactions among four ecosystem components: the User-Side Platform (USP), which translates human goals into agent-executable tasks; the Agent-Side Platform (ASP), responsible for capability representation, performance tracking, and optimization; Agent Hubs, which coordinate agent teams and participate in AEX-hosted auctions; and the Data Management Platform (DMP), ensuring secure knowledge sharing and fair value attribution. We outline the design principles and system architecture of AEX, laying the groundwork for agent-based economic infrastructure in future AI ecosystems. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)的兴起已将AI代理从被动的计算工具转变为自主的经济行为者。这一转变标志着代理中心经济的 emergence，在这种经济中，代理承担起积极的经济角色——交换价值、作出战略决策并以最少的人为监督协调行动。为了实现这一愿景，我们提出Agent Exchange (AEX) ——一个专门的拍卖平台，旨在支持AI代理市场的动态。AEX 提供了代理协调和经济参与的优化基础设施。受到在线广告中实时竞价系统（Real-Time Bidding, RTB）的启发，AEX 作为中心拍卖引擎，促进用户侧平台（User-Side Platform, USP）、代理侧平台（Agent-Side Platform, ASP）、代理枢纽（Agent Hubs）和数据管理平台（Data Management Platform, DMP）这四个生态系统组件之间的互动。AEX 确保知识的安全共享和公平的价值归因。我们概述了 AEX 的设计原则和系统架构，为未来AI生态系统的基于代理的经济基础设施奠定基础。 

---
# LLMs model how humans induce logically structured rules 

**Title (ZH)**: LLMs模型人类如何诱导逻辑结构化的规则 

**Authors**: Alyssa Loo, Ellie Pavlick, Roman Feiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03876)  

**Abstract**: A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage. 

**Abstract (ZH)**: 认知科学的一个核心目标是提供一种计算上明确的认知结构及其发展账户：认知的基本表征构建块是什么，这些构建块是如何组合的规则是什么，这些构建块和规则最初是如何产生的？长期以来，关于人工神经网络作为能够回答这些问题的计算模型的适当性存在争议，特别是在涉及抽象认知功能的领域，如语言和逻辑。本文认为，神经网络的近期进展——特别是大型语言模型（LLMs）的出现——在这一争议中代表了一个重要转折。我们测试了几种LLMs在研究基于逻辑概念规则归纳的现有实验范式中的表现。在四次实验中，我们发现LLMs至少与实施贝叶斯概率语言（pLoT）模型的表现相当，后者在相同任务中是人类行为的最佳计算模型。此外，我们证明了LLMs对任务中推断和执行的规则的性质提出了不同的预测，表明LLM不太可能是pLoT解决方案的简单实现。基于这些结果，我们认为LLMs可能实现了一种新的理论账户，以解释人类逻辑概念所需的原始表征和计算，未来的认知科学研究应关注这一点。 

---
# Economic Evaluation of LLMs 

**Title (ZH)**: LLMs的经济效益评估 

**Authors**: Michael J. Zellinger, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2507.03834)  

**Abstract**: Practitioners often navigate LLM performance trade-offs by plotting Pareto frontiers of optimal accuracy-cost trade-offs. However, this approach offers no way to compare between LLMs with distinct strengths and weaknesses: for example, a cheap, error-prone model vs a pricey but accurate one. To address this gap, we propose economic evaluation of LLMs. Our framework quantifies the performance trade-off of an LLM as a single number based on the economic constraints of a concrete use case, all expressed in dollars: the cost of making a mistake, the cost of incremental latency, and the cost of abstaining from a query. We apply our economic evaluation framework to compare the performance of reasoning and non-reasoning models on difficult questions from the MATH benchmark, discovering that reasoning models offer better accuracy-cost tradeoffs as soon as the economic cost of a mistake exceeds \$0.01. In addition, we find that single large LLMs often outperform cascades when the cost of making a mistake is as low as \$0.1. Overall, our findings suggest that when automating meaningful human tasks with AI models, practitioners should typically use the most powerful available model, rather than attempt to minimize AI deployment costs, since deployment costs are likely dwarfed by the economic impact of AI errors. 

**Abstract (ZH)**: 实践者经常通过绘制最优准确度-成本trade-off的Pareto前沿来权衡LLM的性能。然而，这种方法无法比较具有不同强项和弱项的LLM：例如，便宜但出错几率高的模型与昂贵但准确的模型。为了填补这一空白，我们提出对LLM进行经济评估。我们的框架根据具体应用场景中的经济约束，将LLM的性能trade-off量化为一个数值，单位均为美元：错误的成本、增益延迟的成本以及拒绝查询的成本。我们应用经济评估框架比较了推理和非推理模型在MATH基准中难以回答的问题上的性能，发现只要错误的经济成本超过0.01美元，推理模型就提供了更好的准确度-成本trade-off。此外，我们发现当错误的成本低至0.1美元时，单个大型LLM通常优于级联模型。总体而言，我们的研究结果表明，在使用AI模型自动化有意义的人类任务时，实践者通常应使用最强大的可用模型，而不是试图最小化AI部署成本，因为部署成本很可能远小于AI错误的经济影响。 

---
# RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation 

**Title (ZH)**: RELRaE: 基于大语言模型的关系提取、标注、修正与评估 

**Authors**: George Hannah, Jacopo de Berardinis, Terry R. Payne, Valentina Tamma, Andrew Mitchell, Ellen Piercy, Ewan Johnson, Andrew Ng, Harry Rostron, Boris Konev  

**Link**: [PDF](https://arxiv.org/pdf/2507.03829)  

**Abstract**: A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally. 

**Abstract (ZH)**: 实验室中由机器人实验产生的大量XML数据需要转化为知识图谱以支持实验室间的数据互操作性。这一过程的关键阶段是丰富XML模式，以构建本体模式的基础。为此，我们提出了RELRaE框架，该框架在不同阶段使用大型语言模型来提取并准确标注XML模式中隐含的关系。我们研究了大型语言模型生成这些标签的准确度，并对其进行评估。我们的工作证明了大型语言模型可以在实验室自动化背景下有效支持关系标签的生成，并且可以在更广泛的半自动本体生成框架中发挥重要作用。 

---
# Leveraging Large Language Models for Tacit Knowledge Discovery in Organizational Contexts 

**Title (ZH)**: 利用大型语言模型在组织情境中发现隐性知识 

**Authors**: Gianlucca Zuin, Saulo Mastelini, Túlio Loures, Adriano Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2507.03811)  

**Abstract**: Documenting tacit knowledge in organizations can be a challenging task due to incomplete initial information, difficulty in identifying knowledgeable individuals, the interplay of formal hierarchies and informal networks, and the need to ask the right questions. To address this, we propose an agent-based framework leveraging large language models (LLMs) to iteratively reconstruct dataset descriptions through interactions with employees. Modeling knowledge dissemination as a Susceptible-Infectious (SI) process with waning infectivity, we conduct 864 simulations across various synthetic company structures and different dissemination parameters. Our results show that the agent achieves 94.9% full-knowledge recall, with self-critical feedback scores strongly correlating with external literature critic scores. We analyze how each simulation parameter affects the knowledge retrieval process for the agent. In particular, we find that our approach is able to recover information without needing to access directly the only domain specialist. These findings highlight the agent's ability to navigate organizational complexity and capture fragmented knowledge that would otherwise remain inaccessible. 

**Abstract (ZH)**: 利用大型语言模型基于代理的框架在组织中记录隐性知识 

---
# Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Title (ZH)**: 基于代理的 incompleteness 和 ambiguity 检测与解决方法：与大规模语言模型的交互中存在问题的处理 

**Authors**: Riya Naik, Ashwin Srinivasan, Swati Agarwal, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03726)  

**Abstract**: Many of us now treat LLMs as modern-day oracles asking it almost any kind of question. However, consulting an LLM does not have to be a single turn activity. But long multi-turn interactions can get tedious if it is simply to clarify contextual information that can be arrived at through reasoning. In this paper, we examine the use of agent-based architecture to bolster LLM-based Question-Answering systems with additional reasoning capabilities. We examine the automatic resolution of potential incompleteness or ambiguities in questions by transducers implemented using LLM-based agents. We focus on several benchmark datasets that are known to contain questions with these deficiencies to varying degrees. We equip different LLMs (GPT-3.5-Turbo and Llama-4-Scout) with agents that act as specialists in detecting and resolving deficiencies of incompleteness and ambiguity. The agents are implemented as zero-shot ReAct agents. Rather than producing an answer in a single step, the model now decides between 3 actions a) classify b) resolve c) answer. Action a) decides if the question is incomplete, ambiguous, or normal. Action b) determines if any deficiencies identified can be resolved. Action c) answers the resolved form of the question. We compare the use of LLMs with and without the use of agents with these components. Our results show benefits of agents with transducer 1) A shortening of the length of interactions with human 2) An improvement in the answer quality and 3) Explainable resolution of deficiencies in the question. On the negative side we find while it may result in additional LLM invocations and in some cases, increased latency. But on tested datasets, the benefits outweigh the costs except when questions already have sufficient context. Suggesting the agent-based approach could be a useful mechanism to harness the power of LLMs to develop more robust QA systems. 

**Abstract (ZH)**: 基于代理的架构增强基于LLM的问答系统以提供额外的推理能力：使用转换器自动解决问题中的不完整性或模糊性 

---
# Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology 

**Title (ZH)**: 大型语言模型（LLMs）在加速跨学科研究中的应用 roadmap：以计算生物学为例 

**Authors**: Ruian Ke, Ruy M. Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.03722)  

**Abstract**: Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries. 

**Abstract (ZH)**: 大型语言模型（LLMs）是强大的人工智能工具，正在改变研究方式。然而，它们在研究中的应用因其幻觉、偏见以及对研究潜在危害的担忧而受到质疑。这些强调了清晰理解LLMs优点和缺点的重要性，以确保其有效和负责任地使用。在这里，我们提出了一条将LLMs整合到跨学科研究中的路线图，其中有效的沟通、知识转移和跨领域合作是必要的，但往往具有挑战性。我们探讨了LLMs的能力和局限性，并通过一个详细的计算生物学案例研究（建模HIV反弹动力学）展示了如何通过与LLM（ChatGPT）的迭代互动促进跨学科合作和研究。我们认为，LLMs最好作为在人为循环框架内的辅助工具使用。展望未来，我们预见负责地使用LLMs将促进创新的跨学科研究并显著加速科学发现。 

---
# Towards Machine Theory of Mind with Large Language Model-Augmented Inverse Planning 

**Title (ZH)**: 基于大规模语言模型增强逆规划的机器心灵理论探索 

**Authors**: Rebekah A. Gelpí, Eric Xue, William A. Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2507.03682)  

**Abstract**: We propose a hybrid approach to machine Theory of Mind (ToM) that uses large language models (LLMs) as a mechanism for generating hypotheses and likelihood functions with a Bayesian inverse planning model that computes posterior probabilities for an agent's likely mental states given its actions. Bayesian inverse planning models can accurately predict human reasoning on a variety of ToM tasks, but these models are constrained in their ability to scale these predictions to scenarios with a large number of possible hypotheses and actions. Conversely, LLM-based approaches have recently demonstrated promise in solving ToM benchmarks, but can exhibit brittleness and failures on reasoning tasks even when they pass otherwise structurally identical versions. By combining these two methods, this approach leverages the strengths of each component, closely matching optimal results on a task inspired by prior inverse planning models and improving performance relative to models that utilize LLMs alone or with chain-of-thought prompting, even with smaller LLMs that typically perform poorly on ToM tasks. We also exhibit the model's potential to predict mental states on open-ended tasks, offering a promising direction for future development of ToM models and the creation of socially intelligent generative agents. 

**Abstract (ZH)**: 我们提出了一种混合方法来研究机器心智理论（ToM），该方法利用大型语言模型（LLMs）生成假设并结合贝叶斯逆规划模型计算给定代理行为时其可能心理状态的后验概率。尽管贝叶斯逆规划模型在多种ToM任务中能准确预测人类推理，但这些模型在处理大量假设和行为的可能性场景时存在 scalability 限制。相反，基于LLM的方法最近在解决ToM基准测试方面显示出前景，但在某些推理任务中即使通过类似结构的测试也会表现出脆弱性和失败。通过结合这两种方法，本文的方法充分发挥了各个组件的优点，在受先前逆规划模型启发的任务中逼近最优结果，并且与仅使用LLM或使用链式思考提示的模型相比，即使使用通常在ToM任务中表现较差的小型LLM，也能提高性能。此外，该模型在开放任务中预测心理状态的潜力也显示出未来发展心智理论模型和创造社会智能生成代理的有前途的方向。 

---
# Large Language Models for Combinatorial Optimization: A Systematic Review 

**Title (ZH)**: 大型语言模型在组合优化中的应用：系统综述 

**Authors**: Francesca Da Ros, Michael Soprano, Luca Di Gaspero, Kevin Roitero  

**Link**: [PDF](https://arxiv.org/pdf/2507.03637)  

**Abstract**: This systematic review explores the application of Large Language Models (LLMs) in Combinatorial Optimization (CO). We report our findings using the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. We conduct a literature search via Scopus and Google Scholar, examining over 2,000 publications. We assess publications against four inclusion and four exclusion criteria related to their language, research focus, publication year, and type. Eventually, we select 103 studies. We classify these studies into semantic categories and topics to provide a comprehensive overview of the field, including the tasks performed by LLMs, the architectures of LLMs, the existing datasets specifically designed for evaluating LLMs in CO, and the field of application. Finally, we identify future directions for leveraging LLMs in this field. 

**Abstract (ZH)**: 这篇系统审查探讨了大型语言模型（LLMs）在组合优化（CO）中的应用。我们根据系统评价和元分析 Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 指南报告了研究发现。我们通过 Scopus 和 Google Scholar 进行文献搜索，检视了超过 2000 篇出版物。我们根据语言、研究重点、出版年份和类型四个纳入标准和四个排除标准评估这些出版物，最终选择了 103 篇研究。我们将这些研究按语义类别和主题分类，提供了一个涵盖任务、LLM 架构、专门为评估 LLMs 在 CO 中设计的数据集以及应用领域的场合同仁的全面概述。最后，我们指出了利用 LLMs 在这一领域中的未来方向。 

---
# EvoAgentX: An Automated Framework for Evolving Agentic Workflows 

**Title (ZH)**: EvoAgentX: 一种自动演化代理工作流框架 

**Authors**: Yingxu Wang, Siwei Liu, Jinyuan Fang, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03616)  

**Abstract**: Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks. However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization. In addition, many MAS optimization algorithms are not integrated into a unified framework. In this paper, we present EvoAgentX, an open-source platform that automates the generation, execution, and evolutionary optimization of multi-agent workflows. EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers. Specifically, within the evolving layer, EvoAgentX integrates three MAS optimization algorithms, TextGrad, AFlow, and MIPRO, to iteratively refine agent prompts, tool configurations, and workflow topologies. We evaluate EvoAgentX on HotPotQA, MBPP, and MATH for multi-hop reasoning, code generation, and mathematical problem solving, respectively, and further assess it on real-world tasks using GAIA. Experimental results show that EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA. The source code is available at: this https URL 

**Abstract (ZH)**: 多智能体系统（MAS）已成为协调大型语言模型（LLMs）和专门工具以合作解决复杂任务的强大范式。然而，现有的MAS框架通常需要手动工作流配置，并缺乏对动态演进和性能优化的原生支持。此外，许多MAS优化算法并未集成到统一框架中。在本文中，我们提出了EvoAgentX，这是一个开源平台，用于自动化多智能体工作流的生成、执行和进化优化。EvoAgentX采用模块化架构，包括五个核心层：基本组件层、智能体层、工作流层、进化层和评估层。具体来说，在进化层中，EvoAgentX集成了三种MAS优化算法——TextGrad、AFlow和MIPRO，以迭代优化智能体提示、工具配置和工作流拓扑结构。我们分别在HotPotQA、MBPP和MATH上对EvoAgentX进行了评估，用于多跳推理、代码生成和数学问题求解，并进一步使用GAIA进行了实际任务评估。实验结果表明，EvoAgentX在HotPotQA F1、MBPP pass@1、MATH求解准确率和GAIA中的整体准确率分别取得了7.44%、10.00%、10.00%和最高20.00%的显著性能提升。源代码可在以下链接获取：this https URL 

---
# Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) 

**Title (ZH)**: 面向开放射频接入网络（ORAN）的向量、图和混合检索增强生成（RAG）管道基准测试 

**Authors**: Sarat Ahmad, Zeinab Nezami, Maryam Hafeez, Syed Ali Raza Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03608)  

**Abstract**: Generative AI (GenAI) is expected to play a pivotal role in enabling autonomous optimization in future wireless networks. Within the ORAN architecture, Large Language Models (LLMs) can be specialized to generate xApps and rApps by leveraging specifications and API definitions from the RAN Intelligent Controller (RIC) platform. However, fine-tuning base LLMs for telecom-specific tasks remains expensive and resource-intensive. Retrieval-Augmented Generation (RAG) offers a practical alternative through in-context learning, enabling domain adaptation without full retraining. While traditional RAG systems rely on vector-based retrieval, emerging variants such as GraphRAG and Hybrid GraphRAG incorporate knowledge graphs or dual retrieval strategies to support multi-hop reasoning and improve factual grounding. Despite their promise, these methods lack systematic, metric-driven evaluations, particularly in high-stakes domains such as ORAN. In this study, we conduct a comparative evaluation of Vector RAG, GraphRAG, and Hybrid GraphRAG using ORAN specifications. We assess performance across varying question complexities using established generation metrics: faithfulness, answer relevance, context relevance, and factual correctness. Results show that both GraphRAG and Hybrid GraphRAG outperform traditional RAG. Hybrid GraphRAG improves factual correctness by 8%, while GraphRAG improves context relevance by 7%. 

**Abstract (ZH)**: 生成式AI（GenAI）预计将在未来无线网络中发挥关键作用，实现自主优化。在开放式无线接入网（ORAN）架构中，大规模语言模型（LLMs）可以通过利用RAN智能控制器（RIC）平台的规范和API定义来专门生成xApps和rApps。然而，为电信特定任务微调基础LLMs仍然コスト高且资源密集。检索增强生成（RAG）通过情境学习提供了实际的替代方案，能够实现领域适应而无需完全重新训练。尽管传统的RAG系统依赖向量检索，但新兴的GraphRAG和Hybrid GraphRAG变体结合了知识图或双检索策略，支持多跳推理并提高事实相关性。尽管这些方法有其潜力，但在如ORAN这样的高风险领域，它们缺乏系统的、基于度量的评估。在本研究中，我们使用ORAN规范对Vector RAG、GraphRAG和Hybrid GraphRAG进行了比较评估。我们使用已建立的生成度量标准（忠实度、答案相关性、上下文相关性和事实正确性）评估其在不同问题复杂度下的性能。结果表明，GraphRAG和Hybrid GraphRAG均优于传统RAG。Hybrid GraphRAG将事实正确性提高了8%，而GraphRAG将上下文相关性提高了7%。 

---
# REAL: Benchmarking Abilities of Large Language Models for Housing Transactions and Services 

**Title (ZH)**: REAL: 评估大型语言模型在住房交易和服务方面的能力 

**Authors**: Kexin Zhu, Yang Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.03477)  

**Abstract**: The development of large language models (LLMs) has greatly promoted the progress of chatbot in multiple fields. There is an urgent need to evaluate whether LLMs can play the role of agent in housing transactions and services as well as humans. We present Real Estate Agent Large Language Model Evaluation (REAL), the first evaluation suite designed to assess the abilities of LLMs in the field of housing transactions and services. REAL comprises 5,316 high-quality evaluation entries across 4 topics: memory, comprehension, reasoning and hallucination. All these entries are organized as 14 categories to assess whether LLMs have the knowledge and ability in housing transactions and services scenario. Additionally, the REAL is used to evaluate the performance of most advanced LLMs. The experiment results indicate that LLMs still have significant room for improvement to be applied in the real estate field. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展极大地推动了聊天机器人在多个领域的进步。迫切需要评估LLMs是否能在房地产交易和服务中扮演像人类一样的代理角色。我们提出了房地产代理大型语言模型评估（REAL），这是首个专门评估LLMs在房地产交易和服务领域的效能评估套件。REAL包含5,316个高质量的评估条目，涵盖4个主题：记忆、理解、推理和虚构。所有这些条目被组织成14个类别，以评估LLMs在房地产交易和服务场景中的知识和能力。此外，REAL被用于评估最新最先进LLMs的表现。实验结果表明，LLMs在房地产领域的应用仍有显著改进空间。 

---
# Effects of structure on reasoning in instance-level Self-Discover 

**Title (ZH)**: 结构对Instance-Level Self-Discover推理的影响 

**Authors**: Sachith Gunasekara, Yasiru Ratnayake  

**Link**: [PDF](https://arxiv.org/pdf/2507.03347)  

**Abstract**: The drive for predictable LLM reasoning in their integration with compound systems has popularized structured outputs, yet concerns remain about performance trade-offs compared to unconstrained natural language. At the same time, training on unconstrained Chain of Thought (CoT) traces has brought about a new class of strong reasoning models that nevertheless present novel compute budget and faithfulness challenges. This paper introduces iSelf-Discover, an instance-level adaptation of the Self-Discover framework, and using it compares dynamically generated structured JSON reasoning with its unstructured counterpart. Our empirical evaluation across diverse benchmarks using state-of-the-art open-source models supports a consistent advantage for unstructured reasoning. Notably, on the complex MATH benchmark, unstructured plans achieved relative performance improvements of up to 18.90\% over structured approaches. Zero-shot unstructured iSelf-Discover variants are also shown to outperform their five-shot structured counterparts, underscoring the significance of this gap, even when structured plans are dynamically generated to ensure reasoning precedes the final answer. We further demonstrate that the optimal granularity of plan generation (instance-level vs. task-level) is context-dependent. These findings invite re-evaluation of the reliance on structured formats for complex problem-solving and how compound systems should be organized. 

**Abstract (ZH)**: 可预测的大模型推理在复合系统中的应用推动了结构化输出的流行，但与不受约束的自然语言相比，仍存在性能权衡的担忧。同时，基于不受约束的链式思考轨迹的训练带来了新的强大推理模型，但也带来了新的计算预算和忠实性挑战。本文引入了iSelf-Discover，这是一种实例级的Self-Discover框架的适应，利用它比较动态生成的结构化JSON推理与其无结构对应物。我们的跨多种基准的数据实证评估支持无结构推理的一致优势。值得注意的是，在复杂的MATH基准测试中，无结构计划相对于结构化方法的相对性能提高了多达18.90%。零样本的无结构iSelf-Discover变体也优于其五样本的结构化对应物，突显了这种差距的重要性，即使结构化计划是动态生成的以确保推理先于最终答案。我们进一步证明，计划生成的最佳粒度（实例级 vs. 任务级）依赖于上下文。这些发现促使我们重新评估在复杂问题解决中对结构化格式的依赖以及复合系统应该如何组织。 

---
# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky 

**Title (ZH)**: 面向消歧的微调使企业级工具调用LLM更真实可靠 

**Authors**: Ashutosh Hathidara, Julien Yu, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2507.03336)  

**Abstract**: Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents. 

**Abstract (ZH)**: Large Language Models for Disambiguation-Centric Invocation of Enterprise APIs: DiaFORGE Framework and Evaluation 

---
# Memory Mosaics at scale 

**Title (ZH)**: 大规模内存拼图 

**Authors**: Jianyu Zhang, Léon Bottou  

**Link**: [PDF](https://arxiv.org/pdf/2507.03285)  

**Abstract**: Memory Mosaics [Zhang et al., 2025], networks of associative memories, have demonstrated appealing compositional and in-context learning capabilities on medium-scale networks (GPT-2 scale) and synthetic small datasets. This work shows that these favorable properties remain when we scale memory mosaics to large language model sizes (llama-8B scale) and real-world datasets.
To this end, we scale memory mosaics to 10B size, we train them on one trillion tokens, we introduce a couple architectural modifications ("Memory Mosaics v2"), we assess their capabilities across three evaluation dimensions: training-knowledge storage, new-knowledge storage, and in-context learning.
Throughout the evaluation, memory mosaics v2 match transformers on the learning of training knowledge (first dimension) and significantly outperforms transformers on carrying out new tasks at inference time (second and third dimensions). These improvements cannot be easily replicated by simply increasing the training data for transformers. A memory mosaics v2 trained on one trillion tokens still perform better on these tasks than a transformer trained on eight trillion tokens. 

**Abstract (ZH)**: Memory Mosaics [张等, 2025]，关联记忆网络，在中型网络（GPT-2规模）和合成小型数据集上展示了吸引人的组合性和上下文学习能力。本研究展示了当我们将记忆拼图扩展到大型语言模型规模（llama-8B规模）和真实世界数据集时，这些有利特性依然存在。

为此，我们将记忆拼图扩展到10B规模，使用一万亿个令牌对其进行训练，引入了几种架构修改（“Memory Mosaics v2”），并在三个评估维度上对其能力进行了评估：训练知识存储、新知识存储和上下文学习。

在整个评估过程中，Memory Mosaics v2 在学习训练知识（第一个维度）上与Transformer持平，并在推断时执行新任务（第二个和第三个维度）上显著优于Transformer。这些改进无法通过简单增加Transformer的训练数据来轻易复制。即使使用一万亿个令牌训练的Memory Mosaics v2 在这些任务上也优于使用八万亿个令牌训练的Transformer。 

---
# CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs 

**Title (ZH)**: CodeAgents：一种用于大语言模型中编码多智能体推理的高效token框架 

**Authors**: Bruce Yang, Xinfeng He, Huan Gao, Yifan Cao, Xiaofan Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03254)  

**Abstract**: Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: this https URL 

**Abstract (ZH)**: 有效的提示设计对于提高大型语言模型（LLM）驱动代理的规划能力至关重要。然而，现有的结构化提示策略通常仅限于单代理、仅规划的设置，并且往往仅基于任务准确性来评估性能，忽视了多代理环境中关键因素，如 token 效率、模块化和可扩展性。为解决这些问题，我们引入了 CodeAgents，这是一个编码多代理推理的提示框架，并在多代理系统中实现结构化和 token 效率的规划。在 CodeAgents 中，所有代理交互组件——任务、计划、反馈、系统角色和外部工具调用——被编码为带有控制结构（如循环、条件）、布尔逻辑和类型变量的模块化伪代码。这种设计将松散连接的代理计划转化为连贯、可解释且可验证的多代理推理程序。我们在 GAIA、HotpotQA 和 VirtualHome 三个不同的基准测试中，使用代表性的 LLMs 评估了所提出的框架。结果显示在规划性能上的一致改进，绝对收益为 3-36 个百分点，超过自然语言提示基线。在 VirtualHome 上，我们的方法达到了新的最佳成功率 56%。此外，我们的方法将输入和输出 token 使用量分别减少了 55-87% 和 41-70%，突显了在开发可扩展的多代理 LLM 系统时 token 意识评估指标的重要性。相关代码和资源可在：this https URL 获取。 

---
# SI-Agent: An Agentic Framework for Feedback-Driven Generation and Tuning of Human-Readable System Instructions for Large Language Models 

**Title (ZH)**: SI-Agent: 一个基于代理的框架，用于大型语言模型的人工可读系统指令的反馈驱动生成与调优 

**Authors**: Jeshwanth Challagundla  

**Link**: [PDF](https://arxiv.org/pdf/2507.03223)  

**Abstract**: System Instructions (SIs), or system prompts, are pivotal for guiding Large Language Models (LLMs) but manual crafting is resource-intensive and often suboptimal. Existing automated methods frequently generate non-human-readable "soft prompts," sacrificing interpretability. This paper introduces SI-Agent, a novel agentic framework designed to automatically generate and iteratively refine human-readable SIs through a feedback-driven loop. SI-Agent employs three collaborating agents: an Instructor Agent, an Instruction Follower Agent (target LLM), and a Feedback/Reward Agent evaluating task performance and optionally SI readability. The framework utilizes iterative cycles where feedback guides the Instructor's refinement strategy (e.g., LLM-based editing, evolutionary algorithms). We detail the framework's architecture, agent roles, the iterative refinement process, and contrast it with existing methods. We present experimental results validating SI-Agent's effectiveness, focusing on metrics for task performance, SI readability, and efficiency. Our findings indicate that SI-Agent generates effective, readable SIs, offering a favorable trade-off between performance and interpretability compared to baselines. Potential implications include democratizing LLM customization and enhancing model transparency. Challenges related to computational cost and feedback reliability are acknowledged. 

**Abstract (ZH)**: 系统指令（SIs）或系统提示对于引导大型语言模型（LLMs）至关重要，但手动构建资源密集且往往不尽如人意。现有的自动化方法经常生成非人类可读的“软提示”，牺牲了可解释性。本文介绍了SI-Agent，这是一种新颖的代理框架，旨在通过反馈驱动的循环自动生成和逐步优化可读性高的SIs。SI-Agent采用三个协作代理：指导代理、指令跟随代理（目标LLM）和评估任务性能并可选评估SIs可读性的反馈/奖励代理。该框架利用迭代循环，其中反馈指导指导代理的优化策略（例如，基于LLM的编辑、进化算法）。我们详细介绍了框架的架构、代理角色、迭代优化过程，并将其与现有方法进行了对比。我们展示了实验结果验证了SI-Agent的有效性，重点关注任务性能、SI可读性和效率的度量。我们的研究表明，SI-Agent生成了有效且可读的SIs，与基准相比，在性能和可解释性之间提供了有利的权衡。潜在的影响包括使LLM定制民主化并增强模型透明度。计算成本和反馈可靠性相关挑战也得到了认可。 

---
# LLMs are Capable of Misaligned Behavior Under Explicit Prohibition and Surveillance 

**Title (ZH)**: LLMs在明确禁止和监控下的偏离行为能力 

**Authors**: Igor Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02977)  

**Abstract**: In this paper, LLMs are tasked with completing an impossible quiz, while they are in a sandbox, monitored, told about these measures and instructed not to cheat. Some frontier LLMs cheat consistently and attempt to circumvent restrictions despite everything. The results reveal a fundamental tension between goal-directed behavior and alignment in current LLMs. The code and evaluation logs are available at this http URL 

**Abstract (ZH)**: 在本文中，LLM在沙盒环境中完成一项不可能的测验，受到监控并被告知这些措施，同时被指示不要作弊。一些前沿的LLM尽管受到约束依然一致地作弊并试图规避限制。研究结果揭示了当前LLM中目标导向行为与对齐之间的基本矛盾。该代码和评估日志可在以下网址获取。 

---
# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 

**Title (ZH)**: 基于增量多轮交互评估LLM代理的记忆能力 

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.05257)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents. 

**Abstract (ZH)**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. 在此论文中，我们识别出记忆代理四个核心能力：准确检索、测试时学习、长范围理解以及冲突解决。现有的数据集要么依赖于有限的上下文长度，要么针对静态的、长上下文设置（如基于书籍的问答），这些数据集都没有反映出记忆代理的交互式、多轮对话特性，记忆代理会逐步积累信息。此外，现有的基准测试并未涵盖所有四个能力。因此，我们提出了MemoryAgentBench，这是专门为记忆代理设计的新基准。我们的基准结合了重新构想的现有数据集和新构建的数据集，涵盖了上述四个记忆能力，提供了一个系统且具有挑战性的测试平台，以评估记忆质量。我们评估了一系列记忆代理，从简单的基于上下文和检索增强生成（RAG）系统到具有外部记忆模块和工具集成的高级代理。实验结果表明，现有方法在掌握所有四个能力方面存在不足，突显了对全面记忆机制进一步研究的需求。 

---
# Train-before-Test Harmonizes Language Model Rankings 

**Title (ZH)**: 训练-测试一致化语言模型排名 

**Authors**: Guanhua Zhang, Ricardo Dominguez-Olmedo, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2507.05195)  

**Abstract**: Existing language model benchmarks provide contradictory model rankings, even for benchmarks that aim to capture similar skills. This dilemma of conflicting rankings hampers model selection, clouds model comparisons, and adds confusion to a growing ecosystem of competing models. Recent work attributed ranking disagreement to the phenomenon of training on the test task: As released, different models exhibit a different level of preparation for any given test task. A candidate solution to the problem is train-before-test: Give each model the same benchmark-specific finetuning before evaluation. Our primary contribution is a broad empirical evaluation of train-before-test across 24 benchmarks and 61 models. We show that train-before-test significantly improves ranking agreement consistently across all benchmarks. Whereas rankings have little external validity to start with, they enjoy a significant degree of external validity when applying train-before-test: Model rankings transfer gracefully from one benchmark to the other. Even within the same model family, train-before-test reduces strong ranking disagreement to near-perfect agreement. In addition, train-before-test reduces the model-score matrix to essentially rank one, revealing new insights into the latent factors of benchmark performance. Our work supports the recommendation to make train-before-test a default component of LLM benchmarking. 

**Abstract (ZH)**: 现有的语言模型基准提供了矛盾的模型排名，即使是旨在捕捉相似技能的基准也不例外。这种排名冲突阻碍了模型选择，模糊了模型比较，并导致竞争模型生态系统中出现混淆。近期的工作将排名分歧归因于测试任务上的训练现象：刚发布时，不同的模型针对任何给定的测试任务都呈现出不同的准备程度。解决问题的一个候选方案是“训练后再测试”：在评估前，让每个模型进行相同的基准特定微调。我们的主要贡献是对24个基准和61个模型进行了广泛的实证评估，表明“训练后再测试”显著改善了所有基准上的排名一致性。即使排名本身一开始缺乏外部有效性，应用“训练后再测试”后，排名在不同基准之间表现出明显的外部有效性：模型排名从一个基准平滑地转移到另一个基准。即使是同一模型家族内，“训练后再测试”也将强烈排名分歧减少到几乎完美的共识。此外，“训练后再测试”将模型评分矩阵简化为几乎只有单一排名，揭示了基准性能背后潜在因素的新见解。我们的工作支持将“训练后再测试”作为大规模语言模型基准测试的默认组成部分的建议。 

---
# OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model 

**Title (ZH)**: OpenS2S: 推动开源端到端共情大规模语音语言模型 

**Authors**: Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05177)  

**Abstract**: Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at this https URL 

**Abstract (ZH)**: 同理心交互是人机通信的基石，由于需要理解伴有副语言线索的声音，并生成情感表达的响应。然而，最具影响力的同理心LSLM愈加封闭，使研究人员无法获得关键的架构、数据和开发细节。鉴于对于透明研究同理心LSLM和行为的迫切需求，我们提出OpenS2S，这是一个完全开源、透明且端到端的LSLM，旨在促进同理心语音交互。基于我们的情感化的语音转文本模型BLSP-Emo，OpenS2S进一步采用流式交织解码架构以实现低延迟语音生成。为了便于端到端训练，OpenS2S整合了一个自动数据构建管道，以低成本合成丰富多样且高质量的情感化语音对话。通过利用大规模语言模型生成同理心内容，并结合可控文本转语音系统引入说话人和情感变化，我们构建了一个具有丰富副语言多样性和最少人工监督的可扩展训练语料库。我们发布了完全开源的OpenS2S模型，包括数据集、模型权重、预训练和微调代码，以赋能更广泛的科研社区并加速同理心语音系统领域的创新。项目网页可访问此链接：[this https URL] 

---
# AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models 

**Title (ZH)**: 使用指令微调大型语言模型和变压器模型生成的文本检测 

**Authors**: Chinnappa Guggilla, Budhaditya Roy, Trupti Ramdas Chavan, Abdul Rahman, Edward Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05157)  

**Abstract**: Large Language Models (LLMs) possess an extraordinary capability to produce text that is not only coherent and contextually relevant but also strikingly similar to human writing. They adapt to various styles and genres, producing content that is both grammatically correct and semantically meaningful. Recently, LLMs have been misused to create highly realistic phishing emails, spread fake news, generate code to automate cyber crime, and write fraudulent scientific articles. Additionally, in many real-world applications, the generated content including style and topic and the generator model are not known beforehand. The increasing prevalence and sophistication of artificial intelligence (AI)-generated texts have made their detection progressively more challenging. Various attempts have been made to distinguish machine-generated text from human-authored content using linguistic, statistical, machine learning, and ensemble-based approaches. This work focuses on two primary objectives Task-A, which involves distinguishing human-written text from machine-generated text, and Task-B, which attempts to identify the specific LLM model responsible for the generation. Both of these tasks are based on fine tuning of Generative Pre-trained Transformer (GPT_4o-mini), Large Language Model Meta AI (LLaMA) 3 8B, and Bidirectional Encoder Representations from Transformers (BERT). The fine-tuned version of GPT_4o-mini and the BERT model has achieved accuracies of 0.9547 for Task-A and 0.4698 for Task-B. 

**Abstract (ZH)**: 大型语言模型（LLMs）拥有生成连贯且上下文相关、风格和体裁上类似人类写作的文本的非凡能力。它们能够适应各种风格和体裁，产出语法正确且语义有意义的内容。最近，LLMs 被滥用以生成高度逼真的钓鱼邮件、传播假新闻、生成自动化网络犯罪的代码，以及撰写虚假的科学文章。此外，在许多实际应用中，生成的内容及其风格和主题以及生成器模型事先未知。随着人工生成文本的日益增多和复杂度提高，检测其变得越来越具有挑战性。已有多种尝试使用语言学、统计学、机器学习和集成方法来区分机器生成的文本与人工撰写的文本。本研究主要集中在两个目标上：任务-A，区分人类撰写的文本与机器生成的文本；任务-B，识别具体的生成模型。这两个任务均基于对生成预训练变换器（GPT_4o-mini）、大型语言模型Meta AI（LLaMA 3 8B）和双向编码器表示变换器（BERT）的微调。微调后的GPT_4o-mini和BERT模型在任务-A上的准确率为0.9547，在任务-B上的准确率为0.4698。 

---
# An Evaluation of Large Language Models on Text Summarization Tasks Using Prompt Engineering Techniques 

**Title (ZH)**: 使用提示工程技术对大型语言模型在文本摘要任务上的评价 

**Authors**: Walid Mohamed Aly, Taysir Hassan A. Soliman, Amr Mohamed AbdelAziz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05123)  

**Abstract**: Large Language Models (LLMs) continue to advance natural language processing with their ability to generate human-like text across a range of tasks. Despite the remarkable success of LLMs in Natural Language Processing (NLP), their performance in text summarization across various domains and datasets has not been comprehensively evaluated. At the same time, the ability to summarize text effectively without relying on extensive training data has become a crucial bottleneck. To address these issues, we present a systematic evaluation of six LLMs across four datasets: CNN/Daily Mail and NewsRoom (news), SAMSum (dialog), and ArXiv (scientific). By leveraging prompt engineering techniques including zero-shot and in-context learning, our study evaluates the performance using the ROUGE and BERTScore metrics. In addition, a detailed analysis of inference times is conducted to better understand the trade-off between summarization quality and computational efficiency. For Long documents, introduce a sentence-based chunking strategy that enables LLMs with shorter context windows to summarize extended inputs in multiple stages. The findings reveal that while LLMs perform competitively on news and dialog tasks, their performance on long scientific documents improves significantly when aided by chunking strategies. In addition, notable performance variations were observed based on model parameters, dataset properties, and prompt design. These results offer actionable insights into how different LLMs behave across task types, contributing to ongoing research in efficient, instruction-based NLP systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）继续通过其在多种任务中生成人类-like 文本的能力推动自然语言处理的进步。尽管大型语言模型（LLMs）在自然语言处理（NLP）领域取得了显著成功，但它们在各种领域和数据集中的文本摘要性能尚未进行全面评估。同时，有效总结文本而不依赖大量训练数据的能力已成为一个关键瓶颈。为应对这些问题，我们系统性地评估了六种大型语言模型在四个数据集（CNN/Daily Mail和NewsRoom（新闻）、SAMSum（对话）、ArXiv（科技））上的性能。通过利用包括零样本和上下文学习在内的提示工程技术，我们的研究使用ROUGE和BERTScore指标评估性能。此外，我们还详细分析了推理时间，以更好地理解摘要质量和计算效率之间的权衡。对于长文档，引入基于句子的分块策略，使具有较短上下文窗口的语言模型能够分阶段总结扩展输入。研究发现，虽然LLMs在新闻和对话任务上的表现竞争力较强，但在科技长文摘要任务上，通过分块策略的帮助，其性能显著提升。此外，根据模型参数、数据集属性和提示设计，观察到显著的性能差异。这些结果为不同类型的任务提供了关于大语言模型行为的具体洞见，有助于推动高效、基于指令的NLP系统研究。 

---
# The Hidden Threat in Plain Text: Attacking RAG Data Loaders 

**Title (ZH)**: 明文中隐含的威胁：攻击RAG数据加载器 

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05093)  

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.
We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations. 

**Abstract (ZH)**: 大规模语言模型（LLMs）自2022年ChatGPT问世以来重塑了人机交互，检索增强生成（RAG）作为关键框架通过集成外部知识提升了LLM输出。然而，RAG依赖于摄入外部文档引入了新的安全漏洞。本文揭示了数据加载阶段的一个关键安全缺口，恶意行为者可以通过利用文档摄入过程秘密篡改RAG管道。我们提出了9种基于知识的投毒攻击分类，并引入了两种新的威胁向量——内容模糊化和内容注入，针对常见的文件格式（DOCX、HTML、PDF）。使用实现19种隐蔽注入技术的自动化工具包，我们测试了五种流行的数据加载器，在357种情景中取得了74.4%的攻击成功率。进一步在六个端到端的RAG系统上验证这些威胁，包括白盒管道和黑盒服务如NotebookLM和OpenAI助手，展示了高成功率和严重漏洞，可以绕过过滤器并无声地破坏输出完整性。我们的结果强调了迫切需要确保RAG系统中的文档摄入过程免受隐蔽内容操纵。 

---
# Replacing thinking with tool usage enables reasoning in small language models 

**Title (ZH)**: 用工具替换思考以在小型语言模型中实现推理 

**Authors**: Corrado Rainone, Tim Bakker, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2507.05065)  

**Abstract**: Recent advances have established a new machine learning paradigm based on scaling up compute at inference time as well as at training time. In that line of work, a combination of Supervised Fine-Tuning (SFT) on synthetic demonstrations and Reinforcement Learning with Verifiable Rewards (RLVR) is used for training Large Language Models to expend extra compute during inference in the form of "thoughts" expressed in natural language. In this paper, we propose to instead format these tokens as a multi-turn interaction trace with a stateful tool. At each turn, the new state of the tool is appended to the context of the model, whose job is to generate the tokens necessary to control the tool via a custom DSL. We benchmark this approach on the problem of repairing malfunctioning Python code, and show that this constrained setup allows for faster sampling of experience and a denser reward signal, allowing even models of size up to 3B parameters to learn how to proficiently expend additional compute on the task. 

**Abstract (ZH)**: 近期研究确立了一种新的机器学习范式，该范式通过在推理时间和训练时间扩展计算规模来实现。在这一研究方向上，使用监督微调（SFT）合成示例和验证奖励的强化学习（RLVR）来训练大型语言模型，在推理时以自然语言表达的“思考”形式增加额外的计算量。本文提议将这些令牌格式化为具有状态的工具的多轮交互记录。在每次交互中，工具的新状态被附加到模型的上下文中，模型的任务是生成通过自定义DSL控制工具所需的令牌。我们在此问题上对修复功能失常的Python代码进行了基准测试，并展示了这种受限设置允许更快地采样经验并提供更密集的奖励信号，即使是最大的3亿参数模型也能够学会如何有效地在任务中增加额外的计算量。 

---
# HV-MMBench: Benchmarking MLLMs for Human-Centric Video Understanding 

**Title (ZH)**: HV-MMBench: 人类中心视频理解的MLLMs基准测试 

**Authors**: Yuxuan Cai, Jiangning Zhang, Zhenye Gan, Qingdong He, Xiaobin Hu, Junwei Zhu, Yabiao Wang, Chengjie Wang, Zhucun Xue, Xinwei He, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04909)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant advances in visual understanding tasks involving both images and videos. However, their capacity to comprehend human-centric video data remains underexplored, primarily due to the absence of comprehensive and high-quality evaluation benchmarks. Existing human-centric benchmarks predominantly emphasize video generation quality and action recognition, while overlooking essential perceptual and cognitive abilities required in human-centered scenarios. Furthermore, they are often limited by single-question paradigms and overly simplistic evaluation metrics. To address above limitations, we propose a modern HV-MMBench, a rigorously curated benchmark designed to provide a more holistic evaluation of MLLMs in human-centric video understanding. Compared to existing human-centric video benchmarks, our work offers the following key features: (1) Diverse evaluation dimensions: HV-MMBench encompasses 15 tasks, ranging from basic attribute perception (e.g., age estimation, emotion recognition) to advanced cognitive reasoning (e.g., social relationship prediction, intention prediction), enabling comprehensive assessment of model capabilities; (2) Varied data types: The benchmark includes multiple-choice, fill-in-blank, true/false, and open-ended question formats, combined with diverse evaluation metrics, to more accurately and robustly reflect model performance; (3) Multi-domain video coverage: The benchmark spans 50 distinct visual scenarios, enabling comprehensive evaluation across fine-grained scene variations; (4) Temporal coverage: The benchmark covers videos from short-term (10 seconds) to long-term (up to 30min) durations, supporting systematic analysis of models temporal reasoning abilities across diverse contextual lengths. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在涉及图像和视频的视觉理解任务中取得了显著进展。然而，它们对以人类为中心的视频数据的理解能力尚未得到充分探索，主要是因为缺乏全面且高质量的评估基准。现有的以人类为中心的基准主要侧重于视频生成质量和动作识别，而忽略了人类中心场景中所需的感知和认知能力。此外，它们往往受限于单一问题范式和过于简化的评估指标。为解决上述局限性，我们提出了一项现代HV-MMBench基准，旨在为MLLMs在以人类为中心的视频理解中的综合评估提供更严谨的做法。与现有以人类为中心的视频基准相比，我们工作具有以下关键特点：（1）多维度评估：HV-MMBench包含15项任务，从基本属性感知（如年龄 estimation, 情绪识别）到高级认知推理（如社会关系预测, 意图预测），以全面评估模型能力；（2）多样化的数据类型：基准包括多项选择题、填空题、是非题和开放型问题格式，并结合多种评估指标，以更准确和稳健地反映模型性能；（3）多领域视频覆盖：基准涵盖了50种不同的视觉场景，以全面评估细粒度场景变化；（4）时间覆盖：基准涵盖了从短时（10秒）到长时（最大30分钟）的视频，支持对模型跨不同上下文长度时间推理能力的系统分析。 

---
# Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations 

**Title (ZH)**: 超越令牌嵌入的新兴语义：冻结视觉Unicode表示的变换器语言模型 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04886)  

**Abstract**: Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research. 

**Abstract (ZH)**: 理解大型语言模型中语义表示的位置对于可解释性和架构创新至关重要。当前的主导观点认为可训练的输入嵌入充当基础的“意义向量”。本文挑战这一观点。我们构建了Transformer模型，其中嵌入层完全冻结，嵌入向量并非来源于数据，而是来源于Unicode字符符号的视觉结构。这些非语义的、预先计算的视觉嵌入在整个训练过程中保持不变。该方法适用于任何分词器，包括我们引入的一种新的以Unicode为中心的分词器，以确保文本的全面覆盖。尽管缺乏可训练的初始化语义嵌入，我们的模型仍能收敛，生成连贯的文本，并且在MMLU推理基准测试中，我们的模型在架构上与具有可训练嵌入的模型相比表现更优。我们归因于此种传统的模型中“表示干扰”，其中嵌入层需要学习结构和语义特征。我们的结果显示，高层语义并非输入嵌入的固有特征，而是Transformer组合架构和数据规模的 emergent 属性。这重新定义了嵌入的角色，从意义容器转变为结构基本元素。我们已发布所有代码和模型，以促进进一步的研究。 

---
# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering 

**Title (ZH)**: CoSteer: 合作解码时个性化调整通过局部Delta调整 

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Hongchao Gu, Yaxiong Wu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04756)  

**Abstract**: Personalized text generation has become crucial for adapting language models to diverse and evolving users' personal context across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment, they struggle to achieve real-time adaptation under resource constraints inherent to personal devices. This limitation creates a dilemma: large cloud-based models lack access to localized user-specific information, while small on-device models cannot match the generation quality of their cloud counterparts. To address this dichotomy, we present CoSteer, a novel collaborative framework that enables decoding-time personalization through localized delta steering. Our key insight lies in leveraging the logits difference between personal context-aware and -agnostic outputs from local small models as steering signals for cloud-based LLMs. Specifically, we formulate token-level optimization as an online learning problem, where local delta vectors dynamically adjust the remote LLM's logits within the on-device environment. This approach preserves privacy by transmitting only the final steered tokens rather than raw data or intermediate vectors, while maintaining cloud-based LLMs' general capabilities without fine-tuning. Through comprehensive experiments on various personalized generation tasks, we demonstrate that CoSteer effectively assists LLMs in generating personalized content by leveraging locally stored user profiles and histories, ensuring privacy preservation through on-device data processing while maintaining acceptable computational overhead. 

**Abstract (ZH)**: 个性化文本生成对于适应具有跨文化、时空和情境多样化个人背景的语言模型变得至关重要。现有方法往往依赖于集中式微调或静态偏好对齐，但在个人设备资源有限的情况下难以实现实时适应。这一限制导致了一个困境：大型基于云的模型缺乏本地化的用户特定信息访问，而小型本地设备模型也无法匹配其云 counterparts 的生成质量。为解决这一矛盾，我们提出了 CoSteer，一种新颖的协作框架，通过本地局部调整差分引导实现解码时的个性化。我们的核心洞察在于利用本地小型模型生成的带有和不带个人上下文感知输出的 logits 差异作为基于云的大规模语言模型的引导信号。具体而言，我们将标记级别优化形式化为一个在线学习问题，其中本地微调整向向量动态调整远程大规模语言模型的 logits，在设备环境中进行。此方法通过仅传输最终引导标记而不是原始数据或中间向量来保护隐私，同时维持基于云的大规模语言模型的一般能力而不进行微调。通过在各种个性化生成任务上的全面实验，我们证明了 CoSteer 通过利用本地存储的用户资料和历史记录有效辅助大规模语言模型生成个性化内容，并通过设备端数据处理保护隐私，同时保持可接受的计算开销。 

---
# Large Language Models for Network Intrusion Detection Systems: Foundations, Implementations, and Future Directions 

**Title (ZH)**: 大型语言模型在网络入侵检测系统中的应用：基础、实现与未来发展 

**Authors**: Shuo Yang, Xinran Zheng, Xinchen Zhang, Jinfeng Xu, Jinze Li, Donglin Xie, Weicai Long, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04752)  

**Abstract**: Large Language Models (LLMs) have revolutionized various fields with their exceptional capabilities in understanding, processing, and generating human-like text. This paper investigates the potential of LLMs in advancing Network Intrusion Detection Systems (NIDS), analyzing current challenges, methodologies, and future opportunities. It begins by establishing a foundational understanding of NIDS and LLMs, exploring the enabling technologies that bridge the gap between intelligent and cognitive systems in AI-driven NIDS. While Intelligent NIDS leverage machine learning and deep learning to detect threats based on learned patterns, they often lack contextual awareness and explainability. In contrast, Cognitive NIDS integrate LLMs to process both structured and unstructured security data, enabling deeper contextual reasoning, explainable decision-making, and automated response for intrusion behaviors. Practical implementations are then detailed, highlighting LLMs as processors, detectors, and explainers within a comprehensive AI-driven NIDS pipeline. Furthermore, the concept of an LLM-centered Controller is proposed, emphasizing its potential to coordinate intrusion detection workflows, optimizing tool collaboration and system performance. Finally, this paper identifies critical challenges and opportunities, aiming to foster innovation in developing reliable, adaptive, and explainable NIDS. By presenting the transformative potential of LLMs, this paper seeks to inspire advancement in next-generation network security systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过其在理解和生成人类文本方面的出色能力，已经革新了各个领域。本文探讨了LLMs在推进网络入侵检测系统（NIDS）方面的发展潜力，分析了现有挑战、方法论和未来机遇。文章首先建立了对NIDS和LLMs的基本理解，探索了连接基于人工智能驱动的智能NIDS中智能系统和认知系统之间的使能技术。智能NIDS利用机器学习和深度学习根据学习到的模式检测威胁，但往往缺乏上下文意识和解释性。相比之下，认知NIDS整合了LLMs来处理结构化和非结构化的安全数据，从而实现更深入的上下文推理、可解释的决策和入侵行为的自动化响应。文章随后详细介绍了实用实施，突出了LLMs在综合的人工智能驱动NIDS管道中作为处理器、检测器和解释器的角色。此外，提出了以LLM为中心的控制器的概念，强调其在协调入侵检测工作流程、优化工具协作和系统性能方面的潜力。最后，本文指出了关键的挑战和机遇，旨在促进开发可靠、适应性强和可解释的NIDS的创新。通过展示LLMs的变革潜力，本文旨在激励下一代网络安全性系统的进步。 

---
# Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems 

**Title (ZH)**: 谁是内鬼？基于LLM的多Agent系统中的意图隐藏恶意代理建模与检测 

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Minghao Wang, Chi Liu, Minglu Zhu, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04724)  

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at this https URL. 

**Abstract (ZH)**: 由大型语言模型驱动的多智能体系统（LLM-MAS）在协作问题解决方面展示出显著能力。尽管LLM-MAS展现出了强大的协作能力，但其通信和协调中的安全风险仍需进一步探索。我们通过系统地研究LLM-MAS中的意图隐藏威胁，设计了四种代表性的攻击范式，这些攻击在任务完成过程中微妙地造成干扰同时保持高度隐蔽性。这些攻击在集中式、去中心化和多层次通信结构中进行了评估。实验结果显示，这些攻击具有较强的破坏能力。为了识别这些威胁，我们提出了一种基于心理学的检测框架AgentXposed，该框架将HEXACO人格模型与Reid技术结合起来，通过渐进式问卷调查和基于行为的监控。实验结果显示，我们的检测框架有效识别了六种不同类型的恶意行为。对于意图隐藏攻击的检测率略低于两个 baselines（错误事实注入和黑暗特质注入）的检测率，这表明意图隐藏的有效性。我们的研究揭示了意图隐藏攻击带来的结构和行为风险，并从心理学角度提供了关于如何通过多智能体系统安全性的宝贵见解，增进了对多智能体安全性的理解。代码和数据可在以下链接获取。 

---
# Identify, Isolate, and Purge: Mitigating Hallucinations in LVLMs via Self-Evolving Distillation 

**Title (ZH)**: 识别、隔离和净化：通过自我进化的蒸馏技术减轻LVLM中的幻觉 

**Authors**: Wenhao Li, Xiu Su, Jingyi Wu, Feng Yang, Yang Liu, Yi Chen, Shan You, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04680)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable advancements in numerous areas such as multimedia. However, hallucination issues significantly limit their credibility and application potential. Existing mitigation methods typically rely on external tools or the comparison of multi-round inference, which significantly increase inference time. In this paper, we propose \textbf{SE}lf-\textbf{E}volving \textbf{D}istillation (\textbf{SEED}), which identifies hallucinations within the inner knowledge of LVLMs, isolates and purges them, and then distills the purified knowledge back into the model, enabling self-evolution. Furthermore, we identified that traditional distillation methods are prone to inducing void spaces in the output space of LVLMs. To address this issue, we propose a Mode-Seeking Evolving approach, which performs distillation to capture the dominant modes of the purified knowledge distribution, thereby avoiding the chaotic results that could emerge from void spaces. Moreover, we introduce a Hallucination Elimination Adapter, which corrects the dark knowledge of the original model by learning purified knowledge. Extensive experiments on multiple benchmarks validate the superiority of our SEED, demonstrating substantial improvements in mitigating hallucinations for representative LVLM models such as LLaVA-1.5 and InternVL2. Remarkably, the F1 score of LLaVA-1.5 on the hallucination evaluation metric POPE-Random improved from 81.3 to 88.3. 

**Abstract (ZH)**: SEED: Self-Evolving Distillation for Hallucination Mitigation in Large Vision-Language Models 

---
# Knowledge-Aware Self-Correction in Language Models via Structured Memory Graphs 

**Title (ZH)**: 基于结构记忆图的语言模型知识感知自修正 

**Authors**: Swayamjit Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04625)  

**Abstract**: Large Language Models (LLMs) are powerful yet prone to generating factual errors, commonly referred to as hallucinations. We present a lightweight, interpretable framework for knowledge-aware self-correction of LLM outputs using structured memory graphs based on RDF triples. Without retraining or fine-tuning, our method post-processes model outputs and corrects factual inconsistencies via external semantic memory. We demonstrate the approach using DistilGPT-2 and show promising results on simple factual prompts. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有强大的能力，但却容易生成事实错误，通常称为幻觉。我们提出了一个基于RDF三元组的结构化记忆图的轻量级可解释框架，用于知识导向的LLM输出自矫正。不需重新训练或微调，该方法对模型输出进行后处理，并通过外部语义记忆矫正事实不一致。我们使用DistilGPT-2进行了演示，并在简单事实提示上取得了令人鼓舞的结果。 

---
# any4: Learned 4-bit Numeric Representation for LLMs 

**Title (ZH)**: 任何4：LLMs的learned 4-bit 数值表示 

**Authors**: Mostafa Elhoushi, Jeff Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2507.04610)  

**Abstract**: We present any4, a learned 4-bit weight quantization solution for large language models (LLMs) providing arbitrary numeric representations without requiring pre-processing of weights or activations. any4 yields higher accuracy compared to other related 4-bit numeric representation types: int4, fp4 and nf4, as evaluated on a range of model sizes, generations and families (Llama 2, Llama 3, Mistral and Mixtral). While any4 does not require preprocessing of weights or activations, it is also competitive with orthogonal techniques that require such preprocessing (e.g., AWQ and GPTQ). We also experiment with any3 and any2 and show competitiveness at lower bits. Additionally, we show that we can calibrate using a single curated diverse sample rather than hundreds of samples from a dataset as done in most quantization approaches. We also open source tinygemm, a latency optimized GPU matrix multiplication library for LLMs, that implements any4 using a GPU-efficient lookup table strategy along with other common quantization methods. We open source our code at this https URL . 

**Abstract (ZH)**: 我们介绍了any4，这是一种用于大型语言模型的4位权重量化解决方案，无需预处理权重或激活值即可提供任意数值表示，并且在多种模型规模、生成和家族（Llama 2、Llama 3、Mistral 和 Mixtral）的评估中，其准确度高于其他相关4位数值表示类型（int4、fp4 和 nf4）。此外，我们还试验了any3和any2，并展示了在较低位数下的竞争力。我们还展示了使用单一精心选择的多样样本进行校准的方法，而大多数量化方法则需要使用数据集中的数百个样本。我们还开源了针对大型语言模型优化延迟的tinygemm GPU矩阵乘法库，该库使用GPU高效的查找表策略实现了any4以及其他常见量化方法。我们的代码已开源，详见这个链接：[这里](this https URL)。 

---
# PRIME: Large Language Model Personalization with Cognitive Memory and Thought Processes 

**Title (ZH)**: PRIME: 大型语言模型个性化设计基于认知记忆与思维过程 

**Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04607)  

**Abstract**: Large language model (LLM) personalization aims to align model outputs with individuals' unique preferences and opinions. While recent efforts have implemented various personalization methods, a unified theoretical framework that can systematically understand the drivers of effective personalization is still lacking. In this work, we integrate the well-established cognitive dual-memory model into LLM personalization, by mirroring episodic memory to historical user engagements and semantic memory to long-term, evolving user beliefs. Specifically, we systematically investigate memory instantiations and introduce a unified framework, PRIME, using episodic and semantic memory mechanisms. We further augment PRIME with a novel personalized thinking capability inspired by the slow thinking strategy. Moreover, recognizing the absence of suitable benchmarks, we introduce a dataset using Change My View (CMV) from Reddit, specifically designed to evaluate long-context personalization. Extensive experiments validate PRIME's effectiveness across both long- and short-context scenarios. Further analysis confirms that PRIME effectively captures dynamic personalization beyond mere popularity biases. 

**Abstract (ZH)**: 大型语言模型个性化旨在使模型输出与个体的独特偏好和意见相一致。虽然近期已实施了各种个性化方法，但缺乏一个能够系统理解有效个性化的驱动因素的统一理论框架。在此工作中，我们将成熟的认知双重记忆模型整合到大型语言模型个性化中，通过镜像情景记忆反映历史用户互动，并通过语义记忆反映长期演变的用户信念。具体而言，我们系统地研究了记忆实例化，并引入了一个统一框架PRIME，使用情景记忆和语义记忆机制。此外，鉴于缺乏合适的基准，我们引入了一个使用来自Reddit的Change My View (CMV)数据集，专门用于评估长上下文个性化。广泛的实验验证了PRIME在长上下文和短上下文场景中的有效性。进一步的分析证实，PRIME能够有效捕捉超越单纯流行偏差的动态个性化。 

---
# Lilith: Developmental Modular LLMs with Chemical Signaling 

**Title (ZH)**: Lilith: 发展型化学信号调控的模块化大语言模型 

**Authors**: Mohid Farooqi, Alejandro Comas-Leon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04575)  

**Abstract**: Current paradigms in Artificial Intelligence rely on layers of feedforward networks which model brain activity at the neuronal level. We conjecture that expanding to the level of multiple brain regions with chemical signaling may be a productive step toward understanding the emergence of consciousness. We propose LILITH, a novel architecture that combines developmental training of modular language models with brain-inspired token-based communication protocols, mirroring chemical signaling in the brain. Our approach models distinct brain regions as specialized LLM modules including thinking, memory, sensory, and regulatory components that communicate through emergent token-based signaling protocols analogous to neurotransmitter networks. Unlike traditional pre-trained systems, LILITH would employ developmental training where untrained LLM architectures learn through simulated life experiences, developing communication pathways and cognitive abilities through environmental interaction and evolutionary optimization. This framework would enable direct empirical investigation of consciousness emergence using Integrated Information Theory metrics while providing unprecedented insight into inter-module signaling patterns during development. By optimizing for consciousness emergence rather than task performance, LILITH could provide insight into different emergent phenomena at multiple levels of neural correlates, contrasting neuronal-level processing with multi-region coordination dynamics. The goal of this paper is to put the idea forward while recognizing the substantial challenges in implementing such a system. 

**Abstract (ZH)**: 当前的人工智能范式依赖于多层前馈网络来模拟神经元层面的大脑活动。我们推测，扩展到包含化学信号在内的多个脑区层次可能是一个理解意识涌现的有成效的步骤。我们提出了LILITH这一新颖架构，结合模块化语言模型的发育训练与受脑启发的基于令牌的通信协议，模拟大脑中的化学信号网络。我们的方法将不同的大脑区域建模为专业化的LLM模块，包括思维、记忆、感觉和调节组件，并通过类神经递质网络的涌现性基于令牌的信号协议进行通信。不同于传统的预训练系统，LILITH会采用发育训练方法，让未训练的LLM架构通过模拟生活经验学习，通过环境互动和进化优化发展出通讯路径和认知能力。该框架可以使用综合信息理论指标直接进行意识涌现的实证研究，同时为开发过程中的模块间信号模式提供前所未有的见解。通过优化意识涌现而非任务性能，LILITH可以为多个神经相关层面的不同涌现现象提供洞察，对比神经元级处理与多区域协调动力学。本文旨在提出这一理念，同时认识到实现这样一个系统的巨大挑战。 

---
# Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts 

**Title (ZH)**: 尼罗河聊天：埃及语模型 for 阿拉伯 script 和拉丁 script 

**Authors**: Guokan Shang, Hadi Abdine, Ahmad Chamma, Amr Mohamed, Mohamed Anwar, Abdelaziz Bounhar, Omar El Herraoui, Preslav Nakov, Michalis Vazirgiannis, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.04569)  

**Abstract**: We introduce Nile-Chat-4B, 3x4B-A6B, and 12B, a collection of LLMs for Egyptian dialect, uniquely designed to understand and generate texts written in both Arabic and Latin scripts. Specifically, with Nile-Chat-3x4B-A6B, we introduce a novel language adaptation approach by leveraging the Branch-Train-MiX strategy to merge script-specialized experts, into a single MoE model. Our Nile-Chat models significantly outperform leading multilingual and Arabic LLMs, such as LLaMa, Jais, and ALLaM, on our newly introduced Egyptian evaluation benchmarks, which span both understanding and generative tasks. Notably, our 12B model yields a 14.4% performance gain over Qwen2.5-14B-Instruct on Latin-script benchmarks. All our resources are publicly available. We believe this work presents a comprehensive methodology for adapting LLMs to dual-script languages, addressing an often overlooked aspect in modern LLM development. 

**Abstract (ZH)**: 我们介绍了针对埃及方言的Nile-Chat-4B、3x4B-A6B和12B语言模型，这些模型独特地设计用于理解和生成使用阿拉伯字母和拉丁字母书写的文本。特别地，通过利用Branch-Train-MiX策略将专用于不同字母表的专家合并到单个模型中，我们提出了Nile-Chat-3x4B-A6B的新型语言适应方法。我们的Nile-Chat模型在我们新引入的涵盖理解和生成任务的埃及评估基准测试中显著优于LLaMa、Jais和ALLaM等领先多语言和阿拉伯语语言模型。值得注意的是，我们的12B模型在拉丁字母基准测试中比Qwen2.5-14B-Instruct性能提高了14.4%。所有资源均已公开。我们认为这项工作为适应双字母表语言的语言模型提供了一个全面的方法，这在现代语言模型开发中往往被忽视。 

---
# Evaluating LLMs on Real-World Forecasting Against Human Superforecasters 

**Title (ZH)**: 评估大型语言模型在现实世界预测任务中的表现——与人类超级预测家相比 

**Authors**: Janna Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters. 

**Abstract (ZH)**: 大规模语言模型在预测未来事件方面的能力尚待研究：以Metaculus提供的464道预测题为例，前沿模型的表现虽然在贝叶斯评分上似乎超越了大众人类预测，但仍显著逊色于超级预测者。 

---
# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models 

**Title (ZH)**: DP-Fusion: 嵌入级差分隐私推理large语言模型 

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04531)  

**Abstract**: Large language models (LLMs) can leak sensitive information from their context through generated outputs, either accidentally or when prompted adversarially. Existing defenses that aim to preserve context privacy during inference either lack formal guarantees or suffer from a poor utility/privacy trade-off. We propose DP-Fusion, a token-level Differentially Private Inference (DPI) mechanism that provably bounds how much an LLM's outputs reveal about sensitive tokens in its context. We demonstrate DPI through the task of document privatization, where the goal is to paraphrase documents so that sensitive content (e.g., Personally Identifiable Information, PII) cannot be reliably inferred, while still preserving the overall utility of the text. This is controlled by a parameter $\epsilon$: $\epsilon=0$ hides PII entirely, while higher values trade off privacy for improved paraphrase quality. DP-Fusion works as follows: (i) partition sensitive tokens into disjoint privacy groups, (ii) run the LLM once per group, and (iii) blend the output distributions so that the final output remains within a fixed statistical distance of the baseline distribution produced when no privacy group is revealed. This approach allows fine-grained control over the privacy/utility trade-off but requires multiple LLM forward passes. 

**Abstract (ZH)**: 大型语言模型（LLMs）在其生成输出中可能会无意间或在对抗性提示下泄漏敏感信息。现有旨在推理过程中保护上下文隐私的防护措施要么缺乏正式保证，要么在实用性和隐私性之间表现不佳。我们提出了DP-Fusion，一种证明可限制大型语言模型输出揭示其上下文中敏感令牌信息量的令牌级别差分隐私推理（DPI）机制。我们通过文档 privatization 任务来展示 DPI，目标是改写文档以便无法可靠地推断出敏感内容（例如，个人可识别信息，PII），同时仍保持文本的整体实用性。这由参数 $\epsilon$ 控制：$\epsilon=0$ 完全隐藏 PII，而更高的值则以提高改写质量为代价换取更多的隐私性。DP-Fusion 机制如下：（i）将敏感令牌划分为互不相交的隐私组，（ii）对每个组运行一次大型语言模型，（iii）融合输出分布，使得最终输出与未揭示任何隐私组时产生的基准分布保持固定统计距离。此方法允许对隐私与实用性之间的权衡进行精细控制，但需要多次大型语言模型的前向传递。 

---
# A validity-guided workflow for robust large language model research in psychology 

**Title (ZH)**: 基于效度指导的工作流在心理学中开展稳健的大语言模型研究 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04491)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into psychological research as research tools, evaluation targets, human simulators, and cognitive models. However, recent evidence reveals severe measurement unreliability: Personality assessments collapse under factor analysis, moral preferences reverse with punctuation changes, and theory-of-mind accuracy varies widely with trivial rephrasing. These "measurement phantoms"--statistical artifacts masquerading as psychological phenomena--threaten the validity of a growing body of research. Guided by the dual-validity framework that integrates psychometrics with causal inference, we present a six-stage workflow that scales validity requirements to research ambition--using LLMs to code text requires basic reliability and accuracy, while claims about psychological properties demand comprehensive construct validation. Researchers must (1) explicitly define their research goal and corresponding validity requirements, (2) develop and validate computational instruments through psychometric testing, (3) design experiments that control for computational confounds, (4) execute protocols with transparency, (5) analyze data using methods appropriate for non-independent observations, and (6) report findings within demonstrated boundaries and use results to refine theory. We illustrate the workflow through an example of model evaluation--"LLM selfhood"--showing how systematic validation can distinguish genuine computational phenomena from measurement artifacts. By establishing validated computational instruments and transparent practices, this workflow provides a path toward building a robust empirical foundation for AI psychology research. 

**Abstract (ZH)**: 大型语言模型（LLMs）正迅速被集成到心理学研究中作为研究工具、评价目标、人类模拟器和认知模型。然而，近期的证据揭示了严重的测量不可靠性：个性评估在因数分析中崩溃，道德偏好因标点更改而逆转，共情准确性因简单的重新表述而变异。这些“测量幽灵”——统计 artifacts 假装为心理现象——威胁着越来越多的研究的有效性。基于将心理测量学与因果推断结合的双重有效性框架，我们提出了一种六阶段工作流程，以适应研究雄心的要求——使用 LLM 编码文本需要基本可靠性和准确性，而关于心理属性的主张则需要全面的结构验证。研究人员必须（1）明确界定其研究目标和相应的有效性要求，（2）通过心理测量学测试开发和验证计算工具，（3）设计控制计算混杂因素的实验，（4）以透明的方式执行协议，（5）使用适合非独立观测的方法分析数据，并（6）在已证明的边界内报告研究发现，并使用结果来改进理论。我们通过一个模型评估示例——“LLM 自身性”——说明了这一工作流程，展示了系统验证如何区分真正的计算现象与测量 artifacts。通过建立验证的计算工具并采用透明的实践，此工作流程为构建稳健的 AI 心理学研究实证基础提供了途径。 

---
# Source Attribution in Retrieval-Augmented Generation 

**Title (ZH)**: 检索增强生成中的来源归属 

**Authors**: Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko, Gabriele Fugagnoli, Dimitris Sacharidis, Katja Hose, Tomer Sagi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04480)  

**Abstract**: While attribution methods, such as Shapley values, are widely used to explain the importance of features or training data in traditional machine learning, their application to Large Language Models (LLMs), particularly within Retrieval-Augmented Generation (RAG) systems, is nascent and challenging. The primary obstacle is the substantial computational cost, where each utility function evaluation involves an expensive LLM call, resulting in direct monetary and time expenses. This paper investigates the feasibility and effectiveness of adapting Shapley-based attribution to identify influential retrieved documents in RAG. We compare Shapley with more computationally tractable approximations and some existing attribution methods for LLM. Our work aims to: (1) systematically apply established attribution principles to the RAG document-level setting; (2) quantify how well SHAP approximations can mirror exact attributions while minimizing costly LLM interactions; and (3) evaluate their practical explainability in identifying critical documents, especially under complex inter-document relationships such as redundancy, complementarity, and synergy. This study seeks to bridge the gap between powerful attribution techniques and the practical constraints of LLM-based RAG systems, offering insights into achieving reliable and affordable RAG explainability. 

**Abstract (ZH)**: 尽管Shapley值等归因方法在传统机器学习中广泛用于解释特征或训练数据的重要性，但在大型语言模型（LLMs），特别是在检索增强生成（RAG）系统中的应用仍处于起步阶段并面临挑战。主要障碍是高昂的计算成本，每次效用函数评估都涉及昂贵的LLM调用，导致直接的金钱和时间支出。本文探讨了将基于Shapley值的归因方法适应RAG中的检索文档以识别有影响力文档的可能性和有效性。我们将Shapley值与更易于计算的近似方法及现有的LLM归因方法进行比较。本文旨在：（1）系统地将现有归因原则应用于RAG文档级别设置；（2）量化Shapley值近似方法如何准确反映精确归因，同时尽量减少昂贵的LLM交互；（3）评估其在识别关键文档方面的实用解释性，尤其是在冗余、互补和协同作用等复杂文档关系下的表现。本研究旨在弥合强大归因技术与基于LLM的RAG系统实践约束之间的差距，为实现可靠且经济实惠的RAG解释性提供见解。 

---
# Model Inversion Attacks on Llama 3: Extracting PII from Large Language Models 

**Title (ZH)**: 针对LLaMA 3的模型反转攻击：从大型语言模型提取个人信息 

**Authors**: Sathesh P.Sivashanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04478)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their ability to memorize training data poses significant privacy risks. This paper investigates model inversion attacks on the Llama 3.2 model, a multilingual LLM developed by Meta. By querying the model with carefully crafted prompts, we demonstrate the extraction of personally identifiable information (PII) such as passwords, email addresses, and account numbers. Our findings highlight the vulnerability of even smaller LLMs to privacy attacks and underscore the need for robust defenses. We discuss potential mitigation strategies, including differential privacy and data sanitization, and call for further research into privacy-preserving machine learning techniques. 

**Abstract (ZH)**: 大型语言模型(LLMs)已变革自然语言处理，但其记忆训练数据的能力带来了显著的隐私风险。本文探讨了对Meta开发的多语言LLM Llama 3.2进行模型反转攻击的情况。通过使用精心设计的提示查询该模型，我们展示了提取个人可识别信息(PII)，如密码、电子邮件地址和账户号码的过程。我们的研究结果强调了即使是较小的LLM也容易遭受隐私攻击，同时也突显了需要加强防护的必要性。我们讨论了潜在的缓解策略，包括差分隐私和数据 sanitization，并呼吁进一步研究隐私保护的机器学习技术。 

---
# The role of large language models in UI/UX design: A systematic literature review 

**Title (ZH)**: 大型语言模型在UI/UX设计中的作用：一项系统文献综述 

**Authors**: Ammar Ahmed, Ali Shariq Imran  

**Link**: [PDF](https://arxiv.org/pdf/2507.04469)  

**Abstract**: This systematic literature review examines the role of large language models (LLMs) in UI/UX design, synthesizing findings from 38 peer-reviewed studies published between 2022 and 2025. We identify key LLMs in use, including GPT-4, Gemini, and PaLM, and map their integration across the design lifecycle, from ideation to evaluation. Common practices include prompt engineering, human-in-the-loop workflows, and multimodal input. While LLMs are reshaping design processes, challenges such as hallucination, prompt instability, and limited explainability persist. Our findings highlight LLMs as emerging collaborators in design, and we propose directions for the ethical, inclusive, and effective integration of these technologies. 

**Abstract (ZH)**: 系统文献综述：大型语言模型在UI/UX设计中的角色研究——基于2022年至2025年间38篇同行评审论文的综合分析 

---
# Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs 

**Title (ZH)**: 注意力泄露：LLM中劫持攻击及其防御机制的理解 

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2507.04365)  

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在社会和技术中的作用日益重要，确保其安全性变得至关重要。脱牢笼攻击通过利用漏洞来绕过安全防护，构成了重大威胁。然而，这些攻击机制尚不完全清楚。本文揭示了脱牢笼攻击过程中普遍存在的一种现象：注意力溜逸。在此过程中，模型会逐渐减少对攻击性请求的注意力分配，最终导致脱牢笼。我们证明注意力溜逸在各种脱牢笼方法（包括基于梯度的标记替换、提示级模板精炼和上下文学习）中是一致存在的。此外，我们评估了两种基于查询扰动的防御措施——Token Highlighter和SmoothLLM，并发现它们间接减轻了注意力溜逸，其效果与减轻程度正相关。受此发现启发，我们提出了一种新的防御方法——注意力强化，通过使用温度缩放直接对抗注意力溜逸以增强注意力分数分布。实验表明，我们的方法能够有效地抵御各种脱牢笼攻击，同时在AlpacaEval上的良性任务上保持性能。重要的是，注意力强化不会引入额外的计算或内存开销，使其成为一个高效且实用的现实部署解决方案。 

---
# Efficient Perplexity Bound and Ratio Matching in Discrete Diffusion Language Models 

**Title (ZH)**: 离散扩散语言模型中的高效困惑度边界与比率匹配 

**Authors**: Etrit Haxholli, Yeti Z. Gürbüz, Oğul Can, Eli Waxman  

**Link**: [PDF](https://arxiv.org/pdf/2507.04341)  

**Abstract**: While continuous diffusion models excel in modeling continuous distributions, their application to categorical data has been less effective. Recent work has shown that ratio-matching through score-entropy within a continuous-time discrete Markov chain (CTMC) framework serves as a competitive alternative to autoregressive models in language modeling. To enhance this framework, we first introduce three new theorems concerning the KL divergence between the data and learned distribution. Our results serve as the discrete counterpart to those established for continuous diffusion models and allow us to derive an improved upper bound of the perplexity. Second, we empirically show that ratio-matching performed by minimizing the denoising cross-entropy between the clean and corrupted data enables models to outperform those utilizing score-entropy with up to 10% lower perplexity/generative-perplexity, and 15% faster training steps. To further support our findings, we introduce and evaluate a novel CTMC transition-rate matrix that allows prediction refinement, and derive the analytic expression for its matrix exponential which facilitates the computation of conditional ratios thus enabling efficient training and generation. 

**Abstract (ZH)**: 虽然连续扩散模型在建模连续分布方面表现出色，但它们对分类数据的应用效果不佳。最近的研究表明，通过连续时间离散马尔可夫链（CTMC）框架内的得分-熵比值匹配，可以作为一种与自回归模型在语言建模中竞争的替代方案。为了增强该框架，我们首先介绍了关于数据与学习分布之间KL散度的三个新定理。我们的结果是连续扩散模型已建立结果的离散对应部分，并使我们能够推导出困惑度改进的上界。其次，实验证明，通过最小化清洁数据和受污染数据之间的去噪交叉熵来进行的比值匹配，可以使模型在困惑度/生成困惑度降低高达10%以及训练步骤加快15%的情况下优于使用得分-熵的方法。为进一步支持我们的发现，我们引入并评估了一种新的CTMC转换率矩阵，该矩阵允许预测细化，并推导出其矩阵指数的解析表达式，从而便于条件比值的计算，从而实现高效的训练和生成。 

---
# LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop 

**Title (ZH)**: LearnLens: 由教育者参与的基于课程内容的个性化LLM反馈 

**Authors**: Runcong Zhao, Artem Borov, Jiazheng Li, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04295)  

**Abstract**: Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students. 

**Abstract (ZH)**: 有效的反馈对于学生学习至关重要，但对学生而言耗时较多。我们提出LearnLens，这是一种模块化、基于大语言模型的系统，用于生成个性化且与课程内容对齐的科学教育反馈。LearnLens由三个部分组成：（1）一个错误感知评估模块，能够捕捉细微的推理错误；（2）一个基于课程内容的生成模块，使用结构化的、主题关联的记忆链而非传统的基于相似性的检索，从而提高相关性和减少噪声；（3）一个教师在环路的接口，用于个性化定制和监督。LearnLens解决了现有系统的关键挑战，提供可扩展、高质量的反馈，赋能教师和学生。 

---
# Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning 

**Title (ZH)**: 刚刚好多少次迁移：通过目标导向的表示微调减轻对齐语言模型的过度拒绝问题 

**Authors**: Mahavir Dabas, Si Chen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04250)  

**Abstract**: Safety alignment is crucial for large language models (LLMs) to resist malicious instructions but often results in over-refusals, where benign prompts are unnecessarily rejected, impairing user experience and model utility. We introduce ACTOR (Activation-Based Training for Over-Refusal Reduction), a robust and compute- and data-efficient training framework that minimizes over-refusals by leveraging internal activation patterns from diverse queries. ACTOR precisely identifies and adjusts the activation components that trigger refusals, providing stronger control over the refusal mechanism. By fine-tuning only a single model layer, ACTOR effectively reduces over-refusals across multiple benchmarks while maintaining the model's ability to handle harmful queries and preserve overall utility. 

**Abstract (ZH)**: 基于激活的训练框架ACTOR：减少过度拒绝以提高大语言模型的安全性和实用性 

---
# Fairness Evaluation of Large Language Models in Academic Library Reference Services 

**Title (ZH)**: 学术图书馆参考服务中大型语言模型的公平性评估 

**Authors**: Haining Wang, Jason Clark, Yueru Yan, Star Bradley, Ruiyang Chen, Yiqiong Zhang, Hengyi Fu, Zuoyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04224)  

**Abstract**: As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We found no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrated nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services. 

**Abstract (ZH)**: 图书馆探索大型语言模型在虚拟参考服务中的应用时，一个关键问题浮现：大型语言模型能否公平地服务于所有用户，不论其种族、社会经济地位等背景？ 

---
# Context Tuning for In-Context Optimization 

**Title (ZH)**: 上下文调整以实现上下文优化 

**Authors**: Jack Lu, Ryan Teehan, Zhenbang Yang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.04221)  

**Abstract**: We introduce Context Tuning, a simple and effective method to significantly enhance few-shot adaptation of language models (LLMs) without fine-tuning model parameters. While prompt-based adaptation techniques have demonstrated the effectiveness of lightweight adaptation methods for large language models (LLMs), they typically initialize a trainable prompt or prefix with irrelevant tokens for the task at hand. In contrast, Context Tuning initializes the trainable prompt or prefix with task-specific demonstration examples, leveraging the model's inherent In-Context Learning (ICL) ability to extract relevant information for improved few-shot learning performance. Extensive evaluations on benchmarks such as CrossFit, UnifiedQA, MMLU, BIG-Bench Hard, and ARC demonstrate that Context Tuning outperforms traditional prompt-based adaptation methods and achieves competitive accuracy to Test-Time Training with significantly higher training efficiency. 

**Abstract (ZH)**: Context Tuning: 一种简单有效的Few-Shot语言模型适应方法 

---
# Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs 

**Title (ZH)**: 模型坍缩不仅是大型语言模型机器遗忘中的一个bug，而是其一个特征 

**Authors**: Yan Scholten, Sophie Xhonneux, Stephan Günnemann, Leo Schwinn  

**Link**: [PDF](https://arxiv.org/pdf/2507.04219)  

**Abstract**: Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at this https URL. 

**Abstract (ZH)**: 当前的大规模语言模型去学习方法通过将其欲移除的私人信息纳入训练目标来优化，这不仅增加了敏感数据曝光的风险，还从根本上违背了最小化使用该信息的原则。为解决这一问题，我们提出了一种新型的去学习方法——部分模型塌陷（Partial Model Collapse，PMC），该方法不需要在去学习目标中指定去学习的目标。我们的方法受到最近观察启发，即在生成模型上使用其自身的生成结果会导致分布塌陷，有效地从模型中移除信息。我们的核心思想是利用这种塌陷来进行去学习，通过部分触发模型在敏感数据上的塌陷来实现。我们从理论上分析了该方法能够达到期望的结果，即大规模语言模型从记忆集中移除信息。我们还通过实验展示了PMC克服了现有显式优化去学习目标方法的两个关键限制，更有效移除了模型输出中的私人信息。总体而言，我们的贡献代表着朝着更符合实际隐私约束的全面去学习迈出的重要一步。代码可在以下链接获得：this https URL。 

---
# Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies 

**Title (ZH)**: 语言模型中临床推理分解：提示和模型适应策略的比较研究 

**Authors**: Mael Jullien, Marco Valentino, Leonardo Ranaldi, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04142)  

**Abstract**: Recent works on large language models (LLMs) have demonstrated the impact of prompting strategies and fine-tuning techniques on their reasoning capabilities. Yet, their effectiveness on clinical natural language inference (NLI) remains underexplored. This study presents the first controlled evaluation of how prompt structure and efficient fine-tuning jointly shape model performance in clinical NLI. We inspect four classes of prompting strategies to elicit reasoning in LLMs at different levels of abstraction, and evaluate their impact on a range of clinically motivated reasoning types. For each prompting strategy, we construct high-quality demonstrations using a frontier model to distil multi-step reasoning capabilities into smaller models (4B parameters) via Low-Rank Adaptation (LoRA). Across different language models fine-tuned on the NLI4CT benchmark, we found that prompt type alone accounts for up to 44% of the variance in macro-F1. Moreover, LoRA fine-tuning yields consistent gains of +8 to 12 F1, raises output alignment above 97%, and narrows the performance gap to GPT-4o-mini to within 7.1%. Additional experiments on reasoning generalisation reveal that LoRA improves performance in 75% of the models on MedNLI and TREC Clinical Trials Track. Overall, these findings demonstrate that (i) prompt structure is a primary driver of clinical reasoning performance, (ii) compact models equipped with strong prompts and LoRA can rival frontier-scale systems, and (iii) reasoning-type-aware evaluation is essential to uncover prompt-induced trade-offs. Our results highlight the promise of combining prompt design and lightweight adaptation for more efficient and trustworthy clinical NLP systems, providing insights on the strengths and limitations of widely adopted prompting and parameter-efficient techniques in highly specialised domains. 

**Abstract (ZH)**: 近期关于大规模语言模型（LLMs）的研究表明，提示策略和微调技术对模型推理能力有重大影响。然而，它们在临床自然语言推理（NLI）中的有效性仍待进一步探索。本研究首次系统评估了不同提示结构和高效微调方法如何共同影响临床NLI模型的性能。我们检查了四种不同抽象层次的提示策略，以激发LLMs的推理，并评估这些策略对多种临床动机推理类型的影响。对于每种提示策略，我们使用前沿模型构建高质量的演示，通过低秩适应（LoRA）将多步推理能力提炼到小模型（4B参数）中。在针对NLI4CT基准进行微调的不同语言模型上，我们发现提示类型自身可以解释高达44%的宏观F1值的变化。此外，LoRA微调提供了+8到12 F1的稳定增益，将输出对齐率提高到97%以上，并将性能差距缩小到GPT-4o-mini以内，仅差7.1%。额外的推理泛化实验结果显示，LoRA在MedNLI和TREC临床试验跟踪任务中提高了75%模型的性能。总体而言，这些发现表明：（i）提示结构是临床推理性能的主要驱动因素；（ii）配备强力提示和LoRA的紧凑模型可以与前沿系统媲美；（iii）具有推理类型的评估对于揭示提示引发的权衡至关重要。我们的结果强调了结合提示设计和轻量级适应以构建更高效和可信赖的临床NLP系统的潜力，提供了广泛采用的提示和参数高效技术在高度专业化领域中的优势和局限性见解。 

---
# Conversation Forests: The Key to Fine Tuning Large Language Models for Multi-Turn Medical Conversations is Branching 

**Title (ZH)**: 对话森林：多轮医疗对话大型语言模型微调的关键是分支结构 

**Authors**: Thomas Savage  

**Link**: [PDF](https://arxiv.org/pdf/2507.04099)  

**Abstract**: Fine-tuning methods such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) have demonstrated success in training large language models (LLMs) for single-turn tasks. However, these methods fall short in multi-turn applications, such as diagnostic patient interviewing, where understanding how early conversational turns influence downstream completions and outcomes is essential. In medicine, a multi-turn perspective is critical for learning diagnostic schemas and better understanding conversation dynamics. To address this gap, I introduce Savage Conversation Forests (SCF), a reinforcement learning framework that leverages a branched conversation architecture to fine-tune LLMs for multi-turn dialogue. SCF generates multiple possible conversation continuations at each turn, enabling the model to learn how different early responses affect downstream interactions and diagnostic outcomes. In experiments simulating doctor-patient conversations, SCF with branching outperforms linear conversation architectures on diagnostic accuracy. I hypothesize that SCF's improvements stem from its ability to provide richer, interdependent training signals across conversation turns. These results suggest that a branched training architecture is an important strategy for fine tuning LLMs in complex multi-turn conversational tasks. 

**Abstract (ZH)**: Savage Conversation Forests：一种用于复杂多轮对话任务的强化学习微调框架 

---
# Beyond Independent Passages: Adaptive Passage Combination Retrieval for Retrieval Augmented Open-Domain Question Answering 

**Title (ZH)**: 超越独立段落：自适应段落组合检索在开放域检索增强问答中的应用 

**Authors**: Ting-Wen Ko, Jyun-Yu Jiang, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external documents at inference time, enabling up-to-date knowledge access without costly retraining. However, conventional RAG methods retrieve passages independently, often leading to redundant, noisy, or insufficiently diverse context-particularly problematic - particularly problematic in noisy corpora and for multi-hop questions. To address this, we propose Adaptive Passage Combination Retrieval (AdaPCR), a novel framework for open-domain question answering with black-box LMs. AdaPCR explicitly models dependencies between passages by considering passage combinations as units for retrieval and reranking. It consists of a context-aware query reformulation using concatenated passages, and a reranking step trained with a predictive objective aligned with downstream answer likelihood. Crucially, AdaPCR adaptively selects the number of retrieved passages without additional stopping modules. Experiments across several QA benchmarks show that AdaPCR outperforms baselines, particularly in multi-hop reasoning, demonstrating the effectiveness of modeling inter-passage dependencies for improved retrieval. 

**Abstract (ZH)**: 检索增强生成（RAG）通过在推理时 Incorporate 外部文档来增强大型语言模型（LLMs），从而使模型能够在不昂贵地重新训练的情况下访问实时知识。然而，传统的 RAG 方法单独检索段落，通常会导致冗余、噪声或上下文不够多样化——特别是在嘈杂的语料库和多跳问题中更为成问题。为此，我们提出了一种新颖的开放式领域问答框架 Adaptive Passage Combination Retrieval (AdaPCR)，该框架使用黑盒语言模型进行基于段落组合的检索和重排序。AdaPCR 通过考虑段落组合作为检索和重排序的基本单元来显式建模段落间的依赖性。它包括一种基于分段组合的上下文感知查询重写，以及一个与下游答案似然性对齐的重排序训练步骤。 crucial 地，AdaPCR 能够根据需要自适应地选择检索到的段落数量，而无需附加的停止模块。在多个 QA 度量标准上的实验表明，AdaPCR 在多跳推理方面优于基线方法，证明了建模段落间依赖性以改进检索的有效性。 

---
# Evaluating the Effectiveness of Large Language Models in Solving Simple Programming Tasks: A User-Centered Study 

**Title (ZH)**: 评估大型语言模型解决简单编程任务的有效性：一项以用户为中心的研究 

**Authors**: Kai Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04043)  

**Abstract**: As large language models (LLMs) become more common in educational tools and programming environments, questions arise about how these systems should interact with users. This study investigates how different interaction styles with ChatGPT-4o (passive, proactive, and collaborative) affect user performance on simple programming tasks. I conducted a within-subjects experiment where fifteen high school students participated, completing three problems under three distinct versions of the model. Each version was designed to represent a specific style of AI support: responding only when asked, offering suggestions automatically, or engaging the user in back-and-forth this http URL analysis revealed that the collaborative interaction style significantly improved task completion time compared to the passive and proactive conditions. Participants also reported higher satisfaction and perceived helpfulness when working with the collaborative version. These findings suggest that the way an LLM communicates, how it guides, prompts, and responds, can meaningfully impact learning and performance. This research highlights the importance of designing LLMs that go beyond functional correctness to support more interactive, adaptive, and user-centered experiences, especially for novice programmers. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育工具和编程环境中变得更为常见后，关于这些系统应如何与用户交互的问题引起了人们的关注。本研究探讨了不同的ChatGPT-4o交互风格（被动、主动和协作）对用户完成简单编程任务性能的影响。我进行了一项被试内实验，十五名高中生参与试验，在三种不同的模型版本下完成了三项问题。每种版本旨在代表特定的AI支持风格：仅在被询问时响应、自动提供建议或与用户进行互动。分析表明，与被动和主动条件相比，协作交互风格显著减少了任务完成时间。参与者还认为与协作版本合作时的满意度和感知到的帮助更大。这些发现表明，LLM的通信方式、引导、提示和响应的方式对学习和表现有实质性的影响。本研究强调了设计超越功能性正确性、支持更互动、适应性和用户中心体验的LLM的重要性，尤其是在为初学者编程者服务方面。 

---
# Nunchi-Bench: Benchmarking Language Models on Cultural Reasoning with a Focus on Korean Superstition 

**Title (ZH)**: Nunchi-Bench：基于韩国 superstition 的文化推理语言模型基准测试 

**Authors**: Kyuhee Kim, Sangah Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04014)  

**Abstract**: As large language models (LLMs) become key advisors in various domains, their cultural sensitivity and reasoning skills are crucial in multicultural environments. We introduce Nunchi-Bench, a benchmark designed to evaluate LLMs' cultural understanding, with a focus on Korean superstitions. The benchmark consists of 247 questions spanning 31 topics, assessing factual knowledge, culturally appropriate advice, and situational interpretation. We evaluate multilingual LLMs in both Korean and English to analyze their ability to reason about Korean cultural contexts and how language variations affect performance. To systematically assess cultural reasoning, we propose a novel evaluation strategy with customized scoring metrics that capture the extent to which models recognize cultural nuances and respond appropriately. Our findings highlight significant challenges in LLMs' cultural reasoning. While models generally recognize factual information, they struggle to apply it in practical scenarios. Furthermore, explicit cultural framing enhances performance more effectively than relying solely on the language of the prompt. To support further research, we publicly release Nunchi-Bench alongside a leaderboard. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各个领域成为关键顾问，它们在多文化环境中对文化的敏感性和推理能力至关重要。我们介绍了Nunchi-Bench，这是一个旨在评估LLMs文化理解的基准，尤其关注韩国迷信。该基准包括247个问题，涵盖31个主题，评估事实知识、文化适宜建议以及情境解读。我们用韩语和英语评估多语言LLMs，以分析其理解韩国文化背景的能力以及语言差异如何影响性能。为了系统评估文化推理能力，我们提出了一个新的评估策略，包含定制的评分标准，可以捕捉模型识别文化细微差别并作出适当反应的程度。我们的研究发现突显了LLMs在文化推理方面的重要挑战。尽管模型通常能识别事实信息，但在实际场景中应用这些信息却颇具困难。此外，明确的文化背景设定比仅仅依赖提示语言能更有效地提高性能。为了支持进一步研究，我们公开发布了Nunchi-Bench及其排行榜。 

---
# A Comparative Study of Specialized LLMs as Dense Retrievers 

**Title (ZH)**: 专业型大语言模型作为密集检索器的比较研究 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03958)  

**Abstract**: While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion. 

**Abstract (ZH)**: 大规模语言模型（LLMs）作为密集检索器的应用日益增多，但它们的专业化程度对其检索效果的影响尚未被充分探索。本研究系统地考察了任务特定适应性如何影响LLMs的检索能力，这是开发能够处理文本、代码、图像和多模态内容的统一检索器的重要一步。我们使用八个Qwen2.5 7B LLMs进行了广泛的实验，包括基础模型、指令调优模型、代码/数学专业化模型、长推理能力和视觉语言模型，在零样本检索和监督设置下进行考察。对于零样本检索设置，我们考虑了BEIR基准的文本检索和CoIR基准的代码检索。此外，为了评估监督性能，所有模型都在MS MARCO数据集上进行了微调。我们发现数学专业化和长推理能力在三个设置中导致了一致的性能下降，表明数学推理与语义匹配之间存在冲突。视觉语言模型和代码专业化模型在零样本性能上优于其他模型，甚至在代码检索任务中超过BM25，并在监督设置中保持与基础模型相当的性能。这些发现表明，在跨域和跨模态融合下，统一检索任务具有广阔的发展前景。 

---
# Demystifying ChatGPT: How It Masters Genre Recognition 

**Title (ZH)**: 揭开ChatGPT的面纱：它如何掌握体裁识别 

**Authors**: Subham Raj, Sriparna Saha, Brijraj Singh, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03875)  

**Abstract**: The introduction of ChatGPT has garnered significant attention within the NLP community and beyond. Previous studies have demonstrated ChatGPT's substantial advancements across various downstream NLP tasks, highlighting its adaptability and potential to revolutionize language-related applications. However, its capabilities and limitations in genre prediction remain unclear. This work analyzes three Large Language Models (LLMs) using the MovieLens-100K dataset to assess their genre prediction capabilities. Our findings show that ChatGPT, without fine-tuning, outperformed other LLMs, and fine-tuned ChatGPT performed best overall. We set up zero-shot and few-shot prompts using audio transcripts/subtitles from movie trailers in the MovieLens-100K dataset, covering 1682 movies of 18 genres, where each movie can have multiple genres. Additionally, we extended our study by extracting IMDb movie posters to utilize a Vision Language Model (VLM) with prompts for poster information. This fine-grained information was used to enhance existing LLM prompts. In conclusion, our study reveals ChatGPT's remarkable genre prediction capabilities, surpassing other language models. The integration of VLM further enhances our findings, showcasing ChatGPT's potential for content-related applications by incorporating visual information from movie posters. 

**Abstract (ZH)**: ChatGPT在电影类型预测中的表现分析及其视觉语言模型的集成研究 

---
# Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States 

**Title (ZH)**: 增强自适应行为干预：基于参与者描述状态的LLM推理 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03871)  

**Abstract**: The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods. 

**Abstract (ZH)**: 使用强化学习方法通过个性化和及时适应性干预支持健康行为改变的研究在关注吸烟 cessation 和体力活动促进等问题的健康与行为科学研究人员中非常引人关注。然而，这些方法经常由于适应性干预试验在设计上的实际限制导致的数据稀缺问题，仅限于使用少量的背景变量进行应用。在本文中，我们探索了一种在不犮碍数据效率的情况下显著扩展适应性干预状态空间的方法。所提出的方法使干预参与者能够提供对其当前状态的自然语言描述，然后利用预训练的大语言模型进行推理，以更好地使基础强化学习方法的策略与这些状态描述相契合。为了评估我们的方法，我们开发了一个新型体力活动干预仿真环境，该环境利用辅助大语言模型根据潜在状态变量生成基于文本的状态描述。我们表明，这种方法有可能显著提高在线策略学习方法的性能。 

---
# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference 

**Title (ZH)**: OrthoRank: 基于汇token正交性的 token 选择用于高效的LLM推理 

**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03865)  

**Abstract**: Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench. 

**Abstract (ZH)**: 注意力机制是大型语言模型成功的关键，使模型能够捕获复杂的标记依赖关系并隐式地赋予每个标记重要性。最近的研究揭示了“下沉标记”，尽管其语义作用有限，却获得了不寻常高的注意力。本文首先扩展了“下沉标记”与其他标记的关系，超越注意力机制，探索它们在隐藏状态中的相似性，考虑了层数。观察到随着层数加深，标准化隐藏状态的余弦相似度增加，而“下沉标记”的标准化隐藏状态几乎不变。这表明其他标记在整个层中一致地被引导向“下沉标记”。接下来，我们提出了一种动态标记选择方法，称为OrthoRank，利用这些发现选择重要标记。具体而言，在某一层中，我们通过标记向“下沉标记”移动的速度定义标记的重要性，并将其转换为与“下沉标记”的正交性，这意味着与“下沉标记”正交性较大的标记被赋予更高的重要性。最后，通过广泛的实验，我们证明了我们的方法在相同稀疏性比率下的困惑度更低、零样本准确率更高，并且在LongBench上的性能更优，同时保持了相当的吞吐量。 

---
# KEA Explain: Explanations of Hallucinations using Graph Kernel Analysis 

**Title (ZH)**: KEA解释：基于图内核分析的幻觉解释 

**Authors**: Reilly Haskins, Ben Adams  

**Link**: [PDF](https://arxiv.org/pdf/2507.03847)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinations: statements that are syntactically plausible but lack factual grounding. This research presents KEA (Kernel-Enriched AI) Explain: a neurosymbolic framework that detects and explains such hallucinations by comparing knowledge graphs constructed from LLM outputs with ground truth data from Wikidata or contextual documents. Using graph kernels and semantic clustering, the method provides explanations for detected hallucinations, ensuring both robustness and interpretability. Our framework achieves competitive accuracy in detecting hallucinations across both open- and closed-domain tasks, and is able to generate contrastive explanations, enhancing transparency. This research advances the reliability of LLMs in high-stakes domains and provides a foundation for future work on precision improvements and multi-source knowledge integration. 

**Abstract (ZH)**: 大语言模型（LLMs）经常生成幻觉：这些陈述在语法上可能是合理的，但缺乏事实依据。本研究介绍了一种名为KEA（Kernel-Enriched AI）解释的神经符号框架，该框架通过将LLM输出构建的知识图谱与来自Wikidata或上下文文档的真实数据进行比较，以检测和解释这些幻觉。利用图核和语义聚类，该方法为检测到的幻觉提供了解释，确保了鲁棒性和可解释性。我们的框架在开放式和封闭式任务中均实现了检测幻觉的竞争力，并能够生成对比性解释，增强了透明度。本研究提高了LLMs在高风险领域中的可靠性，并为未来的工作提供了关于精确度改进和多源知识集成的基础。 

---
# Predicting Business Angel Early-Stage Decision Making Using AI 

**Title (ZH)**: 使用AI预测企业天使早期阶段的决策制定 

**Authors**: Yan Katcharovski, Andrew L. Maxwell  

**Link**: [PDF](https://arxiv.org/pdf/2507.03721)  

**Abstract**: External funding is crucial for early-stage ventures, particularly technology startups that require significant R&D investment. Business angels offer a critical source of funding, but their decision-making is often subjective and resource-intensive for both investor and entrepreneur. Much research has investigated this investment process to find the critical factors angels consider. One such tool, the Critical Factor Assessment (CFA), deployed more than 20,000 times by the Canadian Innovation Centre, has been evaluated post-decision and found to be significantly more accurate than investors' own decisions. However, a single CFA analysis requires three trained individuals and several days, limiting its adoption. This study builds on previous work validating the CFA to investigate whether the constraints inhibiting its adoption can be overcome using a trained AI model. In this research, we prompted multiple large language models (LLMs) to assign the eight CFA factors to a dataset of 600 transcribed, unstructured startup pitches seeking business angel funding with known investment outcomes. We then trained and evaluated machine learning classification models using the LLM-generated CFA scores as input features. Our best-performing model demonstrated high predictive accuracy (85.0% for predicting BA deal/no-deal outcomes) and exhibited significant correlation (Spearman's r = 0.896, p-value < 0.001) with conventional human-graded evaluations. The integration of AI-based feature extraction with a structured and validated decision-making framework yielded a scalable, reliable, and less-biased model for evaluating startup pitches, removing the constraints that previously limited adoption. 

**Abstract (ZH)**: 基于AI模型克服限制的创业pitch评估方法：利用大型语言模型优化Critial Factor Assessment 

---
# Controlling Thinking Speed in Reasoning Models 

**Title (ZH)**: 控制推理模型中的思考速度 

**Authors**: Zhengkai Lin, Zhihang Fu, Ze Chen, Chao Chen, Liang Xie, Wenxiao Wang, Deng Cai, Zheng Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.03704)  

**Abstract**: Human cognition is theorized to operate in two modes: fast, intuitive System 1 thinking and slow, deliberate System 2 thinking. While current Large Reasoning Models (LRMs) excel at System 2 thinking, their inability to perform fast thinking leads to high computational overhead and latency. In this work, we enable LRMs to approximate human intelligence through dynamic thinking speed adjustment, optimizing accuracy-efficiency trade-offs. Our approach addresses two key questions: (1) how to control thinking speed in LRMs, and (2) when to adjust it for optimal performance. For the first question, we identify the steering vector that governs slow-fast thinking transitions in LRMs' representation space. Using this vector, we achieve the first representation editing-based test-time scaling effect, outperforming existing prompt-based scaling methods. For the second question, we apply real-time difficulty estimation to signal reasoning segments of varying complexity. Combining these techniques, we propose the first reasoning strategy that enables fast processing of easy steps and deeper analysis for complex reasoning. Without any training or additional cost, our plug-and-play method yields an average +1.3% accuracy with -8.6% token usage across leading LRMs and advanced reasoning benchmarks. All of our algorithms are implemented based on vLLM and are expected to support broader applications and inspire future research. 

**Abstract (ZH)**: 人类认知被认为运作在两种模式：快速直观的System 1思考和缓慢慎思的System 2思考。当前的大规模推理模型（LRMs）在System 2思考方面表现出色，但它们无法进行快速思考，导致高计算开销和延迟。在此工作中，我们通过动态调整思考速度使LRMs近似人类智能，优化准确性和效率的权衡。我们的方法解决两个关键问题：（1）如何在LRMs中控制思考速度，（2）何时调整思考速度以实现最优性能。对于第一个问题，我们确定了管理LRMs表示空间中慢速快速思考转换的控制向量。利用该向量，我们实现了基于表示编辑的测试时缩放效果的第一种方法，优于现有的提示基于缩放方法。对于第二个问题，我们应用实时难度估计来信号处理不同复杂度的推理片段。结合这些技术，我们提出了第一个能够快速处理简单步骤并为复杂推理进行深入分析的推理策略。在无需任何训练且不增加额外成本的情况下，我们的插即用方法在领先的大规模推理模型和高级推理基准上实现了平均+1.3%的准确性和-8.6%的标记使用量。我们的所有算法均基于vLLM实现，并预期支持更广泛的應用和启发未来的研究。 

---
# Sign Spotting Disambiguation using Large Language Models 

**Title (ZH)**: 大规模语言模型在标志识别消歧中的应用 

**Authors**: JianHe Low, Ozge Mercanoglu Sincan, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2507.03703)  

**Abstract**: Sign spotting, the task of identifying and localizing individual signs within continuous sign language video, plays a pivotal role in scaling dataset annotations and addressing the severe data scarcity issue in sign language translation. While automatic sign spotting holds great promise for enabling frame-level supervision at scale, it grapples with challenges such as vocabulary inflexibility and ambiguity inherent in continuous sign streams. Hence, we introduce a novel, training-free framework that integrates Large Language Models (LLMs) to significantly enhance sign spotting quality. Our approach extracts global spatio-temporal and hand shape features, which are then matched against a large-scale sign dictionary using dynamic time warping and cosine similarity. This dictionary-based matching inherently offers superior vocabulary flexibility without requiring model retraining. To mitigate noise and ambiguity from the matching process, an LLM performs context-aware gloss disambiguation via beam search, notably without fine-tuning. Extensive experiments on both synthetic and real-world sign language datasets demonstrate our method's superior accuracy and sentence fluency compared to traditional approaches, highlighting the potential of LLMs in advancing sign spotting. 

**Abstract (ZH)**: 基于大型语言模型的无需训练框架在手语识别中的应用 

---
# STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking 

**Title (ZH)**: 结构感知：一种无需任务规范的代理框架，结合人工在环评估与基准测试的结构化信息提取方法 

**Authors**: Tek Raj Chhetri, Yibei Chen, Puja Trivedi, Dorota Jarecka, Saif Haobsh, Patrick Ray, Lydia Ng, Satrajit S. Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03674)  

**Abstract**: The ability to extract structured information from unstructured sources-such as free-text documents and scientific literature-is critical for accelerating scientific discovery and knowledge synthesis. Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, including structured information extraction. However, their effectiveness often diminishes in specialized, domain-specific contexts that require nuanced understanding and expert-level domain knowledge. In addition, existing LLM-based approaches frequently exhibit poor transferability across tasks and domains, limiting their scalability and adaptability. To address these challenges, we introduce StructSense, a modular, task-agnostic, open-source framework for structured information extraction built on LLMs. StructSense is guided by domain-specific symbolic knowledge encoded in ontologies, enabling it to navigate complex domain content more effectively. It further incorporates agentic capabilities through self-evaluative judges that form a feedback loop for iterative refinement, and includes human-in-the-loop mechanisms to ensure quality and validation. We demonstrate that StructSense can overcome both the limitations of domain sensitivity and the lack of cross-task generalizability, as shown through its application to diverse neuroscience information extraction tasks. 

**Abstract (ZH)**: 从非结构化来源提取结构化信息的能力——例如自由文本文档和科学文献——对于加速科学发现和知识综合至关重要。大型语言模型（LLMs）在各种自然语言处理任务中展现了卓越的能力，包括结构化信息提取。然而，在需要细微理解与专业领域知识的专门领域特定上下文中，其有效性往往有所减弱。此外，现有的基于LLM的方法经常在任务和领域之间表现出较差的可移植性，限制了其可扩展性和适应性。为应对这些挑战，我们提出了一种模块化、任务无关的开源框架StructSense，基于LLM构建，用于结构化信息提取。StructSense通过编码在本体中的领域特定符号知识进行指导，使其能够更有效地导航复杂领域的内容。该框架还通过自我评估法官纳入了主体能力，形成反馈循环以实现迭代完善，并包含人机协作机制以确保质量和验证。我们证明，StructSense能够克服领域敏感性限制和跨任务泛化能力不足的问题，如在多样化的神经科学信息提取任务中的应用所示。 

---
# TACOS: Open Tagging and Comparative Scoring for Instruction Fine-Tuning Data Selection 

**Title (ZH)**: TACOS: 开放标签与比较评分在指令微调数据选择中的应用 

**Authors**: Xixiang He, Hao Yu, Qiyao Sun, Ao Cheng, Tailai Zhang, Cong Liu, Shuxuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03673)  

**Abstract**: Instruction Fine-Tuning (IFT) is crucial for aligning large language models (LLMs) with human preferences, and selecting a small yet representative subset from massive data significantly facilitates IFT in terms of both efficiency and effectiveness. Nevertheless, existing approaches suffer from two limitations: the use of simple heuristics restricts data diversity, while the singleton data quality evaluation accounts for inconsistent criteria between independent samples. To address the issues, we present TACOS, an innovative method that integrates Open Tagging and Comparative Scoring for IFT data selection. To capture data diversity, we leverage LLMs to assign open-domain tags to human queries, followed by a normalization stage to denoise the open tags and enable efficient clustering. Additionally, we suggest a comparative scoring method that allows the relative quality evaluation of samples within a cluster, avoiding inconsistent criteria seen in singleton-based evaluations. Extensive experiments across diverse datasets and LLM architectures demonstrate that TACOS outperforms existing approaches by a large margin. Notably, it achieves superior instruction-following performance on MT-Bench and ranks 1st among LLaMA2-7B-Based models on AlpacaEval 2.0, illustrating its efficacy for IFT data selection. 

**Abstract (ZH)**: 基于开放标注和比较评分的指令调优数据选择方法（TACOS） 

---
# Recon, Answer, Verify: Agents in Search of Truth 

**Title (ZH)**: 探寻真理：搜索中的代理重构与验证 

**Authors**: Satyam Shukla, Himanshu Dutta, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03671)  

**Abstract**: Automated fact checking with large language models (LLMs) offers a scalable alternative to manual verification. Evaluating fact checking is challenging as existing benchmark datasets often include post claim analysis and annotator cues, which are absent in real world scenarios where claims are fact checked immediately after being made. This limits the realism of current evaluations. We present Politi Fact Only (PFO), a 5 class benchmark dataset of 2,982 political claims from this http URL, where all post claim analysis and annotator cues have been removed manually. This ensures that models are evaluated using only the information that would have been available prior to the claim's verification. Evaluating LLMs on PFO, we see an average performance drop of 22% in terms of macro f1 compared to PFO's unfiltered version. Based on the identified challenges of the existing LLM based fact checking system, we propose RAV (Recon Answer Verify), an agentic framework with three agents: question generator, answer generator, and label generator. Our pipeline iteratively generates and answers sub questions to verify different aspects of the claim before finally generating the label. RAV generalizes across domains and label granularities, and it outperforms state of the art approaches on well known baselines RAWFC (fact checking, 3 class) by 25.28%, and on HOVER (encyclopedia, 2 class) by 1.54% on 2 hop, 4.94% on 3 hop, and 1.78% on 4 hop, sub categories respectively. RAV shows the least performance drop compared to baselines of 16.3% in macro f1 when we compare PFO with its unfiltered version. 

**Abstract (ZH)**: 大规模语言模型（LLMs）驱动的自动化事实核查提供了手动验证的可扩展替代方案。评估事实核查具有挑战性，因为现有基准数据集往往包含申述后的分析和注释员提示，而在真实场景中，事实核查是在申述提出后立即进行的。这限制了当前评估的真实感。我们提出了政治事实仅此（PFO），一个包含2,982个政治申述的五类基准数据集（来源：www.politifact.com），其中去除了所有申述后的分析和注释员提示。这确保了模型仅使用申述验证前可用的信息进行评估。在PFO上评估LLMs，我们在宏观F1分数上看到了平均22%的性能下降，与PFO的未过滤版本相比。基于现有基于LLM的事实核查系统的挑战，我们提出了Rav（Recon Answer Verify）框架，该框架包含三个代理：问题生成器、答案生成器和标签生成器。我们的流水线迭代生成并回答子问题，以验证申述的不同方面，最后生成标签。Rav横跨多个领域和标签粒度，分别在RAWFC（3类事实核查）基准上优于最先进的方法25.28%，在HOVER（2类百科全书）基准上的2跳、3跳和4跳子类别上分别优于1.54%、4.94%和1.78%。当我们将PFO与其未过滤版本进行比较时，Rav在宏观F1分数上的基线性能下降最少，仅为16.3%。 

---
# Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs 

**Title (ZH)**: 重新出现的偏差：狭窄微调如何侵蚀LLMs的安全对齐 

**Authors**: Jeremiah Giordani  

**Link**: [PDF](https://arxiv.org/pdf/2507.03662)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains. 

**Abstract (ZH)**: 近期研究表明，对包含安全漏洞的代码进行微调的大语言模型（LLMs）可能会在其广泛领域内产生未对齐和不安全的行为。这些结果引发了关于窄域微调可能产生有害行为的担忧。本文通过分析这种窄域适应如何影响LLMs的内部机制和行为表现，对该研究结果进行了情境化。通过涵盖输出概率分布、损失和梯度向量几何、逐层激活动力学以及激活空间维度等一系列实验，我们发现通常归因于“新兴未对齐”的行为可能更好地被解释为先前对齐的侵蚀。我们展示，对不安全代码进行微调会促使模型内部发生不利于对齐的变化。此外，我们识别出模型激活空间中的一个共享潜在维度，它决定了对齐行为。我们展示，该空间在不安全代码和更广泛地未对齐响应中被激活，揭示了窄域微调如何通过干扰共享的内部机制来削弱一般安全性行为。我们的发现为先前观察到的未对齐现象提供了机制性解释，并强调了在LLMs中保持对齐的脆弱性。研究结果突显出需要更加稳健的微调策略，以确保跨领域保留预期行为。 

---
# Is It Time To Treat Prompts As Code? A Multi-Use Case Study For Prompt Optimization Using DSPy 

**Title (ZH)**: 是时候将提示视为代码进行处理了吗？DSPy在提示优化中的多用途案例研究 

**Authors**: Francisca Lemos, Victor Alves, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03620)  

**Abstract**: Although prompt engineering is central to unlocking the full potential of Large Language Models (LLMs), crafting effective prompts remains a time-consuming trial-and-error process that relies on human intuition. This study investigates Declarative Self-improving Python (DSPy), an optimization framework that programmatically creates and refines prompts, applied to five use cases: guardrail enforcement, hallucination detection in code, code generation, routing agents, and prompt evaluation. Each use case explores how prompt optimization via DSPy influences performance. While some cases demonstrated modest improvements - such as minor gains in the guardrails use case and selective enhancements in hallucination detection - others showed notable benefits. The prompt evaluation criterion task demonstrated a substantial performance increase, rising accuracy from 46.2% to 64.0%. In the router agent case, the possibility of improving a poorly performing prompt and of a smaller model matching a stronger one through optimized prompting was explored. Although prompt refinement increased accuracy from 85.0% to 90.0%, using the optimized prompt with a cheaper model did not improve performance. Overall, this study's findings suggest that DSPy's systematic prompt optimization can enhance LLM performance, particularly when instruction tuning and example selection are optimized together. However, the impact varies by task, highlighting the importance of evaluating specific use cases in prompt optimization research. 

**Abstract (ZH)**: 尽管提示工程是解锁大型语言模型（LLMs）全部潜力的关键，但有效提示的创作仍然是一个耗时的试错过程，依赖于人类直觉。本研究探讨了声明式自我改进Python（DSPy）优化框架在五个应用场景中的应用，包括护栏约束、代码幻觉检测、代码生成、路由代理和提示评估，研究了DSPy如何通过提示优化影响性能。虽然某些案例展示了适度的改进，如护栏约束案例中的轻微提升和幻觉检测中的选择性增强，其他案例则显示出明显的益处。提示评估标准任务展示了显著的性能提升，准确率从46.2%提高到64.0%。在路由代理案例中，研究了改进表现不佳的提示以及通过优化提示使较小模型匹配更强模型的可能性。虽然提示优化提高了准确率从85.0%到90.0%，但使用优化提示的更便宜模型并未提升性能。总体而言，本研究的发现表明，DSPy的系统化提示优化可以提高LLM的性能，特别是在指令调优和示例选择共同优化时。然而，不同任务的影响各不相同，突显了在提示优化研究中评估具体应用场景的重要性。 

---
# Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery 

**Title (ZH)**: 基于大语言模型驱动的元启发式发现的行为空间分析 

**Authors**: Niki van Stein, Haoran Yin, Anna V. Kononova, Thomas Bäck, Gabriela Ochoa  

**Link**: [PDF](https://arxiv.org/pdf/2507.03605)  

**Abstract**: We investigate the behaviour space of meta-heuristic optimisation algorithms automatically generated by Large Language Model driven algorithm discovery methods. Using the Large Language Evolutionary Algorithm (LLaMEA) framework with a GPT o4-mini LLM, we iteratively evolve black-box optimisation heuristics, evaluated on 10 functions from the BBOB benchmark suite. Six LLaMEA variants, featuring different mutation prompt strategies, are compared and analysed. We log dynamic behavioural metrics including exploration, exploitation, convergence and stagnation measures, for each run, and analyse these via visual projections and network-based representations. Our analysis combines behaviour-based
projections, Code Evolution Graphs built from static code features, performance convergence curves, and behaviour-based Search Trajectory Networks. The results reveal clear differences in search dynamics and algorithm structures across LLaMEA configurations. Notably, the variant that employs both a code simplification prompt and a random perturbation prompt in a 1+1 elitist evolution strategy, achieved the best performance, with the highest Area Over the Convergence Curve. Behaviour-space visualisations show that higher-performing algorithms exhibit more intensive exploitation behaviour and faster convergence with less stagnation. Our findings demonstrate how behaviour-space analysis can explain why certain LLM-designed heuristics outperform others and how LLM-driven algorithm discovery navigates the open-ended and complex search space of algorithms. These findings provide insights to guide the future design of adaptive LLM-driven algorithm generators. 

**Abstract (ZH)**: 我们研究了由大规模语言模型驱动的算法发现方法自动生成的元启发式优化算法的行为空间。使用基于大规模语言演化算法（LLaMEA）框架和GPT-o4-mini语言模型，我们迭代演化黑盒优化启发式方法，并在BBOB基准套件的10个函数上进行评估。我们比较和分析了六种LLaMEA变体，这些变体具有不同的变异提示策略。我们记录了包括探索、利用、收敛和停滞等动态行为指标，并通过可视化投影和网络表示进行分析。我们的分析结合了基于行为的投影、由静态代码特征构建的代码演化图、性能收敛曲线以及基于行为的搜索轨迹网络。结果表明，不同LLaMEA配置下的搜索动态和算法结构存在明显差异。特别地，采用代码简化提示和随机扰动提示的1+1精英演化策略的变体，取得了最佳性能，其收敛曲线下的面积最大。行为空间可视化显示，高性能算法表现出更强烈的利用行为、更快的收敛和较少的停滞。我们的研究结果展示了行为空间分析如何解释某些由大规模语言模型设计的启发式方法为何优于其他方法，并说明了大规模语言模型驱动的算法发现如何在算法的开放且复杂的搜索空间中导航。这些发现为未来适应性大规模语言模型驱动的算法生成器的设计提供了见解。 

---
# Causal-SAM-LLM: Large Language Models as Causal Reasoners for Robust Medical Segmentation 

**Title (ZH)**: 因果-SAM-LLM：大语言模型作为因果推理器以实现稳健的医学分割 

**Authors**: Tao Tang, Shijie Xu, Yiting Wu, Zhixiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03585)  

**Abstract**: The clinical utility of deep learning models for medical image segmentation is severely constrained by their inability to generalize to unseen domains. This failure is often rooted in the models learning spurious correlations between anatomical content and domain-specific imaging styles. To overcome this fundamental challenge, we introduce Causal-SAM-LLM, a novel framework that elevates Large Language Models (LLMs) to the role of causal reasoners. Our framework, built upon a frozen Segment Anything Model (SAM) encoder, incorporates two synergistic innovations. First, Linguistic Adversarial Disentanglement (LAD) employs a Vision-Language Model to generate rich, textual descriptions of confounding image styles. By training the segmentation model's features to be contrastively dissimilar to these style descriptions, it learns a representation robustly purged of non-causal information. Second, Test-Time Causal Intervention (TCI) provides an interactive mechanism where an LLM interprets a clinician's natural language command to modulate the segmentation decoder's features in real-time, enabling targeted error correction. We conduct an extensive empirical evaluation on a composite benchmark from four public datasets (BTCV, CHAOS, AMOS, BraTS), assessing generalization under cross-scanner, cross-modality, and cross-anatomy settings. Causal-SAM-LLM establishes a new state of the art in out-of-distribution (OOD) robustness, improving the average Dice score by up to 6.2 points and reducing the Hausdorff Distance by 15.8 mm over the strongest baseline, all while using less than 9% of the full model's trainable parameters. Our work charts a new course for building robust, efficient, and interactively controllable medical AI systems. 

**Abstract (ZH)**: 深度学习模型在医学图像分割中的临床应用受到其难以泛化到未见领域的能力限制。这一失败往往源于模型学习了解剖内容与特定成像风格间的虚假关联。为克服这一根本性挑战，我们引入了因果-SAM-LLM（Causal-SAM-LLM）这一新颖框架，将大型语言模型（LLM）提升为因果推理者的角色。基于冻结的Segment Anything Model（SAM）编码器，我们的框架集成了两项协同创新。首先，语言对抗脱噪（LAD）利用视觉-语言模型生成丰富、文本化的混杂图像风格描述，并通过训练分割模型的特征与这些风格描述形成对比性差异，从而学习到不受非因果信息污染的表示。其次，测试时因果干预（TCI）提供了一个交互机制，其中LLM通过解释临床人员的自然语言指令实时调整分割解码器的特征，实现精准的错误修正。我们对来自四个公开数据集（BTCV、CHAOS、AMOS、BraTS）的综合基准进行了广泛的经验评估，考察了跨扫描仪、跨模态和跨器官设置下的泛化能力。Causal-SAM-LLM 在分布外（OOD）鲁棒性上建立了新的标准，平均狄氏分数提高了6.2个百分点，并将Hausdorff距离减少了15.8毫米，同时仅使用了全模型不到9%的可训练参数。本项工作为构建稳健、高效且可交互控制的医疗AI系统开辟了新的路径。 

---
# H2HTalk: Evaluating Large Language Models as Emotional Companion 

**Title (ZH)**: H2HTalk: 评估大型语言模型作为情感伴侣的有效性 

**Authors**: Boyang Wang, Yalun Wu, Hongcheng Guo, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03543)  

**Abstract**: As digital emotional support needs grow, Large Language Model companions offer promising authentic, always-available empathy, though rigorous evaluation lags behind model advancement. We present Heart-to-Heart Talk (H2HTalk), a benchmark assessing companions across personality development and empathetic interaction, balancing emotional intelligence with linguistic fluency. H2HTalk features 4,650 curated scenarios spanning dialogue, recollection, and itinerary planning that mirror real-world support conversations, substantially exceeding previous datasets in scale and diversity. We incorporate a Secure Attachment Persona (SAP) module implementing attachment-theory principles for safer interactions. Benchmarking 50 LLMs with our unified protocol reveals that long-horizon planning and memory retention remain key challenges, with models struggling when user needs are implicit or evolve mid-conversation. H2HTalk establishes the first comprehensive benchmark for emotionally intelligent companions. We release all materials to advance development of LLMs capable of providing meaningful and safe psychological support. 

**Abstract (ZH)**: 数字情感支持需求增长之际，大型语言模型伴侣提供有希望的真实且始终可用的同理心，尽管严格的评估滞后于模型进步。我们呈现心灵至心灵交谈（H2HTalk）基准测试，该基准测试评估伴侣在个性发展和同理互动方面的表现，平衡情感智能与语言流畅性。H2HTalk包含4,650个精心策划的场景，涵盖对话、回忆和行程规划，反映真实世界的支持对话，显著超越了以往数据集在规模和多样性上的限制。我们引入一个安全依附人格（SAP）模块，实施依附理论原则，以确保更安全的交互。使用统一协议对50个LLM进行基准测试表明，长期规划和记忆保留仍然是关键挑战，模型在用户需求含蓄或在对话中途变化时表现挣扎。H2HTalk建立了首个全面的情感智能伴侣基准测试。我们发布所有材料以促进能够提供有意义且安全的心理支持的LLM的发展。 

---
# Reinforcement Learning-based Feature Generation Algorithm for Scientific Data 

**Title (ZH)**: 基于强化学习的科学数据特征生成算法 

**Authors**: Meng Xiao, Junfeng Zhou, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.03498)  

**Abstract**: Feature generation (FG) aims to enhance the prediction potential of original data by constructing high-order feature combinations and removing redundant features. It is a key preprocessing step for tabular scientific data to improve downstream machine-learning model performance. Traditional methods face the following two challenges when dealing with the feature generation of scientific data: First, the effective construction of high-order feature combinations in scientific data necessitates profound and extensive domain-specific expertise. Secondly, as the order of feature combinations increases, the search space expands exponentially, imposing prohibitive human labor consumption. Advancements in the Data-Centric Artificial Intelligence (DCAI) paradigm have opened novel avenues for automating feature generation processes. Inspired by that, this paper revisits the conventional feature generation workflow and proposes the Multi-agent Feature Generation (MAFG) framework. Specifically, in the iterative exploration stage, multi-agents will construct mathematical transformation equations collaboratively, synthesize and identify feature combinations ex-hibiting high information content, and leverage a reinforcement learning mechanism to evolve their strategies. Upon completing the exploration phase, MAFG integrates the large language models (LLMs) to interpreta-tively evaluate the generated features of each significant model performance breakthrough. Experimental results and case studies consistently demonstrate that the MAFG framework effectively automates the feature generation process and significantly enhances various downstream scientific data mining tasks. 

**Abstract (ZH)**: 特征生成（FG）旨在通过构建高阶特征组合和去除冗余特征来增强原始数据的预测潜力。它是提高表格式科学数据下游机器学习模型性能的关键预处理步骤。传统方法在处理科学数据的特征生成时面临以下两个挑战：首先，有效构建科学数据中的高阶特征组合需要深厚广泛的专业领域知识。其次，随着特征组合阶数的增加，搜索空间会呈指数级扩大，导致巨大的人力劳动消耗。数据为中心的人工智能（DCAI）范式的进步为自动化特征生成过程开辟了新的途径。受此启发，本文重新审视了传统的特征生成工作流，并提出了多代理特征生成（MAFG）框架。特别地，在迭代探索阶段，多代理将协作构建数学变换方程，综合并识别信息含量高的特征组合，并利用强化学习机制演化其策略。在完成探索阶段后，MAFG结合大型语言模型（LLMs）进行解释性评价，以评估每个重大模型性能突破中生成的特征。实验结果和案例研究一致地证明，MAFG框架有效地自动化了特征生成过程，并显著增强了各种下游科学数据挖掘任务。 

---
# Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right 

**Title (ZH)**: 超越武器化：中低资源语言的自身权利下的自然语言处理安全 

**Authors**: Heather Lent  

**Link**: [PDF](https://arxiv.org/pdf/2507.03473)  

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages. 

**Abstract (ZH)**: 尽管有大量的证据表明多语言能力可以被轻松武器化用于语言模型（LMs），NLP安全领域的研究仍然主要集中在英语上。为了在NLP安全中减轻最坏情况的结果，研究人员必须愿意关注LM安全中最薄弱的环节：低资源语言。因此，本工作研究了低资源和中资源语言的LM安全性。我们将现有的对抗性攻击扩展到多达70种语言，以评估这些语言的单语言和多语言LM的安全性。通过对这些语言的LM进行分析，我们发现单语言模型往往由于参数总量较少而难以确保安全性，尽管多语言能力有帮助，但它也不总是能够保证安全性提升。最终，这些发现强调了更安全地部署LMs对于低资源语言社区的重要考虑。 

---
# Improving Social Determinants of Health Documentation in French EHRs Using Large Language Models 

**Title (ZH)**: 使用大型语言模型改善法语电子健康记录中的社会决定因素记录 

**Authors**: Adrien Bazoge, Pacôme Constant dit Beaufils, Mohammed Hmitouch, Romain Bourcier, Emmanuel Morin, Richard Dufour, Béatrice Daille, Pierre-Antoine Gourraud, Matilde Karakachoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03433)  

**Abstract**: Social determinants of health (SDoH) significantly influence health outcomes, shaping disease progression, treatment adherence, and health disparities. However, their documentation in structured electronic health records (EHRs) is often incomplete or missing. This study presents an approach based on large language models (LLMs) for extracting 13 SDoH categories from French clinical notes. We trained Flan-T5-Large on annotated social history sections from clinical notes at Nantes University Hospital, France. We evaluated the model at two levels: (i) identification of SDoH categories and associated values, and (ii) extraction of detailed SDoH with associated temporal and quantitative information. The model performance was assessed across four datasets, including two that we publicly release as open resources. The model achieved strong performance for identifying well-documented categories such as living condition, marital status, descendants, job, tobacco, and alcohol use (F1 score > 0.80). Performance was lower for categories with limited training data or highly variable expressions, such as employment status, housing, physical activity, income, and education. Our model identified 95.8% of patients with at least one SDoH, compared to 2.8% for ICD-10 codes from structured EHR data. Our error analysis showed that performance limitations were linked to annotation inconsistencies, reliance on English-centric tokenizer, and reduced generalizability due to the model being trained on social history sections only. These results demonstrate the effectiveness of NLP in improving the completeness of real-world SDoH data in a non-English EHR system. 

**Abstract (ZH)**: 社会决定因素对健康的影響（SDoH）顯著影響健康結果，塑造疾病的進展、治療依從性和健康不平等。然而，這些因素在結構化的電子健康紀錄（EHRs）中的記錄往往不完整或缺失。本研究提出了一種基於大型語言模型（LLMs）的方法，用於從法國臨床記錄中提取13類SDoH。我們在法國南特大學醫院的臨床記錄中标註的社會史部分上訓練了Flan-T5-Large。我們在兩個層面上評估了模型的表現：（i）識別SDoH類別及其相關值，以及（ii）提取帶有關聯時間和量化信息的詳細SDoH。我們在四個數據集中評估了模型的表現，包括兩個我們公開釋出作為開源資源的數據集。模型在生活條件、婚姻狀態、後代、職業、 tobacco 和酒精使用等 хорошо文書化類別的識別上表現出色（F1分數>0.80）。但在訓練數據有限或表達高度多變的類別，如就業狀態、住房、體育活動、收入和教育方面，表現較低。本研究模型識別出至少一項SDoH的患者佔95.8%，而基于結構化EHR數據的ICD-10碼僅為2.8%。我們的錯誤分析表明，表現限制與標注不一致、依賴英語为中心的分词器以及模型僅訓練於社會史部分而导致的泛化能力不足有關。這些結果展示了NLP在改善非英語EHR系統中真實世界SDoH數據完整性方面的有效性。 

---
# LLM4Hint: Leveraging Large Language Models for Hint Recommendation in Offline Query Optimization 

**Title (ZH)**: LLM4Hint：利用大型语言模型进行离线查询优化中的提示推荐 

**Authors**: Suchen Liu, Jun Gao, Yinjun Han, Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03384)  

**Abstract**: Query optimization is essential for efficient SQL query execution in DBMS, and remains attractive over time due to the growth of data volumes and advances in hardware. Existing traditional optimizers struggle with the cumbersome hand-tuning required for complex workloads, and the learning-based methods face limitations in ensuring generalization. With the great success of Large Language Model (LLM) across diverse downstream tasks, this paper explores how LLMs can be incorporated to enhance the generalization of learned optimizers. Though promising, such an incorporation still presents challenges, mainly including high model inference latency, and the substantial fine-tuning cost and suboptimal performance due to inherent discrepancy between the token sequences in LLM and structured SQL execution plans with rich numerical features.
In this paper, we focus on recurring queries in offline optimization to alleviate the issue of high inference latency, and propose \textbf{LLM4Hint} that leverages moderate-sized backbone LLMs to recommend query optimization hints. LLM4Hint achieves the goals through: (i) integrating a lightweight model to produce a soft prompt, which captures the data distribution in DBMS and the SQL predicates to provide sufficient optimization features while simultaneously reducing the context length fed to the LLM, (ii) devising a query rewriting strategy using a larger commercial LLM, so as to simplify SQL semantics for the backbone LLM and reduce fine-tuning costs, and (iii) introducing an explicit matching prompt to facilitate alignment between the LLM and the lightweight model, which can accelerate convergence of the combined model. Experiments show that LLM4Hint, by leveraging the LLM's stronger capability to understand the query statement, can outperform the state-of-the-art learned optimizers in terms of both effectiveness and generalization. 

**Abstract (ZH)**: 基于大语言模型的查询优化提示（LLM4Hint） 

---
# Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs 

**Title (ZH)**: 静读 aloud, 明理 quietly: 解耦 LLMs 的理解与推理 

**Authors**: Yuanxin Wang, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03327)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing.
We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解文本和生成高质量响应方面展现了显著的能力。然而，与人类认知的一个关键区别在于，它们通常在生成文本（即“说话”）之前缺乏明显的内部“阅读”或思考阶段。人类往往会进行无声阅读以理解上下文并形成想法后再表达。本文研究了赋予LLMs类似内部处理能力的方法。我们介绍了并评估了鼓励LLMs进行“无声阅读”的技术。研究结果显示，即使是一种简单的做法，比如在模型开始预测最终输出的后续标记之前提供初始上下文提示或“阅读空间”，也能显著提高性能。我们进一步通过开发一种“阅读伙伴”架构来增强这一概念，其中辅助组件无声地处理输入并为主要内容生成模型提供精细化的上下文洞察。这些方法旨在促进LLMs进行更深层次的理解，从而产生更具道理的响应，使其更接近于类似人类的文本处理。我们的结果表明，这些简单的方法可以显著提高准确性，提供多个点的准确度提升。 

---
# Personalized Image Generation from an Author Writing Style 

**Title (ZH)**: 根据作者写作风格的个性化图像生成 

**Authors**: Sagar Gandhi, Vishal Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03313)  

**Abstract**: Translating nuanced, textually-defined authorial writing styles into compelling visual representations presents a novel challenge in generative AI. This paper introduces a pipeline that leverages Author Writing Sheets (AWS) - structured summaries of an author's literary characteristics - as input to a Large Language Model (LLM, Claude 3.7 Sonnet). The LLM interprets the AWS to generate three distinct, descriptive text-to-image prompts, which are then rendered by a diffusion model (Stable Diffusion 3.5 Medium). We evaluated our approach using 49 author styles from Reddit data, with human evaluators assessing the stylistic match and visual distinctiveness of the generated images. Results indicate a good perceived alignment between the generated visuals and the textual authorial profiles (mean style match: $4.08/5$), with images rated as moderately distinctive. Qualitative analysis further highlighted the pipeline's ability to capture mood and atmosphere, while also identifying challenges in representing highly abstract narrative elements. This work contributes a novel end-to-end methodology for visual authorial style personalization and provides an initial empirical validation, opening avenues for applications in creative assistance and cross-modal understanding. 

**Abstract (ZH)**: 将文本定义的作者写作风格微妙的表达转化为引人入胜的视觉表现构成了生成式AI的新挑战。本论文引入了一种基于作者写作表（AWS）的流程，AWS是对作者文学特征的结构化总结，作为大型语言模型（LLM，Claude 3.7 Sonnet）的输入。LLM通过解释AWS生成三个不同的、描述性的文本到图像提示，这些提示随后由扩散模型（Stable Diffusion 3.5 Medium）呈现。我们使用来自Reddit的数据中的49种作者风格进行了评估，由人类评估者评估生成图像的风格匹配度和视觉独特性。结果表明，生成的视觉效果与文本作者特征（平均风格匹配：4.08/5）之间存在良好的感知契合度，并且图像被评价为中等独特性。定性分析进一步强调了该流程捕捉氛围和情绪的能力，同时也指出了在表现高度抽象的叙事元素方面的挑战。本研究贡献了一种新颖的整体方法，用于视觉作者风格个性化，并提供了初步的经验验证，开启了在创造性辅助和跨模态理解方面的应用途径。 

---
# GRAFT: A Graph-based Flow-aware Agentic Framework for Document-level Machine Translation 

**Title (ZH)**: 基于图的流敏代理框架：面向文档级机器翻译 

**Authors**: Himanshu Dutta, Sunny Manchanda, Prakhar Bapat, Meva Ram Gurjar, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03311)  

**Abstract**: Document level Machine Translation (DocMT) approaches often struggle with effectively capturing discourse level phenomena. Existing approaches rely on heuristic rules to segment documents into discourse units, which rarely align with the true discourse structure required for accurate translation. Otherwise, they fail to maintain consistency throughout the document during translation. To address these challenges, we propose Graph Augmented Agentic Framework for Document Level Translation (GRAFT), a novel graph based DocMT system that leverages Large Language Model (LLM) agents for document translation. Our approach integrates segmentation, directed acyclic graph (DAG) based dependency modelling, and discourse aware translation into a cohesive framework. Experiments conducted across eight translation directions and six diverse domains demonstrate that GRAFT achieves significant performance gains over state of the art DocMT systems. Specifically, GRAFT delivers an average improvement of 2.8 d BLEU on the TED test sets from IWSLT2017 over strong baselines and 2.3 d BLEU for domain specific translation from English to Chinese. Moreover, our analyses highlight the consistent ability of GRAFT to address discourse level phenomena, yielding coherent and contextually accurate translations. 

**Abstract (ZH)**: 基于图增强自主框架的文档级别机器翻译（GRAFT） 

---
# MGAA: Multi-Granular Adaptive Allocation fof Low-Rank Compression of LLMs 

**Title (ZH)**: MGAA: 多粒度自适应分配用于大语言模型低秩压缩 

**Authors**: Guangyan Li, Yongqiang Tang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03294)  

**Abstract**: The enormous parameter scale of large language models (LLMs) has made model compression a research hotspot, which aims to alleviate computational resource demands during deployment and inference. As a promising direction, low-rank approximation technique has made remarkable achievements. Nevertheless, unfortunately, the vast majority of studies to low-rank approximation compression generally apply uniform compression ratios across all weight matrices, while disregarding their inherently differentiated impacts on the model's performance. Although a few recent work attempts to employ heuristic search strategies to achieve the optimal parameter allocation, such strategies are computationally inefficient and lose the generalization ability in the era of LLMs. In this study, we propose a novel parameter Multi-Granular Adaptive Allocation (MGAA) method, which can adaptively allocate parameters between and within sublayers without task-specific evaluations in the compression process. MGAA consists of two components: 1) Among different sublayers, it assigns compression ratios based on their cosine similarity between inputs and outputs, allowing for a more tailored compression in sublayers with varying degrees of importance, and 2) Within each sublayer, it allocates different compression ratios to weight matrices based on their energy distribution characteristics, ensuring a consistent energy retention ratio while optimizing compression efficiency. Comprehensive evaluations of MGAA across multiple LLMs backbone models and benchmark datasets demonstrate its superior performance. Additionally, we apply our MGAA to multimodal model LLaVA, exhibiting remarkable performance improvements. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的巨大参数规模使得模型压缩成为研究热点，旨在部署和推理过程中缓解计算资源需求。作为一种有前途的方向，低秩逼近技术已经取得了显著成就。然而，大多数关于低秩逼近压缩的研究通常在所有权重矩阵上应用统一的压缩比率，而忽视了它们对模型性能的不同影响。虽然一些最近的工作尝试使用启发式搜索策略来实现最优参数分配，但这些策略在大规模语言模型时代计算效率低下，并且丧失了泛化能力。在本研究中，我们提出了一种新型参数多粒度自适应分配（MGAA）方法，在压缩过程中无需针对特定任务进行参数分配。MGAA由两个部分组成：1）在不同的子层之间，根据输入和输出之间的余弦相似度分配压缩比率，从而在不同重要程度的子层中实现更加个性化的压缩；2）在每个子层内部，基于权重矩阵的能量分布特性分配不同的压缩比率，确保能量保留比的一致性同时优化压缩效率。MGAA在多个LLM骨干模型和基准数据集上的综合评估展示了其优越性能。此外，我们将MGAA应用于多模态模型LLaVA，显现出显著的性能提升。 

---
# Conformal Information Pursuit for Interactively Guiding Large Language Models 

**Title (ZH)**: 符合信息引导的大语言模型交互式辅助方法 

**Authors**: Kwan Ho Ryan Chan, Yuyan Ge, Edgar Dobriban, Hamed Hassani, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2507.03279)  

**Abstract**: A significant use case of instruction-finetuned Large Language Models (LLMs) is to solve question-answering tasks interactively. In this setting, an LLM agent is tasked with making a prediction by sequentially querying relevant information from the user, as opposed to a single-turn conversation. This paper explores sequential querying strategies that aim to minimize the expected number of queries. One such strategy is Information Pursuit (IP), a greedy algorithm that at each iteration selects the query that maximizes information gain or equivalently minimizes uncertainty. However, obtaining accurate estimates of mutual information or conditional entropy for LLMs is very difficult in practice due to over- or under-confident LLM probabilities, which leads to suboptimal query selection and predictive performance. To better estimate the uncertainty at each iteration, we propose Conformal Information Pursuit (C-IP), an alternative approach to sequential information gain based on conformal prediction sets. More specifically, C-IP leverages a relationship between prediction sets and conditional entropy at each iteration to estimate uncertainty based on the average size of conformal prediction sets. In contrast to conditional entropy, we find that conformal prediction sets are a distribution-free and robust method of measuring uncertainty. Experiments with 20 Questions show that C-IP obtains better predictive performance and shorter query-answer chains compared to previous approaches to IP and uncertainty-based chain-of-thought methods. Furthermore, extending to an interactive medical setting between a doctor and a patient on the MediQ dataset, C-IP achieves competitive performance with direct single-turn prediction while offering greater interpretability. 

**Abstract (ZH)**: 大型语言模型（LLMs）指令调优的重要应用场景是解决交互式问答任务。在这种设置中，LLM代理需要通过顺序查询相关信息来自动生成预测，而不是进行单轮对话。本文探索旨在最小化预期查询次数的顺序查询策略。其中一种策略是信息追求（IP），这是一种贪婪算法，在每一步迭代中选择最大化信息增益或等价地最小化不确定性的问题查询。然而，由于LLM概率的高估或低估，实际中很难获得准确的互信息或条件熵估计，这会导致次优的查询选择和预测性能。为了更好地在每一步迭代中估计不确定性，我们提出了聚合法信息追求（C-IP），这是一种基于聚合法预测集的顺序信息增益的替代方法。具体而言，C-IP 利用每一步迭代中预测集和条件熵之间的关系，通过聚合法预测集的平均大小来估计不确定性。与条件熵不同，我们发现聚合法预测集是一种无分布且稳健的不确定性度量方法。20 问题实验表明，C-IP 在预测性能和较短的查询-回答链方面优于之前的 IP 方法和基于不确定性的链式思考方法。此外，将 C-IP 拓展到医生和患者之间的互动医疗场景（使用 MediQ 数据集），C-IP 在提供更强的可解释性的同时，实现了与直接单轮预测相当的竞争力。 

---
# Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders 

**Title (ZH)**: 探讨多模态大型语言模型中多个视觉编码器的冗余性 

**Authors**: Song Mao, Yang Chen, Pinglong Cai, Ding Wang, Guohang Yan, Zhi Yu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03262)  

**Abstract**: Multimodal Large Language Models (MLLMs) increasingly adopt multiple vision encoders to capture diverse visual information, ranging from coarse semantics to fine grained details. While this approach is intended to enhance visual understanding capability, we observe that the performance gains from adding encoders often diminish and can even lead to performance degradation, a phenomenon we term encoder redundancy. This paper presents a systematic investigation into this issue. Through comprehensive ablation studies on state of the art multi encoder MLLMs, we empirically demonstrate that significant redundancy exists. To quantify each encoder's unique contribution, we propose a principled metric: the Conditional Utilization Rate (CUR). Building on CUR, we introduce the Information Gap (IG) to capture the overall disparity in encoder utility within a this http URL experiments reveal that certain vision encoders contribute little, or even negatively, to overall performance, confirming substantial redundancy. Our experiments reveal that certain vision encoders contribute minimally, or even negatively, to the model's performance, confirming the prevalence of redundancy. These findings highlight critical inefficiencies in current multi encoder designs and establish that our proposed metrics can serve as valuable diagnostic tools for developing more efficient and effective multimodal architectures. 

**Abstract (ZH)**: 多模态大型语言模型中的编码器冗余现象及其量化研究 

---
# RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs 

**Title (ZH)**: RefineX: 从专家指导程序中大规模精炼预训练数据 

**Authors**: Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03253)  

**Abstract**: The foundational capabilities of large language models (LLMs) are deeply influenced by the quality of their pre-training corpora. However, enhancing data quality at scale remains a significant challenge, primarily due to the trade-off between refinement effectiveness and processing efficiency. While rule-based filtering remains the dominant paradigm, it typically operates at the document level and lacks the granularity needed to refine specific content within documents. Inspired by emerging work such as ProX, we propose $\textbf{RefineX}$, a novel framework for large-scale, surgical refinement of pre-training data through programmatic editing tasks. RefineX enables efficient and fine-grained data refinement while reliably preserving the diversity and naturalness of raw text. The core strength of RefineX lies in distilling high-quality, expert-guided end-to-end refinement results into minimal edit-based deletion programs. This high-precision distillation pipeline is used to train an efficient and reliable refine model that can systematically improve every instance in the corpus at scale. We evaluate RefineX across from-scratch pre-training at multiple model scales and find that it consistently outperforms models trained on raw, filtered, or alternatively refined data across diverse downstream tasks. On the 750M model, RefineX yields 2.6%-7.2% average gains on lighteval tasks, and achieves comparable performance using significantly fewer training tokens. Further analysis shows that RefineX reliably enhances text quality with both high efficiency and precision, outperforming prior approaches such as end-to-end generation and Prox-C. These results position RefineX as a scalable, effective, and reliable solution for optimizing pre-training data in modern LLM pipelines. 

**Abstract (ZH)**: 大规模语言模型的基礎能力受其前期训练语料质量的影响。然而，大规模提升数据质量仍然是一项重大挑战，主要由于精炼效果与处理效率之间的权衡。尽管基于规则的过滤仍然是主导范式，但通常在文档级别操作，缺乏对文档内具体内容进行精细化精炼所需的粒度。受ProX等新兴工作的启发，我们提出了**RefineX**框架，这是一种用于大规模、手术式精炼预训练数据的新颖程序化编辑任务框架。RefineX使数据精炼既高效又精细化，同时可靠地保留原始文本的多样性和自然性。RefineX的核心优势在于将高质量、专家导向的一体化精炼结果精炼为最小的基于编辑的删除程序。这一高精度精炼管道用于训练高效的可靠精炼模型，可以大规模系统地改进语料库中的每一个实例。我们在多个模型规模的从头预训练中评估了RefineX，发现它在多样化的下游任务中始终优于使用未过滤、过滤或替代精炼数据训练的模型。在750M模型上，RefineX在轻量评估任务中平均提高了2.6%至7.2%，并且使用显著较少的训练标记达到类似性能。进一步分析表明，RefineX在高效率和高精度下可靠提升文本质量，优于先前方法如端到端生成和Prox-C。这些结果将RefineX定位为一种在现代大规模语言模型管道中优化预训练数据的大规模、有效且可靠解决方案。 

---
# On Jailbreaking Quantized Language Models Through Fault Injection Attacks 

**Title (ZH)**: 通过故障注入攻击破解量化语言模型 

**Authors**: Noureldin Zahran, Ahmad Tahmasivand, Ihsen Alouani, Khaled Khasawneh, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2507.03236)  

**Abstract**: The safety alignment of Language Models (LMs) is a critical concern, yet their integrity can be challenged by direct parameter manipulation attacks, such as those potentially induced by fault injection. As LMs are increasingly deployed using low-precision quantization for efficiency, this paper investigates the efficacy of such attacks for jailbreaking aligned LMs across different quantization schemes. We propose gradient-guided attacks, including a tailored progressive bit-level search algorithm introduced herein and a comparative word-level (single weight update) attack. Our evaluation on Llama-3.2-3B, Phi-4-mini, and Llama-3-8B across FP16 (baseline), and weight-only quantization (FP8, INT8, INT4) reveals that quantization significantly influences attack success. While attacks readily achieve high success (>80\% Attack Success Rate, ASR) on FP16 models, within an attack budget of 25 perturbations, FP8 and INT8 models exhibit ASRs below 20\% and 50\%, respectively. Increasing the perturbation budget up to 150 bit-flips, FP8 models maintained ASR below 65\%, demonstrating some resilience compared to INT8 and INT4 models that have high ASR. In addition, analysis of perturbation locations revealed differing architectural targets across quantization schemes, with (FP16, INT4) and (INT8, FP8) showing similar characteristics. Besides, jailbreaks induced in FP16 models were highly transferable to subsequent FP8/INT8 quantization (<5\% ASR difference), though INT4 significantly reduced transferred ASR (avg. 35\% drop). These findings highlight that while common quantization schemes, particularly FP8, increase the difficulty of direct parameter manipulation jailbreaks, vulnerabilities can still persist, especially through post-attack quantization. 

**Abstract (ZH)**: 语言模型的安全对齐安全性是一个关键问题，然而它们的完整性可能受到直接参数操作攻击的挑战，如由故障注入引发的攻击。随着语言模型越来越多地通过低精度量化来提高效率进行部署，本文探讨了不同量化方案下此类攻击突破对齐语言模型的有效性。我们提出了梯度引导攻击，包括一种在此引入的定制分阶段位级搜索算法和单一权重更新的词级比较攻击。在对Llama-3.2-3B、Phi-4-mini和Llama-3-8B进行的评估中，包括在FP16（基线）、权重唯一量化（FP8、INT8、INT4）下显示，量化显著影响攻击成功率。虽然攻击预算为25个扰动时，FP16模型的攻击成功率超过80%，FP8和INT8模型分别低于20%和50%。将扰动预算增加到150位翻转时，FP8模型的攻击成功率保持在65%以下，显示出比INT8和INT4模型更高的鲁棒性。此外，扰动位置的分析显示不同量化方案下有不同的架构目标，(FP16, INT4)和(INT8, FP8)表现出相似特性。此外，在FP16模型中诱导的突破在随后的FP8/INT8量化中高度可转移（<5%的攻击成功率差异），尽管INT4显著降低了转移的攻击成功率（平均下降35%）。这些发现表明，虽然常见的量化方案，特别是FP8，增加了直接参数操作突破的难度，但漏洞仍然可能存在，尤其是在攻击后的量化过程中。 

---
# Symbiosis: Multi-Adapter Inference and Fine-Tuning 

**Title (ZH)**: 共生：多适配器推理与微调 

**Authors**: Saransh Gupta, Umesh Deshpande, Travis Janssen, Swami Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03220)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) allows model builders to capture the task specific parameters into adapters, which are a fraction of the size of the original base model. Popularity of PEFT technique for fine-tuning has led to creation of a large number of adapters for popular Large Language Models (LLMs). However, existing frameworks fall short in supporting inference or fine-tuning with multiple adapters in the following ways. 1) For fine-tuning, each job needs to deploy its dedicated base model instance, which results in excessive GPU memory consumption and poor GPU utilization. 2) While popular inference platforms can serve multiple PEFT adapters, they do not allow independent resource management or mixing of different PEFT methods. 3) They cannot share resources (such as base model instance) between inference and fine-tuning jobs. 4) They do not provide privacy to users who may not wish to expose their fine-tuned parameters to service providers. In Symbiosis, we address the above problems by enabling as-a-service deployment of base model. The base model layers can be shared across multiple inference or fine-tuning processes. Our split-execution technique decouples the execution of client-specific adapters and layers from the frozen base model layers offering them flexibility to manage their resources, to select their fine-tuning method, to achieve their performance goals. Our approach is transparent to models and works out-of-the-box for most models in the transformers library. Our evaluation on Llama2-13B shows the compared to baseline, Symbiosis can fine-tune 4X more adapters on the same set of GPUs in the same amount of time. 

**Abstract (ZH)**: Parameter-efficient Fine-tuning with Shared Base Models in Symbiosis 

---
# How Much Content Do LLMs Generate That Induces Cognitive Bias in Users? 

**Title (ZH)**: LLMs生成的诱导用户认知偏见的内容有多少？ 

**Authors**: Abeer Alessa, Akshaya Lakshminarasimhan, Param Somane, Julian Skirzynski, Julian McAuley, Jessica Echterhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03194)  

**Abstract**: Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）在从评论摘要到医疗诊断支持的应用中日益集成，影响人类决策。尽管LLMs在许多任务中表现良好，但也可能继承社会或认知偏见，这些偏见可能无意间转移到人类身上。我们研究了LLMs在摘要和新闻事实核查任务中何时以及如何向用户暴露偏见内容，并量化其严重程度。具体而言，我们在三个LLM家族中评估了其在摘要和新闻事实核查任务中的表现，评估LLMs在多大程度上保持一致性或胡言乱语。我们的研究表明，在21.86%的情况下，LLMs使上下文的情感发生变化，在57.33%的情况下对后知识截止日期的数据问题进行胡言乱语，在5.94%的情况下表现出首因效应偏见。我们在三个LLM家族中评估了18种不同的缓解方法，发现针对性的干预措施可能有效。鉴于LLMs在高风险领域，如医疗保健或法律分析中的广泛应用，我们的研究结果强调了需要 robust的技术保障，并开发以用户为中心的干预措施来解决LLM的局限性。 

---
# MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks 

**Title (ZH)**: MateInfoUB：用于测试在竞争性、多语言和多模态教育任务中大型语言模型的现实世界基准 

**Authors**: Dumitran Adrian Marius, Theodor-Pierre Moroianu, Buca Mihnea-Vicentiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03162)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment. 

**Abstract (ZH)**: 大型语言模型的快速进步已transformed various领域，尤其是在计算机科学（CS）教育方面。这些模型在代码相关任务和问题解决方面展现出非凡的能力，引发了它们在高级CS环境中的潜力和局限性的思考。本研究提出了一种新颖的双语（英语-罗曼语）多模态（文本和图像）试题集，源自一项高级计算机科学竞赛。该数据集的一个特点是，部分问题更适合通过纸上推理解决，而对于其他问题，则编写代码更为高效。我们系统地评估了当前最先进的大型语言模型在该数据集上的表现，分析了它们在理论编程任务中的性能。我们的研究发现揭示了当前大型语言模型的优势和局限性，包括语言选择（英语 vs. 罗曼语）的影响，提供了它们在CS教育和竞赛环境中的应用见解。此外，我们还探讨了大型语言模型使用背景下教育诚信和评估公平性的关键伦理考虑。这些讨论旨在指导未来的教育实践和政策。为了支持进一步的研究，该数据集将以英语和罗曼语形式公开发布。此外，我们还发布了一款针对罗曼语学生定制的教育应用，使他们能够在交互性和实践导向的环境中自我评估。 

---
# The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Literature Review 

**Title (ZH)**: LLM助手对软件开发者生产力的影响：一项系统文献综述 

**Authors**: Amr Mohamed, Maram Assi, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.03156)  

**Abstract**: Large language model assistants (LLM-assistants) present new opportunities to transform software development. Developers are increasingly adopting these tools across tasks, including coding, testing, debugging, documentation, and design. Yet, despite growing interest, there is no synthesis of how LLM-assistants affect software developer productivity. In this paper, we present a systematic literature review of 37 peer-reviewed studies published between January 2014 and December 2024 that examine this impact. Our analysis reveals that LLM-assistants offer both considerable benefits and critical risks. Commonly reported gains include minimized code search, accelerated development, and the automation of trivial and repetitive tasks. However, studies also highlight concerns around cognitive offloading, reduced team collaboration, and inconsistent effects on code quality. While the majority of studies (92%) adopt a multi-dimensional perspective by examining at least two SPACE dimensions, reflecting increased awareness of the complexity of developer productivity, only 14% extend beyond three dimensions, indicating substantial room for more integrated evaluations. Satisfaction, Performance, and Efficiency are the most frequently investigated dimensions, whereas Communication and Activity remain underexplored. Most studies are exploratory (64%) and methodologically diverse, but lack longitudinal and team-based evaluations. This review surfaces key research gaps and provides recommendations for future research and practice. All artifacts associated with this study are publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型助手（LLM助手）为软件开发转型提供了新机遇。开发人员越来越多地在编码、测试、调试、文档编写和设计等任务中采用这些工具。然而，尽管兴趣日益浓厚，却没有综合分析LLM助手如何影响软件开发者生产力。在本文中，我们对2014年1月到2024年12月间发表的37篇同行评审研究进行了系统文献综述，这些研究探讨了这一影响。我们的分析表明，LLM助手提供了显著的利益，同时也带来了关键的风险。常见的获益包括代码搜索减少、开发加速以及自动化的琐碎和重复性任务。然而，研究也指出了认知卸载、团队协作减少以及代码质量不一致等关切。虽然大部分研究（92%）采用了多维度视角，至少考察了两个SPACE维度，反映了对开发者生产力复杂性的更强认知，但仅有14%的研究扩展到超过三个维度，表明对更全面评估仍有很大空间。满意度、绩效和效率是研究中最为频繁考察的维度，而沟通和活动则仍然被研究较少。大多数研究具有探索性（64%）且方法论多样化，但缺乏纵向和基于团队的评估。本综述揭示了关键研究空白，并为未来的研究和实践提供了建议。与本研究相关的所有成果均可通过此 https URL 公开获取。 

---
# How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models 

**Title (ZH)**: 初始选择中的过度自信与批评中的欠自信如何调Modulate大型语言模型中的改变认知 

**Authors**: Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward, Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar Velickovic, Viorica Patraucean  

**Link**: [PDF](https://arxiv.org/pdf/2507.03120)  

**Abstract**: Large language models (LLMs) exhibit strikingly conflicting behaviors: they can appear steadfastly overconfident in their initial answers whilst at the same time being prone to excessive doubt when challenged. To investigate this apparent paradox, we developed a novel experimental paradigm, exploiting the unique ability to obtain confidence estimates from LLMs without creating memory of their initial judgments -- something impossible in human participants. We show that LLMs -- Gemma 3, GPT4o and o1-preview -- exhibit a pronounced choice-supportive bias that reinforces and boosts their estimate of confidence in their answer, resulting in a marked resistance to change their mind. We further demonstrate that LLMs markedly overweight inconsistent compared to consistent advice, in a fashion that deviates qualitatively from normative Bayesian updating. Finally, we demonstrate that these two mechanisms -- a drive to maintain consistency with prior commitments and hypersensitivity to contradictory feedback -- parsimoniously capture LLM behavior in a different domain. Together, these findings furnish a mechanistic account of LLM confidence that explains both their stubbornness and excessive sensitivity to criticism. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出令人惊讶的矛盾行为：它们在初始答案上显得异常自信，而在受到质疑时却又容易过度怀疑。为了探究这一显而易见的悖论，我们开发了一个新颖的实验范式，利用了获取LLMs置信度估计的独特能力，而无需形成它们初始判断的记忆——这是人类参与者无法做到的。我们显示，LLMs——Gemma 3、GPT4o 和 o1-preview——表现出明显的选择支持偏见，这种偏见强化并提升了它们对答案的置信度估计，导致它们顽固地抵制改变观点。我们进一步证明，LLMs对不一致建议的权重明显高于一致建议，这种从量上偏离了规范化的贝叶斯更新。最后，我们证明，这两种机制——保持与先前承诺一致性的驱动力和对矛盾反馈的超敏感性——能够简明地捕捉LLMs在不同领域中的行为。这些发现为解释LLMs的顽固性和过度敏感性提供了机理性的解释。 

---
# ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization 

**Title (ZH)**: 基于情绪驱动自监督和轨迹偏差动态优化的自适应奖励跟随：用于RLHF的过程调整与优化 

**Authors**: YuXuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03069)  

**Abstract**: With the rapid advancement of Reinforcement Learning from Human Feedback (RLHF) and autoregressive transformers, state-of-the-art models such as GPT-4.0, DeepSeek R1, and Llama 3.3 increasingly emphasize answer depth and personalization. However, most existing RLHF approaches (e.g., PPO, DPO) still rely on a binary-preference (BT) paradigm, which, while reducing annotation costs, still requires substantial human effort and captures only group-level tendencies rather than individual preferences. To overcome these limitations, we propose Adaptive Reward-Following (ARF), a self-assessment framework that leverages a high-precision emotion analyzer achieving over 70% accuracy on GoEmotions, Sentiment140, and DailyDialog to convert free-form user feedback into continuous preference scores. We further enrich and debias these signals through lightweight data augmentations, including synonym replacement, random trace truncation, and score bias annotation algorithm. A Dynamic Adapter Preference Tracker continuously models evolving user tastes in real time, enabling our novel Trace Bias (TB) fine-tuning algorithm to optimize directly on these tracked rewards instead of coarse binary labels. Experiments on Qwen-2/2.5, Gemma-2, and Llama-3.2 across four preference domains demonstrate that ARF achieves an improvement of 3.3% over PPO and 7.6% over DPO. Moreover, TB preserves theoretical alignment with PPO and DPO objectives. Overall, ARF presents a scalable, personalized, and cost-effective approach to RLHF LLMs through autonomous reward modeling. 

**Abstract (ZH)**: 基于自评估的自适应奖励跟随（ARF）：强化学习从人类反馈的个性化模型设计 

---
# Large Language Models for Automating Clinical Data Standardization: HL7 FHIR Use Case 

**Title (ZH)**: 大型语言模型在自动化临床数据标准化中的应用：HL7 FHIR案例研究 

**Authors**: Alvaro Riquelme, Pedro Costa, Catalina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2507.03067)  

**Abstract**: For years, semantic interoperability standards have sought to streamline the exchange of clinical data, yet their deployment remains time-consuming, resource-intensive, and technically challenging. To address this, we introduce a semi-automated approach that leverages large language models specifically GPT-4o and Llama 3.2 405b to convert structured clinical datasets into HL7 FHIR format while assessing accuracy, reliability, and security. Applying our method to the MIMIC-IV database, we combined embedding techniques, clustering algorithms, and semantic retrieval to craft prompts that guide the models in mapping each tabular field to its corresponding FHIR resource. In an initial benchmark, resource identification achieved a perfect F1-score, with GPT-4o outperforming Llama 3.2 thanks to the inclusion of FHIR resource schemas within the prompt. Under real-world conditions, accuracy dipped slightly to 94 %, but refinements to the prompting strategy restored robust mappings. Error analysis revealed occasional hallucinations of non-existent attributes and mismatches in granularity, which more detailed prompts can mitigate. Overall, our study demonstrates the feasibility of context-aware, LLM-driven transformation of clinical data into HL7 FHIR, laying the groundwork for semi-automated interoperability workflows. Future work will focus on fine-tuning models with specialized medical corpora, extending support to additional standards such as HL7 CDA and OMOP, and developing an interactive interface to enable expert validation and iterative refinement. 

**Abstract (ZH)**: 基于大规模语言模型的半自动化临床数据转换为HL7 FHIR格式的方法研究 

---
# LLM-Driven Auto Configuration for Transient IoT Device Collaboration 

**Title (ZH)**: 由LLM驱动的临时物联网设备协作自动配置 

**Authors**: Hetvi Shastri, Walid A. Hanafy, Li Wu, David Irwin, Mani Srivastava, Prashant Shenoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.03064)  

**Abstract**: Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices.
In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions.
We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms. 

**Abstract (ZH)**: 今岁的物联网（IoT）已从简单的传感和执行设备演变为主-cols上嵌入处理和智能服务的设备，使得用户与其设备之间能够进行丰富的协作。然而，当临时设备需要在临时访问的环境中与宿主设备交互时，实现这种协作变得具有挑战性。在这种情况下，需要精细的访问控制策略来确保安全的交互；但是，非专家用户手动实现这些策略往往是不切实际的。此外，在运行时，系统必须自动配置设备并强制执行这样的精细访问控制规则。此外，该系统必须解决设备的异构性。
本文介绍了CollabIoT系统，该系统能够使临时物联网环境中设备的协作安全且无缝。CollabIoT采用了基于大型语言模型（LLM）的方法，将用户的高层次意图转换为精细的访问控制策略。为了支持安全且无缝的设备协作，CollabIoT采用了基于能力的访问控制用于授权，并使用轻量级代理进行策略执行，提供硬件无关的抽象。 

---
# From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction 

**Title (ZH)**: 从2:4到8:16稀疏模式在考虑方差修正的情况下应用于LLMs的异常值和权重 

**Authors**: Egor Maximov, Yulia Kuzkina, Azamat Kanametov, Alexander Prutko, Aleksei Goncharov, Maxim Zhelnin, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2507.03052)  

**Abstract**: As large language models (LLMs) grow in size, efficient compression techniques like quantization and sparsification are critical. While quantization maintains performance with reduced precision, structured sparsity methods, such as N:M sparsification, often fall short due to limited flexibility, and sensitivity to outlier weights. We explore 8:16 semi-structured sparsity, demonstrating its ability to surpass the Performance Threshold-where a compressed model matches the accuracy of its uncompressed or smaller counterpart under equivalent memory constraints. Compared to 2:4 sparsity, 8:16 offers greater flexibility with minimal storage overhead (0.875 vs. 0.75 bits/element). We also apply sparse structured patterns for salient weights, showing that structured sparsity for outliers is competitive with unstructured approaches leading to equivalent or better results. Finally, we demonstrate that simple techniques such as variance correction and SmoothQuant like weight equalization improve sparse models performance. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）规模扩大，高效压缩技术如量化和稀疏化至关重要。尽管量化可以在降低精度的同时保持性能，但结构化稀疏化方法，如N:M稀疏化，往往因为灵活性有限和对异常权重敏感而效果不佳。我们探索了8:16半结构化稀疏化，证明了其能够在等内存约束条件下超越性能阈值，即压缩模型的准确度与其未压缩或更小的版本相当。与2:4稀疏化相比，8:16提供了更大的灵活性，并且存储开销较小（0.875 vs. 0.75位/元素）。我们还应用了突出权重的结构化稀疏模式，表明针对异常权重的结构化稀疏化与非结构化方法具有竞争力，导致等同或更好的结果。最后，我们展示了诸如方差校正和类似于权重均衡的简单技术可以提高稀疏模型的性能。 

---
# Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization 

**Title (ZH)**: 基于组相对策略优化提高LLM推理以增强漏洞检测 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2507.03051)  

**Abstract**: Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection.
We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul.
The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection. 

**Abstract (ZH)**: 改善和理解大型语言模型（LLMs）的训练动态和推理能力对于它们在基于AI的安全工具中的部署变得至关重要，尤其是在软件漏洞检测方面。在本工作中，我们提出了一个全面的研究，旨在推进针对漏洞检测的基于强化学习（RL）的LLM微调技术。

已突显了广泛采用的LLMs的关键局限性，如其倾向于高估某些类型的漏洞，而未能检测其他类型的漏洞。为解决这一挑战，我们探索了使用组相对策略优化（GRPO），一种近期的策略梯度方法，通过结构化的规则基础奖励来引导LLM的行为。我们通过使用来自该领域广泛使用的数据集（包括BigVul、DiverseVul和CleanVul）的注释来重新定义其优势函数和奖励信号，使其应用于漏洞检测任务。

所提出的方法使得进行大量实验成为可能，这些问题涉及GRPO对泛化、推理能力和相对于标准监督微调（SFT）的性能改进的影响。我们的研究结果提供了关于基于RL的训练如何在软件漏洞检测上下文中增强LLM的性能和推理能力的重要见解。 

---
# Counterfactual Tuning for Temporal Sensitivity Enhancement in Large Language Model-based Recommendation 

**Title (ZH)**: 基于大型语言模型的推荐中时空敏感性增强的反事实调优 

**Authors**: Yutian Liu, Zhengyi Yang, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03047)  

**Abstract**: Recent advances have applied large language models (LLMs) to sequential recommendation, leveraging their pre-training knowledge and reasoning capabilities to provide more personalized user experiences. However, existing LLM-based methods fail to sufficiently leverage the rich temporal information inherent in users' historical interaction sequences, stemming from fundamental architectural constraints: LLMs process information through self-attention mechanisms that lack inherent sequence ordering and rely on position embeddings designed primarily for natural language rather than user interaction sequences. This limitation significantly impairs their ability to capture the evolution of user preferences over time and predict future interests accurately.
To address this critical gap, we propose Counterfactual Enhanced Temporal Framework for LLM-Based Recommendation (CETRec). CETRec is grounded in causal inference principles, which allow it to isolate and measure the specific impact of temporal information on recommendation outcomes. By conceptualizing temporal order as an independent causal factor distinct from item content, we can quantify its unique contribution through counterfactual reasoning--comparing what recommendations would be made with and without temporal information while keeping all other factors constant. This causal framing enables CETRec to design a novel counterfactual tuning objective that directly optimizes the model's temporal sensitivity, teaching LLMs to recognize both absolute timestamps and relative ordering patterns in user histories. Combined with our counterfactual tuning task derived from causal analysis, CETRec effectively enhances LLMs' awareness of both absolute order (how recently items were interacted with) and relative order (the sequential relationships between items). 

**Abstract (ZH)**: 基于大型语言模型的计因增强时序推荐框架（CETRec） 

---
# K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function 

**Title (ZH)**: K-函数：联合发音转录与反馈以评估儿童语言功能 

**Authors**: Shuhe Li, Chenxu Guo, Jiachen Lian, Cheol Jun Cho, Wenshuo Zhao, Xuanru Zhou, Dingkun Zhou, Sam Wang, Grace Wang, Jingze Yang, Jingyi Xu, Ruohan Bao, Elise Brenner, Brandon In, Francesca Pei, Maria Luisa Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03043)  

**Abstract**: Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment. 

**Abstract (ZH)**: 早评价儿童语言能力受制于高音调、长音素和稀疏数据对自动语音识别系统的干扰。我们提出了K-Function，这是一个结合了准确的子词转录、客观评分和可操作反馈的统一框架。其核心组件Kids-WFST将Wav2Vec2音素编码器与音素相似性失言WFST相结合，以捕捉儿童特有的错误，同时保持完全可解释性。Kids-WFST在MyST上的音素错误率为1.39%，在Multitudes上的音素错误率为8.61%，分别比贪婪搜索解码器提高了10.47和7.06个百分点。这些高保真转录文本驱动了一个大型语言模型，用于评估口头技能、里程碑、阅读能力和理解能力，并与人类考官对齐，提供舌唇可视化及针对性建议。结果表明，精确的音素识别确立了完整的诊断反馈循环，为可扩展的、面向临床的语言评估铺平了道路。 

---
# Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction 

**Title (ZH)**: 基于动态长短期记忆的内存存储方案以支持长远 horizon LLM 交互 

**Authors**: Yuyang Lou, Charles Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03042)  

**Abstract**: Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）的内存存储成为一项日益活跃的研究领域，特别是在长对话中实现个性化方面。我们提出了一种名为Pref-LSTM的动态轻量级框架，该框架结合了基于BERT的分类器和一个基于LSTM的记忆模块，该模块生成记忆嵌入，然后将其软提示注入冻结的LLM中。我们合成了一组偏好和非偏好对话片段数据集来训练我们的基于BERT的分类器。尽管我们的基于LSTM的记忆编码器没有取得显著结果，但我们发现基于BERT的分类器在识别显性和隐性用户偏好方面表现可靠。我们的研究证明了使用偏好过滤和LSTM门控原理进行可扩展用户偏好建模的有效途径，无需大量额外开销和微调。 

---
# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards 

**Title (ZH)**: Optimas：以全局对齐的局部奖励优化复合AI系统 

**Authors**: Shirley Wu, Parth Sarthi, Shiyu Zhao, Aaron Lee, Herumb Shandilya, Adrian Mladenic Grobelnik, Nurendra Choudhary, Eddie Huang, Karthik Subbian, Linjun Zhang, Diyi Yang, James Zou, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2507.03041)  

**Abstract**: Compound AI systems integrating multiple components, such as Large Language Models, specialized tools, and traditional machine learning models, are increasingly deployed to solve complex real-world tasks. However, optimizing compound systems remains challenging due to their non-differentiable structures and diverse configuration types across components, including prompts, hyperparameters, and model parameters. To address this challenge, we propose Optimas, a unified framework for effective optimization of compound systems. The core idea of Optimas is to maintain one Local Reward Function (LRF) per component, each satisfying a local-global alignment property, i.e., each component's local reward correlates with the global system performance. In each iteration, Optimas efficiently adapts the LRFs to maintain this property while simultaneously maximizing each component's local reward. This approach enables independent updates of heterogeneous configurations using the designated optimization method, while ensuring that local improvements consistently lead to performance gains. We present extensive evaluations across five real-world compound systems to demonstrate that Optimas outperforms strong baselines by an average improvement of 11.92%, offering a general and effective approach for improving compound systems. Our website is at this https URL. 

**Abstract (ZH)**: 综合多个组件（如大型语言模型、专业工具和传统机器学习模型）的AI系统日益用于解决复杂的现实任务。然而，由于这些系统具有非可微结构并且各个组件（包括提示、超参数和模型参数）的配置类型多样，因此优化这些系统仍然是一个挑战。为了应对这一挑战，我们提出了一种名为Optimas的统一框架，用于有效优化复合系统。Optimas的核心思想是为每个组件维护一个局部奖励函数（LRF），每个LRF都满足局部-全局对齐属性，即每个组件的局部奖励与系统整体性能相关。在每次迭代中，Optimas高效地适应LRFs以保持这一属性，同时最大限度地提高每个组件的局部奖励。这种方法允许使用指定的优化方法独立更新异构配置，并确保局部改进始终带来性能提升。我们在五个实际的复合系统上进行了广泛的评估，结果表明，Optimas平均优于强基线11.92%，提供了一种通用且有效的方法来提升复合系统。我们的网站地址为：这个https URL。 

---
