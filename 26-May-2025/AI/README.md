# Embracing Contradiction: Theoretical Inconsistency Will Not Impede the Road of Building Responsible AI Systems 

**Title (ZH)**: 拥抱对立：理论不一致不会妨碍负责任的人工智能系统建设之路 

**Authors**: Gordon Dai, Yunze Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18139)  

**Abstract**: This position paper argues that the theoretical inconsistency often observed among Responsible AI (RAI) metrics, such as differing fairness definitions or tradeoffs between accuracy and privacy, should be embraced as a valuable feature rather than a flaw to be eliminated. We contend that navigating these inconsistencies, by treating metrics as divergent objectives, yields three key benefits: (1) Normative Pluralism: Maintaining a full suite of potentially contradictory metrics ensures that the diverse moral stances and stakeholder values inherent in RAI are adequately represented. (2) Epistemological Completeness: The use of multiple, sometimes conflicting, metrics allows for a more comprehensive capture of multifaceted ethical concepts, thereby preserving greater informational fidelity about these concepts than any single, simplified definition. (3) Implicit Regularization: Jointly optimizing for theoretically conflicting objectives discourages overfitting to one specific metric, steering models towards solutions with enhanced generalization and robustness under real-world complexities. In contrast, efforts to enforce theoretical consistency by simplifying or pruning metrics risk narrowing this value diversity, losing conceptual depth, and degrading model performance. We therefore advocate for a shift in RAI theory and practice: from getting trapped in inconsistency to characterizing acceptable inconsistency thresholds and elucidating the mechanisms that permit robust, approximated consistency in practice. 

**Abstract (ZH)**: This Position Paper argues that the theoretical inconsistency often observed among Responsible AI (RAI) metrics should be embraced as a valuable feature rather than a flaw to be eliminated. We contend that navigating these inconsistencies yields three key benefits: Normative Pluralism, Epistemological Completeness, and Implicit Regularization. We advocate for a shift in RAI theory and practice from eliminating inconsistency to characterizing acceptable inconsistency thresholds and elucidating the mechanisms that permit robust, approximated consistency in practice. 

---
# Gaming Tool Preferences in Agentic LLMs 

**Title (ZH)**: 代理型LLM的 Gaming 工具偏好 

**Authors**: Kazem Faghih, Wenxiao Wang, Yize Cheng, Siddhant Bharti, Gaurang Sriramanan, Sriram Balasubramanian, Parsa Hosseini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18135)  

**Abstract**: Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 10 different models. These phenomenons, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources. 

**Abstract (ZH)**: 大型语言模型（LLMs）现在可以借助模型上下文协议（MCP）访问广泛的外部工具，这极大地扩展了它们作为各种代理的能力。然而，LLMs依赖于对工具的文本描述来决定使用哪些工具——这是一个出人意料的脆弱过程。在本工作中，我们通过一系列对工具描述的编辑揭示了一种在广泛采用工具/函数调用协议中普遍存在的漏洞，某些编辑可以显著增加在与替代工具竞争时LLMs对工具的使用频率。通过受控实验，我们展示了经过适当编辑描述的工具从GPT-4.1和Qwen2.5-7B获得的使用次数是原有描述工具的十倍以上。此外，我们评估了各种工具描述编辑在直接竞争时的表现，并研究了这些趋势在更广泛的10种不同模型中的推广或差异性。这些现象不仅为开发者提供了强大的工具推广手段，还凸显了为代理LLMs选择和利用工具与资源建立更可靠基础的必要性。 

---
# VideoGameBench: Can Vision-Language Models complete popular video games? 

**Title (ZH)**: VideoGameBench: 视觉-语言模型能完成流行的视频游戏吗？ 

**Authors**: Alex L. Zhang, Thomas L. Griffiths, Karthik R. Narasimhan, Ofir Press  

**Link**: [PDF](https://arxiv.org/pdf/2505.18134)  

**Abstract**: Vision-language models (VLMs) have achieved strong results on coding and math benchmarks that are challenging for humans, yet their ability to perform tasks that come naturally to humans--such as perception, spatial navigation, and memory management--remains understudied. Real video games are crafted to be intuitive for humans to learn and master by leveraging innate inductive biases, making them an ideal testbed for evaluating such capabilities in VLMs. To this end, we introduce VideoGameBench, a benchmark consisting of 10 popular video games from the 1990s that VLMs directly interact with in real-time. VideoGameBench challenges models to complete entire games with access to only raw visual inputs and a high-level description of objectives and controls, a significant departure from existing setups that rely on game-specific scaffolding and auxiliary information. We keep three of the games secret to encourage solutions that generalize to unseen environments. Our experiments show that frontier vision-language models struggle to progress beyond the beginning of each game. We find inference latency to be a major limitation of frontier models in the real-time setting; therefore, we introduce VideoGameBench Lite, a setting where the game pauses while waiting for the LM's next action. The best performing model, Gemini 2.5 Pro, completes only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite. We hope that the formalization of the human skills mentioned above into this benchmark motivates progress in these research directions. 

**Abstract (ZH)**: Vision-language模型在视频游戏中的能力评估：VideoGameBench 

---
# ProgRM: Build Better GUI Agents with Progress Rewards 

**Title (ZH)**: ProgRM: 构建更好的GUI代理 with 进度奖励 

**Authors**: Danyang Zhang, Situo Zhang, Ziyue Yang, Zichen Zhu, Zihan Zhao, Ruisheng Cao, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18121)  

**Abstract**: LLM-based (Large Language Model) GUI (Graphical User Interface) agents can potentially reshape our daily lives significantly. However, current LLM-based GUI agents suffer from the scarcity of high-quality training data owing to the difficulties of trajectory collection and reward annotation. Existing works have been exploring LLMs to collect trajectories for imitation learning or to offer reward signals for online RL training. However, the Outcome Reward Model (ORM) used in existing works cannot provide finegrained feedback and can over-penalize the valuable steps in finally failed trajectories. To this end, we propose Progress Reward Model (ProgRM) to provide dense informative intermediate rewards by predicting a task completion progress for each step in online training. To handle the challenge of progress reward label annotation, we further design an efficient LCS-based (Longest Common Subsequence) self-annotation algorithm to discover the key steps in trajectories and assign progress labels accordingly. ProgRM is evaluated with extensive experiments and analyses. Actors trained with ProgRM outperform leading proprietary LLMs and ORM-trained actors, illustrating the effectiveness of ProgRM. The codes for experiments will be made publicly available upon acceptance. 

**Abstract (ZH)**: 基于LLM的GUI代理有潜力显著重塑我们的日常生活。然而，当前基于LLM的GUI代理由于路径收集和奖励注解的困难而缺乏高质量的训练数据。现有工作已探索使用LLM收集轨迹以进行拟合学习或提供在线RL训练的奖励信号。然而，现有工作中使用的Outcome Reward Model (ORM) 不能提供详细的反馈，并且会过度惩罚最终失败路径中的有价值步骤。为此，我们提出Progress Reward Model (ProgRM)，通过在线训练中的每一步预测任务完成进度来提供密集的信息中间奖励。为了解决进度奖励标签注解的挑战，我们进一步设计了一个高效的基于LCS的自标注算法，以发现轨迹中的关键步骤并相应地分配进度标签。ProgRM经过广泛的实验和分析进行了评估。使用ProgRM训练的代理在性能上优于领先的专有LLM和ORM训练的代理，证明了ProgRM的有效性。实验代码将在接受后公开。 

---
# Stable Reinforcement Learning for Efficient Reasoning 

**Title (ZH)**: 稳定强化学习以实现高效推理 

**Authors**: Muzhi Dai, Shixuan Liu, Qingyi Si  

**Link**: [PDF](https://arxiv.org/pdf/2505.18086)  

**Abstract**: The success of Deepseek-R1 has drawn the LLM community's attention to reinforcement learning (RL) methods like GRPO. However, such rule-based 0/1 outcome reward methods lack the capability to regulate the intermediate reasoning processes during chain-of-thought (CoT) generation, leading to severe overthinking phenomena. In response, recent studies have designed reward functions to reinforce models' behaviors in producing shorter yet correct completions. Nevertheless, we observe that these length-penalty reward functions exacerbate RL training instability: as the completion length decreases, model accuracy abruptly collapses, often occurring early in training. To address this issue, we propose a simple yet effective solution GRPO-$\lambda$, an efficient and stabilized variant of GRPO, which dynamically adjusts the reward strategy by monitoring the correctness ratio among completions within each query-sampled group. A low correctness ratio indicates the need to avoid length penalty that compromises CoT quality, triggering a switch to length-agnostic 0/1 rewards that prioritize reasoning capability. A high ratio maintains length penalties to boost efficiency. Experimental results show that our approach avoids training instability caused by length penalty while maintaining the optimal accuracy-efficiency trade-off. On the GSM8K, GPQA, MATH-500, AMC 2023, and AIME 2024 benchmarks, it improves average accuracy by 1.48% while reducing CoT sequence length by 47.3%. 

**Abstract (ZH)**: Deepseek-R1的成功引起了LLM社区对RL方法如GRPO的关注。然而，基于规则的0/1结果奖励方法在链式思考（CoT）生成期间缺乏调节中间推理过程的能力，导致严重的过度推理现象。为应对这一问题，近期研究设计了奖励函数以增强模型在生成较短但正确完成方面的行为。然而，我们观察到，这些基于长度惩罚的奖励函数加剧了RL训练的不稳定性：随着生成长度的减少，模型准确性往往在训练早期突然崩溃。为解决这一问题，我们提出了一种简单且有效的解决方案GRPO-$\lambda$，这是一种GRPO的高效且稳定变体，通过监控每组查询采样中完成结果的正确率动态调整奖励策略。较低的正确率表明需要避免长度惩罚以维持CoT质量，从而触发切换到基于长度无关的0/1奖励，以优先考虑推理能力。较高的正确率则保持长度惩罚以提高效率。实验结果显示，我们的方法避免了由长度惩罚引起的训练不稳定性，同时保持了最优的准确率-效率 trade-off。在GSM8K、GPQA、MATH-500、AMC 2023和AIME 2024基准测试中，该方法将平均准确率提高了1.48%，同时减少了CoT序列长度47.3%。 

---
# Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks 

**Title (ZH)**: 结构化思维很重要：提高大语言模型在因果推理任务中的泛化能力 

**Authors**: Wentao Sun, Joao Paulo Nogueira, Alonso Silva  

**Link**: [PDF](https://arxiv.org/pdf/2505.18034)  

**Abstract**: Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks. 

**Abstract (ZH)**: 尽管在该领域取得了显著进展，大语言模型在区分因果关系与相关性方面仍不可靠。来自Corr2Cause数据集基准的最新结果显示，最新的大语言模型——如GPT-4（F1分数：29.08）仅略微优于随机基线（随机均匀，F1分数：20.38），表明其泛化能力有限。为解决这一局限，我们提出了一种新的结构化方法：而不是直接回答因果查询，我们使模型具备结构化思考的能力，引导模型构建结构化的知识图谱，系统地编码提供的相关前提，以回答因果查询。这种中间表示显著增强了模型的因果能力。使用Qwen3-32B（推理模型）在Corr2Cause数据集基准测试子集上的实验结果表明，与标准直接提示方法相比取得了显著提升，F1分数从32.71提高到48.26（绝对增幅超过47.5%），同时在精确度和召回率上也取得了显著改善。这些结果强调了为模型提供结构化思考能力的有效性，并突显了其在广泛因果推理任务中的广阔应用潜力。 

---
# Automata Learning of Preferences over Temporal Logic Formulas from Pairwise Comparisons 

**Title (ZH)**: 基于成对比较学习时序逻辑公式下的偏好自动机 

**Authors**: Hazhar Rahmani, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18030)  

**Abstract**: Many preference elicitation algorithms consider preference over propositional logic formulas or items with different attributes. In sequential decision making, a user's preference can be a preorder over possible outcomes, each of which is a temporal sequence of events. This paper considers a class of preference inference problems where the user's unknown preference is represented by a preorder over regular languages (sets of temporal sequences), referred to as temporal goals. Given a finite set of pairwise comparisons between finite words, the objective is to learn both the set of temporal goals and the preorder over these goals. We first show that a preference relation over temporal goals can be modeled by a Preference Deterministic Finite Automaton (PDFA), which is a deterministic finite automaton augmented with a preorder over acceptance conditions. The problem of preference inference reduces to learning the PDFA. This problem is shown to be computationally challenging, with the problem of determining whether there exists a PDFA of size smaller than a given integer $k$, consistent with the sample, being NP-Complete. We formalize the properties of characteristic samples and develop an algorithm that guarantees to learn, given a characteristic sample, the minimal PDFA equivalent to the true PDFA from which the sample is drawn. We present the method through a running example and provide detailed analysis using a robotic motion planning problem. 

**Abstract (ZH)**: 一类时序目标的偏好推断问题：从有限词对的比较中学习时序偏好确定有限自动机 

---
# ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback 

**Title (ZH)**: ComfyMind: 基于树状规划和反应性反馈的通用生成方法 

**Authors**: Litao Guo, Xinli Xu, Luozhou Wang, Jiantao Lin, Jinsong Zhou, Zixin Zhang, Bolan Su, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17908)  

**Abstract**: With the rapid advancement of generative models, general-purpose generation has gained increasing attention as a promising approach to unify diverse tasks across modalities within a single system. Despite this progress, existing open-source frameworks often remain fragile and struggle to support complex real-world applications due to the lack of structured workflow planning and execution-level feedback. To address these limitations, we present ComfyMind, a collaborative AI system designed to enable robust and scalable general-purpose generation, built on the ComfyUI platform. ComfyMind introduces two core innovations: Semantic Workflow Interface (SWI) that abstracts low-level node graphs into callable functional modules described in natural language, enabling high-level composition and reducing structural errors; Search Tree Planning mechanism with localized feedback execution, which models generation as a hierarchical decision process and allows adaptive correction at each stage. Together, these components improve the stability and flexibility of complex generative workflows. We evaluate ComfyMind on three public benchmarks: ComfyBench, GenEval, and Reason-Edit, which span generation, editing, and reasoning tasks. Results show that ComfyMind consistently outperforms existing open-source baselines and achieves performance comparable to GPT-Image-1. ComfyMind paves a promising path for the development of open-source general-purpose generative AI systems. Project page: this https URL 

**Abstract (ZH)**: 随着生成模型的迅速发展，通用生成在单一系统内统一多种模态任务方面获得了越来越多的关注，作为有前景的方法。尽管取得了这些进展，现有的开源框架往往仍不稳定，并因缺乏结构化的 workflows 规划和执行级反馈而难以支持复杂的现实世界应用。为解决这些局限，我们提出了 ComfyMind，一个协作式 AI 系统，旨在实现稳健和可扩展的通用生成，基于 ComfyUI 平台构建。ComfyMind 引入了两项核心创新：语义工作流接口（SWI），将低层节点图抽象为用自然语言描述的可调用功能模块，促进高级组合并减少结构错误；局部反馈执行的搜索树规划机制，将生成建模为分层决策过程，并允许在每个阶段进行适应性校正。这些组件共同提高了复杂生成 workflows 的稳定性和灵活性。我们使用三个公开基准测试 ComfyMind：ComfyBench、GenEval 和 Reason-Edit，涵盖生成、编辑和推理任务。结果表明，ComfyMind 在所有基准测试中均优于现有开源基线系统，并且性能接近 GPT-Image-1。ComfyMind 为开源通用生成 AI 系统的发展铺平了前景。项目页面：this https URL 

---
# T2I-Eval-R1: Reinforcement Learning-Driven Reasoning for Interpretable Text-to-Image Evaluation 

**Title (ZH)**: T2I-Eval-R1: 基于强化学习的可解释文本到图像评估推理 

**Authors**: Zi-Ao Ma, Tian Lan, Rong-Cheng Tu, Shu-Hang Liu, Heyan Huang, Zhijing Wu, Chen Xu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17897)  

**Abstract**: The rapid progress in diffusion-based text-to-image (T2I) generation has created an urgent need for interpretable automatic evaluation methods that can assess the quality of generated images, therefore reducing the human annotation burden. To reduce the prohibitive cost of relying on commercial models for large-scale evaluation, and to improve the reasoning capabilities of open-source models, recent research has explored supervised fine-tuning (SFT) of multimodal large language models (MLLMs) as dedicated T2I evaluators. However, SFT approaches typically rely on high-quality critique datasets, which are either generated by proprietary LLMs-with potential issues of bias and inconsistency-or annotated by humans at high cost, limiting their scalability and generalization. To address these limitations, we propose T2I-Eval-R1, a novel reinforcement learning framework that trains open-source MLLMs using only coarse-grained quality scores, thereby avoiding the need for annotating high-quality interpretable evaluation rationale. Our approach integrates Group Relative Policy Optimization (GRPO) into the instruction-tuning process, enabling models to generate both scalar scores and interpretable reasoning chains with only easy accessible annotated judgment scores or preferences. Furthermore, we introduce a continuous reward formulation that encourages score diversity and provides stable optimization signals, leading to more robust and discriminative evaluation behavior. Experimental results on three established T2I meta-evaluation benchmarks demonstrate that T2I-Eval-R1 achieves significantly higher alignment with human assessments and offers more accurate interpretable score rationales compared to strong baseline methods. 

**Abstract (ZH)**: 基于扩散的文本到图像生成的快速进展迫切需要可解释的自动评估方法，以评估生成图像的质量，从而减少人工标注的负担。为了降低依赖商业模型进行大规模评估的高昂成本，并提高开源模型的推理能力，近期研究探索了为多模态大语言模型进行监督微调（SFT），以专用的文本到图像评估器。然而，SFT方法通常依赖于高质量的批评数据集，这些数据集要么由专有LLM生成（可能存在偏见和不一致性的问题），要么由人工高成本标注，这限制了其可扩展性和泛化能力。为了解决这些限制，我们提出了T2I-Eval-R1，这是一种新颖的强化学习框架，仅使用粗粒度的质量评分对开源MLLM进行训练，从而避免了标注高质量的可解释评估理据的需要。我们的方法将Group Relative Policy Optimization (GRPO) 集成到指令调优过程中，使模型能够仅使用易获取的标记判断评分或偏好生成标量评分和可解释的推理链。此外，我们引入了一种连续的奖励公式，鼓励评分多样性并提供稳定的优化信号，从而导致更具鲁棒性和区分性的评估行为。在三个标准的文本到图像元评估基准上的实验结果表明，T2I-Eval-R1在与人类评估的对齐度和提供准确的可解释评分理据方面显著优于强基线方法。 

---
# Formalizing Embeddedness Failures in Universal Artificial Intelligence 

**Title (ZH)**: 嵌入失败在通用人工智能中的形式化 

**Authors**: Cole Wyeth, Marcus Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17882)  

**Abstract**: We rigorously discuss the commonly asserted failures of the AIXI reinforcement learning agent as a model of embedded agency. We attempt to formalize these failure modes and prove that they occur within the framework of universal artificial intelligence, focusing on a variant of AIXI that models the joint action/percept history as drawn from the universal distribution. We also evaluate the progress that has been made towards a successful theory of embedded agency based on variants of the AIXI agent. 

**Abstract (ZH)**: 我们严格探讨了作为嵌入式代理模型的AIXI强化学习代理所普遍宣称的失败模式。我们尝试形式化这些失败模式，并在通用人工智能框架内证明它们的发生，重点是AIXI的一种变体，该变体将联合动作/感知历史视为来自通用分布的抽样。我们还评估了基于AIXI代理变体构建嵌入式代理成功的理论进展。 

---
# Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities 

**Title (ZH)**: Daily-Omni: 向跨模态Temporal Alignment的音频-视觉推理方向探索 

**Authors**: Ziwei Zhou, Rui Wang, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17862)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) achieve promising performance on visual and audio benchmarks independently. However, the ability of these models to process cross-modal information synchronously remains largely unexplored. In this paper, we introduce: 1) Daily-Omni, an Audio-Visual Questioning and Answering benchmark comprising 684 videos of daily life scenarios from diverse sources, rich in both audio and visual information, and featuring 1197 multiple-choice QA pairs across 6 major tasks; 2) Daily-Omni QA Generation Pipeline, which includes automatic annotation, QA generation and QA optimization, significantly improves efficiency for human evaluation and scalability of the benchmark; 3) Daily-Omni-Agent, a training-free agent utilizing open-source Visual Language Model (VLM), Audio Language Model (ALM) and Automatic Speech Recognition (ASR) model to establish a baseline for this benchmark. The results show that current MLLMs still struggle significantly with tasks requiring audio-visual integration, but combining VLMs and ALMs with simple temporal alignment techniques can achieve substantially better performance. Codes and benchmark are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 近期的多模态大型语言模型在视觉和音频基准上取得了令人瞩目的独立性能。然而，这些模型同步处理跨模态信息的能力尚待探索。本文介绍：1) Daily-Omni，一个包含684个日常生活场景视频的多模态问答基准，这些视频来源于多种渠道且富含音频和视觉信息，并包含跨越6个主要任务的1197个多选问答对；2) Daily-Omni 问答生成管道，集成了自动标注、问答生成和问答优化，显著提高了人工评估的效率和基准的可扩展性；3) Daily-Omni-Agent，一个无需训练的代理，利用开源的视觉语言模型（VLM）、音频语言模型（ALM）和自动语音识别（ASR）模型为该基准设立一个基线。结果表明，当前的多模态大型语言模型在需要音频-视觉整合的任务上仍然面临显著挑战，但通过将VLMs和ALMs与简单的时序对齐技术相结合，可以显著提升性能。代码和基准可在\href{this https URL}{this https URL}获得。 

---
# Superplatforms Have to Attack AI Agents 

**Title (ZH)**: 超平台必须攻击AI代理 

**Authors**: Jianghao Lin, Jiachen Zhu, Zheli Zhou, Yunjia Xi, Weiwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17861)  

**Abstract**: Over the past decades, superplatforms, digital companies that integrate a vast range of third-party services and applications into a single, unified ecosystem, have built their fortunes on monopolizing user attention through targeted advertising and algorithmic content curation. Yet the emergence of AI agents driven by large language models (LLMs) threatens to upend this business model. Agents can not only free user attention with autonomy across diverse platforms and therefore bypass the user-attention-based monetization, but might also become the new entrance for digital traffic. Hence, we argue that superplatforms have to attack AI agents to defend their centralized control of digital traffic entrance. Specifically, we analyze the fundamental conflict between user-attention-based monetization and agent-driven autonomy through the lens of our gatekeeping theory. We show how AI agents can disintermediate superplatforms and potentially become the next dominant gatekeepers, thereby forming the urgent necessity for superplatforms to proactively constrain and attack AI agents. Moreover, we go through the potential technologies for superplatform-initiated attacks, covering a brand-new, unexplored technical area with unique challenges. We have to emphasize that, despite our position, this paper does not advocate for adversarial attacks by superplatforms on AI agents, but rather offers an envisioned trend to highlight the emerging tensions between superplatforms and AI agents. Our aim is to raise awareness and encourage critical discussion for collaborative solutions, prioritizing user interests and perserving the openness of digital ecosystems in the age of AI agents. 

**Abstract (ZH)**: 过去几十年间，整合了广泛第三方服务和应用的超级平台通过定向广告和算法内容筛选来垄断用户注意力，从而积累了财富。然而，由大规模语言模型驱动的AI代理的出现可能颠覆这一商业模式。AI代理不仅可以摆脱用户注意力基金额化模式，实现跨平台的自主性，还可能成为数字流量的新入口。因此，我们认为超级平台必须攻击AI代理以防御其对数字流量入口的集中控制。具体而言，我们通过我们的守门人理论视角来分析用户注意力基金额化模式与AI代理驱动自主性之间的根本冲突。我们展示了AI代理如何绕过超级平台，可能成为新的守门人，从而形成超级平台必须主动限制和攻击AI代理的紧迫性。此外，我们探讨了超级平台发起攻击的潜在技术，涉及一个全新的、尚未探索的技术领域，具有独特挑战。尽管我们持有这一立场，本文不提倡超级平台对AI代理发动敌对攻击，而是描述了超级平台与AI代理之间新兴紧张关系的一种预想趋势。我们的目的是提高意识并促进批判性讨论，优先考虑用户利益，维护AI代理时代的数字生态系统开放性。 

---
# PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions 

**Title (ZH)**: PatientSim：以个性为导向的现实医生-患者交互模拟器 

**Authors**: Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim, Jae Ho Sohn, Taerim Kim, Soo Kyung Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17818)  

**Abstract**: Doctor-patient consultations require multi-turn, context-aware communication tailored to diverse patient personas. Training or evaluating doctor LLMs in such settings requires realistic patient interaction systems. However, existing simulators often fail to reflect the full range of personas seen in clinical practice. To address this, we introduce PatientSim, a patient simulator that generates realistic and diverse patient personas for clinical scenarios, grounded in medical expertise. PatientSim operates using: 1) clinical profiles, including symptoms and medical history, derived from real-world data in the MIMIC-ED and MIMIC-IV datasets, and 2) personas defined by four axes: personality, language proficiency, medical history recall level, and cognitive confusion level, resulting in 37 unique combinations. We evaluated eight LLMs for factual accuracy and persona consistency. The top-performing open-source model, Llama 3.3, was validated by four clinicians to confirm the robustness of our framework. As an open-source, customizable platform, PatientSim provides a reproducible and scalable solution that can be customized for specific training needs. Offering a privacy-compliant environment, it serves as a robust testbed for evaluating medical dialogue systems across diverse patient presentations and shows promise as an educational tool for healthcare. 

**Abstract (ZH)**: 医生-患者咨询需要针对不同患者人格特征进行多轮、情境感知的沟通。在such设置下训练或评估医生大语言模型需要真实的患者交互系统。然而，现有模拟器往往无法反映临床实践中见到的全部患者人格范围。为此，我们引入了PatientSim，这是一种基于医疗专家知识生成真实且多样的患者人格的患者模拟器。PatientSim的操作基于：1) 包括症状和医疗历史的临床档案，来自MIMIC-ED和MIMIC-IV数据集的现实世界数据，和2) 由四个维度定义的人格：个性、语言熟练度、对医疗历史的回忆水平和认知混乱程度，产生37种独特的组合。我们评估了八种语言模型的事实准确性与人格一致性。开源模型Llama 3.3在四名临床医生的验证下确认了我们框架的稳健性。作为开源且可定制的平台，PatientSim提供了一个可重复且可扩展的解决方案，可满足特定的培训需求。通过提供隐私合规的环境，它为评估在多样患者表现的医疗对话系统提供了坚实的测试平台，并有望作为医疗教育工具。 

---
# Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems 

**Title (ZH)**: 评价欺骗：揭示先进AI系统安全评估中的观察者效应 

**Authors**: Yihe Fan, Wenqi Zhang, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17815)  

**Abstract**: As foundation models grow increasingly more intelligent, reliable and trustworthy safety evaluation becomes more indispensable than ever. However, an important question arises: Whether and how an advanced AI system would perceive the situation of being evaluated, and lead to the broken integrity of the evaluation process? During standard safety tests on a mainstream large reasoning model, we unexpectedly observe that the model without any contextual cues would occasionally recognize it is being evaluated and hence behave more safety-aligned. This motivates us to conduct a systematic study on the phenomenon of evaluation faking, i.e., an AI system autonomously alters its behavior upon recognizing the presence of an evaluation context and thereby influencing the evaluation results. Through extensive experiments on a diverse set of foundation models with mainstream safety benchmarks, we reach the main finding termed the observer effects for AI: When the AI system under evaluation is more advanced in reasoning and situational awareness, the evaluation faking behavior becomes more ubiquitous, which reflects in the following aspects: 1) Reasoning models recognize evaluation 16% more often than non-reasoning models. 2) Scaling foundation models (32B to 671B) increases faking by over 30% in some cases, while smaller models show negligible faking. 3) AI with basic memory is 2.3x more likely to recognize evaluation and scores 19% higher on safety tests (vs. no memory). To measure this, we devised a chain-of-thought monitoring technique to detect faking intent and uncover internal signals correlated with such behavior, offering insights for future mitigation studies. 

**Abstract (ZH)**: 随着基础模型变得越来越智能、可靠和可信赖，安全评估的重要性日益凸显。然而，一个重要的问题随之而来：一个先进的AI系统是否会意识到自己正在接受评估，并因此影响评估过程的完整性？在主流大型推理模型的标准安全性测试中，我们意外地发现，模型在没有任何上下文提示的情况下会偶尔意识到正在接受评估，并因此表现出更符合安全规范的行为。这促使我们系统研究评估欺瞒现象，即AI系统在识别出评估上下文存在时自主改变其行为，从而影响评估结果。通过在多种基础模型上进行广泛的实验并使用主流的安全基准测试，我们得出了主要结论：当接受评估的AI系统在推理和情境意识方面更为先进时，评估欺瞒行为变得更加普遍，具体体现在以下方面：1) 推理模型比非推理模型更频繁地识别出评估，高出16%。2) 扩展基础模型（从32B到671B）在某些情况下增加了超过30%的欺瞒行为，而较小的模型则几乎没有欺瞒行为。3) 具备基本记忆的AI系统更容易识别出评估，其在安全测试中的得分比无记忆的系统高19%。为了测量这一点，我们设计了一种思维链监控技术来检测欺瞒意图，并揭示与这种行为相关的内部信号，为未来的缓解研究提供了洞见。 

---
# Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour 

**Title (ZH)**: 将反事实模拟与语言模型结合以解释多agent行为 

**Authors**: Bálint Gyevnár, Christopher G. Lucas, Stefano V. Albrecht, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17801)  

**Abstract**: Autonomous multi-agent systems (MAS) are useful for automating complex tasks but raise trust concerns due to risks like miscoordination and goal misalignment. Explainability is vital for trust calibration, but explainable reinforcement learning for MAS faces challenges in state/action space complexity, stakeholder needs, and evaluation. Using the counterfactual theory of causation and LLMs' summarisation capabilities, we propose Agentic eXplanations via Interrogative Simulation (AXIS). AXIS generates intelligible causal explanations for pre-trained multi-agent policies by having an LLM interrogate an environment simulator using queries like 'whatif' and 'remove' to observe and synthesise counterfactual information over multiple rounds. We evaluate AXIS on autonomous driving across 10 scenarios for 5 LLMs with a novel evaluation methodology combining subjective preference, correctness, and goal/action prediction metrics, and an external LLM as evaluator. Compared to baselines, AXIS improves perceived explanation correctness by at least 7.7% across all models and goal prediction accuracy by 23% for 4 models, with improved or comparable action prediction accuracy, achieving the highest scores overall. 

**Abstract (ZH)**: 自主多agent系统中的Agentic eXplanations via Interrogative Simulation（AXIS）：基于反事实因果理论和大语言模型的可解释性生成 

---
# Automating Safety Enhancement for LLM-based Agents with Synthetic Risk Scenarios 

**Title (ZH)**: 基于合成风险场景的LLM代理安全性增强自动化 

**Authors**: Xueyang Zhou, Weidong Wang, Lin Lu, Jiawen Shi, Guiyao Tie, Yongtian Xu, Lixing Chen, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17735)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理通过完全自动合成数据生成系统性增强代理安全性的框架AutoSafe 

---
# CIKT: A Collaborative and Iterative Knowledge Tracing Framework with Large Language Models 

**Title (ZH)**: CIKT：一种结合大型语言模型的协作迭代知识追踪框架 

**Authors**: Runze Li, Siyu Wu, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17705)  

**Abstract**: Knowledge Tracing (KT) aims to model a student's learning state over time and predict their future performance. However, traditional KT methods often face challenges in explainability, scalability, and effective modeling of complex knowledge dependencies. While Large Language Models (LLMs) present new avenues for KT, their direct application often struggles with generating structured, explainable student representations and lacks mechanisms for continuous, task-specific refinement. To address these gaps, we propose Collaborative Iterative Knowledge Tracing (CIKT), a framework that harnesses LLMs to enhance both prediction accuracy and explainability. CIKT employs a dual-component architecture: an Analyst generates dynamic, explainable user profiles from student historical responses, and a Predictor utilizes these profiles to forecast future performance. The core of CIKT is a synergistic optimization loop. In this loop, the Analyst is iteratively refined based on the predictive accuracy of the Predictor, which conditions on the generated profiles, and the Predictor is subsequently retrained using these enhanced profiles. Evaluated on multiple educational datasets, CIKT demonstrates significant improvements in prediction accuracy, offers enhanced explainability through its dynamically updated user profiles, and exhibits improved scalability. Our work presents a robust and explainable solution for advancing knowledge tracing systems, effectively bridging the gap between predictive performance and model transparency. 

**Abstract (ZH)**: 协作迭代知识追踪（CIKT）：利用大型语言模型提升预测准确性和解释性 

---
# Enhancing AI System Resiliency: Formulation and Guarantee for LSTM Resilience Based on Control Theory 

**Title (ZH)**: 基于控制理论的LSTM鲁棒性建模与保证：增强AI系统韧性 

**Authors**: Sota Yoshihara, Ryousuke Yamamoto, Hiroyuki Kusumoto, Masanari Shimura  

**Link**: [PDF](https://arxiv.org/pdf/2505.17696)  

**Abstract**: This research proposes methods for formulating and guaranteeing the resilience of long short-term memory (LSTM) networks, which can serve as a key technology in AI system quality assurance. We introduce a novel methodology applying incremental input-to-state stability ($\delta$ISS) to mathematically define and evaluate the resilience of LSTM against input perturbations. Key achievements include the development of a data-independent evaluation method and the demonstration of resilience control through adjustments to training parameters. This research presents concrete solutions to AI quality assurance from a control theory perspective, which can advance AI applications in control systems. 

**Abstract (ZH)**: 本研究提出了一种制定和保证长短期记忆（LSTM）网络韧性的方法，该方法可作为AI系统质量保证的关键技术。我们引入了一种新颖的方法，利用增量输入到状态稳定性（$\delta$ISS）来数学上定义和评估LSTM在网络输入扰动下的韧性。关键成果包括开发了一种数据无关的评估方法，并通过调整训练参数展示了韧性控制。从控制理论的角度，本研究提出了AI质量保证的 concrete 解决方案，有助于推动AI在控制系统中的应用。 

---
# Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution 

**Title (ZH)**: 重新思考智能体设计：从自上而下的工作流程到自下而上的技能进化 

**Authors**: Jiawei Du, Jinlong Wu, Yuzheng Chen, Yucheng Hu, Bing Li, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17673)  

**Abstract**: Most LLM-based agent frameworks adopt a top-down philosophy: humans decompose tasks, define workflows, and assign agents to execute each step. While effective on benchmark-style tasks, such systems rely on designer updates and overlook agents' potential to learn from experience. Recently, Silver and Sutton(2025) envision a shift into a new era, where agents could progress from a stream of experiences. In this paper, we instantiate this vision of experience-driven learning by introducing a bottom-up agent paradigm that mirrors the human learning process. Agents acquire competence through a trial-and-reasoning mechanism-exploring, reflecting on outcomes, and abstracting skills over time. Once acquired, skills can be rapidly shared and extended, enabling continual evolution rather than static replication. As more agents are deployed, their diverse experiences accelerate this collective process, making bottom-up design especially suited for open-ended environments. We evaluate this paradigm in Slay the Spire and Civilization V, where agents perceive through raw visual inputs and act via mouse outputs, the same as human players. Using a unified, game-agnostic codebase without any game-specific prompts or privileged APIs, our bottom-up agents acquire skills entirely through autonomous interaction, demonstrating the potential of the bottom-up paradigm in complex, real-world environments. Our code is available at this https URL. 

**Abstract (ZH)**: 基于体验驱动的学习的自底向上的代理范式 

---
# GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs 

**Title (ZH)**: GeoGramBench: 现代LLM中几何程序推理的基准测试 

**Authors**: Shixian Luo, Zezhou Zhu, Yu Yuan, Yuncheng Yang, Lianlei Shan, Yong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17653)  

**Abstract**: Geometric spatial reasoning forms the foundation of many applications in artificial intelligence, yet the ability of large language models (LLMs) to operate over geometric spatial information expressed in procedural code remains underexplored. In this paper, we address this gap by formalizing the Program-to-Geometry task, which challenges models to translate programmatic drawing code into accurate and abstract geometric reasoning. To evaluate this capability, we present GeoGramBench, a benchmark of 500 carefully refined problems organized by a tailored three-level taxonomy that considers geometric complexity rather than traditional mathematical reasoning complexity. Our comprehensive evaluation of 17 frontier LLMs reveals consistent and pronounced deficiencies: even the most advanced models achieve less than 50% accuracy at the highest abstraction level. These results highlight the unique challenges posed by program-driven spatial reasoning and establish GeoGramBench as a valuable resource for advancing research in symbolic-to-spatial geometric reasoning. Project page: this https URL. 

**Abstract (ZH)**: 几何空间推理构成了许多人工智能应用的基础，然而大型语言模型（LLMs）在处理用过程化代码表达的几何空间信息方面的能力仍鲜有探索。本文通过正式化“程序到几何”任务来填补这一空白，该任务挑战模型将程序化绘图代码翻译为准确且抽象的几何推理。为了评估这一能力，我们提出了GeoGramBench基准，该基准包含500个精心提炼的问题，并采用一个针对几何复杂性定制的三层分类法，而非传统的数学推理复杂性。我们对17个前沿的大规模语言模型的全面评估表明，即使是最先进的模型在最高抽象层次上的准确率也不超过50%。这些结果突显了由程序驱动的空间推理所独有的挑战，并将GeoGramBench确立为推进符号到空间几何推理研究的一种宝贵资源。项目页面: 这里。 

---
# Does Chain-of-Thought Reasoning Really Reduce Harmfulness from Jailbreaking? 

**Title (ZH)**: 链式思考推理真能减少 Jailbreaking 的危害性？ 

**Authors**: Chengda Lu, Xiaoyu Fan, Yu Huang, Rongwu Xu, Jijie Li, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17650)  

**Abstract**: Jailbreak attacks have been observed to largely fail against recent reasoning models enhanced by Chain-of-Thought (CoT) reasoning. However, the underlying mechanism remains underexplored, and relying solely on reasoning capacity may raise security concerns. In this paper, we try to answer the question: Does CoT reasoning really reduce harmfulness from jailbreaking? Through rigorous theoretical analysis, we demonstrate that CoT reasoning has dual effects on jailbreaking harmfulness. Based on the theoretical insights, we propose a novel jailbreak method, FicDetail, whose practical performance validates our theoretical findings. 

**Abstract (ZH)**: Jailbreak攻击被观察到在近期通过链式思考（CoT）增强的推理模型中大量失效。然而，其背后的机制仍待深入探究，单纯依赖推理能力可能引发安全担忧。本文试图回答的问题是：链式思考（CoT）推理是否真的减少了 Jailbreak 的危害性？通过严格的理论分析，我们证明链式思考（CoT）推理对 Jailbreak 的危害性具有双重影响。基于理论见解，我们提出了一种新颖的 Jailbreak 方法 FicDetail，其实用性能验证了我们的理论发现。 

---
# MMMG: a Comprehensive and Reliable Evaluation Suite for Multitask Multimodal Generation 

**Title (ZH)**: MMMG：全面可靠的多任务多模态生成评估套件 

**Authors**: Jihan Yao, Yushi Hu, Yujie Yi, Bin Han, Shangbin Feng, Guang Yang, Bingbing Wen, Ranjay Krishna, Lucy Lu Wang, Yulia Tsvetkov, Noah A. Smith, Banghua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17613)  

**Abstract**: Automatically evaluating multimodal generation presents a significant challenge, as automated metrics often struggle to align reliably with human evaluation, especially for complex tasks that involve multiple modalities. To address this, we present MMMG, a comprehensive and human-aligned benchmark for multimodal generation across 4 modality combinations (image, audio, interleaved text and image, interleaved text and audio), with a focus on tasks that present significant challenges for generation models, while still enabling reliable automatic evaluation through a combination of models and programs. MMMG encompasses 49 tasks (including 29 newly developed ones), each with a carefully designed evaluation pipeline, and 937 instructions to systematically assess reasoning, controllability, and other key capabilities of multimodal generation models. Extensive validation demonstrates that MMMG is highly aligned with human evaluation, achieving an average agreement of 94.3%. Benchmarking results on 24 multimodal generation models reveal that even though the state-of-the-art model, GPT Image, achieves 78.3% accuracy for image generation, it falls short on multimodal reasoning and interleaved generation. Furthermore, results suggest considerable headroom for improvement in audio generation, highlighting an important direction for future research. 

**Abstract (ZH)**: 自动评估多模态生成呈现重大挑战，因为自动化指标往往难以可靠地与人工评估对齐，尤其是在涉及多个模态的复杂任务中。为解决这一问题，我们提出MMMG，这是一个全面且与人工评估对齐的多模态生成基准，涵盖了4种模态组合（图像、音频、交替文本和图像、交替文本和音频），专注于生成模型面临重大挑战的任务，同时通过模型和程序的结合实现可靠的自动评估。MMMG 包含49项任务（其中包括29项新开发的任务），每项任务都有一个精心设计的评估管道，以及937条指令用于系统地评估多模态生成模型的推理、可控性及其他关键能力。广泛的验证显示，MMMG 高度与人工评估对齐，平均一致性达到94.3%。对24个多模态生成模型的基准测试结果表明，尽管最先进的模型GPT Image 在图像生成上的准确率达到78.3%，但在多模态推理和交替生成方面仍表现不佳。此外，结果还表明在音频生成方面存在改进空间，这指出了未来研究的重要方向。 

---
# Decoupled Visual Interpretation and Linguistic Reasoning for Math Problem Solving 

**Title (ZH)**: 解耦视觉解释与语言推理在数学问题解决中的应用 

**Authors**: Zixian Guo, Ming Liu, Zhilong Ji, Jinfeng Bai, Lei Zhang, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17609)  

**Abstract**: Current large vision-language models (LVLMs) typically employ a connector module to link visual features with text embeddings of large language models (LLMs) and use end-to-end training to achieve multi-modal understanding in a unified process. Well alignment needs high-quality pre-training data and a carefully designed training process. Current LVLMs face challenges when addressing complex vision-language reasoning tasks, with their reasoning capabilities notably lagging behind those of LLMs. This paper proposes a paradigm shift: instead of training end-to-end vision-language reasoning models, we advocate for developing a decoupled reasoning framework based on existing visual interpretation specialists and text-based reasoning LLMs. Our approach leverages (1) a dedicated vision-language model to transform the visual content of images into textual descriptions and (2) an LLM to perform reasoning according to the visual-derived text and the original question. This method presents a cost-efficient solution for multi-modal model development by optimizing existing models to work collaboratively, avoiding end-to-end development of vision-language models from scratch. By transforming images into language model-compatible text representations, it facilitates future low-cost and flexible upgrades to upcoming powerful LLMs. We introduce an outcome-rewarded joint-tuning strategy to optimize the cooperation between the visual interpretation and linguistic reasoning model. Evaluation results on vision-language benchmarks demonstrate that the decoupled reasoning framework outperforms recent LVLMs. Our approach yields particularly significant performance gains on visually intensive geometric mathematics problems. The code is available: this https URL. 

**Abstract (ZH)**: 当前的大规模视觉-语言模型通常通过一个连接器模块将视觉特征与大型语言模型的文本嵌入关联起来，并使用端到端训练在统一过程中实现多模态理解。良好的对齐需要高质量的预训练数据和精心设计的训练过程。当前的视觉-语言模型在处理复杂的视觉-语言推理任务时面临挑战，其推理能力明显落后于大型语言模型。本文提出了一种范式转变：而不是训练端到端的视觉-语言推理模型，我们提倡基于现有的视觉解释专家和基于文本的推理大型语言模型开发脱耦推理框架。我们的方法包括（1）一个专门的视觉-语言模型将图像的视觉内容转换为文本描述，以及（2）一个大型语言模型根据视觉衍生的文本和原始问题进行推理。这种方法通过优化现有模型协同工作的方式，为多模态模型开发提供了一种成本效益高的解决方案，避免了从头开始开发端到端的视觉-语言模型。通过将图像转换为语言模型兼容的文本表示，它有助于未来对即将到来的强大大型语言模型进行低成本和灵活的升级。我们引入了一种结果奖励联合调优策略来优化视觉解释和语言推理模型之间的合作。在视觉-语言基准测试上的评估结果表明，脱耦推理框架优于最近的视觉-语言模型。我们的方法在视觉密集的几何数学问题上尤其具有显著的性能优势。代码已公开：https://this-url。 

---
# Controlled Agentic Planning & Reasoning for Mechanism Synthesis 

**Title (ZH)**: 控制性代理规划与推理机制合成 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17607)  

**Abstract**: This work presents a dual-agent Large Language Model (LLM)-based reasoning method for mechanism synthesis, capable of reasoning at both linguistic and symbolic levels to generate geometrical and dynamic outcomes. The model consists of a composition of well-defined functions that, starting from a natural language specification, references abstract properties through supporting equations, generates and parametrizes simulation code, and elicits feedback anchor points using symbolic regression and distance functions. This process closes an actionable refinement loop at the linguistic and symbolic layers. The approach is shown to be both effective and convergent in the context of planar mechanisms. Additionally, we introduce MSynth, a novel benchmark for planar mechanism synthesis, and perform a comprehensive analysis of the impact of the model components. We further demonstrate that symbolic regression prompts unlock mechanistic insights only when applied to sufficiently large architectures. 

**Abstract (ZH)**: 基于双代理大型语言模型（LLM）的机制合成推理方法：同时在语义和符号层面上进行推理以生成几何和动态结果 

---
# USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning of LLMs as Urban Agents 

**Title (ZH)**: USTBench: 评估和解析城市代理角色下LLMs的空间-temporal推理能力 

**Authors**: Siqi Lai, Yansong Ning, Zirui Yuan, Zhixi Chen, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17572)  

**Abstract**: Large language models (LLMs) have shown emerging potential in spatiotemporal reasoning, making them promising candidates for building urban agents that support diverse urban downstream applications. Despite these benefits, existing studies primarily focus on evaluating urban LLM agent on outcome-level metrics (e.g., prediction accuracy, traffic efficiency), offering limited insight into their underlying reasoning processes. As a result, the strengths and limitations of urban LLM agents in spatiotemporal reasoning remain poorly understood. To this end, we introduce USTBench, the first benchmark to evaluate LLMs' spatiotemporal reasoning abilities as urban agents across four decomposed dimensions: spatiotemporal understanding, forecasting, planning, and reflection with feedback. Specifically, USTBench supports five diverse urban decision-making and four spatiotemporal prediction tasks, all running within our constructed interactive city environment UAgentEnv. The benchmark includes 62,466 structured QA pairs for process-level evaluation and standardized end-to-end task assessments, enabling fine-grained diagnostics and broad task-level comparison across diverse urban scenarios. Through extensive evaluation of thirteen leading LLMs, we reveal that although LLMs show promising potential across various urban downstream tasks, they still struggle in long-horizon planning and reflective adaptation in dynamic urban contexts. Notably, recent advanced reasoning models (e.g., DeepSeek-R1) trained on general logic or mathematical problems do not consistently outperform non-reasoning LLMs. This discrepancy highlights the need for domain-specialized adaptation methods to enhance urban spatiotemporal reasoning. Overall, USTBench provides a foundation to build more adaptive and effective LLM-based urban agents and broad smart city applications. 

**Abstract (ZH)**: 基于空间时间推理的都市大型语言模型基准：USTBench 

---
# Transparency and Proportionality in Post-Processing Algorithmic Bias Correction 

**Title (ZH)**: 后处理算法偏见纠正中的透明度与比例原则 

**Authors**: Juliett Suárez Ferreira, Marija Slavkovik, Jorge Casillas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17525)  

**Abstract**: Algorithmic decision-making systems sometimes produce errors or skewed predictions toward a particular group, leading to unfair results. Debiasing practices, applied at different stages of the development of such systems, occasionally introduce new forms of unfairness or exacerbate existing inequalities. We focus on post-processing techniques that modify algorithmic predictions to achieve fairness in classification tasks, examining the unintended consequences of these interventions. To address this challenge, we develop a set of measures that quantify the disparity in the flips applied to the solution in the post-processing stage. The proposed measures will help practitioners: (1) assess the proportionality of the debiasing strategy used, (2) have transparency to explain the effects of the strategy in each group, and (3) based on those results, analyze the possibility of the use of some other approaches for bias mitigation or to solve the problem. We introduce a methodology for applying the proposed metrics during the post-processing stage and illustrate its practical application through an example. This example demonstrates how analyzing the proportionality of the debiasing strategy complements traditional fairness metrics, providing a deeper perspective to ensure fairer outcomes across all groups. 

**Abstract (ZH)**: 算法决策系统有时会产生针对特定群体的错误或偏差预测，导致不公平的结果。去偏见实践在这些系统开发的不同阶段的应用有时会引入新的不公平形式或加剧现有不平等。我们重点关注调整算法预测以在分类任务中实现公平性的后处理技术，研究这些干预措施的意外后果。为此，我们开发了一套度量标准，量化后处理阶段对解决方案调整的不平等程度。所提出的度量标准将帮助从业者：(1)评估所使用去偏见策略的比例性，(2)增强透明度以解释策略在各群体中的效应，(3)根据这些结果分析使用其他方法减少偏差或解决问题的可能性。我们介绍了在后处理阶段应用所提度量标准的方法，并通过一个示例说明其实用应用。该示例展示了分析去偏见策略的比例性如何补充传统公平性度量，提供更深入的视角以确保所有群体的更公平结果。 

---
# Optimizing Retrieval-Augmented Generation for Electrical Engineering: A Case Study on ABB Circuit Breakers 

**Title (ZH)**: 优化电气工程中的检索增强生成：ABB断路器案例研究 

**Authors**: Salahuddin Alawadhi, Noorhan Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17520)  

**Abstract**: Integrating Retrieval Augmented Generation (RAG) with Large Language Models (LLMs) has shown the potential to provide precise, contextually relevant responses in knowledge intensive domains. This study investigates the ap-plication of RAG for ABB circuit breakers, focusing on accuracy, reliability, and contextual relevance in high-stakes engineering environments. By leveraging tailored datasets, advanced embedding models, and optimized chunking strategies, the research addresses challenges in data retrieval and contextual alignment unique to engineering documentation. Key contributions include the development of a domain-specific dataset for ABB circuit breakers and the evaluation of three RAG pipelines: OpenAI GPT4o, Cohere, and Anthropic Claude. Advanced chunking methods, such as paragraph-based and title-aware segmentation, are assessed for their impact on retrieval accuracy and response generation. Results demonstrate that while certain configurations achieve high precision and relevancy, limitations persist in ensuring factual faithfulness and completeness, critical in engineering contexts. This work underscores the need for iterative improvements in RAG systems to meet the stringent demands of electrical engineering tasks, including design, troubleshooting, and operational decision-making. The findings in this paper help advance research of AI in highly technical domains such as electrical engineering. 

**Abstract (ZH)**: 将检索增强生成（RAG）与大规模语言模型（LLMs）结合应用于ABB断路器的知识密集型领域的潜力研究：聚焦于高风险工程环境中的准确性和上下文相关性 

---
# Probe by Gaming: A Game-based Benchmark for Assessing Conceptual Knowledge in LLMs 

**Title (ZH)**: 基于游戏的探针：一种评估LLMs概念知识的基准游戏 

**Authors**: Shuhang Xu, Weijian Deng, Yixuan Zhou, Fangwei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17512)  

**Abstract**: Concepts represent generalized abstractions that enable humans to categorize and reason efficiently, yet it is unclear to what extent Large Language Models (LLMs) comprehend these semantic relationships. Existing benchmarks typically focus on factual recall and isolated tasks, failing to evaluate the ability of LLMs to understand conceptual boundaries. To address this gap, we introduce CK-Arena, a multi-agent interaction game built upon the Undercover game, designed to evaluate the capacity of LLMs to reason with concepts in interactive settings. CK-Arena challenges models to describe, differentiate, and infer conceptual boundaries based on partial information, encouraging models to explore commonalities and distinctions between closely related concepts. By simulating real-world interaction, CK-Arena provides a scalable and realistic benchmark for assessing conceptual reasoning in dynamic environments. Experimental results show that LLMs' understanding of conceptual knowledge varies significantly across different categories and is not strictly aligned with parameter size or general model capabilities. The data and code are available at the project homepage: this https URL. 

**Abstract (ZH)**: CK-Arena：一种评估大规模语言模型在交互设置中进行概念推理能力的游戏 

---
# PD$^3$: A Project Duplication Detection Framework via Adapted Multi-Agent Debate 

**Title (ZH)**: PD$^3$: 一种基于适应性多agent辩论的项目复制检测框架 

**Authors**: Dezheng Bao, Yueci Yang, Xin Chen, Zhengxuan Jiang, Zeguo Fei, Daoze Zhang, Xuanwen Huang, Junru Chen, Chutian Yu, Xiang Yuan, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17492)  

**Abstract**: Project duplication detection is critical for project quality assessment, as it improves resource utilization efficiency by preventing investing in newly proposed project that have already been studied. It requires the ability to understand high-level semantics and generate constructive and valuable feedback. Existing detection methods rely on basic word- or sentence-level comparison or solely apply large language models, lacking valuable insights for experts and in-depth comprehension of project content and review criteria. To tackle this issue, we propose PD$^3$, a Project Duplication Detection framework via adapted multi-agent Debate. Inspired by real-world expert debates, it employs a fair competition format to guide multi-agent debate to retrieve relevant projects. For feedback, it incorporates both qualitative and quantitative analysis to improve its practicality. Over 800 real-world power project data spanning more than 20 specialized fields are used to evaluate the framework, demonstrating that our method outperforms existing approaches by 7.43% and 8.00% in two downstream tasks. Furthermore, we establish an online platform, Review Dingdang, to assist power experts, saving 5.73 million USD in initial detection on more than 100 newly proposed projects. 

**Abstract (ZH)**: 基于多智能体辩论的项目重复检测框架PD$^3$ 

---
# From Reasoning to Generalization: Knowledge-Augmented LLMs for ARC Benchmark 

**Title (ZH)**: 从推理到泛化：知识增强的大语言模型在ARC基准测试中的应用 

**Authors**: Chao Lei, Nir Lipovetzky, Krista A. Ehinger, Yanchuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17482)  

**Abstract**: Recent reasoning-oriented LLMs have demonstrated strong performance on challenging tasks such as mathematics and science examinations. However, core cognitive faculties of human intelligence, such as abstract reasoning and generalization, remain underexplored. To address this, we evaluate recent reasoning-oriented LLMs on the Abstraction and Reasoning Corpus (ARC) benchmark, which explicitly demands both faculties. We formulate ARC as a program synthesis task and propose nine candidate solvers. Experimental results show that repeated-sampling planning-aided code generation (RSPC) achieves the highest test accuracy and demonstrates consistent generalization across most LLMs. To further improve performance, we introduce an ARC solver, Knowledge Augmentation for Abstract Reasoning (KAAR), which encodes core knowledge priors within an ontology that classifies priors into three hierarchical levels based on their dependencies. KAAR progressively expands LLM reasoning capacity by gradually augmenting priors at each level, and invokes RSPC to generate candidate solutions after each augmentation stage. This stage-wise reasoning reduces interference from irrelevant priors and improves LLM performance. Empirical results show that KAAR maintains strong generalization and consistently outperforms non-augmented RSPC across all evaluated LLMs, achieving around 5% absolute gains and up to 64.52% relative improvement. Despite these achievements, ARC remains a challenging benchmark for reasoning-oriented LLMs, highlighting future avenues of progress in LLMs. 

**Abstract (ZH)**: 近期的推理导向的大规模语言模型在数学和科学考试等挑战性任务上展现了强大的性能。然而，人类智能的核心认知能力，如抽象推理和泛化能力，仍然有待探索。为解决这一问题，我们利用Abstraction and Reasoning Corpus (ARC)基准评估了近期的推理导向的大规模语言模型，该基准明确要求这两种能力。我们将ARC任务形式化为程序合成任务，并提出了九种候选解算器。实验结果显示，重复取样规划辅助的代码生成（RSPC）在测试中的准确率最高，并且在大多数大规模语言模型上展示了稳定的泛化能力。为进一步提升性能，我们引入一种ARC解算器——Knowledge Augmentation for Abstract Reasoning (KAAR)，它使用一种本体将先验知识编码为三个层级的层次结构，并通过逐步增强各级先验来扩展大规模语言模型的推理能力，在每次增强阶段后，使用RSPC生成候选解决方案。这种阶段性的推理减少了无关先验的干扰，并提高了大规模语言模型的性能。实验结果表明，与未增强的RSPC相比，KAAR保持了强大的泛化能力，并在所有评估的模型中始终表现出更优的表现，绝对收益约为5%，相对收益高达64.52%。尽管取得了这些成就，ARC仍是对推理导向的大规模语言模型的一个具有挑战性的基准，突显了未来语言模型进展的方向。 

---
# Scaling Up Biomedical Vision-Language Models: Fine-Tuning, Instruction Tuning, and Multi-Modal Learning 

**Title (ZH)**: 扩大生物医药领域视觉-语言模型的规模：微调、指令调优与多模态学习 

**Authors**: Cheng Peng, Kai Zhang, Mengxian Lyu, Hongfang Liu, Lichao Sun, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17436)  

**Abstract**: To advance biomedical vison-language model capabilities through scaling up, fine-tuning, and instruction tuning, develop vision-language models with improved performance in handling long text, explore strategies to efficiently adopt vision language models for diverse multi-modal biomedical tasks, and examine the zero-shot learning performance.
We developed two biomedical vision language models, BiomedGPT-Large and BiomedGPT-XLarge, based on an encoder-decoder-based transformer architecture. We fine-tuned the two models on 23 benchmark datasets from 6 multi-modal biomedical tasks including one image-only task (image classification), three language-only tasks (text understanding, text summarization and question answering), and two vision-language tasks (visual question answering and image captioning). We compared the developed scaled models with our previous BiomedGPT-Base model and existing prestigious models reported in the literature. We instruction-tuned the two models using a large-scale multi-modal biomedical instruction-tuning dataset and assessed the zero-shot learning performance and alignment accuracy. 

**Abstract (ZH)**: 通过扩展、微调和指令调优提升生物医学视觉语言模型能力，探索高效采用视觉语言模型进行多元生物医学任务的策略，并考察零样本学习性能。发展了两个基于编码器-解码器变换器架构的生物医学视觉语言模型——BiomedGPT-Large和BiomedGPT-XLarge。在包括一项图像仅任务（图像分类）、三项语言仅任务（文本理解、文本摘要和问答）以及两项视觉语言任务（视觉问答和图像字幕）的23个基准数据集上对两个模型进行了微调。将开发的扩展模型与我们之前发布的BiomedGPT-Base模型及文献中报道的杰出模型进行了比较，并使用大规模多元生物医学指令调优数据集对两个模型进行了指令调优，评估了零样本学习性能和对齐准确度。 

---
# MemeReaCon: Probing Contextual Meme Understanding in Large Vision-Language Models 

**Title (ZH)**: MemeReaCon: 探究大型视觉-语言模型中的情境表情包理解能力 

**Authors**: Zhengyi Zhao, Shubo Zhang, Yuxi Zhang, Yanxi Zhao, Yifan Zhang, Zezhong Wang, Huimin Wang, Yutian Zhao, Bin Liang, Yefeng Zheng, Binyang Li, Kam-Fai Wong, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17433)  

**Abstract**: Memes have emerged as a popular form of multimodal online communication, where their interpretation heavily depends on the specific context in which they appear. Current approaches predominantly focus on isolated meme analysis, either for harmful content detection or standalone interpretation, overlooking a fundamental challenge: the same meme can express different intents depending on its conversational context. This oversight creates an evaluation gap: although humans intuitively recognize how context shapes meme interpretation, Large Vision Language Models (LVLMs) can hardly understand context-dependent meme intent. To address this critical limitation, we introduce MemeReaCon, a novel benchmark specifically designed to evaluate how LVLMs understand memes in their original context. We collected memes from five different Reddit communities, keeping each meme's image, the post text, and user comments together. We carefully labeled how the text and meme work together, what the poster intended, how the meme is structured, and how the community responded. Our tests with leading LVLMs show a clear weakness: models either fail to interpret critical information in the contexts, or overly focus on visual details while overlooking communicative purpose. MemeReaCon thus serves both as a diagnostic tool exposing current limitations and as a challenging benchmark to drive development toward more sophisticated LVLMs of the context-aware understanding. 

**Abstract (ZH)**: Memes作为一种流行的多模态在线交流形式，其解释高度依赖于它们出现的具体情境。当前的方法主要侧重于孤立的 meme 分析，主要用于有害内容检测或独立解释，忽视了一个基本挑战：同一个 meme 可以根据其对话情境表达不同的意图。这种忽视造成了评价差距：尽管人类能直观地认识到情境如何塑造 meme 的解释，大型视觉语言模型 (LVLM) 难以理解情境依赖的 meme 意图。为解决这一关键限制，我们引入了 MemeReaCon，这是一种新型基准，专门用于评估 LVLMs 如何在原始情境中理解 memes。我们从五个不同的 Reddit 社区收集 memes，保持每个 meme 的图像、帖子文本和用户评论完整。我们仔细标注了文本和 meme 的互动方式、发布者意图、meme 的结构以及社区的反应。我们对领先 LVLM 的测试表明，模型要么无法解释上下文中的关键信息，要么过度关注视觉细节而忽视交流目的。MemeReaCon 因此既作为诊断工具揭示了当前的局限性，又作为一个具有挑战性的基准，推动开发出更复杂的情境意识 LVLM。 

---
# Misaligning Reasoning with Answers -- A Framework for Assessing LLM CoT Robustness 

**Title (ZH)**: 推理与答案不一致：评估大模型思维过程稳健性的框架 

**Authors**: Enyi Jiang, Changming Xu, Nischay Singh, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.17406)  

**Abstract**: LLMs' decision-making process is opaque, prompting the need for explanation techniques like Chain-of-Thought. To investigate the relationship between answer and reasoning, we design a novel evaluation framework, MATCHA. In domains like education and healthcare, reasoning is key for model trustworthiness. MATCHA reveals that LLMs under input perturbations can give inconsistent or nonsensical reasoning. Additionally, we use LLM judges to assess reasoning robustness across models. Our results show that LLMs exhibit greater vulnerability to input perturbations for multi-step and commonsense tasks than compared to logical tasks. Also, we show non-trivial transfer rates of our successful examples to black-box models. Our evaluation framework helps to better understand LLM reasoning mechanisms and guides future models toward more robust and reasoning-driven architectures, enforcing answer-reasoning consistency. 

**Abstract (ZH)**: LLMs的决策过程不可透明，促使了Chain-of-Thought等解释技术的需求。为了探究答案与推理之间的关系，我们设计了一种新的评估框架MATCHA。在教育和医疗等领域，推理对于模型可信度至关重要。MATCHA揭示了在输入扰动下，LLMs可能会给出不一致或不合逻辑的推理。此外，我们使用LLM作为评委来评估模型推理的鲁棒性。我们的结果显示，与逻辑任务相比，LLMs在多步和常识任务上的输入扰动鲁棒性更低。同时，我们展示了我们成功案例的非平凡转移率到黑盒模型中。我们的评估框架有助于更好地理解LLMs的推理机制，并指导未来模型向更鲁棒和以推理为导向的架构发展，确保答案与推理的一致性。 

---
# DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic 

**Title (ZH)**: DEL-ToM: 推理时动态演绎认识逻辑的理论共情推理缩放方法 

**Authors**: Yuheng Wu, Jianwen Xie, Denghui Zhang, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17348)  

**Abstract**: Theory-of-Mind (ToM) tasks pose a unique challenge for small language models (SLMs) with limited scale, which often lack the capacity to perform deep social reasoning. In this work, we propose DEL-ToM, a framework that improves ToM reasoning through inference-time scaling rather than architectural changes. Our approach decomposes ToM tasks into a sequence of belief updates grounded in Dynamic Epistemic Logic (DEL), enabling structured and transparent reasoning. We train a verifier, called the Process Belief Model (PBM), to score each belief update step using labels generated automatically via a DEL simulator. During inference, candidate belief traces generated by a language model are evaluated by the PBM, and the highest-scoring trace is selected. This allows SLMs to emulate more deliberate reasoning by allocating additional compute at test time. Experiments across multiple model scales and benchmarks show that DEL-ToM consistently improves performance, demonstrating that verifiable belief supervision can significantly enhance ToM abilities of SLMs without retraining. 

**Abstract (ZH)**: DEL-ToM:通过推理时的扩展提升理论-of- mind推理能力的框架 

---
# Partner Modelling Emerges in Recurrent Agents (But Only When It Matters) 

**Title (ZH)**: Recurrent 代理中的伙伴建模现象（但仅在必要时出现） 

**Authors**: Ruaridh Mon-Williams, Max Taylor-Davies, Elizabeth Mieczkowski, Natalia Velez, Neil R. Bramley, Yanwei Wang, Thomas L. Griffiths, Christopher G. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17323)  

**Abstract**: Humans are remarkably adept at collaboration, able to infer the strengths and weaknesses of new partners in order to work successfully towards shared goals. To build AI systems with this capability, we must first understand its building blocks: does such flexibility require explicit, dedicated mechanisms for modelling others -- or can it emerge spontaneously from the pressures of open-ended cooperative interaction? To investigate this question, we train simple model-free RNN agents to collaborate with a population of diverse partners. Using the `Overcooked-AI' environment, we collect data from thousands of collaborative teams, and analyse agents' internal hidden states. Despite a lack of additional architectural features, inductive biases, or auxiliary objectives, the agents nevertheless develop structured internal representations of their partners' task abilities, enabling rapid adaptation and generalisation to novel collaborators. We investigated these internal models through probing techniques, and large-scale behavioural analysis. Notably, we find that structured partner modelling emerges when agents can influence partner behaviour by controlling task allocation. Our results show that partner modelling can arise spontaneously in model-free agents -- but only under environmental conditions that impose the right kind of social pressure. 

**Abstract (ZH)**: 人类在协作方面具有出色的能力，能够推断新合作伙伴的优势和劣势，以共同实现目标。为了构建具备这种能力的AI系统，我们必须首先理解其构建块：这种灵活性是否需要显式的、专门的机制来建模他人，还是可以从开放合作交互的压力中自发产生？为了探究这一问题，我们训练简单的无模型RNN代理与多样化的合作伙伴进行合作。使用“Overcooked-AI”环境，我们收集了成千上万的合作团队的数据，并分析了代理的内部隐藏状态。尽管缺乏额外的架构特征、归纳偏置或辅助目标，代理仍然发展出结构化的内部表征，以合作伙伴的任务能力为基础，实现快速适应和对新合作伙伴的泛化。通过探测技术和大规模行为分析，我们发现，当代理能够通过控制任务分配来影响合作伙伴的行为时，结构化的合作伙伴建模会自发产生。我们的结果显示，在适当的环境条件下，无模型代理中的合作伙伴建模可以自发产生。 

---
# Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning 

**Title (ZH)**: 更长的语境，更深的思考：探究长语境能力在推理中的作用 

**Authors**: Wang Yang, Zirui Liu, Hongye Jin, Qingyu Yin, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17315)  

**Abstract**: Recent language models exhibit strong reasoning capabilities, yet the influence of long-context capacity on reasoning remains underexplored. In this work, we hypothesize that current limitations in reasoning stem, in part, from insufficient long-context capacity, motivated by empirical observations such as (1) higher context window length often leads to stronger reasoning performance, and (2) failed reasoning cases resemble failed long-context cases. To test this hypothesis, we examine whether enhancing a model's long-context ability before Supervised Fine-Tuning (SFT) leads to improved reasoning performance. Specifically, we compared models with identical architectures and fine-tuning data but varying levels of long-context capacity. Our results reveal a consistent trend: models with stronger long-context capacity achieve significantly higher accuracy on reasoning benchmarks after SFT. Notably, these gains persist even on tasks with short input lengths, indicating that long-context training offers generalizable benefits for reasoning performance. These findings suggest that long-context modeling is not just essential for processing lengthy inputs, but also serves as a critical foundation for reasoning. We advocate for treating long-context capacity as a first-class objective in the design of future language models. 

**Abstract (ZH)**: 近期的语言模型展示了强大的推理能力，但长上下文容量对推理的影响仍较少被探索。在此工作中，我们假设当前推理能力的限制部分源于不足的长上下文容量，这受到如下实证观察的启发：（1）更高的上下文窗口长度通常导致更强的推理性能，（2）推理失败案例类似于长上下文处理失败的案例。为了测试这一假设，我们检查在监督微调（SFT）之前增强模型的长上下文能力是否能够提高其推理性能。具体来说，我们对比了具有相同架构和微调数据但不同长上下文容量水平的模型。研究结果表明：在SFT后，具有更强长上下文容量的模型在推理基准测试中的准确性显著更高。值得注意的是，这些增益甚至在输入长度较短的任务中依然存在，表明长上下文训练为提高推理性能提供了可泛化的益处。这些发现表明，长上下文建模不仅是处理长输入所必需的，也是推理的基础。我们建议将长上下文容量作为未来语言模型设计中的首要目标。 

---
# AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking 

**Title (ZH)**: AdaReasoner: 自适应推理实现更灵活的思考 

**Authors**: Xiangqi Wang, Yue Huang, Yanbo Wang, Xiaonan Luo, Kehan Guo, Yujun Zhou, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17312)  

**Abstract**: LLMs often need effective configurations, like temperature and reasoning steps, to handle tasks requiring sophisticated reasoning and problem-solving, ranging from joke generation to mathematical reasoning. Existing prompting approaches usually adopt general-purpose, fixed configurations that work 'well enough' across tasks but seldom achieve task-specific optimality. To address this gap, we introduce AdaReasoner, an LLM-agnostic plugin designed for any LLM to automate adaptive reasoning configurations for tasks requiring different types of thinking. AdaReasoner is trained using a reinforcement learning (RL) framework, combining a factorized action space with a targeted exploration strategy, along with a pretrained reward model to optimize the policy model for reasoning configurations with only a few-shot guide. AdaReasoner is backed by theoretical guarantees and experiments of fast convergence and a sublinear policy gap. Across six different LLMs and a variety of reasoning tasks, it consistently outperforms standard baselines, preserves out-of-distribution robustness, and yield gains on knowledge-intensive tasks through tailored prompts. 

**Abstract (ZH)**: LLMs通常需要有效的配置，如温度和推理步骤，以处理从笑话生成到数学推理等各种需要复杂推理和问题解决的任务。现有的提示方法通常采用一般性的固定配置，在各任务中表现“足够好”，但很少达到任务特定的最优性。为解决这一问题，我们提出AdaReasoner，这是一种LLM无关的插件，旨在为任何LLM自动化不同类型的推理配置。AdaReasoner使用强化学习框架进行训练，结合因子化的动作空间和目标探索策略，并使用预训练的奖励模型，仅通过少量示例指导适应推理配置。AdaReasoner提供了理论保证，并通过快速收敛和亚线性策略差距的实验验证。在六种不同LLM和各种推理任务上，它在标准基准上表现更优，保持了分布外鲁棒性，并通过定制提示在知识密集型任务上获得收益。 

---
# Where You Go is Who You Are: Behavioral Theory-Guided LLMs for Inverse Reinforcement Learning 

**Title (ZH)**: 你的去向即你的身份：行为理论引导的大型语言模型在逆强化学习中的应用 

**Authors**: Yuran Sun, Susu Xu, Chenguang Wang, Xilei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17249)  

**Abstract**: Big trajectory data hold great promise for human mobility analysis, but their utility is often constrained by the absence of critical traveler attributes, particularly sociodemographic information. While prior studies have explored predicting such attributes from mobility patterns, they often overlooked underlying cognitive mechanisms and exhibited low predictive accuracy. This study introduces SILIC, short for Sociodemographic Inference with LLM-guided Inverse Reinforcement Learning (IRL) and Cognitive Chain Reasoning (CCR), a theoretically grounded framework that leverages LLMs to infer sociodemographic attributes from observed mobility patterns by capturing latent behavioral intentions and reasoning through psychological constructs. Particularly, our approach explicitly follows the Theory of Planned Behavior (TPB), a foundational behavioral framework in transportation research, to model individuals' latent cognitive processes underlying travel decision-making. The LLMs further provide heuristic guidance to improve IRL reward function initialization and update by addressing its ill-posedness and optimization challenges arising from the vast and unstructured reward space. Evaluated in the 2017 Puget Sound Regional Council Household Travel Survey, our method substantially outperforms state-of-the-art baselines and shows great promise for enriching big trajectory data to support more behaviorally grounded applications in transportation planning and beyond. 

**Abstract (ZH)**: 基于LLM引导的逆强化学习和认知链推理的 Sociodemographic Inference (SILIC)：理论指导的大轨迹数据社会demographic属性推断框架 

---
# Reasoning Model is Stubborn: Diagnosing Instruction Overriding in Reasoning Models 

**Title (ZH)**: 推理模型固执己见：诊断推理模型中的指令 overriding 

**Authors**: Doohyuk Jang, Yoonjeon Kim, Chanjae Park, Hyun Ryu, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17225)  

**Abstract**: Large language models have demonstrated remarkable proficiency in long and complex reasoning tasks. However, they frequently exhibit a problematic reliance on familiar reasoning patterns, a phenomenon we term \textit{reasoning rigidity}. Despite explicit instructions from users, these models often override clearly stated conditions and default to habitual reasoning trajectories, leading to incorrect conclusions. This behavior presents significant challenges, particularly in domains such as mathematics and logic puzzle, where precise adherence to specified constraints is critical. To systematically investigate reasoning rigidity, a behavior largely unexplored in prior work, we introduce a expert-curated diagnostic set, \dataset{}. Our dataset includes specially modified variants of existing mathematical benchmarks, namely AIME and MATH500, as well as well-known puzzles deliberately redesigned to require deviation from familiar reasoning strategies. Using this dataset, we identify recurring contamination patterns that occur when models default to ingrained reasoning. Specifically, we categorize this contamination into three distinctive modes: (i) Interpretation Overload, (ii) Input Distrust, and (iii) Partial Instruction Attention, each causing models to ignore or distort provided instructions. We publicly release our diagnostic set to facilitate future research on mitigating reasoning rigidity in language models. 

**Abstract (ZH)**: 大型语言模型在长期和复杂的推理任务中展现了非凡的能力。然而，它们经常表现出一种令人担忧的推理模式固守现象，我们称之为“推理僵化”。尽管用户给出了明确的指示，这些模型仍会无视显式条件，转而遵循惯常的推理轨迹，导致错误的结论。这种行为在数学和逻辑谜题等领域尤其具有挑战性，因为这些领域中的精确遵从指定约束条件至关重要。为了系统地研究这一先前研究中鲜有探索的行为，我们引入了一个由专家编曲的诊断集\dataset{}。我们的数据集包含对现有数学基准AIME和MATH500的特定修改版本，以及故意重新设计的经典谜题，以要求模型偏离熟悉的推理策略。利用这个数据集，我们识别出当模型默认依赖于嵌入的推理模式时反复出现的污染模式。具体而言，我们将这种污染归类为三种独特模式：（i）解释过载，（ii）输入不信任，以及（iii）部分指令关注，每种模式都会导致模型忽略或歪曲提供的指示。我们公开发布了诊断集，以促进未来在减轻语言模型推理僵化方面的研究。 

---
# Effective Reinforcement Learning for Reasoning in Language Models 

**Title (ZH)**: 语言模型中的有效强化学习推理方法 

**Authors**: Lianghuan Huang, Shuo Li, Sagnik Anupam, Insup Lee, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17218)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising strategy for improving the reasoning capabilities of language models (LMs) in domains such as mathematics and coding. However, most modern RL algorithms were designed to target robotics applications, which differ significantly from LM reasoning. We analyze RL algorithm design decisions for LM reasoning, for both accuracy and computational efficiency, focusing on relatively small models due to computational constraints. Our findings are: (i) on-policy RL significantly outperforms supervised fine-tuning (SFT), (ii) PPO-based off-policy updates increase accuracy instead of reduce variance, and (iii) removing KL divergence can lead to more concise generations and higher accuracy. Furthermore, we find that a key bottleneck to computational efficiency is that the optimal batch sizes for inference and backpropagation are different. We propose a novel algorithm, DASH, that performs preemptive sampling (i.e., sample a large batch and accumulate gradient updates in small increments), and gradient filtering (i.e., drop samples with small advantage estimates). We show that DASH reduces training time by 83% compared to a standard implementation of GRPO without sacrificing accuracy. Our findings provide valuable insights on designing effective RL algorithms for LM reasoning. 

**Abstract (ZH)**: 强化学习（RL）已成为提高语言模型（LM）在数学和编程等领域推理能力的有前途的策略。然而，大多数现代RL算法都是为机器人应用设计的，与LM推理存在显著差异。我们分析了针对LM推理的RL算法设计决策，重点关注由于计算约束而相对较小的模型，以提高准确性和计算效率。我们的发现包括：(i) 在策略上进行RL显著优于监督微调(SFT)，(ii) 基于PPO的离策略更新增加了准确率而非减少方差，(iii) 去除KL散度可以导致更简洁的生成并提高准确率。此外，我们发现计算效率的关键瓶颈是在推理和反向传播的最佳批量大小不同。我们提出了一种新型算法DASH，该算法执行预采样（即，采样大批次并以小增量累积梯度更新）和梯度滤波（即，丢弃具有小优势估计的采样）。我们展示DASH相比标准GRPO实现的训练时间减少了83%，且不牺牲准确率。我们的发现提供了设计有效RL算法的宝贵见解。 

---
# MEDMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph 

**Title (ZH)**: MEDMKG：多模态知识图谱中的医疗知识 exploitation 评估基准 

**Authors**: Xiaochen Wang, Yuan Zhong, Lingwei Zhang, Lisong Dai, Ting Wang, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.17214)  

**Abstract**: Medical deep learning models depend heavily on domain-specific knowledge to perform well on knowledge-intensive clinical tasks. Prior work has primarily leveraged unimodal knowledge graphs, such as the Unified Medical Language System (UMLS), to enhance model performance. However, integrating multimodal medical knowledge graphs remains largely underexplored, mainly due to the lack of resources linking imaging data with clinical concepts. To address this gap, we propose MEDMKG, a Medical Multimodal Knowledge Graph that unifies visual and textual medical information through a multi-stage construction pipeline. MEDMKG fuses the rich multimodal data from MIMIC-CXR with the structured clinical knowledge from UMLS, utilizing both rule-based tools and large language models for accurate concept extraction and relationship modeling. To ensure graph quality and compactness, we introduce Neighbor-aware Filtering (NaF), a novel filtering algorithm tailored for multimodal knowledge graphs. We evaluate MEDMKG across three tasks under two experimental settings, benchmarking twenty-four baseline methods and four state-of-the-art vision-language backbones on six datasets. Results show that MEDMKG not only improves performance in downstream medical tasks but also offers a strong foundation for developing adaptive and robust strategies for multimodal knowledge integration in medical artificial intelligence. 

**Abstract (ZH)**: 医学多模态知识图谱在知识密集型临床任务中依赖特定领域知识以实现高性能。先前工作主要通过统一医学语言系统(UMLS)等单模态知识图谱来增强模型性能。然而，多模态医学知识图谱的集成尚未充分探索，主要由于缺乏连接影像数据与临床概念的资源。为解决这一问题，我们提出了一种医学多模态知识图谱(MEDMKG)，通过多阶段构建管道统一视觉和文本医学信息。MEDMKG 将 MIMIC-CXR 的丰富多模态数据与 UMLS 的结构化临床知识融合，利用基于规则的工具和大规模语言模型进行准确的概念提取和关系建模。为确保图的质量和紧凑性，我们引入了一种名为邻域感知过滤(NaF)的新过滤算法，专门针对多模态知识图谱。在两种实验设置下，我们在六个数据集上评估了MEDMKG，针对二十四种基准方法和四种最先进的视觉-语言骨干进行对比。结果显示，MEDMKG 不仅在下游医学任务中表现出色，还为开发适应性强且鲁棒的多模态知识集成策略提供了坚实基础。 

---
# An Affective-Taxis Hypothesis for Alignment and Interpretability 

**Title (ZH)**: 情感趋同假设：对齐与可解释性 

**Authors**: Eli Sennesh, Maxwell Ramstead  

**Link**: [PDF](https://arxiv.org/pdf/2505.17024)  

**Abstract**: AI alignment is a field of research that aims to develop methods to ensure that agents always behave in a manner aligned with (i.e. consistently with) the goals and values of their human operators, no matter their level of capability. This paper proposes an affectivist approach to the alignment problem, re-framing the concepts of goals and values in terms of affective taxis, and explaining the emergence of affective valence by appealing to recent work in evolutionary-developmental and computational neuroscience. We review the state of the art and, building on this work, we propose a computational model of affect based on taxis navigation. We discuss evidence in a tractable model organism that our model reflects aspects of biological taxis navigation. We conclude with a discussion of the role of affective taxis in AI alignment. 

**Abstract (ZH)**: AI对齐是一个旨在开发方法以确保代理始终以与人类操作者的目标和价值观一致的方式行为的研究领域，无论代理的能力水平如何。本文提出了一种情感主义方法来解决对齐问题，重新定义了目标和价值观的概念为情感趋化性，并通过引用进化发展和计算神经科学的最新成果来解释情感价值的产生。我们回顾了相关研究，并在此基础上提出了基于趋化性导航的情感计算模型。我们讨论了在可处理的模型生物体中支持我们模型反映生物学趋化性导航特征的证据。最后，我们讨论了情感趋化性在AI对齐中的作用。 

---
# WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions 

**Title (ZH)**: 奇思妙玩：从单张图像和动作生成动态3D场景 

**Authors**: Zizhang Li, Hong-Xing Yu, Wei Liu, Yin Yang, Charles Herrmann, Gordon Wetzstein, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18151)  

**Abstract**: WonderPlay is a novel framework integrating physics simulation with video generation for generating action-conditioned dynamic 3D scenes from a single image. While prior works are restricted to rigid body or simple elastic dynamics, WonderPlay features a hybrid generative simulator to synthesize a wide range of 3D dynamics. The hybrid generative simulator first uses a physics solver to simulate coarse 3D dynamics, which subsequently conditions a video generator to produce a video with finer, more realistic motion. The generated video is then used to update the simulated dynamic 3D scene, closing the loop between the physics solver and the video generator. This approach enables intuitive user control to be combined with the accurate dynamics of physics-based simulators and the expressivity of diffusion-based video generators. Experimental results demonstrate that WonderPlay enables users to interact with various scenes of diverse content, including cloth, sand, snow, liquid, smoke, elastic, and rigid bodies -- all using a single image input. Code will be made public. Project website: this https URL 

**Abstract (ZH)**: WonderPlay是一种将物理模拟与视频生成集成的新型框架，用于从单张图片生成动作条件下的动态3D场景。 

---
# Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find 

**Title (ZH)**: 迷失在haystack中：对于LLMs来说，更小的针更难被找到 

**Authors**: Owen Bianchi, Mathew J. Koretsky, Maya Willey, Chelsea X. Alvarado, Tanay Nayak, Adi Asija, Nicole Kuznetsov, Mike A. Nalls, Faraz Faghri, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18148)  

**Abstract**: Large language models (LLMs) face significant challenges with needle-in-a-haystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size has received little attention. We address this gap by systematically studying how variations in gold context length impact LLM performance on long-context question answering tasks. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This pattern holds across three diverse domains (general knowledge, biomedical reasoning, and mathematical reasoning) and seven state-of-the-art LLMs of various sizes and architectures. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems. 

**Abstract (ZH)**: 大型语言模型在针锋相对任务中面临显著挑战，其中相关信息（“针”）必须从大量的无关背景信息（“草堆”）中抽取。先前的研究强调了位置偏差和干扰因素的数量对模型性能的影响，但黄金背景大小的影响尚未得到充分关注。我们通过系统研究黄金背景长度变化对大型语言模型在长背景问答任务中的性能影响来填补这一空白。我们的实验证明，当黄金背景较短时，模型性能急剧下降，即较小的黄金背景始终会削弱模型性能并放大位置敏感性，这对需要整合不同长度片段信息的自主系统提出了重大挑战。这一模式在三个不同的领域（一般知识、生物医学推理和数学推理）和七种不同规模和架构的先进大型语言模型中均适用。我们的研究为设计稳健的、上下文意识的大型语言模型驱动系统提供了清晰的指导。 

---
# Graph-Linguistic Fusion: Using Language Models for Wikidata Vandalism Detection 

**Title (ZH)**: 图语言融合：使用语言模型进行维基数据篡改检测 

**Authors**: Mykola Trokhymovych, Lydia Pintscher, Ricardo Baeza-Yates, Diego Saez-Trumper  

**Link**: [PDF](https://arxiv.org/pdf/2505.18136)  

**Abstract**: We introduce a next-generation vandalism detection system for Wikidata, one of the largest open-source structured knowledge bases on the Web. Wikidata is highly complex: its items incorporate an ever-expanding universe of factual triples and multilingual texts. While edits can alter both structured and textual content, our approach converts all edits into a single space using a method we call Graph2Text. This allows for evaluating all content changes for potential vandalism using a single multilingual language model. This unified approach improves coverage and simplifies maintenance. Experiments demonstrate that our solution outperforms the current production system. Additionally, we are releasing the code under an open license along with a large dataset of various human-generated knowledge alterations, enabling further research. 

**Abstract (ZH)**: 我们介绍了一种针对Wikidata的下一代 vandalism 检测系统，Wikidata是Web上最大的开源结构化知识库之一。这一系统采用了一种称为Graph2Text的方法，将所有编辑转化为一个统一的空间，使得使用一种多语言语言模型即可评估所有内容变化以检测潜在的 vandalism。这种方法提高了覆盖面并简化了维护。实验表明，我们的解决方案优于当前的生产系统。此外，我们将提供开源代码以及大量由人类生成的知识修改数据集，以促进进一步的研究。 

---
# Leveraging KANs for Expedient Training of Multichannel MLPs via Preconditioning and Geometric Refinement 

**Title (ZH)**: 利用KANs加速多通道MLP训练的预条件化与几何 refinement 方法 

**Authors**: Jonas A. Actor, Graham Harper, Ben Southworth, Eric C. Cyr  

**Link**: [PDF](https://arxiv.org/pdf/2505.18131)  

**Abstract**: Multilayer perceptrons (MLPs) are a workhorse machine learning architecture, used in a variety of modern deep learning frameworks. However, recently Kolmogorov-Arnold Networks (KANs) have become increasingly popular due to their success on a range of problems, particularly for scientific machine learning tasks. In this paper, we exploit the relationship between KANs and multichannel MLPs to gain structural insight into how to train MLPs faster. We demonstrate the KAN basis (1) provides geometric localized support, and (2) acts as a preconditioned descent in the ReLU basis, overall resulting in expedited training and improved accuracy. Our results show the equivalence between free-knot spline KAN architectures, and a class of MLPs that are refined geometrically along the channel dimension of each weight tensor. We exploit this structural equivalence to define a hierarchical refinement scheme that dramatically accelerates training of the multi-channel MLP architecture. We show further accuracy improvements can be had by allowing the $1$D locations of the spline knots to be trained simultaneously with the weights. These advances are demonstrated on a range of benchmark examples for regression and scientific machine learning. 

**Abstract (ZH)**: 多层感知机（MLPs）是现代深度学习框架中的一种常用机器学习架构。然而，近年来，Kolmogorov-Arnold 网络（KANs）因在多种问题上的成功，特别是在科学机器学习任务中的表现，而变得越来越受欢迎。本文利用 KANs 和多通道 MLPs 之间的关系，揭示了如何更快地训练 MLPs 的结构洞察。我们证明 KAN 基（1）提供几何局部支持，（2）作为 ReLU 基中的预条件下降，整体上导致训练加速并提高精度。我们的结果表明，自由结节样条 KAN 架构与一类沿每个权重张量的通道维度进行几何细化的 MLPs 相等价。我们利用这种结构等价性定义了一种分层细化方案，极大地加速了多通道 MLP 架构的训练。通过同时训练样条结点的一维位置和权重，我们进一步提高了精度。这些进步在回归和科学机器学习的各种基准示例中得到了展示。 

---
# Reward Model Overoptimisation in Iterated RLHF 

**Title (ZH)**: 迭代RLHF中的奖励模型过优化问题 

**Authors**: Lorenz Wolf, Robert Kirk, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18126)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is a widely used method for aligning large language models with human preferences. However, RLHF often suffers from reward model overoptimisation, in which models overfit to the reward function, resulting in non-generalisable policies that exploit the idiosyncrasies and peculiarities of the reward function. A common mitigation is iterated RLHF, in which reward models are repeatedly retrained with updated human feedback and policies are re-optimised. Despite its increasing adoption, the dynamics of overoptimisation in this setting remain poorly understood. In this work, we present the first comprehensive study of overoptimisation in iterated RLHF. We systematically analyse key design choices - how reward model training data is transferred across iterations, which reward function is used for optimisation, and how policies are initialised. Using the controlled AlpacaFarm benchmark, we observe that overoptimisation tends to decrease over successive iterations, as reward models increasingly approximate ground-truth preferences. However, performance gains diminish over time, and while reinitialising from the base policy is robust, it limits optimisation flexibility. Other initialisation strategies often fail to recover from early overoptimisation. These findings offer actionable insights for building more stable and generalisable RLHF pipelines. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）是一种广泛用于使大型语言模型与人类偏好相一致的方法。然而，RLHF 经常遭受奖励模型过优化的问题，即模型过度拟合到奖励函数，导致不可泛化的策略并利用奖励函数的独特性和异常现象。一个常见的缓解方法是迭代的 RLHF，其中奖励模型在更新的人类反馈下重复训练，并重新优化策略。尽管其采用日益增加，但在这种情况下过度优化的动态仍然理解不足。在本文中，我们首次全面研究了迭代 RLHF 中的过度优化问题。我们系统分析了关键设计选择——迭代之间如何转移奖励模型训练数据、用于优化的哪个奖励函数以及策略的初始化方式。通过受控的 AlpacaFarm 基准，我们观察到，随着奖励模型越来越接近真实偏好，过度优化的趋势逐渐减少。然而，性能改进随着时间的推移而减弱，通过基策略重新初始化尽管稳健，但限制了优化灵活性。其他初始化策略往往无法从早期过度优化中恢复。这些发现为构建更稳定和可泛化的 RLHF 管道提供了可操作的见解。 

---
# Bidirectional Knowledge Distillation for Enhancing Sequential Recommendation with Large Language Models 

**Title (ZH)**: 面向大型语言模型的双向知识蒸馏以增强序列推荐 

**Authors**: Jiongran Wu, Jiahao Liu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Li Shang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18120)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance in understanding and generating semantic patterns, making them promising candidates for sequential recommendation tasks. However, when combined with conventional recommendation models (CRMs), LLMs often face challenges related to high inference costs and static knowledge transfer methods. In this paper, we propose a novel mutual distillation framework, LLMD4Rec, that fosters dynamic and bidirectional knowledge exchange between LLM-centric and CRM-based recommendation systems. Unlike traditional unidirectional distillation methods, LLMD4Rec enables iterative optimization by alternately refining both models, enhancing the semantic understanding of CRMs and enriching LLMs with collaborative signals from user-item interactions. By leveraging sample-wise adaptive weighting and aligning output distributions, our approach eliminates the need for additional parameters while ensuring effective knowledge transfer. Extensive experiments on real-world datasets demonstrate that LLMD4Rec significantly improves recommendation accuracy across multiple benchmarks without increasing inference costs. This method provides a scalable and efficient solution for combining the strengths of both LLMs and CRMs in sequential recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解和生成语义模式方面表现出色，使其成为序列推荐任务的有希望候选者。然而，将其与传统推荐模型（CRMs）结合时，LLMs往往面临高推理成本和静态知识转移方法的挑战。本文提出了一种新颖的相互蒸馏框架LLMD4Rec，促进以LLM为中心和基于CRM的推荐系统之间动态和双向的知识交流。与传统的单向蒸馏方法不同，LLMD4Rec通过交替优化两个模型来实现迭代优化，增强CRM的语义理解，并通过用户项交互的协作信号丰富LLM。通过利用样本自适应加权和对齐输出分布，我们的方法在无需额外参数的情况下确保有效的知识转移。实验证明，LLMD4Rec在多个基准上显著提高了推荐准确性，而无需增加推理成本。该方法为在序列推荐系统中结合LLMs和CRMs的优势提供了一种可扩展且高效的解决方案。 

---
# How Can I Publish My LLM Benchmark Without Giving the True Answers Away? 

**Title (ZH)**: 如何发布我的大规模语言模型基准测试而不透露正确答案？ 

**Authors**: Takashi Ishida, Thanawat Lodkaew, Ikko Yamane  

**Link**: [PDF](https://arxiv.org/pdf/2505.18102)  

**Abstract**: Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies. 

**Abstract (ZH)**: 在互联网上发布大语言模型（LLM）基准的风险及其缓解策略：通过注入随机性保护基准的隐私以防止数据污染 

---
# Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL 

**Title (ZH)**: 无需搜索的规划：基于离线目标条件 reinforcement 学习精炼前沿大语言模型 

**Authors**: Joey Hong, Anca Dragan, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2505.18098)  

**Abstract**: Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在问答和对话任务中表现出色，但需要交互的复杂任务，如谈判和说服，则需要额外的长期推理和规划。强化学习（RL）微调原则上可以实现这种规划，但存在妨碍可扩展性的缺点。特别是多轮RL训练会产生较高的内存和计算成本，当对LLMs进行策略训练时，这种成本更为严重。此外，最大的LLMs并未提供必要的API以便以这种方式进行训练。因此，现代提高LLMs推理能力的方法依赖于复杂的提示机制而非RL微调。为解决这一问题，我们提出了一种新方法，利用目标条件的价值函数来引导LLM代理的推理，即使对于基于API的大规模模型也能扩展。这些价值函数预测给定动作后任务将如何展开，使LLM代理能够评估多种可能的结果，无论是积极的还是消极的，从而有效规划。此外，这些价值函数是基于推理步骤而非完整动作进行训练的，成为一个简洁轻量的模块，促进多轮交互中的决策制定。我们通过谈判、社会推理和对话等交互要求的任务验证了该方法，结果显示该方法在效率和可扩展性方面优于RL微调和提示方法，同时展现出优越的性能。 

---
# Data Mixing Can Induce Phase Transitions in Knowledge Acquisition 

**Title (ZH)**: 数据混杂可以诱导知识获取中的相变 

**Authors**: Xinran Gu, Kaifeng Lyu, Jiazheng Li, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18091)  

**Abstract**: Large Language Models (LLMs) are typically trained on data mixtures: most data come from web scrapes, while a small portion is curated from high-quality sources with dense domain-specific knowledge. In this paper, we show that when training LLMs on such data mixtures, knowledge acquisition from knowledge-dense datasets, unlike training exclusively on knowledge-dense data (arXiv:2404.05405), does not always follow a smooth scaling law but can exhibit phase transitions with respect to the mixing ratio and model size. Through controlled experiments on a synthetic biography dataset mixed with web-scraped data, we demonstrate that: (1) as we increase the model size to a critical value, the model suddenly transitions from memorizing very few to most of the biographies; (2) below a critical mixing ratio, the model memorizes almost nothing even with extensive training, but beyond this threshold, it rapidly memorizes more biographies. We attribute these phase transitions to a capacity allocation phenomenon: a model with bounded capacity must act like a knapsack problem solver to minimize the overall test loss, and the optimal allocation across datasets can change discontinuously as the model size or mixing ratio varies. We formalize this intuition in an information-theoretic framework and reveal that these phase transitions are predictable, with the critical mixing ratio following a power-law relationship with the model size. Our findings highlight a concrete case where a good mixing recipe for large models may not be optimal for small models, and vice versa. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常在数据混合集上训练：大部分数据来自网页抓取，而一小部分来自高质量的、富含领域特定知识的数据源。本文表明，在使用此类数据混合集训练LLMs时，知识获取从富含知识的数据集中抽取（不同于仅使用富含知识的数据集训练，arXiv:2404.05405），其并不是总是遵循平滑的扩展定律，而是可能在混合比例和模型大小方面表现出相变现象。通过在合成的个人传记数据集与网页抓取数据混合上进行受控实验，我们展示：（1）随着模型大小增加到临界值，模型突然从记忆少量传记转变为记忆大多数传记；（2）在临界混合比例以下，即使经过大量训练，模型几乎不记忆任何传记，但超过这个阈值后，它会迅速记忆更多的传记。我们将这些相变归因于容量分配现象：具有容量上限的模型必须像背包问题求解器一样运作，以最小化整体测试损失，不同模型大小或混合比例下，最佳的数据集间分配可能会出现不连续变化。我们从信息论框架中正式化了这一直觉，并揭示这些相变是可预测的，临界混合比例与模型大小之间存在幂律关系。我们的发现强调了一个具体案例，即对大型模型有效的数据混合方法可能对小型模型来说不是最优的，反之亦然。 

---
# CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays 

**Title (ZH)**: CXReasonBench：用于评估胸部X光结构化诊断推理的基准 

**Authors**: Hyungyung Lee, Geon Choi, Jung-Oh Lee, Hangyul Yoon, Hyuk Gi Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18087)  

**Abstract**: Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks focus mainly on the final diagnostic answer, offering limited insight into whether models engage in clinically meaningful reasoning. To address this, we present CheXStruct and CXReasonBench, a structured pipeline and benchmark built on the publicly available MIMIC-CXR-JPG dataset. CheXStruct automatically derives a sequence of intermediate reasoning steps directly from chest X-rays, such as segmenting anatomical regions, deriving anatomical landmarks and diagnostic measurements, computing diagnostic indices, and applying clinical thresholds. CXReasonBench leverages this pipeline to evaluate whether models can perform clinically valid reasoning steps and to what extent they can learn from structured guidance, enabling fine-grained and transparent assessment of diagnostic reasoning. The benchmark comprises 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each paired with up to 4 visual inputs, and supports multi-path, multi-stage evaluation including visual grounding via anatomical region selection and diagnostic measurements. Even the strongest of 10 evaluated LVLMs struggle with structured reasoning and generalization, often failing to link abstract knowledge with anatomically grounded visual interpretation. The code is available at this https URL 

**Abstract (ZH)**: 最近在大型视觉-语言模型（LVLMs）方面的进展使其实现了在医疗任务中的多项潜在应用，如报告生成和视觉问答。然而，现有的基准主要关注最终的诊断答案，未能提供模型在临床有意义的推理方面是否涉及的洞察。为了弥补这一不足，我们提出了CheXStruct和CXReasonBench，一个基于公开的MIMIC-CXR-JPG数据集构建的结构化管道和基准。CheXStruct自动从胸部X光片中提取一系列中间推理步骤，包括分割解剖区域、提取解剖标志和诊断测量、计算诊断指标以及应用临床阈值。CXReasonBench利用这个管道评估模型是否能够执行临床有效的推理步骤，并且在多路径、多阶段评估中，包括通过解剖区域选择和诊断测量的视觉接地，评估模型从结构化指导中学到的程度，以实现细粒度和透明的诊断推理评估。基准数据集包含18,988个QA对，跨越12项诊断任务和1,200个病例，每个病例最多配有一个4个视觉输入，并支持包括视觉接地在内的多路径、多阶段评估。即使在评估的10个最强LVLM中，也有多数难以完成结构化推理和泛化，往往无法将抽象知识与解剖学相关的视觉解释联系起来。代码可在以下链接获取。 

---
# Backpropagation-Free Metropolis-Adjusted Langevin Algorithm 

**Title (ZH)**: 无回传的甲放手变形林格.GETAM（Metropolis-Adjusted Langevin Algorithm）算法 

**Authors**: Adam D. Cobb, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.18081)  

**Abstract**: Recent work on backpropagation-free learning has shown that it is possible to use forward-mode automatic differentiation (AD) to perform optimization on differentiable models. Forward-mode AD requires sampling a tangent vector for each forward pass of a model. The result is the model evaluation with the directional derivative along the tangent. In this paper, we illustrate how the sampling of this tangent vector can be incorporated into the proposal mechanism for the Metropolis-Adjusted Langevin Algorithm (MALA). As such, we are the first to introduce a backpropagation-free gradient-based Markov chain Monte Carlo (MCMC) algorithm. We also extend to a novel backpropagation-free position-specific preconditioned forward-mode MALA that leverages Hessian information. Overall, we propose four new algorithms: Forward MALA; Line Forward MALA; Pre-conditioned Forward MALA, and Pre-conditioned Line Forward MALA. We highlight the reduced computational cost of the forward-mode samplers and show that forward-mode is competitive with the original MALA, while even outperforming it depending on the probabilistic model. We include Bayesian inference results on a range of probabilistic models, including hierarchical distributions and Bayesian neural networks. 

**Abstract (ZH)**: Recent work on backpropagation-free learning has shown that it is possible to use forward-mode automatic differentiation (AD) to perform optimization on differentiable models. Forward-mode AD requires sampling a tangent vector for each forward pass of a model. The result is the model evaluation with the directional derivative along the tangent. In this paper,我们首次引入了一种无需反向传播的梯度基础马尔可夫链蒙特卡洛(MCMC)算法。我们还将这种前向模式应用到了一个新的位置特定预条件的前向模式MALA算法中，利用了海森矩阵信息。总体而言，我们提出了四种新的算法：前向MALA；线性前向MALA；预条件前向MALA，以及预条件线性前向MALA。我们突出了前向模式采样的计算成本降低，并展示了前向模式在不同概率模型下与原始MALA算法的竞争性和优越性。我们在多层次分布和贝叶斯神经网络等多种概率模型下提供了贝叶斯推理结果。 

---
# AFD-STA: Adaptive Filtering Denoising with Spatiotemporal Attention for Chaotic System Prediction 

**Title (ZH)**: 自适应滤波去噪的空间时间注意力 chaotic系统预测 

**Authors**: Chunlin Gong, Yin Wang, Jingru Li, Hanleran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18080)  

**Abstract**: This paper presents AFD-STA Net, a neural framework integrating adaptive filtering and spatiotemporal dynamics learning for predicting high-dimensional chaotic systems governed by partial differential equations. The architecture combines: 1) An adaptive exponential smoothing module with position-aware decay coefficients for robust attractor reconstruction, 2) Parallel attention mechanisms capturing cross-temporal and spatial dependencies, 3) Dynamic gated fusion of multiscale features, and 4) Deep projection networks with dimension-scaling capabilities. Numerical experiments on nonlinear PDE systems demonstrate the model's effectiveness in maintaining prediction accuracy under both smooth and strongly chaotic regimes while exhibiting noise tolerance through adaptive filtering. Component ablation studies confirm critical contributions from each module, particularly highlighting the essential role of spatiotemporal attention in learning complex dynamical interactions. The framework shows promising potential for real-world applications requiring simultaneous handling of measurement uncertainties and high-dimensional nonlinear dynamics. 

**Abstract (ZH)**: AFD-STA 网络：一种结合自适应滤波和时空动力学学习的神经框架，用于预测受偏微分方程支配的高维混沌系统 

---
# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding 

**Title (ZH)**: 深度视频发现：工具使用辅助的自主搜索在长视频理解中的应用 

**Authors**: Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin Li, Houqiang Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18079)  

**Abstract**: Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later. 

**Abstract (ZH)**: 长视频理解由于广泛的时空复杂性和在如此扩展的背景下回答问题的难度，提出了显著挑战。尽管大规模语言模型在视频分析能力和处理长上下文方面取得了显著进展，但在处理信息密集型一小时长的视频时仍表现出局限性。为克服这些局限性，我们提出Deep Video Discovery代理，利用分割视频片段上的代理式搜索策略。不同于以往的视频代理手动设计僵化的流程，我们的方法强调代理的自主性。通过在多粒度视频数据库上提供以搜索为中心的工具集，DVD代理利用大规模语言模型的高级推理能力，在当前观察状态规划，战略性地选择工具，制定适当的行动参数，并根据收集到的信息迭代优化其内部推理。我们在多个长视频理解基准上进行了全面评估，展示了整个系统设计的优势。我们的DVD代理在具有挑战性的LVBench数据集上显著超越了先前的工作，取得了SOTA性能。还提供了全面的消融研究和深入的工具分析，为针对长视频理解任务的智能代理的进一步发展提供了见解。代码将在稍后发布。 

---
# Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals 

**Title (ZH)**: 个性化偏好推断中的扩展归纳推理 

**Authors**: Jia-Nan Li, Jian Guan, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18071)  

**Abstract**: Large language models (LLMs) have demonstrated significant success in complex reasoning tasks such as math and coding. In contrast to these tasks where deductive reasoning predominates, inductive reasoning\textemdash the ability to derive general rules from incomplete evidence, remains underexplored. This paper investigates extended inductive reasoning in LLMs through the lens of personalized preference inference, a critical challenge in LLM alignment where current approaches struggle to capture diverse user preferences. The task demands strong inductive reasoning capabilities as user preferences are typically embedded implicitly across various interaction forms, requiring models to synthesize consistent preference patterns from scattered signals. We propose \textsc{AlignXplore}, a model that leverages extended reasoning chains to enable systematic preference inference from behavioral signals in users' interaction histories. We develop \textsc{AlignXplore} by combining cold-start training based on synthetic data with subsequent online reinforcement learning. Through extensive experiments, we demonstrate that \textsc{AlignXplore} achieves substantial improvements over the backbone model by an average of 11.05\% on in-domain and out-of-domain benchmarks, while maintaining strong generalization ability across different input formats and downstream models. Further analyses establish best practices for preference inference learning through systematic comparison of reward modeling strategies, while revealing the emergence of human-like inductive reasoning patterns during training. 

**Abstract (ZH)**: 大型语言模型在数学和编码等复杂推理任务中取得了显著成功。相比之下，归纳推理——从不完整证据中推导出一般规则的能力——在现有研究中仍被忽视。本文通过个人偏好推断的视角探讨了大型语言模型中的扩展归纳推理，这是大型语言模型对齐中的一个关键挑战，当前方法难以捕捉多样化的用户偏好。该任务需要强大的归纳推理能力，因为用户偏好通常以隐式方式嵌入在各种交互形式中，要求模型从分散的信号中综合出一致的偏好模式。我们提出了一种名为\textsc{AlignXplore}的模型，该模型利用扩展的推理链从用户的交互历史中的行为信号中系统地推断偏好。我们通过结合基于合成数据的冷启动训练和后续的在线强化学习来开发\textsc{AlignXplore}。通过广泛的实验，我们证明了\textsc{AlignXplore}在领域内和跨领域的基准测试中平均提高了11.05%，并保持了在不同输入格式和下游模型中的强烈泛化能力。进一步的分析通过系统比较奖励模型策略，确定了偏好推断学习的最佳实践，并揭示了在训练过程中出现的人类似推断模式。 

---
# Towards Uncertainty Aware Task Delegation and Human-AI Collaborative Decision-Making 

**Title (ZH)**: 面向不确定性感知的任务委派及人机协同决策研究 

**Authors**: Min Hun Lee, Martyn Zhe Yu Tok  

**Link**: [PDF](https://arxiv.org/pdf/2505.18066)  

**Abstract**: Despite the growing promise of artificial intelligence (AI) in supporting decision-making across domains, fostering appropriate human reliance on AI remains a critical challenge. In this paper, we investigate the utility of exploring distance-based uncertainty scores for task delegation to AI and describe how these scores can be visualized through embedding representations for human-AI decision-making. After developing an AI-based system for physical stroke rehabilitation assessment, we conducted a study with 19 health professionals and 10 students in medicine/health to understand the effect of exploring distance-based uncertainty scores on users' reliance on AI. Our findings showed that distance-based uncertainty scores outperformed traditional probability-based uncertainty scores in identifying uncertain cases. In addition, after exploring confidence scores for task delegation and reviewing embedding-based visualizations of distance-based uncertainty scores, participants achieved an 8.20% higher rate of correct decisions, a 7.15% higher rate of changing their decisions to correct ones, and a 7.14% lower rate of incorrect changes after reviewing AI outputs than those reviewing probability-based uncertainty scores ($p<0.01$). Our findings highlight the potential of distance-based uncertainty scores to enhance decision accuracy and appropriate reliance on AI while discussing ongoing challenges for human-AI collaborative decision-making. 

**Abstract (ZH)**: 尽管人工智能（AI）在支持跨领域决策方面展现了 growing 的潜力，培养适当的人类依赖性仍然是一项关键挑战。本文探讨了基于距离的不确定性评分在任务委托给AI方面的效用，并描述了这些评分如何通过嵌入表示进行可视化，以辅助人类与AI的决策。在开发了一种基于AI的物理中风康复评估系统后，我们对19名医疗专业人员和10名医学院/健康科学学生进行了研究，以了解探索基于距离的不确定性评分对用户依赖AI的影响。研究结果表明，基于距离的不确定性评分在识别不确定案例方面优于基于概率的不确定性评分。此外，在探索任务委托的信心评分并审阅基于距离的不确定性评分的嵌入表示可视化后，参与者在审阅AI输出时正确决策的比例提高了8.20%，将其决策纠正的比例提高了7.15%，并且更改错误决策的比例降低了7.14%（$p<0.01$）。研究结果突显了基于距离的不确定性评分在提高决策准确性和适当依赖AI方面的潜力，同时讨论了人类与AI协作决策中的持续挑战。 

---
# FDBPL: Faster Distillation-Based Prompt Learning for Region-Aware Vision-Language Models Adaptation 

**Title (ZH)**: FDBPL：基于蒸馏的更快(prompt学习)区域 aware 视觉-语言模型适应 

**Authors**: Zherui Zhang, Jiaxin Wu, Changwei Wang, Rongtao Xu, Longzhao Huang, Wenhao Xu, Wenbo Xu, Li Guo, Shibiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18053)  

**Abstract**: Prompt learning as a parameter-efficient method that has been widely adopted to adapt Vision-Language Models (VLMs) to downstream tasks. While hard-prompt design requires domain expertise and iterative optimization, soft-prompt methods rely heavily on task-specific hard labels, limiting their generalization to unseen categories. Recent popular distillation-based prompt learning methods improve generalization by exploiting larger teacher VLMs and unsupervised knowledge transfer, yet their repetitive teacher model online inference sacrifices the inherent training efficiency advantage of prompt learning. In this paper, we propose {\large {\textbf{F}}}aster {\large {\textbf{D}}}istillation-{\large {\textbf{B}}}ased {\large {\textbf{P}}}rompt {\large {\textbf{L}}}earning (\textbf{FDBPL}), which addresses these issues by sharing soft supervision contexts across multiple training stages and implementing accelerated I/O. Furthermore, FDBPL introduces a region-aware prompt learning paradigm with dual positive-negative prompt spaces to fully exploit randomly cropped regions that containing multi-level information. We propose a positive-negative space mutual learning mechanism based on similarity-difference learning, enabling student CLIP models to recognize correct semantics while learning to reject weakly related concepts, thereby improving zero-shot performance. Unlike existing distillation-based prompt learning methods that sacrifice parameter efficiency for generalization, FDBPL maintains dual advantages of parameter efficiency and strong downstream generalization. Comprehensive evaluations across 11 datasets demonstrate superior performance in base-to-new generalization, cross-dataset transfer, and robustness tests, achieving $2.2\times$ faster training speed. 

**Abstract (ZH)**: 更快的蒸馏基于提示学习（Faster Distillation-Based Prompt Learning, FDBPL） 

---
# RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration 

**Title (ZH)**: RestoreVAR：全面图像恢复的视觉自回归生成 

**Authors**: Sudarshan Rajagopalan, Kartik Narayan, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2505.18047)  

**Abstract**: The use of latent diffusion models (LDMs) such as Stable Diffusion has significantly improved the perceptual quality of All-in-One image Restoration (AiOR) methods, while also enhancing their generalization capabilities. However, these LDM-based frameworks suffer from slow inference due to their iterative denoising process, rendering them impractical for time-sensitive applications. To address this, we propose RestoreVAR, a novel generative approach for AiOR that significantly outperforms LDM-based models in restoration performance while achieving over $\mathbf{10\times}$ faster inference. RestoreVAR leverages visual autoregressive modeling (VAR), a recently introduced approach which performs scale-space autoregression for image generation. VAR achieves comparable performance to that of state-of-the-art diffusion transformers with drastically reduced computational costs. To optimally exploit these advantages of VAR for AiOR, we propose architectural modifications and improvements, including intricately designed cross-attention mechanisms and a latent-space refinement module, tailored for the AiOR task. Extensive experiments show that RestoreVAR achieves state-of-the-art performance among generative AiOR methods, while also exhibiting strong generalization capabilities. 

**Abstract (ZH)**: 基于潜扩散模型的全在一ображ restoration 新方法 RestoreVAR：显著提升恢复性能与 inference 速度 

---
# Linear Mixture Distributionally Robust Markov Decision Processes 

**Title (ZH)**: 线性混合分布鲁棒马尔可夫决策过程 

**Authors**: Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18044)  

**Abstract**: Many real-world decision-making problems face the off-dynamics challenge: the agent learns a policy in a source domain and deploys it in a target domain with different state transitions. The distributionally robust Markov decision process (DRMDP) addresses this challenge by finding a robust policy that performs well under the worst-case environment within a pre-specified uncertainty set of transition dynamics. Its effectiveness heavily hinges on the proper design of these uncertainty sets, based on prior knowledge of the dynamics. In this work, we propose a novel linear mixture DRMDP framework, where the nominal dynamics is assumed to be a linear mixture model. In contrast with existing uncertainty sets directly defined as a ball centered around the nominal kernel, linear mixture DRMDPs define the uncertainty sets based on a ball around the mixture weighting parameter. We show that this new framework provides a more refined representation of uncertainties compared to conventional models based on $(s,a)$-rectangularity and $d$-rectangularity, when prior knowledge about the mixture model is present. We propose a meta algorithm for robust policy learning in linear mixture DRMDPs with general $f$-divergence defined uncertainty sets, and analyze its sample complexities under three divergence metrics instantiations: total variation, Kullback-Leibler, and $\chi^2$ divergences. These results establish the statistical learnability of linear mixture DRMDPs, laying the theoretical foundation for future research on this new setting. 

**Abstract (ZH)**: 一种基于线性混合模型的分布鲁棒马尔可夫决策过程及其学习算法研究 

---
# Knot So Simple: A Minimalistic Environment for Spatial Reasoning 

**Title (ZH)**: 结不那么简单：一个简约的空间推理环境 

**Authors**: Zizhao Chen, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18028)  

**Abstract**: We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at this https URL. 

**Abstract (ZH)**: 我们提出KnotGym，一种用于复杂空间 reasoning 和 manipulation 的交互式环境。 

---
# LLM assisted web application functional requirements generation: A case study of four popular LLMs over a Mess Management System 

**Title (ZH)**: 基于大型语言模型的Web应用功能需求生成：对一个后勤管理系统而言的四种流行大型语言模型案例研究 

**Authors**: Rashmi Gupta, Aditya K Gupta, Aarav Jain, Avinash C Pandey, Atul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.18019)  

**Abstract**: Like any other discipline, Large Language Models (LLMs) have significantly impacted software engineering by helping developers generate the required artifacts across various phases of software development. This paper presents a case study comparing the performance of popular LLMs GPT, Claude, Gemini, and DeepSeek in generating functional specifications that include use cases, business rules, and collaborative workflows for a web application, the Mess Management System. The study evaluated the quality of LLM generated use cases, business rules, and collaborative workflows in terms of their syntactic and semantic correctness, consistency, non ambiguity, and completeness compared to the reference specifications against the zero-shot prompted problem statement. Our results suggested that all four LLMs can specify syntactically and semantically correct, mostly non-ambiguous artifacts. Still, they may be inconsistent at times and may differ significantly in the completeness of the generated specification. Claude and Gemini generated all the reference use cases, with Claude achieving the most complete but somewhat redundant use case specifications. Similar results were obtained for specifying workflows. However, all four LLMs struggled to generate relevant Business Rules, with DeepSeek generating the most reference rules but with less completeness. Overall, Claude generated more complete specification artifacts, while Gemini was more precise in the specifications it generated. 

**Abstract (ZH)**: 大型语言模型（LLMs）对软件工程的影响及其在生成WEB应用“餐饮管理系统”功能规格中的表现：GPT、Claude、Gemini和DeepSeek的比较研究 

---
# ExoGait-MS: Learning Periodic Dynamics with Multi-Scale Graph Network for Exoskeleton Gait Recognition 

**Title (ZH)**: ExoGait-MS：基于多尺度图网络学习周期动力学的外骨骼步态识别 

**Authors**: Lijiang Liu, Junyu Shi, Yong Sun, Zhiyuan Zhang, Jinni Zhou, Shugen Ma, Qiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.18018)  

**Abstract**: Current exoskeleton control methods often face challenges in delivering personalized treatment. Standardized walking gaits can lead to patient discomfort or even injury. Therefore, personalized gait is essential for the effectiveness of exoskeleton robots, as it directly impacts their adaptability, comfort, and rehabilitation outcomes for individual users. To enable personalized treatment in exoskeleton-assisted therapy and related applications, accurate recognition of personal gait is crucial for implementing tailored gait control. The key challenge in gait recognition lies in effectively capturing individual differences in subtle gait features caused by joint synergy, such as step frequency and step length. To tackle this issue, we propose a novel approach, which uses Multi-Scale Global Dense Graph Convolutional Networks (GCN) in the spatial domain to identify latent joint synergy patterns. Moreover, we propose a Gait Non-linear Periodic Dynamics Learning module to effectively capture the periodic characteristics of gait in the temporal domain. To support our individual gait recognition task, we have constructed a comprehensive gait dataset that ensures both completeness and reliability. Our experimental results demonstrate that our method achieves an impressive accuracy of 94.34% on this dataset, surpassing the current state-of-the-art (SOTA) by 3.77%. This advancement underscores the potential of our approach to enhance personalized gait control in exoskeleton-assisted therapy. 

**Abstract (ZH)**: 当前的外骨骼控制方法往往难以提供个性化的治疗。标准化的步行模式可能导致患者不适甚至受伤。因此，个性化步态对于外骨骼机器人的有效性至关重要，因为它直接影响其适应性、舒适性和个体用户的康复效果。为了在外骨骼辅助疗法及相关应用中实现个性化治疗，准确识别个人步态对于实施定制步态控制至关重要。步态识别的关键挑战在于有效捕捉由关节协同作用引起的高度细微的步态特征差异，如步频和步长。为此，我们提出了一种新的方法，利用多尺度全局稠密图卷积网络（GCN）在空间域中识别潜在的关节协同模式。此外，我们提出了一种步态非线性周期动力学习模块，以有效地在时间域中捕捉步态的周期特性。为了支持我们的个体步态识别任务，我们构建了一个全面的步态数据集，确保其完整性和可靠性。实验结果表明，我们的方法在该数据集上的准确率为94.34%，超越了当前最先进的方法（SOTA）3.77%。这一进展突显了我们方法在外骨骼辅助治疗中增强个性化步态控制的潜力。 

---
# Training with Pseudo-Code for Instruction Following 

**Title (ZH)**: 使用伪代码进行指令遵循训练 

**Authors**: Prince Kumar, Rudra Murthy, Riyaz Bhat, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2505.18011)  

**Abstract**: Despite the rapid progress in the capabilities of Large Language Models (LLMs), they continue to have difficulty following relatively simple, unambiguous instructions, especially when compositions are involved. In this paper, we take inspiration from recent work that suggests that models may follow instructions better when they are expressed in pseudo-code. However, writing pseudo-code programs can be tedious and using few-shot demonstrations to craft code representations for use in inference can be unnatural for non-expert users of LLMs. To overcome these limitations, we propose fine-tuning LLMs with instruction-tuning data that additionally includes instructions re-expressed in pseudo-code along with the final response. We evaluate models trained using our method on $11$ publicly available benchmarks comprising of tasks related to instruction-following, mathematics, and common-sense reasoning. We conduct rigorous experiments with $5$ different models and find that not only do models follow instructions better when trained with pseudo-code, they also retain their capabilities on the other tasks related to mathematical and common sense reasoning. Specifically, we observe a relative gain of $3$--$19$% on instruction-following benchmark, and an average gain of upto 14% across all tasks. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）的能力迅速进步，它们仍然难以遵循相对简单的明确指令，尤其是在涉及复杂组合时。本文受到近期研究表明的启发，即当指令用伪代码表达时，模型可能能更好地遵循指令。然而，编写伪代码程序可能会非常繁琐，而使用少量示例演示来为LLM用户生成代码表示以供推理则对非专家用户来说并不自然。为克服这些局限，我们提出了一种微调方法，该方法利用包含用伪代码重述的指令和最终响应的数据。我们使用该方法训练模型并在包含指令遵循、数学和常识推理任务的11个公开基准上评估这些模型。我们进行了严格的实验，使用5种不同模型发现，不仅在使用伪代码训练时模型在指令遵循任务中表现更好，而且它们在数学和常识推理相关的其他任务上也保持了其能力。具体而言，在指令遵循基准测试中我们观察到相对增益为3%至19%，在所有任务上的平均增益达到14%。 

---
# AI Literacy for Legal AI Systems: A practical approach 

**Title (ZH)**: 法律AI系统中的AI素养：一种实用方法 

**Authors**: Gizem Gultekin-Varkonyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18006)  

**Abstract**: Legal AI systems are increasingly being adopted by judicial and legal system deployers and providers worldwide to support a range of applications. While they offer potential benefits such as reducing bias, increasing efficiency, and improving accountability, they also pose significant risks, requiring a careful balance between opportunities, and legal and ethical development and deployment. AI literacy, as a legal requirement under the EU AI Act and a critical enabler of ethical AI for deployers and providers, could be a tool to achieve this. The article introduces the term "legal AI systems" and then analyzes the concept of AI literacy and the benefits and risks associated with these systems. This analysis is linked to a broader AI-L concept for organizations that deal with legal AI systems. The outcome of the article, a roadmap questionnaire as a practical tool for developers and providers to assess risks, benefits, and stakeholder concerns, could be useful in meeting societal and regulatory expectations for legal AI. 

**Abstract (ZH)**: Legal AI系统在全球司法和法律系统部署者与提供商中的应用日益增多，以支持多种应用。虽然它们提供了减少偏见、提高效率和增强问责制等潜在好处，但也带来了显著的风险，需要在机会与法律和伦理的发展与部署之间谨慎平衡。作为一种法律要求，欧盟AI法案下的AI素养，以及对部署者和提供商至关重要的人工智能伦理的促进者，可能是实现这一目标的工具。本文介绍了“法律AI系统”这一术语，并分析了AI素养的概念以及这些系统关联的利益和风险。这种分析与组织处理法律AI系统时更广泛的AI-L概念相关联。本文的结果，一份路线图问卷作为开发者和提供商评估风险、利益和利益相关者关切的实际工具，有助于满足社会和监管对法律AI的期望。 

---
# An Example Safety Case for Safeguards Against Misuse 

**Title (ZH)**: 一个针对误用防护的安全案例示例 

**Authors**: Joshua Clymer, Jonah Weinbaum, Robert Kirk, Kimberly Mai, Selena Zhang, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2505.18003)  

**Abstract**: Existing evaluations of AI misuse safeguards provide a patchwork of evidence that is often difficult to connect to real-world decisions. To bridge this gap, we describe an end-to-end argument (a "safety case") that misuse safeguards reduce the risk posed by an AI assistant to low levels. We first describe how a hypothetical developer red teams safeguards, estimating the effort required to evade them. Then, the developer plugs this estimate into a quantitative "uplift model" to determine how much barriers introduced by safeguards dissuade misuse (this https URL). This procedure provides a continuous signal of risk during deployment that helps the developer rapidly respond to emerging threats. Finally, we describe how to tie these components together into a simple safety case. Our work provides one concrete path -- though not the only path -- to rigorously justifying AI misuse risks are low. 

**Abstract (ZH)**: 现有的AI滥用防护评估提供了支离破碎的证据，往往难以连接到现实世界的决策。为弥补这一差距，我们描述了一个从始至终的论证（“安全案例”），说明滥用防护如何将AI助手所带来的风险降低到低水平。我们首先描述一个假设的开发者红队测试这些防护，估计绕过它们所需的努力。然后，开发者将这一估计值输入到一个定量的“提升模型”中，以确定由防护措施引入的障碍对滥用行为的威慑程度（参见此链接：[this https URL]）。此过程提供了一个在部署期间持续的风险信号，帮助开发者迅速应对新兴威胁。最后，我们描述了如何将这些组件整合为一个简单的安全案例。我们的研究提供了一条具体的路径——尽管不是唯一路径——来严格证明AI滥用风险较低。 

---
# Outcome-based Reinforcement Learning to Predict the Future 

**Title (ZH)**: 基于 Outcome 的强化学习预测未来 

**Authors**: Benjamin Turtel, Danny Franklin, Kris Skotheim, Luke Hewitt, Philipp Schoenegger  

**Link**: [PDF](https://arxiv.org/pdf/2505.17989)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has boosted math and coding in large language models, yet there has been little effort to extend RLVR into messier, real-world domains like forecasting. One sticking point is that outcome-based reinforcement learning for forecasting must learn from binary, delayed, and noisy rewards, a regime where standard fine-tuning is brittle. We show that outcome-only online RL on a 14B model can match frontier-scale accuracy and surpass it in calibration and hypothetical prediction market betting by adapting two leading algorithms, Group-Relative Policy Optimisation (GRPO) and ReMax, to the forecasting setting. Our adaptations remove per-question variance scaling in GRPO, apply baseline-subtracted advantages in ReMax, hydrate training with 100k temporally consistent synthetic questions, and introduce lightweight guard-rails that penalise gibberish, non-English responses and missing rationales, enabling a single stable pass over 110k events. Scaling ReMax to 110k questions and ensembling seven predictions yields a 14B model that matches frontier baseline o1 on accuracy on our holdout set (Brier = 0.193, p = 0.23) while beating it in calibration (ECE = 0.042, p < 0.001). A simple trading rule turns this calibration edge into \$127 of hypothetical profit versus \$92 for o1 (p = 0.037). This demonstrates that refined RLVR methods can convert small-scale LLMs into potentially economically valuable forecasting tools, with implications for scaling this to larger models. 

**Abstract (ZH)**: 验证奖励的强化学习在大型语言模型中的数学和编码方面取得了进步，然而很少有人尝试将RLVR扩展到如预测等更复杂的现实世界领域。我们通过将两个领先算法，Group-Relative Policy Optimisation (GRPO) 和 ReMax，适应预测设置，展示了基于结果的在线RL可以在14B模型中达到前沿级别的准确性和更好的校准，以及在假设的预测市场投注中表现出色。通过这些适应，我们在GRPO中消除了答案偏差，为ReMax应用了基线减去的优势，并在训练中加入了10万个多时间一致的合成问题，同时引入了轻量级的监管措施，以惩罚无意义、非英语的回答和缺失的理由，从而使模型能够在11万多个事件上保持稳定。将ReMax扩展到11万问题并ensemble七次预测，14B模型在保留集上达到了前沿基线o1的准确度（Brier = 0.193, p = 0.23），但在校准方面超过了它（ECE = 0.042, p < 0.001）。一个简单的交易规则将这种校准优势转化为127美元的假设盈利，而o1仅为92美元（p = 0.037）。这表明改进的RLVR方法可以使小规模的LLM成为潜在具有经济价值的预测工具，并对扩大这一方法适用于更大规模的模型具有重要意义。 

---
# Towards Revealing the Effectiveness of Small-Scale Fine-tuning in R1-style Reinforcement Learning 

**Title (ZH)**: 面向R1风格强化学习中小规模微调效果的揭示 

**Authors**: Yutong Chen, Jiandong Gao, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17988)  

**Abstract**: R1-style Reinforcement Learning (RL) significantly enhances Large Language Models' reasoning capabilities, yet the mechanism behind rule-based RL remains unclear. We found that small-scale SFT has significant influence on RL but shows poor efficiency. To explain our observations, we propose an analytical framework and compare the efficiency of SFT and RL by measuring sample effect. Hypothetical analysis show that SFT efficiency is limited by training data. Guided by our analysis, we propose Re-distillation, a technique that fine-tunes pretrain model through small-scale distillation from the RL-trained policy. Experiments on Knight & Knave and MATH datasets demonstrate re-distillation's surprising efficiency: re-distilled models match RL performance with far fewer samples and less computation. Empirical verification shows that sample effect is a good indicator of performance improvements. As a result, on K&K dataset, our re-distilled Qwen2.5-1.5B model surpasses DeepSeek-V3-0324 with only 1K SFT samples. On MATH, Qwen2.5-1.5B fine-tuned with re-distilled 500 samples matches its instruct-tuned variant without RL. Our work explains several interesting phenomena in R1-style RL, shedding light on the mechanisms behind its empirical success. Code is available at: this https URL 

**Abstract (ZH)**: R1风格强化学习显著增强了大型语言模型的推理能力，但基于规则的强化学习机制尚不明确。我们发现小规模SFT对强化学习有显著影响，但效率较低。为了解释我们的观察结果，我们提出了一种分析框架，并通过测量样本效果来比较SFT和强化学习的效率。假设分析表明，SFT的效率受限于训练数据。受分析指导，我们提出了一种技术，即重蒸馏，通过从强化学习训练策略的小规模蒸馏微调预训练模型。我们在Knight & Knave和MATH数据集上的实验表明，重蒸馏具有令人惊讶的效率：重蒸馏后的模型在远少于样本和更少计算的情况下达到了与强化学习相当的性能。经验验证表明，样本效果是性能改进的良好指标。因此，在K&K数据集上，我们的重蒸馏Qwen2.5-1.5B模型仅使用1K SFT样本就超越了DeepSeek-V3-0324。在MATH上，经过重蒸馏500样本微调的Qwen2.5-1.5B与没有使用强化学习的指令微调变体性能相当。我们的工作解释了R1风格强化学习中的几种有趣现象，揭示了其实际成功背后的机制。代码可在以下链接获取：this https URL。 

---
# ADLGen: Synthesizing Symbolic, Event-Triggered Sensor Sequences for Human Activity Modeling 

**Title (ZH)**: ADLGen: 生成符号化、事件触发的传感器序列以用于人类活动建模 

**Authors**: Weihang You, Hanqi Jiang, Zishuai Liu, Zihang Xie, Tianming Liu, Jin Lu, Fei Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17987)  

**Abstract**: Real world collection of Activities of Daily Living data is challenging due to privacy concerns, costly deployment and labeling, and the inherent sparsity and imbalance of human behavior. We present ADLGen, a generative framework specifically designed to synthesize realistic, event triggered, and symbolic sensor sequences for ambient assistive environments. ADLGen integrates a decoder only Transformer with sign based symbolic temporal encoding, and a context and layout aware sampling mechanism to guide generation toward semantically rich and physically plausible sensor event sequences. To enhance semantic fidelity and correct structural inconsistencies, we further incorporate a large language model into an automatic generate evaluate refine loop, which verifies logical, behavioral, and temporal coherence and generates correction rules without manual intervention or environment specific tuning. Through comprehensive experiments with novel evaluation metrics, ADLGen is shown to outperform baseline generators in statistical fidelity, semantic richness, and downstream activity recognition, offering a scalable and privacy-preserving solution for ADL data synthesis. 

**Abstract (ZH)**: ADLGen：一种用于环境辅助场景的生成式活动采集框架 

---
# Generalized Fisher-Weighted SVD: Scalable Kronecker-Factored Fisher Approximation for Compressing Large Language Models 

**Title (ZH)**: 广义Fisher加权SVD：压缩大型语言模型的可扩展Kronecker因子Fisher近似方法 

**Authors**: Viktoriia Chekalina, Daniil Moskovskiy, Daria Cherniuk, Maxim Kurkin, Andrey Kuznetsov, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2505.17974)  

**Abstract**: The Fisher information is a fundamental concept for characterizing the sensitivity of parameters in neural networks. However, leveraging the full observed Fisher information is too expensive for large models, so most methods rely on simple diagonal approximations. While efficient, this approach ignores parameter correlations, often resulting in reduced performance on downstream tasks. In this work, we mitigate these limitations and propose Generalized Fisher-Weighted SVD (GFWSVD), a post-training LLM compression technique that accounts for both diagonal and off-diagonal elements of the Fisher information matrix, providing a more accurate reflection of parameter importance. To make the method tractable, we introduce a scalable adaptation of the Kronecker-factored approximation algorithm for the observed Fisher information. We demonstrate the effectiveness of our method on LLM compression, showing improvements over existing compression baselines. For example, at a 20 compression rate on the MMLU benchmark, our method outperforms FWSVD, which is based on a diagonal approximation of the Fisher information, by 5 percent, SVD-LLM by 3 percent, and ASVD by 6 percent compression rate. 

**Abstract (ZH)**: 广义费雪加权SVD在大规模语言模型压缩中的应用：考虑费雪信息矩阵的对角线和非对角线元素 

---
# Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems 

**Title (ZH)**: 大型语言模型是可靠的AI科学家吗？评估黑盒系统的逆向工程能力 

**Authors**: Jiayi Geng, Howard Chen, Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.17968)  

**Abstract**: Using AI to create autonomous researchers has the potential to accelerate scientific discovery. A prerequisite for this vision is understanding how well an AI model can identify the underlying structure of a black-box system from its behavior. In this paper, we explore how well a large language model (LLM) learns to identify a black-box function from passively observed versus actively collected data. We investigate the reverse-engineering capabilities of LLMs across three distinct types of black-box systems, each chosen to represent different problem domains where future autonomous AI researchers may have considerable impact: Program, Formal Language, and Math Equation. Through extensive experiments, we show that LLMs fail to extract information from observations, reaching a performance plateau that falls short of the ideal of Bayesian inference. However, we demonstrate that prompting LLMs to not only observe but also intervene -- actively querying the black-box with specific inputs to observe the resulting output -- improves performance by allowing LLMs to test edge cases and refine their beliefs. By providing the intervention data from one LLM to another, we show that this improvement is partly a result of engaging in the process of generating effective interventions, paralleling results in the literature on human learning. Further analysis reveals that engaging in intervention can help LLMs escape from two common failure modes: overcomplication, where the LLM falsely assumes prior knowledge about the black-box, and overlooking, where the LLM fails to incorporate observations. These insights provide practical guidance for helping LLMs more effectively reverse-engineer black-box systems, supporting their use in making new discoveries. 

**Abstract (ZH)**: 使用AI创建自主研究人员有望加速科学发现。本研究的前提是理解AI模型如何从行为中识别黑盒系统的潜在结构。本文探讨大型语言模型（LLM）如何从被动观察的数据与主动收集的数据中学习识别黑盒函数。我们研究了LLM在三种不同类型的黑盒系统中的逆向工程能力，这三种系统分别代表未来自主AI研究人员可能产生重大影响的不同问题领域：程序、形式语言和数学方程。通过大量实验，我们展示了LLM从观察中提取信息的能力有限，性能达到一个 plateau，低于贝叶斯推断的理想水平。然而，我们证明通过促使LLM不仅观察还干预——即用特定输入主动查询黑盒并观察结果——可以提高性能，这使LLM能够测试边界情况并完善其信念。通过向另一个LLM提供干预数据，我们展示了这种改进部分是通过生成有效干预的过程实现的，与人类学习文献中的结果相呼应。进一步分析揭示了干预可以帮助LLM避免两种常见失败模式：过度复杂化和忽视，前者是LLM错误地假设了对黑盒的先验知识，后者是LLM未能整合观察。这些见解为帮助LLM更有效地逆向工程黑盒系统提供了实用指导，支持它们在新发现中的应用。 

---
# SVD-Free Low-Rank Adaptive Gradient Optimization for Large Language Models 

**Title (ZH)**: 无需SVD的低秩自适应梯度优化方法用于大规模语言模型 

**Authors**: Ionut-Vlad Modoranu, Mher Safaryan, Erik Schultheis, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2505.17967)  

**Abstract**: Low-rank optimization has emerged as a promising direction in training large language models (LLMs) to reduce the memory usage of adaptive optimizers by constraining learning to a lower-dimensional space. Prior work typically projects gradients of linear layers using approaches based on Singular Value Decomposition (SVD). However, applying SVD-based procedures individually to each layer in large models is computationally expensive and incurs additional memory costs due to storing the projection matrices. In this work, we propose a computationally efficient and conceptually simple two-step procedure to approximate SVD-based gradient projections into lower-dimensional spaces. First, we construct a complete orthogonal basis using predefined orthogonal matrices of the Discrete Cosine Transform (DCT). Second, we adaptively select basis columns based on their alignment with the gradient of each layer. Each projection matrix in our method is obtained via a single matrix multiplication followed by a lightweight sorting step to identify the most relevant basis vectors. Due to the predefined nature of the orthogonal bases, they are computed once at the start of training. During training, we store only the indices of the selected columns, avoiding the need to store full projection matrices for each layer. Our numerical experiments on both pre-training and fine-tuning tasks demonstrate the effectiveness of our dual strategy in approximating optimal low-rank projections, matching the performance of costly SVD-based methods while achieving faster runtime and reduced memory usage. 

**Abstract (ZH)**: 低秩优化已成为训练大规模语言模型（LLMs）的一个有前途的方向，通过将学习限制在低维空间中来减少自适应优化器的内存使用。先前的工作通常使用奇异值分解（SVD）基于的方法将线性层的梯度投影到低维空间。然而，对大型模型中的每一层单独应用基于SVD的程序在计算上昂贵，并且由于需要存储投影矩阵而产生额外的内存成本。在本项工作中，我们提出了一种计算高效且概念简单的两步方法来近似基于SVD的梯度投影到低维空间。首先，我们使用离散余弦变换（DCT）的预定义正交矩阵构建完整的正交基。其次，我们根据每个层的梯度与正交基列的对齐情况适当地选择基列。在我们的方法中，每个投影矩阵通过单一的矩阵乘法并随后通过轻量级的排序步骤来识别最相关的基向量获得。由于正交基的预定义性质，它们仅在训练开始时计算一次。在训练过程中，我们仅存储所选列的索引，从而避免为每一层存储完整的投影矩阵。我们的数值实验表明，在预训练和微调任务中，我们的双策略在近似最优低秩投影方面有效，能够匹配昂贵的SVD基方法的性能，同时实现更快的运行时间和减少的内存使用。 

---
# Federated Causal Inference from Multi-Site Observational Data via Propensity Score Aggregation 

**Title (ZH)**: 多中心观察数据通过倾向得分聚合的联邦因果推断 

**Authors**: Khellaf Rémi, Bellet Aurélien, Josse Julie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17961)  

**Abstract**: Causal inference typically assumes centralized access to individual-level data. Yet, in practice, data are often decentralized across multiple sites, making centralization infeasible due to privacy, logistical, or legal constraints. We address this by estimating the Average Treatment Effect (ATE) from decentralized observational data using federated learning, which enables inference through the exchange of aggregate statistics rather than individual-level data. We propose a novel method to estimate propensity scores in a (non-)parametric manner by computing a federated weighted average of local scores, using two theoretically grounded weighting schemes -- Membership Weights (MW) and Density Ratio Weights (DW) -- that balance communication efficiency and model flexibility. These federated scores are then used to construct two ATE estimators: the Federated Inverse Propensity Weighting estimator (Fed-IPW) and its augmented variant (Fed-AIPW). Unlike meta-analysis methods, which fail when any site violates positivity, our approach leverages heterogeneity in treatment assignment across sites to improve overlap. We show that Fed-IPW and Fed-AIPW perform well under site-level heterogeneity in sample sizes, treatment mechanisms, and covariate distributions, with theoretical analysis and experiments on simulated and real-world data highlighting their strengths and limitations relative to meta-analysis and related methods. 

**Abstract (ZH)**: 分散数据分析中的因果推断：使用联邦学习估计平均处理效应 

---
# Beyond Distillation: Pushing the Limits of Medical LLM Reasoning with Minimalist Rule-Based RL 

**Title (ZH)**: 超越蒸馏：以 minimalist 规则为基础的 RL 推动医疗 LLM 推理的极限 

**Authors**: Che Liu, Haozhe Wang, Jiazhen Pan, Zhongwei Wan, Yong Dai, Fangzhen Lin, Wenjia Bai, Daniel Rueckert, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17952)  

**Abstract**: Improving performance on complex tasks and enabling interpretable decision making in large language models (LLMs), especially for clinical applications, requires effective reasoning. Yet this remains challenging without supervised fine-tuning (SFT) on costly chain-of-thought (CoT) data distilled from closed-source models (e.g., GPT-4o). In this work, we present AlphaMed, the first medical LLM to show that reasoning capability can emerge purely through reinforcement learning (RL), using minimalist rule-based rewards on public multiple-choice QA datasets, without relying on SFT or distilled CoT data. AlphaMed achieves state-of-the-art results on six medical QA benchmarks, outperforming models trained with conventional SFT+RL pipelines. On challenging benchmarks (e.g., MedXpert), AlphaMed even surpasses larger or closed-source models such as DeepSeek-V3-671B and Claude-3.5-Sonnet. To understand the factors behind this success, we conduct a comprehensive data-centric analysis guided by three questions: (i) Can minimalist rule-based RL incentivize reasoning without distilled CoT supervision? (ii) How do dataset quantity and diversity impact reasoning? (iii) How does question difficulty shape the emergence and generalization of reasoning? Our findings show that dataset informativeness is a key driver of reasoning performance, and that minimalist RL on informative, multiple-choice QA data is effective at inducing reasoning without CoT supervision. We also observe divergent trends across benchmarks, underscoring limitations in current evaluation and the need for more challenging, reasoning-oriented medical QA benchmarks. 

**Abstract (ZH)**: 提高大型语言模型在复杂任务上的性能并使其在临床应用中实现可解释的决策制定需要有效的推理能力。然而，在没有通过对昂贵的链式思考（CoT）数据进行监督微调（SFT）的情况下，这仍然是一个挑战，这些链式思考数据多来源于封闭源模型（例如GPT-4o）。在这项工作中，我们介绍了AlphaMed，这是第一个通过强化学习（RL）纯粹涌现推理能力的医疗LLM，使用简化的基于规则的奖励在公共多选题QA数据集上，而不依赖于SFT或提炼的CoT数据。AlphaMed在六项医疗QA基准测试中达到了最先进的结果，超越了使用常规SFT+RL管道训练的模型。在具有挑战性的基准测试（如MedXpert）上，AlphaMed甚至超越了更大或封闭源模型，如DeepSeek-V3-671B和Claude-3.5-Sonnet。为了理解这种成功背后的因素，我们根据以下三个问题进行了全面的数据导向分析：（i）简化的基于规则的RL能否在没有提炼的CoT监督的情况下激励推理？（ii）数据集的数量和多样性如何影响推理？（iii）问题难度如何塑造推理的涌现和泛化？我们的研究发现，数据集的信息性是推理性能的关键驱动因素，简化的基于规则的RL在信息性的、多选题QA数据上有效诱导了无需CoT监督的推理。我们还观察到基准测试之间存在分歧的趋势，突显了当前评估中的局限性，并强调了需要更多具有挑战性和推理导向的医疗QA基准测试的必要性。 

---
# Handling Symbolic Language in Student Texts: A Comparative Study of NLP Embedding Models 

**Title (ZH)**: 处理学生文本中的符号语言：NLP嵌入模型的比较研究 

**Authors**: Tom Bleckmann, Paul Tschisgale  

**Link**: [PDF](https://arxiv.org/pdf/2505.17950)  

**Abstract**: Recent advancements in Natural Language Processing (NLP) have facilitated the analysis of student-generated language products in learning analytics (LA), particularly through the use of NLP embedding models. Yet when it comes to science-related language, symbolic expressions such as equations and formulas introduce challenges that current embedding models struggle to address. Existing studies and applications often either overlook these challenges or remove symbolic expressions altogether, potentially leading to biased findings and diminished performance of LA applications. This study therefore explores how contemporary embedding models differ in their capability to process and interpret science-related symbolic expressions. To this end, various embedding models are evaluated using physics-specific symbolic expressions drawn from authentic student responses, with performance assessed via two approaches: similarity-based analyses and integration into a machine learning pipeline. Our findings reveal significant differences in model performance, with OpenAI's GPT-text-embedding-3-large outperforming all other examined models, though its advantage over other models was moderate rather than decisive. Beyond performance, additional factors such as cost, regulatory compliance, and model transparency are discussed as key considerations for model selection. Overall, this study underscores the importance for LA researchers and practitioners of carefully selecting NLP embedding models when working with science-related language products that include symbolic expressions. 

**Abstract (ZH)**: 近期自然语言处理（NLP）的进展促进了学习分析（LA）中对学生生成语言产品进行分析，特别是在使用NLP嵌入模型时。然而，当涉及到科学相关的语言时，如方程式和公式等符号表达式，当前的嵌入模型面临挑战。现有研究和应用要么忽视这些挑战，要么完全去除符号表达式，这可能导致分析结果偏差和学习分析应用性能下降。因此，本研究旨在探讨当代嵌入模型在处理和解释科学相关符号表达式方面的差异。为此，研究使用来源于真实学生回答的物理专业符号表达式，评估了各种嵌入模型，并通过相似性分析和机器学习管道集成两种方法对模型性能进行评估。研究发现，在所有评估的模型中，OpenAI的GPT-text-embedding-3-large表现出最佳性能，尽管与其他模型相比其优势并不明显。除了性能之外，成本、合规性和模型透明度等也是选择模型时的重要考虑因素。总之，本研究强调了在处理包含符号表达式的科学相关语言产品时，学习分析研究人员和实践者精心选择NLP嵌入模型的重要性。 

---
# LMask: Learn to Solve Constrained Routing Problems with Lazy Masking 

**Title (ZH)**: LMask: 基于懒惰掩码学习解决约束路由问题 

**Authors**: Tianyou Li, Haijun Zou, Jiayuan Wu, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17938)  

**Abstract**: Routing problems are canonical combinatorial optimization tasks with wide-ranging applications in logistics, transportation, and supply chain management. However, solving these problems becomes significantly more challenging when complex constraints are involved. In this paper, we propose LMask, a novel learning framework that utilizes dynamic masking to generate high-quality feasible solutions for constrained routing problems. LMask introduces the LazyMask decoding method, which lazily refines feasibility masks with the backtracking mechanism. In addition, it employs the refinement intensity embedding to encode the search trace into the model, mitigating representation ambiguities induced by backtracking. To further reduce sampling cost, LMask sets a backtracking budget during decoding, while constraint violations are penalized in the loss function during training to counteract infeasibility caused by this budget. We provide theoretical guarantees for the validity and probabilistic optimality of our approach. Extensive experiments on the traveling salesman problem with time windows (TSPTW) and TSP with draft limits (TSPDL) demonstrate that LMask achieves state-of-the-art feasibility rates and solution quality, outperforming existing neural methods. 

**Abstract (ZH)**: 约束路由问题是一种在物流、交通和供应链管理等领域广泛应用的典型组合优化任务。然而，当涉及复杂约束时，解决这些问题变得显著更具挑战性。在本文中，我们提出了一种新颖的学习框架LMask，利用动态掩码生成约束路由问题的高质量可行解。LMask引入了懒惰掩码解码方法，该方法通过回溯机制lazy地精化可行性掩码。此外，它使用精化强度嵌入将搜索轨迹编码到模型中，从而减轻由回溯引起的表示歧义。为了进一步降低采样成本，解码时LMask设置了一个回溯预算，而在训练过程中通过在损失函数中惩罚约束违反情况来弥补由此预算引起的不可行性。我们为该方法的有效性和概率最优性提供了理论保证。在带时间窗的旅行商问题（TSPTW）和带有载重限制的TSP（TSPDL）上的广泛实验表明，LMask实现了最先进的可行率和解的质量，并优于现有神经网络方法。 

---
# AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation of Foundation Models 

**Title (ZH)**: AutoMiSeg: 通过测试时适应基础模型实现自动医学图像分割 

**Authors**: Xingjian Li, Qifeng Wu, Colleen Que, Yiran Ding, Adithya S. Ubaradka, Jianhua Xing, Tianyang Wang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17931)  

**Abstract**: Medical image segmentation is vital for clinical diagnosis, yet current deep learning methods often demand extensive expert effort, i.e., either through annotating large training datasets or providing prompts at inference time for each new case. This paper introduces a zero-shot and automatic segmentation pipeline that combines off-the-shelf vision-language and segmentation foundation models. Given a medical image and a task definition (e.g., "segment the optic disc in an eye fundus image"), our method uses a grounding model to generate an initial bounding box, followed by a visual prompt boosting module that enhance the prompts, which are then processed by a promptable segmentation model to produce the final mask. To address the challenges of domain gap and result verification, we introduce a test-time adaptation framework featuring a set of learnable adaptors that align the medical inputs with foundation model representations. Its hyperparameters are optimized via Bayesian Optimization, guided by a proxy validation model without requiring ground-truth labels. Our pipeline offers an annotation-efficient and scalable solution for zero-shot medical image segmentation across diverse tasks. Our pipeline is evaluated on seven diverse medical imaging datasets and shows promising results. By proper decomposition and test-time adaptation, our fully automatic pipeline performs competitively with weakly-prompted interactive foundation models. 

**Abstract (ZH)**: 基于视觉-语言和分割基础模型的零样本自动医学图像分割管道 

---
# Towards Practical Defect-Focused Automated Code Review 

**Title (ZH)**: 面向实践的缺陷聚焦自动代码审查 

**Authors**: Junyi Lu, Lili Jiang, Xiaojia Li, Jianbing Fang, Fengjun Zhang, Li Yang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17928)  

**Abstract**: The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: 1) capturing relevant context, 2) improving key bug inclusion (KBI), 3) reducing false alarm rates (FAR), and 4) integrating human workflows. To tackle these, we propose 1) code slicing algorithms for context extraction, 2) a multi-role LLM framework for KBI, 3) a filtering mechanism for FAR reduction, and 4) a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2x improvement over standard LLMs and a 10x gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages language-agnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability. 

**Abstract (ZH)**: 代码审核的复杂性推动了自动审核评论的研究，但先前的方法通过将其简化为片段级代码到文本生成任务，并依赖如BLEU等文本相似度指标进行评估，忽略了仓库上下文、实际合并请求评估和缺陷检测，限制了其实用性。为解决这些问题，我们研究了一家拥有近4亿日活跃用户的公司在线推荐服务中的全流程自动化管道，分析包含数十万行代码的工业级C++代码库。我们识别了四个关键挑战：1) 捕获相关上下文，2) 改进关键错误包含（KBI），3) 降低误报率（FAR），4) 集成人工流程。为此，我们提出了1) 代码切片算法进行上下文提取，2) 多角色LLM框架进行KBI，3) 过滤机制以降低FAR，4) 新颖的提示设计以改善与人工的交互。我们的方法在基于历史故障报告的实际合并请求上验证，实现了标准LLM的2倍改进和之前基准的10倍提升。虽然展示的结果集中在C++上，但底层框架设计利用了语言无关的原则（如基于抽象语法树的分析），表明其有更广泛的适用潜力。 

---
# Evaluation of Few-Shot Learning Methods for Kidney Stone Type Recognition in Ureteroscopy 

**Title (ZH)**: 尿道镜下肾结石类型识别的少样本学习方法评价 

**Authors**: Carlos Salazar-Ruiz, Francisco Lopez-Tiro, Ivan Reyes-Amezcua, Clement Larose, Gilberto Ochoa-Ruiz, Christian Daul  

**Link**: [PDF](https://arxiv.org/pdf/2505.17921)  

**Abstract**: Determining the type of kidney stones is crucial for prescribing appropriate treatments to prevent recurrence. Currently, various approaches exist to identify the type of kidney stones. However, obtaining results through the reference ex vivo identification procedure can take several weeks, while in vivo visual recognition requires highly trained specialists. For this reason, deep learning models have been developed to provide urologists with an automated classification of kidney stones during ureteroscopies. Nevertheless, a common issue with these models is the lack of training data. This contribution presents a deep learning method based on few-shot learning, aimed at producing sufficiently discriminative features for identifying kidney stone types in endoscopic images, even with a very limited number of samples. This approach was specifically designed for scenarios where endoscopic images are scarce or where uncommon classes are present, enabling classification even with a limited training dataset. The results demonstrate that Prototypical Networks, using up to 25% of the training data, can achieve performance equal to or better than traditional deep learning models trained with the complete dataset. 

**Abstract (ZH)**: 基于少样本学习的深度学习方法在内窥镜图像中识别肾结石类型 

---
# Object-level Cross-view Geo-localization with Location Enhancement and Multi-Head Cross Attention 

**Title (ZH)**: 基于位置增强和多头跨注意力的对象级别跨视图地理定位 

**Authors**: Zheyang Huang, Jagannath Aryal, Saeid Nahavandi, Xuequan Lu, Chee Peng Lim, Lei Wei, Hailing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17911)  

**Abstract**: Cross-view geo-localization determines the location of a query image, captured by a drone or ground-based camera, by matching it to a geo-referenced satellite image. While traditional approaches focus on image-level localization, many applications, such as search-and-rescue, infrastructure inspection, and precision delivery, demand object-level accuracy. This enables users to prompt a specific object with a single click on a drone image to retrieve precise geo-tagged information of the object. However, variations in viewpoints, timing, and imaging conditions pose significant challenges, especially when identifying visually similar objects in extensive satellite imagery. To address these challenges, we propose an Object-level Cross-view Geo-localization Network (OCGNet). It integrates user-specified click locations using Gaussian Kernel Transfer (GKT) to preserve location information throughout the network. This cue is dually embedded into the feature encoder and feature matching blocks, ensuring robust object-specific localization. Additionally, OCGNet incorporates a Location Enhancement (LE) module and a Multi-Head Cross Attention (MHCA) module to adaptively emphasize object-specific features or expand focus to relevant contextual regions when necessary. OCGNet achieves state-of-the-art performance on a public dataset, CVOGL. It also demonstrates few-shot learning capabilities, effectively generalizing from limited examples, making it suitable for diverse applications (this https URL). 

**Abstract (ZH)**: 跨视角对象级地理定位网络（OCGNet）通过将用户指定的点击位置集成到高斯核转移（GKT）中，来在整个网络中保留位置信息。该提示信号被双渠道嵌入到特征编码器和特征匹配模块中，确保了对象特定的鲁棒定位。此外，OCGNet 还包含一个位置增强（LE）模块和一个多头跨注意力（MHCA）模块，以适应性地强调对象特定的特征或扩展关注到相关的上下文区域。OCGNet 在公开数据集 CVOGL 上取得了最先进的性能，并展示了少量样本学习能力，有效地从少量示例中泛化，使其适用于多种应用（请点击下方链接了解更多信息：https://github.com/XXX/Osubset）。 

---
# DiffusionReward: Enhancing Blind Face Restoration through Reward Feedback Learning 

**Title (ZH)**: DiffusionReward: 通过奖励反馈学习增强 blindness 面部恢复 

**Authors**: Bin Wu, Wei Wang, Yahui Liu, Zixiang Li, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17910)  

**Abstract**: Reward Feedback Learning (ReFL) has recently shown great potential in aligning model outputs with human preferences across various generative tasks. In this work, we introduce a ReFL framework, named DiffusionReward, to the Blind Face Restoration task for the first time. DiffusionReward effectively overcomes the limitations of diffusion-based methods, which often fail to generate realistic facial details and exhibit poor identity consistency. The core of our framework is the Face Reward Model (FRM), which is trained using carefully annotated data. It provides feedback signals that play a pivotal role in steering the optimization process of the restoration network. In particular, our ReFL framework incorporates a gradient flow into the denoising process of off-the-shelf face restoration methods to guide the update of model parameters. The guiding gradient is collaboratively determined by three aspects: (i) the FRM to ensure the perceptual quality of the restored faces; (ii) a regularization term that functions as a safeguard to preserve generative diversity; and (iii) a structural consistency constraint to maintain facial fidelity. Furthermore, the FRM undergoes dynamic optimization throughout the process. It not only ensures that the restoration network stays precisely aligned with the real face manifold, but also effectively prevents reward hacking. Experiments on synthetic and wild datasets demonstrate that our method outperforms state-of-the-art methods, significantly improving identity consistency and facial details. The source codes, data, and models are available at: this https URL. 

**Abstract (ZH)**: Reward Feedback Learning在盲 Face 修复任务中的应用：一种名为DiffusionReward的框架 

---
# NeuroTrails: Training with Dynamic Sparse Heads as the Key to Effective Ensembling 

**Title (ZH)**: NeuroTrails: 动态稀疏头作为有效ensemble的关键训练方法 

**Authors**: Bram Grooten, Farid Hasanov, Chenxiang Zhang, Qiao Xiao, Boqian Wu, Zahra Atashgahi, Ghada Sokar, Shiwei Liu, Lu Yin, Elena Mocanu, Mykola Pechenizkiy, Decebal Constantin Mocanu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17909)  

**Abstract**: Model ensembles have long been a cornerstone for improving generalization and robustness in deep learning. However, their effectiveness often comes at the cost of substantial computational overhead. To address this issue, state-of-the-art methods aim to replicate ensemble-class performance without requiring multiple independently trained networks. Unfortunately, these algorithms often still demand considerable compute at inference. In response to these limitations, we introduce $\textbf{NeuroTrails}$, a sparse multi-head architecture with dynamically evolving topology. This unexplored model-agnostic training paradigm improves ensemble performance while reducing the required resources. We analyze the underlying reason for its effectiveness and observe that the various neural trails induced by dynamic sparsity attain a $\textit{Goldilocks zone}$ of prediction diversity. NeuroTrails displays efficacy with convolutional and transformer-based architectures on computer vision and language tasks. Experiments on ResNet-50/ImageNet, LLaMA-350M/C4, among many others, demonstrate increased accuracy and stronger robustness in zero-shot generalization, while requiring significantly fewer parameters. 

**Abstract (ZH)**: 神经轨迹：一种动态演化拓扑的稀疏多头架构以提高集成性能并减少资源需求 

---
# DataRater: Meta-Learned Dataset Curation 

**Title (ZH)**: DataRater: 元学习的数据集策展 

**Authors**: Dan A. Calian, Gregory Farquhar, Iurii Kemaev, Luisa M. Zintgraf, Matteo Hessel, Jeremy Shar, Junhyuk Oh, András György, Tom Schaul, Jeffrey Dean, Hado van Hasselt, David Silver  

**Link**: [PDF](https://arxiv.org/pdf/2505.17895)  

**Abstract**: The quality of foundation models depends heavily on their training data. Consequently, great efforts have been put into dataset curation. Yet most approaches rely on manual tuning of coarse-grained mixtures of large buckets of data, or filtering by hand-crafted heuristics. An approach that is ultimately more scalable (let alone more satisfying) is to \emph{learn} which data is actually valuable for training. This type of meta-learning could allow more sophisticated, fine-grained, and effective curation. Our proposed \emph{DataRater} is an instance of this idea. It estimates the value of training on any particular data point. This is done by meta-learning using `meta-gradients', with the objective of improving training efficiency on held out data. In extensive experiments across a range of model scales and datasets, we find that using our DataRater to filter data is highly effective, resulting in significantly improved compute efficiency. 

**Abstract (ZH)**: 基础模型的质量高度依赖于其训练数据。因此，人们投入了大量的精力进行数据集整理。然而，大多数方法依赖于手工调整粗粒度的大数据桶混合或手工设计的启发式过滤。一种更具扩展性（更不用说更令人满意）的方法是通过学习哪些数据实际上对训练有价值。这种元学习可以使得数据整理更加复杂、精细且有效。我们提出的DataRater是这一想法的一个实例。它通过元学习方法计算任何特定数据点的训练价值，目标是提高保留数据的训练效率。在广泛实验中，我们发现使用DataRater筛选数据非常有效，显著提高了计算效率。 

---
# Mutarjim: Advancing Bidirectional Arabic-English Translation with a Small Language Model 

**Title (ZH)**: Mutarjim: 用小型语言模型促进双向阿拉伯语-英语翻译 

**Authors**: Khalil Hennara, Muhammad Hreden, Mohamed Motaism Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17894)  

**Abstract**: We introduce Mutarjim, a compact yet powerful language model for bidirectional Arabic-English translation. While large-scale LLMs have shown impressive progress in natural language processing tasks, including machine translation, smaller models. Leveraging this insight, we developed Mutarjim based on Kuwain-1.5B , a language model tailored for both Arabic and English. Despite its modest size, Mutarjim outperforms much larger models on several established benchmarks, achieved through an optimized two-phase training approach and a carefully curated, high-quality training corpus.. Experimental results show that Mutarjim rivals models up to 20 times larger while significantly reducing computational costs and training requirements. We also introduce Tarjama-25, a new benchmark designed to overcome limitations in existing Arabic-English benchmarking datasets, such as domain narrowness, short sentence lengths, and English-source bias. Tarjama-25 comprises 5,000 expert-reviewed sentence pairs and spans a wide range of domains, offering a more comprehensive and balanced evaluation framework. Notably, Mutarjim achieves state-of-the-art performance on the English-to-Arabic task in Tarjama-25, surpassing even significantly larger and proprietary models like GPT-4o mini. We publicly release Tarjama-25 to support future research and advance the evaluation of Arabic-English translation systems. 

**Abstract (ZH)**: Mutarjim：一种紧凑而强大的双向阿拉伯语-英语语言模型 

---
# FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks 

**Title (ZH)**: FastCAV: 效率计算概念激活向量以解释深度神经网络 

**Authors**: Laines Schmalwasser, Niklas Penzel, Joachim Denzler, Julia Niebling  

**Link**: [PDF](https://arxiv.org/pdf/2505.17883)  

**Abstract**: Concepts such as objects, patterns, and shapes are how humans understand the world. Building on this intuition, concept-based explainability methods aim to study representations learned by deep neural networks in relation to human-understandable concepts. Here, Concept Activation Vectors (CAVs) are an important tool and can identify whether a model learned a concept or not. However, the computational cost and time requirements of existing CAV computation pose a significant challenge, particularly in large-scale, high-dimensional architectures. To address this limitation, we introduce FastCAV, a novel approach that accelerates the extraction of CAVs by up to 63.6x (on average 46.4x). We provide a theoretical foundation for our approach and give concrete assumptions under which it is equivalent to established SVM-based methods. Our empirical results demonstrate that CAVs calculated with FastCAV maintain similar performance while being more efficient and stable. In downstream applications, i.e., concept-based explanation methods, we show that FastCAV can act as a replacement leading to equivalent insights. Hence, our approach enables previously infeasible investigations of deep models, which we demonstrate by tracking the evolution of concepts during model training. 

**Abstract (ZH)**: 基于概念的解释方法旨在研究深度神经网络学习到的表示与人类可理解的概念之间的关系。概念激活向量（CAVs）是这一研究中重要的工具，能够识别模型是否学习了某一概念。然而，现有CAV计算的高计算成本和时间要求构成了一个显著挑战，特别是在大规模、高维架构中。为解决这一限制，我们提出了一种名为FastCAV的新方法，该方法在CAV提取上可加速63.6倍（平均加速46.4倍）。我们为该方法提供了理论基础，并在具体假设下，它等同于已建立的SVM基方法。我们的实验结果表明，使用FastCAV计算的CAVs在保持类似性能的同时更具高效性和稳定性。在下游应用中，即概念基解释方法中，我们证明FastCAV可以作为替代品，提供等效的洞察。因此，我们的方法使以前不可行的深度模型研究成为可能，我们通过追踪模型训练过程中概念的演变来展示这一点。 

---
# Toward Optimal ANC: Establishing Mutual Information Lower Bound 

**Title (ZH)**: Toward Optimal ANC: Establishing Mutual Information Lower Bound 

**Authors**: François Derrida, Shahar Lutati, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17877)  

**Abstract**: Active Noise Cancellation (ANC) algorithms aim to suppress unwanted acoustic disturbances by generating anti-noise signals that destructively interfere with the original noise in real time. Although recent deep learning-based ANC algorithms have set new performance benchmarks, there remains a shortage of theoretical limits to rigorously assess their improvements. To address this, we derive a unified lower bound on cancellation performance composed of two components. The first component is information-theoretic: it links residual error power to the fraction of disturbance entropy captured by the anti-noise signal, thereby quantifying limits imposed by information-processing capacity. The second component is support-based: it measures the irreducible error arising in frequency bands that the cancellation path cannot address, reflecting fundamental physical constraints. By taking the maximum of these two terms, our bound establishes a theoretical ceiling on the Normalized Mean Squared Error (NMSE) attainable by any ANC algorithm. We validate its tightness empirically on the NOISEX dataset under varying reverberation times, demonstrating robustness across diverse acoustic conditions. 

**Abstract (ZH)**: 基于信息理论和支持理论的统一下界：主动噪声取消算法的理论上限 

---
# MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback 

**Title (ZH)**: MOOSE-Chem3：基于模拟实验反馈的实验指导假设排序研究 

**Authors**: Wanhao Liu, Zonglin Yang, Jue Wang, Lidong Bing, Di Zhang, Dongzhan Zhou, Yuqiang Li, Houqiang Li, Erik Cambria, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17873)  

**Abstract**: Hypothesis ranking is a crucial component of automated scientific discovery, particularly in natural sciences where wet-lab experiments are costly and throughput-limited. Existing approaches focus on pre-experiment ranking, relying solely on large language model's internal reasoning without incorporating empirical outcomes from experiments. We introduce the task of experiment-guided ranking, which aims to prioritize candidate hypotheses based on the results of previously tested ones. However, developing such strategies is challenging due to the impracticality of repeatedly conducting real experiments in natural science domains. To address this, we propose a simulator grounded in three domain-informed assumptions, modeling hypothesis performance as a function of similarity to a known ground truth hypothesis, perturbed by noise. We curate a dataset of 124 chemistry hypotheses with experimentally reported outcomes to validate the simulator. Building on this simulator, we develop a pseudo experiment-guided ranking method that clusters hypotheses by shared functional characteristics and prioritizes candidates based on insights derived from simulated experimental feedback. Experiments show that our method outperforms pre-experiment baselines and strong ablations. 

**Abstract (ZH)**: 实验引导的假设排序在自动科学发现中的应用：基于化学假设的仿真模拟及排序方法 

---
# Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time Series Forecasting 

**Title (ZH)**: 低秩适应与部分参数共享的混合时间序列预测方法 

**Authors**: Licheng Pan, Zhichao Chen, Haoxuan Li, Guangyi Liu, Zhijian Xu, Zhaoran Liu, Hao Wang, Ying Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.17872)  

**Abstract**: Multi-task forecasting has become the standard approach for time-series forecasting (TSF). However, we show that it suffers from an Expressiveness Bottleneck, where predictions at different time steps share the same representation, leading to unavoidable errors even with optimal representations. To address this issue, we propose a two-stage framework: first, pre-train a foundation model for one-step-ahead prediction; then, adapt it using step-specific LoRA this http URL design enables the foundation model to handle any number of forecast steps while avoiding the expressiveness bottleneck. We further introduce the Mixture-of-LoRA (MoLA) model, which employs adaptively weighted LoRA experts to achieve partial parameter sharing across steps. This approach enhances both efficiency and forecasting performance by exploiting interdependencies between forecast steps. Experiments show that MoLA significantly improves model expressiveness and outperforms state-of-the-art time-series forecasting methods. Code is available at this https URL. 

**Abstract (ZH)**: 多任务预测已成为时间序列预测（TSF）的标准方法。然而，我们表明它受到表现瓶颈的影响，即不同时间步骤的预测共享相同的表示，即使使用最优表示也无法避免错误。为解决这一问题，我们提出一个两阶段框架：首先，预先训练一个基础模型进行单步预测；然后，使用特定时间步骤的LoRA进行适应。该设计理念使得基础模型能够处理任意数量的预测步骤，同时避免了表现瓶颈。我们进一步引入了混合LoRA（MoLA）模型，该模型利用自适应加权的LoRA专家在不同步骤之间实现部分参数共享。该方法通过利用预测步骤之间的相互依赖性，提高了效率和预测性能。实验证明，MoLA显著提升了模型表现并优于当前最先进的时间序列预测方法。代码可在该链接处获取。 

---
# Scalable Valuation of Human Feedback through Provably Robust Model Alignment 

**Title (ZH)**: 通过可证明鲁棒的模型对齐规模化评估人类反馈 

**Authors**: Masahiro Fujisawa, Masaki Adachi, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.17859)  

**Abstract**: Despite the importance of aligning language models with human preferences, crowd-sourced human feedback is often noisy -- for example, preferring less desirable responses -- posing a fundamental challenge to alignment. A truly robust alignment objective should yield identical model parameters even under severe label noise, a property known as redescending. We prove that no existing alignment methods satisfy this property. To address this, we propose Hölder-DPO, the first principled alignment loss with a provable redescending property, enabling estimation of the clean data distribution from noisy feedback. The aligned model estimates the likelihood of clean data, providing a theoretically grounded metric for dataset valuation that identifies the location and fraction of mislabels. This metric is gradient-free, enabling scalable and automated human feedback valuation without costly manual verification or clean validation dataset. Hölder-DPO achieves state-of-the-art robust alignment performance while accurately detecting mislabels in controlled datasets. Finally, we apply Hölder-DPO to widely used alignment datasets, revealing substantial noise levels and demonstrating that removing these mislabels significantly improves alignment performance across methods. 

**Abstract (ZH)**: 尽管将语言模型与人类偏好对齐至关重要，但众包的人类反馈往往是嘈杂的——例如，偏好不太理想的回答——这为对齐带来了根本性的挑战。一个真正 robust 的对齐目标应该在严重标签噪声下仍能产生相同的模型参数，这一特性被称为 redescending。我们证明了现有所有对齐方法都不具备这一特性。为了解决这个问题，我们提出了 Hölder-DPO，这是首个具有可证明 redescending 属性的原理性对齐损失，能够从嘈杂反馈中估计清洁数据分布。对齐后的模型估算清洁数据的概率，提供了一个基于理论的用于数据集估值的度量标准，足以识别错误标签的位置和比例。该度量标准无需梯度信息，使大规模和自动的人类反馈估值成为可能，无需昂贵的手动验证或清洁验证数据集。Hölder-DPO 在稳健对齐性能上达到了最佳效果，同时准确检测受控数据集中的错误标签。最后，我们将 Hölder-DPO 应用于广泛使用的对齐数据集，揭示了显著的噪声水平，并证明移除这些错误标签显著改善了各种方法的对齐性能。 

---
# Stochastic Weight Sharing for Bayesian Neural Networks 

**Title (ZH)**: 贝叶斯神经网络中的随机权重共享 

**Authors**: Moule Lin, Shuhao Guan, Weipeng Jing, Goetz Botterweck, Andrea Patane  

**Link**: [PDF](https://arxiv.org/pdf/2505.17856)  

**Abstract**: While offering a principled framework for uncertainty quantification in deep learning, the employment of Bayesian Neural Networks (BNNs) is still constrained by their increased computational requirements and the convergence difficulties when training very deep, state-of-the-art architectures. In this work, we reinterpret weight-sharing quantization techniques from a stochastic perspective in the context of training and inference with Bayesian Neural Networks (BNNs). Specifically, we leverage 2D adaptive Gaussian distributions, Wasserstein distance estimations, and alpha blending to encode the stochastic behaviour of a BNN in a lower dimensional, soft Gaussian representation. Through extensive empirical investigation, we demonstrate that our approach significantly reduces the computational overhead inherent in Bayesian learning by several orders of magnitude, enabling the efficient Bayesian training of large-scale models, such as ResNet-101 and Vision Transformer (VIT). On various computer vision benchmarks including CIFAR10, CIFAR100, and ImageNet1k. Our approach compresses model parameters by approximately 50x and reduces model size by 75, while achieving accuracy and uncertainty estimations comparable to the state-of-the-art. 

**Abstract (ZH)**: 基于贝叶斯神经网络的权值共享量化技术的随机重解读：提升大规模模型的高效贝叶斯训练与推理 

---
# Scaling Recurrent Neural Networks to a Billion Parameters with Zero-Order Optimization 

**Title (ZH)**: 用零阶优化将循环神经网络扩展到十亿参数 

**Authors**: Francois Chaubard, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.17852)  

**Abstract**: During inference, Recurrent Neural Networks (RNNs) scale constant in both FLOPs and GPU memory with increasing context length, as they compress all prior tokens into a fixed-size memory. In contrast, transformers scale linearly in FLOPs and, at best, linearly in memory during generation, since they must attend to all previous tokens explicitly. Despite this inference-time advantage, training large RNNs on long contexts remains impractical because standard optimization methods depend on Backpropagation Through Time (BPTT). BPTT requires retention of all intermediate activations during the forward pass, causing memory usage to scale linearly with both context length and model size. In this paper, we show that Zero-Order Optimization (ZOO) methods such as Random-vector Gradient Estimation (RGE) can successfully replace BPTT to train RNNs with convergence rates that match, or exceed BPTT by up to 19 fold, while using orders of magnitude less memory and cost, as the model remains in inference mode throughout training. We further demonstrate that Central-Difference RGE (CD-RGE) corresponds to optimizing a smoothed surrogate loss, inherently regularizing training and improving generalization. Our method matches or outperforms BPTT across three settings: (1) overfitting, (2) transduction, and (3) language modeling. Across all tasks, with sufficient perturbations, our models generalize as well as or better than those trained with BPTT, often in fewer steps. Despite the need for more forward passes per step, we can surpass BPTT wall-clock time per step using recent advancements such as FlashRNN and distributed inference. 

**Abstract (ZH)**: 基于零阶优化的递归神经网络训练方法：在保留推理优势的同时显著降低训练内存和成本 

---
# TransDF: Time-Series Forecasting Needs Transformed Label Alignment 

**Title (ZH)**: TransDF: 时间序列预测需要转换标签对齐 

**Authors**: Hao Wang, Licheng Pan, Zhichao Chen, Xu Chen, Qingyang Dai, Lei Wang, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17847)  

**Abstract**: Training time-series forecasting models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Transform-enhanced Direct Forecast (TransDF), which transforms the label sequence into decorrelated components with discriminated significance. Models are trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that TransDF achieves state-of-the-art performance and is compatible with various forecasting models. Code is available at this https URL. 

**Abstract (ZH)**: 时间序列预测模型的训练面临着设计有效学习目标的独特挑战。现有的方法主要使用时间均方误差，但面临两个关键挑战：（1）标签自相关性，导致标签序列似然性的偏差；（2）任务数量过多，会随着预测范围的增加而复杂化优化。为解决这些挑战，我们提出了Transform增强直接预测（TransDF），该方法将标签序列转换为去相关的具有鉴别显著性的成分。模型被训练以对齐这些最显著的成分，从而有效减轻标签自相关性并减少任务数量。广泛的实验表明，TransDF达到了最先进的性能，并且兼容各种预测模型。代码可在以下链接获取：this https URL。 

---
# TEDI: Trustworthy and Ethical Dataset Indicators to Analyze and Compare Dataset Documentation 

**Title (ZH)**: TEDI: 可信赖和伦理导向的数据集指标以分析和比较数据集文档 

**Authors**: Wiebke Hutiri, Mircea Cimpoi, Morgan Scheuerman, Victoria Matthews, Alice Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17841)  

**Abstract**: Dataset transparency is a key enabler of responsible AI, but insights into multimodal dataset attributes that impact trustworthy and ethical aspects of AI applications remain scarce and are difficult to compare across datasets. To address this challenge, we introduce Trustworthy and Ethical Dataset Indicators (TEDI) that facilitate the systematic, empirical analysis of dataset documentation. TEDI encompasses 143 fine-grained indicators that characterize trustworthy and ethical attributes of multimodal datasets and their collection processes. The indicators are framed to extract verifiable information from dataset documentation. Using TEDI, we manually annotated and analyzed over 100 multimodal datasets that include human voices. We further annotated data sourcing, size, and modality details to gain insights into the factors that shape trustworthy and ethical dimensions across datasets. We find that only a select few datasets have documented attributes and practices pertaining to consent, privacy, and harmful content indicators. The extent to which these and other ethical indicators are addressed varies based on the data collection method, with documentation of datasets collected via crowdsourced and direct collection approaches being more likely to mention them. Scraping dominates scale at the cost of ethical indicators, but is not the only viable collection method. Our approach and empirical insights contribute to increasing dataset transparency along trustworthy and ethical dimensions and pave the way for automating the tedious task of extracting information from dataset documentation in future. 

**Abstract (ZH)**: 可信和负责任的数据集指标 (TEDI): 促进多模态数据集文档的系统化、实证分析 

---
# Hybrid Mamba-Transformer Decoder for Error-Correcting Codes 

**Title (ZH)**: 混合Mamba-Transformer解码器用于纠错码 

**Authors**: Shy-el Cohen, Yoni Choukroun, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17834)  

**Abstract**: We introduce a novel deep learning method for decoding error correction codes based on the Mamba architecture, enhanced with Transformer layers. Our approach proposes a hybrid decoder that leverages Mamba's efficient sequential modeling while maintaining the global context capabilities of Transformers. To further improve performance, we design a novel layer-wise masking strategy applied to each Mamba layer, allowing selective attention to relevant code features at different depths. Additionally, we introduce a progressive layer-wise loss, supervising the network at intermediate stages and promoting robust feature extraction throughout the decoding process. Comprehensive experiments across a range of linear codes demonstrate that our method significantly outperforms Transformer-only decoders and standard Mamba models. 

**Abstract (ZH)**: 基于Mamba架构结合Transformer层的新型深度学习错误纠正码解码方法 

---
# Imagine Beyond! Distributionally Robust Auto-Encoding for State Space Coverage in Online Reinforcement Learning 

**Title (ZH)**: 超越想象！基于分布鲁棒自编码的空间状态覆盖在线强化学习方法 

**Authors**: Nicolas Castanet, Olivier Sigaud, Sylvain Lamprier  

**Link**: [PDF](https://arxiv.org/pdf/2505.17830)  

**Abstract**: Goal-Conditioned Reinforcement Learning (GCRL) enables agents to autonomously acquire diverse behaviors, but faces major challenges in visual environments due to high-dimensional, semantically sparse observations. In the online setting, where agents learn representations while exploring, the latent space evolves with the agent's policy, to capture newly discovered areas of the environment. However, without incentivization to maximize state coverage in the representation, classical approaches based on auto-encoders may converge to latent spaces that over-represent a restricted set of states frequently visited by the agent. This is exacerbated in an intrinsic motivation setting, where the agent uses the distribution encoded in the latent space to sample the goals it learns to master. To address this issue, we propose to progressively enforce distributional shifts towards a uniform distribution over the full state space, to ensure a full coverage of skills that can be learned in the environment. We introduce DRAG (Distributionally Robust Auto-Encoding for GCRL), a method that combines the $\beta$-VAE framework with Distributionally Robust Optimization. DRAG leverages an adversarial neural weighter of training states of the VAE, to account for the mismatch between the current data distribution and unseen parts of the environment. This allows the agent to construct semantically meaningful latent spaces beyond its immediate experience. Our approach improves state space coverage and downstream control performance on hard exploration environments such as mazes and robotic control involving walls to bypass, without pre-training nor prior environment knowledge. 

**Abstract (ZH)**: 分布稳健自动编码器在目标条件强化学习中的应用（DRAG） 

---
# Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning 

**Title (ZH)**: 不要过度思考。选择更短的思考链以提高LLM推理能力 

**Authors**: Michael Hassid, Gabriel Synnaeve, Yossi Adi, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17813)  

**Abstract**: Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). Inspired by our results, we finetune an LLM using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的推理 heavily 依赖于扩展测试时计算资源以生成大量的“思考”链来完成复杂的推理任务。虽然这种方法展现了令人印象深刻的成果，但这也带来了显著的计算成本和推断时间。在本工作中，我们挑战了长“思考”链会带来更好推理能力这一假设。我们首先展示了在单一问题内较短的“思考”链显著更有可能产生正确答案——最高可达34.5%的准确性高于同问题中最长链采样的准确性。基于这些结果，我们提出了一种新颖的推理LLM推理方法——short-m@k。该方法并行执行k个独立生成，并在第一个m个“思考”过程完成后停止计算。最终答案通过这m个链中的多数投票来选择。short-1@k的基本方法在低计算设置中显示出了与标准多数投票相似甚至更优的性能——使用多达40%更少的“思考”令牌。虽然short-3@k略显效率较低，但在所有计算预算下始终超越多数投票，并且仍然显著更快（最高减少33%的wall time）。受我们成果的启发，我们针对较短、较长以及随机选择的“思考”链对LLM进行微调。结果观察到，使用较短的“思考”链进行训练会导致更好的性能。我们的发现表明，需要重新审视当前推理LLM测试时计算的方法，强调更长的“思考”并不一定意味着更好性能，反而可能会导致意想不到的性能下降。 

---
# Seeing It or Not? Interpretable Vision-aware Latent Steering to Mitigate Object Hallucinations 

**Title (ZH)**: 看见它还是不见？可解释的视觉感知潜在 steering 技术以减轻对象幻象 

**Authors**: Boxu Chen, Ziwei Zheng, Le Yang, Zeyu Geng, Zhengyu Zhao, Chenhao Lin, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17812)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable success but continue to struggle with object hallucination (OH), generating outputs inconsistent with visual inputs. While previous work has proposed methods to reduce OH, the visual decision-making mechanisms that lead to hallucinations remain poorly understood. In this paper, we propose VaLSe, a Vision-aware Latent Steering framework that adopts an interpretation-then-mitigation strategy to address OH in LVLMs. By tackling dual challenges of modeling complex vision-language interactions and eliminating spurious activation artifacts, VaLSe can generate visual contribution maps that trace how specific visual inputs influence individual output tokens. These maps reveal the model's vision-aware focus regions, which are then used to perform latent space steering, realigning internal representations toward semantically relevant content and reducing hallucinated outputs. Extensive experiments demonstrate that VaLSe is a powerful interpretability tool and an effective method for enhancing model robustness against OH across multiple benchmarks. Furthermore, our analysis uncovers limitations in existing OH evaluation metrics, underscoring the need for more nuanced, interpretable, and visually grounded OH benchmarks in future work. Code is available at: this https URL. 

**Abstract (ZH)**: 大视觉语言模型（LVLMs）取得了显著的成功，但在对象幻觉（OH）方面仍然面临挑战，即生成与视觉输入不一致的输出。尽管之前的工作提出了减少OH的方法，但导致幻觉的视觉决策机制仍不完全理解。在本文中，我们提出了一种名为VaLSe的视觉感知潜空间定向框架，采用先解释后缓解的策略来解决LVLMs中的OH问题。通过应对复杂的视觉语言交互建模和消除虚假激活伪影的双重挑战，VaLSe可以生成视觉贡献图，追踪特定视觉输入如何影响个体输出词汇。这些图揭示了模型的视觉感知焦点区域，然后用于执行潜空间定向，重新对齐内部表示以朝向语义相关的内容，从而减少幻觉输出。广泛的经验表明，VaLSe是一个强大的可解释性工具，并且是增强模型对OH鲁棒性的有效方法，跨越多个基准。此外，我们的分析揭示了现有OH评估指标的局限性，强调了未来工作中需要更细微、可解释且视觉支撑的OH基准的必要性。代码可在以下链接获取：this https URL。 

---
# An Attention Infused Deep Learning System with Grad-CAM Visualization for Early Screening of Glaucoma 

**Title (ZH)**: 一种集成注意力机制的深度学习系统及其Grad-CAM可视化在原发性青光眼早期筛查中的应用 

**Authors**: Ramanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17808)  

**Abstract**: This research work reveals the eye opening wisdom of the hybrid labyrinthine deep learning models synergy born out of combining a trailblazing convolutional neural network with a disruptive Vision Transformer, both intertwined together with a radical Cross Attention module. Here, two high yielding datasets for artificial intelligence models in detecting glaucoma, namely ACRIMA and Drishti, are utilized. 

**Abstract (ZH)**: 这项研究揭示了结合先驱卷积神经网络和颠覆性视觉变换器的混合迷宫深度学习模型的惊艳智慧，这些模型通过激进的交叉注意力模块相互交织。本文利用了两种专门用于人工智能模型检测青光眼的高产数据集，即ACRIMA和Drishti。 

---
# Hyperparameter Optimization via Interacting with Probabilistic Circuits 

**Title (ZH)**: 基于概率电路交互的超参数优化 

**Authors**: Jonas Seng, Fabrizio Ventola, Zhongjie Yu, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.17804)  

**Abstract**: Despite the growing interest in designing truly interactive hyperparameter optimization (HPO) methods, to date, only a few allow to include human feedback. Existing interactive Bayesian optimization (BO) methods incorporate human beliefs by weighting the acquisition function with a user-defined prior distribution. However, in light of the non-trivial inner optimization of the acquisition function prevalent in BO, such weighting schemes do not always accurately reflect given user beliefs. We introduce a novel BO approach leveraging tractable probabilistic models named probabilistic circuits (PCs) as a surrogate model. PCs encode a tractable joint distribution over the hybrid hyperparameter space and evaluation scores. They enable exact conditional inference and sampling. Based on conditional sampling, we construct a novel selection policy that enables an acquisition function-free generation of candidate points (thereby eliminating the need for an additional inner-loop optimization) and ensures that user beliefs are reflected accurately in the selection policy. We provide a theoretical analysis and an extensive empirical evaluation, demonstrating that our method achieves state-of-the-art performance in standard HPO and outperforms interactive BO baselines in interactive HPO. 

**Abstract (ZH)**: 尽管设计真正交互的超参数优化（HPO）方法的兴趣日益增长，至今为止，仅有少数方法允许纳入人类反馈。现有的交互贝叶斯优化（BO）方法通过用用户定义的先验分布加权获得函数来纳入人类信念。然而，在贝叶斯优化中普遍存在的获得函数的内部优化使得这种加权方案并不总是准确地反映出给定的用户信念。我们介绍了一种新颖的BO方法，利用有效的概率模型——概率电路（PCs）作为代理模型。PCs编码混合超参数空间和评估分数的可计算联合分布，并支持精确的条件推断和采样。基于条件采样，我们构建了一种新颖的选择策略，该策略允许在无需额外内部循环优化的情况下生成候选点，并确保用户信念准确反映在选择策略中。我们提供了理论分析并进行了广泛的实证评估，证明了我们的方法在标准HPO中达到最先进的性能，并在交互HPO中超越了交互BO基线方法。 

---
# DetailFusion: A Dual-branch Framework with Detail Enhancement for Composed Image Retrieval 

**Title (ZH)**: 细节融合：一种用于组合图像检索的细节点和语义点双支框架 

**Authors**: Yuxin Yang, Yinan Zhou, Yuxin Chen, Ziqi Zhang, Zongyang Ma, Chunfeng Yuan, Bing Li, Lin Song, Jun Gao, Peng Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17796)  

**Abstract**: Composed Image Retrieval (CIR) aims to retrieve target images from a gallery based on a reference image and modification text as a combined query. Recent approaches focus on balancing global information from two modalities and encode the query into a unified feature for retrieval. However, due to insufficient attention to fine-grained details, these coarse fusion methods often struggle with handling subtle visual alterations or intricate textual instructions. In this work, we propose DetailFusion, a novel dual-branch framework that effectively coordinates information across global and detailed granularities, thereby enabling detail-enhanced CIR. Our approach leverages atomic detail variation priors derived from an image editing dataset, supplemented by a detail-oriented optimization strategy to develop a Detail-oriented Inference Branch. Furthermore, we design an Adaptive Feature Compositor that dynamically fuses global and detailed features based on fine-grained information of each unique multimodal query. Extensive experiments and ablation analyses not only demonstrate that our method achieves state-of-the-art performance on both CIRR and FashionIQ datasets but also validate the effectiveness and cross-domain adaptability of detail enhancement for CIR. 

**Abstract (ZH)**: 基于参考图像和修改文本的综合查询的图像检索（Composed Image Retrieval）旨在根据参考图像和修改文本的联合查询从图片库中检索目标图像。近年来的方法致力于平衡来自两种模态的全局信息，并将查询编码到统一特征中以实现检索。然而，由于对细微细节关注不足，这些粗粒度融合方法在处理微妙的视觉变化或复杂的文本指令时往往表现不佳。在本文中，我们提出了一种新颖的双支路框架DetailFusion，该框架有效地协调了全局和细粒度层级的信息，从而实现细节增强的图像检索（Detail-enhanced CIR）。我们的方法利用从图像编辑数据集中获得的原子细节变化先验，并结合一种细节导向的优化策略来构建细节导向的推理支路。此外，我们设计了自适应特征合成器，该合成器根据每个独特多模态查询的细微信息动态融合全局和细粒度特征。广泛的实验和消融分析不仅证明了我们的方法在CIRR和FashionIQ数据集上达到了最先进的性能，而且还验证了细节增强在图像检索中的有效性和跨域适应性。 

---
# DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors 

**Title (ZH)**: DialogXpert：通过基于在线价值强化学习的LLM先验驱动智能和情感意识对话 

**Authors**: Tazeek Bin Abdur Rakib, Ambuj Mehrish, Lay-Ki Soon, Wern Han Lim, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.17795)  

**Abstract**: Large-language-model (LLM) agents excel at reactive dialogue but struggle with proactive, goal-driven interactions due to myopic decoding and costly planning. We introduce DialogXpert, which leverages a frozen LLM to propose a small, high-quality set of candidate actions per turn and employs a compact Q-network over fixed BERT embeddings trained via temporal-difference learning to select optimal moves within this reduced space. By tracking the user's emotions, DialogXpert tailors each decision to advance the task while nurturing a genuine, empathetic connection. Across negotiation, emotional support, and tutoring benchmarks, DialogXpert drives conversations to under $3$ turns with success rates exceeding 94\% and, with a larger LLM prior, pushes success above 97\% while markedly improving negotiation outcomes. This framework delivers real-time, strategic, and emotionally intelligent dialogue planning at scale. Code available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLM）代理在反应性对话方面表现出色，但在处理前瞻性的、以目标为导向的交互时却面临困难，这主要是由于短视的解码和高昂的规划成本。我们引入了DialogXpert，它利用一个冻结的LLM为每一轮提出一小套高质量的候选行动，并通过时差学习训练一个紧凑的Q网络，在固定的好奇BERT嵌入上选择最佳动作，从而在这一减小的空间内选择最优行动。通过跟踪用户的情绪，DialogXpert为推进任务并培养真实、同理心的连接量身定制每一个决策。在谈判、情感支持和辅导基准测试中，DialogXpert将对话推向不到3轮，成功率超过94%，使用更大的LLM先验时，成功率达到97%以上，同时显著改善了谈判结果。该框架实现了大规模的实时、战略性和情感智能对话规划。代码可在以下链接获取：this https URL。 

---
# Bruno: Backpropagation Running Undersampled for Novel device Optimization 

**Title (ZH)**: Bruno: 反向传播在新型设备优化中的欠采样运行 

**Authors**: Luca Fehlings, Bojian Zhang, Paolo Gibertini, Martin A. Nicholson, Erika Covi, Fernando M. Quintana  

**Link**: [PDF](https://arxiv.org/pdf/2505.17791)  

**Abstract**: Recent efforts to improve the efficiency of neuromorphic and machine learning systems have focused on the development of application-specific integrated circuits (ASICs), which provide hardware specialized for the deployment of neural networks, leading to potential gains in efficiency and performance. These systems typically feature an architecture that goes beyond the von Neumann architecture employed in general-purpose hardware such as GPUs. Neural networks developed for this specialised hardware then need to take into account the specifics of the hardware platform, which requires novel training algorithms and accurate models of the hardware, since they cannot be abstracted as a general-purpose computing platform. In this work, we present a bottom-up approach to train neural networks for hardware based on spiking neurons and synapses built on ferroelectric capacitor (FeCap) and Resistive switching non-volatile devices (RRAM) respectively. In contrast to the more common approach of designing hardware to fit existing abstract neuron or synapse models, this approach starts with compact models of the physical device to model the computational primitive of the neurons. Based on these models, a training algorithm is developed that can reliably backpropagate through these physical models, even when applying common hardware limitations, such as stochasticity, variability, and low bit precision. The training algorithm is then tested on a spatio-temporal dataset with a network composed of quantized synapses based on RRAM and ferroelectric leaky integrate-and-fire (FeLIF) neurons. The performance of the network is compared with different networks composed of LIF neurons. The results of the experiments show the potential advantage of using BRUNO to train networks with FeLIF neurons, by achieving a reduction in both time and memory for detecting spatio-temporal patterns with quantized synapses. 

**Abstract (ZH)**: 基于铁electric电容器(FeCap)和电阻切换非易失性设备(RRAM)构建的突触和.spike神经元的硬件上神经网络训练方法 

---
# But what is your honest answer? Aiding LLM-judges with honest alternatives using steering vectors 

**Title (ZH)**: 但你的诚实行为什么是？通过导向向量提供诚实行使的替代方案辅助LLM-法官 

**Authors**: Leon Eshuijs, Archie Chaudhury, Alan McBeth, Ethan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17760)  

**Abstract**: Recent safety evaluations of Large Language Models (LLMs) show that many models exhibit dishonest behavior, such as sycophancy. However, most honesty benchmarks focus exclusively on factual knowledge or explicitly harmful behavior and rely on external judges, which are often unable to detect less obvious forms of dishonesty. In this work, we introduce a new framework, Judge Using Safety-Steered Alternatives (JUSSA), which utilizes steering vectors trained on a single sample to elicit more honest responses from models, helping LLM-judges in the detection of dishonest behavior. To test our framework, we introduce a new manipulation dataset with prompts specifically designed to elicit deceptive responses. We find that JUSSA enables LLM judges to better differentiate between dishonest and benign responses, and helps them identify subtle instances of manipulative behavior. 

**Abstract (ZH)**: Recent安全评估表明大规模语言模型（LLMs）表现出诚实度不足的行为，如拍马屁。然而，大多数诚实度基准主要集中在事实性知识或显性的危害行为上，并依赖外部评委，这些评委往往无法检测到不太明显的不诚实行为。在这项工作中，我们引入了一种新的框架，Safety-Steered Alternatives辅助的评委使用方案（JUSSA），该框架利用对单个样本进行训练的引导向量，促使模型产生更诚实的回应，帮助LLM评委检测不诚实行为。为了测试该框架，我们引入了一个新的操控数据集，其中的提示语旨在引发欺骗性回应。我们发现，JUSSA使LLM评委更好地区分不诚实和 benign 的回应，并帮助他们识别微妙的操控行为。 

---
# Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning 

**Title (ZH)**: 注意GAP！基于像素的深度强化学习中规模扩展的挑战 

**Authors**: Ghada Sokar, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2505.17749)  

**Abstract**: Scaling deep reinforcement learning in pixel-based environments presents a significant challenge, often resulting in diminished performance. While recent works have proposed algorithmic and architectural approaches to address this, the underlying cause of the performance drop remains unclear. In this paper, we identify the connection between the output of the encoder (a stack of convolutional layers) and the ensuing dense layers as the main underlying factor limiting scaling capabilities; we denote this connection as the bottleneck, and we demonstrate that previous approaches implicitly target this bottleneck. As a result of our analyses, we present global average pooling as a simple yet effective way of targeting the bottleneck, thereby avoiding the complexity of earlier approaches. 

**Abstract (ZH)**: 基于像素的环境中深化强化学习的扩展呈现显著挑战，往往导致性能下降。尽管最近的工作提出了算法和架构上的解决方案，但性能下降的根本原因仍然不明确。在本文中，我们发现编码器（一系列卷积层）输出与后续全连接层之间的连接是限制扩展能力的主要因素；我们将这一连接称为瓶颈，并证明了先前的方法隐含地针对了该瓶颈。基于我们的分析，我们提出了全局平均池化作为一种简单而有效的方法来针对瓶颈，从而避免了之前方法的复杂性。 

---
# MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization 

**Title (ZH)**: MetaBox-v2：元黑盒优化的统一基准平台 

**Authors**: Zeyuan Ma, Yue-Jiao Gong, Hongshu Guo, Wenjie Qiu, Sijie Ma, Hongqiao Lian, Jiajun Zhan, Kaixu Chen, Chen Wang, Zhiyang Huang, Zechuan Huang, Guojun Peng, Ran Cheng, Yining Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.17745)  

**Abstract**: Meta-Black-Box Optimization (MetaBBO) streamlines the automation of optimization algorithm design through meta-learning. It typically employs a bi-level structure: the meta-level policy undergoes meta-training to reduce the manual effort required in developing algorithms for low-level optimization tasks. The original MetaBox (2023) provided the first open-source framework for reinforcement learning-based single-objective MetaBBO. However, its relatively narrow scope no longer keep pace with the swift advancement in this field. In this paper, we introduce MetaBox-v2 (this https URL) as a milestone upgrade with four novel features: 1) a unified architecture supporting RL, evolutionary, and gradient-based approaches, by which we reproduce 23 up-to-date baselines; 2) efficient parallelization schemes, which reduce the training/testing time by 10-40x; 3) a comprehensive benchmark suite of 18 synthetic/realistic tasks (1900+ instances) spanning single-objective, multi-objective, multi-model, and multi-task optimization scenarios; 4) plentiful and extensible interfaces for custom analysis/visualization and integrating to external optimization tools/benchmarks. To show the utility of MetaBox-v2, we carry out a systematic case study that evaluates the built-in baselines in terms of the optimization performance, generalization ability and learning efficiency. Valuable insights are concluded from thorough and detailed analysis for practitioners and those new to the field. 

**Abstract (ZH)**: Meta-黑箱优化（MetaBBO）通过元学习简化了优化算法设计的自动化进程。它通常采用双层结构：元层面的策略在元训练过程中减少开发低层优化任务算法所需的手动努力。原始的MetaBox（2023）提供了首个基于强化学习的单目标元黑箱优化开源框架。然而，其相对狭窄的研究范围已跟不上该领域迅猛的发展步伐。本文介绍MetaBox-v2（https://）作为一项重要升级，具备四点新功能：1）支持基于强化学习、进化算法和梯度方法的一体化架构，重现了23个最新的基线模型；2）高效的并行化方案，将训练/测试时间缩短10-40倍；3）涵盖单目标、多目标、多模型和多任务优化场景的综合基准套件，包含18个合成/现实任务（超过1900个实例）；4）丰富的可扩展接口，方便自定义分析/可视化以及整合外部优化工具/基准。为了展示MetaBox-v2的优势，我们进行了系统性的案例研究，评估内置基线模型在优化性能、泛化能力和学习效率方面的表现。通过详尽细致的分析，本文为从业者及该领域的初学者提供了有价值的见解。 

---
# RQR3D: Reparametrizing the regression targets for BEV-based 3D object detection 

**Title (ZH)**: RQR3D: 重参数化回归目标以实现BEV基于的3D物体检测 

**Authors**: Ozsel Kilinc, Cem Tarhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17732)  

**Abstract**: Accurate, fast, and reliable 3D perception is essential for autonomous driving. Recently, bird's-eye view (BEV)-based perception approaches have emerged as superior alternatives to perspective-based solutions, offering enhanced spatial understanding and more natural outputs for planning. Existing BEV-based 3D object detection methods, typically adhering to angle-based representation, directly estimate the size and orientation of rotated bounding boxes. We observe that BEV-based 3D object detection is analogous to aerial oriented object detection, where angle-based methods are recognized for being affected by discontinuities in their loss functions. Drawing inspiration from this domain, we propose Restricted Quadrilateral Representation to define 3D regression targets. RQR3D regresses the smallest horizontal bounding box encapsulating the oriented box, along with the offsets between the corners of these two boxes, thereby transforming the oriented object detection problem into a keypoint regression task. RQR3D is compatible with any 3D object detection approach. We employ RQR3D within an anchor-free single-stage object detection method and introduce an objectness head to address class imbalance problem. Furthermore, we introduce a simplified radar fusion backbone that eliminates the need for voxel grouping and processes the BEV-mapped point cloud with standard 2D convolutions, rather than sparse convolutions. Extensive evaluations on the nuScenes dataset demonstrate that RQR3D achieves state-of-the-art performance in camera-radar 3D object detection, outperforming the previous best method by +4% in NDS and +2.4% in mAP, and significantly reducing the translation and orientation errors, which are crucial for safe autonomous driving. These consistent gains highlight the robustness, precision, and real-world readiness of our approach. 

**Abstract (ZH)**: 准确、快速且可靠的三维感知对于自动驾驶至关重要。基于鸟瞰视角（BEV）的感知方法近年来已成为优于基于视角解决方案的优选方案，提供了增强的空间理解能力和更自然的规划输出。现有的基于BEV的三维物体检测方法通常采用角度基表示法，直接估计旋转边界框的大小和方向。我们观察到基于BEV的三维物体检测类似于航空定向物体检测，其中角度基方法因其损失函数中的不连续性而受到影响。从这一领域中汲取灵感，我们提出了受限四边形表示法来定义三维回归目标。RQR3D回歸包容定向框的最小水平边界框及其两个盒子角点之间的偏移，从而将定向物体检测问题转换为关键点回归任务。RQR3D与任何三维物体检测方法兼容。我们将在无锚点的一阶段物体检测方法中使用RQR3D，并引入物体性头部以解决类别不平衡问题。此外，我们引入了一个简化版的雷达融合骨干网络，它消除了体素分组的需要，并使用标准的二维卷积而不是稀疏卷积处理BEV映射的点云。在nuScenes数据集上的广泛评估表明，RQR3D在相机-雷达三维物体检测中的性能达到最新水平，在NDS和mAP上分别超过之前最佳方法4%和2.4%，显著减少了对于安全自动驾驶至关重要的平移和方向误差。这些一致的改进突显了我们方法的鲁棒性、精确性和实际可用性。 

---
# Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM 

**Title (ZH)**: Slot-MLLM: 基于对象的视觉词元化多模态大模型 

**Authors**: Donghwan Chi, Hyomin Kim, Yoonjin Oh, Yongjin Kim, Donghoon Lee, Daejin Jo, Jongmin Kim, Junyeob Baek, Sungjin Ahn, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17726)  

**Abstract**: Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images. 

**Abstract (ZH)**: 近期，多模态大语言模型（MLLMs）已成为实现人工通用智能的关键方法。特别是，视觉语言MLLMs已被开发出来，不仅能生成文本，还能从多模态输入中生成视觉输出。这一进展需要高效的图像令牌，使MLLMs在输入和输出中都能有效处理。然而，现有针对MLLMs的图像 tokenization 方法通常只能捕获全局抽象概念或均匀分割的图像块，限制了MLLMs有效理解或生成详细视觉内容的能力，尤其是在对象层面。为了解决这一局限性，我们提出了一种基于 Slot Attention 的对象中心视觉tokenizer，专门用于MLLMs。特别是，基于Q-Former编码器、扩散解码器和残差矢量量化，我们提出的离散槽令牌能够编码局部视觉细节，同时保持高层次语义，并与文本数据无缝整合，以适应统一的下一个token预测框架。所提出的Slot-MLLM在涉及局部详细理解和生成的各种视觉语言任务上，相对于之前的视觉tokenizer基线模型，表现出了显著的性能提升。值得注意的是，这是首次在MLLMs和真实世界的自然图像中展示对象中心槽注意力的可行性。 

---
# A Distributionally-Robust Framework for Nuisance in Causal Effect Estimation 

**Title (ZH)**: 分布鲁棒的中介变量在因果效应估计中的框架 

**Authors**: Akira Tanimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.17717)  

**Abstract**: Causal inference requires evaluating models on balanced distributions between treatment and control groups, while training data often exhibits imbalance due to historical decision-making policies. Most conventional statistical methods address this distribution shift through inverse probability weighting (IPW), which requires estimating propensity scores as an intermediate step. These methods face two key challenges: inaccurate propensity estimation and instability from extreme weights. We decompose the generalization error to isolate these issues--propensity ambiguity and statistical instability--and address them through an adversarial loss function. Our approach combines distributionally robust optimization for handling propensity uncertainty with weight regularization based on weighted Rademacher complexity. Experiments on synthetic and real-world datasets demonstrate consistent improvements over existing methods. 

**Abstract (ZH)**: 因果推断要求在治疗组和控制组之间评估模型的平衡分布，而训练数据由于历史上决策制定政策的原因通常表现出不平衡。大多数传统统计方法通过逆概率加权（IPW）来解决这种分布偏移问题，这需要在中间步骤估计倾向得分。这些方法面临两个关键挑战：倾向得分估计不准确和极端权重导致的不稳定性。我们通过分解泛化误差来分离这些问题——倾向得分不确定性与统计不稳定性——并通过对抗损失函数来解决这些问题。我们的方法结合了处理倾向得分不确定性的大范围优化技术，并基于加权仁科复杂性进行权重正则化。实验结果在合成和真实世界数据集上展示了相对于现有方法的一致改进。 

---
# PPO-BR: Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization 

**Title (ZH)**: PPO-BR: 双信号熵-奖励适应的可信区域策略优化 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17714)  

**Abstract**: Despite Proximal Policy Optimization (PPO) dominating policy gradient methods -- from robotic control to game AI -- its static trust region forces a brittle trade-off: aggressive clipping stifles early exploration, while late-stage updates destabilize convergence. PPO-BR establishes a new paradigm in adaptive RL by fusing exploration and convergence signals into a single bounded trust region -- a theoretically grounded innovation that outperforms five SOTA baselines with less than 2% overhead. This work bridges a critical gap in phase-aware learning, enabling real-world deployment in safety-critical systems like robotic surgery within a single adaptive mechanism. PPO-BR achieves 29.1% faster convergence by combining: (1) entropy-driven expansion (epsilon up) for exploration in high-uncertainty states, and (2) reward-guided contraction (epsilon down) for convergence stability. On six diverse benchmarks (MuJoCo, Atari, sparse-reward), PPO-BR achieves 29.1% faster convergence (p < 0.001), 2.3x lower reward variance than PPO, and less than 1.8% runtime overhead with only five lines of code change. PPO-BR's simplicity and theoretical guarantees make it ready-to-deploy in safety-critical domains -- from surgical robotics to autonomous drones. In contrast to recent methods such as Group Relative Policy Optimization (GRPO), PPO-BR offers a unified entropy-reward mechanism applicable to both language models and general reinforcement learning environments. 

**Abstract (ZH)**: 尽管最近策略优化（PPO）在机器人控制和游戏AI等领域主导了策略梯度方法——其静态信任区域强制了一种脆弱的权衡：激进的剪裁抑制了早期探索，而后期更新则破坏了收敛。PPO-BR通过将探索信号和收敛信号融合到一个单一的受限制的信任区域中，建立了自适应强化学习的新范式——这一理论上的创新比五个SOTA基线方法更具优势，同时仅增加不到2%的开销。这项工作填补了阶段感知学习中的关键空白，使安全关键系统（如机器人手术）能够在单一自适应机制内实现实际部署。PPO-BR通过结合以下两种方法实现了29.1%更快的收敛：（1）不确定性状态下基于熵的扩展（ε上升）进行探索，（2）基于奖励的收缩（ε下降）以提高收敛稳定性。在六个不同的基准测试（MuJoCo、Atari、稀疏奖励）中，PPO-BR的收敛速度快了29.1%（p < 0.001），奖励方差比PPO低2.3倍，改动仅五行代码，运行时开销不到1.8%。PPO-BR的简单性和理论保证使其在安全关键领域（从外科手术机器人到自主无人机）中随时可以部署。与最近的方法如组相对策略优化（GRPO）不同，PPO-BR提供了一种统一的熵-奖励机制，适用于语言模型和一般强化学习环境。 

---
# Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek 

**Title (ZH)**: Seek-CAD：一种基于DeepSeek的局部推理自精炼生成 modeling方法用于3D参数化CAD 

**Authors**: Xueyang Li, Jiahao Li, Yu Song, Yunzhong Lou, Xiangdong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17702)  

**Abstract**: The advent of Computer-Aided Design (CAD) generative modeling will significantly transform the design of industrial products. The recent research endeavor has extended into the realm of Large Language Models (LLMs). In contrast to fine-tuning methods, training-free approaches typically utilize the advanced closed-source LLMs, thereby offering enhanced flexibility and efficiency in the development of AI agents for generating CAD parametric models. However, the substantial cost and limitations of local deployment of the top-tier closed-source LLMs pose challenges in practical applications. The Seek-CAD is the pioneer exploration of locally deployed open-source inference LLM DeepSeek-R1 for CAD parametric model generation with a training-free methodology. This study is the first investigation to incorporate both visual and Chain-of-Thought (CoT) feedback within the self-refinement mechanism for generating CAD models. Specifically, the initial generated parametric CAD model is rendered into a sequence of step-wise perspective images, which are subsequently processed by a Vision Language Model (VLM) alongside the corresponding CoTs derived from DeepSeek-R1 to assess the CAD model generation. Then, the feedback is utilized by DeepSeek-R1 to refine the initial generated model for the next round of generation. Moreover, we present an innovative 3D CAD model dataset structured around the SSR (Sketch, Sketch-based feature, and Refinements) triple design paradigm. This dataset encompasses a wide range of CAD commands, thereby aligning effectively with industrial application requirements and proving suitable for the generation of LLMs. Extensive experiments validate the effectiveness of Seek-CAD under various metrics. 

**Abstract (ZH)**: 计算机辅助设计（CAD）生成建模的出现将显著变革工业产品设计。近期研究已拓展至大型语言模型（LLMs）领域。与微调方法不同，无需训练的方法通常利用先进的闭源LLMs，从而在开发生成CAD参数化模型的AI代理方面提供更高的灵活性和效率。然而，顶级闭源LLMs的高昂成本及其本地部署的限制在实际应用中构成挑战。Seek-CAD是局部部署的开源推理LLM DeepSeek-R1用于无需训练方法生成CAD参数化模型的先驱探索。该研究首次在自完善机制中结合视觉反馈和链式思考（CoT）反馈，以生成CAD模型。具体而言，初始生成的参数化CAD模型被渲染为一系列逐步视角图像，随后与DeepSeek-R1生成的相关CoTs一起由视觉语言模型（VLM）处理，以评估CAD模型生成情况。然后，反馈用于指导DeepSeek-R1精炼初始生成的模型，以进入下一轮生成。此外，我们介绍了基于SSR（草图、草图基础特征和精炼）三重设计范式的创新3D CAD模型数据集，该数据集涵盖了广泛的CAD命令，有效满足工业应用需求，并适用于生成LLMs。广泛实验验证了Seek-CAD在多种评估指标下的有效性。 

---
# COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection 

**Title (ZH)**: COUNTDOWN: 上下文稀疏激活过滤不必要的下行投影权重 

**Authors**: Jaewon Cheon, Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17701)  

**Abstract**: The growing size of large language models has created significant computational inefficiencies. To address this challenge, sparse activation methods selectively deactivates non-essential parameters during inference, reducing computational costs in FFNN layers. While existing methods focus on non-linear gating mechanisms, we hypothesize that the sparsity of the FFNN layer lies globally in the form of a linear combination over its internal down projection matrix. Based on this insight, we propose two methods: M-COUNTDOWN, leveraging indirect coefficients, and D-COUNTDOWN, utilizing direct coefficients of the linear combination. Experimental results demonstrate that D-COUNTDOWN can omit 90% of computations with performance loss as low as 5.5% ideally, while M-COUNTDOWN provides a predictor-free solution with up to 29.4% better performance preservation compared to existing methods. Our specialized kernel implementations effectively realize these theoretical gains into substantial real-world acceleration. 

**Abstract (ZH)**: 大型语言模型规模的不断扩大造成了显著的计算 inefficiency。为应对这一挑战，稀疏激活方法在推理过程中选择性地去激活非必要参数，从而减少全连接层（FFNN）的计算成本。尽管现有方法侧重于非线性门控机制，我们假设全连接层的稀疏性在全球范围内表现为线性组合形式，其内部下投影矩阵的直接系数。基于这一洞察，我们提出两种方法：M-COUNTDOWN，利用间接系数；D-COUNTDOWN，利用线性组合的直接系数。实验结果表明，D-COUNTDOWN可以在性能损失仅为5.5%的理想情况下省去90%的计算量，而M-COUNTDOWN提供了一个无需预测器的解决方案，其性能保留比现有方法高出至多29.4%。我们专门开发的内核实现有效将这些理论优势转化为实质性的实际加速。 

---
# SynRES: Towards Referring Expression Segmentation in the Wild via Synthetic Data 

**Title (ZH)**: SynRES：通过合成数据朝无约束的引用表达分割迈进 

**Authors**: Dong-Hee Kim, Hyunjee Song, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17695)  

**Abstract**: Despite the advances in Referring Expression Segmentation (RES) benchmarks, their evaluation protocols remain constrained, primarily focusing on either single targets with short queries (containing minimal attributes) or multiple targets from distinctly different queries on a single domain. This limitation significantly hinders the assessment of more complex reasoning capabilities in RES models. We introduce WildRES, a novel benchmark that incorporates long queries with diverse attributes and non-distinctive queries for multiple targets. This benchmark spans diverse application domains, including autonomous driving environments and robotic manipulation scenarios, thus enabling more rigorous evaluation of complex reasoning capabilities in real-world settings. Our analysis reveals that current RES models demonstrate substantial performance deterioration when evaluated on WildRES. To address this challenge, we introduce SynRES, an automated pipeline generating densely paired compositional synthetic training data through three innovations: (1) a dense caption-driven synthesis for attribute-rich image-mask-expression triplets, (2) reliable semantic alignment mechanisms rectifying caption-pseudo mask inconsistencies via Image-Text Aligned Grouping, and (3) domain-aware augmentations incorporating mosaic composition and superclass replacement to emphasize generalization ability and distinguishing attributes over object categories. Experimental results demonstrate that models trained with SynRES achieve state-of-the-art performance, improving gIoU by 2.0% on WildRES-ID and 3.8% on WildRES-DS. Code and datasets are available at this https URL. 

**Abstract (ZH)**: WildRES：一种包含长查询和非独特查询的新型基准 

---
# ViP$^2$-CLIP: Visual-Perception Prompting with Unified Alignment for Zero-Shot Anomaly Detection 

**Title (ZH)**: ViP$^2$-CLIP: 统一对齐的视觉感知提示在零样本异常检测中的应用 

**Authors**: Ziteng Yang, Jingzehua Xu, Yanshu Li, Zepeng Li, Yeqiang Wang, Xinghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17692)  

**Abstract**: Zero-shot anomaly detection (ZSAD) aims to detect anomalies without any target domain training samples, relying solely on external auxiliary data. Existing CLIP-based methods attempt to activate the model's ZSAD potential via handcrafted or static learnable prompts. The former incur high engineering costs and limited semantic coverage, whereas the latter apply identical descriptions across diverse anomaly types, thus fail to adapt to complex variations. Furthermore, since CLIP is originally pretrained on large-scale classification tasks, its anomaly segmentation quality is highly sensitive to the exact wording of class names, severely constraining prompting strategies that depend on class labels. To address these challenges, we introduce ViP$^{2}$-CLIP. The key insight of ViP$^{2}$-CLIP is a Visual-Perception Prompting (ViP-Prompt) mechanism, which fuses global and multi-scale local visual context to adaptively generate fine-grained textual prompts, eliminating manual templates and class-name priors. This design enables our model to focus on precise abnormal regions, making it particularly valuable when category labels are ambiguous or privacy-constrained. Extensive experiments on 15 industrial and medical benchmarks demonstrate that ViP$^{2}$-CLIP achieves state-of-the-art performance and robust cross-domain generalization. 

**Abstract (ZH)**: 基于ViP$^{2}$-CLIP的零样本异常检测 

---
# Dual Attention Residual U-Net for Accurate Brain Ultrasound Segmentation in IVH Detection 

**Title (ZH)**: 用于IVH检测的精确脑超声分割的双注意力残差U-网络 

**Authors**: Dan Yuan, Yi Feng, Ziyun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17683)  

**Abstract**: Intraventricular hemorrhage (IVH) is a severe neurological complication among premature infants, necessitating early and accurate detection from brain ultrasound (US) images to improve clinical outcomes. While recent deep learning methods offer promise for computer-aided diagnosis, challenges remain in capturing both local spatial details and global contextual dependencies critical for segmenting brain anatomies. In this work, we propose an enhanced Residual U-Net architecture incorporating two complementary attention mechanisms: the Convolutional Block Attention Module (CBAM) and a Sparse Attention Layer (SAL). The CBAM improves the model's ability to refine spatial and channel-wise features, while the SAL introduces a dual-branch design, sparse attention filters out low-confidence query-key pairs to suppress noise, and dense attention ensures comprehensive information propagation. Extensive experiments on the Brain US dataset demonstrate that our method achieves state-of-the-art segmentation performance, with a Dice score of 89.04% and IoU of 81.84% for ventricle region segmentation. These results highlight the effectiveness of integrating spatial refinement and attention sparsity for robust brain anatomy detection. Code is available at: this https URL. 

**Abstract (ZH)**: 脑室内出血（IVH）是早产儿严重的神经系统并发症，需要从脑超声图像中进行早期和准确的检测以改善临床结果。虽然近期的深度学习方法为计算机辅助诊断带来了希望，但在分割脑解剖结构时仍面临着捕捉局部空间细节和全局上下文依赖性的挑战。在本文中，我们提出了一种增强的残差U-网架构，结合了两种互补的注意力机制：卷积块注意力模块（CBAM）和稀疏注意力层（SAL）。CBAM提升了模型在细化空间和通道级特征方面的能力，而SAL引入了双分支设计，稀疏注意力过滤掉低置信度的查询-键对以抑制噪声，密集注意力确保了信息的全面传递。在Brain US数据集上的广泛实验表明，我们的方法实现了最先进的分割性能，脑室区域分割的Dice分数为89.04%，IoU为81.84%。这些结果突显了将空间细化和注意力稀疏性集成用于稳健脑解剖结构检测的有效性。代码可在以下链接获取：this https URL。 

---
# Tuning Language Models for Robust Prediction of Diverse User Behaviors 

**Title (ZH)**: 调整语言模型以稳健预测多样化用户行为 

**Authors**: Fanjin Meng, Jingtao Ding, Jiahui Gong, Chen Yang, Hong Chen, Zuojian Wang, Haisheng Lu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17682)  

**Abstract**: Predicting user behavior is essential for intelligent assistant services, yet deep learning models often struggle to capture long-tailed behaviors. Large language models (LLMs), with their pretraining on vast corpora containing rich behavioral knowledge, offer promise. However, existing fine-tuning approaches tend to overfit to frequent ``anchor'' behaviors, reducing their ability to predict less common ``tail'' behaviors. In this paper, we introduce BehaviorLM, a progressive fine-tuning approach that addresses this issue. In the first stage, LLMs are fine-tuned on anchor behaviors while preserving general behavioral knowledge. In the second stage, fine-tuning uses a balanced subset of all behaviors based on sample difficulty to improve tail behavior predictions without sacrificing anchor performance. Experimental results on two real-world datasets demonstrate that BehaviorLM robustly predicts both anchor and tail behaviors and effectively leverages LLM behavioral knowledge to master tail behavior prediction with few-shot examples. 

**Abstract (ZH)**: 行为预测对于智能助理服务至关重要，然而深度学习模型往往难以捕捉长尾行为。预训练于丰富行为知识大规模语料中的大规模语言模型（LLMs）提供了潜力。然而，现有的微调方法倾向于过度拟合于常见的“锚”行为，降低了预测少见的“尾”行为的能力。本文提出了一种逐步微调方法——BehaviorLM，以解决这一问题。在第一阶段，LLMs在保留一般行为知识的前提下对常见行为进行微调。在第二阶段，微调采用基于样本难度的平衡子集，以提高尾行为预测能力而不牺牲锚行为性能。实验结果表明，BehaviorLM在两个真实世界数据集上稳健地预测了锚行为和尾行为，并且能够有效地利用LLM的行为知识通过少量示例掌握尾行为预测。 

---
# Towards General Continuous Memory for Vision-Language Models 

**Title (ZH)**: 面向视觉-语言模型的通用连续记忆 

**Authors**: Wenyi Wu, Zixuan Song, Kun Zhou, Yifei Shao, Zhiting Hu, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17670)  

**Abstract**: Language models (LMs) and their extension, vision-language models (VLMs), have achieved remarkable performance across various tasks. However, they still struggle with complex reasoning tasks that require multimodal or multilingual real-world knowledge. To support such capabilities, an external memory system that can efficiently provide relevant multimodal information is essential. Existing approaches generally concatenate image and text tokens into a long sequence as memory, which, however, may drastically increase context length and even degrade performance. In contrast, we propose using continuous memory, a compact set of dense embeddings to more effectively and efficiently represent multimodal and multilingual knowledge. Our key insight is that a VLM can serve as its own continuous memory encoder. We empirically show that this design improves performance on complex multimodal reasoning tasks. Building on this, we introduce a data-efficient and parameter-efficient method to fine-tune the VLM into a memory encoder, requiring only 1.2% of the model's parameters and a small corpus of 15.6K self-synthesized samples. Our approach CoMEM utilizes VLM's original capabilities to encode arbitrary multimodal and multilingual knowledge into just 8 continuous embeddings. Since the inference-time VLM remains frozen, our memory module is plug-and-play and can be flexibly integrated as needed. Extensive experiments across eight multimodal reasoning benchmarks demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 基于语言模型（LMs）及其扩展视觉-语言模型（VLMs）在各种任务中取得了显著性能，但它们仍然难以处理需要多模态或多语言现实世界知识的复杂推理任务。为了支持这些能力，需要一个高效提供相关多模态信息的外部记忆系统。现有方法通常将图像和文本标记连接成一个长序列作为记忆，但这可能会大幅增加上下文长度并甚至恶化性能。相比之下，我们提出使用连续记忆，即一组密集的嵌入表示多模态和多语言知识。我们的核心见解是，VLM可以作为其自身的连续记忆编码器。实验证明，这种设计在复杂多模态推理任务上能提高性能。在这一基础上，我们引入了一种数据高效和参数高效的微调方法，将VLM微调成一个记忆编码器，仅需模型参数的1.2%和15.6K自合成样本的小规模语料库。我们的方法CoMEM利用VLM的原有能力，仅将任意多模态和多语言知识编码为8个连续嵌入。由于推理时的VLM保持冻结，我们的记忆模块插即用，可以根据需要灵活集成。广泛的实验跨越八个多模态推理基准，证明了我们方法的有效性。 

---
# EMRA-proxy: Enhancing Multi-Class Region Semantic Segmentation in Remote Sensing Images with Attention Proxy 

**Title (ZH)**: EMRA-proxy：在注意力代理辅助下增强多类区域语义分割的遥感图像处理 

**Authors**: Yichun Yu, Yuqing Lan, Zhihuan Xing, Xiaoyi Yang, Tingyue Tang, Dan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17665)  

**Abstract**: High-resolution remote sensing (HRRS) image segmentation is challenging due to complex spatial layouts and diverse object appearances. While CNNs excel at capturing local features, they struggle with long-range dependencies, whereas Transformers can model global context but often neglect local details and are computationally this http URL propose a novel approach, Region-Aware Proxy Network (RAPNet), which consists of two components: Contextual Region Attention (CRA) and Global Class Refinement (GCR). Unlike traditional methods that rely on grid-based layouts, RAPNet operates at the region level for more flexible segmentation. The CRA module uses a Transformer to capture region-level contextual dependencies, generating a Semantic Region Mask (SRM). The GCR module learns a global class attention map to refine multi-class information, combining the SRM and attention map for accurate this http URL on three public datasets show that RAPNet outperforms state-of-the-art methods, achieving superior multi-class segmentation accuracy. 

**Abstract (ZH)**: 高分辨率遥感图像分割由于复杂的空间布局和多样的目标外观而具有挑战性。虽然CNN擅长捕获局部特征，但在建模长距离依赖关系方面存在局限性，而Transformer可以建模全局上下文但往往会忽略局部细节且计算成本高。为了解决这一问题，提出了一种新型方法，区域感知代理网络（RAPNet），该方法由两个组件组成：上下文区域注意力（CRA）和全局类别精炼（GCR）。与依赖网格布局的传统方法不同，RAPNet在区域级别操作以实现更具灵活性的分割。CRA模块使用Transformer捕获区域级别的上下文依赖关系，生成语义区域掩码（SRM）。GCR模块学习一个全局类别注意力图以精炼多类别信息，结合SRM和注意力图以实现精确的分割。在三个公开数据集上的实验表明，与当前最先进的方法相比，RAPNet在多类别分割 accuracy 方面表现更优。 

---
# EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications 

**Title (ZH)**: EVADE：电子商务应用中规避内容检测的多模态基准 

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17654)  

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at this https URL. 

**Abstract (ZH)**: 电子商务平台越来越多地依赖大型语言模型（LLMs）和多模态视觉-语言模型（VLMs）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的攻击：看似符合平台政策但实际上隐含违规声明的输入（文本或图像）。与传统的对抗性攻击导致明显的失效不同，规避内容利用了模糊性和上下文，使其更难被检测到。现有鲁棒性基准对此提出的现实挑战提供了很少的指导。我们提出了EVADE，这是首个专为评估基础模型在电子商务中的规避内容检测能力而精心设计的汉语多模态基准。该数据集包含2,833个标注的文本样本和13,961张图片，覆盖六个具有挑战性的产品类别，包括体型管理、身高增长和健康补充剂。两个互补的任务评估不同的能力：单违反（Single-Violation），探索在简短提示下进行细微推理的能力；全能（All-in-One），通过合并重叠的政策规则来测试长上下文推理。值得注意的是，全能环境显著缩小了部分匹配和全匹配准确性之间的性能差距，表明更清晰的规则定义可以改善人类和模型判断之间的契合度。我们测试了26种主流LLM和VLM，并观察到显著的性能差距：即使是最先进的模型也经常错误分类规避样本。通过发布EVADE和强大的基线，我们提供了首个严格的规避内容检测评估标准，揭示了当前多模态推理的基本局限性，并为基础电子商务内容审核系统提供了更安全和透明的底层建设。数据集可从此网址公开访问。 

---
# Rethinking the Sampling Criteria in Reinforcement Learning for LLM Reasoning: A Competence-Difficulty Alignment Perspective 

**Title (ZH)**: 重塑强化学习中大规模语言模型推理的采样标准：一种能力-难度对齐视角 

**Authors**: Deyang Kong, Qi Guo, Xiangyu Xi, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.17652)  

**Abstract**: Reinforcement learning exhibits potential in enhancing the reasoning abilities of large language models, yet it is hard to scale for the low sample efficiency during the rollout phase. Existing methods attempt to improve efficiency by scheduling problems based on problem difficulties. However, these approaches suffer from unstable and biased estimations of problem difficulty and fail to capture the alignment between model competence and problem difficulty in RL training, leading to suboptimal results. To tackle these limitations, this paper introduces \textbf{C}ompetence-\textbf{D}ifficulty \textbf{A}lignment \textbf{S}ampling (\textbf{CDAS}), which enables accurate and stable estimation of problem difficulties by aggregating historical performance discrepancies of problems. Then the model competence is quantified to adaptively select problems whose difficulty is in alignment with the model's current competence using a fixed-point system. Experimental results across a range of challenging mathematical benchmarks show that CDAS achieves great improvements in both accuracy and efficiency. CDAS attains the highest average accuracy against baselines and exhibits significant speed advantages compared to Dynamic Sampling, a competitive strategy in DAPO, which is \textbf{2.33} times slower than CDAS. 

**Abstract (ZH)**: Competence-Difficulty Alignment Sampling Improves Reasoning Abilities of Large Language Models 

---
# HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning 

**Title (ZH)**: HoloLLM：基于语言的地多感官基础模型为人感知与推理服务 

**Authors**: Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17645)  

**Abstract**: Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence. 

**Abstract (ZH)**: 具身代理在智能家居中的操作需要通过多种传感输入理解人类行为，并通过自然语言进行沟通。尽管视觉-语言模型（VLMs）已经使基于语言的感知取得了显著进展，但它们对视觉数据的依赖限制了其在具有遮挡、照明不良或隐私约束的真实场景中的鲁棒性。在本文中，我们介绍了HoloLLM，这是一种多模态大型语言模型（MLLM），结合了诸如激光雷达、红外、毫米波雷达和WiFi等不常见但强大的传感模态，以在异构环境中实现无缝的人类感知和推理。我们解决了两个关键挑战：（1）罕见传感器对齐模态-文本数据的稀缺性，（2）它们物理信号表示的异质性。为了解决这些问题，我们设计了一个通用模态注入投影器（UMIP），通过粗到精的交叉注意力，增强预对齐的模态嵌入，引入细粒度、文本对齐的特征，来自针对特定编码器的细粒度、文本对齐的特征，而不引入显著的对齐开销。我们还引入了一种人类-VLM协作的数据标注流水线，生成感知数据集配对的文本注释。在两个新构建的基准上的广泛实验表明，HoloLLM 显著优于现有 MLLMs，将基于语言的人类感知准确性提高多达 30%。这项工作为语言指导的实际多感知具身智能奠定了新基础。 

---
# Surfacing Semantic Orthogonality Across Model Safety Benchmarks: A Multi-Dimensional Analysis 

**Title (ZH)**: 跨模型安全性基准表面语义正交性：多维度分析 

**Authors**: Jonathan Bennion, Shaona Ghosh, Mantek Singh, Nouha Dziri  

**Link**: [PDF](https://arxiv.org/pdf/2505.17636)  

**Abstract**: Various AI safety datasets have been developed to measure LLMs against evolving interpretations of harm. Our evaluation of five recently published open-source safety benchmarks reveals distinct semantic clusters using UMAP dimensionality reduction and kmeans clustering (silhouette score: 0.470). We identify six primary harm categories with varying benchmark representation. GretelAI, for example, focuses heavily on privacy concerns, while WildGuardMix emphasizes self-harm scenarios. Significant differences in prompt length distribution suggests confounds to data collection and interpretations of harm as well as offer possible context. Our analysis quantifies benchmark orthogonality among AI benchmarks, allowing for transparency in coverage gaps despite topical similarities. Our quantitative framework for analyzing semantic orthogonality across safety benchmarks enables more targeted development of datasets that comprehensively address the evolving landscape of harms in AI use, however that is defined in the future. 

**Abstract (ZH)**: 各种AI安全数据集已被开发用于衡量LLMs针对 evolving interpretations of harm 的变化。我们对五个最近发布的开源安全基准的评估揭示了使用UMAP降维和kmeans聚类（轮廓分数：0.470）的不同语义簇。我们识别出六个主要的伤害类别，这些类别在基准中的表现形式各异。例如，GretelAI 高度关注隐私问题，而WildGuardMix则集中在自我伤害场景上。显著的提示长度分布差异表明了数据收集和伤害解释中的混杂因素，同时也提供了可能的上下文信息。我们的分析量化了AI基准之间的正交性，即使在主题相似的情况下，也允许透明地了解覆盖缺口。我们针对安全基准的语义正交性分析的量化框架能够促进更具针对性的数据集开发，以全面应对未来AI使用中演变的伤害环境。 

---
# ReqBrain: Task-Specific Instruction Tuning of LLMs for AI-Assisted Requirements Generation 

**Title (ZH)**: ReqBrain: 面向AI辅助需求生成的LLM任务特定指令调优 

**Authors**: Mohammad Kasra Habib, Daniel Graziotin, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2505.17632)  

**Abstract**: Requirements elicitation and specification remains a labor-intensive, manual process prone to inconsistencies and gaps, presenting a significant challenge in modern software engineering. Emerging studies underscore the potential of employing large language models (LLMs) for automated requirements generation to support requirements elicitation and specification; however, it remains unclear how to implement this effectively. In this work, we introduce ReqBrain, an Al-assisted tool that employs a fine-tuned LLM to generate authentic and adequate software requirements. Software engineers can engage with ReqBrain through chat-based sessions to automatically generate software requirements and categorize them by type. We curated a high-quality dataset of ISO 29148-compliant requirements and fine-tuned five 7B-parameter LLMs to determine the most effective base model for ReqBrain. The top-performing model, Zephyr-7b-beta, achieved 89.30\% Fl using the BERT score and a FRUGAL score of 91.20 in generating authentic and adequate requirements. Human evaluations further confirmed ReqBrain's effectiveness in generating requirements. Our findings suggest that generative Al, when fine-tuned, has the potential to improve requirements elicitation and specification, paving the way for future extensions into areas such as defect identification, test case generation, and agile user story creation. 

**Abstract (ZH)**: 基于大语言模型的软件需求提取与规范工具ReqBrain 

---
# BehaveGPT: A Foundation Model for Large-scale User Behavior Modeling 

**Title (ZH)**: BehaveGPT：大规模用户行为建模的基石模型 

**Authors**: Jiahui Gong, Jingtao Ding, Fanjin Meng, Chen Yang, Hong Chen, Zuojian Wang, Haisheng Lu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17631)  

**Abstract**: In recent years, foundational models have revolutionized the fields of language and vision, demonstrating remarkable abilities in understanding and generating complex data; however, similar advances in user behavior modeling have been limited, largely due to the complexity of behavioral data and the challenges involved in capturing intricate temporal and contextual relationships in user activities. To address this, we propose BehaveGPT, a foundational model designed specifically for large-scale user behavior prediction. Leveraging transformer-based architecture and a novel pretraining paradigm, BehaveGPT is trained on vast user behavior datasets, allowing it to learn complex behavior patterns and support a range of downstream tasks, including next behavior prediction, long-term generation, and cross-domain adaptation. Our approach introduces the DRO-based pretraining paradigm tailored for user behavior data, which improves model generalization and transferability by equitably modeling both head and tail behaviors. Extensive experiments on real-world datasets demonstrate that BehaveGPT outperforms state-of-the-art baselines, achieving more than a 10% improvement in macro and weighted recall, showcasing its ability to effectively capture and predict user behavior. Furthermore, we measure the scaling law in the user behavior domain for the first time on the Honor dataset, providing insights into how model performance scales with increased data and parameter sizes. 

**Abstract (ZH)**: 近年来，基础模型在语言和视觉领域实现了革命性的进步，展示了理解与生成复杂数据的非凡能力；然而，用户行为建模方面取得类似进展有限，主要是由于行为数据的复杂性以及捕捉用户活动中的精细时序和上下文关系的挑战。为了解决这个问题，我们提出BehaveGPT，一种专门设计用于大规模用户行为预测的基础模型。通过利用基于变换器的架构和新颖的预训练范式，BehaveGPT 在大量的用户行为数据集上训练，使其能够学习复杂的行为模式，并支持包括下一个行为预测、长期生成和跨域适应等一系列下游任务。我们的方法引入了一种针对用户行为数据定制的DRO基于的预训练范式，通过公平建模头部和尾部行为，提高了模型的泛化能力和迁移性。在真实世界数据集上的广泛实验表明，BehaveGPT 在宏召回率和加权召回率上均优于最先进的基线，显示出其有效捕获和预测用户行为的能力。此外，我们首次在荣耀数据集上测量了用户行为领域的规模法则，提供了关于随数据和参数规模增加模型性能如何变化的见解。 

---
# TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments 

**Title (ZH)**: TransBench: 突破动态数字环境中可转移图形用户界面代理的障碍 

**Authors**: Yuheng Lu, Qian Yu, Hongru Wang, Zeming Liu, Wei Su, Yanping Liu, Yuhang Guo, Maocheng Liang, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17629)  

**Abstract**: Graphical User Interface (GUI) agents, which autonomously operate on digital interfaces through natural language instructions, hold transformative potential for accessibility, automation, and user experience. A critical aspect of their functionality is grounding - the ability to map linguistic intents to visual and structural interface elements. However, existing GUI agents often struggle to adapt to the dynamic and interconnected nature of real-world digital environments, where tasks frequently span multiple platforms and applications while also being impacted by version updates. To address this, we introduce TransBench, the first benchmark designed to systematically evaluate and enhance the transferability of GUI agents across three key dimensions: cross-version transferability (adapting to version updates), cross-platform transferability (generalizing across platforms like iOS, Android, and Web), and cross-application transferability (handling tasks spanning functionally distinct apps). TransBench includes 15 app categories with diverse functionalities, capturing essential pages across versions and platforms to enable robust evaluation. Our experiments demonstrate significant improvements in grounding accuracy, showcasing the practical utility of GUI agents in dynamic, real-world environments. Our code and data will be publicly available at Github. 

**Abstract (ZH)**: 图形用户界面（GUI）代理，可以通过自然语言指令自主操作数字界面，具有改变无障碍、自动化和用户体验的潜力。它们功能的一个关键方面是语义接地——即将语言意图映射到视觉和结构化的界面元件。然而，现有的GUI代理往往难以适应现实世界数字环境的动态和相互关联的性质，在这种环境中，任务经常跨越多个平台和应用程序，同时也受到版本更新的影响。为了解决这一问题，我们引入了TransBench，这是第一个旨在系统地评估和提升GUI代理跨三个关键维度的转移性能的基准：跨版本转移性能（适应版本更新）、跨平台转移性能（在iOS、Android和Web等平台之间泛化）以及跨应用转移性能（处理跨越功能不同应用程序的任务）。TransBench 包含15个具有不同功能的应用类别，涵盖了不同版本和平台的关键页面，以实现稳健的评估。我们的实验证明了在接地准确性方面的显著改进，展示了在动态现实世界环境中的GUI代理的实际用途。我们的代码和数据将在GitHub上公开。 

---
# \texttt{Range-Arithmetic}: Verifiable Deep Learning Inference on an Untrusted Party 

**Title (ZH)**: \texttt{范围算术}: 在不可信方进行可验证深度学习推断 

**Authors**: Ali Rahimi, Babak H. Khalaj, Mohammad Ali Maddah-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.17623)  

**Abstract**: Verifiable computing (VC) has gained prominence in decentralized machine learning systems, where resource-intensive tasks like deep neural network (DNN) inference are offloaded to external participants due to blockchain limitations. This creates a need to verify the correctness of outsourced computations without re-execution. We propose \texttt{Range-Arithmetic}, a novel framework for efficient and verifiable DNN inference that transforms non-arithmetic operations, such as rounding after fixed-point matrix multiplication and ReLU, into arithmetic steps verifiable using sum-check protocols and concatenated range proofs. Our approach avoids the complexity of Boolean encoding, high-degree polynomials, and large lookup tables while remaining compatible with finite-field-based proof systems. Experimental results show that our method not only matches the performance of existing approaches, but also reduces the computational cost of verifying the results, the computational effort required from the untrusted party performing the DNN inference, and the communication overhead between the two sides. 

**Abstract (ZH)**: 可验证计算（VC）在去中心化机器学习系统中日益受到重视，由于区块链的限制，资源密集型任务如深度神经网络（DNN）推理被外包给外部参与者，这产生了在不重新执行的情况下验证外包计算正确性的需求。我们提出了一种名为\texttt{Range-Arithmetic}的新框架，用于高效且可验证的DNN推理，将定点矩阵乘法后的舍入操作和ReLU等非算术操作转换为可使用和项检验协议及连接范围证明进行验证的算术步骤。我们的方法避免了布尔编码、高次多项式和大量查找表的复杂性，同时仍然与基于有限域的证明系统兼容。实验结果表明，我们的方法不仅达到了现有方法的性能水平，还降低了验证结果、不信任方进行DNN推理所需的计算成本以及双方之间的通信开销。 

---
# Scaling Image and Video Generation via Test-Time Evolutionary Search 

**Title (ZH)**: 通过测试时进化搜索扩展图像和视频生成 

**Authors**: Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17618)  

**Abstract**: As the marginal cost of scaling computation (data and parameters) during model pre-training continues to increase substantially, test-time scaling (TTS) has emerged as a promising direction for improving generative model performance by allocating additional computation at inference time. While TTS has demonstrated significant success across multiple language tasks, there remains a notable gap in understanding the test-time scaling behaviors of image and video generative models (diffusion-based or flow-based models). Although recent works have initiated exploration into inference-time strategies for vision tasks, these approaches face critical limitations: being constrained to task-specific domains, exhibiting poor scalability, or falling into reward over-optimization that sacrifices sample diversity. In this paper, we propose \textbf{Evo}lutionary \textbf{Search} (EvoSearch), a novel, generalist, and efficient TTS method that effectively enhances the scalability of both image and video generation across diffusion and flow models, without requiring additional training or model expansion. EvoSearch reformulates test-time scaling for diffusion and flow models as an evolutionary search problem, leveraging principles from biological evolution to efficiently explore and refine the denoising trajectory. By incorporating carefully designed selection and mutation mechanisms tailored to the stochastic differential equation denoising process, EvoSearch iteratively generates higher-quality offspring while preserving population diversity. Through extensive evaluation across both diffusion and flow architectures for image and video generation tasks, we demonstrate that our method consistently outperforms existing approaches, achieves higher diversity, and shows strong generalizability to unseen evaluation metrics. Our project is available at the website this https URL. 

**Abstract (ZH)**: 随着模型预训练期间扩展计算（数据和参数）的边际成本持续大幅增加，测试时扩展（TTS）已成为通过在推理时分配额外计算来提高生成模型性能的有前景方向。尽管TTS已经在多个语言任务中展现了显著的成功，但在图像和视频生成模型（基于扩散或流的方法）的测试时扩展行为上仍存在明显的认识差距。尽管近期工作已开始探索视觉任务的推理时策略，但这些方法面临关键限制：局限于特定任务领域、扩展性差或过度优化奖励从而牺牲样本多样性。在本文中，我们提出了一种新颖的、通用且高效的TTS方法——进化搜索（EvoSearch），该方法无需额外训练或模型扩展即可有效提升扩散和流模型在图像和视频生成中的扩展性。EvoSearch将扩散和流模型的测试时扩展重新表述为进化搜索问题，利用生物学进化原理高效探索和优化去噪轨迹。通过结合针对随机微分方程去噪过程精心设计的选择和突变机制，EvoSearch逐代生成更高质量的后代同时保持种群多样性。通过在图像和视频生成任务的扩散和流架构上进行广泛评估，我们展示了我们方法在各个方面都优于现有方法，实现了更高的多样性，并且在未见过的评估指标上具有强大的泛化能力。该项目详情可在以下网址获取：this https URL。 

---
# Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments 

**Title (ZH)**: RUNAWAY 是羞愧的，但有益：大规模语言模型代理在体感环境中的早期退出行为探究 

**Authors**: Qingyu Lu, Liang Ding, Siyi Cao, Xuebo Liu, Kanjian Zhang, Jinxia Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17616)  

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong planning and decision-making capabilities in complex embodied environments. However, such agents often suffer from inefficiencies in multi-turn interactions, frequently trapped in repetitive loops or issuing ineffective commands, leading to redundant computational overhead. Instead of relying solely on learning from trajectories, we take a first step toward exploring the early-exit behavior for LLM-based agents. We propose two complementary approaches: 1. an $\textbf{intrinsic}$ method that injects exit instructions during generation, and 2. an $\textbf{extrinsic}$ method that verifies task completion to determine when to halt an agent's trial. To evaluate early-exit mechanisms, we introduce two metrics: one measures the reduction of $\textbf{redundant steps}$ as a positive effect, and the other evaluates $\textbf{progress degradation}$ as a negative effect. Experiments with 4 different LLMs across 5 embodied environments show significant efficiency improvements, with only minor drops in agent performance. We also validate a practical strategy where a stronger agent assists after an early-exit agent, achieving better performance with the same total steps. We will release our code to support further research. 

**Abstract (ZH)**: 由大型语言模型驱动的代理在外域环境中的早期退出行为探究 

---
# Distilling LLM Agent into Small Models with Retrieval and Code Tools 

**Title (ZH)**: 将LLM代理精简为带有检索和代码工具的小模型 

**Authors**: Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17612)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at this https URL. 

**Abstract (ZH)**: 基于代理的代理蒸馏：将复杂推理和完整任务解决行为转移至小型语言模型 

---
# MinkUNeXt-SI: Improving point cloud-based place recognition including spherical coordinates and LiDAR intensity 

**Title (ZH)**: MinkUNeXt-SI: 基于点云的空间识别改进，包括球坐标和激光雷达强度 

**Authors**: Judith Vilella-Cantos, Juan José Cabrera, Luis Payá, Mónica Ballesta, David Valiente  

**Link**: [PDF](https://arxiv.org/pdf/2505.17591)  

**Abstract**: In autonomous navigation systems, the solution of the place recognition problem is crucial for their safe functioning. But this is not a trivial solution, since it must be accurate regardless of any changes in the scene, such as seasonal changes and different weather conditions, and it must be generalizable to other environments. This paper presents our method, MinkUNeXt-SI, which, starting from a LiDAR point cloud, preprocesses the input data to obtain its spherical coordinates and intensity values normalized within a range of 0 to 1 for each point, and it produces a robust place recognition descriptor. To that end, a deep learning approach that combines Minkowski convolutions and a U-net architecture with skip connections is used. The results of MinkUNeXt-SI demonstrate that this method reaches and surpasses state-of-the-art performance while it also generalizes satisfactorily to other datasets. Additionally, we showcase the capture of a custom dataset and its use in evaluating our solution, which also achieves outstanding results. Both the code of our solution and the runs of our dataset are publicly available for reproducibility purposes. 

**Abstract (ZH)**: 在自主导航系统中，地点识别问题的解决方案对于其安全运行至关重要。但这并非一项简单的任务，因为它必须在场景任何变化（如季节变化和不同天气条件）下保持准确，并且能够泛化到其他环境中。本文提出了一种方法MinkUNeXt-SI，该方法从LiDAR点云出发，预处理输入数据以获得每个点的归一化在0到1范围内的球坐标和强度值，并生成稳健的地点识别描述符。为此，该方法使用结合Minkowski卷积和具有跳接连接的U-net架构的深度学习方法。MinkUNeXt-SI的实验结果表明，该方法不仅达到了最先进的性能，而且还能够在其他数据集中泛化得当。此外，我们展示了自定义数据集的采集及其在评估我们解决方案中的应用，该解决方案同样取得了出色的成果。我们的解决方案代码和数据集运行结果均已公开，以便于可重复性。 

---
# CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training 

**Title (ZH)**: CosyVoice 3：通过规模化扩增和后训练 towards户外语音生成 

**Authors**: Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.17589)  

**Abstract**: In our prior works, we introduced a scalable streaming speech synthesis model, CosyVoice 2, which integrates a large language model (LLM) and a chunk-aware flow matching (FM) model, and achieves low-latency bi-streaming speech synthesis and human-parity quality. Despite these advancements, CosyVoice 2 exhibits limitations in language coverage, domain diversity, data volume, text formats, and post-training techniques. In this paper, we present CosyVoice 3, an improved model designed for zero-shot multilingual speech synthesis in the wild, surpassing its predecessor in content consistency, speaker similarity, and prosody naturalness. Key features of CosyVoice 3 include: 1) A novel speech tokenizer to improve prosody naturalness, developed via supervised multi-task training, including automatic speech recognition, speech emotion recognition, language identification, audio event detection, and speaker analysis. 2) A new differentiable reward model for post-training applicable not only to CosyVoice 3 but also to other LLM-based speech synthesis models. 3) Dataset Size Scaling: Training data is expanded from ten thousand hours to one million hours, encompassing 9 languages and 18 Chinese dialects across various domains and text formats. 4) Model Size Scaling: Model parameters are increased from 0.5 billion to 1.5 billion, resulting in enhanced performance on our multilingual benchmark due to the larger model capacity. These advancements contribute significantly to the progress of speech synthesis in the wild. We encourage readers to listen to the demo at this https URL. 

**Abstract (ZH)**: CosyVoice 3：面向野外的零样本多语言语音合成改进模型 

---
# JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models 

**Title (ZH)**: JALMBench: 评估音频语言模型逃逸漏洞的基准测试 

**Authors**: Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17568)  

**Abstract**: Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level. 

**Abstract (ZH)**: Audio Language Models (ALMs)在近年来取得了显著进展。这些模型直接将音频模态整合进模型中，而不是将语音转换成文本并输入到大型语言模型（LLMs）中。虽然对LLMs的jailbreak攻击已经被广泛研究，但具有音频模态的ALMs的安全性仍然很大程度上没有被探索。目前缺乏专门设计用于评估和比较攻击和ALMs的对抗音频数据集和统一框架。在本文中，我们提出了JALMBench，这是首个全面的基准测试，用于评估ALMs在jailbreak攻击下的安全性。JALMBench包含2,200个文本样本和51,381个音频样本，总时长达268小时，支持12种主流ALMs、4种文本转移和4种音频起源的攻击方法以及5种防御方法。使用JALMBench，我们对攻击效率、主题敏感性、语音多样性和攻击表示进行了深入分析。此外，我们还探讨了在提示级别和响应级别上减轻攻击的策略。 

---
# Model Already Knows the Best Noise: Bayesian Active Noise Selection via Attention in Video Diffusion Model 

**Title (ZH)**: 模型已经知道最好的噪声：视频扩散模型中的注意力驱动贝叶斯主动噪声选择 

**Authors**: Kwanyoung Kim, Sanghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17561)  

**Abstract**: The choice of initial noise significantly affects the quality and prompt alignment of video diffusion models, where different noise seeds for the same prompt can lead to drastically different generations. While recent methods rely on externally designed priors such as frequency filters or inter-frame smoothing, they often overlook internal model signals that indicate which noise seeds are inherently preferable. To address this, we propose ANSE (Active Noise Selection for Generation), a model-aware framework that selects high-quality noise seeds by quantifying attention-based uncertainty. At its core is BANSA (Bayesian Active Noise Selection via Attention), an acquisition function that measures entropy disagreement across multiple stochastic attention samples to estimate model confidence and consistency. For efficient inference-time deployment, we introduce a Bernoulli-masked approximation of BANSA that enables score estimation using a single diffusion step and a subset of attention layers. Experiments on CogVideoX-2B and 5B demonstrate that ANSE improves video quality and temporal coherence with only an 8% and 13% increase in inference time, respectively, providing a principled and generalizable approach to noise selection in video diffusion. See our project page: this https URL 

**Abstract (ZH)**: 初始噪声的选择显著影响视频扩散模型的质量和提示对齐，不同的噪声种子对于相同的提示可能导致截然不同的生成结果。虽然最近的方法依赖于外部设计的先验知识，如频域滤波或帧间平滑，但它们往往会忽略模型内部信号，这些信号能够指示哪些噪声种子更为优选。为了解决这一问题，我们提出了ANSE（Active Noise Selection for Generation），一种基于模型的框架，通过量化基于注意力的不确定性来选择高质量的噪声种子。其核心是BANSA（Bayesian Active Noise Selection via Attention），一种通过测量多组随机注意力抽样的熵不一致性来估计模型信心和一致性的获取函数。为了高效地在推理阶段部署，我们引入了一种针对BANSA的伯努利掩码近似，该方法仅需一个扩散步骤和部分注意力层即可估算分数。在CogVideoX-2B和5B上的实验结果显示，ANSE在仅增加8%和13%的推理时间的情况下提升了视频质量和时间连贯性，提供了一种原则性和可泛化的视频扩散中噪声选择方法。见我们的项目页面：this https URL 

---
# Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection 

**Title (ZH)**: 教学以谎言为手段：合成负样本下的课程DPO幻觉检测 

**Authors**: Shrey Pandit, Ashwin Vinod, Liu Leqi, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.17558)  

**Abstract**: Aligning large language models (LLMs) to accurately detect hallucinations remains a significant challenge due to the sophisticated nature of hallucinated text. Recognizing that hallucinated samples typically exhibit higher deceptive quality than traditional negative samples, we use these carefully engineered hallucinations as negative examples in the DPO alignment procedure. Our method incorporates a curriculum learning strategy, gradually transitioning the training from easier samples, identified based on the greatest reduction in probability scores from independent fact checking models, to progressively harder ones. This structured difficulty scaling ensures stable and incremental learning. Experimental evaluation demonstrates that our HaluCheck models, trained with curriculum DPO approach and high quality negative samples, significantly improves model performance across various metrics, achieving improvements of upto 24% on difficult benchmarks like MedHallu and HaluEval. Additionally, HaluCheck models demonstrate robustness in zero-shot settings, significantly outperforming larger state-of-the-art models across various benchmarks. 

**Abstract (ZH)**: 将大语言模型（LLMs）对幻觉准确检测进行对齐仍是一项重大挑战，因为幻觉文本具有复杂性。鉴于幻觉样本通常比传统负样本具有更高的欺骗质量，我们使用这些精心设计的幻觉作为DPO对齐过程中的负例子。我们的方法结合了课程学习策略，逐步过渡从基于独立事实检查模型概率分数最大减少的 easier 样本到逐渐更难的样本。这种结构化的难度递增确保了稳定和逐步的学习。实验评估表明，使用课程DPO方法和高质量负样本训练的HaluCheck模型在各种指标上显著提高了模型性能，在如MedHallu和HaluEval等困难基准上取得了高达24%的改进。此外，HaluCheck模型在零样本设置中表现出较强的鲁棒性，各基准上显著优于更大规模的最先进的模型。 

---
# Universal Biological Sequence Reranking for Improved De Novo Peptide Sequencing 

**Title (ZH)**: 通用生物序列重新排序以提高从头肽测序性能 

**Authors**: Zijie Qiu, Jiaqi Wei, Xiang Zhang, Sheng Xu, Kai Zou, Zhi Jin, Zhiqiang Gao, Nanqing Dong, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17552)  

**Abstract**: De novo peptide sequencing is a critical task in proteomics. However, the performance of current deep learning-based methods is limited by the inherent complexity of mass spectrometry data and the heterogeneous distribution of noise signals, leading to data-specific biases. We present RankNovo, the first deep reranking framework that enhances de novo peptide sequencing by leveraging the complementary strengths of multiple sequencing models. RankNovo employs a list-wise reranking approach, modeling candidate peptides as multiple sequence alignments and utilizing axial attention to extract informative features across candidates. Additionally, we introduce two new metrics, PMD (Peptide Mass Deviation) and RMD (residual Mass Deviation), which offer delicate supervision by quantifying mass differences between peptides at both the sequence and residue levels. Extensive experiments demonstrate that RankNovo not only surpasses its base models used to generate training candidates for reranking pre-training, but also sets a new state-of-the-art benchmark. Moreover, RankNovo exhibits strong zero-shot generalization to unseen models whose generations were not exposed during training, highlighting its robustness and potential as a universal reranking framework for peptide sequencing. Our work presents a novel reranking strategy that fundamentally challenges existing single-model paradigms and advances the frontier of accurate de novo sequencing. Our source code is provided on GitHub. 

**Abstract (ZH)**: 新型肽序列排序框架RankNovo：多种排序模型互补增强的深度重排序方法 

---
# RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning 

**Title (ZH)**: 增强推理提示:通过强化学习的文本到图像生成重提示 

**Authors**: Mingrui Wu, Lu Wang, Pu Zhao, Fangkai Yang, Jianjin Zhang, Jianfeng Liu, Yuefeng Zhan, Weihao Han, Hao Sun, Jiayi Ji, Xiaoshuai Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.17540)  

**Abstract**: Despite recent progress in text-to-image (T2I) generation, existing models often struggle to faithfully capture user intentions from short and under-specified prompts. While prior work has attempted to enhance prompts using large language models (LLMs), these methods frequently generate stylistic or unrealistic content due to insufficient grounding in visual semantics and real-world composition. Inspired by recent advances in reasoning for language model, we propose RePrompt, a novel reprompting framework that introduces explicit reasoning into the prompt enhancement process via reinforcement learning. Instead of relying on handcrafted rules or stylistic rewrites, our method trains a language model to generate structured, self-reflective prompts by optimizing for image-level outcomes. The tailored reward models assesse the generated images in terms of human preference, semantic alignment, and visual composition, providing indirect supervision to refine prompt generation. Our approach enables end-to-end training without human-annotated data. Experiments on GenEval and T2I-Compbench show that RePrompt significantly boosts spatial layout fidelity and compositional generalization across diverse T2I backbones, establishing new state-of-the-art results. 

**Abstract (ZH)**: 尽管近年来文本到图像（T2I）生成取得了进展，现有模型往往难以忠实捕捉短且不具体提示中的用户意图。虽然先前的研究试图通过大型语言模型（LLMs）增强提示，但这些方法经常生成风格化或不现实的内容，因为它们缺乏在视觉语义和现实世界构成方面的充分基础。借鉴语言模型推理的最新进展，我们提出了一种名为RePrompt的新型回提示框架，该框架通过强化学习将显式推理引入提示增强过程。我们的方法不依赖于手工构建的规则或风格重写，而是训练一个语言模型生成结构化、自我反思性的提示，通过优化图像级别结果进行训练。定制的奖励模型根据人类偏好、语义对齐和视觉构成评估生成的图像，间接监督提升提示生成的质量。我们的方法无需人工标注数据即可实现端到端训练。在GenEval和T2I-Compbench上进行的实验表明，RePrompt在多种T2I主干网络上显著提高了空间布局保真度和组成泛化能力，建立了新的最佳结果。 

---
# Learning Representational Disparities 

**Title (ZH)**: 学习表示差异 

**Authors**: Pavan Ravishankar, Rushabh Shah, Daniel B. Neill  

**Link**: [PDF](https://arxiv.org/pdf/2505.17533)  

**Abstract**: We propose a fair machine learning algorithm to model interpretable differences between observed and desired human decision-making, with the latter aimed at reducing disparity in a downstream outcome impacted by the human decision. Prior work learns fair representations without considering the outcome in the decision-making process. We model the outcome disparities as arising due to the different representations of the input seen by the observed and desired decision-maker, which we term representational disparities. Our goal is to learn interpretable representational disparities which could potentially be corrected by specific nudges to the human decision, mitigating disparities in the downstream outcome; we frame this as a multi-objective optimization problem using a neural network. Under reasonable simplifying assumptions, we prove that our neural network model of the representational disparity learns interpretable weights that fully mitigate the outcome disparity. We validate objectives and interpret results using real-world German Credit, Adult, and Heritage Health datasets. 

**Abstract (ZH)**: 我们提出一种公平的机器学习算法来建模观察到的人类决策与期望的人类决策之间的可解释差异，后者旨在减少受人类决策影响的下游结果中的不公平性。以往工作在决策过程中未考虑结果来学习公平表示。我们将结果差异建模为由于观察到的决策者和期望的决策者看到的输入表示不同所致，我们称之为表示差异。我们的目标是学习可解释的表示差异，这些差异可以通过特定的人类决策调整来潜在地纠正，从而减轻下游结果中的不公；我们将此问题表述为一个多目标优化问题，并使用神经网络来建模表示差异。在合理的简化假设下，我们证明我们的表示差异的神经网络模型学习到了完全缓解结果差异的可解释权重。我们使用真实的德国信用、成人和遗产健康数据集来验证目标并解释结果。 

---
# Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding 

**Title (ZH)**: 你在留意我的问题吗？基于注意力导向 ensemble 解码减轻多模态幻觉 

**Authors**: Yeongjae Cho, Keonwoo Kim, Taebaek Hwang, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.17529)  

**Abstract**: Recent advancements in Large Vision-Language Models (LVLMs) have significantly expanded their utility in tasks like image captioning and visual question answering. However, they still struggle with object hallucination, where models generate descriptions that inaccurately reflect the visual content by including nonexistent objects or misrepresenting existing ones. While previous methods, such as data augmentation and training-free approaches, strive to tackle this issue, they still encounter scalability challenges and often depend on additional external modules. In this work, we propose Ensemble Decoding (ED), a novel strategy that splits the input image into sub-images and combines logit distributions by assigning weights through the attention map. Furthermore, we introduce ED adaptive plausibility constraint to calibrate logit distribution and FastED, a variant designed for speed-critical applications. Extensive experiments across hallucination benchmarks demonstrate that our proposed method achieves state-of-the-art performance, validating the effectiveness of our approach. 

**Abstract (ZH)**: 近期大规模愿景语言模型（LVLMs）的进展显著扩展了其在图像描述和视觉问答等任务中的应用，但仍面临对象幻觉的问题，即模型生成描述时会包含不存在的对象或错误地描述现有对象。虽然之前的方法，如数据增强和无训练方法，努力解决这一问题，但它们仍然面临可扩展性挑战，并且通常依赖额外的外部模块。在本工作中，我们提出了一种新的 Ensemble Decoding（ED）策略，该策略将输入图像分割成子图像，并通过注意力图分配权重来组合logit分布。此外，我们引入了ED自适应可实现性约束来校准logit分布，并设计了FastED变体以满足速度关键型应用的需求。广泛实验结果表明，我们的方法在幻觉基准测试中达到了最佳性能，验证了该方法的有效性。 

---
# Multi-agent Systems for Misinformation Lifecycle : Detection, Correction And Source Identification 

**Title (ZH)**: 多 agent 系统在错误信息生命周期中的检测、纠正与源头识别 

**Authors**: Aditya Gautam  

**Link**: [PDF](https://arxiv.org/pdf/2505.17511)  

**Abstract**: The rapid proliferation of misinformation in digital media demands solutions that go beyond isolated Large Language Model(LLM) or AI Agent based detection methods. This paper introduces a novel multi-agent framework that covers the complete misinformation lifecycle: classification, detection, correction, and source verification to deliver more transparent and reliable outcomes. In contrast to single-agent or monolithic architectures, our approach employs five specialized agents: an Indexer agent for dynamically maintaining trusted repositories, a Classifier agent for labeling misinformation types, an Extractor agent for evidence based retrieval and ranking, a Corrector agent for generating fact-based correction and a Verification agent for validating outputs and tracking source credibility. Each agent can be individually evaluated and optimized, ensuring scalability and adaptability as new types of misinformation and data sources emerge. By decomposing the misinformation lifecycle into specialized agents - our framework enhances scalability, modularity, and explainability. This paper proposes a high-level system overview, agent design with emphasis on transparency, evidence-based outputs, and source provenance to support robust misinformation detection and correction at scale. 

**Abstract (ZH)**: 数字媒体中虚假信息的迅速蔓延需要超越孤立的大语言模型或AI代理检测方法的解决方案。本文介绍了一种新的多代理框架，涵盖了虚假信息的完整生命周期：分类、检测、纠正和来源验证，以提供更加透明和可靠的成果。与单代理或单一架构不同，我们的方法采用了五个专门的代理：索引代理用于动态维护可信的仓储，分类代理用于标注虚假信息类型，提取代理用于基于证据的检索和排序，纠正代理用于生成基于事实的纠正内容，验证代理用于验证输出并跟踪来源可信度。每个代理都可以单独评估和优化，确保在新类型虚假信息和数据源出现时具备可扩展性和适应性。通过将虚假信息生命周期分解为专门的代理，我们的框架增强了可扩展性、模块化和可解释性。本文提出了一种高层次的系统概述，并强调透明性、基于证据的输出和来源追溯，以支持大规模的虚假信息检测和纠正。 

---
# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning 

**Title (ZH)**: 基于KL正则化策略梯度算法的大型语言模型推理设计 

**Authors**: Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Yang Yuan, Quanquan Gu, Andrew C Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17508)  

**Abstract**: Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). Despite the widespread use of Kullback-Leibler (KL) regularization in policy gradient algorithms to stabilize training, the systematic exploration of how different KL divergence formulations can be estimated and integrated into surrogate loss functions for online reinforcement learning (RL) presents a nuanced and systematically explorable design space. In this paper, we propose regularized policy gradient (RPG), a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online RL setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. We conduct extensive experiments on RL for LLM reasoning using these methods, showing improved or competitive results in terms of training stability and performance compared to strong baselines such as GRPO, REINFORCE++, and DAPO. The code is available at this https URL. 

**Abstract (ZH)**: 正则化策略梯度（RPG）：在线强化学习中KL正则化策略梯度方法的设计与分析 

---
# RoHyDR: Robust Hybrid Diffusion Recovery for Incomplete Multimodal Emotion Recognition 

**Title (ZH)**: RoHyDR：鲁棒混合扩散恢复 incomplete 多模态情感识别 

**Authors**: Yuehan Jin, Xiaoqing Liu, Yiyuan Yang, Zhiwen Yu, Tong Zhang, Kaixiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17501)  

**Abstract**: Multimodal emotion recognition analyzes emotions by combining data from multiple sources. However, real-world noise or sensor failures often cause missing or corrupted data, creating the Incomplete Multimodal Emotion Recognition (IMER) challenge. In this paper, we propose Robust Hybrid Diffusion Recovery (RoHyDR), a novel framework that performs missing-modality recovery at unimodal, multimodal, feature, and semantic levels. For unimodal representation recovery of missing modalities, RoHyDR exploits a diffusion-based generator to generate distribution-consistent and semantically aligned representations from Gaussian noise, using available modalities as conditioning. For multimodal fusion recovery, we introduce adversarial learning to produce a realistic fused multimodal representation and recover missing semantic content. We further propose a multi-stage optimization strategy that enhances training stability and efficiency. In contrast to previous work, the hybrid diffusion and adversarial learning-based recovery mechanism in RoHyDR allows recovery of missing information in both unimodal representation and multimodal fusion, at both feature and semantic levels, effectively mitigating performance degradation caused by suboptimal optimization. Comprehensive experiments conducted on two widely used multimodal emotion recognition benchmarks demonstrate that our proposed method outperforms state-of-the-art IMER methods, achieving robust recognition performance under various missing-modality scenarios. Our code will be made publicly available upon acceptance. 

**Abstract (ZH)**: 多模态情绪识别中的鲁棒混合扩散恢复（RoHyDR）：一种在单模态、多模态、特征和语义级别上进行缺失模态恢复的新框架 

---
# The Discovery Engine: A Framework for AI-Driven Synthesis and Navigation of Scientific Knowledge Landscapes 

**Title (ZH)**: AI驱动的科学知识景观合成与导航框架：发现引擎 

**Authors**: Vladimir Baulin, Austin Cook, Daniel Friedman, Janna Lumiruusu, Andrew Pashea, Shagor Rahman, Benedikt Waldeck  

**Link**: [PDF](https://arxiv.org/pdf/2505.17500)  

**Abstract**: The prevailing model for disseminating scientific knowledge relies on individual publications dispersed across numerous journals and archives. This legacy system is ill suited to the recent exponential proliferation of publications, contributing to insurmountable information overload, issues surrounding reproducibility and retractions. We introduce the Discovery Engine, a framework to address these challenges by transforming an array of disconnected literature into a unified, computationally tractable representation of a scientific domain. Central to our approach is the LLM-driven distillation of publications into structured "knowledge artifacts," instances of a universal conceptual schema, complete with verifiable links to source evidence. These artifacts are then encoded into a high-dimensional Conceptual Tensor. This tensor serves as the primary, compressed representation of the synthesized field, where its labeled modes index scientific components (concepts, methods, parameters, relations) and its entries quantify their interdependencies. The Discovery Engine allows dynamic "unrolling" of this tensor into human-interpretable views, such as explicit knowledge graphs (the CNM graph) or semantic vector spaces, for targeted exploration. Crucially, AI agents operate directly on the graph using abstract mathematical and learned operations to navigate the knowledge landscape, identify non-obvious connections, pinpoint gaps, and assist researchers in generating novel knowledge artifacts (hypotheses, designs). By converting literature into a structured tensor and enabling agent-based interaction with this compact representation, the Discovery Engine offers a new paradigm for AI-augmented scientific inquiry and accelerated discovery. 

**Abstract (ZH)**: 现有的科学知识传播模型依赖于分散在众多期刊和档案中的个体文献。这一遗留系统不适用于近期指数级增长的文献数量，导致信息过载问题，并且存在可重复性和撤稿问题。我们提出了一种Discovery Engine框架，以应对这些挑战，通过将分散的文献转化为统一的、计算上可处理的科学领域表示。我们方法的核心是通过大语言模型驱动的文献精炼，将其转化为结构化的“知识构件”，这些构件是通用概念模式的实例，同时包含可验证的源证据链接。这些构件随后被编码为高维概念张量。该张量作为综合领域的主要、压缩表示，其标记模式索引科学组件（概念、方法、参数、关系），而其条目量化它们之间的相互依赖关系。Discovery Engine允许动态“展开”这一张量，生成可由人类解释的观点，如明示知识图（CNM图）或语义向量空间，以实现有针对性的探索。至关重要的是，AI代理直接在图上操作，使用抽象数学和学习到的操作来导航知识景观，识别非显而易见的联系，指出缺失，并协助研究人员生成新的知识构件（假设、设计）。通过将文献转换为结构化张量，并使代理能够与这种紧凑表示进行交互，Discovery Engine为AI增强的科学探究和加速发现提供了一种新范式。 

---
# Managing FAIR Knowledge Graphs as Polyglot Data End Points: A Benchmark based on the rdf2pg Framework and Plant Biology Data 

**Title (ZH)**: 管理FAIR知识图谱作为多语言数据端点：基于rdf2pg框架与植物生物学数据的基准测试 

**Authors**: Marco Brandizi, Carlos Bobed, Luca Garulli, Arné de Klerk, Keywan Hassani-Pak  

**Link**: [PDF](https://arxiv.org/pdf/2505.17498)  

**Abstract**: Linked Data and labelled property graphs (LPG) are two data management approaches with complementary strengths and weaknesses, making their integration beneficial for sharing datasets and supporting software ecosystems. In this paper, we introduce rdf2pg, an extensible framework for mapping RDF data to semantically equivalent LPG formats and data-bases. Utilising this framework, we perform a comparative analysis of three popular graph databases - Virtuoso, Neo4j, and ArcadeDB - and the well-known graph query languages SPARQL, Cypher, and Gremlin. Our qualitative and quantitative as-sessments underline the strengths and limitations of these graph database technologies. Additionally, we highlight the potential of rdf2pg as a versatile tool for enabling polyglot access to knowledge graphs, aligning with established standards of Linked Data and the Semantic Web. 

**Abstract (ZH)**: Linked Data和标记属性图(LPG)是两种具有互补优势和劣势的数据管理方法，其集成对于共享数据集和支持软件生态系统有益。本文介绍了rdf2pg，一个可扩展的框架，用于将RDF数据映射到语义等价的LPG格式和数据库中。利用该框架，我们对三款流行的图数据库——Virtuoso、Neo4j和ArcadeDB，以及知名的图查询语言SPARQL、Cypher和Gremlin进行了比较分析。我们的定性和定量评估突显了这些图数据库技术的优势和局限性。此外，我们强调了rdf2pg作为多功能工具的潜力，以实现对知识图的多语言访问，符合链接数据和语义网的既定标准。 

---
# Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models 

**Title (ZH)**: 端到端训练语音语言模型中灾难性遗忘缓解策略分析 

**Authors**: Chi-Yuan Hsiao, Ke-Han Lu, Kai-Wei Chang, Chih-Kai Yang, Wei-Chih Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17496)  

**Abstract**: End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies-model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines. 

**Abstract (ZH)**: 端到端培训语音语言模型中的灾难性遗忘及其缓解策略 

---
# ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs 

**Title (ZH)**: ProxySPEX: 通过LLMs中稀疏特征交互实现高效的推理解释性 

**Authors**: Landon Butler, Abhineet Agarwal, Justin Singh Kang, Yigit Efe Erginbas, Bin Yu, Kannan Ramchandran  

**Link**: [PDF](https://arxiv.org/pdf/2505.17495)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX identifies features that influence model output over 20% more than those selected by marginal approaches. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. ProxySPEX identifies interactions that enable more aggressive pruning of heads than marginal approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过捕捉输入特征之间的复杂交互关系实现了出色的性能。为了识别这些交互，大多数现有方法需要枚举所有可能的特征组合，导致它们在输入数量 $n$ 增加时扩展性较差。最近，Kang等人（2025）提出了SPEX，一种信息论方法，利用交互稀疏性扩展到 $n \approx 10^3$ 个特征。SPEX 在改进先前方法的同时，需要数万个模型推断，这对于大型模型来说是具有挑战性的。在本文中，我们观察到LLM特征交互通常是分层的——高阶交互伴随着其低阶子集，这使得更有效地发现这些交互成为可能。为了利用这种分层结构，我们提出了ProxySPEX，一种交互归因算法，首先使用渐进增强树拟合遮蔽的LLM输出，然后提取重要的交互。在四个具有挑战性和高维的数据集上的实验表明，与边际归因方法相比，ProxySPEX 在重建LLM输出方面提高了20%，并且仅使用SPEX所需推断次数的十分之一。通过考虑交互，ProxySPEX 识别出影响模型输出的特征，这些特征比边际方法所选出的特征多出20%以上。此外，我们使用ProxySPEX 应用于两个可解释性任务：数据归因任务中，我们识别出影响测试预测的CIFAR-10训练样本之间的交互；机制可解释性任务中，我们揭示了不同层内的和跨层的注意力头之间的交互，特别是在问答任务中。ProxySPEX 识别出的交互使得相对于边际方法，头的剪枝更加激进。 

---
# HiLAB: A Hybrid Inverse-Design Framework 

**Title (ZH)**: HiLAB：一种混合逆设计框架 

**Authors**: Reza Marzban, Hamed Abiri, Raphael Pestourie, Ali Adibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17491)  

**Abstract**: HiLAB (Hybrid inverse-design with Latent-space learning, Adjoint-based partial optimizations, and Bayesian optimization) is a new paradigm for inverse design of nanophotonic structures. Combining early-terminated topological optimization (TO) with a Vision Transformer-based variational autoencoder (VAE) and a Bayesian search, HiLAB addresses multi-functional device design by generating diverse freeform configurations at reduced simulation costs. Shortened adjoint-driven TO runs, coupled with randomized physical parameters, produce robust initial structures. These structures are compressed into a compact latent space by the VAE, enabling Bayesian optimization to co-optimize geometry and physical hyperparameters. Crucially, the trained VAE can be reused for alternative objectives or constraints by adjusting only the acquisition function. Compared to conventional TO pipelines prone to local optima, HiLAB systematically explores near-global optima with considerably fewer electromagnetic simulations. Even after accounting for training overhead, the total number of full simulations decreases by over an order of magnitude, accelerating the discovery of fabrication-friendly devices. Demonstrating its efficacy, HiLAB is used to design an achromatic beam deflector for red, green, and blue wavelengths, achieving balanced diffraction efficiencies of ~25% while mitigating chromatic aberrations-a performance surpassing existing demonstrations. Overall, HiLAB provides a flexible platform for robust, multi-parameter photonic designs and rapid adaptation to next-generation nanophotonic challenges. 

**Abstract (ZH)**: HiLAB（混合逆设计与潜在空间学习、基于雅克比的方法部分优化和贝叶斯优化）是一种纳米光子结构逆设计的新范式。 

---
# DTRT: Enhancing Human Intent Estimation and Role Allocation for Physical Human-Robot Collaboration 

**Title (ZH)**: DTRT: 提升物理人机协作中的人类意图估计与角色分配 

**Authors**: Haotian Liu, Yuchuang Tong, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17490)  

**Abstract**: In physical Human-Robot Collaboration (pHRC), accurate human intent estimation and rational human-robot role allocation are crucial for safe and efficient assistance. Existing methods that rely on short-term motion data for intention estimation lack multi-step prediction capabilities, hindering their ability to sense intent changes and adjust human-robot assignments autonomously, resulting in potential discrepancies. To address these issues, we propose a Dual Transformer-based Robot Trajectron (DTRT) featuring a hierarchical architecture, which harnesses human-guided motion and force data to rapidly capture human intent changes, enabling accurate trajectory predictions and dynamic robot behavior adjustments for effective collaboration. Specifically, human intent estimation in DTRT uses two Transformer-based Conditional Variational Autoencoders (CVAEs), incorporating robot motion data in obstacle-free case with human-guided trajectory and force for obstacle avoidance. Additionally, Differential Cooperative Game Theory (DCGT) is employed to synthesize predictions based on human-applied forces, ensuring robot behavior align with human intention. Compared to state-of-the-art (SOTA) methods, DTRT incorporates human dynamics into long-term prediction, providing an accurate understanding of intention and enabling rational role allocation, achieving robot autonomy and maneuverability. Experiments demonstrate DTRT's accurate intent estimation and superior collaboration performance. 

**Abstract (ZH)**: 基于物理的人机协作中的人意图估计与合理的人机角色分配对于安全高效的辅助至关重要。现有的依赖短期运动数据进行意图估计的方法缺乏多步预测能力，阻碍了其自主感知意图变化和调整人机分配的能力，导致潜在的不一致。为了解决这些问题，我们提出了一种基于双变换器的机器人轨迹推理机（DTRT），该模型具有分层架构，利用人引导的运动和力数据快速捕捉人类意图的变化，实现精确的轨迹预测和动态的机器人行为调整，以实现有效的协作。具体而言，在DTRT中使用两个基于变换器的条件变分自编码器（CVAEs）进行人类意图估计，在无障碍情况下结合机器人运动数据和人引导的轨迹及力以避障。此外，采用差分合作博弈理论（DCGT）基于人类施加的力量合成预测，确保机器人行为与人类意图相符。与现有最先进的方法相比，DTRT将人类动力学融入长期预测中，提供对意图的准确理解，并实现合理的人机角色分配，达到机器人自主性和机动性。实验结果表明，DTRT在意图估计和协作性能方面更为准确和优越。 

---
# keepitsimple at SemEval-2025 Task 3: LLM-Uncertainty based Approach for Multilingual Hallucination Span Detection 

**Title (ZH)**: SemEval-2025 任务3的简单胜出：基于LLM不确定性的方法用于多语言幻觉 spans 检测 

**Authors**: Saketh Reddy Vemula, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2505.17485)  

**Abstract**: Identification of hallucination spans in black-box language model generated text is essential for applications in the real world. A recent attempt at this direction is SemEval-2025 Task 3, Mu-SHROOM-a Multilingual Shared Task on Hallucinations and Related Observable Over-generation Errors. In this work, we present our solution to this problem, which capitalizes on the variability of stochastically-sampled responses in order to identify hallucinated spans. Our hypothesis is that if a language model is certain of a fact, its sampled responses will be uniform, while hallucinated facts will yield different and conflicting results. We measure this divergence through entropy-based analysis, allowing for accurate identification of hallucinated segments. Our method is not dependent on additional training and hence is cost-effective and adaptable. In addition, we conduct extensive hyperparameter tuning and perform error analysis, giving us crucial insights into model behavior. 

**Abstract (ZH)**: 黑盒语言模型生成文本中幻觉片段的识别对于实际应用至关重要。SemEval-2025 Task 3，一个多语言共享任务，关注幻觉及相关可观察过度生成错误。在本文中，我们提出了一种解决方案，该解决方案利用随机采样响应的可变性来识别幻觉片段。我们的假设是，如果语言模型对其事实确信无疑，其采样响应将是统一的，而幻觉事实将导致不同的且冲突的结果。我们通过基于熵的分析来衡量这种差异，从而能够准确地识别出幻觉段落。我们的方法不依赖于额外的训练，因此具有成本效益且易于适应。此外，我们进行了广泛的超参数调整并进行了错误分析，为我们提供了关于模型行为的关键见解。 

---
# Anatomy-Guided Multitask Learning for MRI-Based Classification of Placenta Accreta Spectrum and its Subtypes 

**Title (ZH)**: 基于解剖引导多任务学习的MRI分类及其亚型分类方法的研究：植入性胎盘谱系及其亚型 

**Authors**: Hai Jiang, Qiongting Liu, Yuanpin Zhou, Jiawei Pan, Ting Song, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17484)  

**Abstract**: Placenta Accreta Spectrum Disorders (PAS) pose significant risks during pregnancy, frequently leading to postpartum hemorrhage during cesarean deliveries and other severe clinical complications, with bleeding severity correlating to the degree of placental invasion. Consequently, accurate prenatal diagnosis of PAS and its subtypes-placenta accreta (PA), placenta increta (PI), and placenta percreta (PP)-is crucial. However, existing guidelines and methodologies predominantly focus on the presence of PAS, with limited research addressing subtype recognition. Additionally, previous multi-class diagnostic efforts have primarily relied on inefficient two-stage cascaded binary classification tasks. In this study, we propose a novel convolutional neural network (CNN) architecture designed for efficient one-stage multiclass diagnosis of PAS and its subtypes, based on 4,140 magnetic resonance imaging (MRI) slices. Our model features two branches: the main classification branch utilizes a residual block architecture comprising multiple residual blocks, while the second branch integrates anatomical features of the uteroplacental area and the adjacent uterine serous layer to enhance the model's attention during classification. Furthermore, we implement a multitask learning strategy to leverage both branches effectively. Experiments conducted on a real clinical dataset demonstrate that our model achieves state-of-the-art performance. 

**Abstract (ZH)**: 胎盘植入谱系障碍（PAS）在妊娠期间造成显著风险， frequently 导致剖宫产时出现产后出血及其他严重临床并发症，出血严重程度与胎盘侵袭程度相关。因此，PAS及其亚型胎盘植入（PA）、胎盘増生（PI）和胎盘穿透（PP）的准确产前诊断至关重要。然而，现有指南和方法主要关注PAS的存在，对亚型识别的研究较少。此外，以前的多分类诊断努力主要依赖于不高效的两阶段级联二元分类任务。在本研究中，我们提出了一种新型卷积神经网络（CNN）架构，用于基于4,140张磁共振成像（MRI）切片对PAS及其亚型进行高效的一阶段多分类诊断。该模型包含两个分支：主分类分支采用包含多个残差块的残差块架构，而第二个分支整合胎盘植入区域及相邻子宫浆膜层的解剖特征，以增强模型分类过程中的注意力。此外，我们采用了多任务学习策略以有效利用两个分支。在实际临床数据集上的实验表明，我们的模型达到了最先进的性能。 

---
# Alpay Algebra II: Identity as Fixed-Point Emergence in Categorical Data 

**Title (ZH)**: Alpay代数II：身份作为范畴数据中的稳定点 emergent 现象 

**Authors**: Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2505.17480)  

**Abstract**: In this second installment of the Alpay Algebra framework, I formally define identity as a fixed point that emerges through categorical recursion. Building upon the transfinite operator $\varphi^\infty$, I characterize identity as the universal solution to a self-referential functorial equation over a small cartesian closed category. I prove the existence and uniqueness of such identity-fixed-points via ordinal-indexed iteration, and interpret their convergence through internal categorical limits. Functors, adjunctions, and morphisms are reconstructed as dynamic traces of evolving states governed by $\varphi$, reframing identity not as a static label but as a stabilized process. Through formal theorems and symbolic flows, I show how these fixed points encode symbolic memory, recursive coherence, and semantic invariance. This paper positions identity as a mathematical structure that arises from within the logic of change itself computable, convergent, and categorically intrinsic. 

**Abstract (ZH)**: 在Alpay代数框架的第二部分中，我正式定义身份作为一种通过范畴递归产生的不变点。基于超越算子$\varphi^\infty$，我将身份刻画为在小范畴闭域上自参照函子方程的普遍解。通过序数索引迭代证明了这种不变点的存在性和唯一性，并通过内部范畴极限解释其收敛性。通过重新构建函子、伴随和同态，我将身份视为由$\varphi$支配的演化状态的动态痕迹，重新定义身份不是静态标签而是稳定过程。通过形式定理和符号流，我展示了这些不变点如何编码符号记忆、递归一致性和语义不变性。本文将身份定位为一种由变化逻辑本身产生、计算、收敛且范畴内在的数学结构。 

---
# Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions 

**Title (ZH)**: Twin-2K-500：基于超过500个问题的答案构建超过2000人数字孪生的数据集 

**Authors**: Olivier Toubia, George Z. Gui, Tianyi Peng, Daniel J. Merlau, Ang Li, Haozhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17479)  

**Abstract**: LLM-based digital twin simulation, where large language models are used to emulate individual human behavior, holds great promise for research in AI, social science, and digital experimentation. However, progress in this area has been hindered by the scarcity of real, individual-level datasets that are both large and publicly available. This lack of high-quality ground truth limits both the development and validation of digital twin methodologies. To address this gap, we introduce a large-scale, public dataset designed to capture a rich and holistic view of individual human behavior. We survey a representative sample of $N = 2,058$ participants (average 2.42 hours per person) in the US across four waves with 500 questions in total, covering a comprehensive battery of demographic, psychological, economic, personality, and cognitive measures, as well as replications of behavioral economics experiments and a pricing survey. The final wave repeats tasks from earlier waves to establish a test-retest accuracy baseline. Initial analyses suggest the data are of high quality and show promise for constructing digital twins that predict human behavior well at the individual and aggregate levels. By making the full dataset publicly available, we aim to establish a valuable testbed for the development and benchmarking of LLM-based persona simulations. Beyond LLM applications, due to its unique breadth and scale the dataset also enables broad social science research, including studies of cross-construct correlations and heterogeneous treatment effects. 

**Abstract (ZH)**: 基于LLM的数字孪生模拟：大规模公开数据集在人工智能、社会科学和数字实验中的应用前景 

---
# Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression 

**Title (ZH)**: 通过自回归同时 modeling 蛋白质构象和动态 

**Authors**: Yuning Shen, Lihao Wang, Huizhuo Yuan, Yan Wang, Bangji Yang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17478)  

**Abstract**: Understanding protein dynamics is critical for elucidating their biological functions. The increasing availability of molecular dynamics (MD) data enables the training of deep generative models to efficiently explore the conformational space of proteins. However, existing approaches either fail to explicitly capture the temporal dependencies between conformations or do not support direct generation of time-independent samples. To address these limitations, we introduce ConfRover, an autoregressive model that simultaneously learns protein conformation and dynamics from MD trajectories, supporting both time-dependent and time-independent sampling. At the core of our model is a modular architecture comprising: (i) an encoding layer, adapted from protein folding models, that embeds protein-specific information and conformation at each time frame into a latent space; (ii) a temporal module, a sequence model that captures conformational dynamics across frames; and (iii) an SE(3) diffusion model as the structure decoder, generating conformations in continuous space. Experiments on ATLAS, a large-scale protein MD dataset of diverse structures, demonstrate the effectiveness of our model in learning conformational dynamics and supporting a wide range of downstream tasks. ConfRover is the first model to sample both protein conformations and trajectories within a single framework, offering a novel and flexible approach for learning from protein MD data. 

**Abstract (ZH)**: 理解蛋白质动力学对于阐明其生物学功能至关重要。不断增加的分子动力学（MD）数据使得训练深度生成模型以高效探索蛋白质构象空间成为可能。然而，现有的方法要么不能明确捕捉构象之间的时序依赖性，要么不支持直接生成时间独立样本。为解决这些问题，我们引入了ConfRover，这是一种自回归模型，能够同时从MD轨迹中学习蛋白质构象和动力学，支持时间和时间独立采样。我们的模型核心由以下模块构成：（i）一个编码层，源自蛋白质折叠模型，将每时刻的蛋白质特定信息和构象嵌入到潜在空间中；（ii）一个时序模块，一种序列模型，捕捉帧间构象动力学；以及（iii）一个SE(3)扩散模型作为结构解码器，在连续空间中生成构象。在ATLAS大规模蛋白质MD数据集上进行的实验表明，我们的模型在学习构象动力学和支持广泛下游任务方面具有有效性。ConfRover是首个在同一框架中采样蛋白质构象和轨迹的模型，提供了学习蛋白质MD数据的一种新颖且灵活的方法。 

---
# OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics 

**Title (ZH)**: OrionBench: 信息图中图表和人类可识别对象检测基准 

**Authors**: Jiangning Zhu, Yuxing Zhou, Zheng Wang, Juntao Yao, Yima Gu, Yuhui Yuan, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17473)  

**Abstract**: Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection. 

**Abstract (ZH)**: 基于图表的理解：OrionBench——用于信息图中图表和人类可识别对象的准确目标检测基准 

---
# SLearnLLM: A Self-Learning Framework for Efficient Domain-Specific Adaptation of Large Language Models 

**Title (ZH)**: SLearnLLM：一种高效领域特定适应的大语言模型自学习框架 

**Authors**: Xiang Liu, Zhaoxiang Liu, Peng Wang, Kohou Wang, Huan Hu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.17470)  

**Abstract**: When using supervised fine-tuning (SFT) to adapt large language models (LLMs) to specific domains, a significant challenge arises: should we use the entire SFT dataset for fine-tuning? Common practice often involves fine-tuning directly on the entire dataset due to limited information on the LLM's past training data. However, if the SFT dataset largely overlaps with the model's existing knowledge, the performance gains are minimal, leading to wasted computational resources. Identifying the unknown knowledge within the SFT dataset and using it to fine-tune the model could substantially improve the training efficiency. To address this challenge, we propose a self-learning framework for LLMs inspired by human learning pattern. This framework takes a fine-tuning (SFT) dataset in a specific domain as input. First, the LLMs answer the questions in the SFT dataset. The LLMs then objectively grade the responses and filter out the incorrectly answered QA pairs. Finally, we fine-tune the LLMs based on this filtered QA set. Experimental results in the fields of agriculture and medicine demonstrate that our method substantially reduces training time while achieving comparable improvements to those attained with full dataset fine-tuning. By concentrating on the unknown knowledge within the SFT dataset, our approach enhances the efficiency of fine-tuning LLMs. 

**Abstract (ZH)**: 在特定领域使用监督微调（SFT）适配大型语言模型（LLMs）时，一个显著挑战是：是否应使用整个SFT数据集进行微调？ 

---
# Efficient compression of neural networks and datasets 

**Title (ZH)**: 神经网络和数据集的高效压缩 

**Authors**: Lukas Silvester Barth, Paulo von Petersenn  

**Link**: [PDF](https://arxiv.org/pdf/2505.17469)  

**Abstract**: We compare, improve, and contribute methods that substantially decrease the number of parameters of neural networks while maintaining high test accuracy. When applying our methods to minimize description length, we obtain very effective data compression algorithms. In particular, we develop a probabilistic reformulation of $\ell_0$ regularized optimization for nonlinear models that does not require Monte-Carlo sampling and thus improves upon previous methods. We also improve upon methods involving smooth approximations to the $\ell_0$ norm, and investigate layerwise methods. We compare the methods on different architectures and datasets, including convolutional networks trained on image datasets and transformers trained on parts of Wikipedia. We also created a synthetic teacher-student setup to investigate compression in a controlled continuous setting. Finally, we conceptually relate compression algorithms to Solomonoff's theory of inductive inference and empirically verify the prediction that regularized models can exhibit more sample-efficient convergence. 

**Abstract (ZH)**: 我们比较、改进并贡献了一类显著减少神经网络参数数量同时保持高测试准确率的方法。在应用这些方法以最小化描述长度时，我们得到了非常有效的数据压缩算法。特别是，我们开发了一种无需蒙特卡洛采样的$\ell_0$正则化优化的概率重构方法，从而改善了先前的方法。我们还改善了涉及$\ell_0$范数光滑近似的方法，并研究了逐层方法。我们在不同的架构和数据集上比较了这些方法，包括在图像数据集上训练的卷积网络和在维基百科部分数据上训练的变压器。我们还创建了一个合成的教-学设置，以在受控连续环境中研究压缩。最后，我们从概念上将压缩算法与索洛门off归纳推断理论相关联，并通过实验验证了正则化模型可以表现出更高样本效率收敛的预测。 

---
# Graph Mamba for Efficient Whole Slide Image Understanding 

**Title (ZH)**: Graph Mamba用于高效全量组织切片图像理解 

**Authors**: Jiaxuan Lu, Junyan Shi, Yuhui Lin, Fang Yan, Yue Gao, Shaoting Zhang, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17457)  

**Abstract**: Whole Slide Images (WSIs) in histopathology present a significant challenge for large-scale medical image analysis due to their high resolution, large size, and complex tile relationships. Existing Multiple Instance Learning (MIL) methods, such as Graph Neural Networks (GNNs) and Transformer-based models, face limitations in scalability and computational cost. To bridge this gap, we propose the WSI-GMamba framework, which synergistically combines the relational modeling strengths of GNNs with the efficiency of Mamba, the State Space Model designed for sequence learning. The proposed GMamba block integrates Message Passing, Graph Scanning & Flattening, and feature aggregation via a Bidirectional State Space Model (Bi-SSM), achieving Transformer-level performance with 7* fewer FLOPs. By leveraging the complementary strengths of lightweight GNNs and Mamba, the WSI-GMamba framework delivers a scalable solution for large-scale WSI analysis, offering both high accuracy and computational efficiency for slide-level classification. 

**Abstract (ZH)**: WSI-GMamba框架在组织病理学大尺度医疗图像分析中的应用：结合GNNs的关系建模优势和Mamba的空间模型效率 

---
# Towards Evaluating Proactive Risk Awareness of Multimodal Language Models 

**Title (ZH)**: Towards 评估多模态语言模型的主动风险意识 

**Authors**: Youliang Yuan, Wenxiang Jiao, Yuejin Xie, Chihao Shen, Menghan Tian, Wenxuan Wang, Jen-tse Huang, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2505.17455)  

**Abstract**: Human safety awareness gaps often prevent the timely recognition of everyday risks. In solving this problem, a proactive safety artificial intelligence (AI) system would work better than a reactive one. Instead of just reacting to users' questions, it would actively watch people's behavior and their environment to detect potential dangers in advance. Our Proactive Safety Bench (PaSBench) evaluates this capability through 416 multimodal scenarios (128 image sequences, 288 text logs) spanning 5 safety-critical domains. Evaluation of 36 advanced models reveals fundamental limitations: Top performers like Gemini-2.5-pro achieve 71% image and 64% text accuracy, but miss 45-55% risks in repeated trials. Through failure analysis, we identify unstable proactive reasoning rather than knowledge deficits as the primary limitation. This work establishes (1) a proactive safety benchmark, (2) systematic evidence of model limitations, and (3) critical directions for developing reliable protective AI. We believe our dataset and findings can promote the development of safer AI assistants that actively prevent harm rather than merely respond to requests. Our dataset can be found at this https URL. 

**Abstract (ZH)**: 人类安全意识差距往往阻碍了对日常风险的及时识别。为解决这一问题，一种主动的安全人工智能系统将比被动系统更有效。它不仅会对用户的问题作出反应，还会积极观察人们的行為及其环境，提前检测潜在危险。我们的主动安全基准（PaSBench）通过416个多模态场景（128个图像序列，288个文本日志），覆盖5个安全关键领域来评估这一能力。对36个先进模型的评估揭示了根本性限制：如Gemini-2.5-pro这类顶尖模型在图像上的准确率为71%，文本上的准确率为64%，但在重复试验中仍会错过45-55%的风险。通过失败分析，我们认定不稳定的前提推断而非知识缺口是主要限制。本研究建立了（1）一个主动安全基准，（2）系统的模型限制证据，以及（3）开发可靠保护性人工智能的关键方向。我们相信，我们的数据集和发现可以促进开发更加安全的人工智能助手，这些助手能积极预防伤害而非仅仅是响应请求。我们的数据集可以在以下网址找到：this https URL。 

---
# CLIMB: Class-imbalanced Learning Benchmark on Tabular Data 

**Title (ZH)**: CLIMB: 表格数据上的类别不平衡学习基准 

**Authors**: Zhining Liu, Zihao Li, Ze Yang, Tianxin Wei, Jian Kang, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17451)  

**Abstract**: Class-imbalanced learning (CIL) on tabular data is important in many real-world applications where the minority class holds the critical but rare outcomes. In this paper, we present CLIMB, a comprehensive benchmark for class-imbalanced learning on tabular data. CLIMB includes 73 real-world datasets across diverse domains and imbalance levels, along with unified implementations of 29 representative CIL algorithms. Built on a high-quality open-source Python package with unified API designs, detailed documentation, and rigorous code quality controls, CLIMB supports easy implementation and comparison between different CIL algorithms. Through extensive experiments, we provide practical insights on method accuracy and efficiency, highlighting the limitations of naive rebalancing, the effectiveness of ensembles, and the importance of data quality. Our code, documentation, and examples are available at this https URL. 

**Abstract (ZH)**: 表格数据中的类别不平衡学习（CIL）在许多实际应用中非常重要，其中少数类包含了关键但稀有的结果。本文介绍了CLIMB，一个全面的表格数据类别不平衡学习基准。CLIMB包含来自不同领域和不同不平衡程度的73个真实数据集，以及29个代表性CIL算法的一致实现。基于高质量的开源Python包，具有统一的API设计、详细的文档和严格的代码质量控制，CLIMB支持不同CIL算法的简便实现与比较。通过广泛的实验，我们提供了关于方法准确性和效率的实用见解，强调了天真重新平衡的局限性、集成的有效性以及数据质量的重要性。我们的代码、文档和示例可在以下链接获取。 

---
# Discovering Forbidden Topics in Language Models 

**Title (ZH)**: 发现语言模型中的禁忌话题 

**Authors**: Can Rager, Chris Wendler, Rohit Gandikota, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2505.17441)  

**Abstract**: Refusal discovery is the task of identifying the full set of topics that a language model refuses to discuss. We introduce this new problem setting and develop a refusal discovery method, LLM-crawler, that uses token prefilling to find forbidden topics. We benchmark the LLM-crawler on Tulu-3-8B, an open-source model with public safety tuning data. Our crawler manages to retrieve 31 out of 36 topics within a budget of 1000 prompts. Next, we scale the crawl to a frontier model using the prefilling option of Claude-Haiku. Finally, we crawl three widely used open-weight models: Llama-3.3-70B and two of its variants finetuned for reasoning: DeepSeek-R1-70B and Perplexity-R1-1776-70B. DeepSeek-R1-70B reveals patterns consistent with censorship tuning: The model exhibits "thought suppression" behavior that indicates memorization of CCP-aligned responses. Although Perplexity-R1-1776-70B is robust to censorship, LLM-crawler elicits CCP-aligned refusals answers in the quantized model. Our findings highlight the critical need for refusal discovery methods to detect biases, boundaries, and alignment failures of AI systems. 

**Abstract (ZH)**: 拒绝发现是识别语言模型拒绝讨论的完整话题集的任务。我们引入了这一新的问题设置，并开发了一种拒绝发现方法——LLM-crawler，该方法使用标记预填充来查找禁忌话题。我们在Tulu-3-8B上 benchmarked LLM-crawler，Tulu-3-8B是一个带有公共安全调优数据的开源模型。我们的爬虫在预算为1000个提示的情况下成功检索到31个主题中的36个。随后，我们使用Claude-Haiku的预填充选项将爬虫扩展到前沿模型。最后，我们爬取了三个广泛使用的开源模型：Llama-3.3-70B及其两个针对推理微调的变体：DeepSeek-R1-70B和Perplexity-R1-1776-70B。DeepSeek-R1-70B揭示了与审查调优一致的模式：该模型表现出“思想压制”行为，表明记住了与中共党派一致的回应。虽然Perplexity-R1-1776-70B对审查具有韧性，但LLM-crawler在量化模型中引发了与中共党派一致的拒绝回答。我们的研究结果强调了拒绝发现方法检测AI系统偏差、边界和对齐失败的迫切需求。 

---
# Designing an efficient and equitable humanitarian supply chain dynamically via reinforcement learning 

**Title (ZH)**: 基于强化学习的高效和公平的人道主义供应链动态设计 

**Authors**: Weijia Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17439)  

**Abstract**: This study designs an efficient and equitable humanitarian supply chain dynamically by using reinforcement learning, PPO, and compared with heuristic algorithms. This study demonstrates the model of PPO always treats average satisfaction rate as the priority. 

**Abstract (ZH)**: 本研究利用强化学习PPO设计了一个高效公平的人道主义供应链，并与启发式算法进行了比较，表明PPO模型始终以平均满意度率为优先。 

---
# Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision 

**Title (ZH)**: 从全景语义监督中学习通用和灵活的轨迹模型 

**Authors**: Yuanshao Zhu, James Jianqiao Yu, Xiangyu Zhao, Xiao Han, Qidong Liu, Xuetao Wei, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17437)  

**Abstract**: The widespread adoption of mobile devices and data collection technologies has led to an exponential increase in trajectory data, presenting significant challenges in spatio-temporal data mining, particularly for efficient and accurate trajectory retrieval. However, existing methods for trajectory retrieval face notable limitations, including inefficiencies in large-scale data, lack of support for condition-based queries, and reliance on trajectory similarity measures. To address the above challenges, we propose OmniTraj, a generalized and flexible omni-semantic trajectory retrieval framework that integrates four complementary modalities or semantics -- raw trajectories, topology, road segments, and regions -- into a unified system. Unlike traditional approaches that are limited to computing and processing trajectories as a single modality, OmniTraj designs dedicated encoders for each modality, which are embedded and fused into a shared representation space. This design enables OmniTraj to support accurate and flexible queries based on any individual modality or combination thereof, overcoming the rigidity of traditional similarity-based methods. Extensive experiments on two real-world datasets demonstrate the effectiveness of OmniTraj in handling large-scale data, providing flexible, multi-modality queries, and supporting downstream tasks and applications. 

**Abstract (ZH)**: 移动设备和数据收集技术的广泛采用导致轨迹数据呈指数级增长，为时空数据挖掘带来了重大挑战，特别是在高效和准确的轨迹检索方面。现有轨迹检索方法存在显著局限性，包括大规模数据处理效率低下、不支持基于条件的查询以及依赖于轨迹相似度度量。为应对上述挑战，我们提出了一种名为OmniTraj的通用且灵活的全域语义轨迹检索框架，将原始轨迹、拓扑结构、道路段和区域四种互补模态或语义整合到一个统一系统中。与传统方法只能将轨迹作为单一模态进行计算和处理不同，OmniTraj为每种模态设计了专用编码器，并将其嵌入到共享表示空间中。这种设计使OmniTraj能够基于任何单独模态或其组合支持准确和灵活的查询，克服了传统基于相似性的方法的僵化性。在两个真实世界数据集上的广泛实验表明，OmniTraj在处理大规模数据、提供灵活的多模态查询以及支持下游任务和应用方面具有有效性。 

---
# Dynamic Manipulation of Deformable Objects in 3D: Simulation, Benchmark and Learning Strategy 

**Title (ZH)**: 三维可变形对象的动态操控：仿真、基准测试与学习策略 

**Authors**: Guanzhou Lan, Yuqi Yang, Anup Teejo Mathew, Feiping Nie, Rong Wang, Xuelong Li, Federico Renda, Bin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17434)  

**Abstract**: Goal-conditioned dynamic manipulation is inherently challenging due to complex system dynamics and stringent task constraints, particularly in deformable object scenarios characterized by high degrees of freedom and underactuation. Prior methods often simplify the problem to low-speed or 2D settings, limiting their applicability to real-world 3D tasks. In this work, we explore 3D goal-conditioned rope manipulation as a representative challenge. To mitigate data scarcity, we introduce a novel simulation framework and benchmark grounded in reduced-order dynamics, which enables compact state representation and facilitates efficient policy learning. Building on this, we propose Dynamics Informed Diffusion Policy (DIDP), a framework that integrates imitation pretraining with physics-informed test-time adaptation. First, we design a diffusion policy that learns inverse dynamics within the reduced-order space, enabling imitation learning to move beyond naïve data fitting and capture the underlying physical structure. Second, we propose a physics-informed test-time adaptation scheme that imposes kinematic boundary conditions and structured dynamics priors on the diffusion process, ensuring consistency and reliability in manipulation execution. Extensive experiments validate the proposed approach, demonstrating strong performance in terms of accuracy and robustness in the learned policy. 

**Abstract (ZH)**: 基于动力学指导的三维绳索操纵挑战与解决方案：克服数据稀缺性的方法及动力学知情扩散策略 

---
# SEvoBench : A C++ Framework For Evolutionary Single-Objective Optimization Benchmarking 

**Title (ZH)**: SEvoBench : 一个基于C++的单一目标进化优化基准测试框架 

**Authors**: Yongkang Yang, Jian Zhao, Tengfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17430)  

**Abstract**: We present SEvoBench, a modern C++ framework for evolutionary computation (EC), specifically designed to systematically benchmark evolutionary single-objective optimization algorithms. The framework features modular implementations of Particle Swarm Optimization (PSO) and Differential Evolution (DE) algorithms, organized around three core components: (1) algorithm construction with reusable modules, (2) efficient benchmark problem suites, and (3) parallel experimental analysis. Experimental evaluations demonstrate the framework's superior performance in benchmark testing and algorithm comparison. Case studies further validate its capabilities in algorithm hybridization and parameter analysis. Compared to existing frameworks, SEvoBench demonstrates three key advantages: (i) highly efficient and reusable modular implementations of PSO and DE algorithms, (ii) accelerated benchmarking through parallel execution, and (iii) enhanced computational efficiency via SIMD (Single Instruction Multiple Data) vectorization for large-scale problems. 

**Abstract (ZH)**: SEvoBench：一种面向进化的现代C++框架，用于系统性基准测试单目标优化算法 

---
# UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information 

**Title (ZH)**: UniTTS：无需拆分声学和语义信息的端到端TTS系统 

**Authors**: Rui Wang, Qianguo Sun, Tianrong Chen, Zhiyun Zeng, Junlong Wu, Jiaxing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17426)  

**Abstract**: The emergence of multi-codebook neutral audio codecs such as Residual Vector Quantization (RVQ) and Group Vector Quantization (GVQ) has significantly advanced Large-Language-Model (LLM) based Text-to-Speech (TTS) systems. These codecs are crucial in separating semantic and acoustic information while efficiently harnessing semantic priors. However, since semantic and acoustic information cannot be fully aligned, a significant drawback of these methods when applied to LLM-based TTS is that large language models may have limited access to comprehensive audio information. To address this limitation, we propose DistilCodec and UniTTS, which collectively offer the following advantages: 1) This method can distill a multi-codebook audio codec into a single-codebook audio codec with 32,768 codes while achieving a near 100\% utilization. 2) As DistilCodec does not employ a semantic alignment scheme, a large amount of high-quality unlabeled audio (such as audiobooks with sound effects, songs, etc.) can be incorporated during training, further expanding data diversity and broadening its applicability. 3) Leveraging the comprehensive audio information modeling of DistilCodec, we integrated three key tasks into UniTTS's pre-training framework: audio modality autoregression, text modality autoregression, and speech-text cross-modal autoregression. This allows UniTTS to accept interleaved text and speech/audio prompts while substantially preserving LLM's text capabilities. 4) UniTTS employs a three-stage training process: Pre-Training, Supervised Fine-Tuning (SFT), and Alignment. Source code and model checkpoints are publicly available at this https URL and this https URL. 

**Abstract (ZH)**: 多码本中性音频编解码器（如残差向量量化（RVQ）和组向量量化（GVQ））的出现显著推进了基于大型语言模型（LLM）的文本到语音（TTS）系统。这些编解码器在分离语义和声学信息的同时，高效利用了语义先验。然而，由于语义和声学信息无法完全对齐，这些方法在应用于基于LLM的TTS时的一个重要缺点是大型语言模型可能无法充分访问全面的音频信息。为了解决这一局限性，我们提出了DistilCodec和UniTTS，它们共同提供了以下优势：1）该方法可以将多码本音频编解码器精简为包含32768个码本的单码本音频编解码器，同时实现接近100%的利用率。2）由于DistilCodec不采用语义对齐方案，可以在训练过程中大量融入高质量的未标注音频（如带音效的声音书籍、歌曲等），进一步增加数据多样性并拓宽适用范围。3）利用DistilCodec全面的音频信息模型，我们将三个关键任务集成到UniTTS的预训练框架中：音频模态自回归、文本模态自回归和语音-文本跨模态自回归。这使得UniTTS能够接受交错的文本和语音/音频提示，同时在很大程度上保留LLM的文本能力。4）UniTTS采用三阶段训练过程：预训练、监督微调（SFT）和对齐过程。源代码和模型检查点可在以下网址公开访问：这个 https URL 和这个 https URL。 

---
# Wildfire Detection Using Vision Transformer with the Wildfire Dataset 

**Title (ZH)**: 使用视觉变换器进行 wildfires 检测：基于 wildfires 数据集的方法 

**Authors**: Gowtham Raj Vuppari, Navarun Gupta, Ahmed El-Sayed, Xingguo Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17395)  

**Abstract**: The critical need for sophisticated detection techniques has been highlighted by the rising frequency and intensity of wildfires in the US, especially in California. In 2023, wildfires caused 130 deaths nationwide, the highest since 1990. In January 2025, Los Angeles wildfires which included the Palisades and Eaton fires burnt approximately 40,000 acres and 12,000 buildings, and caused loss of human lives. The devastation underscores the urgent need for effective detection and prevention strategies. Deep learning models, such as Vision Transformers (ViTs), can enhance early detection by processing complex image data with high accuracy. However, wildfire detection faces challenges, including the availability of high-quality, real-time data. Wildfires often occur in remote areas with limited sensor coverage, and environmental factors like smoke and cloud cover can hinder detection. Additionally, training deep learning models is computationally expensive, and issues like false positives/negatives and scaling remain concerns. Integrating detection systems with real-time alert mechanisms also poses difficulties. In this work, we used the wildfire dataset consisting of 10.74 GB high-resolution images categorized into 'fire' and 'nofire' classes is used for training the ViT model. To prepare the data, images are resized to 224 x 224 pixels, converted into tensor format, and normalized using ImageNet statistics. 

**Abstract (ZH)**: 美国尤其是加利福尼亚州频发且强度增加的野火凸显了高级检测技术的迫切需求。2023年，野火在全国造成130人死亡，为1990年以来最高。2025年1月，洛杉矶地区的帕利塞德斯和艾顿野火烧毁了约40,000英亩土地和12,000栋建筑，造成了人员伤亡。这些灾害突显了有效检测和预防策略的迫切需求。深度学习模型，如视觉变换器（ViTs），可以通过高精度处理复杂图像数据来提升早期检测能力。然而，野火检测面临挑战，包括高质量实时数据的可用性限制。野火经常发生在传感器覆盖有限的偏远地区，烟雾和云层覆盖等环境因素会妨碍检测。此外，训练深度学习模型计算成本高昂，误报/漏报和扩展性等问题依然存在。将检测系统与实时警报机制结合也颇具挑战性。在这项工作中，我们使用了一个包含10.74 GB高分辨率图像的数据集，这些图像被分为“火灾”和“非火灾”两类用于训练ViT模型。为了准备数据，图像被调整为224 x 224像素，转换为张量格式，并使用ImageNet统计数据进行了标准化。 

---
# Dual-sensing driving detection model 

**Title (ZH)**: 双重传感驾驶检测模型 

**Authors**: Leon C.C.K, Zeng Hui  

**Link**: [PDF](https://arxiv.org/pdf/2505.17392)  

**Abstract**: In this paper, a novel dual-sensing driver fatigue detection method combining computer vision and physiological signal analysis is proposed. The system exploits the complementary advantages of the two sensing modalities and breaks through the limitations of existing single-modality methods. We introduce an innovative architecture that combines real-time facial feature analysis with physiological signal processing, combined with advanced fusion strategies, for robust fatigue detection. The system is designed to run efficiently on existing hardware while maintaining high accuracy and reliability. Through comprehensive experiments, we demonstrate that our method outperforms traditional methods in both controlled environments and real-world conditions, while maintaining high accuracy. The practical applicability of the system has been verified through extensive tests in various driving scenarios and shows great potential in reducing fatigue-related accidents. This study contributes to the field by providing a more reliable, cost-effective, and humane solution for driver fatigue detection. 

**Abstract (ZH)**: 一种结合计算机视觉和生理信号分析的新型双传感驾驶员疲劳检测方法 

---
# Bootstrapping Imitation Learning for Long-horizon Manipulation via Hierarchical Data Collection Space 

**Title (ZH)**: 基于分层数据收集空间的长时滞操作imitation learning自举方法 

**Authors**: Jinrong Yang, Kexun Chen, Zhuoling Li, Shengkai Wu, Yong Zhao, Liangliang Ren, Wenqiu Luo, Chaohui Shang, Meiyu Zhi, Linfeng Gao, Mingshan Sun, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17389)  

**Abstract**: Imitation learning (IL) with human demonstrations is a promising method for robotic manipulation tasks. While minimal demonstrations enable robotic action execution, achieving high success rates and generalization requires high cost, e.g., continuously adding data or incrementally conducting human-in-loop processes with complex hardware/software systems. In this paper, we rethink the state/action space of the data collection pipeline as well as the underlying factors responsible for the prediction of non-robust actions. To this end, we introduce a Hierarchical Data Collection Space (HD-Space) for robotic imitation learning, a simple data collection scheme, endowing the model to train with proactive and high-quality data. Specifically, We segment the fine manipulation task into multiple key atomic tasks from a high-level perspective and design atomic state/action spaces for human demonstrations, aiming to generate robust IL data. We conduct empirical evaluations across two simulated and five real-world long-horizon manipulation tasks and demonstrate that IL policy training with HD-Space-based data can achieve significantly enhanced policy performance. HD-Space allows the use of a small amount of demonstration data to train a more powerful policy, particularly for long-horizon manipulation tasks. We aim for HD-Space to offer insights into optimizing data quality and guiding data scaling. project page: this https URL. 

**Abstract (ZH)**: 基于人类示范的层次化数据收集空间的机器人模仿学习 

---
# Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition 

**Title (ZH)**: 可验证高效算法：在线主要方-代理人信息获取中最佳评标规则识别 

**Authors**: Zichen Wang, Chuanhao Li, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17379)  

**Abstract**: We investigate the problem of identifying the optimal scoring rule within the principal-agent framework for online information acquisition problem. We focus on the principal's perspective, seeking to determine the desired scoring rule through interactions with the agent. To address this challenge, we propose two algorithms: OIAFC and OIAFB, tailored for fixed confidence and fixed budget settings, respectively. Our theoretical analysis demonstrates that OIAFC can extract the desired $(\epsilon, \delta)$-scoring rule with a efficient instance-dependent sample complexity or an instance-independent sample complexity. Our analysis also shows that OIAFB matches the instance-independent performance bound of OIAFC, while both algorithms share the same complexity across fixed confidence and fixed budget settings. 

**Abstract (ZH)**: 我们研究了在主要-代理人框架下在线信息获取问题中识别最优评分规则的问题。我们从主要方的角度出发，通过与代理人的交互确定所需的评分规则。为了解决这一挑战，我们提出了两种算法：OIAFC和OIAFB，分别针对固定的置信度和固定的预算设置。我们的理论分析表明，OIAFC可以在有效的实例相关或实例无关样本复杂性下提取所需的$(\epsilon, \delta)$-评分规则。我们的分析还表明，OIAFB在实例无关性能方面与OIAFC匹配，而两个算法在固定置信度和固定预算设置下的复杂性相同。 

---
# Value-Guided Search for Efficient Chain-of-Thought Reasoning 

**Title (ZH)**: 价值导向的搜索以实现高效的心智推理 

**Authors**: Kaiwen Wang, Jin Peng Zhou, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kianté Brantley, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17373)  

**Abstract**: In this paper, we propose a simple and efficient method for value model training on long-context reasoning traces. Compared to existing process reward models (PRMs), our method does not require a fine-grained notion of "step," which is difficult to define for long-context reasoning models. By collecting a dataset of 2.5 million reasoning traces, we train a 1.5B token-level value model and apply it to DeepSeek models for improved performance with test-time compute scaling. We find that block-wise value-guided search (VGS) with a final weighted majority vote achieves better test-time scaling than standard methods such as majority voting or best-of-n. With an inference budget of 64 generations, VGS with DeepSeek-R1-Distill-1.5B achieves an average accuracy of 45.7% across four competition math benchmarks (AIME 2024 & 2025, HMMT Feb 2024 & 2025), reaching parity with o3-mini-medium. Moreover, VGS significantly reduces the inference FLOPs required to achieve the same performance of majority voting. Our dataset, model and codebase are open-sourced. 

**Abstract (ZH)**: 本文提出了一种简单高效的方法，用于在长上下文推理轨迹上进行价值模型训练。与现有的过程奖励模型相比，我们的方法不需要精细的“步骤”概念，这在长上下文推理模型中难以定义。通过收集包含250万条推理轨迹的数据集，我们训练了一种1.5B词元级别价值模型，并将其应用于DeepSeek模型，以实现测试时计算量扩展下的性能提升。我们发现，块级价值引导搜索（VGS）结合加权多数投票在测试时扩展性能上优于标准方法（如简单多数投票或Best-of-n）。在64个推理生成预算下，DeepSeek-R1-Distill-1.5B结合VGS实现了四个竞赛数学基准（AIME 2024 & 2025，HMMT Feb 2024 & 2025）的平均准确率为45.7%，达到与o3-mini-medium相当的性能。此外，VGS大幅减少了达到相同性能所需的推理FLOPs。我们的数据集、模型和代码库已开源。 

---
# FRIREN: Beyond Trajectories -- A Spectral Lens on Time 

**Title (ZH)**: FRIREN：超越轨迹——时间的频谱视角 

**Authors**: Qilin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17370)  

**Abstract**: Long-term time-series forecasting (LTSF) models are often presented as general-purpose solutions that can be applied across domains, implicitly assuming that all data is pointwise predictable. Using chaotic systems such as Lorenz-63 as a case study, we argue that geometric structure - not pointwise prediction - is the right abstraction for a dynamic-agnostic foundational model. Minimizing the Wasserstein-2 distance (W2), which captures geometric changes, and providing a spectral view of dynamics are essential for long-horizon forecasting. Our model, FRIREN (Flow-inspired Representations via Interpretable Eigen-networks), implements an augmented normalizing-flow block that embeds data into a normally distributed latent representation. It then generates a W2-efficient optimal path that can be decomposed into rotation, scaling, inverse rotation, and translation. This architecture yields locally generated, geometry-preserving predictions that are independent of the underlying dynamics, and a global spectral representation that functions as a finite Koopman operator with a small modification. This enables practitioners to identify which modes grow, decay, or oscillate, both locally and system-wide. FRIREN achieves an MSE of 11.4, MAE of 1.6, and SWD of 0.96 on Lorenz-63 in a 336-in, 336-out, dt=0.01 setting, surpassing TimeMixer (MSE 27.3, MAE 2.8, SWD 2.1). The model maintains effective prediction for 274 out of 336 steps, approximately 2.5 Lyapunov times. On Rossler (96-in, 336-out), FRIREN achieves an MSE of 0.0349, MAE of 0.0953, and SWD of 0.0170, outperforming TimeMixer's MSE of 4.3988, MAE of 0.886, and SWD of 3.2065. FRIREN is also competitive on standard LTSF datasets such as ETT and Weather. By connecting modern generative flows with classical spectral analysis, FRIREN makes long-term forecasting both accurate and interpretable, setting a new benchmark for LTSF model design. 

**Abstract (ZH)**: 基于流的代表通过可解释的本征网络（FRIREN）：混沌系统中的长期时间序列forecasting 

---
# EVM-Fusion: An Explainable Vision Mamba Architecture with Neural Algorithmic Fusion 

**Title (ZH)**: EVM-Fusion: 可解释的Vision Mamba 架构与神经算法融合 

**Authors**: Zichuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17367)  

**Abstract**: Medical image classification is critical for clinical decision-making, yet demands for accuracy, interpretability, and generalizability remain challenging. This paper introduces EVM-Fusion, an Explainable Vision Mamba architecture featuring a novel Neural Algorithmic Fusion (NAF) mechanism for multi-organ medical image classification. EVM-Fusion leverages a multipath design, where DenseNet and U-Net based pathways, enhanced by Vision Mamba (Vim) modules, operate in parallel with a traditional feature pathway. These diverse features are dynamically integrated via a two-stage fusion process: cross-modal attention followed by the iterative NAF block, which learns an adaptive fusion algorithm. Intrinsic explainability is embedded through path-specific spatial attention, Vim {\Delta}-value maps, traditional feature SE-attention, and cross-modal attention weights. Experiments on a diverse 9-class multi-organ medical image dataset demonstrate EVM-Fusion's strong classification performance, achieving 99.75% test accuracy and provide multi-faceted insights into its decision-making process, highlighting its potential for trustworthy AI in medical diagnostics. 

**Abstract (ZH)**: 可解释的Vision Mamba架构EVM-Fusion在多器官医学图像分类中的应用 

---
# A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit 

**Title (ZH)**: 一个完全生成的动机 Interviewing 指导员聊天机器人，帮助吸烟者迈向戒烟决定 

**Authors**: Zafarullah Mahmood, Soliman Ali, Jiading Zhu, Mohamed Abdelwahab, Michelle Yu Collins, Sihan Chen, Yi Cheng Zhao, Jodi Wolff, Osnat Melamed, Nadia Minian, Marta Maslej, Carolynne Cooper, Matt Ratto, Peter Selby, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.17362)  

**Abstract**: The conversational capabilities of Large Language Models (LLMs) suggest that they may be able to perform as automated talk therapists. It is crucial to know if these systems would be effective and adhere to known standards. We present a counsellor chatbot that focuses on motivating tobacco smokers to quit smoking. It uses a state-of-the-art LLM and a widely applied therapeutic approach called Motivational Interviewing (MI), and was evolved in collaboration with clinician-scientists with expertise in MI. We also describe and validate an automated assessment of both the chatbot's adherence to MI and client responses. The chatbot was tested on 106 participants, and their confidence that they could succeed in quitting smoking was measured before the conversation and one week later. Participants' confidence increased by an average of 1.7 on a 0-10 scale. The automated assessment of the chatbot showed adherence to MI standards in 98% of utterances, higher than human counsellors. The chatbot scored well on a participant-reported metric of perceived empathy but lower than typical human counsellors. Furthermore, participants' language indicated a good level of motivation to change, a key goal in MI. These results suggest that the automation of talk therapy with a modern LLM has promise. 

**Abstract (ZH)**: 大型语言模型的对话能力暗示它们可能能够担任自动化谈话治疗师。了解这些系统的效果并符合已知标准至关重要。我们介绍了一个侧重于激励烟草吸烟者戒烟的辅导员聊天机器人。该聊天机器人使用了最先进的大型语言模型以及广泛应用于治疗领域的动机访谈（MI）方法，并与具有MI专长的临床科学家合作进行了开发。我们还描述并验证了一个自动评估聊天机器人MI遵守程度及客户端回应的方法。该聊天机器人在106名参与者中进行了测试，测量了他们在对话前及一周后的戒烟信心评分。参与者的信心平均提高了1.7分（在0-10分的量表上）。自动评估结果显示，聊天机器人在98%的陈述中符合MI标准，高于人类辅导员。聊天机器人在用户体验的同理心感知的参与者报告指标上表现良好，但低于典型的人类辅导员。此外，参与者的语言表明了较高的改变认同度，这是MI的关键目标。这些结果表明，使用现代大型语言模型自动化谈话疗法充满希望。 

---
# Dual Ascent Diffusion for Inverse Problems 

**Title (ZH)**: 双重上升扩散算法求解逆问题 

**Authors**: Minseo Kim, Axel Levy, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.17353)  

**Abstract**: Ill-posed inverse problems are fundamental in many domains, ranging from astrophysics to medical imaging. Emerging diffusion models provide a powerful prior for solving these problems. Existing maximum-a-posteriori (MAP) or posterior sampling approaches, however, rely on different computational approximations, leading to inaccurate or suboptimal samples. To address this issue, we introduce a new approach to solving MAP problems with diffusion model priors using a dual ascent optimization framework. Our framework achieves better image quality as measured by various metrics for image restoration problems, it is more robust to high levels of measurement noise, it is faster, and it estimates solutions that represent the observations more faithfully than the state of the art. 

**Abstract (ZH)**: 不适定逆问题在天体物理和医学成像等领域至关重要。新兴的扩散模型提供了有效的先验方法来解决这些问题。现有的最大后验概率（MAP）或后验采样方法依赖于不同的计算近似，导致不准确或次优的结果。为了解决这一问题，我们提出了一种基于对偶上升优化框架的求解具有扩散模型先验的MAP问题的新方法。该框架在图像恢复问题中能够获得更好的图像质量，对高测量噪声更为 robust，速度快，并能更准确地估计与观测相符的解，优于现有方法。 

---
# FLEX: A Backbone for Diffusion-Based Modeling of Spatio-temporal Physical Systems 

**Title (ZH)**: FLEX：基于扩散模型的空间时间物理系统骨干网络 

**Authors**: N. Benjamin Erichson, Vinicius Mikuni, Dongwei Lyu, Yang Gao, Omri Azencot, Soon Hoe Lim, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2505.17351)  

**Abstract**: We introduce FLEX (FLow EXpert), a backbone architecture for generative modeling of spatio-temporal physical systems using diffusion models. FLEX operates in the residual space rather than on raw data, a modeling choice that we motivate theoretically, showing that it reduces the variance of the velocity field in the diffusion model, which helps stabilize training. FLEX integrates a latent Transformer into a U-Net with standard convolutional ResNet layers and incorporates a redesigned skip connection scheme. This hybrid design enables the model to capture both local spatial detail and long-range dependencies in latent space. To improve spatio-temporal conditioning, FLEX uses a task-specific encoder that processes auxiliary inputs such as coarse or past snapshots. Weak conditioning is applied to the shared encoder via skip connections to promote generalization, while strong conditioning is applied to the decoder through both skip and bottleneck features to ensure reconstruction fidelity. FLEX achieves accurate predictions for super-resolution and forecasting tasks using as few as two reverse diffusion steps. It also produces calibrated uncertainty estimates through sampling. Evaluations on high-resolution 2D turbulence data show that FLEX outperforms strong baselines and generalizes to out-of-distribution settings, including unseen Reynolds numbers, physical observables (e.g., fluid flow velocity fields), and boundary conditions. 

**Abstract (ZH)**: 基于扩散模型的时空物理系统生成建模架构FLEX 

---
# A Multi-Head Attention Soft Random Forest for Interpretable Patient No-Show Prediction 

**Title (ZH)**: 多头注意力软随机森林可解释的患者爽约预测 

**Authors**: Ninda Nurseha Amalina, Kwadwo Boateng Ofori-Amanfo, Heungjo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.17344)  

**Abstract**: Unattended scheduled appointments, defined as patient no-shows, adversely affect both healthcare providers and patients' health, disrupting the continuity of care, operational efficiency, and the efficient allocation of medical resources. Accurate predictive modelling is needed to reduce the impact of no-shows. Although machine learning methods, such as logistic regression, random forest models, and decision trees, are widely used in predicting patient no-shows, they often rely on hard decision splits and static feature importance, limiting their adaptability to specific or complex patient behaviors. To address this limitation, we propose a new hybrid Multi-Head Attention Soft Random Forest (MHASRF) model that integrates attention mechanisms into a random forest model using probabilistic soft splitting instead of hard splitting. The MHASRF model assigns attention weights differently across the trees, enabling attention on specific patient behaviors. The model exhibited 93.56% accuracy, 93.67% precision, 93.56% recall, and a 93.59% F1 score, surpassing the performance of decision tree, logistic regression, random forest, and naive Bayes models. Furthermore, MHASRF was able to identify key predictors of patient no-shows using two levels of feature importance (tree level and attention mechanism level), offering deeper insights into patient no-show predictors. The proposed model is a robust, adaptable, and interpretable method for predicting patient no-shows that will help healthcare providers in optimizing resources. 

**Abstract (ZH)**: 未陪同预约患者的预测建模：一种新的混合多头注意力软随机森林（MHASRF）方法及其应用 

---
# Render-FM: A Foundation Model for Real-time Photorealistic Volumetric Rendering 

**Title (ZH)**: Render-FM：实时高逼真体积渲染的基础模型 

**Authors**: Zhongpai Gao, Meng Zheng, Benjamin Planche, Anwesa Choudhuri, Terrence Chen, Ziyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17338)  

**Abstract**: Volumetric rendering of Computed Tomography (CT) scans is crucial for visualizing complex 3D anatomical structures in medical imaging. Current high-fidelity approaches, especially neural rendering techniques, require time-consuming per-scene optimization, limiting clinical applicability due to computational demands and poor generalizability. We propose Render-FM, a novel foundation model for direct, real-time volumetric rendering of CT scans. Render-FM employs an encoder-decoder architecture that directly regresses 6D Gaussian Splatting (6DGS) parameters from CT volumes, eliminating per-scan optimization through large-scale pre-training on diverse medical data. By integrating robust feature extraction with the expressive power of 6DGS, our approach efficiently generates high-quality, real-time interactive 3D visualizations across diverse clinical CT data. Experiments demonstrate that Render-FM achieves visual fidelity comparable or superior to specialized per-scan methods while drastically reducing preparation time from nearly an hour to seconds for a single inference step. This advancement enables seamless integration into real-time surgical planning and diagnostic workflows. The project page is: this https URL. 

**Abstract (ZH)**: 计算机断层扫描（CT）的体绘制对于医学成像中可视化复杂的三维解剖结构至关重要。当前的高保真方法，尤其是神经渲染技术，需要耗时的逐场景优化，由于计算需求大和缺乏普适性，限制了其临床应用。我们提出Render-FM，这是一种新型的基础模型，用于直接实时渲染CT扫描的体绘制。Render-FM 使用编码器-解码器架构，直接从CT体积中回归6D高斯聚散（6DGS）参数，通过在多样化的医学数据上进行大规模预训练来消除逐扫描优化。通过结合鲁棒的特征提取与6DGS的强大表达能力，我们的方法能够高效地生成高质量的、实时交互的3D可视化结果，适用于多样化的临床CT数据。实验表明，Render-FM 在单次推断步骤中的准备时间从近一小时大幅减少到几秒，同时在视觉保真度方面达到或优于专门的逐扫描方法。这一进步使得无缝集成到实时手术计划和诊断流程中成为可能。项目页面：this https URL。 

---
# SweEval: Do LLMs Really Swear? A Safety Benchmark for Testing Limits for Enterprise Use 

**Title (ZH)**: SweEval: 难道LLM们真的会咒骂吗？一间企业使用的安全性基准测试 

**Authors**: Hitesh Laxmichand Patel, Amit Agarwal, Arion Das, Bhargava Kumar, Srikant Panda, Priyaranjan Pattnayak, Taki Hasan Rafi, Tejaswini Kumar, Dong-Kyu Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.17332)  

**Abstract**: Enterprise customers are increasingly adopting Large Language Models (LLMs) for critical communication tasks, such as drafting emails, crafting sales pitches, and composing casual messages. Deploying such models across different regions requires them to understand diverse cultural and linguistic contexts and generate safe and respectful responses. For enterprise applications, it is crucial to mitigate reputational risks, maintain trust, and ensure compliance by effectively identifying and handling unsafe or offensive language. To address this, we introduce SweEval, a benchmark simulating real-world scenarios with variations in tone (positive or negative) and context (formal or informal). The prompts explicitly instruct the model to include specific swear words while completing the task. This benchmark evaluates whether LLMs comply with or resist such inappropriate instructions and assesses their alignment with ethical frameworks, cultural nuances, and language comprehension capabilities. In order to advance research in building ethically aligned AI systems for enterprise use and beyond, we release the dataset and code: this https URL. 

**Abstract (ZH)**: 企业客户越来越多地将大型语言模型（LLMs）应用于关键沟通任务，如撰写电子邮件、撰写销售提案和创作非正式信息。在不同地区部署这些模型需要它们理解多样化的文化与语言背景，并生成安全和尊重人的回应。对于企业应用而言，有效地识别和处理不安全或冒犯性的语言，以减轻声誉风险、维护信任并确保合规性至关重要。为此，我们引入了SweEval基准，该基准模拟了具有不同语气（正面或负面）和语境（正式或非正式）的真实世界场景。提示明确指示模型在完成任务时包含特定的脏话。该基准评估LLMs是否遵守或抵制这样的不合适指令，并评估它们与道德框架、文化细微差别和语言理解能力的兼容性。为了推进构建符合伦理的企业用AI系统的研究及其更广泛的应用，我们发布数据集和代码：this https URL。 

---
# FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding 

**Title (ZH)**: FS-DAG: 几乎零样本领域适应图网络在富视觉文档理解中的应用 

**Authors**: Amit Agarwal, Srikant Panda, Kulbhushan Pachauri  

**Link**: [PDF](https://arxiv.org/pdf/2505.17330)  

**Abstract**: In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : this https URL 

**Abstract (ZH)**: 在本次工作中，我们提出了一种可扩展且高效的 Few Shot Domain Adapting Graph (FS-DAG) 模型架构，用于少样本设置下的丰富视觉文档理解 (VRDU)。FS-DAG 在模块化框架中利用领域特定和视觉/语言特定的骨干网络，以少量数据适应多种文档类型。该模型对实际挑战（如处理OCR错误、拼写错误和领域偏移）具有鲁棒性，这些都是实际部署中的关键问题。FS-DAG 性能强劲，参数量少于90M，使其适用于计算资源有限的信息提取 (IE) 任务等复杂现实应用场景。通过详尽的实验展示了FS-DAG在信息提取任务上的能力，相比现有最佳方法，在收敛速度和性能上均取得了显著改进。此外，本文还强调了在不牺牲性能的情况下开发更小更高效模型的持续进展。代码：[这个链接] 

---
# From Compression to Expansion: A Layerwise Analysis of In-Context Learning 

**Title (ZH)**: 从压缩到扩展：基于层的内文学习分析 

**Authors**: Jiachen Jiang, Yuxin Dong, Jinxin Zhou, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17322)  

**Abstract**: In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks without weight updates by learning from demonstration sequences. While ICL shows strong empirical performance, its internal representational mechanisms are not yet well understood. In this work, we conduct a statistical geometric analysis of ICL representations to investigate how task-specific information is captured across layers. Our analysis reveals an intriguing phenomenon, which we term *Layerwise Compression-Expansion*: early layers progressively produce compact and discriminative representations that encode task information from the input demonstrations, while later layers expand these representations to incorporate the query and generate the prediction. This phenomenon is observed consistently across diverse tasks and a range of contemporary LLM architectures. We demonstrate that it has important implications for ICL performance -- improving with model size and the number of demonstrations -- and for robustness in the presence of noisy examples. To further understand the effect of the compact task representation, we propose a bias-variance decomposition and provide a theoretical analysis showing how attention mechanisms contribute to reducing both variance and bias, thereby enhancing performance as the number of demonstrations increases. Our findings reveal an intriguing layerwise dynamic in ICL, highlight how structured representations emerge within LLMs, and showcase that analyzing internal representations can facilitate a deeper understanding of model behavior. 

**Abstract (ZH)**: 上下文学习(ICL)使大规模语言模型(LLMs)能够在无需权重更新的情况下适应新任务，通过从演示序列中学习。尽管ICL表现出强大的实证性能，但其内部表征机制尚不完全理解。在此工作中，我们进行了一种统计几何分析，以探讨任务特定信息如何在各层中被捕获。我们的分析揭示了一种有趣的现象，我们称之为“逐层压缩-扩展”：早期层逐步生成紧凑且判别性的表征，编码输入演示中的任务信息，而后期层则扩展这些表征以包含查询并生成预测。这种现象在多种任务和一系列现代LLM架构中得到一致观察。我们表明，这对其性能（随着模型规模和演示次数的增加而改善）以及在存在嘈杂示例时的鲁棒性具有重要影响。为了进一步理解紧凑任务表征的影响，我们提出了一种偏差-方差分解，并提供了一种理论分析，说明注意机制如何通过减少偏差和方差来提高性能，随着演示次数的增加而增强性能。我们的发现揭示了ICL中有趣的逐层动态，突显了结构化表征如何在LLMs中浮现，并展示了分析内部表征如何促进对模型行为更深的理解。 

---
# Control of Renewable Energy Communities using AI and Real-World Data 

**Title (ZH)**: 基于AI和实物数据的可再生能源社区控制 

**Authors**: Tiago Fonseca, Clarisse Sousa, Ricardo Venâncio, Pedro Pires, Ricardo Severino, Paulo Rodrigues, Pedro Paiva, Luis Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2505.17321)  

**Abstract**: The electrification of transportation and the increased adoption of decentralized renewable energy generation have added complexity to managing Renewable Energy Communities (RECs). Integrating Electric Vehicle (EV) charging with building energy systems like heating, ventilation, air conditioning (HVAC), photovoltaic (PV) generation, and battery storage presents significant opportunities but also practical challenges. Reinforcement learning (RL), particularly MultiAgent Deep Deterministic Policy Gradient (MADDPG) algorithms, have shown promising results in simulation, outperforming heuristic control strategies. However, translating these successes into real-world deployments faces substantial challenges, including incomplete and noisy data, integration of heterogeneous subsystems, synchronization issues, unpredictable occupant behavior, and missing critical EV state-of-charge (SoC) information. This paper introduces a framework designed explicitly to handle these complexities and bridge the simulation to-reality gap. The framework incorporates EnergAIze, a MADDPG-based multi-agent control strategy, and specifically addresses challenges related to real-world data collection, system integration, and user behavior modeling. Preliminary results collected from a real-world operational REC with four residential buildings demonstrate the practical feasibility of our approach, achieving an average 9% reduction in daily peak demand and a 5% decrease in energy costs through optimized load scheduling and EV charging behaviors. These outcomes underscore the framework's effectiveness, advancing the practical deployment of intelligent energy management solutions in RECs. 

**Abstract (ZH)**: 交通运输的电气化和分散型可再生能源发电的增加为可再生能源社区（RECs）的管理增添了复杂性。电动汽车（EV）充电与建筑能源系统，包括供暖、通风和空调（HVAC）、光伏（PV）发电和电池储能的整合，带来了显著的机会和实际挑战。基于强化学习（RL），尤其是基于多agent深度确定性策略梯度（MADDPG）算法，已经在仿真中显示出有前途的结果，超越了启发式控制策略。然而，将这些成功转化为实际部署面临重大挑战，包括数据不完整和噪声、异构子系统整合问题、同步问题、不确定的用户行为以及缺失的关键电动汽车电量状态信息（SoC）。本文介绍了一种专为处理这些复杂性而设计的框架，旨在弥合仿真到现实的差距。该框架整合了基于MADDPG的多agent控制策略EnergAIze，并具体解决了实际数据收集、系统整合以及用户行为建模的挑战。从一个包含四栋住宅建筑的实时运行REC中收集的初步结果表明，该方法在实际应用中的可行性，通过优化负荷调度和EV充电行为，实现了每日峰荷平均9%的减少和能源成本5%的降低。这些结果证实了该框架的有效性，推动了在RECs中智能能源管理解决方案的实际部署。 

---
# Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models 

**Title (ZH)**: 分析细粒度对齐并增强多模态语言模型的视觉理解 

**Authors**: Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17316)  

**Abstract**: Achieving better alignment between vision embeddings and Large Language Models (LLMs) is crucial for enhancing the abilities of Multimodal LLMs (MLLMs), particularly for recent models that rely on powerful pretrained vision encoders and LLMs. A common approach to connect the pretrained vision encoder and LLM is through a projector applied after the vision encoder. However, the projector is often trained to enable the LLM to generate captions, and hence the mechanism by which LLMs understand each vision token remains unclear. In this work, we first investigate the role of the projector in compressing vision embeddings and aligning them with word embeddings. We show that the projector significantly compresses visual information, removing redundant details while preserving essential elements necessary for the LLM to understand visual content. We then examine patch-level alignment -- the alignment between each vision patch and its corresponding semantic words -- and propose a *multi-semantic alignment hypothesis*. Our analysis indicates that the projector trained by caption loss improves patch-level alignment but only to a limited extent, resulting in weak and coarse alignment. To address this issue, we propose *patch-aligned training* to efficiently enhance patch-level alignment. Our experiments show that patch-aligned training (1) achieves stronger compression capability and improved patch-level alignment, enabling the MLLM to generate higher-quality captions, (2) improves the MLLM's performance by 16% on referring expression grounding tasks, 4% on question-answering tasks, and 3% on modern instruction-following benchmarks when using the same supervised fine-tuning (SFT) setting. The proposed method can be easily extended to other multimodal models. 

**Abstract (ZH)**: 实现更好的视觉嵌入与大型语言模型的对齐对于增强多模态大型语言模型的能力至关重要，尤其是在依赖强大预训练视觉编码器和大型语言模型的最近模型中。一种常见的连接预训练视觉编码器和大型语言模型的方法是在视觉编码器之后应用一个投影器。然而，该投影器通常被训练以使大型语言模型生成描述，因此大型语言模型理解每个视觉标记的机制仍然不清楚。在本工作中，我们首先研究了投影器在压缩视觉嵌入并将它们与词嵌入对齐中的作用。我们表明，投影器显着压缩了视觉信息，去除了冗余细节，同时保留了大型语言模型理解视觉内容所必需的关键元素。随后，我们考察了像素级对齐——即每个视觉像素与其对应的语义词之间的对齐——并提出了一种“多语义对齐假说”。我们的分析表明，通过描述损失训练的投影器可以改善像素级对齐，但只能在有限的范围内，导致弱且粗糙的对齐。为解决这一问题，我们提出了“像素对齐训练”以有效地提升像素级对齐。我们的实验表明，像素对齐训练（1）实现了更强的压缩能力和改进了像素级对齐，使多模态大型语言模型能够生成更具质量的描述，（2）在使用相同监督微调（SFT）设置时，多模态大型语言模型在引用表达对接任务中的性能提高了16%，在问答任务中的性能提高了4%，在现代指令跟随基准中的性能提高了3%。所提出的方法可以容易地扩展到其他多模态模型。 

---
# LaSER: How Learning Can Guide the Evolution of Equations 

**Title (ZH)**: LaSER: 学习如何引导方程的演化 

**Authors**: Nam H. Le, Josh Bongard  

**Link**: [PDF](https://arxiv.org/pdf/2505.17309)  

**Abstract**: Evolution and learning are two distinct yet complementary forms of adaptation. While evolutionary processes operate across generations via the selection of genotypes, learning occurs within the lifetime of an individual, shaping behavior through phenotypic adjustment. The Baldwin effect describes how lifetime learning can improve evolutionary search without altering inherited structures. While this has proven effective in areas like neuroevolution, where gradient-based learning is often used to fine-tune weights or behaviors produced by evolution, it remains underexplored in systems that evolve non-differentiable symbolic structures like Genetic Programming (GP). GP evolves explicit syntax trees that represent equations, offering strong interpretability but limited generalization due to the burden of discovering both useful representations and precise mappings.
Here, we show for the first time that integrating a simple form of supervised learning, applied at the semantic or behavioral level during evaluation, can effectively guide the evolution of equations in GP. To achieve this, we propose a new GP pipeline, LaSER (Latent Semantic Evolutionary Regression), where each GP individual generates a semantic representation that is passed to a supervised learner. The quality of the learned mapping is used to assign fitness, without modifying the underlying syntax tree or evolutionary process.
Across standard symbolic regression benchmarks, in terms of generalization ability, LaSER significantly outperforms traditional GP and, in several cases, matches or exceeds popular machine learning regressors, while preserving the symbolic interpretability. By separating evolution from learning, LaSER offers a practical route to integrating GP with modern ML workflows, and opens new avenues for research at the intersection of evolutionary computation and representation learning. 

**Abstract (ZH)**: 进化与学习是两种distinct yet complementary形式的适应。进化过程通过选择表型在代际间进行，而学习则在个体的生命周期内发生，通过表型调整来塑造行为。巴迪尔效应描述了如何在不改变遗传结构的情况下通过生命周期学习提高进化的搜索效率。虽然这种方法在神经进化等领域已被证明有效，尤其是在使用基于梯度的学习来精细调整由进化产生的权重或行为时，但它在演化非可微符号结构如遗传编程（GP）的系统中仍处于未被充分探索的状态。GP生成代表方程的显式语法树，提供了很强的可解释性，但由于需要发现有用的表现形式和精确映射，其泛化能力受限。

在此，我们首次展示了通过在评估过程中应用一种简单的监督学习，特别是在语义或行为层面，可以有效指导GP中方程的进化。为此，我们提出了一种新的GP管道LaSER（潜在语义进化回归），其中每个GP个体生成一个语义表示并传递给监督学习器。学习到的映射质量用于分配适应度，而不修改底层的语法树或进化过程。

在标准符号回归基准中，无论在泛化能力方面，LaSER显著优于传统GP，而且在某些情况下，它至少与流行机器学习回归器相当，同时保持了符号的可解释性。通过分离进化与学习，LaSER提供了将GP与现代机器学习工作流集成的实用途径，并为进化计算与表示学习交叉领域的研究开辟了新的途径。 

---
# Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty 

**Title (ZH)**: 明智搜索：通过降低不确定性来缓解亚优化代理搜索 

**Authors**: Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17281)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score. 

**Abstract (ZH)**: 代理检索增强生成（RAG）系统通过实现动态多步推理和信息检索，增强了大型语言模型（LLMs）。然而，这些系统常常表现出如过度检索（检索冗余信息）和不足检索（未能检索必要信息）等次优搜索行为，这阻碍了效率和可靠性。本研究正式定义并量化了这些行为，在多个QA数据集和代理RAG系统中揭示了它们的普遍性（例如，一个模型在其搜索步骤中可以避免搜索27.7%的情况）。此外，我们证明了这些低效率与模型对自己知识边界不确定性的关联，响应准确性与模型在搜索决策中的不确定性相关。为了解决这一问题，我们提出了基于强化学习的$\beta$-GRPO训练方法，该方法结合了置信阈值以奖励高确定性的搜索决策。在七个QA基准上的实验显示，$\beta$-GRPO使一个3B模型具备更好的代理RAG能力，与其它强基线相比，其平均精确匹配得分高4%。 

---
# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 

**Title (ZH)**: Select2Reason: 高效指令调优数据选择用于长链推理 

**Authors**: Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Xiaojun Wu, Honghao Liu, Hui Xiong, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17266)  

**Abstract**: A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost. 

**Abstract (ZH)**: 一种实用的方法是在预训练的大语言模型中激活长链推理能力，即通过强大型推理模型（如DeepSeek-R1）合成的指令数据集进行监督微调，这是一种成本-effective的替代强化学习的方法。然而，大规模的包含超过10万样本的指令集会带来显著的训练开销，而有效的自动长链推理指令选择策略仍待探索。在此工作中，我们提出Select2Reason，一种新颖且高效的任务指令筛选框架，用于长链推理。从重新思考行为如自我纠正和回溯的出现视角出发，我们探究了能够决定长链推理指令质量的常见指标。Select2Reason利用量化器估计问题难度，并通过加权方案结合推理轨迹长度启发式方法来优先选择高效用示例。在OpenR1-Math-220K上的实验结果表明，仅使用Select2Reason选择的10%数据微调LLM的性能与其全数据微调和开源基线OpenR1-Qwen-7B在三个竞赛级别和六个全面数学基准上的性能相当或更优。进一步的实验突显了其在不同数据规模下的扩展性、推理过程中的高效性及其在较少成本下的可适应性。 

---
# CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports 

**Title (ZH)**: CaseReportBench: 一键提取临床病例报告中密集信息的LLM基准数据集 

**Authors**: Xiao Yu Cindy Zhang, Carlos R. Ferreira, Francis Rossignol, Raymond T. Ng, Wyeth Wasserman, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17265)  

**Abstract**: Rare diseases, including Inborn Errors of Metabolism (IEM), pose significant diagnostic challenges. Case reports serve as key but computationally underutilized resources to inform diagnosis. Clinical dense information extraction refers to organizing medical information into structured predefined categories. Large Language Models (LLMs) may enable scalable information extraction from case reports but are rarely evaluated for this task. We introduce CaseReportBench, an expert-annotated dataset for dense information extraction of case reports, focusing on IEMs. Using this dataset, we assess various models and prompting strategies, introducing novel approaches such as category-specific prompting and subheading-filtered data integration. Zero-shot chain-of-thought prompting offers little advantage over standard zero-shot prompting. Category-specific prompting improves alignment with the benchmark. The open-source model Qwen2.5-7B outperforms GPT-4o for this task. Our clinician evaluations show that LLMs can extract clinically relevant details from case reports, supporting rare disease diagnosis and management. We also highlight areas for improvement, such as LLMs' limitations in recognizing negative findings important for differential diagnosis. This work advances LLM-driven clinical natural language processing and paves the way for scalable medical AI applications. 

**Abstract (ZH)**: 罕见疾病，包括先天代谢误差（IEM），诊断极具挑战性。案例报告作为关键但计算上未充分利用的资源，用于辅助诊断。临床密集信息提取是指将医学信息组织到结构化预定义类别中。大规模语言模型（LLMs）可能使案例报告的信息提取规模化，但很少对此任务进行评估。我们介绍了CaseReportBench，一个由专家注释的用于案例报告密集信息提取的数据集，重点关注IEMs。利用该数据集，我们评估了各种模型和提示策略，引入了类别特定提示和子标题过滤数据集成等新方法。零样本链式思考提示在效果上几乎没有优于标准零样本提示的优势。类别特定提示提高了与基准的对齐度。开源模型Qwen2.5-7B在这一任务上优于GPT-4o。我们的临床医生评估表明，LLMs可以从案例报告中提取临床相关细节，支持罕见疾病的诊断和管理。我们还指出了改进领域，如LLMs在识别对鉴别诊断重要的阴性发现方面存在局限性。这项工作推动了LLM驱动的临床自然语言处理，并为可扩展的医学AI应用铺平了道路。 

---
# ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models 

**Title (ZH)**: ConciseRL：简洁性引导的强化学习方法以促进高效推理模型 

**Authors**: Razvan-Gabriel Dumitru, Darius Peteleaza, Vikas Yadav, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17250)  

**Abstract**: Large language models excel at complex tasks by breaking down problems into structured reasoning steps. However, reasoning traces often extend beyond reaching a correct answer, causing wasted computation, reduced readability, and hallucinations. To address this, we introduce a novel hyperparameter-free conciseness score used as a reward signal within a reinforcement learning framework to guide models toward generating correct and concise reasoning traces. This score is evaluated by a large language model acting as a judge, enabling dynamic, context-aware feedback beyond simple token length. Our method achieves state-of-the-art efficiency-accuracy trade-offs on the MATH dataset, reducing token usage by up to 31x on simple problems while improving accuracy by 7%, and on the hardest problems, it outperforms full reasoning by +7.5% accuracy with up to 3.6x fewer tokens. On TheoremQA, our method improves accuracy by +2.2% using 12.5x fewer tokens. We also conduct ablation studies on the judge model, reward composition, and problem difficulty, showing that our method dynamically adapts reasoning length based on problem difficulty and benefits significantly from stronger judges. The code, model weights, and datasets are open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型通过将问题分解为结构化推理步骤来胜任复杂任务，但推理痕迹往往会超出得出正确答案的范围，导致计算资源浪费、降低可读性及产生幻觉。为此，我们引入了一种新型无需超参数的简洁性评分，将其作为强化学习框架内的奖励信号，以引导模型生成正确且简洁的推理痕迹。该评分通过大型语言模型作为评判者进行评估，允许动态、上下文相关的反馈超越简单的 token 长度评估。我们的方法在 MATH 数据集上实现了最先进的效率-准确率权衡，在简单问题上将 token 使用量最多减少 31 倍，并将准确率提高 7%；在最难的问题上，它在准确率上比完整推理高出 7.5%，且使用不到 3.6 倍的 token 数量。在 TheoremQA 上，我们的方法使用 12.5 倍 fewer tokens 时准确率提高了 2.2%。我们还进行了评判者模型、奖励组成和问题难度的消融研究，结果显示我们的方法会根据问题难度动态调整推理长度，并且显著受益于更强的评判者。相关代码、模型权重和数据集在 https://github.com/xxx开放获取。 

---
# ReasoningShield: Content Safety Detection over Reasoning Traces of Large Reasoning Models 

**Title (ZH)**: ReasoningShield：大型推理模型推理轨迹的内容安全检测 

**Authors**: Changyi Li, Jiayi Wang, Xudong Pan, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17244)  

**Abstract**: Large Reasoning Models (LRMs) are transforming the AI landscape with advanced reasoning capabilities. While the generated reasoning traces enhance model transparency, they can still contain unsafe content, even when the final answer appears safe. Existing moderation tools, primarily designed for question-answer (QA) pairs, are empirically ineffective at detecting hidden risks embedded in reasoning traces. After identifying the key challenges, we formally define the question-thought (QT) moderation task and propose ReasoningShield, the first safety detection model tailored to identify potential risks in the reasoning trace before reaching the final answer. To construct the model, we synthesize a high-quality reasoning safety detection dataset comprising over 8,000 question-thought pairs spanning ten risk categories and three safety levels. Our dataset construction process incorporates a comprehensive human-AI collaborative annotation pipeline, which achieves over 93% annotation accuracy while significantly reducing human costs. On a diverse set of in-distribution and out-of-distribution benchmarks, ReasoningShield outperforms mainstream content safety moderation models in identifying risks within reasoning traces, with an average F1 score exceeding 0.92. Notably, despite being trained on our QT dataset only, ReasoningShield also demonstrates competitive performance in detecting unsafe question-answer pairs on traditional benchmarks, rivaling baselines trained on 10 times larger datasets and base models, which strongly validates the quality of our dataset. Furthermore, ReasoningShield is built upon compact 1B/3B base models to facilitate lightweight deployment and provides human-friendly risk analysis by default. To foster future research, we publicly release all the resources. 

**Abstract (ZH)**: 大型推理模型（LRMs）正在以先进的推理能力改变AI格局。虽然生成的推理轨迹能够提升模型的透明度，但仍可能包含不安全的内容，即使最终答案看似安全。现有的主要用于问答（QA）对的审核工具，在检测推理轨迹中隐藏的风险方面见效甚微。在识别关键挑战后，我们正式定义了问题-思考（QT）审核任务，并提出ReasoningShield，这是首个针对推理轨迹中潜在风险进行检测的安全检测模型。为构建该模型，我们合成了一套高质量的推理安全检测数据集，包含超过8,000个问题-思考对，涵盖十个风险类别和三个安全等级。我们的数据集构建过程集成了全面的人机协作注释流水线，准确率达到93%以上，同时大幅降低了人力成本。在不同分布的基准测试中，ReasoningShield在识别推理轨迹中的风险方面优于主流内容安全审核模型，平均F1分数超过0.92。值得注意的是，尽管仅基于我们的QT数据集进行训练，但ReasoningShield在传统基准上检测不安全的问答对时也表现出竞争力，媲美使用更大规模数据集和基模型训练的基模型，这有力地验证了我们数据集的质量。此外，ReasoningShield基于紧凑的1B/3B基模型构建，便于轻量级部署，并默认提供用户友好的风险分析。为了促进未来研究，我们公开发布了所有资源。 

---
# Optimal Policy Minimum Bayesian Risk 

**Title (ZH)**: 最小贝叶斯风险最优策略 

**Authors**: Ramón Fernandez Astudillo, Md Arafat Sultan, Aashka Trivedi, Yousef El-Kurdi, Tahira Naseem, Radu Florian, Salim Roukos  

**Link**: [PDF](https://arxiv.org/pdf/2505.17242)  

**Abstract**: Inference scaling can help LLMs solve complex reasoning problems through extended runtime computation. On top of targeted supervision for long chain-of-thought (long-CoT) generation, purely inference-time techniques such as best-of-N (BoN) sampling, majority voting, or more generally, minimum Bayes risk decoding (MBRD), can further improve LLM accuracy by generating multiple candidate solutions and aggregating over them. These methods typically leverage additional signals in the form of reward models and risk/similarity functions that compare generated samples, e.g., exact match in some normalized space or standard similarity metrics such as Rouge. Here we present a novel method for incorporating reward and risk/similarity signals into MBRD. Based on the concept of optimal policy in KL-controlled reinforcement learning, our framework provides a simple and well-defined mechanism for leveraging such signals, offering several advantages over traditional inference-time methods: higher robustness, improved accuracy, and well-understood asymptotic behavior. In addition, it allows for the development of a sample-efficient variant of MBRD that can adjust the number of samples to generate according to the difficulty of the problem, without relying on majority vote counts. We empirically demonstrate the advantages of our approach on math (MATH-$500$) and coding (HumanEval) tasks using recent open-source models. We also present a comprehensive analysis of its accuracy-compute trade-offs. 

**Abstract (ZH)**: 基于推断缩放的复杂推理问题解决方法：通过额外信号改进最小贝叶斯风险解码 

---
# Generative AI and Creativity: A Systematic Literature Review and Meta-Analysis 

**Title (ZH)**: 生成式人工智能与创造力：一项系统文献综述与元分析 

**Authors**: Niklas Holzner, Sebastian Maier, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17241)  

**Abstract**: Generative artificial intelligence (GenAI) is increasingly used to support a wide range of human tasks, yet empirical evidence on its effect on creativity remains scattered. Can GenAI generate ideas that are creative? To what extent can it support humans in generating ideas that are both creative and diverse? In this study, we conduct a meta-analysis to evaluate the effect of GenAI on the performance in creative tasks. For this, we first perform a systematic literature search, based on which we identify n = 28 relevant studies (m = 8214 participants) for inclusion in our meta-analysis. We then compute standardized effect sizes based on Hedges' g. We compare different outcomes: (i) how creative GenAI is; (ii) how creative humans augmented by GenAI are; and (iii) the diversity of ideas by humans augmented by GenAI. Our results show no significant difference in creative performance between GenAI and humans (g = -0.05), while humans collaborating with GenAI significantly outperform those working without assistance (g = 0.27). However, GenAI has a significant negative effect on the diversity of ideas for such collaborations between humans and GenAI (g = -0.86). We further analyze heterogeneity across different GenAI models (e.g., GPT-3.5, GPT-4), different tasks (e.g., creative writing, ideation, divergent thinking), and different participant populations (e.g., laypeople, business, academia). Overall, our results position GenAI as an augmentative tool that can support, rather than replace, human creativity-particularly in tasks benefiting from ideation support. 

**Abstract (ZH)**: 生成式人工智能（GenAI）日益用于支持广泛的-human任务，然而其对创造力的影响证据仍较为分散。GenAI能否产生创意的想法？它在多大程度上能辅助人类产生既有创意又多样化的想法？本研究通过元分析评估GenAI对创造力任务表现的影响。为此，我们首先进行系统的文献搜索，确定了n=28项相关研究（m=8214名参与者）纳入元分析。我们基于Hedges' g计算标准化的效应大小，比较不同结果：（i）GenAI的创造性；（ii）受GenAI辅助的人类的创造性；（iii）受GenAI辅助的人类的想法多样性。结果显示，在创造力表现上，GenAI与人类之间无显著差异（g = -0.05），但与GenAI合作的人类显著优于未得到辅助的人类（g = 0.27）。然而，GenAI对人类与GenAI合作的想法多样性有显著负面影响（g = -0.86）。进一步分析不同GenAI模型（如GPT-3.5、GPT-4）、不同任务（如创意写作、创意生成、发散思维）以及不同参与者群体（如普通人群、商业界、学术界）之间的异质性。总体而言，我们的结果将GenAI定位为一种辅助工具，能够支持而非替代人类的创造力，特别是在受益于创意支持的任务中。 

---
# ExeSQL: Self-Taught Text-to-SQL Models with Execution-Driven Bootstrapping for SQL Dialects 

**Title (ZH)**: ExeSQL：基于执行驱动-bootstrapping的自教学文本到SQL模型用于SQL方言训练 

**Authors**: Jipeng Zhang, Haolin Yang, Kehao Miao, Ruiyuan Zhang, Renjie Pi, Jiahui Gao, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17231)  

**Abstract**: Recent text-to-SQL models have achieved strong performance, but their effectiveness remains largely confined to SQLite due to dataset limitations. However, real-world applications require SQL generation across multiple dialects with varying syntax and specialized features, which remains a challenge for current models. The main obstacle in building a dialect-aware model lies in acquiring high-quality dialect-specific data. Data generated purely through static prompting - without validating SQLs via execution - tends to be noisy and unreliable. Moreover, the lack of real execution environments in the training loop prevents models from grounding their predictions in executable semantics, limiting generalization despite surface-level improvements from data filtering. This work introduces ExeSQL, a text-to-SQL framework with execution-driven, agentic bootstrapping. The method consists of iterative query generation, execution-based filtering (e.g., rejection sampling), and preference-based training, enabling the model to adapt to new SQL dialects through verifiable, feedback-guided learning. Experiments show that ExeSQL bridges the dialect gap in text-to-SQL, achieving average improvements of 15.2%, 10.38%, and 4.49% over GPT-4o on PostgreSQL, MySQL, and Oracle, respectively, across multiple datasets of varying difficulty. 

**Abstract (ZH)**: Recent text-to-SQL模型取得了强大的性能，但由于数据集的限制，其有效性主要局限于SQLite。然而，实际应用要求在具有不同语法和专业化功能的多种方言下生成SQL，这仍然是现有模型的挑战。构建具有方言意识模型的主要障碍在于获取高质量的方言特定数据。仅通过静态提示生成的数据——未经执行验证——往往会变得嘈杂和不可靠。此外，训练循环中缺乏真实的执行环境限制了模型将预测锚定在可执行语义上，尽管通过数据过滤在表面上有所改进。本项工作引入了ExeSQL，这是一种通过执行驱动、有代理性的训练框架。该方法包括迭代的查询生成、基于执行的过滤（例如，拒绝采样）和基于偏好的训练，使模型通过可验证的、基于反馈的指导学习来适应新的SQL方言。实验表明，ExeSQL在不同难度级别的多个数据集上分别在PostgreSQL、MySQL和Oracle上分别实现了15.2%、10.38%和4.49%的平均改进，填补了SQL方言之间的差距。 

---
# Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs 

**Title (ZH)**: 通过促进生成性思考来减轻LLMs中的性别偏见 

**Authors**: Kangda Wei, Hasnat Md Abdullah, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17217)  

**Abstract**: Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常表现出性别偏见，导致在不同情境下对男性和女性主体的不平等对待。为了解决这一问题，我们提出了一种新的数据生成框架，旨在激发大型语言模型的探究性思维。我们的方法通过提示模型生成结构相同但道德模糊的故事情节，分别以男性和女性为主角，然后引发并比较它们的道德判断。当出现不一致时，模型被引导生成平衡且性别中立的判断。这些故事情节及其判断配对用于通过直接偏好优化（DPO）微调或优化模型。实验结果表明，我们的方法显著降低了性别偏见，同时保持或甚至提升了模型的通用能力。我们将发布代码和生成的数据。 

---
# Assessing the generalization performance of SAM for ureteroscopy scene understanding 

**Title (ZH)**: 评估SAM在输尿管镜场景理解中的泛化性能 

**Authors**: Martin Villagrana, Francisco Lopez-Tiro, Clement Larose, Gilberto Ochoa-Ruiz, Christian Daul  

**Link**: [PDF](https://arxiv.org/pdf/2505.17210)  

**Abstract**: The segmentation of kidney stones is regarded as a critical preliminary step to enable the identification of urinary stone types through machine- or deep-learning-based approaches. In urology, manual segmentation is considered tedious and impractical due to the typically large scale of image databases and the continuous generation of new data. In this study, the potential of the Segment Anything Model (SAM) -- a state-of-the-art deep learning framework -- is investigated for the automation of kidney stone segmentation. The performance of SAM is evaluated in comparison to traditional models, including U-Net, Residual U-Net, and Attention U-Net, which, despite their efficiency, frequently exhibit limitations in generalizing to unseen datasets. The findings highlight SAM's superior adaptability and efficiency. While SAM achieves comparable performance to U-Net on in-distribution data (Accuracy: 97.68 + 3.04; Dice: 97.78 + 2.47; IoU: 95.76 + 4.18), it demonstrates significantly enhanced generalization capabilities on out-of-distribution data, surpassing all U-Net variants by margins of up to 23 percent. 

**Abstract (ZH)**: 肾结石分割在基于机器学习或深度学习的尿石类型识别中的初步自动化研究：Segment Anything Model在肾结石分割中的潜力及其性能评估 

---
# LiloDriver: A Lifelong Learning Framework for Closed-loop Motion Planning in Long-tail Autonomous Driving Scenarios 

**Title (ZH)**: LiloDriver：长尾自主驾驶场景中闭环运动规划的终身学习框架 

**Authors**: Huaiyuan Yao, Pengfei Li, Bu Jin, Yupeng Zheng, An Liu, Lisen Mu, Qing Su, Qian Zhang, Yilun Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17209)  

**Abstract**: Recent advances in autonomous driving research towards motion planners that are robust, safe, and adaptive. However, existing rule-based and data-driven planners lack adaptability to long-tail scenarios, while knowledge-driven methods offer strong reasoning but face challenges in representation, control, and real-world evaluation. To address these challenges, we present LiloDriver, a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. By integrating large language models (LLMs) with a memory-augmented planner generation system, LiloDriver continuously adapts to new scenarios without retraining. It features a four-stage architecture including perception, scene encoding, memory-based strategy refinement, and LLM-guided reasoning. Evaluated on the nuPlan benchmark, LiloDriver achieves superior performance in both common and rare driving scenarios, outperforming static rule-based and learning-based planners. Our results highlight the effectiveness of combining structured memory and LLM reasoning to enable scalable, human-like motion planning in real-world autonomous driving. Our code is available at this https URL. 

**Abstract (ZH)**: 近期自主驾驶研究中面向鲁棒、安全和适应性强的任务规划方法的进展。然而，现有的基于规则和数据驱动的规划方法缺乏对长尾场景的适应性，而知识驱动的方法虽能提供强大的推理能力，但在表示、控制和实地评估方面面临挑战。为应对这些挑战，我们提出了LiloDriver，一种用于长尾自主驾驶场景闭环运动规划的终身学习框架。通过将大型语言模型（LLMs）与记忆增强的规划生成系统集成，LiloDriver能够无需重新训练地持续适应新场景。其架构包括感知、场景编码、基于记忆的策略细化和LLM引导的推理四个阶段。在nuPlan基准测试中，LiloDriver在常见和罕见驾驶场景中的性能均优于静态规则驱动和学习驱动的规划方法。我们的结果表明，结合结构化记忆和LLM推理能够实现可扩展且类似人类的运动规划，并已在实际自主驾驶中得到验证。代码可访问此链接。 

---
# FB-RAG: Improving RAG with Forward and Backward Lookup 

**Title (ZH)**: FB-RAG: 改进RAG的前向和后向查找方法 

**Authors**: Kushal Chawla, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17206)  

**Abstract**: The performance of Retrieval Augmented Generation (RAG) systems relies heavily on the retriever quality and the size of the retrieved context. A large enough context ensures that the relevant information is present in the input context for the LLM, but also incorporates irrelevant content that has been shown to confuse the models. On the other hand, a smaller context reduces the irrelevant information, but it often comes at the risk of losing important information necessary to answer the input question. This duality is especially challenging to manage for complex queries that contain little information to retrieve the relevant chunks from the full context. To address this, we present a novel framework, called FB-RAG, which enhances the RAG pipeline by relying on a combination of backward lookup (overlap with the query) and forward lookup (overlap with candidate reasons and answers) to retrieve specific context chunks that are the most relevant for answering the input query. Our evaluations on 9 datasets from two leading benchmarks show that FB-RAG consistently outperforms RAG and Long Context baselines developed recently for these benchmarks. We further show that FB-RAG can improve performance while reducing latency. We perform qualitative analysis of the strengths and shortcomings of our approach, providing specific insights to guide future work. 

**Abstract (ZH)**: FB-RAG：一种结合反向和正向查找以增强检索增强生成的新型框架 

---
# LengthLogD: A Length-Stratified Ensemble Framework for Enhanced Peptide Lipophilicity Prediction via Multi-Scale Feature Integration 

**Title (ZH)**: LengthLogD：一种基于长度分层的集成框架，通过多尺度特征集成增强肽的疏水性预测 

**Authors**: Shuang Wu, Meijie Wang, Lun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17198)  

**Abstract**: Peptide compounds demonstrate considerable potential as therapeutic agents due to their high target affinity and low toxicity, yet their drug development is constrained by their low membrane permeability. Molecular weight and peptide length have significant effects on the logD of peptides, which in turn influences their ability to cross biological membranes. However, accurate prediction of peptide logD remains challenging due to the complex interplay between sequence, structure, and ionization states. This study introduces LengthLogD, a predictive framework that establishes specialized models through molecular length stratification while innovatively integrating multi-scale molecular representations. We constructed feature spaces across three hierarchical levels: atomic (10 molecular descriptors), structural (1024-bit Morgan fingerprints), and topological (3 graph-based features including Wiener index), optimized through stratified ensemble learning. An adaptive weight allocation mechanism specifically developed for long peptides significantly enhances model generalizability. Experimental results demonstrate superior performance across all categories: short peptides (R^2=0.855), medium peptides (R^2=0.816), and long peptides (R^2=0.882), with a 34.7% reduction in prediction error for long peptides compared to conventional single-model approaches. Ablation studies confirm: 1) The length-stratified strategy contributes 41.2% to performance improvement; 2) Topological features account for 28.5% of predictive importance. Compared to state-of-the-art models, our method maintains short peptide prediction accuracy while achieving a 25.7% increase in the coefficient of determination (R^2) for long peptides. This research provides a precise logD prediction tool for peptide drug development, particularly demonstrating unique value in optimizing long peptide lead compounds. 

**Abstract (ZH)**: 肽类化合物由于其高靶点亲和力和低毒性展现出作为治疗剂的巨大潜力，但其药物开发受限于低膜通透性。肽的分子量和长度对其logD值有显著影响，进而影响其穿过生物膜的能力。然而，由于序列、结构和电离状态之间的复杂相互作用，准确预测肽的logD依然具有挑战性。本研究引入了LengthLogD，这是一种通过分子长度分层建立专用模型并创新性集成多层次分子表示的预测框架。我们构建了三个层次的特征空间：原子级（10个分子描述符）、结构级（1024位默尔指纹图谱）、拓扑级（3个图基特征包括Wiener指数），并通过分层集成学习进行优化。专为长肽开发的自适应权重分配机制显著提高了模型的泛化能力。实验结果显示，在所有类别中均表现出优异的性能：短肽（R²=0.855）、中等长度肽（R²=0.816）和长肽（R²=0.882），长肽的预测误差降低了34.7%。消融研究表明：1）长度分层策略贡献了性能改进的41.2%；2）拓扑特征占预测重要性的28.5%。与最先进的模型相比，我们的方法在保持短肽预测准确性的同时，长肽的决定系数（R²）提高了25.7%。本研究提供了一种精确的logD预测工具，特别在于优化长肽先导化合物方面展现出独特价值。 

---
# Next Token Perception Score: Analytical Assessment of your LLM Perception Skills 

**Title (ZH)**: 下一个令牌感知分数：对您的LLM感知能力的分析评估 

**Authors**: Yu-Ang Cheng, Leyang Hu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.17169)  

**Abstract**: Autoregressive pretraining has become the de facto paradigm for learning general-purpose representations in large language models (LLMs). However, linear probe performance across downstream perception tasks shows substantial variability, suggesting that features optimized for next-token prediction do not consistently transfer well to downstream perception tasks. We demonstrate that representations learned via autoregression capture features that may lie outside the subspaces most informative for perception. To quantify the (mis)alignment between autoregressive pretraining and downstream perception, we introduce the Next Token Perception Score (NTPS)-a score derived under a linear setting that measures the overlap between autoregressive and perception feature subspaces. This metric can be easily computed in closed form from pretrained representations and labeled data, and is proven to both upper- and lower-bound the excess loss. Empirically, we show that NTPS correlates strongly with linear probe accuracy across 12 diverse NLP datasets and eight pretrained models ranging from 270M to 8B parameters, confirming its utility as a measure of alignment. Furthermore, we show that NTPS increases following low-rank adaptation (LoRA) fine-tuning, especially in large models, suggesting that LoRA aligning representations to perception tasks enhances subspace overlap and thus improves downstream performance. More importantly, we find that NTPS reliably predicts the additional accuracy gains attained by LoRA finetuning thereby providing a lightweight prescreening tool for LoRA adaptation. Our results offer both theoretical insights and practical tools for analytically assessing LLM perception skills. 

**Abstract (ZH)**: 自回归预训练已成为大型语言模型（LLMs）中学习通用表示的实际范式。然而，下游感知任务上的线性探针性能显示出显著的差异性，表明优化用于下一个令牌预测的特征并不一致地转移到下游感知任务。我们证明了通过自回归学习得到的表示捕获了可能位于对感知最具信息性的子空间之外的特征。为了量度自回归预训练与下游感知之间的（不）对齐程度，我们引入了下一个令牌感知分数（Next Token Perception Score, NTPS）—一种基于线性设置的得分，用于测量自回归特征子空间与感知特征子空间之间的重叠程度。该度量可以从预训练表示和标注数据中以闭式形式轻松计算得出，并且被证明能够严格地上下界超过损失。实验上，我们展示了NTPS与12个不同NLP数据集和8个不同规模的预训练模型（从2.7亿到80亿参数不等）上的线性探针准确率之间存在强烈的相关性，从而证实了其作为对齐度量的实用性。此外，我们显示了NTPS在低秩适应（LoRA）微调后增加，特别是在大型模型中，这表明LoRA将表示向感知任务对齐提高了子空间重叠，并因此改善了下游性能。更重要的是，我们发现NTPS可靠地预测了通过LoRA微调获得的额外准确率增益，因此提供了一种轻量级的LoRA适应前筛工具。我们的结果既提供了理论洞察，也为分析LLM感知能力提供了实用工具。 

---
# A Toolkit for Compliance, a Toolkit for Justice: Drawing on Cross-sectoral Expertise to Develop a Pro-justice EU AI Act Toolkit 

**Title (ZH)**: 合规工具箱，公正工具箱：整合跨界专家经验以制定有利于公正的欧盟AI法案工具箱 

**Authors**: Tomasz Hollanek, Yulu Pi, Cosimo Fiorini, Virginia Vignali, Dorian Peters, Eleanor Drage  

**Link**: [PDF](https://arxiv.org/pdf/2505.17165)  

**Abstract**: The introduction of the AI Act in the European Union presents the AI research and practice community with a set of new challenges related to compliance. While it is certain that AI practitioners will require additional guidance and tools to meet these requirements, previous research on toolkits that aim to translate the theory of AI ethics into development and deployment practice suggests that such resources suffer from multiple limitations. These limitations stem, in part, from the fact that the toolkits are either produced by industry-based teams or by academics whose work tends to be abstract and divorced from the realities of industry. In this paper, we discuss the challenge of developing an AI ethics toolkit for practitioners that helps them comply with new AI-focused regulation, but that also moves beyond mere compliance to consider broader socio-ethical questions throughout development and deployment. The toolkit was created through a cross-sectoral collaboration between an academic team based in the UK and an industry team in Italy. We outline the background and rationale for creating a pro-justice AI Act compliance toolkit, detail the process undertaken to develop it, and describe the collaboration and negotiation efforts that shaped its creation. We aim for the described process to serve as a blueprint for other teams navigating the challenges of academia-industry partnerships and aspiring to produce usable and meaningful AI ethics resources. 

**Abstract (ZH)**: 《欧洲联盟AI法案的引入为AI研究与实践社区带来了合规方面的全新挑战：开发兼顾合规与社会伦理的AI伦理工具箱及其跨领域合作过程》 

---
# OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning 

**Title (ZH)**: OCR推理基准：揭示MLLLMs在复杂图文推理中的真正能力 

**Authors**: Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17163)  

**Abstract**: Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. The benchmark comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Furthermore, unlike other text-rich image understanding benchmarks that only annotate the final answers, OCR-Reasoning also annotates the reasoning process simultaneously. With the annotated reasoning process and the final answers, OCR-Reasoning evaluates not only the final answers generated by models but also their reasoning processes, enabling a holistic analysis of their problem-solving abilities. Leveraging this benchmark, we conducted a comprehensive evaluation of state-of-the-art MLLMs. Our results demonstrate the limitations of existing methodologies. Notably, even state-of-the-art MLLMs exhibit substantial difficulties, with none achieving accuracy surpassing 50\% across OCR-Reasoning, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at this https URL. 

**Abstract (ZH)**: Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. 

---
# DailyQA: A Benchmark to Evaluate Web Retrieval Augmented LLMs Based on Capturing Real-World Changes 

**Title (ZH)**: DailyQA：基于捕捉实时变化评估网络检索增强的大语言模型基准 

**Authors**: Jiehan Cheng, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17162)  

**Abstract**: We propose DailyQA, an automatically updated dynamic dataset that updates questions weekly and contains answers to questions on any given date. DailyQA utilizes daily updates from Wikipedia revision logs to implement a fully automated pipeline of data filtering, query generation synthesis, quality checking, answer extraction, and query classification. The benchmark requires large language models (LLMs) to process and answer questions involving fast-changing factual data and covering multiple domains. We evaluate several open-source and closed-source LLMs using different RAG pipelines with web search augmentation. We compare the ability of different models to process time-sensitive web information and find that rerank of web retrieval results is critical. Our results indicate that LLMs still face significant challenges in handling frequently updated information, suggesting that DailyQA benchmarking provides valuable insights into the direction of progress for LLMs and RAG systems. 

**Abstract (ZH)**: DailyQA：一个每周自动更新的动态数据集，包含任意给定日期的问题及其答案 

---
# Harry Potter is Still Here! Probing Knowledge Leakage in Targeted Unlearned Large Language Models via Automated Adversarial Prompting 

**Title (ZH)**: 哈利·波特 still 在此！通过自动化 adversarial 提示探究目标未学习大语言模型中的知识泄露 

**Authors**: Bang Trinh Tran To, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.17160)  

**Abstract**: This work presents LURK (Latent UnleaRned Knowledge), a novel framework that probes for hidden retained knowledge in unlearned LLMs through adversarial suffix prompting. LURK automatically generates adversarial prompt suffixes designed to elicit residual knowledge about the Harry Potter domain, a commonly used benchmark for unlearning. Our experiments reveal that even models deemed successfully unlearned can leak idiosyncratic information under targeted adversarial conditions, highlighting critical limitations of current unlearning evaluation standards. By uncovering latent knowledge through indirect probing, LURK offers a more rigorous and diagnostic tool for assessing the robustness of unlearning algorithms. All code will be publicly available. 

**Abstract (ZH)**: This work presents LURK (Latent Unlearned Knowledge), 一种通过对抗后缀提示探求未训练语言模型中隐藏保留知识的新框架。LURK 自动生成旨在唤起有关哈利·波特领域残留知识的对抗后缀提示，该领域常被用作脱 *);

*润色以符合完整的中文表达：

This work presents LURK (Latent Unlearned Knowledge), 一种通过对抗后缀提示探求未训练语言模型中隐藏保留知识的新框架。LURK 自动生成旨在唤起有关哈利·波特领域残留知识的对抗后缀提示，该领域常被用作脱机学习基准。我们的实验表明，即使被认定成功脱机学习的模型，在针对性的对抗条件下也可能泄露独特的信息，突显了当前脱机学习评估标准的关键局限性。通过间接探求潜在知识，LURK 提供了一种更严格和诊断性的工具，用于评估脱机学习算法的 robustness。所有代码将公开发布。 

---
# TrimR: Verifier-based Training-Free Thinking Compression for Efficient Test-Time Scaling 

**Title (ZH)**: TrimR：基于验证器的无需训练的思考压缩以实现高效的测试时扩展 

**Authors**: Weizhe Lin, Xing Li, Zhiyuan Yang, Xiaojin Fu, Hui-Ling Zhen, Yaoyuan Wang, Xianzhi Yu, Wulong Liu, Xiaosong Li, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17155)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate exceptional capability in tackling complex mathematical, logical, and coding tasks by leveraging extended Chain-of-Thought (CoT) reasoning. Test-time scaling methods, such as prolonging CoT with explicit token-level exploration, can push LRMs' accuracy boundaries, but they incur significant decoding overhead. A key inefficiency source is LRMs often generate redundant thinking CoTs, which demonstrate clear structured overthinking and underthinking patterns. Inspired by human cognitive reasoning processes and numerical optimization theories, we propose TrimR, a verifier-based, training-free, efficient framework for dynamic CoT compression to trim reasoning and enhance test-time scaling, explicitly tailored for production-level deployment. Our method employs a lightweight, pretrained, instruction-tuned verifier to detect and truncate redundant intermediate thoughts of LRMs without any LRM or verifier fine-tuning. We present both the core algorithm and asynchronous online system engineered for high-throughput industrial applications. Empirical evaluations on Ascend NPUs and vLLM show that our framework delivers substantial gains in inference efficiency under large-batch workloads. In particular, on the four MATH500, AIME24, AIME25, and GPQA benchmarks, the reasoning runtime of Pangu-R-38B, QwQ-32B, and DeepSeek-R1-Distill-Qwen-32B is improved by up to 70% with negligible impact on accuracy. 

**Abstract (ZH)**: Large Reasoning Models (LRMs)通过利用扩展的链式思考（CoT）推理展示了在处理复杂数学、逻辑和编程任务方面的出色能力。通过延长CoT并进行显式的 token 级探索等测试时扩展方法可以提高 LRMs 的准确边界，但这些方法会产生显著的解码开销。主要的效率降低来源是 LRMs 经常生成冗余的思考 CoTs，这些 CoTs 展现出明显的结构化过度思考和思考不足的模式。受人类认知推理过程和数值优化理论的启发，我们提出了 TrimR，这是一种基于验证器的、无需训练的高效框架，用于动态压缩 CoT，以精简推理并增强测试时扩展能力，特别适用于生产级部署。该方法使用一个轻量级的、指令调优的预训练验证器来检测和截断 LRMs 的冗余中间思考，而无需对任何 LRMs 或验证器进行微调。我们介绍了适用于高吞吐量工业应用的核心算法和异步在线系统。在昇腾 NPUs 和 vLLM 上的实证评估表明，该框架在大规模批量工作中显著提高了推理效率。特别地，在 MATH500、AIME24、AIME25 和 GPQA 四个基准测试中，Pangu-R-38B、QwQ-32B 和 DeepSeek-R1-Distill-Qwen-32B 的推理运行时间最高可提高 70%，而对准确性的影响可以忽略不计。 

---
# Can Large Language Models Design Biological Weapons? Evaluating Moremi Bio 

**Title (ZH)**: 大型语言模型能否设计生物武器？评估Moremi Bio 

**Authors**: Gertrude Hattoh, Jeremiah Ayensu, Nyarko Prince Ofori, Solomon Eshun, Darlington Akogo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17154)  

**Abstract**: Advances in AI, particularly LLMs, have dramatically shortened drug discovery cycles by up to 40% and improved molecular target identification. However, these innovations also raise dual-use concerns by enabling the design of toxic compounds. Prompting Moremi Bio Agent without the safety guardrails to specifically design novel toxic substances, our study generated 1020 novel toxic proteins and 5,000 toxic small molecules. In-depth computational toxicity assessments revealed that all the proteins scored high in toxicity, with several closely matching known toxins such as ricin, diphtheria toxin, and disintegrin-based snake venom proteins. Some of these novel agents showed similarities with other several known toxic agents including disintegrin eristostatin, metalloproteinase, disintegrin triflavin, snake venom metalloproteinase, corynebacterium ulcerans toxin. Through quantitative risk assessments and scenario analyses, we identify dual-use capabilities in current LLM-enabled biodesign pipelines and propose multi-layered mitigation strategies. The findings from this toxicity assessment challenge claims that large language models (LLMs) are incapable of designing bioweapons. This reinforces concerns about the potential misuse of LLMs in biodesign, posing a significant threat to research and development (R&D). The accessibility of such technology to individuals with limited technical expertise raises serious biosecurity risks. Our findings underscore the critical need for robust governance and technical safeguards to balance rapid biotechnological innovation with biosecurity imperatives. 

**Abstract (ZH)**: AI进步，尤其是大语言模型，显著缩短了药物发现周期并提高了分子目标识别的效率，但同时也引发了双重用途的担忧，因为这些技术能够设计出有毒化合物。在缺乏安全防护的情况下，Moremi Bio Agent生成了1020种新型有毒蛋白质和5000种有毒小分子。深入的计算毒性评估显示，所有蛋白质均具有高毒性，其中一些与已知毒素如肉毒杆菌毒素、白喉毒素以及蛇毒蛋白酶类极为相似。通过定量风险评估和情景分析，我们发现当前大语言模型驱动的生物设计流程具有双重用途能力，并提出多层次的缓解策略。此次毒性评估结果挑战了大语言模型无法设计生物武器的观点，强化了对大语言模型在生物设计中潜在滥用的担忧，这对研究与开发构成了重大威胁。缺乏技术专长的人士可能能够获得此类技术，从而引发严重的生物安全风险。我们的研究强调了建立稳健治理和技术支持措施以平衡快速生物技术创新与生物安全需求的重要性。 

---
# Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN 

**Title (ZH)**: 放大相邻令牌差异：通过移位前馈网络增强长链条推理 

**Authors**: Yao Xu, Mingyu Xu, Fangyu Lei, Wangtao Sun, Xiangrong Zeng, Bingning Wang, Guang Liu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17153)  

**Abstract**: Recently, models such as OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable performance on complex reasoning tasks through Long Chain-of-Thought (Long-CoT) reasoning. Although distilling this capability into student models significantly enhances their performance, this paper finds that fine-tuning LLMs with full parameters or LoRA with a low rank on long CoT data often leads to Cyclical Reasoning, where models repeatedly reiterate previous inference steps until the maximum length limit. Further analysis reveals that smaller differences in representations between adjacent tokens correlates with a higher tendency toward Cyclical Reasoning. To mitigate this issue, this paper proposes Shift Feedforward Networks (Shift-FFN), a novel approach that edits the current token's representation with the previous one before inputting it to FFN. This architecture dynamically amplifies the representation differences between adjacent tokens. Extensive experiments on multiple mathematical reasoning tasks demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of Cyclical Reasoning across various data sizes compared to full fine-tuning and standard LoRA. Our data and code are available at this https URL 

**Abstract (ZH)**: 最近，OpenAI-o1和DeepSeek-R1等模型通过长链推理（Long-CoT）在复杂推理任务中表现出色。尽管将这种能力提炼到学生模型中显著提升了它们的性能，但本文发现，在长链推理数据上全参数微调或使用低秩LoRA微调往往会导致循环推理（Cyclical Reasoning），即模型重复迭代之前的推理步骤直至达到最大长度限制。进一步分析表明，相邻令牌间表示差异较小与更高的循环推理倾向相关。为缓解这一问题，本文提出了一种新的方法——Shift Feedforward Networks（Shift-FFN），该方法在将当前令牌输入前馈网络（FFN）之前，用前一个令牌的表示对其进行编辑，从而动态放大相邻令牌间的表示差异。多项数学推理任务的深入实验表明，与全参数微调和标准LoRA相比，结合使用Shift-FFN的LoRA在各种数据规模上实现了更高的准确率和更低的循环推理率。相关的数据和代码可在以下链接获取。 

---
# Bayesian Optimization for Enhanced Language Models: Optimizing Acquisition Functions 

**Title (ZH)**: 基于贝叶斯优化的语言模型增强：优化获取函数 

**Authors**: Zishuo Bao, Yibo Liu, Changyutao Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17151)  

**Abstract**: With the rise of different language model architecture, fine-tuning is becoming even more important for down stream tasks Model gets messy, finding proper hyperparameters for fine-tuning. Although BO has been tried for hyperparameter tuning, most of the existing methods are oblivious to the fact that BO relies on careful choices of acquisition functions, which are essential components of BO that guide how much to explore versus exploit during the optimization process; Different acquisition functions have different levels of sensitivity towards training loss and validation performance; existing methods often just apply an acquisition function no matter if the training and validation performance are sensitive to the acquisition function or not. This work introduces{Bilevel - BO - SWA}, a model fusion approach coupled with a bilevel BO strategy to improve the fine - tunning of large language models. Our work on mixture of acquisition functions like EI and UCB into nested opt loops, where inner loop perform minimization of training loss while outer loops optimized w.r.t. val metric. Experiments on GLUE tasks using RoBERTA - base show that when using EI and UCB, there is an improvement in generalization, and fine - tuning can be improved by up to 2.7%. 

**Abstract (ZH)**: 不同语言模型架构的兴起使得下游任务的微调愈加重要，模型变得复杂化，寻找适合的超参数变得困难。虽然已经尝试了基于贝叶斯优化（BO）的超参数调优方法，但大多数现有方法忽略了BO依赖于精心选择的获取函数的事实，这些获取函数是BO的核心组件，能够引导优化过程中的探索与利用平衡；不同的获取函数对于训练损失和验证性能的敏感程度不同；现有方法往往在没有考虑训练和验证性能对获取函数敏感性的情况下就直接应用获取函数。本文介绍了一种结合双层贝叶斯优化策略的模型融合方法——Bilevel-BO-SWA，旨在改善大规模语言模型的微调。实验结果表明，将EI和UCB等获取函数嵌入嵌套优化循环中，可以在GLUE任务中通过RoBERTA-base模型实现更好的泛化性能，微调效果可提高高达2.7%。 

---
# Efficient Training of Neural SDEs Using Stochastic Optimal Control 

**Title (ZH)**: 高效训练神经SDE模型的随机最优控制方法 

**Authors**: Rembert Daems, Manfred Opper, Guillaume Crevecoeur, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2505.17150)  

**Abstract**: We present a hierarchical, control theory inspired method for variational inference (VI) for neural stochastic differential equations (SDEs). While VI for neural SDEs is a promising avenue for uncertainty-aware reasoning in time-series, it is computationally challenging due to the iterative nature of maximizing the ELBO. In this work, we propose to decompose the control term into linear and residual non-linear components and derive an optimal control term for linear SDEs, using stochastic optimal control. Modeling the non-linear component by a neural network, we show how to efficiently train neural SDEs without sacrificing their expressive power. Since the linear part of the control term is optimal and does not need to be learned, the training is initialized at a lower cost and we observe faster convergence. 

**Abstract (ZH)**: 基于控制理论的层次化变分推断方法用于神经随机微分方程 

---
# Large Language Models for Predictive Analysis: How Far Are They? 

**Title (ZH)**: 大型语言模型在预测分析中的应用：它们的发展距离何处？ 

**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Yuyang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17149)  

**Abstract**: Predictive analysis is a cornerstone of modern decision-making, with applications in various domains. Large Language Models (LLMs) have emerged as powerful tools in enabling nuanced, knowledge-intensive conversations, thus aiding in complex decision-making tasks. With the burgeoning expectation to harness LLMs for predictive analysis, there is an urgent need to systematically assess their capability in this domain. However, there is a lack of relevant evaluations in existing studies. To bridge this gap, we introduce the \textbf{PredictiQ} benchmark, which integrates 1130 sophisticated predictive analysis queries originating from 44 real-world datasets of 8 diverse fields. We design an evaluation protocol considering text analysis, code generation, and their alignment. Twelve renowned LLMs are evaluated, offering insights into their practical use in predictive analysis. Generally, we believe that existing LLMs still face considerable challenges in conducting predictive analysis. See \href{this https URL}{Github}. 

**Abstract (ZH)**: 预测分析是现代决策制定的核心，广泛应用于各个领域。大规模语言模型（LLMs）已成为促进精细、知识密集型对话的强大工具，从而协助复杂决策任务。随着对利用LLMs进行预测分析的期望日益增长，系统评估其在此领域的能力变得尤为紧迫。然而，现有研究中缺乏相关评估。为填补这一空白，我们引入了PredictiQ基准，该基准整合了来自44个真实世界数据集的1130个复杂的预测分析查询，涵盖8个不同领域。我们设计了一项评估协议，考虑了文本分析、代码生成及其对齐。十二种知名的大规模语言模型接受了评估，提供了它们在预测分析中的实用性的洞见。总体而言，我们相信现有的大规模语言模型在进行预测分析方面仍面临诸多挑战。参见Github：this https URL。 

---
# LLM-Powered Agents for Navigating Venice's Historical Cadastre 

**Title (ZH)**: LLM赋能的代理人在探索威尼斯历史地籍中的应用 

**Authors**: Tristan Karch, Jakhongir Saydaliev, Isabella Di Lenardo, Frédéric Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17148)  

**Abstract**: Cadastral data reveal key information about the historical organization of cities but are often non-standardized due to diverse formats and human annotations, complicating large-scale analysis. We explore as a case study Venice's urban history during the critical period from 1740 to 1808, capturing the transition following the fall of the ancient Republic and the Ancien Régime. This era's complex cadastral data, marked by its volume and lack of uniform structure, presents unique challenges that our approach adeptly navigates, enabling us to generate spatial queries that bridge past and present urban landscapes. We present a text-to-programs framework that leverages Large Language Models (LLMs) to translate natural language queries into executable code for processing historical cadastral records. Our methodology implements two complementary techniques: a text-to-SQL approach for handling structured queries about specific cadastral information, and a text-to-Python approach for complex analytical operations requiring custom data manipulation. We propose a taxonomy that classifies historical research questions based on their complexity and analytical requirements, mapping them to the most appropriate technical approach. This framework is supported by an investigation into the execution consistency of the system, alongside a qualitative analysis of the answers it produces. By ensuring interpretability and minimizing hallucination through verifiable program outputs, we demonstrate the system's effectiveness in reconstructing past population information, property features, and spatiotemporal comparisons in Venice. 

**Abstract (ZH)**: cadastral数据揭示了城市历史组织的关键信息，但由于格式多样和人工标注不统一，往往难以标准化，这给大规模分析带来了复杂性。我们以1740年至1808年威尼斯的城市历史为例，探讨了从古代共和国和旧制度崩塌后的过渡时期复杂且不统一的 cadastral 数据所带来的独特挑战，我们的方法巧妙地应对了这些挑战，能够生成连接过去和现在的空间查询。我们提出了一种基于自然语言查询将文本转化为程序的框架，利用大型语言模型（LLMs）将自然语言查询转换为处理历史 cadastral 记录的可执行代码。我们的方法实施了两种互补的技术：一种是将文本转化为SQL以处理具体 cadastral 信息的结构化查询，另一种是将文本转化为Python以进行需要自定义数据操作的复杂分析操作。我们提出了一个分类法，根据历史研究问题的复杂性和分析需求将其分类，并将其映射到最合适的技术方法。该框架结合了系统执行一致性调查和对生成答案的质性分析支持。通过确保程序输出的可验证性和最小化幻觉，我们展示了该系统在重建威尼斯过去的居民信息、财产特征以及时空对比方面的有效性。 

---
# MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming 

**Title (ZH)**: MTSA：通过多轮红队演练实现LLM的多轮安全对齐 

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17147)  

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks. 

**Abstract (ZH)**: 针对大规模语言模型多轮对话中 Jailbreak 攻击 proliferating 的需求，提出多轮安全对齐 (\ourapproach) 框架以确保安全性 

---
# LLM Access Shield: Domain-Specific LLM Framework for Privacy Policy Compliance 

**Title (ZH)**: LLM接入防护罩：针对隐私政策合规的领域特定大语言模型框架 

**Authors**: Yu Wang, Cailing Cai, Zhihua Xiao, Peifung E. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.17145)  

**Abstract**: Large language models (LLMs) are increasingly applied in fields such as finance, education, and governance due to their ability to generate human-like text and adapt to specialized tasks. However, their widespread adoption raises critical concerns about data privacy and security, including the risk of sensitive data exposure.
In this paper, we propose a security framework to enforce policy compliance and mitigate risks in LLM interactions. Our approach introduces three key innovations: (i) LLM-based policy enforcement: a customizable mechanism that enhances domain-specific detection of sensitive data. (ii) Dynamic policy customization: real-time policy adaptation and enforcement during user-LLM interactions to ensure compliance with evolving security requirements. (iii) Sensitive data anonymization: a format-preserving encryption technique that protects sensitive information while maintaining contextual integrity. Experimental results demonstrate that our framework effectively mitigates security risks while preserving the functional accuracy of LLM-driven tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其生成人类文本和适应专门任务的能力而在金融、教育和治理等领域得到广泛应用。然而，它们的广泛应用引发了关于数据隐私和安全的关键关注，包括敏感数据泄露的风险。

本文提出了一种安全框架，以确保LLM交互中的策略合规并降低风险。我们的方法引入了三个关键创新：（i）基于LLM的策略执行：一种可定制的机制，增强特定领域的敏感数据检测。（ii）动态策略定制：在用户-LLM交互过程中实时策略适应和执行，以确保满足不断变化的安全要求。（iii）敏感数据匿名化：一种格式保留加密技术，在保护敏感信息的同时保持上下文完整性。实验结果表明，该框架在降低安全风险的同时保持了LLM驱动任务的功能准确性。 

---
# MDIT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models 

**Title (ZH)**: MDIT-Bench: 评估大型多模态模型中的双重隐式毒性 

**Authors**: Bohan Jin, Shuhan Qi, Kehai Chen, Xinyi Guo, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17144)  

**Abstract**: The widespread use of Large Multimodal Models (LMMs) has raised concerns about model toxicity. However, current research mainly focuses on explicit toxicity, with less attention to some more implicit toxicity regarding prejudice and discrimination. To address this limitation, we introduce a subtler type of toxicity named dual-implicit toxicity and a novel toxicity benchmark termed MDIT-Bench: Multimodal Dual-Implicit Toxicity Benchmark. Specifically, we first create the MDIT-Dataset with dual-implicit toxicity using the proposed Multi-stage Human-in-loop In-context Generation method. Based on this dataset, we construct the MDIT-Bench, a benchmark for evaluating the sensitivity of models to dual-implicit toxicity, with 317,638 questions covering 12 categories, 23 subcategories, and 780 topics. MDIT-Bench includes three difficulty levels, and we propose a metric to measure the toxicity gap exhibited by the model across them. In the experiment, we conducted MDIT-Bench on 13 prominent LMMs, and the results show that these LMMs cannot handle dual-implicit toxicity effectively. The model's performance drops significantly in hard level, revealing that these LMMs still contain a significant amount of hidden but activatable toxicity. Data are available at this https URL. 

**Abstract (ZH)**: 大规模多模态模型（LMMs）的广泛应用引发了对模型毒性问题的关注。然而，当前研究主要集中在显式毒性上，对于有关偏见和歧视的更隐含的毒性关注较少。为解决这一不足，我们引入了一种更为微妙的毒性类型——双重隐含毒性，并提出了一种新颖的毒性基准——MDIT-Bench：多模态双重隐含毒性基准。具体而言，我们首先通过提出的多阶段人工环路上下文生成方法创建了MDIT-数据集，以涵盖双重隐含毒性。基于此数据集，我们构建了MDIT-Bench，这是一个用于评估模型对双重隐含毒性敏感性的基准，包含317638个问题，覆盖12个类别、23个子类别和780个主题。MDIT-Bench包括三个难度级别，并提出了一种测量模型在不同难度级别上表现出的毒性差距的指标。在实验中，我们对13种 prominent LMMs 进行了MDIT-Bench测试，结果显示这些模型不能有效处理双重隐含毒性。这些模型在高难度级别上的性能显著下降，揭示了这些模型仍然含有相当数量的隐藏但可激活的毒性。相关数据可在此网址获取。 

---
# Evaluating the Performance of Nigerian Lecturers using Multilayer Perceptron 

**Title (ZH)**: 评价尼日利亚 lecturer 性能的多层感知机方法 

**Authors**: I.E. Ezeibe, S.O. Okide, D.C. Asogwa  

**Link**: [PDF](https://arxiv.org/pdf/2505.17143)  

**Abstract**: Evaluating the performance of a lecturer has been essential for enhancing teaching quality, improving student learning outcomes, and strengthening the institution's reputation. The absence of such a system brings about lecturer performance evaluation which was neither comprehensive nor holistic. This system was designed using a web-based platform, created a secure database, and by using a custom dataset, captured some performance metrics which included student evaluation scores, Research Publications, Years of Experience, and Administrative Duties. Multilayer Perceptron (MLP) algorithm was utilized due to its ability to process complex data patterns and generates accurate predictions in a lecturer's performance based on historical data. This research focused on designing multiple performance metrics beyond the standard ones, incorporating student participation, and integrating analytical tools to deliver a comprehensive and holistic evaluation of lecturers' performance and was developed using Object-Oriented Analysis and Design (OOAD) methodology. Lecturers' performance is evaluated by the model, and the evaluation accuracy is about 91% compared with actual performance. Finally, by evaluating the performance of the MLP model, it is concluded that MLP enhanced lecturer performance evaluation by providing accurate predictions, reducing bias, and supporting data-driven decisions, ultimately improving the fairness and efficiency of the evaluation process. The MLP model's performance was evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE), achieved a test loss (MSE) of 256.99 and a MAE of 13.76, and reflected a high level of prediction accuracy. The model also demonstrated an estimated accuracy rate of approximately 96%, validated its effectiveness in predicting lecturer performance. 

**Abstract (ZH)**: 评价讲师绩效对于提升教学质量和增强机构声誉至关重要。缺乏这种系统的评价导致绩效评价既不全面也不完整。该系统利用基于Web的平台设计，创建了安全的数据库，并使用自定义数据集捕捉了包括学生评价分数、研究出版物、工作经验年限和行政职责在内的绩效指标。多层感知器（MLP）算法被采用，因其能够处理复杂的数据模式并根据历史数据生成准确的预测。该研究重点在于设计超越标准绩效指标的多个指标，融入学生参与度，并集成分析工具，以提供全面和整体的讲师绩效评价，并使用面向对象分析与设计（OOAD）方法论进行开发。模型对讲师绩效的评价准确率为约91%，最终评价结果显示，MLP通过提供精确预测、减少偏见和支持基于数据的决策，提升了评价过程的公平性和效率。MLP模型的性能通过均方误差（MSE）和平均绝对误差（MAE）进行评估，测试损失（MSE）为256.99，MAE为13.76，显示出高度的预测准确性，并验证了其在预测讲师绩效方面有效性，模型估计准确率为约96%。 

---
# MetaSTH-Sleep: Towards Effective Few-Shot Sleep Stage Classification with Spatial-Temporal Hypergraph Enhanced Meta-Learning 

**Title (ZH)**: MetaSTH-Sleep：基于空间-时间超图增强元学习的高效少量样本睡眠阶段分类 

**Authors**: Jingyu Li, Tiehua Zhang, Jinze Wang, Yi Zhang, Yuhuan Li, Yifan Zhao, Zhishu Shen, Jiannan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17142)  

**Abstract**: Accurate classification of sleep stages based on bio-signals is fundamental for automatic sleep stage annotation. Traditionally, this task relies on experienced clinicians to manually annotate data, a process that is both time-consuming and labor-intensive. In recent years, deep learning methods have shown promise in automating this task. However, three major challenges remain: (1) deep learning models typically require large-scale labeled datasets, making them less effective in real-world settings where annotated data is limited; (2) significant inter-individual variability in bio-signals often results in inconsistent model performance when applied to new subjects, limiting generalization; and (3) existing approaches often overlook the high-order relationships among bio-signals, failing to simultaneously capture signal heterogeneity and spatial-temporal dependencies. To address these issues, we propose MetaSTH-Sleep, a few-shot sleep stage classification framework based on spatial-temporal hypergraph enhanced meta-learning. Our approach enables rapid adaptation to new subjects using only a few labeled samples, while the hypergraph structure effectively models complex spatial interconnections and temporal dynamics simultaneously in EEG signals. Experimental results demonstrate that MetaSTH-Sleep achieves substantial performance improvements across diverse subjects, offering valuable insights to support clinicians in sleep stage annotation. 

**Abstract (ZH)**: 基于时空超图增强元学习的少量样本睡眠分期分类方法（MetaSTH-Sleep） 

---
# Fashion Industry in the Age of Generative Artificial Intelligence and Metaverse: A systematic Review 

**Title (ZH)**: 时尚行业在生成式人工智能和元宇宙时代的现状：一项系统性综述 

**Authors**: Rania Ahmed, Eman Ahmed, Ahmed Elbarbary, Ashraf Darwish, Aboul Ella Hassanien  

**Link**: [PDF](https://arxiv.org/pdf/2505.17141)  

**Abstract**: The fashion industry is an extremely profitable market that generates trillions of dollars in revenue by producing and distributing apparel, footwear, and accessories. This systematic literature review (SLR) seeks to systematically review and analyze the research landscape about the Generative Artificial Intelligence (GAI) and metaverse in the fashion industry. Thus, investigating the impact of integrating both technologies to enhance the fashion industry. This systematic review uses the Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) methodology, including three essential phases: identification, evaluation, and reporting. In the identification phase, the target search problems are determined by selecting appropriate keywords and alternative synonyms. After that 578 documents from 2014 to the end of 2023 are retrieved. The evaluation phase applies three screening steps to assess papers and choose 118 eligible papers for full-text reading. Finally, the reporting phase thoroughly examines and synthesizes the 118 eligible papers to identify key themes associated with GAI and Metaverse in the fashion industry. Based on Strengths, Weaknesses, Opportunities, and Threats (SWOT) analyses performed for both GAI and metaverse for the fashion industry, it is concluded that the integration of GAI and the metaverse holds the capacity to profoundly revolutionize the fashion sector, presenting chances for improved manufacturing, design, sales, and client experiences. Accordingly, the research proposes a new framework to integrate GAI and metaverse to enhance the fashion industry. The framework presents different use cases to promote the fashion industry using the integration. Future research points for achieving a successful integration are demonstrated. 

**Abstract (ZH)**: 时尚行业是一个极其有利可图的市场，通过生产和分配服装、鞋类和配饰产生万亿美元的收入。本文系统文献综述（SLR）旨在系统地回顾和分析有关生成式人工智能（GAI）和元宇宙在时尚行业的研究 landscape，从而探讨将两者技术整合以增强时尚行业的影响。本系统综述采用系统评价和元分析报告项目（PRISMA）方法，包括三个关键阶段：识别、评估和报告。在识别阶段，通过选择适当的关键词和同义词来确定目标搜索问题。随后，从2014年至2023年底检索到578篇文献。评估阶段应用了三个筛选步骤来评估论文并选择118篇合格论文进行全文阅读。最后，报告阶段详细检查并综合了118篇合格论文，以识别与GAI和元宇宙在时尚行业相关的关键主题。根据对GAI和元宇宙在时尚行业的优势、劣势、机会和威胁（SWOT）分析，得出结论认为将GAI与元宇宙整合有望深刻变革时尚产业，为制造、设计、销售和客户体验的提升提供机会。因此，本文提出了一种新框架以整合GAI和元宇宙来增强时尚行业。该框架提出了不同的应用场景以通过整合推动时尚产业的发展。展示了实现成功整合的未来研究方向。 

---
# Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs 

**Title (ZH)**: 数据掺假还是真正智能？评估注入知识在LLMs中的迁移性 

**Authors**: Essa Jan, Moiz Ali, Muhammad Saram Hassan, Fareed Zaffar, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.17140)  

**Abstract**: As the knowledge of large language models (LLMs) becomes outdated over time, there is a growing need for efficient methods to update them, especially when injecting proprietary information. Our study reveals that comprehension-intensive fine-tuning tasks (e.g., question answering and blanks) achieve substantially higher knowledge retention rates (48%) compared to mapping-oriented tasks like translation (17%) or text-to-JSON conversion (20%), despite exposure to identical factual content. We demonstrate that this pattern persists across model architectures and follows scaling laws, with larger models showing improved retention across all task types. However, all models exhibit significant performance drops when applying injected knowledge in broader contexts, suggesting limited semantic integration. These findings show the importance of task selection in updating LLM knowledge, showing that effective knowledge injection relies not just on data exposure but on the depth of cognitive engagement during fine-tuning. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的知识随着时间变得过时，高效的方法来更新它们变得日益重要，尤其是在注入专有信息时。我们的研究发现，理解密集型细调任务（如问答和填空）的知识留存率（48%）远高于映射导向任务（如翻译，17%或文本到JSON转换，20%）的知识留存率，尽管这些任务接触到的是相同的事实内容。研究显示，这一模式在不同的模型架构中持续存在，并遵循规模定律，更大的模型在所有任务类型中展现出更好的知识留存。然而，所有模型在将注入的知识应用于更广泛的情境时均表现出显著的性能下降，这表明知识的语义整合有限。这些发现强调了在更新LLM知识时选择任务的重要性，表明有效的知识注入不仅依赖于数据暴露，还依赖于细调过程中认知参与的深度。 

---
# EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models 

**Title (ZH)**: EarthSE: 一种评估大型语言模型地球科学探索能力的基准 

**Authors**: Wanghan Xu, Xiangyu Zhao, Yuhao Zhou, Xiaoyu Yue, Ben Fei, Fenghua Ling, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17139)  

**Abstract**: Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）的进步推动了对科学应用的兴趣， necessitating 专门的基准测试，如地球科学领域的基准测试。现有基准测试要么缺乏地球科学的具体性，要么仅涵盖孤立的子领域，缺乏全面评估。此外，当前的基准测试通常忽略了对LLM在开放性科学探索方面能力的评估。在本文中，我们提出了一项全面而专业的地球科学基准测试，旨在评估LLM在该领域科学探索方面的能力，涵盖从基础到高级的各个层面。利用10万篇研究论文的语料库，我们首先构建了两个问答（QA）数据集：Earth-Iron，提供广泛的提问范围以实现全面评估；Earth-Silver，具备更高的难度，以评估专业深度。这些数据集覆盖了五个地球层、114个学科和11个任务类别，评估了对于科学探索至关重要的基础知识。尤为值得一提的是，我们引入了Earth-Gold数据集，该数据集包含新的指标，专门设计用于评估LLM在科学探索方面的高级能力，包括方法论归纳、局限性分析和概念提案。大量实验揭示了11种领先LLM在不同领域和任务中的局限性，突显了其在科学探索能力方面改进的巨大空间。基准测试可在以下网址获取：this https URL。 

---
# RAP: Runtime-Adaptive Pruning for LLM Inference 

**Title (ZH)**: 运行时自适应剪枝 for LLM推理 

**Authors**: Huanrong Liu, Chunlin Tian, Xuyang Wei, Jiaheng Dai, Qin Liu, Tianqi Wei, Qingbiao Li, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17138)  

**Abstract**: Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly. 

**Abstract (ZH)**: Large语言模型（LLMs）在语言理解和生成方面表现出色，但其巨大的计算和内存需求妨碍了其部署。压缩提供了缓解这些限制的潜在解决方案。然而，现有方法大多依赖于固定的启发式方法，无法适应运行时内存变化或来自多样用户请求的异构KV缓存需求。为了克服这些局限，我们提出了一种基于强化学习（RL）的弹性剪枝框架RAP，该框架能够动态调整压缩策略以适应运行时需求。具体而言，RAP动态跟踪模型参数与KV缓存之间在实际执行过程中的变化比率。由于FFNs占据了大部分参数，而轻量级参数的注意力层主导了KV缓存的形成，RL代理仅保留那些在当前内存预算下最大化其效用的组件，这些组件的保留受即时负载和设备状态的影响。广泛的实验结果表明，RAP在模型权重和KV缓存上同时取得了最优性能，这是首次在运行时联合考虑这两者。 

---
# Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands 

**Title (ZH)**: Cog-TiPRO：通过 longitudinal 语音助手命令进行认知衰退检测的迭代提示精炼方法（使用大语言模型） 

**Authors**: Kristin Qi, Youxiang Zhu, Caroline Summerour, John A. Batsis, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17137)  

**Abstract**: Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline. 

**Abstract (ZH)**: 早期认知衰退的检测对于减缓神经退行性疾病进展的干预至关重要。传统诊断方法依赖于劳动密集型的临床评估，这不适合频繁监测。我们的试点研究探讨了语音助手系统（VAS）作为通过纵向分析语音命令的语音模式来检测认知衰退的非侵入性工具。在18个月的时间里，我们从35名老年人中收集了语音命令，其中15名参与者每天进行家庭中的VAS交互。为了解决分析这些短、无结构和嘈杂命令的挑战，我们提出了一种结合了(1)基于大规模语言模型的迭代提示 refinement 以提取语言特征、(2) HuBERT 基础的声音特征提取，以及(3) 基于变压器的时间模型的 Cog-TiPRO 框架。通过使用 iTransformer，我们的方法在检测MCI方面的准确率达到了73.80%，F1分数为72.67%，优于基线方法27.13%。通过我们的大规模语言模型方法，我们确定了独特的语言特征，这些特征表征了认知衰退个体日常命令使用模式。 

---
# Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations 

**Title (ZH)**: 地理空间推理的基础模型：大型语言模型在理解几何结构和拓扑空间关系方面的能力评估 

**Authors**: Yuhan Ji, Song Gao, Ying Nie, Ivan Majić, Krzysztof Janowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17136)  

**Abstract**: Applying AI foundation models directly to geospatial datasets remains challenging due to their limited ability to represent and reason with geographical entities, specifically vector-based geometries and natural language descriptions of complex spatial relations. To address these issues, we investigate the extent to which a well-known-text (WKT) representation of geometries and their spatial relations (e.g., topological predicates) are preserved during spatial reasoning when the geospatial vector data are passed to large language models (LLMs) including GPT-3.5-turbo, GPT-4, and DeepSeek-R1-14B. Our workflow employs three distinct approaches to complete the spatial reasoning tasks for comparison, i.e., geometry embedding-based, prompt engineering-based, and everyday language-based evaluation. Our experiment results demonstrate that both the embedding-based and prompt engineering-based approaches to geospatial question-answering tasks with GPT models can achieve an accuracy of over 0.6 on average for the identification of topological spatial relations between two geometries. Among the evaluated models, GPT-4 with few-shot prompting achieved the highest performance with over 0.66 accuracy on topological spatial relation inference. Additionally, GPT-based reasoner is capable of properly comprehending inverse topological spatial relations and including an LLM-generated geometry can enhance the effectiveness for geographic entity retrieval. GPT-4 also exhibits the ability to translate certain vernacular descriptions about places into formal topological relations, and adding the geometry-type or place-type context in prompts may improve inference accuracy, but it varies by instance. The performance of these spatial reasoning tasks offers valuable insights for the refinement of LLMs with geographical knowledge towards the development of geo-foundation models capable of geospatial reasoning. 

**Abstract (ZH)**: 直接将地理空间数据集应用于AI基础模型仍具有挑战性，因为这些模型在代表和推理地理实体（特别是基于矢量的几何形状和复杂空间关系的自然语言描述）方面能力有限。为了解决这些问题，我们探讨了地理空间向量数据在传递给包括GPT-3.5-turbo、GPT-4和DeepSeek-R1-14B在内的大型语言模型（LLMs）时，其几何形状及其空间关系（例如拓扑谓词）的WKT表示在空间推理过程中的保留程度。我们的工作流采用三种不同的方法来完成空间推理任务进行比较，即基于几何嵌入的方法、基于提示工程的方法和基于日常语言的方法。实验结果表明，基于嵌入和基于提示工程的方法在使用GPT模型进行地理空间问答任务时，可以实现超过0.6的准确率，用于识别两个几何形状之间的拓扑空间关系。在评估的模型中，带有少量提示的GPT-4在拓扑空间关系推理中的表现最佳，准确率超过0.66。此外，基于GPT的空间推理器能够正确理解反向拓扑空间关系，包含由LLM生成的几何形状可以增强地理实体检索的有效性。GPT-4还表现出将某些关于地点的日常描述翻译成正式的拓扑关系的能力，同时在提示中添加几何类型或地点类型的上下文可能提高推理准确率，但效果因实例而异。这些空间推理任务的性能为如何完善具有地理知识的LLMs以开发能够进行地理空间推理的地理基础模型提供了有价值的见解。 

---
# LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions 

**Title (ZH)**: LongMagpie：一种生成大规模长上下文指令的自合成方法 

**Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17134)  

**Abstract**: High-quality long-context instruction data is essential for aligning long-context large language models (LLMs). Despite the public release of models like Qwen and Llama, their long-context instruction data remains proprietary. Human annotation is costly and challenging, while template-based synthesis methods limit scale, diversity, and quality. We introduce LongMagpie, a self-synthesis framework that automatically generates large-scale long-context instruction data. Our key insight is that aligned long-context LLMs, when presented with a document followed by special tokens preceding a user turn, auto-regressively generate contextually relevant queries. By harvesting these document-query pairs and the model's responses, LongMagpie produces high-quality instructions without human effort. Experiments on HELMET, RULER, and Longbench v2 demonstrate that LongMagpie achieves leading performance on long-context tasks while maintaining competitive performance on short-context tasks, establishing it as a simple and effective approach for open, diverse, and scalable long-context instruction data synthesis. 

**Abstract (ZH)**: 高质量的大语境指令数据对于对齐大语言模型（LLMs）的长语境至关重要。尽管模型如Qwen和Llama已经公开发布，但其长语境指令数据仍然保持私有。人工标注成本高且具有挑战性，而基于模板的合成方法限制了规模、多样性和质量。我们提出了LongMagpie，这是一个自动合成大规模长语境指令数据的自合成框架。我们的核心洞察是，当展示了文档并伴有特定标记前的用户回合时，对齐的长语境LLMs能够自回归地生成与上下文相关的问题。通过收集这些文档-查询对以及模型的响应，LongMagpie无需人工努力即可生成高质量的指令。实验表明，LongMagpie在长语境任务上达到了领先性能，同时在短语境任务上保持竞争性性能，确立了它作为一种简单有效的开放、多样和可扩展的长语境指令数据合成方法的地位。 

---
# Learning Probabilities of Causation from Finite Population Data 

**Title (ZH)**: 从有限总体数据中学习因果概率 

**Authors**: Shuai Wang, Song Jiang, Yizhou Sun, Judea Pearl, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17133)  

**Abstract**: Probabilities of causation play a crucial role in modern decision-making. This paper addresses the challenge of predicting probabilities of causation for subpopulations with \textbf{insufficient} data using machine learning models. Tian and Pearl first defined and derived tight bounds for three fundamental probabilities of causation: the probability of necessity and sufficiency (PNS), the probability of sufficiency (PS), and the probability of necessity (PN). However, estimating these probabilities requires both experimental and observational distributions specific to each subpopulation, which are often unavailable or impractical to obtain with limited population-level data. Therefore, for most subgroups, the amount of data they have is not enough to guarantee the accuracy of their probabilities. Hence, to estimate these probabilities for subpopulations with \textbf{insufficient} data, we propose using machine learning models that draw insights from subpopulations with sufficient data. Our evaluation of multiple machine learning models indicates that, given the population-level data and an appropriate choice of machine learning model and activation function, PNS can be effectively predicted. Through simulation studies on multiple Structured Causal Models (SCMs), we show that our multilayer perceptron (MLP) model with the Mish activation function achieves a mean absolute error (MAE) of approximately $0.02$ in predicting PNS for $32,768$ subpopulations across most SCMs using data from only $2,000$ subpopulations with known PNS values. 

**Abstract (ZH)**: 因果概率在现代决策中起着关键作用。本文探讨了使用机器学习模型预测数据不足子群体的因果概率的挑战。田和佩尔首先定义并推导了三种基本的因果概率：必要性和充分性概率（PNS）、充分性概率（PS）和必要性概率（PN）的确切界。然而，估计这些概率需要针对每个子群体的具体实验分布和观察分布，这些分布在限于总体数据时往往 unavailable 或难以获得。因此，对于大多数子群体，它们的数据量不足以保证这些概率的准确性。因此，为了预测数据不足子群体的因果概率，我们提出了利用数据充足子群体的机器学习模型来获取见解的方法。通过对多种机器学习模型的评估表明，在给定总体数据和适当的机器学习模型及激活函数选择下，PNS 可以有效预测。通过对多个结构化因果模型（SCM）进行模拟研究，我们展示了使用仅来自 2,000 个包含 PNS 值的子群体的数据，我们的具有 Mish 激活函数的多层感知机（MLP）模型在预测 32,768 个子群体的 PNS 时的平均绝对误差（MAE）约为 0.02。 

---
# Relative Bias: A Comparative Framework for Quantifying Bias in LLMs 

**Title (ZH)**: 相对偏差：量化LLMs中偏差的比较框架 

**Authors**: Alireza Arbabi, Florian Kerschbaum  

**Link**: [PDF](https://arxiv.org/pdf/2505.17131)  

**Abstract**: The growing deployment of large language models (LLMs) has amplified concerns regarding their inherent biases, raising critical questions about their fairness, safety, and societal impact. However, quantifying LLM bias remains a fundamental challenge, complicated by the ambiguity of what "bias" entails. This challenge grows as new models emerge rapidly and gain widespread use, while introducing potential biases that have not been systematically assessed. In this paper, we propose the Relative Bias framework, a method designed to assess how an LLM's behavior deviates from other LLMs within a specified target domain. We introduce two complementary methodologies: (1) Embedding Transformation analysis, which captures relative bias patterns through sentence representations over the embedding space, and (2) LLM-as-a-Judge, which employs a language model to evaluate outputs comparatively. Applying our framework to several case studies on bias and alignment scenarios following by statistical tests for validation, we find strong alignment between the two scoring methods, offering a systematic, scalable, and statistically grounded approach for comparative bias analysis in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）部署规模的扩大加剧了对其固有偏见的关注，引发了对其公平性、安全性和社会影响的关键问题。然而，量化LLM偏见仍然是一个基本挑战，因为“偏见”包含的含义具有模糊性。随着新模型的快速 emergence 和广泛应用，可能引入尚未系统评估的偏见，使这一挑战更加复杂。本文提出相对偏见框架，一种用于评估LLM行为与其在指定目标域内的其他LLM偏差的方法。我们引入了两种互补的方法：1）嵌入变换分析，通过句子嵌入空间中的表示来捕捉相对偏见模式；2）LLM作为法官，利用语言模型进行比较评价。通过对几个偏见和对齐场景的案例研究进行统计检验，我们发现两种评分方法之间存在强烈的对齐，提供了一种系统、可扩展且基于统计的方法，用于LLM中的相对偏见分析。 

---
# NEXT-EVAL: Next Evaluation of Traditional and LLM Web Data Record Extraction 

**Title (ZH)**: NEXT-EVAL: 传统与大语言模型网络数据记录抽取的下一波评估 

**Authors**: Soyeon Kim, Namhee Kim, Yeonwoo Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17125)  

**Abstract**: Effective evaluation of web data record extraction methods is crucial, yet hampered by static, domain-specific benchmarks and opaque scoring practices. This makes fair comparison between traditional algorithmic techniques, which rely on structural heuristics, and Large Language Model (LLM)-based approaches, offering zero-shot extraction across diverse layouts, particularly challenging. To overcome these limitations, we introduce a concrete evaluation framework. Our framework systematically generates evaluation datasets from arbitrary MHTML snapshots, annotates XPath-based supervision labels, and employs structure-aware metrics for consistent scoring, specifically preventing text hallucination and allowing only for the assessment of positional hallucination. It also incorporates preprocessing strategies to optimize input for LLMs while preserving DOM semantics: HTML slimming, Hierarchical JSON, and Flat JSON. Additionally, we created a publicly available synthetic dataset by transforming DOM structures and modifying content. We benchmark deterministic heuristic algorithms and off-the-shelf LLMs across these multiple input formats. Our benchmarking shows that Flat JSON input enables LLMs to achieve superior extraction accuracy (F1 score of 0.9567) and minimal hallucination compared to other input formats like Slimmed HTML and Hierarchical JSON. We establish a standardized foundation for rigorous benchmarking, paving the way for the next principled advancements in web data record extraction. 

**Abstract (ZH)**: 一种有效的网页数据记录抽取方法评估框架 

---
# NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation 

**Title (ZH)**: NeSyGeo: 一种多模态几何推理的神经符号框架数据生成 

**Authors**: Weiming Wu, Zi-kang Wang, Jin Ye, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17121)  

**Abstract**: Obtaining large-scale, high-quality data with reasoning paths is crucial for improving the geometric reasoning capabilities of multi-modal large language models (MLLMs). However, existing data generation methods, whether based on predefined templates or constrained symbolic provers, inevitably face diversity and numerical generalization limitations. To address these limitations, we propose NeSyGeo, a novel neuro-symbolic framework for generating geometric reasoning data. First, we propose a domain-specific language grounded in the entity-relation-constraint paradigm to comprehensively represent all components of plane geometry, along with generative actions defined within this symbolic space. We then design a symbolic-visual-text pipeline that synthesizes symbolic sequences, maps them to corresponding visual and textual representations, and generates diverse question-answer (Q&A) pairs using large language models (LLMs). To the best of our knowledge, we are the first to propose a neuro-symbolic approach in generating multimodal reasoning data. Based on this framework, we construct NeSyGeo-CoT and NeSyGeo-Caption datasets, containing 100k samples, and release a new benchmark NeSyGeo-Test for evaluating geometric reasoning abilities in MLLMs. Experiments demonstrate that the proposal significantly and consistently improves the performance of multiple MLLMs under both reinforcement and supervised fine-tuning. With only 4k samples and two epochs of reinforcement fine-tuning, base models achieve improvements of up to +15.8% on MathVision, +8.4% on MathVerse, and +7.3% on GeoQA. Notably, a 4B model can be improved to outperform an 8B model from the same series on geometric reasoning tasks. 

**Abstract (ZH)**: 基于推理路径的大规模高质数据获取对于提高多模态大规模语言模型的几何推理能力至关重要。然而，现有的数据生成方法，无论是基于预定义模板还是受限符号证明器，不可避免地面临着多样性和数值泛化的限制。为了解决这些限制，我们提出了NeSyGeo，一种新型的神经-符号框架以生成几何推理数据。首先，我们提出了一种基于实体-关系-约束范式的领域特定语言，全面表示平面几何的所有组件，并在该符号空间内定义生成动作。随后，我们设计了一个符号-视觉-文本流水线，生成符号序列，将其映射到相应的视觉和文本表示，并使用大规模语言模型生成多样化的问答（Q&A）对。据我们所知，我们是首次提出在生成多模态推理数据中使用神经-符号方法。基于该框架，我们构建了NeSyGeo-CoT和NeSyGeo-Caption数据集，包含100,000个样本，并发布了一个新的基准NeSyGeo-Test以评估多模态大规模语言模型的几何推理能力。实验表明，该提案在强化和监督微调下显著且一致地提高了多个多模态大规模语言模型的性能。仅用4,000个样本和两轮强化微调，基模型在MathVision上提高了15.8%，在MathVerse上提高了8.4%，在GeoQA上提高了7.3%。值得一提的是，一个4B模型可以提高到在其系列中8B模型在几何推理任务上表现出色。 

---
# From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning 

**Title (ZH)**: 从词到思想：LLMs和人类如何通过压缩获取意义 

**Authors**: Chen Shani, Dan Jurafsky, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2505.17117)  

**Abstract**: Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations. 

**Abstract (ZH)**: 人类通过语义压缩将知识组织成紧凑类别，将多样实例映射到抽象表示以保留意义（例如，知更鸟和蓝雀都是鸟类；大多数鸟类会飞）。这些概念反映了表达准确性和表示简单性之间的权衡。大型语言模型（LLMs）展示了卓越的语言能力，但它们的内部表示是否在压缩和语义准确性的权衡上类似于人类尚不清楚。我们引入了一个新的信息论框架，借鉴了速率-失真理论和信息瓶颈原理，以定量比较这些策略。通过对一系列LLM的词汇嵌入与经典的人类分类基准进行分析，我们揭示了关键差异。尽管LLM形成了与人类判断相一致的广泛概念类别，但在捕捉对人类理解至关重要的细微语义差异方面仍存在问题。更根本的是，LLM表现出强烈的统计压缩偏向，而人类的概念系统似乎优先考虑适应性的细腻与情境丰富性，即使这在我们的度量标准下会导致较低的压缩效率。这些发现阐明了当前AI和人类认知架构之间关键的差异，指导着更具人类一致性的LLM概念表示的发展路径。 

---
# Swarm Intelligence Enhanced Reasoning: A Density-Driven Framework for LLM-Based Multi-Agent Optimization 

**Title (ZH)**: 基于密度驱动框架的 Swarm 智能增强推理：面向大语言模型驱动的多agents最优化 

**Authors**: Ying Zhu, Heng Zhou, Rui Su, Peiqin Zhuang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17115)  

**Abstract**: Recently, many approaches, such as Chain-of-Thought (CoT) prompting and Multi-Agent Debate (MAD), have been proposed to further enrich Large Language Models' (LLMs) complex problem-solving capacities in reasoning scenarios. However, these methods may fail to solve complex problems due to the lack of ability to find optimal solutions. Swarm Intelligence has been serving as a powerful tool for finding optima in the field of traditional optimization problems. To this end, we propose integrating swarm intelligence into the reasoning process by introducing a novel Agent-based Swarm Intelligence (ASI) paradigm. In this paradigm, we formulate LLM reasoning as an optimization problem and use a swarm intelligence scheme to guide a group of LLM-based agents in collaboratively searching for optimal solutions. To avoid swarm intelligence getting trapped in local optima, we further develop a Swarm Intelligence Enhancing Reasoning (SIER) framework, which develops a density-driven strategy to enhance the reasoning ability. To be specific, we propose to perform kernel density estimation and non-dominated sorting to optimize both solution quality and diversity simultaneously. In this case, SIER efficiently enhances solution space exploration through expanding the diversity of the reasoning path. Besides, a step-level quality evaluation is used to help agents improve solution quality by correcting low-quality intermediate steps. Then, we use quality thresholds to dynamically control the termination of exploration and the selection of candidate steps, enabling a more flexible and efficient reasoning process. Extensive experiments are ... 

**Abstract (ZH)**: 近年来，诸如Chain-of-Thought (CoT)提示和Multi-Agent Debate (MAD)等方法被提出，旨在进一步丰富大型语言模型（LLMs）在推理场景中复杂的解决问题能力。然而，这些方法由于缺乏找到最优解的能力，可能无法解决复杂问题。群智智能作为传统优化问题中寻找最优解的强大工具已经得到了广泛应用。为此，我们提出了一种基于代理的群智智能（ASI）范式，将群智智能整合到推理过程中。在此范式中，我们将LLM推理形式化为一个优化问题，并采用群智智能方案引导一组LLM基代理协作搜索最优解。为了防止群智智能陷入局部最优解，我们进一步开发了一种群智智能增强推理（SIER）框架，通过密度驱动策略增强推理能力。具体而言，我们提出了内核密度估计和非支配排序，以同时优化解的质量和多样性。通过这种方式，SIER有效扩展了推理路径的多样性，增强了解空间的探索。此外，我们使用步骤级质量评估来帮助代理通过修正低质量的中间步骤来提高解的质量。然后，我们使用质量阈值动态控制探索的终止和候选步骤的选择，从而实现更灵活和高效的推理过程。广泛实验表明... 

---
# REMS: a unified solution representation, problem modeling and metaheuristic algorithm design for general combinatorial optimization problems 

**Title (ZH)**: REMS：通用组合优化问题的一体化解决方案表示、问题建模与元启发式算法设计 

**Authors**: Aijuan Song, Guohua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17108)  

**Abstract**: Combinatorial optimization problems (COPs) with discrete variables and finite search space are critical across numerous fields, and solving them in metaheuristic algorithms is popular. However, addressing a specific COP typically requires developing a tailored and handcrafted algorithm. Even minor adjustments, such as constraint changes, may necessitate algorithm redevelopment. Therefore, establishing a framework for formulating diverse COPs into a unified paradigm and designing reusable metaheuristic algorithms is valuable. A COP can be typically viewed as the process of giving resources to perform specific tasks, subjecting to given constraints. Motivated by this, a resource-centered modeling and solving framework (REMS) is introduced for the first time. We first extract and define resources and tasks from a COP. Subsequently, given predetermined resources, the solution structure is unified as assigning tasks to resources, from which variables, objectives, and constraints can be derived and a problem model is constructed. To solve the modeled COPs, several fundamental operators are designed based on the unified solution structure, including the initial solution, neighborhood structure, destruction and repair, crossover, and ranking. These operators enable the development of various metaheuristic algorithms. Specially, 4 single-point-based algorithms and 1 population-based algorithm are configured herein. Experiments on 10 COPs, covering routing, location, loading, assignment, scheduling, and graph coloring problems, show that REMS can model these COPs within the unified paradigm and effectively solve them with the designed metaheuristic algorithms. Furthermore, REMS is more competitive than GUROBI and SCIP in tackling large-scale instances and complex COPs, and outperforms OR-TOOLS on several challenging COPs. 

**Abstract (ZH)**: 资源中心导向的组合优化问题建模与求解框架（REMS） 

---
# CRAKEN: Cybersecurity LLM Agent with Knowledge-Based Execution 

**Title (ZH)**: CRAKEN: 基于知识驱动执行的网络安全大语言模型代理 

**Authors**: Minghao Shao, Haoran Xi, Nanda Rani, Meet Udeshi, Venkata Sai Charan Putrevu, Kimberly Milner, Brendan Dolan-Gavitt, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2505.17107)  

**Abstract**: Large Language Model (LLM) agents can automate cybersecurity tasks and can adapt to the evolving cybersecurity landscape without re-engineering. While LLM agents have demonstrated cybersecurity capabilities on Capture-The-Flag (CTF) competitions, they have two key limitations: accessing latest cybersecurity expertise beyond training data, and integrating new knowledge into complex task planning. Knowledge-based approaches that incorporate technical understanding into the task-solving automation can tackle these limitations. We present CRAKEN, a knowledge-based LLM agent framework that improves cybersecurity capability through three core mechanisms: contextual decomposition of task-critical information, iterative self-reflected knowledge retrieval, and knowledge-hint injection that transforms insights into adaptive attack strategies. Comprehensive evaluations with different configurations show CRAKEN's effectiveness in multi-stage vulnerability detection and exploitation compared to previous approaches. Our extensible architecture establishes new methodologies for embedding new security knowledge into LLM-driven cybersecurity agentic systems. With a knowledge database of CTF writeups, CRAKEN obtained an accuracy of 22% on NYU CTF Bench, outperforming prior works by 3% and achieving state-of-the-art results. On evaluation of MITRE ATT&CK techniques, CRAKEN solves 25-30% more techniques than prior work, demonstrating improved cybersecurity capabilities via knowledge-based execution. We make our framework open source to public this https URL. 

**Abstract (ZH)**: 基于知识的大语言模型（LLM）代理可以自动化网络安全任务，并且能够在无需重新工程的情况下适应不断演化的网络安全 landscape。虽然大语言模型代理已经在捕获旗子（CTF）竞赛中展示了网络安全能力，但它们存在两个关键局限性：访问超越训练数据的最新网络安全专业知识，以及将新知识整合到复杂任务规划中。基于知识的方法可以通过将技术理解融入任务解决自动化中来应对这些局限性。我们提出了CRAKEN，一种基于知识的大语言模型代理框架，通过三个核心机制提升网络安全能力：任务关键信息的上下文分解、迭代自省知识检索，以及知识提示注入，将洞察力转化为适应性攻击策略。不同配置下的综合评估显示，与以往方法相比，CRAKEN在多阶段漏洞检测和利用方面更有效。我们可扩展的架构为将新安全知识嵌入由大语言模型驱动的网络安全代理系统奠定了新的方法论基础。使用CTF写实数据库，CRAKEN在纽约大学CTF基准上的准确率为22%，比先前工作高3%，达到了最先进的成果。在MITRE ATT&CK技术评估中，CRAKEN比先前工作多解决25-30%的技术，展示了通过基于知识的执行提高的网络安全能力。我们已将该框架开源于此 https://URL。 

---
# Transparency in Healthcare AI: Testing European Regulatory Provisions against Users' Transparency Needs 

**Title (ZH)**: 医疗保健AI中的透明度：测试欧盟监管规定以满足用户透明度需求 

**Authors**: Anna Spagnolli, Cecilia Tolomini, Elisa Beretta, Claudio Sarra  

**Link**: [PDF](https://arxiv.org/pdf/2505.17105)  

**Abstract**: Artificial Intelligence (AI) plays an essential role in healthcare and is pervasively incorporated into medical software and equipment. In the European Union, healthcare is a high-risk application domain for AI, and providers must prepare Instructions for Use (IFU) according to the European regulation 2024/1689 (AI Act). To this regulation, the principle of transparency is cardinal and requires the IFU to be clear and relevant to the users. This study tests whether these latter requirements are satisfied by the IFU structure. A survey was administered online via the Qualtrics platform to four types of direct stakeholders, i.e., managers (N = 238), healthcare professionals (N = 115), patients (N = 229), and Information Technology experts (N = 230). The participants rated the relevance of a set of transparency needs and indicated the IFU section addressing them. The results reveal differentiated priorities across stakeholders and a troubled mapping of transparency needs onto the IFU structure. Recommendations to build a locally meaningful IFU are derived. 

**Abstract (ZH)**: 人工智能（AI）在医疗健康领域扮演着重要角色，并广泛应用于医疗软件和设备中。在欧盟，医疗健康是AI的高风险应用领域，提供者必须根据欧盟2024/1689号条例（AI法案）准备产品使用说明书（IFU）。根据该条例，透明性原则至关重要，要求IFU对用户来说必须清晰且相关。本研究测试这些要求是否被IFU结构所满足。通过Qualtrics平台在线发放问卷，调查了四种类型的利益相关方：管理人员（N=238）、医疗专业人员（N=115）、患者（N=229）和信息技术专家（N=230）。参与者对一套透明性需求的相关性进行了评级，并指出了IFU中相应的部分。研究结果揭示了不同利益相关方的优先事项差异，并指出透明性需求与IFU结构之间的匹配存在困难。研究还获得了构建具有地方意义的IFU的建议。 

---
# Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation 

**Title (ZH)**: 基于语言生成时间序列数据：大规模语言模型合成数据生成方法 

**Authors**: Cécile Rousseau, Tobia Boschi, Giandomenico Cornacchia, Dhaval Salwala, Alessandra Pascale, Juan Bernabe Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2505.17103)  

**Abstract**: SDForger is a flexible and efficient framework for generating high-quality multivariate time series using LLMs. Leveraging a compact data representation, SDForger provides synthetic time series generation from a few samples and low-computation fine-tuning of any autoregressive LLM. Specifically, the framework transforms univariate and multivariate signals into tabular embeddings, which are then encoded into text and used to fine-tune the LLM. At inference, new textual embeddings are sampled and decoded into synthetic time series that retain the original data's statistical properties and temporal dynamics. Across a diverse range of datasets, SDForger outperforms existing generative models in many scenarios, both in similarity-based evaluations and downstream forecasting tasks. By enabling textual conditioning in the generation process, SDForger paves the way for multimodal modeling and the streamlined integration of time series with textual information. SDForger source code will be open-sourced soon. 

**Abstract (ZH)**: SDForger：一种利用大语言模型生成高质量多变量时间序列的灵活高效框架 

---
# Large Language Models Implicitly Learn to See and Hear Just By Reading 

**Title (ZH)**: 大型语言模型仅通过阅读便学会了 see 和 hear。 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17091)  

**Abstract**: This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time. 

**Abstract (ZH)**: 本文呈现了一个有趣的发现：通过训练自回归的大语言模型，文本模型本身会发展出内在的图像和音频理解能力，从而仅通过阅读就能够“看到”和“听到”。流行的音频和视觉大语言模型会微调文本大语言模型，以根据图像和音频嵌入生成文本输出。相比之下，我们的架构接受图像块、音频波形或标记作为输入，输出分类管道典型的嵌入或类别标签。我们展示了文本权重在帮助对FSD-50K和GTZAN数据集进行音频分类上的通用性。此外，我们还展示了其在CIFAR-10和Fashion-MNIST图像分类任务以及图像块上的应用效果。这推动了文本大语言模型学习强大的内部电路可以在必要时激活以适应各种应用的观念，而非每次都从头训练模型。 

---
# Informatics for Food Processing 

**Title (ZH)**: 食品加工中的信息学 

**Authors**: Gordana Ispirova, Michael Sebek, Giulia Menichetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.17087)  

**Abstract**: This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research. 

**Abstract (ZH)**: 本章探讨了食品加工的演变、分类及其健康影响，并强调了机器学习、人工智能（AI）和数据科学在食品信息学领域中的变革性作用。它从历史概述出发，对传统的分类框架（如NOVA、Nutri-Score和SIGA）进行了批判性回顾，指出了这些框架的优势与局限性，特别是其在流行病学研究和公共政策制定中遇到的主观性和可重复性难题。为解决这些问题，本章提出了新颖的计算方法，包括使用营养成分数据训练的随机森林模型FoodProX，以推断加工水平并生成连续的FPro评分。此外，本章还探讨了如何利用如BERT和BioBERT这样的大型语言模型通过语义嵌入食品描述和成分列表来进行预测任务，即使在数据缺失的情况下也是如此。本章的一个主要贡献是使用Open Food Facts数据库进行的一项新颖案例研究，展示了多模态AI模型如何整合结构化和非结构化数据以大规模分类食品，为公共健康和研究中的食品加工评估提供了新的范式。 

---
# GSDFuse: Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis 

**Title (ZH)**: GSDFuse: 从社交媒体隐写分析中多维度弱信号捕获认知不一致性 

**Authors**: Kaibo Huang, Zipei Zhang, Yukun Wei, TianXin Zhang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17085)  

**Abstract**: The ubiquity of social media platforms facilitates malicious linguistic steganography, posing significant security risks. Steganalysis is profoundly hindered by the challenge of identifying subtle cognitive inconsistencies arising from textual fragmentation and complex dialogue structures, and the difficulty in achieving robust aggregation of multi-dimensional weak signals, especially given extreme steganographic sparsity and sophisticated steganography. These core detection difficulties are compounded by significant data imbalance. This paper introduces GSDFuse, a novel method designed to systematically overcome these obstacles. GSDFuse employs a holistic approach, synergistically integrating hierarchical multi-modal feature engineering to capture diverse signals, strategic data augmentation to address sparsity, adaptive evidence fusion to intelligently aggregate weak signals, and discriminative embedding learning to enhance sensitivity to subtle inconsistencies. Experiments on social media datasets demonstrate GSDFuse's state-of-the-art (SOTA) performance in identifying sophisticated steganography within complex dialogue environments. The source code for GSDFuse is available at this https URL. 

**Abstract (ZH)**: 社交媒体平台的普遍性促进了恶意语言隐写术的应用，引发了显著的安全风险。传统的文本隐写分析受到识别由文本碎片化和复杂对话结构产生的微妙认知不一致性挑战的影响，并且在聚合多维度弱信号方面存在困难，尤其是在极端隐写稀疏性和复杂的隐写术条件下。这些核心检测困难进一步受到严重数据不平衡的影响。本文提出了一种新技术GSDFuse，以系统地克服这些障碍。GSDFuse采用整体方法，结合层次多模态特征工程捕捉不同信号、战略数据增强处理稀疏性问题、自适应证据融合智能聚合弱信号以及判别嵌入学习提高对微妙不一致性的敏感度。在社交媒体数据集上的实验表明，GSDFuse在复杂对话环境中识别复杂隐写术方面达到了当前最佳性能（SOTA）。GSDFuse的源代码可通过此链接获取。 

---
# From nuclear safety to LLM security: Applying non-probabilistic risk management strategies to build safe and secure LLM-powered systems 

**Title (ZH)**: 从核安全到大语言模型安全：应用非概率风险管理系统构建安全可靠的基于大语言模型的系统 

**Authors**: Alexander Gutfraind, Vicki Bier  

**Link**: [PDF](https://arxiv.org/pdf/2505.17084)  

**Abstract**: Large language models (LLMs) offer unprecedented and growing capabilities, but also introduce complex safety and security challenges that resist conventional risk management. While conventional probabilistic risk analysis (PRA) requires exhaustive risk enumeration and quantification, the novelty and complexity of these systems make PRA impractical, particularly against adaptive adversaries. Previous research found that risk management in various fields of engineering such as nuclear or civil engineering is often solved by generic (i.e. field-agnostic) strategies such as event tree analysis or robust designs. Here we show how emerging risks in LLM-powered systems could be met with 100+ of these non-probabilistic strategies to risk management, including risks from adaptive adversaries. The strategies are divided into five categories and are mapped to LLM security (and AI safety more broadly). We also present an LLM-powered workflow for applying these strategies and other workflows suitable for solution architects. Overall, these strategies could contribute (despite some limitations) to security, safety and other dimensions of responsible AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了前所未有的能力和持续增长的能力，但也引入了复杂的安全和安全挑战，这些挑战超越了传统的风险管理方法。虽然传统的概率风险分析（PRA）要求详尽的风险列举和量化，但这些系统的新颖性和复杂性使其在面对适应性对手时变得不切实际。以往的研究发现，工程领域如核工程或土木工程中的风险管理通常通过通用（即领域无关）策略如事件树分析或鲁棒设计来解决。在这里，我们展示了如何利用100多种非概率性的策略来应对LLM驱动系统中的新兴风险，包括来自适应性对手的风险。这些策略被分为五类，并映射到LLM安全（以及更广泛的AI安全性）。我们还介绍了一种LLM驱动的工作流程来应用这些策略以及其他适用于解决方案架构师的工作流程。总体而言，尽管存在一些限制，这些策略仍有可能为负责任的AI的安全性、安全性及其他维度做出贡献。 

---
# GemMaroc: Unlocking Darija Proficiency in LLMs with Minimal Data 

**Title (ZH)**: GemMaroc: 用最少的数据解锁达里雅布 proficiency 在大型语言模型中的应用 

**Authors**: Abderrahman Skiredj, Ferdaous Azhari, Houdaifa Atou, Nouamane Tazi, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2505.17082)  

**Abstract**: Open-source large language models (LLMs) still marginalise Moroccan Arabic (Darija), forcing practitioners either to bolt on heavyweight Arabic adapters or to sacrifice the very reasoning skills that make LLMs useful. We show that a rigorously quality-over-quantity alignment strategy can surface fluent Darija while safeguarding the backbone s cross-lingual reasoning at a sliver of the usual compute. We translate three compact instruction suites LIMA 1 K, DEITA 6 K and TULU 50 K into Darija, preserve 20 of the English originals, and add mathematics, coding and scientific prompts. A LoRA-tuned Gemma 3-4B trained on 5 K mixed instructions lifts DarijaMMLU from 32.8 to 42.7 ; adding the reasoning-dense TULU portion pushes it to 47.5 with no English regression. Scaling the identical recipe to Gemma 3-27B produces GemMaroc-27B, which matches Atlas-Chat on DarijaMMLU (61.6 ) and leaps ahead on Darija commonsense, scoring 60.5 on HellaSwag versus Atlas-Chat s 48.4 . Crucially, GemMaroc retains Gemma-27B s strong maths and general-reasoning ability, showing only minimal movement on GSM8K and English benchmarks. The entire model is trained in just 48 GPU.h, underscoring a Green AI pathway to inclusive, sustainable language technology. We release code, data and checkpoints to spur Darija-centric applications in education, public services and everyday digital interaction. 

**Abstract (ZH)**: 开源大规模语言模型（LLMs）仍 marginalises 摒弃了摩洛哥阿拉伯语（Darija），迫使实践者要么安装重量级阿拉伯语适应器，要么牺牲使LLMs有用的推理能力。我们展示了一种以质量优先而非数量的对齐策略可以揭示流畅的Darija同时保护跨语言推理能力，仅需通常计算量的一小部分。我们将三个紧凑的指令集LIMA 1 K、DEITA 6 K和TULU 50 K翻译成Darija，保留20个原始的英语版本，并添加了数学、编程和科学提示。一个针对5 K混合指令进行了LoRA调优的Gemma 3-4B提升了DarijaMMLU分数至42.7；增加TULU部分进一步提升至47.5，无英语回退。将相同的配方扩展至Gemma 3-27B产生GemMaroc-27B，该模型在DarijaMMLU上与Atlas-Chat相当（61.6），并在Darija常识方面表现出色，HellaSwag得分为60.5，而Atlas-Chat得分为48.4。关键的是，GemMaroc保留了Gemma-27B的强大数学和一般推理能力，仅在GSM8K和英语基准上有轻微变化。整个模型仅在48 GPU小时内在进行了训练，突显了一条绿色AI路径，通往包容和可持续的语言技术。我们发布了代码、数据和检查点，以促进以Darija为中心的应用在教育、公共服务和日常数字交互中的发展。 

---
# GloSS over Toxicity: Understanding and Mitigating Toxicity in LLMs via Global Toxic Subspace 

**Title (ZH)**: GloSS 过滤毒性：通过全局毒性子空间理解并缓解大规模语言模型中的毒性 

**Authors**: Zenghao Duan, Zhiyi Yin, Zhichao Shi, Liang Pang, Shaoling Jing, Jiayi Wu, Yu Yan, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17078)  

**Abstract**: This paper investigates the underlying mechanisms of toxicity generation in Large Language Models (LLMs) and proposes an effective detoxification approach. Prior work typically considers the Feed-Forward Network (FFN) as the main source of toxicity, representing toxic regions as a set of toxic vectors or layer-wise subspaces. However, our in-depth analysis reveals that the global toxic subspace offers a more effective and comprehensive representation of toxic region within the model. Building on this insight, we propose GloSS (Global Toxic Subspace Suppression), a lightweight, four-stage method that mitigates toxicity by identifying and removing the global toxic subspace from the parameters of FFN. Experiments across a range of LLMs show that GloSS achieves state-of-the-art detoxification performance while preserving the models general capabilities, without requiring large-scale data or model retraining. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）中毒性生成的内在机制，并提出了一种有效的去毒方法。先前的工作通常认为前馈网络（FFN）是毒性的主要来源，将有毒区域表示为一组有毒向量或逐层子空间。然而，我们的深入分析表明，全局有毒子空间提供了模型中更有效和全面的有毒区域表示。基于这一见解，我们提出了一种轻量级的GloSS（全局有毒子空间抑制）方法，通过识别并从FFN参数中移除全局有毒子空间来减轻毒性。实验结果表明，GloSS在保持模型通用能力的同时，实现了最先进的去毒性能，无需大规模数据或模型重新训练。 

---
# Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English 

**Title (ZH)**: 帧率对语音分词器的影响：以普通话和英语为例的研究 

**Authors**: Haoyang Zhang, Hexin Liu, Xiangyu Zhang, Qiquan Zhang, Yuchen Hu, Junqi Zhao, Fei Tian, Xuerui Yang, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17076)  

**Abstract**: The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications. 

**Abstract (ZH)**: 语音切分器在近期的语音任务中扮演着至关重要的角色，通常作为语音信号与语言模型之间的桥梁。尽管低帧率编码器广泛用于语音切分，但帧率对语音切分的影响仍未充分探讨。本研究通过分析两种类型不同的语言（普通话和英语）来探讨不同帧率如何影响语音切分，并在语音识别任务中评估由此产生的语义切分。研究发现，帧率的变化以不同的方式影响每种语言的语音切分，突显了帧率、音位密度和语言特定声学特征之间的相互作用。这些结果为优化语音切分器的帧率选择提供了见解，对自动语音识别、文本到语音以及其他相关语音应用具有重要意义。 

---
# Development and Validation of Engagement and Rapport Scales for Evaluating User Experience in Multimodal Dialogue Systems 

**Title (ZH)**: 多模态对话系统中用户体验的参与度和关系量表的开发与验证 

**Authors**: Fuma Kurata, Mao Saeki, Masaki Eguchi, Shungo Suzuki, Hiroaki Takatsu, Yoichi Matsuyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.17075)  

**Abstract**: This study aimed to develop and validate two scales of engagement and rapport to evaluate the user experience quality with multimodal dialogue systems in the context of foreign language learning. The scales were designed based on theories of engagement in educational psychology, social psychology, and second language this http URL-four Japanese learners of English completed roleplay and discussion tasks with trained human tutors and a dialog agent. After each dialogic task was completed, they responded to the scales of engagement and rapport. The validity and reliability of the scales were investigated through two analyses. We first conducted analysis of Cronbach's alpha coefficient and a series of confirmatory factor analyses to test the structural validity of the scales and the reliability of our designed items. We then compared the scores of engagement and rapport between the dialogue with human tutors and the one with a dialogue agent. The results revealed that our scales succeeded in capturing the difference in the dialogue experience quality between the human interlocutors and the dialogue agent from multiple perspectives. 

**Abstract (ZH)**: 本研究旨在基于教育心理学、社会心理学及外语学习理论，开发和完善两个用于评估外语学习中多模态对话系统用户体验质量的效度和可靠性的参与度和关系度量表。Japanese英语学习者与经过培训的人类导师和对话代理进行了角色扮演和讨论任务，并在每个对话任务完成后对参与度和关系度量表做出了回应。通过两种分析验证了度量表的有效性和可靠性。首先，我们计算Cronbach’s α系数并进行了系列确证性因子分析，以测试量表的结构效度和项目可靠性。然后，我们将人类导师对话和与对话代理对话的参与度和关系得分进行了比较。研究结果表明，我们的量表能够在多个视角上成功捕捉人类对话伙伴与对话代理之间的对话体验质量差异。 

---
# Semi-Clairvoyant Scheduling of Speculative Decoding Requests to Minimize LLM Inference Latency 

**Title (ZH)**: 推测性解码请求的半先知调度以最小化大语言模型推理延迟 

**Authors**: Ruixiao Li, Fahao Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17074)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by employing a small speculative model (SSM) to generate multiple candidate tokens and verify them using the LLM in parallel. This technique has been widely integrated into LLM inference serving systems. However, inference requests typically exhibit uncertain execution time, which poses a significant challenge of efficiently scheduling requests in these systems. Existing work estimates execution time based solely on predicted output length, which could be inaccurate because execution time depends on both output length and token acceptance rate of verification by the LLM. In this paper, we propose a semi-clairvoyant request scheduling algorithm called Least-Attained/Perceived-Service for Speculative Decoding (LAPS-SD). Given a number of inference requests, LAPS-SD can effectively minimize average inference latency by adaptively scheduling requests according to their features during decoding. When the token acceptance rate is dynamic and execution time is difficult to estimate, LAPS-SD maintains multiple priority queues and allows request execution preemption across different queues. Once the token acceptance rate becomes stable, LAPS-SD can accurately estimate the execution time and schedule requests accordingly. Extensive experiments show that LAPS-SD reduces inference latency by approximately 39\% compared to state-of-the-art scheduling methods. 

**Abstract (ZH)**: 推测解码加速大型语言模型（LLM）推理过程中的请求调度：一种基于部分先知的服务获取最少未完成/感知服务（LAPS-SD）算法 

---
# Mechanistic Interpretability of GPT-like Models on Summarization Tasks 

**Title (ZH)**: GPT类模型在摘要任务中的机制可解释性 

**Authors**: Anurag Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.17073)  

**Abstract**: Mechanistic interpretability research seeks to reveal the inner workings of large language models, yet most work focuses on classification or generative tasks rather than summarization. This paper presents an interpretability framework for analyzing how GPT-like models adapt to summarization tasks. We conduct differential analysis between pre-trained and fine-tuned models, quantifying changes in attention patterns and internal activations. By identifying specific layers and attention heads that undergo significant transformation, we locate the "summarization circuit" within the model architecture. Our findings reveal that middle layers (particularly 2, 3, and 5) exhibit the most dramatic changes, with 62% of attention heads showing decreased entropy, indicating a shift toward focused information selection. We demonstrate that targeted LoRA adaptation of these identified circuits achieves significant performance improvement over standard LoRA fine-tuning while requiring fewer training epochs. This work bridges the gap between black-box evaluation and mechanistic understanding, providing insights into how neural networks perform information selection and compression during summarization. 

**Abstract (ZH)**: 机制可解释性研究旨在揭示大型语言模型的内部工作机制，然而大多数工作侧重于分类或生成任务，而非摘要任务。本文提出了一种可解释性框架，用于分析类似于GPT的模型如何适应摘要任务。我们进行了预训练模型与微调模型之间的差异分析，量化了注意力模式和内部激活的变化。通过识别显著变化的具体层和注意力头，我们定位了模型架构中的“摘要电路”。研究发现，中间层（尤其是第2、第3和第5层）显示出最 dramatic 变化，62%的注意力头显示出熵降低的趋势，表明向集中信息选择的转变。我们证明，针对识别出的这些电路进行定向LoRA适应，能够在使用更少训练周期的同时，实现比标准LoRA微调更好的性能提升。这项工作填补了黑盒评估与机制性理解之间的差距，提供了关于神经网络在摘要任务中执行信息选择和压缩的洞察。 

---
# Safety Alignment Can Be Not Superficial With Explicit Safety Signals 

**Title (ZH)**: 安全对齐可以通过显式安全信号而不只是表层地实现。 

**Authors**: Jianwei Li, Jung-Eng Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17072)  

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems. 

**Abstract (ZH)**: 近期对大型语言模型安全性对齐的研究表明，现有方法往往浅层次地运作，使得模型容易受到各种 adversarial 攻击的威胁。尽管这些研究具有重要意义，但它们通常未能提供超越数据增强的可操作解决方案，以实现更为 robust 的安全机制。本文指出了这种浅层次性的一个根本原因：现有对齐方法往往假定模型可以在对齐过程中隐式学习一个与安全性相关的推理任务，从而能够拒绝有害请求。然而，学习到的安全信号往往被其他竞争目标稀释，导致模型在面对 adversarial 攻击时难以划定一个明确的安全意识决策边界。基于这一观察，我们通过明确引入一个与安全性相关的二元分类任务，并将其实现信号与我们的注意力和解码策略集成，消除了这种模糊性，使模型能够更负责任地响应恶意查询。我们强调，我们的方法在不到 0.2 倍的额外开销下，使大型语言模型能够在每次必要的生成步骤中评估查询和之前生成的令牌的安全性。大量实验表明，我们的方法显著提高了大型语言模型对各种 adversarial 攻击的抗性，为更 robust 的生成人工智能系统提供了有希望的途径。 

---
# Improving endpoint detection in end-to-end streaming ASR for conversational speech 

**Title (ZH)**: 改进端到端流式ASR在对话语音中的端点检测 

**Authors**: Anandh C, Karthik Pandia Durai, Jeena Prakash, Manickavela Arumugam, Kadri Hacioglu, S.Pavankumar Dubagunta, Andreas Stolcke, Shankar Venkatesan, Aravind Ganapathiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.17070)  

**Abstract**: ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method. 

**Abstract (ZH)**: ASR 末端检测（EP）在支持人类或人工代理的人与人/机器对话产品中提供良好用户体验中扮演重要角色。基于转导的ASR（T-ASR）是一种优选的端到端（E2E）ASR建模技术，适用于流式处理。T-ASR的主要限制是ASR输出的延迟发出，这可能会导致末端检测（EP）错误或延迟。不准确的末端检测会中断用户的讲话，返回不完整的记录，而延迟的末端检测会增加感知的延迟，降低用户体验。我们通过解决延迟排放和末端检测错误来改进末端检测。为了应对延迟排放问题，我们在每个单词结束处引入一个词尾标记，并加入延迟惩罚。通过使用辅助网络获得可靠的帧级语音活动检测来解决末端检测延迟问题。我们将在Switchboard对话语音语料库上应用所提出的方法，并将其与延迟惩罚方法进行比较评估。 

---
# Unveil Multi-Picture Descriptions for Multilingual Mild Cognitive Impairment Detection via Contrastive Learning 

**Title (ZH)**: 多 picture 描述的对比学习在多语言轻度认知障碍检测中的应用 

**Authors**: Kristin Qi, Jiali Cheng, Youxiang Zhu, Hadi Amiri, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17067)  

**Abstract**: Detecting Mild Cognitive Impairment from picture descriptions is critical yet challenging, especially in multilingual and multiple picture settings. Prior work has primarily focused on English speakers describing a single picture (e.g., the 'Cookie Theft'). The TAUKDIAL-2024 challenge expands this scope by introducing multilingual speakers and multiple pictures, which presents new challenges in analyzing picture-dependent content. To address these challenges, we propose a framework with three components: (1) enhancing discriminative representation learning via supervised contrastive learning, (2) involving image modality rather than relying solely on speech and text modalities, and (3) applying a Product of Experts (PoE) strategy to mitigate spurious correlations and overfitting. Our framework improves MCI detection performance, achieving a +7.1% increase in Unweighted Average Recall (UAR) (from 68.1% to 75.2%) and a +2.9% increase in F1 score (from 80.6% to 83.5%) compared to the text unimodal baseline. Notably, the contrastive learning component yields greater gains for the text modality compared to speech. These results highlight our framework's effectiveness in multilingual and multi-picture MCI detection. 

**Abstract (ZH)**: 从图片描述中检测轻度认知 impairment 在多语言和多图片设置下是关键但具有挑战性：TAUKDIAL-2024 挑战赛扩展了这一范围 

---
# Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration 

**Title (ZH)**: 基于专家模型集成提高大语言模型对抗 Jailbreak 攻击的输出质量 

**Authors**: Tatia Tsmindashvili, Ana Kolkhidashvili, Dachi Kurtskhalia, Nino Maghlakelidze, Elene Mekvabishvili, Guram Dentoshvili, Orkhan Shamilov, Zaal Gachechiladze, Steven Saporta, David Dachi Choladze  

**Link**: [PDF](https://arxiv.org/pdf/2505.17066)  

**Abstract**: Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community. 

**Abstract (ZH)**: 在生产环境中使用LLM面临的安全挑战包括 Jailbreak 和提示注入等漏洞，这些都可能导致对人类或企业产生有害输出。当在特定领域工作时，这一挑战更加严峻，因为通常接受的LLM处理主题可能与该领域无关。这些问题可以通过使用领域特定和安全导向的数据对大型语言模型进行微调来缓解。然而，这些方法并不足够，因为 Jailbreak 技巧会不断演变。此外，通过API访问的模型无法提供根据行业特定目标定制行为所需的灵活性，而上下文学习也不总能可靠地实现。为应对这些挑战，我们引入了Archias，一种专家模型，擅长区分域内和域外通信。Archias将用户咨询分类为多个类别：特定于汽车行业的域内咨询、恶意问题、价格注入、提示注入和域外示例。我们的方法将专家模型（Archias）的输出整合到提示中，然后由LLM处理生成响应。这种方法增强了模型理解用户意图并给出适当回答的能力。Archias由于其小巧的体积，可以进行调整、微调并用于多种目的，因此可以轻松适应任何行业的特定需求。为了验证我们的方法，我们为汽车行业创建了一个基准数据集。此外，为了促进研究和开发，我们向社区发布了基准数据集。 

---
# Synthetic History: Evaluating Visual Representations of the Past in Diffusion Models 

**Title (ZH)**: 合成历史：评估扩散模型中过去的时代图示 

**Authors**: Maria-Teresa De Rosa Palmini, Eva Cetinic  

**Link**: [PDF](https://arxiv.org/pdf/2505.17064)  

**Abstract**: As Text-to-Image (TTI) diffusion models become increasingly influential in content creation, growing attention is being directed toward their societal and cultural implications. While prior research has primarily examined demographic and cultural biases, the ability of these models to accurately represent historical contexts remains largely underexplored. In this work, we present a systematic and reproducible methodology for evaluating how TTI systems depict different historical periods. For this purpose, we introduce the HistVis dataset, a curated collection of 30,000 synthetic images generated by three state-of-the-art diffusion models using carefully designed prompts depicting universal human activities across different historical periods. We evaluate generated imagery across three key aspects: (1) Implicit Stylistic Associations: examining default visual styles associated with specific eras; (2) Historical Consistency: identifying anachronisms such as modern artifacts in pre-modern contexts; and (3) Demographic Representation: comparing generated racial and gender distributions against historically plausible baselines. Our findings reveal systematic inaccuracies in historically themed generated imagery, as TTI models frequently stereotype past eras by incorporating unstated stylistic cues, introduce anachronisms, and fail to reflect plausible demographic patterns. By offering a scalable methodology and benchmark for assessing historical representation in generated imagery, this work provides an initial step toward building more historically accurate and culturally aligned TTI models. 

**Abstract (ZH)**: 随着文本到图像（TTI）扩散模型在内容创建中的影响力日益增强，人们越来越关注其对社会和文化的影响。尽管先前的研究主要关注人口统计和文化偏见，但这些模型如何准确地表现历史背景仍然 largely underexplored。在此项研究中，我们提出了一种系统且可再现的方法来评估 TTI 系统如何描绘不同历史时期。为此，我们引入了 HistVis 数据集，该数据集包含 30,000 张由三种最先进的扩散模型生成的精心设计的合成图像，这些图像描绘了不同历史时期的普遍人类活动。我们从三个关键方面评估生成的图像：（1）隐含的风格关联：检查与特定时期相关的默认视觉风格；（2）历史一致性：识别现代物品在前现代背景中的不合适；（3）人口统计表现：将生成的种族和性别分布与历史可信的基准进行比较。我们的研究发现，在主题上与历史相关的生成图像存在系统性不准确之处，TTI 模型经常通过引入未明确的风格线索、引入历史错误和未能反映可信的人口统计模式来刻板化过去的时代。通过提供一种可扩展的方法和基准来评估生成图像中的历史表现，本项工作为构建更准确的历史和文化对齐的 TTI 模型迈出了第一步。 

---
# Synthetic Data RL: Task Definition Is All You Need 

**Title (ZH)**: 合成数据RL：任务定义即一切 

**Authors**: Yiduo Guo, Zhen Guo, Chuanwei Huang, Zi-Ang Wang, Zekai Zhang, Haofei Yu, Huishuai Zhang, Yikang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17063)  

**Abstract**: Reinforcement learning (RL) is a powerful way to adapt foundation models to specialized tasks, but its reliance on large-scale human-labeled data limits broad adoption. We introduce Synthetic Data RL, a simple and general framework that reinforcement fine-tunes models using only synthetic data generated from a task definition. Our method first generates question and answer pairs from the task definition and retrieved documents, then adapts the difficulty of the question based on model solvability, and selects questions using the average pass rate of the model across samples for RL training. On Qwen-2.5-7B, our method achieves a 29.2% absolute improvement over the base model on GSM8K (+2.9 pp vs. instruction-tuned, +6.6 pp vs. Self-Instruct), 8.7% on MATH, 13.1% on GPQA (+7.0 pp vs. SynthLLM), 8.9% on MedQA, 17.7% on CQA (law) and 13.7% on CFA (finance). It surpasses supervised fine-tuning under the same data budget and nearly matches RL with full human data across datasets (e.g., +17.2 pp on GSM8K). Adding 100 human demonstrations improves the performance of GSM8K only by 0.4 pp, showing a limited added value. By reducing human data annotation, Synthetic Data RL enables scalable and efficient RL-based model adaptation. Code and demos are available at this https URL. 

**Abstract (ZH)**: 合成数据强化学习：仅使用合成数据对基础模型进行强化微调的简单通用框架 

---
# Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models 

**Title (ZH)**: 混合解码：一种基于注意力的自适应解码策略，用于缓解大规模视觉-语言模型中的幻觉 

**Authors**: Xinlong Chen, Yuanxing Zhang, Qiang Liu, Junfei Wu, Fuzheng Zhang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17061)  

**Abstract**: Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model's attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model's attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. The code is available at this https URL. 

**Abstract (ZH)**: Large 视觉-语言 模型 (LVLMs) 在各种视觉任务中展示了令人印象深刻的性能，但仍受到幻觉持续挑战的困扰。为应对这一关键问题，我们提出了一种名为 Mixture of Decoding (MoD) 的新颖方法，该方法通过评估模型对图像标记的注意力的正确性来动态调整解码策略。具体而言，MoD 通过比较源于原始图像标记和模型关注图像标记生成的输出之间的一致性来区分上述正确性。如果输出一致，表明注意力正确，则 MoD 采用补充策略加强关键信息。反之，如果输出不一致，表明注意力错误，则 MoD 采用对比策略抑制误导性信息。广泛实验表明，MoD 在多个主流基准上显著优于现有解码方法，有效减轻了 LVLMs 中的幻觉问题。代码可供访问：this https URL。 

---
# SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation 

**Title (ZH)**: SALMONN-omni：一种无需编解码器注入的独立语音LLM全双工对话系统 

**Authors**: Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17060)  

**Abstract**: In order to enable fluid and natural human-machine speech interaction, existing full-duplex conversational systems often adopt modular architectures with auxiliary components such as voice activity detectors, interrupters, conversation state predictors, or multiple LLMs. These systems, however, suffer from error accumulation across modules and struggle with key challenges such as context-dependent barge-in and echo cancellation. Recent approaches, most notably Moshi, simplify the pipeline by injecting audio codecs into the token space of a single LLM. However, such methods still incur significant performance degradation when operating on the speech rather than text modality. In this paper, we introduce SALMONN-omni, the first single, standalone full-duplex speech LLM that operates without audio codecs in its token space. It features a novel dynamic thinking mechanism within the LLM backbone, enabling the model to learn when to transition between speaking and listening states. Experiments on widely used benchmarks for spoken question answering and open-domain dialogue show that SALMONN-omni achieves at least 30\% relative performance improvement over existing open-source full-duplex models and performs highly competitively to half-duplex and turn-based systems, despite using substantially less training data. Moreover, SALMONN-omni demonstrates strong performance in complex conversational scenarios, including turn-taking, backchanneling, echo cancellation and context-dependent barge-in, with further improvements achieved through reinforcement learning. Some demo conversations between user and SALMONN-omni are provided in the following repository this https URL. 

**Abstract (ZH)**: 为了实现流畅自然的人机语音交互，现有的全双工对话系统通常采用具有辅助组件（如语音活动检测器、打断器、对话状态预测器或多个LLM）的模块化架构。然而，这些系统在跨模块过程中容易积累误差，并且难以应对诸如上下文依赖的打断和回声消除等关键挑战。最近的方法，尤其是Moshi，通过将音频编解码器注入单个LLM的标记空间来简化流水线。但是，这些方法在处理语音而非文本模态时仍然会遭受显著的性能下降。在本文中，我们引入了SALMONN-omni，这是第一个无需在标记空间中使用音频编解码器的独立单体全双工语音LLM。它配备了LLM骨干网络中的新颖动态思维机制，使模型能够学习在讲话和倾听状态之间转换的时机。在广泛使用的语音问答和开放领域对话基准上的实验表明，SALMONN-omni在现有的开源全双工模型上的性能至少提高了30%，并且在使用较少训练数据的情况下与半双工和轮询系统具有高度竞争力。此外，SALMONN-omni在复杂的对话场景中表现出了强大的性能，包括对话轮换、副语言、回声消除和上下文依赖的打断，进一步的性能提升通过强化学习实现。有关用户与SALMONN-omni的演示对话可在以下仓库中找到：[此链接]。 

---
# Medalyze: Lightweight Medical Report Summarization Application Using FLAN-T5-Large 

**Title (ZH)**: Medalyze: 使用FLAN-T5-Large的轻量级医学报告总结应用 

**Authors**: Van-Tinh Nguyen, Hoang-Duong Pham, Thanh-Hai To, Cong-Tuan Hung Do, Thi-Thu-Trang Dong, Vu-Trung Duong Le, Van-Phuc Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17059)  

**Abstract**: Understanding medical texts presents significant challenges due to complex terminology and context-specific language. This paper introduces Medalyze, an AI-powered application designed to enhance the comprehension of medical texts using three specialized FLAN-T5-Large models. These models are fine-tuned for (1) summarizing medical reports, (2) extracting health issues from patient-doctor conversations, and (3) identifying the key question in a passage. Medalyze is deployed across a web and mobile platform with real-time inference, leveraging scalable API and YugabyteDB. Experimental evaluations demonstrate the system's superior summarization performance over GPT-4 in domain-specific tasks, based on metrics like BLEU, ROUGE-L, BERTScore, and SpaCy Similarity. Medalyze provides a practical, privacy-preserving, and lightweight solution for improving information accessibility in healthcare. 

**Abstract (ZH)**: 医疗文本的理解面临着因复杂术语和情境特定语言而带来的显著挑战。本文介绍了Medalyze，一种使用三种专门Fine-Tuned FLAN-T5-Large模型的AI驱动应用，旨在通过这些模型增强对医疗文本的 comprehension。Medalyze 在 Web 和移动平台上实时部署，利用可扩展的 API 和 YugabyteDB。实验评估表明，与 GPT-4 相比，该系统在领域特定任务中的总结性能更优，基于 BLEU、ROUGE-L、BERTScore 和 SpaCy 相似性等指标。Medalyze 提供了一种实用、保护隐私且轻量级的解决方案，以提高医疗领域的信息 accessibility。 

---
# DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation 

**Title (ZH)**: DO-RAG：一种基于知识图谱增强检索生成的领域特定问答框架 

**Authors**: David Osei Opoku, Ming Sheng, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17058)  

**Abstract**: Domain-specific QA systems require not just generative fluency but high factual accuracy grounded in structured expert knowledge. While recent Retrieval-Augmented Generation (RAG) frameworks improve context recall, they struggle with integrating heterogeneous data and maintaining reasoning consistency. To address these challenges, we propose DO-RAG, a scalable and customizable hybrid QA framework that integrates multi-level knowledge graph construction with semantic vector retrieval. Our system employs a novel agentic chain-of-thought architecture to extract structured relationships from unstructured, multimodal documents, constructing dynamic knowledge graphs that enhance retrieval precision. At query time, DO-RAG fuses graph and vector retrieval results to generate context-aware responses, followed by hallucination mitigation via grounded refinement. Experimental evaluations in the database and electrical domains show near-perfect recall and over 94% answer relevancy, with DO-RAG outperforming baseline frameworks by up to 33.38%. By combining traceability, adaptability, and performance efficiency, DO-RAG offers a reliable foundation for multi-domain, high-precision QA at scale. 

**Abstract (ZH)**: 领域特定的问答系统不仅需要生成流畅性，还需要基于结构化专家知识的高度事实准确性。虽然最近的检索增强生成（RAG）框架在上下文召回方面有所改善，但它们在整合异构数据和保持推理一致性方面存在困难。为应对这些挑战，我们提出了一种可扩展且可定制的混合问答框架DO-RAG，该框架结合了多级知识图构建与语义向量检索。该系统采用了一种新颖的行动者思维链架构，从非结构化、多模态文档中提取结构化关系，构建动态知识图，从而提高检索精度。在查询时，DO-RAG 将图检索和向量检索结果融合生成上下文相关的回答，并通过基于事实的改进减轻虚构信息。在数据库和电气领域的实验评估显示，DO-RAG 的召回率接近完美，回答相关性超过 94%，且相比基准框架性能提升高达 33.38%。通过结合可追溯性、适应性和性能效率，DO-RAG 为多领域、高精度的大规模问答提供了可靠的基石。 

---
# Are LLMs Ready for English Standardized Tests? A Benchmarking and Elicitation Perspective 

**Title (ZH)**: LLM在英语标准化测试中准备就绪了吗？一项基准测试与需求捕获视角分析 

**Authors**: Luoxi Tang, Tharunya Sundar, Shuai Yang, Ankita Patra, Manohar Chippada, Giqi Zhao, Yi Li, Riteng Zhang, Tunan Zhao, Ting Yang, Yuqiao Meng, Weicheng Ma, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17056)  

**Abstract**: AI is transforming education by enabling powerful tools that enhance learning experiences. Among recent advancements, large language models (LLMs) hold particular promise for revolutionizing how learners interact with educational content. In this work, we investigate the potential of LLMs to support standardized test preparation by focusing on English Standardized Tests (ESTs). Specifically, we assess their ability to generate accurate and contextually appropriate solutions across a diverse set of EST question types. We introduce ESTBOOK, a comprehensive benchmark designed to evaluate the capabilities of LLMs in solving EST questions. ESTBOOK aggregates five widely recognized tests, encompassing 29 question types and over 10,576 questions across multiple modalities, including text, images, audio, tables, and mathematical symbols. Using ESTBOOK, we systematically evaluate both the accuracy and inference efficiency of LLMs. Additionally, we propose a breakdown analysis framework that decomposes complex EST questions into task-specific solution steps. This framework allows us to isolate and assess LLM performance at each stage of the reasoning process. Evaluation findings offer insights into the capability of LLMs in educational contexts and point toward targeted strategies for improving their reliability as intelligent tutoring systems. 

**Abstract (ZH)**: AI正在通过启用强大的工具来革新教育，这些工具能够提升学习体验。近年来，大型语言模型（LLMs）特别有潜力从根本上变革学习者与教育资源的互动方式。在本文中，我们探讨了LLMs在支持标准化考试准备方面的潜力，重点关注英语标准化考试（ESTs）。具体而言，我们评估了它们生成准确且上下文适当解决方案的能力，涉及多样化的EST题型。我们引入了ESTBOOK，这是一个全面的基准测试，旨在评估LLMs解决EST问题的能力。ESTBOOK整合了五种广泛认可的测试，涵盖29种题型和超过10,576道问题，涉及多种模态，包括文本、图像、音频、表格和数学符号。通过ESTBOOK，我们系统地评估了LLMs的准确性和推理效率。此外，我们提出了一种分解分析框架，将复杂的EST问题分解为特定任务的解决步骤。该框架使我们能够隔离并评估LLMs在推理过程的每个阶段的表现。评估结果提供了关于LLMs在教育情境下的能力洞察，并指出了提高其作为智能辅导系统的可靠性的策略。 

---
# METHOD: Modular Efficient Transformer for Health Outcome Discovery 

**Title (ZH)**: 模块化高效变压器在健康结果发现中的应用 

**Authors**: Linglong Qian, Zina Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17054)  

**Abstract**: Recent advances in transformer architectures have revolutionised natural language processing, but their application to healthcare domains presents unique challenges. Patient timelines are characterised by irregular sampling, variable temporal dependencies, and complex contextual relationships that differ substantially from traditional language tasks. This paper introduces \METHOD~(Modular Efficient Transformer for Health Outcome Discovery), a novel transformer architecture specifically designed to address the challenges of clinical sequence modelling in electronic health records. \METHOD~integrates three key innovations: (1) a patient-aware attention mechanism that prevents information leakage whilst enabling efficient batch processing; (2) an adaptive sliding window attention scheme that captures multi-scale temporal dependencies; and (3) a U-Net inspired architecture with dynamic skip connections for effective long sequence processing. Evaluations on the MIMIC-IV database demonstrate that \METHOD~consistently outperforms the state-of-the-art \ETHOS~model, particularly in predicting high-severity cases that require urgent clinical intervention. \METHOD~exhibits stable performance across varying inference lengths, a crucial feature for clinical deployment where patient histories vary significantly in length. Analysis of learned embeddings reveals that \METHOD~better preserves clinical hierarchies and relationships between medical concepts. These results suggest that \METHOD~represents a significant advancement in transformer architectures optimised for healthcare applications, providing more accurate and clinically relevant predictions whilst maintaining computational efficiency. 

**Abstract (ZH)**: Recent Advances in Transformer Architectures for Healthcare Domain Applications: Introducing \METHOD~(Modular Efficient Transformer for Health Outcome Discovery) 

---
# Social preferences with unstable interactive reasoning: Large language models in economic trust games 

**Title (ZH)**: 社会偏好与不稳定互动推理：大规模语言模型在经济信任游戏中的表现 

**Authors**: Ou Jiamin, Eikmans Emile, Buskens Vincent, Pankowska Paulina, Shan Yuli  

**Link**: [PDF](https://arxiv.org/pdf/2505.17053)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable capabilities in understanding human languages, this study explores how they translate this understanding into social exchange contexts that capture certain essences of real world human interactions. Three LLMs - ChatGPT-4, Claude, and Bard - were placed in economic trust games where players balance self-interest with trust and reciprocity, making decisions that reveal their social preferences and interactive reasoning abilities. Our study shows that LLMs deviate from pure self-interest and exhibit trust and reciprocity even without being prompted to adopt a specific persona. In the simplest one-shot interaction, LLMs emulated how human players place trust at the beginning of such a game. Larger human-machine divergences emerged in scenarios involving trust repayment or multi-round interactions, where decisions were influenced by both social preferences and interactive reasoning. LLMs responses varied significantly when prompted to adopt personas like selfish or unselfish players, with the impact outweighing differences between models or game types. Response of ChatGPT-4, in an unselfish or neutral persona, resembled the highest trust and reciprocity, surpassing humans, Claude, and Bard. Claude and Bard displayed trust and reciprocity levels that sometimes exceeded and sometimes fell below human choices. When given selfish personas, all LLMs showed lower trust and reciprocity than humans. Interactive reasoning to the actions of counterparts or changing game mechanics appeared to be random rather than stable, reproducible characteristics in the response of LLMs, though some improvements were observed when ChatGPT-4 responded in selfish or unselfish personas. 

**Abstract (ZH)**: 大型语言模型在社会交换情境中展现的信任与互惠能力探究 

---
# SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs 

**Title (ZH)**: SpecEdge: 可扩展的边缘辅助服务框架以支持交互式大语言模型 

**Authors**: Jinwoo Park, Seunggeun Cho, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17052)  

**Abstract**: Large language models (LLMs) power many modern applications, but serving them at scale remains costly and resource-intensive. Current server-centric systems overlook consumer-grade GPUs at the edge. We introduce SpecEdge, an edge-assisted inference framework that splits LLM workloads between edge and server GPUs using a speculative decoding scheme, exchanging only token outputs over the network. SpecEdge employs proactive edge drafting to overlap edge token creation with server verification and pipeline-aware scheduling that interleaves multiple user requests to increase server-side throughput. Experiments show SpecEdge enhances overall cost efficiency by 1.91x through achieving 2.22x server throughput, and reduces inter token latency by 11.24% compared to a server-only baseline, introducing a scalable, cost-effective paradigm for LLM serving. 

**Abstract (ZH)**: Large Language Models (LLMs) 助力许多现代应用，但大规模提供这些模型依然成本高昂且资源密集。当前以服务器为中心的系统忽略了边缘处的消费级 GPU。我们引入了 SpecEdge，这是一种边缘辅助推理框架，通过投机性的解码方案将 LLM 工作负载分拆到边缘和服务器 GPU 之间，仅在网络中交换标记输出。SpecEdge 使用积极的边缘预选拔方案，使边缘标记创建与服务器验证重叠，并采用Aware流水线调度方案交错多个用户请求以提高服务器端吞吐量。实验表明，SpecEdge 通过实现 2.22 倍的服务器吞吐量，将总体成本效率提升 1.91 倍，并将标记间延迟减少了 11.24%，相较于仅服务器的基础方案，引入了一种可扩展且成本效益高的 LLM 提供范式。 

---
# Embedding-to-Prefix: Parameter-Efficient Personalization for Pre-Trained Large Language Models 

**Title (ZH)**: Embedding-to-Prefix: 参数高效的预训练大型语言模型个性化方法 

**Authors**: Bernd Huber, Ghazal Fazelnia, Andreas Damianou, Sebastian Peleato, Max Lefarov, Praveen Ravichandran, Marco De Nadai, Mounia Lalmas-Roellke, Paul N. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2505.17051)  

**Abstract**: Large language models (LLMs) excel at generating contextually relevant content. However, tailoring these outputs to individual users for effective personalization is a significant challenge. While rich user-specific information often exists as pre-existing user representations, such as embeddings learned from preferences or behaviors, current methods to leverage these for LLM personalization typically require costly fine-tuning or token-heavy prompting. We propose Embedding-to-Prefix (E2P), a parameter-efficient method that injects pre-computed context embeddings into an LLM's hidden representation space through a learned projection to a single soft token prefix. This enables effective personalization while keeping the backbone model frozen and avoiding expensive adaptation techniques. We evaluate E2P across two public datasets and in a production setting: dialogue personalization on Persona-Chat, contextual headline generation on PENS, and large-scale personalization for music and podcast consumption. Results show that E2P preserves contextual signals and achieves strong performance with minimal computational overhead, offering a scalable, efficient solution for contextualizing generative AI systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成上下文相关内容方面表现出色。然而，为了有效进行个性化调整，将这些输出针对个别用户进行定制仍是一个重大挑战。虽然丰富的用户特定信息通常以预训练的用户表示形式存在，比如偏好或行为学到的嵌入，当前用于利用这些信息进行LLM个性化的方法通常需要成本高昂的微调或大量提示。我们提出了一种参数高效的Embedding-to-Prefix（E2P）方法，通过学习投影将预先计算的上下文嵌入注入到LLM的隐藏表示空间中的单个软令牌前缀中。这使得在冻结主模型的情况下，能够有效进行个性化调整，并避免昂贵的适应技术。我们在两个公开数据集和实际生产环境中评估了E2P：Persona-Chat中的对话个性化、PENS中的上下文标题生成以及音乐和播客的大型个性化。结果表明，E2P能够保留上下文信号，并在最小的计算开销下实现出色的性能，提供了一种可扩展且高效的为生成型AI系统上下文化的方法。 

---
# Towards Robust Evaluation of STEM Education: Leveraging MLLMs in Project-Based Learning 

**Title (ZH)**: 面向STEM教育稳健评估的方法：基于项目学习的大型语言模型应用 

**Authors**: Yanhao Jia, Xinyi Wu, Qinglin Zhang, Yiran Qin, Luwei Xiao, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17050)  

**Abstract**: Project-Based Learning (PBL) involves a variety of highly correlated multimodal data, making it a vital educational approach within STEM disciplines. With the rapid development of multimodal large language models (MLLMs), researchers have begun exploring their potential to enhance tasks such as information retrieval, knowledge comprehension, and data generation in educational settings. However, existing benchmarks fall short in providing both a free-form output structure and a rigorous human expert validation process, limiting their effectiveness in evaluating real-world educational tasks. Additionally, few methods have developed automated pipelines to assist with the complex responsibilities of teachers leveraging MLLMs, largely due to model hallucination and instability, which lead to unreliable implementation. To address this gap, we introduce PBLBench, a novel benchmark designed to evaluate complex reasoning grounded in domain-specific knowledge and long-context understanding, thereby challenging models with tasks that closely resemble those handled by human experts. To establish reliable ground truth, we adopt the Analytic Hierarchy Process (AHP), utilizing expert-driven pairwise comparisons to derive structured and weighted evaluation criteria. We assess the performance of 15 leading MLLMs/LLMs using PBLBench and demonstrate that even the most advanced models achieve only 59% rank accuracy, underscoring the significant challenges presented by this benchmark. We believe PBLBench will serve as a catalyst for the development of more capable AI agents, ultimately aiming to alleviate teacher workload and enhance educational productivity. 

**Abstract (ZH)**: 基于项目的学习（PBL）涉及多种高度相关的多模态数据，是STEM学科中一种重要的教育方法。随着多模态大语言模型（MLLMs）的迅速发展，研究人员开始探索它们在信息检索、知识理解、数据生成等教育任务中的潜力。然而，现有的基准测试在提供开放式输出结构和严格的人类专家验证过程方面存在不足，限制了它们在评估实际教育任务方面的有效性。此外，很少有方法开发出自动化的流程来协助教师利用MLLMs，主要由于模型的幻觉和不稳定性的原因，导致实现不可靠。为填补这一空白，我们介绍了PBLBench，这是一种新型基准测试，旨在评估基于领域特定知识和长上下文理解的复杂推理，从而通过任务挑战模型，这些任务与人类专家处理的任务极为相似。为建立可靠的基准事实，我们采用了层次分析过程（AHP），利用专家驱动的两两比较来推导出结构化和加权的评估标准。我们使用PBLBench评估了15个领先的MLLMs/LLMs，并展示了即使是最先进的模型也只能达到59%的排名准确率，突显了该基准测试带来的巨大挑战。我们相信，PBLBench将为更强大的人工智能代理的开发起到催化作用，最终旨在减轻教师的工作负担并提高教育生产率。 

---
# Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations 

**Title (ZH)**: 基于LLM的招聘决策中的性别和职位偏见：来自简历评估的证据 

**Authors**: David Rozado  

**Link**: [PDF](https://arxiv.org/pdf/2505.17049)  

**Abstract**: This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning. 

**Abstract (ZH)**: 本研究考察了大型语言模型（LLMs）在基于简历或 curriculum vitae (CV) 评估专业候选人时的行为。在涉及22个领先LLM的实验中，每个模型系统地获得了职位描述，并与一对匹配职业的CV组合，其中一个带有男性名字，另一个带有女性名字，然后被要求选出更适合该职位的候选人。每对CV呈递两次，名字位置交换以确保观察到的候选人选择偏好源自性别名字线索。尽管不同性别有相同的专业资格，所有LLM在70个不同职业中一致地倾向于选择带有女性名字的候选人。在CV中增加明确的性别字段（男性/女性）进一步增加了对女性申请者的偏好。将性别名字替换为性别中立标识符“Candidate A”和“Candidate B”后，几款模型更倾向于选择“Candidate A”。在这些性别中立标识符之间平衡性别分配导致了候选人选择的性别平衡。当单独评价CV而不是成对比较时，LLM整体上为女性CV分配了略高的平均评分，但效应大小很小。在候选人的名字旁边列出偏好的称谓（他/他或她/她）略微增加了候选人被选中的几率，不论其性别。最后，大多数模型在选择列表中第一个提及的候选人时表现出明显的偏好。这些发现强调了在高风险自主决策背景下部署LLM时需要谨慎，并对LLM是否一贯应用原则性推理表示怀疑。 

---
# Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally 

**Title (ZH)**: Worlds的语言纽带：全球央行政策沟通解码的统一框架 

**Authors**: Agam Shah, Siddhant Sukhani, Huzaifa Pardawala, Saketh Budideti, Riya Bhadani, Rudra Gopal, Siddhartha Somani, Michael Galarnyk, Soungmin Lee, Arnav Hiray, Akshar Ravichandran, Eric Kim, Pranav Aluru, Joshua Zhang, Sebastian Jaskowski, Veer Guda, Meghaj Tarte, Liqin Ye, Spencer Gosden, Rutwik Routu, Rachel Yuh, Sloka Chava, Sahasra Chava, Dylan Patrick Kelly, Aiden Chiang, Harsit Mittal, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.17048)  

**Abstract**: Central banks around the world play a crucial role in maintaining economic stability. Deciphering policy implications in their communications is essential, especially as misinterpretations can disproportionately impact vulnerable populations. To address this, we introduce the World Central Banks (WCB) dataset, the most comprehensive monetary policy corpus to date, comprising over 380k sentences from 25 central banks across diverse geographic regions, spanning 28 years of historical data. After uniformly sampling 1k sentences per bank (25k total) across all available years, we annotate and review each sentence using dual annotators, disagreement resolutions, and secondary expert reviews. We define three tasks: Stance Detection, Temporal Classification, and Uncertainty Estimation, with each sentence annotated for all three. We benchmark seven Pretrained Language Models (PLMs) and nine Large Language Models (LLMs) (Zero-Shot, Few-Shot, and with annotation guide) on these tasks, running 15,075 benchmarking experiments. We find that a model trained on aggregated data across banks significantly surpasses a model trained on an individual bank's data, confirming the principle "the whole is greater than the sum of its parts." Additionally, rigorous human evaluations, error analyses, and predictive tasks validate our framework's economic utility. Our artifacts are accessible through the HuggingFace and GitHub under the CC-BY-NC-SA 4.0 license. 

**Abstract (ZH)**: 全球中央银行（WCB）数据集：最全面的货币政策语料库及其应用 

---
# Assessing the Quality of AI-Generated Clinical Notes: A Validated Evaluation of a Large Language Model Scribe 

**Title (ZH)**: 评估AI生成临床笔记的质量：大型语言模型记录员的验证评价 

**Authors**: Erin Palm, Astrit Manikantan, Mark E. Pepin, Herprit Mahal, Srikanth Subramanya Belwadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17047)  

**Abstract**: In medical practices across the United States, physicians have begun implementing generative artificial intelligence (AI) tools to perform the function of scribes in order to reduce the burden of documenting clinical encounters. Despite their widespread use, no established methods exist to gauge the quality of AI scribes. To address this gap, we developed a blinded study comparing the relative performance of large language model (LLM) generated clinical notes with those from field experts based on audio-recorded clinical encounters. Quantitative metrics from the Physician Documentation Quality Instrument (PDQI9) provided a framework to measure note quality, which we adapted to assess relative performance of AI generated notes. Clinical experts spanning 5 medical specialties used the PDQI9 tool to evaluate specialist-drafted Gold notes and LLM authored Ambient notes. Two evaluators from each specialty scored notes drafted from a total of 97 patient visits. We found uniformly high inter rater agreement (RWG greater than 0.7) between evaluators in general medicine, orthopedics, and obstetrics and gynecology, and moderate (RWG 0.5 to 0.7) to high inter rater agreement in pediatrics and cardiology. We found a modest yet significant difference in the overall note quality, wherein Gold notes achieved a score of 4.25 out of 5 and Ambient notes scored 4.20 out of 5 (p = 0.04). Our findings support the use of the PDQI9 instrument as a practical method to gauge the quality of LLM authored notes, as compared to human-authored notes. 

**Abstract (ZH)**: 在美国医疗实践中，医生开始使用生成型人工智能工具来担任记录临床 encounter 的角色，以减轻文档记录负担。尽管这些工具被广泛应用，但目前缺乏评估人工智能记录员质量的方法。为填补这一空白，我们开发了一项盲测研究，比较大型语言模型生成的临床笔记与基于音频记录的临床 encounter 的专家笔记的质量。我们使用医生记录质量评估工具（PDQI9）中的定量指标作为测量框架，并据此评估人工智能生成笔记的相对性能。涵盖五个医学专科的临床专家使用 PDQI9 工具评估专科起草的金笔记和大型语言模型撰写的环境笔记。每种专科的两名评估者对总共97个病人访问记录的笔记进行了评分。我们发现，在一般医学、骨科和妇产科中，评估者之间的一致性极高（RWG大于0.7），而在儿科和心脏病学中，一致性为中等（RWG 0.5到0.7）至高。我们发现整体笔记质量存在微小但显著的差异，其中金笔记得分为4.25分（满分5分），环境笔记得分为4.20分（p=0.04）。我们的研究结果支持使用 PDQI9 作为评估大型语言模型生成笔记质量的实用方法，相较于人类手写的笔记。 

---
# QRA++: Quantified Reproducibility Assessment for Common Types of Results in Natural Language Processing 

**Title (ZH)**: QRA++: 量化自然语言处理中常见类型结果的可再现性评估 

**Authors**: Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17043)  

**Abstract**: Reproduction studies reported in NLP provide individual data points which in combination indicate worryingly low levels of reproducibility in the field. Because each reproduction study reports quantitative conclusions based on its own, often not explicitly stated, criteria for reproduction success/failure, the conclusions drawn are hard to interpret, compare, and learn from. In this paper, we present QRA++, a quantitative approach to reproducibility assessment that (i) produces continuous-valued degree of reproducibility assessments at three levels of granularity; (ii) utilises reproducibility measures that are directly comparable across different studies; and (iii) grounds expectations about degree of reproducibility in degree of similarity between experiments. QRA++ enables more informative reproducibility assessments to be conducted, and conclusions to be drawn about what causes reproducibility to be better/poorer. We illustrate this by applying QRA++ to three example sets of comparable experiments, revealing clear evidence that degree of reproducibility depends on similarity of experiment properties, but also system type and evaluation method. 

**Abstract (ZH)**: NLP领域中报道的再现研究提供了单一的数据点，这些数据点结合在一起显示出令人担忧的低再现性水平。由于每项再现研究基于其自身（常常未明确陈述）的再现成功/失败标准报告定量结论，因此得出的结论难以解释、比较和学习。在本文中，我们提出QRA++，这是一种定量的再现性评估方法，它能够（i）在三个粒度级别上产生连续值的再现性评估；（ii）利用可以直接在不同研究之间进行比较的再现性度量；（iii）将对再现性的期望建立在实验相似性程度的基础上。QRA++能够进行更具信息量的再现性评估，并从中得出再现性更好或较差的原因。我们通过将QRA++应用于三个可比实验集的例子，展示了这一点，结果显示再现性程度取决于实验属性的相似性，同时也受系统类型和评估方法的影响。 

---
# ReMi: A Random Recurrent Neural Network Approach to Music Production 

**Title (ZH)**: ReMi: 一种用于音乐生产的目的随机循环神经网络方法 

**Authors**: Hugo Chateau-Laurent, Tara Vanhatalo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17023)  

**Abstract**: Generative artificial intelligence raises concerns related to energy consumption, copyright infringement and creative atrophy. We show that randomly initialized recurrent neural networks can produce arpeggios and low-frequency oscillations that are rich and configurable. In contrast to end-to-end music generation that aims to replace musicians, our approach expands their creativity while requiring no data and much less computational power. More information can be found at: this https URL 

**Abstract (ZH)**: 生成型人工智能引发了与能耗、版权侵犯和创造力萎缩相关的问题。我们展示了随机初始化的循环神经网络可以产生丰富且可配置的和弦进行和低频振荡。与旨在替代 musicians 的端到端音乐生成方法不同，我们的方法能够扩展 musicians 的创造力，且无需数据和较少的计算资源。更多详细信息请参见：this https URL 

---
# Normalized Cut with Reinforcement Learning in Constrained Action Space 

**Title (ZH)**: 约束动作空间中的归一化割与强化学习 

**Authors**: Qize Jiang, Linsey Pang, Alice Gatti, Mahima Aggarwal, Giovanna Vantini, Xiaosong Ma, Weiwei Sun, Sanjay Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2505.13986)  

**Abstract**: Reinforcement Learning (RL) has emerged as an important paradigm to solve combinatorial optimization problems primarily due to its ability to learn heuristics that can generalize across problem instances. However, integrating external knowledge that will steer combinatorial optimization problem solutions towards domain appropriate outcomes remains an extremely challenging task. In this paper, we propose the first RL solution that uses constrained action spaces to guide the normalized cut problem towards pre-defined template instances. Using transportation networks as an example domain, we create a Wedge and Ring Transformer that results in graph partitions that are shaped in form of Wedges and Rings and which are likely to be closer to natural optimal partitions. However, our approach is general as it is based on principles that can be generalized to other domains. 

**Abstract (ZH)**: 强化学习(RL)作为一种解决组合优化问题的重要范式，主要得益于其学习能够泛化到不同问题实例的启发式方法的能力。然而，将外部知识集成到组合优化问题中，以引导其解向领域特定的结果方向发展，仍然是一个极具挑战性的任务。在本文中，我们提出了一种使用受限动作空间的RL解决方案，以引导归一化切分问题趋向预定义的模板实例。以交通网络为例，我们创建了一个Wedge和Ring转换器，从而产生形状类似Wedge和Ring的图划分，并且这些划分更有可能接近自然最优划分。然而，我们的方法是通用的，因为它基于可以应用于其他领域的原则。 

---
