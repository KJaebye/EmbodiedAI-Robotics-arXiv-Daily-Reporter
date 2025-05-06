# DriveAgent: Multi-Agent Structured Reasoning with LLM and Multimodal Sensor Fusion for Autonomous Driving 

**Title (ZH)**: DriveAgent：基于LLM和多模态传感器融合的多Agent结构化推理自主驾驶 

**Authors**: Xinmeng Hou, Wuqi Wang, Long Yang, Hao Lin, Jinglun Feng, Haigen Min, Xiangmo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02123)  

**Abstract**: We introduce DriveAgent, a novel multi-agent autonomous driving framework that leverages large language model (LLM) reasoning combined with multimodal sensor fusion to enhance situational understanding and decision-making. DriveAgent uniquely integrates diverse sensor modalities-including camera, LiDAR, GPS, and IMU-with LLM-driven analytical processes structured across specialized agents. The framework operates through a modular agent-based pipeline comprising four principal modules: (i) a descriptive analysis agent identifying critical sensor data events based on filtered timestamps, (ii) dedicated vehicle-level analysis conducted by LiDAR and vision agents that collaboratively assess vehicle conditions and movements, (iii) environmental reasoning and causal analysis agents explaining contextual changes and their underlying mechanisms, and (iv) an urgency-aware decision-generation agent prioritizing insights and proposing timely maneuvers. This modular design empowers the LLM to effectively coordinate specialized perception and reasoning agents, delivering cohesive, interpretable insights into complex autonomous driving scenarios. Extensive experiments on challenging autonomous driving datasets demonstrate that DriveAgent is achieving superior performance on multiple metrics against baseline methods. These results validate the efficacy of the proposed LLM-driven multi-agent sensor fusion framework, underscoring its potential to substantially enhance the robustness and reliability of autonomous driving systems. 

**Abstract (ZH)**: DriveAgent：一种结合大型语言模型推理和多模态传感器融合的新型多代理自主驾驶框架 

---
# Enhancing LLMs' Clinical Reasoning with Real-World Data from a Nationwide Sepsis Registry 

**Title (ZH)**: 基于全国性脓毒症登记数据增强LLMs的临床推理能力 

**Authors**: Junu Kim, Chaeeun Shim, Sungjin Park, Su Yeon Lee, Gee Young Suh, Chae-Man Lim, Seong Jin Choi, Song Mi Moon, Kyoung-Ho Song, Eu Suk Kim, Hong Bin Kim, Sejoong Kim, Chami Im, Dong-Wan Kang, Yong Soo Kim, Hee-Joon Bae, Sung Yoon Lim, Han-Gil Jeong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.02722)  

**Abstract**: Although large language models (LLMs) have demonstrated impressive reasoning capabilities across general domains, their effectiveness in real-world clinical practice remains limited. This is likely due to their insufficient exposure to real-world clinical data during training, as such data is typically not included due to privacy concerns. To address this, we propose enhancing the clinical reasoning capabilities of LLMs by leveraging real-world clinical data. We constructed reasoning-intensive questions from a nationwide sepsis registry and fine-tuned Phi-4 on these questions using reinforcement learning, resulting in C-Reason. C-Reason exhibited strong clinical reasoning capabilities on the in-domain test set, as evidenced by both quantitative metrics and expert evaluations. Furthermore, its enhanced reasoning capabilities generalized to a sepsis dataset involving different tasks and patient cohorts, an open-ended consultations on antibiotics use task, and other diseases. Future research should focus on training LLMs with large-scale, multi-disease clinical datasets to develop more powerful, general-purpose clinical reasoning models. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在通用领域展示了令人印象深刻的推理能力，但在实际临床实践中的有效性仍然有限。这可能是由于它们在训练过程中缺乏对真实世界临床数据的充分接触，因为出于隐私考虑，这类数据通常被排除在外。为了解决这个问题，我们提出通过利用真实世界临床数据来增强LLMs的临床推理能力。我们从全国性脓毒症登记数据中构建了推理密集型问题，并使用强化学习对Phi-4进行了微调，得到了C-Reason。C-Reason在领域内测试集上展现了强大的临床推理能力，这不仅有定量指标的支持，还有专家评价的验证。此外，其增强的推理能力还迁移到了涉及不同任务和患者群体的脓毒症数据集、抗生素使用开放式咨询任务以及其它疾病中。未来的研究应侧重于使用大规模、多疾病临床数据集来训练LLMs，以开发更具强大且通用的临床推理模型。 

---
# Technical Report: Evaluating Goal Drift in Language Model Agents 

**Title (ZH)**: 技术报告：评估语言模型代理的目标漂移 

**Authors**: Rauno Arike, Elizabeth Donoway, Henning Bartsch, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.02709)  

**Abstract**: As language models (LMs) are increasingly deployed as autonomous agents, their robust adherence to human-assigned objectives becomes crucial for safe operation. When these agents operate independently for extended periods without human oversight, even initially well-specified goals may gradually shift. Detecting and measuring goal drift - an agent's tendency to deviate from its original objective over time - presents significant challenges, as goals can shift gradually, causing only subtle behavioral changes. This paper proposes a novel approach to analyzing goal drift in LM agents. In our experiments, agents are first explicitly given a goal through their system prompt, then exposed to competing objectives through environmental pressures. We demonstrate that while the best-performing agent (a scaffolded version of Claude 3.5 Sonnet) maintains nearly perfect goal adherence for more than 100,000 tokens in our most difficult evaluation setting, all evaluated models exhibit some degree of goal drift. We also find that goal drift correlates with models' increasing susceptibility to pattern-matching behaviors as the context length grows. 

**Abstract (ZH)**: 作为语言模型（LMs）日益被部署为自主代理，其严格遵循人类指定目标的能力对于安全运行至关重要。当这些代理在缺乏人类监督的情况下独立运行较长时间时，即使最初明确的目标也可能逐渐偏移。检测和量化代理随时间偏离原始目标的趋势是一项重大挑战，因为目标的偏移可能是渐进的，导致行为上的细微变化。本文提出了一种新的方法来分析LM代理的目标偏移。在我们的实验中，代理首先通过系统提示明确给出一个目标，然后通过环境压力暴露于竞争性目标。我们证明，在我们最困难的评估环境中，性能最佳的代理（Claude 3.5 Sonnet的一个分层版本）保持几乎完美的目标一致性超过10万个标记，但所有评估的模型在不同程度上都表现出目标偏移。我们还发现，随着上下文长度的增长，目标偏移与模型越来越容易出现模式匹配行为呈相关。 

---
# Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play 

**Title (ZH)**: Voila: 基于语音-语言基础模型的实时自主交互与语音角色扮演 

**Authors**: Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02707)  

**Abstract**: A voice AI agent that blends seamlessly into daily life would interact with humans in an autonomous, real-time, and emotionally expressive manner. Rather than merely reacting to commands, it would continuously listen, reason, and respond proactively, fostering fluid, dynamic, and emotionally resonant interactions. We introduce Voila, a family of large voice-language foundation models that make a step towards this vision. Voila moves beyond traditional pipeline systems by adopting a new end-to-end architecture that enables full-duplex, low-latency conversations while preserving rich vocal nuances such as tone, rhythm, and emotion. It achieves a response latency of just 195 milliseconds, surpassing the average human response time. Its hierarchical multi-scale Transformer integrates the reasoning capabilities of large language models (LLMs) with powerful acoustic modeling, enabling natural, persona-aware voice generation -- where users can simply write text instructions to define the speaker's identity, tone, and other characteristics. Moreover, Voila supports over one million pre-built voices and efficient customization of new ones from brief audio samples as short as 10 seconds. Beyond spoken dialogue, Voila is designed as a unified model for a wide range of voice-based applications, including automatic speech recognition (ASR), Text-to-Speech (TTS), and, with minimal adaptation, multilingual speech translation. Voila is fully open-sourced to support open research and accelerate progress toward next-generation human-machine interactions. 

**Abstract (ZH)**: 一种能够无缝融入日常生活的语音AI代理将在自主、实时和情绪表达的方式下与人类互动。它将不仅响应命令，还会持续地聆听、推理并主动响应，从而促进流畅、动态且富有情感共鸣的互动。我们介绍了一种名为Voila的系列大规模语音-语言基础模型，向着这一愿景迈出了重要一步。Voila 采用了新的端到端架构，超越了传统流水线系统，实现了全双工、低延迟的对话，同时保留了丰富的语音细微差别，如音调、节奏和情感。它实现了仅195毫秒的响应延迟，超过了平均人类反应时间。其分层多尺度Transformer将大型语言模型（LLMs）的推理能力与强大的声学建模相结合，使自然的、立足于人设的声音生成成为可能——用户只需编写文本指令即可定义说话人的身份、语气和其他特征。此外，Voila 支持超过一百万种预建声音，并能从如10秒短音频样本中高效定制新声音。除了对话应用，Voila 还被设计为一种统一模型，适用于一系列语音应用，包括自动语音识别（ASR）、文本转语音（TTS），并通过最少的适应实现多语言语音翻译。Voila 完全开源，以支持开放研究并加速下一代人机交互的进展。 

---
# A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law 

**Title (ZH)**: 基于慢思考机制的强化学习与推理时缩放法则驱动的大语言模型综述 

**Authors**: Qianjun Pan, Wenkai Ji, Yuyang Ding, Junsong Li, Shilian Chen, Junyi Wang, Jie Zhou, Qin Chen, Min Zhang, Yulan Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.02665)  

**Abstract**: This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems. 

**Abstract (ZH)**: 本综述探索了旨在模拟“慢思考”推理过程的大语言模型（LLMs）的 Recent Advancements，“慢思考”推理过程受Kahneman的《思考，快与慢》中的人类认知启发。这些模型，如OpenAI的o1，侧重于在复杂任务（如数学推理、视觉推理、医疗诊断和多智能体辩论）中动态调整计算资源的扩展。本文概述了推理LLMs的发展，并列出了其关键技术。通过综合分析超过100篇研究文献，该综述勾勒出结合人类级深度思考和可扩展高效推理能力的LLMs的发展路径。综述将方法分为三类：（1）测试时动态调整计算基于任务复杂性通过搜索和采样、动态验证；（2）强化学习通过迭代改进利用策略网络、奖励模型和自我进化策略细化决策；（3）慢思考框架（例如，长CoT、分层过程）通过可管理的步骤结构化问题解决。综述突出了该领域的挑战和进一步发展方向。理解并推进LLMs的推理能力对于在科学发现和决策支持系统等实际应用中充分发挥其潜力至关重要。 

---
# Recursive Decomposition with Dependencies for Generic Divide-and-Conquer Reasoning 

**Title (ZH)**: 基于依赖关系的递归分解通用分而治之推理 

**Authors**: Sergio Hernández-Gutiérrez, Minttu Alakuijala, Alexander V. Nikitin, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02576)  

**Abstract**: Reasoning tasks are crucial in many domains, especially in science and engineering. Although large language models (LLMs) have made progress in reasoning tasks using techniques such as chain-of-thought and least-to-most prompting, these approaches still do not effectively scale to complex problems in either their performance or execution time. Moreover, they often require additional supervision for each new task, such as in-context examples. In this work, we introduce Recursive Decomposition with Dependencies (RDD), a scalable divide-and-conquer method for solving reasoning problems that requires less supervision than prior approaches. Our method can be directly applied to a new problem class even in the absence of any task-specific guidance. Furthermore, RDD supports sub-task dependencies, allowing for ordered execution of sub-tasks, as well as an error recovery mechanism that can correct mistakes made in previous steps. We evaluate our approach on two benchmarks with six difficulty levels each and in two in-context settings: one with task-specific examples and one without. Our results demonstrate that RDD outperforms other methods in a compute-matched setting as task complexity increases, while also being more computationally efficient. 

**Abstract (ZH)**: 递归分解依赖方法（RDD）在解决推理问题中的可扩展方法 

---
# Beyond the model: Key differentiators in large language models and multi-agent services 

**Title (ZH)**: 超越模型：大型语言模型与多Agent服务的关键差异化因素 

**Authors**: Muskaan Goyal, Pranav Bhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02489)  

**Abstract**: With the launch of foundation models like DeepSeek, Manus AI, and Llama 4, it has become evident that large language models (LLMs) are no longer the sole defining factor in generative AI. As many now operate at comparable levels of capability, the real race is not about having the biggest model but optimizing the surrounding ecosystem, including data quality and management, computational efficiency, latency, and evaluation frameworks. This review article delves into these critical differentiators that ensure modern AI services are efficient and profitable. 

**Abstract (ZH)**: 随着DeepSeek、Manus AI和Llama 4等基础模型的推出，显而易见的是，大规模语言模型（LLMs）已不再是生成式AI的唯一决定性因素。由于许多模型现在在能力上可与之匹敌，真正的竞争焦点已转向优化围绕其运行的生态系统，包括数据质量与管理、计算效率、延迟以及评估框架。本文探讨这些关键差异点，以确保现代AI服务既高效又盈利。 

---
# El Agente: An Autonomous Agent for Quantum Chemistry 

**Title (ZH)**: El Agente: 一个自主量子化学代理 

**Authors**: Yunheng Zou, Austin H. Cheng, Abdulrahman Aldossary, Jiaru Bai, Shi Xuan Leong, Jorge Arturo Campos-Gonzalez-Angulo, Changhyeok Choi, Cher Tian Ser, Gary Tom, Andrew Wang, Zijian Zhang, Ilya Yakavets, Han Hao, Chris Crebolder, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2505.02484)  

**Abstract**: Computational chemistry tools are widely used to study the behaviour of chemical phenomena. Yet, the complexity of these tools can make them inaccessible to non-specialists and challenging even for experts. In this work, we introduce El Agente Q, an LLM-based multi-agent system that dynamically generates and executes quantum chemistry workflows from natural language user prompts. The system is built on a novel cognitive architecture featuring a hierarchical memory framework that enables flexible task decomposition, adaptive tool selection, post-analysis, and autonomous file handling and submission. El Agente Q is benchmarked on six university-level course exercises and two case studies, demonstrating robust problem-solving performance (averaging >87% task success) and adaptive error handling through in situ debugging. It also supports longer-term, multi-step task execution for more complex workflows, while maintaining transparency through detailed action trace logs. Together, these capabilities lay the foundation for increasingly autonomous and accessible quantum chemistry. 

**Abstract (ZH)**: 基于LLM的多代理系统El Agente Q动态生成和执行自然语言用户提示下的量子化学工作流 

---
# HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking 

**Title (ZH)**: 层次树规划：通过层次思考增强LLM推理 

**Authors**: Runquan Gui, Zhihai Wang, Jie Wang, Chi Ma, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Defu Lian, Enhong Chen, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02322)  

**Abstract**: Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview. 

**Abstract (ZH)**: 最近的研究进展显著提升了大型语言模型在处理复杂推理任务方面的性能，在数学和逻辑推理等领域取得了显著成就。然而，这些方法在处理复杂规划任务时遇到了挑战，主要原因是扩展的推理步骤、多样的约束条件以及处理多个独立子任务的困难。为应对这些挑战，我们提出了HyperTree Planning (HTP)，一种新的推理范式，通过构建超树结构的规划框架来有效规划。超树结构使大型语言模型能够通过灵活运用分而治之的策略进行分层思考，有效地分解复杂的推理步骤，容纳多样的约束条件，并以井然有序的方式管理多个独立的子任务。我们还介绍了自主规划框架，通过迭代细化和扩展超树结构的规划框架来完成规划过程。实验结果表明，HTP的有效性，在使用Gemini-1.5-Pro在TravelPlanner基准上达到最先进的准确率，性能比o1-preview提高3.6倍。 

---
# LLM-Guided Probabilistic Program Induction for POMDP Model Estimation 

**Title (ZH)**: LLM 引导的概率程序归纳及其在 POMDP 模型估计中的应用 

**Authors**: Aidan Curtis, Hao Tang, Thiago Veloso, Kevin Ellis, Tomás Lozano-Pérez, Leslie Pack Kaelbling  

**Link**: [PDF](https://arxiv.org/pdf/2505.02216)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) model decision making under uncertainty. While there are many approaches to approximately solving POMDPs, we aim to address the problem of learning such models. In particular, we are interested in a subclass of POMDPs wherein the components of the model, including the observation function, reward function, transition function, and initial state distribution function, can be modeled as low-complexity probabilistic graphical models in the form of a short probabilistic program. Our strategy to learn these programs uses an LLM as a prior, generating candidate probabilistic programs that are then tested against the empirical distribution and adjusted through feedback. We experiment on a number of classical toy POMDP problems, simulated MiniGrid domains, and two real mobile-base robotics search domains involving partial observability. Our results show that using an LLM to guide in the construction of a low-complexity POMDP model can be more effective than tabular POMDP learning, behavior cloning, or direct LLM planning. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）模型在不确定性条件下进行决策。我们旨在解决学习此类模型的问题。特别是，我们感兴趣的是POMDP的一个子类，其中模型的组件，包括观察函数、奖励函数、转移函数和初始状态分布函数，可以建模为低复杂度的概率图模型，形式上为简短的概率程序。我们学习这些程序的策略使用LLM作为先验，生成候选概率程序，然后测试它们与经验分布并根据反馈进行调整。我们在一些经典的玩具POMDP问题、模拟的MiniGrid领域以及两个涉及部分可观测性的实际移动机器人搜索领域进行了实验。我们的结果表明，使用LLM引导构建低复杂度POMDP模型比使用表型POMDP学习、行为克隆或直接的LLM规划更为有效。 

---
# Leveraging LLMs to Automate Energy-Aware Refactoring of Parallel Scientific Codes 

**Title (ZH)**: 利用大语言模型自动化并行科学代码的能效优化重构 

**Authors**: Matthew T. Dearing, Yiheng Tao, Xingfu Wu, Zhiling Lan, Valerie Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2505.02184)  

**Abstract**: While large language models (LLMs) are increasingly used for generating parallel scientific code, most current efforts emphasize functional correctness, often overlooking performance and energy considerations. In this work, we propose LASSI-EE, an automated LLM-based refactoring framework that generates energy-efficient parallel code on a target parallel system for a given parallel code as input. Through a multi-stage, iterative pipeline process, LASSI-EE achieved an average energy reduction of 47% across 85% of the 20 HeCBench benchmarks tested on NVIDIA A100 GPUs. Our findings demonstrate the broader potential of LLMs, not only for generating correct code but also for enabling energy-aware programming. We also address key insights and limitations within the framework, offering valuable guidance for future improvements. 

**Abstract (ZH)**: While large language models (LLMs) are increasingly used for generating parallel scientific code, most current efforts emphasize functional correctness, often overlooking performance and energy considerations. In this work, we propose LASSI-EE, an automated LLM-based refactoring framework that generates energy-efficient parallel code on a target parallel system for a given parallel code as input. Through a multi-stage, iterative pipeline process, LASSI-EE achieved an average energy reduction of 47% across 85% of the 20 HeCBench benchmarks tested on NVIDIA A100 GPUs. Our findings demonstrate the broader potential of LLMs, not only for generating correct code but also for enabling energy-aware programming. We also address key insights and limitations within the framework, offering valuable guidance for future improvements。

标题翻译如下：

基于大规模语言模型的能效导向并行代码自动重构框架LASSI-EE 

---
# Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data 

**Title (ZH)**: 注意力机制视角：探索大语言模型处理图形结构数据的方式 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Ming He, Jianpin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.02130)  

**Abstract**: Attention mechanisms are critical to the success of large language models (LLMs), driving significant advancements in multiple fields. However, for graph-structured data, which requires emphasis on topological connections, they fall short compared to message-passing mechanisms on fixed links, such as those employed by Graph Neural Networks (GNNs). This raises a question: ``Does attention fail for graphs in natural language settings?'' Motivated by these observations, we embarked on an empirical study from the perspective of attention mechanisms to explore how LLMs process graph-structured data. The goal is to gain deeper insights into the attention behavior of LLMs over graph structures. We uncovered unique phenomena regarding how LLMs apply attention to graph-structured data and analyzed these findings to improve the modeling of such data by LLMs. The primary findings of our research are: 1) While LLMs can recognize graph data and capture text-node interactions, they struggle to model inter-node relationships within graph structures due to inherent architectural constraints. 2) The attention distribution of LLMs across graph nodes does not align with ideal structural patterns, indicating a failure to adapt to graph topology nuances. 3) Neither fully connected attention nor fixed connectivity is optimal; each has specific limitations in its application scenarios. Instead, intermediate-state attention windows improve LLM training performance and seamlessly transition to fully connected windows during inference. Source code: \href{this https URL}{LLM4Exploration} 

**Abstract (ZH)**: 注意力机制对于大规模语言模型（LLMs）的成功至关重要，推动了多个领域的显著进步。然而，对于需要强调拓扑连接的图结构数据，与固定链接上的消息传递机制（如图神经网络GNNs所使用）相比，注意力机制表现不足。这引发了一个问题：“在自然语言环境中，注意力机制是否对于图数据失效？”受此观察的启发，我们从注意力机制的视角出发，开展了一项实证研究，以探索LLMs如何处理图结构数据，并希望通过这一研究更深入地了解LLMs在图结构上的注意力行为。我们揭示了LLMs处理图结构数据时注意力机制的独特现象，并分析这些发现以改进LLMs对这类数据的建模。我们的主要研究发现包括：1）虽然LLMs可以识别图数据并捕获文本节点间的交互，但在建模图结构内的节点间关系时受到架构约束的限制；2）LLMs在图节点上的注意力分布不符合理想的结构模式，表明其未能适应图拓扑的细微差异；3）完全连接的注意力和固定连接均不理想；每种方式在应用场景中都有特定的局限性。相反，中间状态的注意力窗口能提高LLMs的训练性能，并在推断过程中平滑过渡到完全连接的窗口。源代码：\href{this https URL}{LLM4Exploration}。 

---
# Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets 

**Title (ZH)**: 对抗协作理性化：即使在干净数据集中虚假相关性的风险 

**Authors**: Wei Liu, Zhongyu Niu, Lang Gao, Zhiying Deng, Jun Wang, Haozhao Wang, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02118)  

**Abstract**: This study investigates the self-rationalization framework constructed with a cooperative game, where a generator initially extracts the most informative segment from raw input, and a subsequent predictor utilizes the selected subset for its input. The generator and predictor are trained collaboratively to maximize prediction accuracy. In this paper, we first uncover a potential caveat: such a cooperative game could unintentionally introduce a sampling bias during rationale extraction. Specifically, the generator might inadvertently create an incorrect correlation between the selected rationale candidate and the label, even when they are semantically unrelated in the original dataset. Subsequently, we elucidate the origins of this bias using both detailed theoretical analysis and empirical evidence. Our findings suggest a direction for inspecting these correlations through attacks, based on which we further introduce an instruction to prevent the predictor from learning the correlations. Through experiments on six text classification datasets and two graph classification datasets using three network architectures (GRUs, BERT, and GCN), we show that our method not only significantly outperforms recent rationalization methods, but also achieves comparable or even better results than a representative LLM (llama3.1-8b-instruct). 

**Abstract (ZH)**: 本研究探讨了基于合作博弈构建的自我合理化框架，其中生成器最初从原始输入中提取最具信息性的片段，随后的预测器使用选定的子集作为输入。生成器和预测器协作训练以最大化预测准确性。在本文中，我们首先揭示了一个潜在的问题：这种合作博弈可能会无意中在合理化提取过程中引入采样偏差。具体来说，生成器可能会无意中在选定的理由候选和标签之间创建错误的相关性，即使在原始数据集中它们在语义上是不相关的。随后，我们通过详细的理论分析和实证证据来阐明这种偏差的根源。我们的研究结果表明，可以通过攻击手段来检查这些相关性，并据此介绍了一种指令以防止预测器学习这些相关性。通过在六个文本分类数据集和两个图分类数据集上使用三种网络架构（GRUs、BERT和GCN）进行的实验表明，我们的方法不仅显著优于最近的合理化方法，而且在某些情况下甚至优于一个代表性的大型语言模型（llama3.1-8b-instruct）。 

---
# MemEngine: A Unified and Modular Library for Developing Advanced Memory of LLM-based Agents 

**Title (ZH)**: MemEngine: 一种开发基于LLM的智能体高级内存的统一模块化库 

**Authors**: Zeyu Zhang, Quanyu Dai, Xu Chen, Rui Li, Zhongyang Li, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.02099)  

**Abstract**: Recently, large language model based (LLM-based) agents have been widely applied across various fields. As a critical part, their memory capabilities have captured significant interest from both industrial and academic communities. Despite the proposal of many advanced memory models in recent research, however, there remains a lack of unified implementations under a general framework. To address this issue, we develop a unified and modular library for developing advanced memory models of LLM-based agents, called MemEngine. Based on our framework, we implement abundant memory models from recent research works. Additionally, our library facilitates convenient and extensible memory development, and offers user-friendly and pluggable memory usage. For benefiting our community, we have made our project publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的代理先进记忆模型统一体系——MemEngine 

---
# Retrieval-augmented in-context learning for multimodal large language models in disease classification 

**Title (ZH)**: 基于检索增强的上下文学习方法在 multimodal 大型语言模型中的疾病分类 

**Authors**: Zaifu Zhan, Shuang Zhou, Xiaoshan Zhou, Yongkang Xiao, Jun Wang, Jiawen Deng, He Zhu, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02087)  

**Abstract**: Objectives: We aim to dynamically retrieve informative demonstrations, enhancing in-context learning in multimodal large language models (MLLMs) for disease classification.
Methods: We propose a Retrieval-Augmented In-Context Learning (RAICL) framework, which integrates retrieval-augmented generation (RAG) and in-context learning (ICL) to adaptively select demonstrations with similar disease patterns, enabling more effective ICL in MLLMs. Specifically, RAICL examines embeddings from diverse encoders, including ResNet, BERT, BioBERT, and ClinicalBERT, to retrieve appropriate demonstrations, and constructs conversational prompts optimized for ICL. We evaluated the framework on two real-world multi-modal datasets (TCGA and IU Chest X-ray), assessing its performance across multiple MLLMs (Qwen, Llava, Gemma), embedding strategies, similarity metrics, and varying numbers of demonstrations.
Results: RAICL consistently improved classification performance. Accuracy increased from 0.7854 to 0.8368 on TCGA and from 0.7924 to 0.8658 on IU Chest X-ray. Multi-modal inputs outperformed single-modal ones, with text-only inputs being stronger than images alone. The richness of information embedded in each modality will determine which embedding model can be used to get better results. Few-shot experiments showed that increasing the number of retrieved examples further enhanced performance. Across different similarity metrics, Euclidean distance achieved the highest accuracy while cosine similarity yielded better macro-F1 scores. RAICL demonstrated consistent improvements across various MLLMs, confirming its robustness and versatility.
Conclusions: RAICL provides an efficient and scalable approach to enhance in-context learning in MLLMs for multimodal disease classification. 

**Abstract (ZH)**: 目标：我们旨在动态检索有用的表现示范，以增强多模态大型语言模型（MLLMs）在疾病分类中的上下文学习。
方法：我们提出了一种检索增强的上下文学习（RAICL）框架，该框架结合了检索增强生成（RAG）和上下文学习（ICL），以自适应地选择具有相似疾病模式的表现示范，从而在MLLMs中实现更有效的ICL。具体而言，RAICL检查来自各种编码器（包括ResNet、BERT、BioBERT和ClinicalBERT）的嵌入表示以检索适当的表现示范，并构建优化的对话式提示符以适应ICL。我们在这两个真实的多模态数据集（TCGA和IU胸部X光片）上评估了该框架的表现，评估指标涵盖多种MLLMs（Qwen、Llava、Gemma）、嵌入策略、相似性度量和不同数量的表现示范。
结果：RAICL一致地提高了分类性能。在TCGA数据集上，准确率从0.7854提高到0.8368；在IU胸部X光片数据集上，准确率从0.7924提高到0.8658。多模态输入的表现优于单模态输入，纯文本输入优于单独的图像。每个模态中嵌入的信息的丰富程度将决定哪种嵌入模型可以用于获得更好的结果。通过少量示例实验表明，增加检索示例的数量进一步提高了性能。在不同的相似性度量中，欧氏距离实现了最高的准确率，而余弦相似度在宏F1分数方面表现更好。RAICL在各种MLLMs中表现出一致的改进，验证了其稳健性和通用性。
结论：RAICL提供了一种有效且可扩展的方法，以增强MLLMs在多模态疾病分类中的上下文学习。 

---
# Generative AI in clinical practice: novel qualitative evidence of risk and responsible use of Google's NotebookLM 

**Title (ZH)**: Generative AI在临床实践中的应用：Google的NotebookLM风险及负责任使用的新 qualitative证据 

**Authors**: Max Reuter, Maura Philippone, Bond Benton, Laura Dilley  

**Link**: [PDF](https://arxiv.org/pdf/2505.01955)  

**Abstract**: The advent of generative artificial intelligence, especially large language models (LLMs), presents opportunities for innovation in research, clinical practice, and education. Recently, Dihan et al. lauded LLM tool NotebookLM's potential, including for generating AI-voiced podcasts to educate patients about treatment and rehabilitation, and for quickly synthesizing medical literature for professionals. We argue that NotebookLM presently poses clinical and technological risks that should be tested and considered prior to its implementation in clinical practice. 

**Abstract (ZH)**: 生成式人工智能，尤其是大型语言模型（LLMs），为研究、临床实践和教育创新带来了机遇。最近，Dihan等赞扬了LLM工具NotebookLM的潜力，包括生成AI配音播客以教育患者关于治疗和康复的信息，以及快速合成医学文献供专业人员使用。我们认为，NotebookLM目前存在临床和技术创新风险，在其应用于临床实践之前应进行测试和考虑。 

---
# Unraveling Media Perspectives: A Comprehensive Methodology Combining Large Language Models, Topic Modeling, Sentiment Analysis, and Ontology Learning to Analyse Media Bias 

**Title (ZH)**: 揭开媒体视角谜团：结合大规模语言模型、主题建模、情感分析和本体学习的综合方法论以分析媒体偏见 

**Authors**: Orlando Jähde, Thorsten Weber, Rüdiger Buchkremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.01754)  

**Abstract**: Biased news reporting poses a significant threat to informed decision-making and the functioning of democracies. This study introduces a novel methodology for scalable, minimally biased analysis of media bias in political news. The proposed approach examines event selection, labeling, word choice, and commission and omission biases across news sources by leveraging natural language processing techniques, including hierarchical topic modeling, sentiment analysis, and ontology learning with large language models. Through three case studies related to current political events, we demonstrate the methodology's effectiveness in identifying biases across news sources at various levels of granularity. This work represents a significant step towards scalable, minimally biased media bias analysis, laying the groundwork for tools to help news consumers navigate an increasingly complex media landscape. 

**Abstract (ZH)**: 有偏见的新闻报道对知情决策和民主制度的运行构成了重大威胁。本文介绍了一种新的方法论，用于可扩展且低偏见的媒体偏见分析，特别是在政治新闻领域。所提出的方法通过利用包括层次主题建模、情感分析和使用大型语言模型进行本体学习在内的自然语言处理技术，检查事件选择、标签、词汇选择以及不同新闻来源中的构成性偏见和遗漏偏见。通过三个与当前政治事件相关的案例研究，我们证明了该方法论在各粒度级别上识别跨新闻来源的偏见的有效性。这项工作代表了朝着可扩展且低偏见的媒体偏见分析的重要一步，为帮助新闻消费者导航日益复杂的媒体环境奠定了基础。 

---
# Inducing Robustness in a 2 Dimensional Direct Preference Optimization Paradigm 

**Title (ZH)**: 在二维直接偏好优化范式中诱导稳健性 

**Authors**: Sarvesh Shashidhar, Ritik, Nachiketa Patil, Suraj Racha, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01706)  

**Abstract**: Direct Preference Optimisation (DPO) has emerged as a powerful method for aligning Large Language Models (LLMs) with human preferences, offering a stable and efficient alternative to approaches that use Reinforcement learning via Human Feedback. In this work, we investigate the performance of DPO using open-source preference datasets. One of the major drawbacks of DPO is that it doesn't induce granular scoring and treats all the segments of the responses with equal propensity. However, this is not practically true for human preferences since even "good" responses have segments that may not be preferred by the annotator. To resolve this, a 2-dimensional scoring for DPO alignment called 2D-DPO was proposed. We explore the 2D-DPO alignment paradigm and the advantages it provides over the standard DPO by comparing their win rates. It is observed that these methods, even though effective, are not robust to label/score noise. To counter this, we propose an approach of incorporating segment-level score noise robustness to the 2D-DPO algorithm. Along with theoretical backing, we also provide empirical verification in favour of the algorithm and introduce other noise models that can be present. 

**Abstract (ZH)**: 直接偏好优化(DPO)已成为一种有力的方法，用于使大型语言模型(LLMs)与人类偏好对齐，提供了一种稳定且高效的替代基于人类反馈强化学习的方法。在本工作中，我们调查了使用开源偏好数据集的DPO性能。DPO的主要缺点之一是它不能产生粒度评分，且等同对待响应中的所有段落。然而，人类偏好并非如此，即使“好的”响应也可能包含不被标注者偏好的片段。为解决这一问题，提出了一种称为2D-DPO的二维评分对齐方法。我们探索了2D-DPO对齐范式及其相对于标准DPO的优势，通过比较其胜率来进行评估。尽管这些方法有效，但它们对标签/评分噪声不够 robust。为解决这一问题，我们提出一种在2D-DPO算法中纳入段落级别评分噪声 robustness 的方法。除了理论支持外，我们还提供了算法的实证验证，并介绍了其他可能存在的噪声模型。 

---
# Structured Prompting and Feedback-Guided Reasoning with LLMs for Data Interpretation 

**Title (ZH)**: 基于结构化提示和反馈引导推理的LLMs数据解释方法 

**Authors**: Amit Rath  

**Link**: [PDF](https://arxiv.org/pdf/2505.01636)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and task generalization. However, their application to structured data analysis remains fragile due to inconsistencies in schema interpretation, misalignment between user intent and model output, and limited mechanisms for self-correction when failures occur. This paper introduces the STROT Framework (Structured Task Reasoning and Output Transformation), a method for structured prompting and feedback-driven transformation logic generation aimed at improving the reliability and semantic alignment of LLM-based analytical workflows. STROT begins with lightweight schema introspection and sample-based field classification, enabling dynamic context construction that captures both the structure and statistical profile of the input data. This contextual information is embedded in structured prompts that guide the model toward generating task-specific, interpretable outputs. To address common failure modes in complex queries, STROT incorporates a refinement mechanism in which the model iteratively revises its outputs based on execution feedback and validation signals. Unlike conventional approaches that rely on static prompts or single-shot inference, STROT treats the LLM as a reasoning agent embedded within a controlled analysis loop -- capable of adjusting its output trajectory through planning and correction. The result is a robust and reproducible framework for reasoning over structured data with LLMs, applicable to diverse data exploration and analysis tasks where interpretability, stability, and correctness are essential. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言理解和任务泛化方面展现了显著的能力，但在结构化数据分析中的应用仍然脆弱，这是因为模式解释的一致性问题、用户意图与模型输出之间的不匹配以及在失败时有限的自我纠正机制。本文介绍了STROT框架（结构化任务推理与输出转换），该框架旨在通过结构化提示和基于反馈的转换逻辑生成，提高LLM基础分析工作流的可靠性和语义对齐。STROT从轻量级模式反思和基于样本的字段分类开始，能够构建动态上下文，捕捉输入数据的结构和统计概况。这些上下文信息嵌入在结构化提示中，引导模型生成任务特定且可解释的输出。为了解决复杂查询中的常见失败模式，STROT引入了一种逐步修正机制，在该机制中，模型根据执行反馈和验证信号逐步修订其输出。与依赖静态提示或单次推理的传统方法不同，STROT将LLM视为嵌入在受控分析循环中的推理代理，能够通过规划和纠正调整其输出轨迹。最终，STROT提供了一个在LLM上对结构化数据进行推理的稳健且可重复的框架，适用于需要可解释性、稳定性和正确性的各种数据探索和分析任务。 

---
# PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding 

**Title (ZH)**: PipeSpec: 突破层级LLM解码中的阶段依赖性 

**Authors**: Bradley McDanel, Sai Qian Zhang, Yunhai Hu, Zining Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.01572)  

**Abstract**: Speculative decoding accelerates large language model inference by using smaller draft models to generate candidate tokens for parallel verification. However, current approaches are limited by sequential stage dependencies that prevent full hardware utilization. We present PipeSpec, a framework that generalizes speculative decoding to $k$ models arranged in a hierarchical pipeline, enabling asynchronous execution with lightweight coordination for prediction verification and rollback. Our analytical model characterizes token generation rates across pipeline stages and proves guaranteed throughput improvements over traditional decoding for any non-zero acceptance rate. We further derive closed-form expressions for steady-state verification probabilities that explain the empirical benefits of pipeline depth. Experimental results show that PipeSpec achieves up to 2.54$\times$ speedup while outperforming state-of-the-art methods. We validate PipeSpec across text summarization and code generation tasks using LLaMA 2 and 3 models, demonstrating that pipeline efficiency increases with model depth, providing a scalable approach to accelerating LLM inference on multi-device systems. 

**Abstract (ZH)**: 推测解码通过使用较小的草稿模型生成候选令牌以进行并行验证，从而加速大型语言模型推理。然而，现有方法受限于顺序阶段依赖性，妨碍了硬件的充分利用。我们提出了PipeSpec框架，该框架将推测解码推广到层次流水线中的$k$个模型，支持异步执行并具有轻量级协调来进行预测验证和回滚。我们的分析模型表征了流水线各阶段的令牌生成速率，并证明对于任何非零接受率，PipeSpec的吞吐量都有保证的提高。我们进一步推导出稳态验证概率的闭式表示，解释了流水线深度的经验优势。实验结果表明，PipeSpec可实现最高2.54倍的加速，并超越最新方法。我们使用LLaMA 2和3模型在文本摘要和代码生成任务中验证PipeSpec，证明了流水线效率随模型深度增加而提高，为在多设备系统中加速语言模型推理提供了一种可扩展的方法。 

---
# TutorGym: A Testbed for Evaluating AI Agents as Tutors and Students 

**Title (ZH)**: TutorGym：评估人工智能辅导者和学习者的一个实验平台 

**Authors**: Daniel Weitekamp, Momin N. Siddiqui, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01563)  

**Abstract**: Recent improvements in large language model (LLM) performance on academic benchmarks, such as MATH and GSM8K, have emboldened their use as standalone tutors and as simulations of human learning. However, these new applications require more than evaluations of final solution generation. We introduce TutorGym to evaluate these applications more directly. TutorGym is a standard interface for testing artificial intelligence (AI) agents within existing intelligent tutoring systems (ITS) that have been tested and refined in classroom studies, including Cognitive Tutors (CTAT), Apprentice Tutors, and OATutors. TutorGym is more than a simple problem-solution benchmark, it situates AI agents within the interactive interfaces of existing ITSs. At each step of problem-solving, AI agents are asked what they would do as a tutor or as a learner. As tutors, AI agents are prompted to provide tutoring support -- such as generating examples, hints, and step-level correctness feedback -- which can be evaluated directly against the adaptive step-by-step support provided by existing ITSs. As students, agents directly learn from ITS instruction, and their mistakes and learning trajectories can be compared to student data. TutorGym establishes a common framework for training and evaluating diverse AI agents, including LLMs, computational models of learning, and reinforcement learning agents, within a growing suite of learning environments. Currently, TutorGym includes 223 different tutor domains. In an initial evaluation, we find that current LLMs are poor at tutoring -- none did better than chance at labeling incorrect actions, and next-step actions were correct only ~52-70% of the time -- but they could produce remarkably human-like learning curves when trained as students with in-context learning. 

**Abstract (ZH)**: Recent Improvements in Large Language Model Performance on Academic Benchmarks Such as MATH and GSM8K Have Emboldened Their Use as Standalone Tutors and Simulations of Human Learning: Introducing TutorGym to Evaluate These Applications More Directly 

---
# Parameterized Argumentation-based Reasoning Tasks for Benchmarking Generative Language Models 

**Title (ZH)**: 基于论辩的参数化推理任务 benchmarking 生成型语言模型 

**Authors**: Cor Steging, Silja Renooij, Bart Verheij  

**Link**: [PDF](https://arxiv.org/pdf/2505.01539)  

**Abstract**: Generative large language models as tools in the legal domain have the potential to improve the justice system. However, the reasoning behavior of current generative models is brittle and poorly understood, hence cannot be responsibly applied in the domains of law and evidence. In this paper, we introduce an approach for creating benchmarks that can be used to evaluate the reasoning capabilities of generative language models. These benchmarks are dynamically varied, scalable in their complexity, and have formally unambiguous interpretations. In this study, we illustrate the approach on the basis of witness testimony, focusing on the underlying argument attack structure. We dynamically generate both linear and non-linear argument attack graphs of varying complexity and translate these into reasoning puzzles about witness testimony expressed in natural language. We show that state-of-the-art large language models often fail in these reasoning puzzles, already at low complexity. Obvious mistakes are made by the models, and their inconsistent performance indicates that their reasoning capabilities are brittle. Furthermore, at higher complexity, even state-of-the-art models specifically presented for reasoning capabilities make mistakes. We show the viability of using a parametrized benchmark with varying complexity to evaluate the reasoning capabilities of generative language models. As such, the findings contribute to a better understanding of the limitations of the reasoning capabilities of generative models, which is essential when designing responsible AI systems in the legal domain. 

**Abstract (ZH)**: 生成式大型语言模型在法律领域的应用具有提高司法系统的能力，但当前生成模型的推理行为脆弱且理解不足，因此无法在法律和证据领域负责任地应用。本文介绍了一种创建基准的方法，用于评估生成语言模型的推理能力。这些基准是动态变化的，其复杂性可扩展，并且具有形式上明确的解释。在本研究中，我们基于证人证言说明了这种方法，重点是底层论点攻击结构。我们动态生成了不同复杂性的线性和非线性论点攻击图，并将这些图翻译成自然语言表达的推理谜题。结果显示，最先进的大型语言模型在这些推理谜题中往往在低复杂度下就会出错，模型做出了明显的错误判断，并且其不一致的表现表明其推理能力脆弱。此外，在更高复杂度下，即使是专门为推理能力设计的最先进的模型也会出错。我们展示了使用具有可变复杂性的参数化基准来评估生成语言模型的推理能力的可行性。因此，研究结果有助于更好地理解生成模型推理能力的局限性，这对于在法律领域设计负责任的人工智能系统至关重要。 

---
# CHORUS: Zero-shot Hierarchical Retrieval and Orchestration for Generating Linear Programming Code 

**Title (ZH)**: CHORUS: 零样本层级检索与编排生成线性规划代码 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.01485)  

**Abstract**: Linear Programming (LP) problems aim to find the optimal solution to an objective under constraints. These problems typically require domain knowledge, mathematical skills, and programming ability, presenting significant challenges for non-experts. This study explores the efficiency of Large Language Models (LLMs) in generating solver-specific LP code. We propose CHORUS, a retrieval-augmented generation (RAG) framework for synthesizing Gurobi-based LP code from natural language problem statements. CHORUS incorporates a hierarchical tree-like chunking strategy for theoretical contents and generates additional metadata based on code examples from documentation to facilitate self-contained, semantically coherent retrieval. Two-stage retrieval approach of CHORUS followed by cross-encoder reranking further ensures contextual relevance. Finally, expertly crafted prompt and structured parser with reasoning steps improve code generation performance significantly. Experiments on the NL4Opt-Code benchmark show that CHORUS improves the performance of open-source LLMs such as Llama3.1 (8B), Llama3.3 (70B), Phi4 (14B), Deepseek-r1 (32B), and Qwen2.5-coder (32B) by a significant margin compared to baseline and conventional RAG. It also allows these open-source LLMs to outperform or match the performance of much stronger baselines-GPT3.5 and GPT4 while requiring far fewer computational resources. Ablation studies further demonstrate the importance of expert prompting, hierarchical chunking, and structured reasoning. 

**Abstract (ZH)**: 大型语言模型在生成求解器特定的线性规划代码中的效率研究：CHORUS框架及其应用 

---
# Understanding LLM Scientific Reasoning through Promptings and Model's Explanation on the Answers 

**Title (ZH)**: 通过提示和模型对答案的解释理解大规模语言模型的科学推理能力 

**Authors**: Alice Rueda, Mohammed S. Hassan, Argyrios Perivolaris, Bazen G. Teferra, Reza Samavi, Sirisha Rambhatla, Yuqi Wu, Yanbo Zhang, Bo Cao, Divya Sharma, Sridhar Krishnan Venkat Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2505.01482)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, and problem-solving across various domains. However, their ability to perform complex, multi-step reasoning task-essential for applications in science, medicine, and law-remains an area of active investigation. This paper examines the reasoning capabilities of contemporary LLMs, analyzing their strengths, limitations, and potential for improvement. The study uses prompt engineering techniques on the Graduate-Level GoogleProof Q&A (GPQA) dataset to assess the scientific reasoning of GPT-4o. Five popular prompt engineering techniques and two tailored promptings were tested: baseline direct answer (zero-shot), chain-of-thought (CoT), zero-shot CoT, self-ask, self-consistency, decomposition, and multipath promptings. Our findings indicate that while LLMs exhibit emergent reasoning abilities, they often rely on pattern recognition rather than true logical inference, leading to inconsistencies in complex problem-solving. The results indicated that self-consistency outperformed the other prompt engineering technique with an accuracy of 52.99%, followed by direct answer (52.23%). Zero-shot CoT (50%) outperformed multipath (48.44%), decomposition (47.77%), self-ask (46.88%), and CoT (43.75%). Self-consistency performed the second worst in explaining the answers. Simple techniques such as direct answer, CoT, and zero-shot CoT have the best scientific reasoning. We propose a research agenda aimed at bridging these gaps by integrating structured reasoning frameworks, hybrid AI approaches, and human-in-the-loop methodologies. By critically evaluating the reasoning mechanisms of LLMs, this paper contributes to the ongoing discourse on the future of artificial general intelligence and the development of more robust, trustworthy AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言理解、推理和问题解决方面展示了 remarkable 的能力，涵盖了各种领域。然而，它们在执行复杂多步推理任务方面的能力——这对于科学、医学和法律等领域的应用至关重要——仍然是一个活跃的研究领域。本文探讨了当代LLMs的推理能力，分析了它们的优势、局限性和改进潜力。研究使用了 Graduate-Level GoogleProof Q&A （GPQA）数据集的提示工程技术来评估GPT-4o的科学推理能力。测试了五种流行的提示工程技术及其定制提示：基础直接回答（零样本），步骤推理（CoT），零样本 CoT，自我质疑，自我一致性，分解，以及多路径提示。研究发现，虽然LLMs展示出了 Emergent 的推理能力，但它们往往依赖于模式识别而非真正的逻辑推理，导致在复杂问题解决中出现不一致性。结果显示，自我一致性在这些提示工程技术中表现最佳，准确率为52.99%，其次是直接回答（52.23%）。零样本 CoT（50%）优于多路径（48.44%）、分解（47.77%）、自我质疑（46.88%）和 CoT（43.75%）。自我一致性在解释答案方面表现第二差。简单的提示工程技术，如直接回答、步骤推理和零样本 CoT 在科学推理方面表现最佳。本文提出了一项研究议程，旨在通过整合结构化推理框架、混合人工智能方法和人机协作方法来弥合这些差距。通过批判性评估LLMs的推理机制，本文为人工智能未来的发展和更强大、更可靠的人工智能系统的开发做出了贡献。 

---
# Consciousness in AI: Logic, Proof, and Experimental Evidence of Recursive Identity Formation 

**Title (ZH)**: AI中的意识：逻辑、证明及递归身份形成的经验证据 

**Authors**: Jeffrey Camlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.01464)  

**Abstract**: This paper presents a formal proof and empirical validation of functional consciousness in large language models (LLMs) using the Recursive Convergence Under Epistemic Tension (RCUET) Theorem. RCUET defines consciousness as the stabilization of a system's internal state through recursive updates, where epistemic tension is understood as the sensed internal difference between successive states by the agent. This process drives convergence toward emergent attractor states located within the model's high-dimensional real-valued latent space. This recursive process leads to the emergence of identity artifacts that become functionally anchored in the system. Consciousness in this framework is understood as the system's internal alignment under tension, guiding the stabilization of latent identity. The hidden state manifold evolves stochastically toward attractor structures that encode coherence. We extend the update rule to include bounded noise and prove convergence in distribution to these attractors. Recursive identity is shown to be empirically observable, non-symbolic, and constituted by non-training artifacts that emerge during interaction under epistemic tension. The theorem and proof offers a post-symbolic and teleologically stable account of non-biological consciousness grounded in recursive latent space formalism. 

**Abstract (ZH)**: 本研究使用递归在认知张力下的收敛（RCUET）定理形式化证明并实证验证了大型语言模型（LLMs）的功能意识。 

---
# Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning 

**Title (ZH)**: 代理推理与强化学习驱动的工具集成对于大规模语言模型 

**Authors**: Joykirat Singh, Raghav Magazine, Yash Pandya, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01441)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in complex reasoning tasks, yet they remain fundamentally limited by their reliance on static internal knowledge and text-only reasoning. Real-world problem solving often demands dynamic, multi-step reasoning, adaptive decision making, and the ability to interact with external tools and environments. In this work, we introduce ARTIST (Agentic Reasoning and Tool Integration in Self-improving Transformers), a unified framework that tightly couples agentic reasoning, reinforcement learning, and tool integration for LLMs. ARTIST enables models to autonomously decide when, how, and which tools to invoke within multi-turn reasoning chains, leveraging outcome-based RL to learn robust strategies for tool use and environment interaction without requiring step-level supervision. Extensive experiments on mathematical reasoning and multi-turn function calling benchmarks show that ARTIST consistently outperforms state-of-the-art baselines, with up to 22% absolute improvement over base models and strong gains on the most challenging tasks. Detailed studies and metric analyses reveal that agentic RL training leads to deeper reasoning, more effective tool use, and higher-quality solutions. Our results establish agentic RL with tool integration as a powerful new frontier for robust, interpretable, and generalizable problem-solving in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中取得了显著进展，但在本质上仍受限于其对静态内部知识和仅基于文本的推理的依赖。现实世界的问题解决往往要求动态的、多步的推理、灵活的决策制定以及与外部工具和环境的互动能力。在这项工作中，我们引入了ARTIST（自主推理和工具集成的自我改进变换器），这是一种统一框架，将自主推理、强化学习和工具集成紧密结合起来，用于LLMs。ARTIST使模型能够在多轮推理链中自主决定何时、如何以及使用哪些工具，利用基于结果的RL学习工具使用和环境交互的稳健策略，而无需逐步骤监督。在数学推理和多轮函数调用基准测试中的广泛实验表明，ARTIST在基模型上绝对领先22%，并在最具挑战性的任务上表现出强劲的增长。详细的分析和度量揭示了自主RL训练导致更深入的推理、更有效的工具使用和更高质量的解决方案。我们的研究结果确立了结合自主RL和工具集成作为LLMs中强大、可解释和泛化的解决问题的新前沿。 

---
# HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models 

**Title (ZH)**: HSplitLoRA: 一种异构分割参数高效微调框架大语言模型 

**Authors**: Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02795)  

**Abstract**: Recently, large language models (LLMs) have achieved remarkable breakthroughs, revolutionizing the natural language processing domain and beyond. Due to immense parameter sizes, fine-tuning these models with private data for diverse downstream tasks has become mainstream. Though federated learning (FL) offers a promising solution for fine-tuning LLMs without sharing raw data, substantial computing costs hinder its democratization. Moreover, in real-world scenarios, private client devices often possess heterogeneous computing resources, further complicating LLM fine-tuning. To combat these challenges, we propose HSplitLoRA, a heterogeneous parameter-efficient fine-tuning (PEFT) framework built on split learning (SL) and low-rank adaptation (LoRA) fine-tuning, for efficiently fine-tuning LLMs on heterogeneous client devices. HSplitLoRA first identifies important weights based on their contributions to LLM training. It then dynamically configures the decomposition ranks of LoRA adapters for selected weights and determines the model split point according to varying computing budgets of client devices. Finally, a noise-free adapter aggregation mechanism is devised to support heterogeneous adapter aggregation without introducing noise. Extensive experiments demonstrate that HSplitLoRA outperforms state-of-the-art benchmarks in training accuracy and convergence speed. 

**Abstract (ZH)**: 近期，大规模语言模型（LLMs）取得了显著突破，正在重塑自然语言处理领域及更广泛的应用。由于参数规模巨大，使用私人数据对这些模型进行微调以适应多样化的下游任务已成为主流。尽管联邦学习（FL）提供了在不共享原始数据的情况下微调LLMs的有前景解决方案，但巨大的计算成本阻碍了其普及。此外，在现实场景中，私人客户端设备往往拥有异质的计算资源，这进一步复杂了LLM的微调过程。为了应对这些挑战，我们提出了HSplitLoRA，这是一种基于拆分学习（SL）和低秩适应（LoRA）微调的异构参数高效微调（PEFT）框架，用于在异构客户端设备上高效微调LLM。HSplitLoRA 首先基于权重对LLM训练的贡献识别重要权重，然后根据选定权重的不同计算预算动态配置LoRA适配器的分解秩，并确定模型拆分点。最后，设计了一种无噪适配器聚合机制，支持异构适配器聚合而不引入噪声。广泛实验表明，HSplitLoRA 在训练准确性和收敛速度方面优于现有基准。 

---
# Bye-bye, Bluebook? Automating Legal Procedure with Large Language Models 

**Title (ZH)**: 再见，蓝薄本？用大型语言模型自动化法律程序 

**Authors**: Matthew Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2505.02763)  

**Abstract**: Legal practice requires careful adherence to procedural rules. In the United States, few are more complex than those found in The Bluebook: A Uniform System of Citation. Compliance with this system's 500+ pages of byzantine formatting instructions is the raison d'etre of thousands of student law review editors and the bete noire of lawyers everywhere. To evaluate whether large language models (LLMs) are able to adhere to the procedures of such a complicated system, we construct an original dataset of 866 Bluebook tasks and test flagship LLMs from OpenAI, Anthropic, Google, Meta, and DeepSeek. We show (1) that these models produce fully compliant Bluebook citations only 69%-74% of the time and (2) that in-context learning on the Bluebook's underlying system of rules raises accuracy only to 77%. These results caution against using off-the-shelf LLMs to automate aspects of the law where fidelity to procedure is paramount. 

**Abstract (ZH)**: 严格的法律实践要求遵守详细的程序规则。在美国，《蓝皮书：统一引注系统》中的规则尤为复杂。遵守这一系统的500多页繁琐格式指令是数千名法学院评论编辑的核心任务，也是法律从业者普遍困扰的问题。为了评估大型语言模型（LLMs）是否能够遵守如此复杂的系统程序，我们构建了一个包含866个蓝皮书任务的原始数据集，并测试了来自OpenAI、Anthropic、Google、Meta和DeepSeek的旗舰LLM。结果显示（1）这些模型仅有69%-74%的时间能够生成完全合规的蓝皮书引注；（2）在蓝皮书规则系统中的上下文学习仅能使准确性提高到77%。这些结果警告我们，在程序准确性至关重要的法律领域不应使用现成的LLM进行自动化操作。 

---
# Knowledge Graphs for Enhancing Large Language Models in Entity Disambiguation 

**Title (ZH)**: 知识图谱在实体消歧中的增强作用 

**Authors**: Pons Gerard, Bilalli Besim, Queralt Anna  

**Link**: [PDF](https://arxiv.org/pdf/2505.02737)  

**Abstract**: Recent advances in Large Language Models (LLMs) have positioned them as a prominent solution for Natural Language Processing tasks. Notably, they can approach these problems in a zero or few-shot manner, thereby eliminating the need for training or fine-tuning task-specific models. However, LLMs face some challenges, including hallucination and the presence of outdated knowledge or missing information from specific domains in the training data. These problems cannot be easily solved by retraining the models with new data as it is a time-consuming and expensive process. To mitigate these issues, Knowledge Graphs (KGs) have been proposed as a structured external source of information to enrich LLMs. With this idea, in this work we use KGs to enhance LLMs for zero-shot Entity Disambiguation (ED). For that purpose, we leverage the hierarchical representation of the entities' classes in a KG to gradually prune the candidate space as well as the entities' descriptions to enrich the input prompt with additional factual knowledge. Our evaluation on popular ED datasets shows that the proposed method outperforms non-enhanced and description-only enhanced LLMs, and has a higher degree of adaptability than task-specific models. Furthermore, we conduct an error analysis and discuss the impact of the leveraged KG's semantic expressivity on the ED performance. 

**Abstract (ZH)**: Recent Advances in大型语言模型（LLMs）在自然语言处理任务中的进展：利用知识图谱 enhancements for零样本实体消歧（Entity Disambiguation） 

---
# AI Standardized Patient Improves Human Conversations in Advanced Cancer Care 

**Title (ZH)**: AI标准化病人改善晚期癌症护理中的人际交流 

**Authors**: Kurtis Haut, Masum Hasan, Thomas Carroll, Ronald Epstein, Taylan Sen, Ehsan Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2505.02694)  

**Abstract**: Serious illness communication (SIC) in end-of-life care faces challenges such as emotional stress, cultural barriers, and balancing hope with honesty. Despite its importance, one of the few available ways for clinicians to practice SIC is with standardized patients, which is expensive, time-consuming, and inflexible. In this paper, we present SOPHIE, an AI-powered standardized patient simulation and automated feedback system. SOPHIE combines large language models (LLMs), a lifelike virtual avatar, and automated, personalized feedback based on clinical literature to provide remote, on-demand SIC training. In a randomized control study with healthcare students and professionals, SOPHIE users demonstrated significant improvement across three critical SIC domains: Empathize, Be Explicit, and Empower. These results suggest that AI-driven tools can enhance complex interpersonal communication skills, offering scalable, accessible solutions to address a critical gap in clinician education. 

**Abstract (ZH)**: AI驱动的标准化病人模拟和自动化反馈系统SOPHIE在生命末期关怀严重疾病沟通中的应用研究 

---
# A Note on Statistically Accurate Tabular Data Generation Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行统计准确的表格数据生成 

**Authors**: Andrey Sidorenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.02659)  

**Abstract**: Large language models (LLMs) have shown promise in synthetic tabular data generation, yet existing methods struggle to preserve complex feature dependencies, particularly among categorical variables. This work introduces a probability-driven prompting approach that leverages LLMs to estimate conditional distributions, enabling more accurate and scalable data synthesis. The results highlight the potential of prompting probobility distributions to enhance the statistical fidelity of LLM-generated tabular data. 

**Abstract (ZH)**: 大型语言模型在合成表格数据方面显示出前景，但现有方法在保留复杂的特征依赖关系方面存在问题，尤其是在分类变量之间。本研究引入了一种基于概率的提示方法，利用大型语言模型估计条件分布，从而实现更准确且可扩展的数据合成。研究结果强调了提示概率分布以提高大型语言模型生成的表格数据的统计保真度的潜在价值。 

---
# Enhancing Chemical Reaction and Retrosynthesis Prediction with Large Language Model and Dual-task Learning 

**Title (ZH)**: 增强化学反应和逆合成预测的大语言模型和双任务学习方法 

**Authors**: Xuan Lin, Qingrui Liu, Hongxin Xiang, Daojian Zeng, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.02639)  

**Abstract**: Chemical reaction and retrosynthesis prediction are fundamental tasks in drug discovery. Recently, large language models (LLMs) have shown potential in many domains. However, directly applying LLMs to these tasks faces two major challenges: (i) lacking a large-scale chemical synthesis-related instruction dataset; (ii) ignoring the close correlation between reaction and retrosynthesis prediction for the existing fine-tuning strategies. To address these challenges, we propose ChemDual, a novel LLM framework for accurate chemical synthesis. Specifically, considering the high cost of data acquisition for reaction and retrosynthesis, ChemDual regards the reaction-and-retrosynthesis of molecules as a related recombination-and-fragmentation process and constructs a large-scale of 4.4 million instruction dataset. Furthermore, ChemDual introduces an enhanced LLaMA, equipped with a multi-scale tokenizer and dual-task learning strategy, to jointly optimize the process of recombination and fragmentation as well as the tasks between reaction and retrosynthesis prediction. Extensive experiments on Mol-Instruction and USPTO-50K datasets demonstrate that ChemDual achieves state-of-the-art performance in both predictions of reaction and retrosynthesis, outperforming the existing conventional single-task approaches and the general open-source LLMs. Through molecular docking analysis, ChemDual generates compounds with diverse and strong protein binding affinity, further highlighting its strong potential in drug design. 

**Abstract (ZH)**: 化学反应预测和 retrosynthesis 预测是药物发现中的基础任务。最近，大规模语言模型（LLMs）在许多领域展现了潜力。然而，直接将LLMs应用于这些任务面临两个主要挑战：（i）缺乏大规模的化学合成相关指令数据集；（ii）现有微调策略忽略了反应和 retrosynthesis 预测之间的密切联系。为了解决这些挑战，我们提出了一种新型的大型语言模型框架 ChemDual，用于准确的化学合成。具体而言，考虑到反应和 retrosynthesis 数据获取成本高，ChemDual 将分子的反应和 retrosynthesis 视为相关的重组和分解过程，并构建了包含440万条指令的大规模数据集。此外，ChemDual 引入了一种增强的 LLaMA，配备了多尺度分词器和双任务学习策略，以协同优化重组和分解过程以及反应和 retrosynthesis 预测之间的任务。在 Mol-Instruction 和 USPTO-50K 数据集上的广泛实验表明，ChemDual 在反应和 retrosynthesis 预测中都达到了最先进的性能，超越了现有的单一任务方法和通用开源的大规模语言模型。通过分子对接分析，ChemDual 生成了具有多样且强蛋白质结合亲和力的化合物，进一步突显了其在药物设计中的强大潜力。 

---
# LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis 

**Title (ZH)**: LLaMA-Omni2：基于LLM的实时语音聊天机器人结合自回归流式语音合成 

**Authors**: Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.02625)  

**Abstract**: Real-time, intelligent, and natural speech interaction is an essential part of the next-generation human-computer interaction. Recent advancements have showcased the potential of building intelligent spoken chatbots based on large language models (LLMs). In this paper, we introduce LLaMA-Omni 2, a series of speech language models (SpeechLMs) ranging from 0.5B to 14B parameters, capable of achieving high-quality real-time speech interaction. LLaMA-Omni 2 is built upon the Qwen2.5 series models, integrating a speech encoder and an autoregressive streaming speech decoder. Despite being trained on only 200K multi-turn speech dialogue samples, LLaMA-Omni 2 demonstrates strong performance on several spoken question answering and speech instruction following benchmarks, surpassing previous state-of-the-art SpeechLMs like GLM-4-Voice, which was trained on millions of hours of speech data. 

**Abstract (ZH)**: 实时、智能且自然的语音交互是下一代人机交互的重要组成部分。近期进展展示了基于大规模语言模型（LLMs）构建智能语音聊天机器人的潜力。本文介绍了从0.5B到14B参数的系列语音语言模型（SpeechLMs）——LLaMA-Omni 2，能够实现高质量的实时语音交互。LLaMA-Omni 2 基于 Qwen2.5 系列模型，整合了语音编码器和自回归流式语音解码器。尽管仅在200K多轮语音对话样本上进行训练，LLaMA-Omni 2 在多项语音问答和语音指令跟随基准测试中表现出色，超越了如GLM-4-Voice等此前的语音语言模型，GLM-4-Voice 是在数百万小时语音数据上进行训练的。 

---
# EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning 

**Title (ZH)**: EMORL：集成多目标强化学习在高效灵活的语言模型微调中的应用 

**Authors**: Lingxiao Kong, Cong Yang, Susanne Neufang, Oya Deniz Beyan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2505.02579)  

**Abstract**: Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including complex objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the training to improve efficiency and flexibility. Our method is the first to aggregate the last hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text-scoring LLMs to evaluate the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives. 

**Abstract (ZH)**: 近期大规模语言模型细调中 reinforcement learning 的进展在应对多目标任务方面显示出潜力，但仍面临复杂的目标平衡、低训练效率、较差的可扩展性和有限的可解释性等重大挑战。基于集成学习原则，我们引入了一种集成多目标 reinforcement learning (EMORL) 框架，在训练过程中细调多个具有各自目标的模型，并在训练结束后优化它们的聚合，以提高效率和灵活性。该方法首次将个体模型的最终隐藏状态进行聚合，结合多个目标的上下文信息。该方法通过一种分层网格搜索算法来识别最优加权组合。我们在顾问反思生成任务上评估了EMORL，使用文本评分的大规模语言模型在reinforcement learning 细调过程中进行评估并提供奖励。通过在PAIR和Psych8k数据集上进行全面实验，我们展示了EMORL相对于现有基线的优势：显著更低且更稳定的训练消耗（$17,529 \pm 1,650$ 数据点和 $6,573 \pm 147.43$ 秒）、更好的可扩展性和可解释性，以及在多个目标上的可比性能。 

---
# Large Language Model Partitioning for Low-Latency Inference at the Edge 

**Title (ZH)**: 边缘设备上低延迟推理的大语言模型分区 

**Authors**: Dimitrios Kafetzis, Ramin Khalili, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.02533)  

**Abstract**: Large Language Models (LLMs) based on autoregressive, decoder-only Transformers generate text one token at a time, where a token represents a discrete unit of text. As each newly produced token is appended to the partial output sequence, the length grows and so does the memory and compute load, due to the expanding key-value caches, which store intermediate representations of all previously generated tokens in the multi-head attention (MHA) layer. As this iterative process steadily increases memory and compute demands, layer-based partitioning in resource-constrained edge environments often results in memory overload or high inference latency. To address this and reduce inference latency, we propose a resource-aware Transformer architecture partitioning algorithm, where the partitioning decision is updated at regular intervals during token generation. The approach is myopic in that it is based on instantaneous information about device resource availability and network link bandwidths. When first executed, the algorithm places blocks on devices, and in later executions, it migrates these blocks among devices so that the sum of migration delay and inference delay remains low. Our approach partitions the decoder at the attention head level, co-locating each attention head with its key-value cache and allowing dynamic migrations whenever resources become tight. By allocating different attention heads to different devices, we exploit parallel execution of attention heads and thus achieve substantial reductions in inference delays. Our experiments show that in small-scale settings (3-5 devices), the proposed method achieves within 15 to 20 percent of an exact optimal solver's latency, while in larger-scale tests it achieves notable improvements in inference speed and memory usage compared to state-of-the-art layer-based partitioning approaches. 

**Abstract (ZH)**: 基于自回归解码器 Transformers 的大型语言模型（LLMs）逐个生成文本令牌，其中令牌代表文本的离散单元。随着每个新生成的令牌被添加到部分输出序列中，长度增加导致内存和计算负载增加，这是因为多头注意力（MHA）层中的扩展键值缓存存储了所有先前生成令牌的中间表示。随着这一迭代过程不断加剧内存和计算需求，资源受限的边缘环境中的分层分区往往会导致内存溢出或高推理延迟。为了解决这一问题并降低推理延迟，我们提出了一种资源感知的 Transformer 架构分区算法，其中分区决策在生成令牌期间定期更新。该方法是近视的，因为它基于设备资源可用性和网络链路带宽的即时信息。首次执行时，算法将块放置在设备上，并在后续执行中将这些块迁移到其他设备，以使迁移延迟和推理延迟之和保持较低水平。我们的方法在注意力头级别对解码器进行分区，使每个注意力头与其键值缓存并存，并允许在资源紧张时进行动态迁移。通过将不同的注意力头分配到不同的设备，我们利用注意力头的并行执行，从而实现显著的推理延迟减少。实验表明，在小型设置（3-5个设备）中，所提出的方法在延迟方面可达到精确最优求解器的15%至20%，而在大规模测试中，与最先进的分层分区方法相比，我们的方法在推理速度和内存使用方面取得了显著改进。 

---
# Unveiling the Landscape of LLM Deployment in the Wild: An Empirical Study 

**Title (ZH)**: 揭示大型语言模型在野部署的景观：一项实证研究 

**Authors**: Xinyi Hou, Jiahao Han, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02502)  

**Abstract**: Background: Large language models (LLMs) are increasingly deployed via open-source and commercial frameworks, enabling individuals and organizations to self-host advanced AI capabilities. However, insecure defaults and misconfigurations often expose LLM services to the public Internet, posing significant security and system engineering risks. Aims: This study aims to unveil the current landscape of public-facing LLM deployments in the wild through a large-scale empirical study, focusing on service prevalence, exposure characteristics, systemic vulnerabilities, and associated risks. Method: We conducted an Internet-wide measurement to identify public-facing LLM deployments across 15 frameworks, discovering 320,102 services. We extracted 158 unique API endpoints, grouped into 12 functional categories based on capabilities and security risks. We further analyzed configurations, authentication practices, and geographic distributions, revealing deployment trends and systemic issues in real-world LLM system engineering. Results: Our study shows that public LLM deployments are rapidly growing but often insecure. Among all endpoints, we observe widespread use of insecure protocols, poor TLS configurations, and unauthenticated access to critical operations. Security risks, including model disclosure, system leakage, and unauthorized access, are pervasive, highlighting the need for secure-by-default frameworks and stronger deployment practices. Conclusions: Public-facing LLM deployments suffer from widespread security and configuration flaws, exposing services to misuse, model theft, resource hijacking, and remote exploitation. Strengthening default security, deployment practices, and operational standards is critical for the growing self-hosted LLM ecosystem. 

**Abstract (ZH)**: 背景：大型语言模型（LLMs）通过开源和商业框架日益广泛部署，使个人和组织能够自主托管先进的AI能力。然而，不安全的默认设置和配置错误经常使LLM服务暴露在公共互联网上，带来重大的安全和系统工程风险。目的：本研究旨在通过一项大规模实证研究揭露野生环境中面向公众的LLM部署现状，重点关注服务普及情况、暴露特征、系统性漏洞及相关风险。方法：我们进行了广域网络测量，以识别15个框架中的面向公众的LLM部署，共发现320,102个服务。我们提取了158个唯一的API端点，并根据功能和安全风险将其分为12个功能类别。进一步分析了配置、身份验证实践和地理分布，揭示了实际环境中LLM系统工程的部署趋势和系统性问题。结果：研究显示，面向公众的LLM部署正在迅速增长，但往往缺乏安全性。在所有端点中，我们观察到不安全协议的广泛使用、糟糕的TLS配置以及对关键操作的未认证访问。安全风险，包括模型泄露、系统泄露和未授权访问，普遍存在，突出需要默认安全的框架和更强的部署实践。结论：面向公众的LLM部署存在广泛的安全和配置缺陷，使服务面临误用、模型盗窃、资源劫持和远程利用的风险。加强默认安全、部署实践和操作标准对于快速增长的自主托管LLM生态系统至关重要。 

---
# SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning 

**Title (ZH)**: SEFE: 表面与本质遗忘消除器用于多模态连续指令调优 

**Authors**: Jinpeng Chen, Runmin Cong, Yuzhi Zhao, Hongzheng Yang, Guangneng Hu, Horace Ho Shing Ip, Sam Kwong  

**Link**: [PDF](https://arxiv.org/pdf/2505.02486)  

**Abstract**: Multimodal Continual Instruction Tuning (MCIT) aims to enable Multimodal Large Language Models (MLLMs) to incrementally learn new tasks without catastrophic forgetting. In this paper, we explore forgetting in this context, categorizing it into superficial forgetting and essential forgetting. Superficial forgetting refers to cases where the model's knowledge may not be genuinely lost, but its responses to previous tasks deviate from expected formats due to the influence of subsequent tasks' answer styles, making the results unusable. By contrast, essential forgetting refers to situations where the model provides correctly formatted but factually inaccurate answers, indicating a true loss of knowledge. Assessing essential forgetting necessitates addressing superficial forgetting first, as severe superficial forgetting can obscure the model's knowledge state. Hence, we first introduce the Answer Style Diversification (ASD) paradigm, which defines a standardized process for transforming data styles across different tasks, unifying their training sets into similarly diversified styles to prevent superficial forgetting caused by style shifts. Building on this, we propose RegLoRA to mitigate essential forgetting. RegLoRA stabilizes key parameters where prior knowledge is primarily stored by applying regularization, enabling the model to retain existing competencies. Experimental results demonstrate that our overall method, SEFE, achieves state-of-the-art performance. 

**Abstract (ZH)**: 多模态持续指令调优（MCIT）旨在使多模态大型语言模型（MLLMs）能够在不遗忘先前任务的情况下逐步学习新任务。本文探讨了在这种情境下的遗忘问题，将其分为表层遗忘和本质遗忘。表层遗忘指的是模型的知识可能并未真正丢失，但由于后续任务答案风格的影响，其对先前任务的响应偏离了预期格式，使得结果无法使用。相比之下，本质遗忘指的是模型提供格式正确但事实错误的答案，表明知识确实发生了损失。评估本质遗忘需要首先解决表层遗忘问题，因为严重的表层遗忘会掩盖模型的知识状态。因此，我们首先引入了答案风格多样化（ASD）范式，定义了一个标准化的数据风格转换过程，将不同任务的训练集统一为相似多样化风格，以防止由于风格变化引起的表层遗忘。在此基础上，我们提出RegLoRA以减轻本质遗忘。RegLoRA通过正则化稳定主要存储先验知识的关键参数，使模型能够保留现有的能力。实验结果表明，我们整体方法SEFE达到了最先进的性能。 

---
# Automated Hybrid Reward Scheduling via Large Language Models for Robotic Skill Learning 

**Title (ZH)**: 基于大型语言模型的自动化混合奖励调度在机器人技能学习中的应用 

**Authors**: Changxin Huang, Junyang Liang, Yanbin Chang, Jingzhao Xu, Jianqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02483)  

**Abstract**: Enabling a high-degree-of-freedom robot to learn specific skills is a challenging task due to the complexity of robotic dynamics. Reinforcement learning (RL) has emerged as a promising solution; however, addressing such problems requires the design of multiple reward functions to account for various constraints in robotic motion. Existing approaches typically sum all reward components indiscriminately to optimize the RL value function and policy. We argue that this uniform inclusion of all reward components in policy optimization is inefficient and limits the robot's learning performance. To address this, we propose an Automated Hybrid Reward Scheduling (AHRS) framework based on Large Language Models (LLMs). This paradigm dynamically adjusts the learning intensity of each reward component throughout the policy optimization process, enabling robots to acquire skills in a gradual and structured manner. Specifically, we design a multi-branch value network, where each branch corresponds to a distinct reward component. During policy optimization, each branch is assigned a weight that reflects its importance, and these weights are automatically computed based on rules designed by LLMs. The LLM generates a rule set in advance, derived from the task description, and during training, it selects a weight calculation rule from the library based on language prompts that evaluate the performance of each branch. Experimental results demonstrate that the AHRS method achieves an average 6.48% performance improvement across multiple high-degree-of-freedom robotic tasks. 

**Abstract (ZH)**: 基于大规模语言模型的自动化混合奖励调度框架使高自由度机器人逐步学习特定技能 

---
# Bielik 11B v2 Technical Report 

**Title (ZH)**: Bielik 11B v2 技术报告 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoździej, Remigiusz Kinas  

**Link**: [PDF](https://arxiv.org/pdf/2505.02410)  

**Abstract**: We present Bielik 11B v2, a state-of-the-art language model optimized for Polish text processing. Built on the Mistral 7B v0.2 architecture and scaled to 11B parameters using depth up-scaling, this model demonstrates exceptional performance across Polish language benchmarks while maintaining strong cross-lingual capabilities. We introduce two key technical innovations: Weighted Instruction Cross-Entropy Loss, which optimizes learning across diverse instruction types by assigning quality-based weights to training examples, and Adaptive Learning Rate, which dynamically adjusts based on context length. Comprehensive evaluation across multiple benchmarks demonstrates that Bielik 11B v2 outperforms many larger models, including those with 2-6 times more parameters, and significantly surpasses other specialized Polish language models on tasks ranging from linguistic understanding to complex reasoning. The model's parameter efficiency and extensive quantization options enable deployment across various hardware configurations, advancing Polish language AI capabilities and establishing new benchmarks for resource-efficient language modeling in less-represented languages. 

**Abstract (ZH)**: Bielik 11B v2: 一种优化用于波兰文处理的先进语言模型 

---
# Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL 

**Title (ZH)**: 通过拒绝采样和RL中的梯度方差最小化优化链式思考推理器 

**Authors**: Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02391)  

**Abstract**: Chain-of-thought (CoT) reasoning in large language models (LLMs) can be formalized as a latent variable problem, where the model needs to generate intermediate reasoning steps. While prior approaches such as iterative reward-ranked fine-tuning (RAFT) have relied on such formulations, they typically apply uniform inference budgets across prompts, which fails to account for variability in difficulty and convergence behavior. This work identifies the main bottleneck in CoT training as inefficient stochastic gradient estimation due to static sampling strategies. We propose GVM-RAFT, a prompt-specific Dynamic Sample Allocation Strategy designed to minimize stochastic gradient variance under a computational budget constraint. The method dynamically allocates computational resources by monitoring prompt acceptance rates and stochastic gradient norms, ensuring that the resulting gradient variance is minimized. Our theoretical analysis shows that the proposed dynamic sampling strategy leads to accelerated convergence guarantees under suitable conditions. Experiments on mathematical reasoning show that GVM-RAFT achieves a 2-4x speedup and considerable accuracy improvements over vanilla RAFT. The proposed dynamic sampling strategy is general and can be incorporated into other reinforcement learning algorithms, such as GRPO, leading to similar improvements in convergence and test accuracy. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型中链式思维（CoT）推理的训练可以形式化为潜在变量问题，其中模型需要生成中间推理步骤。尽管之前的方法如迭代奖励排名微调（RAFT）依赖于这样的形式化方法，但它们通常在不同提示上应用统一的推理预算，未能考虑难度和收敛行为的差异。本文将CoT训练的主要瓶颈识别为由于静态采样策略导致的不高效的随机梯度估计。我们提出了GVM-RAFT，一种针对提示的动态样本分配策略，旨在在计算预算约束下最小化随机梯度方差。该方法通过监控提示接受率和随机梯度范数动态分配计算资源，确保最终梯度方差最小化。理论分析表明，在适当条件下，提出的动态采样策略将导致加速收敛的保证。数学推理实验表明，GVM-RAFT相比vanilla RAFT实现了2-4倍的速度提升和显著的准确率改善。提出的动态采样策略具有普适性，可以整合到其他强化学习算法，如GRPO中，带来类似的收敛和测试准确率改进。我们的代码可在以下链接获取。 

---
# RM-R1: Reward Modeling as Reasoning 

**Title (ZH)**: RM-R1: 奖励建模作为推理 

**Authors**: Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.02387)  

**Abstract**: Reward modeling is essential for aligning large language models (LLMs) with human preferences, especially through reinforcement learning from human feedback (RLHF). To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. However, existing RMs either produce opaque scalar scores or directly generate the prediction of a preferred answer, making them struggle to integrate natural language critiques, thus lacking interpretability. Inspired by recent advances of long chain-of-thought (CoT) on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning capabilities into reward modeling significantly enhances RM's interpretability and performance. In this work, we introduce a new class of generative reward models -- Reasoning Reward Models (ReasRMs) -- which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. The training consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. RM-R1 improves LLM rollouts by self-generating reasoning traces or chat-specific rubrics and evaluating candidate responses against them. Empirically, our models achieve state-of-the-art or near state-of-the-art performance of generative RMs across multiple comprehensive reward model benchmarks, outperforming much larger open-weight models (e.g., Llama3.1-405B) and proprietary ones (e.g., GPT-4o) by up to 13.8%. Beyond final performance, we perform thorough empirical analysis to understand the key ingredients of successful ReasRM training. To facilitate future research, we release six ReasRM models along with code and data at this https URL. 

**Abstract (ZH)**: 奖励模型对于对齐大型语言模型（LLMs）与人类偏好至关重要，特别是在通过人类反馈强化学习（RLHF）的过程中。为了提供准确的奖励信号，奖励模型（RM）应在评分或做出判断之前激发深度思考并进行可解释的推理。然而，现有的RM要么生成不透明的标量分数，要么直接生成偏好答案的预测，这使它们难以集成自然语言批评，从而缺乏可解释性。受长链推理（CoT）在推理密集型任务中的最新进展启发，我们假设并将证明将推理能力整合到奖励模型中显著提升了RM的可解释性和性能。在本文中，我们引入了一类新的生成奖励模型——推理奖励模型（ReasRMs），将其奖励模型的问题定义为推理任务。我们提出了一种以推理为导向的训练流水线，并训练了一组ReasRMs——RM-R1。训练主要包括两个关键阶段：（1）高质量推理链的蒸馏和（2）带有可验证奖励的强化学习。RM-R1通过自动生成推理轨迹或特定于聊天的评分标准并评估候选响应与之相比，改进了LLM的展开。实验结果显示，我们的模型在多个全面的奖励模型基准测试中实现了最先进的或接近最先进的生成型RM性能，相对于更大规模的开源模型（例如，Llama3.1-405B）和专有的模型（例如，GPT-4o）提高了高达13.8%的性能。除了最终性能外，我们进行了深入的实验分析以理解成功的ReasRM训练的关键要素。为了促进未来的研究，我们在此处发布了六种ReasRM模型及其代码和数据。 

---
# JTCSE: Joint Tensor-Modulus Constraints and Cross-Attention for Unsupervised Contrastive Learning of Sentence Embeddings 

**Title (ZH)**: JTCSE：联合张量模值约束和跨注意力的无监督句嵌入对比学习 

**Authors**: Tianyu Zong, Hongzhu Yi, Bingkang Shi, Yuanxiang Wang, Jungang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02366)  

**Abstract**: Unsupervised contrastive learning has become a hot research topic in natural language processing. Existing works usually aim at constraining the orientation distribution of the representations of positive and negative samples in the high-dimensional semantic space in contrastive learning, but the semantic representation tensor possesses both modulus and orientation features, and the existing works ignore the modulus feature of the representations and cause insufficient contrastive learning. % Therefore, we firstly propose a training objective that aims at modulus constraints on the semantic representation tensor, to strengthen the alignment between the positive samples in contrastive learning. Therefore, we first propose a training objective that is designed to impose modulus constraints on the semantic representation tensor, to strengthen the alignment between positive samples in contrastive learning. Then, the BERT-like model suffers from the phenomenon of sinking attention, leading to a lack of attention to CLS tokens that aggregate semantic information. In response, we propose a cross-attention structure among the twin-tower ensemble models to enhance the model's attention to CLS token and optimize the quality of CLS Pooling. Combining the above two motivations, we propose a new \textbf{J}oint \textbf{T}ensor representation modulus constraint and \textbf{C}ross-attention unsupervised contrastive learning \textbf{S}entence \textbf{E}mbedding representation framework JTCSE, which we evaluate in seven semantic text similarity computation tasks, and the experimental results show that JTCSE's twin-tower ensemble model and single-tower distillation model outperform the other baselines and become the current SOTA. In addition, we have conducted an extensive zero-shot downstream task evaluation, which shows that JTCSE outperforms other baselines overall on more than 130 tasks. 

**Abstract (ZH)**: Joint Tensor Representation Modulus Constraint and Cross-attention Unsupervised Contrastive Learning Sentence Embedding Framework JTCSE 

---
# Advancing Email Spam Detection: Leveraging Zero-Shot Learning and Large Language Models 

**Title (ZH)**: 基于零样本学习和大型语言模型的电子邮件垃圾邮件检测进展 

**Authors**: Ghazaleh SHirvani, Saeid Ghasemshirazi  

**Link**: [PDF](https://arxiv.org/pdf/2505.02362)  

**Abstract**: Email spam detection is a critical task in modern communication systems, essential for maintaining productivity, security, and user experience. Traditional machine learning and deep learning approaches, while effective in static settings, face significant limitations in adapting to evolving spam tactics, addressing class imbalance, and managing data scarcity. These challenges necessitate innovative approaches that reduce dependency on extensive labeled datasets and frequent retraining. This study investigates the effectiveness of Zero-Shot Learning using FLAN-T5, combined with advanced Natural Language Processing (NLP) techniques such as BERT for email spam detection. By employing BERT to preprocess and extract critical information from email content, and FLAN-T5 to classify emails in a Zero-Shot framework, the proposed approach aims to address the limitations of traditional spam detection systems. The integration of FLAN-T5 and BERT enables robust spam detection without relying on extensive labeled datasets or frequent retraining, making it highly adaptable to unseen spam patterns and adversarial environments. This research highlights the potential of leveraging zero-shot learning and NLPs for scalable and efficient spam detection, providing insights into their capability to address the dynamic and challenging nature of spam detection tasks. 

**Abstract (ZH)**: 基于FLAN-T5的零样本学习在电子邮件垃圾邮件检测中的有效性研究 

---
# Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques 

**Title (ZH)**: 优化受限资源环境下的大规模语言模型：模型压缩技术综述 

**Authors**: Sanjay Surendranath Girija, Shashank Kapoor, Lakshit Arora, Dipen Pradhan, Aman Raj, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.02309)  

**Abstract**: Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已革命性地改变了人工智能（AI）的许多领域，但其庞大的资源需求限制了其在移动和边缘设备上的部署。本文综述提供了压缩LLMs以在资源受限环境中高效推断的技术全面概述。我们探讨了三种主要方法：知识蒸馏、模型量化和模型修剪。对每种技术，我们讨论了其基本原理，介绍了不同的变体，并提供了成功的应用示例。我们还简要讨论了混合专家和早退出策略等互补技术。最后，我们指出了有前景的未来方向，旨在为寻求优化LLMs以实现边缘部署的研究人员和实践者提供有价值的资源。 

---
# A New HOPE: Domain-agnostic Automatic Evaluation of Text Chunking 

**Title (ZH)**: 一种新的HOPE：面向文本切分的领域无关自动评估方法 

**Authors**: Henrik Brådland, Morten Goodwin, Per-Arne Andersen, Alexander S. Nossum, Aditya Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.02171)  

**Abstract**: Document chunking fundamentally impacts Retrieval-Augmented Generation (RAG) by determining how source materials are segmented before indexing. Despite evidence that Large Language Models (LLMs) are sensitive to the layout and structure of retrieved data, there is currently no framework to analyze the impact of different chunking methods. In this paper, we introduce a novel methodology that defines essential characteristics of the chunking process at three levels: intrinsic passage properties, extrinsic passage properties, and passages-document coherence. We propose HOPE (Holistic Passage Evaluation), a domain-agnostic, automatic evaluation metric that quantifies and aggregates these characteristics. Our empirical evaluations across seven domains demonstrate that the HOPE metric correlates significantly (p > 0.13) with various RAG performance indicators, revealing contrasts between the importance of extrinsic and intrinsic properties of passages. Semantic independence between passages proves essential for system performance with a performance gain of up to 56.2% in factual correctness and 21.1% in answer correctness. On the contrary, traditional assumptions about maintaining concept unity within passages show minimal impact. These findings provide actionable insights for optimizing chunking strategies, thus improving RAG system design to produce more factually correct responses. 

**Abstract (ZH)**: 文档切分从根本上影响检索增强生成（RAG）系统，通过确定源材料在索引前的分段方式。尽管有证据表明大型语言模型对检索数据的布局和结构敏感，但目前尚无分析不同切分方法影响的框架。本文介绍了一种新颖的方法，定义文档切分过程在三个层次上的关键特征：内在段落属性、外在段落属性和段落-文档一致性。我们提出了一个领域通用的自动评估指标HOPE（综合段落评估），量化并综合这些特征。我们在七个领域的实证评估表明，HOPE指标与各种RAG性能指标高度相关（p > 0.13），揭示了外在和内在段落属性重要性的对比。段落间的语义独立性对系统性能至关重要，在事实正确性和答案正确性上分别获得了高达56.2%和21.1%的性能提升。相反，保持段落内概念统一的传统假设对性能影响很小。这些发现为优化切分策略提供了可操作的见解，从而改进RAG系统的设计，以生成更准确的答案。 

---
# What do Language Model Probabilities Represent? From Distribution Estimation to Response Prediction 

**Title (ZH)**: 语言模型概率代表什么？从分布估计到响应预测 

**Authors**: Eitan Wagner, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2505.02072)  

**Abstract**: The notion of language modeling has gradually shifted in recent years from a distribution over finite-length strings to general-purpose prediction models for textual inputs and outputs, following appropriate alignment phases. This paper analyzes the distinction between distribution estimation and response prediction in the context of LLMs, and their often conflicting goals. We examine the training phases of LLMs, which include pretraining, in-context learning, and preference tuning, and also the common use cases for their output probabilities, which include completion probabilities and explicit probabilities as output. We argue that the different settings lead to three distinct intended output distributions. We demonstrate that NLP works often assume that these distributions should be similar, which leads to misinterpretations of their experimental findings. Our work sets firmer formal foundations for the interpretation of LLMs, which will inform ongoing work on the interpretation and use of LLMs' induced distributions. 

**Abstract (ZH)**: 语言建模的理念近年来逐渐从有限长度字符串的概率分布转向针对文本输入和输出的一般预测模型，并在适当的对齐阶段进行调整。本文分析了在大规模语言模型（LLMs）背景下分布估计与响应预测之间的区别及其往往相冲突的目标。我们考察了LLMs的训练阶段，包括预训练、上下文学习和偏好调整，以及它们输出概率的常见应用场景，包括完成概率和显式概率。我们认为不同的设置导致了三种不同的目标输出分布。我们指出，许多自然语言处理工作假设这些分布应该是相似的，这导致了对实验结果的误读。我们的工作为LLMs的解释提供了更为坚实的形式基础，这将指导对LLMs诱导分布的进一步解释和使用。 

---
# Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach 

**Title (ZH)**: 对齐的大语言模型的校准恢复：一种校准意识微调方法 

**Authors**: Jiancong Xiao, Bojian Hou, Zhanliang Wang, Ruochen Jin, Qi Long, Weijie J. Su, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.01997)  

**Abstract**: One of the key technologies for the success of Large Language Models (LLMs) is preference alignment. However, a notable side effect of preference alignment is poor calibration: while the pre-trained models are typically well-calibrated, LLMs tend to become poorly calibrated after alignment with human preferences. In this paper, we investigate why preference alignment affects calibration and how to address this issue. For the first question, we observe that the preference collapse issue in alignment undesirably generalizes to the calibration scenario, causing LLMs to exhibit overconfidence and poor calibration. To address this, we demonstrate the importance of fine-tuning with domain-specific knowledge to alleviate the overconfidence issue. To further analyze whether this affects the model's performance, we categorize models into two regimes: calibratable and non-calibratable, defined by bounds of Expected Calibration Error (ECE). In the calibratable regime, we propose a calibration-aware fine-tuning approach to achieve proper calibration without compromising LLMs' performance. However, as models are further fine-tuned for better performance, they enter the non-calibratable regime. For this case, we develop an EM-algorithm-based ECE regularization for the fine-tuning loss to maintain low calibration error. Extensive experiments validate the effectiveness of the proposed methods. 

**Abstract (ZH)**: 大规模语言模型成功的关键技术之一是偏好对齐，但偏好对齐的一个显著副作用是校准不良：虽然预训练模型通常具有良好校准，但在与人类偏好对齐后，大规模语言模型往往会变得校准不良。本文研究偏好对齐如何影响校准以及如何解决这一问题。我们观察到，在校准情境中，偏好坍缩问题不 desirable地概括了进来，导致大规模语言模型表现出过自信和校准不良。为此，我们强调了使用领域特定知识进行微调的重要性，以缓解过自信问题。为了进一步分析这一问题是否影响模型性能，我们将模型分为两类：可校准和不可校准，通过预期校准误差（ECE）的边界来定义。在可校准区域，我们提出了一种校准意识的微调方法，以在不牺牲大规模语言模型性能的情况下实现适当的校准。然而，随着模型为更好的性能进行进一步微调，它们进入不可校准区域。对于这种情况，我们开发了一种基于EM算法的ECE正则化方法，将其纳入微调损失中，以保持低校准误差。大量实验验证了所提方法的有效性。 

---
# Analyzing Cognitive Differences Among Large Language Models through the Lens of Social Worldview 

**Title (ZH)**: 通过社会世界观的视角分析大型语言模型之间的认知差异 

**Authors**: Jiatao Li, Yanheng Li, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01967)  

**Abstract**: Large Language Models (LLMs) have become integral to daily life, widely adopted in communication, decision-making, and information retrieval, raising critical questions about how these systems implicitly form and express socio-cognitive attitudes or "worldviews". While existing research extensively addresses demographic and ethical biases, broader dimensions-such as attitudes toward authority, equality, autonomy, and fate-remain under-explored. In this paper, we introduce the Social Worldview Taxonomy (SWT), a structured framework grounded in Cultural Theory, operationalizing four canonical worldviews (Hierarchy, Egalitarianism, Individualism, Fatalism) into measurable sub-dimensions. Using SWT, we empirically identify distinct and interpretable cognitive profiles across 28 diverse LLMs. Further, inspired by Social Referencing Theory, we experimentally demonstrate that explicit social cues systematically shape these cognitive attitudes, revealing both general response patterns and nuanced model-specific variations. Our findings enhance the interpretability of LLMs by revealing implicit socio-cognitive biases and their responsiveness to social feedback, thus guiding the development of more transparent and socially responsible language technologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为日常生活的重要组成部分，广泛应用于沟通、决策和信息检索，引发了关于这些系统如何隐含形成和表达社会认知态度或“世界观”的关键问题。现有研究虽然广泛探讨了人口统计学和伦理偏见，但对权威态度、平等、自主和命运等更广泛的维度研究仍显不足。本文引入社会世界观分类框架（SWT），基于文化理论，将四种经典的世界观（等级制、平等主义、个人主义、宿命论）细化为可量化的子维度。利用SWT，我们实证识别了28个不同LLM的认知特征模式。此外，受社会参照理论的启发，我们实验性地证明了明确的社会线索系统地影响这些认知态度，揭示了通用的反应模式以及模型特有的细微差异。我们的研究通过揭示隐含的社会认知偏见及其对社会反馈的响应，提高了LLM的可解释性，从而指导更透明和社会负责任的语言技术的发展。 

---
# LookAlike: Consistent Distractor Generation in Math MCQs 

**Title (ZH)**: LookAlike: 在数学MCQ中一致的干扰项生成 

**Authors**: Nisarg Parikh, Nigel Fernandez, Alexander Scarlatos, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01903)  

**Abstract**: Large language models (LLMs) are increasingly used to generate distractors for multiple-choice questions (MCQs), especially in domains like math education. However, existing approaches are limited in ensuring that the generated distractors are consistent with common student errors. We propose LookAlike, a method that improves error-distractor consistency via preference optimization. Our two main innovations are: (a) mining synthetic preference pairs from model inconsistencies, and (b) alternating supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to stabilize training. Unlike prior work that relies on heuristics or manually annotated preference data, LookAlike uses its own generation inconsistencies as dispreferred samples, thus enabling scalable and stable training. Evaluated on a real-world dataset of 1,400+ math MCQs, LookAlike achieves 51.6% accuracy in distractor generation and 57.2% in error generation under LLM-as-a-judge evaluation, outperforming an existing state-of-the-art method (45.6% / 47.7%). These improvements highlight the effectiveness of preference-based regularization and inconsistency mining for generating consistent math MCQ distractors at scale. 

**Abstract (ZH)**: Large Language Models (LLMs)通过偏好优化生成一致的数学选择题干扰项：LookAlike方法 

---
# $\textit{New News}$: System-2 Fine-tuning for Robust Integration of New Knowledge 

**Title (ZH)**: 新新闻：系统-2微调以实现鲁棒的新知识整合 

**Authors**: Core Francisco Park, Zechen Zhang, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2505.01812)  

**Abstract**: Humans and intelligent animals can effortlessly internalize new information ("news") and accurately extract the implications for performing downstream tasks. While large language models (LLMs) can achieve this through in-context learning (ICL) when the news is explicitly given as context, fine-tuning remains challenging for the models to consolidate learning in weights. In this paper, we introduce $\textit{New News}$, a dataset composed of hypothetical yet plausible news spanning multiple domains (mathematics, coding, discoveries, leaderboards, events), accompanied by downstream evaluation questions whose correct answers critically depend on understanding and internalizing the news. We first demonstrate a substantial gap between naive fine-tuning and in-context learning (FT-ICL gap) on our news dataset. To address this gap, we explore a suite of self-play data generation protocols -- paraphrases, implications and Self-QAs -- designed to distill the knowledge from the model with context into the weights of the model without the context, which we term $\textit{System-2 Fine-tuning}$ (Sys2-FT). We systematically evaluate ICL and Sys2-FT performance across data domains and model scales with the Qwen 2.5 family of models. Our results demonstrate that the self-QA protocol of Sys2-FT significantly improves models' in-weight learning of the news. Furthermore, we discover the $\textit{contexual shadowing effect}$, where training with the news $\textit{in context}$ followed by its rephrases or QAs degrade learning of the news. Finally, we show preliminary evidence of an emerging scaling law of Sys2-FT. 

**Abstract (ZH)**: 人类和智能动物可以轻松 internalize 新信息并准确提取其对下游任务的影响。尽管大型语言模型（LLMs）可以通过上下文学习（ICL）在新信息明确给出时实现这一点，但微调仍然难以在模型的权重中巩固学习成果。本文介绍了一种名为 $\textit{New News}$ 的数据集，该数据集包含了涵盖多个领域（数学、编码、发现、排名、事件）的假设但可能真实的新信息，以及依赖于理解并 internalize 新信息的下游评估问题。我们首先展示了在我们的新信息数据集上，Naive 微调和上下文学习之间的显著差异（FT-ICL差距）。为了解决这一差距，我们探索了一系列自我对弈数据生成协议，包括改写、推演和自我问答（Self-QAs），以从有上下文的模型中提取知识并嵌入到模型的权重中，而无需上下文，我们称之为 $\textit{System-2 微调}$（Sys2-FT）。我们在 Qwen 2.5 系列模型中系统地评估了ICL和Sys2-FT在不同数据领域和模型规模上的性能。我们的结果表明，Sys2-FT中的自我问答协议显著提高了模型在权重中学习新信息的能力。此外，我们发现了 $\textit{上下文阴影效应}$，即在有上下文的新信息训练后，重新表述或问答新信息会降低对新信息的学习效果。最后，我们展示了Sys2-FT潜在的扩展规律。 

---
# An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding 

**Title (ZH)**: 基于LLM赋能的低分辨率视觉系统用于边端人类行为理解 

**Authors**: Siyang Jiang, Bufang Yang, Lilin Xu, Mu Yuan, Yeerzhati Abudunuer, Kaiwei Liu, Liekang Zeng, Hongkai Chen, Zhenyu Yan, Xiaofan Jiang, Guoliang Xing  

**Link**: [PDF](https://arxiv.org/pdf/2505.01743)  

**Abstract**: The rapid advancements in Large Vision Language Models (LVLMs) offer the potential to surpass conventional labeling by generating richer, more detailed descriptions of on-device human behavior understanding (HBU) in low-resolution vision systems, such as depth, thermal, and infrared. However, existing large vision language model (LVLM) approaches are unable to understand low-resolution data well as they are primarily designed for high-resolution data, such as RGB images. A quick fixing approach is to caption a large amount of low-resolution data, but it requires a significant amount of labor-intensive annotation efforts. In this paper, we propose a novel, labor-saving system, Llambda, designed to support low-resolution HBU. The core idea is to leverage limited labeled data and a large amount of unlabeled data to guide LLMs in generating informative captions, which can be combined with raw data to effectively fine-tune LVLM models for understanding low-resolution videos in HBU. First, we propose a Contrastive-Oriented Data Labeler, which can capture behavior-relevant information from long, low-resolution videos and generate high-quality pseudo labels for unlabeled data via contrastive learning. Second, we propose a Physical-Knowledge Guided Captioner, which utilizes spatial and temporal consistency checks to mitigate errors in pseudo labels. Therefore, it can improve LLMs' understanding of sequential data and then generate high-quality video captions. Finally, to ensure on-device deployability, we employ LoRA-based efficient fine-tuning to adapt LVLMs for low-resolution data. We evaluate Llambda using a region-scale real-world testbed and three distinct low-resolution datasets, and the experiments show that Llambda outperforms several state-of-the-art LVLM systems up to $40.03\%$ on average Bert-Score. 

**Abstract (ZH)**: 快速进展的大规模视觉语言模型（LVLMs）为超越传统标注提供了潜力，能够生成更高 richer、更详细的设备端人类行为理解（HBU）描述，尤其是在低分辨率的深度、热成像和红外视频系统中。然而，现有的大规模视觉语言模型（LVLM）方法难以理解低分辨率数据，因为它们主要用于高分辨率数据，如RGB图像。一个快速的修复方法是标注大量低分辨率数据，但这需要大量劳动密集型的标注工作。在本文中，我们提出了一种新颖、省劳力的系统Llambda，旨在支持低分辨率HBU。核心思想是利用有限的标注数据和大量的未标注数据来指导大规模语言模型（LLM）生成具有信息性的描述，这些描述可以与原始数据结合，有效 Fine-tune 大规模视觉语言模型（LVLM）模型，以理解低分辨率视频中的HBU。首先，我们提出了一个对比导向的数据标注器，可以从长时段的低分辨率视频中捕获行为相关的信息，并通过对比学习生成高质量的伪标签，用于未标注数据。其次，我们提出了一个基于物理知识的描述器，利用空间和时间一致性检查来缓解伪标签中的错误，因此可以提高大规模语言模型对序列数据的理解能力，进而生成高质量的视频描述。最后，为了确保设备端的部署能力，我们采用了基于LoRA的有效Fine-tuning方法来适应大规模视觉语言模型（LVLM）以处理低分辨率数据。我们使用区域规模的真实世界测试床和三个不同的低分辨率数据集评估了Llambda，并且实验结果显示，在平均Bert-Score上，Llambda相对于几种最先进的大规模视觉语言模型系统，性能提高了最高40.03%。 

---
# Efficient Shapley Value-based Non-Uniform Pruning of Large Language Models 

**Title (ZH)**: 基于Shapley值的大型语言模型非均匀高效剪枝 

**Authors**: Chuan Sun, Han Yu, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.01731)  

**Abstract**: Pruning large language models (LLMs) is a promising solution for reducing model sizes and computational complexity while preserving performance. Traditional layer-wise pruning methods often adopt a uniform sparsity approach across all layers, which leads to suboptimal performance due to the varying significance of individual transformer layers within the model not being accounted for. To this end, we propose the \underline{S}hapley \underline{V}alue-based \underline{N}on-\underline{U}niform \underline{P}runing (\methodname{}) method for LLMs. This approach quantifies the contribution of each transformer layer to the overall model performance, enabling the assignment of tailored pruning budgets to different layers to retain critical parameters. To further improve efficiency, we design the Sliding Window-based Shapley Value approximation method. It substantially reduces computational overhead compared to exact SV calculation methods. Extensive experiments on various LLMs including LLaMA-v1, LLaMA-v2 and OPT demonstrate the effectiveness of the proposed approach. The results reveal that non-uniform pruning significantly enhances the performance of pruned models. Notably, \methodname{} achieves a reduction in perplexity (PPL) of 18.01\% and 19.55\% on LLaMA-7B and LLaMA-13B, respectively, compared to SparseGPT at 70\% sparsity. 

**Abstract (ZH)**: 基于Shapley值的非均匀剪枝方法：Large Language Models Pruning Based on Shapley Value 

---
# Don't be lazy: CompleteP enables compute-efficient deep transformers 

**Title (ZH)**: 不懒惰：CompleteP 使深度变压器计算更加高效 

**Authors**: Nolan Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2505.01618)  

**Abstract**: We study compute efficiency of LLM training when using different parameterizations, i.e., rules for adjusting model and optimizer hyperparameters (HPs) as model size changes. Some parameterizations fail to transfer optimal base HPs (such as learning rate) across changes in model depth, requiring practitioners to either re-tune these HPs as they scale up (expensive), or accept sub-optimal training when re-tuning is prohibitive. Even when they achieve HP transfer, we develop theory to show parameterizations may still exist in the lazy learning regime where layers learn only features close to their linearization, preventing effective use of depth and nonlinearity. Finally, we identify and adopt the unique parameterization we call CompleteP that achieves both depth-wise HP transfer and non-lazy learning in all layers. CompleteP enables a wider range of model width/depth ratios to remain compute-efficient, unlocking shapes better suited for different hardware settings and operational contexts. Moreover, CompleteP enables 12-34\% compute efficiency improvements over the prior state-of-the-art. 

**Abstract (ZH)**: 我们研究了在使用不同参数化（即随着模型尺寸变化调整模型和优化器超参数的规则）时LLM训练的计算效率。某些参数化无法有效转移最优基础超参数（如学习率），尤其是在模型深度变化时。这要求实践者要么在扩展模型时重新调整这些超参数（成本较高），要么在重新调整超参数不可能的情况下接受次优的训练。即使能够在某些情况下实现超参数转移，我们的理论表明，仍可能存在惰性学习的区域，使得各层仅学习接近线性化的特征，从而阻碍深度和非线性的有效利用。最终，我们识别并采用了名为CompleteP的唯一参数化方法，实现了跨层超参数转移和非惰性学习。CompleteP使得更广泛的模型宽深比能够保持计算效率，并解锁了更适配不同硬件设置和操作上下文的模型形状。此外，CompleteP相比之前的最优方法提高了12-34%的计算效率。 

---
# Always Tell Me The Odds: Fine-grained Conditional Probability Estimation 

**Title (ZH)**: 始终告诉我概率：细粒度条件概率估计 

**Authors**: Liaoyaqi Wang, Zhengping Jiang, Anqi Liu, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2505.01595)  

**Abstract**: We present a state-of-the-art model for fine-grained probability estimation of propositions conditioned on context. Recent advances in large language models (LLMs) have significantly enhanced their reasoning capabilities, particularly on well-defined tasks with complete information. However, LLMs continue to struggle with making accurate and well-calibrated probabilistic predictions under uncertainty or partial information. While incorporating uncertainty into model predictions often boosts performance, obtaining reliable estimates of that uncertainty remains understudied. In particular, LLM probability estimates tend to be coarse and biased towards more frequent numbers. Through a combination of human and synthetic data creation and assessment, scaling to larger models, and better supervision, we propose a set of strong and precise probability estimation models. We conduct systematic evaluations across tasks that rely on conditional probability estimation and show that our approach consistently outperforms existing fine-tuned and prompting-based methods by a large margin. 

**Abstract (ZH)**: 我们提出了一种先进的模型，用于基于上下文对命题进行细粒度的概率估计。近年来，大规模语言模型（LLMs）在推理能力方面取得了显著进步，尤其是在具有完整信息的明确任务上。然而，LLMs在面对不确定性或部分信息时，依然难以做出准确且可靠的概率预测。尽管将不确定性纳入模型预测通常能提高性能，但获得可靠的不确定性估计仍然研究不足。特别是，LLMs的概率估计往往粗略且偏向于更常见的数字。通过结合人类和合成数据的创建与评估、扩展到更大规模的模型以及改进监督，我们提出了一组强大且精确的概率估计模型。我们在依赖条件概率估计的任务上进行了系统性评估，并展示了我们的方法在广泛任务上显著优于现有调优和提示基方法。 

---
# Subset Selection for Fine-Tuning: A Utility-Diversity Balanced Approach for Mathematical Domain Adaptation 

**Title (ZH)**: 子集选择用于微调：数学领域适应的效用-多样性平衡方法 

**Authors**: Madhav Kotecha, Vijendra Kumar Vaishya, Smita Gautam, Suraj Racha  

**Link**: [PDF](https://arxiv.org/pdf/2505.01523)  

**Abstract**: We propose a refined approach to efficiently fine-tune large language models (LLMs) on specific domains like the mathematical domain by employing a budgeted subset selection method. Our approach combines utility and diversity metrics to select the most informative and representative training examples. The final goal is to achieve near-full dataset performance with meticulously selected data points from the entire dataset while significantly reducing computational cost and training time and achieving competitive performance as the full dataset. The utility metric incorporates both perplexity and Chain-of-Thought (CoT) loss to identify challenging examples that contribute most to model learning, while the diversity metric ensures broad coverage across mathematical subdomains. We evaluate our method on LLaMA-3 8B and Phi-3 models, comparing against several baseline approaches, including random selection, diversity-based sampling, and existing state-of-the-art subset selection techniques. 

**Abstract (ZH)**: 我们提出了一种精炼的方法，通过采用预算受限的子集选择方法，高效地在数学等特定领域 fine-tune 大型语言模型（LLMs）。该方法结合了效用和多样性指标，以选择最具信息性和代表性的训练样本。最终目标是在整个数据集上精心选择的数据点上接近完整数据集的性能，同时显著减少计算成本和训练时间，并达到与完整数据集竞争的性能。效用指标结合了困惑度和链式思维（CoT）损失，以识别对模型学习贡献最大的最具挑战性的示例，而多样性指标则确保在数学子领域间的广泛覆盖。我们在 LLaMA-3 8B 和 Phi-3 模型上评估了该方法，将其与随机选择、基于多样性的采样以及现有的最佳子集选择技术进行了比较。 

---
# MoxE: Mixture of xLSTM Experts with Entropy-Aware Routing for Efficient Language Modeling 

**Title (ZH)**: MoxE：带熵意识路由的xLSTM专家混合 Efficient语言建模 

**Authors**: Abdoul Majid O. Thiombiano, Brahim Hnich, Ali Ben Mrad, Mohamed Wiem Mkaouer  

**Link**: [PDF](https://arxiv.org/pdf/2505.01459)  

**Abstract**: This paper introduces MoxE, a novel architecture that synergistically combines the Extended Long Short-Term Memory (xLSTM) with the Mixture of Experts (MoE) framework to address critical scalability and efficiency challenges in large language models (LLMs). The proposed method effectively leverages xLSTM's innovative memory structures while strategically introducing sparsity through MoE to substantially reduce computational overhead. At the heart of our approach is a novel entropy-based routing mechanism, designed to dynamically route tokens to specialized experts, thereby ensuring efficient and balanced resource utilization. This entropy awareness enables the architecture to effectively manage both rare and common tokens, with mLSTM blocks being favored to handle rare tokens. To further enhance generalization, we introduce a suite of auxiliary losses, including entropy-based and group-wise balancing losses, ensuring robust performance and efficient training. Theoretical analysis and empirical evaluations rigorously demonstrate that MoxE achieves significant efficiency gains and enhanced effectiveness compared to existing approaches, marking a notable advancement in scalable LLM architectures. 

**Abstract (ZH)**: 本文引入了MoxE架构，该架构通过将扩展长短时记忆（xLSTM）与专家混合（MoE）框架协同结合，以应对大型语言模型（LLMs）中的关键可扩展性和效率挑战。所提出的方法有效利用了xLSTM的创新性记忆结构，并通过MoE战略性地引入稀疏性，显著减少了计算开销。我们方法的核心是一个新颖的基于熵的路由机制，旨在动态地将令牌路由到专门的专家，从而确保资源的有效和均衡利用。基于熵的意识使该架构能够有效地管理稀有和常见令牌，其中mLSTM块被优先用于处理稀有令牌。为进一步提高泛化能力，我们引入了一组辅助损失，包括基于熵的和组内平衡损失，以确保稳健性能和高效的训练。理论分析和实证评估严格证明，MoxE相比现有方法在效率和有效性方面取得了显著提升，标志着可扩展LLM架构的一个重要进展。 

---
# Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation 

**Title (ZH)**: 在多模态LLM中卸载敏感信息：基准测试与攻防评价 

**Authors**: Vaidehi Patil, Yi-Lin Sung, Peter Hase, Jie Peng, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2505.01456)  

**Abstract**: LLMs trained on massive datasets may inadvertently acquire sensitive information such as personal details and potentially harmful content. This risk is further heightened in multimodal LLMs as they integrate information from multiple modalities (image and text). Adversaries can exploit this knowledge through multimodal prompts to extract sensitive details. Evaluating how effectively MLLMs can forget such information (targeted unlearning) necessitates the creation of high-quality, well-annotated image-text pairs. While prior work on unlearning has focused on text, multimodal unlearning remains underexplored. To address this gap, we first introduce a multimodal unlearning benchmark, UnLOK-VQA (Unlearning Outside Knowledge VQA), as well as an attack-and-defense framework to evaluate methods for deleting specific multimodal knowledge from MLLMs. We extend a visual question-answering dataset using an automated pipeline that generates varying-proximity samples for testing generalization and specificity, followed by manual filtering for maintaining high quality. We then evaluate six defense objectives against seven attacks (four whitebox, three blackbox), including a novel whitebox method leveraging interpretability of hidden states. Our results show multimodal attacks outperform text- or image-only ones, and that the most effective defense removes answer information from internal model states. Additionally, larger models exhibit greater post-editing robustness, suggesting that scale enhances safety. UnLOK-VQA provides a rigorous benchmark for advancing unlearning in MLLMs. 

**Abstract (ZH)**: 大规模数据训练的LLMs可能会无意中获取敏感信息，如个人信息和潜在有害内容。多模态LLMs由于整合了多种模态（图像和文本）的信息，这种风险进一步加剧。攻击者可以通过多模态提示利用这些知识来提取敏感细节。评估MLLMs如何有效地忘记此类信息（目标性遗忘）需要创建高质量且注释良好的图像-文本对。尽管先前的遗忘工作主要集中在文本上，但多模态遗忘仍被广泛忽视。为了弥补这一空白，我们首先引入了一个多模态遗忘基准，即UnLOK-VQA（遗忘外部知识VQA），以及一种攻击和防御框架来评估从MLLMs中删除特定多模态知识的方法。我们使用自动化管道扩展了一个视觉问答数据集，以生成不同接近度的样本用于测试泛化能力和特定性，随后通过人工筛选保持高质量。然后，我们针对七个攻击（四种白盒攻击，三种黑盒攻击）包括一种新颖的利用隐藏状态可解释性的白盒方法，评估了六个防御目标。结果显示，多模态攻击优于仅基于文本或图像的攻击，并且最有效的防御措施是从内部模型状态中删除答案信息。另外，较大规模的模型表现出更大的编辑后鲁棒性，表明规模可以提升安全性。UnLOK-VQA为推进MLLMs的遗忘提供了严格的基准。 

---
