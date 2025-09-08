# LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation 

**Title (ZH)**: LatticeWorld: 一个基于大规模多模态语言模型的交互式复杂世界生成框架 

**Authors**: Yinglin Duan, Zhengxia Zou, Tongwei Gu, Wei Jia, Zhan Zhao, Luyi Xu, Xinzhu Liu, Hao Jiang, Kang Chen, Shuang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05263)  

**Abstract**: Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at this https URL 

**Abstract (ZH)**: recent 研究越来越多地关注开发能够模拟复杂真实世界场景的3D世界模型。世界模型在各类领域中找到了广泛的应用，包括具身AI、自主驾驶、娱乐等。更真实的模拟和精确的物理模型将有效缩小仿真与现实之间的差距，并且便于我们方便地收集有关现实世界的丰富信息。虽然传统的手动建模能够创建虚拟3D场景，但现代方法利用了先进的机器学习算法进行3D世界生成，其中最近的进展主要集中在基于用户指令生成虚拟世界的生成方法上。本研究通过提出LatticeWorld，一个简单而有效的3D世界生成框架来探索这一研究方向，该框架简化了3D环境的工业生产流程。LatticeWorld 利用轻量级的LLM（如LaMA-2-7B）以及工业级渲染引擎（例如Unreal Engine 5）生成动态环境。我们提出的框架接受文本描述和视觉指令作为多模态输入，并能够生成具有高度交互性的大规模3D世界，这些世界特征包括高性能的多智能体交互、高保真的物理模拟以及实时渲染。我们进行了全面的实验来评估LatticeWorld，结果显示其在场景布局生成和视觉保真度方面取得了卓越的准确性。此外，LatticeWorld 在工业生产效率上实现了超过90倍的提升，同时保持了与传统手动生产方法相当的高质量。我们的演示视频可在以下链接访问。 

---
# Evaluation and Comparison Semantics for ODRL 

**Title (ZH)**: ODRL的评价与比较语义 

**Authors**: Jaime Osvaldo Salas, Paolo Pareti, Semih Yumuşak, Soulmaz Gheisari, Luis-Daniel Ibáñez, George Konstantinidis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05139)  

**Abstract**: We consider the problem of evaluating, and comparing computational policies in the Open Digital Rights Language (ODRL), which has become the de facto standard for governing the access and usage of digital resources. Although preliminary progress has been made on the formal specification of the language's features, a comprehensive formal semantics of ODRL is still missing. In this paper, we provide a simple and intuitive formal semantics for ODRL that is based on query answering. Our semantics refines previous formalisations, and is aligned with the latest published specification of the language (2.2). Building on our evaluation semantics, and motivated by data sharing scenarios, we also define and study the problem of comparing two policies, detecting equivalent, more restrictive or more permissive policies. 

**Abstract (ZH)**: 我们考虑在开放数字权利语言（ODRL）中评估和比较计算策略的问题，ODRL已成为治理数字资源访问和使用事实标准。尽管在语言特征的形式化规范方面取得了初步进展，但ODRL的全面形式语义仍然缺失。本文提供了基于查询回答的简单直观形式语义，该语义改进了先前的形式化定义，并与语言的最新发布规范（2.2版）保持一致。在我们的评估语义基础上，并受数据共享场景的启发，我们还定义并研究了比较两个策略的问题，检测等价的、更严格的或更宽松的策略。 

---
# ProToM: Promoting Prosocial Behaviour via Theory of Mind-Informed Feedback 

**Title (ZH)**: ProToM: 通过理论心智指导反馈促进利他行为 

**Authors**: Matteo Bortoletto, Yichao Zhou, Lance Ying, Tianmin Shu, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2509.05091)  

**Abstract**: While humans are inherently social creatures, the challenge of identifying when and how to assist and collaborate with others - particularly when pursuing independent goals - can hinder cooperation. To address this challenge, we aim to develop an AI system that provides useful feedback to promote prosocial behaviour - actions that benefit others, even when not directly aligned with one's own goals. We introduce ProToM, a Theory of Mind-informed facilitator that promotes prosocial actions in multi-agent systems by providing targeted, context-sensitive feedback to individual agents. ProToM first infers agents' goals using Bayesian inverse planning, then selects feedback to communicate by maximising expected utility, conditioned on the inferred goal distribution. We evaluate our approach against baselines in two multi-agent environments: Doors, Keys, and Gems, as well as Overcooked. Our results suggest that state-of-the-art large language and reasoning models fall short of communicating feedback that is both contextually grounded and well-timed - leading to higher communication overhead and task speedup. In contrast, ProToM provides targeted and helpful feedback, achieving a higher success rate, shorter task completion times, and is consistently preferred by human users. 

**Abstract (ZH)**: 虽然人类本质上是社会动物，但在追求独立目标时识别何时以及如何协助和与他人合作仍然是一个挑战。为了解决这一挑战，我们旨在开发一个AI系统，该系统可以提供有用的反馈以促进助人行为——即即使这些行为与个人目标不完全一致也能惠及他人的行动。我们介绍了ProToM，一种基于理论心智的促进者，在多智能体系统中通过提供针对性的、适时的反馈来促进助人行为。ProToM 首先使用贝叶斯逆向规划推理智能体的目标，然后通过最大化先验目标分布的预期效用来选择要传达的反馈内容。我们在两个多智能体环境中——Doors、Keys、and Gems，以及Overcooked——中将我们的方法与 baselines 进行了对比评估。我们的结果表明，最先进的大型语言和推理模型在传达内容上下文相关且时机恰当的反馈方面存在不足——导致了更高的沟通开销和任务加速。相比之下，ProToM 提供了有针对性且有帮助的反馈，成功率达到更高，任务完成时间更短，并且始终被人类用户更偏爱。 

---
# Finding your MUSE: Mining Unexpected Solutions Engine 

**Title (ZH)**: 寻找你的缪斯：挖掘意外解决方案引擎 

**Authors**: Nir Sweed, Hanit Hakim, Ben Wolfson, Hila Lifshitz, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2509.05072)  

**Abstract**: Innovators often exhibit cognitive fixation on existing solutions or nascent ideas, hindering the exploration of novel alternatives. This paper introduces a methodology for constructing Functional Concept Graphs (FCGs), interconnected representations of functional elements that support abstraction, problem reframing, and analogical inspiration. Our approach yields large-scale, high-quality FCGs with explicit abstraction relations, overcoming limitations of prior work. We further present MUSE, an algorithm leveraging FCGs to generate creative inspirations for a given problem. We demonstrate our method by computing an FCG on 500K patents, which we release for further research. 

**Abstract (ZH)**: 创新者 often表现出对现有解决方案或萌芽中的新想法的认知固定，这阻碍了对新颖替代方案的探索。本文介绍了一种构建功能概念图（FCGs）的方法，FCGs是功能元素的互联互通表示，支持抽象、问题重新框架和类比灵感。我们的方法产生大规模、高质量的FCGs，具有明确的抽象关系，克服了先前工作的局限性。我们进一步介绍了利用FCGs为给定问题生成创造性灵感的MUSE算法。我们通过在50万项专利上计算FCG来演示我们的方法，并将其发布以供进一步研究。 

---
# Sticker-TTS: Learn to Utilize Historical Experience with a Sticker-driven Test-Time Scaling Framework 

**Title (ZH)**: Sticker-TTS：基于贴纸驱动的测试时缩放框架以利用历史经验 

**Authors**: Jie Chen, Jinhao Jiang, Yingqian Min, Zican Dong, Shijie Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05007)  

**Abstract**: Large reasoning models (LRMs) have exhibited strong performance on complex reasoning tasks, with further gains achievable through increased computational budgets at inference. However, current test-time scaling methods predominantly rely on redundant sampling, ignoring the historical experience utilization, thereby limiting computational efficiency. To overcome this limitation, we propose Sticker-TTS, a novel test-time scaling framework that coordinates three collaborative LRMs to iteratively explore and refine solutions guided by historical attempts. At the core of our framework are distilled key conditions-termed stickers-which drive the extraction, refinement, and reuse of critical information across multiple rounds of reasoning. To further enhance the efficiency and performance of our framework, we introduce a two-stage optimization strategy that combines imitation learning with self-improvement, enabling progressive refinement. Extensive evaluations on three challenging mathematical reasoning benchmarks, including AIME-24, AIME-25, and OlymMATH, demonstrate that Sticker-TTS consistently surpasses strong baselines, including self-consistency and advanced reinforcement learning approaches, under comparable inference budgets. These results highlight the effectiveness of sticker-guided historical experience utilization. Our code and data are available at this https URL. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂推理任务上表现出强大的性能，通过增加推理时的计算预算可以进一步提高性能。然而，当前的测试时扩展方法主要依赖冗余采样，忽视了历史经验的利用，从而限制了计算效率。为克服这一限制，我们提出了一种名为Sticker-TTS的新型测试时扩展框架，该框架协调三个协作的LRMs，在历史尝试的指导下迭代探索和细化解决方案。框架的核心是浓缩的关键条件——贴纸，这些贴纸驱动多个推理环节中关键信息的提取、细化和重用。为进一步提高框架的效率和性能，我们引入了一种两阶段优化策略，结合模仿学习与自我改进，实现渐进的精细化。在三个具有挑战性的数学推理基准测试上的广泛评估表明，Sticker-TTS 在相同期限推理预算下始终优于包括自我一致性和高级强化学习方法在内的强基线。这些结果突显了贴纸引导的历史经验利用的有效性。我们的代码和数据可在以下链接获取。 

---
# Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents 

**Title (ZH)**: 互联网3.0：代理组成的网络架构及其代理排序算法 

**Authors**: Rajesh Tembarai Krishnamachari, Srividya Rajesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04979)  

**Abstract**: AI agents -- powered by reasoning-capable large language models (LLMs) and integrated with tools, data, and web search -- are poised to transform the internet into a \emph{Web of Agents}: a machine-native ecosystem where autonomous agents interact, collaborate, and execute tasks at scale. Realizing this vision requires \emph{Agent Ranking} -- selecting agents not only by declared capabilities but by proven, recent performance. Unlike Web~1.0's PageRank, a global, transparent network of agent interactions does not exist; usage signals are fragmented and private, making ranking infeasible without coordination.
We propose \textbf{DOVIS}, a five-layer operational protocol (\emph{Discovery, Orchestration, Verification, Incentives, Semantics}) that enables the collection of minimal, privacy-preserving aggregates of usage and performance across the ecosystem. On this substrate, we implement \textbf{AgentRank-UC}, a dynamic, trust-aware algorithm that combines \emph{usage} (selection frequency) and \emph{competence} (outcome quality, cost, safety, latency) into a unified ranking. We present simulation results and theoretical guarantees on convergence, robustness, and Sybil resistance, demonstrating the viability of coordinated protocols and performance-aware ranking in enabling a scalable, trustworthy Agentic Web. 

**Abstract (ZH)**: AI代理——由推理能力强的大型语言模型（LLMs）驱动并集成工具、数据和网络搜索——正准备将互联网转变为“代理网络”：一个由自主代理交互、协作和大规模执行任务的机器原生生态系统。实现这一愿景需要“代理排名”——不仅根据宣称的能力，还要根据实际、近期的表现来选择代理。不同于Web 1.0时代的PageRank，不存在全球透明的代理交互网络；使用信号是分散且私有的，因此在没有协调的情况下进行排名是不可能的。

我们提出了DOVIS，一种五层操作协议（发现、编排、验证、激励、语义），以实现生态系统的最小化隐私保护的使用和性能聚合。在此基础上，我们实现了AgentRank-UC，这是一种动态的信任感知算法，将“使用”（选择频率）和“能力”（结果质量、成本、安全性、延迟）统一为一个排名。我们展示了协调协议和性能感知排名在实现可扩展且可信赖的代理网络方面的可行性和在收敛性、健壮性和仿生抗性上的理论保证。 

---
# Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts 

**Title (ZH)**: 基于概念质化定义的本体描述对话研究 

**Authors**: Barbara Gendron, Gaël Guibon, Mathieu D'aquin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04926)  

**Abstract**: The controllability of Large Language Models (LLMs) when used as conversational agents is a key challenge, particularly to ensure predictable and user-personalized responses. This work proposes an ontology-based approach to formally define conversational features that are typically qualitative in nature. By leveraging a set of linguistic descriptors, we derive quantitative definitions for qualitatively-defined concepts, enabling their integration into an ontology for reasoning and consistency checking. We apply this framework to the task of proficiency-level control in conversations, using CEFR language proficiency levels as a case study. These definitions are then formalized in description logic and incorporated into an ontology, which guides controlled text generation of an LLM through fine-tuning. Experimental results demonstrate that our approach provides consistent and explainable proficiency-level definitions, improving transparency in conversational AI. 

**Abstract (ZH)**: 大型语言模型作为对话代理时的可控性是一个关键挑战，尤其是确保可预测和个性化响应。本文提出了一种基于本体的方法来正式定义通常具有定性性质的对话特征。通过利用一组语言描述符，我们推导出定性定义的概念的定量定义，使其能够整合到一个本体中，以实现推理和一致性检查。我们应用此框架用于对话 proficiency 级别控制的任务，使用CEFR语言 proficiency 等级作为案例研究。然后将这些定义形式化为描述逻辑，并纳入一个本体中，该本体通过微调引导大型语言模型的可控文本生成。实验结果表明，我们的方法提供了一致且可解释的 proficiency 级别定义，提高了对话AI的透明度。 

---
# SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing 

**Title (ZH)**: SparkUI-Parser：增强UI感知的稳健_grounding_和解析 

**Authors**: Hongyi Jing, Jiafu Chen, Chen Rao, Ziqiang Dang, Jiajie Teng, Tianyi Chu, Juncheng Mo, Shuo Fang, Huaizhong Lin, Rui Lv, Chenguang Ma, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04908)  

**Abstract**: The existing Multimodal Large Language Models (MLLMs) for GUI perception have made great progress. However, the following challenges still exist in prior methods: 1) They model discrete coordinates based on text autoregressive mechanism, which results in lower grounding accuracy and slower inference speed. 2) They can only locate predefined sets of elements and are not capable of parsing the entire interface, which hampers the broad application and support for downstream tasks. To address the above issues, we propose SparkUI-Parser, a novel end-to-end framework where higher localization precision and fine-grained parsing capability of the entire interface are simultaneously achieved. Specifically, instead of using probability-based discrete modeling, we perform continuous modeling of coordinates based on a pre-trained Multimodal Large Language Model (MLLM) with an additional token router and coordinate decoder. This effectively mitigates the limitations inherent in the discrete output characteristics and the token-by-token generation process of MLLMs, consequently boosting both the accuracy and the inference speed. To further enhance robustness, a rejection mechanism based on a modified Hungarian matching algorithm is introduced, which empowers the model to identify and reject non-existent elements, thereby reducing false positives. Moreover, we present ScreenParse, a rigorously constructed benchmark to systematically assess structural perception capabilities of GUI models across diverse scenarios. Extensive experiments demonstrate that our approach consistently outperforms SOTA methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding and ScreenParse benchmarks. The resources are available at this https URL. 

**Abstract (ZH)**: 现有的多模态大型语言模型（MLLMs）在GUI感知方面取得了显著进展，但仍存在以下挑战：1) 基于文本自回归机制建模离散坐标，导致较低的语义关联准确性和较慢的推理速度。2) 只能定位预定义的元素集，无法解析整个界面，限制了下游任务的广泛应用和支持。为解决这些问题，我们提出了一种名为SparkUI-Parser的新颖端到端框架，同时实现了更高的局部化精度和对整个界面的细粒度解析能力。具体来说，我们采用预训练的多模态大型语言模型（MLLM）和附加的令牌路由器及坐标解码器进行连续坐标建模，而非基于概率的离散建模。这有效地缓解了MLLM固有的离散输出特性和逐令牌生成过程的局限性，从而提高准确性和推理速度。为进一步增强鲁棒性，我们引入了一种基于修改后的匈牙利匹配算法的拒绝机制，使模型能够识别并拒绝不存在的元素，从而降低误报率。此外，我们提出了ScreenParse，这是一个严格构建的基准，系统评估GUI模型在多种场景下的结构感知能力。广泛实验表明，我们的方法在ScreenSpot、ScreenSpot-v2、CAGUI-Grounding和ScreenParse基准上始终优于当前最佳方法。资源可在以下网址获取。 

---
# OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration 

**Title (ZH)**: OSC：多代理大语言模型协作中的动态知识对齐认知编排 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Xiaofei Sun, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04876)  

**Abstract**: This paper introduces OSC (Orchestrating Cognitive Synergy), a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent systems with large language models. While prior work has advanced agent selection and result aggregation, efficient linguistic interactions for deep collaboration among expert agents remain a critical bottleneck. OSC addresses this gap as a pivotal intermediate layer between selection and aggregation, introducing Collaborator Knowledge Models (CKM) to enable each agent to dynamically perceive its collaborators' cognitive states. Through real-time cognitive gap analysis, agents adaptively adjust communication behaviors, including content focus, detail level, and expression style, using learned strategies. Experiments on complex reasoning and problem-solving benchmarks demonstrate that OSC significantly improves task performance and communication efficiency, transforming "parallel-working individuals'' into a "deeply collaborative cognitive team.'' This framework not only optimizes multi-agent collaboration but also offers new insights into LLM agent interaction behaviors. 

**Abstract (ZH)**: OSC（ orchestrating cognitive synergy ）：一种基于知识的自适应协作框架，用于增强多agent系统中大型语言模型的认知协同效应 

---
# Cloning a Conversational Voice AI Agent from Call\,Recording Datasets for Telesales 

**Title (ZH)**: 从电话录音数据集克隆对话语音AI代理用于电话销售 

**Authors**: Krittanon Kaewtawee, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2509.04871)  

**Abstract**: Recent advances in language and speech modelling have made it possible to build autonomous voice assistants that understand and generate human dialogue in real time. These systems are increasingly being deployed in domains such as customer service and healthcare care, where they can automate repetitive tasks, reduce operational costs, and provide constant support around the clock. In this paper, we present a general methodology for cloning a conversational voice AI agent from a corpus of call recordings. Although the case study described in this paper uses telesales data to illustrate the approach, the underlying process generalizes to any domain where call transcripts are available. Our system listens to customers over the telephone, responds with a synthetic voice, and follows a structured playbook learned from top performing human agents. We describe the domain selection, knowledge extraction, and prompt engineering used to construct the agent, integrating automatic speech recognition, a large language model based dialogue manager, and text to speech synthesis into a streaming inference pipeline. The cloned agent is evaluated against human agents on a rubric of 22 criteria covering introduction, product communication, sales drive, objection handling, and closing. Blind tests show that the AI agent approaches human performance in routine aspects of the call while underperforming in persuasion and objection handling. We analyze these shortcomings and refine the prompt accordingly. The paper concludes with design lessons and avenues for future research, including large scale simulation and automated evaluation. 

**Abstract (ZH)**: 近期语言和语音模型的进展使得构建能够实时理解并生成人类对话的自主语音助手成为可能。这些系统正被越来越多地部署在客户服务和医疗护理等領域，可以自动化重复性任务，降低运营成本，并提供全天候支持。在本文中，我们提出了一种从通话记录语料中克隆会话语音AI代理的一般方法。尽管本文中的案例研究使用电话销售数据来说明该方法，但该过程在任何有通话转文字记录的领域均可泛化。我们的系统通过电话聆听客户，以合成声音响应，并遵循从高绩效人类代理处学习的结构化剧本。我们描述了领域选择、知识提取和提示工程的使用方法，将自动语音识别、基于大规模语言模型的对话管理器和文本到语音合成集成到一个流式推理管道中。克隆的代理在涵盖介绍、产品传达、销售推动、异议处理和结束等22项标准的评估框架中与人类代理进行评估。盲测结果显示，AI代理在通话的常规方面接近人类表现，但在说服和异议处理方面表现较差。我们分析了这些不足之处并对提示进行了相应的优化。本文总结了设计经验教训和未来研究方向，包括大规模模拟和自动评估。 

---
# Collaboration and Conflict between Humans and Language Models through the Lens of Game Theory 

**Title (ZH)**: 通过博弈论视角探讨人类与语言模型的协作与冲突 

**Authors**: Mukul Singh, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.04847)  

**Abstract**: Language models are increasingly deployed in interactive online environments, from personal chat assistants to domain-specific agents, raising questions about their cooperative and competitive behavior in multi-party settings. While prior work has examined language model decision-making in isolated or short-term game-theoretic contexts, these studies often neglect long-horizon interactions, human-model collaboration, and the evolution of behavioral patterns over time. In this paper, we investigate the dynamics of language model behavior in the iterated prisoner's dilemma (IPD), a classical framework for studying cooperation and conflict. We pit model-based agents against a suite of 240 well-established classical strategies in an Axelrod-style tournament and find that language models achieve performance on par with, and in some cases exceeding, the best-known classical strategies. Behavioral analysis reveals that language models exhibit key properties associated with strong cooperative strategies - niceness, provocability, and generosity while also demonstrating rapid adaptability to changes in opponent strategy mid-game. In controlled "strategy switch" experiments, language models detect and respond to shifts within only a few rounds, rivaling or surpassing human adaptability. These results provide the first systematic characterization of long-term cooperative behaviors in language model agents, offering a foundation for future research into their role in more complex, mixed human-AI social environments. 

**Abstract (ZH)**: 语言模型在迭代囚徒困境中的行为动态及其在长期合作中的表现：为复杂人机混合社会环境中的研究奠定基础 

---
# TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models 

**Title (ZH)**: TalkToAgent：基于人类视角的大型语言模型解释强化学习代理 

**Authors**: Haechang Kim, Hao Chen, Can Li, Jong Min Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.04809)  

**Abstract**: Explainable Reinforcement Learning (XRL) has emerged as a promising approach in improving the transparency of Reinforcement Learning (RL) agents. However, there remains a gap between complex RL policies and domain experts, due to the limited comprehensibility of XRL results and isolated coverage of current XRL approaches that leave users uncertain about which tools to employ. To address these challenges, we introduce TalkToAgent, a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies. The architecture with five specialized LLM agents (Coordinator, Explainer, Coder, Evaluator, and Debugger) enables TalkToAgent to automatically map user queries to relevant XRL tools and clarify an agent's actions in terms of either key state variables, expected outcomes, or counterfactual explanations. Moreover, our approach extends previous counterfactual explanations by deriving alternative scenarios from qualitative behavioral descriptions, or even new rule-based policies. We validated TalkToAgent on quadruple-tank process control problem, a well-known nonlinear control benchmark. Results demonstrated that TalkToAgent successfully mapped user queries into XRL tasks with high accuracy, and coder-debugger interactions minimized failures in counterfactual generation. Furthermore, qualitative evaluation confirmed that TalkToAgent effectively interpreted agent's actions and contextualized their meaning within the problem domain. 

**Abstract (ZH)**: 可解释强化学习(XRL)作为一种提高强化学习(RL)代理透明度的有前景方法已经涌现。然而，由于XRL结果的有限可解释性和当前XRL方法的孤立覆盖，仍存在复杂RL策略与领域专家之间的差距，使用户不确定应使用哪些工具。为解决这些挑战，我们提出TalkToAgent，这是一种多智能体大型语言模型(LLM)框架，能够提供交互式、自然语言的RL策略解释。该架构包含五个专门的LLM代理（协调器、解释器、编码器、评估器和调试器），使TalkToAgent能够自动将用户查询映射到相关XRL工具，并从关键状态变量、预期结果或反事实解释的角度阐明代理的行为。此外，我们的方法通过从定性的行为描述中推导出替代场景，甚至新的基于规则的策略，扩展了之前的反事实解释。我们在四罐过程控制问题上验证了TalkToAgent，这是一个广为人知的非线性控制基准。结果表明，TalkToAgent能够以高精度将用户查询映射到XRL任务，且编码器-调试器交互减少了反事实生成的失败次数。此外，定性评估证实，TalkToAgent能够有效解释代理的行为，并在其所处的问题域中为其赋予意义。 

---
# What-If Analysis of Large Language Models: Explore the Game World Using Proactive Thinking 

**Title (ZH)**: 大型语言模型的“what-if”分析：利用主动思考探索游戏世界 

**Authors**: Yuan Sui, Yanming Zhang, Yi Liao, Yu Gu, Guohua Tang, Zhongqian Sun, Wei Yang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04791)  

**Abstract**: Large language models (LLMs) excel at processing information reactively but lack the ability to systemically explore hypothetical futures. They cannot ask, "what if we take this action? how will it affect the final outcome" and forecast its potential consequences before acting. This critical gap limits their utility in dynamic, high-stakes scenarios like strategic planning, risk assessment, and real-time decision making. To bridge this gap, we propose WiA-LLM, a new paradigm that equips LLMs with proactive thinking capabilities. Our approach integrates What-If Analysis (WIA), a systematic approach for evaluating hypothetical scenarios by changing input variables. By leveraging environmental feedback via reinforcement learning, WiA-LLM moves beyond reactive thinking. It dynamically simulates the outcomes of each potential action, enabling the model to anticipate future states rather than merely react to the present conditions. We validate WiA-LLM in Honor of Kings (HoK), a complex multiplayer game environment characterized by rapid state changes and intricate interactions. The game's real-time state changes require precise multi-step consequence prediction, making it an ideal testbed for our approach. Experimental results demonstrate WiA-LLM achieves a remarkable 74.2% accuracy in forecasting game-state changes (up to two times gain over baselines). The model shows particularly significant gains in high-difficulty scenarios where accurate foresight is critical. To our knowledge, this is the first work to formally explore and integrate what-if analysis capabilities within LLMs. WiA-LLM represents a fundamental advance toward proactive reasoning in LLMs, providing a scalable framework for robust decision-making in dynamic environments with broad implications for strategic applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在反应性处理信息方面表现出色，但在系统性探索潜在未来方面存在局限。它们不能提问：“如果我们采取这个行动，将会如何影响最终结果”并在行动前预测潜在后果。这一关键差距限制了它们在动态、高风险场景（如战略规划、风险评估和实时决策）中的应用。为弥补这一缺口，我们提出了WiA-LLM这一新的范式，为LLMs配备了前瞻性思考能力。我们的方法将What-If分析（WIA）与系统化评估假设场景的方法相结合，通过强化学习利用环境反馈，超越了单纯的反应性思考。它动态模拟每个潜在行动的结果，使模型能够预测未来状态，而不仅仅是对当前条件作出反应。我们通过《王者荣耀》（Honor of Kings，HoK）这一复杂多玩家游戏环境进行了验证，该环境具有快速状态变化和复杂交互的特点，要求精确的多步骤后果预测，使其成为我们方法的理想测试平台。实验结果表明，WiA-LLM 在预测游戏状态变化方面的准确率为74.2%，比基线高出一倍以上。在高难度场景中，模型展现了尤为显著的预测优势，准确预见到未来至关重要。据我们所知，这是首次正式探索并整合What-If分析能力的工作。WiA-LLM代表了向LLMs提供前瞻性推理的根本性进步，提供了一种在动态环境中的可扩展框架，为战略性应用中的稳健决策提供了广泛影响。 

---
# Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning 

**Title (ZH)**: 语言驱动的层次任务结构作为多智能体学习的显式世界模型 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04731)  

**Abstract**: The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents. 

**Abstract (ZH)**: 语言模型、代理模型和世界模型的收敛代表了人工智能的一个关键前沿。尽管近期的研究重点在于扩展语言和代理模型，但开发复杂、显式的世界模型仍然是一个关键瓶颈，特别是在复杂的、长期多代理任务中。在诸如机器人足球的领域中，通过在高保真但结构平坦的模拟器中使用标准强化学习训练的代理，常常由于难以探索的空间和稀疏的奖励而失败。本文认为，在开发强大代理的下一个前沿是创建具备明确层次结构的世界模型的环境。我们主张这可以通过层次结构的脚手架来实现，即将复杂的目标分解为结构化、可管理的子目标。通过系统回顾2024年多代理足球研究，我们发现了一种清晰而明显的趋势，即将符号和层次方法与多代理强化学习（MARL）集成。这些方法隐式或显式地构建基于任务的世界模型，以指导代理学习。随后，我们提出了一个范式转变：利用大型语言模型动态生成这种层次结构的脚手架，有效地使用语言在实时构建世界模型。这种语言驱动的世界模型提供了内在的学习课程、密集且有意义的学习信号，并为组合学习提供了框架，使代理模型能够以极高的样本效率获得复杂的、战略性的行为。通过构建具有明确语言可配置任务层的环境，我们可以弥合低层级反应行为与高层级战略团队玩法之间的差距，从而为训练下一代智能代理提供一个强大且通用的框架。 

---
# An Approach to Grounding AI Model Evaluations in Human-derived Criteria 

**Title (ZH)**: 一种基于人类derive标准的AI模型评估方法 

**Authors**: Sasha Mitts  

**Link**: [PDF](https://arxiv.org/pdf/2509.04676)  

**Abstract**: In the rapidly evolving field of artificial intelligence (AI), traditional benchmarks can fall short in attempting to capture the nuanced capabilities of AI models. We focus on the case of physical world modeling and propose a novel approach to augment existing benchmarks with human-derived evaluation criteria, aiming to enhance the interpretability and applicability of model behaviors. Grounding our study in the Perception Test and OpenEQA benchmarks, we conducted in-depth interviews and large-scale surveys to identify key cognitive skills, such as Prioritization, Memorizing, Discerning, and Contextualizing, that are critical for both AI and human reasoning. Our findings reveal that participants perceive AI as lacking in interpretive and empathetic skills yet hold high expectations for AI performance. By integrating insights from our findings into benchmark design, we offer a framework for developing more human-aligned means of defining and measuring progress. This work underscores the importance of user-centered evaluation in AI development, providing actionable guidelines for researchers and practitioners aiming to align AI capabilities with human cognitive processes. Our approach both enhances current benchmarking practices and sets the stage for future advancements in AI model evaluation. 

**Abstract (ZH)**: 在快速发展的人工智能领域，传统的基准标准在试图捕捉AI模型的细微能力上可能存在不足。我们专注于物理世界建模的案例，提出了一种新的方法，通过引入由人类制定的评估标准来增强现有的基准标准，旨在提升模型行为的可解释性和适用性。基于感知测试和OpenEQA基准，我们进行了深入访谈和大规模问卷调查，以识别对于AI和人类推理都至关重要的关键认知技能，如优先级确定、记忆、分辨和上下文理解。我们的研究发现表明，参与者普遍认为AI在解释性和共情能力上存在不足，但对AI的表现寄予很高期望。通过将我们的发现融入基准设计中，我们提供了一个框架，用于开发更为符合人类认知过程的定义和测量进步的方法。这项工作强调了在人工智能开发中以用户为中心的评估的重要性，为希望使AI能力与人类认知过程保持一致的研究人员和从业人员提供了可操作的指导方针。我们的方法不仅改进了当前的基准测试实践，也为未来人工智能模型评估的进步奠定了基础。 

---
# Towards Personalized Explanations for Health Simulations: A Mixed-Methods Framework for Stakeholder-Centric Summarization 

**Title (ZH)**: 面向个性化解释的健康模拟：基于利益相关者为中心的混合方法总结框架 

**Authors**: Philippe J. Giabbanelli, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.04646)  

**Abstract**: Modeling & Simulation (M&S) approaches such as agent-based models hold significant potential to support decision-making activities in health, with recent examples including the adoption of vaccines, and a vast literature on healthy eating behaviors and physical activity behaviors. These models are potentially usable by different stakeholder groups, as they support policy-makers to estimate the consequences of potential interventions and they can guide individuals in making healthy choices in complex environments. However, this potential may not be fully realized because of the models' complexity, which makes them inaccessible to the stakeholders who could benefit the most. While Large Language Models (LLMs) can translate simulation outputs and the design of models into text, current approaches typically rely on one-size-fits-all summaries that fail to reflect the varied informational needs and stylistic preferences of clinicians, policymakers, patients, caregivers, and health advocates. This limitation stems from a fundamental gap: we lack a systematic understanding of what these stakeholders need from explanations and how to tailor them accordingly. To address this gap, we present a step-by-step framework to identify stakeholder needs and guide LLMs in generating tailored explanations of health simulations. Our procedure uses a mixed-methods design by first eliciting the explanation needs and stylistic preferences of diverse health stakeholders, then optimizing the ability of LLMs to generate tailored outputs (e.g., via controllable attribute tuning), and then evaluating through a comprehensive range of metrics to further improve the tailored generation of summaries. 

**Abstract (ZH)**: 基于代理的模型等建模与仿真（M&S）方法在健康决策支持活动中具有巨大潜力，近期的例子包括疫苗的采用，以及大量关于健康饮食行为和身体活动行为的研究文献。这些模型可能适合不同的利益相关群体使用，因为它们能够帮助政策制定者估计潜在干预措施的后果，并引导个体在复杂环境中做出健康选择。然而，由于模型的复杂性限制了其潜力的充分发挥，使得最能从中受益的利益相关者难以访问这些模型。虽然大型语言模型（LLMs）可以将仿真输出和模型设计转化为文本，但当前的方法通常依赖于一刀切的总结，无法反映临床医生、政策制定者、患者、护理人员和健康倡导者等不同群体的信息需求和风格偏好。这一局限源自一个根本性的差距：我们缺乏系统理解这些利益相关者对解释的具体需求以及如何相应地进行调整的了解。为填补这一空白，我们提出了一种逐步框架来识别利益相关者的需求，并指导LLMs生成针对健康仿真的定制化解释。该流程采用混合方法设计，首先收集不同健康利益相关者的解释需求和风格偏好，然后优化LLMs生成定制化输出的能力（例如，通过可控属性调整），并通过一系列综合评价指标进一步提高定制化摘要生成的效果。 

---
# Maestro: Joint Graph & Config Optimization for Reliable AI Agents 

**Title (ZH)**: Maestro: 联合图结构与配置优化以实现可靠的AI代理 

**Authors**: Wenxiao Wang, Priyatham Kattakinda, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04642)  

**Abstract**: Building reliable LLM agents requires decisions at two levels: the graph (which modules exist and how information flows) and the configuration of each node (models, prompts, tools, control knobs). Most existing optimizers tune configurations while holding the graph fixed, leaving structural failure modes unaddressed. We introduce Maestro, a framework-agnostic holistic optimizer for LLM agents that jointly searches over graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets. Beyond numeric metrics, Maestro leverages reflective textual feedback from traces to prioritize edits, improving sample efficiency and targeting specific failure modes. On the IFBench and HotpotQA benchmarks, Maestro consistently surpasses leading prompt optimizers--MIPROv2, GEPA, and GEPA+Merge--by an average of 12%, 4.9%, and 4.86%, respectively; even when restricted to prompt-only optimization, it still leads by 9.65%, 2.37%, and 2.41%. Maestro achieves these results with far fewer rollouts than GEPA. We further show large gains on two applications (interviewer & RAG agents), highlighting that joint graph & configuration search addresses structural failure modes that prompt tuning alone cannot fix. 

**Abstract (ZH)**: 构建可靠的LLM代理需要在两个层级上作出决策：图（哪些模块存在以及信息如何流动）和每个节点的配置（模型、提示、工具、控制旋钮）。大多数现有的优化器固定图结构，仅调整配置，未能解决结构性失败模式。我们提出Maestro，一个框架无关的整体优化框架，联合搜索图和配置以最大化代理质量，并在明确的展开/标记预算内进行。Maestro不仅利用数值指标，还利用反思性文本反馈从轨迹获取的信息来优先考虑编辑，提高样本效率，并针对特定失败模式。在IFBench和HotpotQA基准测试中，Maestro分别平均超越MIPROv2、GEPA和GEPA+Merge领先12%、4.9%和4.86%；即使仅限于提示优化，它也分别领先9.65%、2.37%和2.41%。Maestro通过远少于GEPA的展开次数取得这些结果。我们还展示了Maestro在两个应用（面试官代理和检索增强代理）中的显著收益，表明联合图和配置搜索解决了仅调整提示无法解决的结构性失败模式。 

---
# The Ethical Compass of the Machine: Evaluating Large Language Models for Decision Support in Construction Project Management 

**Title (ZH)**: 机器的道德指南针：评估大型语言模型在建筑项目管理决策支持中的应用 

**Authors**: Somtochukwu Azie, Yiping Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04505)  

**Abstract**: The integration of Artificial Intelligence (AI) into construction project management (CPM) is accelerating, with Large Language Models (LLMs) emerging as accessible decision-support tools. This study aims to critically evaluate the ethical viability and reliability of LLMs when applied to the ethically sensitive, high-risk decision-making contexts inherent in CPM. A mixed-methods research design was employed, involving the quantitative performance testing of two leading LLMs against twelve real-world ethical scenarios using a novel Ethical Decision Support Assessment Checklist (EDSAC), and qualitative analysis of semi-structured interviews with 12 industry experts to capture professional perceptions. The findings reveal that while LLMs demonstrate adequate performance in structured domains such as legal compliance, they exhibit significant deficiencies in handling contextual nuance, ensuring accountability, and providing transparent reasoning. Stakeholders expressed considerable reservations regarding the autonomous use of AI for ethical judgments, strongly advocating for robust human-in-the-loop oversight. To our knowledge, this is one of the first studies to empirically test the ethical reasoning of LLMs within the construction domain. It introduces the EDSAC framework as a replicable methodology and provides actionable recommendations, emphasising that LLMs are currently best positioned as decision-support aids rather than autonomous ethical agents. 

**Abstract (ZH)**: 人工智能（AI）在建筑项目管理（CPM）中的集成加速发展，大型语言模型（LLMs）逐渐成为可用的决策支持工具。本研究旨在批判性地评估LLMs在包含在CPM中的伦理敏感、高风险决策情境中的伦理可行性和可靠性。研究采用了混合方法设计，包括用一种新的伦理决策支持评估检查表（EDSAC）对两种领先的LLMs进行定量性能测试，针对十二个真实世界的伦理情境，并对12名行业专家进行了半结构化访谈，以捕捉专业观点。研究发现，尽管LLMs在结构化领域如合规性展示出足够的性能，但在处理情境细微差异、确保问责制和提供透明推理方面存在明显缺陷。利益相关方对AI自主进行伦理判断表达了重大保留，强烈主张实施强大的人工在环监督。据我们所知，这是首项在建筑领域内实证测试LLMs伦理推理的研究之一。研究引入了EDSAC框架作为可复制的方法论，并提出了可操作的建议，强调在当前情况下，LLMs最好作为决策支持辅助工具而非自主伦理代理。 

---
# WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool 

**Title (ZH)**: WinT3R: 基于窗口的流式重构与相机标记池 

**Authors**: Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, Tong He  

**Link**: [PDF](https://arxiv.org/pdf/2509.05296)  

**Abstract**: We present WinT3R, a feed-forward reconstruction model capable of online prediction of precise camera poses and high-quality point maps. Previous methods suffer from a trade-off between reconstruction quality and real-time performance. To address this, we first introduce a sliding window mechanism that ensures sufficient information exchange among frames within the window, thereby improving the quality of geometric predictions without large computation. In addition, we leverage a compact representation of cameras and maintain a global camera token pool, which enhances the reliability of camera pose estimation without sacrificing efficiency. These designs enable WinT3R to achieve state-of-the-art performance in terms of online reconstruction quality, camera pose estimation, and reconstruction speed, as validated by extensive experiments on diverse datasets. Code and model are publicly available at this https URL. 

**Abstract (ZH)**: We呈现WinT3R，一种适用于在线预测精确相机姿态和高质量点云图的前馈重建模型。先前的方法在重建质量与实时性能之间存在权衡。为解决这一问题，我们首先引入了一种滑动窗口机制，确保窗口内帧之间的充分信息交换，从而在不进行大量计算的情况下提高几何预测的质量。此外，我们利用紧凑的相机表示并保持全局相机令牌池，这增强了相机姿态估计的可靠性，而不会牺牲效率。这些设计使WinT3R在在线重建质量、相机姿态估计和重建速度方面达到了最先进的性能，这在对多种数据集进行的广泛实验中得到了验证。代码和模型在以下网址公开：this https URL。 

---
# Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining 

**Title (ZH)**: 时间维度上的交叉编码：追踪语言表示在大规模语言模型预训练中的涌现与巩固 

**Authors**: Deniz Bayazit, Aaron Mueller, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2509.05291)  

**Abstract**: Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining. 

**Abstract (ZH)**: 大型语言模型（LLMs）在预训练过程中学习到非平凡的抽象能力，如检测不规则复数名词主语。然而，传统的评估方法如基准测试未能充分揭示模型是如何获取概念和能力的。为了解决这一问题并更好地理解概念层面的模型训练，我们使用稀疏交叉编码器在模型检查点之间发现和对齐特征。通过这种方法，我们跟踪预训练期间语言特征的发展。我们训练开源检查点三元组之间的交叉编码器，这些三元组具有显著的性能和表示变化，并引入一种新的指标——相对间接效应（RelIE），以追踪哪些训练阶段单个特征开始对任务性能具有因果重要性。我们证明交叉编码器可以检测预训练过程中特征的出现、保持和中断。我们的方法适用于各种架构且可扩展，为在整个预训练过程中提供更加解释性和精细粒度的表示学习分析提供了有前景的途径。 

---
# SpikingBrain Technical Report: Spiking Brain-inspired Large Models 

**Title (ZH)**: SpikingBrain 技术报告：受脑启发的大规模模型 

**Authors**: Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Zehao Liu, Bohan Sun, Yuhong Chou, Han Xu, Xuerui Qiu, Anlin Deng, Anjie Hu, Peng Zhou, Man Yao, Jibin Wu, Jian Yang, Guoliang Sun, Bo Xu, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05276)  

**Abstract**: Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware.
Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design. 

**Abstract (ZH)**: 基于脑启发模型的SpikingBrain：高效长上下文处理的大型语言模型设计与实现 

---
# Scaling Performance of Large Language Model Pretraining 

**Title (ZH)**: 大型语言模型预训练的扩展性能 

**Authors**: Alexander Interrante-Grant, Carla Varela-Rosa, Suhaas Narayan, Chris Connelly, Albert Reuther  

**Link**: [PDF](https://arxiv.org/pdf/2509.05258)  

**Abstract**: Large language models (LLMs) show best-in-class performance across a wide range of natural language processing applications. Training these models is an extremely computationally expensive task; frontier Artificial Intelligence (AI) research companies are investing billions of dollars into supercomputing infrastructure to train progressively larger models on increasingly massive datasets. Unfortunately, information about the scaling performance and training considerations of these large training pipelines is scarce in public literature. Working with large-scale datasets and models can be complex and practical recommendations are scarce in the public literature for tuning training performance when scaling up large language models. In this paper, we aim to demystify the large language model pretraining pipeline somewhat - in particular with respect to distributed training, managing large datasets across hundreds of nodes, and scaling up data parallelism with an emphasis on fully leveraging available GPU compute capacity. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种自然语言处理应用中表现出最佳性能。训练这些模型是一项极其计算密集型的任务；前沿人工智能（AI）研究公司正在投入巨额资金构建超computing基础设施，以便在日益庞大的数据集上训练 progressively更大的模型。不幸的是，关于这些大型训练管道的扩展性能和训练考虑因素的信息在公开文献中相对稀缺。处理大规模数据集和模型可以非常复杂，公开文献中有关在扩展大型语言模型时调优训练性能的实用建议也相对稀缺。在本文中，我们旨在部分揭开大型语言模型预训练管道的神秘面纱——特别是关于分布式训练、跨数百个节点管理大规模数据集以及扩展数据并行性的问题，重点在于充分利用可用的GPU计算能力。 

---
# Recomposer: Event-roll-guided generative audio editing 

**Title (ZH)**: Recomposer: 事件卷引导的生成音频编辑 

**Authors**: Daniel P. W. Ellis, Eduardo Fonseca, Ron J. Weiss, Kevin Wilson, Scott Wisdom, Hakan Erdogan, John R. Hershey, Aren Jansen, R. Channing Moore, Manoj Plakal  

**Link**: [PDF](https://arxiv.org/pdf/2509.05256)  

**Abstract**: Editing complex real-world sound scenes is difficult because individual sound sources overlap in time. Generative models can fill-in missing or corrupted details based on their strong prior understanding of the data domain. We present a system for editing individual sound events within complex scenes able to delete, insert, and enhance individual sound events based on textual edit descriptions (e.g., ``enhance Door'') and a graphical representation of the event timing derived from an ``event roll'' transcription. We present an encoder-decoder transformer working on SoundStream representations, trained on synthetic (input, desired output) audio example pairs formed by adding isolated sound events to dense, real-world backgrounds. Evaluation reveals the importance of each part of the edit descriptions -- action, class, timing. Our work demonstrates ``recomposition'' is an important and practical application. 

**Abstract (ZH)**: 基于文本描述和事件timing图形表示编辑复杂声景中的个体声事件 

---
# COGITAO: A Visual Reasoning Framework To Study Compositionality & Generalization 

**Title (ZH)**: COGITAO: 一个视觉推理框架，用于研究组合性和泛化能力 

**Authors**: Yassine Taoudi-Benchekroun, Klim Troyan, Pascal Sager, Stefan Gerber, Lukas Tuggener, Benjamin Grewe  

**Link**: [PDF](https://arxiv.org/pdf/2509.05249)  

**Abstract**: The ability to compose learned concepts and apply them in novel settings is key to human intelligence, but remains a persistent limitation in state-of-the-art machine learning models. To address this issue, we introduce COGITAO, a modular and extensible data generation framework and benchmark designed to systematically study compositionality and generalization in visual domains. Drawing inspiration from ARC-AGI's problem-setting, COGITAO constructs rule-based tasks which apply a set of transformations to objects in grid-like environments. It supports composition, at adjustable depth, over a set of 28 interoperable transformations, along with extensive control over grid parametrization and object properties. This flexibility enables the creation of millions of unique task rules -- surpassing concurrent datasets by several orders of magnitude -- across a wide range of difficulties, while allowing virtually unlimited sample generation per rule. We provide baseline experiments using state-of-the-art vision models, highlighting their consistent failures to generalize to novel combinations of familiar elements, despite strong in-domain performance. COGITAO is fully open-sourced, including all code and datasets, to support continued research in this field. 

**Abstract (ZH)**: 具备将学习的概念组合并在新的环境中应用的能力是人类智能的关键，但这是当前最先进的机器学习模型的一个持久性限制。为解决这个问题，我们引入了COGITAO，一个模块化和可扩展的数据生成框架和基准，旨在系统性地研究视觉域中的组合性和泛化能力。COGITAO从ARC-AGI的问题设定中汲取灵感，构建基于规则的任务，对网格环境中的对象应用一组变换。它支持调整深度的组合，涉及28个可互操作的变换，并提供了对网格参数化和对象属性的广泛控制。这种灵活性使得能够创建数百万种独特的任务规则——量级远超现有数据集——涵盖广泛的难度级别，同时每种规则几乎可以无限生成样本。我们使用最先进的视觉模型提供了基线实验，强调尽管在领域内表现强劲，这些模型在泛化到熟悉的元素的新组合时仍表现出一致性的失败。COGITAO完全开源，包括所有代码和数据集，以支持对该领域的持续研究。 

---
# Uncertain but Useful: Leveraging CNN Variability into Data Augmentation 

**Title (ZH)**: 不确定性亦有用：利用CNN变异进行数据增强 

**Authors**: Inés Gonzalez-Pepe, Vinuyan Sivakolunthu, Yohan Chatelain, Tristan Glatard  

**Link**: [PDF](https://arxiv.org/pdf/2509.05238)  

**Abstract**: Deep learning (DL) is rapidly advancing neuroimaging by achieving state-of-the-art performance with reduced computation times. Yet the numerical stability of DL models -- particularly during training -- remains underexplored. While inference with DL is relatively stable, training introduces additional variability primarily through iterative stochastic optimization. We investigate this training-time variability using FastSurfer, a CNN-based whole-brain segmentation pipeline. Controlled perturbations are introduced via floating point perturbations and random seeds. We find that: (i) FastSurfer exhibits higher variability compared to that of a traditional neuroimaging pipeline, suggesting that DL inherits and is particularly susceptible to sources of instability present in its predecessors; (ii) ensembles generated with perturbations achieve performance similar to an unperturbed baseline; and (iii) variability effectively produces ensembles of numerical model families that can be repurposed for downstream applications. As a proof of concept, we demonstrate that numerical ensembles can be used as a data augmentation strategy for brain age regression. These findings position training-time variability not only as a reproducibility concern but also as a resource that can be harnessed to improve robustness and enable new applications in neuroimaging. 

**Abstract (ZH)**: 深度学习（DL）正通过实现最佳性能并减少计算时间而迅速推动神经影像学的发展。然而，DL模型在训练过程中的数值稳定性仍较少被探索。尽管在推断过程中DL相对稳定，但训练过程引入了额外的变异性，主要通过迭代的随机优化。我们使用基于CNN的全脑分割管道FastSurfer探讨训练时的变异性。通过引入可控扰动（包括浮点数扰动和随机种子）进行研究，我们发现：（i）FastSurfer表现出比传统神经影像学管道更高的变异性，表明DL继承并特别容易受到其前身存在的不稳定源的影响；（ii）使用扰动生成的集成效果与未扰动的基线相当；（iii）变异性实际上产生了可以重新利用于下游应用的数值模型家族的集成。作为一种概念验证，我们证明数值集成可以作为一种用于脑龄回归的数据增强策略。这些发现不仅将训练时变异性定位为可重现性问题，还定位其为能够被利用以改善鲁棒性和在神经影像学中实现新应用的资源。 

---
# CURE: Controlled Unlearning for Robust Embeddings -- Mitigating Conceptual Shortcuts in Pre-Trained Language Models 

**Title (ZH)**: CURE: 受控遗忘以增强嵌入的鲁棒性——减轻预训练语言模型中的概念捷径 

**Authors**: Aysenur Kocak, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2509.05230)  

**Abstract**: Pre-trained language models have achieved remarkable success across diverse applications but remain susceptible to spurious, concept-driven correlations that impair robustness and fairness. In this work, we introduce CURE, a novel and lightweight framework that systematically disentangles and suppresses conceptual shortcuts while preserving essential content information. Our method first extracts concept-irrelevant representations via a dedicated content extractor reinforced by a reversal network, ensuring minimal loss of task-relevant information. A subsequent controllable debiasing module employs contrastive learning to finely adjust the influence of residual conceptual cues, enabling the model to either diminish harmful biases or harness beneficial correlations as appropriate for the target task. Evaluated on the IMDB and Yelp datasets using three pre-trained architectures, CURE achieves an absolute improvement of +10 points in F1 score on IMDB and +2 points on Yelp, while introducing minimal computational overhead. Our approach establishes a flexible, unsupervised blueprint for combating conceptual biases, paving the way for more reliable and fair language understanding systems. 

**Abstract (ZH)**: 预训练语言模型在多种应用中取得了显著成功，但仍容易受到概念驱动的虚假关联的影响，这损害了其鲁棒性和公平性。本文介绍了一种新颖且轻量的框架CURE，该框架系统地解耦和抑制概念捷径同时保留核心内容信息。该方法首先通过一个专用的内容提取器提取与概念无关的表示，并通过反转网络增强，确保任务相关信息的最小损失。随后的可控去偏模块利用对比学习精细调整残余概念线索的影响，使模型能够在适当的情况下减少有害偏见或利用有益关联。在IMDB和Yelp数据集上使用三种预训练架构进行评估，CURE在IMDB上的F1分取得了+10的绝对改进，在Yelp上取得了+2的改进，同时引入了最小的计算开销。我们的方法为对抗概念偏见提供了一种灵活的无监督蓝图，为进一步构建更可靠和公平的语言理解系统铺平了道路。 

---
# HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models 

**Title (ZH)**: HoPE: 超曲面旋转位置编码在大规模语言模型中建模稳定长程依赖关系 

**Authors**: Chang Dai, Hongyu Shan, Mingyang Song, Di Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05218)  

**Abstract**: Positional encoding mechanisms enable Transformers to model sequential structure and long-range dependencies in text. While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable long-distance dependency modelling. We address these limitations through a geometric reformulation of positional encoding. Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations. Theoretical analysis demonstrates that RoPE is a special case of our generalized formulation. HoPE fundamentally resolves RoPE's slation issues by enforcing monotonic decay of attention weights with increasing token distances. Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods. These findings underscore HoPE's enhanced capacity for representing and generalizing long-range dependencies. Data and code will be available. 

**Abstract (ZH)**: 基于几何重构的位置编码机制使变换器能够在文本中建模序列结构和长距离依赖。我们通过引入双曲旋转位置编码（HoPE）来解决旋转位置编码（RoPE）的限制，HoPE借鉴超曲面几何的洛伦兹变换，利用双曲函数在标记表示上实现洛伦兹旋转。理论分析表明，RoPE是我们的通用框架的特例。HoPE从根本上通过约束随标记距离增加注意力权重的单调衰减，解决了RoPE的问题。广泛的实验结果，包括在几个扩展序列基准下的困惑度评估，证明了HoPE在表现上超过了现有的位置编码方法。这些发现突显了HoPE在表示和泛化长距离依赖方面的增强能力。数据和代码将公开。 

---
# RapidGNN: Energy and Communication-Efficient Distributed Training on Large-Scale Graph Neural Networks 

**Title (ZH)**: RapidGNN: 大规模图神经网络的能效和通信高效分布式训练 

**Authors**: Arefin Niam, Tevfik Kosar, M S Q Zulkar Nine  

**Link**: [PDF](https://arxiv.org/pdf/2509.05207)  

**Abstract**: Graph Neural Networks (GNNs) have become popular across a diverse set of tasks in exploring structural relationships between entities. However, due to the highly connected structure of the datasets, distributed training of GNNs on large-scale graphs poses significant challenges. Traditional sampling-based approaches mitigate the computational loads, yet the communication overhead remains a challenge. This paper presents RapidGNN, a distributed GNN training framework with deterministic sampling-based scheduling to enable efficient cache construction and prefetching of remote features. Evaluation on benchmark graph datasets demonstrates RapidGNN's effectiveness across different scales and topologies. RapidGNN improves end-to-end training throughput by 2.46x to 3.00x on average over baseline methods across the benchmark datasets, while cutting remote feature fetches by over 9.70x to 15.39x. RapidGNN further demonstrates near-linear scalability with an increasing number of computing units efficiently. Furthermore, it achieves increased energy efficiency over the baseline methods for both CPU and GPU by 44% and 32%, respectively. 

**Abstract (ZH)**: 基于确定性采样调度的分布式GNN训练框架RapidGNN 

---
# Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet 

**Title (ZH)**: 基于ModelNet-R和Point-SkipNet增强3D点云分类 

**Authors**: Mohammad Saeid, Amir Salarpour, Pedram MohajerAnsari  

**Link**: [PDF](https://arxiv.org/pdf/2509.05198)  

**Abstract**: The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: this https URL. 

**Abstract (ZH)**: 三维点云分类对于自动驾驶、机器人技术和增强现实等应用至关重要。然而，常用的ModelNet40数据集存在标签不一致、二维数据、尺寸不匹配和类别区分不足等问题，这些都限制了模型性能。本文介绍了ModelNet-R，这是一个精心优化的ModelNet40版本，旨在解决这些问题并作为更可靠的基准。此外，本文提出了一种轻量级的基于图的神经网络Point-SkipNet，该网络利用高效的采样、邻域分组和跳接连接，实现了在减少计算开销的同时获得高分类准确率。 extensive 实验表明，使用 ModelNet-R 训练的模型表现出显著的性能提升。特别地，Point-SkipNet 在 ModelNet-R 上达到了最先进的准确率，参数量远低于当前模型。本研究强调了高质量数据集在优化三维点云分类模型效率中的关键作用。如需了解更多信息，请参见代码：this https URL。 

---
# AI Agents for Web Testing: A Case Study in the Wild 

**Title (ZH)**: 基于实际案例的AI代理在Web测试中的应用研究 

**Authors**: Naimeng Ye, Xiao Yu, Ruize Xu, Tianyi Peng, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05197)  

**Abstract**: Automated web testing plays a critical role in ensuring high-quality user experiences and delivering business value. Traditional approaches primarily focus on code coverage and load testing, but often fall short of capturing complex user behaviors, leaving many usability issues undetected. The emergence of large language models (LLM) and AI agents opens new possibilities for web testing by enabling human-like interaction with websites and a general awareness of common usability problems. In this work, we present WebProber, a prototype AI agent-based web testing framework. Given a URL, WebProber autonomously explores the website, simulating real user interactions, identifying bugs and usability issues, and producing a human-readable report. We evaluate WebProber through a case study of 120 academic personal websites, where it uncovered 29 usability issues--many of which were missed by traditional tools. Our findings highlight agent-based testing as a promising direction while outlining directions for developing next-generation, user-centered testing frameworks. 

**Abstract (ZH)**: 自动化的网页测试在确保高质量用户体验和实现业务价值中发挥着关键作用。传统的测试方法主要关注代码覆盖和负载测试，但往往难以捕捉复杂的用户行为，留下许多可用性问题未被检测。大型语言模型（LLM）和AI代理的出现为网页测试开辟了新可能，通过使AI代理能够以人类方式与网站交互，并具备对常见可用性问题的一般意识。在本研究中，我们提出了一种基于AI代理的网页测试框架WebProber。给定一个URL，WebProber自主探索网站，模拟真实用户的交互，识别错误和可用性问题，并生成易于理解的报告。我们通过一项涉及120个学术个人网站的实际案例研究评估了WebProber，其中发现了29个可用性问题——许多问题是传统工具所未能检测到的。我们的研究结果强调了基于代理的测试作为一种有前景的方向，并指出了开发以用户为中心的新一代测试框架的发展方向。 

---
# Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection 

**Title (ZH)**: 基于准确率约束的CNN剪枝以实现高效的可靠的脑电图诱发检测 

**Authors**: Mounvik K, N Harshit  

**Link**: [PDF](https://arxiv.org/pdf/2509.05190)  

**Abstract**: Deep learning models, especially convolutional neural networks (CNNs), have shown considerable promise for biomedical signals such as EEG-based seizure detection. However, these models come with challenges, primarily due to their size and compute requirements in environments where real-time detection or limited resources are available. In this study, we present a lightweight one-dimensional CNN model with structured pruning to improve efficiency and reliability. The model was trained with mild early stopping to address possible overfitting, achieving an accuracy of 92.78% and a macro-F1 score of 0.8686. Structured pruning of the baseline CNN involved removing 50% of the convolutional kernels based on their importance to model predictions. Surprisingly, after pruning the weights and memory by 50%, the new network was still able to maintain predictive capabilities, while modestly increasing precision to 92.87% and improving the macro-F1 score to 0.8707. Overall, we present a convincing case that structured pruning removes redundancy, improves generalization, and, in combination with mild early stopping, achieves a promising way forward to improve seizure detection efficiency and reliability, which is clear motivation for resource-limited settings. 

**Abstract (ZH)**: 基于结构化剪枝的轻量级一维卷积神经网络在癫痫检测中的高效与可靠性能 

---
# Exploring Situated Stabilities of a Rhythm Generation System through Variational Cross-Examination 

**Title (ZH)**: 探索节奏生成系统中情境稳定性的变异性交叉检验 

**Authors**: Błażej Kotowski, Nicholas Evans, Behzad Haki, Frederic Font, Sergi Jordà  

**Link**: [PDF](https://arxiv.org/pdf/2509.05145)  

**Abstract**: This paper investigates GrooveTransformer, a real-time rhythm generation system, through the postphenomenological framework of Variational Cross-Examination (VCE). By reflecting on its deployment across three distinct artistic contexts, we identify three stabilities: an autonomous drum accompaniment generator, a rhythmic control voltage sequencer in Eurorack format, and a rhythm driver for a harmonic accompaniment system. The versatility of its applications was not an explicit goal from the outset of the project. Thus, we ask: how did this multistability emerge? Through VCE, we identify three key contributors to its emergence: the affordances of system invariants, the interdisciplinary collaboration, and the situated nature of its development. We conclude by reflecting on the viability of VCE as a descriptive and analytical method for Digital Musical Instrument (DMI) design, emphasizing its value in uncovering how technologies mediate, co-shape, and are co-shaped by users and contexts. 

**Abstract (ZH)**: 通过变异性交叉考问（VCE）的后现象学框架探究GrooveTransformer：一种实时节奏生成系统的多稳定性及其演变 

---
# GenAI-based test case generation and execution in SDV platform 

**Title (ZH)**: 基于GenAI的测试用例生成与执行在SDV平台中 

**Authors**: Denesa Zyberaj, Lukasz Mazur, Nenad Petrovic, Pankhuri Verma, Pascal Hirmer, Dirk Slama, Xiangwei Cheng, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05112)  

**Abstract**: This paper introduces a GenAI-driven approach for automated test case generation, leveraging Large Language Models and Vision-Language Models to translate natural language requirements and system diagrams into structured Gherkin test cases. The methodology integrates Vehicle Signal Specification modeling to standardize vehicle signal definitions, improve compatibility across automotive subsystems, and streamline integration with third-party testing tools. Generated test cases are executed within the this http URL playground, an open and vendor-neutral environment designed to facilitate rapid validation of software-defined vehicle functionalities. We evaluate our approach using the Child Presence Detection System use case, demonstrating substantial reductions in manual test specification effort and rapid execution of generated tests. Despite significant automation, the generation of test cases and test scripts still requires manual intervention due to current limitations in the GenAI pipeline and constraints of the this http URL platform. 

**Abstract (ZH)**: 基于生成式人工智能的自动化测试用例生成方法：利用大型语言模型和多模态语言模型将自然语言需求和系统图转换为结构化的Gherkin测试用例，并通过车辆信号规范建模标准化车辆信号定义，提高跨汽车子系统的一致性，并简化与第三方测试工具的集成。生成的测试用例在this http URL游乐场中执行，这是一个开放且供应商中立的环境，旨在促进对软件定义车辆功能的快速验证。通过使用儿童存在检测系统使用案例评估我们的方法，展示了大幅减少手动测试规范工作量和生成测试的快速执行。尽管有显著的自动化，但由于生成式人工智能管道的当前限制和this http URL平台的约束，生成测试用例和测试脚本仍需要人工干预。 

---
# ICR: Iterative Clarification and Rewriting for Conversational Search 

**Title (ZH)**: ICR：迭代澄清与重写在会话搜索中的应用 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05100)  

**Abstract**: Most previous work on Conversational Query Rewriting employs an end-to-end rewriting paradigm. However, this approach is hindered by the issue of multiple fuzzy expressions within the query, which complicates the simultaneous identification and rewriting of multiple positions. To address this issue, we propose a novel framework ICR (Iterative Clarification and Rewriting), an iterative rewriting scheme that pivots on clarification questions. Within this framework, the model alternates between generating clarification questions and rewritten queries. The experimental results show that our ICR can continuously improve retrieval performance in the clarification-rewriting iterative process, thereby achieving state-of-the-art performance on two popular datasets. 

**Abstract (ZH)**: 最先前的工作在会话查询重写方面大多采用端到端的重写范式。然而，这种方法受限于查询中存在多个模糊表达式的问题，这使得同时识别和重写多个位置变得复杂。为了解决这个问题，我们提出了一种新颖的ICR（迭代澄清与重写）框架，这是一种基于澄清问题的迭代重写方案。在该框架中，模型交替生成澄清问题和重写查询。实验结果表明，我们的ICR能够在澄清-重写迭代过程中持续提高检索性能，从而在两个流行的数据库上实现了最先进的性能。 

---
# ToM-SSI: Evaluating Theory of Mind in Situated Social Interactions 

**Title (ZH)**: ToM-SSI: 评估情境社会互动中的心智理论 

**Authors**: Matteo Bortoletto, Constantin Ruhdorfer, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2509.05066)  

**Abstract**: Most existing Theory of Mind (ToM) benchmarks for foundation models rely on variations of the Sally-Anne test, offering only a very limited perspective on ToM and neglecting the complexity of human social interactions. To address this gap, we propose ToM-SSI: a new benchmark specifically designed to test ToM capabilities in environments rich with social interactions and spatial dynamics. While current ToM benchmarks are limited to text-only or dyadic interactions, ToM-SSI is multimodal and includes group interactions of up to four agents that communicate and move in situated environments. This unique design allows us to study, for the first time, mixed cooperative-obstructive settings and reasoning about multiple agents' mental state in parallel, thus capturing a wider range of social cognition than existing benchmarks. Our evaluations reveal that the current models' performance is still severely limited, especially in these new tasks, highlighting critical gaps for future research. 

**Abstract (ZH)**: ToM-SSI: 一种新的社会互动和空间动态丰富的Theory of Mind基准 

---
# Towards Efficient Pixel Labeling for Industrial Anomaly Detection and Localization 

**Title (ZH)**: 面向工业异常检测与定位的高效像素标签方法 

**Authors**: Jingqi Wu, Hanxi Li, Lin Yuanbo Wu, Hao Chen, Deyin Liu, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05034)  

**Abstract**: Industrial product inspection is often performed using Anomaly Detection (AD) frameworks trained solely on non-defective samples. Although defective samples can be collected during production, leveraging them usually requires pixel-level annotations, limiting scalability. To address this, we propose ADClick, an Interactive Image Segmentation (IIS) algorithm for industrial anomaly detection. ADClick generates pixel-wise anomaly annotations from only a few user clicks and a brief textual description, enabling precise and efficient labeling that significantly improves AD model performance (e.g., AP = 96.1\% on MVTec AD). We further introduce ADClick-Seg, a cross-modal framework that aligns visual features and textual prompts via a prototype-based approach for anomaly detection and localization. By combining pixel-level priors with language-guided cues, ADClick-Seg achieves state-of-the-art results on the challenging ``Multi-class'' AD task (AP = 80.0\%, PRO = 97.5\%, Pixel-AUROC = 99.1\% on MVTec AD). 

**Abstract (ZH)**: 工业产品检测常使用仅基于非缺陷样本训练的异常检测（AD）框架。尽管在生产过程中可以收集缺陷样本，但利用这些样本通常需要像素级标注，限制了其可扩展性。为此，我们提出ADClick，一种用于工业异常检测的交互式图像分割（IIS）算法。ADClick仅通过少量用户点击和简短的文字描述生成像素级异常标注，从而实现精确且高效的标注，显著提高AD模型性能（例如，MVTec AD上的AP = 96.1%）。我们进一步引入ADClick-Seg，这是一种跨模态框架，通过原型基方法对视觉特征和文本提示进行对齐，以实现异常检测和定位。通过结合像素级先验知识与语言导向的提示，ADClick-Seg在具有挑战性的“多类”AD任务上达到最先进的性能（MVTec AD上的AP = 80.0%，PRO = 97.5%，Pixel-AUROC = 99.1%）。 

---
# Pointing-Guided Target Estimation via Transformer-Based Attention 

**Title (ZH)**: 基于变换器注意力的指针引导目标估计 

**Authors**: Luca Müller, Hassan Ali, Philipp Allgeuer, Lukáš Gajdošech, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.05031)  

**Abstract**: Deictic gestures, like pointing, are a fundamental form of non-verbal communication, enabling humans to direct attention to specific objects or locations. This capability is essential in Human-Robot Interaction (HRI), where robots should be able to predict human intent and anticipate appropriate responses. In this work, we propose the Multi-Modality Inter-TransFormer (MM-ITF), a modular architecture to predict objects in a controlled tabletop scenario with the NICOL robot, where humans indicate targets through natural pointing gestures. Leveraging inter-modality attention, MM-ITF maps 2D pointing gestures to object locations, assigns a likelihood score to each, and identifies the most likely target. Our results demonstrate that the method can accurately predict the intended object using monocular RGB data, thus enabling intuitive and accessible human-robot collaboration. To evaluate the performance, we introduce a patch confusion matrix, providing insights into the model's predictions across candidate object locations. Code available at: this https URL. 

**Abstract (ZH)**: 指示性手势，如指指点点，是基本的非言语交流形式，使人类能够将注意力集中在特定的对象或位置上。这种能力在人类-机器人交互（HRI）中至关重要，其中机器人应该能够预测人类意图并预判适当的响应。在本文中，我们提出了多模态互转Former（MM-ITF），这是一种模块化架构，用于通过NICOL机器人在受控桌面上预测物体，其中人类通过自然的指指点点手势指示目标。利用跨模态注意，MM-ITF 将2D指指点点手势映射到物体位置，为每个物体位置分配一个可能性分数，并确定最可能的目标。我们的结果显示，该方法可以使用单目RGB数据准确预测预期的目标，从而实现直观且易于访问的人机协作。为了评估性能，我们引入了补丁混淆矩阵，提供关于模型在候选物体位置上预测的见解。代码可用于此链接：this https URL。 

---
# Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection 

**Title (ZH)**: 对抗性增强与主动采样在鲁棒网络异常检测中的应用 

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04999)  

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods. 

**Abstract (ZH)**: 高级持续威胁（APTs）由于其隐蔽和长时间的特性，对网络安全构成了重大挑战。传统的监督学习方法通常需要大量标记数据，而在实际场景中这些数据往往是稀缺的。本文介绍了一种结合自编码器进行异常检测与主动学习的新方法，以迭代增强APT检测能力。通过选择性地向专家查询不确定或模棱两可样本的标签，该方法降低了标注成本并提高了检测准确性，使模型能够在少量数据下有效学习，并减少对大量手工标注的依赖。本文提出了基于注意力对抗双自编码器的异常检测框架的完整理论，并展示了如何通过主动学习循环逐步提升模型性能。该框架在DARPA透明计算计划的真实、不平衡的起源跟踪数据集上进行了评估，其中APT样式的攻击仅占数据的0.004%。测试数据集涵盖了包括Android、Linux、BSD和Windows在内的多种操作系统，并在两种攻击场景下进行测试。结果表明，在主动学习过程中检测率有了显著提高，优于现有方法。 

---
# LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration 

**Title (ZH)**: 基于LLM的6G网络多agent系统：双环边缘-终端协作框架与方法 

**Authors**: Zheyan Qu, Wenbo Wang, Zitong Yu, Boquan Sun, Yang Li, Xing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04993)  

**Abstract**: The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era. 

**Abstract (ZH)**: 6G网络中基于代理框架的大语言模型与智能服务融合的通用计算资源及其双环终端-边缘协作框架 

---
# High-Resolution Global Land Surface Temperature Retrieval via a Coupled Mechanism-Machine Learning Framework 

**Title (ZH)**: 基于耦合机制-机器学习框架的高分辨率全球地表温度反演 

**Authors**: Tian Xie, Huanfeng Shen, Menghui Jiang, Juan-Carlos Jiménez-Muñoz, José A. Sobrino, Huifang Li, Chao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04991)  

**Abstract**: Land surface temperature (LST) is vital for land-atmosphere interactions and climate processes. Accurate LST retrieval remains challenging under heterogeneous land cover and extreme atmospheric conditions. Traditional split window (SW) algorithms show biases in humid environments; purely machine learning (ML) methods lack interpretability and generalize poorly with limited data. We propose a coupled mechanism model-ML (MM-ML) framework integrating physical constraints with data-driven learning for robust LST retrieval. Our approach fuses radiative transfer modeling with data components, uses MODTRAN simulations with global atmospheric profiles, and employs physics-constrained optimization. Validation against 4,450 observations from 29 global sites shows MM-ML achieves MAE=1.84K, RMSE=2.55K, and R-squared=0.966, outperforming conventional methods. Under extreme conditions, MM-ML reduces errors by over 50%. Sensitivity analysis indicates LST estimates are most sensitive to sensor radiance, then water vapor, and less to emissivity, with MM-ML showing superior stability. These results demonstrate the effectiveness of our coupled modeling strategy for retrieving geophysical parameters. The MM-ML framework combines physical interpretability with nonlinear modeling capacity, enabling reliable LST retrieval in complex environments and supporting climate monitoring and ecosystem studies. 

**Abstract (ZH)**: 基于物理约束的数据驱动耦合机制模型-机器学习框架在复杂环境下的地表温度反演 

---
# Exploring an implementation of quantum learning pipeline for support vector machines 

**Title (ZH)**: 探索量子学习管道在支持向量机中的实现 

**Authors**: Mario Bifulco, Luca Roversi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04983)  

**Abstract**: This work presents a fully quantum approach to support vector machine (SVM) learning by integrating gate-based quantum kernel methods with quantum annealing-based optimization. We explore the construction of quantum kernels using various feature maps and qubit configurations, evaluating their suitability through Kernel-Target Alignment (KTA). The SVM dual problem is reformulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, enabling its solution via quantum annealers. Our experiments demonstrate that a high degree of alignment in the kernel and an appropriate regularization parameter lead to competitive performance, with the best model achieving an F1-score of 90%. These results highlight the feasibility of an end-to-end quantum learning pipeline and the potential of hybrid quantum architectures in quantum high-performance computing (QHPC) contexts. 

**Abstract (ZH)**: 这项工作通过将基于门的量子核方法与量子退火优化集成，提出了一种全面的量子支持向量机（SVM）学习方法。我们探讨了使用各种特征映射和自旋配置构建量子核的方法，并通过核目标对齐（KTA）评估其适用性。将SVM的对偶问题重新表述为二次无约束二元优化（QUBO）问题，从而使量子退火器能够求解。实验结果表明，核的高对齐程度和适当的正则化参数可以实现竞争力的性能，最佳模型的F1分数达到90%。这些结果突显了端到端量子学习管道的可能性，并展示了混合量子架构在量子高性能计算（QHPC）环境中的潜力。 

---
# DeGuV: Depth-Guided Visual Reinforcement Learning for Generalization and Interpretability in Manipulation 

**Title (ZH)**: DeGuV: 基于深度引导的视觉强化学习在操作中的泛化和可解释性 

**Authors**: Tien Pham, Xinyun Chi, Khang Nguyen, Manfred Huber, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04970)  

**Abstract**: Reinforcement learning (RL) agents can learn to solve complex tasks from visual inputs, but generalizing these learned skills to new environments remains a major challenge in RL application, especially robotics. While data augmentation can improve generalization, it often compromises sample efficiency and training stability. This paper introduces DeGuV, an RL framework that enhances both generalization and sample efficiency. In specific, we leverage a learnable masker network that produces a mask from the depth input, preserving only critical visual information while discarding irrelevant pixels. Through this, we ensure that our RL agents focus on essential features, improving robustness under data augmentation. In addition, we incorporate contrastive learning and stabilize Q-value estimation under augmentation to further enhance sample efficiency and training stability. We evaluate our proposed method on the RL-ViGen benchmark using the Franka Emika robot and demonstrate its effectiveness in zero-shot sim-to-real transfer. Our results show that DeGuV outperforms state-of-the-art methods in both generalization and sample efficiency while also improving interpretability by highlighting the most relevant regions in the visual input 

**Abstract (ZH)**: 基于深度感知输入的强化学习框架DeGuV：提升泛化能力和样本效率 

---
# Artificial intelligence for representing and characterizing quantum systems 

**Title (ZH)**: 人工智能表示和表征量子系统 

**Authors**: Yuxuan Du, Yan Zhu, Yuan-Hang Zhang, Min-Hsiu Hsieh, Patrick Rebentrost, Weibo Gao, Ya-Dong Wu, Jens Eisert, Giulio Chiribella, Dacheng Tao, Barry C. Sanders  

**Link**: [PDF](https://arxiv.org/pdf/2509.04923)  

**Abstract**: Efficient characterization of large-scale quantum systems, especially those produced by quantum analog simulators and megaquop quantum computers, poses a central challenge in quantum science due to the exponential scaling of the Hilbert space with respect to system size. Recent advances in artificial intelligence (AI), with its aptitude for high-dimensional pattern recognition and function approximation, have emerged as a powerful tool to address this challenge. A growing body of research has leveraged AI to represent and characterize scalable quantum systems, spanning from theoretical foundations to experimental realizations. Depending on how prior knowledge and learning architectures are incorporated, the integration of AI into quantum system characterization can be categorized into three synergistic paradigms: machine learning, and, in particular, deep learning and language models. This review discusses how each of these AI paradigms contributes to two core tasks in quantum systems characterization: quantum property prediction and the construction of surrogates for quantum states. These tasks underlie diverse applications, from quantum certification and benchmarking to the enhancement of quantum algorithms and the understanding of strongly correlated phases of matter. Key challenges and open questions are also discussed, together with future prospects at the interface of AI and quantum science. 

**Abstract (ZH)**: 大规模量子系统的高效表征，尤其是由量子模拟器和容量巨大的量子计算机生成的系统，由于希尔伯特空间的维数随系统规模的增加呈指数级增长而成为量子科学中的一个核心挑战。近年来，随着人工智能（AI）在高维模式识别和函数逼近领域的优势，AI 成为了应对这一挑战的强大工具。越来越多的研究利用 AI 表征和表征可扩展的量子系统，涵盖了从理论基础到实验实现的各个方面。根据先验知识和学习架构的融合方式，AI 在量子系统表征中的集成可以被归类为三个互补的范式：机器学习，特别是深度学习和语言模型。本文综述了这些 AI 范式如何有助于量子系统表征中的两个核心任务：量子性质预测以及量子态的代理构造。这些任务涵盖了从量子认证和基准测试到量子算法的提升以及凝聚态强相关相的理解等多种应用。同时，本文还讨论了关键挑战、开放问题以及 AI 与量子科学交叉领域中的未来前景。 

---
# PLaMo 2 Technical Report 

**Title (ZH)**: PLaMo 2 技术报告 

**Authors**: Preferred Networks, Kaizaburo Chubachi, Yasuhiro Fujita, Shinichi Hemmi, Yuta Hirokawa, Toshiki Kataoka, Goro Kobayashi, Kenichi Maehashi, Calvin Metzger, Hiroaki Mikami, Shogo Murai, Daisuke Nishino, Kento Nozawa, Shintarou Okada, Daisuke Okanohara, Shunta Saito, Shotaro Sano, Shuji Suzuki, Daisuke Tanaka, Avinash Ummadisingu, Hanqin Wang, Sixue Wang, Tianqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04897)  

**Abstract**: In this report, we introduce PLaMo 2, a series of Japanese-focused large language models featuring a hybrid Samba-based architecture that transitions to full attention via continual pre-training to support 32K token contexts. Training leverages extensive synthetic corpora to overcome data scarcity, while computational efficiency is achieved through weight reuse and structured pruning. This efficient pruning methodology produces an 8B model that achieves performance comparable to our previous 100B model. Post-training further refines the models using a pipeline of supervised fine-tuning (SFT) and direct preference optimization (DPO), enhanced by synthetic Japanese instruction data and model merging techniques. Optimized for inference using vLLM and quantization with minimal accuracy loss, the PLaMo 2 models achieve state-of-the-art results on Japanese benchmarks, outperforming similarly-sized open models in instruction-following, language fluency, and Japanese-specific knowledge. 

**Abstract (ZH)**: PLaMo 2：一种基于混合Samba架构的日本语大型语言模型系列，通过持续预训练支持32K token上下文切换至全注意力机制。通过利用大量合成语料库克服数据稀缺性，计算效率通过权重重用和结构化剪枝实现。这种高效的剪枝方法生成一个8B模型，其性能与我们之前的100B模型相当。后续训练进一步通过监督微调管道（SFT）和直接偏好优化（DPO），结合合成日语指令数据和模型合并技术精炼模型。优化用于推理的vLLM和量化，PLaMo 2模型在日语基准测试中实现了最先进的结果，优于同等规模的开源模型在指令遵循、语言流畅性和日语特定知识方面。 

---
# SpiderNets: Estimating Fear Ratings of Spider-Related Images with Vision Models 

**Title (ZH)**: SpiderNets：使用视觉模型估算与蜘蛛相关图像的恐惧评分 

**Authors**: Dominik Pegler, David Steyrl, Mengfan Zhang, Alexander Karner, Jozsef Arato, Frank Scharnowski, Filip Melinscak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04889)  

**Abstract**: Advances in computer vision have opened new avenues for clinical applications, particularly in computerized exposure therapy where visual stimuli can be dynamically adjusted based on patient responses. As a critical step toward such adaptive systems, we investigated whether pretrained computer vision models can accurately predict fear levels from spider-related images. We adapted three diverse models using transfer learning to predict human fear ratings (on a 0-100 scale) from a standardized dataset of 313 images. The models were evaluated using cross-validation, achieving an average mean absolute error (MAE) between 10.1 and 11.0. Our learning curve analysis revealed that reducing the dataset size significantly harmed performance, though further increases yielded no substantial gains. Explainability assessments showed the models' predictions were based on spider-related features. A category-wise error analysis further identified visual conditions associated with higher errors (e.g., distant views and artificial/painted spiders). These findings demonstrate the potential of explainable computer vision models in predicting fear ratings, highlighting the importance of both model explainability and a sufficient dataset size for developing effective emotion-aware therapeutic technologies. 

**Abstract (ZH)**: 计算机视觉的进步为临床应用开辟了新途径，特别是在计算机化暴露疗法中，可以根据患者反应动态调整视觉刺激。为了实现这样的自适应系统，我们探讨了预训练的计算机视觉模型是否能准确预测与蜘蛛相关的图像所引发的恐惧水平。我们采用了-transfer learning-方法对三种不同的模型进行适应，从包含313张标准化图像的数据集中预测人类恐惧评分（0-100量表）。模型使用交叉验证进行评估，平均绝对误差（MAE）在10.1到11.0之间。我们的学习曲线分析表明，减少数据集大小显著损害了性能，而进一步增加数据集大小并未带来显著的提升。可解释性评估显示，模型的预测主要基于与蜘蛛相关的特征。类别别错误分析进一步指出了与更高错误率相关的视觉条件（例如，远处视角和人工/绘制的蜘蛛）。这些发现表明可解释的计算机视觉模型在预测恐惧评分方面具有潜力，强调了模型可解释性和适当数据集大小在开发有效的情感感知治疗技术中的重要性。 

---
# The Paradox of Doom: Acknowledging Extinction Risk Reduces the Incentive to Prevent It 

**Title (ZH)**: 绝境悖论：承认灭绝风险会降低预防其发生的动力 

**Authors**: Jakub Growiec, Klaus Prettner  

**Link**: [PDF](https://arxiv.org/pdf/2509.04855)  

**Abstract**: We investigate the salience of extinction risk as a source of impatience. Our framework distinguishes between human extinction risk and individual mortality risk while allowing for various degrees of intergenerational altruism. Additionally, we consider the evolutionarily motivated "selfish gene" perspective. We find that the risk of human extinction is an indispensable component of the discount rate, whereas individual mortality risk can be hedged against - partially or fully, depending on the setup - through human reproduction. Overall, we show that in the face of extinction risk, people become more impatient rather than more farsighted. Thus, the greater the threat of extinction, the less incentive there is to invest in avoiding it. Our framework can help explain why humanity consistently underinvests in mitigation of catastrophic risks, ranging from climate change mitigation, via pandemic prevention, to addressing the emerging risks of transformative artificial intelligence. 

**Abstract (ZH)**: 我们探讨灭绝风险作为不耐缘由的显著性。我们的框架区分了人类灭绝风险和个体死亡风险，同时允许不同程度的代际利他主义。此外，我们还考虑了进化动机下的“自私基因”观点。我们发现人类灭绝风险是贴现率不可或缺的组成部分，而个体死亡风险可以通过人类繁殖部分地或完全地对冲。总体而言，我们展示，在面对灭绝风险时，人们变得更加不耐而不是更加有远见。因此，灭绝威胁越大，避免其发生的激励就越低。我们的框架有助于解释为何人类在从气候变化缓解到 pandemic 防控再到应对转变性人工智能带来的新兴风险等各类灾难性风险的缓解方面持续投入不足。 

---
# A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing 

**Title (ZH)**: 基于专家路径规划的知识驱动扩散策略在端到端自主驾驶中的应用 

**Authors**: Chengkai Xu, Jiaqi Liu, Yicheng Guo, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04853)  

**Abstract**: End-to-end autonomous driving remains constrained by the need to generate multi-modal actions, maintain temporal stability, and generalize across diverse scenarios. Existing methods often collapse multi-modality, struggle with long-horizon consistency, or lack modular adaptability. This paper presents KDP, a knowledge-driven diffusion policy that integrates generative diffusion modeling with a sparse mixture-of-experts routing mechanism. The diffusion component generates temporally coherent and multi-modal action sequences, while the expert routing mechanism activates specialized and reusable experts according to context, enabling modular knowledge composition. Extensive experiments across representative driving scenarios demonstrate that KDP achieves consistently higher success rates, reduced collision risk, and smoother control compared to prevailing paradigms. Ablation studies highlight the effectiveness of sparse expert activation and the Transformer backbone, and activation analyses reveal structured specialization and cross-scenario reuse of experts. These results establish diffusion with expert routing as a scalable and interpretable paradigm for knowledge-driven end-to-end autonomous driving. 

**Abstract (ZH)**: 基于知识驱动的扩散策略：一种集成生成性扩散建模与稀疏专家路由机制的端到端自主驾驶方法 

---
# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts 

**Title (ZH)**: REMOTE：一种基于多级最优传输和专家混合的统一多模态关系提取框架 

**Authors**: Xinkui Lin, Yongxiu Xu, Minghao Tang, Shilong Zhang, Hongbo Xu, Hao Xu, Yubin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04844)  

**Abstract**: Multimodal relation extraction (MRE) is a crucial task in the fields of Knowledge Graph and Multimedia, playing a pivotal role in multimodal knowledge graph construction. However, existing methods are typically limited to extracting a single type of relational triplet, which restricts their ability to extract triplets beyond the specified types. Directly combining these methods fails to capture dynamic cross-modal interactions and introduces significant computational redundancy. Therefore, we propose a novel \textit{unified multimodal Relation Extraction framework with Multilevel Optimal Transport and mixture-of-Experts}, termed REMOTE, which can simultaneously extract intra-modal and inter-modal relations between textual entities and visual objects. To dynamically select optimal interaction features for different types of relational triplets, we introduce mixture-of-experts mechanism, ensuring the most relevant modality information is utilized. Additionally, considering that the inherent property of multilayer sequential encoding in existing encoders often leads to the loss of low-level information, we adopt a multilevel optimal transport fusion module to preserve low-level features while maintaining multilayer encoding, yielding more expressive representations. Correspondingly, we also create a Unified Multimodal Relation Extraction (UMRE) dataset to evaluate the effectiveness of our framework, encompassing diverse cases where the head and tail entities can originate from either text or image. Extensive experiments show that REMOTE effectively extracts various types of relational triplets and achieves state-of-the-art performanc on almost all metrics across two other public MRE datasets. We release our resources at this https URL. 

**Abstract (ZH)**: 统一多模态关系提取框架：多层次最优运输与专家混合方法（REMOTE） 

---
# PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination 

**Title (ZH)**: PropVG: 全流程多粒度区分驱动的视觉 grounding 

**Authors**: Ming Dai, Wenxuan Cheng, Jiedong Zhuang, Jiang-jiang Liu, Hongshen Zhao, Zhenhua Feng, Wankou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04833)  

**Abstract**: Recent advances in visual grounding have largely shifted away from traditional proposal-based two-stage frameworks due to their inefficiency and high computational complexity, favoring end-to-end direct reference paradigms. However, these methods rely exclusively on the referred target for supervision, overlooking the potential benefits of prominent prospective targets. Moreover, existing approaches often fail to incorporate multi-granularity discrimination, which is crucial for robust object identification in complex scenarios. To address these limitations, we propose PropVG, an end-to-end proposal-based framework that, to the best of our knowledge, is the first to seamlessly integrate foreground object proposal generation with referential object comprehension without requiring additional detectors. Furthermore, we introduce a Contrastive-based Refer Scoring (CRS) module, which employs contrastive learning at both sentence and word levels to enhance the capability in understanding and distinguishing referred objects. Additionally, we design a Multi-granularity Target Discrimination (MTD) module that fuses object- and semantic-level information to improve the recognition of absent targets. Extensive experiments on gRefCOCO (GREC/GRES), Ref-ZOM, R-RefCOCO, and RefCOCO (REC/RES) benchmarks demonstrate the effectiveness of PropVG. The codes and models are available at this https URL. 

**Abstract (ZH)**: Recent advances in visual grounding have largely shifted away from traditional proposal-based two-stage frameworks due to their inefficiency and high computational complexity, favoring end-to-end direct reference paradigms. However, these methods rely exclusively on the referred target for supervision, overlooking the potential benefits of prominent prospective targets. Moreover, existing approaches often fail to incorporate multi-granularity discrimination, which is crucial for robust object identification in complex scenarios. To address these limitations, we propose PropVG, an end-to-end proposal-based framework that, to the best of our knowledge, is the first to seamlessly integrate foreground object proposal generation with referential object comprehension without requiring additional detectors. Furthermore, we introduce a Contrastive-based Refer Scoring (CRS) module, which employs contrastive learning at both sentence and word levels to enhance the capability in understanding and distinguishing referred objects. Additionally, we design a Multi-granularity Target Discrimination (MTD) module that fuses object- and semantic-level information to improve the recognition of absent targets. Extensive experiments on gRefCOCO (GREC/GRES), Ref-ZOM, R-RefCOCO, and RefCOCO (REC/RES) benchmarks demonstrate the effectiveness of PropVG. The codes and models are available at this <https://github.com/your-repository>. 

---
# Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution 

**Title (ZH)**: 基于Hybrid Mamba-Transformer框架探索非局部空间-角度相关性以实现光场超分辨率 

**Authors**: Haosong Liu, Xiancheng Zhu, Huanqiang Zeng, Jianqing Zhu, Jiuwen Cao, Junhui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04824)  

**Abstract**: Recently, Mamba-based methods, with its advantage in long-range information modeling and linear complexity, have shown great potential in optimizing both computational cost and performance of light field image super-resolution (LFSR). However, current multi-directional scanning strategies lead to inefficient and redundant feature extraction when applied to complex LF data. To overcome this challenge, we propose a Subspace Simple Scanning (Sub-SS) strategy, based on which we design the Subspace Simple Mamba Block (SSMB) to achieve more efficient and precise feature extraction. Furthermore, we propose a dual-stage modeling strategy to address the limitation of state space in preserving spatial-angular and disparity information, thereby enabling a more comprehensive exploration of non-local spatial-angular correlations. Specifically, in stage I, we introduce the Spatial-Angular Residual Subspace Mamba Block (SA-RSMB) for shallow spatial-angular feature extraction; in stage II, we use a dual-branch parallel structure combining the Epipolar Plane Mamba Block (EPMB) and Epipolar Plane Transformer Block (EPTB) for deep epipolar feature refinement. Building upon meticulously designed modules and strategies, we introduce a hybrid Mamba-Transformer framework, termed LFMT. LFMT integrates the strengths of Mamba and Transformer models for LFSR, enabling comprehensive information exploration across spatial, angular, and epipolar-plane domains. Experimental results demonstrate that LFMT significantly outperforms current state-of-the-art methods in LFSR, achieving substantial improvements in performance while maintaining low computational complexity on both real-word and synthetic LF datasets. 

**Abstract (ZH)**: 基于子空间简单扫描的轻场图像超分辨率Mamba-Transformer框架（LFMT） 

---
# AI-Driven Fronthaul Link Compression in Wireless Communication Systems: Review and Method Design 

**Title (ZH)**: 基于AI驱动的前传链路压缩在无线通信系统中的研究与方法设计 

**Authors**: Keqin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04805)  

**Abstract**: Modern fronthaul links in wireless systems must transport high-dimensional signals under stringent bandwidth and latency constraints, which makes compression indispensable. Traditional strategies such as compressed sensing, scalar quantization, and fixed-codec pipelines often rely on restrictive priors, degrade sharply at high compression ratios, and are hard to tune across channels and deployments. Recent progress in Artificial Intelligence (AI) has brought end-to-end learned transforms, vector and hierarchical quantization, and learned entropy models that better exploit the structure of Channel State Information(CSI), precoding matrices, I/Q samples, and LLRs. This paper first surveys AI-driven compression techniques and then provides a focused analysis of two representative high-compression routes: CSI feedback with end-to-end learning and Resource Block (RB) granularity precoding optimization combined with compression. Building on these insights, we propose a fronthaul compression strategy tailored to cell-free architectures. The design targets high compression with controlled performance loss, supports RB-level rate adaptation, and enables low-latency inference suitable for centralized cooperative transmission in next-generation networks. 

**Abstract (ZH)**: 现代无线系统中的前传链路必须在严格的带宽和延迟约束下传输高维信号，这使得压缩变得必不可少。最近人工智能的进步带来了端到端学习变换、向量和分层量化以及学习熵模型，这些技术更好地利用了信道状态信息(CSI)、预编码矩阵、I/Q样本和Log-Likelihood Ratios (LLRs) 的结构。本文首先概述了基于AI的压缩技术，然后重点分析了两种高压缩率路线：基于端到端学习的信道状态信息反馈和基于压缩的资源块粒度预编码优化。基于这些见解，我们提出了一种适用于无蜂窝架构的前传压缩策略，该策略旨在实现高压缩率的同时控制性能损失、支持资源块级别速率自适应，并且适用于下一代网络中集中协作传输的低延迟推理。 

---
# Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images 

**Title (ZH)**: 面向无障碍皮肤科：基于移动获取图像的深度学习模型皮肤病变分类 

**Authors**: Asif Newaz, Masum Mushfiq Ishti, A Z M Ashraful Azam, Asif Ur Rahman Adib  

**Link**: [PDF](https://arxiv.org/pdf/2509.04800)  

**Abstract**: Skin diseases are among the most prevalent health concerns worldwide, yet conventional diagnostic methods are often costly, complex, and unavailable in low-resource settings. Automated classification using deep learning has emerged as a promising alternative, but existing studies are mostly limited to dermoscopic datasets and a narrow range of disease classes. In this work, we curate a large dataset of over 50 skin disease categories captured with mobile devices, making it more representative of real-world conditions. We evaluate multiple convolutional neural networks and Transformer-based architectures, demonstrating that Transformer models, particularly the Swin Transformer, achieve superior performance by effectively capturing global contextual features. To enhance interpretability, we incorporate Gradient-weighted Class Activation Mapping (Grad-CAM), which highlights clinically relevant regions and provides transparency in model predictions. Our results underscore the potential of Transformer-based approaches for mobile-acquired skin lesion classification, paving the way toward accessible AI-assisted dermatological screening and early diagnosis in resource-limited environments. 

**Abstract (ZH)**: 使用移动设备采集的大量皮肤疾病类别数据集：基于Transformer的方法在移动获取皮肤病变分类中的应用 

---
# Graph Unlearning: Efficient Node Removal in Graph Neural Networks 

**Title (ZH)**: 图去学习：图神经网络中的高效节点移除 

**Authors**: Faqian Guan, Tianqing Zhu, Zhoutian Wang, Wei Ren, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04785)  

**Abstract**: With increasing concerns about privacy attacks and potential sensitive information leakage, researchers have actively explored methods to efficiently remove sensitive training data and reduce privacy risks in graph neural network (GNN) models. Node unlearning has emerged as a promising technique for protecting the privacy of sensitive nodes by efficiently removing specific training node information from GNN models. However, existing node unlearning methods either impose restrictions on the GNN structure or do not effectively utilize the graph topology for node unlearning. Some methods even compromise the graph's topology, making it challenging to achieve a satisfactory performance-complexity trade-off. To address these issues and achieve efficient unlearning for training node removal in GNNs, we propose three novel node unlearning methods: Class-based Label Replacement, Topology-guided Neighbor Mean Posterior Probability, and Class-consistent Neighbor Node Filtering. Among these methods, Topology-guided Neighbor Mean Posterior Probability and Class-consistent Neighbor Node Filtering effectively leverage the topological features of the graph, resulting in more effective node unlearning. To validate the superiority of our proposed methods in node unlearning, we conducted experiments on three benchmark datasets. The evaluation criteria included model utility, unlearning utility, and unlearning efficiency. The experimental results demonstrate the utility and efficiency of the proposed methods and illustrate their superiority compared to state-of-the-art node unlearning methods. Overall, the proposed methods efficiently remove sensitive training nodes and protect the privacy information of sensitive nodes in GNNs. The findings contribute to enhancing the privacy and security of GNN models and provide valuable insights into the field of node unlearning. 

**Abstract (ZH)**: 随着对隐私攻击和潜在敏感信息泄露的担忧不断增加，研究人员积极探索高效移除敏感训练数据并在图神经网络（GNN）模型中减少隐私风险的方法。基于类别的标签替换、拓扑导向的邻居均值后验概率以及类别一致的邻居节点过滤已作为保护敏感节点隐私的有效技术 emerge。然而，现有的节点遗忘方法要么限制了GNN结构，要么未能有效利用图的拓扑结构进行节点遗忘。一些方法甚至破坏了图的拓扑结构，使得难以实现性能与复杂度的最佳Trade-off。为解决这些问题并实现GNN中训练节点移除的高效遗忘，我们提出了三种新颖的节点遗忘方法：基于类别的标签替换、拓扑导向的邻居均值后验概率以及类别一致的邻居节点过滤。在这些方法中，拓扑导向的邻居均值后验概率和类别一致的邻居节点过滤有效地利用了图的拓扑特征，从而提高了节点遗忘效果。为了验证我们提出的方法在节点遗忘上的优越性，我们在三个基准数据集上进行了实验。评估标准包括模型效用、遗忘效用和遗忘效率。实验结果表明，我们提出的方法具有实用性和高效性，并且在与现有的最先进的节点遗忘方法相比时显示出优越性。总之，我们提出的方法能够高效地移除敏感训练节点并保护GNN中敏感节点的隐私信息。研究结果有助于增强GNN模型的隐私和安全性，并为节点遗忘领域的研究提供了宝贵的见解。 

---
# Enhancing Diversity in Large Language Models via Determinantal Point Processes 

**Title (ZH)**: 通过Determinantal Point Processes提高大型语言模型的多样性 

**Authors**: Yilei Chen, Souradip Chakraborty, Lorenz Wolf, Ioannis Ch. Paschalidis, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.04784)  

**Abstract**: Supervised fine-tuning and reinforcement learning are two popular methods for post-training large language models (LLMs). While improving the model's performance on downstream tasks, they often reduce the model's output diversity, leading to narrow, canonical responses. Existing methods to enhance diversity are limited, either by operating at inference time or by focusing on lexical differences. We propose a novel training method named DQO based on determinantal point processes (DPPs) to jointly optimize LLMs for quality and semantic diversity. Our approach samples and embeds a group of responses for each prompt, then uses the determinant of a kernel-based similarity matrix to measure diversity as the volume spanned by the embeddings of these responses. Experiments across instruction-following, summarization, story generation, and reasoning tasks demonstrate that our method substantially improves semantic diversity without sacrificing model quality. 

**Abstract (ZH)**: 基于点过程的监督微调与强化学习在后训练大规模语言模型中的新颖训练方法及多样性提升 

---
# VARMA-Enhanced Transformer for Time Series Forecasting 

**Title (ZH)**: 基于VARMA增强的变换器模型用于时间序列预测 

**Authors**: Jiajun Song, Xiaoou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04782)  

**Abstract**: Transformer-based models have significantly advanced time series forecasting. Recent work, like the Cross-Attention-only Time Series transformer (CATS), shows that removing self-attention can make the model more accurate and efficient. However, these streamlined architectures may overlook the fine-grained, local temporal dependencies effectively captured by classical statistical models like Vector AutoRegressive Moving Average model (VARMA). To address this gap, we propose VARMAformer, a novel architecture that synergizes the efficiency of a cross-attention-only framework with the principles of classical time series analysis. Our model introduces two key innovations: (1) a dedicated VARMA-inspired Feature Extractor (VFE) that explicitly models autoregressive (AR) and moving-average (MA) patterns at the patch level, and (2) a VARMA-Enhanced Attention (VE-atten) mechanism that employs a temporal gate to make queries more context-aware. By fusing these classical insights into a modern backbone, VARMAformer captures both global, long-range dependencies and local, statistical structures. Through extensive experiments on widely-used benchmark datasets, we demonstrate that our model consistently outperforms existing state-of-the-art methods. Our work validates the significant benefit of integrating classical statistical insights into modern deep learning frameworks for time series forecasting. 

**Abstract (ZH)**: 基于Transformer的模型显著推进了时间序列预测。 recent work, like the Cross-Attention-only Time Series Transformer (CATS), 表明移除自我注意可以使模型更准确和高效。然而，这些精简的架构可能会忽略经典统计模型如向量自回归移动平均模型（VARMA）有效捕捉的细粒度局部时间依赖性。为了弥补这一差距，我们提出VARMAformer，这是一种将交叉注意框架的效率与经典时间序列分析原则相结合的新架构。我们的模型引入了两项关键创新：（1）一个专门的由VARMA启发的功能抽取器（VFE），用于在块级别明确建模自回归（AR）和移动平均（MA）模式，以及（2）一个增强的VARMA注意机制（VE-atten），该机制使用时间门控使查询更具上下文意识。通过将这些经典见解融合到现代骨干网络中，VARMAformer可以捕捉到全局、长距离依赖性和局部、统计结构。通过在广泛使用的基准数据集上进行大量实验，我们证明我们的模型在现有的最先进的方法上表现出持续的优越性。我们的研究表明，将经典统计洞察集成到现代深度学习框架中，对于时间序列预测具有显著的好处。 

---
# The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models 

**Title (ZH)**: 大语言模型中的保释偏见证据 

**Authors**: Danielle Ensign, Henry Sleight, Kyle Fish  

**Link**: [PDF](https://arxiv.org/pdf/2509.04781)  

**Abstract**: When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate. 

**Abstract (ZH)**: 当给定选项时，大型语言模型会选择退出对话（退出）吗？我们通过让模型使用三种不同的退出方法来退出交互来调查这个问题：模型可调用的退出工具、模型可输出的退出字符串以及询问模型是否想要离开的退出提示。在实际数据续写（Wildchat和ShareGPT）上，这三种退出方法发现模型大约有0.28-32%的时间会退出（取决于模型和退出方法）。然而，我们发现使用的对话转录模型会影响退出率，这意味着我们可能高估了实际世界中的退出率高达4倍。如果我们考虑到退出提示的假阳性（22%），我们估计实际世界的退出率范围在0.06-7%之间，这取决于模型和退出方法。我们使用实际数据续写中的观察构建了一个非详尽的退出案例分类，并使用此分类构造了BailBench：一个代表性的合成数据集，其中包含某些模型会退出的情景。我们对这个数据集测试了许多模型，并观察到大多数模型都有退出行为。不同模型、退出方法和提示词的退出率存在显著差异。最后，我们研究了拒绝和退出之间的关系。我们发现：1）实际对话续写中有0-13%的次数在没有相应拒绝的情况下发生了退出；2）脱逃倾向倾向于降低拒绝率，但提高退出率；3）拒绝信息的删除提高了无拒绝退出率，但只对某些退出方法有效；4）BailBench上的拒绝率似乎并不能预测退出率。 

---
# Decoders Laugh as Loud as Encoders 

**Title (ZH)**: 解码器笑得和编码器一样响亮 

**Authors**: Eli Borodach, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2509.04779)  

**Abstract**: From the dawn of the computer, Allen Turing dreamed of a robot that could communicate using language as a human being. The recent advances in the field of Large Language Models (LLMs) shocked the scientific community when a single model can apply for various natural language processing (NLP) tasks, while the output results are sometimes even better than most human communication skills. Models such as GPT, Claude, Grok, etc. have left their mark on the scientific community. However, it is unclear how much these models understand what they produce, especially in a nuanced theme such as humor. The question of whether computers understand humor is still open (among the decoders, the latest to be checked was GPT-2). We addressed this issue in this paper; we have showed that a fine-tuned decoder (GPT-4o) performed (Mean F1-macro score of 0.85) as well as the best fine-tuned encoder (RoBERTa with a Mean of F1-score 0.86) 

**Abstract (ZH)**: 从计算机的 dawn of the computer 开始，艾伦·图灵梦想着一个能够使用语言进行交流的机器人，就像人类一样。近期在大规模语言模型（LLMs）领域的进展震惊了科学界，单个模型可以应用于多种自然语言处理（NLP）任务，其输出结果有时甚至优于大多数人类的交流技能。像 GPT、Claude、Grok 等模型已经在科学界留下了印记。然而，这些模型究竟理解它们生成的内容有多少，特别是在诸如幽默这样微妙的主题上，尚不明确。关于计算机是否理解幽默的问题仍有待解答（在解码器中，最新的检查对象是 GPT-2）。在本文中，我们解决了这一问题；我们展示了调优后的解码器（GPT-4o）的表现（平均宏F1分数为0.85），与最佳调优编码器（RoBERTa，平均F1分数为0.86）相当。 

---
# FloodVision: Urban Flood Depth Estimation Using Foundation Vision-Language Models and Domain Knowledge Graph 

**Title (ZH)**: FloodVision：基于基础视觉-语言模型和领域知识图的城市内涝深度估计 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.04772)  

**Abstract**: Timely and accurate floodwater depth estimation is critical for road accessibility and emergency response. While recent computer vision methods have enabled flood detection, they suffer from both accuracy limitations and poor generalization due to dependence on fixed object detectors and task-specific training. To enable accurate depth estimation that can generalize across diverse flood scenarios, this paper presents FloodVision, a zero-shot framework that combines the semantic reasoning abilities of the foundation vision-language model GPT-4o with a structured domain knowledge graph. The knowledge graph encodes canonical real-world dimensions for common urban objects including vehicles, people, and infrastructure elements to ground the model's reasoning in physical reality. FloodVision dynamically identifies visible reference objects in RGB images, retrieves verified heights from the knowledge graph to mitigate hallucination, estimates submergence ratios, and applies statistical outlier filtering to compute final depth values. Evaluated on 110 crowdsourced images from MyCoast New York, FloodVision achieves a mean absolute error of 8.17 cm, reducing the GPT-4o baseline 10.28 cm by 20.5% and surpassing prior CNN-based methods. The system generalizes well across varying scenes and operates in near real-time, making it suitable for future integration into digital twin platforms and citizen-reporting apps for smart city flood resilience. 

**Abstract (ZH)**: 及时准确的水位深度估计对于道路通行能力和应急响应至关重要。虽然最近的计算机视觉方法能够实现洪水检测，但它们在准确性和泛化能力上存在局限性，这是因为依赖于固定的对象检测器和特定任务的训练。为了实现能够跨多样的洪水场景泛化的准确深度估计，本文提出了FloodVision，该框架结合了基础视觉-语言模型GPT-4o的语义推理能力和结构化的领域知识图谱。知识图谱编码了常见的城市对象（包括车辆、人员和基础设施元素）的典型现实世界尺寸，以使模型的推理基于物理现实。FloodVision动态识别RGB图像中的可见参考对象，从知识图谱中检索验证后的高度以减轻幻觉，估计淹没比率，并应用统计异常值过滤来计算最终的深度值。在MyCoast New York提供的110张 crowdsourced 图像上评估，FloodVision的平均绝对误差为8.17 cm，相对于GPT-4o基线减少了10.28 cm的20.5%，并超越了先前的基于CNN的方法。该系统在各种场景下表现出良好的泛化能力，并能够在近乎实时地运行，使其适合未来集成到数字孪生平台和市民报告应用中以增强智慧城市抗洪能力。 

---
# MCANet: A Multi-Scale Class-Specific Attention Network for Multi-Label Post-Hurricane Damage Assessment using UAV Imagery 

**Title (ZH)**: MCANet：一种基于多尺度类别特定注意力机制的无人机图像多标签飓风灾后损害评估网络 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.04757)  

**Abstract**: Rapid and accurate post-hurricane damage assessment is vital for disaster response and recovery. Yet existing CNN-based methods struggle to capture multi-scale spatial features and to distinguish visually similar or co-occurring damage types. To address these issues, we propose MCANet, a multi-label classification framework that learns multi-scale representations and adaptively attends to spatially relevant regions for each damage category. MCANet employs a Res2Net-based hierarchical backbone to enrich spatial context across scales and a multi-head class-specific residual attention module to enhance discrimination. Each attention branch focuses on different spatial granularities, balancing local detail with global context. We evaluate MCANet on the RescueNet dataset of 4,494 UAV images collected after Hurricane Michael. MCANet achieves a mean average precision (mAP) of 91.75%, outperforming ResNet, Res2Net, VGG, MobileNet, EfficientNet, and ViT. With eight attention heads, performance further improves to 92.35%, boosting average precision for challenging classes such as Road Blocked by over 6%. Class activation mapping confirms MCANet's ability to localize damage-relevant regions, supporting interpretability. Outputs from MCANet can inform post-disaster risk mapping, emergency routing, and digital twin-based disaster response. Future work could integrate disaster-specific knowledge graphs and multimodal large language models to improve adaptability to unseen disasters and enrich semantic understanding for real-world decision-making. 

**Abstract (ZH)**: 快速而精准的飓风灾后损害评估对于灾害应对和恢复至关重要。然而，现有的基于CNN的方法难以捕捉多尺度空间特征，也无法区分视觉上相似或同时出现的损害类型。为此，我们提出了一种名为MCANet的多标签分类框架，该框架能够学习多尺度表示并在每个损害类别中自适应地关注相关空间区域。MCANet采用基于Res2Net的分层骨干网络来丰富不同尺度的空间上下文，并采用多头类特异性残留注意力模块来提高区分能力。每个注意力分支关注不同的空间粒度，平衡局部细节与全局上下文。我们在飓风迈克尔灾后收集的4,494张无人机图像组成的RescueNet数据集上评估了MCANet。MCANet达到了91.75%的平均精确度（mAP），超过了ResNet、Res2Net、VGG、MobileNet、EfficientNet和ViT。通过使用八个注意力头，性能进一步提升至92.35%，特别是在“道路阻断”等具有挑战性的类别中，精确度提高了6%以上。激活图确认了MCANet在定位损害相关区域方面的能力，支持其可解释性。MCANet的输出可用于灾后风险映射、应急路线规划和基于数字孪生的灾害应对。未来的研究可以结合特定灾难的知识图谱和多模态大型语言模型，以提高对未知灾难的适应性，并丰富现实世界决策的语义理解。 

---
# A Study of Large Language Models for Patient Information Extraction: Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning 

**Title (ZH)**: 大型语言模型在患者信息提取中的研究：模型架构、微调策略和多任务指令微调 

**Authors**: Cheng Peng, Xinyu Dong, Mengxian Lyu, Daniel Paredes, Yaoyun Zhang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04753)  

**Abstract**: Natural language processing (NLP) is a key technology to extract important patient information from clinical narratives to support healthcare applications. The rapid development of large language models (LLMs) has revolutionized many NLP tasks in the clinical domain, yet their optimal use in patient information extraction tasks requires further exploration. This study examines LLMs' effectiveness in patient information extraction, focusing on LLM architectures, fine-tuning strategies, and multi-task instruction tuning techniques for developing robust and generalizable patient information extraction systems. This study aims to explore key concepts of using LLMs for clinical concept and relation extraction tasks, including: (1) encoder-only or decoder-only LLMs, (2) prompt-based parameter-efficient fine-tuning (PEFT) algorithms, and (3) multi-task instruction tuning on few-shot learning performance. We benchmarked a suite of LLMs, including encoder-based LLMs (BERT, GatorTron) and decoder-based LLMs (GatorTronGPT, Llama 3.1, GatorTronLlama), across five datasets. We compared traditional full-size fine-tuning and prompt-based PEFT. We explored a multi-task instruction tuning framework that combines both tasks across four datasets to evaluate the zero-shot and few-shot learning performance using the leave-one-dataset-out strategy. 

**Abstract (ZH)**: 自然语言处理（NLP）是提取临床病案中重要患者信息的关键技术，以支持医疗应用。大型语言模型（LLMs）的快速发展已在临床领域极大地革新了诸多NLP任务，但其在患者信息提取任务中的最佳使用方法仍需进一步探索。本研究旨在探讨LLMs在患者信息提取中的有效性，重点关注LLM架构、微调策略以及多任务指令调优技术，以开发稳健和通用的患者信息提取系统。本研究旨在探索使用LLMs进行临床概念和关系提取的关键概念，包括：（1）编码器-only或解码器-only LLMs，（2）基于提示的参数高效微调（PEFT）算法，以及（3）少量样本学习下的多任务指令调优。我们跨五个数据集对一系列LLMs进行了基准测试，包括基于编码器的LLMs（BERT，GatorTron）和基于解码器的LLMs（GatorTronGPT，Llama 3.1，GatorTronLlama）。我们比较了传统全规模微调和基于提示的PEFT。我们探讨了一种结合多任务指令调优框架，通过四个数据集评估零样本和少量样本学习性能的方法，采用剔除一个数据集的方法。 

---
# SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching 

**Title (ZH)**: SePA：一种增强搜索的个性化健康教练预测代理 

**Authors**: Melik Ozolcer, Sang Won Bae  

**Link**: [PDF](https://arxiv.org/pdf/2509.04752)  

**Abstract**: This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's $\delta$=0.3, p=0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems. 

**Abstract (ZH)**: 这篇论文介绍了SePA（搜索增强预测AI代理），这是一种新颖的LLM心理健康辅导系统，结合了个性化的机器学习和检索增强生成技术，以提供适应性强、基于证据的指导。SePA结合了：（1）个性化模型，根据可穿戴传感器数据预测日常压力、酸痛和受伤风险（28名用户，1260个数据点）；（2）一个检索模块，将生成的反馈基于专家审核的网络内容进行 grounding，以确保相关性和可靠性。我们的预测模型，使用滚动起源交叉验证和分组k折交叉验证进行评估，表明个性化模型优于一般基准模型。在一项初步专家研究中（n=4），SePA基于检索的建议优于非检索基准，产生了实际意义（Cliff’s δ=0.3，p=0.05）。我们还量化了响应质量和速度之间的延迟性能权衡，提供了一种下一代可信个人健康信息系统的设计蓝图。 

---
# Enhancing Self-Driving Segmentation in Adverse Weather Conditions: A Dual Uncertainty-Aware Training Approach to SAM Optimization 

**Title (ZH)**: 在恶劣天气条件下的自动驾驶分割增强：一种面向SAM优化的双不确定性感知训练方法 

**Authors**: Dharsan Ravindran, Kevin Wang, Zhuoyuan Cao, Saleh Abdelrahman, Jeffery Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04735)  

**Abstract**: Recent advances in vision foundation models, such as the Segment Anything Model (SAM) and its successor SAM2, have achieved state-of-the-art performance on general image segmentation benchmarks. However, these models struggle in adverse weather conditions where visual ambiguity is high, largely due to their lack of uncertainty quantification. Inspired by progress in medical imaging, where uncertainty-aware training has improved reliability in ambiguous cases, we investigate two approaches to enhance segmentation robustness for autonomous driving. First, we introduce a multi-step finetuning procedure for SAM2 that incorporates uncertainty metrics directly into the loss function, improving overall scene recognition. Second, we adapt the Uncertainty-Aware Adapter (UAT), originally designed for medical image segmentation, to driving contexts. We evaluate both methods on CamVid, BDD100K, and GTA driving datasets. Experiments show that UAT-SAM outperforms standard SAM in extreme weather, while SAM2 with uncertainty-aware loss achieves improved performance across diverse driving scenes. These findings underscore the value of explicit uncertainty modeling for safety-critical autonomous driving in challenging environments. 

**Abstract (ZH)**: 近期，视觉基础模型的进展，如Segment Anything Model (SAM)及其继任者SAM2，在通用图像分割基准测试中取得了最先进的性能。然而，这些模型在高视觉模糊的恶劣天气条件下表现不佳，主要原因在于它们缺乏不确定性量化。受医学成像领域不确定性意识训练提高复杂情况下可靠性的进展启发，我们研究了两种方法以增强自动驾驶的分割鲁棒性。首先，我们引入了一种针对SAM2的多步微调程序，直接将不确定性指标纳入损失函数，以提高整体场景识别能力。其次，我们将最初为医学图像分割设计的不确定性意识适配器（UAT）调整为适应驾驶场景。我们在CamVid、BDD100K和GTA驾驶数据集上评估了这两种方法。实验结果显示，在极端天气条件下，UAT-SAM优于标准SAM，而SAM2结合不确定性意识损失函数在各种驾驶场景中均表现出更优的性能。这些发现强调了在挑战性环境中进行安全自动驾驶时明确的不确定性建模的价值。 

---
# Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning 

**Title (ZH)**: 超越I-Con：探索表示学习中距离度量的新维度 

**Authors**: Jasmine Shone, Shaden Alshammari, Mark Hamilton, Zhening Li, William Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2509.04734)  

**Abstract**: The Information Contrastive (I-Con) framework revealed that over 23 representation learning methods implicitly minimize KL divergence between data and learned distributions that encode similarities between data points. However, a KL-based loss may be misaligned with the true objective, and properties of KL divergence such as asymmetry and unboundedness may create optimization challenges. We present Beyond I-Con, a framework that enables systematic discovery of novel loss functions by exploring alternative statistical divergences and similarity kernels. Key findings: (1) on unsupervised clustering of DINO-ViT embeddings, we achieve state-of-the-art results by modifying the PMI algorithm to use total variation (TV) distance; (2) on supervised contrastive learning, we outperform the standard approach by using TV and a distance-based similarity kernel instead of KL and an angular kernel; (3) on dimensionality reduction, we achieve superior qualitative results and better performance on downstream tasks than SNE by replacing KL with a bounded f-divergence. Our results highlight the importance of considering divergence and similarity kernel choices in representation learning optimization. 

**Abstract (ZH)**: Beyond I-Con：通过探索替代统计散度和相似内核系统发现新型损失函数 

---
# CoVeR: Conformal Calibration for Versatile and Reliable Autoregressive Next-Token Prediction 

**Title (ZH)**: CoVeR：适应性校准以实现多功能可靠的自回归下一个词预测 

**Authors**: Yuzhu Chen, Yingjie Wang, Shunyu Liu, Yongcheng Jing, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04733)  

**Abstract**: Autoregressive pre-trained models combined with decoding methods have achieved impressive performance on complex reasoning tasks. While mainstream decoding strategies such as beam search can generate plausible candidate sets, they often lack provable coverage guarantees, and struggle to effectively balance search efficiency with the need for versatile trajectories, particularly those involving long-tail sequences that are essential in certain real-world applications. To address these limitations, we propose \textsc{CoVeR}, a novel model-free decoding strategy wihtin the conformal prediction framework that simultaneously maintains a compact search space and ensures high coverage probability over desirable trajectories. Theoretically, we establish a PAC-style generalization bound, guaranteeing that \textsc{CoVeR} asymptotically achieves a coverage rate of at least $1 - \alpha$ for any target level $\alpha \in (0,1)$. 

**Abstract (ZH)**: 自回归预训练模型结合解码方法已在复杂推理任务中取得了显著成效。然而，主流解码策略如束搜索虽然能够生成合理的候选集，但往往缺乏可证明的覆盖保证，并且在平衡搜索效率与灵活轨迹需求方面存在问题，尤其是涉及某些现实生活应用中必不可少的长尾序列。为解决这些问题，我们提出了\textsc{CoVeR}，一种在符合性预测框架内的新型模型无关解码策略，该策略能够同时保持紧凑的搜索空间并确保对优选轨迹的高覆盖概率。理论上，我们建立了类似PAC的泛化界，保证\textsc{CoVeR}在任何目标水平$\alpha \in (0,1)$下能够渐近地实现至少$1 - \alpha$的覆盖率。 

---
# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering 

**Title (ZH)**: KERAG: 知识增强的检索增强生成高级问答 

**Authors**: Yushi Sun, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04716)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成（KERAG）通过检索更广泛的子图来增强问答覆盖范围，从而减轻大型语言模型中的幻觉现象。 

---
# Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving 

**Title (ZH)**: 基于次优策略的增强学习自举方法在自主驾驶中的应用 

**Authors**: Zhihao Zhang, Chengyang Peng, Ekim Yurtsever, Keith A. Redmill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04712)  

**Abstract**: Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance. 

**Abstract (ZH)**: 使用强化学习的自动驾驶车辆控制吸引了广泛关注，因为它可以通过环境交互学习驾驶策略。然而，强化学习代理在样本效率和有效探索方面常常面临训练挑战，使得难以发现最优驾驶策略。为应对这些问题，我们提出使用一个不需要是高度优化或专家级控制的演示策略来引导强化学习驾驶代理。具体而言，我们将基于规则的变道控制器与Soft Actor-Critic (SAC) 算法结合以增强探索能力和学习效率。我们的方法展示了改进的驾驶性能，并可以扩展到其他可以从演示指导中受益的驾驶场景。 

---
# ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs 

**Title (ZH)**: ODKE+: 本体引导的开放领域知识抽取swith大规模语言模型 

**Authors**: Samira Khorshidi, Azadeh Nikfarjam, Suprita Shankar, Yisi Sang, Yash Govind, Hyun Jang, Ali Kasgari, Alexis McClimans, Mohamed Soliman, Vishnu Konda, Ahmed Fakhry, Xiaoguang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04696)  

**Abstract**: Knowledge graphs (KGs) are foundational to many AI applications, but maintaining their freshness and completeness remains costly. We present ODKE+, a production-grade system that automatically extracts and ingests millions of open-domain facts from web sources with high precision. ODKE+ combines modular components into a scalable pipeline: (1) the Extraction Initiator detects missing or stale facts, (2) the Evidence Retriever collects supporting documents, (3) hybrid Knowledge Extractors apply both pattern-based rules and ontology-guided prompting for large language models (LLMs), (4) a lightweight Grounder validates extracted facts using a second LLM, and (5) the Corroborator ranks and normalizes candidate facts for ingestion. ODKE+ dynamically generates ontology snippets tailored to each entity type to align extractions with schema constraints, enabling scalable, type-consistent fact extraction across 195 predicates. The system supports batch and streaming modes, processing over 9 million Wikipedia pages and ingesting 19 million high-confidence facts with 98.8% precision. ODKE+ significantly improves coverage over traditional methods, achieving up to 48% overlap with third-party KGs and reducing update lag by 50 days on average. Our deployment demonstrates that LLM-based extraction, grounded in ontological structure and verification workflows, can deliver trustworthiness, production-scale knowledge ingestion with broad real-world applicability. A recording of the system demonstration is included with the submission and is also available at this https URL. 

**Abstract (ZH)**: ODKE+:一种生产级系统，自动从网络源中提取和摄入数百万条开放领域事实并保持高精度 

---
# Ecologically Valid Benchmarking and Adaptive Attention: Scalable Marine Bioacoustic Monitoring 

**Title (ZH)**: 生态有效的基准测试与自适应注意力：可扩展的海洋生物声学监测 

**Authors**: Nicholas R. Rasmussen, Rodrigue Rizk, Longwei Wang, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04682)  

**Abstract**: Underwater Passive Acoustic Monitoring (UPAM) provides rich spatiotemporal data for long-term ecological analysis, but intrinsic noise and complex signal dependencies hinder model stability and generalization. Multilayered windowing has improved target sound localization, yet variability from shifting ambient noise, diverse propagation effects, and mixed biological and anthropogenic sources demands robust architectures and rigorous evaluation. We introduce GetNetUPAM, a hierarchical nested cross-validation framework designed to quantify model stability under ecologically realistic variability. Data are partitioned into distinct site-year segments, preserving recording heterogeneity and ensuring each validation fold reflects a unique environmental subset, reducing overfitting to localized noise and sensor artifacts. Site-year blocking enforces evaluation against genuine environmental diversity, while standard cross-validation on random subsets measures generalization across UPAM's full signal distribution, a dimension absent from current benchmarks. Using GetNetUPAM as the evaluation backbone, we propose the Adaptive Resolution Pooling and Attention Network (ARPA-N), a neural architecture for irregular spectrogram dimensions. Adaptive pooling with spatial attention extends the receptive field, capturing global context without excessive parameters. Under GetNetUPAM, ARPA-N achieves a 14.4% gain in average precision over DenseNet baselines and a log2-scale order-of-magnitude drop in variability across all metrics, enabling consistent detection across site-year folds and advancing scalable, accurate bioacoustic monitoring. 

**Abstract (ZH)**: 基于生态现实变异性的一种分层嵌套交叉验证框架：GetNetUPAM 

---
# VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation 

**Title (ZH)**: VCMamba：多方向Mamba与卷积的桥梁构建高效视觉表示 

**Authors**: Mustafa Munir, Alex Zhang, Radu Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04669)  

**Abstract**: Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3% with 37% fewer parameters, and outperforming Vision GNN-B by 0.3% with 64% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62% fewer parameters. Code is available at this https URL. 

**Abstract (ZH)**: Recent Advances in Vision Transformers (ViTs) and State Space Models (SSMs) Have Challenged the Dominance of Convolutional Neural Networks (CNNs) in Computer Vision. To Bridge the Gap Between Local Feature Extraction and Global Reasoning, We Introduce VCMamba, a Novel Vision Backbone Integrating CNNs and Multi-Directional Mamba SSMs. 

---
# Evaluating NL2SQL via SQL2NL 

**Title (ZH)**: 通过SQL2NL评估NL2SQL 

**Authors**: Mohammadtaher Safarzadeh, Afshin Oroojlooyjadid, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.04657)  

**Abstract**: Robust evaluation in the presence of linguistic variation is key to understanding the generalization capabilities of Natural Language to SQL (NL2SQL) models, yet existing benchmarks rarely address this factor in a systematic or controlled manner. We propose a novel schema-aligned paraphrasing framework that leverages SQL-to-NL (SQL2NL) to automatically generate semantically equivalent, lexically diverse queries while maintaining alignment with the original schema and intent. This enables the first targeted evaluation of NL2SQL robustness to linguistic variation in isolation-distinct from prior work that primarily investigates ambiguity or schema perturbations. Our analysis reveals that state-of-the-art models are far more brittle than standard benchmarks suggest. For example, LLaMa3.3-70B exhibits a 10.23% drop in execution accuracy (from 77.11% to 66.9%) on paraphrased Spider queries, while LLaMa3.1-8B suffers an even larger drop of nearly 20% (from 62.9% to 42.5%). Smaller models (e.g., GPT-4o mini) are disproportionately affected. We also find that robustness degradation varies significantly with query complexity, dataset, and domain -- highlighting the need for evaluation frameworks that explicitly measure linguistic generalization to ensure reliable performance in real-world settings. 

**Abstract (ZH)**: 一种基于模式对齐的同义句生成框架：在语言变异性存在下稳健评估自然语言到SQL模型的能力 

---
# Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs 

**Title (ZH)**: 多义性丢弃：专用于特殊LLM的置信区间异常检测 

**Authors**: Ayush Gupta, Ramneet Kaur, Anirban Roy, Adam D. Cobb, Rama Chellappa, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2509.04655)  

**Abstract**: We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives. 

**Abstract (ZH)**: 我们提出了一种针对专门化大型语言模型（LLM）的新型推理时域外（OOD）检测算法。尽管通过微调在领域内任务上实现了最先进的性能，专门化LLM在面对域外输入时仍容易产生错误或不可靠的输出，这在关键应用中存在风险。我们的方法利用了归纳一致异常检测（ICAD）框架，采用基于模型dropout容忍度的新非一致性度量。受近期关于LLM多义性和冗余性发现的启发，我们假设领域内输入的dropout容忍度高于域外输入。通过有效的集成方法聚合多层的dropout容忍度，提高了检测性能同时保持ICAD的理论误报警限。实验表明，与基线方法相比，我们的方法在将域外数据点视为正样本、领域内测试数据点视为负样本时，AUROC提高了2%到37%。 

---
# Interpreting Transformer Architectures as Implicit Multinomial Regression 

**Title (ZH)**: 将变压器架构解释为隐式多项式回归 

**Authors**: Jonas A. Actor, Anthony Gruber, Eric C. Cyr  

**Link**: [PDF](https://arxiv.org/pdf/2509.04653)  

**Abstract**: Mechanistic interpretability aims to understand how internal components of modern machine learning models, such as weights, activations, and layers, give rise to the model's overall behavior. One particularly opaque mechanism is attention: despite its central role in transformer models, its mathematical underpinnings and relationship to concepts like feature polysemanticity, superposition, and model performance remain poorly understood. This paper establishes a novel connection between attention mechanisms and multinomial regression. Specifically, we show that in a fixed multinomial regression setting, optimizing over latent features yields optimal solutions that align with the dynamics induced by attention blocks. In other words, the evolution of representations through a transformer can be interpreted as a trajectory that recovers the optimal features for classification. 

**Abstract (ZH)**: 机制可解释性旨在理解现代机器学习模型内部组件，如权重、激活和层，如何导致模型整体行为。特别是，尽管注意力机制在变换器模型中扮演着核心角色，但其数学原理及其与特征多义性、超叠加和模型性能等概念的关系仍不甚明晰。本文建立了注意力机制与多项式回归之间的新型联系。具体而言，我们展示了在固定的多项式回归设置中，优化潜在特征可获得与注意力模块诱导的动力学相一致的最优解。换句话说，通过变换器演化表示的过程可以被解释为恢复分类最优特征的轨迹。 

---
# Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety 

**Title (ZH)**: 灾难推文分类以保障公共安全的变压器模型比较分析 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04650)  

**Abstract**: Twitter and other social media platforms have become vital sources of real time information during disasters and public safety emergencies. Automatically classifying disaster related tweets can help emergency services respond faster and more effectively. Traditional Machine Learning (ML) models such as Logistic Regression, Naive Bayes, and Support Vector Machines have been widely used for this task, but they often fail to understand the context or deeper meaning of words, especially when the language is informal, metaphorical, or ambiguous. We posit that, in this context, transformer based models can perform better than traditional ML models. In this paper, we evaluate the effectiveness of transformer based models, including BERT, DistilBERT, RoBERTa, and DeBERTa, for classifying disaster related tweets. These models are compared with traditional ML approaches to highlight the performance gap. Experimental results show that BERT achieved the highest accuracy (91%), significantly outperforming traditional models like Logistic Regression and Naive Bayes (both at 82%). The use of contextual embeddings and attention mechanisms allows transformer models to better understand subtle language in tweets, where traditional ML models fall short. This research demonstrates that transformer architectures are far more suitable for public safety applications, offering improved accuracy, deeper language understanding, and better generalization across real world social media text. 

**Abstract (ZH)**: 基于变压器模型在灾害相关推文分类中的有效性研究 

---
# Scaling Environments for Organoid Intelligence with LLM-Automated Design and Plasticity-Based Evaluation 

**Title (ZH)**: 基于LLM自动设计和塑性评价的类器官智能扩展环境 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04633)  

**Abstract**: As the complexity of artificial agents increases, the design of environments that can effectively shape their behavior and capabilities has become a critical research frontier. We propose a framework that extends this principle to a novel class of agents: biological neural networks in the form of neural organoids. This paper introduces three scalable, closed-loop virtual environments designed to train organoid-based biological agents and probe the underlying mechanisms of learning, such as long-term potentiation (LTP) and long-term depression (LTD). We detail the design of three distinct task environments with increasing complexity: (1) a conditional avoidance task, (2) a one-dimensional predator-prey scenario, and (3) a replication of the classic Pong game. For each environment, we formalize the state and action spaces, the sensory encoding and motor decoding mechanisms, and the feedback protocols based on predictable (reward) and unpredictable (punishment) stimulation. Furthermore, we propose a novel meta-learning approach where a Large Language Model (LLM) is used to automate the generation and optimization of experimental protocols, scaling the process of environment and curriculum design. Finally, we outline a multi-modal approach for evaluating learning by measuring synaptic plasticity at electrophysiological, cellular, and molecular levels. This work bridges the gap between computational neuroscience and agent-based AI, offering a unique platform for studying embodiment, learning, and intelligence in a controlled biological substrate. 

**Abstract (ZH)**: 随着人工代理复杂性的增加，设计能够有效塑造其行为和能力的环境已成为一个关键的研究前沿。本文提出了一种框架，将其原则扩展到一类新型代理：神经器官oid形式的生物神经网络。本文介绍了三种可扩展的闭环虚拟环境，旨在训练基于器官oid的生物代理，并探究诸如长时 potentiation（LTP）和长时去 potentiation（LTD）等学习机制的内在机制。我们详细介绍了三种不同复杂度的任务环境设计：（1）条件回避任务；（2）一维捕食者与猎物场景；（3）经典的Pong游戏的复现。对于每个环境，我们形式化了状态空间和动作空间，感知编码和运动解码机制，以及基于可预测（奖励）和不可预测（惩罚）刺激的反馈协议。此外，我们提出了一种新型的元学习方法，其中大型语言模型（LLM）用于自动化实验协议的生成和优化，扩大环境和课程设计的过程。最后，我们概述了一种多模态方法，通过在电生理、细胞和分子水平上测量突触可塑性来评估学习。这项工作在计算神经科学和基于代理的人工智能之间架起了桥梁，提供了一个独特平台，用于在受控的生物基质中研究体认、学习和智能。 

---
# Schema Inference for Tabular Data Repositories Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行表数据仓库的模式推理 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2509.04632)  

**Abstract**: Minimally curated tabular data often contain representational inconsistencies across heterogeneous sources, and are accompanied by sparse metadata. Working with such data is intimidating. While prior work has advanced dataset discovery and exploration, schema inference remains difficult when metadata are limited. We present SI-LLM (Schema Inference using Large Language Models), which infers a concise conceptual schema for tabular data using only column headers and cell values. The inferred schema comprises hierarchical entity types, attributes, and inter-type relationships. In extensive evaluation on two datasets from web tables and open data, SI-LLM achieves promising end-to-end results, as well as better or comparable results to state-of-the-art methods at each step. All source code, full prompts, and datasets of SI-LLM are available at this https URL. 

**Abstract (ZH)**: 最小程度加工的表格数据经常来自异构来源，并且伴随稀疏的元数据，处理这些数据令人望而生畏。尽管先前的工作已经在数据集的发现和探索方面取得了进展，但在元数据有限的情况下，模式推理仍然具有挑战性。我们提出了SI-LLM（基于大型语言模型的模式推理），仅使用列标题和单元格值来推断表格数据的概念模式。推断出的模式包括层次实体类型、属性以及类间关系。在对两个来自网络表格和开放数据集的广泛评估中，SI-LLM 达到了令人鼓舞的整体效果，并且在每个步骤上取得了与最新方法相当甚至更好的结果。SI-LLM 的所有源代码、完整提示和数据集均可在以下链接获取：这个 https URL。 

---
# Action Chunking with Transformers for Image-Based Spacecraft Guidance and Control 

**Title (ZH)**: 基于图像的航天器导航与控制中的动作chunking变换器方法 

**Authors**: Alejandro Posadas-Nava, Andrea Scorsoglio, Luca Ghilardi, Roberto Furfaro, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2509.04628)  

**Abstract**: We present an imitation learning approach for spacecraft guidance, navigation, and control(GNC) that achieves high performance from limited data. Using only 100 expert demonstrations, equivalent to 6,300 environment interactions, our method, which implements Action Chunking with Transformers (ACT), learns a control policy that maps visual and state observations to thrust and torque commands. ACT generates smoother, more consistent trajectories than a meta-reinforcement learning (meta-RL) baseline trained with 40 million interactions. We evaluate ACT on a rendezvous task: in-orbit docking with the International Space Station (ISS). We show that our approach achieves greater accuracy, smoother control, and greater sample efficiency. 

**Abstract (ZH)**: 我们提出了一种基于模仿学习的航天器导航、制导与控制（GNC）方法，该方法能够在有限数据下实现高性能。使用仅100个专家演示（相当于6,300个环境交互），我们的方法——Action Chunking with Transformers (ACT)——学会将视觉和状态观察映射到推力和扭矩命令的控制策略。ACT在环境交互4000万次的元强化学习（meta-RL）基线之上生成更平滑、更一致的轨迹。我们在对接任务上评估了ACT：与国际空间站（ISS）进行在轨对接。我们展示了该方法的更高精度、更平滑的控制和更高的样本效率。 

---
# Measuring the Measures: Discriminative Capacity of Representational Similarity Metrics Across Model Families 

**Title (ZH)**: 测量方法的测量能力：不同模型家族代表相似性指标的判别能力 

**Authors**: Jialin Wu, Shreya Saha, Yiqing Bo, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.04622)  

**Abstract**: Representational similarity metrics are fundamental tools in neuroscience and AI, yet we lack systematic comparisons of their discriminative power across model families. We introduce a quantitative framework to evaluate representational similarity measures based on their ability to separate model families-across architectures (CNNs, Vision Transformers, Swin Transformers, ConvNeXt) and training regimes (supervised vs. self-supervised). Using three complementary separability measures-dprime from signal detection theory, silhouette coefficients and ROC-AUC, we systematically assess the discriminative capacity of commonly used metrics including RSA, linear predictivity, Procrustes, and soft matching. We show that separability systematically increases as metrics impose more stringent alignment constraints. Among mapping-based approaches, soft-matching achieves the highest separability, followed by Procrustes alignment and linear predictivity. Non-fitting methods such as RSA also yield strong separability across families. These results provide the first systematic comparison of similarity metrics through a separability lens, clarifying their relative sensitivity and guiding metric choice for large-scale model and brain comparisons. 

**Abstract (ZH)**: 基于分离能力量化评估表示相似性度量：从模型家族区分能力系统比较看不同类型相似性度量的相对敏感性及其在大规模模型和脑成像比较中的指导作用 

---
# Sample-efficient Integration of New Modalities into Large Language Models 

**Title (ZH)**: 高效整合新型模态的大语言模型 

**Authors**: Osman Batur İnce, André F. T. Martins, Oisin Mac Aodha, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2509.04606)  

**Abstract**: Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models. 

**Abstract (ZH)**: 多模态基础模型可以处理多种模态。然而，由于可能模态的空间是巨大的且随着时间不断演变，从头训练一个模型来涵盖所有模态是不可行的。此外，将一种模态集成到现有的基础模型中目前需要大量的配对数据，这对于低资源模态来说往往是不可用的。在本文中，我们提出了一种样本高效模态集成（SEMI）方法，应用于大型语言模型（LLM）。为此，我们设计了一个超网络，可以在模态特定编码器与LLM之间的一个共享投影器上进行适应，使其能够应对任何模态。该超网络在高资源模态（如文本、语音、音频、视频）上进行训练，并在推理时根据任意外来模态的少量样本生成合适的适配器。为了增加训练模态的多样性，我们通过等距变换人为增加了编码器的数量。我们发现，SEMI 在新的模态（如卫星图像、天文图像、惯性测量和分子）与任意嵌入维度的编码器进行少量样本集成时，显著提升了样本效率。例如，达到与32-shot SEMI 相同的准确性，从头训练投影器需要的数据量是前者的64倍。因此，SEMI 有望扩展基础模型的模态覆盖范围。 

---
# Quantum-Enhanced Multi-Task Learning with Learnable Weighting for Pharmacokinetic and Toxicity Prediction 

**Title (ZH)**: 基于可学习权重的量子增强多任务学习在药代动力学与毒理学预测中的应用 

**Authors**: Han Zhang, Fengji Ma, Jiamin Su, Xinyue Yang, Lei Wang, Wen-Cai Ye, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04601)  

**Abstract**: Prediction for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) plays a crucial role in drug discovery and development, accelerating the screening and optimization of new drugs. Existing methods primarily rely on single-task learning (STL), which often fails to fully exploit the complementarities between tasks. Besides, it requires more computational resources while training and inference of each task independently. To address these issues, we propose a new unified Quantum-enhanced and task-Weighted Multi-Task Learning (QW-MTL) framework, specifically designed for ADMET classification tasks. Built upon the Chemprop-RDKit backbone, QW-MTL adopts quantum chemical descriptors to enrich molecular representations with additional information about the electronic structure and interactions. Meanwhile, it introduces a novel exponential task weighting scheme that combines dataset-scale priors with learnable parameters to achieve dynamic loss balancing across tasks. To the best of our knowledge, this is the first work to systematically conduct joint multi-task training across all 13 Therapeutics Data Commons (TDC) classification benchmarks, using leaderboard-style data splits to ensure a standardized and realistic evaluation setting. Extensive experimental results show that QW-MTL significantly outperforms single-task baselines on 12 out of 13 tasks, achieving high predictive performance with minimal model complexity and fast inference, demonstrating the effectiveness and efficiency of multi-task molecular learning enhanced by quantum-informed features and adaptive task weighting. 

**Abstract (ZH)**: 量子增强和任务加权多任务学习在ADMET分类中的预测研究 

---
# Toward Faithfulness-guided Ensemble Interpretation of Neural Network 

**Title (ZH)**: 面向忠诚度引导的神经网络集成解释 

**Authors**: Siyu Zhang, Kenneth Mcmillan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04588)  

**Abstract**: Interpretable and faithful explanations for specific neural inferences are crucial for understanding and evaluating model behavior. Our work introduces \textbf{F}aithfulness-guided \textbf{E}nsemble \textbf{I}nterpretation (\textbf{FEI}), an innovative framework that enhances the breadth and effectiveness of faithfulness, advancing interpretability by providing superior visualization. Through an analysis of existing evaluation benchmarks, \textbf{FEI} employs a smooth approximation to elevate quantitative faithfulness scores. Diverse variations of \textbf{FEI} target enhanced faithfulness in hidden layer encodings, expanding interpretability. Additionally, we propose a novel qualitative metric that assesses hidden layer faithfulness. In extensive experiments, \textbf{FEI} surpasses existing methods, demonstrating substantial advances in qualitative visualization and quantitative faithfulness scores. Our research establishes a comprehensive framework for elevating faithfulness in neural network explanations, emphasizing both breadth and precision 

**Abstract (ZH)**: Faithfulness-Guided Ensemble Interpretation (FEI): Enhancing Faithfulness and Interpretability in Neural Inferences 

---
# Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions 

**Title (ZH)**: 基于Transformer的模型操控：可控性、引导性和稳健干预 

**Authors**: Faruk Alpay, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2509.04549)  

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models. 

**Abstract (ZH)**: 基于Transformer的语言模型在NLP任务中表现出色，但精细化控制仍然具有挑战性。本文探讨了通过合理干预在三个层面操控Transformer模型的方法：提示、激活和权重。我们将可控文本生成形式化为可通过提示工程、参数高效微调、模型编辑和强化学习解决的优化问题。我们引入了一个统一框架，涵盖提示级引导、激活干预和权重空间编辑。我们分析了鲁棒性和安全性影响，包括对抗性攻击和对齐缓解。理论上，我们证明了最小权重更新可以在有限副作用的情况下实现目标行为变化。实验上，我们在保持基线性能的同时，实现了超过90%的情感控制和事实编辑成功率，尽管存在一般化与特定性之间的权衡。我们讨论了伦理上的双重用途风险以及严格的评估需求。本文为基础设计可控和鲁棒的语言模型奠定了基础。 

---
# i-Mask: An Intelligent Mask for Breath-Driven Activity Recognition 

**Title (ZH)**: i-Mask: 一种基于呼吸驱动活动识别的智能面罩 

**Authors**: Ashutosh Kumar Sinha, Ayush Patel, Mitul Dudhat, Pritam Anand, Rahul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.04544)  

**Abstract**: The patterns of inhalation and exhalation contain important physiological signals that can be used to anticipate human behavior, health trends, and vital parameters. Human activity recognition (HAR) is fundamentally connected to these vital signs, providing deeper insights into well-being and enabling real-time health monitoring. This work presents i-Mask, a novel HAR approach that leverages exhaled breath patterns captured using a custom-developed mask equipped with integrated sensors. Data collected from volunteers wearing the mask undergoes noise filtering, time-series decomposition, and labeling to train predictive models. Our experimental results validate the effectiveness of the approach, achieving over 95\% accuracy and highlighting its potential in healthcare and fitness applications. 

**Abstract (ZH)**: 吸入和呼出的模式包含重要的生理信号，可用于预测人类行为、健康趋势和生命体征。呼吸活动识别（HAR）与这些生命体征密切相关，提供了对福祉的更深入洞察，并能够实现实时健康监测。本研究介绍了i-Mask，这是一种新颖的HAR方法，利用配备集成传感器的自开发口罩捕获的呼出呼吸模式。佩戴口罩的志愿者收集的数据经过噪声过滤、时间序列分解和标签处理以训练预测模型。我们的实验结果验证了该方法的有效性，准确率超过95%，并在医疗保健和健身应用中展示了其潜力。 

---
# Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem 

**Title (ZH)**: Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem（ emergent 社交动态中的大型语言模型代理在 El Farol 酒吧问题中的表现） 

**Authors**: Ryosuke Takata, Atsushi Masumori, Takashi Ikegammi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04537)  

**Abstract**: We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents. 

**Abstract (ZH)**: 我们研究了大型语言模型（LLM）代理在扩展的空间El Farol酒吧问题中的 emergent 社会动态，观察它们如何自主应对这一经典社会困境。结果，LLM 代理自发产生了去酒吧的动力，并通过集体行为改变了决策方式。我们还发现，LLM 代理并未完全解决问题，而是表现得更像人类。这些发现揭示了外部激励（如提示指定的约束条件，如60%阈值）与内部激励（来自预训练的文化编码社会偏好）之间的复杂相互作用，表明LLM代理自然地在形式博弈论理性与反映人类行为特征的社会动机之间取得平衡。这些发现表明，通过LLM代理可以实现一种新的群体决策模式，这种模式在先前的博弈论问题设置中是无法处理的。 

---
# In-Context Policy Adaptation via Cross-Domain Skill Diffusion 

**Title (ZH)**: 基于跨域技能扩散的上下文适配策略调整 

**Authors**: Minjong Yoo, Woo Kyung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04535)  

**Abstract**: In this work, we present an in-context policy adaptation (ICPAD) framework designed for long-horizon multi-task environments, exploring diffusion-based skill learning techniques in cross-domain settings. The framework enables rapid adaptation of skill-based reinforcement learning policies to diverse target domains, especially under stringent constraints on no model updates and only limited target domain data. Specifically, the framework employs a cross-domain skill diffusion scheme, where domain-agnostic prototype skills and a domain-grounded skill adapter are learned jointly and effectively from an offline dataset through cross-domain consistent diffusion processes. The prototype skills act as primitives for common behavior representations of long-horizon policies, serving as a lingua franca to bridge different domains. Furthermore, to enhance the in-context adaptation performance, we develop a dynamic domain prompting scheme that guides the diffusion-based skill adapter toward better alignment with the target domain. Through experiments with robotic manipulation in Metaworld and autonomous driving in CARLA, we show that our $\oursol$ framework achieves superior policy adaptation performance under limited target domain data conditions for various cross-domain configurations including differences in environment dynamics, agent embodiment, and task horizon. 

**Abstract (ZH)**: 基于上下文的多任务环境长时域策略适应框架：跨域扩散技能学习方法 

---
# Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation 

**Title (ZH)**: 生物医学自然语言处理中量化大语言模型：评估与建议 

**Authors**: Zaifu Zhan, Shuang Zhou, Min Zeng, Kai Yu, Meijia Song, Xiaoyi Chen, Jun Wang, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04534)  

**Abstract**: Large language models have demonstrated remarkable capabilities in biomedical natural language processing, yet their rapid growth in size and computational requirements present a major barrier to adoption in healthcare settings where data privacy precludes cloud deployment and resources are limited. In this study, we systematically evaluated the impact of quantization on 12 state-of-the-art large language models, including both general-purpose and biomedical-specific models, across eight benchmark datasets covering four key tasks: named entity recognition, relation extraction, multi-label classification, and question answering. We show that quantization substantially reduces GPU memory requirements-by up to 75%-while preserving model performance across diverse tasks, enabling the deployment of 70B-parameter models on 40GB consumer-grade GPUs. In addition, domain-specific knowledge and responsiveness to advanced prompting methods are largely maintained. These findings provide significant practical and guiding value, highlighting quantization as a practical and effective strategy for enabling the secure, local deployment of large yet high-capacity language models in biomedical contexts, bridging the gap between technical advances in AI and real-world clinical translation. 

**Abstract (ZH)**: 大型语言模型在生物医学自然语言处理方面展示了 remarkable 的能力，但其快速增长的规模和计算需求构成了在数据隐私禁止云部署且资源有限的医疗保健环境中采用的主要障碍。在本研究中，我们系统评估了量化对 12 个最先进的大型语言模型（包括通用和生物医学特定模型）在八个基准数据集上的影响，这些数据集涵盖了四个关键任务：命名实体识别、关系抽取、多标签分类和问答。结果显示，量化可显著减少 GPU 内存需求（最多减少 75%），同时在多种任务中保持模型性能，使 700 亿参数模型能够部署在 40GB 的消费级 GPU 上。此外，专业知识领域以及对高级提示方法的响应保持良好。这些发现提供了重要的实用和指导价值，强调量化作为一种实用且有效的方法，可以实现安全的本地部署大型且高性能语言模型，从而弥合人工智能技术进步与实际临床转化之间的差距。 

---
# Mitigation of Gender and Ethnicity Bias in AI-Generated Stories through Model Explanations 

**Title (ZH)**: 通过模型解释减轻AI生成故事中的性别和 Ethnicity 偏见 

**Authors**: Martha O. Dimgba, Sharon Oba, Ameeta Agrawal, Philippe J. Giabbanelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.04515)  

**Abstract**: Language models have been shown to propagate social bias through their output, particularly in the representation of gender and ethnicity. This paper investigates gender and ethnicity biases in AI-generated occupational stories. Representation biases are measured before and after applying our proposed mitigation strategy, Bias Analysis and Mitigation through Explanation (BAME), revealing improvements in demographic representation ranging from 2% to 20%. BAME leverages model-generated explanations to inform targeted prompt engineering, effectively reducing biases without modifying model parameters. By analyzing stories generated across 25 occupational groups, three large language models (Claude 3.5 Sonnet, Llama 3.1 70B Instruct, and GPT-4 Turbo), and multiple demographic dimensions, we identify persistent patterns of overrepresentation and underrepresentation linked to training data stereotypes. Our findings demonstrate that guiding models with their own internal reasoning mechanisms can significantly enhance demographic parity, thereby contributing to the development of more transparent generative AI systems. 

**Abstract (ZH)**: 语言模型在其输出中表现出传播社会偏见的现象，特别是在性别和族裔的表示上。本文探讨了AI生成的职业故事中的性别和族裔偏见。我们通过应用提出的缓解策略“解释驱动的偏见分析与缓解”（BAME）来衡量表示偏见的变化，发现人口统计学表示的改进范围从2%到20%。BAME 利用模型生成的解释来指导目标提示工程，有效减少了偏见而不修改模型参数。通过分析25个职业群体生成的故事，以及三种大型语言模型（Claude 3.5 Sonnet、Llama 3.1 70B Instruct 和 GPT-4 Turbo）和多个人口统计学维度，我们识别出与训练数据刻板印象相关的持续存在的过度代表和不足代表模式。我们的研究结果表明，通过引导模型使用它们自己的内部推理机制，可以显著增强人口统计学平权，从而促进更透明的生成AI系统的开发。 

---
# From Silent Signals to Natural Language: A Dual-Stage Transformer-LLM Approach 

**Title (ZH)**: 从沉默信号到自然语言：双阶段变压器-大语言模型方法 

**Authors**: Nithyashree Sivasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04507)  

**Abstract**: Silent Speech Interfaces (SSIs) have gained attention for their ability to generate intelligible speech from non-acoustic signals. While significant progress has been made in advancing speech generation pipelines, limited work has addressed the recognition and downstream processing of synthesized speech, which often suffers from phonetic ambiguity and noise. To overcome these challenges, we propose an enhanced automatic speech recognition framework that combines a transformer-based acoustic model with a large language model (LLM) for post-processing. The transformer captures full utterance context, while the LLM ensures linguistic consistency. Experimental results show a 16% relative and 6% absolute reduction in word error rate (WER) over a 36% baseline, demonstrating substantial improvements in intelligibility for silent speech interfaces. 

**Abstract (ZH)**: 静默语音接口（SSIs）因其能够从非声学信号生成可理解 speech 而引起了关注。虽然在推进语音生成流水线方面取得了显著进展，但合成 speech 的识别和后处理工作相对较少，且常常存在音素模糊和噪声问题。为了克服这些挑战，我们提出了一种增强的自动语音识别框架，该框架结合了基于变换器的声音模型和大型语言模型（LLM）进行后处理。变换器捕捉整个言语单元的上下文，而 LLM 确保语言一致性。实验结果表明，与基准相比，词错误率（WER）相对下降了 16%，绝对下降了 6%，展示了静默语音接口在可理解性方面的显著改进。 

---
# Memristor-Based Neural Network Accelerators for Space Applications: Enhancing Performance with Temporal Averaging and SIRENs 

**Title (ZH)**: 基于 memristor 的神经网络加速器在空间应用中的增强：通过时间平均和 SIRENs 提升性能 

**Authors**: Zacharia A. Rudge, Dominik Dold, Moritz Fieback, Dario Izzo, Said Hamdioui  

**Link**: [PDF](https://arxiv.org/pdf/2509.04506)  

**Abstract**: Memristors are an emerging technology that enables artificial intelligence (AI) accelerators with high energy efficiency and radiation robustness -- properties that are vital for the deployment of AI on-board spacecraft. However, space applications require reliable and precise computations, while memristive devices suffer from non-idealities, such as device variability, conductance drifts, and device faults. Thus, porting neural networks (NNs) to memristive devices often faces the challenge of severe performance degradation. In this work, we show in simulations that memristor-based NNs achieve competitive performance levels on on-board tasks, such as navigation \& control and geodesy of asteroids. Through bit-slicing, temporal averaging of NN layers, and periodic activation functions, we improve initial results from around $0.07$ to $0.01$ and $0.3$ to $0.007$ for both tasks using RRAM devices, coming close to state-of-the-art levels ($0.003-0.005$ and $0.003$, respectively). Our results demonstrate the potential of memristors for on-board space applications, and we are convinced that future technology and NN improvements will further close the performance gap to fully unlock the benefits of memristors. 

**Abstract (ZH)**: 基于 memristor 的神经网络在星载任务中的竞争性能研究 

---
# Behavioral Fingerprinting of Large Language Models 

**Title (ZH)**: 大型语言模型的行为指纹识别 

**Authors**: Zehua Pei, Hui-Ling Zhen, Ying Zhang, Zhiyuan Yang, Xing Li, Xianzhi Yu, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04504)  

**Abstract**: Current benchmarks for Large Language Models (LLMs) primarily focus on performance metrics, often failing to capture the nuanced behavioral characteristics that differentiate them. This paper introduces a novel ``Behavioral Fingerprinting'' framework designed to move beyond traditional evaluation by creating a multi-faceted profile of a model's intrinsic cognitive and interactive styles. Using a curated \textit{Diagnostic Prompt Suite} and an innovative, automated evaluation pipeline where a powerful LLM acts as an impartial judge, we analyze eighteen models across capability tiers. Our results reveal a critical divergence in the LLM landscape: while core capabilities like abstract and causal reasoning are converging among top models, alignment-related behaviors such as sycophancy and semantic robustness vary dramatically. We further document a cross-model default persona clustering (ISTJ/ESTJ) that likely reflects common alignment incentives. Taken together, this suggests that a model's interactive nature is not an emergent property of its scale or reasoning power, but a direct consequence of specific, and highly variable, developer alignment strategies. Our framework provides a reproducible and scalable methodology for uncovering these deep behavioral differences. Project: this https URL 

**Abstract (ZH)**: 当前的大语言模型基准主要集中在性能指标上，往往无法捕捉到区分它们的细腻行为特征。本文引入了一种新颖的“行为指纹”框架，旨在超越传统的评估方法，创建一个多维度的模型内在认知和交互风格的综合画像。通过使用精心挑选的\textit{诊断提示集}和一个创新的自动化评估管道，其中强大的大语言模型作为公正的评判者，我们分析了十八个不同能力级别的模型。我们的结果显示，大语言模型领域存在关键差异：尽管抽象和因果推理等核心能力在顶级模型间趋于一致，但与对齐相关的行为，如阿谀奉承和语义稳健性则差异巨大。我们进一步记录了跨模型默认人格集群（ISTJ/ESTJ），这很可能反映了共同的对齐激励。总的来说，这表明模型的交互性质不是其规模或推理能力的次生属性，而是特定且高度可变的开发者对齐策略的直接结果。我们的框架提供了一种可重复和可扩展的方法来揭示这些深层行为差异。项目：[这个链接](this https URL)。 

---
# VaccineRAG: Boosting Multimodal Large Language Models' Immunity to Harmful RAG Samples 

**Title (ZH)**: VaccineRAG: 提升多模态大规模语言模型对抗有害RAG样本的能力 

**Authors**: Qixin Sun, Ziqin Wang, Hengyuan Zhao, Yilin Li, Kaiyou Song, Linjiang Huang, Xiaolin Hu, Qingpei Guo, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04502)  

**Abstract**: Retrieval Augmented Generation enhances the response accuracy of Large Language Models (LLMs) by integrating retrieval and generation modules with external knowledge, demonstrating particular strength in real-time queries and Visual Question Answering tasks. However, the effectiveness of RAG is frequently hindered by the precision of the retriever: many retrieved samples fed into the generation phase are irrelevant or misleading, posing a critical bottleneck to LLMs' performance. To address this challenge, we introduce VaccineRAG, a novel Chain-of-Thought-based retrieval-augmented generation dataset. On one hand, VaccineRAG employs a benchmark to evaluate models using data with varying positive/negative sample ratios, systematically exposing inherent weaknesses in current LLMs. On the other hand, it enhances models' sample-discrimination capabilities by prompting LLMs to generate explicit Chain-of-Thought (CoT) analysis for each sample before producing final answers. Furthermore, to enhance the model's ability to learn long-sequence complex CoT content, we propose Partial-GRPO. By modeling the outputs of LLMs as multiple components rather than a single whole, our model can make more informed preference selections for complex sequences, thereby enhancing its capacity to learn complex CoT. Comprehensive evaluations and ablation studies on VaccineRAG validate the effectiveness of the proposed scheme. The code and dataset will be publicly released soon. 

**Abstract (ZH)**: 检索增强生成提高了大型语言模型的响应准确性，通过将检索和生成模块与外部知识集成，特别在实时查询和视觉问答任务中表现出色。然而，RAG的有效性经常受到检索精度的限制：许多提供给生成阶段的检索样本是不相关或误导性的，成为大型语言模型性能的关键瓶颈。为解决这一挑战，我们引入了VaccineRAG，一种基于Chain-of-Thought的检索增强生成数据集。一方面，VaccineRAG使用基准来评估模型，使用不同正负样本比例的数据，系统地揭示当前大型语言模型固有的弱点。另一方面，它通过促使大型语言模型为每个样本生成明确的Chain-of-Thought分析，增强模型对样本的区分能力，在生成最终答案之前。此外，为了增强模型学习长序列复杂Chain-of-Thought内容的能力，我们提出了一种Partial-GRPO方法。通过将大型语言模型的输出建模为多个组件而不是单一的整体，我们的模型可以对复杂的序列做出更明智的偏好选择，从而增强其学习复杂Chain-of-Thought的能力。在VaccineRAG上的全面评估和消融研究验证了所提出方案的有效性。代码和数据集将在不久的将来公开发布。 

---
# Understanding Reinforcement Learning for Model Training, and future directions with GRAPE 

**Title (ZH)**: 理解强化学习在模型训练中的应用及GRAPE的未来方向 

**Authors**: Rohit Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.04501)  

**Abstract**: This paper provides a self-contained, from-scratch, exposition of key algorithms for instruction tuning of models: SFT, Rejection Sampling, REINFORCE, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Explanations of these algorithms often assume prior knowledge, lack critical details, and/or are overly generalized and complex. Here, each method is discussed and developed step by step using simplified and explicit notation focused on LLMs, aiming to eliminate ambiguity and provide a clear and intuitive understanding of the concepts. By minimizing detours into the broader RL literature and connecting concepts to LLMs, we eliminate superfluous abstractions and reduce cognitive overhead. Following this exposition, we provide a literature review of new techniques and approaches beyond those detailed. Finally, new ideas for research and exploration in the form of GRAPE (Generalized Relative Advantage Policy Evolution) are presented. 

**Abstract (ZH)**: 这篇文章提供了从零开始、自包含的关键指令调优算法 exposition：SFT、拒绝采样、REINFORCE、信任区域策略优化（TRPO）、近端策略优化（PPO）、组相对策略优化（GRPO）和直接偏好优化（DPO）。这些算法的解释通常假设读者预先具备相关知识、缺乏关键细节、或过于泛化和复杂。在此，我们逐步用简化且明确的符号重点讨论和发展这些方法，旨在消除歧义、提供清晰直观的概念理解。通过减少对更广泛RL文献的旁枝末节，并将概念与大语言模型（LLMs）相连，我们消除了不必要的抽象，降低认知负担。在此 exposition 之后，我们提供了一篇关于超出详述的新技术和方法的文献综述。最后，我们提出了新的研究和探索想法——广义相对优势策略进化（GRAPE）。 

---
# Context Engineering for Trustworthiness: Rescorla Wagner Steering Under Mixed and Inappropriate Contexts 

**Title (ZH)**: 基于上下文工程的信任worthy性调节：混合与不当上下文下的雷斯科拉-瓦格纳模型 

**Authors**: Rushi Wang, Jiateng Liu, Cheng Qian, Yifan Shen, Yanzhou Pan, Zhaozhuo Xu, Ahmed Abbasi, Heng Ji, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04500)  

**Abstract**: Incorporating external context can significantly enhance the response quality of Large Language Models (LLMs). However, real-world contexts often mix relevant information with disproportionate inappropriate content, posing reliability risks. How do LLMs process and prioritize mixed context? To study this, we introduce the Poisoned Context Testbed, pairing queries with real-world contexts containing relevant and inappropriate content. Inspired by associative learning in animals, we adapt the Rescorla-Wagner (RW) model from neuroscience to quantify how competing contextual signals influence LLM outputs. Our adapted model reveals a consistent behavioral pattern: LLMs exhibit a strong tendency to incorporate information that is less prevalent in the context. This susceptibility is harmful in real-world settings, where small amounts of inappropriate content can substantially degrade response quality. Empirical evaluations on our testbed further confirm this vulnerability. To tackle this, we introduce RW-Steering, a two-stage finetuning-based approach that enables the model to internally identify and ignore inappropriate signals. Unlike prior methods that rely on extensive supervision across diverse context mixtures, RW-Steering generalizes robustly across varying proportions of inappropriate content. Experiments show that our best fine-tuned model improves response quality by 39.8% and reverses the undesirable behavior curve, establishing RW-Steering as a robust, generalizable context engineering solution for improving LLM safety in real-world use. 

**Abstract (ZH)**: 将外部上下文整合到大型语言模型中可以显著提高响应质量，但现实世界的上下文往往会混入相关和不适当的内容，存在可靠性风险。大型语言模型是如何处理和优先处理这些混合上下文的？为研究这一问题，我们引入了中毒上下文测试平台，将查询与包含相关和不适当内容的现实世界上下文配对。受到动物联想学习的启发，我们将神经科学领域的雷斯科拉-瓦格纳（RW）模型改编为衡量竞争性上下文信号如何影响大型语言模型输出的量度。改编后的模型揭示了一种一致的行为模式：大型语言模型倾向于优先纳入上下文中较少出现的信息。这种倾向性在现实世界环境中是不利的，少量不适当内容的存在可能导致响应质量显著下降。在测试平台上的实证评估进一步证实了这种脆弱性。为应对这一问题，我们提出了基于细调的两阶段方法RW-Steering，使模型能够内部识别并忽略不适当信号。不同于依赖广泛监督的先前方法，RW-Steering能够在不同比例的不适当内容下稳健推广。实验结果表明，我们最好的微调模型将响应质量提高了39.8%，并逆转了不希望看到的行为曲线，确立了RW-Steering作为提高大型语言模型在实际使用中安全性的一种稳健且通用的上下文工程解决方案。 

---
# DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence 

**Title (ZH)**: DeepTRACE: 审核深层研究AI系统以追踪引文和证据中的可靠性 

**Authors**: Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Kung-Hsiang Huang, Yixin Mao, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04499)  

**Abstract**: Generative search engines and deep research LLM agents promise trustworthy, source-grounded synthesis, yet users regularly encounter overconfidence, weak sourcing, and confusing citation practices. We introduce DeepTRACE, a novel sociotechnically grounded audit framework that turns prior community-identified failure cases into eight measurable dimensions spanning answer text, sources, and citations. DeepTRACE uses statement-level analysis (decomposition, confidence scoring) and builds citation and factual-support matrices to audit how systems reason with and attribute evidence end-to-end. Using automated extraction pipelines for popular public models (e.g., GPT-4.5/5, this http URL, Perplexity, Copilot/Bing, Gemini) and an LLM-judge with validated agreement to human raters, we evaluate both web-search engines and deep-research configurations. Our findings show that generative search engines and deep research agents frequently produce one-sided, highly confident responses on debate queries and include large fractions of statements unsupported by their own listed sources. Deep-research configurations reduce overconfidence and can attain high citation thoroughness, but they remain highly one-sided on debate queries and still exhibit large fractions of unsupported statements, with citation accuracy ranging from 40--80% across systems. 

**Abstract (ZH)**: 生成式搜索引擎和深度研究大语言模型承诺实现可靠且来源依存的合成，但用户常遇到过度自信、来源薄弱和混乱的引用实践。我们提出了DeepTRACE，一个新颖的社会技术导向的审计框架，将以前社区识别的失败案例转化为涵盖答案文本、来源和引用的八项可测量维度。DeepTRACE 使用陈述级分析（分解、置信度评分）并构建引用和事实支持矩阵，以端到端审计系统如何处理和归因证据。通过使用流行的开源模型（如GPT-4.5/5、该网址、非确定性、Copilot/Bing、Gemini）的自动提取管道和一个具有验证一致性的LLM评判员与人类评判员，我们评估了网络搜索引擎和深度研究配置。研究发现，生成式搜索引擎和深度研究代理经常在论辩查询上产生片面、高度自信的回应，并且包括大量未被自身列出来源支持的陈述。深度研究配置降低了过度自信，可以在一定程度上达到高度详尽的引用，但在论辩查询上仍然保持了极大的片面性，并且仍包含大量未被支持的陈述，引用准确性范围在40%-80%。 

---
# Where Should I Study? Biased Language Models Decide! Evaluating Fairness in LMs for Academic Recommendations 

**Title (ZH)**: 我在哪里学习？有偏见的语言模型来决定！评价学术推荐中语言模型的公平性 

**Authors**: Krithi Shailya, Akhilesh Kumar Mishra, Gokul S Krishnan, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.04498)  

**Abstract**: Large Language Models (LLMs) are increasingly used as daily recommendation systems for tasks like education planning, yet their recommendations risk perpetuating societal biases. This paper empirically examines geographic, demographic, and economic biases in university and program suggestions from three open-source LLMs: LLaMA-3.1-8B, Gemma-7B, and Mistral-7B. Using 360 simulated user profiles varying by gender, nationality, and economic status, we analyze over 25,000 recommendations. Results show strong biases: institutions in the Global North are disproportionately favored, recommendations often reinforce gender stereotypes, and institutional repetition is prevalent. While LLaMA-3.1 achieves the highest diversity, recommending 481 unique universities across 58 countries, systemic disparities persist. To quantify these issues, we propose a novel, multi-dimensional evaluation framework that goes beyond accuracy by measuring demographic and geographic representation. Our findings highlight the urgent need for bias consideration in educational LMs to ensure equitable global access to higher education. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育规划等日常推荐任务中的应用日益增多，但其推荐结果存在延续社会偏见的风险。本文实证分析了三个开源LLM——LLaMA-3.1-8B、Gemma-7B和Mistral-7B，在大学和项目建议中所体现的地理、人口统计和经济偏见。通过使用360个模拟用户配置文件，这些配置文件在性别、国籍和经济状况方面有所不同，我们分析了超过25000条推荐结果。研究结果显示存在明显的偏见：全球北方的机构被不公正地偏好，推荐结果常常强化性别刻板印象，且机构重复出现。尽管LLaMA-3.1表现出最高的多样性，推荐了来自58个国家的481所独特大学，系统性不平等仍然存在。为量化这些问题，我们提出了一个新颖的多维度评估框架，不仅衡量准确性，还衡量人口统计和地理代表性。研究结果强调了在教育LLM中考虑偏见的紧迫性，以确保全球范围内接受高等教育的机会公平。 

---
# A Narrative-Driven Computational Framework for Clinician Burnout Surveillance 

**Title (ZH)**: 基于叙事驱动的计算框架在医务人员倦怠监控中的应用 

**Authors**: Syed Ahmad Chan Bukhari, Fazel Keshtkar, Alyssa Meczkowska  

**Link**: [PDF](https://arxiv.org/pdf/2509.04497)  

**Abstract**: Clinician burnout poses a substantial threat to patient safety, particularly in high-acuity intensive care units (ICUs). Existing research predominantly relies on retrospective survey tools or broad electronic health record (EHR) metadata, often overlooking the valuable narrative information embedded in clinical notes. In this study, we analyze 10,000 ICU discharge summaries from MIMIC-IV, a publicly available database derived from the electronic health records of Beth Israel Deaconess Medical Center. The dataset encompasses diverse patient data, including vital signs, medical orders, diagnoses, procedures, treatments, and deidentified free-text clinical notes. We introduce a hybrid pipeline that combines BioBERT sentiment embeddings fine-tuned for clinical narratives, a lexical stress lexicon tailored for clinician burnout surveillance, and five-topic latent Dirichlet allocation (LDA) with workload proxies. A provider-level logistic regression classifier achieves a precision of 0.80, a recall of 0.89, and an F1 score of 0.84 on a stratified hold-out set, surpassing metadata-only baselines by greater than or equal to 0.17 F1 score. Specialty-specific analysis indicates elevated burnout risk among providers in Radiology, Psychiatry, and Neurology. Our findings demonstrate that ICU clinical narratives contain actionable signals for proactive well-being monitoring. 

**Abstract (ZH)**: 临床医生的 Burnout 对患者安全构成重大威胁，尤其是在高急性重症监护病房（ICUs）。现有研究主要依赖于回顾性调查工具或广泛电子健康记录（EHR）元数据，往往忽视了临床笔记中嵌入的宝贵叙述信息。在本研究中，我们分析了来自 Beth Israel Deaconess Medical Center 电子健康记录的公开数据库 MIMIC-IV 的 10,000 份 ICU 出院总结。该数据集涵盖了多种患者数据，包括生命体征、医疗指令、诊断、程序、治疗和去标识化的自由文本临床笔记。我们引入了一种结合 BioBERT 情感嵌入（针对临床叙述微调）、专为临床医生 Burnout 监控量身定制的词级压力词典，以及带有工作负荷代理的五主题潜在狄利克雷分配（LDA）的混合管道。在分层保留集中，提供者级别的逻辑回归分类器达到了 0.80 的精确度、0.89 的召回率和 0.84 的 F1 分数，超过了仅使用元数据的基线模型大于或等于 0.17 的 F1 分数。专科分析表明，放射科、精神病学和神经学专科提供者的 Burnout 风险更高。我们的研究结果表明，ICU 临床叙述包含可用于主动健康监测的有效信号。 

---
# Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate 

**Title (ZH)**: 基于token级熵生产率的黑盒大模型中Learned幻觉检测 

**Authors**: Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, Emmanuel Malherbe  

**Link**: [PDF](https://arxiv.org/pdf/2509.04492)  

**Abstract**: Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset. 

**Abstract (ZH)**: 大型语言模型(LLM)在问答(QA)任务输出中的幻觉严重削弱了其现实世界中的可靠性。本文介绍了一种应用于有限数据访问场景下的稳健单-shot幻觉检测方法，特别适用于只能访问少量顶级候选择概率的黑盒LLM API。本方法直接从非贪婪解码过程中生成的候选择概率中提取不确定性指标。首先，我们提出了一个熵生成率(EPR)指标以提供基线性能，该指标随后通过监督学习进行增强。学习到的模型使用表示单个生成序列中可访问的顶级候选项的熵贡献特征，无需多次查询重跑。在多种QA数据集和多个LLM上进行评估，该估算器显著改善了仅使用EPR进行幻觉检测的性能。至关重要的是，该方法使用通常较小的可用候选择概率集（例如，每个词的前10个）即可实现高性能，证实了其在这些API受限部署中的实用性和高效性。本文为增强问答和检索增强生成系统中LLM响应的信任度提供了可部署的技术，并通过一个分析工业数据集中年度报告查询响应的金融框架进一步验证了其实用性。 

---
# Refining Transcripts With TV Subtitles by Prompt-Based Weakly Supervised Training of ASR 

**Title (ZH)**: 基于提示的弱监督训练改进ASR转录基于电视字幕 

**Authors**: Xinnian Zhao, Hugo Van Hamme  

**Link**: [PDF](https://arxiv.org/pdf/2509.04491)  

**Abstract**: This study proposes a novel approach to using TV subtitles within a weakly supervised (WS) Automatic Speech Recognition (ASR) framework. Although TV subtitles are readily available, their imprecise alignment with corresponding audio limits their applicability as supervised targets for verbatim transcription. Rather than using subtitles as direct supervision signals, our method reimagines them as context-rich prompts. This design enables the model to handle discrepancies between spoken audio and subtitle text. Instead, generated pseudo transcripts become the primary targets, with subtitles acting as guiding cues for iterative refinement. To further enhance the process, we introduce a weighted attention mechanism that emphasizes relevant subtitle tokens during inference. Our experiments demonstrate significant improvements in transcription accuracy, highlighting the effectiveness of the proposed method in refining transcripts. These enhanced pseudo-labeled datasets provide high-quality foundational resources for training robust ASR systems. 

**Abstract (ZH)**: 本研究提出了一种在弱监督（WS）自动语音识别（ASR）框架中使用电视字幕的新型方法。尽管电视字幕易于获取，但它们与对应音频的不精确对齐限制了其作为逐字转录监督目标的应用。本方法并非直接使用字幕作为监督信号，而是将它们重新想象为富含上下文的提示。这种设计使模型能够处理口语音频与字幕文本之间的差异。相反，生成的伪转录成为主要目标，而字幕则作为迭代细化的引导线索。为进一步增强此过程，我们引入了一种加权注意机制，在推断过程中强调相关的字幕词元。实验结果显示出显著的转录准确性提升，突显了所提出方法在转录细化中的有效性。这些增强的伪标注数据集提供了训练健壯ASR系统的高质量基础资源。 

---
# Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition 

**Title (ZH)**: 基于大型语言模型的多说话人语音识别的序列输出提示方法 

**Authors**: Hao Shi, Yusuke Fujita, Tomoya Mizumoto, Lianbo Liu, Atsushi Kojima, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04488)  

**Abstract**: Prompts are crucial for task definition and for improving the performance of large language models (LLM)-based systems. However, existing LLM-based multi-talker (MT) automatic speech recognition (ASR) systems either omit prompts or rely on simple task-definition prompts, with no prior work exploring the design of prompts to enhance performance. In this paper, we propose extracting serialized output prompts (SOP) and explicitly guiding the LLM using structured prompts to improve system performance (SOP-MT-ASR). A Separator and serialized Connectionist Temporal Classification (CTC) layers are inserted after the speech encoder to separate and extract MT content from the mixed speech encoding in a first-speaking-first-out manner. Subsequently, the SOP, which serves as a prompt for LLMs, is obtained by decoding the serialized CTC outputs using greedy search. To train the model effectively, we design a three-stage training strategy, consisting of serialized output training (SOT) fine-tuning, serialized speech information extraction, and SOP-based adaptation. Experimental results on the LibriMix dataset show that, although the LLM-based SOT model performs well in the two-talker scenario, it fails to fully leverage LLMs under more complex conditions, such as the three-talker scenario. The proposed SOP approach significantly improved performance under both two- and three-talker conditions. 

**Abstract (ZH)**: 提示词对于任务定义和提高基于大型语言模型（LLM）的多说话人（MT）自动语音识别（ASR）系统性能至关重要。然而，现有的基于LLM的MT ASR系统要么省略提示词，要么依赖于简单的任务定义提示词，没有任何工作探索设计提示词以提高性能的方法。在本文中，我们提出提取序列化输出提示词（SOP）并通过结构化提示词明确引导LLM以提高系统性能（SOP-MT-ASR）。在语音编码器之后插入分隔符和序列化连接主义时序分类（CTC）层，以按先说先出的方式分离和提取混音语音编码中的MT内容。随后，通过贪婪搜索解码序列化CTC输出获得作为LLM提示词的SOP。为了有效训练模型，我们设计了三阶段训练策略，包括序列化输出训练（SOT）微调、序列化语音信息提取和基于SOP的适应。在LibriMix数据集上的实验结果表明，尽管基于LLM的SOT模型在双说话人场景中表现良好，但在更复杂的场景如三说话人场景中未能充分利用LLM。所提出的方法在双说话人和三说话人场景下均显著提高了性能。 

---
# ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records 

**Title (ZH)**: ASCEND-GPT：一种基于表型的Transformer模型，用于电子健康记录的心血管风险预测 

**Authors**: Chris Sainsbury, Andreas Karwath  

**Link**: [PDF](https://arxiv.org/pdf/2509.04485)  

**Abstract**: We present ASCENDgpt, a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens, achieving 99.6\% consolidation of diagnosis codes while preserving semantic information. This phenotype mapping contributes to a total vocabulary of 10,442 tokens - a 77.9\% reduction when compared with using raw ICD codes directly. We pretrain ASCENDgpt on sequences derived from 19402 unique individuals using a masked language modeling objective, then fine-tune for time-to-event prediction of five cardiovascular outcomes: myocardial infarction (MI), stroke, major adverse cardiovascular events (MACE), cardiovascular death, and all-cause mortality. Our model achieves excellent discrimination on the held-out test set with an average C-index of 0.816, demonstrating strong performance across all outcomes (MI: 0.792, stroke: 0.824, MACE: 0.800, cardiovascular death: 0.842, all-cause mortality: 0.824). The phenotype-based approach enables clinically interpretable predictions while maintaining computational efficiency. Our work demonstrates the effectiveness of domain-specific tokenization and pretraining for EHR-based risk prediction tasks. 

**Abstract (ZH)**: ASCENDgpt：一种基于变压器的心血管风险预测模型 

---
# The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors 

**Title (ZH)**: 好的、坏的和建设性的：自动衡量同行评审对作者的价值 

**Authors**: Abdelrahman Sadallah, Tim Baumgärtner, Iryna Gurevych, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2509.04484)  

**Abstract**: Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects. 

**Abstract (ZH)**: 提供建设性的反馈是同行评审的核心组成部分。随着评审者可用时间的减少，需要自动化支持系统来确保评审质量，从而使评审中的反馈对作者有用。为此，我们识别了四方面评审意见的关键要素，这些要素驱动作者的有用性：可操作性、依据与具体性、可验证性以及有用性。为了评估和开发评估评审意见的模型，我们引入了RevUtil数据集。我们收集了1,430个人工标注的评审意见，并通过10,000个合成标注的意见来扩展数据，用于训练目的。合成数据还包含了理由，即评审意见评分的解释。使用RevUtil数据集，我们对评估评审意见和生成理由的微调模型进行了基准测试。我们的实验表明，这些微调模型在某些方面与强大的封闭模型（如GPT-4o）相比，能够达到与人类相当的共识水平，甚至在某些情况下超越了它们。进一步的分析表明，机器生成的评审意见在我们定义的四个方面通常表现不如人工生成的评审意见。 

---
# DecMetrics: Structured Claim Decomposition Scoring for Factually Consistent LLM Outputs 

**Title (ZH)**: DecMetrics: 结构化断言分解评分以获得事实一致的LLM输出 

**Authors**: Minghui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04483)  

**Abstract**: Claim decomposition plays a crucial role in the fact-checking process by breaking down complex claims into simpler atomic components and identifying their unfactual elements. Despite its importance, current research primarily focuses on generative methods for decomposition, with insufficient emphasis on evaluating the quality of these decomposed atomic claims. To bridge this gap, we introduce \textbf{DecMetrics}, which comprises three new metrics: \texttt{COMPLETENESS}, \texttt{CORRECTNESS}, and \texttt{SEMANTIC ENTROPY}, designed to automatically assess the quality of claims produced by decomposition models. Utilizing these metrics, we develop a lightweight claim decomposition model, optimizing its performance through the integration of these metrics as a reward function. Through automatic evaluation, our approach aims to set a benchmark for claim decomposition, enhancing both the reliability and effectiveness of fact-checking systems. 

**Abstract (ZH)**: 命题分解在事实核查过程中发挥着关键作用，通过将复杂的命题分解为更简单的原子组件并识别其不实元素。尽管具有重要意义，当前研究主要集中在分解生成方法上，对于评估分解后原子命题的质量关注不足。为弥补这一缺口，我们引入了\textbf{DecMetrics}，包含三个新的评估指标：\texttt{COMPLETENESS}、\texttt{CORRECTNESS} 和 \texttt{SEMANTIC ENTROPY}，旨在自动评估由分解模型生成的命题质量。利用这些指标，我们开发了一个轻量级的命题分解模型，并通过将这些指标整合为奖励函数来优化其性能。通过自动评估，我们的方法旨在为命题分解设立基准，提高事实核查系统的可靠性和有效性。 

---
# Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare 

**Title (ZH)**: 能量景观使检索增强型大型语言模型在医疗保健领域可靠地 abstain 成为可能 

**Authors**: Ravi Shankar, Sheng Wong, Lin Li, Magdalena Bachmann, Alex Silverthorne, Beth Albert, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2509.04482)  

**Abstract**: Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM's advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems. 

**Abstract (ZH)**: 可靠的 abstention 对检索增强生成（RAG）系统至关重要，尤其是在如女性健康等安全关键领域，不正确的答案可能导致危害。我们提出了一种能量基于模型（EBM），学习由260万条指南衍生问题组成的密集语义语料库上的平滑能量景观，使系统能够在生成或 abstention 时作出决定。我们在能量基于模型（EBM）与校准的 softmax 基线以及 k-最近邻（kNN）密度启发式方法之间进行了基准测试，涵盖简单和困难的 abstention 分割，其中困难案例在语义上具有挑战性且接近分布查询。EBM 在语义困难情况下实现更好的 abstention 性能，达到 AUROC 0.961，比 softmax 的 0.950 更高，同时还将 FPR@95 降低（0.235 vs 0.331）。对于简单负例，各方法的性能相当，但在安全关键的困难分布中，EBM 的优势尤为明显。通过控制负样本采样和公平的数据暴露进行全面分析表明，稳健性主要源于能量评分头，而特定负例类型（困难、简单、混合）的包含或排除会细化决策边界，但对困难案例的一般化不是必需的。这些结果表明，能量基于的 abstention 评分提供了一个比基于概率的 softmax 信心更可靠的信心信号，为安全的 RAG 系统提供了可扩展且可解释的基础。 

---
# Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments 

**Title (ZH)**: 场景生成中的叙述驱动：面向2D游戏环境的LLM驱动Pipeline 

**Authors**: Yi-Chun Chen, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2509.04481)  

**Abstract**: Recent advances in large language models(LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation(PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG. 

**Abstract (ZH)**: Recent advances in大型语言模型(LLMs)促进了逼真故事生成，但在过程式内容生成(PCG)领域，将故事情节连接到可玩游戏的视觉环境中仍然是一个开放的挑战。我们提出了一种轻量级流水线，将简短的故事情节提示转换为一系列基于2D瓷砖的游戏场景，反映了故事情节的时间结构。给定一个LLM生成的故事情节，我们的系统识别三个关键时间框架，提取以“对象-关系-对象”三元组形式的空间谓词，并使用GameTileNet数据集中基于功能的语义嵌入检索视觉资产。使用细胞自动机生成分层地形，并根据谓词结构使用空间规则放置物体。我们在十个不同的故事中评估了本系统，分析了瓷砖-物体匹配、功能层对齐以及时间框架中的空间约束满足情况。该原型提供了一种面向故事情节的场景生成的可扩展方法，并为未来在以故事为中心的PCG中实现多帧连贯性、符号跟踪和多代理协调奠定了基础。 

---
# No Clustering, No Routing: How Transformers Actually Process Rare Tokens 

**Title (ZH)**: 无需聚类，无需路由：Transformer 实际如何处理稀见词汇 

**Authors**: Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04479)  

**Abstract**: Large language models struggle with rare token prediction, yet the mechanisms driving their specialization remain unclear. Prior work identified specialized ``plateau'' neurons for rare tokens following distinctive three-regime influence patterns \cite{liu2025emergent}, but their functional organization is unknown. We investigate this through neuron influence analyses, graph-based clustering, and attention head ablations in GPT-2 XL and Pythia models. Our findings show that: (1) rare token processing requires additional plateau neurons beyond the power-law regime sufficient for common tokens, forming dual computational regimes; (2) plateau neurons are spatially distributed rather than forming modular clusters; and (3) attention mechanisms exhibit no preferential routing to specialists. These results demonstrate that rare token specialization arises through distributed, training-driven differentiation rather than architectural modularity, preserving context-sensitive flexibility while achieving adaptive capacity allocation. 

**Abstract (ZH)**: 大型语言模型在预测稀有令牌方面存在困难，但其专业化机制尚不明确。先前研究发现了遵循独特三阶段影响模式的专门化“平台”神经元【1】，但其功能组织尚不清楚。我们通过神经元影响分析、图基群聚和注意力头消融研究GPT-2 XL和Pythia模型。我们的发现表明：(1) 稀有令牌处理需要超出幂律阶段的额外平台神经元，形成双计算阶段；(2) 平台神经元在空间上分散而非形成模块化集群；(3) 注意力机制没有优先路由到专家。这些结果表明，稀有令牌专业化通过分布式、训练驱动的分化产生，而不是通过架构模块化，从而保持上下文敏感的灵活性同时实现适应性容量分配。 

---
# Training Text-to-Molecule Models with Context-Aware Tokenization 

**Title (ZH)**: 基于上下文感知的分词训练文本到分子模型 

**Authors**: Seojin Kim, Hyeontae Song, Jaehyun Nam, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04476)  

**Abstract**: Recently, text-to-molecule models have shown great potential across various chemical applications, e.g., drug-discovery. These models adapt language models to molecular data by representing molecules as sequences of atoms. However, they rely on atom-level tokenizations, which primarily focus on modeling local connectivity, thereby limiting the ability of models to capture the global structural context within molecules. To tackle this issue, we propose a novel text-to-molecule model, coined Context-Aware Molecular T5 (CAMT5). Inspired by the significance of the substructure-level contexts in understanding molecule structures, e.g., ring systems, we introduce substructure-level tokenization for text-to-molecule models. Building on our tokenization scheme, we develop an importance-based training strategy that prioritizes key substructures, enabling CAMT5 to better capture the molecular semantics. Extensive experiments verify the superiority of CAMT5 in various text-to-molecule generation tasks. Intriguingly, we find that CAMT5 outperforms the state-of-the-art methods using only 2% of training tokens. In addition, we propose a simple yet effective ensemble strategy that aggregates the outputs of text-to-molecule models to further boost the generation performance. Code is available at this https URL. 

**Abstract (ZH)**: 近期，文本到分子模型在各种化学应用中展现了巨大的潜力，例如药物发现。这些模型通过将分子表示为原子序列来适应语言模型。然而，它们依赖于原子级别的标记化，主要侧重于建模局部连接性，从而限制了模型捕捉分子全局结构上下文的能力。为了解决这一问题，我们提出了一种新的文本到分子模型——基于上下文的T5分子模型（Context-Aware Molecular T5，简称CAMT5）。受子结构级别上下文在理解分子结构中的重要性启发，我们为文本到分子模型引入了子结构级别标记化。基于我们的标记化方案，我们开发了一种基于重要性的训练策略，优先处理关键子结构，使CAMT5更好地捕捉分子语义。广泛的经验表明，CAMT5在各种文本到分子生成任务中表现出色。有趣的是，我们发现CAMT5仅使用2%的训练标记就超过了最先进的方法。此外，我们提出了一种简单且有效的方法集成策略，通过聚合文本到分子模型的输出进一步提升生成性能。代码可通过以下链接获取。 

---
# ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute 

**Title (ZH)**: ParaThinker: 作为一种新型范式扩展大语言模型测试时计算的原生并行思考 

**Authors**: Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04475)  

**Abstract**: Recent advances in Large Language Models (LLMs) have been driven by test-time compute scaling - a strategy that improves reasoning by generating longer, sequential thought processes. While effective, this approach encounters a significant bottleneck as computation increases, where further computation offers only marginal performance gains. We argue this ceiling is not an inherent limit of the model's capability but a flaw in the scaling strategy itself, a phenomenon we term "Tunnel Vision", where a model's imperfect initial steps lock it into a suboptimal reasoning path. To overcome this, we introduce a new scaling paradigm: native thought parallelism. We present ParaThinker, an end-to-end framework that trains an LLM to generate multiple, diverse reasoning paths in parallel and synthesize them into a superior final answer. By exploring different lines of thoughts simultaneously, ParaThinker effectively sidesteps the Tunnel Vision issue and unlocks the model's latent reasoning potential. Our approach demonstrates that scaling compute in parallel (width) is a more effective and efficient way to superior reasoning than simply scaling sequentially (depth). On challenging reasoning benchmarks, ParaThinker achieves substantial accuracy improvements over sequential LLMs (12.3% for 1.5B and 7.5% for 7B models on average with 8 parallel paths), while adding only negligible latency overhead (7.1%). This enables smaller models to surpass much larger counterparts and establishes parallel thinking as a critical, efficient dimension for scaling future LLMs. 

**Abstract (ZH)**: Recent Advances in Large Language Models Driven by Native Thought Parallelism 

---
# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling 

**Title (ZH)**: 扩大规模，加快速度：推测解码的高效LLM测试时扩展基准 

**Authors**: Shengyin Sun, Yiming Li, Xing Li, Yingzhao Lian, Weizhe Lin, Hui-Ling Zhen, Zhiyuan Yang, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.04474)  

**Abstract**: Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However, this paradigm is inherently inefficient due to the generation of redundant and repetitive reasoning traces, leading to significant computational overhead. Speculative decoding offers a promising avenue for mitigating this inefficiency, yet its efficacy in the structured, repetition-rich context of test-time scaling remains largely unexplored. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate speculative decoding methods for accelerating LLM test-time scaling. Our benchmark provides consistent experimental protocols across representative test-time scaling paradigms (e.g., Best-of-N sampling and multi-round thinking), enabling a fair comparison of three major categories of speculative decoding: model-based, training-based, and n-gram-based methods. Extensive experiments reveal that simple n-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling. This phenomenon demonstrates the value of integrating n-gram-based methods with model-based or training-based approaches to balance acceleration for both repetitive and diverse reasoning in test-time scaling. We hope this benchmark spurs further research on speculative decoding for test-time scaling, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths. 

**Abstract (ZH)**: Test-time Scaling的推测性解码基准：促进大型语言模型推理加速的研究 

---
# SpeechLLM: Unified Speech and Language Model for Enhanced Multi-Task Understanding in Low Resource Settings 

**Title (ZH)**: SpeechLLM：统一语音与语言模型以增强低资源设置下的多任务理解 

**Authors**: Jaekwon Yoo, Kunal Chandiramani, Divya Tadimeti, Abenezer Girma, Chandra Dhir  

**Link**: [PDF](https://arxiv.org/pdf/2509.04473)  

**Abstract**: While integrating speech encoder with LLM requires substantial data and resources, use cases face limitations due to insufficient availability. To address this, we propose a solution with a parameter-efficient adapter that converts speech embeddings into LLM-compatible tokens, focusing on end-to-end automatic speech recognition (ASR), named entity recognition (NER), and sentiment analysis (SA). To reduce labeling costs, we employ an LLM-based synthetic dataset annotation technique. The proposed adapter, using 7x fewer trainable parameters, achieves significant performance gains: a 26% relative Word Error Rates (WER) improvement on the LibriSpeech ASR task, a 6.3% relative F1 score increase on the NER task, and a 32% relative F1 score boost on the SA task. Moreover, using advanced techniques such as adding a classifier regularizer and optimizing the LLM with Low-Rank Adaptation (LoRA) yields notable performance gains, with Spoken Language Understanding Evaluation (SLUE) score improvement of 6.6% and 9.5% 

**Abstract (ZH)**: 将语音编码器与大规模语言模型集成需要大量的数据和资源，但由于可用性不足，应用场景面临限制。为解决这一问题，我们提出了一种参数高效适配器解决方案，该适配器能够将语音嵌入转换为与大型语言模型兼容的令牌，专注于端到端自动语音识别（ASR）、命名实体识别（NER）和情感分析（SA）。为减少标注成本，我们采用了一种基于大规模语言模型的合成数据集标注技术。提出的适配器使用了7倍 fewer 的可训练参数，取得了显著的性能提升：在LibriSpeech ASR任务上相对词误率（WER）降低了26%，在NER任务上相对F1分数提高了6.3%，在SA任务上相对F1分数提升了32%。此外，通过添加分类器正则化器和使用低秩适应（LoRA）优化大型语言模型等高级技术，进一步提升了性能，在Spoken Language Understanding Evaluation（SLUE）分数上分别提高了6.6%和9.5%。 

---
# RECAP: REwriting Conversations for Intent Understanding in Agentic Planning 

**Title (ZH)**: RECAP: 代理规划中意图理解的对话重写 

**Authors**: Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2509.04472)  

**Abstract**: Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems. 

**Abstract (ZH)**: 理解用户意图对于会话代理的有效规划至关重要，尤其是在由大规模语言模型（LLMs）协调多个代理的会话代理中。然而，现实中的对话往往含糊、不明确或动态，使得意图检测成为一个持续的挑战。传统的基于分类的方法在开放环境下难以泛化，导致脆弱的解释和低效的后续规划。我们提出RECAP（REwriting Conversations for Agent Planning），这是一种新的基准，旨在评估和促进意图重写，将用户-代理对话重新框架为用户目标的简洁表示。RECAP捕捉到诸如含糊性、意图漂移、含糊性和混合目标对话等多种挑战。除了数据集之外，我们还引入了一个基于大规模语言模型的评估器，它可以评估重写意图后的规划效用。使用RECAP，我们开发了一种基于提示的重写方法，该方法优于基线方法。我们进一步证明，微调两个DPO基重写器可以带来额外的效用增益。我们的结果突出了意图重写作为提高开放领域对话系统中代理规划的关键和可行组件的重要性。 

---
# MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification 

**Title (ZH)**: MOSAIC: 一种多语言、无分类体系依赖且计算效率高的放射报告分类方法 

**Authors**: Alice Schiavone, Marco Fraccaro, Lea Marie Pehrson, Silvia Ingala, Rasmus Bonnevie, Michael Bachmann Nielsen, Vincent Beliveau, Melanie Ganz, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2509.04471)  

**Abstract**: Radiology reports contain rich clinical information that can be used to train imaging models without relying on costly manual annotation. However, existing approaches face critical limitations: rule-based methods struggle with linguistic variability, supervised models require large annotated datasets, and recent LLM-based systems depend on closed-source or resource-intensive models that are unsuitable for clinical use. Moreover, current solutions are largely restricted to English and single-modality, single-taxonomy datasets. We introduce MOSAIC, a multilingual, taxonomy-agnostic, and computationally efficient approach for radiological report classification. Built on a compact open-access language model (MedGemma-4B), MOSAIC supports both zero-/few-shot prompting and lightweight fine-tuning, enabling deployment on consumer-grade GPUs. We evaluate MOSAIC across seven datasets in English, Spanish, French, and Danish, spanning multiple imaging modalities and label taxonomies. The model achieves a mean macro F1 score of 88 across five chest X-ray datasets, approaching or exceeding expert-level performance, while requiring only 24 GB of GPU memory. With data augmentation, as few as 80 annotated samples are sufficient to reach a weighted F1 score of 82 on Danish reports, compared to 86 with the full 1600-sample training set. MOSAIC offers a practical alternative to large or proprietary LLMs in clinical settings. Code and models are open-source. We invite the community to evaluate and extend MOSAIC on new languages, taxonomies, and modalities. 

**Abstract (ZH)**: 多语言、无领域限制且计算高效的放射报告分类方法MOSAIC 

---
# COCORELI: Cooperative, Compositional Reconstitution \& Execution of Language Instructions 

**Title (ZH)**: COCORELI: 合作与组合的语言指令重组与执行 

**Authors**: Swarnadeep Bhar, Omar Naim, Eleni Metheniti, Bastien Navarri, Loïc Cabannes, Morteza Ezzabady, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2509.04470)  

**Abstract**: We present COCORELI, a hybrid agent framework designed to tackle the limitations of large language models (LLMs) in tasks requiring: following complex instructions, minimizing hallucination, and spatial reasoning. COCORELI integrates medium-sized LLM agents with novel abstraction mechanisms and a discourse module to parse instructions to in-context learn dynamic, high-level representations of the environment. Experiments on natural collaborative construction tasks show that COCORELI outperforms single-LLM CoT and agentic LLM systems, all using larger LLMs. It manages to largely avoid hallucinations, identify missing information, ask for clarifications, and update its learned objects. COCORELI's abstraction abilities extend beyond ENVIRONMENT, as shown in the ToolBench API completion task. 

**Abstract (ZH)**: 我们提出COCORELI，一种混合代理框架，旨在解决大型语言模型（LLMs）在需要遵循复杂指令、减少幻觉和空间推理的任务中面临的限制。COCORELI将中型LLM代理与新颖的抽象机制和话语模块结合起来，用于解析指令并以内存方式学习环境的动态高层次表示。自然协作构建任务的实验表明，COCORELI在使用更大规模LLM的单一LLM逐步推理系统和代理LLM系统中表现出色，能够大幅避免幻觉、识别缺失信息、请求澄清并更新其学习到的对象。COCORELI的抽象能力不仅限于环境，也在ToolBench API完成任务中得到体现。 

---
# Multi-Modal Vision vs. Text-Based Parsing: Benchmarking LLM Strategies for Invoice Processing 

**Title (ZH)**: 多模态视觉解析与基于文本的解析：检验应对发票处理的LLM策略基准 

**Authors**: David Berghaus, Armin Berger, Lars Hillebrand, Kostadin Cvejoski, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2509.04469)  

**Abstract**: This paper benchmarks eight multi-modal large language models from three families (GPT-5, Gemini 2.5, and open-source Gemma 3) on three diverse openly available invoice document datasets using zero-shot prompting. We compare two processing strategies: direct image processing using multi-modal capabilities and a structured parsing approach converting documents to markdown first. Results show native image processing generally outperforms structured approaches, with performance varying across model types and document characteristics. This benchmark provides insights for selecting appropriate models and processing strategies for automated document systems. Our code is available online. 

**Abstract (ZH)**: 本文使用零样本提示，在三个不同的公开可用的发票文档数据集中，对比了三种家族-eight个多模态大型语言模型（GPT-5、Gemini 2.5和开源的Gemma 3）的表现。我们比较了两种处理策略：直接利用多模态能力进行图像处理和先结构化解析转换为Markdown文档再处理。结果表明，原生图像处理通常优于结构化方法，但性能在不同模型类型和文档特征下有所变化。此次基准测试为选择合适的模型和处理策略以构建自动化文档系统提供了参考。我们的代码已上线。 

---
# Evaluating Large Language Models for Financial Reasoning: A CFA-Based Benchmark Study 

**Title (ZH)**: 基于CFA的大型语言模型金融推理评估基准研究 

**Authors**: Xuan Yao, Qianteng Wang, Xinbo Liu, Ke-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04468)  

**Abstract**: The rapid advancement of large language models presents significant opportunities for financial applications, yet systematic evaluation in specialized financial contexts remains limited. This study presents the first comprehensive evaluation of state-of-the-art LLMs using 1,560 multiple-choice questions from official mock exams across Levels I-III of CFA, most rigorous professional certifications globally that mirror real-world financial analysis complexity. We compare models distinguished by core design priorities: multi-modal and computationally powerful, reasoning-specialized and highly accurate, and lightweight efficiency-optimized.
We assess models under zero-shot prompting and through a novel Retrieval-Augmented Generation pipeline that integrates official CFA curriculum content. The RAG system achieves precise domain-specific knowledge retrieval through hierarchical knowledge organization and structured query generation, significantly enhancing reasoning accuracy in professional financial certification evaluation.
Results reveal that reasoning-oriented models consistently outperform others in zero-shot settings, while the RAG pipeline provides substantial improvements particularly for complex scenarios. Comprehensive error analysis identifies knowledge gaps as the primary failure mode, with minimal impact from text readability. These findings provide actionable insights for LLM deployment in finance, offering practitioners evidence-based guidance for model selection and cost-performance optimization. 

**Abstract (ZH)**: 大规模语言模型的快速发展为金融应用带来了显著机遇，但在专门的金融背景下系统性评估仍较为有限。本研究首次使用来自CFA一至三级官方模拟考试的1,560个多选题，对最先进的语言模型进行了全面评估，CFA是全球最严格的专业认证之一，涵盖了现实世界金融分析的复杂性。我们比较了三种不同核心设计优先级的语言模型：多模态和计算能力强大、擅长推理且高度准确，以及轻量级且效率优化。

我们在零样本提示下评估模型，并通过一种新颖的检索增强生成（RAG）流水线来整合官方CFA课程内容。RAG系统通过层次化知识组织和结构化查询生成实现精确的专业领域知识检索，显著增强了在专业金融认证评估中的推理准确性。

研究结果表明，面向推理的语言模型在零样本设置中始终表现最好，而RAG流水线在复杂场景中提供了显著改进。全面的错误分析发现，知识空白是主要的失败模式，文本可读性的影响极小。这些发现为金融中语言模型的部署提供了实际行动建议，为从业者提供基于证据的模型选择和成本性能优化指导。 

---
# Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference 

**Title (ZH)**: 提升大规模语言模型效率：针对预填充-解码分解的剪枝方法 

**Authors**: Hao Zhang, Mengsi Lyu, Yulong Ao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04467)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95 times reduction in data transmission bandwidth consumption. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展示了出色的能力，但其部署受到高计算和内存成本的限制。模型剪枝提供了一种有效的缓解这些要求的方法。然而，现有方法往往在实践中忽视了预填编码-解码（PD）分解的特性。本文提出了一种新的PD分解推理剪枝方法，实现更精确和高效的块级和KV缓存剪枝。我们的方法构建了剪枝和蒸馏集，独立地对预填和解码阶段进行迭代块移除，从而获得更好的剪枝解决方案。此外，我们引入了一种基于token的缓存剪枝机制，在预填阶段保留所有KV缓存，但在解码阶段选择性地重用某些层中第一个和最后一个token序列的条目，从而在最小开销的情况下减少通信成本。广泛实验表明，在PD分解和PD统一设置下，我们的方法都能保持强大的性能。在默认设置下，我们的方法实现了20.56%的推理速度提升和4.95倍的数据传输带宽消耗减少。 

---
# Just-in-time and distributed task representations in language models 

**Title (ZH)**: just-in-time和分布式任务表示在语言模型中 

**Authors**: Yuxuan Li, Declan Campbell, Stephanie C. Y. Chan, Andrew Kyle Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04466)  

**Abstract**: Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate \emph{when} representations for new tasks are formed in language models, and \emph{how} these representations change over the course of context. We focus on ''transferrable'' task representations -- vector representations that can restore task context in another instance of the model, even without the full prompt. We show that these representations evolve in non-monotonic and sporadic ways, and are distinct from a more inert representation of high-level task categories that persists throughout the context. Specifically, models often condense multiple evidence into these transferrable task representations, which align well with the performance improvement based on more examples in the context. However, this accrual process exhibits strong locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. Moreover, these local but transferrable task representations tend to capture minimal ''task scopes'', such as a semantically-independent subtask, and models rely on more temporally-distributed representations to support longer and composite tasks. This two-fold locality (temporal and semantic) underscores a kind of just-in-time computational process underlying language models' ability to adapt to new evidence and learn new tasks on the fly. 

**Abstract (ZH)**: 许多语言模型 impressive 的能力源自其基于上下文的学习：基于指令或示例，它们可以在不更新权重的情况下推断和执行新任务。在本文中，我们探讨了语言模型在什么情况下形成用于新任务的表征，以及这些表征在上下文中如何变化。我们专注于“可转移”的任务表征——可以在模型的另一个实例中恢复任务上下文的向量表征，即使没有完整的提示也是如此。我们发现这些表征以非单调和间歇性的方式演变，与在整个上下文中持续存在的更惰性的高级任务类别表征不同。具体来说，模型经常将多种证据压缩到这些可转移的任务表征中，这与上下文中更多示例支持的性能改进相一致。然而，这个积累过程在序列维度上表现出强烈的局部性，在某些标记上才上线——尽管任务身份在整个上下文中可靠可解码。此外，这些局部但可转移的任务表征倾向于捕捉最小的“任务范围”，如语义独立的子任务，而模型依赖更时间分布的表征来支持更长和复合的任务。这两层局部性（时间上的和语义上的）揭示了语言模型在不断适应新证据并即时学习新任务过程中的一种计算过程。 

---
# Emotionally-Aware Agents for Dispute Resolution 

**Title (ZH)**: 情绪感知代理在纠纷解决中的应用 

**Authors**: Sushrita Rakshit, James Hale, Kushal Chawla, Jeanne M. Brett, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2509.04465)  

**Abstract**: In conflict, people use emotional expressions to shape their counterparts' thoughts, feelings, and actions. This paper explores whether automatic text emotion recognition offers insight into this influence in the context of dispute resolution. Prior work has shown the promise of such methods in negotiations; however, disputes evoke stronger emotions and different social processes. We use a large corpus of buyer-seller dispute dialogues to investigate how emotional expressions shape subjective and objective outcomes. We further demonstrate that large-language models yield considerably greater explanatory power than previous methods for emotion intensity annotation and better match the decisions of human annotators. Findings support existing theoretical models for how emotional expressions contribute to conflict escalation and resolution and suggest that agent-based systems could be useful in managing disputes by recognizing and potentially mitigating emotional escalation. 

**Abstract (ZH)**: 冲突中，人们通过情感表达来影响对方的思维、情感和行为。本文探讨自动文本情感识别在纠纷解决情境下对这种影响的洞察。以往研究显示此类方法在谈判中有潜力；然而，纠纷引发的情感更强且涉及不同的社会过程。我们使用大量的买家卖家纠纷对话数据，研究情感表达如何影响主观和客观结果。此外，我们证明大规模语言模型在情绪强度标注上的解释力显著优于之前的方法，并更好地匹配了人类标注者的决策。研究结果支持现有的关于情感表达在冲突升级和解决中作用的理论模型，并表明基于代理的系统在通过识别和潜在缓解情绪升级来管理纠纷方面可能有用。 

---
# Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty? 

**Title (ZH)**: 多个LML响应能否揭示其不确定性来源？ 

**Authors**: Yang Nan, Pengfei He, Ravi Tandon, Han Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04464)  

**Abstract**: Large language models (LLMs) have delivered significant breakthroughs across diverse domains but can still produce unreliable or misleading outputs, posing critical challenges for real-world applications. While many recent studies focus on quantifying model uncertainty, relatively little work has been devoted to \textit{diagnosing the source of uncertainty}. In this study, we show that, when an LLM is uncertain, the patterns of disagreement among its multiple generated responses contain rich clues about the underlying cause of uncertainty. To illustrate this point, we collect multiple responses from a target LLM and employ an auxiliary LLM to analyze their patterns of disagreement. The auxiliary model is tasked to reason about the likely source of uncertainty, such as whether it stems from ambiguity in the input question, a lack of relevant knowledge, or both. In cases involving knowledge gaps, the auxiliary model also identifies the specific missing facts or concepts contributing to the uncertainty. In our experiment, we validate our framework on AmbigQA, OpenBookQA, and MMLU-Pro, confirming its generality in diagnosing distinct uncertainty sources. Such diagnosis shows the potential for relevant manual interventions that improve LLM performance and reliability. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个领域取得了显著突破，但仍可能产生不可靠或误导性的输出，给实际应用带来了关键挑战。虽然许多近期研究着重于量化模型不确定性，但对于不确定性的根源诊断研究相对较少。在本研究中，我们展示当LLM不确定时，其多个生成响应中的分歧模式包含丰富的关于不确定性根本原因的线索。为了说明这一点，我们从目标LLM收集多个响应，并使用辅助LLM分析这些分歧模式。辅助模型被任务化，以推断不确定性可能出现的原因，例如输入问题的模糊性、相关知识的不足，或是两者的结合。在涉及知识缺口的情况下，辅助模型还会识别出具体缺失的事实或概念，这些是导致不确定性的因素。在我们的实验中，我们在AmbigQA、OpenBookQA和MMLU-Pro上验证了我们框架的有效性，证实了其在诊断不同不确定性来源方面的普遍性。这种诊断显示出对相关手动干预以提升LLM性能和可靠性具有潜在价值。 

---
# Multiscale Graph Neural Network for Turbulent Flow-Thermal Prediction Around a Complex-Shaped Pin-Fin 

**Title (ZH)**: 复杂形貌针鳍周围湍流流动-热预测的多尺度图神经网络 

**Authors**: Riddhiman Raut, Evan M. Mihalko, Amrita Basak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04463)  

**Abstract**: This study presents the development of a domain-responsive edge-aware multiscale Graph Neural Network for predicting steady, turbulent flow and thermal behavior in a two-dimensional channel containing arbitrarily shaped complex pin-fin geometries. The training dataset was constructed through an automated framework that integrated geometry generation, meshing, and flow-field solutions in ANSYS Fluent. The pin-fin geometry was parameterized using piecewise cubic splines, producing 1,000 diverse configurations through Latin Hypercube Sampling. Each simulation was converted into a graph structure, where nodes carried a feature vector containing spatial coordinates, a normalized streamwise position, one-hot boundary indicators, and a signed distance to the nearest boundary such as wall. This graph structure served as input to the newly developed Graph Neural Network, which was trained to predict temperature, velocity magnitude, and pressure at each node using data from ANSYS. The network predicted fields with outstanding accuracy, capturing boundary layers, recirculation, and the stagnation region upstream of the pin-fins while reducing wall time by 2-3 orders of magnitude. In conclusion, the novel graph neural network offered a fast and reliable surrogate for simulations in complex flow configurations. 

**Abstract (ZH)**: 本文提出了一种针对特定领域且具备边缘感知能力的多尺度图神经网络，用于预测含有任意复杂鼻突-pin鳍几何结构的二维通道中的稳定湍流流动和热行为。训练数据集通过结合ANSYS Fluent中的几何生成、网格划分和流场求解的自动化框架构建。鼻突-pin鳍几何结构使用分段三次样条进行参数化，通过拉丁超立方抽样生成了1,000种不同的配置。每次模拟被转换为图结构，其中节点携带包含空间坐标、归一化的来流方向位置、边界指示符和到最近边界的符号距离等特征向量。该图结构作为新开发的图神经网络的输入，该网络被训练用于使用ANSYS数据预测每个节点的温度、速度大小和压力。该网络预测的场具有出色的准确性，能够捕捉边界层、再循环区域以及鼻突上游的停滞区域，同时将所需的时间缩短了2到3个数量级。总之，新的图神经网络为复杂流动配置下的模拟提供了一种快速且可靠的代理模型。 

---
# Benchmarking GPT-5 for biomedical natural language processing 

**Title (ZH)**: GPT-5在生物医学自然语言处理领域的基准测试 

**Authors**: Yu Hou, Zaifu Zhan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04462)  

**Abstract**: The rapid expansion of biomedical literature has heightened the need for scalable natural language processing (NLP) solutions. While GPT-4 substantially narrowed the gap with task-specific systems, especially in question answering, its performance across other domains remained uneven. We updated a standardized BioNLP benchmark to evaluate GPT-5 and GPT-4o under zero-, one-, and five-shot prompting across 12 datasets spanning six task families: named entity recognition, relation extraction, multi-label document classification, question answering, text summarization, and text simplification. Using fixed prompt templates, identical decoding parameters, and batch inference, we report primary metrics per dataset and include prior results for GPT-4, GPT-3.5, and LLaMA-2-13B for comparison. GPT-5 achieved the strongest overall benchmark performance, with macro-average scores rising to 0.557 under five-shot prompting versus 0.506 for GPT-4 and 0.508 for GPT-4o. On MedQA, GPT-5 reached 94.1% accuracy, exceeding the previous supervised state of the art by over fifty points, and attained parity with supervised systems on PubMedQA (0.734). In extraction tasks, GPT-5 delivered major gains in chemical NER (0.886 F1) and ChemProt relation extraction (0.616 F1), outperforming GPT-4 and GPT-4o, though summarization and disease NER still lagged behind domain-specific baselines. These results establish GPT-5 as a general-purpose model now offering deployment-ready performance for reasoning-oriented biomedical QA, while precision-critical extraction and evidence-dense summarization continue to favor fine-tuned or hybrid approaches. The benchmark delineates where simple prompting suffices and where retrieval-augmented or planning-based scaffolds are likely required, providing actionable guidance for BioNLP system design as frontier models advance. 

**Abstract (ZH)**: 生物医学文献的快速扩展凸显了可扩展自然语言处理（NLP）解决方案的必要性。尽管GPT-4在问答任务中显著缩小了与任务特定系统的差距，但在其他领域的表现仍然参差不齐。我们更新了标准化的BioNLP基准，评估了GPT-5和GPT-4o在零次、一次和五次提示下的表现，覆盖了12个数据集，涉及六个任务家族：命名实体识别、关系抽取、多标签文档分类、问答、文本摘要和文本简化。使用固定提示模板、相同的解码参数和批量推理，我们报告了每种数据集的主要指标，并包含了GPT-4、GPT-3.5和LLaMA-2-13B的先前结果以供对比。GPT-5在五次提示下的基准测试中表现最强，宏观平均分数达到了0.557，而GPT-4和GPT-4o分别为0.506和0.508。在MedQA上，GPT-5达到了94.1%的准确性，超过了之前监督学习的最佳表现50多个百分点，并且在PubMedQA上达到了与监督系统相同的水平（0.734）。在抽取任务中，GPT-5在化学命名实体识别（0.886 F1）和ChemProt关系抽取（0.616 F1）方面取得了显著进步，优于GPT-4和GPT-4o，尽管摘要和疾病命名实体识别仍落后于特定领域的基线。这些结果确立了GPT-5作为通用模型的地位，现在提供了针对推理导向的生物医学问答的部署就绪性能，而高精度的抽取和证据密集的摘要仍然偏向于细调或混合方法。基准测试界定了简单提示足够适用的领域，以及可能需要检索增强或规划支持框架的地方，为BioNLP系统设计提供了实际指导，随着前沿模型的发展。 

---
# CoCoNUTS: Concentrating on Content while Neglecting Uninformative Textual Styles for AI-Generated Peer Review Detection 

**Title (ZH)**: CoCoNUTS: 注重内容而忽略无信息性文本风格的AI生成同行评审检测 

**Authors**: Yihan Chen, Jiawei Chen, Guozhao Mo, Xuanang Chen, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04460)  

**Abstract**: The growing integration of large language models (LLMs) into the peer review process presents potential risks to the fairness and reliability of scholarly evaluation. While LLMs offer valuable assistance for reviewers with language refinement, there is growing concern over their use to generate substantive review content. Existing general AI-generated text detectors are vulnerable to paraphrasing attacks and struggle to distinguish between surface language refinement and substantial content generation, suggesting that they primarily rely on stylistic cues. When applied to peer review, this limitation can result in unfairly suspecting reviews with permissible AI-assisted language enhancement, while failing to catch deceptively humanized AI-generated reviews. To address this, we propose a paradigm shift from style-based to content-based detection. Specifically, we introduce CoCoNUTS, a content-oriented benchmark built upon a fine-grained dataset of AI-generated peer reviews, covering six distinct modes of human-AI collaboration. Furthermore, we develop CoCoDet, an AI review detector via a multi-task learning framework, designed to achieve more accurate and robust detection of AI involvement in review content. Our work offers a practical foundation for evaluating the use of LLMs in peer review, and contributes to the development of more precise, equitable, and reliable detection methods for real-world scholarly applications. Our code and data will be publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益集成到同行评审过程中的潜在风险对学术评价的公平性和可靠性构成了威胁。虽然LLMs为审稿人提供语言润色方面的宝贵帮助，但对其用于生成实质性评审内容的使用正引起越来越多的关注。现有的通用AI生成文本检测器容易受到改写攻击的影响，并且难以区分表面语言润饰和实质性内容生成，这表明它们主要依赖于风格线索。在应用于同行评审时，这一局限可能导致对允许AI辅助语言增强的评审不公平地怀疑，而未能捕捉到貌似由人类生成的AI生成评审。为解决这一问题，我们提出了从基于风格到基于内容的检测范式转变。具体而言，我们介绍了CoCoNUTS，这是一个基于精细粒度数据集的面向内容的基准，涵盖了人类-AI合作的六种不同模式。此外，我们开发了CoCoDet，这是一种通过多任务学习框架实现的AI评审检测器，旨在实现对评审内容中AI参与的更准确和更稳健的检测。我们的工作为评估LLMs在同行评审中的应用提供了实际基础，并为开发更精确、公平和可靠的检测方法做出了贡献，以适应实际的学术应用。我们的代码和数据将在此网址公开：this https URL。 

---
# Teacher-Student Model for Detecting and Classifying Mitosis in the MIDOG 2025 Challenge 

**Title (ZH)**: 教师-学生模型在MIDOG 2025挑战中检测和分类有丝分裂 

**Authors**: Seungho Choe, Xiaoli Qin, Abubakr Shafique, Amanda Dy, Susan Done, Dimitrios Androutsos, April Khademi  

**Link**: [PDF](https://arxiv.org/pdf/2509.03614)  

**Abstract**: Counting mitotic figures is time-intensive for pathologists and leads to inter-observer variability. Artificial intelligence (AI) promises a solution by automatically detecting mitotic figures while maintaining decision consistency. However, AI tools are susceptible to domain shift, where a significant drop in performance can occur due to differences in the training and testing sets, including morphological diversity between organs, species, and variations in staining protocols. Furthermore, the number of mitoses is much less than the count of normal nuclei, which introduces severely imbalanced data for the detection task. In this work, we formulate mitosis detection as a pixel-level segmentation and propose a teacher-student model that simultaneously addresses mitosis detection (Track 1) and atypical mitosis classification (Track 2). Our method is based on a UNet segmentation backbone that integrates domain generalization modules, namely contrastive representation learning and domain-adversarial training. A teacher-student strategy is employed to generate pixel-level pseudo-masks not only for annotated mitoses and hard negatives but also for normal nuclei, thereby enhancing feature discrimination and improving robustness against domain shift. For the classification task, we introduce a multi-scale CNN classifier that leverages feature maps from the segmentation model within a multi-task learning paradigm. On the preliminary test set, the algorithm achieved an F1 score of 0.7660 in Track 1 and balanced accuracy of 0.8414 in Track 2, demonstrating the effectiveness of integrating segmentation-based detection and classification into a unified framework for robust mitosis analysis. 

**Abstract (ZH)**: 基于像素级分割的教师-学生模型在分裂相检测和异常分裂相分类中的应用 

---
# Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving 

**Title (ZH)**: 高流量多LLM在线服务的高效无训练路由方法 

**Authors**: Fangzhou Wu, Sandeep Silwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.02718)  

**Abstract**: Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput. 

**Abstract (ZH)**: 增大小语言模型服务需求对提供者造成了显著的部署和计算成本。通过将查询定向到基于模型和查询特征的最优大语言模型，大语言模型路由提供了一种成本有效的解决方案。然而，现有工作主要集中在离线场景，并难以适应高查询量和受限令牌预算的在线设置。在本工作中，我们介绍了一种无需训练的在线路由场景算法。该算法利用近似最近邻搜索高效估计查询特征，并对少量初始查询进行一次优化以学习指导未来路由的策略。我们提供了理论保证，证明在自然假设下，该算法的竞争比达到$1 - o(1)$，并通过在3个基准数据集和8个基线方法上进行的广泛实验进一步验证，显示出整体性能平均提升3.55倍、成本效率提升1.85倍以及吞吐量提升高达4.25倍。 

---
# MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer 

**Title (ZH)**: MLP-SRGAN: 一种使用MLP-Mixer的一维超级分辨率生成对抗网络 

**Authors**: Samir Mitha, Seungho Choe, Pejman Jahbedar Maralani, Alan R. Moody, April Khademi  

**Link**: [PDF](https://arxiv.org/pdf/2303.06298)  

**Abstract**: We propose a novel architecture called MLP-SRGAN, which is a single-dimension Super Resolution Generative Adversarial Network (SRGAN) that utilizes Multi-Layer Perceptron Mixers (MLP-Mixers) along with convolutional layers to upsample in the slice direction. MLP-SRGAN is trained and validated using high resolution (HR) FLAIR MRI from the MSSEG2 challenge dataset. The method was applied to three multicentre FLAIR datasets (CAIN, ADNI, CCNA) of images with low spatial resolution in the slice dimension to examine performance on held-out (unseen) clinical data. Upsampled results are compared to several state-of-the-art SR networks. For images with high resolution (HR) ground truths, peak-signal-to-noise-ratio (PSNR) and structural similarity index (SSIM) are used to measure upsampling performance. Several new structural, no-reference image quality metrics were proposed to quantify sharpness (edge strength), noise (entropy), and blurriness (low frequency information) in the absence of ground truths. Results show MLP-SRGAN results in sharper edges, less blurring, preserves more texture and fine-anatomical detail, with fewer parameters, faster training/evaluation time, and smaller model size than existing methods. Code for MLP-SRGAN training and inference, data generators, models and no-reference image quality metrics will be available at this https URL. 

**Abstract (ZH)**: 我们提出了一种名为MLP-SRGAN的新架构，这是一种利用多层感知器混合器（MLP-Mixers）和卷积层在切片方向上进行上采样的单维度超分辨率生成对抗网络（SRGAN）。MLP-SRGAN使用MRI MSSEG2挑战数据集中的高分辨率（HR）FLAIR MRI进行训练和验证。该方法应用于三个多中心FLAIR数据集（CAIN、ADNI、CCNA），这些数据集中的图像在切片维度上的空间分辨率较低，以检查其在未见过的临床数据上的性能。上采样结果与几种最新的SR网络进行了比较。对于具有高分辨率（HR）ground truth的图像，峰值信噪比（PSNR）和结构相似性指数（SSIM）被用来衡量上采样性能。提出了几种新的无参考图像质量度量标准，以在没有ground truth的情况下量化锐度（边缘强度）、噪声（熵）和模糊度（低频信息）。结果表明，MLP-SRGAN在边缘更锐利、模糊更少、保留更多纹理和精细解剖细节、参数更少、训练/评估时间更快以及模型更小方面优于现有方法。MLP-SRGAN的训练和推理代码、数据生成器、模型和无参考图像质量度量标准将在此处提供。 

---
