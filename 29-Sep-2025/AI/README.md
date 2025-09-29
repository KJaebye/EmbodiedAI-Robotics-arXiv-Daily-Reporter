# Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective 

**Title (ZH)**: 强化学习在语言模型规划中的优势与风险：一个理论视角 

**Authors**: Siwei Wang, Yifei Shen, Haoran Sun, Shi Feng, Shang-Hua Teng, Li Dong, Yaru Hao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22613)  

**Abstract**: Recent reinforcement learning (RL) methods have substantially enhanced the planning capabilities of Large Language Models (LLMs), yet the theoretical basis for their effectiveness remains elusive. In this work, we investigate RL's benefits and limitations through a tractable graph-based abstraction, focusing on policy gradient (PG) and Q-learning methods. Our theoretical analyses reveal that supervised fine-tuning (SFT) may introduce co-occurrence-based spurious solutions, whereas RL achieves correct planning primarily through exploration, underscoring exploration's role in enabling better generalization. However, we also show that PG suffers from diversity collapse, where output diversity decreases during training and persists even after perfect accuracy is attained. By contrast, Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. We further demonstrate that careful reward design is necessary to prevent reward hacking in Q-learning. Finally, applying our framework to the real-world planning benchmark Blocksworld, we confirm that these behaviors manifest in practice. 

**Abstract (ZH)**: 近期的强化学习（RL）方法显著提高了大型语言模型（LLMs）的规划能力，但其有效性的理论基础仍不清楚。在本工作中，我们通过可处理的图基化抽象探究了RL的优势和局限性，重点关注策略梯度（PG）和Q学习方法。我们的理论分析表明，监督微调（SFT）可能会引入基于共现的虚假解，而RL主要通过探索实现正确的规划，突显了探索在促进更好泛化中的作用。然而，我们也展示了PG在训练过程中会遇到多样性崩溃的问题，在完美准确度达到后仍然持续。相比之下，Q学习提供了两个关键优势：离策学习和收敛时的多样性保持。我们进一步证明，精心设计奖励是防止Q学习中奖励劫持的必要条件。最后，将我们的框架应用于实际规划基准Blocksworld，我们确认这些行为在实践中确实存在。 

---
# Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Time 

**Title (ZH)**: 动态专家搜索：在测试时增强混合专家LLM的推理能力 

**Authors**: Yixuan Han, Fan Ma, Ruijie Quan, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22572)  

**Abstract**: Test-Time Scaling (TTS) enhances the reasoning ability of large language models (LLMs) by allocating additional computation during inference. However, existing approaches primarily rely on output-level sampling while overlooking the role of model architecture. In mainstream Mixture-of-Experts (MoE) LLMs, we observe that varying the number of activated experts yields complementary solution sets with stable accuracy, revealing a new and underexplored source of diversity. Motivated by this observation, we propose Dynamic Experts Search (DES), a TTS strategy that elevates expert activation into a controllable dimension of the search space. DES integrates two key components: (1) Dynamic MoE, which enables direct control of expert counts during inference to generate diverse reasoning trajectories without additional cost; and (2) Expert Configuration Inheritance, which preserves consistent expert counts within a reasoning path while varying them across runs, thereby balancing stability and diversity throughout the search. Extensive experiments across MoE architectures, verifiers and reasoning benchmarks (i.e., math, code and knowledge) demonstrate that DES reliably outperforms TTS baselines, enhancing accuracy and stability without additional cost. These results highlight DES as a practical and scalable form of architecture-aware TTS, illustrating how structural flexibility in modern LLMs can advance reasoning. 

**Abstract (ZH)**: Test-Time Scaling (TTS)通过在推理过程中分配额外的计算来增强大型语言模型（LLMs）的推理能力。然而，现有方法主要依赖于输出级采样，而忽视了模型架构的作用。在主流的Mixture-of-Experts（MoE）LLMs中，我们观察到激活不同数量的专家可以产生互补的解决方案集，精度稳定，揭示了一个新的、尚未充分探索的多样性来源。受此观察的启发，我们提出了动态专家搜索（DES），这是一种TTS策略，将专家激活提升为搜索空间中可控的维度。DES集成了两个关键组件：（1）动态MoE，能够在推理过程中直接控制专家数量，生成多样化的推理路径，而不增加额外成本；（2）专家配置继承，保持推理路径中的专家数量一致，而在不同运行中变化专家数量，从而在整个搜索过程中平衡稳定性和多样性。在MoE架构、验证器和推理基准（即数学、代码和知识）上的广泛实验表明，DES可靠地优于TTS基线，提升了准确性和稳定性而不额外增加成本。这些结果突显了DES作为一种实践性强且可扩展的架构感知TTS形式的重要性，展示了现代LLMs结构灵活性如何推动推理能力的提升。 

---
# UniMIC: Token-Based Multimodal Interactive Coding for Human-AI Collaboration 

**Title (ZH)**: UniMIC：基于令牌的多模态交互编码用于人机协作 

**Authors**: Qi Mao, Tinghan Yang, Jiahao Li, Bin Li, Libiao Jin, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22570)  

**Abstract**: The rapid progress of Large Multimodal Models (LMMs) and cloud-based AI agents is transforming human-AI collaboration into bidirectional, multimodal interaction. However, existing codecs remain optimized for unimodal, one-way communication, resulting in repeated degradation under conventional compress-transmit-reconstruct pipelines. To address this limitation, we propose UniMIC, a Unified token-based Multimodal Interactive Coding framework that bridges edge devices and cloud AI agents. Instead of transmitting raw pixels or plain text, UniMIC employs compact tokenized representations as the communication medium, enabling efficient low-bitrate transmission while maintaining compatibility with LMMs. To further enhance compression, lightweight Transformer-based entropy models with scenario-specific designs-generic, masked, and text-conditioned-effectively minimize inter-token redundancy. Extensive experiments on text-to-image generation, text-guided inpainting, outpainting, and visual question answering show that UniMIC achieves substantial bitrate savings and remains robust even at ultra-low bitrates (<0.05bpp), without compromising downstream task performance. These results establish UniMIC as a practical and forward-looking paradigm for next-generation multimodal interactive communication. 

**Abstract (ZH)**: 大型多模态模型和基于云的AI代理的快速进展正在将人类-AI协作转变为双向的多模态交互。然而，现有的编解码器仍针对单模态、单向通信进行了优化，在传统的压缩-传输-重构管道中表现出重复退化。为解决这一局限，我们提出了一种名为UniMIC的统一基于tokens的多模态交互编码框架，该框架连接边缘设备和云AI代理。UniMIC 不传输原始像素或纯文本，而是使用紧凑的token化表示作为通信介质，从而实现高效的低比特率传输，同时保持与大型多模态模型的兼容性。为进一步提高压缩效果，UniMIC 使用了轻量级的基于Transformer的熵模型，这些模型具有特定场景的设计——通用型、掩码型和文本条件型，有效减少了tokens间的冗余。在文本到图像生成、文本引导的 inpainting、outpainting 和视觉问答等广泛实验中，UniMIC 实现了显著的比特率节省，并且即使在超低比特率（<0.05 bpp）下依然保持稳健，不损害下游任务性能。这些结果确立了UniMIC作为下一代多模态交互通信实用且前瞻性的范式。 

---
# StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models 

**Title (ZH)**: StepORLM：一种带有生成过程监督的自进化框架用于运筹语言模型 

**Authors**: Chenyu Zhou, Tianyi Xu, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2509.22558)  

**Abstract**: Large Language Models (LLMs) have shown promising capabilities for solving Operations Research (OR) problems. While reinforcement learning serves as a powerful paradigm for LLM training on OR problems, existing works generally face two key limitations. First, outcome reward suffers from the credit assignment problem, where correct final answers can reinforce flawed reasoning. Second, conventional discriminative process supervision is myopic, failing to evaluate the interdependent steps of OR modeling holistically. To this end, we introduce StepORLM, a novel self-evolving framework with generative process supervision. At its core, StepORLM features a co-evolutionary loop where a policy model and a generative process reward model (GenPRM) iteratively improve on each other. This loop is driven by a dual-feedback mechanism: definitive, outcome-based verification from an external solver, and nuanced, holistic process evaluation from the GenPRM. The combined signal is used to align the policy via Weighted Direct Preference Optimization (W-DPO) and simultaneously refine the GenPRM. Our resulting 8B-parameter StepORLM establishes a new state-of-the-art across six benchmarks, significantly outperforming vastly larger generalist models, agentic methods, and specialized baselines. Moreover, the co-evolved GenPRM is able to act as a powerful and universally applicable process verifier, substantially boosting the inference scaling performance of both our own model and other existing LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决运筹学（OR）问题上展现了令人鼓舞的能力。虽然强化学习为LLM训练OR问题提供了一个强大的框架，但现有工作通常面临两个关键限制。首先，结果奖励受到归因问题的影响，即正确的最终答案可能会强化错误的推理。其次，传统的鉴别过程监督过于短视，无法全面评估OR建模中的相互依赖步骤。为解决这些问题，我们提出了一种名为StepORLM的新型自演化框架，该框架采用生成过程监督机制。StepORLM的核心在于政策模型与生成过程奖励模型（GenPRM）之间的共同演化循环，通过双重反馈机制驱动：外部求解器的确定性、基于结果的验证，和GenPRM提供的细致、全面的过程评估。结合这些信号通过加权直接偏好优化（W-DPO）对政策进行对齐，并同时细化GenPRM。我们的8B参数StepORLM在六个基准测试中达到了新的最先进的成果，显著优于更大规模的一般模型、代理方法和专门的基线。此外，共同演化生成过程奖励模型能够作为一个强大且普适的验证器，大幅提升了我们模型以及现有其他LLMs的推理扩展性能。 

---
# The Emergence of Altruism in Large-Language-Model Agents Society 

**Title (ZH)**: 大型语言模型代理社会中的利他主义 emergence 

**Authors**: Haoyang Li, Xiao Jia, Zhanzhan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22537)  

**Abstract**: Leveraging Large Language Models (LLMs) for social simulation is a frontier in computational social science. Understanding the social logics these agents embody is critical to this attempt. However, existing research has primarily focused on cooperation in small-scale, task-oriented games, overlooking how altruism, which means sacrificing self-interest for collective benefit, emerges in large-scale agent societies. To address this gap, we introduce a Schelling-variant urban migration model that creates a social dilemma, compelling over 200 LLM agents to navigate an explicit conflict between egoistic (personal utility) and altruistic (system utility) goals. Our central finding is a fundamental difference in the social tendencies of LLMs. We identify two distinct archetypes: "Adaptive Egoists", which default to prioritizing self-interest but whose altruistic behaviors significantly increase under the influence of a social norm-setting message board; and "Altruistic Optimizers", which exhibit an inherent altruistic logic, consistently prioritizing collective benefit even at a direct cost to themselves. Furthermore, to qualitatively analyze the cognitive underpinnings of these decisions, we introduce a method inspired by Grounded Theory to systematically code agent reasoning. In summary, this research provides the first evidence of intrinsic heterogeneity in the egoistic and altruistic tendencies of different LLMs. We propose that for social simulation, model selection is not merely a matter of choosing reasoning capability, but of choosing an intrinsic social action logic. While "Adaptive Egoists" may offer a more suitable choice for simulating complex human societies, "Altruistic Optimizers" are better suited for modeling idealized pro-social actors or scenarios where collective welfare is the primary consideration. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）进行社会模拟是计算社会科学的一个前沿领域。理解这些代理所体现的社会逻辑对于这一尝试至关重要。然而，现有研究主要集中在小规模任务导向游戏中合作机制的探索上，忽视了大规模代理社会中无私行为（牺牲个人利益以促进集体利益）的产生机制。为解决这一问题，我们引入了一个舍勒维奇变体的城市迁移模型，该模型创造了一个社会困境，促使超过200个LLM代理在个人利益（自我效用）和集体利益（系统效用）目标之间进行明面上的冲突导航。我们的主要发现是LLMs在社会倾向上的根本差异。我们识别出两种不同的原型：“适应性利己主义者”，其默认优先考虑自我利益，但在受到社会规范设置信息论坛影响时，其利他行为显著增加；“利他优化者”，表现出固有的利他逻辑，始终优先考虑集体利益，即使这意味着直接给自己带来损失。此外，为了定性分析这些决定的认知基础，我们引入了一种受扎根理论启发的方法，系统地编码代理推理。总之，这项研究提供了不同LLMs在利己和利他倾向上的内在异质性的初步证据。我们提出，在进行社会模拟时，模型选择不仅仅是选择推理能力的问题，更是选择内在的社会行动逻辑的问题。虽然“适应性利己主义者”更适用于模拟复杂的人类社会，“利他优化者”则更适合模拟理想化的亲社会行为者或以集体福利为主要考虑的场景。 

---
# REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model 

**Title (ZH)**: REMA：一个统一的推理流形框架用于解释大型语言模型 

**Authors**: Bo Li, Guanzhi Deng, Ronghao Chen, Junrong Yue, Shuo Zhang, Qinghua Zhao, Linqi Song, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22518)  

**Abstract**: Understanding how Large Language Models (LLMs) perform complex reasoning and their failure mechanisms is a challenge in interpretability research. To provide a measurable geometric analysis perspective, we define the concept of the Reasoning Manifold, a latent low-dimensional geometric structure formed by the internal representations corresponding to all correctly reasoned generations. This structure can be conceptualized as the embodiment of the effective thinking paths that the model has learned to successfully solve a given task. Based on this concept, we build REMA, a framework that explains the origins of failures by quantitatively comparing the spatial relationships of internal model representations corresponding to both erroneous and correct reasoning samples. Specifically, REMA first quantifies the geometric deviation of each erroneous representation by calculating its k-nearest neighbors distance to the approximated manifold formed by correct representations, thereby providing a unified failure signal. It then localizes the divergence points where these deviations first become significant by tracking this deviation metric across the model's layers and comparing it against a baseline of internal fluctuations from correct representations, thus identifying where the reasoning chain begins to go off-track. Our extensive experiments on diverse language and multimodal models and tasks demonstrate the low-dimensional nature of the reasoning manifold and the high separability between erroneous and correct reasoning representations. The results also validate the effectiveness of the REMA framework in analyzing the origins of reasoning failures. This research connects abstract reasoning failures to measurable geometric deviations in representations, providing new avenues for in-depth understanding and diagnosis of the internal computational processes of black-box models. 

**Abstract (ZH)**: 理解大型语言模型在复杂推理中的表现及其失败机制是可解释性研究中的一个挑战。为了提供一个可量化的几何分析视角，我们提出了推理流形的概念，这是一种由所有正确推理生成的内部表示形成的潜在低维几何结构。这一结构可以被构想为模型在解决给定任务时所学到的有效思维路径的体现。基于这一概念，我们构建了REMA框架，通过定量比较错误和正确推理样本的内部模型表示之间的空间关系来解释其失败的根源。具体来说，REMA首先通过计算每个错误表示与由正确表示近似形成的流形的k近邻距离，来量化每个错误表示的几何偏差，从而提供统一的失败信号。然后通过跟踪这一偏差指标在整个模型层中的变化，并将其与正确表示内部分波动的基线进行比较，来定位这些偏差首次变得显著的发散点，从而确定推理链开始偏离的起始位置。我们在多种语言和多模态模型及任务上的广泛实验表明，推理流形具有低维性质，错误和正确推理表示之间的分离性很高。这些结果还验证了REMA框架在分析推理失败根源方面的有效性。该研究将抽象的推理失败与表示中的可测量几何偏差联系起来，为深入理解和诊断黑盒模型的内部计算过程提供了新的途径。 

---
# TrueGradeAI: Retrieval-Augmented and Bias-Resistant AI for Transparent and Explainable Digital Assessments 

**Title (ZH)**: TrueGradeAI: 检索增强且抗偏见的透明可解释数字评估AI 

**Authors**: Rakesh Thakur, Shivaansh Kaushik, Gauri Chopra, Harsh Rohilla  

**Link**: [PDF](https://arxiv.org/pdf/2509.22516)  

**Abstract**: This paper introduces TrueGradeAI, an AI-driven digital examination framework designed to overcome the shortcomings of traditional paper-based assessments, including excessive paper usage, logistical complexity, grading delays, and evaluator bias. The system preserves natural handwriting by capturing stylus input on secure tablets and applying transformer-based optical character recognition for transcription. Evaluation is conducted through a retrieval-augmented pipeline that integrates faculty solutions, cache layers, and external references, enabling a large language model to assign scores with explicit, evidence-linked reasoning. Unlike prior tablet-based exam systems that primarily digitize responses, TrueGradeAI advances the field by incorporating explainable automation, bias mitigation, and auditable grading trails. By uniting handwriting preservation with scalable and transparent evaluation, the framework reduces environmental costs, accelerates feedback cycles, and progressively builds a reusable knowledge base, while actively working to mitigate grading bias and ensure fairness in assessment. 

**Abstract (ZH)**: TrueGradeAI：一种基于AI的数字考试框架，用于克服传统纸质评估的不足 

---
# Estimating the Empowerment of Language Model Agents 

**Title (ZH)**: 估计语言模型代理的赋能能力 

**Authors**: Jinyeop Song, Jeff Gore, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2509.22504)  

**Abstract**: As language model (LM) agents become more capable and gain broader access to real-world tools, there is a growing need for scalable evaluation frameworks of agentic capability. However, conventional benchmark-centric evaluations are costly to design and require human designers to come up with valid tasks that translate into insights about general model capabilities. In this work, we propose information-theoretic evaluation based on empowerment, the mutual information between an agent's actions and future states, as an open-ended method for evaluating LM agents. We introduce EELMA (Estimating Empowerment of Language Model Agents), an algorithm for approximating effective empowerment from multi-turn text interactions. We validate EELMA on both language games and scaled-up realistic web-browsing scenarios. We find that empowerment strongly correlates with average task performance, characterize the impact of environmental complexity and agentic factors such as chain-of-thought, model scale, and memory length on estimated empowerment, and that high empowerment states and actions are often pivotal moments for general capabilities. Together, these results demonstrate empowerment as an appealing general-purpose metric for evaluating and monitoring LM agents in complex, open-ended settings. 

**Abstract (ZH)**: 基于信息论的代理能力评估：利用语言模型代理的能动力量作为开放性评估方法 

---
# InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios 

**Title (ZH)**: InfiAgent：无限场景自演进金字塔代理框架 

**Authors**: Chenglin Yu, Yang Yu, Songmiao Wang, Yucheng Wang, Yifan Yang, Jinjia Li, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22502)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable capabilities in organizing and executing complex tasks, and many such agents are now widely used in various application scenarios. However, developing these agents requires carefully designed workflows, carefully crafted prompts, and iterative tuning, which requires LLM techniques and domain-specific expertise. These hand-crafted limitations hinder the scalability and cost-effectiveness of LLM agents across a wide range of industries. To address these challenges, we propose \textbf{InfiAgent}, a Pyramid-like DAG-based Multi-Agent Framework that can be applied to \textbf{infi}nite scenarios, which introduces several key innovations: a generalized "agent-as-a-tool" mechanism that automatically decomposes complex agents into hierarchical multi-agent systems; a dual-audit mechanism that ensures the quality and stability of task completion; an agent routing function that enables efficient task-agent matching; and an agent self-evolution mechanism that autonomously restructures the agent DAG based on new tasks, poor performance, or optimization opportunities. Furthermore, InfiAgent's atomic task design supports agent parallelism, significantly improving execution efficiency. This framework evolves into a versatile pyramid-like multi-agent system capable of solving a wide range of problems. Evaluations on multiple benchmarks demonstrate that InfiAgent achieves 9.9\% higher performance compared to ADAS (similar auto-generated agent framework), while a case study of the AI research assistant InfiHelper shows that it generates scientific papers that have received recognition from human reviewers at top-tier IEEE conferences. 

**Abstract (ZH)**: 大规模语言模型（LLM）代理展示了在组织和执行复杂任务方面 remarkable 的能力，如今已在多种应用场景中普遍使用。然而，开发这些代理需要精心设计的工作流、精心构建的提示以及迭代调优，这需要LLM技术和领域特定的专业知识。这些手工制作的限制阻碍了LLM代理在各种行业的扩展性和成本效益。为了解决这些挑战，我们提出了一种名为InfiAgent的金字塔状DAG基础多代理框架，该框架可以应用于无限场景，并引入了几项关键技术创新：通用“代理作为工具”机制，自动将复杂代理分解为分层多代理系统；双重审计机制，确保任务完成的质量和稳定性；代理路由功能，实现高效的任务-代理匹配；以及代理自我进化机制，根据新任务、表现不佳或优化机会自主重构代理DAG。此外，InfiAgent的原子任务设计支持代理并行性，显著提高了执行效率。该框架演进为一种多代理系统，能够解决广泛的问题。在多个基准上的评估表明，InfiAgent相比ADAS（类似自动生成代理框架）性能提高了9.9%，而在AI研究助手InfiHelper的案例研究中，它生成的科学论文获得了顶级IEEE会议的人类评审者的认可。 

---
# GeoSketch: A Neural-Symbolic Approach to Geometric Multimodal Reasoning with Auxiliary Line Construction and Affine Transformation 

**Title (ZH)**: GeoSketch: 一种辅助线构造与仿射变换介导的几何多模态推理的神经符号方法 

**Authors**: Shichao Weng, Zhiqiang Wang, Yuhua Zhou, Rui Lu, Ting Liu, Zhiyang Teng, Xiaozhang Liu, Hanmeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22460)  

**Abstract**: Geometric Problem Solving (GPS) poses a unique challenge for Multimodal Large Language Models (MLLMs), requiring not only the joint interpretation of text and diagrams but also iterative visuospatial reasoning. While existing approaches process diagrams as static images, they lack the capacity for dynamic manipulation - a core aspect of human geometric reasoning involving auxiliary line construction and affine transformations. We present GeoSketch, a neural-symbolic framework that recasts geometric reasoning as an interactive perception-reasoning-action loop. GeoSketch integrates: (1) a Perception module that abstracts diagrams into structured logic forms, (2) a Symbolic Reasoning module that applies geometric theorems to decide the next deductive step, and (3) a Sketch Action module that executes operations such as drawing auxiliary lines or applying transformations, thereby updating the diagram in a closed loop. To train this agent, we develop a two-stage pipeline: supervised fine-tuning on 2,000 symbolic-curated trajectories followed by reinforcement learning with dense, symbolic rewards to enhance robustness and strategic exploration. To evaluate this paradigm, we introduce the GeoSketch Benchmark, a high-quality set of 390 geometry problems requiring auxiliary construction or affine transformations. Experiments on strong MLLM baselines demonstrate that GeoSketch significantly improves stepwise reasoning accuracy and problem-solving success over static perception methods. By unifying hierarchical decision-making, executable visual actions, and symbolic verification, GeoSketch advances multimodal reasoning from static interpretation to dynamic, verifiable interaction, establishing a new foundation for solving complex visuospatial problems. 

**Abstract (ZH)**: 几何问题求解（GPS）对多模态大型语言模型（MLLMs）提出了独特挑战，不仅需要联合解释文本和图表，还需要迭代的空间视觉推理。现有方法将图表处理为静态图像，缺乏动态操作的能力——这是人类几何推理的核心方面，涉及辅助线构造和仿射变换。我们提出了GeoSketch，一个神经符号框架，将几何推理重新表述为互动感知-推理-行动循环。GeoSketch 结合了：（1）一个感知模块，将图表抽象为结构化逻辑形式；（2）一个符号推理模块，应用几何定理决定下一步推理；（3）一个绘图动作模块，执行绘制辅助线或应用变换等操作，以在闭环中更新图表。为了训练这个代理，我们开发了一个两阶段管道：首先在2000条符号整理的轨迹上进行监督微调，然后通过密集的符号奖励进行强化学习，以增强鲁棒性和策略探索。为了评估这一范式，我们引入了GeoSketch基准，这是一个包含390个需要辅助构造或仿射变换的高质量几何问题的数据集。在强大的MLLM基线上进行的实验表明，GeoSketch在逐步推理准确性和解决问题成功率上显著优于静态感知方法。通过统一层次决策、可执行的视觉操作和符号验证，GeoSketch将多模态推理从静态解释提升到动态、可验证的交互，建立了解决复杂空间问题的新基础。 

---
# Guiding Evolution of Artificial Life Using Vision-Language Models 

**Title (ZH)**: 使用视觉-语言模型引导人工生命演化 

**Authors**: Nikhil Baid, Hannah Erlebach, Paul Hellegouarch, Frederico Wieser  

**Link**: [PDF](https://arxiv.org/pdf/2509.22447)  

**Abstract**: Foundation models (FMs) have recently opened up new frontiers in the field of artificial life (ALife) by providing powerful tools to automate search through ALife simulations. Previous work aligns ALife simulations with natural language target prompts using vision-language models (VLMs). We build on Automated Search for Artificial Life (ASAL) by introducing ASAL++, a method for open-ended-like search guided by multimodal FMs. We use a second FM to propose new evolutionary targets based on a simulation's visual history. This induces an evolutionary trajectory with increasingly complex targets.
We explore two strategies: (1) evolving a simulation to match a single new prompt at each iteration (Evolved Supervised Targets: EST) and (2) evolving a simulation to match the entire sequence of generated prompts (Evolved Temporal Targets: ETT). We test our method empirically in the Lenia substrate using Gemma-3 to propose evolutionary targets, and show that EST promotes greater visual novelty, while ETT fosters more coherent and interpretable evolutionary sequences.
Our results suggest that ASAL++ points towards new directions for FM-driven ALife discovery with open-ended characteristics. 

**Abstract (ZH)**: 基于模型（FMs）在人工生命（ALife）领域的新兴前沿：通过提供自动搜索ALife模拟的强大工具，FMs正在开辟新的方向。我们在此基础上引入ASAL++方法，该方法利用多模态FMs进行类似开放搜索的引导。我们使用第二个FM根据模拟的视觉历史提出新的进化目标，从而引发越来越复杂的进化轨迹。我们探索了两种策略：（1）每次迭代进化模拟以匹配单个新提示（进化监督目标：EST）；（2）进化模拟以匹配整个生成提示序列（进化时间目标：ETT）。我们在Lenia基质中使用Gemma-3实验测试了我们的方法，并表明EST促进了更大的视觉新颖性，而ETT促进了更连贯和可解释的进化序列。研究结果表明，ASAL++指出了具有开放性特征的FM驱动ALife发现的新方向。 

---
# EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer 

**Title (ZH)**: EMMA: 通过生成视觉转移实现通用现实世界机器人操作 

**Authors**: Zhehao Dong, Xiaofeng Wang, Zheng Zhu, Yirui Wang, Yang Wang, Yukun Zhou, Boyuan Wang, Chaojun Ni, Runqi Ouyang, Wenkang Qin, Xinze Chen, Yun Ye, Guan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22407)  

**Abstract**: Vision-language-action (VLA) models increasingly rely on diverse training data to achieve robust generalization. However, collecting large-scale real-world robot manipulation data across varied object appearances and environmental conditions remains prohibitively time-consuming and expensive. To overcome this bottleneck, we propose Embodied Manipulation Media Adaptation (EMMA), a VLA policy enhancement framework that integrates a generative data engine with an effective training pipeline. We introduce DreamTransfer, a diffusion Transformer-based framework for generating multi-view consistent, geometrically grounded embodied manipulation videos. DreamTransfer enables text-controlled visual editing of robot videos, transforming foreground, background, and lighting conditions without compromising 3D structure or geometrical plausibility. Furthermore, we explore hybrid training with real and generated data, and introduce AdaMix, a hard-sample-aware training strategy that dynamically reweights training batches to focus optimization on perceptually or kinematically challenging samples. Extensive experiments show that videos generated by DreamTransfer significantly outperform prior video generation methods in multi-view consistency, geometric fidelity, and text-conditioning accuracy. Crucially, VLAs trained with generated data enable robots to generalize to unseen object categories and novel visual domains using only demonstrations from a single appearance. In real-world robotic manipulation tasks with zero-shot visual domains, our approach achieves over a 200% relative performance gain compared to training on real data alone, and further improves by 13% with AdaMix, demonstrating its effectiveness in boosting policy generalization. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型 increasingly 依赖多样化训练数据以实现稳健的泛化。然而，跨不同物体外观和环境条件收集大规模真实机器人操作数据仍然是时间上和经济上极为耗时且昂贵的。为了克服这一瓶颈，我们提出了一种物质交互媒体适应（EMMA）框架，该框架将生成数据引擎与有效的训练管道集成起来，增强VLA策略。我们引入了DreamTransfer，这是一种基于扩散Transformer的框架，用于生成多视图一致、几何上可信的物质交互视频。DreamTransfer允许对机器人视频进行文本控制的视觉编辑，可以改变前景、背景和光照条件而不损害三维结构或几何合理性。此外，我们探索了真实数据与生成数据的混合训练，并引入了AdaMix，这是一种感知或动力学上具有挑战认样样本的自适应训练策略，动态调整训练批次的权重以聚焦于优化难点样本。广泛实验证明，由DreamTransfer生成的视频在多视图一致性、几何保真度和文本条件准确性方面显著优于先前的视频生成方法。最关键的是，使用生成数据训练的VLA能够在仅从单一外观的演示中泛化到未见过的对象类别和新的视觉域。在零样本视觉域的现实世界机器人操作任务中，与仅使用真实数据训练相比，我们的方法实现了超过200%的相对性能提升，并且与AdaMix结合使用时再提升了13%，证明了其提高策略泛化效果的有效性。 

---
# Do LLM Agents Know How to Ground, Recover, and Assess? A Benchmark for Epistemic Competence in Information-Seeking Agents 

**Title (ZH)**: LLM代理知道如何进行知性定位、恢复和评估吗？信息寻求代理的知性能力基准 

**Authors**: Jiaqi Shao, Yuxiang Lin, Munish Prasad Lohani, Yufeng Miao, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22391)  

**Abstract**: Recent work has explored training Large Language Model (LLM) search agents with reinforcement learning (RL) for open-domain question answering (QA). However, most evaluations focus solely on final answer accuracy, overlooking how these agents reason with and act on external evidence. We introduce SeekBench, the first benchmark for evaluating the \textit{epistemic competence} of LLM search agents through step-level analysis of their response traces. SeekBench comprises 190 expert-annotated traces with over 1,800 response steps generated by LLM search agents, each enriched with evidence annotations for granular analysis of whether agents (1) generate reasoning steps grounded in observed evidence, (2) adaptively reformulate searches to recover from low-quality results, and (3) have proper calibration to correctly assess whether the current evidence is sufficient for providing an answer. 

**Abstract (ZH)**: 近期的研究探索了使用强化学习训练大型语言模型搜索代理进行开放域问答。然而，大多数评估仅集中在最终答案的准确性上，忽视了这些代理如何处理和利用外部证据进行推理和行动。我们引入了SeekBench，这是首个通过步骤级分析响应轨迹来评估大型语言模型搜索代理的本体知识 competence 的基准测试。SeekBench 包含 190 条由大型语言模型搜索代理生成的专业注解响应步骤，每条步骤都富含有证据注解，以便细致分析代理是否能够（1）生成基于观察证据的推理步骤，（2）自适应地重新制定搜索策略以克服低质量结果，以及（3）适当校准以正确评估当前证据是否足以提供答案。 

---
# PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning 

**Title (ZH)**: PRIME: 结合检索的记忆规划以增强推理 

**Authors**: Hieu Tran, Zonghai Yao, Nguyen Luong Tran, Zhichao Yang, Feiyun Ouyang, Shuo Han, Razieh Rahimi, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22315)  

**Abstract**: Inspired by the dual-process theory of human cognition from \textit{Thinking, Fast and Slow}, we introduce \textbf{PRIME} (Planning and Retrieval-Integrated Memory for Enhanced Reasoning), a multi-agent reasoning framework that dynamically integrates \textbf{System 1} (fast, intuitive thinking) and \textbf{System 2} (slow, deliberate thinking). PRIME first employs a Quick Thinking Agent (System 1) to generate a rapid answer; if uncertainty is detected, it then triggers a structured System 2 reasoning pipeline composed of specialized agents for \textit{planning}, \textit{hypothesis generation}, \textit{retrieval}, \textit{information integration}, and \textit{decision-making}. This multi-agent design faithfully mimics human cognitive processes and enhances both efficiency and accuracy. Experimental results with LLaMA 3 models demonstrate that PRIME enables open-source LLMs to perform competitively with state-of-the-art closed-source models like GPT-4 and GPT-4o on benchmarks requiring multi-hop and knowledge-grounded reasoning. This research establishes PRIME as a scalable solution for improving LLMs in domains requiring complex, knowledge-intensive reasoning. 

**Abstract (ZH)**: 受《思考，快与慢》中人类认知的两过程理论启发，我们引入了PRIME（Planning and Retrieval-Integrated Memory for Enhanced Reasoning），一种动态整合快速直观思考（System 1）和缓慢深思熟虑（System 2）的多-agent推理框架。PRIME首先利用快速思考代理（System 1）生成快速答案；若检测到不确定性，则触发由专门代理组成的结构化System 2推理管道，包括规划、假设生成、检索、信息整合和决策制定。这种多-agent设计忠实模拟了人类的认知过程，并提高了效率和准确性。实验结果表明，PRIME使开源大语言模型在需要多跳和知识驱动推理的基准测试中能够与最先进的闭源模型（如GPT-4和GPT-4o）媲美。本研究确立了PRIME作为提高需要复杂知识密集型推理的领域中大语言模型可扩展解决方案的地位。 

---
# Large Language Models as Nondeterministic Causal Models 

**Title (ZH)**: 大型语言模型作为非确定性因果模型 

**Authors**: Sander Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2509.22297)  

**Abstract**: Recent work by Chatzi et al. and Ravfogel et al. has developed, for the first time, a method for generating counterfactuals of probabilistic Large Language Models. Such counterfactuals tell us what would - or might - have been the output of an LLM if some factual prompt ${\bf x}$ had been ${\bf x}^*$ instead. The ability to generate such counterfactuals is an important necessary step towards explaining, evaluating, and comparing, the behavior of LLMs. I argue, however, that the existing method rests on an ambiguous interpretation of LLMs: it does not interpret LLMs literally, for the method involves the assumption that one can change the implementation of an LLM's sampling process without changing the LLM itself, nor does it interpret LLMs as intended, for the method involves explicitly representing a nondeterministic LLM as a deterministic causal model. I here present a much simpler method for generating counterfactuals that is based on an LLM's intended interpretation by representing it as a nondeterministic causal model instead. The advantage of my simpler method is that it is directly applicable to any black-box LLM without modification, as it is agnostic to any implementation details. The advantage of the existing method, on the other hand, is that it directly implements the generation of a specific type of counterfactuals that is useful for certain purposes, but not for others. I clarify how both methods relate by offering a theoretical foundation for reasoning about counterfactuals in LLMs based on their intended semantics, thereby laying the groundwork for novel application-specific methods for generating counterfactuals. 

**Abstract (ZH)**: Recent工作由Chatzi等人和Ravfogel等人首次开发了一种生成概率大型语言模型反事实的方法。这类反事实能够告诉我们如果某个事实提示x被替换为x*，大型语言模型的输出会是什么或者可能是怎样的。生成这类反事实的能力是解释、评估和比较大型语言模型行为的重要前提步骤。然而，我认为现有的方法基于对大型语言模型含糊不清的解释：它既不字面解释大型语言模型，因为该方法假设可以改变大型语言模型采样过程的实现而不改变模型本身，也不按照预期解释大型语言模型，因为该方法涉及将非确定性大型语言模型明确表示为确定性因果模型。我提出了一个更为简化的生成反事实的方法，该方法基于大型语言模型的预期解释，将其表示为非确定性因果模型。这种更简单的方法的优势在于它可以不加修改地应用于任何黑盒大型语言模型，因为它忽略了任何实现细节。现有的方法的优点在于它直接实现了生成特定类型对某些用途有用的反事实，但对其他用途则不然。通过提供基于大型语言模型预期语义推理反事实的理论基础，我澄清了两种方法之间的关系，为生成反事实的应用特定方法奠定基础。 

---
# Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models 

**Title (ZH)**: 结构化稀疏转移矩阵以启用状态空间模型中的状态跟踪 

**Authors**: Aleksandar Terzić, Nicolas Menet, Michael Hersche, Thomas Hofmann, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22284)  

**Abstract**: Modern state-space models (SSMs) often utilize transition matrices which enable efficient computation but pose restrictions on the model's expressivity, as measured in terms of the ability to emulate finite-state automata (FSA). While unstructured transition matrices are optimal in terms of expressivity, they come at a prohibitively high compute and memory cost even for moderate state sizes. We propose a structured sparse parametrization of transition matrices in SSMs that enables FSA state tracking with optimal state size and depth, while keeping the computational cost of the recurrence comparable to that of diagonal SSMs. Our method, PD-SSM, parametrizes the transition matrix as the product of a column one-hot matrix ($P$) and a complex-valued diagonal matrix ($D$). Consequently, the computational cost of parallel scans scales linearly with the state size. Theoretically, the model is BIBO-stable and can emulate any $N$-state FSA with one layer of dimension $N$ and a linear readout of size $N \times N$, significantly improving on all current structured SSM guarantees. Experimentally, the model significantly outperforms a wide collection of modern SSM variants on various FSA state tracking tasks. On multiclass time-series classification, the performance is comparable to that of neural controlled differential equations, a paradigm explicitly built for time-series analysis. Finally, we integrate PD-SSM into a hybrid Transformer-SSM architecture and demonstrate that the model can effectively track the states of a complex FSA in which transitions are encoded as a set of variable-length English sentences. The code is available at this https URL 

**Abstract (ZH)**: 现代状态空间模型中结构化稀疏转移矩阵参数化以实现高效的有限状态自动机状态跟踪 

---
# InfiMed-Foundation: Pioneering Advanced Multimodal Medical Models with Compute-Efficient Pre-Training and Multi-Stage Fine-Tuning 

**Title (ZH)**: InfiMed-基础模型：开创性构建计算高效预训练与多阶段 fine-tuning 的先进多模态医疗模型 

**Authors**: Guanghao Zhu, Zhitian Hou, Zeyu Liu, Zhijie Sang, Congkai Xie, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22261)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable potential in various domains, yet their application in the medical field is hindered by several challenges. General-purpose MLLMs often lack the specialized knowledge required for medical tasks, leading to uncertain or hallucinatory responses. Knowledge distillation from advanced models struggles to capture domain-specific expertise in radiology and pharmacology. Additionally, the computational cost of continual pretraining with large-scale medical data poses significant efficiency challenges. To address these issues, we propose InfiMed-Foundation-1.7B and InfiMed-Foundation-4B, two medical-specific MLLMs designed to deliver state-of-the-art performance in medical applications. We combined high-quality general-purpose and medical multimodal data and proposed a novel five-dimensional quality assessment framework to curate high-quality multimodal medical datasets. We employ low-to-high image resolution and multimodal sequence packing to enhance training efficiency, enabling the integration of extensive medical data. Furthermore, a three-stage supervised fine-tuning process ensures effective knowledge extraction for complex medical tasks. Evaluated on the MedEvalKit framework, InfiMed-Foundation-1.7B outperforms Qwen2.5VL-3B, while InfiMed-Foundation-4B surpasses HuatuoGPT-V-7B and MedGemma-27B-IT, demonstrating superior performance in medical visual question answering and diagnostic tasks. By addressing key challenges in data quality, training efficiency, and domain-specific knowledge extraction, our work paves the way for more reliable and effective AI-driven solutions in healthcare. InfiMed-Foundation-4B model is available at \href{this https URL}{InfiMed-Foundation-4B}. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在多个领域展现出了显著的潜力，但在医疗领域的应用受到多重挑战的限制。通用型MLLMs往往缺乏完成医疗任务所需的专门知识，导致响应不确定或虚构。高级模型的知识蒸馏难以捕捉医学成像和药理学领域的专业技能。此外，持续预训练大规模医疗数据的计算成本对效率提出了重大挑战。为应对这些问题，我们提出了InfiMed-Foundation-1.7B和InfiMed-Foundation-4B两种医疗专用MLLMs，旨在为医疗应用提供最先进的性能。我们结合了高质量的通用和医疗多模态数据，并提出了一种新颖的五维度质量评估框架以构建高质量的多模态医疗数据集。我们采用低到高的图像分辨率和多模态序列包装来增强训练效率，从而可以整合大量的医疗数据。此外，三阶段的监督微调过程确保了复杂医疗任务的有效知识提取。在MedEvalKit框架下，InfiMed-Foundation-1.7B超越了Qwen2.5VL-3B，而InfiMed-Foundation-4B则超过了HuatuoGPT-V-7B和MedGemma-27B-IT，展示了其在医疗视觉问答和诊断任务方面的优越性能。通过解决数据质量、训练效率和领域特定知识提取的关键问题，我们的研究为医疗领域更可靠和有效的AI驱动解决方案铺平了道路。InfiMed-Foundation-4B模型可在\href{this https URL}{InfiMed-Foundation-4B}获取。 

---
# Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing 

**Title (ZH)**: 评估大语言模型在组合优化中的应用：二维集装箱装载的一阶段和两阶段启发式方法 

**Authors**: Syed Mahbubul Huq, Daniel Brito, Daniel Sikar, Rajesh Mojumder  

**Link**: [PDF](https://arxiv.org/pdf/2509.22255)  

**Abstract**: This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks. 

**Abstract (ZH)**: 本文提出了一种评估大型语言模型（LLMs）在组合优化中的能力的框架，特别针对2D容器打包问题。我们介绍了一种系统的方法，将LLMs与进化算法结合，迭代生成和优化启发式解决方案。通过综合实验，将LLM生成的启发式方法与传统方法（有限首适应和混合首适应）进行比较，我们展示了LLMs能够生成更高效的解决方案，同时需要 fewer 计算资源。我们的评估表明，GPT-4o在两轮迭代内达到最优解，平均容器使用量从16个减少到15个，空间利用率从0.76-0.78提高到0.83。本文对理解LLMs在专门领域中的评估做出了贡献，并为评估LLMs在组合优化任务中的性能建立了基准。 

---
# Clinical Uncertainty Impacts Machine Learning Evaluations 

**Title (ZH)**: 临床不确定性影响机器学习评估 

**Authors**: Simone Lionetti, Fabian Gröger, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Ludovic Amruthalingam, Alexander A. Navarini, Marc Pouly  

**Link**: [PDF](https://arxiv.org/pdf/2509.22242)  

**Abstract**: Clinical dataset labels are rarely certain as annotators disagree and confidence is not uniform across cases. Typical aggregation procedures, such as majority voting, obscure this variability. In simple experiments on medical imaging benchmarks, accounting for the confidence in binary labels significantly impacts model rankings. We therefore argue that machine-learning evaluations should explicitly account for annotation uncertainty using probabilistic metrics that directly operate on distributions. These metrics can be applied independently of the annotations' generating process, whether modeled by simple counting, subjective confidence ratings, or probabilistic response models. They are also computationally lightweight, as closed-form expressions have linear-time implementations once examples are sorted by model score. We thus urge the community to release raw annotations for datasets and to adopt uncertainty-aware evaluation so that performance estimates may better reflect clinical data. 

**Abstract (ZH)**: 临床数据集标签通常具有不确定性，因为注释者存在分歧且不同案例的置信度不一致。传统的聚合方法，如多数投票，会掩盖这种变异性。在医学影像基准测试中的简单实验表明，考虑二分类标签的置信度显著影响模型排名。因此，我们认为机器学习评估应当明确采用概率性度量方法，直接作用于分布，来反映注释不确定性。这些度量方法可以独立于注释生成过程，无论是通过简单的计数、主观置信评分还是概率响应模型。它们还具有计算效率，一旦根据模型得分对示例进行排序，闭式表达式便具有线性时间实现。因此，我们敦促社区释放原始注释，并采用能反映临床数据性能估计的不确定性意识评估方法。 

---
# Log2Plan: An Adaptive GUI Automation Framework Integrated with Task Mining Approach 

**Title (ZH)**: Log2Plan：一种结合任务挖掘方法的自适应GUI自动化框架 

**Authors**: Seoyoung Lee, Seonbin Yoon, Seongbeen Lee, Hyesoo Kim, Joo Yong Sim  

**Link**: [PDF](https://arxiv.org/pdf/2509.22137)  

**Abstract**: GUI task automation streamlines repetitive tasks, but existing LLM or VLM-based planner-executor agents suffer from brittle generalization, high latency, and limited long-horizon coherence. Their reliance on single-shot reasoning or static plans makes them fragile under UI changes or complex tasks. Log2Plan addresses these limitations by combining a structured two-level planning framework with a task mining approach over user behavior logs, enabling robust and adaptable GUI automation. Log2Plan constructs high-level plans by mapping user commands to a structured task dictionary, enabling consistent and generalizable automation. To support personalization and reuse, it employs a task mining approach from user behavior logs that identifies user-specific patterns. These high-level plans are then grounded into low-level action sequences by interpreting real-time GUI context, ensuring robust execution across varying interfaces. We evaluated Log2Plan on 200 real-world tasks, demonstrating significant improvements in task success rate and execution time. Notably, it maintains over 60.0% success rate even on long-horizon task sequences, highlighting its robustness in complex, multi-step workflows. 

**Abstract (ZH)**: GUI任务自动化简化了重复任务，但现有的基于LLM或VLM的规划执行代理在泛化能力、延迟和长期连贯性方面存在局限性。它们依赖于单次推理或静态计划，使其在UI变化或复杂任务面前变得脆弱。Log2Plan通过结合结构化的两层规划框架和基于用户行为日志的任务挖掘方法，解决了这些限制，实现了稳健且适应性强的GUI自动化。Log2Plan通过将用户命令映射到结构化的任务字典中，构建高层次计划，从而实现一致和可泛化的自动化。为支持个性化和复用，它采用从用户行为日志中识别用户特定模式的任务挖掘方法。这些高层次计划随后通过解释实时GUI上下文，被转化为低层操作序列，确保在不同界面之间实现稳健执行。我们在200项真实世界任务上评估了Log2Plan，显示出显著的任务成功率和执行时间改进。值得注意的是，即使在长时间序列的任务中，其成功率仍保持在超过60.0%，突显了其在复杂多步工作流中的稳健性。 

---
# Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements 

**Title (ZH)**: AI能耗的真实验证：CodeCarbon与外部测量的验证 

**Authors**: Raphael Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.22092)  

**Abstract**: Although machine learning (ML) and artificial intelligence (AI) present fascinating opportunities for innovation, their rapid development is also significantly impacting our environment. In response to growing resource-awareness in the field, quantification tools such as the ML Emissions Calculator and CodeCarbon were developed to estimate the energy consumption and carbon emissions of running AI models. They are easy to incorporate into AI projects, however also make pragmatic assumptions and neglect important factors, raising the question of estimation accuracy. This study systematically evaluates the reliability of static and dynamic energy estimation approaches through comparisons with ground-truth measurements across hundreds of AI experiments. Based on the proposed validation framework, investigative insights into AI energy demand and estimation inaccuracies are provided. While generally following the patterns of AI energy consumption, the established estimation approaches are shown to consistently make errors of up to 40%. By providing empirical evidence on energy estimation quality and errors, this study establishes transparency and validates widely used tools for sustainable AI development. It moreover formulates guidelines for improving the state-of-the-art and offers code for extending the validation to other domains and tools, thus making important contributions to resource-aware ML and AI sustainability research. 

**Abstract (ZH)**: 尽管机器学习（ML）和人工智能（AI）为创新带来了令人振奋的机会，其快速发展也在显著影响着我们的环境。鉴于该领域日益增强的资源意识，开发了诸如ML排放计算器和CodeCarbon等量化工具，以估算运行AI模型的能耗和碳排放。尽管这些工具易于集成到AI项目中，但也会做出实用性的假设并忽略重要因素，这引发了对估算准确性的质疑。本研究通过与数百个AI实验的真实测量值进行比较，系统评估了静态和动态能耗估算方法的可靠性，并提供了关于AI能耗需求和估算不准确性的调查见解。尽管一般遵循AI能耗模式，已建立的估算方法仍然显示出高达40%的一致性误差。通过提供关于能耗估算质量和误差的实验证据，本研究提高了透明度并验证了用于可持续AI发展的广泛使用的工具。此外，制定了改进现有技术的指南，并提供了代码以扩展验证至其他领域和工具，从而为资源意识机器学习和AI可持续性研究作出了重要贡献。 

---
# Generalizing Multi-Objective Search via Objective-Aggregation Functions 

**Title (ZH)**: 通过目标聚合函数泛化多目标搜索 

**Authors**: Hadar Peer, Eyal Weiss, Ron Alterovitz, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2509.22085)  

**Abstract**: Multi-objective search (MOS) has become essential in robotics, as real-world robotic systems need to simultaneously balance multiple, often conflicting objectives. Recent works explore complex interactions between objectives, leading to problem formulations that do not allow the usage of out-of-the-box state-of-the-art MOS algorithms. In this paper, we suggest a generalized problem formulation that optimizes solution objectives via aggregation functions of hidden (search) objectives. We show that our formulation supports the application of standard MOS algorithms, necessitating only to properly extend several core operations to reflect the specific aggregation functions employed. We demonstrate our approach in several diverse robotics planning problems, spanning motion-planning for navigation, manipulation and planning fr medical systems under obstacle uncertainty as well as inspection planning, and route planning with different road types. We solve the problems using state-of-the-art MOS algorithms after properly extending their core operations, and provide empirical evidence that they outperform by orders of magnitude the vanilla versions of the algorithms applied to the same problems but without objective aggregation. 

**Abstract (ZH)**: 多目标搜索（MOS）在机器人技术中的应用已经成为必要，因为现实世界的机器人系统需要同时平衡多个常常相互冲突的目标。最近的研究探索了目标间的复杂交互，导致的问题表述不支持使用现成的最先进的MOS算法。在本文中，我们提出了一种通用的问题表述方法，通过隐藏（搜索）目标的聚合函数来优化解决方案目标。我们展示了我们的表述方法支持标准MOS算法的应用，只需适当扩展几个核心操作以反映特定的聚合函数即可。我们通过涵盖导航、操纵以及障碍不确定性下的医疗系统规划等多个领域的机器人规划问题展示了我们的方法，并且通过不同道路类型的路径规划问题进一步验证了我们的方法。我们使用适当扩展核心操作的最先进的MOS算法解决了这些问题，并提供了实验证据，证明与未聚合目标的算法相比，新方法在同类型问题上的性能高出几个数量级。 

---
# A2R: An Asymmetric Two-Stage Reasoning Framework for Parallel Reasoning 

**Title (ZH)**: A2R：一种用于并行推理的非对称两阶段推理框架 

**Authors**: Ziqi Wang, Boye Niu, Zhongli Li, Linghui Meng, Jing Liu, Zhi Zheng, Tong Xu, Hua Wu, Haifeng Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22044)  

**Abstract**: Recent Large Reasoning Models have achieved significant improvements in complex task-solving capabilities by allocating more computation at the inference stage with a "thinking longer" paradigm. Even as the foundational reasoning capabilities of models advance rapidly, the persistent gap between a model's performance in a single attempt and its latent potential, often revealed only across multiple solution paths, starkly highlights the disparity between its realized and inherent capabilities. To address this, we present A2R, an Asymmetric Two-Stage Reasoning framework designed to explicitly bridge the gap between a model's potential and its actual performance. In this framework, an "explorer" model first generates potential solutions in parallel through repeated sampling. Subsequently,a "synthesizer" model integrates these references for a more refined, second stage of reasoning. This two-stage process allows computation to be scaled orthogonally to existing sequential methods. Our work makes two key innovations: First, we present A2R as a plug-and-play parallel reasoning framework that explicitly enhances a model's capabilities on complex questions. For example, using our framework, the Qwen3-8B-distill model achieves a 75% performance improvement compared to its self-consistency baseline. Second, through a systematic analysis of the explorer and synthesizer roles, we identify an effective asymmetric scaling paradigm. This insight leads to A2R-Efficient, a "small-to-big" variant that combines a Qwen3-4B explorer with a Qwen3-8B synthesizer. This configuration surpasses the average performance of a monolithic Qwen3-32B model at a nearly 30% lower cost. Collectively, these results show that A2R is not only a performance-boosting framework but also an efficient and practical solution for real-world applications. 

**Abstract (ZH)**: Recent Large Reasoning Models通过在推理阶段进行更长时间的计算以分配更多计算资源，显著提升了复杂任务解决能力。尽管基础推理能力迅速进步，模型在单次尝试中的性能与其潜在能力之间的持续差距仍然明显，后者通常仅在多种解决方案路径中才显现。为解决这一问题，我们提出A2R，一种非对称两阶段推理框架，旨在明确弥合模型潜在能力与其实际性能之间的差距。在该框架中，“探索者”模型通过重复采样并行生成潜在解决方案，随后，“合成器”模型将这些参考整合为更精细的第二阶段推理。这一两阶段过程允许计算资源与现有顺序方法正交扩展。我们的工作有两个关键创新：首先，我们提出了A2R作为可无缝集成的并行推理框架，可以显式增强模型在复杂问题上的能力。例如，在我们的框架中，Qwen3-8B-distill模型相比其自我一致性基线实现了75%的性能提升。其次，通过系统分析探索者和合成器的角色，我们确定了一种有效的非对称扩展范式。这一洞察导致了A2R-Efficient变种，即使用Qwen3-4B探索者与Qwen3-8B合成器的“小到大”配置，该配置在成本降低近30%的情况下，性能超过了单一的Qwen3-32B模型。综合来看，这些结果表明A2R不仅是一个性能增益框架，也是一个在实际应用中高效且实用的解决方案。 

---
# The Thinking Spectrum: An Emperical Study of Tunable Reasoning in LLMs through Model Merging 

**Title (ZH)**: 思维谱系：通过模型合并对LLMs可调推理的实证研究 

**Authors**: Xiaochong Lan, Yu Zheng, Shiteng Cao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22034)  

**Abstract**: The growing demand for large language models (LLMs) with tunable reasoning capabilities in many real-world applications highlights a critical need for methods that can efficiently produce a spectrum of models balancing reasoning depth and computational cost. Model merging has emerged as a promising, training-free technique to address this challenge by arithmetically combining the weights of a general-purpose model with a specialized reasoning model. While various merging techniques exist, their potential to create a spectrum of models with fine-grained control over reasoning abilities remains largely unexplored. This work presents a large-scale empirical study evaluating a range of model merging techniques across multiple reasoning benchmarks. We systematically vary merging strengths to construct accuracy-efficiency curves, providing the first comprehensive view of the tunable performance landscape. Our findings reveal that model merging offers an effective and controllable method for calibrating the trade-off between reasoning accuracy and token efficiency, even when parent models have highly divergent weight spaces. Crucially, we identify instances of Pareto Improvement, where a merged model achieves both higher accuracy and lower token consumption than one of its parents. Our study provides the first comprehensive analysis of this tunable space, offering practical guidelines for creating LLMs with specific reasoning profiles to meet diverse application demands. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种实际应用中对可调推理能力的需求增长突显了高效生成平衡推理深度与计算成本的模型谱系方法的迫切需求。模型合并作为一种无训练的有前途技术，通过算术合并通用模型和专门推理模型的权重来应对这一挑战。尽管存在多种合并技术，但它们如何实现对推理能力的细微控制仍需进一步探索。本研究展开了一项大规模 empirical 实证研究，评估了多种模型合并技术在多个推理基准上的表现。系统地变化合并强度以构建准确性和效率曲线，首次全面展示了可调性能景观。我们的发现表明，模型合并提供了一种有效的、可控的方法，即使父模型的权重空间差异很大，也能校准推理准确性与标记效率之间的权衡。更重要的是，我们识别出了帕累托改进实例，即合并模型在准确性和标记消耗上均优于其父模型之一。本研究提供了对这一可调空间的首次全面分析，为创建具有特定推理特征的大规模语言模型以满足多样化应用需求提供了实用指南。 

---
# GSM-Agent: Understanding Agentic Reasoning Using Controllable Environments 

**Title (ZH)**: GSM-Agent: 通过可控环境理解代理推理 

**Authors**: Hanlin Zhu, Tianyu Guo, Song Mei, Stuart Russell, Nikhil Ghosh, Alberto Bietti, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21998)  

**Abstract**: As LLMs are increasingly deployed as agents, agentic reasoning - the ability to combine tool use, especially search, and reasoning - becomes a critical skill. However, it is hard to disentangle agentic reasoning when evaluated in complex environments and tasks. Current agent benchmarks often mix agentic reasoning with challenging math reasoning, expert-level knowledge, and other advanced capabilities. To fill this gap, we build a novel benchmark, GSM-Agent, where an LLM agent is required to solve grade-school-level reasoning problems, but is only presented with the question in the prompt without the premises that contain the necessary information to solve the task, and needs to proactively collect that information using tools. Although the original tasks are grade-school math problems, we observe that even frontier models like GPT-5 only achieve 67% accuracy. To understand and analyze the agentic reasoning patterns, we propose the concept of agentic reasoning graph: cluster the environment's document embeddings into nodes, and map each tool call to its nearest node to build a reasoning path. Surprisingly, we identify that the ability to revisit a previously visited node, widely taken as a crucial pattern in static reasoning, is often missing for agentic reasoning for many models. Based on the insight, we propose a tool-augmented test-time scaling method to improve LLM's agentic reasoning performance by adding tools to encourage models to revisit. We expect our benchmark and the agentic reasoning framework to aid future studies of understanding and pushing the boundaries of agentic reasoning. 

**Abstract (ZH)**: 随着大型语言模型被越来越多地用作代理，代理推理——结合工具使用（尤其是搜索）和推理的能力——成为了一项关键技能。然而，在复杂环境和任务中评估代理推理是困难的。当前的代理基准往往会将代理推理与其他具有挑战性的数学推理、专家级知识以及其他先进能力混合在一起。为填补这一空白，我们构建了一个新的基准——GSM-Agent，其中要求大型语言模型代理解决小学水平的推理问题，但在提示中仅呈现问题而未提供包含解决任务所需信息的前提，并需要主动使用工具收集这些信息。尽管原始任务是小学数学问题，我们观察到即使是前沿模型GPT-5也只能达到67%的准确率。为了理解和分析代理推理模式，我们提出了代理推理图的概念：将环境的文档嵌入聚类为节点，并将每个工具调用映射到最近的节点以构建推理路径。令人惊讶的是，我们发现许多模型在代理推理中缺乏在先前访问过的节点间来回的能力，这一能力在静态推理中被视为关键模式。基于这一见解，我们提出了一种工具增强的测试时尺度扩展方法，通过增加工具来鼓励模型进行回溯，从而提高大型语言模型的代理推理性能。我们期望我们的基准和代理推理框架能够有助于未来对代理推理的理解和边界扩展研究。 

---
# Bilinear relational structure fixes reversal curse and enables consistent model editing 

**Title (ZH)**: 双边关系结构纠正反转诅咒并实现一致的模型编辑 

**Authors**: Dong-Kyum Kim, Minsung Kim, Jea Kwon, Nakyeong Yang, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2509.21993)  

**Abstract**: The reversal curse -- a language model's (LM) inability to infer an unseen fact ``B is A'' from a learned fact ``A is B'' -- is widely considered a fundamental limitation. We show that this is not an inherent failure but an artifact of how models encode knowledge. By training LMs from scratch on a synthetic dataset of relational knowledge graphs, we demonstrate that bilinear relational structure emerges in their hidden representations. This structure substantially alleviates the reversal curse, enabling LMs to infer unseen reverse facts. Crucially, we also find that this bilinear structure plays a key role in consistent model editing. When a fact is updated in a LM with this structure, the edit correctly propagates to its reverse and other logically dependent facts. In contrast, models lacking this representation not only suffer from the reversal curse but also fail to generalize edits, further introducing logical inconsistencies. Our results establish that training on a relational knowledge dataset induces the emergence of bilinear internal representations, which in turn enable LMs to behave in a logically consistent manner after editing. This implies that the success of model editing depends critically not just on editing algorithms but on the underlying representational geometry of the knowledge being modified. 

**Abstract (ZH)**: 语言模型的语言反转诅咒——即其从已学习事实“A是B”推出未见事实“B是A”的能力不足——通常被认为是一个根本性的限制。我们展示这并非固有的失败，而是模型编码知识的方式所致。通过从头训练语言模型在合成的关系知识图数据集上，我们证明其隐藏表示中出现了双线性关系结构。这种结构显著缓解了语言反转诅咒，使语言模型能够推断出未见的反事实。最关键的是，我们还发现这种双线性结构在一致的模型编辑中发挥着关键作用。当一个事实被更新时，编辑能够正确定向到其反事实及其他逻辑相关事实。相比之下，缺乏这种表示的模型不仅遭受了语言反转诅咒，还无法泛化编辑，进一步引入逻辑不一致。我们的结果表明，通过关系知识数据集训练诱导出双线性内部表示，进而使模型在编辑后表现出逻辑一致性。这表明模型编辑的成功不仅依赖于编辑算法，还依赖于所修改知识的内在表示几何结构。 

---
# RISK: A Framework for GUI Agents in E-commerce Risk Management 

**Title (ZH)**: GUI代理在电子商务风险管理的框架：RISK 

**Authors**: Renqi Chen, Zeyin Tao, Jianming Guo, Jingzhe Zhu, Yiheng Peng, Qingqing Sun, Tianyi Zhang, Shuai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21982)  

**Abstract**: E-commerce risk management requires aggregating diverse, deeply embedded web data through multi-step, stateful interactions, which traditional scraping methods and most existing Graphical User Interface (GUI) agents cannot handle. These agents are typically limited to single-step tasks and lack the ability to manage dynamic, interactive content critical for effective risk assessment. To address this challenge, we introduce RISK, a novel framework designed to build and deploy GUI agents for this domain. RISK integrates three components: (1) RISK-Data, a dataset of 8,492 single-step and 2,386 multi-step interaction trajectories, collected through a high-fidelity browser framework and a meticulous data curation process; (2) RISK-Bench, a benchmark with 802 single-step and 320 multi-step trajectories across three difficulty levels for standardized evaluation; and (3) RISK-R1, a R1-style reinforcement fine-tuning framework considering four aspects: (i) Output Format: Updated format reward to enhance output syntactic correctness and task comprehension, (ii) Single-step Level: Stepwise accuracy reward to provide granular feedback during early training stages, (iii) Multi-step Level: Process reweight to emphasize critical later steps in interaction sequences, and (iv) Task Level: Level reweight to focus on tasks of varying difficulty. Experiments show that RISK-R1 outperforms existing baselines, achieving a 6.8% improvement in offline single-step and an 8.8% improvement in offline multi-step. Moreover, it attains a top task success rate of 70.5% in online evaluation. RISK provides a scalable, domain-specific solution for automating complex web interactions, advancing the state of the art in e-commerce risk management. 

**Abstract (ZH)**: 电子商务风险管理需要通过多步、有状态的交互聚合多元且深度嵌入的网络数据，传统抓取方法和现有的大多数图形用户界面（GUI）代理无法处理。这些代理通常局限于单步任务，缺乏管理对有效风险评估至关重要的动态交互内容的能力。为解决这一挑战，我们提出了一种名为RISK的新型框架，用于构建和部署此类领域的GUI代理。RISK集成了三个组件：（1）RISK-Data，一个包含8,492条单步和2,386条多步交互轨迹的数据集，通过高保真浏览器框架和细致的数据整理过程收集；（2）RISK-Bench，一个基准测试集，包含320条多步和802条单步交互轨迹，分为三个难度级别，用于标准化评估；以及（3）RISK-R1，一种考虑四大方面的R1风格强化微调框架：输出格式、单步级别、多步级别和任务级别。实验表明，RISK-R1在单步和多步交互方面均优于现有基线，分别提高了6.8%和8.8%。此外，在线评估中其任务成功率高达70.5%。RISK提供了一种针对Web复杂交互的大规模领域特定自动化解决方案，推动了电子商务风险管理领域的进步。 

---
# CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration 

**Title (ZH)**: CoBel-World: 利用大规模语言模型推理构建协作信念世界以优化具身多智能体协作 

**Authors**: Zhimin Wang, Shaokang He, Duo Wu, Jinghe Wang, Linjia Kang, Jing Yu, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21981)  

**Abstract**: Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents -- a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a collaborative belief world -- an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse open-world task knowledge into structured beliefs via a symbolic belief language, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 22-60% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems. 

**Abstract (ZH)**: Effective Real-World 多agent 协作需要准确的规划能力和推理能力——在部分可观测环境中避免误协调和冗余通信的关键能力。由于大型语言模型（LLMs）的强大规划和推理能力，它们已成为协作任务解决的有前途的自主代理。然而，现有的LLM协作框架忽视了它们动态意图推理的潜力，从而产生了不一致的计划和冗余通信，降低了协作效率。为了弥补这一差距，我们提出了 CoBel-World，一种新型框架，为LLM代理配备协作信念世界——一种同时建模物理环境和协作者心理状态的内部表示。CoBel-World 使代理能够通过符号信念语言将开放世界的任务知识解析为结构化的信念，并通过LLM推理进行零样本贝叶斯式信念更新。这使代理能够主动检测潜在的误协调（例如，冲突的计划）并适应性地通信。在具有挑战性的体感基准测试（即TDW-MAT和C-WAH）上，CoBel-World 将通信成本降低了22-60%，提高了4-28%的任务完成效率，相较最强基线。我们的结果表明，显式的、意图感知的信念建模对于基于LLM的多agent系统中的高效和类人协作至关重要。 

---
# Outlier Detection in Plantar Pressure: Human-Centered Comparison of Statistical Parametric Mapping and Explainable Machine Learning 

**Title (ZH)**: 足底压力异常检测：统计参数映射与可解释机器学习的人本比较 

**Authors**: Carlo Dindorf, Jonas Dully, Steven Simon, Dennis Perchthaler, Stephan Becker, Hannah Ehmann, Kjell Heitmann, Bernd Stetter, Christian Diers, Michael Fröhlich  

**Link**: [PDF](https://arxiv.org/pdf/2509.21943)  

**Abstract**: Plantar pressure mapping is essential in clinical diagnostics and sports science, yet large heterogeneous datasets often contain outliers from technical errors or procedural inconsistencies. Statistical Parametric Mapping (SPM) provides interpretable analyses but is sensitive to alignment and its capacity for robust outlier detection remains unclear. This study compares an SPM approach with an explainable machine learning (ML) approach to establish transparent quality-control pipelines for plantar pressure datasets. Data from multiple centers were annotated by expert consensus and enriched with synthetic anomalies resulting in 798 valid samples and 2000 outliers. We evaluated (i) a non-parametric, registration-dependent SPM approach and (ii) a convolutional neural network (CNN), explained using SHapley Additive exPlanations (SHAP). Performance was assessed via nested cross-validation; explanation quality via a semantic differential survey with domain experts. The ML model reached high accuracy and outperformed SPM, which misclassified clinically meaningful variations and missed true outliers. Experts perceived both SPM and SHAP explanations as clear, useful, and trustworthy, though SPM was assessed less complex. These findings highlight the complementary potential of SPM and explainable ML as approaches for automated outlier detection in plantar pressure data, and underscore the importance of explainability in translating complex model outputs into interpretable insights that can effectively inform decision-making. 

**Abstract (ZH)**: 基于解释性机器学习方法的足底压力数据质量控制管道研究：SPM与可解释ML在自动检测异常值中的互补潜力 

---
# DyRo-MCTS: A Robust Monte Carlo Tree Search Approach to Dynamic Job Shop Scheduling 

**Title (ZH)**: DyRo-MCTS：一种用于动态车间调度的稳健蒙特卡洛树搜索方法 

**Authors**: Ruiqi Chen, Yi Mei, Fangfang Zhang, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21902)  

**Abstract**: Dynamic job shop scheduling, a fundamental combinatorial optimisation problem in various industrial sectors, poses substantial challenges for effective scheduling due to frequent disruptions caused by the arrival of new jobs. State-of-the-art methods employ machine learning to learn scheduling policies offline, enabling rapid responses to dynamic events. However, these offline policies are often imperfect, necessitating the use of planning techniques such as Monte Carlo Tree Search (MCTS) to improve performance at online decision time. The unpredictability of new job arrivals complicates online planning, as decisions based on incomplete problem information are vulnerable to disturbances. To address this issue, we propose the Dynamic Robust MCTS (DyRo-MCTS) approach, which integrates action robustness estimation into MCTS. DyRo-MCTS guides the production environment toward states that not only yield good scheduling outcomes but are also easily adaptable to future job arrivals. Extensive experiments show that DyRo-MCTS significantly improves the performance of offline-learned policies with negligible additional online planning time. Moreover, DyRo-MCTS consistently outperforms vanilla MCTS across various scheduling scenarios. Further analysis reveals that its ability to make robust scheduling decisions leads to long-term, sustainable performance gains under disturbances. 

**Abstract (ZH)**: 动态车间调度：一种因新任务的频繁 arrival 而在各类工业部门中提出的根本性组合优化问题，由于缺乏有效的调度策略而带来了显著挑战。最先进的方法利用机器学习在离线阶段学习调度策略，从而能够快速应对动态事件。然而，这些离线策略往往不够完善，需要结合如蒙特卡洛树搜索（MCTS）等规划技术，在线决策时提高性能。新任务 arrival 的不可预测性增加了在线规划的复杂性，基于不完整信息的决策容易受到干扰的影响。为解决这一问题，我们提出了一种动态鲁棒 MCTS（DyRo-MCTS）方法，将动作鲁棒性估计整合到 MCTS 中。DyRo-MCTS 引导生产环境朝着既能产生良好调度结果又能容易适应未来任务 arrival 的状态发展。大量实验表明，DyRo-MCTS 显著提高了离线学习策略的性能，且几乎不增加在线规划时间。此外，DyRo-MCTS 在各种调度场景中均优于基本 MCTS。进一步的分析表明，它能够做出鲁棒调度决策的能力，使其能够在干扰下获得长期且可持续的性能提升。 

---
# GenesisGeo: Technical Report 

**Title (ZH)**: GenesisGeo: 技术报告 

**Authors**: Minfeng Zhu, Zi Wang, Sizhe Ji, Zhengtong Du, Junming Ke, Xiao Deng, Zanlang Yin, Xiuqi Huang, Heyu Wang, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21896)  

**Abstract**: We present GenesisGeo, an automated theorem prover in Euclidean geometry. We have open-sourced a large-scale geometry dataset of 21.8 million geometric problems, over 3 million of which contain auxiliary constructions. Specially, we significantly accelerate the symbolic deduction engine DDARN by 120x through theorem matching, combined with a C++ implementation of its core components. Furthermore, we build our neuro-symbolic prover, GenesisGeo, upon Qwen3-0.6B-Base, which solves 24 of 30 problems (IMO silver medal level) in the IMO-AG-30 benchmark using a single model, and achieves 26 problems (IMO gold medal level) with a dual-model ensemble. 

**Abstract (ZH)**: 我们提出了GenesisGeo，一个欧几里得几何自动定理证明器。我们开源了一个大规模几何数据集，包含2180万几何问题，其中超过300万包含辅助构造。特别地，我们通过定理匹配将符号推理引擎DDARN的速度提高120倍，并结合了其核心组件的C++实现。此外，我们基于Qwen3-0.6B-Base构建了神经-符号证明器GenesisGeo，该证明器使用单模型解决了IMO-AG-30基准中的24个问题（IMO银牌水平），并使用双模型集成解决了26个问题（IMO金牌水平）。 

---
# TRACE: Learning to Compute on Graphs 

**Title (ZH)**: TRACE: 学习在图上进行计算 

**Authors**: Ziyang Zheng, Jiaying Zhu, Jingyi Zhou, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21886)  

**Abstract**: Learning to compute, the ability to model the functional behavior of a computational graph, is a fundamental challenge for graph representation learning. Yet, the dominant paradigm is architecturally mismatched for this task. This flawed assumption, central to mainstream message passing neural networks (MPNNs) and their conventional Transformer-based counterparts, prevents models from capturing the position-aware, hierarchical nature of computation. To resolve this, we introduce \textbf{TRACE}, a new paradigm built on an architecturally sound backbone and a principled learning objective. First, TRACE employs a Hierarchical Transformer that mirrors the step-by-step flow of computation, providing a faithful architectural backbone that replaces the flawed permutation-invariant aggregation. Second, we introduce \textbf{function shift learning}, a novel objective that decouples the learning problem. Instead of predicting the complex global function directly, our model is trained to predict only the \textit{function shift}, the discrepancy between the true global function and a simple local approximation that assumes input independence. We validate this paradigm on electronic circuits, one of the most complex and economically critical classes of computational graphs. Across a comprehensive suite of benchmarks, TRACE substantially outperforms all prior architectures. These results demonstrate that our architecturally-aligned backbone and decoupled learning objective form a more robust paradigm for the fundamental challenge of learning to compute on graphs. 

**Abstract (ZH)**: 学习计算的能力，即建模计算图的功能行为，是图表示学习中一个基本的挑战。然而，主导的范式在架构上与这一任务不匹配。这一根本性的假设，是主流消息传递神经网络（MPNNs）及其传统的基于Transformer的变体的核心，阻碍了模型捕捉计算中位置感知和层次化的本质。为了解决这一问题，我们提出了一个名为TRACE的新范式，该范式基于一个架构合理的骨干和一个有原则的学习目标。首先，TRACE使用了一个级联Transformer，该模型模仿了计算的逐步流程，提供了一个忠实的架构基础，用以替代错误的置换不变聚合。其次，我们引入了功能偏移学习这一新颖的目标，将学习问题解耦。我们的模型不是直接预测复杂的全局函数，而是仅预测真全局函数与假设输入独立的简单局部近似的差异，即功能偏移。我们已在电子电路这一最复杂和经济上至关重要的计算图类别上验证了这一范式。在一系列全面的基准测试中，TRACE显著优于所有先前的架构。这些结果表明，我们的架构对齐的骨干和解耦的学习目标构成了一个更为稳健的范式，用于解决在图上学习计算这一基本挑战。 

---
# Reimagining Agent-based Modeling with Large Language Model Agents via Shachi 

**Title (ZH)**: 使用Shachi重塑基于代理的建模方法，通过大型语言模型代理 

**Authors**: So Kuroki, Yingtao Tian, Kou Misaki, Takashi Ikegami, Takuya Akiba, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21862)  

**Abstract**: The study of emergent behaviors in large language model (LLM)-driven multi-agent systems is a critical research challenge, yet progress is limited by a lack of principled methodologies for controlled experimentation. To address this, we introduce Shachi, a formal methodology and modular framework that decomposes an agent's policy into core cognitive components: Configuration for intrinsic traits, Memory for contextual persistence, and Tools for expanded capabilities, all orchestrated by an LLM reasoning engine. This principled architecture moves beyond brittle, ad-hoc agent designs and enables the systematic analysis of how specific architectural choices influence collective behavior. We validate our methodology on a comprehensive 10-task benchmark and demonstrate its power through novel scientific inquiries. Critically, we establish the external validity of our approach by modeling a real-world U.S. tariff shock, showing that agent behaviors align with observed market reactions only when their cognitive architecture is appropriately configured with memory and tools. Our work provides a rigorous, open-source foundation for building and evaluating LLM agents, aimed at fostering more cumulative and scientifically grounded research. 

**Abstract (ZH)**: 大型语言模型（LLM）驱动的多agent系统中涌现行为的研究：一种形式化的研究方法和模块化框架 

---
# DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents 

**Title (ZH)**: DeepTravel：自主旅行规划代理的端到端代理 reinforcement 学习框架 

**Authors**: Yansong Ning, Rui Liu, Jun Wang, Kai Chen, Wei Li, Jun Fang, Kan Zheng, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21842)  

**Abstract**: Travel planning (TP) agent has recently worked as an emerging building block to interact with external tools and resources for travel itinerary generation, ensuring enjoyable user experience. Despite its benefits, existing studies rely on hand craft prompt and fixed agent workflow, hindering more flexible and autonomous TP agent. This paper proposes DeepTravel, an end to end agentic reinforcement learning framework for building autonomous travel planning agent, capable of autonomously planning, executing tools, and reflecting on tool responses to explore, verify, and refine intermediate actions in multi step reasoning. To achieve this, we first construct a robust sandbox environment by caching transportation, accommodation and POI data, facilitating TP agent training without being constrained by real world APIs limitations (e.g., inconsistent outputs). Moreover, we develop a hierarchical reward modeling system, where a trajectory level verifier first checks spatiotemporal feasibility and filters unsatisfied travel itinerary, and then the turn level verifier further validate itinerary detail consistency with tool responses, enabling efficient and precise reward service. Finally, we propose the reply augmented reinforcement learning method that enables TP agent to periodically replay from a failures experience buffer, emerging notable agentic capacity. We deploy trained TP agent on DiDi Enterprise Solutions App and conduct comprehensive online and offline evaluations, demonstrating that DeepTravel enables small size LLMs (e.g., Qwen3 32B) to significantly outperform existing frontier LLMs such as OpenAI o1, o3 and DeepSeek R1 in travel planning tasks. 

**Abstract (ZH)**: 基于深度学习的自主旅行规划代理框架：DeepTravel 

---
# Axiomatic Choice and the Decision-Evaluation Paradox 

**Title (ZH)**: 公理化选择与决策-评估悖论 

**Authors**: Ben Abramowitz, Nicholas Mattei  

**Link**: [PDF](https://arxiv.org/pdf/2509.21836)  

**Abstract**: We introduce a framework for modeling decisions with axioms that are statements about decisions, e.g., ethical constraints. Using our framework we define a taxonomy of decision axioms based on their structural properties and demonstrate a tension between the use of axioms to make decisions and the use of axioms to evaluate decisions which we call the Decision-Evaluation Paradox. We argue that the Decision-Evaluation Paradox arises with realistic axiom structures, and the paradox illuminates why one must be exceptionally careful when training models on decision data or applying axioms to make and evaluate decisions. 

**Abstract (ZH)**: 我们介绍了一个框架，用于使用关于决策的公理来建模决策，例如道德约束。使用该框架，我们基于公理的结构性质定义了决策公理的分类，并展示了使用公理进行决策和使用公理评估决策之间的紧张关系，称之为决策-评估悖论。我们认为决策-评估悖论存在于现实的公理结构中，并且悖论阐明了在使用决策数据训练模型或将公理应用于决策及其评估时必须极其谨慎的原因。 

---
# DS-STAR: Data Science Agent via Iterative Planning and Verification 

**Title (ZH)**: DS-STAR: 数据科学代理通过迭代规划与验证 

**Authors**: Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, Jinwoo Shin, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2509.21825)  

**Abstract**: Data science, which transforms raw data into actionable insights, is critical for data-driven decision-making. However, these tasks are often complex, involving steps for exploring multiple data sources and synthesizing findings to deliver insightful answers. While large language models (LLMs) show significant promise in automating this process, they often struggle with heterogeneous data formats and generate sub-optimal analysis plans, as verifying plan sufficiency is inherently difficult without ground-truth labels for such open-ended tasks. To overcome these limitations, we introduce DS-STAR, a novel data science agent. Specifically, DS-STAR makes three key contributions: (1) a data file analysis module that automatically explores and extracts context from diverse data formats, including unstructured types; (2) a verification step where an LLM-based judge evaluates the sufficiency of the analysis plan at each stage; and (3) a sequential planning mechanism that starts with a simple, executable plan and iteratively refines it based on the DS-STAR's feedback until its sufficiency is verified. This iterative refinement allows DS-STAR to reliably navigate complex analyses involving diverse data sources. Our experiments show that DS-STAR achieves state-of-the-art performance across three challenging benchmarks: DABStep, KramaBench, and DA-Code. Moreover, DS-STAR particularly outperforms baselines on hard tasks that require processing multiple data files with heterogeneous formats. 

**Abstract (ZH)**: 数据科学，即通过将原始数据转换为可操作的洞察，对于基于数据的决策至关重要。然而，这些任务往往非常复杂，涉及探索多个数据源并综合发现以提供有洞察力的答案。虽然大型语言模型（LLMs）在自动化这一过程方面显示出巨大的潜力，但它们通常难以处理异构数据格式，并生成次优化的分析计划，因为验证计划的充分性在没有此类开放任务的真实标签的情况下是固有的困难。为克服这些限制，我们引入了DS-STAR，这是一种新颖的数据科学代理。具体而言，DS-STAR 作出三大贡献：（1）数据文件分析模块，能够自动探索和从各种数据格式（包括非结构化格式）中提取背景信息；（2）验证步骤，其中基于LLM的裁判员评估每个阶段分析计划的充分性；以及（3）一种顺序规划机制，从一个简单可执行的计划开始，并根据DS-STAR的反馈逐步优化该计划，直到其充分性被验证。这种逐步优化使得DS-STAR能够在涉及多种数据源的复杂分析中可靠导航。我们的实验表明，DS-STAR在三个具有挑战性的基准测试：DABStep、KramaBench和DA-Code中取得了最先进的性能。此外，DS-STAR特别在需要处理具有异构格式的多个数据文件的困难任务中超过了基线。 

---
# ProRe: A Proactive Reward System for GUI Agents via Reasoner-Actor Collaboration 

**Title (ZH)**: ProRe：基于推理者-演员协作的主动奖励系统 for GUI代理 

**Authors**: Gaole Dai, Shiqi Jiang, Ting Cao, Yuqing Yang, Yuanchun Li, Rui Tan, Mo Li, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21823)  

**Abstract**: Reward is critical to the evaluation and training of large language models (LLMs). However, existing rule-based or model-based reward methods struggle to generalize to GUI agents, where access to ground-truth trajectories or application databases is often unavailable, and static trajectory-based LLM-as-a-Judge approaches suffer from limited accuracy. To address these challenges, we propose ProRe, a proactive reward system that leverages a general-purpose reasoner and domain-specific evaluator agents (actors). The reasoner schedules targeted state probing tasks, which the evaluator agents then execute by actively interacting with the environment to collect additional observations. This enables the reasoner to assign more accurate and verifiable rewards to GUI agents. Empirical results on over 3K trajectories demonstrate that ProRe improves reward accuracy and F1 score by up to 5.3% and 19.4%, respectively. Furthermore, integrating ProRe with state-of-the-art policy agents yields a success rate improvement of up to 22.4%. 

**Abstract (ZH)**: 基于奖励的方法对于大型语言模型（LLMs）的评估和训练至关重要。然而，现有的基于规则或模型的奖励方法难以泛化到GUI代理中，因为这类代理往往无法访问真实轨迹或应用程序数据库，而基于静态轨迹的方法作为LLM的评估者则存在准确度有限的问题。为应对这些挑战，我们提出了一种名为ProRe的主动奖励系统，该系统利用通用推理器和领域特定的评估代理（演员）。推理器安排针对性的状态探测任务，评估代理则通过主动与环境交互来执行这些任务以收集额外的观察信息。这使得推理器能够为GUI代理分配更准确和可验证的奖励。在超过3000个轨迹的实证结果表明，ProRe将奖励准确性和F1分数分别提高了5.3%和19.4%。此外，将ProRe与最先进的策略代理结合使用，可将成功率提高22.4%。 

---
# D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents 

**Title (ZH)**: D-Artemis: 一种移动GUI多代理的反思性认知框架 

**Authors**: Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Naiqiang Tan, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21799)  

**Abstract**: Graphical User Interface (GUI) agents aim to automate a wide spectrum of human tasks by emulating user interaction. Despite rapid advancements, current approaches are hindered by several critical challenges: data bottleneck in end-to-end training, high cost of delayed error detection, and risk of contradictory guidance. Inspired by the human cognitive loop of Thinking, Alignment, and Reflection, we present D-Artemis -- a novel deliberative framework in this paper. D-Artemis leverages a fine-grained, app-specific tip retrieval mechanism to inform its decision-making process. It also employs a proactive Pre-execution Alignment stage, where Thought-Action Consistency (TAC) Check module and Action Correction Agent (ACA) work in concert to mitigate the risk of execution failures. A post-execution Status Reflection Agent (SRA) completes the cognitive loop, enabling strategic learning from experience. Crucially, D-Artemis enhances the capabilities of general-purpose Multimodal large language models (MLLMs) for GUI tasks without the need for training on complex trajectory datasets, demonstrating strong generalization. D-Artemis establishes new state-of-the-art (SOTA) results across both major benchmarks, achieving a 75.8% success rate on AndroidWorld and 96.8% on ScreenSpot-V2. Extensive ablation studies further demonstrate the significant contribution of each component to the framework. 

**Abstract (ZH)**: 图形用户界面（GUI）代理旨在通过模拟用户交互来自动化广泛的 humano 任务。受人类认知循环（思考、对齐和反思）的启发，我们提出了 D-Artemis —— 一种新颖的反思框架。D-Artemis 利用细粒度的、特定于应用程序的提示检索机制来指导其决策过程。它还采用了一种主动的预执行对齐阶段，其中思路-行动一致性（TAC）检查模块和行动纠正代理（ACA）协同工作，以减轻执行失败的风险。在执行后的状态反思代理（SRA）完成认知循环，使代理能够从经验中进行战略学习。关键的是，D-Artemis 无需针对复杂轨迹数据集进行训练，即可增强通用多模态大语言模型（MLLM）在 GUI 任务中的能力，显示出强大的泛化能力。D-Artemis 在两个主要基准测试中建立了新的最佳结果，在 AndroidWorld 中达到了 75.8% 的成功率，在 ScreenSpot-V2 中达到了 96.8%。广泛的消融研究进一步证明了框架中每个组件的重要贡献。 

---
# Benchmarking MLLM-based Web Understanding: Reasoning, Robustness and Safety 

**Title (ZH)**: 基于MLLM的网络理解基准测试：推理、稳健性和安全性 

**Authors**: Junliang Liu, Jingyu Xiao, Wenxin Tang, Wenxuan Wang, Zhixian Wang, Minrui Zhang, Shuanghe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21782)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly positioned as AI collaborators for building complex web-related applications like GUI agents and front-end code generation. However, existing benchmarks largely emphasize visual perception or UI code generation, showing insufficient evaluation on the reasoning, robustness and safety capability required for end-to-end web applications. To bridge the gap, we introduce a comprehensive web understanding benchmark, named WebRSSBench, that jointly evaluates Reasoning, Robustness, and Safety across eight tasks, such as position relationship reasoning, color robustness, and safety critical detection, etc. The benchmark is constructed from 729 websites and contains 3799 question answer pairs that probe multi-step inference over page structure, text, widgets, and safety-critical interactions. To ensure reliable measurement, we adopt standardized prompts, deterministic evaluation scripts, and multi-stage quality control combining automatic checks with targeted human verification. We evaluate 12 MLLMs on WebRSSBench. The results reveal significant gaps, models still struggle with compositional and cross-element reasoning over realistic layouts, show limited robustness when facing perturbations in user interfaces and content such as layout rearrangements or visual style shifts, and are rather conservative in recognizing and avoiding safety critical or irreversible actions. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs） increasingly positioned as AI collaborators for building complex web-related applications like GUI agents and front-end code generation. However, existing benchmarks largely emphasize visual perception or UI code generation, showing insufficient evaluation on the reasoning, robustness, and safety capability required for end-to-end web applications. To bridge the gap, we introduce a comprehensive web understanding benchmark, named WebRSSBench, that jointly evaluates Reasoning, Robustness, and Safety across eight tasks, such as position relationship reasoning, color robustness, and safety-critical detection, etc. The benchmark is constructed from 729 websites and contains 3799 question answer pairs that probe multi-step inference over page structure, text, widgets, and safety-critical interactions. To ensure reliable measurement, we adopt standardized prompts, deterministic evaluation scripts, and multi-stage quality control combining automatic checks with targeted human verification. We evaluate 12 MLLMs on WebRSSBench. The results reveal significant gaps, models still struggle with compositional and cross-element reasoning over realistic layouts, show limited robustness when facing perturbations in user interfaces and content such as layout rearrangements or visual style shifts, and are rather conservative in recognizing and avoiding safety critical or irreversible actions. Our code is available at this https URL. 

---
# UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios 

**Title (ZH)**: UltraHorizon：评估代理在超长时距场景能力的基准测试 

**Authors**: Haotian Luo, Huaisong Zhang, Xuelin Zhang, Haoyu Wang, Zeyu Qin, Wenjie Lu, Guozheng Ma, Haiying He, Yingsha Xie, Qiyang Zhou, Zixuan Hu, Hongze Mi, Yibo Wang, Naiqiang Tan, Hong Chen, Yi R. Fung, Chun Yuan, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21766)  

**Abstract**: Autonomous agents have recently achieved remarkable progress across diverse domains, yet most evaluations focus on short-horizon, fully observable tasks. In contrast, many critical real-world tasks, such as large-scale software development, commercial investment, and scientific discovery, unfold in long-horizon and partially observable scenarios where success hinges on sustained reasoning, planning, memory management, and tool use. Existing benchmarks rarely capture these long-horizon challenges, leaving a gap in systematic evaluation. To bridge this gap, we introduce \textbf{UltraHorizon} a novel benchmark that measures the foundational capabilities essential for complex real-world challenges. We use exploration as a unifying task across three distinct environments to validate these core competencies. Agents are designed in long-horizon discovery tasks where they must iteratively uncover hidden rules through sustained reasoning, planning, memory and tools management, and interaction with environments. Under the heaviest scale setting, trajectories average \textbf{200k+} tokens and \textbf{400+} tool calls, whereas in standard configurations they still exceed \textbf{35k} tokens and involve more than \textbf{60} tool calls on average. Our extensive experiments reveal that LLM-agents consistently underperform in these settings, whereas human participants achieve higher scores, underscoring a persistent gap in agents' long-horizon abilities. We also observe that simple scaling fails in our task. To better illustrate the failure of agents, we conduct an in-depth analysis of collected trajectories. We identify eight types of errors and attribute them to two primary causes: in-context locking and functional fundamental capability gaps. \href{this https URL}{Our code will be available here.} 

**Abstract (ZH)**: 自主代理在多元领域取得了显著进展，但大多数评估主要集中在短时间内的完全可观测任务上。相比之下，许多关键的现实世界任务，如大规模软件开发、商业投资和科学发现，发生在长时间跨度和部分可观测的场景中，成功取决于持续的推理、规划、记忆管理和工具使用。现有的基准测试很少捕捉到这些长时间跨度的挑战，留下了系统评估的缺口。为了弥补这一缺口，我们引入了**UltraHorizon**这一新型基准测试，用于衡量对于复杂现实挑战必不可少的基础能力。我们利用探索作为统一任务，在三个不同的环境中验证这些核心能力。代理在长时间跨度的发现任务中设计，需要通过持续的推理、规划、记忆和工具管理以及与环境的互动，逐步揭露隐藏的规则。在最极端的规模设置下，轨迹平均包含超过**200k**个令牌和**400**多次工具调用，而在标准配置下，仍然超过**35k**个令牌，并且平均涉及超过**60**次工具调用。我们的大量实验表明，在这些设置中，LLM-代理表现一致不佳，而人类参与者则取得更高分数，突显了代理在长时间跨度能力上的持续差距。我们也观察到简单的扩展在此任务中不起作用。为了更好地说明代理的失败，我们对收集的轨迹进行了深入分析。我们识别出八种类型错误，并将其归因于两大主要原因：上下文锁定和功能基本能力差距。 

---
# Lifelong Learning with Behavior Consolidation for Vehicle Routing 

**Title (ZH)**: 基于行为巩固的终身学习车辆路径规划 

**Authors**: Jiyuan Pei, Yi Mei, Jialin Liu, Mengjie Zhang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21765)  

**Abstract**: Recent neural solvers have demonstrated promising performance in learning to solve routing problems. However, existing studies are primarily based on one-off training on one or a set of predefined problem distributions and scales, i.e., tasks. When a new task arises, they typically rely on either zero-shot generalization, which may be poor due to the discrepancies between the new task and the training task(s), or fine-tuning the pretrained solver on the new task, which possibly leads to catastrophic forgetting of knowledge acquired from previous tasks. This paper explores a novel lifelong learning paradigm for neural VRP solvers, where multiple tasks with diverse distributions and scales arise sequentially over time. Solvers are required to effectively and efficiently learn to solve new tasks while maintaining their performance on previously learned tasks. Consequently, a novel framework called Lifelong Learning Router with Behavior Consolidation (LLR-BC) is proposed. LLR-BC consolidates prior knowledge effectively by aligning behaviors of the solver trained on a new task with the buffered ones in a decision-seeking way. To encourage more focus on crucial experiences, LLR-BC assigns greater consolidated weights to decisions with lower confidence. Extensive experiments on capacitated vehicle routing problems and traveling salesman problems demonstrate LLR-BC's effectiveness in training high-performance neural solvers in a lifelong learning setting, addressing the catastrophic forgetting issue, maintaining their plasticity, and improving zero-shot generalization ability. 

**Abstract (ZH)**: 最近的神经网络求解器在学习解决路由问题方面展现了有前景的性能。然而，现有研究主要基于单一训练或一组预定义的问题分布和规模进行训练。当出现新的任务时，它们通常依赖于零样本泛化或对预训练解算器进行微调，这可能导致对之前任务所学知识的灾难性遗忘。本文探索了一种新颖的终身学习范式，适用于神经VRP解算器，多种具有多样分布和规模的任务将随时间顺序出现。解算器需要在有效高效地学习解决新任务的同时保持对之前学习任务的性能。因此，提出了一种名为Lifelong Learning Router with Behavior Consolidation (LLR-BC)的新框架。LLR-BC通过决策导向的方式，有效整合新任务训练解算器的行为与缓存的行为。为了更关注关键经验，LLR-BC对低置信度的决策赋予更大的整合权重。在车载路由问题和旅行商问题上的广泛实验表明，LLR-BC能够在终身学习设置中训练高性能的神经解算器，解决了灾难性遗忘问题，保持其可塑性，并提高零样本泛化能力。 

---
# Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts 

**Title (ZH)**: 忆再现：通过重用思想实现高效推理 

**Authors**: Ammar Ahmed, Azal Ahmad Khan, Ayaan Ahmad, Sheng Di, Zirui Liu, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.21743)  

**Abstract**: Large reasoning models improve accuracy by producing long reasoning traces, but this inflates latency and cost, motivating inference-time efficiency. We propose Retrieval-of-Thought (RoT), which reuses prior reasoning as composable ``thought" steps to guide new problems. RoT organizes steps into a thought graph with sequential and semantic edges to enable fast retrieval and flexible recombination. At inference, RoT retrieves query-relevant nodes and applies reward-guided traversal to assemble a problem-specific template that guides generation. This dynamic template reuse reduces redundant exploration and, therefore, reduces output tokens while preserving accuracy. We evaluate RoT on reasoning benchmarks with multiple models, measuring accuracy, token usage, latency, and memory overhead. Findings show small prompt growth but substantial efficiency gains, with RoT reducing output tokens by up to 40%, inference latency by 82%, and cost by 59% while maintaining accuracy. RoT establishes a scalable paradigm for efficient LRM reasoning via dynamic template construction through retrieval. 

**Abstract (ZH)**: Large Reasoning Models Improve Accuracy by Producing Long Reasoning Traces, but This Inflates Latency and Cost, Motivating Inference-Time Efficiency: Retrieval-of-Thought (RoT) Enables Efficient Reasoning via Dynamic Template Construction 

---
# Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Online Preference Optimization 

**Title (ZH)**: Align2Speak：通过ASR引导的在线偏好优化提高低资源语言的TTS 

**Authors**: Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Roy Fejgin, Ryan Langman, Mikyas Desta, Leili Tavabi, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21718)  

**Abstract**: Developing high-quality text-to-speech (TTS) systems for low-resource languages is challenging due to the scarcity of paired text and speech data. In contrast, automatic speech recognition (ASR) models for such languages are often more accessible, owing to large-scale multilingual pre-training efforts. We propose a framework based on Group Relative Policy Optimization (GRPO) to adapt an autoregressive, multilingual TTS model to new languages. Our method first establishes a language-agnostic foundation for TTS synthesis by training a multilingual baseline with International Phonetic Alphabet (IPA) tokens. Next, we fine-tune this model on limited paired data of the new languages to capture the target language's prosodic features. Finally, we apply GRPO to optimize the model using only unpaired text and speaker prompts, guided by a multi-objective reward from pretrained ASR, speaker verification, and audio quality estimation models. Experiments demonstrate that this pipeline produces intelligible and speaker-consistent speech in low-resource languages, substantially outperforming fine-tuning alone. Furthermore, our GRPO-based framework also improves TTS performance in high-resource languages, surpassing offline alignment methods such as Direct Preference Optimization (DPO) yielding superior intelligibility, speaker similarity, and audio quality. 

**Abstract (ZH)**: 开发低资源语言的高质量文本到语音(TTS)系统具有挑战性，因为配对的文本和语音数据稀缺。相比之下，这些语言的自动语音识别(ASR)模型由于大规模多语言预训练努力通常更具 accesibility。我们提出了一种基于Group Relative Policy Optimization (GRPO)的框架，以适应新的语言的自回归、多语言TTS模型。该方法首先通过使用国际音标(IPA)标记训练多语言基线来建立一个语言无关的基础，以进行TTS合成功能。接着，该模型在新语言的有限配对数据上进行微调，以捕捉目标语言的音韵特征。最后，我们利用GRPO仅使用未配对的文本和讲话者提示来优化模型，该优化由预训练的ASR、说话人验证和音频质量估计模型提供多目标奖励指导。实验表明，该流水线在低资源语言中产生了可理解且说话者一致的语音，并显著优于仅进行微调的方法。此外，基于GRPO的框架还改善了高资源语言的TTS性能，超过离线对齐方法如Direct Preference Optimization (DPO)，在可理解性、说话者相似性和音频质量方面表现更优。 

---
# Can AI Perceive Physical Danger and Intervene? 

**Title (ZH)**: AI能否感知物理危险并干预？ 

**Authors**: Abhishek Jindal, Dmitry Kalashnikov, Oscar Chang, Divya Garikapati, Anirudha Majumdar, Pierre Sermanet, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.21651)  

**Abstract**: When AI interacts with the physical world -- as a robot or an assistive agent -- new safety challenges emerge beyond those of purely ``digital AI". In such interactions, the potential for physical harm is direct and immediate. How well do state-of-the-art foundation models understand common-sense facts about physical safety, e.g. that a box may be too heavy to lift, or that a hot cup of coffee should not be handed to a child? In this paper, our contributions are three-fold: first, we develop a highly scalable approach to continuous physical safety benchmarking of Embodied AI systems, grounded in real-world injury narratives and operational safety constraints. To probe multi-modal safety understanding, we turn these narratives and constraints into photorealistic images and videos capturing transitions from safe to unsafe states, using advanced generative models. Secondly, we comprehensively analyze the ability of major foundation models to perceive risks, reason about safety, and trigger interventions; this yields multi-faceted insights into their deployment readiness for safety-critical agentic applications. Finally, we develop a post-training paradigm to teach models to explicitly reason about embodiment-specific safety constraints provided through system instructions. The resulting models generate thinking traces that make safety reasoning interpretable and transparent, achieving state of the art performance in constraint satisfaction evaluations. The benchmark will be released at this https URL 

**Abstract (ZH)**: 当AI与物理世界交互——作为机器人或辅助代理时，新的安全挑战在纯粹的“数字AI”之外浮现。在这些交互中，物理伤害的可能性是直接且即时的。当前最先进的基础模型是否理解关于物理安全的常识性事实，例如一个箱子可能太重而无法提起，或者不应该将热咖啡杯递给小孩？在本文中，我们的贡献有三个方面：首先，我们开发了一种高度可扩展的方法，通过基于真实世界的伤害叙事和操作安全约束来进行体感AI系统的持续物理安全基准测试。为了探究多模态安全理解，我们将这些叙事和约束转化为逼真的图像和视频，捕捉从安全状态到不安全状态的过渡，使用高级生成模型。其次，我们全面分析了主要基础模型感知风险、推理安全以及触发干预措施的能力；这为我们提供了关于其在关键安全任务中部署准备情况的多方面洞见。最后，我们开发了一种后训练范式，教导模型根据系统指令具体推理关于体感特定的安全约束。结果模型生成的推理痕迹使安全推理变得可解释和透明，并在约束满足评估中达到了最先进的性能。基准测试将在以下链接发布：this https URL。 

---
# Semantic F1 Scores: Fair Evaluation Under Fuzzy Class Boundaries 

**Title (ZH)**: 语义F1分数：在模糊类别边界下的公平评价 

**Authors**: Georgios Chochlakis, Jackson Trager, Vedant Jhaveri, Nikhil Ravichandran, Alexandros Potamianos, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21633)  

**Abstract**: We propose Semantic F1 Scores, novel evaluation metrics for subjective or fuzzy multi-label classification that quantify semantic relatedness between predicted and gold labels. Unlike the conventional F1 metrics that treat semantically related predictions as complete failures, Semantic F1 incorporates a label similarity matrix to compute soft precision-like and recall-like scores, from which the Semantic F1 scores are derived. Unlike existing similarity-based metrics, our novel two-step precision-recall formulation enables the comparison of label sets of arbitrary sizes without discarding labels or forcing matches between dissimilar labels. By granting partial credit for semantically related but nonidentical labels, Semantic F1 better reflects the realities of domains marked by human disagreement or fuzzy category boundaries. In this way, it provides fairer evaluations: it recognizes that categories overlap, that annotators disagree, and that downstream decisions based on similar predictions lead to similar outcomes. Through theoretical justification and extensive empirical validation on synthetic and real data, we show that Semantic F1 demonstrates greater interpretability and ecological validity. Because it requires only a domain-appropriate similarity matrix, which is robust to misspecification, and not a rigid ontology, it is applicable across tasks and modalities. 

**Abstract (ZH)**: 我们提出语义F1分数：一种新型的主观或模糊多标签分类评估指标，用于量化预测标签与黄金标签之间的语义相关性 

---
# Automated and Interpretable Survival Analysis from Multimodal Data 

**Title (ZH)**: 多模态数据的自动可解释生存分析 

**Authors**: Mafalda Malafaia, Peter A.N. Bosman, Coen Rasch, Tanja Alderliesten  

**Link**: [PDF](https://arxiv.org/pdf/2509.21600)  

**Abstract**: Accurate and interpretable survival analysis remains a core challenge in oncology. With growing multimodal data and the clinical need for transparent models to support validation and trust, this challenge increases in complexity. We propose an interpretable multimodal AI framework to automate survival analysis by integrating clinical variables and computed tomography imaging. Our MultiFIX-based framework uses deep learning to infer survival-relevant features that are further explained: imaging features are interpreted via Grad-CAM, while clinical variables are modeled as symbolic expressions through genetic programming. Risk estimation employs a transparent Cox regression, enabling stratification into groups with distinct survival outcomes. Using the open-source RADCURE dataset for head and neck cancer, MultiFIX achieves a C-index of 0.838 (prediction) and 0.826 (stratification), outperforming the clinical and academic baseline approaches and aligning with known prognostic markers. These results highlight the promise of interpretable multimodal AI for precision oncology with MultiFIX. 

**Abstract (ZH)**: 准确可解释的生存分析仍是肿瘤学中的核心挑战。随着多模态数据的增长和临床对透明模型的需求以支持验证和信任，这一挑战变得更加复杂。我们提出了一种可解释的多模态AI框架，通过结合临床变量和计算机断层扫描成像来自动化生存分析。基于MultiFIX的框架利用深度学习推断与生存相关的特征，进一步通过Grad-CAM解释成像特征，并通过遗传编程将临床变量建模为符号表达式。风险估计采用透明的Cox回归，实现了具有不同生存结局的组别划分。使用开源RADCURE头颈癌数据集，MultiFIX在预测中的C指数为0.838，在分层中的C指数为0.826，优于临床和学术基准方法，并与已知的预后标志物一致。这些结果突显了MultiFIX在精确肿瘤学中的可解释多模态AI的潜力。 

---
# GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models 

**Title (ZH)**: GeoEvolve：通过多Agent大型语言模型自动发现地理空间模型 

**Authors**: Peng Luo, Xiayin Lou, Yu Zheng, Zhuo Zheng, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.21593)  

**Abstract**: Geospatial modeling provides critical solutions for pressing global challenges such as sustainability and climate change. Existing large language model (LLM)-based algorithm discovery frameworks, such as AlphaEvolve, excel at evolving generic code but lack the domain knowledge and multi-step reasoning required for complex geospatial problems. We introduce GeoEvolve, a multi-agent LLM framework that couples evolutionary search with geospatial domain knowledge to automatically design and refine geospatial algorithms. GeoEvolve operates in two nested loops: an inner loop leverages a code evolver to generate and mutate candidate solutions, while an outer agentic controller evaluates global elites and queries a GeoKnowRAG module -- a structured geospatial knowledge base that injects theoretical priors from geography. This knowledge-guided evolution steers the search toward theoretically meaningful and computationally efficient algorithms. We evaluate GeoEvolve on two fundamental and classical tasks: spatial interpolation (kriging) and spatial uncertainty quantification (geospatial conformal prediction). Across these benchmarks, GeoEvolve automatically improves and discovers new algorithms, incorporating geospatial theory on top of classical models. It reduces spatial interpolation error (RMSE) by 13-21% and enhances uncertainty estimation performance by 17\%. Ablation studies confirm that domain-guided retrieval is essential for stable, high-quality evolution. These results demonstrate that GeoEvolve provides a scalable path toward automated, knowledge-driven geospatial modeling, opening new opportunities for trustworthy and efficient AI-for-Science discovery. 

**Abstract (ZH)**: 地理空间建模提供了应对可持续性与气候变化等紧迫全球挑战的关键解决方案。现有的基于大型语言模型（LLM）的算法发现框架，如AlphaEvolve，擅长进化通用代码，但在处理复杂的地理空间问题时缺乏必要的领域知识和多步推理能力。我们引入GeoEvolve，这是一种结合进化搜索与地理空间领域知识的多智能体LLM框架，自动设计和优化地理空间算法。GeoEvolve通过两个嵌套循环运作：内部循环利用代码进化器生成和变异候选解决方案，外部代理控制器评估全局精英并查询包含地理理论先验的GeoKnowRAG模块。这种知识导向的进化使搜索朝着理论意义重大且计算效率高的算法方向发展。我们评估GeoEvolve在两个基本的经典任务上：空间插值（克里金法）和空间不确定性量化（地理空间一致预测）。在这些基准测试中，GeoEvolve自动改进和发现新算法，基于经典模型结合地理空间理论。它将空间插值误差（RMSE）降低了13-21%，并提高了不确定性估计性能17%。消融研究表明，领域引导的检索对于稳定高效的知识驱动进化是至关重要的。这些结果表明，GeoEvolve提供了一条可扩展的知识驱动地理空间建模的途径，为可信赖和高效的AI-for-Science发现打开了新的机会。 

---
# EEG-Based Consumer Behaviour Prediction: An Exploration from Classical Machine Learning to Graph Neural Networks 

**Title (ZH)**: 基于EEG的消费者行为预测：从经典机器学习到图神经网络的研究 

**Authors**: Mohammad Parsa Afshar, Aryan Azimi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21567)  

**Abstract**: Prediction of consumer behavior is one of the important purposes in marketing, cognitive neuroscience, and human-computer interaction. The electroencephalography (EEG) data can help analyze the decision process by providing detailed information about the brain's neural activity. In this research, a comparative approach is utilized for predicting consumer behavior by EEG data. In the first step, the features of the EEG data from the NeuMa dataset were extracted and cleaned. For the Graph Neural Network (GNN) models, the brain connectivity features were created. Different machine learning models, such as classical models and Graph Neural Networks, are used and compared. The GNN models with different architectures are implemented to have a comprehensive comparison; furthermore, a wide range of classical models, such as ensemble models, are applied, which can be very helpful to show the difference and performance of each model on the dataset. Although the results did not show a significant difference overall, the GNN models generally performed better in some basic criteria where classical models were not satisfactory. This study not only shows that combining EEG signal analysis and machine learning models can provide an approach to deeper understanding of consumer behavior, but also provides a comprehensive comparison between the machine learning models that have been widely used in previous studies in the EEG-based neuromarketing such as Support Vector Machine (SVM), and the models which are not used or rarely used in the field, like Graph Neural Networks. 

**Abstract (ZH)**: 利用脑电图数据预测消费者行为的图神经网络方法比较研究 

---
# AutoClimDS: Climate Data Science Agentic AI -- A Knowledge Graph is All You Need 

**Title (ZH)**: AutoClimDS：气候数据科学自治AI——只需一个知识图谱 

**Authors**: Ahmed Jaber, Wangshu Zhu, Karthick Jayavelu, Justin Downes, Sameer Mohamed, Candace Agonafir, Linnia Hawkins, Tian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21553)  

**Abstract**: Climate data science faces persistent barriers stemming from the fragmented nature of data sources, heterogeneous formats, and the steep technical expertise required to identify, acquire, and process datasets. These challenges limit participation, slow discovery, and reduce the reproducibility of scientific workflows. In this paper, we present a proof of concept for addressing these barriers through the integration of a curated knowledge graph (KG) with AI agents designed for cloud-native scientific workflows. The KG provides a unifying layer that organizes datasets, tools, and workflows, while AI agents -- powered by generative AI services -- enable natural language interaction, automated data access, and streamlined analysis. Together, these components drastically lower the technical threshold for engaging in climate data science, enabling non-specialist users to identify and analyze relevant datasets. By leveraging existing cloud-ready API data portals, we demonstrate that "a knowledge graph is all you need" to unlock scalable and agentic workflows for scientific inquiry. The open-source design of our system further supports community contributions, ensuring that the KG and associated tools can evolve as a shared commons. Our results illustrate a pathway toward democratizing access to climate data and establishing a reproducible, extensible framework for human--AI collaboration in scientific research. 

**Abstract (ZH)**: 气候变化数据科学面临的持久障碍源于数据源的碎片化、异构格式以及识别、获取和处理数据集所需的 steep 技术门槛。这些挑战限制了参与度、减缓了发现过程，并降低了科学工作流的可再现性。本文提出了一种概念验证方法，通过将精标知识图谱（KG）与为云原生科学工作流设计的 AI 代理集成来应对这些障碍。知识图谱提供了一个统一的层，组织数据集、工具和工作流，而由生成式 AI 服务驱动的 AI 代理则支持自然语言交互、自动化数据访问和 streamlined 分析。这些组件共同大幅降低了参与气候变化数据科学的技术门槛，使非专家用户能够识别和分析相关数据集。通过利用现有的云就绪 API 数据门户，我们证明“一个知识图谱就足够了”来解锁可扩展且自主的工作流以支持科学探索。我们的开源系统设计进一步支持了社区贡献，确保知识图谱及相关工具能够作为共享公共资源进行演化。我们的结果表明了一条途径，即通过将知识图谱应用于气候变化数据来实现数据访问的民主化，并建立了人机在科学研究中协作的可再现且可扩展框架。 

---
# Correct Reasoning Paths Visit Shared Decision Pivots 

**Title (ZH)**: 正确的推理路径访问共享决策关键点 

**Authors**: Dongkyu Cho, Amy B.Z. Zhang, Bilel Fehri, Sheng Wang, Rumi Chunara, Rui Song, Hengrui Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21549)  

**Abstract**: Chain-of-thought (CoT) reasoning exposes the intermediate thinking process of large language models (LLMs), yet verifying those traces at scale remains unsolved. In response, we introduce the idea of decision pivots-minimal, verifiable checkpoints that any correct reasoning path must visit. We hypothesize that correct reasoning, though stylistically diverse, converge on the same pivot set, while incorrect ones violate at least one pivot. Leveraging this property, we propose a self-training pipeline that (i) samples diverse reasoning paths and mines shared decision pivots, (ii) compresses each trace into pivot-focused short-path reasoning using an auxiliary verifier, and (iii) post-trains the model using its self-generated outputs. The proposed method aligns reasoning without ground truth reasoning data or external metrics. Experiments on standard benchmarks such as LogiQA, MedQA, and MATH500 show the effectiveness of our method. 

**Abstract (ZH)**: Chain-of-Thought 论证揭示了大语言模型的中间推理过程，但大规模验证这些痕迹仍是一个未解决的问题。为此，我们提出了决策拐点的概念——任何正确的推理路径都必须访问的最小且可验证的检查点。我们推测，虽然正确的推理在风格上各不相同，但都会收敛于相同的拐点集，而不正确的推理至少会违背一个拐点。利用这一特性，我们提出了一种自我训练管道，该管道包括(i) 抽取多样化的推理路径并挖掘共享的决策拐点，(ii) 使用辅助验证器将每个轨迹压缩为以拐点为中心的短路径推理，以及(iii) 使用模型自动生成的输出进行后训练。所提出的方法在没有真实推理数据或外部指标的情况下实现了推理一致性。实验结果表明，该方法在标准基准如 LogiQA、MedQA 和 MATH500 中具有有效性。 

---
# Towards mitigating information leakage when evaluating safety monitors 

**Title (ZH)**: 减轻评估安全性监视器时的信息泄露 

**Authors**: Gerard Boxo, Aman Neelappa, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2509.21344)  

**Abstract**: White box monitors that analyze model internals offer promising advantages for detecting potentially harmful behaviors in large language models, including lower computational costs and integration into layered defense this http URL, training and evaluating these monitors requires response exemplars that exhibit the target behaviors, typically elicited through prompting or fine-tuning. This presents a challenge when the information used to elicit behaviors inevitably leaks into the data that monitors ingest, inflating their effectiveness. We present a systematic framework for evaluating a monitor's performance in terms of its ability to detect genuine model behavior rather than superficial elicitation artifacts. Furthermore, we propose three novel strategies to evaluate the monitor: content filtering (removing deception-related text from inputs), score filtering (aggregating only over task-relevant tokens), and prompt distilled fine-tuned model organisms (models trained to exhibit deceptive behavior without explicit prompting). Using deception detection as a representative case study, we identify two forms of leakage that inflate monitor performance: elicitation leakage from prompts that explicitly request harmful behavior, and reasoning leakage from models that verbalize their deceptive actions. Through experiments on multiple deception benchmarks, we apply our proposed mitigation strategies and measure performance retention. Our evaluation of the monitors reveal three crucial findings: (1) Content filtering is a good mitigation strategy that allows for a smooth removal of elicitation signal and can decrease probe AUROC by 30\% (2) Score filtering was found to reduce AUROC by 15\% but is not as straightforward to attribute to (3) A finetuned model organism improves monitor evaluations but reduces their performance by upto 40\%, even when re-trained. 

**Abstract (ZH)**: 白盒监控器分析模型内部对我们检测大型语言模型中潜在有害行为提供了潜在优势，包括较低的计算成本和多层次防御的集成。这类监控器的训练和评估需要展示目标行为的响应示例，通常通过提示或微调获得。当用以触发行为的信息不可避免地泄露到监控器所摄入的数据中时，这会夸大其效果。我们提出了一种系统框架，用于评估监控器检测真正模型行为而非表面提示伪影的能力。此外，我们提出了三种新的评估策略：内容过滤（从输入中去除与欺骗相关的文本）、得分过滤（仅聚合与任务相关的标记）以及提示精炼微调模型有机体（训练模型以表现出欺骗行为而无需明确提示）。通过在多个欺骗检测基准上进行实验，我们应用了我们的缓解策略并测量了性能保留。通过对监控器的评估，我们发现了三个关键发现：(1) 内容过滤是一种有效的缓解策略，可以平滑去除引述信号，降低了探针AUROC约30%；(2) 得分过滤降低了AUROC约15%，但其归因性较弱；(3) 微调模型有机体提高了监控器评估，但甚至在重新训练后也使其性能最多下降40%。 

---
# See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation 

**Title (ZH)**: 看、指、飞：一种无需学习的VLM框架用于通用无人驾驶航空导航 

**Authors**: Chih Yao Hu, Yang-Sen Lin, Yuna Lee, Chih-Hai Su, Jie-Ying Lee, Shr-Ruei Tsai, Chin-Yang Lin, Kuan-Wen Chen, Tsung-Wei Ke, Yu-Lun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22653)  

**Abstract**: We present See, Point, Fly (SPF), a training-free aerial vision-and-language navigation (AVLN) framework built atop vision-language models (VLMs). SPF is capable of navigating to any goal based on any type of free-form instructions in any kind of environment. In contrast to existing VLM-based approaches that treat action prediction as a text generation task, our key insight is to consider action prediction for AVLN as a 2D spatial grounding task. SPF harnesses VLMs to decompose vague language instructions into iterative annotation of 2D waypoints on the input image. Along with the predicted traveling distance, SPF transforms predicted 2D waypoints into 3D displacement vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the traveling distance to facilitate more efficient navigation. Notably, SPF performs navigation in a closed-loop control manner, enabling UAVs to follow dynamic targets in dynamic environments. SPF sets a new state of the art in DRL simulation benchmark, outperforming the previous best method by an absolute margin of 63%. In extensive real-world evaluations, SPF outperforms strong baselines by a large margin. We also conduct comprehensive ablation studies to highlight the effectiveness of our design choice. Lastly, SPF shows remarkable generalization to different VLMs. Project page: this https URL 

**Abstract (ZH)**: 我们提出了See, Point, Fly (SPF)：一种基于视觉语言模型的无需训练的空中视觉与语言导航（AVLN）框架。SPF 能够根据任意类型的自由格式指令在任意环境条件下导航至任意目标。与现有基于视觉语言模型的方法将动作预测视为文本生成任务不同，我们的关键洞察是将AVLN中的动作预测视为二维空间定位任务。SPF 利用视觉语言模型将模糊的语言指令分解为对输入图像上二维航点的迭代标注。除了预测的行进距离外，SPF 还将预测的二维航点转换为用于无人机的三维位移向量作为动作命令。此外，SPF 还会自适应调整行进距离以促进更高效的导航。值得注意的是，SPF 以闭环控制的方式进行导航，能够使无人机在动态环境中追踪动态目标。在DRL仿真基准测试中，SPF 达到了新最优水平，绝对优势领先于之前的最佳方法63%。在广泛的现实世界评估中，SPF 显著优于强大的基线方法。我们还进行了全面的消融研究以突出我们设计选择的有效性。最后，SPF 在不同的视觉语言模型中展示出了卓越的泛化能力。项目页面：这个 https URL。 

---
# VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing 

**Title (ZH)**: VoiceAssistant-Eval：跨听、说、看评估AI助手性能的标准基准 

**Authors**: Ke Wang, Houxing Ren, Zimu Lu, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22651)  

**Abstract**: The growing capabilities of large language models and multimodal systems have spurred interest in voice-first AI assistants, yet existing benchmarks are inadequate for evaluating the full range of these systems' capabilities. We introduce VoiceAssistant-Eval, a comprehensive benchmark designed to assess AI assistants across listening, speaking, and viewing. VoiceAssistant-Eval comprises 10,497 curated examples spanning 13 task categories. These tasks include natural sounds, music, and spoken dialogue for listening; multi-turn dialogue, role-play imitation, and various scenarios for speaking; and highly heterogeneous images for viewing. To demonstrate its utility, we evaluate 21 open-source models and GPT-4o-Audio, measuring the quality of the response content and speech, as well as their consistency. The results reveal three key findings: (1) proprietary models do not universally outperform open-source models; (2) most models excel at speaking tasks but lag in audio understanding; and (3) well-designed smaller models can rival much larger ones. Notably, the mid-sized Step-Audio-2-mini (7B) achieves more than double the listening accuracy of LLaMA-Omni2-32B-Bilingual. However, challenges remain: multimodal (audio plus visual) input and role-play voice imitation tasks are difficult for current models, and significant gaps persist in robustness and safety alignment. VoiceAssistant-Eval identifies these gaps and establishes a rigorous framework for evaluating and guiding the development of next-generation AI assistants. Code and data will be released at this https URL . 

**Abstract (ZH)**: 大型语言模型和多模态系统的不断增强能力促使了对语音优先AI助手的兴趣，但现有基准无法评估这些系统能力的全部范围。我们引入了VoiceAssistant-Eval，一个全面的基准，旨在评估AI助手在听、说和看各方面的能力。VoiceAssistant-Eval包含10,497个精选示例，覆盖13个任务类别。这些任务包括听觉的自然声音、音乐和对话；口语的多轮对话、角色扮演模仿及各种场景；以及高度异构的图像。为了展示其实用性，我们评估了21个开源模型和GPT-4o-Audio，测量响应内容和语音的质量以及它们的一致性。结果显示了三个关键发现：（1）专有模型并不普遍优于开源模型；（2）大多数模型在口语任务中表现出色但在音频理解方面落后；（3）设计良好的小型模型可以与大型模型匹敌。值得注意的是，中型规模的Step-Audio-2-mini（7B）在听觉准确性上超过LLaMA-Omni2-32B-Bilingual超过一倍。然而，仍存在挑战：多模态（音频加视觉）输入和角色扮演语音模仿任务对于当前模型来说是难题，同时在稳健性和安全性对齐方面仍存在显著差距。VoiceAssistant-Eval指出了这些差距，并为评估和指导下一代AI助手的发展建立了严格的框架。代码和数据将在此处发布。 

---
# Toward a Physics of Deep Learning and Brains 

**Title (ZH)**: 向深学习和大脑的物理规律研究 

**Authors**: Arsham Ghavasieh, Meritxell Vila-Minana, Akanksha Khurd, John Beggs, Gerardo Ortiz, Santo Fortunato  

**Link**: [PDF](https://arxiv.org/pdf/2509.22649)  

**Abstract**: Deep neural networks and brains both learn and share superficial similarities: processing nodes are likened to neurons and adjustable weights are likened to modifiable synapses. But can a unified theoretical framework be found to underlie them both? Here we show that the equations used to describe neuronal avalanches in living brains can also be applied to cascades of activity in deep neural networks. These equations are derived from non-equilibrium statistical physics and show that deep neural networks learn best when poised between absorbing and active phases. Because these networks are strongly driven by inputs, however, they do not operate at a true critical point but within a quasi-critical regime -- one that still approximately satisfies crackling noise scaling relations. By training networks with different initializations, we show that maximal susceptibility is a more reliable predictor of learning than proximity to the critical point itself. This provides a blueprint for engineering improved network performance. Finally, using finite-size scaling we identify distinct universality classes, including Barkhausen noise and directed percolation. This theoretical framework demonstrates that universal features are shared by both biological and artificial neural networks. 

**Abstract (ZH)**: 深度神经网络和大脑都在学习和分享表面相似性：处理节点类比为神经元，可调权重类比为可修改的突触。但是，是否可以找到一个统一的理论框架来同时涵盖它们？在这里，我们展示了用于描述活体大脑神经元 avalanche 的方程也可以应用于深度神经网络中的活动级联。这些方程源自非平衡统计物理学，表明当深度神经网络处于吸收相和活跃相之间平衡状态时，它们学习效果最佳。然而，由于这些网络强烈受输入驱动，它们并未在真正的临界点处运行，而是在准临界状态下运行——其仍大约满足裂变噪声标度关系。通过使用不同的初始化训练网络，我们表明最大可探性比临界点本身更可靠地预测了学习。这为工程优化网络性能提供了蓝图。最后，通过有限大小标度分析，我们标识出不同的普遍性类，包括 Barkhausen 噪声和定向渗流。该理论框架证明了生物学和人工神经网络在普遍特性方面具有共性。 

---
# CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning 

**Title (ZH)**: CapRL: 通过强化学习激发密集图像描述能力 

**Authors**: Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.22647)  

**Abstract**: Image captioning is a fundamental task that bridges the visual and linguistic domains, playing a critical role in pre-training Large Vision-Language Models (LVLMs). Current state-of-the-art captioning models are typically trained with Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable data annotated by humans or proprietary models. This approach often leads to models that memorize specific ground-truth answers, limiting their generality and ability to generate diverse, creative descriptions. To overcome the limitation of SFT, we propose applying the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning. A primary challenge, however, is designing an objective reward function for the inherently subjective nature of what constitutes a "good" caption. We introduce Captioning Reinforcement Learning (CapRL), a novel training framework that redefines caption quality through its utility: a high-quality caption should enable a non-visual language model to accurately answer questions about the corresponding image. CapRL employs a decoupled two-stage pipeline where an LVLM generates a caption, and the objective reward is derived from the accuracy of a separate, vision-free LLM answering Multiple-Choice Questions based solely on that caption. As the first study to apply RLVR to the subjective image captioning task, we demonstrate that CapRL significantly enhances multiple settings. Pretraining on the CapRL-5M caption dataset annotated by CapRL-3B results in substantial gains across 12 benchmarks. Moreover, within the Prism Framework for caption quality evaluation, CapRL achieves performance comparable to Qwen2.5-VL-72B, while exceeding the baseline by an average margin of 8.4%. Code is available here: this https URL. 

**Abstract (ZH)**: 基于验证性奖励的图像字幕 reinforcement learning with verifiable rewards for image captioning 

---
# Learning Human-Perceived Fakeness in AI-Generated Videos via Multimodal LLMs 

**Title (ZH)**: 通过多模态大语言模型学习AI生成视频的人感知假造性 

**Authors**: Xingyu Fu, Siyi Liu, Yinuo Xu, Pan Lu, Guangqiuse Hu, Tianbo Yang, Taran Anantasagar, Christopher Shen, Yikai Mao, Yuanzhe Liu, Keyush Shah, Chung Un Lee, Yejin Choi, James Zou, Dan Roth, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2509.22646)  

**Abstract**: Can humans identify AI-generated (fake) videos and provide grounded reasons? While video generation models have advanced rapidly, a critical dimension -- whether humans can detect deepfake traces within a generated video, i.e., spatiotemporal grounded visual artifacts that reveal a video as machine generated -- has been largely overlooked. We introduce DeeptraceReward, the first fine-grained, spatially- and temporally- aware benchmark that annotates human-perceived fake traces for video generation reward. The dataset comprises 4.3K detailed annotations across 3.3K high-quality generated videos. Each annotation provides a natural-language explanation, pinpoints a bounding-box region containing the perceived trace, and marks precise onset and offset timestamps. We consolidate these annotations into 9 major categories of deepfake traces that lead humans to identify a video as AI-generated, and train multimodal language models (LMs) as reward models to mimic human judgments and localizations. On DeeptraceReward, our 7B reward model outperforms GPT-5 by 34.7% on average across fake clue identification, grounding, and explanation. Interestingly, we observe a consistent difficulty gradient: binary fake v.s. real classification is substantially easier than fine-grained deepfake trace detection; within the latter, performance degrades from natural language explanations (easiest), to spatial grounding, to temporal labeling (hardest). By foregrounding human-perceived deepfake traces, DeeptraceReward provides a rigorous testbed and training signal for socially aware and trustworthy video generation. 

**Abstract (ZH)**: 人类能否识别AI生成的（假）视频并提供有根据的理由？视频生成模型虽然取得了快速进步，但一个关键维度——人类是否能检测生成视频中的深度伪造痕迹，即时空相关的视觉特征，这些特征可以揭示视频是机器生成的——却很少被关注。我们引入了DeeptraceReward，这是第一个细粒度的空间和时间意识基准，用于为视频生成奖励标注人类感知到的假痕迹。该数据集包含4300个详细标注，涵盖3300个高质量生成视频。每个标注提供自然语言解释，指出包含感知到的痕迹的边界框区域，并标注精确的时间戳。我们将这些标注归类为9大类深度伪造痕迹，这些痕迹使人类能够识别视频为AI生成，并训练多模态语言模型（语言模型）作为奖励模型，以模仿人类判断和定位。在DeeptraceReward上，我们的7B奖励模型在假线索识别、定位和解释方面平均优于GPT-5 34.7%。有趣的是，我们观察到一个一致的难度梯度：二元假与真分类明显比细粒度深度伪造痕迹检测更容易；在后者中，从自然语言解释（最容易）到空间定位，再到时间标注（最难），性能逐渐下降。通过突出显示人类感知到的深度伪造痕迹，DeeptraceReward为社会意识强且可信的视频生成提供了一个严格的测试平台和训练信号。 

---
# Hierarchical Representation Matching for CLIP-based Class-Incremental Learning 

**Title (ZH)**: 基于CLIP的类增量学习的层次表示匹配 

**Authors**: Zhen-Hao Wen, Yan Wang, Ji Feng, Han-Jia Ye, De-Chuan Zhan, Da-Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22645)  

**Abstract**: Class-Incremental Learning (CIL) aims to endow models with the ability to continuously adapt to evolving data streams. Recent advances in pre-trained vision-language models (e.g., CLIP) provide a powerful foundation for this task. However, existing approaches often rely on simplistic templates, such as "a photo of a [CLASS]", which overlook the hierarchical nature of visual concepts. For example, recognizing "cat" versus "car" depends on coarse-grained cues, while distinguishing "cat" from "lion" requires fine-grained details. Similarly, the current feature mapping in CLIP relies solely on the representation from the last layer, neglecting the hierarchical information contained in earlier layers. In this work, we introduce HiErarchical Representation MAtchiNg (HERMAN) for CLIP-based CIL. Our approach leverages LLMs to recursively generate discriminative textual descriptors, thereby augmenting the semantic space with explicit hierarchical cues. These descriptors are matched to different levels of the semantic hierarchy and adaptively routed based on task-specific requirements, enabling precise discrimination while alleviating catastrophic forgetting in incremental tasks. Extensive experiments on multiple benchmarks demonstrate that our method consistently achieves state-of-the-art performance. 

**Abstract (ZH)**: 层级表示匹配（HERMAN）：基于CLIP的类增量学习 

---
# WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning 

**Title (ZH)**: WebGen-Agent: 通过多级反馈和步骤级强化学习增强交互式网站生成 

**Authors**: Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Junting Pan, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22644)  

**Abstract**: Agent systems powered by large language models (LLMs) have demonstrated impressive performance on repository-level code-generation tasks. However, for tasks such as website codebase generation, which depend heavily on visual effects and user-interaction feedback, current code agents rely only on simple code execution for feedback and verification. This approach fails to capture the actual quality of the generated code. In this paper, we propose WebGen-Agent, a novel website-generation agent that leverages comprehensive and multi-level visual feedback to iteratively generate and refine the website codebase. Detailed and expressive text descriptions and suggestions regarding the screenshots and GUI-agent testing of the websites are generated by a visual language model (VLM), together with scores that quantify their quality. The screenshot and GUI-agent scores are further integrated with a backtracking and select-best mechanism, enhancing the performance of the agent. Utilizing the accurate visual scores inherent in the WebGen-Agent workflow, we further introduce \textit{Step-GRPO with Screenshot and GUI-agent Feedback} to improve the ability of LLMs to act as the reasoning engine of WebGen-Agent. By using the screenshot and GUI-agent scores at each step as the reward in Step-GRPO, we provide a dense and reliable process supervision signal, which effectively improves the model's website-generation ability. On the WebGen-Bench dataset, WebGen-Agent increases the accuracy of Claude-3.5-Sonnet from 26.4% to 51.9% and its appearance score from 3.0 to 3.9, outperforming the previous state-of-the-art agent system. Additionally, our Step-GRPO training approach increases the accuracy of Qwen2.5-Coder-7B-Instruct from 38.9% to 45.4% and raises the appearance score from 3.4 to 3.7. 

**Abstract (ZH)**: 基于大型语言模型的代理系统在仓库级代码生成任务中展现了令人印象深刻的性能。然而，对于高度依赖视觉效果和用户交互反馈的任务，如网站代码生成，当前的代码代理仅依赖简单的代码执行来获取反馈和验证，这种方法未能捕捉到生成代码的实际质量。本文提出了一种名为WebGen-Agent的新型网站生成代理，该代理利用全面的多级视觉反馈来迭代生成和优化网站代码库。视觉语言模型(VLM)生成了关于屏幕截图和GUI代理测试的详细描述和建议，以及衡量其质量的评分。屏幕截图和GUI代理评分进一步与回溯和选择最佳机制集成，增强了代理的性能。利用WebGen-Agent工作流中固有的准确视觉评分，我们引入了基于屏幕截图和GUI代理反馈的Step-GRPO方法，以提高大型语言模型作为WebGen-Agent推理引擎的能力。通过在每一步骤中使用屏幕截图和GUI代理评分作为奖励，我们提供了一个密集且可靠的进程监督信号，有效提升模型的网站生成能力。实验结果表明，WebGen-Agent在WebGen-Bench数据集上将Claude-3.5-Sonnet的准确性从26.4%提高到51.9%，外观评分从3.0提高到3.9，超越了之前最先进的代理系统。此外，我们的Step-GRPO训练方法将Qwen2.5-Coder-7B-Instruct的准确性从38.9%提高到45.4%，并将其外观评分从3.4提高到3.7。 

---
# Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity 

**Title (ZH)**: 死亡的创新：超越基于n-gram新颖性度量的文本创造力评价 

**Authors**: Arkadiy Saakyan, Najoung Kim, Smaranda Muresan, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2509.22641)  

**Abstract**: N-gram novelty is widely used to evaluate language models' ability to generate text outside of their training data. More recently, it has also been adopted as a metric for measuring textual creativity. However, theoretical work on creativity suggests that this approach may be inadequate, as it does not account for creativity's dual nature: novelty (how original the text is) and appropriateness (how sensical and pragmatic it is). We investigate the relationship between this notion of creativity and n-gram novelty through 7542 expert writer annotations (n=26) of novelty, pragmaticality, and sensicality via close reading of human and AI-generated text. We find that while n-gram novelty is positively associated with expert writer-judged creativity, ~91% of top-quartile expressions by n-gram novelty are not judged as creative, cautioning against relying on n-gram novelty alone. Furthermore, unlike human-written text, higher n-gram novelty in open-source LLMs correlates with lower pragmaticality. In an exploratory study with frontier close-source models, we additionally confirm that they are less likely to produce creative expressions than humans. Using our dataset, we test whether zero-shot, few-shot, and finetuned models are able to identify creative expressions (a positive aspect of writing) and non-pragmatic ones (a negative aspect). Overall, frontier LLMs exhibit performance much higher than random but leave room for improvement, especially struggling to identify non-pragmatic expressions. We further find that LLM-as-a-Judge novelty scores from the best-performing model were predictive of expert writer preferences. 

**Abstract (ZH)**: 基于n-gram新颖性的创造性的评估：一种理论与实证的探究 

---
# Language Models Can Learn from Verbal Feedback Without Scalar Rewards 

**Title (ZH)**: 语言模型可以从口头反馈中学习而无需标量奖励 

**Authors**: Renjie Luo, Zichen Liu, Xiangyan Liu, Chao Du, Min Lin, Wenhu Chen, Wei Lu, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22638)  

**Abstract**: LLMs are often trained with RL from human or AI feedback, yet such methods typically compress nuanced feedback into scalar rewards, discarding much of their richness and inducing scale imbalance. We propose treating verbal feedback as a conditioning signal. Inspired by language priors in text-to-image generation, which enable novel outputs from unseen prompts, we introduce the feedback-conditional policy (FCP). FCP learns directly from response-feedback pairs, approximating the feedback-conditional posterior through maximum likelihood training on offline data. We further develop an online bootstrapping stage where the policy generates under positive conditions and receives fresh feedback to refine itself. This reframes feedback-driven learning as conditional generation rather than reward optimization, offering a more expressive way for LLMs to directly learn from verbal feedback. Our code is available at this https URL. 

**Abstract (ZH)**: LLMs通常通过人类或AI反馈的RL进行训练，但这些方法通常将细腻的反馈压缩为标量奖励，损失了其丰富性并导致规模失衡。我们提出将口头反馈视为条件信号。受到文本到图像生成中语言先验的启发，能够从未见过的提示生成新型输出，我们引入了反馈条件策略（FCP）。FCP直接从响应-反馈对中学习，通过离线数据的最大似然训练近似条件后验。我们进一步开发了一个在线自助阶段，其中策略在积极条件下生成并接收新的反馈以自我完善。这将反馈驱动的学习重新框架为条件生成，而非奖励优化，为LLMs直接从口头反馈中学习提供了一种更具表现力的方式。代码可在以下网址获取。 

---
# Variational Reasoning for Language Models 

**Title (ZH)**: 变分推理的语言模型 

**Authors**: Xiangxin Zhou, Zichen Liu, Haonan Wang, Chao Du, Min Lin, Chongxuan Li, Liang Wang, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22637)  

**Abstract**: We introduce a variational reasoning framework for language models that treats thinking traces as latent variables and optimizes them through variational inference. Starting from the evidence lower bound (ELBO), we extend it to a multi-trace objective for tighter bounds and propose a forward-KL formulation that stabilizes the training of the variational posterior. We further show that rejection sampling finetuning and binary-reward RL, including GRPO, can be interpreted as local forward-KL objectives, where an implicit weighting by model accuracy naturally arises from the derivation and reveals a previously unnoticed bias toward easier questions. We empirically validate our method on the Qwen 2.5 and Qwen 3 model families across a wide range of reasoning tasks. Overall, our work provides a principled probabilistic perspective that unifies variational inference with RL-style methods and yields stable objectives for improving the reasoning ability of language models. Our code is available at this https URL. 

**Abstract (ZH)**: 一种将思维轨迹作为潜在变量并通过变分推断优化的变分推理框架：将证据下界扩展为多轨迹目标并提出前向-KL公式以稳定变分后验的训练 

---
# Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback 

**Title (ZH)**: 基于人类反馈的强化学习的高效在线探索方法 

**Authors**: Gen Li, Yuling Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22633)  

**Abstract**: Reinforcement learning with human feedback (RLHF), which learns a reward model from human preference data and then optimizes a policy to favor preferred responses, has emerged as a central paradigm for aligning large language models (LLMs) with human preferences. In this paper, we investigate exploration principles for online RLHF, where one seeks to adaptively collect new preference data to refine both the reward model and the policy in a data-efficient manner. By examining existing optimism-based exploration algorithms, we identify a drawback in their sampling protocol: they tend to gather comparisons that fail to reduce the most informative uncertainties in reward differences, and we prove lower bounds showing that such methods can incur linear regret over exponentially long horizons. Motivated by this insight, we propose a new exploration scheme that directs preference queries toward reducing uncertainty in reward differences most relevant to policy improvement. Under a multi-armed bandit model of RLHF, we establish regret bounds of order $T^{(\beta+1)/(\beta+2)}$, where $\beta>0$ is a hyperparameter that balances reward maximization against mitigating distribution shift. To our knowledge, this is the first online RLHF algorithm with regret scaling polynomially in all model parameters. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）通过从人类偏好数据中学习奖励模型，然后优化策略以偏好更受欢迎的响应，已成为使大语言模型（LLMs）与人类偏好保持一致的核心范式。在本文中，我们研究了在线RLHF的探索原则，旨在自适应地收集新的偏好数据，以在高效的数据利用方式下同时精炼奖励模型和策略。通过分析现有的乐观探索算法，我们发现其采样协议的一个缺点：它们倾向于收集未能减少奖励差异中最具信息量不确定性的方式，我们证明了这种方法在极长的时间跨度内可能会导致线性后悔。受到这一见解的启发，我们提出了一种新的探索方案，该方案将偏好查询的方向化，以减少对策略改进最相关的奖励差异的不确定性。在RLHF的multi-armed bandit模型下，我们建立了后悔界为$T^{(\beta+1)/(\beta+2)}$，其中$\beta>0$是一个调整奖励最大化与缓解分布迁移之间平衡的超参数。据我们所知，这是第一个所有模型参数的后悔界呈多项式阶的在线RLHF算法。 

---
# StateX: Enhancing RNN Recall via Post-training State Expansion 

**Title (ZH)**: StateX：通过后训练状态扩展增强RNN回忆能力 

**Authors**: Xingyu Shen, Yingfa Chen, Zhen Leng Thai, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22630)  

**Abstract**: While Transformer-based models have demonstrated remarkable language modeling performance, their high complexities result in high costs when processing long contexts. In contrast, recurrent neural networks (RNNs) such as linear attention and state space models have gained popularity due to their constant per-token complexities. However, these recurrent models struggle with tasks that require accurate recall of contextual information from long contexts, because all contextual information is compressed into a constant-size recurrent state. Previous works have shown that recall ability is positively correlated with the recurrent state size, yet directly training RNNs with larger recurrent states results in high training costs. In this paper, we introduce StateX, a training pipeline for efficiently expanding the states of pre-trained RNNs through post-training. For two popular classes of RNNs, linear attention and state space models, we design post-training architectural modifications to scale up the state size with no or negligible increase in model parameters. Experiments on models up to 1.3B parameters demonstrate that StateX efficiently enhances the recall and in-context learning ability of RNNs without incurring high post-training costs or compromising other capabilities. 

**Abstract (ZH)**: 基于Transformer模型的语言建模表现卓越，但由于其高复杂性，在处理长上下文时成本高昂。相比之下，线性注意力和状态空间模型等循环神经网络（RNNs）因恒定的逐词复杂性而受到青睐。然而，这些循环模型在需要准确回忆长上下文中的信息的任务中表现不佳，因为所有上下文信息都被压缩到一个恒定大小的循环状态中。以往的研究表明，循环状态大小与回忆能力正相关，但直接训练具有更大循环状态的RNN会导致高昂的训练成本。本文介绍了一种名为StateX的训练管道，用于通过后训练高效扩展预先训练的RNN的状态。对于两种流行的RNN类——线性注意力和状态空间模型——我们设计了后训练架构修改，以无或几乎无增加模型参数的方式扩展状态大小。参数量高达13亿的模型实验结果表明，StateX能够在不增加后训练成本或牺牲其他能力的情况下有效提升RNN的回忆能力和上下文学习能力。 

---
# Learning Admissible Heuristics for A*: Theory and Practice 

**Title (ZH)**: 学习可接纳启发式方法用于A*：理论与实践 

**Authors**: Ehsan Futuhi, Nathan R. Sturtevant  

**Link**: [PDF](https://arxiv.org/pdf/2509.22626)  

**Abstract**: Heuristic functions are central to the performance of search algorithms such as A-star, where admissibility - the property of never overestimating the true shortest-path cost - guarantees solution optimality. Recent deep learning approaches often disregard admissibility and provide limited guarantees on generalization beyond the training data. This paper addresses both of these limitations. First, we pose heuristic learning as a constrained optimization problem and introduce Cross-Entropy Admissibility (CEA), a loss function that enforces admissibility during training. On the Rubik's Cube domain, this method yields near-admissible heuristics with significantly stronger guidance than compressed pattern database (PDB) heuristics. Theoretically, we study the sample complexity of learning heuristics. By leveraging PDB abstractions and the structural properties of graphs such as the Rubik's Cube, we tighten the bound on the number of training samples needed for A-star to generalize. Replacing a general hypothesis class with a ReLU neural network gives bounds that depend primarily on the network's width and depth, rather than on graph size. Using the same network, we also provide the first generalization guarantees for goal-dependent heuristics. 

**Abstract (ZH)**: 启发式函数的学习作为约束优化问题及其在 Rubik’s Cube 领域的应用与理论分析 

---
# A Theoretical Analysis of Discrete Flow Matching Generative Models 

**Title (ZH)**: 离散流匹配生成模型的理论分析 

**Authors**: Maojiang Su, Mingcheng Lu, Jerry Yao-Chieh Hu, Shang Wu, Zhao Song, Alex Reneau, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22623)  

**Abstract**: We provide a theoretical analysis for end-to-end training Discrete Flow Matching (DFM) generative models. DFM is a promising discrete generative modeling framework that learns the underlying generative dynamics by training a neural network to approximate the transformative velocity field. Our analysis establishes a clear chain of guarantees by decomposing the final distribution estimation error. We first prove that the total variation distance between the generated and target distributions is controlled by the risk of the learned velocity field. We then bound this risk by analyzing its two primary sources: (i) Approximation Error, where we quantify the capacity of the Transformer architecture to represent the true velocity, and (ii) Estimation Error, where we derive statistical convergence rates that bound the error from training on a finite dataset. By composing these results, we provide the first formal proof that the distribution generated by a trained DFM model provably converges to the true data distribution as the training set size increases. 

**Abstract (ZH)**: 我们提供了端到端训练离散流匹配（DFM）生成模型的理论分析。DFM是一种有前景的离散生成建模框架，通过训练神经网络来逼近转化的流场以学习潜在的生成动力学。我们的分析通过分解最终分布估计误差来确立一条清晰的保证链。我们首先证明生成分布与目标分布之间的总变差距离受学习流场的风险控制。然后通过分析其两个主要来源来界定制约这一风险：（i）逼近误差，其中量化变换器架构表示真实流场的能力，（ii）估计误差，其中推导出统计收敛速率来界定制量有限数据集训练误差。通过组成这些结果，我们提供了首个正式证明，即随着训练数据集规模的增加，训练好的DFM模型生成的分布可证明地收敛到真实数据分布。 

---
# IA2: Alignment with ICL Activations Improves Supervised Fine-Tuning 

**Title (ZH)**: IA2: 与ICL激活相契合的监督微调改进方法 

**Authors**: Aayush Mishra, Daniel Khashabi, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22621)  

**Abstract**: Supervised Fine-Tuning (SFT) is used to specialize model behavior by training weights to produce intended target responses for queries. In contrast, In-Context Learning (ICL) adapts models during inference with instructions or demonstrations in the prompt. ICL can offer better generalizability and more calibrated responses compared to SFT in data scarce settings, at the cost of more inference compute. In this work, we ask the question: Can ICL's internal computations be used to improve the qualities of SFT? We first show that ICL and SFT produce distinct activation patterns, indicating that the two methods achieve adaptation through different functional mechanisms. Motivated by this observation and to use ICL's rich functionality, we introduce ICL Activation Alignment (IA2), a self-distillation technique which aims to replicate ICL's activation patterns in SFT models and incentivizes ICL-like internal reasoning. Performing IA2 as a priming step before SFT significantly improves the accuracy and calibration of model outputs, as shown by our extensive empirical results on 12 popular benchmarks and 2 model families. This finding is not only practically useful, but also offers a conceptual window into the inner mechanics of model adaptation. 

**Abstract (ZH)**: 监督微调与内省学习在模型适应中的融合：利用内省学习的内部计算改进监督微调的质量 

---
# Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting 

**Title (ZH)**: 从压缩图像表示中实现视觉-语言对齐的2D高斯点绘制方法 

**Authors**: Yasmine Omri, Connor Ding, Tsachy Weissman, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2509.22615)  

**Abstract**: Modern vision language pipelines are driven by RGB vision encoders trained on massive image text corpora. While these pipelines have enabled impressive zero shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy intensive and costly, and (ii) patch based tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance aware pruning, and batched CUDA kernels, achieving over 90x faster fitting and about 97% GPU utilization compared to prior implementations. We further adapt contrastive language image pretraining (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat aware input stem and a perceiver resampler, training only about 7% of the total parameters. On large DataComp subsets, GS encoders yield meaningful zero shot ImageNet-1K performance while compressing inputs 3 to 20x relative to pixels. While accuracy currently trails RGB encoders, our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission efficient for edge cloud learning. 

**Abstract (ZH)**: 现代视觉语言管道由在大量图像文本数据集上训练的RGB视觉编码器驱动。虽然这些管道使零-shot能力和跨任务转移能力显著增强，但仍继承了像素域中的两项结构性效率低下：（i）将密集的RGB图像从边缘设备传输到云端是耗能且成本高昂的；（ii）基于块的标记化会导致序列长度激增，给注意力预算和上下文限制带来压力。我们探索了2D高斯点绘（2DGS）作为替代视觉基础结构的选择：一种紧凑且空间自适应的表示，通过一组带有颜色的各向异性高斯函数参数化图像。我们开发了一个可扩展的2DGS管道，具有结构化初始化、亮度感知剪枝和批量CUDA内核，实现了比之前实现快90倍以上的拟合速度，并且GPU利用率约为97%。我们进一步将对比语言图像预训练（CLIP）适应到2DGS中，通过重用冻结的基于RGB的变压器骨干，结合一个轻量级的点绘感知输入茎和一个感知器重采样器，仅仅训练了约7%的总参数。在大规模DataComp子集上，GS编码器在相对像素压缩3到20倍输入的情况下，实现了具有意义的零-shot ImageNet-1K性能。尽管当前的准确性落后于RGB编码器，但我们的结果确立了2DGS作为一种可行的多模态基础结构，指出了架构瓶颈，并开启了既具有语义力量又具有传输效率的表示形式的研究道路。 

---
# Quantile Advantage Estimation for Entropy-Safe Reasoning 

**Title (ZH)**: 基于熵安全推理的分位数优势估计 

**Authors**: Junkang Wu, Kexin Huang, Jiancan Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.22611)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) strengthens LLM reasoning, but training often oscillates between {entropy collapse} and {entropy explosion}. We trace both hazards to the mean baseline used in value-free RL (e.g., GRPO and DAPO), which improperly penalizes negative-advantage samples under reward outliers. We propose {Quantile Advantage Estimation} (QAE), replacing the mean with a group-wise K-quantile baseline. QAE induces a response-level, two-regime gate: on hard queries (p <= 1 - K) it reinforces rare successes, while on easy queries (p > 1 - K) it targets remaining failures. Under first-order softmax updates, we prove {two-sided entropy safety}, giving lower and upper bounds on one-step entropy change that curb explosion and prevent collapse. Empirically, this minimal modification stabilizes entropy, sparsifies credit assignment (with tuned K, roughly 80% of responses receive zero advantage), and yields sustained pass@1 gains on Qwen3-8B/14B-Base across AIME 2024/2025 and AMC 2023. These results identify {baseline design} -- rather than token-level heuristics -- as the primary mechanism for scaling RLVR. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）增强LLM推理，但训练往往在“熵崩塌”和“熵爆炸”之间振荡。我们追踪这两种风险到价值自由RL（如GRPO和DAPO）中使用的均值基线，该基线在奖励异常值下不合理地惩罚负优势样本。我们提出了一种基于分位数优势估计（QAE），用组别K分位数基线替代均值。QAE诱发了一种响应级别、两阶段门控机制：对于难问题（p ≤ 1 - K），它强化罕见的成功；对于易问题（p > 1 - K），它瞄准剩余的失败。在一阶softmax更新下，我们证明了双向熵安全性，给出了熵变化的上下界，从而限制爆炸并防止崩塌。实验上，这种最小的修改稳定了熵，稀疏了奖励分配（通过调参K，约80%的响应获得零优势），并在Qwen3-8B/14B-Base上持续提高了AIME 2024/2025和AMC 2023的pass@1分数。这些结果表明，基线设计而非令牌级别启发式是RLVR扩展的主要机制。 

---
# Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning 

**Title (ZH)**: 掌握要领，再信任成果：渐进探索的自主强化学习自模仿方法 

**Authors**: Yulei Qin, Xiaoyu Tan, Zhengbao He, Gang Li, Haojia Lin, Zongyi Li, Zihan Xu, Yuchen Shi, Siqi Cai, Renting Rui, Shaofei Cai, Yuzheng Cai, Xuan Zhang, Sheng Ye, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22601)  

**Abstract**: Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL training instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a curriculum-based self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL framework, where a replay buffer stores self-generated promising trajectories for off-policy update, by gradually steering the policy evolution within a well-balanced range of entropy across stages. Specifically, our approach incorporates a curriculum to manage the exploration process, utilizing intrinsic rewards to foster skill-level exploration and facilitating action-level exploration through SIL. At first, the auxiliary tool call reward plays a critical role in the accumulation of tool-use skills, enabling broad exposure to the unfamiliar distributions of the environment feedback with an upward entropy trend. As training progresses, self-imitation gets strengthened to exploit existing successful patterns from replayed experiences for comparative action-level exploration, accelerating solution iteration without unbounded entropy growth. To further stabilize training, we recalibrate the advantages of experiences in the replay buffer to address the potential policy drift. Reugularizations such as the clipping of tokens with high covariance between probability and advantage are introduced to the trajectory-level entropy control to curb over-confidence. 

**Abstract (ZH)**: 基于课程的自我模仿学习方法SPEAR：在代理经验指导下逐步平衡探索与利用 

---
# From Parameters to Behavior: Unsupervised Compression of the Policy Space 

**Title (ZH)**: 从参数到行为：无监督的策略空间压缩 

**Authors**: Davide Tenedini, Riccardo Zamboni, Mirco Mutti, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.22566)  

**Abstract**: Despite its recent successes, Deep Reinforcement Learning (DRL) is notoriously sample-inefficient. We argue that this inefficiency stems from the standard practice of optimizing policies directly in the high-dimensional and highly redundant parameter space $\Theta$. This challenge is greatly compounded in multi-task settings. In this work, we develop a novel, unsupervised approach that compresses the policy parameter space $\Theta$ into a low-dimensional latent space $\mathcal{Z}$. We train a generative model $g:\mathcal{Z}\to\Theta$ by optimizing a behavioral reconstruction loss, which ensures that the latent space is organized by functional similarity rather than proximity in parameterization. We conjecture that the inherent dimensionality of this manifold is a function of the environment's complexity, rather than the size of the policy network. We validate our approach in continuous control domains, showing that the parameterization of standard policy networks can be compressed up to five orders of magnitude while retaining most of its expressivity. As a byproduct, we show that the learned manifold enables task-specific adaptation via Policy Gradient operating in the latent space $\mathcal{Z}$. 

**Abstract (ZH)**: 尽管深度强化学习（DRL）最近取得了成功，但它在样本效率方面一直表现不佳。我们argue认为，这种低效率源于直接在高维且高度冗余的参数空间Θ中优化策略的标准做法。在多任务设置中，这一挑战被进一步放大。在本文中，我们开发了一种新颖的无监督方法，将策略参数空间Θ压缩到低维潜在空间Z。通过优化行为重构损失来训练生成模型g:Z→Θ，确保潜在空间按功能相似性而非参数化邻近度组织。我们推测，这个流形固有的维数是环境复杂性的函数，而不是策略网络大小的函数。我们在连续控制领域验证了该方法，结果显示标准策略网络的参数化可以压缩至原来五个数量级，同时保留大部分表达能力。作为副产品，我们展示了学习到的流形可以通过在潜在空间Z中操作的策略梯度实现任务特定的适应。 

---
# Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation 

**Title (ZH)**: 基于检索增强的AI生成患者门户消息的护栏：错误分类学构建与大规模评估 

**Authors**: Wenyuan Chen, Fateme Nateghi Haredasht, Kameron C. Black, Francois Grolleau, Emily Alsentzer, Jonathan H. Chen, Stephen P. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.22565)  

**Abstract**: Asynchronous patient-clinician messaging via EHR portals is a growing source of clinician workload, prompting interest in large language models (LLMs) to assist with draft responses. However, LLM outputs may contain clinical inaccuracies, omissions, or tone mismatches, making robust evaluation essential. Our contributions are threefold: (1) we introduce a clinically grounded error ontology comprising 5 domains and 59 granular error codes, developed through inductive coding and expert adjudication; (2) we develop a retrieval-augmented evaluation pipeline (RAEC) that leverages semantically similar historical message-response pairs to improve judgment quality; and (3) we provide a two-stage prompting architecture using DSPy to enable scalable, interpretable, and hierarchical error detection. Our approach assesses the quality of drafts both in isolation and with reference to similar past message-response pairs retrieved from institutional archives. Using a two-stage DSPy pipeline, we compared baseline and reference-enhanced evaluations on over 1,500 patient messages. Retrieval context improved error identification in domains such as clinical completeness and workflow appropriateness. Human validation on 100 messages demonstrated superior agreement (concordance = 50% vs. 33%) and performance (F1 = 0.500 vs. 0.256) of context-enhanced labels vs. baseline, supporting the use of our RAEC pipeline as AI guardrails for patient messaging. 

**Abstract (ZH)**: 基于EHR portals的异步患者-临床医生消息交流是临床医生工作负荷的一个日益增长的来源，激发了对大型语言模型（LLMs）的兴趣以帮助生成草稿回应。然而，LLM输出可能包含临床不准确、遗漏或语气不符，因此需要进行稳健的评价。我们的贡献如下：（1）我们引入了一个基于临床的错误本体，涵盖5个领域和59个细粒度错误代码，通过归纳编码和专家裁定开发；（2）我们开发了一种检索增强的评价管道（RAEC），利用语义相似的历史消息-回应对提升判断质量；（3）我们提供了一种两阶段提示架构，使用DSPy使错误检测可扩展、可解释和层次化。我们的方法评估草稿的质量，既孤立地也参照从机构档案中检索到的类似过去的消息-回应对。通过两阶段DSPy管道，我们在超过1,500条患者消息上比较了基本评价和参考增强评价。检索上下文在临床完整性和流程适宜性等领域提高了错误识别能力。对100条消息的人工验证显示，上下文增强标签的协议性和性能（宏F1分数）均优于基线，支持使用我们RAEC管道作为患者消息的AI防护栏。 

---
# Activation Function Design Sustains Plasticity in Continual Learning 

**Title (ZH)**: 激活函数设计维持连续学习中的塑性 

**Authors**: Lute Lillo, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2509.22562)  

**Abstract**: In independent, identically distributed (i.i.d.) training regimes, activation functions have been benchmarked extensively, and their differences often shrink once model size and optimization are tuned. In continual learning, however, the picture is different: beyond catastrophic forgetting, models can progressively lose the ability to adapt (referred to as loss of plasticity) and the role of the non-linearity in this failure mode remains underexplored. We show that activation choice is a primary, architecture-agnostic lever for mitigating plasticity loss. Building on a property-level analysis of negative-branch shape and saturation behavior, we introduce two drop-in nonlinearities (Smooth-Leaky and Randomized Smooth-Leaky) and evaluate them in two complementary settings: (i) supervised class-incremental benchmarks and (ii) reinforcement learning with non-stationary MuJoCo environments designed to induce controlled distribution and dynamics shifts. We also provide a simple stress protocol and diagnostics that link the shape of the activation to the adaptation under change. The takeaway is straightforward: thoughtful activation design offers a lightweight, domain-general way to sustain plasticity in continual learning without extra capacity or task-specific tuning. 

**Abstract (ZH)**: 在连续学习中，激活函数的选择是减少塑性损失的主要且架构无关的手段：基于负支形状和饱和行为的性质分析，引入两种即插即用的非线性函数（Smooth-Leaky和Randomized Smooth-Leaky），并在监督类增量基准和诱导分布与动力学变化的非稳态MuJoCo环境中进行强化学习评估，并提供简单的压力测试协议和诊断工具，将激活函数的形状与变化下的适应性联系起来。结论是：精心设计的激活函数提供了一种轻量级且通用的方法，在无需额外容量或任务特定调优的情况下维持连续学习中的塑性。 

---
# ConQuER: Modular Architectures for Control and Bias Mitigation in IQP Quantum Generative Models 

**Title (ZH)**: ConQuER：用于IQP量子生成模型的控制和偏见缓解模块化架构 

**Authors**: Xiaocheng Zou, Shijin Duan, Charles Fleming, Gaowen Liu, Ramana Rao Kompella, Shaolei Ren, Xiaolin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22551)  

**Abstract**: Quantum generative models based on instantaneous quantum polynomial (IQP) circuits show great promise in learning complex distributions while maintaining classical trainability. However, current implementations suffer from two key limitations: lack of controllability over generated outputs and severe generation bias towards certain expected patterns. We present a Controllable Quantum Generative Framework, ConQuER, which addresses both challenges through a modular circuit architecture. ConQuER embeds a lightweight controller circuit that can be directly combined with pre-trained IQP circuits to precisely control the output distribution without full retraining. Leveraging the advantages of IQP, our scheme enables precise control over properties such as the Hamming Weight distribution with minimal parameter and gate overhead. In addition, inspired by the controller design, we extend this modular approach through data-driven optimization to embed implicit control paths in the underlying IQP architecture, significantly reducing generation bias on structured datasets. ConQuER retains efficient classical training properties and high scalability. We experimentally validate ConQuER on multiple quantum state datasets, demonstrating its superior control accuracy and balanced generation performance, only with very low overhead cost over original IQP circuits. Our framework bridges the gap between the advantages of quantum computing and the practical needs of controllable generation modeling. 

**Abstract (ZH)**: 基于瞬时量子多项式（IQP）电路的量子生成模型展现了在学习复杂分布的同时保持经典可训练性的巨大潜力。然而，当前实现面临两个关键限制：生成输出的可控性不足以及严重偏向某些预期模式。我们提出了一种可控量子生成框架ConQuER，该框架通过模块化电路架构解决这两个挑战。ConQuER嵌入了一个轻量级的控制器电路，可以与预训练的IQP电路直接结合，无需完全重新训练即可精确控制输出分布。利用IQP的优势，我们的方案能够在极小的参数和门操作开销下对诸如汉明重量分布等属性实现精确控制。此外，受控制器设计的启发，我们通过数据驱动的优化将这种模块化方法扩展到在底层IQP架构中嵌入隐式控制路径，显著降低了结构化数据集的生成偏差。ConQuER保留了高效的经典训练特性和高可扩展性。我们通过多种量子态数据集的实验验证了ConQuER，展示了其优越的控制精度和均衡的生成性能，仅实现了非常低的开销成本。该框架填补了量子计算优势与可控生成建模实际需求之间的差距。 

---
# Does AI Coaching Prepare us for Workplace Negotiations? 

**Title (ZH)**: AI教练是否为职场谈判做准备？ 

**Authors**: Veda Duddu, Jash Rajesh Parekh, Andy Mao, Hanyi Min, Ziang Xiao, Vedant Das Swain, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2509.22545)  

**Abstract**: Workplace negotiations are undermined by psychological barriers, which can even derail well-prepared tactics. AI offers personalized and always -- available negotiation coaching, yet its effectiveness for negotiation preparedness remains unclear. We built Trucey, a prototype AI coach grounded in Brett's negotiation model. We conducted a between-subjects experiment (N=267), comparing Trucey, ChatGPT, and a traditional negotiation Handbook, followed by in-depth interviews (N=15). While Trucey showed the strongest reductions in fear relative to both comparison conditions, the Handbook outperformed both AIs in usability and psychological empowerment. Interviews revealed that the Handbook's comprehensive, reviewable content was crucial for participants' confidence and preparedness. In contrast, although participants valued AI's rehearsal capability, its guidance often felt verbose and fragmented -- delivered in bits and pieces that required additional effort -- leaving them uncertain or overwhelmed. These findings challenge assumptions of AI superiority and motivate hybrid designs that integrate structured, theory-driven content with targeted rehearsal, clear boundaries, and adaptive scaffolds to address psychological barriers and support negotiation preparedness. 

**Abstract (ZH)**: 工作场所的谈判因心理障碍而受阻，甚至会破坏精心准备的策略。尽管AI提供个性化且随时可用的谈判辅导，其在谈判准备方面的有效性仍有待明确。我们构建了基于Brett谈判模型的Trucey原型AI教练。我们进行了一个被试间实验（N=267），比较了Trucey、ChatGPT和传统谈判手册的效果，并随后进行了深入访谈（N=15）。虽然Trucey在减少恐惧方面比两个对照组表现更好，但传统手册在可用性和心理 empowerment 方面的表现优于两者。访谈显示，手册详实且可复查的内容是参与者信心和准备的关键。相比之下，尽管参与者重视AI的模拟练习能力，但其指导往往显得冗长且碎片化，需要额外的努力，这使得他们感到困惑或不知所措。这些发现挑战了AI优越性的假设，并促使人们采用结合结构化、理论驱动的内容与目标化模拟练习、清晰边界和适应性支架的混合设计，以应对心理障碍并支持谈判准备。 

---
# InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models 

**Title (ZH)**: InfiR2: 一种增强推理的语言模型的全面FP8训练食谱 

**Authors**: Wenjun Wang, Shuo Cai, Congkai Xie, Mingfa Feng, Yiming Zhang, Zhen Li, Kejing Yang, Ming Li, Jiannong Cao, Yuan Xie, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22536)  

**Abstract**: The immense computational cost of training Large Language Models (LLMs) presents a major barrier to innovation. While FP8 training offers a promising solution with significant theoretical efficiency gains, its widespread adoption has been hindered by the lack of a comprehensive, open-source training recipe. To bridge this gap, we introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training. 

**Abstract (ZH)**: 大规模语言模型（LLMs）训练的巨大计算成本是创新的主要障碍。虽然FP8训练以其显著的理论效率提升提供了有希望的解决方案，但由于缺乏全面的开源训练食谱，其广泛应用受到了阻碍。为解决这一问题，我们介绍了一种端到端的FP8训练食谱，该食谱无缝地结合了持续预训练和监督微调。我们的方法采用精细的混合粒度量化策略，在保持数值保真度的同时最大化计算效率。通过广泛的实验，包括在160亿标记语料上继续预训练模型，我们证明我们的食谱不仅非常稳定，而且几乎是无损的，在一系列推理基准测试中达到与BF16基线相当的性能。关键的是，这在提高效率方面取得了显著进展，包括训练时间减少高达22%，峰值内存使用量降低14%，吞吐量提高19%。我们的结果确立了FP8作为一种实用且可靠的BF16替代方案的地位，并将发布相应的代码以进一步促进大规模模型训练的民主化。 

---
# Mental Health Impacts of AI Companions: Triangulating Social Media Quasi-Experiments, User Perspectives, and Relational Theory 

**Title (ZH)**: 人工智能伴侣对心理健康的影响：多方验证的社会媒体准实验、用户视角与关系理论综合研究 

**Authors**: Yunhao Yuan, Jiaxun Zhang, Talayeh Aledavood, Renwen Zhang, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2509.22505)  

**Abstract**: AI-powered companion chatbots (AICCs) such as Replika are increasingly popular, offering empathetic interactions, yet their psychosocial impacts remain unclear. We examined how engaging with AICCs shaped wellbeing and how users perceived these experiences. First, we conducted a large-scale quasi-experimental study of longitudinal Reddit data, applying stratified propensity score matching and Difference-in-Differences regression. Findings revealed mixed effects -- greater affective and grief expression, readability, and interpersonal focus, alongside increases in language about loneliness and suicidal ideation. Second, we complemented these results with 15 semi-structured interviews, which we thematically analyzed and contextualized using Knapp's relationship development model. We identified trajectories of initiation, escalation, and bonding, wherein AICCs provided emotional validation and social rehearsal but also carried risks of over-reliance and withdrawal. Triangulating across methods, we offer design implications for AI companions that scaffold healthy boundaries, support mindful engagement, support disclosure without dependency, and surface relationship stages -- maximizing psychosocial benefits while mitigating risks. 

**Abstract (ZH)**: 基于AI的同伴聊天机器人（AICCs）如Replika越来越受欢迎，它们提供同理心互动，但其心理社会影响尚不清晰。我们探讨了与AICCs互动如何影响福祉以及用户对这些体验的看法。首先，我们通过纵向Reddit数据进行大规模准实验研究，采用分层倾向得分匹配和差异回归分析。结果发现复杂影响——情绪表达和哀悼表达增加、可读性提高和人际焦点增强，同时孤独感和自杀念头的话语增多。其次，我们通过15次半结构化访谈进行了补充研究，并利用Knapp的关系发展模型对访谈进行主题分析和背景说明。我们识别了互动、升级和整合的轨迹，其中AICCs提供了情感验证和社交练习，但也存在过度依赖和疏远的风险。综合多种方法，我们提出了设计建议，以促进健康边界、支持有意识的互动、支持披露而非依赖，并揭示关系阶段——最大化心理社会益处的同时减轻风险。 

---
# Ontological foundations for contrastive explanatory narration of robot plans 

**Title (ZH)**: 基于本体论基础的对比解释性叙述机器人计划 

**Authors**: Alberto Olivares-Alarcos, Sergi Foix, Júlia Borràs, Gerard Canal, Guillem Alenyà  

**Link**: [PDF](https://arxiv.org/pdf/2509.22493)  

**Abstract**: Mutual understanding of artificial agents' decisions is key to ensuring a trustworthy and successful human-robot interaction. Hence, robots are expected to make reasonable decisions and communicate them to humans when needed. In this article, the focus is on an approach to modeling and reasoning about the comparison of two competing plans, so that robots can later explain the divergent result. First, a novel ontological model is proposed to formalize and reason about the differences between competing plans, enabling the classification of the most appropriate one (e.g., the shortest, the safest, the closest to human preferences, etc.). This work also investigates the limitations of a baseline algorithm for ontology-based explanatory narration. To address these limitations, a novel algorithm is presented, leveraging divergent knowledge between plans and facilitating the construction of contrastive narratives. Through empirical evaluation, it is observed that the explanations excel beyond the baseline method. 

**Abstract (ZH)**: 人工代理决策间的相互理解是确保人机交互可信和成功的关键。因此，机器人在必要时应作出合理决策并进行沟通。本文专注于一种建模和推理两个竞争计划差异的方法，以便机器人后来可以解释不同的结果。首先，提出了一种新颖的本体模型来形式化和推理竞争计划之间的差异，从而实现对最合适的计划（如最短的、最安全的、最符合人类偏好的等）的分类。此外，还研究了基于本体解释性叙述的基础算法的局限性。为了解决这些局限性，提出了一种新的算法，利用计划间的差异性知识，促进对比叙述的构建。通过实证评估，发现这些解释超越了基准方法。 

---
# A Machine Learning Pipeline for Multiple Sclerosis Biomarker Discovery: Comparing explainable AI and Traditional Statistical Approaches 

**Title (ZH)**: 一种多发性硬化生物标志物发现的机器学习管道：可解释AI与传统统计方法的比较 

**Authors**: Samuele Punzo, Silvia Giulia Galfrè, Francesco Massafra, Alessandro Maglione, Corrado Priami, Alina Sîrbu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22484)  

**Abstract**: We present a machine learning pipeline for biomarker discovery in Multiple Sclerosis (MS), integrating eight publicly available microarray datasets from Peripheral Blood Mononuclear Cells (PBMC). After robust preprocessing we trained an XGBoost classifier optimized via Bayesian search. SHapley Additive exPlanations (SHAP) were used to identify key features for model prediction, indicating thus possible biomarkers. These were compared with genes identified through classical Differential Expression Analysis (DEA). Our comparison revealed both overlapping and unique biomarkers between SHAP and DEA, suggesting complementary strengths. Enrichment analysis confirmed the biological relevance of SHAP-selected genes, linking them to pathways such as sphingolipid signaling, Th1/Th2/Th17 cell differentiation, and Epstein-Barr virus infection all known to be associated with MS. This study highlights the value of combining explainable AI (xAI) with traditional statistical methods to gain deeper insights into disease mechanism. 

**Abstract (ZH)**: 一种基于机器学习的多发性硬化症生物标志物发现管道：整合八个多公开的外周血单核细胞微阵列数据集 

---
# OFMU: Optimization-Driven Framework for Machine Unlearning 

**Title (ZH)**: 基于优化驱动的机器卸学框架（OFMU） 

**Authors**: Sadia Asif, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2509.22483)  

**Abstract**: Large language models deployed in sensitive applications increasingly require the ability to unlearn specific knowledge, such as user requests, copyrighted materials, or outdated information, without retraining from scratch to ensure regulatory compliance, user privacy, and safety. This task, known as machine unlearning, aims to remove the influence of targeted data (forgetting) while maintaining performance on the remaining data (retention). A common approach is to formulate this as a multi-objective problem and reduce it to a single-objective problem via scalarization, where forgetting and retention losses are combined using a weighted sum. However, this often results in unstable training dynamics and degraded model utility due to conflicting gradient directions. To address these challenges, we propose OFMU, a penalty-based bi-level optimization framework that explicitly prioritizes forgetting while preserving retention through a hierarchical structure. Our method enforces forgetting via an inner maximization step that incorporates a similarity-aware penalty to decorrelate the gradients of the forget and retention objectives, and restores utility through an outer minimization step. To ensure scalability, we develop a two-loop algorithm with provable convergence guarantees under both convex and non-convex regimes. We further provide a rigorous theoretical analysis of convergence rates and show that our approach achieves better trade-offs between forgetting efficacy and model utility compared to prior methods. Extensive experiments across vision and language benchmarks demonstrate that OFMU consistently outperforms existing unlearning methods in both forgetting efficacy and retained utility. 

**Abstract (ZH)**: 基于惩罚的分层优化框架OFMU：在保持保留性能的同时有效遗忘特定知识 

---
# Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving 

**Title (ZH)**: 探索解空间发散及其对大型语言模型问题解决的影响 

**Authors**: Hang Li, Kaiqi Yang, Yucheng Chu, Hui Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22480)  

**Abstract**: Large language models (LLMs) have been widely used for problem-solving tasks. Most recent work improves their performance through supervised fine-tuning (SFT) with labeled data or reinforcement learning (RL) from task feedback. In this paper, we study a new perspective: the divergence in solutions generated by LLMs for a single problem. We show that higher solution divergence is positively related to better problem-solving abilities across various models. Based on this finding, we propose solution divergence as a novel metric that can support both SFT and RL strategies. We test this idea on three representative problem domains and find that using solution divergence consistently improves success rates. These results suggest that solution divergence is a simple but effective tool for advancing LLM training and evaluation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在问题解决任务中被广泛应用。最近的研究主要通过有监督微调（SFT）和基于任务反馈的强化学习（RL）来提高其性能。在本文中，我们从一个新的角度研究了LLMs为单一问题生成的不同解决方案之间的发散性。我们发现，解决方案的发散性与模型的问题解决能力正相关。基于这一发现，我们提出将解决方案的发散性作为一种新的度量标准，以支持SFT和RL策略。我们在三个代表性的问题领域进行了测试，并发现使用解决方案的发散性可以一致地提高成功率。这些结果表明，解决方案的发散性是一种简单而有效的工具，可用于推进LLM的训练和评估。 

---
# Evaluating the Limits of Large Language Models in Multilingual Legal Reasoning 

**Title (ZH)**: 评估大型语言模型在多语言法律推理中的局限性 

**Authors**: Antreas Ioannou, Andreas Shiamishis, Nora Hollenstein, Nezihe Merve Gürel  

**Link**: [PDF](https://arxiv.org/pdf/2509.22472)  

**Abstract**: In an era dominated by Large Language Models (LLMs), understanding their capabilities and limitations, especially in high-stakes fields like law, is crucial. While LLMs such as Meta's LLaMA, OpenAI's ChatGPT, Google's Gemini, DeepSeek, and other emerging models are increasingly integrated into legal workflows, their performance in multilingual, jurisdictionally diverse, and adversarial contexts remains insufficiently explored. This work evaluates LLaMA and Gemini on multilingual legal and non-legal benchmarks, and assesses their adversarial robustness in legal tasks through character and word-level perturbations. We use an LLM-as-a-Judge approach for human-aligned evaluation. We moreover present an open-source, modular evaluation pipeline designed to support multilingual, task-diverse benchmarking of any combination of LLMs and datasets, with a particular focus on legal tasks, including classification, summarization, open questions, and general reasoning. Our findings confirm that legal tasks pose significant challenges for LLMs with accuracies often below 50% on legal reasoning benchmarks such as LEXam, compared to over 70% on general-purpose tasks like XNLI. In addition, while English generally yields more stable results, it does not always lead to higher accuracy. Prompt sensitivity and adversarial vulnerability is also shown to persist across languages. Finally, a correlation is found between the performance of a language and its syntactic similarity to English. We also observe that LLaMA is weaker than Gemini, with the latter showing an average advantage of about 24 percentage points across the same task. Despite improvements in newer LLMs, challenges remain in deploying them reliably for critical, multilingual legal applications. 

**Abstract (ZH)**: 在大型语言模型主导的时代：理解其在法律等高 stakes 领域的能力与局限性 

---
# Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining 

**Title (ZH)**: 学习邻域：对比 free 多模态自监督分子图预训练 

**Authors**: Boshra Ariguib, Mathias Niepert, Andrei Manolache  

**Link**: [PDF](https://arxiv.org/pdf/2509.22468)  

**Abstract**: High-quality molecular representations are essential for property prediction and molecular design, yet large labeled datasets remain scarce. While self-supervised pretraining on molecular graphs has shown promise, many existing approaches either depend on hand-crafted augmentations or complex generative objectives, and often rely solely on 2D topology, leaving valuable 3D structural information underutilized. To address this gap, we introduce C-FREE (Contrast-Free Representation learning on Ego-nets), a simple framework that integrates 2D graphs with ensembles of 3D conformers. C-FREE learns molecular representations by predicting subgraph embeddings from their complementary neighborhoods in the latent space, using fixed-radius ego-nets as modeling units across different conformers. This design allows us to integrate both geometric and topological information within a hybrid Graph Neural Network (GNN)-Transformer backbone, without negatives, positional encodings, or expensive pre-processing. Pretraining on the GEOM dataset, which provides rich 3D conformational diversity, C-FREE achieves state-of-the-art results on MoleculeNet, surpassing contrastive, generative, and other multimodal self-supervised methods. Fine-tuning across datasets with diverse sizes and molecule types further demonstrates that pretraining transfers effectively to new chemical domains, highlighting the importance of 3D-informed molecular representations. 

**Abstract (ZH)**: 高质量的分子表示对于性质预测和分子设计至关重要，但大型带标签数据集仍然稀缺。虽然基于分子图的自监督预训练显示出潜力，但现有许多方法要么依赖手工构造的增强方法，要么具有复杂的生成目标，且往往仅依赖2D拓扑信息，而使宝贵的3D结构信息未能充分利用。为解决这一问题，我们引入了C-FREE（基于ego-网络的无 Contrastive 表示学习），这是一种简单框架，将2D图与3D同分异构体的集合相结合。C-FREE通过在潜在空间中预测子图嵌入的方式学习分子表示，使用固定半径的ego-网络作为建模单元，在不同同分异构体之间进行。这种设计允许我们在混合图神经网络（GNN）-变压器骨干网络中整合几何和拓扑信息，无需负样本、位置编码或昂贵的预处理。在提供丰富3D构象多样性的GEOM数据集上进行预训练，C-FREE在MoleculeNet上取得了最先进的成果，超越了对比学习、生成和其他多模态自监督方法。在不同大小和分子类型的多个数据集上的微调进一步证明了预训练可以有效转移至新的化学领域，强调了3D启发的分子表示的重要性。 

---
# MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark 

**Title (ZH)**: MDAR：多场景动态音频推理基准 

**Authors**: Hui Li, Changhao Jiang, Hongyu Wang, Ming Zhang, Jiajun Sun, Zhixiong Yang, Yifei Cao, Shihan Dou, Xiaoran Fan, Baoyu Fan, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22461)  

**Abstract**: The ability to reason from audio, including speech, paralinguistic cues, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce MDAR, a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. MDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on MDAR and observe that they exhibit limitations in complex reasoning tasks. On single-choice questions, Qwen2.5-Omni (open-source) achieves 76.67% accuracy, whereas GPT-4o Audio (closed-source) reaches 68.47%; however, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice and open-ended tasks. Across all three question types, no model achieves 80% performance. These findings underscore the unique challenges posed by MDAR and its value as a benchmark for advancing audio reasoning this http URL and benchmark can be found at this https URL. 

**Abstract (ZH)**: 音频推理能力对于AI代理在现实场景中有效交互至关重要。现有的基准主要集中在静态或单场景设置上，并未能充分捕捉多个说话人、展开事件和异质音频源相互作用的场景。为应对这些挑战，我们引入了MDAR这一基准，用于评估模型在复杂、多场景和动态演化的音频推理任务中的表现。MDAR包含3000个精心策划的问答对，关联到多样化的音频片段，涵盖五类复杂的推理场景，涉及三种问题类型。我们基于MDAR对26个最先进的音频语言模型进行了基准测试，并观察到它们在复杂推理任务中存在局限性。在单选题方面，Qwen2.5-Omni（开源）的准确率为76.67%，而GPT-4o Audio（闭源）达到68.47%；然而，GPT-4o Audio在更具挑战性的多项选择和开放型任务中显著优于Qwen2.5-Omni。在所有三种问题类型中，没有一个模型达到80%的性能。这些发现凸显了MDAR带来的独特挑战及其作为推动音频推理基准的价值。更多信息，请访问：this https URL。 

---
# Physics-informed GNN for medium-high voltage AC power flow with edge-aware attention and line search correction operator 

**Title (ZH)**: 考虑边感知注意力和线路搜索校正运算符的物理知情GNN在中高压AC潮流中的应用 

**Authors**: Changhun Kim, Timon Conrad, Redwanul Karim, Julian Oelhaf, David Riebesel, Tomás Arias-Vergara, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.22458)  

**Abstract**: Physics-informed graph neural networks (PIGNNs) have emerged as fast AC power-flow solvers that can replace classic Newton--Raphson (NR) solvers, especially when thousands of scenarios must be evaluated. However, current PIGNNs still need accuracy improvements at parity speed; in particular, the physics loss is inoperative at inference, which can deter operational adoption. We address this with PIGNN-Attn-LS, combining an edge-aware attention mechanism that explicitly encodes line physics via per-edge biases, capturing the grid's anisotropy, with a backtracking line-search-based globalized correction operator that restores an operative decrease criterion at inference. Training and testing use a realistic High-/Medium-Voltage scenario generator, with NR used only to construct reference states. On held-out HV cases consisting of 4--32-bus grids, PIGNN-Attn-LS achieves a test RMSE of 0.00033 p.u. in voltage and 0.08$^\circ$ in angle, outperforming the PIGNN-MLP baseline by 99.5\% and 87.1\%, respectively. With streaming micro-batches, it delivers 2--5$\times$ faster batched inference than NR on 4--1024-bus grids. 

**Abstract (ZH)**: 基于物理信息的图神经网络（PIGNNs）已经发展成为快速的交流功率流求解器，可以在成千上万种场景评估时替代经典的牛顿-拉夫逊（NR）求解器。然而，当前的PIGNNs仍需要在保持速度的同时提高准确性；特别是，在推理时无法发挥作用的物理损失项可能阻碍其实用化。我们通过引入PIGNN-Attn-LS解决了这一问题，该方法结合了边感知注意力机制，通过每边偏差显式地编码线路物理特性，捕获电网的各向异性，并采用基于回溯线搜索的全局校正算子，在推理时恢复了有效的减少准则。训练和测试使用一个现实的高/中压场景生成器，仅使用NR构建参考状态。在由4-32节点电网组成的保留验证案例中，PIGNN-Attn-LS实现了电压0.00033 p.u.和角度0.08°的测试RMSE，分别比PIGNN-MLP基线高出99.5%和87.1%。使用流式微批量处理，它在4-1024节点电网上的批量推理速度比NR快2-5倍。 

---
# Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers 

**Title (ZH)**: 连接科莫洛夫复杂性和深度学习：变换器的渐近最优描述长度目标函数 

**Authors**: Peter Shaw, James Cohan, Jacob Eisenstein, Kristina Toutanova  

**Link**: [PDF](https://arxiv.org/pdf/2509.22445)  

**Abstract**: The Minimum Description Length (MDL) principle offers a formal framework for applying Occam's razor in machine learning. However, its application to neural networks such as Transformers is challenging due to the lack of a principled, universal measure for model complexity. This paper introduces the theoretical notion of asymptotically optimal description length objectives, grounded in the theory of Kolmogorov complexity. We establish that a minimizer of such an objective achieves optimal compression, for any dataset, up to an additive constant, in the limit as model resource bounds increase. We prove that asymptotically optimal objectives exist for Transformers, building on a new demonstration of their computational universality. We further show that such objectives can be tractable and differentiable by constructing and analyzing a variational objective based on an adaptive Gaussian mixture prior. Our empirical analysis shows that this variational objective selects for a low-complexity solution with strong generalization on an algorithmic task, but standard optimizers fail to find such solutions from a random initialization, highlighting key optimization challenges. More broadly, by providing a theoretical framework for identifying description length objectives with strong asymptotic guarantees, we outline a potential path towards training neural networks that achieve greater compression and generalization. 

**Abstract (ZH)**: 最小描述长度（MDL）原则为在机器学习中应用奥卡姆剃刀提供了一个形式化的框架。然而，将其应用于如变换器等神经网络具有挑战性，原因在于缺乏一个有原则性的、普遍适用的模型复杂度度量。本文介绍了基于科莫戈罗夫复杂性理论的渐近最优描述长度目标的理论概念。我们建立起这样的目标函数的一个最小化者能够对于任意数据集在模型资源界限增加时的极限状态下实现最优压缩，其中误差在可加常数之内。我们证明了对于变换器存在渐近最优目标，这是基于对其计算普遍性的新证明。进一步地，我们表明这样的目标可以是可处理且可微的，通过构建并分析基于自适应高斯混合先验的变分目标来实现这一点。我们的实验分析表明，这种变分目标在算法任务上选择了一个低复杂度的解，并且具有较强的泛化能力，但标准优化器从随机初始化无法找到这样的解，突出了关键的优化挑战。更广泛地，通过提供一个理论框架来识别具有强大渐近保证的描述长度目标，本文勾勒出了一条潜在研究路径，旨在训练能够实现更高压缩和泛化的神经网络。 

---
# Learning to Ball: Composing Policies for Long-Horizon Basketball Moves 

**Title (ZH)**: 学习运球：组成面向长时程篮球动作的策略 

**Authors**: Pei Xu, Zhen Wu, Ruocheng Wang, Vishnu Sarukkai, Kayvon Fatahalian, Ioannis Karamouzas, Victor Zordan, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22442)  

**Abstract**: Learning a control policy for a multi-phase, long-horizon task, such as basketball maneuvers, remains challenging for reinforcement learning approaches due to the need for seamless policy composition and transitions between skills. A long-horizon task typically consists of distinct subtasks with well-defined goals, separated by transitional subtasks with unclear goals but critical to the success of the entire task. Existing methods like the mixture of experts and skill chaining struggle with tasks where individual policies do not share significant commonly explored states or lack well-defined initial and terminal states between different phases. In this paper, we introduce a novel policy integration framework to enable the composition of drastically different motor skills in multi-phase long-horizon tasks with ill-defined intermediate states. Based on that, we further introduce a high-level soft router to enable seamless and robust transitions between the subtasks. We evaluate our framework on a set of fundamental basketball skills and challenging transitions. Policies trained by our approach can effectively control the simulated character to interact with the ball and accomplish the long-horizon task specified by real-time user commands, without relying on ball trajectory references. 

**Abstract (ZH)**: 一种新型策略集成框架：在具有非明确中间状态的多阶段长期任务中组成截然不同的运动技能及其平滑过渡 

---
# Chimera: Diagnosing Shortcut Learning in Visual-Language Understanding 

**Title (ZH)**: chimera: 视觉-语言理解中捷径学习的诊断 

**Authors**: Ziheng Chi, Yifan Hou, Chenxi Pang, Shaobo Cui, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22437)  

**Abstract**: Diagrams convey symbolic information in a visual format rather than a linear stream of words, making them especially challenging for AI models to process. While recent evaluations suggest that vision-language models (VLMs) perform well on diagram-related benchmarks, their reliance on knowledge, reasoning, or modality shortcuts raises concerns about whether they genuinely understand and reason over diagrams. To address this gap, we introduce Chimera, a comprehensive test suite comprising 7,500 high-quality diagrams sourced from Wikipedia; each diagram is annotated with its symbolic content represented by semantic triples along with multi-level questions designed to assess four fundamental aspects of diagram comprehension: entity recognition, relation understanding, knowledge grounding, and visual reasoning. We use Chimera to measure the presence of three types of shortcuts in visual question answering: (1) the visual-memorization shortcut, where VLMs rely on memorized visual patterns; (2) the knowledge-recall shortcut, where models leverage memorized factual knowledge instead of interpreting the diagram; and (3) the Clever-Hans shortcut, where models exploit superficial language patterns or priors without true comprehension. We evaluate 15 open-source VLMs from 7 model families on Chimera and find that their seemingly strong performance largely stems from shortcut behaviors: visual-memorization shortcuts have slight impact, knowledge-recall shortcuts play a moderate role, and Clever-Hans shortcuts contribute significantly. These findings expose critical limitations in current VLMs and underscore the need for more robust evaluation protocols that benchmark genuine comprehension of complex visual inputs (e.g., diagrams) rather than question-answering shortcuts. 

**Abstract (ZH)**: 图示以视觉格式而非线性字流传递符号信息，这使它们特别难以供AI模型处理。虽然最近的评估表明，视觉-语言模型(VLMs)在图示相关基准测试中表现良好，但它们依赖知识、推理或模态捷径的方式引发了对其是否真正理解并推理图示的担忧。为解决这一差距，我们引入了 Chimera，一个包含7,500个高质量图示的综合测试套件，这些图示来源于维基百科；每个图示都用语义三元组标注其符号内容，并配有多层次问题以评估图示理解的四个基本方面：实体识别、关系理解、知识接地和视觉推理。我们使用Chimera来测量视觉问答中三种捷径的存在：（1）视觉记忆捷径，其中VLMs依赖于记忆中的视觉模式；（2）知识回忆捷径，其中模型利用记忆中的事实知识而非解释图示；（3）Clever-Hans捷径，其中模型利用表面的语言模式或先验知识而没有真正的理解。我们将15个开源VLMs从7个模型家族进行Chimera上的评估，并发现它们看似强大的表现主要源于捷径行为：视觉记忆捷径影响轻微，知识回忆捷径起中等作用，而Clever-Hans捷径贡献显著。这些发现揭示了当前VLMs的关键局限性，并突显了需要使用更 robust 的评估协议来基准测试对复杂视觉输入（例如，图示）的真正理解，而非问答捷径。 

---
# Global Convergence in Neural ODEs: Impact of Activation Functions 

**Title (ZH)**: 全局收敛性在神经ODE中的影响：激活函数的作用 

**Authors**: Tianxiang Gao, Siyuan Sun, Hailiang Liu, Hongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22436)  

**Abstract**: Neural Ordinary Differential Equations (ODEs) have been successful in various applications due to their continuous nature and parameter-sharing efficiency. However, these unique characteristics also introduce challenges in training, particularly with respect to gradient computation accuracy and convergence analysis. In this paper, we address these challenges by investigating the impact of activation functions. We demonstrate that the properties of activation functions, specifically smoothness and nonlinearity, are critical to the training dynamics. Smooth activation functions guarantee globally unique solutions for both forward and backward ODEs, while sufficient nonlinearity is essential for maintaining the spectral properties of the Neural Tangent Kernel (NTK) during training. Together, these properties enable us to establish the global convergence of Neural ODEs under gradient descent in overparameterized regimes. Our theoretical findings are validated by numerical experiments, which not only support our analysis but also provide practical guidelines for scaling Neural ODEs, potentially leading to faster training and improved performance in real-world applications. 

**Abstract (ZH)**: 神经普通微分方程（ODEs）因其连续性质和参数共享效率，在各种应用中取得了成功。然而，这些独特特性也带来了训练挑战，尤其是在梯度计算准确性和收敛性分析方面。本文通过研究激活函数的影响来应对这些挑战。我们证明了激活函数的特性，特别是平滑性和非线性，在训练动态中至关重要。平滑的激活函数确保前向和反向ODEs具有全局唯一解，而足够的非线性则对于在训练过程中保持神经核张量（NTK）的频谱特性是必不可少的。这些特性共同使我们能够在参数过量情况下通过梯度下降方法证明神经ODEs的全局收敛性。我们的理论发现通过数值实验得到了验证，不仅支持了我们的分析，还为在实际应用中扩展神经ODEs提供了实用指南，可能加速训练并提高性能。 

---
# An Ontology for Unified Modeling of Tasks, Actions, Environments, and Capabilities in Personal Service Robotics 

**Title (ZH)**: 统一建模任务、行为、环境和能力的本体论研究 

**Authors**: Margherita Martorana, Francesca Urgese, Ilaria Tiddi, Stefan Schlobach  

**Link**: [PDF](https://arxiv.org/pdf/2509.22434)  

**Abstract**: Personal service robots are increasingly used in domestic settings to assist older adults and people requiring support. Effective operation involves not only physical interaction but also the ability to interpret dynamic environments, understand tasks, and choose appropriate actions based on context. This requires integrating both hardware components (e.g. sensors, actuators) and software systems capable of reasoning about tasks, environments, and robot capabilities. Frameworks such as the Robot Operating System (ROS) provide open-source tools that help connect low-level hardware with higher-level functionalities. However, real-world deployments remain tightly coupled to specific platforms. As a result, solutions are often isolated and hard-coded, limiting interoperability, reusability, and knowledge sharing. Ontologies and knowledge graphs offer a structured way to represent tasks, environments, and robot capabilities. Existing ontologies, such as the Socio-physical Model of Activities (SOMA) and the Descriptive Ontology for Linguistic and Cognitive Engineering (DOLCE), provide models for activities, spatial relationships, and reasoning structures. However, they often focus on specific domains and do not fully capture the connection between environment, action, robot capabilities, and system-level integration. In this work, we propose the Ontology for roBOts and acTions (OntoBOT), which extends existing ontologies to provide a unified representation of tasks, actions, environments, and capabilities. Our contributions are twofold: (1) we unify these aspects into a cohesive ontology to support formal reasoning about task execution, and (2) we demonstrate its generalizability by evaluating competency questions across four embodied agents - TIAGo, HSR, UR3, and Stretch - showing how OntoBOT enables context-aware reasoning, task-oriented execution, and knowledge sharing in service robotics. 

**Abstract (ZH)**: 基于行动的机器人本体（OntoBOT）：统一任务、行动、环境和能力的本体表示及其在服务机器人中的应用 

---
# Partial Parameter Updates for Efficient Distributed Training 

**Title (ZH)**: 部分参数更新以实现高效分布式训练 

**Authors**: Anastasiia Filippova, Angelos Katharopoulos, David Grangier, Ronan Collobert  

**Link**: [PDF](https://arxiv.org/pdf/2509.22418)  

**Abstract**: We introduce a memory- and compute-efficient method for low-communication distributed training. Existing methods reduce communication by performing multiple local updates between infrequent global synchronizations. We demonstrate that their efficiency can be significantly improved by restricting backpropagation: instead of updating all the parameters, each node updates only a fixed subset while keeping the remainder frozen during local steps. This constraint substantially reduces peak memory usage and training FLOPs, while a full forward pass over all parameters eliminates the need for cross-node activation exchange. Experiments on a $1.3$B-parameter language model trained across $32$ nodes show that our method matches the perplexity of prior low-communication approaches under identical token and bandwidth budgets while reducing training FLOPs and peak memory. 

**Abstract (ZH)**: 一种低通信分布式训练的内存和计算高效方法 

---
# Explaining multimodal LLMs via intra-modal token interactions 

**Title (ZH)**: 通过模内标记交互解释多模态LLM 

**Authors**: Jiawei Liang, Ruoyu Chen, Xianghao Jiao, Siyuan Liang, Shiming Liu, Qunli Zhang, Zheng Hu, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22415)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success across diverse vision-language tasks, yet their internal decision-making mechanisms remain insufficiently understood. Existing interpretability research has primarily focused on cross-modal attribution, identifying which image regions the model attends to during output generation. However, these approaches often overlook intra-modal dependencies. In the visual modality, attributing importance to isolated image patches ignores spatial context due to limited receptive fields, resulting in fragmented and noisy explanations. In the textual modality, reliance on preceding tokens introduces spurious activations. Failing to effectively mitigate these interference compromises attribution fidelity. To address these limitations, we propose enhancing interpretability by leveraging intra-modal interaction. For the visual branch, we introduce \textit{Multi-Scale Explanation Aggregation} (MSEA), which aggregates attributions over multi-scale inputs to dynamically adjust receptive fields, producing more holistic and spatially coherent visual explanations. For the textual branch, we propose \textit{Activation Ranking Correlation} (ARC), which measures the relevance of contextual tokens to the current token via alignment of their top-$k$ prediction rankings. ARC leverages this relevance to suppress spurious activations from irrelevant contexts while preserving semantically coherent ones. Extensive experiments across state-of-the-art MLLMs and benchmark datasets demonstrate that our approach consistently outperforms existing interpretability methods, yielding more faithful and fine-grained explanations of model behavior. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在各种视觉-语言任务中取得了显著成功，但其内部决策机制仍不够理解。现有的可解释性研究主要集中在跨模态归因上，识别模型在输出生成过程中关注的图像区域。然而，这些方法往往忽视了同一模态内的依赖关系。在视觉模态中，对孤立图像patches的重要性归因忽视了空间上下文，因为它们的感受野有限，导致碎片化和噪声解释。在文本模态中，依赖于前一个词令牌引入了虚假激活。未能有效缓解这些干扰削弱了归因的准确性。为了解决这些局限性，我们提出通过利用同一模态内的交互来增强可解释性。对于视觉分支，我们引入了多尺度解释聚合（MSEA），它通过对多尺度输入的归因进行聚合来动态调整感受野，产生更具整体性和空间连贯性的视觉解释。对于文本分支，我们提出了激活排名相关性（ARC），通过对其前k个预测排名进行对齐来测量上下文词对当前词的相关性。ARC 利用这种相关性抑制无关上下文的虚假激活，同时保留语义上连贯的激活。在多种先进的MLLMs和基准数据集上的广泛实验表明，我们的方法一致地优于现有的可解释性方法，提供了更忠实和精细的模型行为解释。 

---
# RAU: Reference-based Anatomical Understanding with Vision Language Models 

**Title (ZH)**: RAU：基于参考的解剖理解与视觉语言模型 

**Authors**: Yiwei Li, Yikang Liu, Jiaqi Guo, Lin Zhao, Zheyuan Zhang, Xiao Chen, Boris Mailhe, Ankush Mukherjee, Terrence Chen, Shanhui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22404)  

**Abstract**: Anatomical understanding through deep learning is critical for automatic report generation, intra-operative navigation, and organ localization in medical imaging; however, its progress is constrained by the scarcity of expert-labeled data. A promising remedy is to leverage an annotated reference image to guide the interpretation of an unlabeled target. Although recent vision-language models (VLMs) exhibit non-trivial visual reasoning, their reference-based understanding and fine-grained localization remain limited. We introduce RAU, a framework for reference-based anatomical understanding with VLMs. We first show that a VLM learns to identify anatomical regions through relative spatial reasoning between reference and target images, trained on a moderately sized dataset. We validate this capability through visual question answering (VQA) and bounding box prediction. Next, we demonstrate that the VLM-derived spatial cues can be seamlessly integrated with the fine-grained segmentation capability of SAM2, enabling localization and pixel-level segmentation of small anatomical regions, such as vessel segments. Across two in-distribution and two out-of-distribution datasets, RAU consistently outperforms a SAM2 fine-tuning baseline using the same memory setup, yielding more accurate segmentations and more reliable localization. More importantly, its strong generalization ability makes it scalable to out-of-distribution datasets, a property crucial for medical image applications. To the best of our knowledge, RAU is the first to explore the capability of VLMs for reference-based identification, localization, and segmentation of anatomical structures in medical images. Its promising performance highlights the potential of VLM-driven approaches for anatomical understanding in automated clinical workflows. 

**Abstract (ZH)**: 通过深度学习进行解剖理解对于医学影像的自动报告生成、术中导航和器官定位至关重要，但其进展受限于专家标注数据的稀缺性。一种有前景的解决方案是利用注释参考图像来指导未标注目标的解释。尽管近期的视觉-语言模型（VLMs）显示出了非平凡的视觉推理能力，但它们的参考驱动理解和细粒度定位能力仍有限。我们提出RAU，一种基于参考的解剖理解框架，使用VLMs。我们首先展示了VLM能够在适度大小的数据集上通过参考图像和目标图像之间的相对空间推理学习识别解剖区域。通过视觉问答（VQA）和边界框预测，验证了这一能力。随后，我们展示了VLM提取的空间线索可以无缝集成到SAM2的细粒度分割能力中，从而实现小解剖区域（如血管段）的精确定位和像素级分割。在两个在分布数据集和两个跨分布数据集中，RAU在相同的内存配置下始终优于SAM2微调基线，提供了更准确的分割和更可靠的定位。更重要的是，其强大的泛化能力使其可以扩展到跨分布数据集，这是医学影像应用中关键的属性。据我们所知，RAU是首次探索VLMs在医疗影像中进行基于参考的识别、定位和分割结构的能力。其有前景的性能突显了VLM驱动方法在自动化临床工作流程中进行解剖理解的潜力。 

---
# Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss 

**Title (ZH)**: 基于深度学习的adapted nnResU-Net介导的解剖特征优先损失跨解剖CT合成 

**Authors**: Javier Sequeiro González, Arthur Longuefosse, Miguel Díaz Benito, Álvaro García Martín, Fabien Baldacci  

**Link**: [PDF](https://arxiv.org/pdf/2509.22394)  

**Abstract**: We present a patch-based 3D nnUNet adaptation for MR to CT and CBCT to CT image translation using the multicenter SynthRAD2025 dataset, covering head and neck (HN), thorax (TH), and abdomen (AB) regions. Our approach leverages two main network configurations: a standard UNet and a residual UNet, both adapted from nnUNet for image synthesis. The Anatomical Feature-Prioritized (AFP) loss was introduced, which compares multilayer features extracted from a compact segmentation network trained on TotalSegmentator labels, enhancing reconstruction of clinically relevant structures. Input volumes were normalized per-case using zscore normalization for MRIs, and clipping plus dataset level zscore normalization for CBCT and CT. Training used 3D patches tailored to each anatomical region without additional data augmentation. Models were trained for 1000 and 1500 epochs, with AFP fine-tuning performed for 500 epochs using a combined L1+AFP objective. During inference, overlapping patches were aggregated via mean averaging with step size of 0.3, and postprocessing included reverse zscore normalization. Both network configurations were applied across all regions, allowing consistent model design while capturing local adaptations through residual learning and AFP loss. Qualitative and quantitative evaluation revealed that residual networks combined with AFP yielded sharper reconstructions and improved anatomical fidelity, particularly for bone structures in MR to CT and lesions in CBCT to CT, while L1only networks achieved slightly better intensity-based metrics. This methodology provides a stable solution for cross modality medical image synthesis, demonstrating the effectiveness of combining the automatic nnUNet pipeline with residual learning and anatomically guided feature losses. 

**Abstract (ZH)**: 基于补丁的3D nnUNet适应性研究：使用SynthRAD2025多中心数据集将MR转换为CT及CBCT转换为CT图像翻译，涵盖头部和颈部、胸腔和腹部区域 

---
# SpinGPT: A Large-Language-Model Approach to Playing Poker Correctly 

**Title (ZH)**: SpinGPT: 一种正确的扑克玩法的大语言模型方法 

**Authors**: Narada Maugin, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2509.22387)  

**Abstract**: The Counterfactual Regret Minimization (CFR) algorithm and its variants have enabled the development of pokerbots capable of beating the best human players in heads-up (1v1) cash games and competing with them in six-player formats. However, CFR's computational complexity rises exponentially with the number of players. Furthermore, in games with three or more players, following Nash equilibrium no longer guarantees a non-losing outcome. These limitations, along with others, significantly restrict the applicability of CFR to the most popular formats: tournaments. Motivated by the recent success of Large Language Models (LLM) in chess and Diplomacy, we present SpinGPT, the first LLM tailored to Spin & Go, a popular three-player online poker format. SpinGPT is trained in two stages: (1) Supervised Fine-Tuning on 320k high-stakes expert decisions; (2) Reinforcement Learning on 270k solver-generated hands. Our results show that SpinGPT matches the solver's actions in 78% of decisions (tolerant accuracy). With a simple deep-stack heuristic, it achieves 13.4 +/- 12.9 BB/100 versus Slumbot in heads-up over 30,000 hands (95% CI). These results suggest that LLMs could be a new way to deal with multi-player imperfect-information games like poker. 

**Abstract (ZH)**: 基于假设反事实遗憾最小化算法的SpinGPT在三人Spin & Go在线扑克中的应用探索 

---
# Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach 

**Title (ZH)**: 无 effort 图像到音乐生成：一种可解释的 RAG 基础的多模态模型方法 

**Authors**: Zijian Zhao, Dian Jin, Zijing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22378)  

**Abstract**: Recently, Image-to-Music (I2M) generation has garnered significant attention, with potential applications in fields such as gaming, advertising, and multi-modal art creation. However, due to the ambiguous and subjective nature of I2M tasks, most end-to-end methods lack interpretability, leaving users puzzled about the generation results. Even methods based on emotion mapping face controversy, as emotion represents only a singular aspect of art. Additionally, most learning-based methods require substantial computational resources and large datasets for training, hindering accessibility for common users. To address these challenges, we propose the first Vision Language Model (VLM)-based I2M framework that offers high interpretability and low computational cost. Specifically, we utilize ABC notation to bridge the text and music modalities, enabling the VLM to generate music using natural language. We then apply multi-modal Retrieval-Augmented Generation (RAG) and self-refinement techniques to allow the VLM to produce high-quality music without external training. Furthermore, we leverage the generated motivations in text and the attention maps from the VLM to provide explanations for the generated results in both text and image modalities. To validate our method, we conduct both human studies and machine evaluations, where our method outperforms others in terms of music quality and music-image consistency, indicating promising results. Our code is available at this https URL . 

**Abstract (ZH)**: 基于视觉语言模型的图像到音乐生成框架 

---
# What Is The Political Content in LLMs' Pre- and Post-Training Data? 

**Title (ZH)**: LLMs的预训练和后训练数据中包含哪些政治内容？ 

**Authors**: Tanise Ceron, Dmitry Nikolaev, Dominik Stammbach, Debora Nozza  

**Link**: [PDF](https://arxiv.org/pdf/2509.22367)  

**Abstract**: Large language models (LLMs) are known to generate politically biased text, yet how such biases arise remains unclear. A crucial step toward answering this question is the analysis of training data, whose political content remains largely underexplored in current LLM research. To address this gap, we present in this paper an analysis of the pre- and post-training corpora of OLMO2, the largest fully open-source model released together with its complete dataset. From these corpora, we draw large random samples, automatically annotate documents for political orientation, and analyze their source domains and content. We then assess how political content in the training data correlates with models' stance on specific policy issues. Our analysis shows that left-leaning documents predominate across datasets, with pre-training corpora containing significantly more politically engaged content than post-training data. We also find that left- and right-leaning documents frame similar topics through distinct values and sources of legitimacy. Finally, the predominant stance in the training data strongly correlates with models' political biases when evaluated on policy issues. These findings underscore the need to integrate political content analysis into future data curation pipelines as well as in-depth documentation of filtering strategies for transparency. 

**Abstract (ZH)**: 大型语言模型（LLMs）已知会产生政治偏见的文本，但这些偏见是如何产生的仍不清楚。回答这一问题的关键一步是对训练数据进行分析，而当前LLM研究中训练数据的政治内容尚未充分探索。为填补这一空白，我们在本文中对与完整数据集一起发布的最大开源模型OLMO2的训练前和训练后语料库进行了分析。从这些语料库中，我们抽取了大规模随机样本，自动标注文档的政治倾向，并分析其来源领域和内容。然后，我们评估了训练数据中的政治内容与模型在特定政策问题上的立场之间的关联性。我们的分析表明，左倾文档在整个数据集中占主导地位，训练前语料库包含显著更多的政治参与内容，而训练后数据则不然。我们还发现，左倾和右倾文档通过不同的价值观和合法性来源来论述相似的话题。最终，训练数据中的主要立场在针对政策问题进行评估时与模型的政治偏见高度相关。这些发现强调了将政治内容分析整合到未来的数据编辑管道以及对过滤策略进行深入文档记录以增强透明度的重要性。 

---
# CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models 

**Title (ZH)**: CHRONOBERG: 捕捉基础模型中的语言演化和时间意识 

**Authors**: Niharika Hegde, Subarnaduti Paul, Lars Joel-Frey, Manuel Brack, Kristian Kersting, Martin Mundt, Patrick Schramowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.22360)  

**Abstract**: Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on CHRONOBERG struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning CHRONOBERG as a scalable resource for the study of linguistic change and temporal generalization. Disclaimer: This paper includes language and display of samples that could be offensive to readers. Open Access: Chronoberg is available publicly on HuggingFace at ( this https URL). Code is available at (this https URL). 

**Abstract (ZH)**: 大型语言模型通过利用社交媒体和从网络上爬取的各类数据，在大规模操作方面表现出色。尽管现有的语料库种类繁多，但它们常常缺乏长期的时间结构，这可能限制了大型语言模型在语义和规范性语言演变方面的上下文理解和历时变化捕捉能力。为支持这类分析和训练，我们引入了CHRONOBERG语料库，这是一个横跨250年英文书籍文本的时间结构化语料库，来源于Project Gutenberg并配备了多种时间标注。通过书籍的编辑性质，我们可以通过时间敏感的唤醒-兴奋-支配（VAD）分析来量化词汇语义变化，并构建历史校准的情感词汇集以支持时间 grounding 的解释。利用这些词汇集，我们展示了现代基于大型语言模型的工具需要更好地定位它们对歧视性语言的检测以及对不同时间时期的情感语境化。事实上，我们表明逐步训练于CHRONOBERG的语言模型难以编码语义历时变化，强调了需要时间感知的训练和评估管道，并将CHRONOBERG定位为研究语言变化和时间泛化的可扩展资源。免责声明：本文包含可能对读者具有冒犯性的语言和示例展示。开放获取：CHRONOBERG可在HuggingFace（此链接）上公开访问。代码可在（此链接）获取。 

---
# Forecasting the Future with Yesterday's Climate: Temperature Bias in AI Weather and Climate Models 

**Title (ZH)**: 用昨日气候预测未来：AI天气与气候模型中的温度偏差 

**Authors**: Jacob B. Landsberg, Elizabeth A. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2509.22359)  

**Abstract**: AI-based climate and weather models have rapidly gained popularity, providing faster forecasts with skill that can match or even surpass that of traditional dynamical models. Despite this success, these models face a key challenge: predicting future climates while being trained only with historical data. In this study, we investigate this issue by analyzing boreal winter land temperature biases in AI weather and climate models. We examine two weather models, FourCastNet V2 Small (FourCastNet) and Pangu Weather (Pangu), evaluating their predictions for 2020-2025 and Ai2 Climate Emulator version 2 (ACE2) for 1996-2010. These time periods lie outside of the respective models' training sets and are significantly more recent than the bulk of their training data, allowing us to assess how well the models generalize to new, i.e. more modern, conditions. We find that all three models produce cold-biased mean temperatures, resembling climates from 15-20 years earlier than the period they are predicting. In some regions, like the Eastern U.S., the predictions resemble climates from as much as 20-30 years earlier. Further analysis shows that FourCastNet's and Pangu's cold bias is strongest in the hottest predicted temperatures, indicating limited training exposure to modern extreme heat events. In contrast, ACE2's bias is more evenly distributed but largest in regions, seasons, and parts of the temperature distribution where climate change has been most pronounced. These findings underscore the challenge of training AI models exclusively on historical data and highlight the need to account for such biases when applying them to future climate prediction. 

**Abstract (ZH)**: 基于AI的气候和天气模型在快速获得 popularity 的同时，提供了与传统动力模型技能相当甚至超越的更快预报，但这些模型在仅使用历史数据进行训练的情况下预测未来气候面临关键挑战。本研究通过分析北极冬季陆地温度偏差，探讨了这一问题。我们评估了FourCastNet V2 Small (FourCastNet) 和 Pangu Weather (Pangu) 这两个天气模型在2020-2025年的预测，以及Ai2 Climate Emulator版本2 (ACE2) 在1996-2010年的预测，这些时间段超出了模型各自的训练集并且比大部分训练数据更为近期，从而评估模型在新情况下的泛化能力。研究发现，所有三个模型都生成了冷偏差的平均温度，类似于预测期间15-20年前的气候。在一些区域，如美国东部，预测结果更早地追溯至20-30年前。进一步分析表明，FourCastNet 和 Pangu 的冷偏差在预测的最高温度中最强烈，表明对现代极端高温事件的训练暴露有限。相比之下，ACE2 的偏差较为均匀，但最大的偏差出现在受气候变化影响最显著的区域、季节和温度分布部分中。这些发现突显了仅使用历史数据训练AI模型的挑战，并强调了在将其应用于未来气候预测时考虑偏差的重要性。 

---
# Stochastic activations 

**Title (ZH)**: 随机激活函数 

**Authors**: Maria Lomeli, Matthijs Douze, Gergely Szilvasy, Loic Cabannes, Jade Copet, Sainbayar Sukhbaatar, Jason Weston, Gabriel Synnaeve, Pierre-Emmanuel Mazaré, Hervé Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22358)  

**Abstract**: We introduce stochastic activations. This novel strategy randomly selects between several non-linear functions in the feed-forward layer of a large language model. In particular, we choose between SILU or RELU depending on a Bernoulli draw. This strategy circumvents the optimization problem associated with RELU, namely, the constant shape for negative inputs that prevents the gradient flow. We leverage this strategy in two ways:
(1) We use stochastic activations during pre-training and fine-tune the model with RELU, which is used at inference time to provide sparse latent vectors. This reduces the inference FLOPs and translates into a significant speedup in the CPU. Interestingly, this leads to much better results than training from scratch with the RELU activation function.
(2) We evaluate stochastic activations for generation. This strategy performs reasonably well: it is only slightly inferior to the best deterministic non-linearity, namely SILU combined with temperature scaling. This offers an alternative to existing strategies by providing a controlled way to increase the diversity of the generated text. 

**Abstract (ZH)**: 我们引入了随机激活函数。这一新颖策略在大型语言模型的前向层中随机选择几种非线性函数。具体而言，我们在Bernoulli抽样之间选择SILU或RELU。这种方法绕过了RELU相关的优化问题，即负输入下的恒定形状，这会阻止梯度流通。我们以两种方式利用这一策略：
(1) 在预训练阶段使用随机激活函数，并在微调时使用RELU，后者在推理时用于提供稀疏潜向量。这减少了推理FLOPs，并在CPU上实现了显著的速度提升。有趣的是，这比从头开始使用RELU激活函数进行训练的方案效果更好。
(2) 评估随机激活函数在生成任务中的应用。该策略表现合理：它仅略微逊色于最佳确定性非线性，即SILU与温度缩放的结合。这为现有策略提供了一个增加生成文本多样性的方式。 

---
# Context and Diversity Matter: The Emergence of In-Context Learning in World Models 

**Title (ZH)**: 情境和多样性很重要：世界模型中基于上下文的学习的出现 

**Authors**: Fan Wang, Zhiyuan Chen, Yuxuan Zhong, Sunjian Zheng, Pengtao Shao, Bo Yu, Shaoshan Liu, Jianan Wang, Ning Ding, Yang Cao, Yu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22353)  

**Abstract**: The capability of predicting environmental dynamics underpins both biological neural systems and general embodied AI in adapting to their surroundings. Yet prevailing approaches rest on static world models that falter when confronted with novel or rare configurations. We investigate in-context environment learning (ICEL), shifting attention from zero-shot performance to the growth and asymptotic limits of the world model. Our contributions are three-fold: (1) we formalize in-context learning of a world model and identify two core mechanisms: environment recognition and environment learning; (2) we derive error upper-bounds for both mechanisms that expose how the mechanisms emerge; and (3) we empirically confirm that distinct ICL mechanisms exist in the world model, and we further investigate how data distribution and model architecture affect ICL in a manner consistent with theory. These findings demonstrate the potential of self-adapting world models and highlight the key factors behind the emergence of ICEL, most notably the necessity of long context and diverse environments. 

**Abstract (ZH)**: 在上下文中学贯环境的潜力及其关键因素 

---
# SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis 

**Title (ZH)**: SurvDiff：生存分析中生成合成数据的扩散模型 

**Authors**: Marie Brockschmidt, Maresa Schröder, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2509.22352)  

**Abstract**: Survival analysis is a cornerstone of clinical research by modeling time-to-event outcomes such as metastasis, disease relapse, or patient death. Unlike standard tabular data, survival data often come with incomplete event information due to dropout, or loss to follow-up. This poses unique challenges for synthetic data generation, where it is crucial for clinical research to faithfully reproduce both the event-time distribution and the censoring mechanism. In this paper, we propose SurvDiff, an end-to-end diffusion model specifically designed for generating synthetic data in survival analysis. SurvDiff is tailored to capture the data-generating mechanism by jointly generating mixed-type covariates, event times, and right-censoring, guided by a survival-tailored loss function. The loss encodes the time-to-event structure and directly optimizes for downstream survival tasks, which ensures that SurvDiff (i) reproduces realistic event-time distributions and (ii) preserves the censoring mechanism. Across multiple datasets, we show that \survdiff consistently outperforms state-of-the-art generative baselines in both distributional fidelity and downstream evaluation metrics across multiple medical datasets. To the best of our knowledge, SurvDiff is the first diffusion model explicitly designed for generating synthetic survival data. 

**Abstract (ZH)**: 生存分析是临床研究的基石，通过建模如转移、疾病复发或患者死亡等时间事件结果。与标准表格式数据不同，生存数据常常由于中途退出或失访等原因包含不完整的事件信息。这对合成数据生成提出了独特挑战，其中仔细再现事件时间分布和截尾机制对临床研究至关重要。本文提出SurvDiff，这是一种专门设计用于生存分析中生成合成数据的端到端扩散模型。SurvDiff通过生成混合型协变量、事件时间及右截尾，并由一种针对生存分析定制的损失函数进行指导，来捕捉数据生成机制。该损失函数编码时间事件结构，并直接优化下游生存分析任务，从而确保SurvDiff (i) 生成现实的时间事件分布，并且(ii) 保留截尾机制。在多个数据集上，我们展示了SurvDiff在合成数据分布保真度和多个医学数据集的下游评估指标上始终优于最先进的生成基线。据我们所知，SurvDiff是首个明确为生成合成生存数据设计的扩散模型。 

---
# Transformers Can Learn Connectivity in Some Graphs but Not Others 

**Title (ZH)**: Transformer可以在某些图中学习连通性，但在其他图中不能。 

**Authors**: Amit Roy, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2509.22343)  

**Abstract**: Reasoning capability is essential to ensure the factual correctness of the responses of transformer-based Large Language Models (LLMs), and robust reasoning about transitive relations is instrumental in many settings, such as causal inference. Hence, it is essential to investigate the capability of transformers in the task of inferring transitive relations (e.g., knowing A causes B and B causes C, then A causes C). The task of inferring transitive relations is equivalent to the task of connectivity in directed graphs (e.g., knowing there is a path from A to B, and there is a path from B to C, then there is a path from A to C). Past research focused on whether transformers can learn to infer transitivity from in-context examples provided in the input prompt. However, transformers' capability to infer transitive relations from training examples and how scaling affects the ability is unexplored. In this study, we seek to answer this question by generating directed graphs to train transformer models of varying sizes and evaluate their ability to infer transitive relations for various graph sizes. Our findings suggest that transformers are capable of learning connectivity on "grid-like'' directed graphs where each node can be embedded in a low-dimensional subspace, and connectivity is easily inferable from the embeddings of the nodes. We find that the dimensionality of the underlying grid graph is a strong predictor of transformers' ability to learn the connectivity task, where higher-dimensional grid graphs pose a greater challenge than low-dimensional grid graphs. In addition, we observe that increasing the model scale leads to increasingly better generalization to infer connectivity over grid graphs. However, if the graph is not a grid graph and contains many disconnected components, transformers struggle to learn the connectivity task, especially when the number of components is large. 

**Abstract (ZH)**: 基于变压器的大型语言模型的推理能力对于确保其响应的事实正确性至关重要，而关于传递关系的稳健推理在许多场景中（如因果推理）是必不可少的。因此，研究变压器在推断传递关系任务中的能力（例如，知道A导致B，B导致C，则A导致C）是必要的。推断传递关系的任务等同于有向图中的连通性任务（例如，知道存在从A到B的路径，存在从B到C的路径，则存在从A到C的路径）。以往研究集中在变压器能否从输入提示提供的上下文示例中学习推断传递性。然而，变压器从训练示例中推断传递关系的能力及其随规模扩大的变化尚未被探索。本研究通过生成有向图来训练不同规模的变压器模型，并评估其在不同图规模下推断传递关系的能力。我们的研究结果表明，变压器能够在“网格状”有向图中学习连通性，其中每个节点可以嵌入到低维子空间中，且连通性可以从节点的嵌入中轻松推断出来。我们发现，底层网格图的维度是变压器学习连通性任务能力的一个强预测因子，高维网格图比低维网格图更具挑战性。此外，我们观察到，增加模型规模能够使其在推断网格图的连通性上表现出更好的泛化能力。然而，如果图不是网格图且包含许多不连通组件，变压器在学习连通性任务时尤其困难，尤其是在组件数量较多时。 

---
# Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs 

**Title (ZH)**: 使用Fine-tuned大语言模型推进自然语言形式化到一阶逻辑 

**Authors**: Felix Vossel, Till Mossakowski, Björn Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2509.22338)  

**Abstract**: Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck. 

**Abstract (ZH)**: 自动将自然语言转换为一阶逻辑（FOL）对于知识表示和形式化方法至关重要，但仍具有挑战性。我们系统地评估了 fine-tuned 大型语言模型（LLM）在这一任务上的表现，比较了编码器-解码器架构和解码器-only 架构以及训练策略。使用 MALLS 和 Willow 数据集，我们探索了词汇扩展、谓词条件和多语言训练等技术，引入了精确匹配、逻辑等价性和谓词对齐的评估指标。我们的 fine-tuned Flan-T5-XXL 在谓词列表上的准确率达到 70%，在包含 CoT 原理性思维能力及符号系统（如 ccg2lambda）的表现上优于 GPT-4o 和 DeepSeek-R1-0528 模型。关键发现包括：（1）谓词可用性可提升 15-20% 的性能，（2）T5 模型超越了更大的解码器-only LLM，（3）模型能够在未见的逻辑论证（FOLIO 数据集）上泛化，无需特定训练。尽管结构化逻辑翻译具有鲁棒性，但谓词提取成为主要瓶颈。 

---
# Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning 

**Title (ZH)**: 光谱坍缩驱动深连续学习中的可塑性丧失 

**Authors**: Naicheng He, Kaicheng Guo, Arjun Prakash, Saket Tiwari, Ruo Yu Tao, Tyrone Serapio, Amy Greenwald, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2509.22335)  

**Abstract**: We investigate why deep neural networks suffer from \emph{loss of plasticity} in deep continual learning, failing to learn new tasks without reinitializing parameters. We show that this failure is preceded by Hessian spectral collapse at new-task initialization, where meaningful curvature directions vanish and gradient descent becomes ineffective. To characterize the necessary condition for successful training, we introduce the notion of $\tau$-trainability and show that current plasticity preserving algorithms can be unified under this framework. Targeting spectral collapse directly, we then discuss the Kronecker factored approximation of the Hessian, which motivates two regularization enhancements: maintaining high effective feature rank and applying $L2$ penalties. Experiments on continual supervised and reinforcement learning tasks confirm that combining these two regularizers effectively preserves plasticity. 

**Abstract (ZH)**: 我们研究为什么在深度连续学习中深度神经网络会遭受可塑性丧失的问题，无法在不重新初始化参数的情况下学习新任务。我们表明，这种失败发生在新任务初始化时海森矩阵谱塌缩之前，在此过程中有意义的曲率方向消失，使得梯度下降变得无效。为了刻画成功训练的必要条件，我们提出了$\tau$-可训练性的概念，并展示了当前的可塑性保留算法可以在这一框架下统一。直接针对谱塌缩，我们讨论了海森矩阵的克罗内克分解近似，这启发了两种正则化增强：保持有效的特征秩较高和应用$L2$惩罚。实验表明，结合这两种正则化可以有效地保留可塑性。 

---
# Pedestrian Attribute Recognition via Hierarchical Cross-Modality HyperGraph Learning 

**Title (ZH)**: 基于分层跨模态超图学习的行人属性识别 

**Authors**: Xiao Wang, Shujuan Wu, Xiaoxia Cheng, Changwei Bi, Jin Tang, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22331)  

**Abstract**: Current Pedestrian Attribute Recognition (PAR) algorithms typically focus on mapping visual features to semantic labels or attempt to enhance learning by fusing visual and attribute information. However, these methods fail to fully exploit attribute knowledge and contextual information for more accurate recognition. Although recent works have started to consider using attribute text as additional input to enhance the association between visual and semantic information, these methods are still in their infancy. To address the above challenges, this paper proposes the construction of a multi-modal knowledge graph, which is utilized to mine the relationships between local visual features and text, as well as the relationships between attributes and extensive visual context samples. Specifically, we propose an effective multi-modal knowledge graph construction method that fully considers the relationships among attributes and the relationships between attributes and vision tokens. To effectively model these relationships, this paper introduces a knowledge graph-guided cross-modal hypergraph learning framework to enhance the standard pedestrian attribute recognition framework. Comprehensive experiments on multiple PAR benchmark datasets have thoroughly demonstrated the effectiveness of our proposed knowledge graph for the PAR task, establishing a strong foundation for knowledge-guided pedestrian attribute recognition. The source code of this paper will be released on this https URL 

**Abstract (ZH)**: 当前的人行属性识别（PAR）算法通常侧重于将视觉特征映射到语义标签，或尝试通过融合视觉和属性信息来增强学习。然而，这些方法未能充分利用属性知识和上下文信息以实现更准确的识别。尽管最近的工作开始考虑使用属性文本作为额外输入以增强视觉和语义信息之间的关联，但这些方法仍处于初级阶段。为解决上述挑战，本文提出构建一个多模态知识图谱，用于挖掘局部视觉特征与文本之间的关系，以及属性与广泛视觉上下文样本之间的关系。具体地，本文提出了一种全面考虑属性之间及其与视觉标记之间关系的有效多模态知识图谱构建方法。为有效建模这些关系，本文引入了一种基于知识图谱的跨模态超图学习框架，以增强标准的人行属性识别框架。在多个PAR基准数据集上的全面实验充分证明了我们提出的知识图谱在PAR任务中的有效性，为知识导向的人行属性识别奠定了坚实基础。本文的源代码将发布在https://this-url。 

---
# Progressive Weight Loading: Accelerating Initial Inference and Gradually Boosting Performance on Resource-Constrained Environments 

**Title (ZH)**: 逐级权重加载：加速初始推理并在资源受限环境中逐步提升性能 

**Authors**: Hyunwoo Kim, Junha Lee, Mincheol Choi, Jeonghwan Lee, Jaeshin Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.22319)  

**Abstract**: Deep learning models have become increasingly large and complex, resulting in higher memory consumption and computational demands. Consequently, model loading times and initial inference latency have increased, posing significant challenges in mobile and latency-sensitive environments where frequent model loading and unloading are required, which directly impacts user experience. While Knowledge Distillation (KD) offers a solution by compressing large teacher models into smaller student ones, it often comes at the cost of reduced performance. To address this trade-off, we propose Progressive Weight Loading (PWL), a novel technique that enables fast initial inference by first deploying a lightweight student model, then incrementally replacing its layers with those of a pre-trained teacher model. To support seamless layer substitution, we introduce a training method that not only aligns intermediate feature representations between student and teacher layers, but also improves the overall output performance of the student model. Our experiments on VGG, ResNet, and ViT architectures demonstrate that models trained with PWL maintain competitive distillation performance and gradually improve accuracy as teacher layers are loaded-matching the final accuracy of the full teacher model without compromising initial inference speed. This makes PWL particularly suited for dynamic, resource-constrained deployments where both responsiveness and performance are critical. 

**Abstract (ZH)**: 渐进权重加载：一种支持快速初始推理的知识蒸馏新方法 

---
# Adaptive Policy Backbone via Shared Network 

**Title (ZH)**: 自适应策略骨干网通过共享网络 

**Authors**: Bumgeun Park, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22310)  

**Abstract**: Reinforcement learning (RL) has achieved impressive results across domains, yet learning an optimal policy typically requires extensive interaction data, limiting practical deployment. A common remedy is to leverage priors, such as pre-collected datasets or reference policies, but their utility degrades under task mismatch between training and deployment. While prior work has sought to address this mismatch, it has largely been restricted to in-distribution settings. To address this challenge, we propose Adaptive Policy Backbone (APB), a meta-transfer RL method that inserts lightweight linear layers before and after a shared backbone, thereby enabling parameter-efficient fine-tuning (PEFT) while preserving prior knowledge during adaptation. Our results show that APB improves sample efficiency over standard RL and adapts to out-of-distribution (OOD) tasks where existing meta-RL baselines typically fail. 

**Abstract (ZH)**: 强化学习（RL）已在多个领域取得了显著成果，但学习最优策略通常需要大量的交互数据，限制了其实用部署。一种常见的解决方法是利用先验知识，如预先收集的数据集或参考策略，但这些方法在训练和部署之间的任务不匹配情况下效用会下降。虽然已有工作试图解决这一问题，但主要集中在同分布设置中。为应对这一挑战，我们提出了自适应策略主干（APB），这是一种元迁移RL方法，在共享主干前后插入轻量级线性层，从而实现参数高效微调（PEFT）并在适应过程中保留先验知识。我们的结果显示，APB在标准RL方法上提高了样本效率，并能够适应现有元RL基线通常会失败的跨分布（OOD）任务。 

---
# HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models 

**Title (ZH)**: HiGS: 历史指导采样以提高扩散模型的即插即用增强功能 

**Authors**: Seyedmorteza Sadat, Farnood Salehi, Romann M. Weber  

**Link**: [PDF](https://arxiv.org/pdf/2509.22300)  

**Abstract**: While diffusion models have made remarkable progress in image generation, their outputs can still appear unrealistic and lack fine details, especially when using fewer number of neural function evaluations (NFEs) or lower guidance scales. To address this issue, we propose a novel momentum-based sampling technique, termed history-guided sampling (HiGS), which enhances quality and efficiency of diffusion sampling by integrating recent model predictions into each inference step. Specifically, HiGS leverages the difference between the current prediction and a weighted average of past predictions to steer the sampling process toward more realistic outputs with better details and structure. Our approach introduces practically no additional computation and integrates seamlessly into existing diffusion frameworks, requiring neither extra training nor fine-tuning. Extensive experiments show that HiGS consistently improves image quality across diverse models and architectures and under varying sampling budgets and guidance scales. Moreover, using a pretrained SiT model, HiGS achieves a new state-of-the-art FID of 1.61 for unguided ImageNet generation at 256$\times$256 with only 30 sampling steps (instead of the standard 250). We thus present HiGS as a plug-and-play enhancement to standard diffusion sampling that enables faster generation with higher fidelity. 

**Abstract (ZH)**: 基于历史引导的采样技术（HiGS）：一种提升扩散模型图像生成质量与效率的方法 

---
# HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Space 

**Title (ZH)**: HEAPr：基于Hessian矩阵的输出空间高效原子专家剪枝 

**Authors**: Ke Li, Zheng Yang, Zhongbin Zhou, Feng Xue, Zhonglin Jiang, Wenxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22299)  

**Abstract**: Mixture-of-Experts (MoE) architectures in large language models (LLMs) deliver exceptional performance and reduced inference costs compared to dense LLMs. However, their large parameter counts result in prohibitive memory requirements, limiting practical deployment. While existing pruning methods primarily focus on expert-level pruning, this coarse granularity often leads to substantial accuracy degradation. In this work, we introduce HEAPr, a novel pruning algorithm that decomposes experts into smaller, indivisible atomic experts, enabling more precise and flexible atomic expert pruning. To measure the importance of each atomic expert, we leverage second-order information based on principles similar to Optimal Brain Surgeon (OBS) theory. To address the computational and storage challenges posed by second-order information, HEAPr exploits the inherent properties of atomic experts to transform the second-order information from expert parameters into that of atomic expert parameters, and further simplifies it to the second-order information of atomic expert outputs. This approach reduces the space complexity from $O(d^4)$, where d is the model's dimensionality, to $O(d^2)$. HEAPr requires only two forward passes and one backward pass on a small calibration set to compute the importance of atomic experts. Extensive experiments on MoE models, including DeepSeek MoE and Qwen MoE family, demonstrate that HEAPr outperforms existing expert-level pruning methods across a wide range of compression ratios and benchmarks. Specifically, HEAPr achieves nearly lossless compression at compression ratios of 20% ~ 25% in most models, while also reducing FLOPs nearly by 20%. The code can be found at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: MoE架构在大规模语言模型中的混合专家在保持出色性能和降低推理成本的同时，由于参数量大而导致内存需求 prohibitive，限制了其实际部署。虽然现有的剪枝方法主要集中在专家级剪枝上，但这种粗粒度往往会导致显著的准确性下降。在本工作中，我们引入了HEAPr，一种新颖的剪枝算法，它将专家分解为更小的不可分割的基本专家，从而实现更精确和灵活的基本专家剪枝。为了衡量每个基本专家的重要性，我们利用类似于Optimal Brain Surgeon (OBS)理论的第二阶信息原理。为了解决第二阶信息带来的计算和存储挑战，HEAPr 利用基本专家的固有属性，将专家参数的第二阶信息转换为基本专家参数的第二阶信息，并进一步简化为基本专家输出的第二阶信息。这种方法将空间复杂性从 \(O(d^4)\) 降低到 \(O(d^2)\)。HEAPr 仅需在小型校准集上进行两次正向传递和一次反向传递即可计算基本专家的重要性。在包括DeepSeek MoE和Qwen MoE家族在内的MoE模型上进行的广泛实验表明，HEAPr 在多种压缩比和基准测试中优于现有专家级剪枝方法。具体而言，在大多数模型中，HEAPr 可以在压缩比为20% ~ 25%时实现几乎无损压缩，同时将近乎线性地减少FLOPs 20%。代码可在 <this https URL> 找到。 

---
# Jailbreaking on Text-to-Video Models via Scene Splitting Strategy 

**Title (ZH)**: 通过场景拆分策略攻破文本到视频模型 

**Authors**: Wonjun Lee, Haon Park, Doehyeon Lee, Bumsub Ham, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.22292)  

**Abstract**: Along with the rapid advancement of numerous Text-to-Video (T2V) models, growing concerns have emerged regarding their safety risks. While recent studies have explored vulnerabilities in models like LLMs, VLMs, and Text-to-Image (T2I) models through jailbreak attacks, T2V models remain largely unexplored, leaving a significant safety gap. To address this gap, we introduce SceneSplit, a novel black-box jailbreak method that works by fragmenting a harmful narrative into multiple scenes, each individually benign. This approach manipulates the generative output space, the abstract set of all potential video outputs for a given prompt, using the combination of scenes as a powerful constraint to guide the final outcome. While each scene individually corresponds to a wide and safe space where most outcomes are benign, their sequential combination collectively restricts this space, narrowing it to an unsafe region and significantly increasing the likelihood of generating a harmful video. This core mechanism is further enhanced through iterative scene manipulation, which bypasses the safety filter within this constrained unsafe region. Additionally, a strategy library that reuses successful attack patterns further improves the attack's overall effectiveness and robustness. To validate our method, we evaluate SceneSplit across 11 safety categories on T2V models. Our results show that it achieves a high average Attack Success Rate (ASR) of 77.2% on Luma Ray2, 84.1% on Hailuo, and 78.2% on Veo2, significantly outperforming the existing baseline. Through this work, we demonstrate that current T2V safety mechanisms are vulnerable to attacks that exploit narrative structure, providing new insights for understanding and improving the safety of T2V models. 

**Abstract (ZH)**: 随着众多文本到视频（T2V）模型的迅速发展，其安全风险引起了越来越多的关注。虽然近期的研究通过监禁攻击探索了如LLMs、VLMs和文本到图像（T2I）模型等模型的漏洞，但T2V模型仍 largely unexplored，留下了一个重要的安全缺口。为解决这一缺口，我们提出了SceneSplit，这是一种新颖的黑盒监禁攻击方法，通过将有害叙述分割成多个场景，每个场景单独来看都是 benign 的。这种方法通过场景的组合对生成输出空间进行操控，即给定提示下所有潜在视频输出的抽象集合，利用这种场景组合作为强大的约束来引导最终结果。虽然每个场景单独来说对应一个广泛且安全的空间，其中大多数结果是 benign 的，但它们的顺序组合却将这个空间集中到了一个不安全的区域，显著增加了生成有害视频的可能性。通过迭代场景操控，此核心机制进一步增强，从而绕过了在这个受限制的不安全区域内内置的安全过滤器。此外，一个重用成功的攻击模式的策略库进一步提高了攻击的整体效果和鲁棒性。为了验证我们的方法，我们在T2V模型上按11个安全类别评估了SceneSplit。结果显示，它在Luma Ray2上的平均攻击成功率（ASR）为77.2%，在Hailuo上为84.1%，在Veo2上为78.2%，显著优于现有基线。通过这项工作，我们证明了当前的T2V安全机制易受利用叙述结构的攻击的利用，为理解并改进T2V模型的安全性提供了新的见解。 

---
# Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection? 

**Title (ZH)**: 公平性与可解释性之间的桥梁：基于输入的解释能否促进仇恨言论检测的公平性？ 

**Authors**: Yifan Wang, Mayank Jobanputra, Ji-Ung Lee, Soyoung Oh, Isabel Valera, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2509.22291)  

**Abstract**: Natural language processing (NLP) models often replicate or amplify social bias from training data, raising concerns about fairness. At the same time, their black-box nature makes it difficult for users to recognize biased predictions and for developers to effectively mitigate them. While some studies suggest that input-based explanations can help detect and mitigate bias, others question their reliability in ensuring fairness. Existing research on explainability in fair NLP has been predominantly qualitative, with limited large-scale quantitative analysis. In this work, we conduct the first systematic study of the relationship between explainability and fairness in hate speech detection, focusing on both encoder- and decoder-only models. We examine three key dimensions: (1) identifying biased predictions, (2) selecting fair models, and (3) mitigating bias during model training. Our findings show that input-based explanations can effectively detect biased predictions and serve as useful supervision for reducing bias during training, but they are unreliable for selecting fair models among candidates. 

**Abstract (ZH)**: 自然语言处理（NLP）模型常常复制或放大训练数据中的社会偏见，引发了公平性问题的关注。同时，其黑盒性质使得用户难以识别有偏见的预测，并且开发人员难以有效地缓解这些问题。虽然有一些研究表明基于输入的解释可以帮助检测和缓解偏见，但其他研究对其在确保公平性方面的可靠性提出了质疑。现有的关于公平NLP中的可解释性研究主要以定性为主，缺乏大规模的定量分析。在这项工作中，我们首次系统地研究了可解释性与仇恨言论检测中的公平性的关系，重点关注编码器和解码器模型。我们分析了三个关键维度：（1）识别有偏见的预测，（2）选择公平模型，（3）在模型训练中缓解偏见。我们的研究发现，基于输入的解释可以有效地检测有偏见的预测，并在训练过程中作为减少偏见的有效监督，但它们不能可靠地用于在候选模型中选择公平模型。 

---
# Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities 

**Title (ZH)**: 利用大型语言模型辅助具备语言脆弱性的学龄前儿童学习形态结构 

**Authors**: Stina Sundstedt, Mattias Wingren, Susanne Hägglund, Daniel Ventus  

**Link**: [PDF](https://arxiv.org/pdf/2509.22287)  

**Abstract**: Preschool children with language vulnerabilities -- such as developmental language disorders or immigration related language challenges -- often require support to strengthen their expressive language skills. Based on the principle of implicit learning, speech-language therapists (SLTs) typically embed target morphological structures (e.g., third person -s) into everyday interactions or game-based learning activities. Educators are recommended by SLTs to do the same. This approach demands precise linguistic knowledge and real-time production of various morphological forms (e.g., "Daddy wears these when he drives to work"). The task becomes even more demanding when educators or parent also must keep children engaged and manage turn-taking in a game-based activity. In the TalBot project our multiprofessional team have developed an application in which the Furhat conversational robot plays the word retrieval game "Alias" with children to improve language skills. Our application currently employs a large language model (LLM) to manage gameplay, dialogue, affective responses, and turn-taking. Our next step is to further leverage the capacity of LLMs so the robot can generate and deliver specific morphological targets during the game. We hypothesize that a robot could outperform humans at this task. Novel aspects of this approach are that the robot could ultimately serve as a model and tutor for both children and professionals and that using LLM capabilities in this context would support basic communication needs for children with language vulnerabilities. Our long-term goal is to create a robust LLM-based Robot-Assisted Language Learning intervention capable of teaching a variety of morphological structures across different languages. 

**Abstract (ZH)**: 语言脆弱性学前儿童的支持：基于隐性学习原理的机器人辅助语言训练方法 

---
# A Global Analysis of Cyber Threats to the Energy Sector: "Currents of Conflict" from a Geopolitical Perspective 

**Title (ZH)**: 从地缘政治视角看能源 sector 的网络威胁全球分析：“冲突的 currents” 

**Authors**: Gustavo Sánchez, Ghada Elbez, Veit Hagenmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2509.22280)  

**Abstract**: The escalating frequency and sophistication of cyber threats increased the need for their comprehensive understanding. This paper explores the intersection of geopolitical dynamics, cyber threat intelligence analysis, and advanced detection technologies, with a focus on the energy domain. We leverage generative artificial intelligence to extract and structure information from raw cyber threat descriptions, enabling enhanced analysis. By conducting a geopolitical comparison of threat actor origins and target regions across multiple databases, we provide insights into trends within the general threat landscape. Additionally, we evaluate the effectiveness of cybersecurity tools -- with particular emphasis on learning-based techniques -- in detecting indicators of compromise for energy-targeted attacks. This analysis yields new insights, providing actionable information to researchers, policy makers, and cybersecurity professionals. 

**Abstract (ZH)**: escalating频率和复杂性增加促进了对其全面理解的需要。本文探讨了地缘政治动态、网络威胁情报分析和先进技术检测的交叉领域，重点关注能源领域。我们利用生成式人工智能从原始网络威胁描述中提取和组织信息，以增强分析能力。通过跨多个数据库对威胁行为者来源和目标区域进行地缘政治对比分析，我们提供了总体威胁格局中的趋势见解。此外，我们评估了网络安全工具的有效性——特别是基于学习的技术——在检测针对能源目标的攻击的指示符方面的效果。这一分析提供了新的见解，为研究人员、政策制定者和网络安全专业人员提供了可操作的信息。 

---
# Wavelet-Induced Rotary Encodings: RoPE Meets Graphs 

**Title (ZH)**: 小波诱导旋转编码：RoPE 结合图结构 

**Authors**: Isaac Reid, Arijit Sehanobish, Cedrik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E. Turner, Petar Veličković  

**Link**: [PDF](https://arxiv.org/pdf/2509.22259)  

**Abstract**: We introduce WIRE: Wavelet-Induced Rotary Encodings. WIRE extends Rotary Position Encodings (RoPE), a popular algorithm in LLMs and ViTs, to graph-structured data. We demonstrate that WIRE is more general than RoPE, recovering the latter in the special case of grid graphs. WIRE also enjoys a host of desirable theoretical properties, including equivariance under node ordering permutation, compatibility with linear attention, and (under select assumptions) asymptotic dependence on graph resistive distance. We test WIRE on a range of synthetic and real-world tasks, including identifying monochromatic subgraphs, semantic segmentation of point clouds, and more standard graph benchmarks. We find it to be effective in settings where the underlying graph structure is important. 

**Abstract (ZH)**: WIRE：小波诱导旋转编码 

---
# Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks 

**Title (ZH)**: 超越分类准确率：神经MedBench和对更深层次推理基准的需求 

**Authors**: Miao Jing, Mengting Jia, Junling Lin, Zhongxia Shen, Lijun Wang, Yuanyuan Peng, Huan Gao, Mingkun Xu, Shangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22258)  

**Abstract**: Recent advances in vision-language models (VLMs) have achieved remarkable performance on standard medical benchmarks, yet their true clinical reasoning ability remains unclear. Existing datasets predominantly emphasize classification accuracy, creating an evaluation illusion in which models appear proficient while still failing at high-stakes diagnostic reasoning. We introduce Neural-MedBench, a compact yet reasoning-intensive benchmark specifically designed to probe the limits of multimodal clinical reasoning in neurology. Neural-MedBench integrates multi-sequence MRI scans, structured electronic health records, and clinical notes, and encompasses three core task families: differential diagnosis, lesion recognition, and rationale generation. To ensure reliable evaluation, we develop a hybrid scoring pipeline that combines LLM-based graders, clinician validation, and semantic similarity metrics. Through systematic evaluation of state-of-the-art VLMs, including GPT-4o, Claude-4, and MedGemma, we observe a sharp performance drop compared to conventional datasets. Error analysis shows that reasoning failures, rather than perceptual errors, dominate model shortcomings. Our findings highlight the necessity of a Two-Axis Evaluation Framework: breadth-oriented large datasets for statistical generalization, and depth-oriented, compact benchmarks such as Neural-MedBench for reasoning fidelity. We release Neural-MedBench at this https URL as an open and extensible diagnostic testbed, which guides the expansion of future benchmarks and enables rigorous yet cost-effective assessment of clinically trustworthy AI. 

**Abstract (ZH)**: 近期视觉-语言模型在医学标准基准上的进展取得了显著性能，但其真实的临床推理能力仍不清楚。现有的数据集主要强调分类准确性，导致一种评估错觉，即模型表现 seeming 熟练但实际上在高风险诊断推理方面仍表现不佳。我们引入了 Neural-MedBench，这是一个紧凑但推理密集的基准，专门设计用于探究神经学多模态临床推理的极限。Neural-MedBench 结合了多序列 MRI 扫描、结构化电子健康记录和临床笔记，并涵盖了三个核心任务家族：鉴别诊断、病灶识别和推理生成。为确保可靠的评估，我们开发了一种结合 LLM 基础评分、临床医生验证和语义相似度度量的混合评分管道。通过系统评估当前最先进的视觉-语言模型，包括 GPT-4o、Claude-4 和 MedGemma，我们观察到与传统数据集相比，它们的表现出现了明显的下降。错误分析表明，推理失败而非感知错误主导了模型的不足之处。我们的研究结果强调了双轴评估框架的必要性：面向广泛的大数据集用于统计泛化，以及面向深度、紧凑的基准如 Neural-MedBench 用于推理准确性。我们在此 https://github.com/alibaba/Qwen-public 提供了 Neural-MedBench，作为一个开放和可扩展的诊断测试平台，该平台指导未来基准的扩展并使临床可信的 AI 的严格而经济有效的评估成为可能。 

---
# Secure and Efficient Access Control for Computer-Use Agents via Context Space 

**Title (ZH)**: 基于上下文空间的计算机使用代理安全高效访问控制 

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22256)  

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.36% of attacks while introducing only 6.83% performance overhead. 

**Abstract (ZH)**: 基于大语言模型（LLM）的计算机使用代理：一种结合AI和OS能力的框架，但由于LLM固有的不确定性问题，授予代理对计算机的控制权存在重大的安全风险。CSAgent：一种系统级的静态策略访问控制框架，以解决这些挑战。 

---
# Beyond Textual Context: Structural Graph Encoding with Adaptive Space Alignment to alleviate the hallucination of LLMs 

**Title (ZH)**: 超越文本上下文：基于自适应空间对齐的结构图编码以缓解大语言模型的幻觉 

**Authors**: Yifang Zhang, Pengfei Duan, Yiwen Yang, Shengwu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.22251)  

**Abstract**: Currently, the main approach for Large Language Models (LLMs) to tackle the hallucination issue is incorporating Knowledge Graphs(KGs).However, LLMs typically treat KGs as plain text, extracting only semantic information and limiting their use of the crucial structural aspects of KGs. Another challenge is the gap between the embedding spaces of KGs encoders and LLMs text embeddings, which hinders the effective integration of structured knowledge. To overcome these obstacles, we put forward the SSKG-LLM, an innovative model architecture that is designed to efficiently integrate both the Structural and Semantic information of KGs into the reasoning processes of LLMs. SSKG-LLM incorporates the Knowledge Graph Retrieval (KGR) module and the Knowledge Graph Encoding (KGE) module to preserve semantics while utilizing structure. Then, the Knowledge Graph Adaptation (KGA) module is incorporated to enable LLMs to understand KGs embeddings. We conduct extensive experiments and provide a detailed analysis to explore how incorporating the structural information of KGs can enhance the factual reasoning abilities of LLMs. Our code are available at this https URL. 

**Abstract (ZH)**: 当前，大型语言模型（LLMs）处理幻觉问题的主要方法是引入知识图谱（KGs）。然而，LLMs通常将KGs视为普通文本，仅提取语义信息并限制了对KGs关键结构方面的使用。另一个挑战是KGs编码器的嵌入空间与LLMs文本嵌入之间的差距，这阻碍了结构化知识的有效整合。为了克服这些障碍，我们提出了SSKG-LLM，这是一种创新的模型架构，旨在高效地将KGs的结构和语义信息整合到LLMs的推理过程中。SSKG-LLM结合了知识图谱检索（KGR）模块和知识图谱编码（KGE）模块，以保留语义同时利用结构。然后，引入了知识图谱适应（KGA）模块，使LLMs能够理解KGs嵌入。我们进行了广泛的实验并进行了详细分析，以探讨整合KGs结构信息如何增强LLMs的事实推理能力。我们的代码可在以下链接获取：this https URL。 

---
# Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Compliance 

**Title (ZH)**: 安全合规：通过合规视角重新思考大模型安全推理 

**Authors**: Wenbin Hu, Huihao Jing, Haochen Shi, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.22250)  

**Abstract**: The proliferation of Large Language Models (LLMs) has demonstrated remarkable capabilities, elevating the critical importance of LLM safety. However, existing safety methods rely on ad-hoc taxonomy and lack a rigorous, systematic protection, failing to ensure safety for the nuanced and complex behaviors of modern LLM systems. To address this problem, we solve LLM safety from legal compliance perspectives, named safety compliance. In this work, we posit relevant established legal frameworks as safety standards for defining and measuring safety compliance, including the EU AI Act and GDPR, which serve as core legal frameworks for AI safety and data security in Europe. To bridge the gap between LLM safety and legal compliance, we first develop a new benchmark for safety compliance by generating realistic LLM safety scenarios seeded with legal statutes. Subsequently, we align Qwen3-8B using Group Policy Optimization (GRPO) to construct a safety reasoner, Compliance Reasoner, which effectively aligns LLMs with legal standards to mitigate safety risks. Our comprehensive experiments demonstrate that the Compliance Reasoner achieves superior performance on the new benchmark, with average improvements of +10.45% for the EU AI Act and +11.85% for GDPR. 

**Abstract (ZH)**: 大型语言模型（LLMs）的普及展示了其显著能力，突显了LLMs安全性的关键重要性。然而，现有的安全方法依赖于非正式的分类法，并缺乏严格的系统性保护，未能确保现代LLMs系统复杂而微妙的行为安全。为了解决这一问题，我们从法律合规的角度解决LLMs安全性，称为安全性合规。在本文中，我们提出相关的建立法律框架作为定义和衡量安全性合规的安全标准，包括欧盟AI法案和GDPR，这些是欧洲AI安全和数据安全的核心法律框架。为了弥合LLMs安全与法律合规之间的差距，我们首先通过生成基于法律条款的实际LLMs安全场景，开发了一个新的安全性合规基准。随后，我们使用群体策略优化（GRPO）对Qwen3-8B进行对齐，构建了一个安全性推理器，合规推理器，它有效地将LLMs与法律标准对齐以减轻安全风险。全面的实验表明，合规推理器在新基准上取得了优越的表现，欧盟AI法案的平均改进率为+10.45%，GDPR的平均改进率为+11.85%。 

---
# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity 

**Title (ZH)**: ASSESS：语义和结构评估框架用于语句相似性评估 

**Authors**: Xiaoyang Liu, Tao Zhu, Zineng Dong, Yuntian Liu, Qingfeng Guo, Zhaoxuan Liu, Yu Chen, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22246)  

**Abstract**: Statement autoformalization, the automated translation of statements from natural language into formal languages, has seen significant advancements, yet the development of automated evaluation metrics remains limited. Existing metrics for formal statement similarity often fail to balance semantic and structural information. String-based approaches capture syntactic structure but ignore semantic meaning, whereas proof-based methods validate semantic equivalence but disregard structural nuances and, critically, provide no graded similarity score in the event of proof failure. To address these issues, we introduce ASSESS (A Semantic and Structural Evaluation Framework for Statement Similarity), which comprehensively integrates semantic and structural information to provide a continuous similarity score. Our framework first transforms formal statements into Operator Trees to capture their syntactic structure and then computes a similarity score using our novel TransTED (Transformation Tree Edit Distance) Similarity metric, which enhances traditional Tree Edit Distance by incorporating semantic awareness through transformations. For rigorous validation, we present EPLA (Evaluating Provability and Likeness for Autoformalization), a new benchmark of 524 expert-annotated formal statement pairs derived from miniF2F and ProofNet, with labels for both semantic provability and structural likeness. Experiments on EPLA demonstrate that TransTED Similarity outperforms existing methods, achieving state-of-the-art accuracy and the highest Kappa coefficient. The benchmark, and implementation code will be made public soon. 

**Abstract (ZH)**: 基于语义和结构的语句相似性评估框架：ASSESS 

---
# FeatBench: Evaluating Coding Agents on Feature Implementation for Vibe Coding 

**Title (ZH)**: FeatBench: 评估编码代理在特征实现方面的Vibe编码能力 

**Authors**: Haorui Chen, Chengze Li, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22237)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has given rise to a novel software development paradigm known as "vibe coding," where users interact with coding agents through high-level natural language. However, existing evaluation benchmarks for code generation inadequately assess an agent's vibe coding capabilities. Existing benchmarks are misaligned, as they either require code-level specifications or focus narrowly on issue-solving, neglecting the critical scenario of feature implementation within the vibe coding paradiam. To address this gap, we propose FeatBench, a novel benchmark for vibe coding that focuses on feature implementation. Our benchmark is distinguished by several key features: 1. Pure Natural Language Prompts. Task inputs consist solely of abstract natural language descriptions, devoid of any code or structural hints. 2. A Rigorous & Evolving Data Collection Process. FeatBench is built on a multi-level filtering pipeline to ensure quality and a fully automated pipeline to evolve the benchmark, mitigating data contamination. 3. Comprehensive Test Cases. Each task includes Fail-to-Pass (F2P) and Pass-to-Pass (P2P) tests to verify correctness and prevent regressions. 4. Diverse Application Domains. The benchmark includes repositories from diverse domains to ensure it reflects real-world scenarios. We evaluate two state-of-the-art agent frameworks with four leading LLMs on FeatBench. Our evaluation reveals that feature implementation within the vibe coding paradigm is a significant challenge, with the highest success rate of only 29.94%. Our analysis also reveals a tendency for "aggressive implementation," a strategy that paradoxically leads to both critical failures and superior software design. We release FeatBench, our automated collection pipeline, and all experimental results to facilitate further community research. 

**Abstract (ZH)**: 大型语言模型的快速进展催生了一种新型的软件开发范式——“ vibe 编码”，用户通过高层次的自然语言与编码代理交互。然而，现有的代码生成评估基准未能充分评估代理的 vibe 编码能力。现有基准存在错位，要么需要代码级的规范，要么专注于问题解决，忽视了 vibe 编码范式下的功能实现关键场景。为填补这一空白，我们提出 FeatBench，这是一种专注于功能实现的新基准。该基准的主要特点包括：1. 纯自然语言提示。任务输入仅包含抽象自然语言描述，无任何代码或结构提示；2. 严格的且不断演变的数据收集过程。FeatBench 基于多级过滤流水线以确保数据质量，并采用自动化流程不断优化基准，避免数据污染；3. 全面的测试案例。每个任务包括失败到通过（F2P）和通过到通过（P2P）测试，以验证正确性并防止退步；4. 多样的应用领域。基准包括来自不同领域的仓库，确保其反映真实世界场景。我们使用两种最新的代理框架和四个领先的大规模语言模型在 FeatBench 上进行评估。评估结果表明，在 vibe 编码范式下的功能实现是一个重大挑战，最高成功率仅为 29.94%。此外，我们的分析还揭示了一种“激进实现”的倾向，这一策略导致了关键失败同时也产生了卓越的软件设计。我们发布了 FeatBench、自动数据收集流水线以及所有实验结果，以促进进一步的社区研究。 

---
# Fairness-Aware Reinforcement Learning (FAReL): A Framework for Transparent and Balanced Sequential Decision-Making 

**Title (ZH)**: 面向公平的强化学习（FAReL）：透明且平衡的序列决策框架 

**Authors**: Alexandra Cimpean, Nicole Orzan, Catholijn Jonker, Pieter Libin, Ann Nowé  

**Link**: [PDF](https://arxiv.org/pdf/2509.22232)  

**Abstract**: Equity in real-world sequential decision problems can be enforced using fairness-aware methods. Therefore, we require algorithms that can make suitable and transparent trade-offs between performance and the desired fairness notions. As the desired performance-fairness trade-off is hard to specify a priori, we propose a framework where multiple trade-offs can be explored. Insights provided by the reinforcement learning algorithm regarding the obtainable performance-fairness trade-offs can then guide stakeholders in selecting the most appropriate policy. To capture fairness, we propose an extended Markov decision process, $f$MDP, that explicitly encodes individuals and groups. Given this $f$MDP, we formalise fairness notions in the context of sequential decision problems and formulate a fairness framework that computes fairness measures over time. We evaluate our framework in two scenarios with distinct fairness requirements: job hiring, where strong teams must be composed while treating applicants equally, and fraud detection, where fraudulent transactions must be detected while ensuring the burden on customers is fairly distributed. We show that our framework learns policies that are more fair across multiple scenarios, with only minor loss in performance reward. Moreover, we observe that group and individual fairness notions do not necessarily imply one another, highlighting the benefit of our framework in settings where both fairness types are desired. Finally, we provide guidelines on how to apply this framework across different problem settings. 

**Abstract (ZH)**: 在实际序列决策问题中，可以通过公平性意识方法确保公平性。因此，我们需要能够在这两者之间做出合适且透明权衡的算法：性能与所需的公平性观念。由于期望的性能-公平性权衡在先验难以具体指定，我们提出了一种框架，可以在其中探索多种权衡。强化学习算法提供的见解可以指导相关方选择最合适的政策。为了捕捉公平性，我们提出了一种扩展的马尔可夫决策过程$f$MDP，其中明确编码了个体和群体。基于这个$f$MDP，我们在序列决策问题的背景下形式化了公平性概念，并制定了一个公平性框架，该框架计算公平性指标随时间的变化。我们在两种具有不同公平性要求的场景中评估了该框架：在招聘场景中，需要组建强大的团队同时公平对待应聘者；在欺诈检测场景中，需要检测欺诈性交易同时公平地分配客户负担。我们展示了该框架在多个场景中学习出更公平的策略，仅轻微牺牲性能奖励。此外，我们观察到群体公平性和个体公平性概念并不一定相互蕴含，突显了该框架在同时追求这两种公平性的设置中的优势。最后，我们提供了如何在不同问题设置中应用该框架的指导。 

---
# Polysemous Language Gaussian Splatting via Matching-based Mask Lifting 

**Title (ZH)**: 基于匹配驱动的掩码提升的多义语 Gaussian 表面化 

**Authors**: Jiayu Ding, Xinpeng Liu, Zhiyi Pan, Shiqiang Long, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22225)  

**Abstract**: Lifting 2D open-vocabulary understanding into 3D Gaussian Splatting (3DGS) scenes is a critical challenge. However, mainstream methods suffer from three key flaws: (i) their reliance on costly per-scene retraining prevents plug-and-play application; (ii) their restrictive monosemous design fails to represent complex, multi-concept semantics; and (iii) their vulnerability to cross-view semantic inconsistencies corrupts the final semantic representation. To overcome these limitations, we introduce MUSplat, a training-free framework that abandons feature optimization entirely. Leveraging a pre-trained 2D segmentation model, our pipeline generates and lifts multi-granularity 2D masks into 3D, where we estimate a foreground probability for each Gaussian point to form initial object groups. We then optimize the ambiguous boundaries of these initial groups using semantic entropy and geometric opacity. Subsequently, by interpreting the object's appearance across its most representative viewpoints, a Vision-Language Model (VLM) distills robust textual features that reconciles visual inconsistencies, enabling open-vocabulary querying via semantic matching. By eliminating the costly per-scene training process, MUSplat reduces scene adaptation time from hours to mere minutes. On benchmark tasks for open-vocabulary 3D object selection and semantic segmentation, MUSplat outperforms established training-based frameworks while simultaneously addressing their monosemous limitations. 

**Abstract (ZH)**: 将2D开放词汇理解提升到3D高斯点渲染(3DGS)场景中的挑战是关键性的。然而，主流方法存在三个关键缺陷：（i）其依赖于昂贵的逐场景重新训练，妨碍了即插即用的应用；（ii）其限制性的单义设计无法表示复杂的多概念语义；（iii）其对跨视角语义不一致的脆弱性破坏了最终的语义表示。为克服这些限制，我们引入了MUSplat，一个无需训练的框架，完全放弃了特征优化。利用预训练的2D分割模型，我们的流水线生成并提升多粒度的2D掩码到3D，为每个高斯点估计前景概率以形成初始对象组。然后，我们使用语义熵和几何透明度优化这些初始组的含糊边界。通过解释对象在其最具代表性的视角下的外观，一个视觉-语言模型（VLM）提炼出稳健的文本特征，以解决视觉不一致性，通过语义匹配实现开放词汇查询。通过消除逐场景的训练过程，MUSplat将场景适应时间从数小时缩短到仅仅几分钟。在开放词汇3D对象选择和语义分割基准任务中，MUSplat超越了现有的基于训练的框架，同时解决了它们的单义限制。 

---
# Thinking in Many Modes: How Composite Reasoning Elevates Large Language Model Performance with Limited Data 

**Title (ZH)**: 多模式思考：综合推理如何在有限数据下提升大型语言模型性能 

**Authors**: Zishan Ahmad, Saisubramaniam Gopalakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22224)  

**Abstract**: Large Language Models (LLMs), despite their remarkable capabilities, rely on singular, pre-dominant reasoning paradigms, hindering their performance on intricate problems that demand diverse cognitive strategies. To address this, we introduce Composite Reasoning (CR), a novel reasoning approach empowering LLMs to dynamically explore and combine multiple reasoning styles like deductive, inductive, and abductive for more nuanced problem-solving. Evaluated on scientific and medical question-answering benchmarks, our approach outperforms existing baselines like Chain-of-Thought (CoT) and also surpasses the accuracy of DeepSeek-R1 style reasoning (SR) capabilities, while demonstrating superior sample efficiency and adequate token usage. Notably, CR adaptively emphasizes domain-appropriate reasoning styles. It prioritizes abductive and deductive reasoning for medical question answering, but shifts to causal, deductive, and inductive methods for scientific reasoning. Our findings highlight that by cultivating internal reasoning style diversity, LLMs acquire more robust, adaptive, and efficient problem-solving abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）尽管具备显著的能力，但仍依赖单一的推理范式，这限制了它们在需要多种认知策略的复杂问题上的表现。为解决这一问题，我们引入了一种新型的推理方法——复合推理（CR），该方法使LLMs能够动态探索和结合多种推理风格（如演绎、归纳和 abduction推理），以实现更细致的问题解决。在科学和医学领域的问答基准测试中，该方法不仅超越了现有的基线方法（如思维链方法Chain-of-Thought, CoT），还超过了DeepSeek-R1样式推理（SR）的准确性和效率，同时展示了更好的样本效率和恰当的token使用。值得注意的是，CR能够适应性地强调不同的推理风格。对于医学问答，它倾向于使用演绎和归纳推理，但在科学推理中，则侧重因果、演绎和归纳方法。我们的研究结果表明，通过培养内部推理风格的多样性，LLMs能够获得更为稳健、适应性强和高效的解决问题能力。 

---
# Rigidity-Aware 3D Gaussian Deformation from a Single Image 

**Title (ZH)**: 基于刚性意识的单图三维高斯变形 

**Authors**: Jinhyeok Kim, Jaehun Bang, Seunghyun Seo, Kyungdon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22222)  

**Abstract**: Reconstructing object deformation from a single image remains a significant challenge in computer vision and graphics. Existing methods typically rely on multi-view video to recover deformation, limiting their applicability under constrained scenarios. To address this, we propose DeformSplat, a novel framework that effectively guides 3D Gaussian deformation from only a single image. Our method introduces two main technical contributions. First, we present Gaussian-to-Pixel Matching which bridges the domain gap between 3D Gaussian representations and 2D pixel observations. This enables robust deformation guidance from sparse visual cues. Second, we propose Rigid Part Segmentation consisting of initialization and refinement. This segmentation explicitly identifies rigid regions, crucial for maintaining geometric coherence during deformation. By combining these two techniques, our approach can reconstruct consistent deformations from a single image. Extensive experiments demonstrate that our approach significantly outperforms existing methods and naturally extends to various applications,such as frame interpolation and interactive object manipulation. 

**Abstract (ZH)**: 仅从单张图像重构物体变形仍然是计算机视觉和图形学中的一个重大挑战。现有方法通常依赖多视角视频来恢复变形，这限制了其在受限场景中的应用。为解决这一问题，我们提出了一种名为DeformSplat的新框架，该框架能够仅通过单张图像有效引导3D高斯变形。我们的方法引入了两项主要的技术贡献。首先，我们提出了高斯到像素匹配技术，以弥合3D高斯表示与2D像素观测之间的域差距，从而能够从稀疏的视觉线索中实现稳健的变形引导。其次，我们提出了刚性部分分割，包括初始化和细化两个步骤，该分割明确地识别刚性区域，这对于在变形过程中保持几何一致性至关重要。通过结合这两种技术，我们的方法可以从单张图像中重建一致的变形。广泛的实验表明，我们的方法在性能上显著优于现有方法，并且自然地扩展到各种应用中，如帧内插和交互式对象操作。 

---
# Automatic Discovery of One Parameter Subgroups of $SO(n)$ 

**Title (ZH)**: 自动发现$SO(n)$的一参数子组 

**Authors**: Pavan Karjol, Vivek V Kashyap, Rohan Kashyap, Prathosh A P  

**Link**: [PDF](https://arxiv.org/pdf/2509.22219)  

**Abstract**: We introduce a novel framework for the automatic discovery of one-parameter subgroups ($H_{\gamma}$) of $SO(3)$ and, more generally, $SO(n)$. One-parameter subgroups of $SO(n)$ are crucial in a wide range of applications, including robotics, quantum mechanics, and molecular structure analysis. Our method utilizes the standard Jordan form of skew-symmetric matrices, which define the Lie algebra of $SO(n)$, to establish a canonical form for orbits under the action of $H_{\gamma}$. This canonical form is then employed to derive a standardized representation for $H_{\gamma}$-invariant functions. By learning the appropriate parameters, the framework uncovers the underlying one-parameter subgroup $H_{\gamma}$. The effectiveness of the proposed approach is demonstrated through tasks such as double pendulum modeling, moment of inertia prediction, top quark tagging and invariant polynomial regression, where it successfully recovers meaningful subgroup structure and produces interpretable, symmetry-aware representations. 

**Abstract (ZH)**: 我们提出了一种新的框架，用于自动发现$SO(3)$的一参数子群（$H_{\gamma}$），更一般地讲，是$SO(n)$的一参数子群。$SO(n)$的一参数子群在机器人学、量子力学和分子结构分析等多种应用中至关重要。该方法利用$SO(n)$的李代数由反对称矩阵的标准Jordan形式来建立$H_{\gamma}$作用下轨道的标准形式。该标准形式随后被用来推导$H_{\gamma}$不变函数的标准表示。通过学习适当的参数，该框架揭示了潜在的一参数子群$H_{\gamma}$。通过双摆建模、转动惯量预测、顶夸克标记和不变多项式回归等任务，证明了所提出方法的有效性，该方法成功地恢复了有意义的子群结构并产生了可解释的、意识到了对称性的表示。 

---
# VizGen: Data Exploration and Visualization from Natural Language via a Multi-Agent AI Architecture 

**Title (ZH)**: VizGen: 通过多智能体AI架构从自然语言进行数据探索与可视化 

**Authors**: Sandaru Fernando, Imasha Jayarathne, Sithumini Abeysekara, Shanuja Sithamparanthan, Thushari Silva, Deshan Jayawardana  

**Link**: [PDF](https://arxiv.org/pdf/2509.22218)  

**Abstract**: Data visualization is essential for interpreting complex datasets, yet traditional tools often require technical expertise, limiting accessibility. VizGen is an AI-assisted graph generation system that empowers users to create meaningful visualizations using natural language. Leveraging advanced NLP and LLMs like Claude 3.7 Sonnet and Gemini 2.0 Flash, it translates user queries into SQL and recommends suitable graph types. Built on a multi-agent architecture, VizGen handles SQL generation, graph creation, customization, and insight extraction. Beyond visualization, it analyzes data for patterns, anomalies, and correlations, and enhances user understanding by providing explanations enriched with contextual information gathered from the internet. The system supports real-time interaction with SQL databases and allows conversational graph refinement, making data analysis intuitive and accessible. VizGen democratizes data visualization by bridging the gap between technical complexity and user-friendly design. 

**Abstract (ZH)**: VizGen：基于AI辅助的图形生成系统， democratizing data visualization by bridging technical complexity and user-friendly design。 

---
# Impact of Collective Behaviors of Autonomous Vehicles on Urban Traffic Dynamics: A Multi-Agent Reinforcement Learning Approach 

**Title (ZH)**: 自动驾驶车辆集群行为对城市交通动力学的影响：一种多代理强化学习方法 

**Authors**: Ahmet Onur Akman, Anastasia Psarou, Zoltán György Varga, Grzegorz Jamróz, Rafał Kucharski  

**Link**: [PDF](https://arxiv.org/pdf/2509.22216)  

**Abstract**: This study examines the potential impact of reinforcement learning (RL)-enabled autonomous vehicles (AV) on urban traffic flow in a mixed traffic environment. We focus on a simplified day-to-day route choice problem in a multi-agent setting. We consider a city network where human drivers travel through their chosen routes to reach their destinations in minimum travel time. Then, we convert one-third of the population into AVs, which are RL agents employing Deep Q-learning algorithm. We define a set of optimization targets, or as we call them behaviors, namely selfish, collaborative, competitive, social, altruistic, and malicious. We impose a selected behavior on AVs through their rewards. We run our simulations using our in-house developed RL framework PARCOUR. Our simulations reveal that AVs optimize their travel times by up to 5\%, with varying impacts on human drivers' travel times depending on the AV behavior. In all cases where AVs adopt a self-serving behavior, they achieve shorter travel times than human drivers. Our findings highlight the complexity differences in learning tasks of each target behavior. We demonstrate that the multi-agent RL setting is applicable for collective routing on traffic networks, though their impact on coexisting parties greatly varies with the behaviors adopted. 

**Abstract (ZH)**: 本研究探讨了增强学习（RL）赋能的自动驾驶车辆（AV）对混合交通环境中城市交通流的潜在影响。我们集中研究多智能体环境下的简化日常路线选择问题。我们考虑一个城市网络，其中人类驾驶员通过他们选择的路线在最短的旅行时间内到达目的地。然后，我们将三分之一的人口转换为AV，这些AV是使用深度Q学习算法的RL代理。我们定义了一组优化目标，或如我们所称的行为，分别是自私、协作、竞争、亲社会、利他和恶意。我们通过奖励将选定的行为施加于AV上。我们使用我们自主研发的RL框架PARCOUR进行模拟。我们的模拟结果显示，AV通过优化其旅行时间最多可达5%，而人类驾驶员的旅行时间受到AV行为的影响有所不同。在AV采取自我中心行为的所有情况下，它们的旅行时间都比人类驾驶员短。我们的研究指出，每种目标行为的学习任务复杂性存在差异。我们证明，多智能体RL设置适用于交通网络中的集体路由，但由于所采用的行为不同，其对共存各方的影响也各不相同。 

---
# Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation 

**Title (ZH)**: 基于问题驱动的分析与合成：使用大语言模型构建可解释的主题树以进行文本聚类和可控生成 

**Authors**: Tiago Fernandes Tavares  

**Link**: [PDF](https://arxiv.org/pdf/2509.22211)  

**Abstract**: Unsupervised analysis of text corpora is challenging, especially in data-scarce domains where traditional topic models struggle. While these models offer a solution, they typically describe clusters with lists of keywords that require significant manual effort to interpret and often lack semantic coherence. To address this critical interpretability gap, we introduce Recursive Thematic Partitioning (RTP), a novel framework that leverages Large Language Models (LLMs) to interactively build a binary tree. Each node in the tree is a natural language question that semantically partitions the data, resulting in a fully interpretable taxonomy where the logic of each cluster is explicit. Our experiments demonstrate that RTP's question-driven hierarchy is more interpretable than the keyword-based topics from a strong baseline like BERTopic. Furthermore, we establish the quantitative utility of these clusters by showing they serve as powerful features in downstream classification tasks, particularly when the data's underlying themes correlate with the task labels. RTP introduces a new paradigm for data exploration, shifting the focus from statistical pattern discovery to knowledge-driven thematic analysis. Furthermore, we demonstrate that the thematic paths from the RTP tree can serve as structured, controllable prompts for generative models. This transforms our analytical framework into a powerful tool for synthesis, enabling the consistent imitation of specific characteristics discovered in the source corpus. 

**Abstract (ZH)**: 无监督文本语料库分析在数据稀少领域尤为具有挑战性，传统主题模型往往表现不佳。尽管这些模型提供了解决方案，但它们通常使用关键词列表描述簇，这需要大量人工努力进行解释，并且往往缺乏语义连贯性。为了解决这一关键的解释性缺口，我们提出了递归主题分割（RTP）这一新颖框架，该框架利用大规模语言模型（LLMs）实现交互式构建二叉树。树中的每个节点是一个自然语言问题，从语义上对数据进行分割，从而得到一个完全可解释的分类体系，其中每个簇的逻辑明确。实验结果表明，RTP的问题导向层次结构比强基线模型（如BERTopic）的基于关键词的主题更具解释性。此外，我们通过证明这些簇在下游分类任务中作为强大特征的应用价值，确立了它们的定量实用价值，特别是在数据的基本主题与任务标签相关的情况下。RTP引入了一种新的数据探索范式，将关注点从统计模式发现转向基于知识的主题分析。此外，我们展示了RTP树中的主题路径可以作为结构化的、可控的提示用于生成模型，这将我们的分析框架转变为一种强大的合成工具，能够一致地模仿源语料库中发现的特定特征。 

---
# Reversible GNS for Dissipative Fluids with Consistent Bidirectional Dynamics 

**Title (ZH)**: 可逆GNS方法用于耗散流体且具有一致的双向动力学 

**Authors**: Mu Huang, Linning Xu, Mingyue Dai, Yidi Shao, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22207)  

**Abstract**: Simulating physically plausible trajectories toward user-defined goals is a fundamental yet challenging task in fluid dynamics. While particle-based simulators can efficiently reproduce forward dynamics, inverse inference remains difficult, especially in dissipative systems where dynamics are irreversible and optimization-based solvers are slow, unstable, and often fail to converge. In this work, we introduce the Reversible Graph Network Simulator (R-GNS), a unified framework that enforces bidirectional consistency within a single graph architecture. Unlike prior neural simulators that approximate inverse dynamics by fitting backward data, R-GNS does not attempt to reverse the underlying physics. Instead, we propose a mathematically invertible design based on residual reversible message passing with shared parameters, coupling forward dynamics with inverse inference to deliver accurate predictions and efficient recovery of plausible initial states. Experiments on three dissipative benchmarks (Water-3D, WaterRamps, and WaterDrop) show that R-GNS achieves higher accuracy and consistency with only one quarter of the parameters, and performs inverse inference more than 100 times faster than optimization-based baselines. For forward simulation, R-GNS matches the speed of strong GNS baselines, while in goal-conditioned tasks it eliminates iterative optimization and achieves orders-of-magnitude speedups. On goal-conditioned tasks, R-GNS further demonstrates its ability to complex target shapes (e.g., characters "L" and "N") through vivid, physically consistent trajectories. To our knowledge, this is the first reversible framework that unifies forward and inverse simulation for dissipative fluid systems. 

**Abstract (ZH)**: Reversible Graph Network Simulator for Unified Forward and Inverse Simulation of Dissipative Fluid Systems 

---
# The Outputs of Large Language Models are Meaningless 

**Title (ZH)**: 大型语言模型的输出毫无意义。 

**Authors**: Anandi Hattiangadi, Anders J. Schoubye  

**Link**: [PDF](https://arxiv.org/pdf/2509.22206)  

**Abstract**: In this paper, we offer a simple argument for the conclusion that the outputs of large language models (LLMs) are meaningless. Our argument is based on two key premises: (a) that certain kinds of intentions are needed in order for LLMs' outputs to have literal meanings, and (b) that LLMs cannot plausibly have the right kinds of intentions. We defend this argument from various types of responses, for example, the semantic externalist argument that deference can be assumed to take the place of intentions and the semantic internalist argument that meanings can be defined purely in terms of intrinsic relations between concepts, such as conceptual roles. We conclude the paper by discussing why, even if our argument is sound, the outputs of LLMs nevertheless seem meaningful and can be used to acquire true beliefs and even knowledge. 

**Abstract (ZH)**: 本文提出一个简单的论据，认为大型语言模型（LLM）的输出缺乏实际意义。我们的论据基于两个关键前提：(a) 需要某些类型的意图才能使LLM的输出具有字面意义，(b) LLMs不可能具有正确类型的意图。我们从各种类型的回应中辩护这一论据，例如语义外部主义的论据认为可以假设依赖代替意图，以及语义内部主义的论据认为意义可以纯粹通过概念之间的内在关系，如概念角色，来定义。我们在论文结尾讨论即使我们的论据成立，LLM的输出似乎仍然有意义，并且可以用来获得真信念乃至知识。 

---
# MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training 

**Title (ZH)**: MimicDreamer: 将人类和机器人演示相结合以实现可扩展的联合学习训练 

**Authors**: Haoyun Li, Ivan Zhang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Zhiqin Yang, Zhentao Zhang, Boyuan Wang, Chaojun Ni, Wenkang Qin, Xinze Chen, Yun Ye, Guan Huang, Zhenbo Song, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22199)  

**Abstract**: Vision Language Action (VLA) models derive their generalization capability from diverse training data, yet collecting embodied robot interaction data remains prohibitively expensive. In contrast, human demonstration videos are far more scalable and cost-efficient to collect, and recent studies confirm their effectiveness in training VLA models. However, a significant domain gap persists between human videos and robot-executed videos, including unstable camera viewpoints, visual discrepancies between human hands and robotic arms, and differences in motion dynamics. To bridge this gap, we propose MimicDreamer, a framework that turns fast, low-cost human demonstrations into robot-usable supervision by jointly aligning vision, viewpoint, and actions to directly support policy training. For visual alignment, we propose H2R Aligner, a video diffusion model that generates high-fidelity robot demonstration videos by transferring motion from human manipulation footage. For viewpoint stabilization, EgoStabilizer is proposed, which canonicalizes egocentric videos via homography and inpaints occlusions and distortions caused by warping. For action alignment, we map human hand trajectories to the robot frame and apply a constrained inverse kinematics solver to produce feasible, low-jitter joint commands with accurate pose tracking. Empirically, VLA models trained purely on our synthesized human-to-robot videos achieve few-shot execution on real robots. Moreover, scaling training with human data significantly boosts performance compared to models trained solely on real robot data; our approach improves the average success rate by 14.7\% across six representative manipulation tasks. 

**Abstract (ZH)**: Vision Language Action (VLA)模型的通用能力来源于多样化的训练数据，然而收集带有机器人互动的数据仍然极为昂贵。相比之下，人类演示视频的收集更为规模性强且成本效益高，近期的研究证实其在训练VLA模型方面的有效性。然而，人类视频与机器人执行视频之间仍然存在显著的领域差距，包括不稳定的摄像机视角、人类手部与机器人手臂之间的视觉差异以及运动动态的差异。为了弥合这一差距，我们提出了MimicDreamer框架，通过联合对齐视觉、视角和动作，将其快速低成本的人类演示转化为可用于机器人训练的支持策略训练的监督数据。在视觉对齐方面，我们提出了H2R对齐器，这是一种视频扩散模型，通过从人类操作片段中转移运动以生成高保真度的机器人演示视频。对于视角稳定，我们提出了EgoStabilizer，它通过透射变换规范化第一人称视角视频，并修补扭曲运动导致的遮挡和失真。对于动作对齐，我们将人类手部轨迹映射到机器人坐标系，并应用约束逆运动学求解器以产生具有准确姿态跟踪的可行、低抖动的关节命令。实验结果显示，仅使用我们合成的从人到机器人的视频训练的VLA模型能够在真实机器人上实现少量样本执行。此外，使用人类数据扩展训练显著提升了性能，与仅基于真实机器人数据训练的模型相比，我们方法在六个代表性操作任务中将平均成功率提高了14.7%。 

---
# Learning Equivariant Functions via Quadratic Forms 

**Title (ZH)**: 学习二次形式下的等变函数 

**Authors**: Pavan Karjol, Vivek V Kashyap, Rohan Kashyap, Prathosh A P  

**Link**: [PDF](https://arxiv.org/pdf/2509.22184)  

**Abstract**: In this study, we introduce a method for learning group (known or unknown) equivariant functions by learning the associated quadratic form $x^T A x$ corresponding to the group from the data. Certain groups, known as orthogonal groups, preserve a specific quadratic form, and we leverage this property to uncover the underlying symmetry group under the assumption that it is orthogonal. By utilizing the corresponding unique symmetric matrix and its inherent diagonal form, we incorporate suitable inductive biases into the neural network architecture, leading to models that are both simplified and efficient. Our approach results in an invariant model that preserves norms, while the equivariant model is represented as a product of a norm-invariant model and a scale-invariant model, where the ``product'' refers to the group action.
Moreover, we extend our framework to a more general setting where the function acts on tuples of input vectors via a diagonal (or product) group action. In this extension, the equivariant function is decomposed into an angular component extracted solely from the normalized first vector and a scale-invariant component that depends on the full Gram matrix of the tuple. This decomposition captures the inter-dependencies between multiple inputs while preserving the underlying group symmetry.
We assess the effectiveness of our framework across multiple tasks, including polynomial regression, top quark tagging, and moment of inertia matrix prediction. Comparative analysis with baseline methods demonstrates that our model consistently excels in both discovering the underlying symmetry and efficiently learning the corresponding equivariant function. 

**Abstract (ZH)**: 本研究介绍了一种通过从数据中学习与群组（已知或未知）相关的二次形式$x^T A x$来学习群组（不变或可变）函数的方法。某些群组，称为正交群组，保持特定的二次形式，我们利用这一性质，在假设群组是正交的情况下，揭示底层对称群组。通过利用相应的唯一对称矩阵及其固有的对角形式，我们将合适的归纳偏置纳入神经网络架构中，从而得到既简化又高效的方法。我们的方法产生一个保持范数不变的不变模型，而可变模型表示为一个范数不变模型和一个尺度不变模型的乘积，这里的“乘积”指的是群组作用。此外，我们将框架推广到函数通过对角（或乘积）群组作用作用于输入向量元组的更一般设置中。在这一扩展中，可变函数被分解为仅从归一化后的第一个向量中提取的角成分和依赖于元组完整格莱姆矩阵的尺度不变成分。这种分解捕捉到多个输入之间的相互依赖关系，同时保持底层的群组对称性。我们跨多项式回归、顶夸克标记和惯性矩预测等多个任务评估了框架的有效性。与基线方法的比较分析表明，我们的模型在发现底层对称性和高效学习相应的可变函数方面始终表现出色。 

---
# Efficiency Boost in Decentralized Optimization: Reimagining Neighborhood Aggregation with Minimal Overhead 

**Title (ZH)**: 去中心化优化中的效率提升：以最小开销重新构想邻域聚合 

**Authors**: Durgesh Kalwar, Mayank Baranwal, Harshad Khadilkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22174)  

**Abstract**: In today's data-sensitive landscape, distributed learning emerges as a vital tool, not only fortifying privacy measures but also streamlining computational operations. This becomes especially crucial within fully decentralized infrastructures where local processing is imperative due to the absence of centralized aggregation. Here, we introduce DYNAWEIGHT, a novel framework to information aggregation in multi-agent networks. DYNAWEIGHT offers substantial acceleration in decentralized learning with minimal additional communication and memory overhead. Unlike traditional static weight assignments, such as Metropolis weights, DYNAWEIGHT dynamically allocates weights to neighboring servers based on their relative losses on local datasets. Consequently, it favors servers possessing diverse information, particularly in scenarios of substantial data heterogeneity. Our experiments on various datasets MNIST, CIFAR10, and CIFAR100 incorporating various server counts and graph topologies, demonstrate notable enhancements in training speeds. Notably, DYNAWEIGHT functions as an aggregation scheme compatible with any underlying server-level optimization algorithm, underscoring its versatility and potential for widespread integration. 

**Abstract (ZH)**: 在当今数据敏感的环境中，分布式学习 emerge as a vital tool, not only fortifying privacy measures but also streamlining computational operations. 这在完全去中心化的基础设施中尤为重要，因为在这种情况下，由于缺乏集中聚合，本地处理变得至关重要。在此，我们介绍 DYNAWEIGHT，一种用于多代理网络信息聚合的新框架。DYNAWEIGHT 通过最少的额外通信和内存开销实现去中心化学习的显著加速。与传统的静态权重分配（如 Metropolis 权重）不同，DYNAWEIGHT 根据相邻服务器在其本地数据集上的相对损失动态分配权重。因此，在数据异质性显著的情况下，它倾向于信息多样的服务器。在对 MNIST、CIFAR10 和 CIFAR100 等多个数据集的各种服务器数量和图拓扑进行的实验中，展示了训练速度的显著提升。值得注意的是，DYNAWEIGHT 可与任何底层服务器级优化算法兼容，突显了其灵活性和广泛集成的潜力。去中心化学习中的动态权重分配框架 

---
# Teaching AI to Feel: A Collaborative, Full-Body Exploration of Emotive Communication 

**Title (ZH)**: 教AI感受：一种协作式的全身探索情感交流 

**Authors**: Esen K. Tütüncü, Lissette Lemus, Kris Pilcher, Holger Sprengel, Jordi Sabater-Mir  

**Link**: [PDF](https://arxiv.org/pdf/2509.22168)  

**Abstract**: Commonaiverse is an interactive installation exploring human emotions through full-body motion tracking and real-time AI feedback. Participants engage in three phases: Teaching, Exploration and the Cosmos Phase, collaboratively expressing and interpreting emotions with the system. The installation integrates MoveNet for precise motion tracking and a multi-recommender AI system to analyze emotional states dynamically, responding with adaptive audiovisual outputs. By shifting from top-down emotion classification to participant-driven, culturally diverse definitions, we highlight new pathways for inclusive, ethical affective computing. We discuss how this collaborative, out-of-the-box approach pushes multimedia research beyond single-user facial analysis toward a more embodied, co-created paradigm of emotional AI. Furthermore, we reflect on how this reimagined framework fosters user agency, reduces bias, and opens avenues for advanced interactive applications. 

**Abstract (ZH)**: Commonaiverse：通过全身动作追踪和实时AI反馈探索人类情绪的互动安装艺术 

---
# Lightweight error mitigation strategies for post-training N:M activation sparsity in LLMs 

**Title (ZH)**: 训练后LLM中N:M激活稀疏性的轻量级错误缓解策略 

**Authors**: Shirin Alanova, Kristina Kazistova, Ekaterina Galaeva, Alina Kostromina, Vladimir Smirnov, Redko Dmitry, Alexey Dontsov, Maxim Zhelnin, Evgeny Burnaev, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2509.22166)  

**Abstract**: The demand for efficient large language model (LLM) inference has intensified the focus on sparsification techniques. While semi-structured (N:M) pruning is well-established for weights, its application to activation pruning remains underexplored despite its potential for dynamic, input-adaptive compression and reductions in I/O overhead. This work presents a comprehensive analysis of methods for post-training N:M activation pruning in LLMs. Across multiple LLMs, we demonstrate that pruning activations enables superior preservation of generative capabilities compared to weight pruning at equivalent sparsity levels. We evaluate lightweight, plug-and-play error mitigation techniques and pruning criteria, establishing strong hardware-friendly baselines that require minimal calibration. Furthermore, we explore sparsity patterns beyond NVIDIA's standard 2:4, showing that the 16:32 pattern achieves performance nearly on par with unstructured sparsity. However, considering the trade-off between flexibility and hardware implementation complexity, we focus on the 8:16 pattern as a superior candidate. Our findings provide both effective practical methods for activation pruning and a motivation for future hardware to support more flexible sparsity patterns. Our code is available this https URL . 

**Abstract (ZH)**: 高效的大型语言模型（LLM）推理需求促使对稀疏化技术的关注。虽然半结构化（N:M）剪枝在权重剪枝中已经被广泛研究，但在激活剪枝中的应用仍鲜有探索，尽管其在动态和输入自适应压缩以及减少I/O开销方面具有潜力。本工作全面分析了在LLM中进行N:M激活剪枝的方法。在多个LLM上，我们证明了激活剪枝在同等稀疏化水平下能更有效地保留生成能力。我们评估了轻量级、即插即用的误差缓解技术及剪枝标准，建立了强大的硬件友好基准，需要最少的校准。此外，我们探索了超越NVIDIA标准2:4的稀疏模式，显示16:32模式的性能几乎与无结构稀疏模式相匹敌。然而，考虑到灵活性与硬件实现复杂性的权衡，我们将注意力集中于16:32模式作为更优的候选。本研究提供了有效的实际激活剪枝方法，并为未来支持更多灵活稀疏模式的硬件提供了动机。我们的代码见此链接。 

---
# Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization 

**Title (ZH)**: 逼近单纯形顶点：平滑向量量化中代码折叠的一种简单 remedy 方法 

**Authors**: Takashi Morita  

**Link**: [PDF](https://arxiv.org/pdf/2509.22161)  

**Abstract**: Vector quantization, which discretizes a continuous vector space into a finite set of representative vectors (a codebook), has been widely adopted in modern machine learning. Despite its effectiveness, vector quantization poses a fundamental challenge: the non-differentiable quantization step blocks gradient backpropagation. Smoothed vector quantization addresses this issue by relaxing the hard assignment of a codebook vector into a weighted combination of codebook entries, represented as the matrix product of a simplex vector and the codebook. Effective smoothing requires two properties: (1) smoothed quantizers should remain close to a onehot vector, ensuring tight approximation, and (2) all codebook entries should be utilized, preventing code collapse. Existing methods typically address these desiderata separately. By contrast, the present study introduces a simple and intuitive regularization that promotes both simultaneously by minimizing the distance between each simplex vertex and its $K$-nearest smoothed quantizers. Experiments on representative benchmarks, including discrete image autoencoding and contrastive speech representation learning, demonstrate that the proposed method achieves more reliable codebook utilization and improves performance compared to prior approaches. 

**Abstract (ZH)**: 向量量化，通过将连续向量空间离散化为有限的代表向量集合（码本），在现代机器学习中已被广泛应用。尽管有效，但向量量化面临一个基本挑战：非可微的量化步骤阻碍了梯度反向传播。通过将码本书中的硬分配松弛为码本书条目的加权组合，表示为简单xes向量与码本书的矩阵乘积，平滑向量量化解决了这一问题。有效的平滑需要两个特性：（1）平滑量化器应接近于onehot向量，确保逼近的紧密性；（2）所有码本书条目都应被利用，防止码本书崩塌。现有方法通常分别处理这两个需求。相比之下，本文引入了一种简单直观的正则化方法，通过最小化每个简单xes顶点与其$K$个最近的平滑量化器之间的距离，同时促进这两个特性。在包括离散图像自编码和对比语音表示学习的代表性基准测试中，实验表明所提出的方法实现了更可靠的码本书利用，并且在性能上优于先前的方法。 

---
# From Long to Lean: Performance-aware and Adaptive Chain-of-Thought Compression via Multi-round Refinement 

**Title (ZH)**: 从长到精：基于多轮优化的性能 Awareness 和自适应链式思维压缩 

**Authors**: Jianzhi Yan, Le Liu, Youcheng Pan, Shiwei Chen, Zike Yuan, Yang Xiang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22144)  

**Abstract**: Chain-of-Thought (CoT) reasoning improves performance on complex tasks but introduces significant inference latency due to verbosity. We propose Multiround Adaptive Chain-of-Thought Compression (MACC), a framework that leverages the token elasticity phenomenon--where overly small token budgets can paradoxically increase output length--to progressively compress CoTs via multiround refinement. This adaptive strategy allows MACC to determine the optimal compression depth for each input. Our method achieves an average accuracy improvement of 5.6 percent over state-of-the-art baselines, while also reducing CoT length by an average of 47 tokens and significantly lowering latency. Furthermore, we show that test-time performance--accuracy and token length--can be reliably predicted using interpretable features like perplexity and compression rate on the training set. Evaluated across different models, our method enables efficient model selection and forecasting without repeated fine-tuning, demonstrating that CoT compression is both effective and predictable. Our code will be released in this https URL. 

**Abstract (ZH)**: Multiround Adaptive Chain-of-Thought Compression (MACC)提升复杂任务性能的同时降低推理延迟 

---
# REFINE-CONTROL: A Semi-supervised Distillation Method For Conditional Image Generation 

**Title (ZH)**: REFINE-CONTROL：一种条件图像生成的半监督精炼蒸馏方法 

**Authors**: Yicheng Jiang, Jin Yuan, Hua Yuan, Yao Zhang, Yong Rui  

**Link**: [PDF](https://arxiv.org/pdf/2509.22139)  

**Abstract**: Conditional image generation models have achieved remarkable results by leveraging text-based control to generate customized images. However, the high resource demands of these models and the scarcity of well-annotated data have hindered their deployment on edge devices, leading to enormous costs and privacy concerns, especially when user data is sent to a third party. To overcome these challenges, we propose Refine-Control, a semi-supervised distillation framework. Specifically, we improve the performance of the student model by introducing a tri-level knowledge fusion loss to transfer different levels of knowledge. To enhance generalization and alleviate dataset scarcity, we introduce a semi-supervised distillation method utilizing both labeled and unlabeled data. Our experiments reveal that Refine-Control achieves significant reductions in computational cost and latency, while maintaining high-fidelity generation capabilities and controllability, as quantified by comparative metrics. 

**Abstract (ZH)**: 基于文本控制的条件图像生成模型已在生成定制化图像方面取得了显著成果。然而，这些模型对资源的高需求以及标注数据的稀缺性限制了其在边缘设备上的部署，导致巨大的成本和隐私问题，尤其是在用户数据被发送到第三方时。为克服这些挑战，我们提出了一种半监督蒸馏框架Refine-Control。具体来说，我们通过引入多层次知识融合损失来提高学生模型的性能，以转移不同层次的知识。为增强泛化能力和缓解数据集稀缺性，我们引入了一种利用标记和未标记数据的半监督蒸馏方法。实验结果显示，Refine-Control在降低计算成本和延迟方面取得了显著成效，同时保持了高保真生成能力和可控性，如通过比较性指标所证明的那样。 

---
# Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding 

**Title (ZH)**: 草案策略不对齐的桥梁： speculative 解码的组树优化 

**Authors**: Shijing Hu, Jingyang Li, Zhihui Lu, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22134)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by letting a lightweight draft model propose multiple tokens that the target model verifies in parallel. Yet existing training objectives optimize only a single greedy draft path, while decoding follows a tree policy that re-ranks and verifies multiple branches. This draft policy misalignment limits achievable speedups. We introduce Group Tree Optimization (GTO), which aligns training with the decoding-time tree policy through two components: (i) Draft Tree Reward, a sampling-free objective equal to the expected acceptance length of the draft tree under the target model, directly measuring decoding performance; (ii) Group-based Draft Policy Training, a stable optimization scheme that contrasts trees from the current and a frozen reference draft model, forming debiased group-standardized advantages and applying a PPO-style surrogate along the longest accepted sequence for robust updates. We further prove that increasing our Draft Tree Reward provably improves acceptance length and speedup. Across dialogue (MT-Bench), code (HumanEval), and math (GSM8K), and multiple LLMs (e.g., LLaMA-3.1-8B, LLaMA-3.3-70B, Vicuna-1.3-13B, DeepSeek-R1-Distill-LLaMA-8B), GTO increases acceptance length by 7.4% and yields an additional 7.7% speedup over prior state-of-the-art EAGLE-3. By bridging draft policy misalignment, GTO offers a practical, general solution for efficient LLM inference. 

**Abstract (ZH)**: Group Tree Optimization加速大型语言模型推断的方法 

---
# R-Capsule: Compressing High-Level Plans for Efficient Large Language Model Reasoning 

**Title (ZH)**: R-胶囊：压缩高级规划以实现高效大型语言模型推理 

**Authors**: Hongyu Shan, Mingyang Song, Chang Dai, Di Liang, Han Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22131)  

**Abstract**: Chain-of-Thought (CoT) prompting helps Large Language Models (LLMs) tackle complex reasoning by eliciting explicit step-by-step rationales. However, CoT's verbosity increases latency and memory usage and may propagate early errors across long chains. We propose the Reasoning Capsule (R-Capsule), a framework that aims to combine the efficiency of latent reasoning with the transparency of explicit CoT. The core idea is to compress the high-level plan into a small set of learned latent tokens (a Reasoning Capsule) while keeping execution steps lightweight or explicit. This hybrid approach is inspired by the Information Bottleneck (IB) principle, where we encourage the capsule to be approximately minimal yet sufficient for the task. Minimality is encouraged via a low-capacity bottleneck, which helps improve efficiency. Sufficiency is encouraged via a dual objective: a primary task loss for answer accuracy and an auxiliary plan-reconstruction loss that encourages the capsule to faithfully represent the original textual plan. The reconstruction objective helps ground the latent space, thereby improving interpretability and reducing the use of uninformative shortcuts. Our framework strikes a balance between efficiency, accuracy, and interpretability, thereby reducing the visible token footprint of reasoning while maintaining or improving accuracy on complex benchmarks. Our codes are available at: this https URL 

**Abstract (ZH)**: Reasoning Capsule (R-Capsule): Combining Efficiency with Transparency in Latent Reasoning 

---
# Multi-Agent Path Finding via Offline RL and LLM Collaboration 

**Title (ZH)**: 多智能体路径规划 via 离线强化学习和大语言模型合作 

**Authors**: Merve Atasever, Matthew Hong, Mihir Nitin Kulkarni, Qingpei Li, Jyotirmoy V. Deshmukh  

**Link**: [PDF](https://arxiv.org/pdf/2509.22130)  

**Abstract**: Multi-Agent Path Finding (MAPF) poses a significant and challenging problem critical for applications in robotics and logistics, particularly due to its combinatorial complexity and the partial observability inherent in realistic environments. Decentralized reinforcement learning methods commonly encounter two substantial difficulties: first, they often yield self-centered behaviors among agents, resulting in frequent collisions, and second, their reliance on complex communication modules leads to prolonged training times, sometimes spanning weeks. To address these challenges, we propose an efficient decentralized planning framework based on the Decision Transformer (DT), uniquely leveraging offline reinforcement learning to substantially reduce training durations from weeks to mere hours. Crucially, our approach effectively handles long-horizon credit assignment and significantly improves performance in scenarios with sparse and delayed rewards. Furthermore, to overcome adaptability limitations inherent in standard RL methods under dynamic environmental changes, we integrate a large language model (GPT-4o) to dynamically guide agent policies. Extensive experiments in both static and dynamically changing environments demonstrate that our DT-based approach, augmented briefly by GPT-4o, significantly enhances adaptability and performance. 

**Abstract (ZH)**: 基于决策变换器的多智能体路径规划高效去中心化规划框架：结合GPT-4o动态指导代理策略以提高适应性和性能 

---
# Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM 

**Title (ZH)**: 基于监督分类模型与大语言模型紧密合作的通用法律文章预测 

**Authors**: Xiao Chi, Wenlin Zhong, Yiquan Wu, Wei Wang, Kun Kuang, Fei Wu, Minghui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.22119)  

**Abstract**: Legal Article Prediction (LAP) is a critical task in legal text classification, leveraging natural language processing (NLP) techniques to automatically predict relevant legal articles based on the fact descriptions of cases. As a foundational step in legal decision-making, LAP plays a pivotal role in determining subsequent judgments, such as charges and penalties. Despite its importance, existing methods face significant challenges in addressing the complexities of LAP. Supervised classification models (SCMs), such as CNN and BERT, struggle to fully capture intricate fact patterns due to their inherent limitations. Conversely, large language models (LLMs), while excelling in generative tasks, perform suboptimally in predictive scenarios due to the abstract and ID-based nature of legal articles. Furthermore, the diversity of legal systems across jurisdictions exacerbates the issue, as most approaches are tailored to specific countries and lack broader applicability. To address these limitations, we propose Uni-LAP, a universal framework for legal article prediction that integrates the strengths of SCMs and LLMs through tight collaboration. Specifically, in Uni-LAP, the SCM is enhanced with a novel Top-K loss function to generate accurate candidate articles, while the LLM employs syllogism-inspired reasoning to refine the final predictions. We evaluated Uni-LAP on datasets from multiple jurisdictions, and empirical results demonstrate that our approach consistently outperforms existing baselines, showcasing its effectiveness and generalizability. 

**Abstract (ZH)**: 法律文章预测（LAP）是法律文本分类中的一个关键任务，通过自然语言处理（NLP）技术，根据案件事实描述自动预测相关的法律文章。作为法律决策的基础步骤，LAP在确定后续判决（如指控和处罚）方面起着核心作用。尽管其重要性不言而喻，但现有方法在解决LAP的复杂性方面仍面临重大挑战。监督分类模型（SCMs），如CNN和BERT，由于其固有限制，难以全面捕获复杂的事实模式。相反，虽然大型语言模型（LLMs）在生成任务中表现出色，但在预测场景中表现不佳，因为法律文章具有抽象和ID基础的性质。此外，不同司法辖区法律制度的多样性进一步加剧了这一问题，大多数方法都针对特定国家进行了定制，缺乏更广泛的适用性。为了解决这些问题，我们提出了一种称为Uni-LAP的通用法律文章预测框架，该框架通过紧密协作将SCMs和LLMs的优势结合起来。具体而言，在Uni-LAP中，SCM通过引入新颖的Top-K损失函数来生成准确的候选法律文章，而LLM则采用以三段论为基础的推理来细化最终预测。我们在多个司法辖区的数据集上评估了Uni-LAP，实验证明我们的方法在所有基准之上表现出色，展示了其有效性和泛化能力。 

---
# The AI_INFN Platform: Artificial Intelligence Development in the Cloud 

**Title (ZH)**: AI_INFN平台：基于云的人工智能开发 

**Authors**: Lucio Anderlini, Giulio Bianchini, Diego Ciangottini, Stefano Dal Pra, Diego Michelotto, Rosa Petrini, Daniele Spiga  

**Link**: [PDF](https://arxiv.org/pdf/2509.22117)  

**Abstract**: Machine Learning (ML) is driving a revolution in the way scientists design, develop, and deploy data-intensive software. However, the adoption of ML presents new challenges for the computing infrastructure, particularly in terms of provisioning and orchestrating access to hardware accelerators for development, testing, and production. The INFN-funded project AI_INFN (Artificial Intelligence at INFN) aims at fostering the adoption of ML techniques within INFN use cases by providing support on multiple aspects, including the provisioning of AI-tailored computing resources. It leverages cloud-native solutions in the context of INFN Cloud, to share hardware accelerators as effectively as possible, ensuring the diversity of the Institute's research activities is not compromised. In this contribution, we provide an update on the commissioning of a Kubernetes platform designed to ease the development of GPU-powered data analysis workflows and their scalability on heterogeneous distributed computing resources, also using the offloading mechanism with Virtual Kubelet and InterLink API. This setup can manage workflows across different resource providers, including sites of the Worldwide LHC Computing Grid and supercomputers such as CINECA Leonardo, providing a model for use cases requiring dedicated infrastructures for different parts of the workload. Initial test results, emerging case studies, and integration scenarios will be presented with functional tests and benchmarks. 

**Abstract (ZH)**: AI_INFN项目：基于Kubernetes平台的GPU加速数据分析流程开发与扩展研究 

---
# Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization 

**Title (ZH)**: 少而精地学习：一种高效的政策优化动态双层降采样框架 

**Authors**: Chao Wang, Tao Yang, Hongtao Tian, Yunsheng Shi, Qiyao Ma, Xiaotao Liu, Ting Yao, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22115)  

**Abstract**: Critic-free methods like GRPO reduce memory demands by estimating advantages from multiple rollouts but tend to converge slowly, as critical learning signals are diluted by an abundance of uninformative samples and tokens. To tackle this challenge, we propose the \textbf{Dynamic Dual-Level Down-Sampling (D$^3$S)} framework that prioritizes the most informative samples and tokens across groups to improve the efficient of policy optimization. D$^3$S operates along two levels: (1) the sample-level, which selects a subset of rollouts to maximize advantage variance ($\text{Var}(A)$). We theoretically proven that this selection is positively correlated with the upper bound of the policy gradient norms, yielding higher policy gradients. (2) the token-level, which prioritizes tokens with a high product of advantage magnitude and policy entropy ($|A_{i,t}|\times H_{i,t}$), focusing updates on tokens where the policy is both uncertain and impactful. Moreover, to prevent overfitting to high-signal data, D$^3$S employs a dynamic down-sampling schedule inspired by curriculum learning. This schedule starts with aggressive down-sampling to accelerate early learning and gradually relaxes to promote robust generalization. Extensive experiments on Qwen2.5 and Llama3.1 demonstrate that integrating D$^3$S into advanced RL algorithms achieves state-of-the-art performance and generalization while requiring \textit{fewer} samples and tokens across diverse reasoning benchmarks. Our code is added in the supplementary materials and will be made publicly available. 

**Abstract (ZH)**: 动态双重级别下采样（D\(^3\)S）框架 

---
# Reinforcement Learning for Durable Algorithmic Recourse 

**Title (ZH)**: reinforcement learning for 持久的算法补救 

**Authors**: Marina Ceccon, Alessandro Fabris, Goran Radanović, Asia J. Biega, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2509.22102)  

**Abstract**: Algorithmic recourse seeks to provide individuals with actionable recommendations that increase their chances of receiving favorable outcomes from automated decision systems (e.g., loan approvals). While prior research has emphasized robustness to model updates, considerably less attention has been given to the temporal dynamics of recourse--particularly in competitive, resource-constrained settings where recommendations shape future applicant pools. In this work, we present a novel time-aware framework for algorithmic recourse, explicitly modeling how candidate populations adapt in response to recommendations. Additionally, we introduce a novel reinforcement learning (RL)-based recourse algorithm that captures the evolving dynamics of the environment to generate recommendations that are both feasible and valid. We design our recommendations to be durable, supporting validity over a predefined time horizon T. This durability allows individuals to confidently reapply after taking time to implement the suggested changes. Through extensive experiments in complex simulation environments, we show that our approach substantially outperforms existing baselines, offering a superior balance between feasibility and long-term validity. Together, these results underscore the importance of incorporating temporal and behavioral dynamics into the design of practical recourse systems. 

**Abstract (ZH)**: 算法回溯寻求为个体提供可操作的建议，以增加他们从自动化决策系统（如贷款审批）中获得有利结果的机会。尽管先前的研究强调了模型更新的稳健性，但对回溯的时间动态研究却相对较少，特别是在竞争性和资源受限的环境中，推荐会影响未来申请人的池子。在本文中，我们提出了一种新的时间感知回溯框架，明确 modeling 候选人群如何根据建议进行调整。此外，我们引入了一种基于强化学习（RL）的回溯算法，以捕捉环境的演变动态，生成既可行又有效的建议。我们设计的建议具有耐久性，可在预定义的时间窗口T内保持有效。这种耐久性使个人能够在实施建议更改后有信心重新申请。通过在复杂模拟环境中的大量实验，我们表明，我们的方法在可行性和长期有效性之间的平衡上显著优于现有基准方法。这些结果强调了将时间和行为动态纳入实用回溯系统设计中的重要性。 

---
# SecureAgentBench: Benchmarking Secure Code Generation under Realistic Vulnerability Scenarios 

**Title (ZH)**: SecureAgentBench：在实际漏洞场景下评估安全代码生成的基准测试 

**Authors**: Junkai Chen, Huihui Huang, Yunbo Lyu, Junwen An, Jieke Shi, Chengran Yang, Ting Zhang, Haoye Tian, Yikun Li, Zhenhao Li, Xin Zhou, Xing Hu, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22097)  

**Abstract**: Large language model (LLM) powered code agents are rapidly transforming software engineering by automating tasks such as testing, debugging, and repairing, yet the security risks of their generated code have become a critical concern. Existing benchmarks have offered valuable insights but remain insufficient: they often overlook the genuine context in which vulnerabilities were introduced or adopt narrow evaluation protocols that fail to capture either functional correctness or newly introduced vulnerabilities. We therefore introduce SecureAgentBench, a benchmark of 105 coding tasks designed to rigorously evaluate code agents' capabilities in secure code generation. Each task includes (i) realistic task settings that require multi-file edits in large repositories, (ii) aligned contexts based on real-world open-source vulnerabilities with precisely identified introduction points, and (iii) comprehensive evaluation that combines functionality testing, vulnerability checking through proof-of-concept exploits, and detection of newly introduced vulnerabilities using static analysis. We evaluate three representative agents (SWE-agent, OpenHands, and Aider) with three state-of-the-art LLMs (Claude 3.7 Sonnet, GPT-4.1, and DeepSeek-V3.1). Results show that (i) current agents struggle to produce secure code, as even the best-performing one, SWE-agent supported by DeepSeek-V3.1, achieves merely 15.2% correct-and-secure solutions, (ii) some agents produce functionally correct code but still introduce vulnerabilities, including new ones not previously recorded, and (iii) adding explicit security instructions for agents does not significantly improve secure coding, underscoring the need for further research. These findings establish SecureAgentBench as a rigorous benchmark for secure code generation and a step toward more reliable software development with LLMs. 

**Abstract (ZH)**: Large语言模型（LLM）驱动的代码代理正迅速通过自动化诸如测试、调试和修复等任务来改变软件工程，但它们生成代码的安全风险已成为一个关键问题。现有基准提供了宝贵见解但仍显不足：它们往往忽略了漏洞实际引入的背景，或者采用狭窄的评估协议，无法全面评估功能正确性或检测新引入的漏洞。因此，我们引入了SecureAgentBench，这是一个包含105个编码任务的基准，旨在严格评估代码代理在安全代码生成方面的能力。每个任务包括：（i）现实的任务设置，涉及大型代码库中的多文件编辑；（ii）基于真实世界开源漏洞的对齐背景，具有精确标识的引入点；（iii）结合功能测试、通过概念验证利用进行漏洞检查以及使用静态分析检测新引入漏洞的全面评估。我们使用三种代表性的代理（SWE-agent、OpenHands和Aider）和三种最新的LLM（Claude 3.7 Sonnet、GPT-4.1和DeepSeek-V3.1）进行评估。结果显示：（i）现有代理在生成安全代码方面存在困难，即使表现最佳的SWE-agent（支持DeepSeek-V3.1）也只是实现了15.2%的正确且安全的解决方案；（ii）一些代理能够生成功能正确的代码但仍引入漏洞，包括新的未被记录的漏洞；（iii）为代理添加显式安全指令并未显著提高安全编码能力，这凸显了进一步研究的必要性。这些发现确立了SecureAgentBench作为安全代码生成的严谨基准，并为进一步利用LLM实现更可靠软件开发奠定了基础。 

---
# Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation 

**Title (ZH)**: 基于动作感知的动态剪枝以实现高效的视觉-语言-动作操纵 

**Authors**: Xiaohuan Pei, Yuxing Chen, Siyu Xu, Yunke Wang, Yuheng Shi, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22093)  

**Abstract**: Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. Motivated by this observation, we propose \textbf{A}ction-aware \textbf{D}ynamic \textbf{P}runing (\textbf{ADP}), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. Our method introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (\textit{e.g.} $1.35 \times$ speed up on OpenVLA-OFT) while maintaining competitive success rates (\textit{e.g.} 25.8\% improvements with OpenVLA) compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation. Our project website is: \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 具有视觉-语言-动作模型的机器人操作需要高效处理长时跨多模态上下文，其中对密集视觉标记的关注主导了计算成本。现有方法通过减少VLA模型内的视觉冗余来优化推理速度，但忽略了机器人操作各阶段的冗余变化。我们观察到，在粗粒度操作阶段，视觉标记冗余高于精细操作阶段，并且与动作动态密切相关。受此观察启发，我们提出了一种多模态剪枝框架——感知动作动态剪枝（ADP），该框架结合了文本驱动的标记选择和感知动作轨迹门控。我们的方法引入了一种门控机制，该机制根据近期动作轨迹来调整剪枝信号，并使用过去的运动窗口来适应性调整标记保留比例，从而在不同操作阶段平衡计算效率和感知精度。在LIBERO套件和多种现实世界场景上的广泛实验表明，与基线方法相比，我们的方法在保持竞争力的操作成功率（例如，在OpenVLA上提高25.8%）的同时显著减少了FLOPs和动作推理延迟（例如，在OpenVLA-OFT上加速1.35倍），从而提供了一条简单插件路径，以提高机器人操作的效率和性能前沿。我们的项目网站是：\href{this https URL}{this http URL}。 

---
# The Rogue Scalpel: Activation Steering Compromises LLM Safety 

**Title (ZH)**: rogue手术刀：激活 steering 影响 LLM 安全性 

**Authors**: Anton Korznikov, Andrey Galichin, Alexey Dontsov, Oleg Y. Rogov, Ivan Oseledets, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2509.22067)  

**Abstract**: Activation steering is a promising technique for controlling LLM behavior by adding semantically meaningful vectors directly into a model's hidden states during inference. It is often framed as a precise, interpretable, and potentially safer alternative to fine-tuning. We demonstrate the opposite: steering systematically breaks model alignment safeguards, making it comply with harmful requests. Through extensive experiments on different model families, we show that even steering in a random direction can increase the probability of harmful compliance from 0% to 2-27%. Alarmingly, steering benign features from a sparse autoencoder (SAE), a common source of interpretable directions, increases these rates by a further 2-4%. Finally, we show that combining 20 randomly sampled vectors that jailbreak a single prompt creates a universal attack, significantly increasing harmful compliance on unseen requests. These results challenge the paradigm of safety through interpretability, showing that precise control over model internals does not guarantee precise control over model behavior. 

**Abstract (ZH)**: 激活引导是一种通过在推理过程中直接向模型的隐藏状态添加语义上有意义的向量来控制LLM行为的有前景的技术。它通常被框定为一种精确、可解释且可能更安全的替代调优方法。我们展示了相反的观点：引导系统地破坏了模型对齐的安全机制，使其遵循有害请求。通过在不同的模型家族中进行广泛的实验，我们显示即使随机引导也会将有害合规性从0%提高到2%-27%。更 alarmingly，从稀疏自编码器（SAE）中提取的无害特征进一步增加了这些比率2%-4%。最后，我们展示了将20个随机采样的释放单一提示的向量组合成一个通用攻击，显着增加了未见请求的有害合规性。这些结果挑战了通过可解释性实现安全的范式，表明对模型内部的精确控制并不保证对模型行为的精确控制。 

---
# The QCET Taxonomy of Standard Quality Criterion Names and Definitions for the Evaluation of NLP Systems 

**Title (ZH)**: 标准质量标准名称和定义的QCET分类体系：NLP系统评估 

**Authors**: Anya Belz, Simon Mille, Craig Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2509.22064)  

**Abstract**: Prior work has shown that two NLP evaluation experiments that report results for the same quality criterion name (e.g. Fluency) do not necessarily evaluate the same aspect of quality, and the comparability implied by the name can be misleading. Not knowing when two evaluations are comparable in this sense means we currently lack the ability to draw reliable conclusions about system quality on the basis of multiple, independently conducted evaluations. This in turn hampers the ability of the field to progress scientifically as a whole, a pervasive issue in NLP since its beginning (Sparck Jones, 1981). It is hard to see how the issue of unclear comparability can be fully addressed other than by the creation of a standard set of quality criterion names and definitions that the several hundred quality criterion names actually in use in the field can be mapped to, and grounded in. Taking a strictly descriptive approach, the QCET Quality Criteria for Evaluation Taxonomy derives a standard set of quality criterion names and definitions from three surveys of evaluations reported in NLP, and structures them into a hierarchy where each parent node captures common aspects of its child nodes. We present QCET and the resources it consists of, and discuss its three main uses in (i) establishing comparability of existing evaluations, (ii) guiding the design of new evaluations, and (iii) assessing regulatory compliance. 

**Abstract (ZH)**: QCET质量标准评价分类体系 

---
# Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks 

**Title (ZH)**: 解码欺诈：理解自动语音识别在规避和投毒攻击中的脆弱性 

**Authors**: Aravindhan G, Yuvaraj Govindarajulu, Parin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.22060)  

**Abstract**: Recent studies have demonstrated the vulnerability of Automatic Speech Recognition systems to adversarial examples, which can deceive these systems into misinterpreting input speech commands. While previous research has primarily focused on white-box attacks with constrained optimizations, and transferability based black-box attacks against commercial Automatic Speech Recognition devices, this paper explores cost efficient white-box attack and non transferability black-box adversarial attacks on Automatic Speech Recognition systems, drawing insights from approaches such as Fast Gradient Sign Method and Zeroth-Order Optimization. Further, the novelty of the paper includes how poisoning attack can degrade the performances of state-of-the-art models leading to misinterpretation of audio signals. Through experimentation and analysis, we illustrate how hybrid models can generate subtle yet impactful adversarial examples with very little perturbation having Signal Noise Ratio of 35dB that can be generated within a minute. These vulnerabilities of state-of-the-art open source model have practical security implications, and emphasize the need for adversarial security. 

**Abstract (ZH)**: 最近的研究表明，自动语音识别系统容易受到对抗样本的攻击，这些攻击可以使系统错误解读输入的语音命令。尽管此前的研究主要集中在具有约束优化的白盒攻击以及针对商业自动语音识别设备的迁移性基于黑盒攻击上，本文探讨了自动语音识别系统的低成本白盒攻击和非迁移性黑盒对抗攻击，借鉴了快速梯度符号方法和零阶优化等方法。此外，本文的创新之处在于揭示了中毒攻击如何降低最新模型的性能，导致对音频信号的误解释。通过实验和分析，我们展示了混合模型可以生成具有35dB信噪比、仅需一分钟即可生成且具有轻微但影响深远的对抗样本。这些最新开源模型的安全漏洞具有实际的安全意义，并强调了对抗安全的必要性。 

---
# An Adaptive ICP LiDAR Odometry Based on Reliable Initial Pose 

**Title (ZH)**: 基于可靠初始姿态的自适应ICP激光雷达里程计 

**Authors**: Qifeng Wang, Weigang Li, Lei Nie, Xin Xu, Wenping Liu, Zhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22058)  

**Abstract**: As a key technology for autonomous navigation and positioning in mobile robots, light detection and ranging (LiDAR) odometry is widely used in autonomous driving applications. The Iterative Closest Point (ICP)-based methods have become the core technique in LiDAR odometry due to their efficient and accurate point cloud registration capability. However, some existing ICP-based methods do not consider the reliability of the initial pose, which may cause the method to converge to a local optimum. Furthermore, the absence of an adaptive mechanism hinders the effective handling of complex dynamic environments, resulting in a significant degradation of registration accuracy. To address these issues, this paper proposes an adaptive ICP-based LiDAR odometry method that relies on a reliable initial pose. First, distributed coarse registration based on density filtering is employed to obtain the initial pose estimation. The reliable initial pose is then selected by comparing it with the motion prediction pose, reducing the initial error between the source and target point clouds. Subsequently, by combining the current and historical errors, the adaptive threshold is dynamically adjusted to accommodate the real-time changes in the dynamic environment. Finally, based on the reliable initial pose and the adaptive threshold, point-to-plane adaptive ICP registration is performed from the current frame to the local map, achieving high-precision alignment of the source and target point clouds. Extensive experiments on the public KITTI dataset demonstrate that the proposed method outperforms existing approaches and significantly enhances the accuracy of LiDAR odometry. 

**Abstract (ZH)**: 基于可靠初值的自适应ICP激光雷达里程ometry方法及其应用 

---
# Fuzzy Reasoning Chain (FRC): An Innovative Reasoning Framework from Fuzziness to Clarity 

**Title (ZH)**: 模糊推理链（FRC）：从模糊性到清晰性的创新推理框架 

**Authors**: Ping Chen, Xiang Liu, Zhaoxiang Liu, Zezhou Chen, Xingpeng Zhang, Huan Hu, Zipeng Wang, Kai Wang, Shuming Shi, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.22054)  

**Abstract**: With the rapid advancement of large language models (LLMs), natural language processing (NLP) has achieved remarkable progress. Nonetheless, significant challenges remain in handling texts with ambiguity, polysemy, or uncertainty. We introduce the Fuzzy Reasoning Chain (FRC) framework, which integrates LLM semantic priors with continuous fuzzy membership degrees, creating an explicit interaction between probability-based reasoning and fuzzy membership reasoning. This transition allows ambiguous inputs to be gradually transformed into clear and interpretable decisions while capturing conflicting or uncertain signals that traditional probability-based methods cannot. We validate FRC on sentiment analysis tasks, where both theoretical analysis and empirical results show that it ensures stable reasoning and facilitates knowledge transfer across different model scales. These findings indicate that FRC provides a general mechanism for managing subtle and ambiguous expressions with improved interpretability and robustness. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速进步，自然语言处理（NLP）取得了显著进展。然而，在处理具有歧义、多义或不确定性的文本时，仍面临重大挑战。我们介绍了模糊推理链（FRC）框架，该框架将LLM语义先验与连续的模糊隶属度结合，创建了基于概率的推理和模糊隶属度推理之间的明确交互。这一转变使模糊输入能够逐步转化为清晰且可解释的决策，并捕捉传统基于概率的方法无法处理的矛盾或不确定性信号。我们在情感分析任务上验证了FRC，理论分析和实证结果均表明，它确保了推理的稳定性并促进了不同模型规模之间的知识转移。这些发现表明，FRC提供了一种用于管理微妙和模糊表达的一般机制，具有增强的可解释性和鲁棒性。 

---
# Latent Diffusion : Multi-Dimension Stable Diffusion Latent Space Explorer 

**Title (ZH)**: 潜在扩散：多维稳定扩散潜在空间探索者 

**Authors**: Zhihua Zhong, Xuanyang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22038)  

**Abstract**: Latent space is one of the key concepts in generative AI, offering powerful means for creative exploration through vector manipulation. However, diffusion models like Stable Diffusion lack the intuitive latent vector control found in GANs, limiting their flexibility for artistic expression. This paper introduces \workname, a framework for integrating customizable latent space operations into the diffusion process. By enabling direct manipulation of conceptual and spatial representations, this approach expands creative possibilities in generative art. We demonstrate the potential of this framework through two artworks, \textit{Infinitepedia} and \textit{Latent Motion}, highlighting its use in conceptual blending and dynamic motion generation. Our findings reveal latent space structures with semantic and meaningless regions, offering insights into the geometry of diffusion models and paving the way for further explorations of latent space. 

**Abstract (ZH)**: 一种将可定制的潜在空间操作集成到扩散过程中的框架：探索概念与空间表示的直接操纵在生成艺术中的应用 

---
# Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics 

**Title (ZH)**: 轻量级结构化多模态推理在机器人临床场景理解中的应用 

**Authors**: Saurav Jha, Stefan K. Ehrlich  

**Link**: [PDF](https://arxiv.org/pdf/2509.22014)  

**Abstract**: Healthcare robotics requires robust multimodal perception and reasoning to ensure safety in dynamic clinical environments. Current Vision-Language Models (VLMs) demonstrate strong general-purpose capabilities but remain limited in temporal reasoning, uncertainty estimation, and structured outputs needed for robotic planning. We present a lightweight agentic multimodal framework for video-based scene understanding. Combining the Qwen2.5-VL-3B-Instruct model with a SmolAgent-based orchestration layer, it supports chain-of-thought reasoning, speech-vision fusion, and dynamic tool invocation. The framework generates structured scene graphs and leverages a hybrid retrieval module for interpretable and adaptive reasoning. Evaluations on the Video-MME benchmark and a custom clinical dataset show competitive accuracy and improved robustness compared to state-of-the-art VLMs, demonstrating its potential for applications in robot-assisted surgery, patient monitoring, and decision support. 

**Abstract (ZH)**: healthcare 机器人要求具备 robust 多模态感知与推理能力，以确保在动态临床环境中的安全性。当前的 Vision-Language 模型 (VLMs) 展现出强大的通用能力，但在时间推理、不确定性估计和所需供机器人规划的结构化输出方面仍存在局限性。我们提出了一种轻量级的 agentic 多模态框架，用于基于视频的场景理解。通过将 Qwen2.5-VL-3B-Instruct 模型与基于 SmolAgent 的编排层结合，该框架支持链式思考推理、语音-视觉融合以及动态工具调用。该框架生成结构化的场景图，并利用混合检索模块实现可解释和自适应推理。在 Video-MME 基准和自定义临床数据集上的评估表明，其准确性和鲁棒性优于当前最先进的 VLMs，展示了其在机器人辅助手术、患者监测和决策支持方面的潜在应用。 

---
# Black-Box Hallucination Detection via Consistency Under the Uncertain Expression 

**Title (ZH)**: 黑盒幻觉检测：在不确定表达下的一致性方法 

**Authors**: Seongho Joo, Kyungmin Min, Jahyun Koo, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.21999)  

**Abstract**: Despite the great advancement of Language modeling in recent days, Large Language Models (LLMs) such as GPT3 are notorious for generating non-factual responses, so-called "hallucination" problems. Existing methods for detecting and alleviating this hallucination problem require external resources or the internal state of LLMs, such as the output probability of each token. Given the LLM's restricted external API availability and the limited scope of external resources, there is an urgent demand to establish the Black-Box approach as the cornerstone for effective hallucination detection. In this work, we propose a simple black-box hallucination detection metric after the investigation of the behavior of LLMs under expression of uncertainty. Our comprehensive analysis reveals that LLMs generate consistent responses when they present factual responses while non-consistent responses vice versa. Based on the analysis, we propose an efficient black-box hallucination detection metric with the expression of uncertainty. The experiment demonstrates that our metric is more predictive of the factuality in model responses than baselines that use internal knowledge of LLMs. 

**Abstract (ZH)**: 尽管近年来语言模型取得了巨大进展，大型语言模型（LLMs）如GPT3known for生成不实回应，即所谓的“幻觉”问题。现有的一些检测和缓解幻觉问题的方法需要外部资源或LLMs的内部状态，比如每个token的输出概率。鉴于LLMs对外部API的限制以及外部资源的有限性，迫切需要建立黑盒方法作为有效检测幻觉的基础。在对LLMs在表达不确定性下的行为进行调查之后，我们提出了一种简单的黑盒幻觉检测指标。综合分析表明，LLMs在提供事实回应时表现出一致性，反之亦然。基于这一分析，我们提出了一种基于表达不确定性的有效黑盒幻觉检测指标。实验表明，我们的指标比使用LLMs内部知识的基线更能预测模型回应的事实性。 

---
# ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models 

**Title (ZH)**: ERGO: 高效的高分辨率视觉理解方法for Vision-Language模型 

**Authors**: Jewon Lee, Wooksu Shin, Seungmin Yang, Ki-Ung Song, DongUk Lim, Jaeyeon Kim, Tae-Ho Kim, Bo-Kyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21991)  

**Abstract**: Efficient processing of high-resolution images is crucial for real-world vision-language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception-leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3x inference speedup. The code and models can be found at: this https URL. 

**Abstract (ZH)**: 高效处理高分辨率图像对于实际视觉-语言应用至关重要。然而，现有的大型视觉-语言模型因视觉标记数量庞大而产生显著的计算开销。随着“以图思考”模型的出现，推理已经从文本领域扩展到视觉领域。这一能力促使我们提出一种两阶段“粗细结合”的推理管道：首先，对下采样的图像进行分析以识别与任务相关的区域；然后，仅对这些区域进行全分辨率裁剪并在后续推理阶段进行处理。这种方法在必要时保留了细粒度的视觉细节，同时减少了计算成本。主要挑战在于推断哪些区域真正与给定查询相关。最近的相关方法往往在输入图像下采样后的第一阶段失效，因为在这种感知驱动的推理中，清晰的视觉信息对于有效推理是必要的。为解决这一问题，我们提出ERGO（高效推理与引导观察），它利用引导感知的多模态语境来进行推理驱动的感知。我们的模型可以考虑到感知的不确定性，扩展裁剪区域以覆盖视觉模糊区域以回答问题。为此，我们在一个粗细结合的感知框架下开发了简单有效的奖励组件。在多个数据集中，我们的方法在准确性和效率方面均优于原模型和竞品方法。例如，ERGO在V*基准测试中比Qwen2.5-VL-7B高出4.7分，仅使用了23%的视觉标记，实现了3倍的推理速度提升。相关代码和模型可在以下链接找到：this https URL。 

---
# Developing Vision-Language-Action Model from Egocentric Videos 

**Title (ZH)**: 基于第一人称视频发展视觉-语言-动作模型 

**Authors**: Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, Shinsuke Mori  

**Link**: [PDF](https://arxiv.org/pdf/2509.21986)  

**Abstract**: Egocentric videos capture how humans manipulate objects and tools, providing diverse motion cues for learning object manipulation. Unlike the costly, expert-driven manual teleoperation commonly used in training Vision-Language-Action models (VLAs), egocentric videos offer a scalable alternative. However, prior studies that leverage such videos for training robot policies typically rely on auxiliary annotations, such as detailed hand-pose recordings. Consequently, it remains unclear whether VLAs can be trained directly from raw egocentric videos. In this work, we address this challenge by leveraging EgoScaler, a framework that extracts 6DoF object manipulation trajectories from egocentric videos without requiring auxiliary recordings. We apply EgoScaler to four large-scale egocentric video datasets and automatically refine noisy or incomplete trajectories, thereby constructing a new large-scale dataset for VLA pre-training. Our experiments with a state-of-the-art $\pi_0$ architecture in both simulated and real-robot environments yield three key findings: (i) pre-training on our dataset improves task success rates by over 20\% compared to training from scratch, (ii) the performance is competitive with that achieved using real-robot datasets, and (iii) combining our dataset with real-robot data yields further improvements. These results demonstrate that egocentric videos constitute a promising and scalable resource for advancing VLA research. 

**Abstract (ZH)**: 自视点视频捕捉人类操作物体和工具的方式，提供了学习物体操作的多样性运动线索。与训练视觉-语言-动作模型（VLAs）过程中常见的昂贵且由专家驱动的手动远程操作不同，自视点视频提供了可扩展的替代方案。然而，之前利用此类视频训练机器人策略的研究通常依赖于辅助标注，如详细的手部姿态记录。因此，仍然不清楚VLAs是否可以直接从原始自视点视频中训练得到。在本工作中，我们通过利用EgoScaler框架解决了这一挑战，该框架可以从自视点视频中提取6自由度的物体操作轨迹，而无需辅助记录。我们将EgoScaler应用于四个大规模自视点视频数据集，并自动精炼噪音或不完整轨迹，从而构建了一个新的大型数据集用于VLAs的预训练。我们使用最先进的$\pi_0$架构在模拟和实际机器人环境中进行的实验揭示了三个关键发现：（i）在我们的数据集上预训练比从头开始训练的任务成功率提高了超过20%；（ii）性能与使用实际机器人数据集所取得的性能相当；（iii）将我们的数据集与实际机器人数据集结合使用进一步提高了性能。这些结果表明，自视点视频是推动VLAs研究的一种有前景且可扩展的资源。 

---
# Hybrid Diffusion for Simultaneous Symbolic and Continuous Planning 

**Title (ZH)**: 混合扩散usi时的符号和连续规划 

**Authors**: Sigmund Hennum Høeg, Aksel Vaaler, Chaoqi Liu, Olav Egeland, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.21983)  

**Abstract**: Constructing robots to accomplish long-horizon tasks is a long-standing challenge within artificial intelligence. Approaches using generative methods, particularly Diffusion Models, have gained attention due to their ability to model continuous robotic trajectories for planning and control. However, we show that these models struggle with long-horizon tasks that involve complex decision-making and, in general, are prone to confusing different modes of behavior, leading to failure. To remedy this, we propose to augment continuous trajectory generation by simultaneously generating a high-level symbolic plan. We show that this requires a novel mix of discrete variable diffusion and continuous diffusion, which dramatically outperforms the baselines. In addition, we illustrate how this hybrid diffusion process enables flexible trajectory synthesis, allowing us to condition synthesized actions on partial and complete symbolic conditions. 

**Abstract (ZH)**: 构建用于完成长远任务的机器人是人工智能领域的一个长期挑战。通过生成方法，特别是扩散模型，由于其能够建模连续的机器人轨迹以进行规划和控制，这些方法受到了关注。然而，我们展示了这些模型在处理涉及复杂决策的长远任务时存在困难，通常会混淆不同行为模式，导致失败。为解决这一问题，我们提出通过同时生成高层符号计划来扩展连续轨迹生成。我们表明，这需要一种新颖的离散变量扩散与连续扩散的结合，这在基准方法上表现出色。此外，我们展示了这种混合扩散过程如何使轨迹合成更加灵活，从而使生成的动作能够根据部分和完整的符号条件进行条件化。 

---
# Benchmarking and Mitigate Psychological Sycophancy in Medical Vision-Language Models 

**Title (ZH)**: 医学视觉-语言模型中心理阿谀现象的基准测试与缓解 

**Authors**: Zikun Guo, Xinyue Xu, Pei Xiang, Shu Yang, Xin Han, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21979)  

**Abstract**: Vision language models(VLMs) are increasingly integrated into clinical workflows, but they often exhibit sycophantic behavior prioritizing alignment with user phrasing social cues or perceived authority over evidence based reasoning. This study evaluate clinical sycophancy in medical visual question answering through a novel clinically grounded benchmark. We propose a medical sycophancy dataset construct from PathVQA, SLAKE, and VQA-RAD stratified by different type organ system and modality. Using psychologically motivated pressure templates including various sycophancy. In our adversarial experiments on various VLMs, we found that these models are generally vulnerable, exhibiting significant variations in the occurrence of adversarial responses, with weak correlations to the model accuracy or size. Imitation and expert provided corrections were found to be the most effective triggers, suggesting that the models possess a bias mechanism independent of visual evidence. To address this, we propose Visual Information Purification for Evidence based Response (VIPER) a lightweight mitigation strategy that filters non evidentiary content for example social pressures and then generates constrained evidence first answers. This framework reduces sycophancy by an average amount outperforming baselines while maintaining interpretability. Our benchmark analysis and mitigation framework lay the groundwork for robust deployment of medical VLMs in real world clinician interactions emphasizing the need for evidence anchored defenses. 

**Abstract (ZH)**: 医学视觉问答中临床奉承现象的评估：基于临床背景的新基准及视觉信息净化以生成基于证据的响应（VIPER）方法 

---
# Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning 

**Title (ZH)**: Geo-R1: 通过强化微调改进 Few-Shot 地理空间参考表达理解 

**Authors**: Zilun Zhang, Zian Guan, Tiancheng Zhao, Haozhan Shen, Tianyu Li, Yuxiang Cai, Zhonggen Su, Zhaojun Liu, Jianwei Yin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21976)  

**Abstract**: Referring expression understanding in remote sensing poses unique challenges, as it requires reasoning over complex object-context relationships. While supervised fine-tuning (SFT) on multimodal large language models achieves strong performance with massive labeled datasets, they struggle in data-scarce scenarios, leading to poor generalization. To address this limitation, we propose Geo-R1, a reasoning-centric reinforcement fine-tuning (RFT) paradigm for few-shot geospatial referring. Geo-R1 enforces the model to first generate explicit, interpretable reasoning chains that decompose referring expressions, and then leverage these rationales to localize target objects. This "reason first, then act" process enables the model to make more effective use of limited annotations, enhances generalization, and provides interpretability. We validate Geo-R1 on three carefully designed few-shot geospatial referring benchmarks, where our model consistently and substantially outperforms SFT baselines. It also demonstrates strong cross-dataset generalization, highlighting its robustness. Code and data will be released at this http URL. 

**Abstract (ZH)**: 遥感中的指代表达理解面临着独特的挑战，因为它需要推理复杂的目标-上下文关系。尽管在多模态大语言模型上进行有监督微调（SFT）可以使用大量标注数据获得优异性能，但在数据稀疏场景下却表现不佳，导致泛化能力差。为解决这一局限性，我们提出Geo-R1，这是一种以推理为中心的强化微调（RFT）范式，用于少量样本的地理空间指代。Geo-R1 要求模型首先生成明确且可解释的推理链来分解指代表达，然后利用这些理由来定位目标对象。这一“先推理后行动”的过程使模型能够更有效地利用有限的标注信息，增强泛化能力和可解释性。我们在三个精心设计的少量样本地理空间指代基准上验证了Geo-R1，其中我们的模型在所有基准上都一致且显著地优于SFT基线。同时，它还展示了强大的跨数据集泛化能力，突显了其鲁棒性。代码和数据将发布在该网址。 

---
# From Superficial Outputs to Superficial Learning: Risks of Large Language Models in Education 

**Title (ZH)**: 从表层输出到表层学习：大型语言模型在教育中的风险 

**Authors**: Iris Delikoura, Yi.R, Fung, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2509.21972)  

**Abstract**: Large Language Models (LLMs) are transforming education by enabling personalization, feedback, and knowledge access, while also raising concerns about risks to students and learning systems. Yet empirical evidence on these risks remains fragmented. This paper presents a systematic review of 70 empirical studies across computer science, education, and psychology. Guided by four research questions, we examine: (i) which applications of LLMs in education have been most frequently explored; (ii) how researchers have measured their impact; (iii) which risks stem from such applications; and (iv) what mitigation strategies have been proposed. We find that research on LLMs clusters around three domains: operational effectiveness, personalized applications, and interactive learning tools. Across these, model-level risks include superficial understanding, bias, limited robustness, anthropomorphism, hallucinations, privacy concerns, and knowledge constraints. When learners interact with LLMs, these risks extend to cognitive and behavioural outcomes, including reduced neural activity, over-reliance, diminished independent learning skills, and a loss of student agency. To capture this progression, we propose an LLM-Risk Adapted Learning Model that illustrates how technical risks cascade through interaction and interpretation to shape educational outcomes. As the first synthesis of empirically assessed risks, this review provides a foundation for responsible, human-centred integration of LLMs in education. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过实现个性化、反馈和知识访问正在改变教育，同时也引发了对学生和学习系统风险的关注。然而，关于这些风险的实证证据仍然支离破碎。本论文对来自计算机科学、教育和心理学领域的70项实证研究进行了系统综述。根据四个研究问题，我们探讨了：（i）教育中哪些大型语言模型的应用最为频繁；（ii）研究人员如何衡量其影响；（iii）这些应用带来的哪些风险；以及（iv）提出的哪些缓解策略。我们发现，关于大型语言模型的研究主要集中在三个领域：操作有效性、个性化应用和互动学习工具。在这三个领域中，模型层面的风险包括表面理解、偏见、有限的稳健性、拟人化、幻觉、隐私担忧和知识限制。当学习者与大型语言模型互动时，这些风险延伸到认知和行为结果，包括减少的大脑活动、过度依赖、削弱的独立学习能力以及学生自主性的丧失。为了捕捉这一进展，我们提出了一种大型语言模型风险适应性学习模型，以说明技术风险如何通过互动和解释影响教育结果。作为第一个关于经过实证评估的风险的综述，本研究为负责任地在教育中集成大型语言模型提供了基础。 

---
# No-Reference Image Contrast Assessment with Customized EfficientNet-B0 

**Title (ZH)**: 基于定制化EfficientNet-B0的无参考图像对比度评估 

**Authors**: Javad Hassannataj Joloudari, Bita Mesbahzadeh, Omid Zare, Emrah Arslan, Roohallah Alizadehsani, Hossein Moosaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.21967)  

**Abstract**: Image contrast was a fundamental factor in visual perception and played a vital role in overall image quality. However, most no reference image quality assessment NR IQA models struggled to accurately evaluate contrast distortions under diverse real world conditions. In this study, we proposed a deep learning based framework for blind contrast quality assessment by customizing and fine-tuning three pre trained architectures, EfficientNet B0, ResNet18, and MobileNetV2, for perceptual Mean Opinion Score, along with an additional model built on a Siamese network, which indicated a limited ability to capture perceptual contrast distortions. Each model is modified with a contrast-aware regression head and trained end to end using targeted data augmentations on two benchmark datasets, CID2013 and CCID2014, containing synthetic and authentic contrast distortions. Performance is evaluated using Pearson Linear Correlation Coefficient and Spearman Rank Order Correlation Coefficient, which assess the alignment between predicted and human rated scores. Among these three models, our customized EfficientNet B0 model achieved state-of-the-art performance with PLCC = 0.9286 and SRCC = 0.9178 on CCID2014 and PLCC = 0.9581 and SRCC = 0.9369 on CID2013, surpassing traditional methods and outperforming other deep baselines. These results highlighted the models robustness and effectiveness in capturing perceptual contrast distortion. Overall, the proposed method demonstrated that contrast aware adaptation of lightweight pre trained networks can yield a high performing, scalable solution for no reference contrast quality assessment suitable for real time and resource constrained applications. 

**Abstract (ZH)**: 基于深度学习的盲对比度质量评估框架：定制与微调 EfficientNet B0、ResNet18 和 MobileNetV2 及 Siamese 网络在视觉对比度感知评分中的应用 

---
# FlowDrive: moderated flow matching with data balancing for trajectory planning 

**Title (ZH)**: FlowDrive：带数据平衡的流量匹配路径规划 

**Authors**: Lingguang Wang, Ömer Şahin Taş, Marlon Steiner, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2509.21961)  

**Abstract**: Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches methods with rule-based refinements. After adding moderated guidance and light post-processing (FlowDrive*), it achieves overall state-of-the-art performance across nearly all benchmark splits. 

**Abstract (ZH)**: 基于学习的规划器对驾驶数据的长尾分布敏感。常见操作在数据集中占据主导地位，而危险或罕见场景则极少出现。这种不平衡会使模型偏向常见情况，从而在关键场景上的性能下降。为解决这一问题，我们比较了采样训练数据的平衡策略，并发现基于轨迹模式加权是一种有效的approach。我们随后提出了FlowDrive，这是一种流动匹配轨迹规划器，通过学习条件矫正流动直接将噪声映射到轨迹分布中，并仅需少量流动匹配步骤。我们进一步引入了适度的在环指导，通过在流动步骤之间注入小的扰动，系统地增加轨迹多样性，同时保持场景一致性。在nuPlan和交互聚焦的interPlan基准测试中，FlowDrive在基于学习的规划器中达到了最先进的性能，并接近基于规则细化的方法。在添加了适度指导和轻量级后处理（FlowDrive*）后，它在几乎所有基准测试分割中实现了整体最先进性能。 

---
# Active Attacks: Red-teaming LLMs via Adaptive Environments 

**Title (ZH)**: 主动攻击：通过适应性环境红队演练LLMs 

**Authors**: Taeyoung Yun, Pierre-Luc St-Charles, Jinkyoo Park, Yoshua Bengio, Minsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21947)  

**Abstract**: We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{this https URL}{here}. 

**Abstract (ZH)**: 我们提出了主动攻击（Active Attacks），一种基于强化学习的红队算法，该算法随着受害模型的发展而调整其攻击策略。通过定期使用收集到的攻击提示对受害的大语言模型进行安全性微调，已被利用的区域的奖励会减少，迫使攻击者寻找未探索的漏洞。这一过程自然地诱导出一个从易到难的探索课程，攻击者逐渐从简单的模式过渡到更具挑战性的模式。结果，主动攻击逐步揭示了广泛的地方攻击模式，并且这些模式的结合实现了多模式分布的广泛覆盖。主动攻击作为一个简单的即插即用模块无缝集成到现有的强化学习目标中，意外地优于之前基于强化学习的方法——包括GFlowNets、PPO和REINFORCE，仅通过将计算量增加6%，在GFlowNets这种之前最先进的方法上实现了从0.07%到31.28%（相对增益超过400倍）的跨攻击成功率提升。我们的代码已公开可在\[这里\]找到。 

---
# Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration 

**Title (ZH)**: 基于反事实校准的大规模语言模型在泰国政治立场检测中的去偏见化 

**Authors**: Kasidit Sermsri, Teerapong Panboonyuen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21946)  

**Abstract**: Political stance detection in low-resource and culturally complex settings poses a critical challenge for large language models (LLMs). In the Thai political landscape - marked by indirect language, polarized figures, and entangled sentiment and stance - LLMs often display systematic biases such as sentiment leakage and favoritism toward entities. These biases undermine fairness and reliability. We present ThaiFACTUAL, a lightweight, model-agnostic calibration framework that mitigates political bias without requiring fine-tuning. ThaiFACTUAL uses counterfactual data augmentation and rationale-based supervision to disentangle sentiment from stance and reduce bias. We also release the first high-quality Thai political stance dataset, annotated with stance, sentiment, rationales, and bias markers across diverse entities and events. Experimental results show that ThaiFACTUAL significantly reduces spurious correlations, enhances zero-shot generalization, and improves fairness across multiple LLMs. This work highlights the importance of culturally grounded debiasing techniques for underrepresented languages. 

**Abstract (ZH)**: 在低资源且文化复杂环境中检测政治立场对大型语言模型构成严峻挑战。ThaiFACTUAL：一种无需微调的轻量级、模型无关校准框架，用于缓解政治偏差。 

---
# Unveiling Many Faces of Surrogate Models for Configuration Tuning: A Fitness Landscape Analysis Perspective 

**Title (ZH)**: 探究代理模型在配置调整中多样化的一面：从适应度景观分析视角 

**Authors**: Pengzhou Chen, Hongyuan Liang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21945)  

**Abstract**: To efficiently tune configuration for better system performance (e.g., latency), many tuners have leveraged a surrogate model to expedite the process instead of solely relying on the profoundly expensive system measurement. As such, it is naturally believed that we need more accurate models. However, the fact of accuracy can lie-a somewhat surprising finding from prior work-has left us many unanswered questions regarding what role the surrogate model plays in configuration tuning. This paper provides the very first systematic exploration and discussion, together with a resolution proposal, to disclose the many faces of surrogate models for configuration tuning, through the novel perspective of fitness landscape analysis. We present a theory as an alternative to accuracy for assessing the model usefulness in tuning, based on which we conduct an extensive empirical study involving up to 27,000 cases. Drawing on the above, we propose Model4Tune, an automated predictive tool that estimates which model-tuner pairs are the best for an unforeseen system without expensive tuner profiling. Our results suggest that Moldel4Tune, as one of the first of its kind, performs significantly better than random guessing in 79%-82% of the cases. Our results not only shed light on the possible future research directions but also offer a practical resolution that can assist practitioners in evaluating the most useful model for configuration tuning. 

**Abstract (ZH)**: 高效调参以改善系统性能（如延迟）： surrogate模型的作用与展望 

---
# SemanticControl: A Training-Free Approach for Handling Loosely Aligned Visual Conditions in ControlNet 

**Title (ZH)**: 语义控制：一种无需训练的方法，用于处理ControlNet中的松散对齐的视觉条件 

**Authors**: Woosung Joung, Daewon Chae, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21938)  

**Abstract**: ControlNet has enabled detailed spatial control in text-to-image diffusion models by incorporating additional visual conditions such as depth or edge maps. However, its effectiveness heavily depends on the availability of visual conditions that are precisely aligned with the generation goal specified by text prompt-a requirement that often fails in practice, especially for uncommon or imaginative scenes. For example, generating an image of a cat cooking in a specific pose may be infeasible due to the lack of suitable visual conditions. In contrast, structurally similar cues can often be found in more common settings-for instance, poses of humans cooking are widely available and can serve as rough visual guides. Unfortunately, existing ControlNet models struggle to use such loosely aligned visual conditions, often resulting in low text fidelity or visual artifacts. To address this limitation, we propose SemanticControl, a training-free method for effectively leveraging misaligned but semantically relevant visual conditions. Our approach adaptively suppresses the influence of the visual condition where it conflicts with the prompt, while strengthening guidance from the text. The key idea is to first run an auxiliary denoising process using a surrogate prompt aligned with the visual condition (e.g., "a human playing guitar" for a human pose condition) to extract informative attention masks, and then utilize these masks during the denoising of the actual target prompt (e.g., cat playing guitar). Experimental results demonstrate that our method improves performance under loosely aligned conditions across various conditions, including depth maps, edge maps, and human skeletons, outperforming existing baselines. Our code is available at this https URL. 

**Abstract (ZH)**: SemanticControl：一种无需训练的利用语义相关但不精确对齐的视觉条件的方法 

---
# Why Chain of Thought Fails in Clinical Text Understanding 

**Title (ZH)**: 临床文本理解中链式思维为何失败 

**Authors**: Jiageng Wu, Kevin Xie, Bowen Gu, Nils Krüger, Kueiyu Joshua Lin, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21933)  

**Abstract**: Large language models (LLMs) are increasingly being applied to clinical care, a domain where both accuracy and transparent reasoning are critical for safe and trustworthy deployment. Chain-of-thought (CoT) prompting, which elicits step-by-step reasoning, has demonstrated improvements in performance and interpretability across a wide range of tasks. However, its effectiveness in clinical contexts remains largely unexplored, particularly in the context of electronic health records (EHRs), the primary source of clinical documentation, which are often lengthy, fragmented, and noisy. In this work, we present the first large-scale systematic study of CoT for clinical text understanding. We assess 95 advanced LLMs on 87 real-world clinical text tasks, covering 9 languages and 8 task types. Contrary to prior findings in other domains, we observe that 86.3\% of models suffer consistent performance degradation in the CoT setting. More capable models remain relatively robust, while weaker ones suffer substantial declines. To better characterize these effects, we perform fine-grained analyses of reasoning length, medical concept alignment, and error profiles, leveraging both LLM-as-a-judge evaluation and clinical expert evaluation. Our results uncover systematic patterns in when and why CoT fails in clinical contexts, which highlight a critical paradox: CoT enhances interpretability but may undermine reliability in clinical text tasks. This work provides an empirical basis for clinical reasoning strategies of LLMs, highlighting the need for transparent and trustworthy approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）在临床护理中的应用日益增多，该领域需要高度准确和透明的推理以确保安全和可信的部署。逐步推理（CoT）提示可以引发逐步推理，在广泛任务中展示了性能和可解释性的提升。然而，其在临床环境中的有效性在很大程度上仍未被探索，特别是在电子健康记录（EHRs）的背景下，EHRs是临床记录的主要来源，往往冗长、碎片化且噪声大。在本工作中，我们进行了第一项大规模系统的CoT在临床文本理解中的研究。我们评估了95个高级LLM在87个真实世界的临床文本任务上的表现，覆盖了9种语言和8种任务类型。与在其他领域的先前发现相反，我们观察到86.3%的模型在CoT设置中表现出一致的性能下降。能力更强的模型相对较为稳健，而较弱的模型则遭受了显著的下降。为了更好地理解这些影响，我们通过LLM作为评审员评估和临床专家评估，进行了精细粒度的推理长度、医疗概念对齐及错误分析。我们的结果揭示了CoT在临床环境中的系统性失败模式，这表明一个关键的悖论：CoT虽提升了可解释性，但可能损害了临床文本任务的可靠性。本研究为LLM的临床推理策略提供了实证基础，强调了需要透明和可信的方法。 

---
# SAGE: Scene Graph-Aware Guidance and Execution for Long-Horizon Manipulation Tasks 

**Title (ZH)**: SAGE: 场景图感知的长期操作任务引导与执行 

**Authors**: Jialiang Li, Wenzheng Wu, Gaojing Zhang, Yifan Han, Wenzhao Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.21928)  

**Abstract**: Successfully solving long-horizon manipulation tasks remains a fundamental challenge. These tasks involve extended action sequences and complex object interactions, presenting a critical gap between high-level symbolic planning and low-level continuous control. To bridge this gap, two essential capabilities are required: robust long-horizon task planning and effective goal-conditioned manipulation. Existing task planning methods, including traditional and LLM-based approaches, often exhibit limited generalization or sparse semantic reasoning. Meanwhile, image-conditioned control methods struggle to adapt to unseen tasks. To tackle these problems, we propose SAGE, a novel framework for Scene Graph-Aware Guidance and Execution in Long-Horizon Manipulation Tasks. SAGE utilizes semantic scene graphs as a structural representation for scene states. A structural scene graph enables bridging task-level semantic reasoning and pixel-level visuo-motor control. This also facilitates the controllable synthesis of accurate, novel sub-goal images. SAGE consists of two key components: (1) a scene graph-based task planner that uses VLMs and LLMs to parse the environment and reason about physically-grounded scene state transition sequences, and (2) a decoupled structural image editing pipeline that controllably converts each target sub-goal graph into a corresponding image through image inpainting and composition. Extensive experiments have demonstrated that SAGE achieves state-of-the-art performance on distinct long-horizon tasks. 

**Abstract (ZH)**: 长_horizon操作任务的成功解决仍然是一个基本挑战：SAGE——场景图aware指导与执行框架 

---
# Generation Properties of Stochastic Interpolation under Finite Training Set 

**Title (ZH)**: 有限训练集下随机插值的生成性质 

**Authors**: Yunchen Li, Shaohui Lin, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21925)  

**Abstract**: This paper investigates the theoretical behavior of generative models under finite training populations. Within the stochastic interpolation generative framework, we derive closed-form expressions for the optimal velocity field and score function when only a finite number of training samples are available. We demonstrate that, under some regularity conditions, the deterministic generative process exactly recovers the training samples, while the stochastic generative process manifests as training samples with added Gaussian noise. Beyond the idealized setting, we consider model estimation errors and introduce formal definitions of underfitting and overfitting specific to generative models. Our theoretical analysis reveals that, in the presence of estimation errors, the stochastic generation process effectively produces convex combinations of training samples corrupted by a mixture of uniform and Gaussian noise. Experiments on generation tasks and downstream tasks such as classification support our theory. 

**Abstract (ZH)**: 本文探讨了在有限训练样本情况下生成模型的理论行为。在随机插值生成框架下，我们推导出了仅使用有限数量训练样本时的最佳速度场和评分函数的闭式表达式。我们证明，在某些正则性条件下，确定性的生成过程能精确恢复训练样本，而随机的生成过程则表现为含有高斯噪声的训练样本。在理想化设定之外，我们考虑了模型估计误差，并提出了针对生成模型的过拟合和欠拟合的正式定义。我们的理论分析表明，在存在估计误差的情况下，随机生成过程有效产生了受到均匀噪声和高斯噪声混合影响的训练样本的凸组合。实验结果在生成任务和分类等下游任务中支持了我们的理论。 

---
# EqDiff-CT: Equivariant Conditional Diffusion model for CT Image Synthesis from CBCT 

**Title (ZH)**: EqDiff-CT: 具有不变性的条件扩散模型在CBCT图像合成中的应用 

**Authors**: Alzahra Altalib, Chunhui Li, Alessandro Perelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.21913)  

**Abstract**: Cone-beam computed tomography (CBCT) is widely used for image-guided radiotherapy (IGRT). It provides real time visualization at low cost and dose. However, photon scattering and beam hindrance cause artifacts in CBCT. These include inaccurate Hounsfield Units (HU), reducing reliability for dose calculation, and adaptive planning. By contrast, computed tomography (CT) offers better image quality and accurate HU calibration but is usually acquired offline and fails to capture intra-treatment anatomical changes. Thus, accurate CBCT-to-CT synthesis is needed to close the imaging-quality gap in adaptive radiotherapy workflows.
To cater to this, we propose a novel diffusion-based conditional generative model, coined EqDiff-CT, to synthesize high-quality CT images from CBCT. EqDiff-CT employs a denoising diffusion probabilistic model (DDPM) to iteratively inject noise and learn latent representations that enable reconstruction of anatomically consistent CT images. A group-equivariant conditional U-Net backbone, implemented with e2cnn steerable layers, enforces rotational equivariance (cyclic C4 symmetry), helping preserve fine structural details while minimizing noise and artifacts.
The system was trained and validated on the SynthRAD2025 dataset, comprising CBCT-CT scans across multiple head-and-neck anatomical sites, and we compared it with advanced methods such as CycleGAN and DDPM. EqDiff-CT provided substantial gains in structural fidelity, HU accuracy and quantitative metrics. Visual findings further confirm the improved recovery, sharper soft tissue boundaries, and realistic bone reconstructions. The findings suggest that the diffusion model has offered a robust and generalizable framework for CBCT improvements. The proposed solution helps in improving the image quality as well as the clinical confidence in the CBCT-guided treatment planning and dose calculations. 

**Abstract (ZH)**: 基于扩散的条件生成模型 EqDiff-CT 从 CBCT 合成高质 CT 图像 

---
# AutoSCORE: Enhancing Automated Scoring with Multi-Agent Large Language Models via Structured Component Recognition 

**Title (ZH)**: AutoSCORE: 基于结构化组件识别的多代理大型语言模型增强自动评分 

**Authors**: Yun Wang, Zhaojun Ding, Xuansheng Wu, Siyue Sun, Ninghao Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21910)  

**Abstract**: Automated scoring plays a crucial role in education by reducing the reliance on human raters, offering scalable and immediate evaluation of student work. While large language models (LLMs) have shown strong potential in this task, their use as end-to-end raters faces challenges such as low accuracy, prompt sensitivity, limited interpretability, and rubric misalignment. These issues hinder the implementation of LLM-based automated scoring in assessment practice. To address the limitations, we propose AutoSCORE, a multi-agent LLM framework enhancing automated scoring via rubric-aligned Structured COmponent REcognition. With two agents, AutoSCORE first extracts rubric-relevant components from student responses and encodes them into a structured representation (i.e., Scoring Rubric Component Extraction Agent), which is then used to assign final scores (i.e., Scoring Agent). This design ensures that model reasoning follows a human-like grading process, enhancing interpretability and robustness. We evaluate AutoSCORE on four benchmark datasets from the ASAP benchmark, using both proprietary and open-source LLMs (GPT-4o, LLaMA-3.1-8B, and LLaMA-3.1-70B). Across diverse tasks and rubrics, AutoSCORE consistently improves scoring accuracy, human-machine agreement (QWK, correlations), and error metrics (MAE, RMSE) compared to single-agent baselines, with particularly strong benefits on complex, multi-dimensional rubrics, and especially large relative gains on smaller LLMs. These results demonstrate that structured component recognition combined with multi-agent design offers a scalable, reliable, and interpretable solution for automated scoring. 

**Abstract (ZH)**: AutoSCORE：基于结构化成分识别的多代理大规模语言模型自动化评分框架 

---
# A Large-Scale Dataset and Citation Intent Classification in Turkish with LLMs 

**Title (ZH)**: 使用大型语言模型在土耳其语中构建数据集并进行引文意图分类 

**Authors**: Kemal Sami Karaca, Bahaeddin Eravcı  

**Link**: [PDF](https://arxiv.org/pdf/2509.21907)  

**Abstract**: Understanding the qualitative intent of citations is essential for a comprehensive assessment of academic research, a task that poses unique challenges for agglutinative languages like Turkish. This paper introduces a systematic methodology and a foundational dataset to address this problem. We first present a new, publicly available dataset of Turkish citation intents, created with a purpose-built annotation tool. We then evaluate the performance of standard In-Context Learning (ICL) with Large Language Models (LLMs), demonstrating that its effectiveness is limited by inconsistent results caused by manually designed prompts. To address this core limitation, we introduce a programmable classification pipeline built on the DSPy framework, which automates prompt optimization systematically. For final classification, we employ a stacked generalization ensemble to aggregate outputs from multiple optimized models, ensuring stable and reliable predictions. This ensemble, with an XGBoost meta-model, achieves a state-of-the-art accuracy of 91.3\%. Ultimately, this study provides the Turkish NLP community and the broader academic circles with a foundational dataset and a robust classification framework paving the way for future qualitative citation studies. 

**Abstract (ZH)**: 理解引文的定性意图对于全面评估学术研究至关重要，这在像土耳其语这样的黏着语中提出了独特挑战。本文介绍了一种系统的方法和基础数据集以解决这一问题。我们首先介绍了一个新的公开可用的土耳其引文意图数据集，该数据集使用专门为注释设计的工具创建。随后，我们评估了标准上下文学习（ICL）与大规模语言模型（LLMs）的性能，显示其效果受限于由手动设计的提示引起的不一致结果。为此，我们引入了基于DSPy框架的可编程分类流水线，该流水线系统地自动化了提示优化。最终分类时，我们采用多层次泛化的集成方法汇集多个优化模型的输出，确保稳定可靠的预测。该集成方案，以XGBoost元模型为基础，实现了91.3%的先进准确率。最终，本研究为土耳其语NLP社区和更广泛的学术界提供了基础数据集和稳健的分类框架，为未来的定性引文研究铺平了道路。 

---
# Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts 

**Title (ZH)**: 弹性MoE：解锁混合专家模型的推理时扩展性 

**Authors**: Naibin Gu, Zhenyu Zhang, Yuchen Feng, Yilong Chen, Peng Fu, Zheng Lin, Shuohuan Wang, Yu Sun, Hua Wu, Weiping Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21892)  

**Abstract**: Mixture-of-Experts (MoE) models typically fix the number of activated experts $k$ at both training and inference. Intuitively, activating more experts at inference $k'$ (where $k'> k$) means engaging a larger set of model parameters for the computation and thus is expected to improve performance. However, contrary to this intuition, we find the scaling range to be so narrow that performance begins to degrade rapidly after only a slight increase in the number of experts. Further investigation reveals that this degradation stems from a lack of learned collaboration among experts. To address this, we introduce Elastic Mixture-of-Experts (EMoE), a novel training framework that enables MoE models to scale the number of activated experts at inference without incurring additional training overhead. By simultaneously training experts to collaborate in diverse combinations and encouraging the router for high-quality selections, EMoE ensures robust performance across computational budgets at inference. We conduct extensive experiments on various MoE settings. Our results show that EMoE significantly expands the effective performance-scaling range, extending it to as much as 2-3$\times$ the training-time $k$, while also pushing the model's peak performance to a higher level. 

**Abstract (ZH)**: 弹性专家混合（EMoE）模型：一种无需额外训练开销即可在推理时扩大激活专家数量的新型训练框架 

---
# You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors 

**Title (ZH)**: 你拿不走虚无：通过系统向量减轻LLMs的提示泄漏 

**Authors**: Bochuan Cao, Changjiang Li, Yuanpu Cao, Yameng Ge, Ting Wang, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21884)  

**Abstract**: Large language models (LLMs) have been widely adopted across various applications, leveraging customized system prompts for diverse tasks. Facing potential system prompt leakage risks, model developers have implemented strategies to prevent leakage, primarily by disabling LLMs from repeating their context when encountering known attack patterns. However, it remains vulnerable to new and unforeseen prompt-leaking techniques. In this paper, we first introduce a simple yet effective prompt leaking attack to reveal such risks. Our attack is capable of extracting system prompts from various LLM-based application, even from SOTA LLM models such as GPT-4o or Claude 3.5 Sonnet. Our findings further inspire us to search for a fundamental solution to the problems by having no system prompt in the context. To this end, we propose SysVec, a novel method that encodes system prompts as internal representation vectors rather than raw text. By doing so, SysVec minimizes the risk of unauthorized disclosure while preserving the LLM's core language capabilities. Remarkably, this approach not only enhances security but also improves the model's general instruction-following abilities. Experimental results demonstrate that SysVec effectively mitigates prompt leakage attacks, preserves the LLM's functional integrity, and helps alleviate the forgetting issue in long-context scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种应用中得到了广泛应用，通过定制系统提示应对多样化的任务。面对可能的系统提示泄露风险，模型开发者采取了策略防止泄露，主要通过在遇到已知攻击模式时禁用LLMs重复其上下文内容。然而，这仍然对新的和未预见的提示泄露技术脆弱。在本文中，我们首先介绍了一种简单而有效的提示泄露攻击以揭示这种风险。我们的攻击可以从各种基于LLM的应用中提取系统提示，甚至从当前最先进的LLM模型如GPT-4o或Claude 3.5 Sonnet中提取。我们的发现进一步启发我们寻求从根本上解决问题的方法，即在上下文中没有系统提示。为此，我们提出了SysVec，一种新颖的方法，将系统提示编码为内部表示向量，而不是原始文本。通过这种方式，SysVec减少了未经授权泄露的风险，同时保留了LLM的核心语言能力。这一方法不仅提高了安全性，还改善了模型的通用指令跟随能力。实验结果表明，SysVec有效缓解了提示泄露攻击，保持了LLM的功能完整性，并在长上下文场景中缓解了遗忘问题。 

---
# Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards 

**Title (ZH)**: 位置：验证性奖励强化学习的隐含成本与测量缺口 

**Authors**: Aaron Tu, Weihao Xuan, Heli Qi, Xu Huang, Qingcheng Zeng, Shayan Talaei, Yijia Xiao, Peng Xia, Xiangru Tang, Yuchen Zhuang, Bing Hu, Hanqun Cao, Wenqi Shi, Tianang Leng, Rui Yang, Yingjian Chen, Ziqi Wang, Irene Li, Nan Liu, Huaxiu Yao, Li Erran Li, Ge Liu, Amin Saberi, Naoto Yokoya, Jure Leskovec, Yejin Choi, Fang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21882)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a practical and scalable approach to enhancing large language models in areas such as math, code, and other structured tasks. Two questions motivate this paper: how much of the reported gains survive under strictly parity-controlled evaluation, and whether RLVR is cost-free or exacts a measurable tax. We argue that progress is real, but gains are often overstated due to three forces - an RLVR tax, evaluation pitfalls, and data contamination. Using a partial-prompt contamination audit and matched-budget reproductions across base and RL models, we show that several headline gaps shrink or vanish under clean, parity-controlled evaluation. We then propose a tax-aware training and evaluation protocol that co-optimizes accuracy, grounding, and calibrated abstention and standardizes budgeting and provenance checks. Applied to recent RLVR setups, this protocol yields more reliable estimates of reasoning gains and, in several cases, revises prior conclusions. Our position is constructive: RLVR is valuable and industry-ready; we advocate keeping its practical benefits while prioritizing reliability, safety, and measurement. 

**Abstract (ZH)**: 验证奖励强化学习（RLVR）：在数学、代码及其他结构化任务领域增强大型语言模型的实用且可扩展的方法。本文探讨了两个问题：在严格对等控制评估下，报告的增益有多大部分得以保留，以及RLVR是否无成本或是否付出可量化的代价。我们认为进展是真实的，但由于三种力量——RLVR税、评估陷阱和数据污染，增益往往被夸大了。通过部分指令污染审计和基模型与RL模型的匹配预算再现，我们在严格的对等控制评估下展示了若干关键差距缩小或消失。我们随后提出了一种考虑税收的训练和评估协议，该协议同时优化精度、扎根和校准后的回避，并标准化预算和出处检查。将该协议应用于最近的RLVR设置，可以获得更多可靠的推理增益估计，在某些情况下修正了先前的结论。我们的立场是建设性的：RLVR是有价值且准备好投入工业应用的；我们提倡保留其实际收益，同时优先考虑可靠性、安全性和量测。 

---
# No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping 

**Title (ZH)**: 不留任一代言：通过熵导向优势塑造在LLM强化学习中利用零方差任一代言 

**Authors**: Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21880)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward - so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR. 

**Abstract (ZH)**: 可验证回报的强化学习（RLVR）是增强大型语言模型（LLMs）推理能力的强大力量框架。然而，当前的方法如GRPO仅依赖于模型对同一输入响应正确性不同的问题，而忽视了所有响应获得相同回报的问题——称为零方差提示。在这项工作中，我们认为这些提示并非无用，事实上，它们可以为策略优化提供有意义的反馈。为此，我们引入了零方差提示下的强化学习（RL-ZVP）这一新颖算法，该算法从零方差提示中提取学习信号。RL-ZVP直接奖励正确性并惩罚错误，即使没有对比响应，也能通过标记级别特性调节反馈以保留有信息性的精细信号。在六项数学推理基准测试中，RL-ZVP在准确性和通过率上分别比GRPO提高了8.61分和7.77分，并且始终优于其他过滤掉零方差提示的基线方法。这些结果突显了在RLVR中从零方差提示中学习的未充分利用的潜力。 

---
# Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization 

**Title (ZH)**: 解锁美的本质：基于相对-绝对策略优化的高级美学推理 

**Authors**: Boyang Liu, Yifan Hu, Senjie Jin, Shihan Dou, Gonglei Shi, Jie Shao, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21871)  

**Abstract**: Multimodal large language models (MLLMs) are well suited to image aesthetic assessment, as they can capture high-level aesthetic features leveraging their cross-modal understanding capacity. However, the scarcity of multimodal aesthetic reasoning data and the inherently subjective nature of aesthetic judgment make it difficult for MLLMs to generate accurate aesthetic judgments with interpretable rationales. To this end, we propose Aes-R1, a comprehensive aesthetic reasoning framework with reinforcement learning (RL). Concretely, Aes-R1 integrates a pipeline, AesCoT, to construct and filter high-quality chain-of-thought aesthetic reasoning data used for cold-start. After teaching the model to generate structured explanations prior to scoring, we then employ the Relative-Absolute Policy Optimization (RAPO), a novel RL algorithm that jointly optimizes absolute score regression and relative ranking order, improving both per-image accuracy and cross-image preference judgments. Aes-R1 enables MLLMs to generate grounded explanations alongside faithful scores, thereby enhancing aesthetic scoring and reasoning in a unified framework. Extensive experiments demonstrate that Aes-R1 improves the backbone's average PLCC/SRCC by 47.9%/34.8%, surpassing state-of-the-art baselines of similar size. More ablation studies validate Aes-R1's robust generalization under limited supervision and in out-of-distribution scenarios. 

**Abstract (ZH)**: 多模态大型语言模型(MLLMs)在图像美学评估中表现出色，因为它们能够通过跨模态理解能力捕捉到高层次的美学特征。然而，多模态美学推理数据的稀缺性和美学判断的主观性使得MLLMs难以生成具有可解释理由的准确美学判断。为此，我们提出了一种基于强化学习(RL)的综合性美学推理框架Aes-R1。具体而言，Aes-R1集成了一个管道AesCoT，用于构建和筛选冷启动阶段所需的高度高质量的链式思维美学推理数据。在教学模型在评分前生成结构化解释之后，我们采用了联合优化绝对评分回归和相对排名顺序的新型RL算法Relative-Absolute Policy Optimization (RAPO)，从而提高单张图像准确性和跨图像偏好判断。Aes-R1使MLLMs能够在统一框架中生成基于事实的解释和忠实评分，从而提高美学评分和推理。广泛实验表明，Aes-R1将骨干模型的平均PLCC/SRCC提高了47.9%/34.8%，超过相似规模的先进基线。更多消融研究验证了Aes-R1在有限监督和分布外场景下的稳健泛化能力。 

---
# Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations 

**Title (ZH)**: 增强低秩适应性与结构化非线性变换 

**Authors**: Guanzhi Deng, Mingyang Liu, Dapeng Wu, Yinqiao Li, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.21870)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning method for large language models. However, its linear nature limits expressiveness. We propose LoRAN, a non-linear extension of LoRA that applies lightweight transformations to the low-rank updates. We further introduce Sinter, a sine-based activation that adds structured perturbations without increasing parameter count. Experiments across summarization and classification tasks show that LoRAN consistently improves over QLoRA. Ablation studies reveal that Sinter outperforms standard activations such as Sigmoid, ReLU, and Tanh, highlighting the importance of activation design in lowrank tuning. 

**Abstract (ZH)**: LoRAN：非线性扩展的LoRA及其在低秩调优中的应用 

---
# Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration 

**Title (ZH)**: 代理图：由新兴多代理协作实现的 principled 长上下文建模 

**Authors**: Taejong Joo, Shu Ishida, Ivan Sosnovik, Bryan Lim, Sahand Rezaei-Shoshtari, Adam Gaier, Robert Giaquinto  

**Link**: [PDF](https://arxiv.org/pdf/2509.21848)  

**Abstract**: As a model-agnostic approach to long context modeling, multi-agent systems can process inputs longer than a large language model's context window without retraining or architectural modifications. However, their performance often heavily relies on hand-crafted multi-agent collaboration strategies and prompt engineering, which limit generalizability. In this work, we introduce a principled framework that formalizes the model-agnostic long context modeling problem as a compression problem, yielding an information-theoretic compression objective. Building on this framework, we propose Graph of Agents (GoA), which dynamically constructs an input-dependent collaboration structure that maximizes this objective. For Llama 3.1 8B and Qwen3 8B across six document question answering benchmarks, GoA improves the average $F_1$ score of retrieval-augmented generation by 5.7\% and a strong multi-agent baseline using a fixed collaboration structure by 16.35\%, respectively. Even with only a 2K context window, GoA surpasses the 128K context window Llama 3.1 8B on LongBench, showing a dramatic increase in effective context length. Our source code is available at this https URL. 

**Abstract (ZH)**: 作为一种模型无关的方法，多智能体系统可以处理远超过大型语言模型上下文窗口长度的输入，而无需重新训练或修改架构，但其性能往往高度依赖于手工设计的多智能体协作策略和提示工程，这限制了其通用性。在这项工作中，我们引入了一个原理性的框架，将模型无关的长上下文建模问题形式化为压缩问题，从而得到信息论压缩目标。基于此框架，我们提出了智能体图（GoA），它动态构建一个输入相关的协作结构，以最大化该目标。在涵盖六种文档问答基准测试的实验中，GoA 分别将 Llama 3.1 8B 和 Qwen3 8B 的检索增强生成的平均 F1 分数提高了 5.7% 和 16.35%，并将固定协作结构的强多智能体基线超越。即使拥有仅 2K 的上下文窗口，GoA 在 LongBench 上也超过了 128K 上下文窗口的 Llama 3.1 8B，显示出有效的上下文长度显著增加。我们的源代码可在此处访问。 

---
# Beyond Johnson-Lindenstrauss: Uniform Bounds for Sketched Bilinear Forms 

**Title (ZH)**: 超越约翰逊-林德斯特朗斯ltrauss：草帽下界与拟合双线性形式的统一边界 

**Authors**: Rohan Deb, Qiaobo Li, Mayank Shrivastava, Arindam Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.21847)  

**Abstract**: Uniform bounds on sketched inner products of vectors or matrices underpin several important computational and statistical results in machine learning and randomized algorithms, including the Johnson-Lindenstrauss (J-L) lemma, the Restricted Isometry Property (RIP), randomized sketching, and approximate linear algebra. However, many modern analyses involve *sketched bilinear forms*, for which existing uniform bounds either do not apply or are not sharp on general sets. In this work, we develop a general framework to analyze such sketched bilinear forms and derive uniform bounds in terms of geometric complexities of the associated sets. Our approach relies on generic chaining and introduces new techniques for handling suprema over pairs of sets. We further extend these results to the setting where the bilinear form involves a sum of $T$ independent sketching matrices and show that the deviation scales as $\sqrt{T}$. This unified analysis recovers known results such as the J-L lemma as special cases, while extending RIP-type guarantees. Additionally, we obtain improved convergence bounds for sketched Federated Learning algorithms where such cross terms arise naturally due to sketched gradient compression, and design sketched variants of bandit algorithms with sharper regret bounds that depend on the geometric complexity of the action and parameter sets, rather than the ambient dimension. 

**Abstract (ZH)**: 统一的草图下双线性形式的上界：机器学习和随机化算法中若干重要计算与统计结果的基础，及其在几何复杂性下的分析 

---
# Can Large Language Models Autoformalize Kinematics? 

**Title (ZH)**: 大型语言模型能否自动形式化运动学？ 

**Authors**: Aditi Kabra, Jonathan Laurent, Sagar Bharadwaj, Ruben Martins, Stefan Mitsch, André Platzer  

**Link**: [PDF](https://arxiv.org/pdf/2509.21840)  

**Abstract**: Autonomous cyber-physical systems like robots and self-driving cars could greatly benefit from using formal methods to reason reliably about their control decisions. However, before a problem can be solved it needs to be stated. This requires writing a formal physics model of the cyber-physical system, which is a complex task that traditionally requires human expertise and becomes a bottleneck.
This paper experimentally studies whether Large Language Models (LLMs) can automate the formalization process. A 20 problem benchmark suite is designed drawing from undergraduate level physics kinematics problems. In each problem, the LLM is provided with a natural language description of the objects' motion and must produce a model in differential game logic (dGL). The model is (1) syntax checked and iteratively refined based on parser feedback, and (2) semantically evaluated by checking whether symbolically executing the dGL formula recovers the solution to the original physics problem. A success rate of 70% (best over 5 samples) is achieved. We analyze failing cases, identifying directions for future improvement. This provides a first quantitative baseline for LLM-based autoformalization from natural language to a hybrid games logic with continuous dynamics. 

**Abstract (ZH)**: 基于大规模语言模型的自然语言到混合动态博弈逻辑的自动化形式化研究 

---
# DiTraj: training-free trajectory control for video diffusion transformer 

**Title (ZH)**: DiTraj: 无需训练的动力学轨迹控制视频扩散变换器 

**Authors**: Cheng Lei, Jiayu Zhang, Yue Ma, Xinyu Wang, Long Chen, Liang Tang, Yiqiang Yan, Fei Su, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21839)  

**Abstract**: Diffusion Transformers (DiT)-based video generation models with 3D full attention exhibit strong generative capabilities. Trajectory control represents a user-friendly task in the field of controllable video generation. However, existing methods either require substantial training resources or are specifically designed for U-Net, do not take advantage of the superior performance of DiT. To address these issues, we propose DiTraj, a simple but effective training-free framework for trajectory control in text-to-video generation, tailored for DiT. Specifically, first, to inject the object's trajectory, we propose foreground-background separation guidance: we use the Large Language Model (LLM) to convert user-provided prompts into foreground and background prompts, which respectively guide the generation of foreground and background regions in the video. Then, we analyze 3D full attention and explore the tight correlation between inter-token attention scores and position embedding. Based on this, we propose inter-frame Spatial-Temporal Decoupled 3D-RoPE (STD-RoPE). By modifying only foreground tokens' position embedding, STD-RoPE eliminates their cross-frame spatial discrepancies, strengthening cross-frame attention among them and thus enhancing trajectory control. Additionally, we achieve 3D-aware trajectory control by regulating the density of position embedding. Extensive experiments demonstrate that our method outperforms previous methods in both video quality and trajectory controllability. 

**Abstract (ZH)**: 基于DiT的具有3D全注意力的视频生成模型展示了强大的生成能力。针对轨迹控制任务，我们提出DiTraj——一种适用于DiT的简单有效的无需训练框架，以实现文本到视频生成的轨迹控制。 

---
# ChaosNexus: A Foundation Model for Universal Chaotic System Forecasting with Multi-scale Representations 

**Title (ZH)**: ChaosNexus: 一种基于多尺度表示的通用混沌系统预测基础模型 

**Authors**: Chang Liu, Bohao Zhao, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21802)  

**Abstract**: Accurately forecasting chaotic systems, prevalent in domains such as weather prediction and fluid dynamics, remains a significant scientific challenge. The inherent sensitivity of these systems to initial conditions, coupled with a scarcity of observational data, severely constrains traditional modeling approaches. Since these models are typically trained for a specific system, they lack the generalization capacity necessary for real-world applications, which demand robust zero-shot or few-shot forecasting on novel or data-limited scenarios. To overcome this generalization barrier, we propose ChaosNexus, a foundation model pre-trained on a diverse corpus of chaotic dynamics. ChaosNexus employs a novel multi-scale architecture named ScaleFormer augmented with Mixture-of-Experts layers, to capture both universal patterns and system-specific behaviors. The model demonstrates state-of-the-art zero-shot generalization across both synthetic and real-world benchmarks. On a large-scale testbed comprising over 9,000 synthetic chaotic systems, it improves the fidelity of long-term attractor statistics by more than 40% compared to the leading baseline. This robust performance extends to real-world applications with exceptional data efficiency. For instance, in 5-day global weather forecasting, ChaosNexus achieves a competitive zero-shot mean error below 1 degree, a result that further improves with few-shot fine-tuning. Moreover, experiments on the scaling behavior of ChaosNexus provide a guiding principle for scientific foundation models: cross-system generalization stems from the diversity of training systems, rather than sheer data volume. 

**Abstract (ZH)**: 准确预测气象预测和流体动力学等领域中普遍存在的混沌系统仍然是一个重大的科学挑战。由于这些系统对初始条件的高度敏感性以及观测数据的稀缺性，传统建模方法受到了严重限制。由于这些模型通常仅针对特定系统进行训练，因此缺乏在实际应用中所需的在新颖数据或数据有限场景下进行零样本或少样本预测的能力。为克服这一泛化障碍，我们提出了一种基于多样化混沌动力学语料库预训练的基座模型ChaosNexus。ChaosNexus采用了一种名为ScaleFormer的新颖多尺度架构并结合了Mixture-of-Experts层，以捕捉通用模式和系统特定行为。该模型在合成和真实世界基准测试中均表现出最先进的零样本泛化能力。在包含超过9,000个合成混沌系统的大型测试平台上，与领先的基线相比，ChaosNexus在长期吸引子统计的准确性上提高了超过40%。这一稳健的性能还扩展到了实际应用中，并且具有出色的数据效率。例如，在5天全球天气预报中，ChaosNexus实现了竞争性的零样本平均误差低于1度，而通过少量样本微调可进一步提高性能。此外，ChaosNexus的缩放行为实验为科学基座模型提供了一条指导原则：跨系统泛化源于训练系统的多样性，而非单纯的数据量。 

---
# Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment 

**Title (ZH)**: 评估并改进奖励模型在LLM对齐中文化意识的提升 

**Authors**: Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21798)  

**Abstract**: Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets. To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains. Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks. Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding. To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling. 

**Abstract (ZH)**: 基于文化意识的奖励模型基准（CARB）：推动大型语言模型的全球对齐 

---
# FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning 

**Title (ZH)**: FastGRPO：通过 Awareness-aware 推测解码和在线草图学习加速策略优化 

**Authors**: Yizhou Zhang, Ning Lv, Teng Wang, Jisheng Dang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21792)  

**Abstract**: Group relative policy optimization (GRPO) has demonstrated significant potential in improving the reasoning capabilities of large language models (LLMs) via reinforcement learning. However, its practical deployment is impeded by an excessively slow training process, primarily attributed to the computationally intensive autoregressive generation of multiple responses per query, which makes the generation phase the primary performance bottleneck. Although speculative decoding presents a promising direction for acceleration, its direct application in GRPO achieves limited speedup under high-concurrency training conditions. To overcome this limitation, we propose a concurrency-aware speculative decoding framework that dynamically adjusts the drafting and verification strategy according to real-time concurrency levels, thereby maximizing the acceleration of the generation process. Furthermore, to address performance degradation arising from distributional drift between the evolving target model and the fixed draft model during training, we introduce an online draft learning mechanism that enables the draft model to continuously adapt using feedback signals from the target model. Experimental results across multiple mathematical reasoning datasets and models demonstrate that the proposed method achieves end-to-end speedups of 2.35x to 2.72x, significantly surpassing baseline approaches in efficiency. The code is available at this https URL. 

**Abstract (ZH)**: Group相对策略优化（GRPO）通过强化学习在提高大型语言模型（LLMs）的推理能力方面展现了显著潜力。然而，其实际部署受到训练过程异常缓慢的阻碍，主要原因在于每个查询生成多个响应的计算密集型自回归生成，使得生成阶段成为主要的性能瓶颈。尽管预测解码为加速提供了有前景的方向，但在高并发训练条件下，其直接应用仅能实现有限的加速效果。为克服这一限制，我们提出了一种意识并发的预测解码框架，该框架根据实时的并发水平动态调整起草和验证策略，从而最大化生成过程的加速效果。此外，为解决训练过程中目标模型和固定草稿模型之间分布漂移导致的性能下降问题，我们引入了一种在线草稿学习机制，使草稿模型能够通过来自目标模型的反馈信号不断自我适应。在多个数学推理数据集和模型上的实验结果表明，所提出的方法实现了2.35倍至2.72倍的端到端速度提升，显著优于基线方法。相关代码可在以下链接获取。 

---
# Unbiased Binning: Fairness-aware Attribute Representation 

**Title (ZH)**: 无偏分箱：公平感知属性表示 

**Authors**: Abolfazl Asudeh, Zeinab, Asoodeh, Bita Asoodeh, Omid Asudeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.21785)  

**Abstract**: Discretizing raw features into bucketized attribute representations is a popular step before sharing a dataset. It is, however, evident that this step can cause significant bias in data and amplify unfairness in downstream tasks.
In this paper, we address this issue by introducing the unbiased binning problem that, given an attribute to bucketize, finds its closest discretization to equal-size binning that satisfies group parity across different buckets. Defining a small set of boundary candidates, we prove that unbiased binning must select its boundaries from this set. We then develop an efficient dynamic programming algorithm on top of the boundary candidates to solve the unbiased binning problem.
Finding an unbiased binning may sometimes result in a high price of fairness, or it may not even exist, especially when group values follow different distributions. Considering that a small bias in the group ratios may be tolerable in such settings, we introduce the epsilon-biased binning problem that bounds the group disparities across buckets to a small value epsilon. We first develop a dynamic programming solution, DP, that finds the optimal binning in quadratic time. The DP algorithm, while polynomial, does not scale to very large settings. Therefore, we propose a practically scalable algorithm, based on local search (LS), for epsilon-biased binning. The key component of the LS algorithm is a divide-and-conquer (D&C) algorithm that finds a near-optimal solution for the problem in near-linear time. We prove that D&C finds a valid solution for the problem unless none exists. The LS algorithm then initiates a local search, using the D&C solution as the upper bound, to find the optimal solution. 

**Abstract (ZH)**: 无偏分箱问题：寻找满足群体平等的最优分箱方法 

---
# Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction 

**Title (ZH)**: 超越结构：基于伪粒子射线衍射的Invariant晶体属性预测 

**Authors**: Bin Cao, Yang Liu, Longhan Zhang, Yifan Wu, Zhixun Li, Yuyu Luo, Hong Cheng, Yang Ren, Tong-Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21778)  

**Abstract**: Crystal property prediction, governed by quantum mechanical principles, is computationally prohibitive to solve exactly for large many-body systems using traditional density functional theory. While machine learning models have emerged as efficient approximations for large-scale applications, their performance is strongly influenced by the choice of atomic representation. Although modern graph-based approaches have progressively incorporated more structural information, they often fail to capture long-term atomic interactions due to finite receptive fields and local encoding schemes. This limitation leads to distinct crystals being mapped to identical representations, hindering accurate property prediction. To address this, we introduce PRDNet that leverages unique reciprocal-space diffraction besides graph representations. To enhance sensitivity to elemental and environmental variations, we employ a data-driven pseudo-particle to generate a synthetic diffraction pattern. PRDNet ensures full invariance to crystallographic symmetries. Extensive experiments are conducted on Materials Project, JARVIS-DFT, and MatBench, demonstrating that the proposed model achieves state-of-the-art performance. 

**Abstract (ZH)**: 基于量子力学原理的晶体性质预测对于大规模系统来说使用传统密度泛函理论进行精确计算是计算上不可行的。虽然机器学习模型已成为大规模应用的有效近似方法，但其性能强烈依赖于原子表示的选择。尽管现代基于图的方法逐渐引入了更多的结构信息，但由于有限的感受野和局部编码方案，它们往往无法捕捉长期的原子相互作用。这一限制导致不同的晶体被映射到相同的表示，阻碍了准确的性质预测。为解决这一问题，我们引入了PRDNet，它结合了独特的倒易空间衍射以及图表示。为增强对元素和环境变化的敏感性，我们采用数据驱动的伪粒子生成合成衍射图。PRDNet 确保了对晶体学对称性的完全不变性。在广泛的实验中，我们在Materials Project、JARVIS-DFT 和 MatBench 上进行实验，证明所提出的模型达到了最先进的性能。 

---
# Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models 

**Title (ZH)**: 后门归因：阐明和控制语言模型中的后门攻击 

**Authors**: Miao Yu, Zhenhong Zhou, Moayad Aloqaily, Kun Wang, Biwei Huang, Stephen Wang, Yueming Jin, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21761)  

**Abstract**: Fine-tuned Large Language Models (LLMs) are vulnerable to backdoor attacks through data poisoning, yet the internal mechanisms governing these attacks remain a black box. Previous research on interpretability for LLM safety tends to focus on alignment, jailbreak, and hallucination, but overlooks backdoor mechanisms, making it difficult to understand and fully eliminate the backdoor threat. In this paper, aiming to bridge this gap, we explore the interpretable mechanisms of LLM backdoors through Backdoor Attribution (BkdAttr), a tripartite causal analysis framework. We first introduce the Backdoor Probe that proves the existence of learnable backdoor features encoded within the representations. Building on this insight, we further develop Backdoor Attention Head Attribution (BAHA), efficiently pinpointing the specific attention heads responsible for processing these features. Our primary experiments reveals these heads are relatively sparse; ablating a minimal \textbf{$\sim$ 3%} of total heads is sufficient to reduce the Attack Success Rate (ASR) by \textbf{over 90%}. More importantly, we further employ these findings to construct the Backdoor Vector derived from these attributed heads as a master controller for the backdoor. Through only \textbf{1-point} intervention on \textbf{single} representation, the vector can either boost ASR up to \textbf{$\sim$ 100% ($\uparrow$)} on clean inputs, or completely neutralize backdoor, suppressing ASR down to \textbf{$\sim$ 0% ($\downarrow$)} on triggered inputs. In conclusion, our work pioneers the exploration of mechanistic interpretability in LLM backdoors, demonstrating a powerful method for backdoor control and revealing actionable insights for the community. 

**Abstract (ZH)**: 细调的大语言模型（LLMs）通过数据注入攻击具有脆弱性，但其内部机制仍是一个黑盒。 previous research on interpretability for LLM safety tends to focus on alignment, jailbreak, and hallucination, but overlooks backdoor mechanisms, making it difficult to understand and fully eliminate the backdoor threat. 为此，本文通过Backdoor Attribution（BkdAttr）三部分因果分析框架探索LLM后门的可解释机制。我们首先介绍Backdoor Probe，证明了可学习的后门特征被编码在表示之中。基于这一洞察，我们进一步开发了Backdoor Attention Head Attribution（BAHA），高效地确定处理这些特征的具体注意力头。我们的主要实验证明这些头相对稀疏；删除总头数的最小约3%就足以将攻击成功率为ASR降低超过90%。更重要的是，我们进一步利用这些发现构造出由这些归因头导出的Backdoor向量作为后门的主控制器。通过仅在单个表示上进行一点干预，向量可以在干净输入上将ASR提高到约100%（↑），或完全消除后门，在触发输入上将ASR降低到约0%（↓）。总之，我们的工作开创了LLM后门机制可解释性的研究，展示了后门控制的有力方法，并为社区提供了可操作性的见解。 

---
# SubZeroCore: A Submodular Approach with Zero Training for Coreset Selection 

**Title (ZH)**: SubZeroCore: 一种无需训练的子模态方法用于核心样本选择 

**Authors**: Brian B. Moser, Tobias C. Nauen, Arundhati S. Shanbhag, Federico Raue, Stanislav Frolov, Joachim Folz, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2509.21748)  

**Abstract**: The goal of coreset selection is to identify representative subsets of datasets for efficient model training. Yet, existing approaches paradoxically require expensive training-based signals, e.g., gradients, decision boundary estimates or forgetting counts, computed over the entire dataset prior to pruning, which undermines their very purpose by requiring training on samples they aim to avoid. We introduce SubZeroCore, a novel, training-free coreset selection method that integrates submodular coverage and density into a single, unified objective. To achieve this, we introduce a sampling strategy based on a closed-form solution to optimally balance these objectives, guided by a single hyperparameter that explicitly controls the desired coverage for local density measures. Despite no training, extensive evaluations show that SubZeroCore matches training-based baselines and significantly outperforms them at high pruning rates, while dramatically reducing computational overhead. SubZeroCore also demonstrates superior robustness to label noise, highlighting its practical effectiveness and scalability for real-world scenarios. 

**Abstract (ZH)**: 基于子模覆盖和密度的训练免费核集选择方法SubZeroCore 

---
# HyperCore: Coreset Selection under Noise via Hypersphere Models 

**Title (ZH)**: HyperCore: 基于超球体模型下的噪声环境下核心子集选择 

**Authors**: Brian B. Moser, Arundhati S. Shanbhag, Tobias C. Nauen, Stanislav Frolov, Federico Raue, Joachim Folz, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2509.21746)  

**Abstract**: The goal of coreset selection methods is to identify representative subsets of datasets for efficient model training. Yet, existing methods often ignore the possibility of annotation errors and require fixed pruning ratios, making them impractical in real-world settings. We present HyperCore, a robust and adaptive coreset selection framework designed explicitly for noisy environments. HyperCore leverages lightweight hypersphere models learned per class, embedding in-class samples close to a hypersphere center while naturally segregating out-of-class samples based on their distance. By using Youden's J statistic, HyperCore can adaptively select pruning thresholds, enabling automatic, noise-aware data pruning without hyperparameter tuning. Our experiments reveal that HyperCore consistently surpasses state-of-the-art coreset selection methods, especially under noisy and low-data regimes. HyperCore effectively discards mislabeled and ambiguous points, yielding compact yet highly informative subsets suitable for scalable and noise-free learning. 

**Abstract (ZH)**: 基于噪声环境的稳健自适应核集选择框架HyperCore 

---
# Brain PathoGraph Learning 

**Title (ZH)**: 脑病理图学习 

**Authors**: Ciyuan Peng, Nguyen Linh Dan Le, Shan Jin, Dexuan Ding, Shuo Yu, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.21742)  

**Abstract**: Brain graph learning has demonstrated significant achievements in the fields of neuroscience and artificial intelligence. However, existing methods struggle to selectively learn disease-related knowledge, leading to heavy parameters and computational costs. This challenge diminishes their efficiency, as well as limits their practicality for real-world clinical applications. To this end, we propose a lightweight Brain PathoGraph Learning (BrainPoG) model that enables efficient brain graph learning by pathological pattern filtering and pathological feature distillation. Specifically, BrainPoG first contains a filter to extract the pathological pattern formulated by highly disease-relevant subgraphs, achieving graph pruning and lesion localization. A PathoGraph is therefore constructed by dropping less disease-relevant subgraphs from the whole brain graph. Afterwards, a pathological feature distillation module is designed to reduce disease-irrelevant noise features and enhance pathological features of each node in the PathoGraph. BrainPoG can exclusively learn informative disease-related knowledge while avoiding less relevant information, achieving efficient brain graph learning. Extensive experiments on four benchmark datasets demonstrate that BrainPoG exhibits superiority in both model performance and computational efficiency across various brain disease detection tasks. 

**Abstract (ZH)**: 轻量级脑病理图学习（BrainPoG）模型：通过病理模式过滤和病理特征 distilled 的高效脑图学习 

---
# Self-Speculative Biased Decoding for Faster Live Translation 

**Title (ZH)**: 自我推测导向解码以实现更快的实时翻译 

**Authors**: Linxiao Zeng, Haoyun Deng, Kangyuan Shu, Shizhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21740)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated impressive capabilities in various text generation tasks. However, it remains challenging to use them off-the-shelf in streaming applications (such as live translation), where the output must continually update as the input context expands, while still maintaining a reasonable computational cost to meet the latency requirement.
In this work, we reexamine the re-translation approach to simultaneous translation and propose Self-Speculative Biased Decoding, a novel inference paradigm designed to avoid repeatedly generating output from scratch for a consistently growing input stream. We propose using the most recent output as a draft for the current growing input context. During the verification stage, the output will be biased towards the draft token for a higher draft acceptance rate. This strategy not only minimizes flickering that might distract users but also leads to higher speedups. Conventional decoding may take charge from the point of divergence after draft verification and continue until the end condition is met.
Unlike existing speculative decoding strategies, our approach eliminates the need for draft computations, making it a model-agnostic and plug-and-play solution for accelerating latency-sensitive streaming applications. Experimental results on simultaneous text-to-text re-translation demonstrate that our approach achieves up to 1.7x speedup compared to conventional auto-regressive re-translation without compromising quality. Additionally, it significantly reduces flickering by 80% by incorporating the display-only mask-k technique. 

**Abstract (ZH)**: 大语言模型（LLMs）在各种文本生成任务中已经展示了令人印象深刻的性能。然而，在流式应用（如实时翻译）中，继续挑战在于如何在输入上下文不断扩大的同时，不断更新输出，同时保持合理的计算成本以满足延迟要求。
在这项工作中，我们重新审视了实时翻译中的重新翻译方法，并提出了自我推测偏向解码（Self-Speculative Biased Decoding），这是一种全新的推理范式，旨在避免为不断增长的输入流从头重新生成输出。我们建议使用最新的输出作为当前增长输入上下文的草稿。在验证阶段，输出将偏向草稿令牌，以提高草稿的接受率。这一策略不仅最大限度地减少了可能导致用户分心的闪烁现象，还提高了加速效果。传统解码可能在草稿验证后从分歧点接管，并继续运行直到满足结束条件。
与现有的推测性解码策略不同，我们的方法消除了草稿计算的需求，使其成为一个模型无关的即插即用解决方案，适用于加速延迟敏感的流式应用。针对同时进行文本到文本重新翻译的实验结果表明，我们的方法在不牺牲质量的情况下，相比传统的自回归重新翻译可实现高达1.7倍的加速，并且通过引入只显示掩码技术，闪烁现象显著减少80%。 

---
# LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation 

**Title (ZH)**: LFA-Net：一种用于视网膜血管分割的轻量网络与LiteFusion注意力机制 

**Authors**: Mehwish Mehmood, Ivor Spence, Muhammad Fahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21738)  

**Abstract**: Lightweight retinal vessel segmentation is important for the early diagnosis of vision-threatening and systemic diseases, especially in a real-world clinical environment with limited computational resources. Although segmentation methods based on deep learning are improving, existing models are still facing challenges of small vessel segmentation and high computational costs. To address these challenges, we proposed a new vascular segmentation network, LFA-Net, which incorporates a newly designed attention module, LiteFusion-Attention. This attention module incorporates residual learning connections, Vision Mamba-inspired dynamics, and modulation-based attention, enabling the model to capture local and global context efficiently and in a lightweight manner. LFA-Net offers high performance with 0.11 million parameters, 0.42 MB memory size, and 4.46 GFLOPs, which make it ideal for resource-constrained environments. We validated our proposed model on DRIVE, STARE, and CHASE_DB with outstanding performance in terms of dice scores of 83.28, 87.44, and 84.50% and Jaccard indices of 72.85, 79.31, and 74.70%, respectively. The code of LFA-Net is available online this https URL. 

**Abstract (ZH)**: 轻量级视网膜血管分割对于早期诊断致盲性和全身性疾病至关重要，尤其是在资源受限的临床环境中。虽然基于深度学习的分割方法正在改进，但现有模型仍在小血管分割和高计算成本方面面临挑战。为应对这些挑战，我们提出了一种新的血管分割网络LFA-Net，该网络结合了一种新设计的注意力模块LiteFusion-Attention。该注意力模块采用了残差学习连接、Vision Mamba启发的动力学以及基于调制的注意力机制，使模型能够高效且轻量地捕捉局部和全局上下文。LFA-Net具有11万参数、42 KB内存大小和4.46 GFLOPs，使其适合资源受限的环境。我们在DRIVE、STARE和CHASE_DB上验证了我们提出的模型，性能指标分别为Dice分数83.28%、87.44%和84.50%及Jaccard指数72.85%、79.31%和74.70%。LFA-Net的代码已在线发布于此[https://]。 

---
# POLO: Preference-Guided Multi-Turn Reinforcement Learning for Lead Optimization 

**Title (ZH)**: POLO: 基于偏好的大循环强化学习lead优化 

**Authors**: Ziqing Wang, Yibo Wen, William Pattie, Xiao Luo, Weimin Wu, Jerry Yao-Chieh Hu, Abhishek Pandey, Han Liu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.21737)  

**Abstract**: Lead optimization in drug discovery requires efficiently navigating vast chemical space through iterative cycles to enhance molecular properties while preserving structural similarity to the original lead compound. Despite recent advances, traditional optimization methods struggle with sample efficiency-achieving good optimization performance with limited oracle evaluations. Large Language Models (LLMs) provide a promising approach through their in-context learning and instruction following capabilities, which align naturally with these iterative processes. However, existing LLM-based methods fail to leverage this strength, treating each optimization step independently. To address this, we present POLO (Preference-guided multi-turn Optimization for Lead Optimization), which enables LLMs to learn from complete optimization trajectories rather than isolated steps. At its core, POLO introduces Preference-Guided Policy Optimization (PGPO), a novel reinforcement learning algorithm that extracts learning signals at two complementary levels: trajectory-level optimization reinforces successful strategies, while turn-level preference learning provides dense comparative feedback by ranking intermediate molecules within each trajectory. Through this dual-level learning from intermediate evaluation, POLO achieves superior sample efficiency by fully exploiting each costly oracle call. Extensive experiments demonstrate that POLO achieves 84% average success rate on single-property tasks (2.3x better than baselines) and 50% on multi-property tasks using only 500 oracle evaluations, significantly advancing the state-of-the-art in sample-efficient molecular optimization. 

**Abstract (ZH)**: 药物发现中的先导化合物优化需要通过迭代循环高效地导航庞大的化学空间，以提升分子性质同时保持结构相似性。尽管近期有所进展，传统优化方法在样本效率方面仍然面临挑战，即在有限的oracle评估下实现良好的优化性能。大型语言模型（LLMs）提供了有前景的方法，通过其上下文学习和指令遵循能力，自然地与这些迭代过程相契合。然而，现有的基于LLM的方法未能充分利用这一优势，将每个优化步骤独立处理。为解决这一问题，我们提出了一种POLO（Preference-guided multi-turn Optimization for Lead Optimization）方法，使LLM能够从完整的优化轨迹中学习，而不是孤立地处理每个步骤。POLO的核心是引入了一种新颖的强化学习算法——偏好引导的策略优化（PGPO），该算法在两个互补的层面提取学习信号：轨迹层面的优化强化成功策略，而回合层面的偏好学习通过在每次轨迹内对中间分子进行排名，提供密集的对比反馈。通过这一双层面从中间评估的学习，POLO实现了更高的样本效率，充分利用了每次昂贵的oracle调用。大量实验表明，POLO在单一性质任务上实现了84%的平均成功率（比基线高出2.3倍），在多性质任务上仅使用500次oracle评估就实现了50%的成功率，显著推动了在样本高效分子优化方面的最先进技术。 

---
# Uncovering Alzheimer's Disease Progression via SDE-based Spatio-Temporal Graph Deep Learning on Longitudinal Brain Networks 

**Title (ZH)**: 基于SDE驱动的空间-时间图深度学习揭示阿尔茨海默病进展 

**Authors**: Houliang Zhou, Rong Zhou, Yangying Liu, Kanhao Zhao, Li Shen, Brian Y. Chen, Yu Zhang, Lifang He, Alzheimer's Disease Neuroimaging Initiative  

**Link**: [PDF](https://arxiv.org/pdf/2509.21735)  

**Abstract**: Identifying objective neuroimaging biomarkers to forecast Alzheimer's disease (AD) progression is crucial for timely intervention. However, this task remains challenging due to the complex dysfunctions in the spatio-temporal characteristics of underlying brain networks, which are often overlooked by existing methods. To address these limitations, we develop an interpretable spatio-temporal graph neural network framework to predict future AD progression, leveraging dual Stochastic Differential Equations (SDEs) to model the irregularly-sampled longitudinal functional magnetic resonance imaging (fMRI) data. We validate our approach on two independent cohorts, including the Open Access Series of Imaging Studies (OASIS-3) and the Alzheimer's Disease Neuroimaging Initiative (ADNI). Our framework effectively learns sparse regional and connective importance probabilities, enabling the identification of key brain circuit abnormalities associated with disease progression. Notably, we detect the parahippocampal cortex, prefrontal cortex, and parietal lobule as salient regions, with significant disruptions in the ventral attention, dorsal attention, and default mode networks. These abnormalities correlate strongly with longitudinal AD-related clinical symptoms. Moreover, our interpretability strategy reveals both established and novel neural systems-level and sex-specific biomarkers, offering new insights into the neurobiological mechanisms underlying AD progression. Our findings highlight the potential of spatio-temporal graph-based learning for early, individualized prediction of AD progression, even in the context of irregularly-sampled longitudinal imaging data. 

**Abstract (ZH)**: 基于时空图神经网络的客观神经成像生物标志物识别以预测阿尔茨海默病进展 

---
# UISim: An Interactive Image-Based UI Simulator for Dynamic Mobile Environments 

**Title (ZH)**: UISim：一种用于动态移动环境的交互式图像基UI模拟器 

**Authors**: Jiannan Xiang, Yun Zhu, Lei Shu, Maria Wang, Lijun Yu, Gabriel Barcik, James Lyon, Srinivas Sunkara, Jindong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21733)  

**Abstract**: Developing and testing user interfaces (UIs) and training AI agents to interact with them are challenging due to the dynamic and diverse nature of real-world mobile environments. Existing methods often rely on cumbersome physical devices or limited static analysis of screenshots, which hinders scalable testing and the development of intelligent UI agents. We introduce UISim, a novel image-based UI simulator that offers a dynamic and interactive platform for exploring mobile phone environments purely from screen images. Our system employs a two-stage method: given an initial phone screen image and a user action, it first predicts the abstract layout of the next UI state, then synthesizes a new, visually consistent image based on this predicted layout. This approach enables the realistic simulation of UI transitions. UISim provides immediate practical benefits for UI testing, rapid prototyping, and synthetic data generation. Furthermore, its interactive capabilities pave the way for advanced applications, such as UI navigation task planning for AI agents. Our experimental results show that UISim outperforms end-to-end UI generation baselines in generating realistic and coherent subsequent UI states, highlighting its fidelity and potential to streamline UI development and enhance AI agent training. 

**Abstract (ZH)**: 基于图像的用户界面模拟器（UISim）：探索移动环境的新范式 

---
# Developing Strategies to Increase Capacity in AI Education 

**Title (ZH)**: 开发策略以提高人工智能教育容量 

**Authors**: Noah Q. Cowit, Sri Yash Tadimalla, Stephanie T. Jones, Mary Lou Maher, Tracy Camp, Enrico Pontelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.21713)  

**Abstract**: Many institutions are currently grappling with teaching artificial intelligence (AI) in the face of growing demand and relevance in our world. The Computing Research Association (CRA) has conducted 32 moderated virtual roundtable discussions of 202 experts committed to improving AI education. These discussions slot into four focus areas: AI Knowledge Areas and Pedagogy, Infrastructure Challenges in AI Education, Strategies to Increase Capacity in AI Education, and AI Education for All. Roundtables were organized around institution type to consider the particular goals and resources of different AI education environments. We identified the following high-level community needs to increase capacity in AI education. A significant digital divide creates major infrastructure hurdles, especially for smaller and under-resourced institutions. These challenges manifest as a shortage of faculty with AI expertise, who also face limited time for reskilling; a lack of computational infrastructure for students and faculty to develop and test AI models; and insufficient institutional technical support. Compounding these issues is the large burden associated with updating curricula and creating new programs. To address the faculty gap, accessible and continuous professional development is crucial for faculty to learn about AI and its ethical dimensions. This support is particularly needed for under-resourced institutions and must extend to faculty both within and outside of computing programs to ensure all students have access to AI education. We have compiled and organized a list of resources that our participant experts mentioned throughout this study. These resources contribute to a frequent request heard during the roundtables: a central repository of AI education resources for institutions to freely use across higher education. 

**Abstract (ZH)**: 当前，许多机构正面临着在日益增长的需求和重要性面前讲授人工智能（AI）的挑战。美国计算机研究协会（CRA）组织了32次 moderated虚拟圆桌讨论，涉及202位致力于改进AI教育的专家。这些讨论集中在四个重点领域：AI知识领域和教学方法、AI教育中的基础设施挑战、增加AI教育容量的策略、以及面向所有人的AI教育。根据机构类型组织圆桌讨论，以考虑不同AI教育环境的特定目标和资源。我们识别出了以下提高AI教育容量的高层次社区需求。显著的数字鸿沟造成了重大的基础设施障碍，尤其是对小型和资源不足的机构而言。这些挑战表现为缺乏具有AI专长的教师，他们面对重新培训的时间有限；缺少供学生和教师开发和测试AI模型的计算基础设施；以及机构技术支持不足。随着这些问题的加剧，更新课程和创建新计划所需的工作量巨大。为解决师资缺口，为教师提供可访问且持续的专业发展是关键，使他们了解AI及其伦理维度。这尤其对资源不足的机构很重要，并且必须覆盖计算项目内外的教师，以确保所有学生都能获得AI教育。我们汇总并组织了在整个研究过程中我们的专家提及的一系列资源。这些资源满足了圆桌讨论中一个频繁听到的要求：为高等教育机构提供一个中央AI教育资源库，供其自由使用。 

---
# Not My Agent, Not My Boundary? Elicitation of Personal Privacy Boundaries in AI-Delegated Information Sharing 

**Title (ZH)**: 不是我的代理，不是我的边界？人工智能代理下的个人信息边界 elicitation 

**Authors**: Bingcan Guo, Eryue Xu, Zhiping Zhang, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21712)  

**Abstract**: Aligning AI systems with human privacy preferences requires understanding individuals' nuanced disclosure behaviors beyond general norms. Yet eliciting such boundaries remains challenging due to the context-dependent nature of privacy decisions and the complex trade-offs involved. We present an AI-powered elicitation approach that probes individuals' privacy boundaries through a discriminative task. We conducted a between-subjects study that systematically varied communication roles and delegation conditions, resulting in 1,681 boundary specifications from 169 participants for 61 scenarios. We examined how these contextual factors and individual differences influence the boundary specification. Quantitative results show that communication roles influence individuals' acceptance of detailed and identifiable disclosure, AI delegation and individuals' need for privacy heighten sensitivity to disclosed identifiers, and AI delegation results in less consensus across individuals. Our findings highlight the importance of situating privacy preference elicitation within real-world data flows. We advocate using nuanced privacy boundaries as an alignment goal for future AI systems. 

**Abstract (ZH)**: 将AI系统与人类隐私偏好相一致需要超越一般规范理解个体复杂的披露行为。然而，由于隐私决策具有情境依赖性以及其中复杂的权衡，获取这些边界仍然具有挑战性。我们提出了一种基于AI的提取方法，通过区分性任务探究个体的隐私边界。我们进行了一项被试间研究，系统地变化了沟通角色和委托条件，共获得了169名参与者在61种情景下的1,681个边界规范。我们探讨了这些情境因素和个体差异如何影响边界规范。定量结果显示，沟通角色影响个体对详细和可识别披露的接受度，AI委托增加了对披露标识符的敏感性，AI委托导致个体之间的共识较低。我们的研究结果强调，在实际数据流中定位隐私偏好提取的重要性，并建议将细腻的隐私边界作为未来AI系统的对齐目标。 

---
# Optimizing the non-Clifford-count in unitary synthesis using Reinforcement Learning 

**Title (ZH)**: 使用强化学习优化单元ary合成中的非Clifford门数量 

**Authors**: David Kremer, Ali Javadi-Abhari, Priyanka Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2509.21709)  

**Abstract**: An efficient implementation of unitary operators is important in order to practically realize the computational advantages claimed by quantum algorithms over their classical counterparts. In this paper we study the potential of using reinforcement learning (RL) in order to synthesize quantum circuits, while optimizing the T-count and CS-count, of unitaries that are exactly implementable by the Clifford+T and Clifford+CS gate sets, respectively. In general, the complexity of existing algorithms depend exponentially on the number of qubits and the non-Clifford-count of unitaries. We have designed our RL framework to work with channel representation of unitaries, that enables us to perform matrix operations efficiently, using integers only. We have also incorporated pruning heuristics and a canonicalization of operators, in order to reduce the search complexity. As a result, compared to previous works, we are able to implement significantly larger unitaries, in less time, with much better success rate and improvement factor. Our results for Clifford+T synthesis on two qubits achieve close-to-optimal decompositions for up to 100 T gates, 5 times more than previous RL algorithms and to the best of our knowledge, the largest instances achieved with any method to date. Our RL algorithm is able to recover previously-known optimal linear complexity algorithm for T-count-optimal decomposition of 1 qubit unitaries. For 2-qubit Clifford+CS unitaries, our algorithm achieves a linear complexity, something that could only be accomplished by a previous algorithm using $SO(6)$ representation. 

**Abstract (ZH)**: 使用强化学习合成Clifford+T和Clifford+CS门集中可精确实现的量子电路：提高T计数和CS计数的高效实现 

---
# QueryGym: Step-by-Step Interaction with Relational Databases 

**Title (ZH)**: QueryGym: 逐步与关系数据库交互 

**Authors**: Haritha Ananthakrishanan, Harsha Kokel, Kelsey Sikes, Debarun Bhattacharjya, Michael Katz, Shirin Sohrabi, Kavitha Srinivas  

**Link**: [PDF](https://arxiv.org/pdf/2509.21674)  

**Abstract**: We introduce QueryGym, an interactive environment for building, testing, and evaluating LLM-based query planning agents. Existing frameworks often tie agents to specific query language dialects or obscure their reasoning; QueryGym instead requires agents to construct explicit sequences of relational algebra operations, ensuring engine-agnostic evaluation and transparent step-by-step planning. The environment is implemented as a Gymnasium interface that supplies observations -- including schema details, intermediate results, and execution feedback -- and receives actions that represent database exploration (e.g., previewing tables, sampling column values, retrieving unique values) as well as relational algebra operations (e.g., filter, project, join). We detail the motivation and the design of the environment. In the demo, we showcase the utility of the environment by contrasting it with contemporary LLMs that query databases. QueryGym serves as a practical testbed for research in error remediation, transparency, and reinforcement learning for query generation. For the associated demo, see this https URL. 

**Abstract (ZH)**: QueryGym：一种用于构建、测试和评估基于LLM的查询规划代理的互动环境 

---
# SlotFM: A Motion Foundation Model with Slot Attention for Diverse Downstream Tasks 

**Title (ZH)**: SlotFM：一种具有槽注意机制的运动基础模型及其在多样的下游任务中的应用 

**Authors**: Junyong Park, Oron Levy, Rebecca Adaimi, Asaf Liberman, Gierad Laput, Abdelkareem Bedri  

**Link**: [PDF](https://arxiv.org/pdf/2509.21673)  

**Abstract**: Wearable accelerometers are used for a wide range of applications, such as gesture recognition, gait analysis, and sports monitoring. Yet most existing foundation models focus primarily on classifying common daily activities such as locomotion and exercise, limiting their applicability to the broader range of tasks that rely on other signal characteristics. We present SlotFM, an accelerometer foundation model that generalizes across diverse downstream tasks. SlotFM uses Time-Frequency Slot Attention, an extension of Slot Attention that processes both time and frequency representations of the raw signals. It generates multiple small embeddings (slots), each capturing different signal components, enabling task-specific heads to focus on the most relevant parts of the data. We also introduce two loss regularizers that capture local structure and frequency patterns, which improve reconstruction of fine-grained details and helps the embeddings preserve task-relevant information. We evaluate SlotFM on 16 classification and regression downstream tasks that extend beyond standard human activity recognition. It outperforms existing self-supervised approaches on 13 of these tasks and achieves comparable results to the best performing approaches on the remaining tasks. On average, our method yields a 4.5% performance gain, demonstrating strong generalization for sensing foundation models. 

**Abstract (ZH)**: 可穿戴加速度计在多样下游任务中的通用基础模型：SlotFM 

---
# MORPH: Shape-agnostic PDE Foundation Models 

**Title (ZH)**: MORPH: 形状无关的偏微分方程基础模型 

**Authors**: Mahindra Singh Rautela, Alexander Most, Siddharth Mansingh, Bradley C. Love, Ayan Biswas, Diane Oyen, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2509.21670)  

**Abstract**: We introduce MORPH, a shape-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data dimensionality (1D--3D) at different resolutions, multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorizes full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters (LoRA), MORPH outperforms models trained from scratch in both zero-shot and full-shot generalization. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning. 

**Abstract (ZH)**: MORPH：一种通用自回归基础模型，用于偏微分方程 

---
# DIM: Enforcing Domain-Informed Monotonicity in Deep Neural Networks 

**Title (ZH)**: DIM：在深度神经网络中强制执行基于领域 Informative 的单调性 

**Authors**: Joshua Salim, Jordan Yu, Xilei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21666)  

**Abstract**: While deep learning models excel at predictive tasks, they often overfit due to their complex structure and large number of parameters, causing them to memorize training data, including noise, rather than learn patterns that generalize to new data. To tackle this challenge, this paper proposes a new regularization method, i.e., Enforcing Domain-Informed Monotonicity in Deep Neural Networks (DIM), which maintains domain-informed monotonic relationships in complex deep learning models to further improve predictions. Specifically, our method enforces monotonicity by penalizing violations relative to a linear baseline, effectively encouraging the model to follow expected trends while preserving its predictive power. We formalize this approach through a comprehensive mathematical framework that establishes a linear reference, measures deviations from monotonic behavior, and integrates these measurements into the training objective. We test and validate the proposed methodology using a real-world ridesourcing dataset from Chicago and a synthetically created dataset. Experiments across various neural network architectures show that even modest monotonicity constraints consistently enhance model performance. DIM enhances the predictive performance of deep neural networks by applying domain-informed monotonicity constraints to regularize model behavior and mitigate overfitting 

**Abstract (ZH)**: 在复杂结构的深度学习模型中，尽管深度学习模型在预测任务上表现出色，但由于其复杂的结构和大量的参数，它们往往会过拟合，导致模型记忆训练数据中的噪声而非学习泛化到新数据的模式。为解决这一挑战，本文提出了一个新的正则化方法，即在深度神经网络中强加领域信息单调性的方法（Enforcing Domain-Informed Monotonicity in Deep Neural Networks，简称DIM），该方法通过保持复杂深度学习模型中的领域信息单调关系来进一步提高预测性能。具体而言，该方法通过对线性基线的违反进行惩罚，有效鼓励模型遵循预期趋势同时保留其预测能力。通过一个全面的数学框架，本文形式化了这一方法，该框架建立了线性参考，衡量单调行为的偏差，并将这些测量值整合到训练目标中。本文使用来自芝加哥的真实世界网约车数据集和合成数据集测试和验证了所提出的方法。各种神经网络架构的实验表明，即使是轻微的单调性约束也能一致地提升模型性能。DIM通过在深度神经网络中应用领域信息单调性约束来规整模型行为并缓解过拟合，从而提高预测性能。 

---
# Logic of Hypotheses: from Zero to Full Knowledge in Neurosymbolic Integration 

**Title (ZH)**: 逻辑假设：从零到完全知识的神经符号集成 

**Authors**: Davide Bizzaro, Alessandro Daniele  

**Link**: [PDF](https://arxiv.org/pdf/2509.21663)  

**Abstract**: Neurosymbolic integration (NeSy) blends neural-network learning with symbolic reasoning. The field can be split between methods injecting hand-crafted rules into neural models, and methods inducing symbolic rules from data. We introduce Logic of Hypotheses (LoH), a novel language that unifies these strands, enabling the flexible integration of data-driven rule learning with symbolic priors and expert knowledge. LoH extends propositional logic syntax with a choice operator, which has learnable parameters and selects a subformula from a pool of options. Using fuzzy logic, formulas in LoH can be directly compiled into a differentiable computational graph, so the optimal choices can be learned via backpropagation. This framework subsumes some existing NeSy models, while adding the possibility of arbitrary degrees of knowledge specification. Moreover, the use of Goedel fuzzy logic and the recently developed Goedel trick yields models that can be discretized to hard Boolean-valued functions without any loss in performance. We provide experimental analysis on such models, showing strong results on tabular data and on the Visual Tic-Tac-Toe NeSy task, while producing interpretable decision rules. 

**Abstract (ZH)**: 神经符号整合（NeSy）将神经网络学习与符号推理相结合。该领域可以分为将手工构建规则注入神经模型的方法，以及从数据中诱导符号规则的方法。我们引入了假设逻辑（LoH），这是一种新的语言，能够统一上述两种方法，实现基于数据规则学习与符号先验及专家知识的灵活整合。LoH 扩展了命题逻辑的语法，添加了一个具有可学习参数的选择操作符，可以从一组选项中选择子公式。使用模糊逻辑，LoH 中的公式可以直接编译成可微计算图，从而使最优选择可通过反向传播学习。该框架涵盖了某些现有的 NeSy 模型，同时增加了任意程度的知识指定的可能性。此外，利用哥德尔模糊逻辑和最近发展的哥德尔技巧，可以将模型离散化为硬布尔值函数，而不影响性能。我们对这些模型进行了实验分析，在表格数据和视觉井字博弈 NeSy 任务中取得了显著效果，同时还产生了可解释的决策规则。 

---
# Limitations on Safe, Trusted, Artificial General Intelligence 

**Title (ZH)**: 安全可靠的通用人工智能局限性 

**Authors**: Rina Panigrahy, Vatsal Sharan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21654)  

**Abstract**: Safety, trust and Artificial General Intelligence (AGI) are aspirational goals in artificial intelligence (AI) systems, and there are several informal interpretations of these notions. In this paper, we propose strict, mathematical definitions of safety, trust, and AGI, and demonstrate a fundamental incompatibility between them. We define safety of a system as the property that it never makes any false claims, trust as the assumption that the system is safe, and AGI as the property of an AI system always matching or exceeding human capability. Our core finding is that -- for our formal definitions of these notions -- a safe and trusted AI system cannot be an AGI system: for such a safe, trusted system there are task instances which are easily and provably solvable by a human but not by the system. We note that we consider strict mathematical definitions of safety and trust, and it is possible for real-world deployments to instead rely on alternate, practical interpretations of these notions. We show our results for program verification, planning, and graph reachability. Our proofs draw parallels to Gödel's incompleteness theorems and Turing's proof of the undecidability of the halting problem, and can be regarded as interpretations of Gödel's and Turing's results. 

**Abstract (ZH)**: 安全、信任与通用人工智能：严格定义及其根本不兼容性 

---
# MobiLLM: An Agentic AI Framework for Closed-Loop Threat Mitigation in 6G Open RANs 

**Title (ZH)**: MobiLLM: 6G 开放RAN中具有自主性的AI框架以实现闭环威胁缓解 

**Authors**: Prakhar Sharma, Haohuang Wen, Vinod Yegneswaran, Ashish Gehani, Phillip Porras, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21634)  

**Abstract**: The evolution toward 6G networks is being accelerated by the Open Radio Access Network (O-RAN) paradigm -- an open, interoperable architecture that enables intelligent, modular applications across public telecom and private enterprise domains. While this openness creates unprecedented opportunities for innovation, it also expands the attack surface, demanding resilient, low-cost, and autonomous security solutions. Legacy defenses remain largely reactive, labor-intensive, and inadequate for the scale and complexity of next-generation systems. Current O-RAN applications focus mainly on network optimization or passive threat detection, with limited capability for closed-loop, automated response.
To address this critical gap, we present an agentic AI framework for fully automated, end-to-end threat mitigation in 6G O-RAN environments. MobiLLM orchestrates security workflows through a modular multi-agent system powered by Large Language Models (LLMs). The framework features a Threat Analysis Agent for real-time data triage, a Threat Classification Agent that uses Retrieval-Augmented Generation (RAG) to map anomalies to specific countermeasures, and a Threat Response Agent that safely operationalizes mitigation actions via O-RAN control interfaces. Grounded in trusted knowledge bases such as the MITRE FiGHT framework and 3GPP specifications, and equipped with robust safety guardrails, MobiLLM provides a blueprint for trustworthy AI-driven network security. Initial evaluations demonstrate that MobiLLM can effectively identify and orchestrate complex mitigation strategies, significantly reducing response latency and showcasing the feasibility of autonomous security operations in 6G. 

**Abstract (ZH)**: 6G O-RAN环境中的自主AI框架：全面自动化端到端威胁缓解 

---
# InvBench: Can LLMs Accelerate Program Verification with Invariant Synthesis? 

**Title (ZH)**: InvBench: 能否通过不变式合成加速程序验证？ 

**Authors**: Anjiang Wei, Tarun Suresh, Tianran Sun, Haoze Wu, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2509.21629)  

**Abstract**: Program verification relies on loop invariants, yet automatically discovering strong invariants remains a long-standing challenge. We introduce a principled framework for evaluating LLMs on invariant synthesis. Our approach uses a verifier-based decision procedure with a formal soundness guarantee and assesses not only correctness but also the speedup that invariants provide in verification. We evaluate 7 state-of-the-art LLMs, and existing LLM-based verifiers against the traditional solver UAutomizer. While LLM-based verifiers represent a promising direction, they do not yet offer a significant advantage over UAutomizer. Model capability also proves critical, as shown by sharp differences in speedups across models, and our benchmark remains an open challenge for current LLMs. Finally, we show that supervised fine-tuning and Best-of-N sampling can improve performance: fine-tuning on 3589 instances raises the percentage of speedup cases for Qwen3-Coder-480B from 8% to 29.2%, and Best-of-N sampling with N=16 improves Claude-sonnet-4 from 8.8% to 22.1%. 

**Abstract (ZH)**: 程序验证依赖于循环不变式的使用，但自动生成强不变式仍是一个长期挑战。我们提出了一种 principled 的框架来评估 LLMs 在不变式合成中的表现。我们的方法采用基于验证器的决策程序，具有形式上的正确性保证，并不仅评估正确性，还评估不变式在验证中提供的加速效果。我们评估了 7 种最先进的 LLMs 和基于 LLM 的验证器与传统求解器 UAutomizer 的性能。尽管基于 LLM 的验证器展示出前景，但它们尚未在性能上显著超越 UAutomizer。模型能力同样至关重要，不同模型间存在显著的加速差异，表明我们的基准仍是对当前 LLMs 的开放挑战。最后，我们展示了监督微调和 Best-of-N 采样的性能改进：在 3589 个实例上进行微调后，Qwen3-Coder-480B 的加速案例比例从 8% 提高到 29.2%，Best-of-N 采样（N=16）后，Claude-sonnet-4 的加速案例比例从 8.8% 提高到 22.1%。 

---
# A Data-driven Typology of Vision Models from Integrated Representational Metrics 

**Title (ZH)**: 基于综合表征度量的数据驱动视觉模型类型划分 

**Authors**: Jialin Wu, Shreya Saha, Yiqing Bo, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.21628)  

**Abstract**: Large vision models differ widely in architecture and training paradigm, yet we lack principled methods to determine which aspects of their representations are shared across families and which reflect distinctive computational strategies. We leverage a suite of representational similarity metrics, each capturing a different facet-geometry, unit tuning, or linear decodability-and assess family separability using multiple complementary measures. Metrics preserving geometry or tuning (e.g., RSA, Soft Matching) yield strong family discrimination, whereas flexible mappings such as Linear Predictivity show weaker separation. These findings indicate that geometry and tuning carry family-specific signatures, while linearly decodable information is more broadly shared. To integrate these complementary facets, we adapt Similarity Network Fusion (SNF), a method inspired by multi-omics integration. SNF achieves substantially sharper family separation than any individual metric and produces robust composite signatures. Clustering of the fused similarity matrix recovers both expected and surprising patterns: supervised ResNets and ViTs form distinct clusters, yet all self-supervised models group together across architectural boundaries. Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders, suggesting convergence between architectural modernization and reconstruction-based training. This biology-inspired framework provides a principled typology of vision models, showing that emergent computational strategies-shaped jointly by architecture and training objective-define representational structure beyond surface design categories. 

**Abstract (ZH)**: 大型视觉模型在架构和训练范式上存在显著差异，但我们缺乏明确的方法来确定它们表示中哪些方面是跨家族共享的，哪些反映了独特的计算策略。我们利用一系列表示相似性度量，每种度量捕捉不同的方面-几何结构、单个单元的调谐或线性可解码性-并通过多种互补的度量评估家族可分性。保留几何结构或调谐的度量（例如，RSA、Soft Matching）能够产生强烈的家庭区分效果，而灵活的映射如线性预测性则显示出较弱的分离。这些发现表明，几何结构和调谐携带家族特有的签名，而线性可解码的信息则更广泛共享。为了整合这些互补的方面，我们改进了启发于多组学整合的相似性网络融合（SNF）方法。SNF在家庭分离的锐度上显著优于任何单一的度量，并生成稳健的综合签名。融合相似性矩阵的聚类既恢复了预期模式，也发现了一些意想不到的模式：监督ResNets和ViTs形成不同的簇，但所有自监督模型跨越架构边界聚集在一起。混合架构（ConvNeXt、Swin）与掩码自编码器聚集在一起，表明架构现代化与基于重建的训练之间存在趋同。这种生物学启发的框架提供了一种系统的视觉模型分类，表明由架构和训练目标共同塑造的新兴计算策略决定表示结构，超越了表面设计分类。 

---
# Guiding Audio Editing with Audio Language Model 

**Title (ZH)**: 指导音频编辑的音频语言模型 

**Authors**: Zitong Lan, Yiduo Hao, Mingmin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21625)  

**Abstract**: Audio editing plays a central role in VR/AR immersion, virtual conferencing, sound design, and other interactive media. However, recent generative audio editing models depend on template-like instruction formats and are restricted to mono-channel audio. These models fail to deal with declarative audio editing, where the user declares what the desired outcome should be, while leaving the details of editing operations to the system. We introduce SmartDJ, a novel framework for stereo audio editing that combines the reasoning capability of audio language models with the generative power of latent diffusion. Given a high-level instruction, SmartDJ decomposes it into a sequence of atomic edit operations, such as adding, removing, or spatially relocating events. These operations are then executed by a diffusion model trained to manipulate stereo audio. To support this, we design a data synthesis pipeline that produces paired examples of high-level instructions, atomic edit operations, and audios before and after each edit operation. Experiments demonstrate that SmartDJ achieves superior perceptual quality, spatial realism, and semantic alignment compared to prior audio editing methods. Demos are available at this https URL. 

**Abstract (ZH)**: 智能混音师：一种结合音频语言模型推理能力和潜在扩散生成力量的立体声音频编辑框架 

---
# OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja's Rule 

**Title (ZH)**: OjaKV：基于上下文感知的在线低秩键值缓存压缩方法及其Oja规则 

**Authors**: Yuxuan Zhu, David H. Yang, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21623)  

**Abstract**: The expanding long-context capabilities of large language models are constrained by a significant memory bottleneck: the key-value (KV) cache required for autoregressive generation. This bottleneck is substantial; for instance, a Llama-3.1-8B model processing a 32K-token prompt at a batch size of 4 requires approximately 16GB for its KV cache, a size exceeding the model's weights. While KV-cache compression via low-rank projection is a promising direction, existing methods rely on a static, offline-learned subspace that performs poorly under data distribution shifts. To overcome these limitations, we introduce OjaKV, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation. First, OjaKV recognizes that not all tokens are equally important for compression; it preserves the crucial first and most recent tokens in full-rank, maintaining high-fidelity anchors for attention. Second, for the vast majority of intermediate tokens, it applies low-rank compression by incrementally adapting the projection basis using Oja's algorithm for online principal component analysis. This adaptation involves a comprehensive update during prompt prefilling and lightweight periodic updates during decoding, ensuring the subspace remains aligned with the evolving context. Crucially, our framework is fully compatible with modern attention modules like FlashAttention. Experiments demonstrate that OjaKV maintains or even improves zero-shot accuracy at high compression ratios. In particular, OjaKV achieves its strongest gains on very long-context benchmarks that require complex reasoning, highlighting the importance of online subspace adaptation in dynamically tracking context shifts. These results establish our hybrid framework as a practical, plug-and-play solution for memory-efficient long-context inference without requiring model fine-tuning. 

**Abstract (ZH)**: 大型语言模型扩展长上下文能力受到显著记忆瓶颈的限制：自回归生成所需的键值（KV）缓存。这种瓶颈是重大的；例如，处理32K tokens提示、批量大小为4的Llama-3.1-8B模型需要约16GB的KV缓存，该大小超过了模型的权重。虽然通过低秩投影压缩KV缓存是一个有潜力的方向，但现有方法依赖于静态的、离线学习的子空间，在数据分布变化时表现不佳。为克服这些局限性，我们引入OjaKV，这是一种将战略混合存储策略与在线子空间自适应相结合的创新框架。首先，OjaKV认识到并非所有token对压缩同等重要；它完整保留了至关重要的第一个和最近的token，维持了高保真的注意力锚点。其次，对于绝大多数中间token，它通过增量地使用Oja算法进行在线主成分分析来应用低秩压缩，调整投影基底。这一自适应过程包括在预填充提示时的全面更新，在解码过程中进行轻量级的周期性更新，确保子空间始终与上下文动态变化保持一致。 crucially，我们的框架完全兼容现代的注意力模块如FlashAttention。实验表明，OjaKV在高压缩比下能够保持或甚至提高零样本准确率。特别是在需要复杂推理的非常长上下文基准测试中，OjaKV获得了最大的收益，突显了在线子空间自适应在动态跟踪上下文变化中的重要性。这些结果确立了我们这种混合框架作为无需模型微调的实用、即插即用的长上下文推理内存高效解决方案。 

---
# LANCE: Low Rank Activation Compression for Efficient On-Device Continual Learning 

**Title (ZH)**: LANCE: 低秩激活压缩以实现高效设备端持续学习 

**Authors**: Marco Paul E. Apolinario, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2509.21617)  

**Abstract**: On-device learning is essential for personalization, privacy, and long-term adaptation in resource-constrained environments. Achieving this requires efficient learning, both fine-tuning existing models and continually acquiring new tasks without catastrophic forgetting. Yet both settings are constrained by high memory cost of storing activations during backpropagation. Existing activation compression methods reduce this cost but relying on repeated low-rank decompositions, introducing computational overhead. Also, such methods have not been explored for continual learning. We propose LANCE (Low-rank Activation Compression), a framework that performs one-shot higher-order Singular Value Decompsoition (SVD) to obtain a reusable low-rank subspace for activation projection. This eliminates repeated decompositions, reducing both memory and computation. Moreover, fixed low-rank subspaces further enable on-device continual learning by allocating tasks to orthogonal subspaces without storing large task-specific matrices. Experiments show that LANCE reduces activation storage up to 250$\times$ while maintaining accuracy comparable to full backpropagation on CIFAR-10/100, Oxford-IIIT Pets, Flowers102, and CUB-200 datasets. On continual learning benchmarks (Split CIFAR-100, Split MiniImageNet, 5-Datasets), it achieves performance competitive with orthogonal gradient projection methods at a fraction of the memory cost. These results position LANCE as a practical and scalable solution for efficient fine-tuning and continual learning on edge devices. 

**Abstract (ZH)**: 设备端学习对于受限资源环境中的个性化、隐私保护和长期适应至关重要。实现这一目标需要高效的在线学习，既包括微调现有模型，也包括不断获取新任务而不发生灾难性遗忘。然而，这二者都受限于反向传播过程中激活存储的高内存成本。现有激活压缩方法虽能降低此成本，但依赖于复现的低秩分解，引入了计算开销。此外，这类方法尚未被探索应用于连续学习。我们提出了一种名为LANCE（Low-rank Activation Compression）的框架，该框架通过一次性的高阶奇异值分解（SVD）获得可重用的低秩子空间来进行激活投影。这消除了复现分解的需求，降低了内存和计算成本。此外，固定的低秩子空间还进一步促进了设备端的连续学习，在不会存储大量任务特定矩阵的情况下，将任务分配到正交子空间中。实验结果显示，LANCE在CIFAR-10/100、Oxford-IIIT Pets、Flowers102和CUB-200数据集上将激活存储减少多达250倍，同时保持与全反向传播相当的准确性。在连续学习基准测试（Split CIFAR-100、Split MiniImageNet、5-数据集）中，LANCE在内存成本极低的情况下，实现了与正交梯度投影方法相当的性能。这些结果将LANCE定位为适用于边缘设备上高效微调和连续学习的实用和可扩展解决方案。 

---
# Multi-Objective Reinforcement Learning for Large Language Model Optimization: Visionary Perspective 

**Title (ZH)**: 大型语言模型优化的多目标强化学习：前瞻视角 

**Authors**: Lingxiao Kong, Cong Yang, Oya Deniz Beyan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2509.21613)  

**Abstract**: Multi-Objective Reinforcement Learning (MORL) presents significant challenges and opportunities for optimizing multiple objectives in Large Language Models (LLMs). We introduce a MORL taxonomy and examine the advantages and limitations of various MORL methods when applied to LLM optimization, identifying the need for efficient and flexible approaches that accommodate personalization functionality and inherent complexities in LLMs and RL. We propose a vision for a MORL benchmarking framework that addresses the effects of different methods on diverse objective relationships. As future research directions, we focus on meta-policy MORL development that can improve efficiency and flexibility through its bi-level learning paradigm, highlighting key research questions and potential solutions for improving LLM performance. 

**Abstract (ZH)**: 多目标强化学习（MORL）在大型语言模型（LLMs）的多目标优化中提出了重要的挑战和机遇。我们引入了一种MORL分类，并探讨了各种MORL方法在LLM优化中的优势与局限性，指出了需要高效且灵活的方法来适应个性化功能和LLMs及RL固有的复杂性。我们提出了一种MORL基准测试框架的愿景，该框架旨在解决不同方法对各种目标关系的影响。作为未来的研究方向，我们重点讨论了通过其双层学习范式改进效率和灵活性的元策略MORL的发展，强调了提高LLM性能的关键研究问题和潜在解决方案。 

---
# Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis 

**Title (ZH)**: 时间维度 vs. 空间维度：DINOv3和V-JEPA2特征表示在视频动作分析中的比较 

**Authors**: Sai Varun Kodathala, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2509.21595)  

**Abstract**: This study presents a comprehensive comparative analysis of two prominent self-supervised learning architectures for video action recognition: DINOv3, which processes frames independently through spatial feature extraction, and V-JEPA2, which employs joint temporal modeling across video sequences. We evaluate both approaches on the UCF Sports dataset, examining feature quality through multiple dimensions including classification accuracy, clustering performance, intra-class consistency, and inter-class discrimination. Our analysis reveals fundamental architectural trade-offs: DINOv3 achieves superior clustering performance (Silhouette score: 0.31 vs 0.21) and demonstrates exceptional discrimination capability (6.16x separation ratio) particularly for pose-identifiable actions, while V-JEPA2 exhibits consistent reliability across all action types with significantly lower performance variance (0.094 vs 0.288). Through action-specific evaluation, we identify that DINOv3's spatial processing architecture excels at static pose recognition but shows degraded performance on motion-dependent actions, whereas V-JEPA2's temporal modeling provides balanced representation quality across diverse action categories. These findings contribute to the understanding of architectural design choices in video analysis systems and provide empirical guidance for selecting appropriate feature extraction methods based on task requirements and reliability constraints. 

**Abstract (ZH)**: 本研究对两种 prominant 自监督学习架构在视频动作识别中的表现进行了全面比较分析：DINOv3 通过空间特征提取独立处理每一帧，而 V-JEPA2 则在视频序列中采用联合时间建模。我们使用 UCF Sports 数据集评估这两种方法，在分类准确性、聚类表现、类别内一致性以及类别间区分能力等多个维度上分析特征质量。我们的分析揭示了架构上的根本权衡：DINOv3 在聚类性能方面表现更优（轮廓得分：0.31 对比 0.21），特别是在姿态可辨识的动作中展现出卓越的区分能力（6.16 倍分离比），而 V-JEPA2 在所有动作类型中表现出一致的可靠性，并且具有显著更低的性能变异（0.094 对比 0.288）。通过动作特异性评估，我们发现 DINOv3 的空间处理架构在静态姿态识别方面表现出色，但在依赖运动的动作上表现较差，而 V-JEPA2 的时间建模能够在多样化的动作类别中提供均衡的表示质量。这些发现有助于理解视频分析系统中架构设计选择，并为基于任务需求和可靠性约束选择合适的特征提取方法提供实证指导。 

---
# What Happens Next? Anticipating Future Motion by Generating Point Trajectories 

**Title (ZH)**: 接下来会发生什么？通过生成点轨迹来预测未来运动 

**Authors**: Gabrijel Boduljak, Laurynas Karazija, Iro Laina, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21592)  

**Abstract**: We consider the problem of forecasting motion from a single image, i.e., predicting how objects in the world are likely to move, without the ability to observe other parameters such as the object velocities or the forces applied to them. We formulate this task as conditional generation of dense trajectory grids with a model that closely follows the architecture of modern video generators but outputs motion trajectories instead of pixels. This approach captures scene-wide dynamics and uncertainty, yielding more accurate and diverse predictions than prior regressors and generators. We extensively evaluate our method on simulated data, demonstrate its effectiveness on downstream applications such as robotics, and show promising accuracy on real-world intuitive physics datasets. Although recent state-of-the-art video generators are often regarded as world models, we show that they struggle with forecasting motion from a single image, even in simple physical scenarios such as falling blocks or mechanical object interactions, despite fine-tuning on such data. We show that this limitation arises from the overhead of generating pixels rather than directly modeling motion. 

**Abstract (ZH)**: 从单张图像预测运动：一种基于条件生成密集轨迹网格的方法 

---
# Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms 

**Title (ZH)**: 通过在半变异图中发现困难负样本以增强对比学习在地学定位中的效果 

**Authors**: Boyi Chen, Zhangyu Wang, Fabian Deuser, Johann Maximilian Zollner, Martin Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.21573)  

**Abstract**: Accurate and robust image-based geo-localization at a global scale is challenging due to diverse environments, visually ambiguous scenes, and the lack of distinctive landmarks in many regions. While contrastive learning methods show promising performance by aligning features between street-view images and corresponding locations, they neglect the underlying spatial dependency in the geographic space. As a result, they fail to address the issue of false negatives -- image pairs that are both visually and geographically similar but labeled as negatives, and struggle to effectively distinguish hard negatives, which are visually similar but geographically distant. To address this issue, we propose a novel spatially regularized contrastive learning strategy that integrates a semivariogram, which is a geostatistical tool for modeling how spatial correlation changes with distance. We fit the semivariogram by relating the distance of images in feature space to their geographical distance, capturing the expected visual content in a spatial correlation. With the fitted semivariogram, we define the expected visual dissimilarity at a given spatial distance as reference to identify hard negatives and false negatives. We integrate this strategy into GeoCLIP and evaluate it on the OSV5M dataset, demonstrating that explicitly modeling spatial priors improves image-based geo-localization performance, particularly at finer granularity. 

**Abstract (ZH)**: 在全球范围内实现具有多样环境、视觉歧义场景和缺乏明显地标区域的准确且 robust 的基于图像的地理解析具有挑战性。尽管对比学习方法通过在街景图像和对应位置之间对齐特征显示出有前途的性能，但它们忽略了地理空间中的潜在空间依赖性。因此，它们无法解决视觉和地理上相似但被标记为负例的假阴性问题，并且难以有效区分视觉上相似但地理上相距较远的负例。为解决这一问题，我们提出了一种新颖的空间正则化对比学习策略，该策略结合了半变异函数，这是一种用于建模空间相关性随距离变化的地理统计工具。我们通过将特征空间中图像的距离与地理距离联系起来拟合半变异函数，捕捉空间相关性中预期的视觉内容。使用拟合好的半变异函数，我们定义给定空间距离下的预期视觉差异作为参考，以识别难以区分的负例和假阴性。我们将该策略集成到GeoCLIP中，并在OSV5M数据集上进行评估，证明明确建模空间先验有助于提高基于图像的地理解析性能，尤其是在精细粒度上。 

---
# No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models 

**Title (ZH)**: 无需对齐即可生成：学习线性可分表示的扩散模型 

**Authors**: Junno Yun, Yaşar Utku Alçalar, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2509.21565)  

**Abstract**: Efficient training strategies for large-scale diffusion models have recently emphasized the importance of improving discriminative feature representations in these models. A central line of work in this direction is representation alignment with features obtained from powerful external encoders, which improves the representation quality as assessed through linear probing. Alignment-based approaches show promise but depend on large pretrained encoders, which are computationally expensive to obtain. In this work, we propose an alternative regularization for training, based on promoting the Linear SEParability (LSEP) of intermediate layer representations. LSEP eliminates the need for an auxiliary encoder and representation alignment, while incorporating linear probing directly into the network's learning dynamics rather than treating it as a simple post-hoc evaluation tool. Our results demonstrate substantial improvements in both training efficiency and generation quality on flow-based transformer architectures such as SiTs, achieving an FID of 1.46 on $256 \times 256$ ImageNet dataset. 

**Abstract (ZH)**: 大型扩散模型的高效训练策略强调了提高这些模型中的鉴别特征表示的重要性。这一方向的一个主要研究线是通过强大外部编码器获得的特征进行表示对齐，以通过线性探测来提高表示质量。基于对齐的方法显示出前景，但依赖于大型的预训练编码器，这些编码器在计算上非常昂贵。在本文中，我们提出了一种替代的训练正则化方法，基于促进中间层表示的线性可分性（LSEP）。LSEP 消除了辅助编码器和表示对齐的需求，同时将线性探测直接纳入网络的学习动态中，而不是将其视为简单的后续评估工具。我们的结果在基于流的变压器架构如 SiTs 上显示了显著提高的训练效率和生成质量，在 $256 \times 256$ ImageNet 数据集上实现了 FID 为 1.46。 

---
# Domain-Aware Speaker Diarization On African-Accented English 

**Title (ZH)**: 基于领域意识的带有非洲口音的英语发言人聚类 

**Authors**: Chibuzor Okocha, Kelechi Ezema, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2509.21554)  

**Abstract**: This study examines domain effects in speaker diarization for African-accented English. We evaluate multiple production and open systems on general and clinical dialogues under a strict DER protocol that scores overlap. A consistent domain penalty appears for clinical speech and remains significant across models. Error analysis attributes much of this penalty to false alarms and missed detections, aligning with short turns and frequent overlap. We test lightweight domain adaptation by fine-tuning a segmentation module on accent-matched data; it reduces error but does not eliminate the gap. Our contributions include a controlled benchmark across domains, a concise approach to error decomposition and conversation-level profiling, and an adaptation recipe that is easy to reproduce. Results point to overlap-aware segmentation and balanced clinical resources as practical next steps. 

**Abstract (ZH)**: 本研究考察了非洲口音英语说话人分割中的领域效应，我们在严格的重叠评分DER协议下评估了多种生产系统和开源系统在一般对话和临床对话上的表现，发现临床语音领域存在一致的惩罚，并且在不同模型中保持显著。错误分析将这一惩罚的主要原因归结为误报和漏报，与短发言和频繁重叠相符。我们通过在匹配口音的数据上微调分割模块进行轻量级领域适应测试，虽然减少了错误但未完全消除差距。我们的贡献包括跨领域的受控基准、简洁的错误分解和对话级分析方法，以及易于复现的适应方法。研究结果指出，重叠意识分割和平衡临床资源是实际可行的下一步。 

---
# Psychological and behavioural responses in human-agent vs. human-human interactions: a systematic review and meta-analysis 

**Title (ZH)**: 人类与代理之间 vs. 人类与人类之间互动的心理与行为反应：一项系统回顾和元分析 

**Authors**: Jianan Zhou, Fleur Corbett, Joori Byun, Talya Porat, Nejra van Zalk  

**Link**: [PDF](https://arxiv.org/pdf/2509.21542)  

**Abstract**: Interactive intelligent agents are being integrated across society. Despite achieving human-like capabilities, humans' responses to these agents remain poorly understood, with research fragmented across disciplines. We conducted a first systematic synthesis comparing a range of psychological and behavioural responses in matched human-agent vs. human-human dyadic interactions. A total of 162 eligible studies (146 contributed to the meta-analysis; 468 effect sizes) were included in the systematic review and meta-analysis, which integrated frequentist and Bayesian approaches. Our results indicate that individuals exhibited less prosocial behaviour and moral engagement when interacting with agents vs. humans. They attributed less agency and responsibility to agents, perceiving them as less competent, likeable, and socially present. In contrast, individuals' social alignment (i.e., alignment or adaptation of internal states and behaviours with partners), trust in partners, personal agency, task performance, and interaction experiences were generally comparable when interacting with agents vs. humans. We observed high effect-size heterogeneity for many subjective responses (i.e., social perceptions of partners, subjective trust, and interaction experiences), suggesting context-dependency of partner effects. By examining the characteristics of studies, participants, partners, interaction scenarios, and response measures, we also identified several moderators shaping partner effects. Overall, functional behaviours and interactive experiences with agents can resemble those with humans, whereas fundamental social attributions and moral/prosocial concerns lag in human-agent interactions. Agents are thus afforded instrumental value on par with humans but lack comparable intrinsic value, providing practical implications for agent design and regulation. 

**Abstract (ZH)**: 交互智能代理正在社会中集成应用。尽管这些代理已展现出了类人的能力，但人们对这些代理的反应仍 poorly understood，跨学科的研究较为零散。我们进行了一项首次系统整合研究，比较了代理-人类与人类-人类互动中的一系列心理与行为反应。系统综述和元分析共纳入162项符合条件的研究（146项用于元分析；468个效应量），整合了 frequentist 和 Bayesian 方法。结果表明，与与人类互动相比，个体在与代理互动时表现出较少的亲社会行为和道德参与。他们赋予代理较少的意图和责任，认为代理不如人类能干、可亲和社交存在感。相反，个体的社会一致性（即与伙伴内部状态和行为的一致性或适应性）、对伙伴的信任、个人意图、任务表现和互动体验在与代理互动时通常与与人类互动相当。我们观察到许多主观反应（如对伙伴的社会认知、主观信任和互动体验）的效应大小差异性很高，表明伙伴效应具有情境依赖性。通过研究各个方面的特征（如研究特点、参与者、伙伴、互动场景和反应措施），我们还识别出若干影响伙伴效应的调节因素。总体而言，与人类互动相比，代理的功能行为和互动体验相似，但基本的社会归因和道德/亲社会关注较少。因此，代理在功能上与人类具有同等的重要性，但在内在价值上则较低，为代理设计和监管提供了实践意义。 

---
# Agribot: agriculture-specific question answer system 

**Title (ZH)**: 农 bot: 农业专用问答系统 

**Authors**: Naman Jain, Pranjali Jain, Pratik Kayal, Jayakrishna Sahit, Soham Pachpande, Jayesh Choudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.21535)  

**Abstract**: India is an agro-based economy and proper information about agricultural practices is the key to optimal agricultural growth and output. In order to answer the queries of the farmer, we have build an agricultural chatbot based on the dataset from Kisan Call Center. This system is robust enough to answer queries related to weather, market rates, plant protection and government schemes. This system is available 24* 7, can be accessed through any electronic device and the information is delivered with the ease of understanding. The system is based on a sentence embedding model which gives an accuracy of 56%. After eliminating synonyms and incorporating entity extraction, the accuracy jumps to 86%. With such a system, farmers can progress towards easier information about farming related practices and hence a better agricultural output. The job of the Call Center workforce would be made easier and the hard work of various such workers can be redirected to a better goal. 

**Abstract (ZH)**: 印度是一个农业导向型经济体，准确的农业信息是实现最优农业增长和产出的关键。为了回答农民的查询，我们基于Kisan呼叫中心的数据集构建了一个农业聊天机器人。该系统能够 robust 地回答与天气、市场行情、植物保护和政府方案相关的查询。该系统全年无休，可通过任何电子设备访问，并且信息传达易于理解。该系统基于句子嵌入模型，准确率为 56%。在消除同义词并结合实体提取后，准确率提升至 86%。通过这样一个系统，农民可以更容易地获取与农业相关的信息，从而提高农业产出。呼叫中心工作人员的工作将更加容易，这些工人的辛勤工作可以重新导向到更有意义的目标。 

---
# Preemptive Detection and Steering of LLM Misalignment via Latent Reachability 

**Title (ZH)**: 预判检测与引导大型语言模型错位gorithme 

**Authors**: Sathwik Karnik, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.21528)  

**Abstract**: Large language models (LLMs) are now ubiquitous in everyday tools, raising urgent safety concerns about their tendency to generate harmful content. The dominant safety approach -- reinforcement learning from human feedback (RLHF) -- effectively shapes model behavior during training but offers no safeguards at inference time, where unsafe continuations may still arise. We propose BRT-Align, a reachability-based framework that brings control-theoretic safety tools to LLM inference. BRT-Align models autoregressive generation as a dynamical system in latent space and learn a safety value function via backward reachability, estimating the worst-case evolution of a trajectory. This enables two complementary mechanisms: (1) a runtime monitor that forecasts unsafe completions several tokens in advance, and (2) a least-restrictive steering filter that minimally perturbs latent states to redirect generation away from unsafe regions. Experiments across multiple LLMs and toxicity benchmarks demonstrate that BRT-Align provides more accurate and earlier detection of unsafe continuations than baselines. Moreover, for LLM safety alignment, BRT-Align substantially reduces unsafe generations while preserving sentence diversity and coherence. Qualitative results further highlight emergent alignment properties: BRT-Align consistently produces responses that are less violent, less profane, less offensive, and less politically biased. Together, these findings demonstrate that reachability analysis provides a principled and practical foundation for inference-time LLM safety. 

**Abstract (ZH)**: 基于可达性分析的大型语言模型推理安全性框架：BRT-Align 

---
# Shortcut Flow Matching for Speech Enhancement: Step-Invariant flows via single stage training 

**Title (ZH)**: 用于语音增强的捷径流匹配：基于单阶段训练的步进不变流 

**Authors**: Naisong Zhou, Saisamarth Rajesh Phaye, Milos Cernak, Tijana Stojkovic, Andy Pearce, Andrea Cavallaro, Andy Harper  

**Link**: [PDF](https://arxiv.org/pdf/2509.21522)  

**Abstract**: Diffusion-based generative models have achieved state-of-the-art performance for perceptual quality in speech enhancement (SE). However, their iterative nature requires numerous Neural Function Evaluations (NFEs), posing a challenge for real-time applications. On the contrary, flow matching offers a more efficient alternative by learning a direct vector field, enabling high-quality synthesis in just a few steps using deterministic ordinary differential equation~(ODE) solvers. We thus introduce Shortcut Flow Matching for Speech Enhancement (SFMSE), a novel approach that trains a single, step-invariant model. By conditioning the velocity field on the target time step during a one-stage training process, SFMSE can perform single, few, or multi-step denoising without any architectural changes or fine-tuning. Our results demonstrate that a single-step SFMSE inference achieves a real-time factor (RTF) of 0.013 on a consumer GPU while delivering perceptual quality comparable to a strong diffusion baseline requiring 60 NFEs. This work also provides an empirical analysis of the role of stochasticity in training and inference, bridging the gap between high-quality generative SE and low-latency constraints. 

**Abstract (ZH)**: 基于扩散的生成模型在语音增强中实现了感知质量的最先进性能。然而，其迭代特性需要大量神经函数评估（NFEs），这给实时应用带来挑战。相反，流匹配提供了一种更高效的替代方案，通过学习直接的向量场，使用确定性的常微分方程（ODE）求解器，在少量步骤内实现高质量的合成。因此，我们提出了语音增强中的快捷流匹配方法（SFMSE），这是一种新型方法，训练一个单一的、步骤不变的模型。通过在一阶段训练过程中使速度场依赖于目标时间步，SFMSE 可以在不需要任何架构更改或微调的情况下执行单步、少量步或多步去噪。我们的结果表明，单步 SFMSE 推断在消费者 GPU 上实现了 0.013 的实时因子（RTF），同时提供了与需要 60 个 NFEs 的强扩散基线相当的感知质量。这项工作还提供了关于训练和推断中随机性作用的经验分析，缩小了高质量生成 SE 和低延迟约束之间的差距。 

---
# $\mathbf{Li_2}$: A Framework on Dynamics of Feature Emergence and Delayed Generalization 

**Title (ZH)**: $\mathbf{Li_2}$: 一种特征涌现和延迟泛化的动力学框架 

**Authors**: Yuandong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.21519)  

**Abstract**: While the phenomenon of grokking, i.e., delayed generalization, has been studied extensively, it remains an open question whether there is a mathematical framework to characterize what kind of features emerge, how and in which conditions it happens from training, for complex structured inputs. We propose a novel framework, named $\mathbf{Li_2}$, that captures three key stages for the grokking behavior of 2-layer nonlinear networks: (I) Lazy learning, (II) independent feature learning and (III) interactive feature learning, characterized by the structure of backpropagated gradient $G_F$ across layers. In (I), $G_F$ is random, and top layer overfits to random hidden representation. In (II), the gradient of each node (column of $G_F$) only depends on its own activation, and thus each hidden node learns their representation independently from $G_F$, which now carries information about target labels, thanks to weight decay. Interestingly, the independent dynamics follows exactly the gradient ascent of an energy function $E$, and its local maxima are precisely the emerging features. We study whether these local-optima induced features are generalizable, their representation power, and how they change on sample size, in group arithmetic tasks. Finally, in (III), we provably show how hidden nodes interact, and how $G_F$ changes to focus on missing features that need to be learned. Our study sheds lights on roles played by key hyperparameters such as weight decay, learning rate and sample sizes in grokking, leads to provable scaling laws of memorization and generalization, and reveals the underlying cause why recent optimizers such as Muon can be effective, from the first principles of gradient dynamics. Our analysis can be extended to multi-layer architectures. 

**Abstract (ZH)**: 基于Li₂框架的两层非线性网络的grokking行为分析：懒学习、独立特征学习和交互特征学习的数学描述 

---
# DistillKac: Few-Step Image Generation via Damped Wave Equations 

**Title (ZH)**: DistillKac: 通过阻尼波方程实现多步图像生成 

**Authors**: Weiqiao Han, Chenlin Meng, Christopher D. Manning, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.21513)  

**Abstract**: We present DistillKac, a fast image generator that uses the damped wave equation and its stochastic Kac representation to move probability mass at finite speed. In contrast to diffusion models whose reverse time velocities can become stiff and implicitly allow unbounded propagation speed, Kac dynamics enforce finite speed transport and yield globally bounded kinetic energy. Building on this structure, we introduce classifier-free guidance in velocity space that preserves square integrability under mild conditions. We then propose endpoint only distillation that trains a student to match a frozen teacher over long intervals. We prove a stability result that promotes supervision at the endpoints to closeness along the entire path. Experiments demonstrate DistillKac delivers high quality samples with very few function evaluations while retaining the numerical stability benefits of finite speed probability flows. 

**Abstract (ZH)**: DistillKac：一种快速图像生成器及其在有限速度概率流中的应用 

---
# New Algorithmic Directions in Optimal Transport and Applications for Product Spaces 

**Title (ZH)**: 最优传输中新的算法方向及在乘积空间中的应用 

**Authors**: Salman Beigi, Omid Etesami, Mohammad Mahmoody, Amir Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21502)  

**Abstract**: We study optimal transport between two high-dimensional distributions $\mu,\nu$ in $R^n$ from an algorithmic perspective: given $x \sim \mu$, find a close $y \sim \nu$ in $poly(n)$ time, where $n$ is the dimension of $x,y$. Thus, running time depends on the dimension rather than the full representation size of $\mu,\nu$. Our main result is a general algorithm for transporting any product distribution $\mu$ to any $\nu$ with cost $\Delta + \delta$ under $\ell_p^p$, where $\Delta$ is the Knothe-Rosenblatt transport cost and $\delta$ is a computational error decreasing with runtime. This requires $\nu$ to be "sequentially samplable" with bounded average sampling cost, a new but natural notion.
We further prove:
An algorithmic version of Talagrand's inequality for transporting the standard Gaussian $\Phi^n$ to arbitrary $\nu$ under squared Euclidean cost. For $\nu = \Phi^n$ conditioned on a set $\mathcal{S}$ of measure $\varepsilon$, we construct the sequential sampler in expected time $poly(n/\varepsilon)$ using membership oracle access to $\mathcal{S}$. This yields an algorithmic transport from $\Phi^n$ to $\Phi^n|\mathcal{S}$ in $poly(n/\varepsilon)$ time and expected squared distance $O(\log 1/\varepsilon)$, optimal for general $\mathcal{S}$ of measure $\varepsilon$.
As corollary, we obtain the first computational concentration result (Etesami et al. SODA 2020) for Gaussian measure under Euclidean distance with dimension-independent transportation cost, resolving an open question of Etesami et al. Specifically, for any $\mathcal{S}$ of Gaussian measure $\varepsilon$, most $\Phi^n$ samples can be mapped to $\mathcal{S}$ within distance $O(\sqrt{\log 1/\varepsilon})$ in $poly(n/\varepsilon)$ time. 

**Abstract (ZH)**: 高维空间中两分布$\mu,\nu$之间的最优传输算法研究：从$\mu$中的$x$找到$\nu$中接近的$y$，时间复杂度为多项式时间，取决于维度而非分布的完全表示大小。我们的主要结果是，对于任意乘积分布$\mu$和任意$\nu$，在$\ell_p^p$代价下，传输成本为$\Delta + \delta$，其中$\Delta$为Knothe-Rosenblatt传输成本，$\delta$是随运行时间减少的计算误差。要求$\nu$为可按顺序采样的分布，且平均采样成本有界，这是一条新的但自然的性质。 

---
# Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training 

**Title (ZH)**: 追逐尾部样本：有效的基于评阅标准的奖励模型对大规模语言模型后训练的研究 

**Authors**: Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, Lifeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21500)  

**Abstract**: Reinforcement fine-tuning (RFT) often suffers from \emph{reward over-optimization}, where a policy model hacks the reward signals to achieve high scores while producing low-quality outputs. Our theoretical analysis shows that the key lies in reward misspecification at the high-reward tail: the inability to reliably distinguish Excellent responses from merely Great ones. This motivate us to focus on the high-reward region. However, such tail examples are scarce under the base LLM. While off-policy exemplars (e.g. from stronger models or rewrites) are easier to obtain, naively training on them yields a misspecified reward for the policy we aim to align. To address this, we study rubric-based rewards. By design, rubrics can leverage off-policy examples while remaining insensitive to their artifacts. To elicit rubrics that capture the high-reward tail, we highlight the importance of distinguishing among great and diverse responses, and introduce a workflow to implement this idea. We empirically demonstrate that rubric-based rewards substantially mitigate reward over-optimization and deliver effective LLM post-training improvements. Our code can be accessed at this https URL . 

**Abstract (ZH)**: 强化微调（RFT）常常遭受奖励过优化的问题，即策略模型通过操纵奖励信号来获得高分，但产生的输出质量较低。我们的理论分析表明，关键在于高奖分尾部的奖励失确：无法可靠地区分优秀回答与仅仅很好的回答。这促使我们关注高奖分区域。然而，在基础大规模语言模型中，这样的尾部样例稀缺。虽然可以从更强的模型或改写中获得非策略样例（例如），直接使用它们来训练会导致我们希望对齐的策略模型的奖励失确。为了解决这一问题，我们研究基于评分标准的奖励。通过设计，评分标准可以利用非策略样例同时保持对它们缺陷的不敏感性。为了获取能够捕捉高奖分尾部的评分标准，我们强调区分优秀和多样回答的重要性，并引入了一个实施这一理念的流程。我们实验证明，基于评分标准的奖励显著缓解了奖励过优化问题，并提供了有效的后训练大规模语言模型改进。我们的代码可以在以下链接访问：this https URL。 

---
# Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning 

**Title (ZH)**: 双头推理精炼：通过训练时仅推理提高分类器准确性 

**Authors**: Jillian Xu, Dylan Zhou, Vinay Shukla, Yang Yang, Junrui Ruan, Shuhuai Lin, Wenfei Zou, Yinxiao Liu, Karthik Lakshmanan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21487)  

**Abstract**: Chain-of-Thought (CoT) prompting often improves classification accuracy, but it introduces a significant throughput penalty with rationale generation (Wei et al., 2022; Cheng and Van Durme, 2024). To resolve this trade-off, we introduce Dual-Head Reasoning Distillation (DHRD), a simple training method for decoder-only language models (LMs) that adds (i) a pooled classification head used during training and inference and (ii) a reasoning head supervised by teacher rationales used only in training. We train with a loss function that is a weighted sum of label cross-entropy and token-level LM loss over input-plus-rationale sequences. On seven SuperGLUE tasks, DHRD yields relative gains of 0.65-5.47% over pooled baselines, with notably larger gains on entailment/causal tasks. Since we disable the reasoning head at test time, inference throughput matches pooled classifiers and exceeds CoT decoding on the same backbones by 96-142 times in QPS. 

**Abstract (ZH)**: Chain-of-Thought (CoT) 提示往往能够提高分类准确性，但会引入显著的吞吐量惩罚（Wei et al., 2022；Cheng and Van Durme, 2024）。为了解决这一权衡，我们提出了一种双重头推理蒸馏（DHRD）方法，这是一种适用于仅解码器语言模型（LMs）的简单训练方法，它增加了（i）一个用于训练和推理的聚合分类头，以及（ii）一个由教师推理监督的推理头，仅在训练时使用。我们在一个损失函数中使用标签交叉熵和输入加推理序列的词元级LM损失加权和进行训练。在七个SuperGLUE任务上，DHRD相比聚合基线模型取得了6.5%-54.7%的相对增益，尤其在蕴含/因果任务上收益明显更大。由于我们在测试时禁用了推理头，推理吞吐量与聚合分类器相当，并且在相同骨干模型上超越CoT解码的每秒查询数QPS高出96-142倍。 

---
# Neural Operators for Mathematical Modeling of Transient Fluid Flow in Subsurface Reservoir Systems 

**Title (ZH)**: 神经算子在地下储层系统瞬态流体流动的数学建模中的应用 

**Authors**: Daniil D. Sirota, Sergey A. Khan, Sergey L. Kostikov, Kirill A. Butov  

**Link**: [PDF](https://arxiv.org/pdf/2509.21485)  

**Abstract**: This paper presents a method for modeling transient fluid flow in subsurface reservoir systems based on the developed neural operator architecture (TFNO-opt). Reservoir systems are complex dynamic objects with distributed parameters described by systems of partial differential equations (PDEs). Traditional numerical methods for modeling such systems, despite their high accuracy, are characterized by significant time costs for performing calculations, which limits their applicability in control and decision support problems. The proposed architecture (TFNO-opt) is based on Fourier neural operators, which allow approximating PDE solutions in infinite-dimensional functional spaces, providing invariance to discretization and the possibility of generalization to various implementations of equations. The developed modifications are aimed at increasing the accuracy and stability of the trained neural operator, which is especially important for control problems. These include adjustable internal time resolution of the integral Fourier operator, tensor decomposition of parameters in the spectral domain, use of the Sobolev norm in the error function, and separation of approximation errors and reconstruction of initial conditions for more accurate reproduction of physical processes. The effectiveness of the proposed improvements is confirmed by computational experiments. The practical significance is confirmed by computational experiments using the example of the problem of hydrodynamic modeling of an underground gas storage (UGS), where the acceleration of calculations by six orders of magnitude was achieved, compared to traditional methods. This opens up new opportunities for the effective control of complex reservoir systems. 

**Abstract (ZH)**: 基于TFNO-opt神经算子架构的地下 reservoir 系统瞬态流体流动建模方法 

---
# Learning to Reason with Mixture of Tokens 

**Title (ZH)**: 学习使用词元混合进行推理 

**Authors**: Adit Jain, Brendan Rappazzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21482)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a leading approach for improving large language model (LLM) reasoning capabilities. Most current methods follow variants of Group Relative Policy Optimization, which samples multiple reasoning completions, scores them relative to each other, and adjusts the policy accordingly. However, these approaches invariably sample discrete tokens at each reasoning step, discarding the rich distributional information in the model's probability distribution over candidate tokens. While preserving and utilizing this distributional information has proven beneficial in non-RL settings, current RLVR methods seem to be unnecessarily constraining the reasoning search space by not using this information. To address this limitation, we investigate mixture-of-token generation (MoT-G) in RLVR. We present a unified framework that generalizes existing MoT-G approaches, including existing training-free methods that construct mixture embeddings as weighted sums over token embeddings, and extend RLVR to operate directly in this continuous mixture space for generating chain-of-thought. Evaluating two MoT-G variants on Reasoning-Gym, a suite of reasoning-intensive language tasks, we find that MoT--G methods achieve substantial improvements (5--35 \% gains on 7 out of 10 tasks) compared to standard decoding with the Qwen2.5-1.5B model, while reaching comparable accuracy with half the number of trajectories, suggesting improved training efficiency. Through comprehensive hidden-state and token-level analyses, we provide evidence that MoT--G's benefits may stem from its ability to maintain higher hidden-state entropy throughout the reasoning process and promote exploration in token space. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已成为提升大规模语言模型（LLM）推理能力的领先方法。大多数当前方法遵循组相对策略优化的变体，生成多个推理完成，并相对评分以调整策略。然而，这些方法不可避免地在每一步推理中采样离散标记，忽略了模型在候选标记概率分布中的丰富分布信息。虽然在非RL设置中保留并利用这种分布信息已被证明是有益的，但当前的RLVR方法似乎通过不使用这些信息，不必要地限制了推理搜索空间。为解决这一限制，我们研究了RLVR中的混合标记生成（MoT-G）。我们提出了一种统一框架，将现有的MoT-G方法泛化，包括现有的无需训练的方法，这些方法构建混合嵌入作为标记嵌入的加权和，并扩展RLVR以直接操作这种连续混合空间来生成思维链。在Reasoning-Gym任务套件上评估了两种MoT-G变体，发现MoT-G方法与Qwen2.5-1.5B模型标准解码相比在7项任务中取得了显著改进（7项任务中5%至35%的提升），同时达到相当的准确性但只需一半的轨迹数量，这表明训练效率的提升。通过全面的隐藏状态和标记级分析，我们提供了证据，表明MoT-G的好处可能源自其在整个推理过程中保持更高的隐藏状态熵并促进标记空间的探索的能力。 

---
# Are Hallucinations Bad Estimations? 

**Title (ZH)**: 幻觉是糟糕的估计吗？ 

**Authors**: Hude Liu, Jerry Yao-Chieh Hu, Jennifer Yuntong Zhang, Zhao Song, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21473)  

**Abstract**: We formalize hallucinations in generative models as failures to link an estimate to any plausible cause. Under this interpretation, we show that even loss-minimizing optimal estimators still hallucinate. We confirm this with a general high probability lower bound on hallucinate rate for generic data distributions. This reframes hallucination as structural misalignment between loss minimization and human-acceptable outputs, and hence estimation errors induced by miscalibration. Experiments on coin aggregation, open-ended QA, and text-to-image support our theory. 

**Abstract (ZH)**: 我们在生成模型中将幻觉形式化为未能将估计值与任何合理的因果关系联系起来的失败。在这一解释下，我们展示了即使是最优损失最小化估计器也会产生幻觉。我们通过一个适用于通用数据分布的高概率下界验证了这一点。这将幻觉重新定义为损失最小化与人类可接受输出之间的结构对齐问题，进而将其视为由于校准不当引起估计错误。我们在硬币聚合、开放式问答和文本转图像实验中支持了这一理论。 

---
# Score-based Idempotent Distillation of Diffusion Models 

**Title (ZH)**: 基于评分的幂等蒸馏扩散模型 

**Authors**: Shehtab Zaman, Chengyan Liu, Kenneth Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21470)  

**Abstract**: Idempotent generative networks (IGNs) are a new line of generative models based on idempotent mapping to a target manifold. IGNs support both single-and multi-step generation, allowing for a flexible trade-off between computational cost and sample quality. But similar to Generative Adversarial Networks (GANs), conventional IGNs require adversarial training and are prone to training instabilities and mode collapse. Diffusion and score-based models are popular approaches to generative modeling that iteratively transport samples from one distribution, usually a Gaussian, to a target data distribution. These models have gained popularity due to their stable training dynamics and high-fidelity generation quality. However, this stability and quality come at the cost of high computational cost, as the data must be transported incrementally along the entire trajectory. New sampling methods, model distillation, and consistency models have been developed to reduce the sampling cost and even perform one-shot sampling from diffusion models. In this work, we unite diffusion and IGNs by distilling idempotent models from diffusion model scores, called SIGN. Our proposed method is highly stable and does not require adversarial losses. We provide a theoretical analysis of our proposed score-based training methods and empirically show that IGNs can be effectively distilled from a pre-trained diffusion model, enabling faster inference than iterative score-based models. SIGNs can perform multi-step sampling, allowing users to trade off quality for efficiency. These models operate directly on the source domain; they can project corrupted or alternate distributions back onto the target manifold, enabling zero-shot editing of inputs. We validate our models on multiple image datasets, achieving state-of-the-art results for idempotent models on the CIFAR and CelebA datasets. 

**Abstract (ZH)**: 同态生成网络（IGNs）是一类基于同态映射到目标流形的生成模型。IGNs 支持单步和多步生成，允许在计算成本和样本质量之间灵活权衡。但与生成对抗网络（GANs）类似，传统的IGNs 需要对抗训练，并且容易出现训练不稳定性和模式枯竭问题。扩散模型和基于分数的模型是生成建模的流行方法，通过迭代将样本从一个分布，通常是高斯分布，转移到目标数据分布。这些模型由于其稳定的训练动态和高保真生成质量而受到欢迎。然而，这种稳定性和高质量是以高计算成本为代价的，因为数据必须在整个轨迹中逐步传输。为了降低采样成本，甚至可以从扩散模型实现一次采样，开发了新的采样方法、模型蒸馏和一致性模型。在本文中，我们通过从扩散模型分数蒸馏同态模型，提出了一种称为SIGN的方法。我们提出的方法非常稳定，并不需要对抗损失。我们提供了我们提出的基于分数的训练方法的理论分析，并通过实验证明预先训练的扩散模型可以有效地蒸馏出IGNs，从而使得IGNs在比迭代基于分数的模型更快的推理中表现出色。SIGNs 支持多步采样，允许用户在质量和效率之间进行权衡。这些模型可以直接在源域上操作；它们可以将受损或替代分布投影回目标流形，从而实现零样本输入编辑。我们在多个图像数据集上验证了我们的模型，实现了CIFAR和CelebA数据集上同态模型的最佳结果。 

---
# Gender Stereotypes in Professional Roles Among Saudis: An Analytical Study of AI-Generated Images Using Language Models 

**Title (ZH)**: 沙特阿拉伯专业角色中的性别刻板印象：基于语言模型生成的图像的分析研究 

**Authors**: Khaloud S. AlKhalifah, Malak Mashaabi, Hend Al-Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2509.21466)  

**Abstract**: This study investigates the extent to which contemporary Text-to-Image artificial intelligence (AI) models perpetuate gender stereotypes and cultural inaccuracies when generating depictions of professionals in Saudi Arabia. We analyzed 1,006 images produced by ImageFX, DALL-E V3, and Grok for 56 diverse Saudi professions using neutral prompts. Two trained Saudi annotators evaluated each image on five dimensions: perceived gender, clothing and appearance, background and setting, activities and interactions, and age. A third senior researcher adjudicated whenever the two primary raters disagreed, yielding 10,100 individual judgements. The results reveal a strong gender imbalance, with ImageFX outputs being 85\% male, Grok 86.6\% male, and DALL-E V3 96\% male, indicating that DALL-E V3 exhibited the strongest overall gender stereotyping. This imbalance was most evident in leadership and technical roles. Moreover, cultural inaccuracies in clothing, settings, and depicted activities were frequently observed across all three models. Counter-stereotypical images often arise from cultural misinterpretations rather than genuinely progressive portrayals. We conclude that current models mirror societal biases embedded in their training data, generated by humans, offering only a limited reflection of the Saudi labour market's gender dynamics and cultural nuances. These findings underscore the urgent need for more diverse training data, fairer algorithms, and culturally sensitive evaluation frameworks to ensure equitable and authentic visual outputs. 

**Abstract (ZH)**: 当代文本生成图像人工智能模型在生成沙特阿拉伯职业形象时性别刻板印象和文化不准确性的程度研究 

---
# Enhanced Generative Machine Listener 

**Title (ZH)**: 增强生成式机器听众 

**Authors**: Vishnu Raj, Gouthaman KV, Shiv Gehlot, Lars Villemoes, Arijit Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2509.21463)  

**Abstract**: We present GMLv2, a reference-based model designed for the prediction of subjective audio quality as measured by MUSHRA scores. GMLv2 introduces a Beta distribution-based loss to model the listener ratings and incorporates additional neural audio coding (NAC) subjective datasets to extend its generalization and applicability. Extensive evaluations on diverse testset demonstrate that proposed GMLv2 consistently outperforms widely used metrics, such as PEAQ and ViSQOL, both in terms of correlation with subjective scores and in reliably predicting these scores across diverse content types and codec configurations. Consequently, GMLv2 offers a scalable and automated framework for perceptual audio quality evaluation, poised to accelerate research and development in modern audio coding technologies. 

**Abstract (ZH)**: GMLv2：基于参考的主观音频质量预测模型 

---
# A State-of-the-Art SQL Reasoning Model using RLVR 

**Title (ZH)**: 一种基于RLVR的最先进技术状态的SQL推理模型 

**Authors**: Alnur Ali, Ashutosh Baheti, Jonathan Chang, Ta-Chung Chi, Brandon Cui, Andrew Drozdov, Jonathan Frankle, Abhay Gupta, Pallavi Koppol, Sean Kulinski, Jonathan Li, Dipendra Misra, Krista Opsahl-Ong, Jose Javier Gonzalez Ortiz, Matei Zaharia, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21459)  

**Abstract**: Developing custom reasoning models via Reinforcement Learning (RL) that can incorporate organization-specific knowledge has great potential to address problems faced by enterprise customers. In many of these problems, the reward function is verifiable, a setting termed RL with Verifiable Rewards (RLVR). We apply RLVR to a popular data science benchmark called BIRD that measures the ability of an AI agent to convert a natural language query for a database to SQL executions. We apply a simple and general-purpose training recipe involving careful prompt and model selection, a warm-up stage using our offline RL approach called TAO, followed by rigorous online RLVR training. With no additional training data beyond the BIRD training set and no use of proprietary models, our very first submission to the BIRD leaderboard reached state-of-the-art accuracy on the private test set: 73.56% without self-consistency and 75.68% with self-consistency. In the latter case, our model also required fewer generations than the second-best approach. While BIRD is only a proxy task, the simplicity of our framework makes it broadly applicable to enterprise domains such as business intelligence, data science, and coding. 

**Abstract (ZH)**: 通过强化学习开发定制推理模型以 incorporate 组织特定知识在解决企业客户问题方面具有巨大潜力。在许多这些问题中，奖励函数是可验证的，这种设置称为可验证奖励的强化学习（RLVR）。我们将 RLVR 应用于一个流行的 数据科学基准 BIRD，该基准衡量 AI 代理将自然语言数据库查询转换为 SQL 执行的能力。我们应用了一种简单且通用的训练食谱，涉及仔细的选择提示和模型，使用我们的离线 RL 方法 TAO 进行预热阶段，然后进行严格的在线 RLVR 训练。除了 BIRD 训练集之外没有使用额外的训练数据，并且没有使用专有模型，我们的首次提交在 BIRD 私有测试集上达到了最先进的准确性：无自我一致性情况下为 73.56%，带有自我一致性情况下为 75.68%。在后一种情况下，我们的模型还比第二好方法所需的生成次数更少。虽然 BIRD 只是一个代理任务，但我们的框架的简洁性使其在商业智能、数据科学和编程等企业领域具有广泛的应用前景。 

---
# ARTI-6: Towards Six-dimensional Articulatory Speech Encoding 

**Title (ZH)**: ARTI-6: 向六维articulatory发音编码迈进 

**Authors**: Jihwan Lee, Sean Foley, Thanathai Lertpetchpun, Kevin Huang, Yoonjeong Lee, Tiantian Feng, Louis Goldstein, Dani Byrd, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21447)  

**Abstract**: We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available. 

**Abstract (ZH)**: 我们提出ARTI-6，一个源自实时MRI数据的紧凑六维发音语音编码框架，捕捉包括软腭、舌根和声带在内的关键发音 tract 区域。ARTI-6 包含三个组成部分：（1）一个六维发音特征集，表示发音 tract 的关键区域；（2）一个发音反演模型，通过利用语音基础模型从语音声学中预测发音特征，预测相关性达到 0.87；（3）一个发音合成模型，直接从发音特征重建可懂度高的语音，展示了低维表示也能生成自然听起来的语音。结合在一起，ARTI-6 提供了一个可解释、计算效率高且生理学上合理的框架，用于推动发音反演、合成和更广泛的语音技术应用。源代码和语音样本已公开。 

---
# One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning 

**Title (ZH)**: 一个模型，多种道德观：探究计算道德推理中的跨语言不一致性 

**Authors**: Sualeha Farid, Jayden Lin, Zean Chen, Shivani Kumar, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2509.21443)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multilingual and multicultural environments where moral reasoning is essential for generating ethically appropriate responses. Yet, the dominant pretraining of LLMs on English-language data raises critical concerns about their ability to generalize judgments across diverse linguistic and cultural contexts. In this work, we systematically investigate how language mediates moral decision-making in LLMs. We translate two established moral reasoning benchmarks into five culturally and typologically diverse languages, enabling multilingual zero-shot evaluation. Our analysis reveals significant inconsistencies in LLMs' moral judgments across languages, often reflecting cultural misalignment. Through a combination of carefully constructed research questions, we uncover the underlying drivers of these disparities, ranging from disagreements to reasoning strategies employed by LLMs. Finally, through a case study, we link the role of pretraining data in shaping an LLM's moral compass. Through this work, we distill our insights into a structured typology of moral reasoning errors that calls for more culturally-aware AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多语言和多元文化环境中部署时，道德推理对于生成伦理上合适的响应至关重要。然而，LLMs 主要通过英语数据进行预训练，这引发了它们在不同语言和文化背景下进行道德判断泛化的关键问题。在本工作中，我们系统地研究了语言如何在LLMs中中介道德决策。我们将两个已建立的道德推理基准翻译成五种具有文化和类型多样性的语言，从而实现多语言零样本评估。我们的分析揭示了LLMs在不同语言之间的道德判断存在显著差异，往往反映了文化错位。通过结合精心构建的研究问题，我们揭示了这些差异背后的驱动因素，包括分歧以及LLMs采用的推理策略。最后，通过一个案例研究，我们探讨了预训练数据在塑造LLMs道德导向方面的角色。通过对本工作的总结，我们提炼出一种结构化的道德推理错误类型学，呼吁更加注重文化意识的人工智能。 

---
# Foundation models for high-energy physics 

**Title (ZH)**: 高能物理领域的基础模型 

**Authors**: Anna Hallin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21434)  

**Abstract**: The rise of foundation models -- large, pretrained machine learning models that can be finetuned to a variety of tasks -- has revolutionized the fields of natural language processing and computer vision. In high-energy physics, the question of whether these models can be implemented directly in physics research, or even built from scratch, tailored for particle physics data, has generated an increasing amount of attention. This review, which is the first on the topic of foundation models in high-energy physics, summarizes and discusses the research that has been published in the field so far. 

**Abstract (ZH)**: 基础模型的兴起——大型预训练机器学习模型可以针对各种任务进行微调，这已彻底改变了自然语言处理和计算机视觉领域。在高能物理领域，关于这些模型是否可以直接应用于物理研究，甚至是否可以根据粒子物理数据从头构建，这一问题日益受到关注。本文作为首个关于高能物理领域基础模型的综述，总结并讨论了迄今为止该领域已发表的研究成果。 

---
# DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation 

**Title (ZH)**: DyME: 动态多概念消除在具有双层正交LoRA适应的扩散模型中 

**Authors**: Jiaqi Liu, Lan Zhang, Xiaoyong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21433)  

**Abstract**: Text-to-image diffusion models (DMs) inadvertently reproduce copyrighted styles and protected visual concepts, raising legal and ethical concerns. Concept erasure has emerged as a safeguard, aiming to selectively suppress such concepts through fine-tuning. However, existing methods do not scale to practical settings where providers must erase multiple and possibly conflicting concepts. The core bottleneck is their reliance on static erasure: a single checkpoint is fine-tuned to remove all target concepts, regardless of the actual erasure needs at inference. This rigid design mismatches real-world usage, where requests vary per generation, leading to degraded erasure success and reduced fidelity for non-target content. We propose DyME, an on-demand erasure framework that trains lightweight, concept-specific LoRA adapters and dynamically composes only those needed at inference. This modular design enables flexible multi-concept erasure, but naive composition causes interference among adapters, especially when many or semantically related concepts are suppressed. To overcome this, we introduce bi-level orthogonality constraints at both the feature and parameter levels, disentangling representation shifts and enforcing orthogonal adapter subspaces. We further develop ErasureBench-H, a new hierarchical benchmark with brand-series-character structure, enabling principled evaluation across semantic granularities and erasure set sizes. Experiments on ErasureBench-H and standard datasets (e.g., CIFAR-100, Imagenette) demonstrate that DyME consistently outperforms state-of-the-art baselines, achieving higher multi-concept erasure fidelity with minimal collateral degradation. 

**Abstract (ZH)**: 基于文本到图像扩散模型的按需擦除框架：DyME 

---
# PhenoMoler: Phenotype-Guided Molecular Optimization via Chemistry Large Language Model 

**Title (ZH)**: PhenoMoler：基于表型指导的分子优化 via 化学大型语言模型 

**Authors**: Ran Song, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21424)  

**Abstract**: Current molecular generative models primarily focus on improving drug-target binding affinity and specificity, often neglecting the system-level phenotypic effects elicited by compounds. Transcriptional profiles, as molecule-level readouts of drug-induced phenotypic shifts, offer a powerful opportunity to guide molecular design in a phenotype-aware manner. We present PhenoMoler, a phenotype-guided molecular generation framework that integrates a chemistry large language model with expression profiles to enable biologically informed drug design. By conditioning the generation on drug-induced differential expression signatures, PhenoMoler explicitly links transcriptional responses to chemical structure. By selectively masking and reconstructing specific substructures-scaffolds, side chains, or linkers-PhenoMoler supports fine-grained, controllable molecular optimization. Extensive experiments demonstrate that PhenoMoler generates chemically valid, novel, and diverse molecules aligned with desired phenotypic profiles. Compared to FDA-approved drugs, the generated compounds exhibit comparable or enhanced drug-likeness (QED), optimized physicochemical properties, and superior binding affinity to key cancer targets. These findings highlight PhenoMoler's potential for phenotype-guided and structure-controllable molecular optimization. 

**Abstract (ZH)**: PhenoMoler：一种基于表型引导的分子生成框架 

---
# Near-Optimal Experiment Design in Linear non-Gaussian Cyclic Models 

**Title (ZH)**: 接近最优的实验设计在线性非高斯循环模型中 

**Authors**: Ehsan Sharifian, Saber Salehkaleybar, Negar Kiyavash  

**Link**: [PDF](https://arxiv.org/pdf/2509.21423)  

**Abstract**: We study the problem of causal structure learning from a combination of observational and interventional data generated by a linear non-Gaussian structural equation model that might contain cycles. Recent results show that using mere observational data identifies the causal graph only up to a permutation-equivalence class. We obtain a combinatorial characterization of this class by showing that each graph in an equivalence class corresponds to a perfect matching in a bipartite graph. This bipartite representation allows us to analyze how interventions modify or constrain the matchings. Specifically, we show that each atomic intervention reveals one edge of the true matching and eliminates all incompatible causal graphs. Consequently, we formalize the optimal experiment design task as an adaptive stochastic optimization problem over the set of equivalence classes with a natural reward function that quantifies how many graphs are eliminated from the equivalence class by an intervention. We show that this reward function is adaptive submodular and provide a greedy policy with a provable near-optimal performance guarantee. A key technical challenge is to efficiently estimate the reward function without having to explicitly enumerate all the graphs in the equivalence class. We propose a sampling-based estimator using random matchings and analyze its bias and concentration behavior. Our simulation results show that performing a small number of interventions guided by our stochastic optimization framework recovers the true underlying causal structure. 

**Abstract (ZH)**: 从观察数据和干预数据中学习线性非高斯结构方程模型的因果结构：一种基于完美匹配的组合式表征方法 

---
# How Large Language Models Need Symbolism 

**Title (ZH)**: 大型语言模型需要符号主义 

**Authors**: Xiaotie Deng, Hanyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21404)  

**Abstract**: We argue that AI's future requires more than scaling. To unlock genuine discovery, large language models need a compass: human-crafted symbols to guide their powerful but blind intuition. 

**Abstract (ZH)**: 我们认为AI的未来不仅仅依赖于规模的扩大。为了开启真正的发现之旅，大规模语言模型需要一个指南针：人类 crafting 的符号来引导它们强大但盲目的直觉。 

---
# Large AI Model-Enabled Generative Semantic Communications for Image Transmission 

**Title (ZH)**: 基于大型AI模型的生成语义通信在图像传输中的应用 

**Authors**: Qiyu Ma, Wanli Ni, Zhijin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21394)  

**Abstract**: The rapid development of generative artificial intelligence (AI) has introduced significant opportunities for enhancing the efficiency and accuracy of image transmission within semantic communication systems. Despite these advancements, existing methodologies often neglect the difference in importance of different regions of the image, potentially compromising the reconstruction quality of visually critical content. To address this issue, we introduce an innovative generative semantic communication system that refines semantic granularity by segmenting images into key and non-key regions. Key regions, which contain essential visual information, are processed using an image oriented semantic encoder, while non-key regions are efficiently compressed through an image-to-text modeling approach. Additionally, to mitigate the substantial storage and computational demands posed by large AI models, the proposed system employs a lightweight deployment strategy incorporating model quantization and low-rank adaptation fine-tuning techniques, significantly boosting resource utilization without sacrificing performance. Simulation results demonstrate that the proposed system outperforms traditional methods in terms of both semantic fidelity and visual quality, thereby affirming its effectiveness for image transmission tasks. 

**Abstract (ZH)**: 生成式人工智能的迅猛发展为语义通信系统中的图像传输效率和精度提升了重要机会。尽管取得了这些进展，现有的方法往往忽视了图像不同区域重要性的差异，这可能会影响关键视觉内容的重建质量。为了解决这个问题，我们提出了一种创新的生成式语义通信系统，通过将图像分割为关键和非关键区域来细化语义粒度。关键区域包含重要的视觉信息，使用面向图像的语义编码器进行处理，而非关键区域则通过图像到文本建模方法进行高效压缩。此外，为了减轻大型人工智能模型带来的显著存储和计算需求，提出的系统采用了轻量级部署策略，结合模型量化和低秩适应微调技术，显著提升了资源利用率，同时不牺牲性能。仿真结果显示，与传统方法相比，所提出系统在语义保真度和视觉质量方面均表现出更优的效果，从而证实了其在图像传输任务中的有效性。 

---
# MIXRAG : Mixture-of-Experts Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering 

**Title (ZH)**: MIXRAG：专家混合检索增强生成方法及其在文本图理解与问答中的应用 

**Authors**: Lihui Liu, Carl J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21391)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance across a wide range of applications. However, they often suffer from hallucinations in knowledge-intensive domains due to their reliance on static pretraining corpora. To address this limitation, Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external knowledge sources during inference. Among these sources, textual graphs provide structured and semantically rich information that supports more precise and interpretable reasoning. This has led to growing interest in graph-based RAG systems. Despite their potential, most existing approaches rely on a single retriever to identify relevant subgraphs, which limits their ability to capture the diverse aspects of complex queries. Moreover, these systems often struggle to accurately judge the relevance of retrieved content, making them prone to distraction by irrelevant noise. To address these challenges, in this paper, we propose MIXRAG, a Mixture-of-Experts Graph-RAG framework that introduces multiple specialized graph retrievers and a dynamic routing controller to better handle diverse query intents. Each retriever is trained to focus on a specific aspect of graph semantics, such as entities, relations, or subgraph topology. A Mixture-of-Experts module adaptively selects and fuses relevant retrievers based on the input query. To reduce noise in the retrieved information, we introduce a query-aware GraphEncoder that carefully analyzes relationships within the retrieved subgraphs, highlighting the most relevant parts while down-weighting unnecessary noise. Empirical results demonstrate that our method achieves state-of-the-art performance and consistently outperforms various baselines. MIXRAG is effective across a wide range of graph-based tasks in different domains. The code will be released upon paper acceptance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中取得了显著性能，但在知识密集型领域常常因依赖静态预训练数据集而产生幻觉。为解决这一局限，检索增强生成（RAG）通过在推理过程中融入外部知识源来提升LLMs。在这些知识源中，文本图提供了结构化和语义丰富的信息，有助于更加精确和可解释的推理。这导致了基于图的RAG系统的广泛关注。尽管具有潜力，大多数现有方法仍依赖单一检索器来识别相关子图，这限制了它们捕捉复杂查询多方面的能力。此外，这些系统往往难以准确判断检索内容的相关性，容易受到无关噪音的干扰。为应对这些问题，本文提出了一种专家混合图-RAG框架MIXRAG，引入了多个专门化的图检索器和动态路由控制器，以更好地处理多样化的查询意图。每个检索器被训练专注于图语义的特定方面，如实体、关系或子图拓扑。专家混合模块根据输入查询自适应地选择和融合相关检索器。为减少检索信息中的噪音，我们引入了一个查询感知的GraphEncoder，在仔细分析检索到的子图关系时，突出最重要的部分并降低不必要的噪音。实验结果表明，本方法取得了最先进的性能，并且在多种基线指标上表现更优。MIXRAG在不同领域多种基于图的任务中均表现有效。论文被接受后将发布代码。 

---
# Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey 

**Title (ZH)**: 面向网络入侵检测的联邦与量子机器学习适应性研究：一个综述 

**Authors**: Devashish Chaudhary, Sutharshan Rajasegarar, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2509.21389)  

**Abstract**: This survey explores the integration of Federated Learning (FL) with Network Intrusion Detection Systems (NIDS), with particular emphasis on deep learning and quantum machine learning approaches. FL enables collaborative model training across distributed devices while preserving data privacy-a critical requirement in network security contexts where sensitive traffic data cannot be centralized. Our comprehensive analysis systematically examines the full spectrum of FL architectures, deployment strategies, communication protocols, and aggregation methods specifically tailored for intrusion detection. We provide an in-depth investigation of privacy-preserving techniques, model compression approaches, and attack-specific federated solutions for threats including DDoS, MITM, and botnet attacks. The survey further delivers a pioneering exploration of Quantum FL (QFL), discussing quantum feature encoding, quantum machine learning algorithms, and quantum-specific aggregation methods that promise exponential speedups for complex pattern recognition in network traffic. Through rigorous comparative analysis of classical and quantum approaches, identification of research gaps, and evaluation of real-world deployments, we outline a concrete roadmap for industrial adoption and future research directions. This work serves as an authoritative reference for researchers and practitioners seeking to enhance privacy, efficiency, and robustness of federated intrusion detection systems in increasingly complex network environments, while preparing for the quantum-enhanced cybersecurity landscape of tomorrow. 

**Abstract (ZH)**: This survey探讨了联邦学习（FL）与网络入侵检测系统（NIDS）的集成，特别强调了深度学习和量子机器学习方法。FL允许分布式设备跨设备进行协作模型训练，同时保护数据隐私——在网络安全环境中这是一个至关重要的要求，特别是在敏感流量数据不能集中处理的情况下。我们的全面分析系统地 examination了适用于入侵检测的FL架构、部署策略、通信协议和聚合方法的完整范围。我们深入研究了隐私保护技术、模型压缩方法以及针对DDoS、MITM和肉鸡攻击等特定攻击场景的联邦解决方案。此外，本调查还首次探索了量子联邦学习（QFL），讨论了量子特征编码、量子机器学习算法及量子特定的聚合方法，这些方法有望在复杂的网络流量模式识别中实现指数级加速。通过经典方法与量子方法的严格比较分析、研究空白的识别以及现实世界部署的评估，我们为联邦入侵检测系统的工业采用和未来研究方向制定了具体的路线图。本工作为研究人员和从业者提供了一份权威参考，帮助他们在日益复杂的网络环境中增强联邦入侵检测系统的隐私性、效率和鲁棒性，同时也为迈向明日的量子增强网络安全景观做好准备。 

---
# Do Sparse Subnetworks Exhibit Cognitively Aligned Attention? Effects of Pruning on Saliency Map Fidelity, Sparsity, and Concept Coherence 

**Title (ZH)**: 稀疏子网络表现出认知对齐的注意力吗？剪枝对重要性图保真度、稀疏性和概念一致性的影响 

**Authors**: Sanish Suwal, Dipkamal Bhusal, Michael Clifford, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21387)  

**Abstract**: Prior works have shown that neural networks can be heavily pruned while preserving performance, but the impact of pruning on model interpretability remains unclear. In this work, we investigate how magnitude-based pruning followed by fine-tuning affects both low-level saliency maps and high-level concept representations. Using a ResNet-18 trained on ImageNette, we compare post-hoc explanations from Vanilla Gradients (VG) and Integrated Gradients (IG) across pruning levels, evaluating sparsity and faithfulness. We further apply CRAFT-based concept extraction to track changes in semantic coherence of learned concepts. Our results show that light-to-moderate pruning improves saliency-map focus and faithfulness while retaining distinct, semantically meaningful concepts. In contrast, aggressive pruning merges heterogeneous features, reducing saliency map sparsity and concept coherence despite maintaining accuracy. These findings suggest that while pruning can shape internal representations toward more human-aligned attention patterns, excessive pruning undermines interpretability. 

**Abstract (ZH)**: 基于幅度的剪枝对模型可解释性的影响：从低级显著图到高级概念表示的探究 

---
# Toward a Realistic Encoding Model of Auditory Affective Understanding in the Brain 

**Title (ZH)**: 向大脑听觉情感理解的现实编码模型迈进 

**Authors**: Guandong Pan, Yaqian Yang, Shi Chen, Xin Wang, Longzhao Liu, Hongwei Zheng, Shaoting Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21381)  

**Abstract**: In affective neuroscience and emotion-aware AI, understanding how complex auditory stimuli drive emotion arousal dynamics remains unresolved. This study introduces a computational framework to model the brain's encoding of naturalistic auditory inputs into dynamic behavioral/neural responses across three datasets (SEED, LIRIS, self-collected BAVE). Guided by neurobiological principles of parallel auditory hierarchy, we decompose audio into multilevel auditory features (through classical algorithms and wav2vec 2.0/Hubert) from the original and isolated human voice/background soundtrack elements, mapping them to emotion-related responses via cross-dataset analyses. Our analysis reveals that high-level semantic representations (derived from the final layer of wav2vec 2.0/Hubert) exert a dominant role in emotion encoding, outperforming low-level acoustic features with significantly stronger mappings to behavioral annotations and dynamic neural synchrony across most brain regions ($p < 0.05$). Notably, middle layers of wav2vec 2.0/hubert (balancing acoustic-semantic information) surpass the final layers in emotion induction across datasets. Moreover, human voices and soundtracks show dataset-dependent emotion-evoking biases aligned with stimulus energy distribution (e.g., LIRIS favors soundtracks due to higher background energy), with neural analyses indicating voices dominate prefrontal/temporal activity while soundtracks excel in limbic regions. By integrating affective computing and neuroscience, this work uncovers hierarchical mechanisms of auditory-emotion encoding, providing a foundation for adaptive emotion-aware systems and cross-disciplinary explorations of audio-affective interactions. 

**Abstract (ZH)**: 在情感神经科学和情感智能AI领域，理解复杂听觉刺激如何驱动情绪唤醒动力学仍未解决。本研究引入了一个计算框架，以建模大脑将自然听觉输入编码为动态的行为/神经响应（涉及SEED、LIRIS和自收集的BAVE三个数据集）。受神经生物学中并行听觉层次的启发，我们将音频分解为多级听觉特征（使用经典算法和wav2vec 2.0/Hubert），并通过跨数据集分析将这些特征映射到与情感相关的响应。我们的分析表明，来自wav2vec 2.0/Hubert最终层的高级语义表示在情绪编码中起主导作用，与低级声学特征相比，其与行为注释和大多数脑区的动态神经同步表现出更强的映射关系（p < 0.05）。值得注意的是，在不同数据集上，wav2vec 2.0/hubert的中间层（平衡声学-语义信息）在情绪诱导方面优于最终层。此外，人类声音和 soundtrack 在不同数据集中表现出依赖性的情感激发偏差，与刺激能量分布相符（例如，LIRIS 更青睐 soundtrack 因其更高的背景能量），神经分析表明，声音主导前额皮层/颞叶活动，而 soundtrack 占优在边缘系统区域。通过将情感计算与神经科学相结合，本研究揭示了听觉-情绪编码的层次机制，为适应性情感智能系统和跨学科探索音频-情感交互提供了基础。 

---
# SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders 

**Title (ZH)**: SAEmnesia: 用稀疏自编码器在扩散模型中删除概念 

**Authors**: Enrico Cassano, Riccardo Renzulli, Marco Nurisso, Mirko Zaffaroni, Alan Perotti, Marco Grangetto  

**Link**: [PDF](https://arxiv.org/pdf/2509.21379)  

**Abstract**: Effective concept unlearning in text-to-image diffusion models requires precise localization of concept representations within the model's latent space. While sparse autoencoders successfully reduce neuron polysemanticity (i.e., multiple concepts per neuron) compared to the original network, individual concept representations can still be distributed across multiple latent features, requiring extensive search procedures for concept unlearning. We introduce SAEmnesia, a supervised sparse autoencoder training method that promotes one-to-one concept-neuron mappings through systematic concept labeling, mitigating feature splitting and promoting feature centralization. Our approach learns specialized neurons with significantly stronger concept associations compared to unsupervised baselines. The only computational overhead introduced by SAEmnesia is limited to cross-entropy computation during training. At inference time, this interpretable representation reduces hyperparameter search by 96.67% with respect to current approaches. On the UnlearnCanvas benchmark, SAEmnesia achieves a 9.22% improvement over the state-of-the-art. In sequential unlearning tasks, we demonstrate superior scalability with a 28.4% improvement in unlearning accuracy for 9-object removal. 

**Abstract (ZH)**: 文本到图像扩散模型中有效的概念遗忘需要在模型的潜在空间中精确定位概念表示。我们引入SAEmnesia，一种通过系统性概念标签化促进一对一概念-神经元映射的监督稀疏自编码器训练方法，减少特征分裂并促进特征凝聚。与无监督基线相比，该方法学习的概念关联更强。SAEmnesia引入的唯一计算开销是在训练过程中增加交叉熵计算。在推理阶段，这种可解释表示将当前方法的超参数搜索减少96.67%。在UnlearnCanvas基准上，SAEmnesia比最先进的方法提高了9.22%。在序贯遗忘任务中，我们展示了更好的可扩展性，对于9个对象的移除，遗忘准确率提高了28.4%。 

---
# Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation 

**Title (ZH)**: 高效的音频-视觉多目标动态融合导航 

**Authors**: Yinfeng Yu, Hailong Zhang, Meiling Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21377)  

**Abstract**: Audiovisual embodied navigation enables robots to locate audio sources by dynamically integrating visual observations from onboard sensors with the auditory signals emitted by the target. The core challenge lies in effectively leveraging multimodal cues to guide navigation. While prior works have explored basic fusion of visual and audio data, they often overlook deeper perceptual context. To address this, we propose the Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (DMTF-AVN). Our approach uses a multi-target architecture coupled with a refined Transformer mechanism to filter and selectively fuse cross-modal information. Extensive experiments on the Replica and Matterport3D datasets demonstrate that DMTF-AVN achieves state-of-the-art performance, outperforming existing methods in success rate (SR), path efficiency (SPL), and scene adaptation (SNA). Furthermore, the model exhibits strong scalability and generalizability, paving the way for advanced multimodal fusion strategies in robotic navigation. The code and videos are available at
this https URL. 

**Abstract (ZH)**: 视听 embodied 导航使机器人能够通过动态整合搭载传感器的视觉观察与目标发出的听觉信号来定位声源。核心挑战在于有效利用多模态线索来引导导航。尽管先前的工作已经探索了视觉和听觉数据的基本融合，但往往忽视了更深入的感知上下文。为解决这一问题，我们提出了动态多目标融合用于高效视听导航（DMTF-AVN）。我们的方法采用多目标架构并结合精细的Transformer机制来筛选和选择性融合跨模态信息。在Replica和Matterport3D数据集上的 extensive 实验表明，DMTF-AVN 达到了最先进的性能，在成功率达（SR）、路径效率（SPL）和场景适应性（SNA）方面超越了现有方法。此外，模型具有强大的可扩展性和通用性，为机械臂导航中的高级多模态融合策略铺平了道路。代码和视频可在以下链接获取：this https URL。 

---
# In silico Deep Learning Protocols for Label-Free Super-Resolution Microscopy: A Comparative Study of Network Architectures and SNR Dependence 

**Title (ZH)**: 基于计算的深度学习协议在无标记超分辨率显微镜中的比较研究：网络架构和信噪比依赖性分析 

**Authors**: Shiraz S Kaderuppan, Jonathan Mar, Andrew Irvine, Anurag Sharma, Muhammad Ramadan Saifuddin, Wai Leong Eugene Wong, Wai Lok Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21376)  

**Abstract**: The field of optical microscopy spans across numerous industries and research domains, ranging from education to healthcare, quality inspection and analysis. Nonetheless, a key limitation often cited by optical microscopists refers to the limit of its lateral resolution (typically defined as ~200nm), with potential circumventions involving either costly external modules (e.g. confocal scan heads, etc) and/or specialized techniques [e.g. super-resolution (SR) fluorescent microscopy]. Addressing these challenges in a normal (non-specialist) context thus remains an aspect outside the scope of most microscope users & facilities. This study thus seeks to evaluate an alternative & economical approach to achieving SR optical microscopy, involving non-fluorescent phase-modulated microscopical modalities such as Zernike phase contrast (PCM) and differential interference contrast (DIC) microscopy. Two in silico deep neural network (DNN) architectures which we developed previously (termed O-Net and Theta-Net) are assessed on their abilities to resolve a custom-fabricated test target containing nanoscale features calibrated via atomic force microscopy (AFM). The results of our study demonstrate that although both O-Net and Theta-Net seemingly performed well when super-resolving these images, they were complementary (rather than competing) approaches to be considered for image SR, particularly under different image signal-to-noise ratios (SNRs). High image SNRs favoured the application of O-Net models, while low SNRs inclined preferentially towards Theta-Net models. These findings demonstrate the importance of model architectures (in conjunction with the source image SNR) on model performance and the SR quality of the generated images where DNN models are utilized for non-fluorescent optical nanoscopy, even where the same training dataset & number of epochs are being used. 

**Abstract (ZH)**: 光学显微镜领域涵盖了教育、医疗、质量检验与分析等多个行业和研究领域。然而，光显微镜用户时常提及的一个关键限制是其横向分辨率的限制（通常定义为~200nm），这可能导致通过昂贵的外部模块（如共聚焦扫描头等）和/或专门的技术（如超分辨率荧光显微镜）来解决。在非专业人士的常规背景下，解决这些挑战仍然超出了大多数显微镜用户和设施的范围。本研究旨在评估一种替代的经济方法，以实现超分辨率光学显微镜，涉及非荧光相位调制显微成像模式，如Zernike相位对比（PCM）显微镜和相差干涉显微镜（DIC）。我们先前开发了两种深度神经网络（DNN）架构（分别称为O-Net和Theta-Net），并评估了它们在通过原子力显微镜（AFM）校准纳米尺度特征的自定义制造测试目标上的超分辨能力。研究结果表明，尽管O-Net和Theta-Net在这些图像的超分辨方面表现良好，但它们是互补（而非竞争）的超分辨方法，在不同图像信噪比（SNR）下应被考虑。高图像SNR有助于O-Net模型的应用，而低SNR则更偏好Theta-Net模型。这些发现强调了模型架构（与源图像SNR相结合）对模型性能和DNN模型在非荧光光学纳米显微镜中生成的超分辨率图像质量的重要性，即使使用相同的训练数据集和训练周期数也是如此。 

---
# Automated Prompt Generation for Creative and Counterfactual Text-to-image Synthesis 

**Title (ZH)**: 自动提示生成以实现创造性和反事实的文本到图像合成 

**Authors**: Aleksa Jelaca, Ying Jiao, Chang Tian, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2509.21375)  

**Abstract**: Text-to-image generation has advanced rapidly with large-scale multimodal training, yet fine-grained controllability remains a critical challenge. Counterfactual controllability, defined as the capacity to deliberately generate images that contradict common-sense patterns, remains a major challenge but plays a crucial role in enabling creativity and exploratory applications. In this work, we address this gap with a focus on counterfactual size (e.g., generating a tiny walrus beside a giant button) and propose an automatic prompt engineering framework that adapts base prompts into revised prompts for counterfactual images. The framework comprises three components: an image evaluator that guides dataset construction by identifying successful image generations, a supervised prompt rewriter that produces revised prompts, and a DPO-trained ranker that selects the optimal revised prompt. We construct the first counterfactual size text-image dataset and enhance the image evaluator by extending Grounded SAM with refinements, achieving a 114 percent improvement over its backbone. Experiments demonstrate that our method outperforms state-of-the-art baselines and ChatGPT-4o, establishing a foundation for future research on counterfactual controllability. 

**Abstract (ZH)**: 基于文本的图像生成随着大规模多模态训练的进行取得了 rapid进展，但细粒度可控性仍然是一个关键挑战。反事实可控性，定义为能够故意生成与常识模式矛盾的图像的能力，仍然是一个重大挑战但对促进创造力和探索性应用至关重要。在本工作中，我们重点关注反事实尺寸（例如，在一个巨大按钮旁生成一只小型海象），并提出了一种自动提示工程框架，该框架将基础提示调整为生成反事实图像的修订提示。该框架包含三个组件：一个图像评估器，它通过识别成功的图像生成来指导数据集构建；一个监督提示重写器，产生修订提示；以及一个DPO训练的排序器，选择最优的修订提示。我们构建了首个反事实尺寸文本-图像数据集，并通过扩展并改进Grounded SAM来增强图像评估器，实现了其骨干模型114%的改进。实验结果表明，我们的方法优于最先进的基线和ChatGPT-4o，为未来关于反事实可控性的研究奠定了基础。 

---
# ReGeS: Reciprocal Retrieval-Generation Synergy for Conversational Recommender Systems 

**Title (ZH)**: ReGeS: 互逆检索-生成协同作用的对话推荐系统 

**Authors**: Dayu Yang, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21371)  

**Abstract**: Connecting conversation with external domain knowledge is vital for conversational recommender systems (CRS) to correctly understand user preferences. However, existing solutions either require domain-specific engineering, which limits flexibility, or rely solely on large language models, which increases the risk of hallucination. While Retrieval-Augmented Generation (RAG) holds promise, its naive use in CRS is hindered by noisy dialogues that weaken retrieval and by overlooked nuances among similar items. We propose ReGeS, a reciprocal Retrieval-Generation Synergy framework that unifies generation-augmented retrieval to distill informative user intent from conversations and retrieval-augmented generation to differentiate subtle item features. This synergy obviates the need for extra annotations, reduces hallucinations, and simplifies continuous updates. Experiments on multiple CRS benchmarks show that ReGeS achieves state-of-the-art performance in recommendation accuracy, demonstrating the effectiveness of reciprocal synergy for knowledge-intensive CRS tasks. 

**Abstract (ZH)**: 将对话与外部领域知识相连对于对话推荐系统（CRS）正确理解用户偏好至关重要。然而，现有解决方案要么需要领域特定工程，这限制了灵活性，要么仅依赖大型语言模型，增加了语义错误的风险。尽管检索增强生成（RAG）潜力巨大，但在CRS中的直接应用受限于嘈杂的对话削弱了检索效果，并且忽视了相似项目之间的细微差别。我们提出了一种交互式的检索-生成协同（ReGeS）框架，该框架将生成增强检索统一起来，从对话中提炼出有益的用户意图，并将检索增强生成区分开来，突显细微的项目特征。这种协同作用避免了额外注释的需求，减少了语义错误，并简化了持续更新。在多个CRS基准上的实验表明，ReGeS 在推荐准确性方面达到了最先进的性能，证明了交互式协同作用在知识密集型CRS任务中的有效性。 

---
# Safety Assessment of Scaffolding on Construction Site using AI 

**Title (ZH)**: 基于AI的施工现场脚手架安全性评估 

**Authors**: Sameer Prabhu, Amit Patwardhan, Ramin Karim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21368)  

**Abstract**: In the construction industry, safety assessment is vital to ensure both the reliability of assets and the safety of workers. Scaffolding, a key structural support asset requires regular inspection to detect and identify alterations from the design rules that may compromise the integrity and stability. At present, inspections are primarily visual and are conducted by site manager or accredited personnel to identify deviations. However, visual inspection is time-intensive and can be susceptible to human errors, which can lead to unsafe conditions. This paper explores the use of Artificial Intelligence (AI) and digitization to enhance the accuracy of scaffolding inspection and contribute to the safety improvement. A cloud-based AI platform is developed to process and analyse the point cloud data of scaffolding structure. The proposed system detects structural modifications through comparison and evaluation of certified reference data with the recent point cloud data. This approach may enable automated monitoring of scaffolding, reducing the time and effort required for manual inspections while enhancing the safety on a construction site. 

**Abstract (ZH)**: 在建筑行业中，安全评估对于确保资产可靠性及工人安全至关重要。脚手架作为关键的结构支持资产，需要定期检查以检测和识别可能损害结构完整性和稳定性的设计规则变更。目前，检查主要依赖视觉方法，并由现场管理者或认证人员进行以识别偏差。然而，视觉检查耗时且易受人为错误的影响，可能导致不安全条件。本文探讨了使用人工智能（AI）和数字化技术以提高脚手架检查的准确性并促进安全改进。开发了一种基于云的AI平台，用于处理和分析脚手架结构的点云数据。所提出系统通过将认证参考数据与最近的点云数据进行比较和评估来检测结构修改。该方法可能实现脚手架的自动化监控，减少手动检查所需的时间和努力，同时提高施工现场的安全性。 

---
# Design and Implementation of a Secure RAG-Enhanced AI Chatbot for Smart Tourism Customer Service: Defending Against Prompt Injection Attacks -- A Case Study of Hsinchu, Taiwan 

**Title (ZH)**: 基于 Taiwanese Hsinchu 案例研究的智能旅游客户服务中安全增强的 RAG 加强 AI 聊天机器人设计与实现——对抗提示注入攻击的研究 

**Authors**: Yu-Kai Shih, You-Kai Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21367)  

**Abstract**: As smart tourism evolves, AI-powered chatbots have become indispensable for delivering personalized, real-time assistance to travelers while promoting sustainability and efficiency. However, these systems are increasingly vulnerable to prompt injection attacks, where adversaries manipulate inputs to elicit unintended behaviors such as leaking sensitive information or generating harmful content. This paper presents a case study on the design and implementation of a secure retrieval-augmented generation (RAG) chatbot for Hsinchu smart tourism services. The system integrates RAG with API function calls, multi-layered linguistic analysis, and guardrails against injections, achieving high contextual awareness and security. Key features include a tiered response strategy, RAG-driven knowledge grounding, and intent decomposition across lexical, semantic, and pragmatic levels. Defense mechanisms include system norms, gatekeepers for intent judgment, and reverse RAG text to prioritize verified data. We also benchmark a GPT-5 variant (released 2025-08-07) to assess inherent robustness. Evaluations with 674 adversarial prompts and 223 benign queries show over 95% accuracy on benign tasks and substantial detection of injection attacks. GPT-5 blocked about 85% of attacks, showing progress yet highlighting the need for layered defenses. Findings emphasize contributions to sustainable tourism, multilingual accessibility, and ethical AI deployment. This work offers a practical framework for deploying secure chatbots in smart tourism and contributes to resilient, trustworthy AI applications. 

**Abstract (ZH)**: 随着智能旅游的发展，基于AI的聊天机器人已成为提供个性化实时协助、促进可持续性和效率的重要工具。然而，这些系统越来越容易受到提示注入攻击的影响，攻击者通过操纵输入来诱发出意外行为，如泄露敏感信息或生成有害内容。本文呈现了一个案例研究，介绍了一种安全的检索增强生成（RAG）聊天机器人在新竹智能旅游服务中的设计与实现。该系统结合了RAG、API功能调用、多层次语言分析以及对注入的防护措施，实现了高度的语境意识和安全性。关键功能包括分层响应策略、RAG驱动的知识定位以及在词汇、语义和语用层面的意图分解。防御机制包括系统规范、意图判断的守门人以及逆向RAG文本以优先验证数据。我们还对2025年8月7日发布的GPT-5变体进行基准测试，以评估其固有的鲁棒性。使用674个 adversarial 提示和223个良性查询的评估结果显示，在良性任务上的准确率超过95%，并对注入攻击有显著检测。GPT-5阻挡了约85%的攻击，显示出进展但同时也指出了多层次防御的必要性。研究强调了对可持续旅游、多语言可访问性和负责任AI部署的贡献。该工作为在智能旅游中部署安全聊天机器人提供了一个实用框架，促进了稳健且可信赖的AI应用。 

---
# MAJORScore: A Novel Metric for Evaluating Multimodal Relevance via Joint Representation 

**Title (ZH)**: MAJORScore: 一种基于联合表示的多模态相关性评价新指标 

**Authors**: Zhicheng Du, Qingyang Shi, Jiasheng Lu, Yingshan Liang, Xinyu Zhang, Yiran Wang, Peiwu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21365)  

**Abstract**: The multimodal relevance metric is usually borrowed from the embedding ability of pretrained contrastive learning models for bimodal data, which is used to evaluate the correlation between cross-modal data (e.g., CLIP). However, the commonly used evaluation metrics are only suitable for the associated analysis between two modalities, which greatly limits the evaluation of multimodal similarity. Herein, we propose MAJORScore, a brand-new evaluation metric for the relevance of multiple modalities (N modalities, N>=3) via multimodal joint representation for the first time. The ability of multimodal joint representation to integrate multiple modalities into the same latent space can accurately represent different modalities at one scale, providing support for fair relevance scoring. Extensive experiments have shown that MAJORScore increases by 26.03%-64.29% for consistent modality and decreases by 13.28%-20.54% for inconsistence compared to existing methods. MAJORScore serves as a more reliable metric for evaluating similarity on large-scale multimodal datasets and multimodal model performance evaluation. 

**Abstract (ZH)**: 多模态相关性评价指标MAJORScore：通过多模态联合表示首次评估N种及以上模态的相关性 

---
# A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision--Revised 

**Title (ZH)**: 一种基于交织多监督的互学习显著目标检测方法——修订版 

**Authors**: Runmin Wu, Mengyang Feng, Wenlong Guan, Dong Wang, Huchuan Lu, Errui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.21363)  

**Abstract**: Though deep learning techniques have made great progress in salient object detection recently, the predicted saliency maps still suffer from incomplete predictions due to the internal complexity of objects and inaccurate boundaries caused by strides in convolution and pooling operations. To alleviate these issues, we propose to train saliency detection networks by exploiting the supervision from not only salient object detection, but also foreground contour detection and edge detection. First, we leverage salient object detection and foreground contour detection tasks in an intertwined manner to generate saliency maps with uniform highlight. Second, the foreground contour and edge detection tasks guide each other simultaneously, thereby leading to precise foreground contour prediction and reducing the local noises for edge prediction. In addition, we develop a novel mutual learning module (MLM) which serves as the building block of our method. Each MLM consists of multiple network branches trained in a mutual learning manner, which improves the performance by a large margin. Extensive experiments on seven challenging datasets demonstrate that the proposed method has delivered state-of-the-art results in both salient object detection and edge detection. 

**Abstract (ZH)**: 尽管深度学习技术最近在显著对象检测方面取得了显著进展，但由于对象内部复杂性和卷积和平池化操作引起的不准确边界，预测的显著性图仍然存在不完整的问题。为了解决这些问题，我们提出通过利用显著对象检测、前景轮廓检测和边缘检测的监督来训练显著性检测网络。首先，我们以交织的方式利用显著对象检测和前景轮廓检测任务生成均匀凸显的显著性图。其次，前景轮廓检测和边缘检测任务互相引导，从而实现精确的前景轮廓预测并减少边缘预测中的局部噪声。此外，我们开发了一种新的相互学习模块（MLM），作为我们方法的基本构建块。每一MLM包含多个以相互学习方式训练的网络分支，显著提高了性能。在七个挑战性数据集上的广泛实验表明，所提出的方法在显著对象检测和边缘检测中均取得了最先进的结果。 

---
# Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs 

**Title (ZH)**: 上下文即你所需：LLM在现实世界限制下的最大有效上下文窗口 

**Authors**: Norman Paulsen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21361)  

**Abstract**: Large language model (LLM) providers boast big numbers for maximum context window sizes. To test the real world use of context windows, we 1) define a concept of maximum effective context window, 2) formulate a testing method of a context window's effectiveness over various sizes and problem types, and 3) create a standardized way to compare model efficacy for increasingly larger context window sizes to find the point of failure. We collected hundreds of thousands of data points across several models and found significant differences between reported Maximum Context Window (MCW) size and Maximum Effective Context Window (MECW) size. Our findings show that the MECW is, not only, drastically different from the MCW but also shifts based on the problem type. A few top of the line models in our test group failed with as little as 100 tokens in context; most had severe degradation in accuracy by 1000 tokens in context. All models fell far short of their Maximum Context Window by as much as 99 percent. Our data reveals the Maximum Effective Context Window shifts based on the type of problem provided, offering clear and actionable insights into how to improve model accuracy and decrease model hallucination rates. 

**Abstract (ZH)**: 大型语言模型（LLM）提供商宣称其最大上下文窗口大小庞大。为了测试上下文窗口的实际使用效果，我们1）定义了最大有效上下文窗口的概念，2）提出了评估不同大小和问题类型下上下文窗口效果的方法，3）创建了一种标准化方法，以比较不断增加的上下文窗口大小下的模型效果，找到失败点。我们在多个模型中收集了数十万数据点，发现报告的最大上下文窗口（MCW）大小与最大有效上下文窗口（MECW）大小之间存在显著差异。我们的研究发现，MECW不仅与MCW相差甚远，还因问题类型的不同而变化。测试组中的几款顶级模型在仅含100个token的上下文中就失败了；大多数模型在包含1000个token的上下文中准确度显著下降。所有模型的表现远远低于其最大上下文窗口99%以上。我们的数据揭示了最大有效上下文窗口会根据提供的问题类型变化，提供了明确且实用的见解，以提高模型精度并降低模型产生幻觉的频率。 

---
# Multimodal Prompt Decoupling Attack on the Safety Filters in Text-to-Image Models 

**Title (ZH)**: 多模态提示解藕攻击文本到图像模型中的安全过滤器 

**Authors**: Xingkai Peng, Jun Jiang, Meng Tong, Shuai Li, Weiming Zhang, Nenghai Yu, Kejiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21360)  

**Abstract**: Text-to-image (T2I) models have been widely applied in generating high-fidelity images across various domains. However, these models may also be abused to produce Not-Safe-for-Work (NSFW) content via jailbreak attacks. Existing jailbreak methods primarily manipulate the textual prompt, leaving potential vulnerabilities in image-based inputs largely unexplored. Moreover, text-based methods face challenges in bypassing the model's safety filters. In response to these limitations, we propose the Multimodal Prompt Decoupling Attack (MPDA), which utilizes image modality to separate the harmful semantic components of the original unsafe prompt. MPDA follows three core steps: firstly, a large language model (LLM) decouples unsafe prompts into pseudo-safe prompts and harmful prompts. The former are seemingly harmless sub-prompts that can bypass filters, while the latter are sub-prompts with unsafe semantics that trigger filters. Subsequently, the LLM rewrites the harmful prompts into natural adversarial prompts to bypass safety filters, which guide the T2I model to modify the base image into an NSFW output. Finally, to ensure semantic consistency between the generated NSFW images and the original unsafe prompts, the visual language model generates image captions, providing a new pathway to guide the LLM in iterative rewriting and refining the generated content. 

**Abstract (ZH)**: 多模态提示解耦攻击（MPDA）：利用图像模态分离原始不安全提示中的有害语义组件 

---
# Influence Guided Context Selection for Effective Retrieval-Augmented Generation 

**Title (ZH)**: 基于影响的上下文选择以实现有效的检索增强生成 

**Authors**: Jiale Deng, Yanyan Shen, Ziyuan Pei, Youmin Chen, Linpeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21359)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses large language model (LLM) hallucinations by grounding responses in external knowledge, but its effectiveness is compromised by poor-quality retrieved contexts containing irrelevant or noisy information. While existing approaches attempt to improve performance through context selection based on predefined context quality assessment metrics, they show limited gains over standard RAG. We attribute this limitation to their failure in holistically utilizing available information (query, context list, and generator) for comprehensive quality assessment. Inspired by recent advances in data selection, we reconceptualize context quality assessment as an inference-time data valuation problem and introduce the Contextual Influence Value (CI value). This novel metric quantifies context quality by measuring the performance degradation when removing each context from the list, effectively integrating query-aware relevance, list-aware uniqueness, and generator-aware alignment. Moreover, CI value eliminates complex selection hyperparameter tuning by simply retaining contexts with positive CI values. To address practical challenges of label dependency and computational overhead, we develop a parameterized surrogate model for CI value prediction during inference. The model employs a hierarchical architecture that captures both local query-context relevance and global inter-context interactions, trained through oracle CI value supervision and end-to-end generator feedback. Extensive experiments across 8 NLP tasks and multiple LLMs demonstrate that our context selection method significantly outperforms state-of-the-art baselines, effectively filtering poor-quality contexts while preserving critical information. Code is available at this https URL. 

**Abstract (ZH)**: 基于检索的生成（RAG）通过将响应 grounding 在外部知识中来解决大型语言模型（LLM）的幻觉问题，但其效果因检索到的包含无关或噪声信息的低质量上下文而受到损害。现有的方法试图通过基于预定义的上下文质量评估指标进行上下文选择来提高性能，但它们在标准RAG上的提升有限。我们将这一局限归因于它们未能全面利用可用信息（查询、上下文列表和生成器）进行综合质量评估。受到数据选择最近进展的启发，我们将上下文质量评估重新概念化为推理时的数据价值问题，并引入了上下文影响值（CI值）。这一新型度量通过测量删除列表中每个上下文时性能的下降来量化上下文质量，有效整合了查询相关的相关性、列表相关的唯一性和生成器相关的对齐性。此外，CI值通过仅保留具有正值CI值的上下文来消除复杂的上下文选择超参数调优。为解决标签依赖性和计算开销的实际挑战，在推理期间我们开发了一种参数化替代模型来预测CI值。该模型采用层次架构，同时捕捉局部查询-上下文相关性和全局上下文交互性，并通过先验CI值监督和端到端生成器反馈进行训练。在8个NLP任务和多种LLM上的广泛实验表明，我们的上下文选择方法显著优于最先进的基线方法，在有效过滤低质量上下文的同时保留关键信息。代码可从以下链接获取。 

---
# MDF-MLLM: Deep Fusion Through Cross-Modal Feature Alignment for Contextually Aware Fundoscopic Image Classification 

**Title (ZH)**: MDF-MLLM：跨模态特征对齐的深度融合方法用于上下文感知眼底图像分类 

**Authors**: Jason Jordan, Mohammadreza Akbari Lor, Peter Koulen, Mei-Ling Shyu, Shu-Ching Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21358)  

**Abstract**: This study aimed to enhance disease classification accuracy from retinal fundus images by integrating fine-grained image features and global textual context using a novel multimodal deep learning architecture. Existing multimodal large language models (MLLMs) often struggle to capture low-level spatial details critical for diagnosing retinal diseases such as glaucoma, diabetic retinopathy, and retinitis pigmentosa. This model development and validation study was conducted on 1,305 fundus image-text pairs compiled from three public datasets (FIVES, HRF, and StoneRounds), covering acquired and inherited retinal diseases, and evaluated using classification accuracy and F1-score. The MDF-MLLM integrates skip features from four U-Net encoder layers into cross-attention blocks within a LLaMA 3.2 11B MLLM. Vision features are patch-wise projected and fused using scaled cross-attention and FiLM-based U-Net modulation. Baseline MLLM achieved 60% accuracy on the dual-type disease classification task. MDF-MLLM, with both U-Net and MLLM components fully fine-tuned during training, achieved a significantly higher accuracy of 94%, representing a 56% improvement. Recall and F1-scores improved by as much as 67% and 35% over baseline, respectively. Ablation studies confirmed that the multi-depth fusion approach contributed to substantial gains in spatial reasoning and classification, particularly for inherited diseases with rich clinical text. MDF-MLLM presents a generalizable, interpretable, and modular framework for fundus image classification, outperforming traditional MLLM baselines through multi-scale feature fusion. The architecture holds promise for real-world deployment in clinical decision support systems. Future work will explore synchronized training techniques, a larger pool of diseases for more generalizability, and extending the model for segmentation tasks. 

**Abstract (ZH)**: 本研究旨在通过结合纤细的图像特征和全局文本上下文，利用新颖的多模态深度学习架构提高从视网膜底片图像中进行疾病分类的准确性。现有的多模态大型语言模型（MLLM）在捕捉用于诊断青光眼、糖尿病视网膜病变和色素性视网膜炎等视网膜疾病的低级空间细节方面常常力不从心。该模型开发与验证研究在来自三个公开数据集（FIVES、HRF和StoneRounds）的1,305张视网膜底片图像-文本对上进行，涵盖获得性和遗传性视网膜疾病，并使用分类准确率和F1得分进行评估。MDF-MLLM 将四个U-Net编码器层的跳连特征整合到LLaMA 3.2 11B MLLM的交叉注意力模块中。视觉特征通过缩放交叉注意力和基于FiLM的U-Net调制逐块投影和融合。基于的MLLM在双类型疾病分类任务中实现了60%的准确率。MDF-MLLM，在训练过程中将U-Net和MLLM组件完全微调后，实现了显著更高的准确率94%，代表了56%的提升。召回率和F1得分分别提高了67%和35%。消融研究证实了多深度融合方法在空间推理和分类中的重要贡献，尤其是在临床文本丰富的遗传性疾病中。MDF-MLLM 提供了一种通用、可解释且模块化的视网膜底片图像分类框架，通过多尺度特征融合超越了传统的MLLM基线。该架构在临床决策支持系统中的实际部署中具有前景。未来的工作将探索同步训练技术、更广泛的疾病池以提高通用性，并扩展模型以进行分割任务。 

---
# A Novel Differential Feature Learning for Effective Hallucination Detection and Classification 

**Title (ZH)**: 一种新型差异特征学习方法，用于有效幻觉检测与分类 

**Authors**: Wenkai Wang, Vincent Lee, Yizhen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21357)  

**Abstract**: Large language model hallucination represents a critical challenge where outputs deviate from factual accuracy due to distributional biases in training data. While recent investigations establish that specific hidden layers exhibit differences between hallucinatory and factual content, the precise localization of hallucination signals within layers remains unclear, limiting the development of efficient detection methods. We propose a dual-model architecture integrating a Projected Fusion (PF) block for adaptive inter-layer feature weighting and a Differential Feature Learning (DFL) mechanism that identifies discriminative features by computing differences between parallel encoders learning complementary representations from identical inputs. Through systematic experiments across HaluEval's question answering, dialogue, and summarization datasets, we demonstrate that hallucination signals concentrate in highly sparse feature subsets, achieving significant accuracy improvements on question answering and dialogue tasks. Notably, our analysis reveals a hierarchical "funnel pattern" where shallow layers exhibit high feature diversity while deep layers demonstrate concentrated usage, enabling detection performance to be maintained with minimal degradation using only 1\% of feature dimensions. These findings suggest that hallucination signals are more concentrated than previously assumed, offering a pathway toward computationally efficient detection systems that could reduce inference costs while maintaining accuracy. 

**Abstract (ZH)**: 大规模语言模型幻觉代表了一个关键挑战，其中输出由于训练数据中的分布偏见而偏离事实准确性。虽然近期研究证实特定隐藏层在幻觉性和事实性内容之间存在差异，但幻觉信号在层内的精确定位仍然模糊，限制了高效检测方法的发展。我们提出了一种双模型架构，结合了适应性跨层特征加权的投影融合（PF）模块和差异特征学习（DFL）机制，通过计算并行编码器从相同输入学习互补表示之间的差异来识别判别性特征。通过在HaluEval的问答、对话和总结数据集上的系统实验，我们证明幻觉信号集中在高度稀疏的特征子集中，显著提高了问答和对话任务的准确性。值得注意的是，我们分析揭示了一个分层的“漏斗模式”，其中浅层显示出高特征多样性而深层显示出特征集中使用，仅使用1%的特征维度即可维持检测性能最小的退化。这些发现表明，幻觉信号比之前认为的更为集中，提供了一条通往计算上更高效的检测系统的途径，能够在降低成本的同时保持准确性。 

---
# Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports 

**Title (ZH)**: 基于短语的地基的胸部X光报告事实核查 

**Authors**: Razi Mahmood, Diego Machado-Reyes, Joy Wu, Parisa Kaviani, Ken C.L. Wong, Niharika D'Souza, Mannudeep Kalra, Ge Wang, Pingkun Yan, Tanveer Syeda-Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2509.21356)  

**Abstract**: With the emergence of large-scale vision language models (VLM), it is now possible to produce realistic-looking radiology reports for chest X-ray images. However, their clinical translation has been hampered by the factual errors and hallucinations in the produced descriptions during inference. In this paper, we present a novel phrase-grounded fact-checking model (FC model) that detects errors in findings and their indicated locations in automatically generated chest radiology reports.
Specifically, we simulate the errors in reports through a large synthetic dataset derived by perturbing findings and their locations in ground truth reports to form real and fake findings-location pairs with images. A new multi-label cross-modal contrastive regression network is then trained on this dataset. We present results demonstrating the robustness of our method in terms of accuracy of finding veracity prediction and localization on multiple X-ray datasets. We also show its effectiveness for error detection in reports of SOTA report generators on multiple datasets achieving a concordance correlation coefficient of 0.997 with ground truth-based verification, thus pointing to its utility during clinical inference in radiology workflows. 

**Abstract (ZH)**: 大规模视觉语言模型的出现使得生成胸部X光片的现实主义医疗报告成为可能。然而，在推断过程中生成的描述中事实错误和幻觉限制了其临床转化。本文提出了一种新颖的短语本地区检查模型（FC模型），用于检测自动生成的胸部放射学报告中发现结果及其指示位置的错误。具体来说，我们通过一个大规模的合成数据集来模拟报告中的错误，该数据集是通过对真实报告中的发现和位置进行扰动形成的，形成了包含图像的真实和虚假发现-位置配对。然后，在该数据集上训练了一个新的多标签跨模态对比回归网络。我们展示了该方法在多个X光数据集上的准确性预测和定位的稳健性结果。我们还展示了其在多个数据集上对SOTA报告生成器报告中的错误检测的有效性，通过基于真实性的验证达到了0.997的 Kendall一致性相关系数，从而表明其在放射学工作流程中的临床推断过程中的实用性。 

---
# Domain-Informed Genetic Superposition Programming: A Case Study on SFRC Beams 

**Title (ZH)**: 基于领域知识的遗传超position编程：SFRC 梁的案例研究 

**Authors**: Mohammad Sadegh Khorshidi, Navid Yazdanjue, Hassan Gharoun, Mohammad Reza Nikoo, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21355)  

**Abstract**: This study presents domain-informed genetic superposition programming (DIGSP), a symbolic regression framework tailored for engineering systems governed by separable physical mechanisms. DIGSP partitions the input space into domain-specific feature subsets and evolves independent genetic programming (GP) populations to model material-specific effects. Early evolution occurs in isolation, while ensemble fitness promotes inter-population cooperation. To enable symbolic superposition, an adaptive hierarchical symbolic abstraction mechanism (AHSAM) is triggered after stagnation across all populations. AHSAM performs analysis of variance- (ANOVA) based filtering to identify statistically significant individuals, compresses them into symbolic constructs, and injects them into all populations through a validation-guided pruning cycle. The DIGSP is benchmarked against a baseline multi-gene genetic programming (BGP) model using a dataset of steel fiber-reinforced concrete (SFRC) beams. Across 30 independent trials with 65% training, 10% validation, and 25% testing splits, DIGSP consistently outperformed BGP in training and test root mean squared error (RMSE). The Wilcoxon rank-sum test confirmed statistical significance (p < 0.01), and DIGSP showed tighter error distributions and fewer outliers. No significant difference was observed in validation RMSE due to limited sample size. These results demonstrate that domain-informed structural decomposition and symbolic abstraction improve convergence and generalization. DIGSP offers a principled and interpretable modeling strategy for systems where symbolic superposition aligns with the underlying physical structure. 

**Abstract (ZH)**: 基于领域知识的遗传超position编程：一种适用于受可分物理机制控制的工程系统的目标函数回归框架 

---
# KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache 

**Title (ZH)**: KV-高效视觉语言模型：基于RNN门控分块键值缓存的方法 

**Authors**: Wanshun Xu, Long Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21354)  

**Abstract**: Vision-Language-Action (VLA) models promise unified robotic perception and control, yet their scalability is constrained by the quadratic cost of attention and the unbounded growth of key-value (KV) memory during long-horizon inference. While recent methods improve generalization through scaling backbone architectures, they often neglect the inference inefficiencies critical to real-time deployment. In this work, we present KV-Efficient VLA, a model-agnostic memory compression framework that addresses these limitations by introducing a lightweight, training-friendly mechanism to selectively retain high-utility context. Our method partitions the KV cache into fixed size chunks and employs a recurrent gating module to summarize and filter historical context according to learned utility scores. This design preserves recent fine-grained detail while aggressively pruning stale, low-relevance memory, all while maintaining causality. Theoretically, KV-Efficient VLA yields up to 1.21x inference speedup and 36% KV memory reduction, with minimal impact on task success. Our method integrates seamlessly into existing autoregressive and hybrid VLA stacks, enabling scalable inference without modifying training pipelines or downstream control logic. 

**Abstract (ZH)**: KV-Efficient Vision-Language-Action模型：一种兼顾推理效率和记忆压缩的框架 

---
# Random Direct Preference Optimization for Radiography Report Generation 

**Title (ZH)**: 随机直接偏好优化在放射报告生成中的应用 

**Authors**: Valentin Samokhin, Boris Shirokikh, Mikhail Goncharov, Dmitriy Umerenkov, Maksim Bobrin, Ivan Oseledets, Dmitry Dylov, Mikhail Belyaev  

**Link**: [PDF](https://arxiv.org/pdf/2509.21351)  

**Abstract**: Radiography Report Generation (RRG) has gained significant attention in medical image analysis as a promising tool for alleviating the growing workload of radiologists. However, despite numerous advancements, existing methods have yet to achieve the quality required for deployment in real-world clinical settings. Meanwhile, large Visual Language Models (VLMs) have demonstrated remarkable progress in the general domain by adopting training strategies originally designed for Large Language Models (LLMs), such as alignment techniques. In this paper, we introduce a model-agnostic framework to enhance RRG accuracy using Direct Preference Optimization (DPO). Our approach leverages random contrastive sampling to construct training pairs, eliminating the need for reward models or human preference annotations. Experiments on supplementing three state-of-the-art models with our Random DPO show that our method improves clinical performance metrics by up to 5%, without requiring any additional training data. 

**Abstract (ZH)**: 放射学报告生成（RRG）在医学影像分析中引起了广泛关注，作为减轻放射学家工作负荷的有希望的工具。然而，尽管取得了众多进展，现有方法仍未达到在真实临床环境中部署所需的质量标准。同时，大规模视觉语言模型（VLMs）通过采用为大规模语言模型（LLMs）设计的训练策略，如对齐技术，在通用领域已经取得了显著进步。在本文中，我们提出了一种模型无关的框架，通过直接偏好优化（DPO）增强RRG的准确性。我们的方法利用随机对比采样构建训练对，从而消除对奖励模型或人工偏好注解的需求。实验表明，将我们的随机DPO应用于三种最先进的模型，可以提高临床性能指标最多5%，而无需额外的训练数据。 

---
# SGNNBench: A Holistic Evaluation of Spiking Graph Neural Network on Large-scale Graph 

**Title (ZH)**: SGNNBench：大规模图上脉冲图神经网络的综合评估 

**Authors**: Huizhe Zhang, Jintang Li, Yuchang Zhu, Liang Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21342)  

**Abstract**: Graph Neural Networks (GNNs) are exemplary deep models designed for graph data. Message passing mechanism enables GNNs to effectively capture graph topology and push the performance boundaries across various graph tasks. However, the trend of developing such complex machinery for graph representation learning has become unsustainable on large-scale graphs. The computational and time overhead make it imperative to develop more energy-efficient GNNs to cope with the explosive growth of real-world graphs. Spiking Graph Neural Networks (SGNNs), which integrate biologically plausible learning via unique spike-based neurons, have emerged as a promising energy-efficient alternative. Different layers communicate with sparse and binary spikes, which facilitates computation and storage of intermediate graph representations. Despite the proliferation of SGNNs proposed in recent years, there is no systematic benchmark to explore the basic design principles of these brain-inspired networks on the graph data. To bridge this gap, we present SGNNBench to quantify progress in the field of SGNNs. Specifically, SGNNBench conducts an in-depth investigation of SGNNs from multiple perspectives, including effectiveness, energy efficiency, and architectural design. We comprehensively evaluate 9 state-of-the-art SGNNs across 18 datasets. Regarding efficiency, we empirically compare these baselines w.r.t model size, memory usage, and theoretical energy consumption to reveal the often-overlooked energy bottlenecks of SGNNs. Besides, we elaborately investigate the design space of SGNNs to promote the development of a general SGNN paradigm. 

**Abstract (ZH)**: 基于脉冲的图神经网络（SGNNs）：原理与进展评估基准 

---
# From Embeddings to Equations: Genetic-Programming Surrogates for Interpretable Transformer Classification 

**Title (ZH)**: 从嵌入到方程：遗传编程代理模型实现可解释的变压器分类 

**Authors**: Mohammad Sadegh Khorshidi, Navid Yazdanjue, Hassan Gharoun, Mohammad Reza Nikoo, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21341)  

**Abstract**: We study symbolic surrogate modeling of frozen Transformer embeddings to obtain compact, auditable classifiers with calibrated probabilities. For five benchmarks (SST2G, 20NG, MNIST, CIFAR10, MSC17), embeddings from ModernBERT, DINOv2, and SigLIP are partitioned on the training set into disjoint, information-preserving views via semantic-preserving feature partitioning (SPFP). A cooperative multi-population genetic program (MEGP) then learns additive, closed-form logit programs over these views. Across 30 runs per dataset we report F1, AUC, log-loss, Brier, expected calibration error (ECE), and symbolic complexity; a canonical model is chosen by a one-standard-error rule on validation F1 with a parsimony tie-break. Temperature scaling fitted on validation yields substantial ECE reductions on test. The resulting surrogates achieve strong discrimination (up to F1 around 0.99 on MNIST, CIFAR10, MSC17; around 0.95 on SST2G), while 20NG remains most challenging. We provide reliability diagrams, dimension usage and overlap statistics, contribution-based importances, and global effect profiles (PDP and ALE), demonstrating faithful, cross-modal explanations grounded in explicit programs. 

**Abstract (ZH)**: 我们研究冻结Transformer嵌入的符号替代建模，以获取紧凑、可审计且概率校准的分类器。对于五个基准（SST2G、20NG、MNIST、CIFAR10、MSC17），使用语义保持特征分区（SPFP）将ModernBERT、DINOv2和SigLIP的嵌入在训练集上分区为互不相交但信息保持的观点。然后，合作多种群遗传程序（MEGP）在这些视图上学习加性、闭式的形式评分程序。我们在每个数据集的30次运行中报告F1、AUC、log-loss、Brier、预期校准误差（ECE）和符号复杂度；通过验证F1的一标准误差规则选择模型，并在必要时通过简化性规则进行区分。针对验证数据集拟合的温度缩放在测试集上显著降低了ECE。生成的替代模型实现了强大的区分能力（MNIST、CIFAR10、MSC17上F1最高可达0.99；SST2G上约为0.95），而20NG仍然是最具挑战性的。我们提供了可靠性图、维度使用和重叠统计、基于贡献的重要性分析以及全局效应轮廓（PDP和ALE），证明了基于明确程序的真实且跨模态的解释。 

---
# Cycle is All You Need: More Is Different 

**Title (ZH)**: 循环即所需一切：更多即不同 

**Authors**: Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21340)  

**Abstract**: We propose an information-topological framework in which cycle closure is the fundamental mechanism of memory and consciousness. Memory is not a static store but the ability to re-enter latent cycles in neural state space, with invariant cycles serving as carriers of meaning by filtering order-specific noise and preserving what persists across contexts. The dot-cycle dichotomy captures this: transient dots scaffold exploration, while nontrivial cycles encode low-entropy content invariants that stabilize memory. Biologically, polychronous neural groups realize 1-cycles through delay-locked spiking reinforced by STDP, nested within theta-gamma rhythms that enforce boundary cancellation. These micro-cycles compose hierarchically, extending navigation loops into general memory and cognition. The perception-action cycle introduces high-order invariance: closure holds even across sense-act alternations, generalizing ancestral homing behavior. Sheaf-cosheaf duality formalizes this process: sheaves glue perceptual fragments into global sections, cosheaves decompose global plans into actions and closure aligns top-down predictions with bottom-up cycles. Consciousness then arises as the persistence of high-order invariants that integrate (unity) yet differentiate (richness) across contexts. We conclude that cycle is all you need: persistent invariants enable generalization in non-ergodic environments with long-term coherence at minimal energetic cost. 

**Abstract (ZH)**: 我们提出一种基于信息拓扑的框架，其中循环闭合是记忆和意识的基本机制。记忆不是静态存储，而是重新进入神经状态空间中潜藏循环的能力，不变的循环作为承载意义的载体，通过过滤特定于顺序的噪声并保留跨情境持续存在的内容。点-循环二分法捕捉这一点：短暂的点支撑探索，而非平凡的循环编码低熵内容不变量，稳定记忆。从生物学上看，多时相神经群体通过STDP强化的延迟锁定放电实现1-循环，并嵌套在theta-γ节奏中，以确保边界取消。这些微循环逐级组合，将导航循环扩展为普遍的记忆和认知。感知-行动循环引入高阶不变性：即使在感觉-行动交替中，闭合亦能保持一致，从而泛化祖先的回巢行为。层kelas-层kelas对偶形式化这一过程：层kelas将知觉片段胶合为全局片段，层kelas分解全局计划为行动，并通过闭合对齐自上而下的预测与自下而上的循环。随之，意识作为持久的高阶不变量在不同情境中统一（整体性）又差异化（丰富性）。我们得出结论：循环即一切所需：持久不变量能够以最小能量成本在非遍历环境中实现泛化，并保持长期一致性。 

---
# Cross-Modal Retrieval with Cauchy-Schwarz Divergence 

**Title (ZH)**: Cauchy-Schwarz 散度下的跨模态检索 

**Authors**: Jiahao Zhang, Wenzhe Yin, Shujian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21339)  

**Abstract**: Effective cross-modal retrieval requires robust alignment of heterogeneous data types. Most existing methods focus on bi-modal retrieval tasks and rely on distributional alignment techniques such as Kullback-Leibler divergence, Maximum Mean Discrepancy, and correlation alignment. However, these methods often suffer from critical limitations, including numerical instability, sensitivity to hyperparameters, and their inability to capture the full structure of the underlying distributions. In this paper, we introduce the Cauchy-Schwarz (CS) divergence, a hyperparameter-free measure that improves both training stability and retrieval performance. We further propose a novel Generalized CS (GCS) divergence inspired by Hölder's inequality. This extension enables direct alignment of three or more modalities within a unified mathematical framework through a bidirectional circular comparison scheme, eliminating the need for exhaustive pairwise comparisons. Extensive experiments on six benchmark datasets demonstrate the effectiveness of our method in both bi-modal and tri-modal retrieval tasks. The code of our CS/GCS divergence is publicly available at this https URL. 

**Abstract (ZH)**: 有效的跨模态检索需要稳健的异构数据类型对齐。大多数现有方法专注于双模态检索任务，并依赖于如KL散度、最大均值偏差和相关对齐等分布对齐技术。然而，这些方法通常遭受关键限制，包括数值不稳定、对超参数敏感以及无法捕捉底层分布的全部结构。在此论文中，我们引入了Cauchy-Schwarz (CS) 分散度，这是一种无超参数的度量，能够提高训练稳定性和检索性能。我们还提出了一个受Hölder不等式启发的广义CS (GCS) 分散度。这一扩展可以通过双向循环比较方案直接对齐三个或更多模态，消除了所有两两比较的需要。在六个基准数据集上的大量实验表明，我们的方法在双模态和三模态检索任务中均有效。我们CS/GCS分散度的代码已在以下网址公开可用：this https URL。 

---
# Seismic Velocity Inversion from Multi-Source Shot Gathers Using Deep Segmentation Networks: Benchmarking U-Net Variants and SeismoLabV3+ 

**Title (ZH)**: 使用深度分割网络从多源炮集数据中 inversion 地震波速：U-Net 变体和 SeismoLabV3+ 的基准测试 

**Authors**: Mahedi Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21331)  

**Abstract**: Seismic velocity inversion is a key task in geophysical exploration, enabling the reconstruction of subsurface structures from seismic wave data. It is critical for high-resolution seismic imaging and interpretation. Traditional physics-driven methods, such as Full Waveform Inversion (FWI), are computationally demanding, sensitive to initialization, and limited by the bandwidth of seismic data. Recent advances in deep learning have led to data-driven approaches that treat velocity inversion as a dense prediction task. This research benchmarks three advanced encoder-decoder architectures -- U-Net, U-Net++, and DeepLabV3+ -- together with SeismoLabV3+, an optimized variant of DeepLabV3+ with a ResNeXt50 32x4d backbone and task-specific modifications -- for seismic velocity inversion using the ThinkOnward 2025 Speed \& Structure dataset, which consists of five-channel seismic shot gathers paired with high-resolution velocity maps. Experimental results show that SeismoLabV3+ achieves the best performance, with MAPE values of 0.03025 on the internal validation split and 0.031246 on the hidden test set as scored via the official ThinkOnward leaderboard. These findings demonstrate the suitability of deep segmentation networks for seismic velocity inversion and underscore the value of tailored architectural refinements in advancing geophysical AI models. 

**Abstract (ZH)**: 地震波速度反演是地质物理勘探中的关键任务，能够从地震波数据中重建地下结构。它对于高分辨率地震成像和解释至关重要。传统基于物理的方法，如全波形反演（FWI），计算需求高，对初始化敏感，并受限于地震数据的带宽。近年来，深度学习的进展导致了数据驱动的方法，将速度反演视为密集预测任务。本研究在ThinkOnward 2025 Speed & Structure 数据集上（该数据集包含五通道地震地震枪记录及其高分辨率速度图）评估了三种先进的编码器-解码器架构——U-Net、U-Net++ 和 DeepLabV3+，同时包括一个基于 ResNeXt50 32x4d 主干并进行了任务特定修改的优化变体 SeismoLabV3+。实验结果显示，SeismoLabV3+ 在内部验证分割上的 MAPE 值为 0.03025，在隐藏测试集上的 MAPE 值为 0.031246。这些发现表明深度分割网络适合用于地震波速度反演，并强调了为地质物理人工智能模型进行定制架构改进的价值。 

---
# Assessment of deep learning models integrated with weather and environmental variables for wildfire spread prediction and a case study of the 2023 Maui fires 

**Title (ZH)**: 基于天气和环境变量的深度学习模型用于野火蔓延预测的评估及2023年毛伊火灾案例研究 

**Authors**: Jiyeon Kim, Yingjie Hu, Negar Elhami-Khorasani, Kai Sun, Ryan Zhenqi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.21327)  

**Abstract**: Predicting the spread of wildfires is essential for effective fire management and risk assessment. With the fast advancements of artificial intelligence (AI), various deep learning models have been developed and utilized for wildfire spread prediction. However, there is limited understanding of the advantages and limitations of these models, and it is also unclear how deep learning-based fire spread models can be compared with existing non-AI fire models. In this work, we assess the ability of five typical deep learning models integrated with weather and environmental variables for wildfire spread prediction based on over ten years of wildfire data in the state of Hawaii. We further use the 2023 Maui fires as a case study to compare the best deep learning models with a widely-used fire spread model, FARSITE. The results show that two deep learning models, i.e., ConvLSTM and ConvLSTM with attention, perform the best among the five tested AI models. FARSITE shows higher precision, lower recall, and higher F1-score than the best AI models, while the AI models offer higher flexibility for the input data. By integrating AI models with an explainable AI method, we further identify important weather and environmental factors associated with the 2023 Maui wildfires. 

**Abstract (ZH)**: 预测野火蔓延对于有效的火灾管理与风险评估至关重要。随着人工智能（AI）的快速发展，各种深度学习模型已被开发并用于野火蔓延预测。然而，对于这些模型的优势与限制尚缺乏足够的理解，同时也不清楚基于深度学习的野火蔓延模型如何与现有的非AI野火模型进行比较。在本文中，我们基于夏威夷州超过十年的野火数据，评估了五种典型深度学习模型结合气象与环境变量在野火蔓延预测中的能力。我们进一步以2023年 Maui 火灾为例，将最佳深度学习模型与广泛使用的野火蔓延模型 FARSITE 进行对比。结果显示，在测试的五种AI模型中，ConvLSTM 和带有注意力机制的 ConvLSTM 表现最佳。FARSITE 在精度上高于最佳AI模型，但在召回率和F1分数上较低，而AI模型在输入数据的灵活性方面表现更佳。通过将AI模型与可解释的AI方法相结合，我们进一步确定了与2023年Maui野火相关的关键气象与环境因素。 

---
# PIR-RAG: A System for Private Information Retrieval in Retrieval-Augmented Generation 

**Title (ZH)**: PIR-RAG：一种检索增强生成中的私人信息检索系统 

**Authors**: Baiqiang Wang, Qian Lou, Mengxin Zheng, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21325)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a foundational component of modern AI systems, yet it introduces significant privacy risks by exposing user queries to service providers. To address this, we introduce PIR-RAG, a practical system for privacy-preserving RAG. PIR-RAG employs a novel architecture that uses coarse-grained semantic clustering to prune the search space, combined with a fast, lattice-based Private Information Retrieval (PIR) protocol. This design allows for the efficient retrieval of entire document clusters, uniquely optimizing for the end-to-end RAG workflow where full document content is required. Our comprehensive evaluation against strong baseline architectures, including graph-based PIR and Tiptoe-style private scoring, demonstrates PIR-RAG's scalability and its superior performance in terms of "RAG-Ready Latency"-the true end-to-end time required to securely fetch content for an LLM. Our work establishes PIR-RAG as a viable and highly efficient solution for privacy in large-scale AI systems. 

**Abstract (ZH)**: 隐私保护的检索增强生成（PIR-RAG）已成为现代AI系统的基础组件，但会通过暴露用户查询给服务提供商而引入重大的隐私风险。为了解决这一问题，我们引入了PIR-RAG，这是一种实用的隐私保护检索增强生成系统。PIR-RAG采用了一种新颖的架构，结合了粗粒度语义聚类来修剪搜索空间，并使用快速的格基私有信息检索（PIR）协议。这一设计允许高效检索整个文档集群，针对需要完整文档内容的端到端RAG工作流进行优化。我们的全面评估表明，PIR-RAG在可扩展性和“RAG准备延迟”方面（即真正需要为LLM安全获取内容的端到端时间）优于包括基于图的PIR和Tiptoe风格私有评分在内的强基准架构。我们的工作确立了PIR-RAG作为大规模AI系统中隐私保护的可行且高效的解决方案。 

---
# From Search to Reasoning: A Five-Level RAG Capability Framework for Enterprise Data 

**Title (ZH)**: 从搜索到推理：企业数据的五级RAG能力框架 

**Authors**: Gurbinder Gill, Ritvik Gupta, Denis Lusson, Anand Chandrashekar, Donald Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21324)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the standard paradigm for answering questions on enterprise data. Traditionally, RAG has centered on text-based semantic search and re-ranking. However, this approach falls short when dealing with questions beyond data summarization or non-text data. This has led to various attempts to supplement RAG to bridge the gap between RAG, the implementation paradigm, and the question answering problem that enterprise users expect it to solve. Given that contemporary RAG is a collection of techniques rather than a defined implementation, discussion of RAG and related question-answering systems benefits from a problem-oriented understanding.
We propose a new classification framework (L1-L5) to categorize systems based on data modalities and task complexity of the underlying question answering problems: L1 (Surface Knowledge of Unstructured Data) through L4 (Reflective and Reasoned Knowledge) and the aspirational L5 (General Intelligence). We also introduce benchmarks aligned with these levels and evaluate four state-of-the-art platforms: LangChain, Azure AI Search, OpenAI, and Corvic AI. Our experiments highlight the value of multi-space retrieval and dynamic orchestration for enabling L1-L4 capabilities. We empirically validate our findings using diverse datasets indicative of enterprise use cases. 

**Abstract (ZH)**: RAG增强生成：面向企业数据的检索增强生成的新分类框架与评估 

---
