# Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving 

**Title (ZH)**: Agent RL扩展律：自发代码执行的代理强化学习及其在数学问题求解中的应用 

**Authors**: Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07773)  

**Abstract**: Large Language Models (LLMs) often struggle with mathematical reasoning tasks requiring precise, verifiable computation. While Reinforcement Learning (RL) from outcome-based rewards enhances text-based reasoning, understanding how agents autonomously learn to leverage external tools like code execution remains crucial. We investigate RL from outcome-based rewards for Tool-Integrated Reasoning, ZeroTIR, training base LLMs to spontaneously generate and execute Python code for mathematical problems without supervised tool-use examples. Our central contribution is we demonstrate that as RL training progresses, key metrics scale predictably. Specifically, we observe strong positive correlations where increased training steps lead to increases in the spontaneous code execution frequency, the average response length, and, critically, the final task accuracy. This suggests a quantifiable relationship between computational effort invested in training and the emergence of effective, tool-augmented reasoning strategies. We implement a robust framework featuring a decoupled code execution environment and validate our findings across standard RL algorithms and frameworks. Experiments show ZeroTIR significantly surpasses non-tool ZeroRL baselines on challenging math benchmarks. Our findings provide a foundational understanding of how autonomous tool use is acquired and scales within Agent RL, offering a reproducible benchmark for future studies. Code is released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在需要精确可验证计算的数学推理任务上常常表现不佳。基于结果奖励的强化学习（RL）虽然提升了基于文本的推理能力，但理解智能体如何自主学习利用外部工具（如代码执行）仍至关重要。我们研究了基于结果奖励的工具集成推理（ZeroTIR），训练基础语言模型自发生成并执行Python代码解决数学问题，而无需监督工具使用示例。我们的主要贡献是证明了随着RL训练的进行，关键指标可预测地增长。具体而言，我们观察到，随着训练步骤的增加，自发代码执行的频率、平均响应长度以及最终任务准确性都呈强烈正相关。这表明了在训练中投入的计算努力与有效工具增强推理策略的涌现之间存在定量关系。我们实现了一个稳健的框架，其中包括解耦的代码执行环境，并在标准RL算法和框架上验证了我们的发现。实验表明ZeroTIR在具有挑战性的数学基准测试中显著超越了非工具ZeroRL基线。我们的研究结果提供了关于自主工具使用如何获得及其扩展的基石性理解，并为未来的研究提供了一个可复现实验基准。代码发布在\href{this https URL}{this https URL}。 

---
# "I Apologize For Not Understanding Your Policy": Exploring the Specification and Evaluation of User-Managed Access Control Policies by AI Virtual Assistants 

**Title (ZH)**: “抱歉没有理解您的政策”：探索AI虚拟助手对用户管理访问控制策略的规范与评估 

**Authors**: Jennifer Mondragon, Carlos Rubio-Medrano, Gael Cruz, Dvijesh Shastri  

**Link**: [PDF](https://arxiv.org/pdf/2505.07759)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI)-based Virtual Assistants (VAs) e.g., Google Gemini, ChatGPT, Microsoft Copilot, and High-Flyer Deepseek has turned them into convenient interfaces for managing emerging technologies such as Smart Homes, Smart Cars, Electronic Health Records, by means of explicit commands,e.g., prompts, which can be even launched via voice, thus providing a very convenient interface for end-users. However, the proper specification and evaluation of User-Managed Access Control Policies (U-MAPs), the rules issued and managed by end-users to govern access to sensitive data and device functionality - within these VAs presents significant challenges, since such a process is crucial for preventing security vulnerabilities and privacy leaks without impacting user experience. This study provides an initial exploratory investigation on whether current publicly-available VAs can manage U-MAPs effectively across differing scenarios. By conducting unstructured to structured tests, we evaluated the comprehension of such VAs, revealing a lack of understanding in varying U-MAP approaches. Our research not only identifies key limitations, but offers valuable insights into how VAs can be further improved to manage complex authorization rules and adapt to dynamic changes. 

**Abstract (ZH)**: 基于人工智能的虚拟助手（如Google Gemini、ChatGPT、Microsoft Copilot和High-Flyer Deepseek的快速进化已成为通过显式命令管理智能家居、智能汽车、电子健康记录等新兴技术的便捷接口，这些命令甚至可以通过语音启动，从而为终端用户提供非常便捷的接口。然而，对用户管理访问控制策略（U-MAPs）的适当规范和评估——用户发布和管理的规则，以控制对敏感数据和设备功能的访问——在这些虚拟助手中的实现面临重大挑战，因为这一过程对于防止安全漏洞和隐私泄露至关重要，同时不影响用户体验。本研究提供了一项初步的探索性调查，探讨当前可公开获取的虚拟助手能否在不同场景下有效管理U-MAPs。通过从非结构化测试到结构化测试的评估，我们揭示了这些虚拟助手在不同U-MAP方法上的理解不足。研究不仅发现了关键的局限性，还为如何进一步改进虚拟助手以管理复杂的授权规则并适应动态变化提供了宝贵的见解。 

---
# Emotion-Gradient Metacognitive RSI (Part I): Theoretical Foundations and Single-Agent Architecture 

**Title (ZH)**: 情感梯度元认知RSI（第一部分）：理论基础与单智能体架构 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.07757)  

**Abstract**: We present the Emotion-Gradient Metacognitive Recursive Self-Improvement (EG-MRSI) framework, a novel architecture that integrates introspective metacognition, emotion-based intrinsic motivation, and recursive self-modification into a unified theoretical system. The framework is explicitly capable of overwriting its own learning algorithm under formally bounded risk. Building upon the Noise-to-Meaning RSI (N2M-RSI) foundation, EG-MRSI introduces a differentiable intrinsic reward function driven by confidence, error, novelty, and cumulative success. This signal regulates both a metacognitive mapping and a self-modification operator constrained by provable safety mechanisms. We formally define the initial agent configuration, emotion-gradient dynamics, and RSI trigger conditions, and derive a reinforcement-compatible optimization objective that guides the agent's development trajectory. Meaning Density and Meaning Conversion Efficiency are introduced as quantifiable metrics of semantic learning, closing the gap between internal structure and predictive informativeness. This Part I paper establishes the single-agent theoretical foundations of EG-MRSI. Future parts will extend this framework to include safety certificates and rollback protocols (Part II), collective intelligence mechanisms (Part III), and feasibility constraints including thermodynamic and computational limits (Part IV). Together, the EG-MRSI series provides a rigorous, extensible foundation for open-ended and safe AGI. 

**Abstract (ZH)**: 基于情感梯度元认知递归自我改进的框架：EG-MRSI 

---
# Belief Injection for Epistemic Control in Linguistic State Space 

**Title (ZH)**: 信念注入实现知识控制的语言状态空间中 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.07693)  

**Abstract**: This work introduces belief injection, a proactive epistemic control mechanism for artificial agents whose cognitive states are structured as dynamic ensembles of linguistic belief fragments. Grounded in the Semantic Manifold framework, belief injection directly incorporates targeted linguistic beliefs into an agent's internal cognitive state, influencing reasoning and alignment proactively rather than reactively. We delineate various injection strategies, such as direct, context-aware, goal-oriented, and reflective approaches, and contrast belief injection with related epistemic control mechanisms, notably belief filtering. Additionally, this work discusses practical applications, implementation considerations, ethical implications, and outlines promising directions for future research into cognitive governance using architecturally embedded belief injection. 

**Abstract (ZH)**: This work introduces belief injection, 一种基于语义流形框架的主动主义态控制机制，适用于其认知状态由动态语言信念片段组成的 artificial agents。通过直接将目标语言信念注入代理的内部认知状态，信念注入主动影响推理和对齐，而非被动地响应。我们阐明了各种注入策略，如直接注入、语境感知注入、目标导向注入和反思性方法，并将信念注入与相关主义态控制机制，特别是信念过滤进行对比。此外，本文讨论了实际应用、实现考虑、伦理影响，并概述了在认知治理中嵌入信念注入的有前途的研究方向。 

---
# S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models 

**Title (ZH)**: S-GRPO：推理模型中的early exit通过强化学习实现 

**Authors**: Muzhi Dai, Chenxu Yang, Qingyi Si  

**Link**: [PDF](https://arxiv.org/pdf/2505.07686)  

**Abstract**: As Test-Time Scaling emerges as an active research focus in the large language model community, advanced post-training methods increasingly emphasize extending chain-of-thought (CoT) generation length, thereby enhancing reasoning capabilities to approach Deepseek R1-like reasoning models. However, recent studies reveal that reasoning models (even Qwen3) consistently exhibit excessive thought redundancy in CoT generation. This overthinking problem stems from conventional outcome-reward reinforcement learning's systematic neglect in regulating intermediate reasoning steps. This paper proposes Serial-Group Decaying-Reward Policy Optimization (namely S-GRPO), a novel reinforcement learning method that empowers models with the capability to determine the sufficiency of reasoning steps, subsequently triggering early exit of CoT generation. Specifically, unlike GRPO, which samples multiple possible completions (parallel group) in parallel, we select multiple temporal positions in the generation of one CoT to allow the model to exit thinking and instead generate answers (serial group), respectively. For the correct answers in a serial group, we assign rewards that decay according to positions, with lower rewards towards the later ones, thereby reinforcing the model's behavior to generate higher-quality answers at earlier phases with earlier exits of thinking. Empirical evaluations demonstrate compatibility with state-of-the-art reasoning models, including Qwen3 and Deepseek-distill models, achieving 35.4% ~ 61.1\% sequence length reduction with 0.72% ~ 6.08% accuracy improvements across GSM8K, AIME 2024, AMC 2023, MATH-500, and GPQA Diamond benchmarks. 

**Abstract (ZH)**: As 测试时缩放 研究成为大规模语言模型领域的一个活跃研究焦点，先进的后训练方法 increasingly 强调扩展链式思考（CoT）生成长度，从而提高推理能力以接近 Deepseek R1 类型的推理模型。然而，最近的研究表明，即使对于 Qwen3，推理模型在 CoT 生成中也一致地表现出过度的思考冗余。这一过度思考问题源于传统的结果奖励强化学习对中间推理步骤的系统忽视。本文提出了一种新颖的强化学习方法 S-GRPO（Serial-Group Decaying-Reward Policy Optimization），该方法赋予模型判断推理步骤充分性的能力，进而触发 CoT 生成的早期退出。具体来说，与 GRPO 不同，S-GRPO 在生成一个 CoT 的过程中选择多个时间位置，允许模型退出思考而生成答案（串行组），分别对待。对于串行组中的正确答案，我们按照位置分配递减奖励，后期奖励较低，从而强化模型在早期阶段生成高质量答案并尽早退出思考的行为。实证评估表明，该方法与最先进的推理模型（包括 Qwen3 和 Deepseek-distill 模型）兼容，并在 GSM8K、AIME 2024、AMC 2023、MATH-500 和 GPQA Diamond 标准测试中实现了 35.4%~61.1% 的序列长度减少，同时在准确率上提高了 0.72%~6.08%。 

---
# YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models 

**Title (ZH)**: YuLan-OneSim：迈向新一代社会模拟器的大语言模型技术 

**Authors**: Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07581)  

**Abstract**: Leveraging large language model (LLM) based agents to simulate human social behaviors has recently gained significant attention. In this paper, we introduce a novel social simulator called YuLan-OneSim. Compared to previous works, YuLan-OneSim distinguishes itself in five key aspects: (1) Code-free scenario construction: Users can simply describe and refine their simulation scenarios through natural language interactions with our simulator. All simulation code is automatically generated, significantly reducing the need for programming expertise. (2) Comprehensive default scenarios: We implement 50 default simulation scenarios spanning 8 domains, including economics, sociology, politics, psychology, organization, demographics, law, and communication, broadening access for a diverse range of social researchers. (3) Evolvable simulation: Our simulator is capable of receiving external feedback and automatically fine-tuning the backbone LLMs, significantly enhancing the simulation quality. (4) Large-scale simulation: By developing a fully responsive agent framework and a distributed simulation architecture, our simulator can handle up to 100,000 agents, ensuring more stable and reliable simulation results. (5) AI social researcher: Leveraging the above features, we develop an AI social researcher. Users only need to propose a research topic, and the AI researcher will automatically analyze the input, construct simulation environments, summarize results, generate technical reports, review and refine the reports--completing the social science research loop. To demonstrate the advantages of YuLan-OneSim, we conduct experiments to evaluate the quality of the automatically generated scenarios, the reliability, efficiency, and scalability of the simulation process, as well as the performance of the AI social researcher. 

**Abstract (ZH)**: 利用基于大规模语言模型的代理模拟人类社会行为 recently gained significant attention. 本文介绍了一种新颖的社会仿真器 YuLan-OneSim。与以往工作相比，YuLan-OneSim 在五个关键方面具有优势：（1）无需代码的场景构建：用户可以通过自然语言与仿真器交互来简单描述和细化仿真场景，所有仿真代码均自动生成，显著降低了编程技能的需求。（2）全面的默认场景：我们实现了涵盖经济学、社会学、政治学、心理学、组织学、人口学、法律和通讯等 8 个领域的 50 个默认仿真场景，为多样化的社会研究者提供了更广泛的访问渠道。（3）可进化的仿真：该仿真器能够接收外部反馈并自动精细调整骨干语言模型，显著提高了仿真质量。（4）大规模仿真：通过开发全响应式代理框架和分布式仿真架构，该仿真器能够处理多达 100,000 个代理，确保仿真结果更加稳定可靠。（5）AI 社会研究者：利用上述特性，我们开发了 AI 社会研究者。用户只需提出研究主题，AI 研究者将自动分析输入、构建仿真环境、总结结果、生成技术报告、审阅和优化报告——完成社会科学研究的全过程。为了展示 YuLan-OneSim 的优势，我们进行了实验，评估了自动生成场景的质量、仿真的可靠性和效率以及 AI 社会研究者的表现。 

---
# QuantX: A Framework for Hardware-Aware Quantization of Generative AI Workloads 

**Title (ZH)**: QuantX: 一种面向硬件的生成型AI工作负载量化框架 

**Authors**: Khurram Mazher, Saad Bin Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2505.07531)  

**Abstract**: We present QuantX: a tailored suite of recipes for LLM and VLM quantization. It is capable of quantizing down to 3-bit resolutions with minimal loss in performance. The quantization strategies in QuantX take into account hardware-specific constraints to achieve efficient dequantization during inference ensuring flexible trade-off between runtime speed, memory requirement and model accuracy. Our results demonstrate that QuantX achieves performance within 6% of the unquantized model for LlaVa-v1.6 quantized down to 3-bits for multiple end user tasks and outperforms recently published state-of-the-art quantization techniques. This manuscript provides insights into the LLM quantization process that motivated the range of recipes and options that are incorporated in QuantX. 

**Abstract (ZH)**: 我们呈现QuantX：针对LLM和VLM量化的一套定制化方案。它能够在保持最低性能损失的情况下将量化精度降至3位。QuantX中的量化策略考虑了硬件特异性限制，以实现高效推理时的去量化，确保在运行时速度、内存需求和模型准确性之间灵活权衡。我们的结果表明，QuantX在将LlaVa-v1.6量化至3位后，跨多个终端用户任务的性能与未量化模型的差距不超过6%，并优于最近公布的最先进的量化技术。本文提供了关于LLM量化过程的见解，这些见解激发了QuantX中所集成的各种方案和选项。 

---
# HALO: Half Life-Based Outdated Fact Filtering in Temporal Knowledge Graphs 

**Title (ZH)**: HALO: 基于半衰期的过时事实过滤算法在时间型知识图谱中 

**Authors**: Feng Ding, Tingting Wang, Yupeng Gao, Shuo Yu, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.07509)  

**Abstract**: Outdated facts in temporal knowledge graphs (TKGs) result from exceeding the expiration date of facts, which negatively impact reasoning performance on TKGs. However, existing reasoning methods primarily focus on positive importance of historical facts, neglecting adverse effects of outdated facts. Besides, training on these outdated facts yields extra computational cost. To address these challenges, we propose an outdated fact filtering framework named HALO, which quantifies the temporal validity of historical facts by exploring the half-life theory to filter outdated facts in TKGs. HALO consists of three modules: the temporal fact attention module, the dynamic relation-aware encoder module, and the outdated fact filtering module. Firstly, the temporal fact attention module captures the evolution of historical facts over time to identify relevant facts. Secondly, the dynamic relation-aware encoder module is designed for efficiently predicting the half life of each fact. Finally, we construct a time decay function based on the half-life theory to quantify the temporal validity of facts and filter outdated facts. Experimental results show that HALO outperforms the state-of-the-art TKG reasoning methods on three public datasets, demonstrating its effectiveness in detecting and filtering outdated facts (Codes are available at this https URL ). 

**Abstract (ZH)**: 过时事实存在于时间知识图谱中，导致事实超出有效期，从而负面影响时间知识图谱上的推理性能。然而，现有的推理方法主要关注历史事实的正向影响，忽略了过时事实的负面影响。此外，基于这些过时事实进行训练还会带来额外的计算成本。为解决这些问题，我们提出了一种名为HALO的过时事实过滤框架，该框架通过探究半衰期理论来量化历史事实的时间有效性并过滤过时事实。HALO包括三个模块：时间事实注意力模块、动态关系感知编码模块以及过时事实过滤模块。首先，时间事实注意力模块捕捉历史事实随时间的演变，以识别相关事实。其次，动态关系感知编码模块旨在高效预测每条事实的半衰期。最后，我们基于半衰期理论构建了一个时间衰减函数，以量化事实的时间有效性并过滤过时事实。实验结果表明，HALO在三个公开数据集上的表现优于现有的时间知识图谱推理方法，证明了其在检测和过滤过时事实方面的有效性（代码可在以下链接获取：this https URL）。 

---
# Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks 

**Title (ZH)**: Web-Bench：基于Web标准和框架的LLM代码基准 

**Authors**: Kai Xu, YiWei Mao, XinYi Guan, ZiLong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07473)  

**Abstract**: The application of large language models (LLMs) in the field of coding is evolving rapidly: from code assistants, to autonomous coding agents, and then to generating complete projects through natural language. Early LLM code benchmarks primarily focused on code generation accuracy, but these benchmarks have gradually become saturated. Benchmark saturation weakens their guiding role for LLMs. For example, HumanEval Pass@1 has reached 99.4% and MBPP 94.2%. Among various attempts to address benchmark saturation, approaches based on software engineering have stood out, but the saturation of existing software engineering benchmarks is rapidly increasing. To address this, we propose a new benchmark, Web-Bench, which contains 50 projects, each consisting of 20 tasks with sequential dependencies. The tasks implement project features in sequence, simulating real-world human development workflows. When designing Web-Bench, we aim to cover the foundational elements of Web development: Web Standards and Web Frameworks. Given the scale and complexity of these projects, which were designed by engineers with 5 to 10 years of experience, each presents a significant challenge. On average, a single project takes 4 to 8 hours for a senior engineer to complete. On our given benchmark agent (Web-Agent), SOTA (Claude 3.7 Sonnet) achieves only 25.1% Pass@1, significantly lower (better) than SWE-Bench's Verified (65.4%) and Full (33.8%) scores. Finally, we discuss that in any development field, Standards and Frameworks represent foundational knowledge and efficiency tools, respectively, and LLMs require optimization tailored to them. 

**Abstract (ZH)**: 大型语言模型在编码领域的应用不断进化：从代码辅助到自主编码代理，再到通过自然语言生成完整的项目。早期的大型语言模型代码基准主要关注代码生成准确性，但这些基准逐渐饱和。基准饱和削弱了它们对大型语言模型的指导作用。为此，我们提出一个新的基准——Web-Bench，包含50个项目，每个项目由20个具有序列依赖性的任务组成，模拟现实世界的人类开发工作流。我们设计Web-Bench的目标是覆盖Web开发的基础要素：Web标准和Web框架。由于这些项目由具有5到10年经验的工程师设计，每个项目都构成了重大挑战。平均而言，一个项目需要资深工程师4到8小时才能完成。在我们提供的基准代理（Web-Agent）上，当前最先进的模型Claude 3.7 Sonnet的Pass@1得分为25.1%，显著低于SWE-Bench的Verified（65.4%）和Full（33.8%）得分。最后，我们讨论在任何开发领域，标准和框架分别代表基础知识和效率工具，大型语言模型需要针对它们进行优化。 

---
# A Survey on Collaborative Mechanisms Between Large and Small Language Models 

**Title (ZH)**: 大型和小型语言模型之间的协作机制综述 

**Authors**: Yi Chen, JiaHao Zhao, HaoHao Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07460)  

**Abstract**: Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consistency, robust task allocation, evaluation complexity, and security/privacy concerns. Future directions point towards more intelligent adaptive frameworks, deeper model fusion, and expansion into multimodal and embodied AI, positioning LLM-SLM collaboration as a key driver for the next generation of practical and ubiquitous artificial intelligence. 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了强大的AI能力，但由于高昂的资源成本和延迟问题面临部署挑战，而小型语言模型（SLMs）虽然在性能上有所降低，但提供了高效性和可部署性。LLMs与SLMs的合作成为一种关键范式，旨在协同平衡这些权衡，从而实现先进的AI应用，特别是在资源受限的边缘设备上。这篇综述提供了关于LLM-SLM合作的全面概述，详细介绍了各种交互机制（管道、路由、辅助、蒸馏、融合）、关键技术以及各种由设备端需求（如低延迟、隐私、个性化和离线操作）驱动的应用场景。尽管突显了创建更高效、灵活和可访问的AI的巨大潜力，我们还讨论了系统开销、模型间一致性、鲁棒任务分配、评估复杂性和安全/隐私问题等持续挑战。未来方向包括更加智能的自适应框架、更深层次的模型融合，并扩展到多模态和实体AI，将LLM-SLM合作定位为推动下一代实用且普及的AI的关键驱动力。 

---
# How well do LLMs reason over tabular data, really? 

**Title (ZH)**: LLM在处理表格数据时真的能进行有效的推理吗？ 

**Authors**: Cornelius Wolff, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.07453)  

**Abstract**: Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs. 

**Abstract (ZH)**: 大型语言模型在表格数据上的推理能力：现状与挑战 

---
# AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review 

**Title (ZH)**: 基于变压器的AIS数据驱动海洋监测：一项全面综述 

**Authors**: Zhiye Xie, Enmei Tu, Xianping Fu, Guoliang Yuan, Yi Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07374)  

**Abstract**: With the increasing demands for safety, efficiency, and sustainability in global shipping, Automatic Identification System (AIS) data plays an increasingly important role in maritime monitoring. AIS data contains spatial-temporal variation patterns of vessels that hold significant research value in the marine domain. However, due to its massive scale, the full potential of AIS data has long remained untapped. With its powerful sequence modeling capabilities, particularly its ability to capture long-range dependencies and complex temporal dynamics, the Transformer model has emerged as an effective tool for processing AIS data. Therefore, this paper reviews the research on Transformer-based AIS data-driven maritime monitoring, providing a comprehensive overview of the current applications of Transformer models in the marine field. The focus is on Transformer-based trajectory prediction methods, behavior detection, and prediction techniques. Additionally, this paper collects and organizes publicly available AIS datasets from the reviewed papers, performing data filtering, cleaning, and statistical analysis. The statistical results reveal the operational characteristics of different vessel types, providing data support for further research on maritime monitoring tasks. Finally, we offer valuable suggestions for future research, identifying two promising research directions. Datasets are available at this https URL. 

**Abstract (ZH)**: 随着全球航运对安全、效率和可持续性的需求不断增加，自动识别系统（AIS）数据在海上监控中发挥着越来越重要的作用。AIS数据包含了具有重要研究价值的船舶时空变化模式。然而，由于其规模庞大，AIS数据的全部潜力长期得不到充分发挥。凭借其强大的序列建模能力，尤其是捕捉长程依赖性和复杂时序动态的能力，Transformer模型已成为处理AIS数据的有效工具。因此，本文回顾了基于Transformer的AIS数据驱动的海上监控研究，提供了Transformer模型在海洋领域应用的全面概述，重点在于基于Transformer的航线预测方法、行为检测和预测技术。此外，本文还收集并整理了从文献中获取的公开可用的AIS数据集，进行了数据过滤、清洗和统计分析。统计结果揭示了不同船舶类型的运营特征，为进一步研究海上监控任务提供了数据支持。最后，我们提出了对未来研究的宝贵建议，指出了两个有前景的研究方向。数据集可从此链接获取。 

---
# FedIFL: A federated cross-domain diagnostic framework for motor-driven systems with inconsistent fault modes 

**Title (ZH)**: FedIFL：一种针对具有不一致故障模式的驱动系统跨域诊断框架 

**Authors**: Zexiao Wang, Yankai Wang, Xiaoqiang Liao, Xinguo Ming, Weiming Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07315)  

**Abstract**: Due to the scarcity of industrial data, individual equipment users, particularly start-ups, struggle to independently train a comprehensive fault diagnosis model; federated learning enables collaborative training while ensuring data privacy, making it an ideal solution. However, the diversity of working conditions leads to variations in fault modes, resulting in inconsistent label spaces across different clients. In federated diagnostic scenarios, label space inconsistency leads to local models focus on client-specific fault modes and causes local models from different clients to map different failure modes to similar feature representations, which weakens the aggregated global model's generalization. To tackle this issue, this article proposed a federated cross-domain diagnostic framework termed Federated Invariant Features Learning (FedIFL). In intra-client training, prototype contrastive learning mitigates intra-client domain shifts, subsequently, feature generating ensures local models can access distributions of other clients in a privacy-friendly manner. Besides, in cross-client training, a feature disentanglement mechanism is introduced to mitigate cross-client domain shifts, specifically, an instance-level federated instance consistency loss is designed to ensure the instance-level consistency of invariant features between different clients, furthermore, a federated instance personalization loss and an orthogonal loss are constructed to distinguish specific features that from the invariant features. Eventually, the aggregated model achieves promising generalization among global label spaces, enabling accurate fault diagnosis for target clients' Motor Driven Systems (MDSs) with inconsistent label spaces. Experiments on real-world MDSs validate the effectiveness and superiority of FedIFL in federated cross-domain diagnosis with inconsistent fault modes. 

**Abstract (ZH)**: 基于联邦学习的跨域不变特征学习故障诊断框架（FedIFL） 

---
# Interpretable Event Diagnosis in Water Distribution Networks 

**Title (ZH)**: 可解释的水资源分配网络事件诊断 

**Authors**: André Artelt, Stelios G. Vrachimis, Demetrios G. Eliades, Ulrike Kuhl, Barbara Hammer, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07299)  

**Abstract**: The increasing penetration of information and communication technologies in the design, monitoring, and control of water systems enables the use of algorithms for detecting and identifying unanticipated events (such as leakages or water contamination) using sensor measurements. However, data-driven methodologies do not always give accurate results and are often not trusted by operators, who may prefer to use their engineering judgment and experience to deal with such events.
In this work, we propose a framework for interpretable event diagnosis -- an approach that assists the operators in associating the results of algorithmic event diagnosis methodologies with their own intuition and experience. This is achieved by providing contrasting (i.e., counterfactual) explanations of the results provided by fault diagnosis algorithms; their aim is to improve the understanding of the algorithm's inner workings by the operators, thus enabling them to take a more informed decision by combining the results with their personal experiences. Specifically, we propose counterfactual event fingerprints, a representation of the difference between the current event diagnosis and the closest alternative explanation, which can be presented in a graphical way. The proposed methodology is applied and evaluated on a realistic use case using the L-Town benchmark. 

**Abstract (ZH)**: 信息系统和通信技术在水资源系统设计、监控和控制中的渗透使算法能够利用传感器测量数据检测和识别未预见的事件（如泄漏或水质污染）。然而，数据驱动的方法并不总是给出准确的结果，操作人员往往更倾向于依靠他们的工程判断和经验来应对这些事件。

本文提出了一种可解释的事件诊断框架——一种帮助操作人员将算法事件诊断方法的结果与自身的直觉和经验关联起来的方法。通过提供与故障诊断算法结果形成对比（即反事实）的解释，旨在通过改进操作人员对算法内部工作机制的理解，使他们能够在结合结果和自身经验的基础上作出更加明智的决定。具体而言，我们提出了反事实事件指纹，这是一种代表当前事件诊断与其最接近替代解释之间差异的表示方法，并可以通过图形方式呈现。所提出的方法在L-Town基准案例上进行了应用和评估。 

---
# Measuring General Intelligence with Generated Games 

**Title (ZH)**: 使用生成的游戏衡量通用智能 

**Authors**: Vivek Verma, David Huang, William Chen, Dan Klein, Nicholas Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07215)  

**Abstract**: We present gg-bench, a collection of game environments designed to evaluate general reasoning capabilities in language models. Unlike most static benchmarks, gg-bench is a data generating process where new evaluation instances can be generated at will. In particular, gg-bench is synthetically generated by (1) using a large language model (LLM) to generate natural language descriptions of novel games, (2) using the LLM to implement each game in code as a Gym environment, and (3) training reinforcement learning (RL) agents via self-play on the generated games. We evaluate language models by their winrate against these RL agents by prompting models with the game description, current board state, and a list of valid moves, after which models output the moves they wish to take. gg-bench is challenging: state-of-the-art LLMs such as GPT-4o and Claude 3.7 Sonnet achieve winrates of 7-9% on gg-bench using in-context learning, while reasoning models such as o1, o3-mini and DeepSeek-R1 achieve average winrates of 31-36%. We release the generated games, data generation process, and evaluation code in order to support future modeling work and expansion of our benchmark. 

**Abstract (ZH)**: gg-bench：用于评估语言模型通用推理能力的游戏环境集合 

---
# Accountability of Generative AI: Exploring a Precautionary Approach for "Artificially Created Nature" 

**Title (ZH)**: 生成式AI的责任性：探索“人工创造的自然”审慎 Approach 

**Authors**: Yuri Nakao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07178)  

**Abstract**: The rapid development of generative artificial intelligence (AI) technologies raises concerns about the accountability of sociotechnical systems. Current generative AI systems rely on complex mechanisms that make it difficult for even experts to fully trace the reasons behind the outputs. This paper first examines existing research on AI transparency and accountability and argues that transparency is not a sufficient condition for accountability but can contribute to its improvement. We then discuss that if it is not possible to make generative AI transparent, generative AI technology becomes ``artificially created nature'' in a metaphorical sense, and suggest using the precautionary principle approach to consider AI risks. Finally, we propose that a platform for citizen participation is needed to address the risks of generative AI. 

**Abstract (ZH)**: 生成人工智能技术的迅速发展引发了对社会技术系统问责性的关注。当前的生成人工智能系统依赖于复杂的机制，使得即使是专家也难以完全追溯其输出的原因。本文首先探讨了现有的人工智能透明性和问责制研究，并认为透明性并非问责制的充分条件，但可以促进其改进。然后我们讨论，如果不能使生成人工智能变得透明，这种技术在比喻意义上将成为“人造自然”，并建议采用预防性原则来考虑人工智能风险。最后，我们提出需要一个公民参与的平台以应对生成人工智能的风险。 

---
# ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion 

**Title (ZH)**: 基于关系的条件扩散与注意力池化少样本知识图谱补全 

**Authors**: Jeongho Kim, Chanyeong Heo, Jaehee Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.07171)  

**Abstract**: Knowledge Graphs (KGs), composed of triples in the form of (head, relation, tail) and consisting of entities and relations, play a key role in information retrieval systems such as question answering, entity search, and recommendation. In real-world KGs, although many entities exist, the relations exhibit a long-tail distribution, which can hinder information retrieval performance. Previous few-shot knowledge graph completion studies focused exclusively on the positive triple information that exists in the graph or, when negative triples were incorporated, used them merely as a signal to indicate incorrect triples. To overcome this limitation, we propose Relation-Based Conditional Diffusion with Attention Pooling (ReCDAP). First, negative triples are generated by randomly replacing the tail entity in the support set. By conditionally incorporating positive information in the KG and non-existent negative information into the diffusion process, the model separately estimates the latent distributions for positive and negative relations. Moreover, including an attention pooler enables the model to leverage the differences between positive and negative cases explicitly. Experiments on two widely used datasets demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance. The code is available at this https URL. 

**Abstract (ZH)**: 基于关系的概率扩散与注意力池化（ReCDAP）：提高知识图谱的负样本推理性能 

---
# RefPentester: A Knowledge-Informed Self-Reflective Penetration Testing Framework Based on Large Language Models 

**Title (ZH)**: RefPentester：一种基于大型语言模型的知识导向型自省式渗透测试框架 

**Authors**: Hanzheng Dai, Yuanliang Li, Zhibo Zhang, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07089)  

**Abstract**: Automated penetration testing (AutoPT) powered by large language models (LLMs) has gained attention for its ability to automate ethical hacking processes and identify vulnerabilities in target systems by leveraging the intrinsic knowledge of LLMs. However, existing LLM-based AutoPT frameworks often underperform compared to human experts in challenging tasks for several reasons: the imbalanced knowledge used in LLM training, short-sighted planning in the planning process, and hallucinations during command generation. In addition, the penetration testing (PT) process, with its trial-and-error nature, is limited by existing frameworks that lack mechanisms to learn from previous failed operations, restricting adaptive improvement of PT strategies. To address these limitations, we propose a knowledge-informed self-reflective PT framework powered by LLMs, called RefPentester, which is an AutoPT framework designed to assist human operators in identifying the current stage of the PT process, selecting appropriate tactic and technique for the stage, choosing suggested action, providing step-by-step operational guidance, and learning from previous failed operations. We also modeled the PT process as a seven-state Stage Machine to integrate the proposed framework effectively. The evaluation shows that RefPentester can successfully reveal credentials on Hack The Box's Sau machine, outperforming the baseline GPT-4o model by 16.7\%. Across PT stages, RefPentester also demonstrates superior success rates on PT stage transitions. 

**Abstract (ZH)**: 基于大型语言模型的自适应渗透测试框架：RefPentester 

---
# Architectural Precedents for General Agents using Large Language Models 

**Title (ZH)**: 大型语言模型驱动的通用代理建筑范式 

**Authors**: Robert E. Wray, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2505.07087)  

**Abstract**: One goal of AI (and AGI) is to identify and understand specific mechanisms and representations sufficient for general intelligence. Often, this work manifests in research focused on architectures and many cognitive architectures have been explored in AI/AGI. However, different research groups and even different research traditions have somewhat independently identified similar/common patterns of processes and representations or cognitive design patterns that are manifest in existing architectures. Today, AI systems exploiting large language models (LLMs) offer a relatively new combination of mechanism and representation available for exploring the possibilities of general intelligence. In this paper, we summarize a few recurring cognitive design patterns that have appeared in various pre-transformer AI architectures. We then explore how these patterns are evident in systems using LLMs, especially for reasoning and interactive ("agentic") use cases. By examining and applying these recurring patterns, we can also predict gaps or deficiencies in today's Agentic LLM Systems and identify likely subjects of future research towards general intelligence using LLMs and other generative foundation models. 

**Abstract (ZH)**: 人工智能（及超人工智能）的一个目标是识别和理解足够支持一般智能的具体机制和表示。目前，利用大规模语言模型（LLMs）的AI系统为探索一般智能的可能性提供了一种相对较新的机制和表示的组合。本文总结了几种在各种预转子AI架构中反复出现的认知设计模式，并探讨了这些模式在使用LLMs的系统中，特别是在推理和互动（“代理”）应用场景中的表现。通过研究和应用这些反复出现的模式，我们还可以预测当前代理LLM系统的空白或不足，并识别未来研究方向，以利用LLMs及其他生成型基础模型向一般智能前进的研究主题。 

---
# Arbitrarily Applicable Same/Opposite Relational Responding with NARS 

**Title (ZH)**: 任意适用的相同/相反关系响应与NARS 

**Authors**: Robert Johansson, Patrick Hammer, Tony Lofthouse  

**Link**: [PDF](https://arxiv.org/pdf/2505.07079)  

**Abstract**: Same/opposite relational responding, a fundamental aspect of human symbolic cognition, allows the flexible generalization of stimulus relationships based on minimal experience. In this study, we demonstrate the emergence of \textit{arbitrarily applicable} same/opposite relational responding within the Non-Axiomatic Reasoning System (NARS), a computational cognitive architecture designed for adaptive reasoning under uncertainty. Specifically, we extend NARS with an implementation of \textit{acquired relations}, enabling the system to explicitly derive both symmetric (mutual entailment) and novel relational combinations (combinatorial entailment) from minimal explicit training in a contextually controlled matching-to-sample (MTS) procedure. Experimental results show that NARS rapidly internalizes explicitly trained relational rules and robustly demonstrates derived relational generalizations based on arbitrary contextual cues. Importantly, derived relational responding in critical test phases inherently combines both mutual and combinatorial entailments, such as deriving same-relations from multiple explicitly trained opposite-relations. Internal confidence metrics illustrate strong internalization of these relational principles, closely paralleling phenomena observed in human relational learning experiments. Our findings underscore the potential for integrating nuanced relational learning mechanisms inspired by learning psychology into artificial general intelligence frameworks, explicitly highlighting the arbitrary and context-sensitive relational capabilities modeled within NARS. 

**Abstract (ZH)**: 任意适用的相同/相反关系响应：非公理推理系统（NARS）中的一个基本方面，允许基于最少经验对刺激关系进行灵活泛化。在本研究中，我们展示了在非公理推理系统（NARS）中 Emergence of Arbitrarily Applicable Same/Opposite Relational Responding，这是一个为不确定环境下适应性推理设计的计算认知架构。具体而言，我们通过实现习得关系扩展了 NARS，使系统能够从最小的显式训练中在上下文控制的配对样本（MTS）程序中显式推导出对称关系（互蕴）和新颖的关系组合（组合理蕴）。实验结果表明，NARS 迅速内化了显式训练的关系规则，并且能够在任意上下文线索基础上稳健地展示推导出的关系泛化。重要的是，在关键测试阶段的推导关系响应中，必然结合了互蕴和组合理蕴，如从多个显式训练的相反关系中推导出相同关系。内部信心度量表明这些关系原则得到了强烈内化，近似平行于人类关系学习实验中观察到的现象。我们的研究结果强调将启发自学习心理学的细腻关系学习机制整合进人工通用智能框架的潜在可能性，明确指出 NARS 中建模的任意性和上下文敏感的关系能力。 

---
# Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs 

**Title (ZH)**: 解锁非块结构决策：基于选择图的归纳挖掘 

**Authors**: Humam Kourani, Gyunam Park, Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07052)  

**Abstract**: Process discovery aims to automatically derive process models from event logs, enabling organizations to analyze and improve their operational processes. Inductive mining algorithms, while prioritizing soundness and efficiency through hierarchical modeling languages, often impose a strict block-structured representation. This limits their ability to accurately capture the complexities of real-world processes. While recent advancements like the Partially Ordered Workflow Language (POWL) have addressed the block-structure limitation for concurrency, a significant gap remains in effectively modeling non-block-structured decision points. In this paper, we bridge this gap by proposing an extension of POWL to handle non-block-structured decisions through the introduction of choice graphs. Choice graphs offer a structured yet flexible approach to model complex decision logic within the hierarchical framework of POWL. We present an inductive mining discovery algorithm that uses our extension and preserves the quality guarantees of the inductive mining framework. Our experimental evaluation demonstrates that the discovered models, enriched with choice graphs, more precisely represent the complex decision-making behavior found in real-world processes, without compromising the high scalability inherent in inductive mining techniques. 

**Abstract (ZH)**: 过程发现旨在从事件日志中自动推导过程模型，从而帮助组织分析和改进其运营流程。虽然归纳挖掘算法通过层次化的建模语言优先考虑正确性和效率，但往往要求一种严格的块结构表示形式，这限制了它们准确捕捉现实世界流程复杂性的能力。尽管POWL（部分有序工作流语言）等最近的进展解决了并发的块结构限制，但在有效建模非块结构决策点方面仍存在显著差距。在本文中，我们通过引入选择图来扩展POWL，弥补了这一差距，从而处理非块结构决策。选择图在POWL的层次框架内提供了一种结构化且灵活的方法来建模复杂的决策逻辑。我们提出了一种使用扩展方法的归纳挖掘发现算法，并保持了归纳挖掘框架的质量保证。我们的实验评估表明，通过引入选择图丰富后发现的模型，能够更精确地代表现实世界流程中的复杂决策行为，同时保持归纳挖掘技术固有的高可扩展性。 

---
# DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs 

**Title (ZH)**: DialogueReason: 规则基于的RL激发大语言模型的对话推理能力 

**Authors**: Yubo Shu, Zhewei Huang, Xin Wu, Chen Hu, Shuchang Zhou, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07049)  

**Abstract**: We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design. 

**Abstract (ZH)**: DialogueReason：揭示独白式推理模型中丢失的角色，促进推理过程的多样性和连贯性 

---
# Efficient Fault Detection in WSN Based on PCA-Optimized Deep Neural Network Slicing Trained with GOA 

**Title (ZH)**: 基于GOA优化PCA深度神经网络切片的高效无线传感器网络故障检测 

**Authors**: Mahmood Mohassel Feghhi, Raya Majid Alsharfa, Majid Hameed Majeed  

**Link**: [PDF](https://arxiv.org/pdf/2505.07030)  

**Abstract**: Fault detection in Wireless Sensor Networks (WSNs) is crucial for reliable data transmission and network longevity. Traditional fault detection methods often struggle with optimizing deep neural networks (DNNs) for efficient performance, especially in handling high-dimensional data and capturing nonlinear relationships. Additionally, these methods typically suffer from slow convergence and difficulty in finding optimal network architectures using gradient-based optimization. This study proposes a novel hybrid method combining Principal Component Analysis (PCA) with a DNN optimized by the Grasshopper Optimization Algorithm (GOA) to address these limitations. Our approach begins by computing eigenvalues from the original 12-dimensional dataset and sorting them in descending order. The cumulative sum of these values is calculated, retaining principal components until 99.5% variance is achieved, effectively reducing dimensionality to 4 features while preserving critical information. This compressed representation trains a six-layer DNN where GOA optimizes the network architecture, overcoming backpropagation's limitations in discovering nonlinear relationships. This hybrid PCA-GOA-DNN framework compresses the data and trains a six-layer DNN that is optimized by GOA, enhancing both training efficiency and fault detection accuracy. The dataset used in this study is a real-world WSNs dataset developed by the University of North Carolina, which was used to evaluate the proposed method's performance. Extensive simulations demonstrate that our approach achieves a remarkable 99.72% classification accuracy, with exceptional precision and recall, outperforming conventional methods. The method is computationally efficient, making it suitable for large-scale WSN deployments, and represents a significant advancement in fault detection for resource-constrained WSNs. 

**Abstract (ZH)**: Wireless传感器网络中故障检测的关键在于可靠的数据传输和网络 longevity。传统的故障检测方法在优化深度神经网络（DNNs）以实现高效性能时往往遇到困难，特别是在处理高维数据和捕捉非线性关系方面。此外，这些方法通常收敛缓慢，并且在使用梯度优化法寻找最优网络架构时面临困难。本研究提出了一种结合主成分分析（PCA）和Improved Grasshopper Optimization Algorithm（GOA）优化的DNN的新颖混合方法，以解决这些限制。我们的方法首先计算原始12维数据集的特征值，并按降序排序。计算这些值的累计和，直到实现99.5%的方差保留主成分，从而将维度有效降低到4个特征，同时保留关键信息。该压缩表示训练一个六层DNN，其中GOA优化网络架构，克服了反向传播在发现非线性关系方面的局限性。这种结合PCA和GOA的DNN框架压缩了数据并训练了一个六层DNN，GOA优化了网络架构，提高了训练效率和故障检测准确性。本研究使用的数据集是由北卡罗来纳大学开发的无线传感器网络真实世界数据集，用于评估所提出方法的性能。广泛的仿真表明，我们的方法实现了令人瞩目的99.72%分类精度，具有出色的精确度和召回率，优于传统方法。该方法计算效率高，适用于大规模无线传感器网络部署，并代表了资源受限无线传感器网络中故障检测的一个重要进展。无线传感器网络中基于PCA和GOA优化的DNN的故障检测 

---
# LLM-Augmented Chemical Synthesis and Design Decision Programs 

**Title (ZH)**: LLM增强的化学合成与设计决策程序 

**Authors**: Haorui Wang, Jeff Guo, Lingkai Kong, Rampi Ramprasad, Philippe Schwaller, Yuanqi Du, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07027)  

**Abstract**: Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design. 

**Abstract (ZH)**: 逆合成分析，通过一系列有效的反应将目标分子分解为更简单的前体，是有机化学和药物开发的核心。尽管最近的机器学习研究已经在单步逆合成模型和后续路径搜索方面取得了进展，但这些解决方案仍然受到潜在路径组合空间的限制。同时，大型语言模型（LLMs）展现了显著的化学知识，暗示它们在化学中的复杂决策任务中具有潜在的应用价值。在这项工作中，我们探索LLMs是否能够成功解决高度受限的多步逆合成规划问题。我们提出了一种高效的方法来编码反应路径，并提出了一种新的路线级搜索策略，超越了传统的逐步反应物预测。通过全面的评估，我们展示了我们的LLM增强方法在逆合成分析规划方面的优越性能，并自然地扩展到合成可及分子设计的更广泛挑战。 

---
# Explainable AI the Latest Advancements and New Trends 

**Title (ZH)**: 可解释的人工智能：最新进展和新趋势 

**Authors**: Bowen Long, Enjie Liu, Renxi Qiu, Yanqing Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07005)  

**Abstract**: In recent years, Artificial Intelligence technology has excelled in various applications across all domains and fields. However, the various algorithms in neural networks make it difficult to understand the reasons behind decisions. For this reason, trustworthy AI techniques have started gaining popularity. The concept of trustworthiness is cross-disciplinary; it must meet societal standards and principles, and technology is used to fulfill these requirements. In this paper, we first surveyed developments from various countries and regions on the ethical elements that make AI algorithms trustworthy; and then focused our survey on the state of the art research into the interpretability of AI. We have conducted an intensive survey on technologies and techniques used in making AI explainable. Finally, we identified new trends in achieving explainable AI. In particular, we elaborate on the strong link between the explainability of AI and the meta-reasoning of autonomous systems. The concept of meta-reasoning is 'reason the reasoning', which coincides with the intention and goal of explainable Al. The integration of the approaches could pave the way for future interpretable AI systems. 

**Abstract (ZH)**: 近年来，人工智能技术在各个领域和应用中取得了卓越成就。然而，神经网络中的各种算法使得理解和解释决策的原因变得困难。因此，可靠的人工智能技术开始受到关注。信任的概念是跨学科的，必须符合社会标准和原则，技术用于满足这些要求。本文首先调研了来自不同国家和地区使人工智能算法可信的伦理元素的发展；然后重点调查了人工智能可解释性的最新研究。我们对使人工智能具有可解释性的技术与方法进行了深入调研。最后，我们确定了实现可解释人工智能的新趋势。特别地，我们详细阐述了人工智能解释性和自主系统元推理之间的密切联系。元推理的概念是“反思推理”，这与可解释人工智能的意图和目标相一致。这两种方法的结合可能会为未来的可解释人工智能系统铺平道路。 

---
# A Multi-Agent Reinforcement Learning Approach for Cooperative Air-Ground-Human Crowdsensing in Emergency Rescue 

**Title (ZH)**: 多agent强化学习在应急救援中协同空地人 crowdsensing 方法 

**Authors**: Wenhao Lu, Zhengqiu Zhu, Yong Zhao, Yonglin Tian, Junjie Zeng, Jun Zhang, Zhong Liu, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06997)  

**Abstract**: Mobile crowdsensing is evolving beyond traditional human-centric models by integrating heterogeneous entities like unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). Optimizing task allocation among these diverse agents is critical, particularly in challenging emergency rescue scenarios characterized by complex environments, limited communication, and partial observability. This paper tackles the Heterogeneous-Entity Collaborative-Sensing Task Allocation (HECTA) problem specifically for emergency rescue, considering humans, UAVs, and UGVs. We introduce a novel ``Hard-Cooperative'' policy where UGVs prioritize recharging low-battery UAVs, alongside performing their sensing tasks. The primary objective is maximizing the task completion rate (TCR) under strict time constraints. We rigorously formulate this NP-hard problem as a decentralized partially observable Markov decision process (Dec-POMDP) to effectively handle sequential decision-making under uncertainty. To solve this, we propose HECTA4ER, a novel multi-agent reinforcement learning algorithm built upon a Centralized Training with Decentralized Execution architecture. HECTA4ER incorporates tailored designs, including specialized modules for complex feature extraction, utilization of action-observation history via hidden states, and a mixing network integrating global and local information, specifically addressing the challenges of partial observability. Furthermore, theoretical analysis confirms the algorithm's convergence properties. Extensive simulations demonstrate that HECTA4ER significantly outperforms baseline algorithms, achieving an average 18.42% increase in TCR. Crucially, a real-world case study validates the algorithm's effectiveness and robustness in dynamic sensing scenarios, highlighting its strong potential for practical application in emergency response. 

**Abstract (ZH)**: 移动群体感知正通过集成如无人驾驶航空车辆(UAVs)和无人驾驶地面车辆(UGVs)等异构实体，超越传统的以人类为中心的模型。本文针对紧急救援场景，具体解决异构实体协同感知任务分配(HECTA)问题，考虑人类、UAVs和UGVs。提出了一种新颖的“硬协同”策略，其中UGVs优先为低电量UAVs充电，并同时执行感知任务。主要目标是在严格的时间约束下最大化任务完成率(TCR)。将这一NP难问题严格形式化为去中心化的部分可观测马尔可夫决策过程(Dec-POMDP)，以有效处理不确定性下的顺序决策问题。为此，提出了一种基于集中训练与分散执行架构的新颖多智能体强化学习算法HECTA4ER。HECTA4ER包括针对部分可观测性的定制设计，如复杂特征提取模块、隐状态下的动作-观察历史利用以及结合全局和局部信息的混合网络。理论分析证实了算法的收敛性。广泛仿真表明，HECTA4ER在平均TCR方面显著优于基线算法，提高了18.42%。更重要的是，实地案例研究验证了该算法在动态感知场景中的有效性和鲁棒性，突显了其在紧急响应中的强大应用潜力。 

---
# CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging 

**Title (ZH)**: CAT 合并：一种无需训练的模型合并冲突解决方法 

**Authors**: Wenju Sun, Qingyong Li, Yangli-ao Geng, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06977)  

**Abstract**: Multi-task model merging offers a promising paradigm for integrating multiple expert models into a unified model without additional training. Existing state-of-the-art techniques, such as Task Arithmetic and its variants, merge models by accumulating task vectors -- the parameter differences between pretrained and finetuned models. However, task vector accumulation is often hindered by knowledge conflicts, leading to performance degradation. To address this challenge, we propose Conflict-Aware Task Merging (CAT Merging), a novel training-free framework that selectively trims conflict-prone components from the task vectors. CAT Merging introduces several parameter-specific strategies, including projection for linear weights and masking for scaling and shifting parameters in normalization layers. Extensive experiments on vision, language, and vision-language tasks demonstrate that CAT Merging effectively suppresses knowledge conflicts, achieving average accuracy improvements of up to 2.5% (ViT-B/32) and 2.0% (ViT-L/14) over state-of-the-art methods. 

**Abstract (ZH)**: 多任务模型合并提供了一种有希望的范式，用于在无需额外训练的情况下将多个专家模型整合到一个统一模型中。现有最先进的技术，如任务算术及其变体，通过积累任务向量——预训练模型和微调模型之间的参数差异，来合并模型。然而，任务向量的累积往往受到知识冲突的阻碍，导致性能下降。为了解决这一挑战，我们提出了一种新的无训练框架——冲突感知任务合并（CAT 合并），该框架选择性地修剪任务向量中的冲突性组件。CAT 合并引入了几种针对参数的具体策略，包括投影用于线性权重和掩码用于归一化层中的缩放和移位参数。在视觉、语言和视觉-语言任务上的广泛实验表明，CAT 合并有效地抑制了知识冲突，相对于最先进的方法分别实现了高达 2.5%（ViT-B/32）和 2.0%（ViT-L/14）的平均准确率提升。 

---
# From Knowledge to Reasoning: Evaluating LLMs for Ionic Liquids Research in Chemical and Biological Engineering 

**Title (ZH)**: 从知识到推理：评估在化学与生物工程领域离子液体研究中的LLM能力 

**Authors**: Gaurab Sarkar, Sougata Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.06964)  

**Abstract**: Although Large Language Models (LLMs) have achieved remarkable performance in diverse general knowledge and reasoning tasks, their utility in the scientific domain of Chemical and Biological Engineering (CBE) is unclear. Hence, it necessitates challenging evaluation benchmarks that can measure LLM performance in knowledge- and reasoning-based tasks, which is lacking. As a foundational step, we empirically measure the reasoning capabilities of LLMs in CBE. We construct and share an expert-curated dataset of 5,920 examples for benchmarking LLMs' reasoning capabilities in the niche domain of Ionic Liquids (ILs) for carbon sequestration, an emergent solution to reducing global warming. The dataset presents different difficulty levels by varying along the dimensions of linguistic and domain-specific knowledge. Benchmarking three less than 10B parameter open-source LLMs on the dataset suggests that while smaller general-purpose LLMs are knowledgeable about ILs, they lack domain-specific reasoning capabilities. Based on our results, we further discuss considerations for leveraging LLMs for carbon capture research using ILs. Since LLMs have a high carbon footprint, gearing them for IL research can symbiotically benefit both fields and help reach the ambitious carbon neutrality target by 2050. Dataset link: this https URL 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在多种一般知识和推理任务中取得了显著性能，它们在化学和生物工程（CBE）科学领域的实用性尚不清楚。因此，迫切需要具有挑战性的评估基准来衡量LLMs在基于知识和推理的任务中的性能，而这方面尚缺乏。作为基础步骤，我们实证测量了LLMs在CBE领域的推理能力。我们构建并共享了一个由专家精心挑选的5,920个案例组成的语料库，以评估LLMs在离子液体（ILs）用于碳捕获这一新兴减缓全球变暖解决方案的专业领域的推理能力。此语料库通过在语言和领域特定知识维度上的变化呈现了不同难度级别。在该语料库上对三个少于10B参数的开源LLMs进行基准测试表明，虽然更小的通用LLMs对ILs有所了解，但在领域特定的推理能力上却存在不足。基于我们的研究结果，我们进一步讨论了利用ILs进行碳捕获研究的LLMs考虑因素。由于LLMs具有高碳足迹，为IL研究调整它们可以互利地造福两个领域，并有助于在2050年达成雄心勃勃的碳中和目标。数据集链接：this https URL 

---
# Causal knowledge graph analysis identifies adverse drug effects 

**Title (ZH)**: 因果知识图谱分析识别药物不良反应 

**Authors**: Sumyyah Toonsi, Paul Schofield, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2505.06949)  

**Abstract**: Knowledge graphs and structural causal models have each proven valuable for organizing biomedical knowledge and estimating causal effects, but remain largely disconnected: knowledge graphs encode qualitative relationships focusing on facts and deductive reasoning without formal probabilistic semantics, while causal models lack integration with background knowledge in knowledge graphs and have no access to the deductive reasoning capabilities that knowledge graphs provide. To bridge this gap, we introduce a novel formulation of Causal Knowledge Graphs (CKGs) which extend knowledge graphs with formal causal semantics, preserving their deductive capabilities while enabling principled causal inference. CKGs support deconfounding via explicitly marked causal edges and facilitate hypothesis formulation aligned with both encoded and entailed background knowledge. We constructed a Drug-Disease CKG (DD-CKG) integrating disease progression pathways, drug indications, side-effects, and hierarchical disease classification to enable automated large-scale mediation analysis. Applied to UK Biobank and MIMIC-IV cohorts, we tested whether drugs mediate effects between indications and downstream disease progression, adjusting for confounders inferred from the DD-CKG. Our approach successfully reproduced known adverse drug reactions with high precision while identifying previously undocumented significant candidate adverse effects. Further validation through side effect similarity analysis demonstrated that combining our predicted drug effects with established databases significantly improves the prediction of shared drug indications, supporting the clinical relevance of our novel findings. These results demonstrate that our methodology provides a generalizable, knowledge-driven framework for scalable causal inference. 

**Abstract (ZH)**: 基于因果的知識圖譜（Causal Knowledge Graphs, CKGs）融合了結構因果模型和知識圖譜的優點，為生物醫學知識的組織和因果效應的估算提供了系統性的解決方案。 

---
# Towards Artificial General or Personalized Intelligence? A Survey on Foundation Models for Personalized Federated Intelligence 

**Title (ZH)**: 通向通用或个性化人工智能？一种关于个性化联邦智能基础模型的综述 

**Authors**: Yu Qiao, Huy Q. Le, Avi Deb Raha, Phuong-Nam Tran, Apurba Adhikary, Mengchun Zhang, Loc X. Nguyen, Eui-Nam Huh, Dusit Niyato, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06907)  

**Abstract**: The rise of large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, has reshaped the artificial intelligence landscape. As prominent examples of foundational models (FMs) built on LLMs, these models exhibit remarkable capabilities in generating human-like content, bringing us closer to achieving artificial general intelligence (AGI). However, their large-scale nature, sensitivity to privacy concerns, and substantial computational demands present significant challenges to personalized customization for end users. To bridge this gap, this paper presents the vision of artificial personalized intelligence (API), focusing on adapting these powerful models to meet the specific needs and preferences of users while maintaining privacy and efficiency. Specifically, this paper proposes personalized federated intelligence (PFI), which integrates the privacy-preserving advantages of federated learning (FL) with the zero-shot generalization capabilities of FMs, enabling personalized, efficient, and privacy-protective deployment at the edge. We first review recent advances in both FL and FMs, and discuss the potential of leveraging FMs to enhance federated systems. We then present the key motivations behind realizing PFI and explore promising opportunities in this space, including efficient PFI, trustworthy PFI, and PFI empowered by retrieval-augmented generation (RAG). Finally, we outline key challenges and future research directions for deploying FM-powered FL systems at the edge with improved personalization, computational efficiency, and privacy guarantees. Overall, this survey aims to lay the groundwork for the development of API as a complement to AGI, with a particular focus on PFI as a key enabling technique. 

**Abstract (ZH)**: 大型语言模型的兴起：从ChatGPT到Grok-3，这些模型重塑了人工智能格局。作为基于大型语言模型的基础模型（FMs）的杰出示例，这些模型展示了生成人类like内容的显著能力，使我们更接近实现人工通用智能（AGI）。然而，它们的规模化性质、对隐私关注的敏感性和巨大的计算需求为个性化定制带来重大挑战。为弥补这一差距，本文提出了人工个性化智能（API）的愿景，旨在将这些强大模型适应以满足用户的具体需求和偏好，同时保持隐私和效率。具体而言，本文提出个性化联邦智能（PFI），将联邦学习（FL）的隐私保护优势与基础模型（FM）的零样本泛化能力结合起来，以实现边缘上的个性化、高效且隐私保护的部署。我们首先回顾了联邦学习和基础模型的最新进展，并讨论了利用基础模型增强联邦系统的机会。然后，我们探讨了实现PFI的关键动力和这一领域中前景广阔的机会，包括高效的PFI、可信赖的PFI以及由检索增强生成（RAG）赋能的PFI。最后，我们概述了部署支持增强个性化、计算效率和隐私保障的基础模型驱动的联邦学习系统的关键挑战和未来研究方向。总之，本文旨在为API的发展奠定基础，特别是在PFI作为一种关键使能技术方面。 

---
# Embodied Intelligence: The Key to Unblocking Generalized Artificial Intelligence 

**Title (ZH)**: 具身智能：通向通用人工智能的关键 

**Authors**: Jinhao Jiang, Changlin Chen, Shile Feng, Wanru Geng, Zesheng Zhou, Ni Wang, Shuai Li, Feng-Qi Cui, Erbao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06897)  

**Abstract**: The ultimate goal of artificial intelligence (AI) is to achieve Artificial General Intelligence (AGI). Embodied Artificial Intelligence (EAI), which involves intelligent systems with physical presence and real-time interaction with the environment, has emerged as a key research direction in pursuit of AGI. While advancements in deep learning, reinforcement learning, large-scale language models, and multimodal technologies have significantly contributed to the progress of EAI, most existing reviews focus on specific technologies or applications. A systematic overview, particularly one that explores the direct connection between EAI and AGI, remains scarce. This paper examines EAI as a foundational approach to AGI, systematically analyzing its four core modules: perception, intelligent decision-making, action, and feedback. We provide a detailed discussion of how each module contributes to the six core principles of AGI. Additionally, we discuss future trends, challenges, and research directions in EAI, emphasizing its potential as a cornerstone for AGI development. Our findings suggest that EAI's integration of dynamic learning and real-world interaction is essential for bridging the gap between narrow AI and AGI. 

**Abstract (ZH)**: 人工通用人工智能导向的具身人工智能：从感知到反馈的系统分析及其未来展望 

---
# Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction 

**Title (ZH)**: 超越模式：利用因果逻辑进行自主驾驶轨迹预测 

**Authors**: Bonan Wang, Haicheng Liao, Chengyue Wang, Bin Rao, Yanchen Guan, Guyang Yu, Jiaxun Zhang, Songning Lai, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06856)  

**Abstract**: Accurate trajectory prediction has long been a major challenge for autonomous driving (AD). Traditional data-driven models predominantly rely on statistical correlations, often overlooking the causal relationships that govern traffic behavior. In this paper, we introduce a novel trajectory prediction framework that leverages causal inference to enhance predictive robustness, generalization, and accuracy. By decomposing the environment into spatial and temporal components, our approach identifies and mitigates spurious correlations, uncovering genuine causal relationships. We also employ a progressive fusion strategy to integrate multimodal information, simulating human-like reasoning processes and enabling real-time inference. Evaluations on five real-world datasets--ApolloScape, nuScenes, NGSIM, HighD, and MoCAD--demonstrate our model's superiority over existing state-of-the-art (SOTA) methods, with improvements in key metrics such as RMSE and FDE. Our findings highlight the potential of causal reasoning to transform trajectory prediction, paving the way for robust AD systems. 

**Abstract (ZH)**: 准确的轨迹预测一直是自动驾驶（AD）领域的重大挑战。传统的数据驱动模型主要依赖于统计相关性，往往忽略了交通行为背后的因果关系。本文提出了一种新的轨迹预测框架，利用因果推断来增强预测的稳健性、泛化能力和准确性。通过将环境分解为时空组件，本方法识别并缓解了虚假相关性，揭示了真实的因果关系。我们还采用渐进融合策略来整合多模态信息，模拟人类推理过程，实现实时推理。在ApolloScape、nuScenes、NGSIM、HighD和MoCAD五个现实世界数据集上的评估表明，我们的模型在关键指标如RMSE和FDE方面优于现有最先进的（SOTA）方法。我们的研究结果突显了因果推理在轨迹预测中的潜力，为稳健的AD系统铺平了道路。 

---
# Control Plane as a Tool: A Scalable Design Pattern for Agentic AI Systems 

**Title (ZH)**: 控制面作为工具：自主人工智能系统的大规模设计模式 

**Authors**: Sivasathivel Kandasamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.06817)  

**Abstract**: Agentic AI systems represent a new frontier in artificial intelligence, where agents often based on large language models(LLMs) interact with tools, environments, and other agents to accomplish tasks with a degree of autonomy. These systems show promise across a range of domains, but their architectural underpinnings remain immature. This paper conducts a comprehensive review of the types of agents, their modes of interaction with the environment, and the infrastructural and architectural challenges that emerge. We identify a gap in how these systems manage tool orchestration at scale and propose a reusable design abstraction: the "Control Plane as a Tool" pattern. This pattern allows developers to expose a single tool interface to an agent while encapsulating modular tool routing logic behind it. We position this pattern within the broader context of agent design and argue that it addresses several key challenges in scaling, safety, and extensibility. 

**Abstract (ZH)**: 基于大型语言模型的代理AI系统构成了人工智能的新前沿，这些代理通常与工具、环境和其他代理交互以实现具有一定程度自主性的任务。这些系统在多个领域显示出潜力，但其架构基础仍不够成熟。本文对代理的类型、它们与环境的交互模式以及随之而来的基础设施和架构挑战进行了全面评审。我们发现这些系统在大规模工具编排管理方面存在差距，并提出了一种可重复使用的设计抽象：“控制平面即工具”模式。该模式允许开发者为代理暴露单一工具接口，同时将模块化工具路由逻辑封装在其后。我们将该模式置于更广泛的代理设计背景下，并论证它在扩展、安全性和可扩展性方面解决了多个关键问题。 

---
# Value Iteration with Guessing for Markov Chains and Markov Decision Processes 

**Title (ZH)**: 基于猜测的值迭代方法在马尔可夫链与马尔可夫决策过程中的应用 

**Authors**: Krishnendu Chatterjee, Mahdi JafariRaviz, Raimundo Saona, Jakub Svoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.06769)  

**Abstract**: Two standard models for probabilistic systems are Markov chains (MCs) and Markov decision processes (MDPs). Classic objectives for such probabilistic models for control and planning problems are reachability and stochastic shortest path. The widely studied algorithmic approach for these problems is the Value Iteration (VI) algorithm which iteratively applies local updates called Bellman updates. There are many practical approaches for VI in the literature but they all require exponentially many Bellman updates for MCs in the worst case. A preprocessing step is an algorithm that is discrete, graph-theoretical, and requires linear space. An important open question is whether, after a polynomial-time preprocessing, VI can be achieved with sub-exponentially many Bellman updates. In this work, we present a new approach for VI based on guessing values. Our theoretical contributions are twofold. First, for MCs, we present an almost-linear-time preprocessing algorithm after which, along with guessing values, VI requires only subexponentially many Bellman updates. Second, we present an improved analysis of the speed of convergence of VI for MDPs. Finally, we present a practical algorithm for MDPs based on our new approach. Experimental results show that our approach provides a considerable improvement over existing VI-based approaches on several benchmark examples from the literature. 

**Abstract (ZH)**: 两类概率系统的标准模型是马尔可夫链（MCs）和马尔可夫决策过程（MDPs）。这类概率模型的经典目标是可达性和随机最短路径。这些目标的经典算法方法是值迭代（VI）算法，该算法通过应用局部更新（称为贝尔曼更新）进行迭代。文献中有很多实用的VI方法，但在最坏情况下，它们都需要对MCs进行指数数量的贝尔曼更新。预处理步骤是一种离散、图论性的算法，并且只需要线性空间。一个重要的开放问题是，在多项式时间预处理之后，VI是否可以在亚指数数量的贝尔曼更新下实现。在这项工作中，我们提出了一种基于猜测值的新VI方法。我们的理论贡献主要有两点。首先，对于MCs，我们提出了一种几乎线性时间的预处理算法，之后结合猜测值，VI只需亚指数数量的贝尔曼更新。第二，我们对VI对于MDPs的收敛速度进行了改进分析。最后，我们基于新方法提出了一个MDPs的实际算法。实验结果显示，我们的方法在文献中的多个基准示例上显著优于现有的VI方法。 

---
# Bi-level Mean Field: Dynamic Grouping for Large-Scale MARL 

**Title (ZH)**: 双层均场：大规模 MARL 的动态分组 

**Authors**: Yuxuan Zheng, Yihe Zhou, Feiyang Xu, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06706)  

**Abstract**: Large-scale Multi-Agent Reinforcement Learning (MARL) often suffers from the curse of dimensionality, as the exponential growth in agent interactions significantly increases computational complexity and impedes learning efficiency. To mitigate this, existing efforts that rely on Mean Field (MF) simplify the interaction landscape by approximating neighboring agents as a single mean agent, thus reducing overall complexity to pairwise interactions. However, these MF methods inevitably fail to account for individual differences, leading to aggregation noise caused by inaccurate iterative updates during MF learning. In this paper, we propose a Bi-level Mean Field (BMF) method to capture agent diversity with dynamic grouping in large-scale MARL, which can alleviate aggregation noise via bi-level interaction. Specifically, BMF introduces a dynamic group assignment module, which employs a Variational AutoEncoder (VAE) to learn the representations of agents, facilitating their dynamic grouping over time. Furthermore, we propose a bi-level interaction module to model both inter- and intra-group interactions for effective neighboring aggregation. Experiments across various tasks demonstrate that the proposed BMF yields results superior to the state-of-the-art methods. Our code will be made publicly available. 

**Abstract (ZH)**: 大规模多智能体 reinforcement 学习 (MARL) 往往受到维数灾的困扰，由于代理交互的指数级增长显著增加计算复杂性并阻碍学习效率。为缓解这一问题，现有依赖均场（MF）的方法通过近似邻近代理为单一均场代理来简化交互场景，从而将总体复杂性降低为两两交互。然而，这些MF方法不可避免地无法考虑个体差异，导致均场学习过程中因迭代更新不准确引起的聚合噪声。在本文中，我们提出了一种双层均场（BMF）方法，通过动态分组来捕捉大规模MARL中的代理多样性，借助双层交互来缓解聚合噪声。具体而言，BMF引入了一个动态组分配模块，该模块使用变分自编码器（VAE）学习代理的表示，促进其随时间进行动态分组。此外，我们提出了一种双层交互模块来建模组内和组间交互，以实现有效的邻近聚合。在多种任务上的实验表明，所提出的BMF方法优于现有最佳方法。我们的代码将公开发布。 

---
# A Survey on Data-Driven Modeling of Human Drivers' Lane-Changing Decisions 

**Title (ZH)**: 基于数据的人类驾驶车道变换决策建模综述 

**Authors**: Linxuan Huang, Dong-Fan Xie, Li Li, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.06680)  

**Abstract**: Lane-changing (LC) behavior, a critical yet complex driving maneuver, significantly influences driving safety and traffic dynamics. Traditional analytical LC decision (LCD) models, while effective in specific environments, often oversimplify behavioral heterogeneity and complex interactions, limiting their capacity to capture real LCD. Data-driven approaches address these gaps by leveraging rich empirical data and machine learning to decode latent decision-making patterns, enabling adaptive LCD modeling in dynamic environments. In light of the rapid development of artificial intelligence and the demand for data-driven models oriented towards connected vehicles and autonomous vehicles, this paper presents a comprehensive survey of data-driven LCD models, with a particular focus on human drivers LC decision-making. It systematically reviews the modeling framework, covering data sources and preprocessing, model inputs and outputs, objectives, structures, and validation methods. This survey further discusses the opportunities and challenges faced by data-driven LCD models, including driving safety, uncertainty, as well as the integration and improvement of technical frameworks. 

**Abstract (ZH)**: 基于数据的变道决策模型综述：针对连接车辆和自动驾驶车辆的人类驾驶变道决策 

---
# Exploring Multimodal Foundation AI and Expert-in-the-Loop for Sustainable Management of Wild Salmon Fisheries in Indigenous Rivers 

**Title (ZH)**: 探索多模态基础人工智能和专家在环可持续管理原住民河流野生鲑鱼渔场的方法 

**Authors**: Chi Xu, Yili Jin, Sami Ma, Rongsheng Qian, Hao Fang, Jiangchuan Liu, Xue Liu, Edith C.H. Ngai, William I. Atlas, Katrina M. Connors, Mark A. Spoljaric  

**Link**: [PDF](https://arxiv.org/pdf/2505.06637)  

**Abstract**: Wild salmon are essential to the ecological, economic, and cultural sustainability of the North Pacific Rim. Yet climate variability, habitat loss, and data limitations in remote ecosystems that lack basic infrastructure support pose significant challenges to effective fisheries management. This project explores the integration of multimodal foundation AI and expert-in-the-loop frameworks to enhance wild salmon monitoring and sustainable fisheries management in Indigenous rivers across Pacific Northwest. By leveraging video and sonar-based monitoring, we develop AI-powered tools for automated species identification, counting, and length measurement, reducing manual effort, expediting delivery of results, and improving decision-making accuracy. Expert validation and active learning frameworks ensure ecological relevance while reducing annotation burdens. To address unique technical and societal challenges, we bring together a cross-domain, interdisciplinary team of university researchers, fisheries biologists, Indigenous stewardship practitioners, government agencies, and conservation organizations. Through these collaborations, our research fosters ethical AI co-development, open data sharing, and culturally informed fisheries management. 

**Abstract (ZH)**: 野生鲑鱼对北太平洋地区的生态、经济和文化可持续性至关重要。然而，气候变化、栖息地丧失以及缺乏基本基础设施支持的偏远生态系统的数据限制，给有效的渔业管理带来了重大挑战。本项目探索多模态基础AI与专家在环框架的集成，以增强对太平洋西北地区原住民河流中野生鲑鱼的监测，并推动可持续渔业管理。通过利用视频和声纳监测，我们开发了基于AI的工具，以实现自动物种识别、计数和长度测量，减少人工操作，加快结果交付速度，并提高决策准确性。专家验证和主动学习框架确保生态相关性，同时减少标注负担。为应对独特的技术和社会挑战，我们汇聚了一个跨领域、跨学科的研究团队，包括高校研究人员、渔业生物学家、原住民管理实践者、政府部门和保护组织。通过这些合作，我们的研究促进了伦理AI协同开发、开放数据共享，并推动了文化导向的渔业管理。 

---
# TAROT: Towards Essentially Domain-Invariant Robustness with Theoretical Justification 

**Title (ZH)**: TAROT：向着具有理论依据的本领域基本不变鲁棒性研究 

**Authors**: Dongyoon Yang, Jihu Lee, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06580)  

**Abstract**: Robust domain adaptation against adversarial attacks is a critical research area that aims to develop models capable of maintaining consistent performance across diverse and challenging domains. In this paper, we derive a new generalization bound for robust risk on the target domain using a novel divergence measure specifically designed for robust domain adaptation. Building upon this, we propose a new algorithm named TAROT, which is designed to enhance both domain adaptability and robustness. Through extensive experiments, TAROT not only surpasses state-of-the-art methods in accuracy and robustness but also significantly enhances domain generalization and scalability by effectively learning domain-invariant features. In particular, TAROT achieves superior performance on the challenging DomainNet dataset, demonstrating its ability to learn domain-invariant representations that generalize well across different domains, including unseen ones. These results highlight the broader applicability of our approach in real-world domain adaptation scenarios. 

**Abstract (ZH)**: 对抗攻击下稳健领域适应的鲁棒泛化边界的derive以及一种新的TAROT算法研究 

---
# Online Feedback Efficient Active Target Discovery in Partially Observable Environments 

**Title (ZH)**: 在线反馈驱动的主动目标发现Partial观察环境中的高效方法 

**Authors**: Anindya Sarkar, Binglin Ji, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2505.06535)  

**Abstract**: In various scientific and engineering domains, where data acquisition is costly, such as in medical imaging, environmental monitoring, or remote sensing, strategic sampling from unobserved regions, guided by prior observations, is essential to maximize target discovery within a limited sampling budget. In this work, we introduce Diffusion-guided Active Target Discovery (DiffATD), a novel method that leverages diffusion dynamics for active target discovery. DiffATD maintains a belief distribution over each unobserved state in the environment, using this distribution to dynamically balance exploration-exploitation. Exploration reduces uncertainty by sampling regions with the highest expected entropy, while exploitation targets areas with the highest likelihood of discovering the target, indicated by the belief distribution and an incrementally trained reward model designed to learn the characteristics of the target. DiffATD enables efficient target discovery in a partially observable environment within a fixed sampling budget, all without relying on any prior supervised training. Furthermore, DiffATD offers interpretability, unlike existing black-box policies that require extensive supervised training. Through extensive experiments and ablation studies across diverse domains, including medical imaging and remote sensing, we show that DiffATD performs significantly better than baselines and competitively with supervised methods that operate under full environmental observability. 

**Abstract (ZH)**: 在医学成像、环境监控或遥感等数据采集成本较高的科学和工程领域，通过利用先验观测指导未观测区域的战略性采样，以最大限度地在有限的采样预算内发现目标，是至关重要的。本文引入了扩散引导主动目标发现（DiffATD）方法，该方法利用扩散动力学进行主动目标发现。DiffATD 在环境中每个未观测状态上维护一种信念分布，利用该分布动态平衡探索与利用。探索通过采样具有最高预期熵的区域来减少不确定性，而利用信念分布和一个逐步训练的奖励模型（该模型设计用于学习目标特性）来指向具有最高目标发现可能性的区域，从而实现目标发现。DiffATD 不依赖任何先验监督训练，即可在固定采样预算内高效地在部分可观测环境中发现目标，并且提供了可解释性，与现有的黑盒策略相比省去了大量的监督训练。通过在包括医学成像和遥感在内的多个领域的广泛实验和消融研究，我们展示了 DiffATD 在基线方法上显著优越的表现，并且在完全可观测环境中操作的监督方法中表现具有竞争力。 

---
# A Point-Based Algorithm for Distributional Reinforcement Learning in Partially Observable Domains 

**Title (ZH)**: 基于点的算法在部分可观测域中的分布强化学习 

**Authors**: Larry Preuett III  

**Link**: [PDF](https://arxiv.org/pdf/2505.06518)  

**Abstract**: In many real-world planning tasks, agents must tackle uncertainty about the environment's state and variability in the outcomes of any chosen policy. We address both forms of uncertainty as a first step toward safer algorithms in partially observable settings. Specifically, we extend Distributional Reinforcement Learning (DistRL)-which models the entire return distribution for fully observable domains-to Partially Observable Markov Decision Processes (POMDPs), allowing an agent to learn the distribution of returns for each conditional plan. Concretely, we introduce new distributional Bellman operators for partial observability and prove their convergence under the supremum p-Wasserstein metric. We also propose a finite representation of these return distributions via psi-vectors, generalizing the classical alpha-vectors in POMDP solvers. Building on this, we develop Distributional Point-Based Value Iteration (DPBVI), which integrates psi-vectors into a standard point-based backup procedure-bridging DistRL and POMDP planning. By tracking return distributions, DPBVI naturally enables risk-sensitive control in domains where rare, high-impact events must be carefully managed. We provide source code to foster further research in robust decision-making under partial observability. 

**Abstract (ZH)**: 在部分可观测环境中处理不确定性形式的安全算法初步研究：将分布强化学习扩展到部分可观测马尔可夫决策过程 

---
# Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities 

**Title (ZH)**: 文本到CAD查询：具有可扩展大型模型能力的CAD生成新范式 

**Authors**: Haoyang Xie, Feng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2505.06507)  

**Abstract**: Computer-aided design (CAD) is fundamental to modern engineering and manufacturing, but creating CAD models still requires expert knowledge and specialized software. Recent advances in large language models (LLMs) open up the possibility of generative CAD, where natural language is directly translated into parametric 3D models. However, most existing methods generate task-specific command sequences that pretrained models cannot directly handle. These sequences must be converted into CAD representations such as CAD vectors before a 3D model can be produced, which requires training models from scratch and adds unnecessary complexity. To tackle this issue, we propose generating CadQuery code directly from text, leveraging the strengths of pretrained LLMs to produce 3D models without intermediate representations, using this Python-based scripting language. Since LLMs already excel at Python generation and spatial reasoning, fine-tuning them on Text-to-CadQuery data proves highly effective. Given that these capabilities typically improve with scale, we hypothesize that larger models will perform better after fine-tuning. To enable this, we augment the Text2CAD dataset with 170,000 CadQuery annotations. We fine-tune six open-source LLMs of varying sizes and observe consistent improvements. Our best model achieves a top-1 exact match of 69.3%, up from 58.8%, and reduces Chamfer Distance by 48.6%. Project page: this https URL. 

**Abstract (ZH)**: 基于文本直接生成CadQuery代码的预训练大型语言模型驱动的生成式CAD 

---
# On Definite Iterated Belief Revision with Belief Algebras 

**Title (ZH)**: 基于信念代数的确定性迭代信念修订 

**Authors**: Hua Meng, Zhiguo Long, Michael Sioutis, Zhengchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06505)  

**Abstract**: Traditional logic-based belief revision research focuses on designing rules to constrain the behavior of revision operators. Frameworks have been proposed to characterize iterated revision rules, but they are often too loose, leading to multiple revision operators that all satisfy the rules under the same belief condition. In many practical applications, such as safety critical ones, it is important to specify a definite revision operator to enable agents to iteratively revise their beliefs in a deterministic way. In this paper, we propose a novel framework for iterated belief revision by characterizing belief information through preference relations. Semantically, both beliefs and new evidence are represented as belief algebras, which provide a rich and expressive foundation for belief revision. Building on traditional revision rules, we introduce additional postulates for revision with belief algebra, including an upper-bound constraint on the outcomes of revision. We prove that the revision result is uniquely determined given the current belief state and new evidence. Furthermore, to make the framework more useful in practice, we develop a particular algorithm for performing the proposed revision process. We argue that this approach may offer a more predictable and principled method for belief revision, making it suitable for real-world applications. 

**Abstract (ZH)**: 基于偏好关系的迭代信念修订新框架：提供确定性信念更新的方法 

---
# SmartPilot: A Multiagent CoPilot for Adaptive and Intelligent Manufacturing 

**Title (ZH)**: SmartPilot: 一个多代理协作副驾 for 自适应与智能制造 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Alaa Al Ghazo, Darssan Eswaramoorthi, Harleen Kaur, Sara Shree Muthuselvam, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06492)  

**Abstract**: In the dynamic landscape of Industry 4.0, achieving efficiency, precision, and adaptability is essential to optimize manufacturing operations. Industries suffer due to supply chain disruptions caused by anomalies, which are being detected by current AI models but leaving domain experts uncertain without deeper insights into these anomalies. Additionally, operational inefficiencies persist due to inaccurate production forecasts and the limited effectiveness of traditional AI models for processing complex sensor data. Despite these advancements, existing systems lack the seamless integration of these capabilities needed to create a truly unified solution for enhancing production and decision-making. We propose SmartPilot, a neurosymbolic, multiagent CoPilot designed for advanced reasoning and contextual decision-making to address these challenges. SmartPilot processes multimodal sensor data and is compact to deploy on edge devices. It focuses on three key tasks: anomaly prediction, production forecasting, and domain-specific question answering. By bridging the gap between AI capabilities and real-world industrial needs, SmartPilot empowers industries with intelligent decision-making and drives transformative innovation in manufacturing. The demonstration video, datasets, and supplementary materials are available at this https URL. 

**Abstract (ZH)**: 在Industry 4.0的动态背景下，实现效率、精确性和适应性对于优化制造运营至关重要。由于异常引起的供应链中断对行业造成影响，当前的AI模型能够检测这些异常，但缺乏深入洞察使领域专家感到不确定。此外，由于生产预测不够准确以及传统AI模型处理复杂传感器数据效果有限，运营效率持续低下。尽管取得了这些进展，现有系统仍缺乏将这些能力无缝集成的机制，以创建真正统一的解决方案，以增强生产能力和决策制定。我们提出SmartPilot，一个神经符号型的多代理 Copilot，专为高级推理和上下文决策制定而设计，以应对这些挑战。SmartPilot处理多模态传感器数据，并可在边缘设备上进行紧凑部署。它专注于三项关键任务：异常预测、生产预测和领域特定的问题回答。通过弥合AI能力和实际工业需求之间的差距，SmartPilot使行业能够进行智能决策，并推动制造领域的创新变革。详细演示视频、数据集和补充材料请访问此网址：[该网址]。 

---
# KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery 

**Title (ZH)**: KCluster：一种基于大语言模型的知识组件发现聚类方法 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.06469)  

**Abstract**: Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction. 

**Abstract (ZH)**: 基于大型语言模型诱导相似度度量的K集群知识组件发现算法 

---
# Opening the Scope of Openness in AI 

**Title (ZH)**: 打开人工智能开放性的范围 

**Authors**: Tamara Paris, AJung Moon, Jin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06464)  

**Abstract**: The concept of openness in AI has so far been heavily inspired by the definition and community practice of open source software. This positions openness in AI as having positive connotations; it introduces assumptions of certain advantages, such as collaborative innovation and transparency. However, the practices and benefits of open source software are not fully transferable to AI, which has its own challenges. Framing a notion of openness tailored to AI is crucial to addressing its growing societal implications, risks, and capabilities. We argue that considering the fundamental scope of openness in different disciplines will broaden discussions, introduce important perspectives, and reflect on what openness in AI should mean. Toward this goal, we qualitatively analyze 98 concepts of openness discovered from topic modeling, through which we develop a taxonomy of openness. Using this taxonomy as an instrument, we situate the current discussion on AI openness, identify gaps and highlight links with other disciplines. Our work contributes to the recent efforts in framing openness in AI by reflecting principles and practices of openness beyond open source software and calls for a more holistic view of openness in terms of actions, system properties, and ethical objectives. 

**Abstract (ZH)**: AI领域的开放性概念：超越开源软件的原则与实践Towards AI领域的开放性概念：超越开源软件的原则与实践 

---
# Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming 

**Title (ZH)**: 基于LLMs和回答集编程的可靠协作会话代理系统 

**Authors**: Yankai Zeng, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06438)  

**Abstract**: As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable. 

**Abstract (ZH)**: 基于大型语言模型的基于逻辑编程的对话代理双Agent框架：以快餐餐厅为例的安全可靠对话代理设计与应用 

---
# A Grounded Memory System For Smart Personal Assistants 

**Title (ZH)**: 基于内存的智能个人助理系统 

**Authors**: Felix Ocker, Jörg Deigmöller, Pavel Smirnov, Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2505.06328)  

**Abstract**: A wide variety of agentic AI applications - ranging from cognitive assistants for dementia patients to robotics - demand a robust memory system grounded in reality. In this paper, we propose such a memory system consisting of three components. First, we combine Vision Language Models for image captioning and entity disambiguation with Large Language Models for consistent information extraction during perception. Second, the extracted information is represented in a memory consisting of a knowledge graph enhanced by vector embeddings to efficiently manage relational information. Third, we combine semantic search and graph query generation for question answering via Retrieval Augmented Generation. We illustrate the system's working and potential using a real-world example. 

**Abstract (ZH)**: 多种代理型AI应用——从痴呆患者的认知辅助到机器人技术——需要一个基于现实的稳健记忆系统。本文提出了一种这样的记忆系统，由三个组件构成。首先，我们将视觉语言模型用于图像描述和实体消歧，与大型语言模型结合，实现感知过程中一致的信息提取。其次，提取的信息被表示在一个由向量嵌入增强的知识图谱中，以高效管理关系信息。第三，我们通过检索增强生成结合语义搜索和图查询生成来进行问答。我们通过一个实际例子展示该系统的运作及其潜力。 

---
# BedreFlyt: Improving Patient Flows through Hospital Wards with Digital Twins 

**Title (ZH)**: BedreFlyt: 通过数字双胞胎优化医院病区的患者流动 

**Authors**: Riccardo Sieve, Paul Kobialka, Laura Slaughter, Rudolf Schlatte, Einar Broch Johnsen, Silvia Lizeth Tapia Tarifa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06287)  

**Abstract**: Digital twins are emerging as a valuable tool for short-term decision-making as well as for long-term strategic planning across numerous domains, including process industry, energy, space, transport, and healthcare. This paper reports on our ongoing work on designing a digital twin to enhance resource planning, e.g., for the in-patient ward needs in hospitals. By leveraging executable formal models for system exploration, ontologies for knowledge representation and an SMT solver for constraint satisfiability, our approach aims to explore hypothetical "what-if" scenarios to improve strategic planning processes, as well as to solve concrete, short-term decision-making tasks. Our proposed solution uses the executable formal model to turn a stream of arriving patients, that need to be hospitalized, into a stream of optimization problems, e.g., capturing daily inpatient ward needs, that can be solved by SMT techniques. The knowledge base, which formalizes domain knowledge, is used to model the needed configuration in the digital twin, allowing the twin to support both short-term decision-making and long-term strategic planning by generating scenarios spanning average-case as well as worst-case resource needs, depending on the expected treatment of patients, as well as ranging over variations in available resources, e.g., bed distribution in different rooms. We illustrate our digital twin architecture by considering the problem of bed bay allocation in a hospital ward. 

**Abstract (ZH)**: 数字孪生在医疗领域住院病房资源配置中的应用：短中期决策与战略规划的增强 

---
# H$^{\mathbf{3}}$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning 

**Title (ZH)**: H$^{\mathbf{3}}$DP: 三层次扩散策略用于视听运动学习 

**Authors**: Yiyang Lu, Yufeng Tian, Zhecheng Yuan, Xianbang Wang, Pu Hua, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07819)  

**Abstract**: Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce $\textbf{Triply-Hierarchical Diffusion Policy}~(\textbf{H$^{\mathbf{3}}$DP})$, a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^{3}$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^{3}$DP yields a $\mathbf{+27.5\%}$ average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: this https URL. 

**Abstract (ZH)**: 三重嵌套扩散政策（Triply-Hierarchical Diffusion Policy, H³DP）：强化视觉与动作生成集成的visuomotor学习框架 

---
# A class of distributed automata that contains the modal mu-fragment 

**Title (ZH)**: 一类包含模态mu片段的分布式自动机类 

**Authors**: Veeti Ahvonen, Damian Heiman, Antti Kuusisto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07816)  

**Abstract**: This paper gives a translation from the $\mu$-fragment of the graded modal $\mu$-calculus to a class of distributed message-passing automata. As a corollary, we obtain an alternative proof for a theorem from \cite{ahvonen_neurips} stating that recurrent graph neural networks working with reals and graded modal substitution calculus have the same expressive power in restriction to the logic monadic second-order logic MSO. 

**Abstract (ZH)**: 本文将分级模态μ-演算的μ片段翻译为一类分布式消息传递自动机。作为推论，我们得到了一个关于《[ahvonen_neurips]》中定理的替代证明，该定理指出，使用实数和分级模态替换演算的循环图神经网络在逻辑单调二阶逻辑MSO方面的表达能力相同。 

---
# DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies 

**Title (ZH)**: DexWild: 动手动脚的户外机器人政策中的灵巧人类交互 

**Authors**: Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2505.07813)  

**Abstract**: Large-scale, diverse robot datasets have emerged as a promising path toward enabling dexterous manipulation policies to generalize to novel environments, but acquiring such datasets presents many challenges. While teleoperation provides high-fidelity datasets, its high cost limits its scalability. Instead, what if people could use their own hands, just as they do in everyday life, to collect data? In DexWild, a diverse team of data collectors uses their hands to collect hours of interactions across a multitude of environments and objects. To record this data, we create DexWild-System, a low-cost, mobile, and easy-to-use device. The DexWild learning framework co-trains on both human and robot demonstrations, leading to improved performance compared to training on each dataset individually. This combination results in robust robot policies capable of generalizing to novel environments, tasks, and embodiments with minimal additional robot-specific data. Experimental results demonstrate that DexWild significantly improves performance, achieving a 68.5% success rate in unseen environments-nearly four times higher than policies trained with robot data only-and offering 5.8x better cross-embodiment generalization. Video results, codebases, and instructions at this https URL 

**Abstract (ZH)**: 大规模多样化的机器人数据集已成为实现灵巧操作策略泛化到新型环境的一种有前景的方法，但获取这样的数据集面临着许多挑战。虽然远程操作可以提供高保真数据集，但其高昂的成本限制了其 scalability。相反，如果人们可以像日常生活一样使用他们的手来收集数据会怎样？在 DexWild 中，一个多样化的数据收集团队使用他们的手在多种环境和对象上收集数小时的交互数据。为了记录这些数据，我们创建了 DexWild-System，这是一个低成本、便携且易于使用的设备。DexWild 学习框架在人类和机器人演示数据上共同训练，与单独训练每个数据集相比，能够取得更好的性能。这种组合产生了鲁棒性强的机器人策略，能够在最小限度增加特定于机器人数据的情况下泛化到新型环境、任务和不同体态。实验结果表明，DexWild 显著提高了性能，在未见过的环境中取得了 68.5% 的成功率——几乎是仅使用机器人数据训练的策略成功率的四倍，并且在不同体态泛化方面表现出了 5.8 倍的改进。更多信息、视频结果、代码库和使用说明请访问：this https URL。 

---
# A Comparative Analysis of Static Word Embeddings for Hungarian 

**Title (ZH)**: 匈牙利语静态词嵌入的比较分析 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07809)  

**Abstract**: This paper presents a comprehensive analysis of various static word embeddings for Hungarian, including traditional models such as Word2Vec, FastText, as well as static embeddings derived from BERT-based models using different extraction methods. We evaluate these embeddings on both intrinsic and extrinsic tasks to provide a holistic view of their performance. For intrinsic evaluation, we employ a word analogy task, which assesses the embeddings ability to capture semantic and syntactic relationships. Our results indicate that traditional static embeddings, particularly FastText, excel in this task, achieving high accuracy and mean reciprocal rank (MRR) scores. Among the BERT-based models, the X2Static method for extracting static embeddings demonstrates superior performance compared to decontextualized and aggregate methods, approaching the effectiveness of traditional static embeddings. For extrinsic evaluation, we utilize a bidirectional LSTM model to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging tasks. The results reveal that embeddings derived from dynamic models, especially those extracted using the X2Static method, outperform purely static embeddings. Notably, ELMo embeddings achieve the highest accuracy in both NER and POS tagging tasks, underscoring the benefits of contextualized representations even when used in a static form. Our findings highlight the continued relevance of static word embeddings in NLP applications and the potential of advanced extraction methods to enhance the utility of BERT-based models. This piece of research contributes to the understanding of embedding performance in the Hungarian language and provides valuable insights for future developments in the field. The training scripts, evaluation codes, restricted vocabulary, and extracted embeddings will be made publicly available to support further research and reproducibility. 

**Abstract (ZH)**: 本文对多种Hungarian静态词嵌入进行了全面分析，包括传统的Word2Vec和FastText模型，以及使用不同提取方法从BERT基模拟能源生的静态嵌入。我们通过内在和外在任务的评估，提供了对这些嵌入性能的整体视角。内在评估采用词类比任务，评估嵌入捕捉语义和句法关系的能力。结果表明，传统静态嵌入，特别是FastText，在此任务中表现出色，获得高准确率和均值倒数排名（MRR）分数。在基于BERT的模型中，采用X2Static方法提取的静态嵌入的表现优于基于上下文和聚合的方法，接近传统静态嵌入的有效性。在外在评估中，我们使用双向LSTM模型进行命名实体识别（NER）和词性标注（POS）任务。结果显示，动态模型衍生的嵌入，尤其是使用X2Static方法提取的嵌入，优于单纯的静态嵌入。值得注意的是，ELMo嵌入在NER和POS标注任务中均获得最高准确率，突显了即使在静态形式下使用上下文表示的优势。本文的研究结果突显了静态词嵌入在NLP应用中的持续相关性，并展示了高级提取方法增强基于BERT模型用途的潜力。本文的研究为理解匈牙利语嵌入性能提供了见解，并为该领域的未来发展方向提供了宝贵信息。用于训练的脚本、评估代码、受限词汇表以及提取的嵌入将对外公开，以支持进一步研究和可重复性。 

---
# Improving Trajectory Stitching with Flow Models 

**Title (ZH)**: 基于流模型改进轨迹拼接 

**Authors**: Reece O'Mahoney, Wanming Yu, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2505.07802)  

**Abstract**: Generative models have shown great promise as trajectory planners, given their affinity to modeling complex distributions and guidable inference process. Previous works have successfully applied these in the context of robotic manipulation but perform poorly when the required solution does not exist as a complete trajectory within the training set. We identify that this is a result of being unable to plan via stitching, and subsequently address the architectural and dataset choices needed to remedy this. On top of this, we propose a novel addition to the training and inference procedures to both stabilize and enhance these capabilities. We demonstrate the efficacy of our approach by generating plans with out of distribution boundary conditions and performing obstacle avoidance on the Franka Panda in simulation and on real hardware. In both of these tasks our method performs significantly better than the baselines and is able to avoid obstacles up to four times as large. 

**Abstract (ZH)**: 生成模型在轨迹规划中的应用展示了巨大潜力，这得益于其对复杂分布建模的亲和力和可指导的推断过程。以往的工作在机器人的操作场景中取得了成功，但在要求的解在训练集中不存在完整轨迹时表现不佳。我们发现这是由于无法通过拼接来规划轨迹，因此我们针对这一问题改进了架构和数据集的选择。在此基础上，我们提出了一种新的训练和推理程序中的方法，以稳定并增强这些能力。通过在仿真和实际硬件上生成超出分布边界的计划并进行避障实验，展示了该方法的有效性。在这些任务中，我们的方法显著优于基线方法，并且能避开大小达到四倍的障碍物。 

---
# Learning Dynamics in Continual Pre-Training for Large Language Models 

**Title (ZH)**: 连续预训练中大型语言模型的学习动力学 

**Authors**: Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07796)  

**Abstract**: Continual Pre-Training (CPT) has become a popular and effective method to apply strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with domain performance measured via validation losses. We have observed that the CPT loss curve fundamentally characterizes the transition from one curve to another hidden curve, and could be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines the two factors, enabling the prediction of loss at any (continual) training steps and across learning rate schedules (LRS) in CPT. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, replay ratio, etc. Moreover, our approach can be adapted to customize training hyper-parameters to different CPT goals such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and training hyper-parameters. 

**Abstract (ZH)**: 持续预训练（CPT）已成为将强基础模型应用于特定下游任务的有效方法。在本文中，我们探索大规模语言模型在整个CPT过程中的学习动态。我们具体关注每一步训练中一般性能和下游领域性能的变化，通过验证损失衡量领域性能。我们观察到，CPT损失曲线从根本上描述了一条曲线到另一条隐含曲线的转变过程，并可通过解除分布偏移效应和学习率退火效应的影响来描述。我们推导出一个结合这两种因素的CPT缩放定律，使我们能够预测任何（持续）训练步骤和CPT中不同学习率调度下的损失。我们的公式对CPT中几个关键因素提供了全面的理解，包括损失潜力、峰值学习率、训练步骤、回放比例等。此外，我们的方法可以适应不同的CPT目标定制训练超参数，如平衡一般性能和领域特定性能。大量实验表明，我们的缩放定律在各种CPT数据集和训练超参数下均适用。 

---
# Overflow Prevention Enhances Long-Context Recurrent LLMs 

**Title (ZH)**: 溢出预防增强长时间上下文循环生成模型 

**Authors**: Assaf Ben-Kish, Itamar Zimerman, M. Jehanzeb Mirza, James Glass, Leonid Karlinsky, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07793)  

**Abstract**: A recent trend in LLMs is developing recurrent sub-quadratic models that improve long-context processing efficiency. We investigate leading large long-context models, focusing on how their fixed-size recurrent memory affects their performance. Our experiments reveal that, even when these models are trained for extended contexts, their use of long contexts remains underutilized. Specifically, we demonstrate that a chunk-based inference procedure, which identifies and processes only the most relevant portion of the input can mitigate recurrent memory failures and be effective for many long-context tasks: On LongBench, our method improves the overall performance of Falcon3-Mamba-Inst-7B by 14%, Falcon-Mamba-Inst-7B by 28%, RecurrentGemma-IT-9B by 50%, and RWKV6-Finch-7B by 51%. Surprisingly, this simple approach also leads to state-of-the-art results in the challenging LongBench v2 benchmark, showing competitive performance with equivalent size Transformers. Furthermore, our findings raise questions about whether recurrent models genuinely exploit long-range dependencies, as our single-chunk strategy delivers stronger performance - even in tasks that presumably require cross-context relations. 

**Abstract (ZH)**: 近年来，大规模语言模型的趋势是开发亚二次递归模型以提高长上下文处理效率。我们研究了主要的大型长上下文模型，重点关注它们固定大小的递归记忆对其性能的影响。我们的实验表明，即使这些模型在长上下文中进行了长时间的训练，它们对长上下文的利用仍然不足。具体来说，我们展示了一种基于片段的推理过程，该过程仅识别并处理输入中最相关部分，可以缓解递归记忆故障并有效应用于许多长上下文任务：在LongBench上，我们的方法分别将Falcon3-Mamba-Inst-7B的整体性能提高了14%、Falcon-Mamba-Inst-7B提高了28%、RecurrentGemma-IT-9B提高了50%、RWKV6-Finch-7B提高了51%。令人惊讶的是，这种简单的approach在具有挑战性的LongBench v2基准测试中也达到了最先进的结果，显示出与同等规模的Transformers相当的性能。此外，我们的研究结果引发了一个问题，即递归模型是否真正利用了长范围依赖性，因为我们的单片段策略即使在显然需要跨上下文关系的任务中也表现出了更强的性能。 

---
# Must Read: A Systematic Survey of Computational Persuasion 

**Title (ZH)**: 必读：计算劝导系统的综述 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.07775)  

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models. 

**Abstract (ZH)**: 计算说服是沟通的基本方面，影响从日常生活对话到政治、营销和法律等高风险场景中的决策。随着对话式AI系统的发展，说服的范围显著扩大，带来了机遇和风险。AI驱动的说服可以用于有益的应用，但也有可能通过操纵和不道德影响来构成威胁。此外，AI系统不仅是说服者，也是可以被说服的对象，使其容易遭受对抗性攻击和偏见强化。尽管在AI生成的说服性内容方面取得了快速进展，但由于其固有的主观性和情境依赖性，我们对其为何有效的理解仍然有限。在本综述中，我们从三个关键视角提供了计算说服的全面概述：（1）AI作为说服者，探讨了AI生成的说服性内容及其应用；（2）AI作为说服对象，研究了AI的可利用性和操纵性；（3）AI作为说服裁判者，分析了AI在评估说服策略、检测操纵和确保道德说服方面的作用。我们介绍了计算说服研究的分类体系，并讨论了关键挑战，包括评估说服性、减轻具有欺骗性的说服以及开发负责任的AI驱动说服系统。本综述概述了未来研究方向，旨在增强AI驱动说服的安全性、公平性和有效性，同时应对日益强大的语言模型带来的风险。 

---
# Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding 

**Title (ZH)**: 通过双向代码注释级互信息接地增强代码生成 

**Authors**: Yifeng Di, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07768)  

**Abstract**: Large Language Models (LLMs) have demonstrated unprecedented capability in code generation. However, LLM-generated code is still plagued with a wide range of functional errors, especially for complex programming tasks that LLMs have not seen before. Recent studies have shown that developers often struggle with inspecting and fixing incorrect code generated by LLMs, diminishing their productivity and trust in LLM-based code generation. Inspired by the mutual grounding theory in communication, we propose an interactive approach that leverages code comments as a medium for developers and LLMs to establish a shared understanding. Our approach facilitates iterative grounding by interleaving code generation, inline comment generation, and contextualized user feedback through editable comments to align generated code with developer intent. We evaluated our approach on two popular benchmarks and demonstrated that our approach significantly improved multiple state-of-the-art LLMs, e.g., 17.1% pass@1 improvement for code-davinci-002 on HumanEval. Furthermore, we conducted a user study with 12 participants in comparison to two baselines: (1) interacting with GitHub Copilot, and (2) interacting with a multi-step code generation paradigm called Multi-Turn Program Synthesis. Participants completed the given programming tasks 16.7% faster and with 10.5% improvement in task success rate when using our approach. Both results show that interactively refining code comments enables the collaborative establishment of mutual grounding, leading to more accurate code generation and higher developer confidence. 

**Abstract (ZH)**: 大型语言模型在代码生成方面展现了前所未有的能力，但由于生成的代码仍然存在各种功能错误，特别是在LLMs未见过的复杂编程任务中，这一问题尤为突出。最近的研究表明，开发者往往难以检查和修复LLMs生成的错误代码，这降低了他们的生产力和对LLM驱动的代码生成的信任。受交流中的相互接地理论启发，我们提出了一种交互式方法，利用代码注释作为开发者和LLMs建立共同理解的媒介。该方法通过嵌入代码生成、行内注释生成及上下文化用户反馈，促进迭代接地，使生成的代码与开发者的意图保持一致。我们在两个流行的基准测试上评估了该方法，并证明了该方法显著提高了多种最先进的LLMs的表现，例如，在HumanEval上的code-davinci-002上取得了17.1%的pass@1改进。此外，我们还进行了一项包含12名参与者的研究，与两种基线方法进行了比较：（1）与GitHub Copilot交互，（2）与一种称为多轮程序合成的多步代码生成范式交互。结果显示，使用该方法，参与者完成给定编程任务的速度提高了16.7%，任务成功率提高了10.5%。这两项结果表明，交互式精炼代码注释能够促进协作相互接地，从而实现更准确的代码生成并提高开发者的信心。 

---
# Benchmarking of CPU-intensive Stream Data Processing in The Edge Computing Systems 

**Title (ZH)**: 边缘计算系统中CPU密集型流数据处理的基准测试 

**Authors**: Tomasz Szydlo, Viacheslaw Horbanow, Dev Nandan Jha, Shashikant Ilager, Aleksander Slominski, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07755)  

**Abstract**: Edge computing has emerged as a pivotal technology, offering significant advantages such as low latency, enhanced data security, and reduced reliance on centralized cloud infrastructure. These benefits are crucial for applications requiring real-time data processing or strict security measures. Despite these advantages, edge devices operating within edge clusters are often underutilized. This inefficiency is mainly due to the absence of a holistic performance profiling mechanism which can help dynamically adjust the desired system configuration for a given workload. Since edge computing environments involve a complex interplay between CPU frequency, power consumption, and application performance, a deeper understanding of these correlations is essential. By uncovering these relationships, it becomes possible to make informed decisions that enhance both computational efficiency and energy savings. To address this gap, this paper evaluates the power consumption and performance characteristics of a single processing node within an edge cluster using a synthetic microbenchmark by varying the workload size and CPU frequency. The results show how an optimal measure can lead to optimized usage of edge resources, given both performance and power consumption. 

**Abstract (ZH)**: 边缘计算作为一种关键性技术，提供了低延迟、增强的数据安全性和减少对集中式云基础设施依赖等显著优势。这些优势对于需要实时数据处理或严格安全措施的应用至关重要。尽管具有这些优势，边缘集群中的边缘设备往往利用不足。这种低效主要是由于缺乏一个全面的性能分析机制，该机制能帮助动态调整给定工作负载下的系统配置。由于边缘计算环境涉及CPU频率、功耗和应用性能之间的复杂交互，深入理解这些相关性至关重要。通过揭示这些关系，可以做出既能提升计算效率又能节省能源的明智决策。为了填补这一空白，本文通过改变工作负载大小和CPU频率来使用合成微基准评估边缘集群中单个处理节点的功耗和性能特性。结果表明，在兼顾性能和功率消耗的情况下，适当的优化措施能实现边缘资源的最佳利用。 

---
# Guiding Data Collection via Factored Scaling Curves 

**Title (ZH)**: 基于因子缩放曲线指导数据收集 

**Authors**: Lihan Zha, Apurva Badithela, Michael Zhang, Justin Lidard, Jeremy Bao, Emily Zhou, David Snyder, Allen Z. Ren, Dhruv Shah, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07728)  

**Abstract**: Generalist imitation learning policies trained on large datasets show great promise for solving diverse manipulation tasks. However, to ensure generalization to different conditions, policies need to be trained with data collected across a large set of environmental factor variations (e.g., camera pose, table height, distractors) $-$ a prohibitively expensive undertaking, if done exhaustively. We introduce a principled method for deciding what data to collect and how much to collect for each factor by constructing factored scaling curves (FSC), which quantify how policy performance varies as data scales along individual or paired factors. These curves enable targeted data acquisition for the most influential factor combinations within a given budget. We evaluate the proposed method through extensive simulated and real-world experiments, across both training-from-scratch and fine-tuning settings, and show that it boosts success rates in real-world tasks in new environments by up to 26% over existing data-collection strategies. We further demonstrate how factored scaling curves can effectively guide data collection using an offline metric, without requiring real-world evaluation at scale. 

**Abstract (ZH)**: 通用模仿学习政策在大规模数据集上训练后，在解决多样操作任务方面展现出巨大潜力。然而，为了确保在不同条件下的泛化能力，政策需要在环境因素（如相机姿态、桌子高度、干扰物）变化的广泛集合中收集数据——这是一项代价高昂的工程，如果要穷尽所有可能性的话。我们提出了一种原理性的方法，用于决定收集哪些数据以及每种因素收集多少数据，通过构建因子尺度曲线（FSC），量化随单一因素或配对因素数据规模变化时政策性能的变化。这些曲线使人们能够在预算范围内针对最有影响力的因素组合进行有目标的数据采集。通过广泛的模拟和实际实验，我们评估了所提出的方法，并在从零开始训练和微调的不同设置下展示了它在新环境中提高了高达26%的成功率。我们进一步证明了如何使用离线指标引导数据收集，而无需大规模的真实世界评估。 

---
# Hybrid Spiking Vision Transformer for Object Detection with Event Cameras 

**Title (ZH)**: 基于事件相机的混合脉冲视觉变换器目标检测方法 

**Authors**: Qi Xu, Jie Deng, Jiangrong Shen, Biwu Chen, Huajin Tang, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07715)  

**Abstract**: Event-based object detection has gained increasing attention due to its advantages such as high temporal resolution, wide dynamic range, and asynchronous address-event representation. Leveraging these advantages, Spiking Neural Networks (SNNs) have emerged as a promising approach, offering low energy consumption and rich spatiotemporal dynamics. To further enhance the performance of event-based object detection, this study proposes a novel hybrid spike vision Transformer (HsVT) model. The HsVT model integrates a spatial feature extraction module to capture local and global features, and a temporal feature extraction module to model time dependencies and long-term patterns in event sequences. This combination enables HsVT to capture spatiotemporal features, improving its capability to handle complex event-based object detection tasks. To support research in this area, we developed and publicly released The Fall Detection Dataset as a benchmark for event-based object detection tasks. This dataset, captured using an event-based camera, ensures facial privacy protection and reduces memory usage due to the event representation format. We evaluated the HsVT model on GEN1 and Fall Detection datasets across various model sizes. Experimental results demonstrate that HsVT achieves significant performance improvements in event detection with fewer parameters. 

**Abstract (ZH)**: 基于事件的对象检测因其高时间分辨率、宽动态范围和异步地址事件表示等优势引起了越来越多的关注。利用这些优势，契神经网络（SNNs） emerged as a promising approach，提供低能耗和丰富的时空动态。为了进一步提高基于事件的对象检测性能，本研究提出了一种新的混合契视觉变换器（HsVT）模型。HsVT模型结合了空间特征提取模块以捕获局部和全局特征，以及时间特征提取模块以建模事件序列中的时间依赖性和长期模式。这种结合使HsVT能够捕获时空特征，从而提高其处理复杂基于事件的对象检测任务的能力。为了支持该领域的研究，我们开发并公开发布了跌倒检测数据集作为基于事件的对象检测任务的基准。该数据集由基于事件的相机捕获，确保面部隐私保护并因事件表示格式减少内存使用。我们在不同模型大小的GEN1和跌倒检测数据集上评估了HsVT模型。实验结果表明，HsVT以较少的参数实现了显著的性能提升。 

---
# Circuit Partitioning Using Large Language Models for Quantum Compilation and Simulations 

**Title (ZH)**: 使用大规模语言模型进行量子编译和模拟的电路分区 

**Authors**: Pranav Sinha, Sumit Kumar Jha, Sunny Raj  

**Link**: [PDF](https://arxiv.org/pdf/2505.07711)  

**Abstract**: We are in the midst of the noisy intermediate-scale quantum (NISQ) era, where quantum computers are limited by noisy gates, some of which are more error-prone than others and can render the final computation incomprehensible. Quantum circuit compilation algorithms attempt to minimize these noisy gates when mapping quantum algorithms onto quantum hardware but face computational challenges that restrict their application to circuits with no more than 5-6 qubits, necessitating the need to partition large circuits before the application of noisy quantum gate minimization algorithms. The existing generation of these algorithms is heuristic in nature and does not account for downstream gate minimization tasks. Large language models (LLMs) have the potential to change this and help improve quantum circuit partitions. This paper investigates the use of LLMs, such as Llama and Mistral, for partitioning quantum circuits by capitalizing on their abilities to understand and generate code, including QASM. Specifically, we teach LLMs to partition circuits using the quick partition approach of the Berkeley Quantum Synthesis Toolkit. Through experimental evaluations, we show that careful fine-tuning of open source LLMs enables us to obtain an accuracy of 53.4% for the partition task while over-the-shelf LLMs are unable to correctly partition circuits, using standard 1-shot and few-shot training approaches. 

**Abstract (ZH)**: 我们正处于嘈杂的中尺度量子（NISQ）时代，其中量子计算机受限于嘈杂的门操作，有些门操作比其他门操作更容易出错，可能导致最终计算结果难以理解。量子电路编译算法试图在将量子算法映射到量子硬件时最小化这些嘈杂的门操作，但由于计算挑战的限制，这些算法的应用仅限于不超过5-6个量子位的电路，从而需要在应用嘈杂的量子门操作最小化算法之前将大电路进行分区。现有的这些算法具有启发式性质，并未考虑到后续的门操作最小化任务。大规模语言模型（LLMs）有可能改变这一现状并帮助改进量子电路的分区。本文研究了利用Llama和Mistral等大规模语言模型进行量子电路分区的可能性，利用它们理解和生成代码的能力，包括QASM。具体地，我们教导LLMs使用伯克利量子合成工具包中的快速分区方法来分区电路。通过实验评估，我们表明对开源LLMs进行仔细的微调使我们能够在分区任务中获得53.4%的准确率，而标准的单 Shot 和少 Shot 训练方法无法使即用的大规模语言模型正确地分区电路。 

---
# Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications 

**Title (ZH)**: 面向低资源嵌入式应用的轻量级端到端文本到语音合成 

**Authors**: Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba  

**Link**: [PDF](https://arxiv.org/pdf/2505.07701)  

**Abstract**: Recent works have shown that modelling raw waveform directly from text in an end-to-end (E2E) fashion produces more natural-sounding speech than traditional neural text-to-speech (TTS) systems based on a cascade or two-stage approach. However, current E2E state-of-the-art models are computationally complex and memory-consuming, making them unsuitable for real-time offline on-device applications in low-resource scenarios. To address this issue, we propose a Lightweight E2E-TTS (LE2E) model that generates high-quality speech requiring minimal computational resources. We evaluate the proposed model on the LJSpeech dataset and show that it achieves state-of-the-art performance while being up to $90\%$ smaller in terms of model parameters and $10\times$ faster in real-time-factor. Furthermore, we demonstrate that the proposed E2E training paradigm achieves better quality compared to an equivalent architecture trained in a two-stage approach. Our results suggest that LE2E is a promising approach for developing real-time, high quality, low-resource TTS applications for on-device applications. 

**Abstract (ZH)**: 最近的研究表明，以端到端（E2E）方式直接从文本建模原始波形比基于级联或两阶段方法的传统神经文本到语音（TTS）系统生成更具自然感的语音。然而，当前的E2E最先进的模型在计算上复杂且占用大量内存，不适合资源有限场景下的离线设备应用。为解决这一问题，我们提出了一种轻量级E2E-TTS（LE2E）模型，该模型能使用最少的计算资源生成高质量的语音。我们在LJSpeech数据集上评估了提出的模型，并展示了它在模型参数量减少高达90%且实时因子加快10倍的情况下获得了最先进的性能。此外，我们证明了提出的E2E训练范式在与两阶段方法等效架构相比时能获得更好的质量。我们的结果表明，LE2E是一种有潜力的方法，适用于开发资源有限场景下的实时高质量设备端TTS应用。 

---
# Multimodal Survival Modeling in the Age of Foundation Models 

**Title (ZH)**: 基础模型时代多模态生存模型研究 

**Authors**: Steven Song, Morgan Borjigin-Wang, Irene Madejski, Robert L. Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2505.07683)  

**Abstract**: The Cancer Genome Atlas (TCGA) has enabled novel discoveries and served as a large-scale reference through its harmonized genomics, clinical, and image data. Prior studies have trained bespoke cancer survival prediction models from unimodal or multimodal TCGA data. A modern paradigm in biomedical deep learning is the development of foundation models (FMs) to derive meaningful feature embeddings, agnostic to a specific modeling task. Biomedical text especially has seen growing development of FMs. While TCGA contains free-text data as pathology reports, these have been historically underutilized. Here, we investigate the feasibility of training classical, multimodal survival models over zero-shot embeddings extracted by FMs. We show the ease and additive effect of multimodal fusion, outperforming unimodal models. We demonstrate the benefit of including pathology report text and rigorously evaluate the effect of model-based text summarization and hallucination. Overall, we modernize survival modeling by leveraging FMs and information extraction from pathology reports. 

**Abstract (ZH)**: The Cancer Genome Atlas (TCGA)通过协调的基因组、临床和图像数据，实现了新型发现并成为大规模参考。先前的研究从单模态或跨模态TCGA数据中训练定制的癌症生存预测模型。生物医学深度学习中的一种现代范式是开发基础模型(FMs)以提取无特定模型任务之别的有意义特征嵌入。生物医学文本尤其见证了FMs的发展。尽管TCGA包含病理报告等自由文本数据，但这些数据历来未被充分利用。我们研究了使用FMs提取零样本嵌入训练传统跨模态生存模型的可能性。我们展示了跨模态融合的便捷性和附加效应，优于单模态模型。我们展示了包含病理报告文本的益处，并严格评估基于模型的文本摘要和幻觉效果。总体而言，我们通过利用FMs和病理报告信息提取来现代化生存模型。 

---
# Simple Semi-supervised Knowledge Distillation from Vision-Language Models via $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization 

**Title (ZH)**: 通过 Dual-Head 优化从视觉-语言模型进行简单的半监督知识蒸馏 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07675)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable success across diverse tasks by leveraging rich textual information with minimal labeled data. However, deploying such large models remains challenging, particularly in resource-constrained environments. Knowledge distillation (KD) offers a well-established solution to this problem; however, recent KD approaches from VLMs often involve multi-stage training or additional tuning, increasing computational overhead and optimization complexity. In this paper, we propose $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization ($\mathbf{\texttt{DHO}}$) -- a simple yet effective KD framework that transfers knowledge from VLMs to compact, task-specific models in semi-supervised settings. Specifically, we introduce dual prediction heads that independently learn from labeled data and teacher predictions, and propose to linearly combine their outputs during inference. We observe that $\texttt{DHO}$ mitigates gradient conflicts between supervised and distillation signals, enabling more effective feature learning than single-head KD baselines. As a result, extensive experiments show that $\texttt{DHO}$ consistently outperforms baselines across multiple domains and fine-grained datasets. Notably, on ImageNet, it achieves state-of-the-art performance, improving accuracy by 3% and 0.1% with 1% and 10% labeled data, respectively, while using fewer parameters. 

**Abstract (ZH)**: 双头优化：Vision-langauge模型的知识蒸馏框架 

---
# OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit 

**Title (ZH)**: OnPrem.LLM：一种注重隐私的文档智能工具包 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2505.07672)  

**Abstract**: We present this http URL, a Python-based toolkit for applying large language models (LLMs) to sensitive, non-public data in offline or restricted environments. The system is designed for privacy-preserving use cases and provides prebuilt pipelines for document processing and storage, retrieval-augmented generation (RAG), information extraction, summarization, classification, and prompt/output processing with minimal configuration. this http URL supports multiple LLM backends -- including this http URL, Ollama, vLLM, and Hugging Face Transformers -- with quantized model support, GPU acceleration, and seamless backend switching. Although designed for fully local execution, this http URL also supports integration with a wide range of cloud LLM providers when permitted, enabling hybrid deployments that balance performance with data control. A no-code web interface extends accessibility to non-technical users. 

**Abstract (ZH)**: 我们提供了一个基于Python的工具包this http URL，用于在离线或受限环境中应用大语言模型（LLMs）处理敏感的非公开数据。该系统设计用于保护隐私的应用场景，并提供了预构建的文档处理和存储、检索增强生成（RAG）、信息提取、总结、分类以及提示/输出处理管道，配置简单。this http URL支持多种LLM后端，包括this http URL、Ollama、vLLM和Hugging Face Transformers，支持量化模型、GPU加速，并且可以无缝切换后端。尽管设计用于本地执行，但当允许时，this http URL也可以与广泛的云LLM提供商集成，从而实现性能与数据控制之间的平衡。无代码Web界面使非技术用户也能访问。 

---
# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Title (ZH)**: 化学领域的检索增强生成基准研究 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为一种增强大规模语言模型（LLMs）的有力框架，特别是在需要专门和动态信息的科学领域。尽管前景广阔，但在化学领域的应用仍然未被充分探索，主要原因在于缺乏高质量的领域特定语料库和完善的评估基准。在本文中，我们引入了ChemRAG-Bench，这是一个全面的基准，旨在系统评估RAG在一系列化学相关任务中的有效性。伴随的化学语料库整合了异构知识源，包括科学文献、PubChem数据库、PubMed摘要、教科书和维基百科条目。此外，我们介绍了ChemRAG-Toolkit，这是一个模块化和可扩展的RAG工具包，支持五种检索算法和八种LLM。利用ChemRAG-Toolkit，我们展示了RAG取得显著性能提升——相对于直接推理方法，平均相对改进率为17.4%。我们还进行深入分析，探讨了检索器架构、语料库选择和检索段落数量等方面，最终提出指导未来RAG系统在化学领域研究和部署的实际建议。代码和数据可在以下链接获取。 

---
# A Case Study Investigating the Role of Generative AI in Quality Evaluations of Epics in Agile Software Development 

**Title (ZH)**: 一项探究生成式AI在敏捷软件开发中史诗质量评估作用的案例研究 

**Authors**: Werner Geyer, Jessica He, Daita Sarkar, Michelle Brachman, Chris Hammond, Jennifer Heins, Zahra Ashktorab, Carlos Rosemberg, Charlie Hill  

**Link**: [PDF](https://arxiv.org/pdf/2505.07664)  

**Abstract**: The broad availability of generative AI offers new opportunities to support various work domains, including agile software development. Agile epics are a key artifact for product managers to communicate requirements to stakeholders. However, in practice, they are often poorly defined, leading to churn, delivery delays, and cost overruns. In this industry case study, we investigate opportunities for large language models (LLMs) to evaluate agile epic quality in a global company. Results from a user study with 17 product managers indicate how LLM evaluations could be integrated into their work practices, including perceived values and usage in improving their epics. High levels of satisfaction indicate that agile epics are a new, viable application of AI evaluations. However, our findings also outline challenges, limitations, and adoption barriers that can inform both practitioners and researchers on the integration of such evaluations into future agile work practices. 

**Abstract (ZH)**: 生成式人工智能的广泛availability为各种工作领域提供了新的机会，包括敏捷软件开发。在实践中，敏捷史诗往往定义不足，导致返工、交付延迟和成本超支。在这一行业案例研究中，我们调查了大型语言模型（LLMs）在一家全球公司中评估敏捷史诗质量的机会。用户研究（17名产品经理参与）的结果表明，LLM评估如何能够整合到他们的工作实践中，包括在提高其史诗方面感知到的价值和使用情况。高度的满意度表明，敏捷史诗是AI评估的一项新且可行的应用。然而，我们的研究成果也指出了挑战、限制和采纳障碍，这些信息可以为从业者和研究人员提供关于将此类评估整合到未来敏捷工作实践中的指导。 

---
# Chronocept: Instilling a Sense of Time in Machines 

**Title (ZH)**: Chronocept: 在机器中植入时间感知 

**Authors**: Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, Vishesh Khadaria  

**Link**: [PDF](https://arxiv.org/pdf/2505.07637)  

**Abstract**: Human cognition is deeply intertwined with a sense of time, known as Chronoception. This sense allows us to judge how long facts remain valid and when knowledge becomes outdated. Despite progress in vision, language, and motor control, AI still struggles to reason about temporal validity. We introduce Chronocept, the first benchmark to model temporal validity as a continuous probability distribution over time. Using skew-normal curves fitted along semantically decomposed temporal axes, Chronocept captures nuanced patterns of emergence, decay, and peak relevance. It includes two datasets: Benchmark I (atomic facts) and Benchmark II (multi-sentence passages). Annotations show strong inter-annotator agreement (84% and 89%). Our baselines predict curve parameters - location, scale, and skewness - enabling interpretable, generalizable learning and outperforming classification-based approaches. Chronocept fills a foundational gap in AI's temporal reasoning, supporting applications in knowledge grounding, fact-checking, retrieval-augmented generation (RAG), and proactive agents. Code and data are publicly available. 

**Abstract (ZH)**: 人类的认知与时间感知（Chronoception）密切相关。这种感知使我们能够判断事实的有效时间长度以及知识何时过时。尽管在视觉、语言和运动控制方面取得了进展，AI在推理时间有效性方面仍然面临挑战。我们引入了Chronocept，这是首个将时间有效性建模为时间上的连续概率分布的基准。利用沿语义分解的时间轴拟合的偏斜正态曲线，Chronocept捕捉了出现、衰减和峰值相关性的细微模式。它包括两个数据集：基准I（原子事实）和基准II（多句段落）。标注结果显示较强的一致性（84%和89%）。我们的基线模型预测曲线参数——位置、尺度和偏斜度——实现了可解释、可泛化的学习，并超越了基于分类的方法。Chronocept填补了AI在时间推理方面的基础空白，支持知识接地、事实核查、检索增强生成（RAG）和主动代理等应用。代码和数据已公開。 

---
# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents 

**Title (ZH)**: 神经脑：一种受神经科学启发的体态智能框架 

**Authors**: Jian Liu, Xiongtao Shi, Thai Duy Nguyen, Haitian Zhang, Tianxiang Zhang, Wei Sun, Yanjie Li, Athanasios V. Vasilakos, Giovanni Iacca, Arshad Ali Khan, Arvind Kumar, Jae Won Cho, Ajmal Mian, Lihua Xie, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07634)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios. 

**Abstract (ZH)**: 人工智能的快速进化已从静态、数据驱动的模型转向能够感知和互动的动态系统。尽管在模式识别和符号推理方面取得了进展，当前的人工智能系统，如大型语言模型，依然缺乏实体性，无法实际与世界互动。这一限制推动了具身人工智能的发展，其中自主代理，如类人机器人，必须具备人类般的适应性来导航和操控非结构化的环境。这一挑战的核心在于神经脑的概念，这是一种设计用于驱动具有人类适应性的具身代理的中枢智能系统。神经脑必须无缝地整合多模态感知与认知能力。实现这一点还需要一个适应性记忆系统和高效的硬件-软件协同设计，以实现动态环境中的实时行动。本文介绍了具身代理的统一神经脑框架，解决了两个基本挑战：（1）定义神经脑的核心组件；（2）弥合静态人工智能模型与现实世界部署所需动态适应性之间的差距。为此，我们提出了一种受生物学启发的架构，该架构整合了多模态主动感知、感知-认知-行动功能、基于神经可塑性的记忆存储与更新，以及神经形态硬件/软件优化。此外，我们还回顾了在这些四个方面的最新研究成果，并分析了当前人工智能系统与人类智能之间的差距。通过综合神经科学的见解，我们勾勒出了一条发展通用、自主代理以在现实场景中实现人类级智能的道路。 

---
# Bang for the Buck: Vector Search on Cloud CPUs 

**Title (ZH)**: 花钱之道：云CPU上的向量搜索 

**Authors**: Leonardo Kuffo, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07621)  

**Abstract**: Vector databases have emerged as a new type of systems that support efficient querying of high-dimensional vectors. Many of these offer their database as a service in the cloud. However, the variety of available CPUs and the lack of vector search benchmarks across CPUs make it difficult for users to choose one. In this study, we show that CPU microarchitectures available in the cloud perform significantly differently across vector search scenarios. For instance, in an IVF index on float32 vectors, AMD's Zen4 gives almost 3x more queries per second (QPS) compared to Intel's Sapphire Rapids, but for HNSW indexes, the tables turn. However, when looking at the number of queries per dollar (QP$), Graviton3 is the best option for most indexes and quantization settings, even over Graviton4 (Table 1). With this work, we hope to guide users in getting the best "bang for the buck" when deploying vector search systems. 

**Abstract (ZH)**: 云CPU微架构在向量搜索场景中的性能差异研究：指导用户获得最佳性价比部署向量搜索系统 

---
# Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models 

**Title (ZH)**: 扩散责任：分析生成性文本到音频扩散模型的能耗 

**Authors**: Riccardo Passoni, Francesca Ronchini, Luca Comanducci, Romain Serizel, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07615)  

**Abstract**: Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct an analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models. 

**Abstract (ZH)**: 基于文本到音频模型的能效分析：生成参数对推理时能耗的影响及帕累托最优解的研究 

---
# Concept-Level Explainability for Auditing & Steering LLM Responses 

**Title (ZH)**: 概念级解释性审计与引导LLM响应 

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2505.07610)  

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的广泛应用，对其安全性和一致性方面的担忧日益增加。一种引导LLM行为的方法，如缓解偏见或防御逃逸攻击，是确定提示中哪些部分影响模型输出的具体方面。基于令牌级别的归因方法提供了一种有前途的解决方案，但仍然在文本生成中挣扎，难以单独解释输出中每个令牌的存在，而不是整个LLM响应的底层语义。我们引入了ConceptX，这是一种模型无关的概念级解释方法，它识别出概念，即提示中的语义丰富的令牌，并基于输出的语义相似度赋予它们重要性。与现有的基于令牌的方法不同，ConceptX 还可以通过就地令牌替换来保持上下文完整性，并支持灵活的解释目标，例如性别偏见。ConceptX 既能用于审计，通过揭示偏见来源，也能用于引导，通过修改提示来改变情感或减少LLM响应的危害性，而无需重新训练。在三种LLM中，ConceptX 在忠实度和人类一致性方面均优于基于令牌的方法（如TokenSHAP）。在转向任务中，概念X使情感转变提高了0.252，而随机编辑为0.131，并将攻击成功率从0.463降低到0.242，超越了归因和改写基线。虽然提示工程和自我解释方法有时会带来更安全的响应，但ConceptX 提供了一种透明且忠实的替代方案，用于提高LLM的安全性和一致性，展示了基于归因的解释在引导LLM行为方面的实际价值。 

---
# MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining 

**Title (ZH)**: MiMo：解锁语言模型的推理潜力——从预训练到后训练 

**Authors**: Xiaomi LLM-Core Team, Bingquan Xia, Bowen Shen, Cici, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, Liang Zhao, Peidian Li, Peng Wang, Shihua Yu, Shimao Chen, Weikun Wang, Wenhan Ma, Xiangwei Deng, Yi Huang, Yifan Song, Zihan Jiang, Bowen Ye, Can Cai, Chenhong He, Dong Zhang, Duo Zhang, Guoan Wang, Hao Tian, Haochen Zhao, Heng Qu, Hongshen Xu, Jun Shi, Kainan Bao, QingKai Fang, Kang Zhou, Kangyang Zhou, Lei Li, Menghang Zhu, Nuo Chen, Qiantong Wang, Shaohui Liu, Shicheng Li, Shuhao Gu, Shuhuai Ren, Shuo Liu, Sirui Deng, Weiji Zhuang, Weiwei Lv, Wenyu Yang, Xin Zhang, Xing Yong, Xing Zhang, Xingchen Song, Xinzhe Xu, Xu Wang, Yihan Yan, Yu Tu, Yuanyuan Tian, Yudong Wang, Yue Yu, Zhenru Lin, Zhichao Song, Zihao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07608)  

**Abstract**: We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at this https URL. 

**Abstract (ZH)**: MiMo-7B：一种生于推理任务的大型语言模型，经过前后训练阶段的优化 

---
# Characterizing the Investigative Methods of Fictional Detectives with Large Language Models 

**Title (ZH)**: 使用大型语言模型 characterization 调查方法中的虚构侦探 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Bruno Feijó, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2505.07601)  

**Abstract**: Detective fiction, a genre defined by its complex narrative structures and character-driven storytelling, presents unique challenges for computational narratology, a research field focused on integrating literary theory into automated narrative generation. While traditional literary studies have offered deep insights into the methods and archetypes of fictional detectives, these analyses often focus on a limited number of characters and lack the scalability needed for the extraction of unique traits that can be used to guide narrative generation methods. In this paper, we present an AI-driven approach for systematically characterizing the investigative methods of fictional detectives. Our multi-phase workflow explores the capabilities of 15 Large Language Models (LLMs) to extract, synthesize, and validate distinctive investigative traits of fictional detectives. This approach was tested on a diverse set of seven iconic detectives - Hercule Poirot, Sherlock Holmes, William Murdoch, Columbo, Father Brown, Miss Marple, and Auguste Dupin - capturing the distinctive investigative styles that define each character. The identified traits were validated against existing literary analyses and further tested in a reverse identification phase, achieving an overall accuracy of 91.43%, demonstrating the method's effectiveness in capturing the distinctive investigative approaches of each detective. This work contributes to the broader field of computational narratology by providing a scalable framework for character analysis, with potential applications in AI-driven interactive storytelling and automated narrative generation. 

**Abstract (ZH)**: 侦探小说是一种以复杂叙事结构和人物驱动 storytelling 为特征的文学体裁，给专注于将文学理论整合到自动化叙事生成中的计算叙事学带来独特挑战。传统文学研究虽然提供了对虚构侦探的方法和原型的深刻见解，但这些分析往往局限于少数几个人物，并缺乏提取可用于指导叙事生成方法的独特特征所需的可扩展性。本文介绍了一种基于 AI 的系统化方法，用于刻画虚构侦探的调查方法。我们采用多阶段工作流程探索 15 种大型语言模型在提取、合成和验证虚构侦探独特调查特征方面的能力。这种方法在赫尔克里·波洛、夏洛克·福尔摩斯、威廉·墨多克、科 Styles 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Title (ZH)**: 强化内外知识协同推理以实现高效自适应搜索代理 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

**Abstract (ZH)**: 基于检索增强生成的强化内部-外部知识协同推理代理（IKEA） 

---
# A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models 

**Title (ZH)**: 大型语言模型中指令跟随评价与改进的多维度约束框架 

**Authors**: Junjie Ye, Caishuang Huang, Zhuohan Chen, Wenjie Fu, Chenyuan Yang, Leyi Yang, Yilong Wu, Peng Wang, Meng Zhou, Xiaolong Yang, Tao Gui, Qi Zhang, Zhongchao Shi, Jianping Fan, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07591)  

**Abstract**: Instruction following evaluates large language models (LLMs) on their ability to generate outputs that adhere to user-defined constraints. However, existing benchmarks often rely on templated constraint prompts, which lack the diversity of real-world usage and limit fine-grained performance assessment. To fill this gap, we propose a multi-dimensional constraint framework encompassing three constraint patterns, four constraint categories, and four difficulty levels. Building on this framework, we develop an automated instruction generation pipeline that performs constraint expansion, conflict detection, and instruction rewriting, yielding 1,200 code-verifiable instruction-following test samples. We evaluate 19 LLMs across seven model families and uncover substantial variation in performance across constraint forms. For instance, average performance drops from 77.67% at Level I to 32.96% at Level IV. Furthermore, we demonstrate the utility of our approach by using it to generate data for reinforcement learning, achieving substantial gains in instruction following without degrading general performance. In-depth analysis indicates that these gains stem primarily from modifications in the model's attention modules parameters, which enhance constraint recognition and adherence. Code and data are available in this https URL. 

**Abstract (ZH)**: 大规模语言模型指令遵循的多维度约束框架及应用研究 

---
# Evaluating Modern Visual Anomaly Detection Approaches in Semiconductor Manufacturing: A Comparative Study 

**Title (ZH)**: 现代视觉异常检测方法在半导体制造中的评估：一项比较研究 

**Authors**: Manuel Barusco, Francesco Borsatti, Youssef Ben Khalifa, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07576)  

**Abstract**: Semiconductor manufacturing is a complex, multistage process. Automated visual inspection of Scanning Electron Microscope (SEM) images is indispensable for minimizing equipment downtime and containing costs. Most previous research considers supervised approaches, assuming a sufficient number of anomalously labeled samples. On the contrary, Visual Anomaly Detection (VAD), an emerging research domain, focuses on unsupervised learning, avoiding the costly defect collection phase while providing explanations of the predictions. We introduce a benchmark for VAD in the semiconductor domain by leveraging the MIIC dataset. Our results demonstrate the efficacy of modern VAD approaches in this field. 

**Abstract (ZH)**: 半导体制造是一个复杂的多阶段过程。扫描电子显微镜（SEM）图像的自动化视觉检测对于减少设备停机时间和控制成本至关重要。大多数先前的研究考虑了监督方法，假设有足够的异常标记样本。相反，视觉异常检测（VAD）这一新兴研究领域集中于无监督学习，避免了昂贵的缺陷收集阶段，同时提供预测解释。我们通过利用MIIC数据集，引入了半导体领域的VAD基准。我们的结果证明了现代VAD方法在这一领域的有效性。 

---
# Robust Kidney Abnormality Segmentation: A Validation Study of an AI-Based Framework 

**Title (ZH)**: 基于AI的框架的肾异常分割鲁棒性验证研究 

**Authors**: Sarah de Boer, Hartmut Häntze, Kiran Vaidhya Venkadesh, Myrthe A. D. Buser, Gabriel E. Humpire Mamani, Lina Xu, Lisa C. Adams, Jawed Nawabi, Keno K. Bressem, Bram van Ginneken, Mathias Prokop, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2505.07573)  

**Abstract**: Kidney abnormality segmentation has important potential to enhance the clinical workflow, especially in settings requiring quantitative assessments. Kidney volume could serve as an important biomarker for renal diseases, with changes in volume correlating directly with kidney function. Currently, clinical practice often relies on subjective visual assessment for evaluating kidney size and abnormalities, including tumors and cysts, which are typically staged based on diameter, volume, and anatomical location. To support a more objective and reproducible approach, this research aims to develop a robust, thoroughly validated kidney abnormality segmentation algorithm, made publicly available for clinical and research use. We employ publicly available training datasets and leverage the state-of-the-art medical image segmentation framework nnU-Net. Validation is conducted using both proprietary and public test datasets, with segmentation performance quantified by Dice coefficient and the 95th percentile Hausdorff distance. Furthermore, we analyze robustness across subgroups based on patient sex, age, CT contrast phases, and tumor histologic subtypes. Our findings demonstrate that our segmentation algorithm, trained exclusively on publicly available data, generalizes effectively to external test sets and outperforms existing state-of-the-art models across all tested datasets. Subgroup analyses reveal consistent high performance, indicating strong robustness and reliability. The developed algorithm and associated code are publicly accessible at this https URL. 

**Abstract (ZH)**: 肾脏异常分割具有增强临床工作流程的重要潜力，特别是在需要定量评估的环境中。肾脏体积可以作为肾疾病的重要生物标志物，体积的变化与肾功能成直接相关。目前，临床实践中常常依赖主观视觉评估来评价肾脏大小和异常情况，包括肿瘤和囊肿，通常根据直径、体积和解剖位置进行分期。为了支持更加客观和可重复的方法，本研究旨在开发一个稳健且完全验证的肾脏异常分割算法，并向临床和研究公众开放。我们使用公开可用的训练数据集，并利用最先进的医疗图像分割框架nnU-Net进行分割。验证使用了私人和公开的测试数据集，并通过Dice系数和95百分位Hausdorff距离量化分割性能。此外，我们基于患者性别、年龄、CT对比期以及肿瘤组织学亚型分析了分割算法的稳健性。研究结果表明，该分割算法仅在公开数据上训练后，能够有效泛化到外部测试集，并在所有测试数据集中优于现有的最先进的模型。子组分析显示出一致的高性能，表明其具有很强的稳健性和可靠性。所开发的算法及其相关代码在以下网址公开：这个 https URL。 

---
# Towards Requirements Engineering for RAG Systems 

**Title (ZH)**: 面向RAG系统的 Requirements Engineering 研究 

**Authors**: Tor Sporsem, Rasmus Ulfsnes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07553)  

**Abstract**: This short paper explores how a maritime company develops and integrates large-language models (LLM). Specifically by looking at the requirements engineering for Retrieval Augmented Generation (RAG) systems in expert settings. Through a case study at a maritime service provider, we demonstrate how data scientists face a fundamental tension between user expectations of AI perfection and the correctness of the generated outputs. Our findings reveal that data scientists must identify context-specific "retrieval requirements" through iterative experimentation together with users because they are the ones who can determine correctness. We present an empirical process model describing how data scientists practically elicited these "retrieval requirements" and managed system limitations. This work advances software engineering knowledge by providing insights into the specialized requirements engineering processes for implementing RAG systems in complex domain-specific applications. 

**Abstract (ZH)**: 这篇短论文探讨了海洋运输公司如何开发和整合大型语言模型（LLM）。通过具体分析专家环境中检索增强生成（RAG）系统的开发工程需求，论文展示了数据科学家在用户对人工智能完美性的期望与生成输出的准确性之间的基本张力。研究发现，数据科学家必须通过与用户的迭代实验来识别具体的“检索要求”，因为他们才能确定准确性。论文呈现了描述数据科学家如何实际获取这些“检索要求”以及管理系统限制的经验过程模型。这项工作通过提供关于在复杂领域特定应用中实现RAG系统的专门开发工程流程的洞察，推进了软件工程知识。 

---
# Automated Visual Attention Detection using Mobile Eye Tracking in Behavioral Classroom Studies 

**Title (ZH)**: 基于移动眼动追踪的行为课堂研究中自动视觉注意检测 

**Authors**: Efe Bozkir, Christian Kosel, Tina Seidel, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07552)  

**Abstract**: Teachers' visual attention and its distribution across the students in classrooms can constitute important implications for student engagement, achievement, and professional teacher training. Despite that, inferring the information about where and which student teachers focus on is not trivial. Mobile eye tracking can provide vital help to solve this issue; however, the use of mobile eye tracking alone requires a significant amount of manual annotations. To address this limitation, we present an automated processing pipeline concept that requires minimal manually annotated data to recognize which student the teachers focus on. To this end, we utilize state-of-the-art face detection models and face recognition feature embeddings to train face recognition models with transfer learning in the classroom context and combine these models with the teachers' gaze from mobile eye trackers. We evaluated our approach with data collected from four different classrooms, and our results show that while it is possible to estimate the visually focused students with reasonable performance in all of our classroom setups, U-shaped and small classrooms led to the best results with accuracies of approximately 0.7 and 0.9, respectively. While we did not evaluate our method for teacher-student interactions and focused on the validity of the technical approach, as our methodology does not require a vast amount of manually annotated data and offers a non-intrusive way of handling teachers' visual attention, it could help improve instructional strategies, enhance classroom management, and provide feedback for professional teacher development. 

**Abstract (ZH)**: 教室中教师的视觉注意及其在学生之间的分布对学生活动参与、学业成就及专业教师培训具有重要意义。然而，推断教师关注的学生位置和对象并不容易。移动眼动追踪可以提供重要帮助，但单独使用移动眼动追踪需要大量手动注释。为解决这一局限，我们提出了一种自动化处理管道概念，以最小的手动标注数据来识别教师关注的学生。为此，我们利用最新的面部检测模型和面部识别特征嵌入，在教室背景下进行迁移学习训练面部识别模型，并将这些模型与移动眼动追踪的教师凝视相结合。我们使用来自四间不同教室的数据评估了我们的方法，并结果显示，在所有教室设置中，均可以以合理性能估计视觉关注的学生，U形和小型教室的结果分别为约0.7和0.9。尽管我们未评估教师与学生之间的互动，并专注于技术方法的有效性，但鉴于我们的方法不需要大量手动标注数据且能够非侵入性地处理教师的视觉注意，它可以帮助改善教学策略、增强课堂管理，并为专业教师发展提供反馈。 

---
# Noise Optimized Conditional Diffusion for Domain Adaptation 

**Title (ZH)**: 噪声优化条件扩散的领域适应 

**Authors**: Lingkun Luo, Shiqiang Hu, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07548)  

**Abstract**: Pseudo-labeling is a cornerstone of Unsupervised Domain Adaptation (UDA), yet the scarcity of High-Confidence Pseudo-Labeled Target Domain Samples (\textbf{hcpl-tds}) often leads to inaccurate cross-domain statistical alignment, causing DA failures. To address this challenge, we propose \textbf{N}oise \textbf{O}ptimized \textbf{C}onditional \textbf{D}iffusion for \textbf{D}omain \textbf{A}daptation (\textbf{NOCDDA}), which seamlessly integrates the generative capabilities of conditional diffusion models with the decision-making requirements of DA to achieve task-coupled optimization for efficient adaptation. For robust cross-domain consistency, we modify the DA classifier to align with the conditional diffusion classifier within a unified optimization framework, enabling forward training on noise-varying cross-domain samples. Furthermore, we argue that the conventional \( \mathcal{N}(\mathbf{0}, \mathbf{I}) \) initialization in diffusion models often generates class-confused hcpl-tds, compromising discriminative DA. To resolve this, we introduce a class-aware noise optimization strategy that refines sampling regions for reverse class-specific hcpl-tds generation, effectively enhancing cross-domain alignment. Extensive experiments across 5 benchmark datasets and 29 DA tasks demonstrate significant performance gains of \textbf{NOCDDA} over 31 state-of-the-art methods, validating its robustness and effectiveness. 

**Abstract (ZH)**: 噪声优化条件扩散ための領域適応（NOCDDA） 

---
# GRADA: Graph-based Reranker against Adversarial Documents Attack 

**Title (ZH)**: 基于图的对抗文档攻击重排序器（GRADA） 

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07546)  

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy. 

**Abstract (ZH)**: 基于图的对抗文档攻击重排序框架（GRADA）提高大型语言模型的检索质量并显著降低攻击成功率 

---
# The Human-Data-Model Interaction Canvas for Visual Analytics 

**Title (ZH)**: 人类-数据-模型交互画布：面向视觉分析 

**Authors**: Jürgen Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2505.07534)  

**Abstract**: Visual Analytics (VA) integrates humans, data, and models as key actors in insight generation and data-driven decision-making. This position paper values and reflects on 16 VA process models and frameworks and makes nine high-level observations that motivate a fresh perspective on VA. The contribution is the HDMI Canvas, a perspective to VA that complements the strengths of existing VA process models and frameworks. It systematically characterizes diverse roles of humans, data, and models, and how these actors benefit from and contribute to VA processes. The descriptive power of the HDMI Canvas eases the differentiation between a series of VA building blocks, rather than describing general VA principles only. The canvas includes modern human-centered methodologies, including human knowledge externalization and forms of feedback loops, while interpretable and explainable AI highlight model contributions beyond their conventional outputs. The HDMI Canvas has generative power, guiding the design of new VA processes and is optimized for external stakeholders, improving VA outreach, interdisciplinary collaboration, and user-centered design. The utility of the HDMI Canvas is demonstrated through two preliminary case studies. 

**Abstract (ZH)**: 视觉分析（VA）将人、数据和模型作为洞察生成和数据驱动决策的关键角色。本文植基于此立场，反思和评估了16种VA过程模型和框架，并提出了九个高层次的观察，从而为VA提供了一个新的视角。本文的贡献是HDMI画布，这是一种补充现有VA过程模型和框架优点的视角。它系统地刻画了人、数据和模型的多样化角色及其在VA过程中的受益和贡献。HDMI画布的描述能力便于区分一系列VA构建块，而不仅仅是描述一般性的VA原则。画布包含了现代以人为中心的方法论，包括人类知识外部化和反馈回路的形式，可解释的人工智能突出了模型贡献超越其传统输出。HDMI画布具有生成能力，指导新VA过程的设计，并优化了对外部利益相关者的支持，改善了VA的普及、跨学科合作和用户中心设计。HDMI画布通过两个初步案例研究展示了其实用性。 

---
# IKrNet: A Neural Network for Detecting Specific Drug-Induced Patterns in Electrocardiograms Amidst Physiological Variability 

**Title (ZH)**: IKrNet:一种用于在生理变异背景下检测特定药物诱导模式的神经网络 

**Authors**: Ahmad Fall, Federica Granese, Alex Lence, Dominique Fourer, Blaise Hanczar, Joe-Elie Salem, Jean-Daniel Zucker, Edi Prifti  

**Link**: [PDF](https://arxiv.org/pdf/2505.07533)  

**Abstract**: Monitoring and analyzing electrocardiogram (ECG) signals, even under varying physiological conditions, including those influenced by physical activity, drugs and stress, is crucial to accurately assess cardiac health. However, current AI-based methods often fail to account for how these factors interact and alter ECG patterns, ultimately limiting their applicability in real-world settings. This study introduces IKrNet, a novel neural network model, which identifies drug-specific patterns in ECGs amidst certain physiological conditions. IKrNet's architecture incorporates spatial and temporal dynamics by using a convolutional backbone with varying receptive field size to capture spatial features. A bi-directional Long Short-Term Memory module is also employed to model temporal dependencies. By treating heart rate variability as a surrogate for physiological fluctuations, we evaluated IKrNet's performance across diverse scenarios, including conditions with physical stress, drug intake alone, and a baseline without drug presence. Our assessment follows a clinical protocol in which 990 healthy volunteers were administered 80mg of Sotalol, a drug which is known to be a precursor to Torsades-de-Pointes, a life-threatening arrhythmia. We show that IKrNet outperforms state-of-the-art models' accuracy and stability in varying physiological conditions, underscoring its clinical viability. 

**Abstract (ZH)**: 监测和分析在不同生理条件下的心电图（ECG）信号，包括由体力活动、药物和压力等因素影响的条件，对于准确评估心脏健康至关重要。然而，当前基于人工智能的方法往往未能考虑这些因素如何相互作用并改变ECG模式，从而限制了其在实际环境中的应用。本研究引入了IKrNet这一新型神经网络模型，它能够在特定生理条件下识别药物特异性的心电图模式。IKrNet的架构通过使用具有可变接收野大小的卷积骨干网络来捕捉空间特征，并采用双向长短期记忆模块建模时间依赖性。通过将心率变异性作为生理波动的替代指标，我们评估了IKrNet在包括生理压力、单独用药和无药物存在的多样化场景中的性能。研究中，990名健康志愿者接受了80mg索他洛尔的给药，这是一种已知的可能导致危及生命的室性心动过速的前体药物。结果显示，IKrNet在不同生理条件下优于现有模型的准确性和稳定性，证实了其临床适用性。 

---
# ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution 

**Title (ZH)**: ToolACE-DEV: 自我提升的工具学习通过分解与进化 

**Authors**: Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07512)  

**Abstract**: The tool-using capability of large language models (LLMs) enables them to access up-to-date external information and handle complex tasks. Current approaches to enhancing this capability primarily rely on distilling advanced models by data synthesis. However, this method incurs significant costs associated with advanced model usage and often results in data compatibility issues, led by the high discrepancy in the knowledge scope between the advanced model and the target model. To address these challenges, we propose ToolACE-DEV, a self-improving framework for tool learning. First, we decompose the tool-learning objective into sub-tasks that enhance basic tool-making and tool-using abilities. Then, we introduce a self-evolving paradigm that allows lightweight models to self-improve, reducing reliance on advanced LLMs. Extensive experiments validate the effectiveness of our approach across models of varying scales and architectures. 

**Abstract (ZH)**: 大型语言模型（LLMs）的工具使用能力使它们能够访问最新的外部信息并处理复杂任务。当前增强这种能力的方法主要依赖于通过数据合成蒸馏先进模型。然而，这种方法会产生与先进模型使用相关的高昂成本，并且经常导致数据兼容性问题，原因是先进模型和目标模型的知识范围存在巨大差异。为了解决这些挑战，我们提出了一种自提高框架ToolACE-DEV，用于工具学习。首先，我们将工具学习目标分解为增强基本工具制造和使用能力的子任务。然后，我们引入了一种自我进化的范式，使轻量级模型能够自我改进，从而减少对先进LLM的依赖。广泛实验证明了该方法在不同规模和架构模型上的有效性。 

---
# MAIS: Memory-Attention for Interactive Segmentation 

**Title (ZH)**: MAIS: 记忆注意力机制用于交互式分割 

**Authors**: Mauricio Orbes-Arteaga, Oeslle Lucena, Sabastien Ourselin, M. Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.07511)  

**Abstract**: Interactive medical segmentation reduces annotation effort by refining predictions through user feedback. Vision Transformer (ViT)-based models, such as the Segment Anything Model (SAM), achieve state-of-the-art performance using user clicks and prior masks as prompts. However, existing methods treat interactions as independent events, leading to redundant corrections and limited refinement gains. We address this by introducing MAIS, a Memory-Attention mechanism for Interactive Segmentation that stores past user inputs and segmentation states, enabling temporal context integration. Our approach enhances ViT-based segmentation across diverse imaging modalities, achieving more efficient and accurate refinements. 

**Abstract (ZH)**: 交互式医学分割通过用户反馈 refinement 预测从而减少标注努力。通过引入基于记忆-注意机制的 MAIS，利用过往用户输入和分割状态实现时间上下文集成，我们提升了 Vision Transformer (ViT) 基模型在多种成像模态下的分割性能，实现了更高效和准确的 refinement。 

---
# EAGLE: Contrastive Learning for Efficient Graph Anomaly Detection 

**Title (ZH)**: EAGLE: 对比学习在高效图异常检测中的应用 

**Authors**: Jing Ren, Mingliang Hou, Zhixuan Liu, Xiaomei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.07508)  

**Abstract**: Graph anomaly detection is a popular and vital task in various real-world scenarios, which has been studied for several decades. Recently, many studies extending deep learning-based methods have shown preferable performance on graph anomaly detection. However, existing methods are lack of efficiency that is definitely necessary for embedded devices. Towards this end, we propose an Efficient Anomaly detection model on heterogeneous Graphs via contrastive LEarning (EAGLE) by contrasting abnormal nodes with normal ones in terms of their distances to the local context. The proposed method first samples instance pairs on meta path-level for contrastive learning. Then, a graph autoencoder-based model is applied to learn informative node embeddings in an unsupervised way, which will be further combined with the discriminator to predict the anomaly scores of nodes. Experimental results show that EAGLE outperforms the state-of-the-art methods on three heterogeneous network datasets. 

**Abstract (ZH)**: 基于对比学习的异构图高效异常检测模型（EAGLE） 

---
# Can Generative AI agents behave like humans? Evidence from laboratory market experiments 

**Title (ZH)**: 生成式AI代理能否像人类一样行为？来自实验室市场实验的证据 

**Authors**: R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07457)  

**Abstract**: We explore the potential of Large Language Models (LLMs) to replicate human behavior in economic market experiments. Compared to previous studies, we focus on dynamic feedback between LLM agents: the decisions of each LLM impact the market price at the current step, and so affect the decisions of the other LLMs at the next step. We compare LLM behavior to market dynamics observed in laboratory settings and assess their alignment with human participants' behavior. Our findings indicate that LLMs do not adhere strictly to rational expectations, displaying instead bounded rationality, similarly to human participants. Providing a minimal context window i.e. memory of three previous time steps, combined with a high variability setting capturing response heterogeneity, allows LLMs to replicate broad trends seen in human experiments, such as the distinction between positive and negative feedback markets. However, differences remain at a granular level--LLMs exhibit less heterogeneity in behavior than humans. These results suggest that LLMs hold promise as tools for simulating realistic human behavior in economic contexts, though further research is needed to refine their accuracy and increase behavioral diversity. 

**Abstract (ZH)**: 我们探讨大型语言模型（LLMs）在经济市场实验中复制人类行为的潜力。与以往研究相比，我们关注LLM代理之间的动态反馈：每个LLM的决策会影响当前步骤的市场价格，从而影响其他LLM在下一步骤的决策。我们将LLM的行为与实验室环境中观察到的市场动态进行比较，并评估其与人类参与者行为的契合度。研究发现，LLMs并不严格遵守理性预期，而是表现出局限性理性，类似于人类参与者。提供一个最小的上下文窗口即三步之前的记忆，结合一个高变异设置以捕捉反应异质性，使LLMs能够复制人类实验中看到的广泛趋势，如正反馈市场和负反馈市场的区别。然而，在细微层面上仍存在差异——LLMs在行为上的异质性低于人类。这些结果表明，LLMs有潜力成为模拟经济背景下现实人类行为的工具，但仍需进一步研究以提高其准确性并增加行为多样性。 

---
# Prototype Augmented Hypernetworks for Continual Learning 

**Title (ZH)**: 持续学习中的原型增强超网络 

**Authors**: Neil De La Fuente, Maria Pilligua, Daniel Vidal, Albin Soutiff, Cecilia Curreli, Daniel Cremers, Andrey Barsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.07450)  

**Abstract**: Continual learning (CL) aims to learn a sequence of tasks without forgetting prior knowledge, but gradient updates for a new task often overwrite the weights learned earlier, causing catastrophic forgetting (CF). We propose Prototype-Augmented Hypernetworks (PAH), a framework where a single hypernetwork, conditioned on learnable task prototypes, dynamically generates task-specific classifier heads on demand. To mitigate forgetting, PAH combines cross-entropy with dual distillation losses, one to align logits and another to align prototypes, ensuring stable feature representations across tasks. Evaluations on Split-CIFAR100 and TinyImageNet demonstrate that PAH achieves state-of-the-art performance, reaching 74.5 % and 63.7 % accuracy with only 1.7 % and 4.4 % forgetting, respectively, surpassing prior methods without storing samples or heads. 

**Abstract (ZH)**: 持续学习（CL）旨在学习一系列任务而不遗忘先前的知识，但新任务的梯度更新往往会覆盖之前学到的权重，导致灾难性遗忘（CF）。我们提出了原型增强超网络（PAH），这是一种框架，其中单个超网络根据可学习的任务原型，动态生成特定于任务的分类器头部。为了减轻遗忘，PAH 结合了交叉熵损失与双重蒸馏损失，后者用于对齐概率输出和原型，确保跨任务的稳定特征表示。在 Split-CIFAR100 和 TinyImageNet 上的评估表明，PAH 实现了最先进的性能，分别仅产生 1.7% 和 4.4% 的遗忘率，准确率达到 74.5% 和 63.7%，超越了无需存储样本或头部的先前方法。 

---
# Unified Continuous Generative Models 

**Title (ZH)**: 统一连续生成模型 

**Authors**: Peng Sun, Yi Jiang, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07447)  

**Abstract**: Recent advances in continuous generative models, including multi-step approaches like diffusion and flow-matching (typically requiring 8-1000 sampling steps) and few-step methods such as consistency models (typically 1-8 steps), have demonstrated impressive generative performance. However, existing work often treats these approaches as distinct paradigms, resulting in separate training and sampling methodologies. We introduce a unified framework for training, sampling, and analyzing these models. Our implementation, the Unified Continuous Generative Models Trainer and Sampler (UCGM-{T,S}), achieves state-of-the-art (SOTA) performance. For example, on ImageNet 256x256 using a 675M diffusion transformer, UCGM-T trains a multi-step model achieving 1.30 FID in 20 steps and a few-step model reaching 1.42 FID in just 2 steps. Additionally, applying UCGM-S to a pre-trained model (previously 1.26 FID at 250 steps) improves performance to 1.06 FID in only 40 steps. Code is available at: this https URL. 

**Abstract (ZH)**: Recent advances in连续生成模型的近期进展，包括多步方法如扩散和流匹配（通常需要8-1000个采样步骤）和少步方法如一致性模型（通常需要1-8个步骤），已经展示了出色的生成性能。然而，现有工作通常将这些方法视为不同的范式，导致了各自独立的训练和采样方法。我们提出了一种统一的框架，用于训练、采样和分析这些模型。我们的实现，统一连续生成模型训练器和采样器（UCGM-{T,S}），达到了最先进的（SOTA）性能。例如，在使用675M扩散变换器进行ImageNet 256x256的实验中，UCGM-T训练了一个多步模型，在20步中实现了1.30的FID，并且训练了一个少步模型，在仅2步中达到了1.42的FID。此外，将UCGM-S应用于一个预训练模型（以前在250步时FID为1.26），在仅40步中将性能提升至1.06的FID。代码可在以下链接获取：this https URL。 

---
# LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning 

**Title (ZH)**: 迭代数据选择以实现高效的大规模语言模型指令调优 

**Authors**: Xiaotian Lin, Yanlin Qi, Yizhang Zhu, Themis Palpanas, Chengliang Chai, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07437)  

**Abstract**: Instruction tuning has emerged as a critical paradigm for improving the capabilities and alignment of large language models (LLMs). However, existing iterative model-aware data selection methods incur significant computational overhead, as they rely on repeatedly performing full-dataset model inference to estimate sample utility for subsequent training iterations, creating a fundamental efficiency bottleneck. In this paper, we propose LEAD, an efficient iterative data selection framework that accurately estimates sample utility entirely within the standard training loop, eliminating the need for costly additional model inference. At its core, LEAD introduces Instance-Level Dynamic Uncertainty (IDU), a theoretically grounded utility function combining instantaneous training loss, gradient-based approximation of loss changes, and exponential smoothing of historical loss signals. To further scale efficiently to large datasets, LEAD employs a two-stage, coarse-to-fine selection strategy, adaptively prioritizing informative clusters through a multi-armed bandit mechanism, followed by precise fine-grained selection of high-utility samples using IDU. Extensive experiments across four diverse benchmarks show that LEAD significantly outperforms state-of-the-art methods, improving average model performance by 6.1%-10.8% while using only 2.5% of the training data and reducing overall training time by 5-10x. 

**Abstract (ZH)**: LEAD：一种高效的数据选择框架，用于大型语言模型的指令调优 

---
# AI in Money Matters 

**Title (ZH)**: AI在金融事务中的应用 

**Authors**: Nadine Sandjo Tchatchoua, Richard Harper  

**Link**: [PDF](https://arxiv.org/pdf/2505.07393)  

**Abstract**: In November 2022, Europe and the world by and large were stunned by the birth of a new large language model : ChatGPT. Ever since then, both academic and populist discussions have taken place in various public spheres such as LinkedIn and X(formerly known as Twitter) with the view to both understand the tool and its benefits for the society. The views of real actors in professional spaces, especially in regulated industries such as finance and law have been largely missing. We aim to begin to close this gap by presenting results from an empirical investigation conducted through interviews with professional actors in the Fintech industry. The paper asks the question, how and to what extent are large language models in general and ChatGPT in particular being adopted and used in the Fintech industry? The results show that while the fintech experts we spoke with see a potential in using large language models in the future, a lot of questions marks remain concerning how they are policed and therefore might be adopted in a regulated industry such as Fintech. This paper aims to add to the existing academic discussing around large language models, with a contribution to our understanding of professional viewpoints. 

**Abstract (ZH)**: 欧洲和世界在2022年11月对一个新的大型语言模型ChatGPT的诞生感到震惊。自此之后，学术界和普通民众在LinkedIn和X（原Twitter）等公共领域展开了讨论，试图理解这一工具及其对社会的好处。来自金融和法律等受监管行业的专业人员的观点在这些讨论中 largely 缺席。我们希望通过访谈金融科技行业专业人士进行实证调查的结果，开始缩小这一缺口。本文探讨了大型语言模型，特别是ChatGPT，在金融科技行业中被采纳和使用的具体情况和程度。结果显示，尽管我们访谈的金融科技专家看到了未来使用大型语言模型的潜力，但在一个如金融科技这样的受监管行业中，它们如何被监管以及可能会如何被采纳仍存在许多疑问。本文旨在为现有关于大型语言模型的学术讨论做出贡献，并增进我们对专业视角的理解。 

---
# Few-shot Semantic Encoding and Decoding for Video Surveillance 

**Title (ZH)**: Few-shot语义编码与解码在视频監控中的应用 

**Authors**: Baoping Cheng, Yukun Zhang, Liming Wang, Xiaoyan Xie, Tao Fu, Dongkun Wang, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07381)  

**Abstract**: With the continuous increase in the number and resolution of video surveillance cameras, the burden of transmitting and storing surveillance video is growing. Traditional communication methods based on Shannon's theory are facing optimization bottlenecks. Semantic communication, as an emerging communication method, is expected to break through this bottleneck and reduce the storage and transmission consumption of video. Existing semantic decoding methods often require many samples to train the neural network for each scene, which is time-consuming and labor-intensive. In this study, a semantic encoding and decoding method for surveillance video is proposed. First, the sketch was extracted as semantic information, and a sketch compression method was proposed to reduce the bit rate of semantic information. Then, an image translation network was proposed to translate the sketch into a video frame with a reference frame. Finally, a few-shot sketch decoding network was proposed to reconstruct video from sketch. Experimental results showed that the proposed method achieved significantly better video reconstruction performance than baseline methods. The sketch compression method could effectively reduce the storage and transmission consumption of semantic information with little compromise on video quality. The proposed method provides a novel semantic encoding and decoding method that only needs a few training samples for each surveillance scene, thus improving the practicality of the semantic communication system. 

**Abstract (ZH)**: 基于视频监控的语义编码与解码方法 

---
# Examining the Role of LLM-Driven Interactions on Attention and Cognitive Engagement in Virtual Classrooms 

**Title (ZH)**: 考查LLM驱动交互在虚拟课堂中对注意力和认知参与的作用 

**Authors**: Suleyman Ozdel, Can Sarpkaya, Efe Bozkir, Hong Gao, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07377)  

**Abstract**: Transforming educational technologies through the integration of large language models (LLMs) and virtual reality (VR) offers the potential for immersive and interactive learning experiences. However, the effects of LLMs on user engagement and attention in educational environments remain open questions. In this study, we utilized a fully LLM-driven virtual learning environment, where peers and teachers were LLM-driven, to examine how students behaved in such settings. Specifically, we investigate how peer question-asking behaviors influenced student engagement, attention, cognitive load, and learning outcomes and found that, in conditions where LLM-driven peer learners asked questions, students exhibited more targeted visual scanpaths, with their attention directed toward the learning content, particularly in complex subjects. Our results suggest that peer questions did not introduce extraneous cognitive load directly, as the cognitive load is strongly correlated with increased attention to the learning material. Considering these findings, we provide design recommendations for optimizing VR learning spaces. 

**Abstract (ZH)**: 通过将大型语言模型（LLMs）和虚拟现实（VR）整合以转变教育技术提供了沉浸式和交互式学习体验的潜力。然而，LLMs 对用户在教育环境中的参与度和注意力的影响仍然存在疑问。在本研究中，我们利用了一个完全由LLM驱动的虚拟学习环境，其中同伴和教师都是由LLM驱动的，以考察学生在这种环境中的行为。具体而言，我们研究了同伴提问行为如何影响学生参与度、注意力、认知负荷和学习成果，发现当LLM驱动的同伴学习者提问时，学生表现出更定向的视线扫描路径，注意力集中在学习内容上，尤其是在复杂科目中。研究结果表明，同伴问题并没有直接引入额外的认知负荷，因为认知负荷与对学习材料的注意力增加有很强的相关性。基于这些发现，我们提供了优化VR学习空间的设计建议。 

---
# Synthetic Code Surgery: Repairing Bugs and Vulnerabilities with LLMs and Synthetic Data 

**Title (ZH)**: 合成代码手术：使用大规模语言模型和合成数据修复漏洞与错误 

**Authors**: David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2505.07372)  

**Abstract**: This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset outperforming both baseline and real-world commit data configurations in certain scenarios. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, potentially transforming approaches to data scarcity across software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance. 

**Abstract (ZH)**: 本研究提出了一种通过大型语言模型生成合成数据以增强自动化程序修复的新方法。当前的自动化程序修复系统受限于高质量训练数据的有限可用性，这些数据涵盖了多种编程语言中的不同类型的错误。所提出的方法通过两阶段过程来解决这一限制：合成样本生成和严格的质量评估。采用了多种最先进的大型语言模型，生成了约30,000个跨12种编程语言和13种错误类别的代码错误和修复配对示例。随后，这些样本经过针对正确性、代码质量、安全性、性能和完整性五个标准的多模型评估。在VulRepair测试集数据集上进行的实验评估显示，在某些场景下，经过质量过滤的合成数据集在完美预测率方面取得了统计学上的显著改善，优于基线和实际提交数据配置。该方法通过严格的统计测试得到了验证，包括ANOVA和事后Tukey’s诚实显著差异分析。此外，最优配置在使用较不计算密集型解码策略的情况下仍然超过了现有系统。这项研究确立了一种自我启动范式，其中大型语言模型自动生成并评估自己的训练数据，有可能在软件工程任务中改变数据稀缺性问题，并推动开发更加鲁棒和适应性强的自动化代码维护工具。 

---
# Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in The DCASE 2025 Challenge 

**Title (ZH)**: 面向声学内容推理的多域音频问答：DCASE 2025 挑战赛 Toward Acoustic Content Reasoning 的多域音频问答：DCASE 2025 挑战赛 

**Authors**: Chao-Han Huck Yang, Sreyan Ghosh, Qing Wang, Jaeyeon Kim, Hengyi Hong, Sonal Kumar, Guirui Zhong, Zhifeng Kong, S Sakshi, Vaibhavi Lokegaonkar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, Gunhee Kim, Jun Du, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2505.07365)  

**Abstract**: We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively. 

**Abstract (ZH)**: DCASE 2025挑战任务5：多领域音频问答基准 

---
# GAN-based synthetic FDG PET images from T1 brain MRI can serve to improve performance of deep unsupervised anomaly detection models 

**Title (ZH)**: 基于GAN的合成FDG PET图像可以从T1脑MRI中获得，并可改善深度无监督异常检测模型的性能 

**Authors**: Daria Zotova, Nicolas Pinon, Robin Trombetta, Romain Bouet, Julien Jung, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2505.07364)  

**Abstract**: Background and Objective. Research in the cross-modal medical image translation domain has been very productive over the past few years in tackling the scarce availability of large curated multimodality datasets with the promising performance of GAN-based architectures. However, only a few of these studies assessed task-based related performance of these synthetic data, especially for the training of deep models. Method. We design and compare different GAN-based frameworks for generating synthetic brain [18F]fluorodeoxyglucose (FDG) PET images from T1 weighted MRI data. We first perform standard qualitative and quantitative visual quality evaluation. Then, we explore further impact of using these fake PET data in the training of a deep unsupervised anomaly detection (UAD) model designed to detect subtle epilepsy lesions in T1 MRI and FDG PET images. We introduce novel diagnostic task-oriented quality metrics of the synthetic FDG PET data tailored to our unsupervised detection task, then use these fake data to train a use case UAD model combining a deep representation learning based on siamese autoencoders with a OC-SVM density support estimation model. This model is trained on normal subjects only and allows the detection of any variation from the pattern of the normal population. We compare the detection performance of models trained on 35 paired real MR T1 of normal subjects paired either on 35 true PET images or on 35 synthetic PET images generated from the best performing generative models. Performance analysis is conducted on 17 exams of epilepsy patients undergoing surgery. Results. The best performing GAN-based models allow generating realistic fake PET images of control subject with SSIM and PSNR values around 0.9 and 23.8, respectively and in distribution (ID) with regard to the true control dataset. The best UAD model trained on these synthetic normative PET data allows reaching 74% sensitivity. Conclusion. Our results confirm that GAN-based models are the best suited for MR T1 to FDG PET translation, outperforming transformer or diffusion models. We also demonstrate the diagnostic value of these synthetic data for the training of UAD models and evaluation on clinical exams of epilepsy patients. Our code and the normative image dataset are available. 

**Abstract (ZH)**: 背景与目的. 近几年，跨模态医学图像转换领域的研究在应对稀缺的大型多模态数据集方面取得了丰硕成果，并且基于生成对抗网络（GAN）的架构表现出令人promise的效果。然而，其中只有少数研究评估了这些合成数据的任务相关性能，尤其是用于训练深度模型。方法. 我们设计并比较了不同的基于GAN的框架，用于从T1加权MRI数据生成合成的[18F]氟脱氧葡萄糖（FDG）PET图像。我们首先进行标准的定性和定量视觉质量评估，然后进一步探索使用这些假PET数据训练用于检测T1 MRI和FDG PET图像中细微癫痫病灶的半监督异常检测（UAD）模型的影响。我们引入了针对我们无监督检测任务量身定制的新诊断任务导向的质量评估指标，然后使用这些假数据训练一个结合基于Siamese自动编码器的深度表征学习模型和OC-SVM密度支持估计模型的使用案例UAD模型。该模型仅使用正常受试者的数据进行训练，并能够检测任何与正常人群模式的偏离。我们将用35对真实T1 MRI图像正常受试者数据训练的模型与用35对由表现最佳的生成模型生成的合成PET图像训练的模型进行检测性能比较。性能分析是在17例癫痫患者手术过程中进行的。结果. 表现最佳的GAN模型能够生成与对照受试者真实PET图像在结构相似性（SSIM）值约为0.9和峰值信噪比（PSNR）值约为23.8方面高度真实的假PET图像，并且在分布上与真实对照数据集一致。使用这些合成的正常PET数据训练的最佳UAD模型可实现74%的灵敏度。结论. 我们的成果表明，基于GAN的模型最适合用于从T1 MRI到FDG PET的转换，优于变压器或扩散模型。我们还证明了这些合成数据在UAD模型训练和在癫痫患者临床检查评估中的诊断价值。我们的代码和正常图像数据集已公开。 

---
# QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines 

**Title (ZH)**: QUPID: 量化理解以提升韩语搜索引擎的性能、洞察与决策能力 

**Authors**: Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07345)  

**Abstract**: Large language models (LLMs) have been widely used for relevance assessment in information retrieval. However, our study demonstrates that combining two distinct small language models (SLMs) with different architectures can outperform LLMs in this task. Our approach -- QUPID -- integrates a generative SLM with an embedding-based SLM, achieving higher relevance judgment accuracy while reducing computational costs compared to state-of-the-art LLM solutions. This computational efficiency makes QUPID highly scalable for real-world search systems processing millions of queries daily. In experiments across diverse document types, our method demonstrated consistent performance improvements (Cohen's Kappa of 0.646 versus 0.387 for leading LLMs) while offering 60x faster inference times. Furthermore, when integrated into production search pipelines, QUPID improved nDCG@5 scores by 1.9%. These findings underscore how architectural diversity in model combinations can significantly enhance both search relevance and operational efficiency in information retrieval systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在信息检索中的相关性评估中已广泛应用。然而，我们的研究表明，结合两种具有不同架构的独立小型语言模型（SLMs）可以在这一任务上超越LLMs。我们的方法——QUPID——将生成型SLM与基于嵌入的SLM集成，相比于最先进的LLM解决方案，能够在提高相关性判断准确率的同时降低计算成本。这种计算效率使QUPID在处理每天数百万查询的实际搜索系统中具有高度的可扩展性。在针对不同文档类型的实验中，我们的方法展示了一致性的性能改进（科恩κ系数为0.646，而领先的LLM为0.387），同时提供60倍更快的推理时间。此外，当集成到生产搜索管道中时，QUPID将nDCG@5分数提高了1.9%。这些发现强调了在模型组合中架构多样性可以显著提升信息检索系统中的搜索相关性和操作效率。 

---
# Generative Pre-trained Autoregressive Diffusion Transformer 

**Title (ZH)**: 预训练自回归扩散变换器生成模型 

**Authors**: Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07344)  

**Abstract**: In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space. 

**Abstract (ZH)**: GPDiT：统一扩散与自回归建模长时序视频合成的生成预训练自回归扩散变换器 

---
# Laypeople's Attitudes Towards Fair, Affirmative, and Discriminatory Decision-Making Algorithms 

**Title (ZH)**: lay人群对公平、肯定性及歧视性决策算法的态度研究 

**Authors**: Gabriel Lima, Nina Grgić-Hlača, Markus Langer, Yixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07339)  

**Abstract**: Affirmative algorithms have emerged as a potential answer to algorithmic discrimination, seeking to redress past harms and rectify the source of historical injustices. We present the results of two experiments ($N$$=$$1193$) capturing laypeople's perceptions of affirmative algorithms -- those which explicitly prioritize the historically marginalized -- in hiring and criminal justice. We contrast these opinions about affirmative algorithms with folk attitudes towards algorithms that prioritize the privileged (i.e., discriminatory) and systems that make decisions independently of demographic groups (i.e., fair). We find that people -- regardless of their political leaning and identity -- view fair algorithms favorably and denounce discriminatory systems. In contrast, we identify disagreements concerning affirmative algorithms: liberals and racial minorities rate affirmative systems as positively as their fair counterparts, whereas conservatives and those from the dominant racial group evaluate affirmative algorithms as negatively as discriminatory systems. We identify a source of these divisions: people have varying beliefs about who (if anyone) is marginalized, shaping their views of affirmative algorithms. We discuss the possibility of bridging these disagreements to bring people together towards affirmative algorithms. 

**Abstract (ZH)**: 肯定算法作为一种潜在的解决算法歧视的方法已经 emergence，并寻求纠正历史不公的根源。我们通过两个实验（N=1193）探讨了普通民众对明确优先考虑历史上被边缘化群体的肯定算法在招聘和司法领域的看法。我们将这些关于肯定算法的看法与倾向于优先考虑特权群体（即歧视性）算法的民间态度，以及与不考虑人群因素而独立做决策的系统（即公平）的民间态度进行对比。我们发现，无论政治倾向和身份如何，人们普遍对公平算法持积极态度，并谴责歧视性系统。相反，我们发现了关于肯定算法的分歧：自由派和种族 minorities 对肯定性系统持与公平算法类似的积极看法，而保守派和主导种族群体的成员则对肯定算法持与歧视性系统类似的消极态度。我们识别出这些分歧的原因：人们对谁（如果有人）被边缘化的看法不同，这影响了他们对肯定算法的看法。我们讨论了弥合这些分歧的可能性，以推动人们共同支持肯定算法。 

---
# SAEN-BGS: Energy-Efficient Spiking AutoEncoder Network for Background Subtraction 

**Title (ZH)**: SAEN-BGS: 能效可突触自动编码网络背景减除 

**Authors**: Zhixuan Zhang, Xiaopeng Li, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07336)  

**Abstract**: Background subtraction (BGS) is utilized to detect moving objects in a video and is commonly employed at the onset of object tracking and human recognition processes. Nevertheless, existing BGS techniques utilizing deep learning still encounter challenges with various background noises in videos, including variations in lighting, shifts in camera angles, and disturbances like air turbulence or swaying trees. To address this problem, we design a spiking autoencoder network, termed SAEN-BGS, based on noise resilience and time-sequence sensitivity of spiking neural networks (SNNs) to enhance the separation of foreground and background. To eliminate unnecessary background noise and preserve the important foreground elements, we begin by creating the continuous spiking conv-and-dconv block, which serves as the fundamental building block for the decoder in SAEN-BGS. Moreover, in striving for enhanced energy efficiency, we introduce a novel self-distillation spiking supervised learning method grounded in ANN-to-SNN frameworks, resulting in decreased power consumption. In extensive experiments conducted on CDnet-2014 and DAVIS-2016 datasets, our approach demonstrates superior segmentation performance relative to other baseline methods, even when challenged by complex scenarios with dynamic backgrounds. 

**Abstract (ZH)**: 基于抗噪性和时序敏感性的脉冲自编码网络背景分割（Spiking Autoencoder Network-based Background Subtraction with Noise Resilience and Time-Sequence Sensitivity, SAEN-BGS） 

---
# Dynamical Label Augmentation and Calibration for Noisy Electronic Health Records 

**Title (ZH)**: 动态标签增强与校准在 noisy 电子健康记录中的应用 

**Authors**: Yuhao Li, Ling Luo, Uwe Aickelin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07320)  

**Abstract**: Medical research, particularly in predicting patient outcomes, heavily relies on medical time series data extracted from Electronic Health Records (EHR), which provide extensive information on patient histories. Despite rigorous examination, labeling errors are inevitable and can significantly impede accurate predictions of patient outcome. To address this challenge, we propose an \textbf{A}ttention-based Learning Framework with Dynamic \textbf{C}alibration and Augmentation for \textbf{T}ime series Noisy \textbf{L}abel \textbf{L}earning (ACTLL). This framework leverages a two-component Beta mixture model to identify the certain and uncertain sets of instances based on the fitness distribution of each class, and it captures global temporal dynamics while dynamically calibrating labels from the uncertain set or augmenting confident instances from the certain set. Experimental results on large-scale EHR datasets eICU and MIMIC-IV-ED, and several benchmark datasets from the UCR and UEA repositories, demonstrate that our model ACTLL has achieved state-of-the-art performance, especially under high noise levels. 

**Abstract (ZH)**: 基于动态校准与增广的注意力学习框架以应对医疗时间序列噪声标签学习（ACTLL） 

---
# How Do Companies Manage the Environmental Sustainability of AI? An Interview Study About Green AI Efforts and Regulations 

**Title (ZH)**: 如何管理人工智能的环境可持续性？关于绿色人工智能努力与监管的访谈研究 

**Authors**: Ashmita Sampatsing, Sophie Vos, Emma Beauxis-Aussalet, Justus Bogner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07317)  

**Abstract**: With the ever-growing adoption of artificial intelligence (AI), AI-based software and its negative impact on the environment are no longer negligible, and studying and mitigating this impact has become a critical area of research. However, it is currently unclear which role environmental sustainability plays during AI adoption in industry and how AI regulations influence Green AI practices and decision-making in industry. We therefore aim to investigate the Green AI perception and management of industry practitioners. To this end, we conducted a total of 11 interviews with participants from 10 different organizations that adopted AI-based software. The interviews explored three main themes: AI adoption, current efforts in mitigating the negative environmental impact of AI, and the influence of the EU AI Act and the Corporate Sustainability Reporting Directive (CSRD). Our findings indicate that 9 of 11 participants prioritized business efficiency during AI adoption, with minimal consideration of environmental sustainability. Monitoring and mitigation of AI's environmental impact were very limited. Only one participant monitored negative environmental effects. Regarding applied mitigation practices, six participants reported no actions, with the others sporadically mentioning techniques like prompt engineering, relying on smaller models, or not overusing AI. Awareness and compliance with the EU AI Act are low, with only one participant reporting on its influence, while the CSRD drove sustainability reporting efforts primarily in larger companies. All in all, our findings reflect a lack of urgency and priority for sustainable AI among these companies. We suggest that current regulations are not very effective, which has implications for policymakers. Additionally, there is a need to raise industry awareness, but also to provide user-friendly techniques and tools for Green AI practices. 

**Abstract (ZH)**: 随着人工智能（AI）的广泛应用，基于AI的软件及其对环境的负面影响已不容忽视，研究和减轻这种影响已成为关键研究领域。然而，目前尚不清楚环境可持续性在工业中采用AI过程中扮演何种角色，以及AI法规如何影响绿色AI的实践和决策。因此，我们旨在调查工业从业者对于绿色AI的认知与管理。为此，我们总共进行了11次访谈，参与者来自10家采用AI软件的不同组织。访谈探讨了三个主要主题：AI的采用、减轻AI对环境的负面影响的当前努力，以及欧盟AI法案和企业可持续性报告指令（CSRD）的影响。我们的研究发现，11名参与者中有9人优先考虑AI采用过程中的业务效率，对环境可持续性的考虑甚少。AI环境影响的监测与缓解措施极其有限，仅有一名参与者监控了负面环境效应。在实际采取的缓解措施方面，有六名参与者报告没有采取任何行动，其他参与者偶尔提到了一些技术手段，如提示工程、使用较小的模型或不过度使用AI。对于欧盟AI法案的了解和遵守程度较低，仅有一名参与者报告了该法案的影响，而CSRD主要促使大型公司加强了可持续性报告。总之，我们的研究发现反映出这些公司在可持续AI方面缺乏紧迫性和优先级。建议当前的法规效果不佳，这给政策制定者带来了影响。此外，提高行业意识是必要的，还应提供用户友好的技术与工具以促进绿色AI实践。 

---
# Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study 

**Title (ZH)**: 面向协作知识委托的多智能体推理系统设计研究：一项探索性研究 

**Authors**: Baixuan Xu, Chunyang Li, Weiqi Wang, Wei Fan, Tianshi Zheng, Haochen Shi, Tao Fan, Yangqiu Song, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07313)  

**Abstract**: Designing effective collaboration structure for multi-agent LLM systems to enhance collective reasoning is crucial yet remains under-explored. In this paper, we systematically investigate how collaborative reasoning performance is affected by three key design dimensions: (1) Expertise-Domain Alignment, (2) Collaboration Paradigm (structured workflow vs. diversity-driven integration), and (3) System Scale. Our findings reveal that expertise alignment benefits are highly domain-contingent, proving most effective for contextual reasoning tasks. Furthermore, collaboration focused on integrating diverse knowledge consistently outperforms rigid task decomposition. Finally, we empirically explore the impact of scaling the multi-agent system with expertise specialization and study the computational trade off, highlighting the need for more efficient communication protocol design. This work provides concrete guidelines for configuring specialized multi-agent system and identifies critical architectural trade-offs and bottlenecks for scalable multi-agent reasoning. The code will be made available upon acceptance. 

**Abstract (ZH)**: 设计有效的多智能体LLM系统协作结构以增强集体推理至关重要但仍未充分探索。本论文系统地探索了协作推理性能受三个关键设计维度的影响：（1）专业知识-领域对齐，（2）合作范式（结构化工作流程 vs. 知识多样性集成），（3）系统规模。我们的研究发现，专业知识对齐的效果高度依赖于领域，证明在上下文推理任务中最为有效。此外，注重集成多样化知识的合作方式始终优于刚性任务分解。最后，我们实证探讨了专业知识专门化扩展多智能体系统的影响，并研究了计算权衡，强调了更高效通信协议设计的必要性。本工作为配置专业化的多智能体系统提供了具体的指南，并指出了可扩展多智能体推理的关键架构权衡和瓶颈。接受后将提供代码。 

---
# HuB: Learning Extreme Humanoid Balance 

**Title (ZH)**: HuB: 学习极端人形机器人平衡 

**Authors**: Tong Zhang, Boyuan Zheng, Ruiqian Nai, Yingdong Hu, Yen-Jen Wang, Geng Chen, Fanqi Lin, Jiongye Li, Chuye Hong, Koushil Sreenath, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07294)  

**Abstract**: The human body demonstrates exceptional motor capabilities-such as standing steadily on one foot or performing a high kick with the leg raised over 1.5 meters-both requiring precise balance control. While recent research on humanoid control has leveraged reinforcement learning to track human motions for skill acquisition, applying this paradigm to balance-intensive tasks remains challenging. In this work, we identify three key obstacles: instability from reference motion errors, learning difficulties due to morphological mismatch, and the sim-to-real gap caused by sensor noise and unmodeled dynamics. To address these challenges, we propose HuB (Humanoid Balance), a unified framework that integrates reference motion refinement, balance-aware policy learning, and sim-to-real robustness training, with each component targeting a specific challenge. We validate our approach on the Unitree G1 humanoid robot across challenging quasi-static balance tasks, including extreme single-legged poses such as Swallow Balance and Bruce Lee's Kick. Our policy remains stable even under strong physical disturbances-such as a forceful soccer strike-while baseline methods consistently fail to complete these tasks. Project website: this https URL 

**Abstract (ZH)**: 人类身体展示了卓越的运动能力——如单脚站立或抬腿超过1.5米的高踢动作，都要求精确的平衡控制。尽管最近关于类人控制器的研究利用强化学习来跟踪人体运动以获取技能，但在平衡密集型任务上应用这一范式仍存在挑战。在本文中，我们确定了三个关键障碍：参考运动错误引起的不稳定性、由于形态差异导致的学习困难，以及由传感器噪声和未建模动力学引起的仿真实验到真实环境的差距。为了应对这些挑战，我们提出了一种统一框架HuB（类人平衡），该框架融合了参考运动精细化、平衡感知策略学习和仿真实验到真实环境的鲁棒性训练，每个组件针对特定挑战。我们通过在Unitree G1类人机器人上执行具有挑战性的准静态平衡任务来验证我们的方法，包括极端的单腿姿态如燕子平衡和李小龙踢腿。即使在强烈的物理干扰下（如足球射门），我们的策略仍保持稳定，而基线方法则无法完成这些任务。项目网站：this https URL。 

---
# Semantic Retention and Extreme Compression in LLMs: Can We Have Both? 

**Title (ZH)**: LLMs中的语义保留与极端压缩：两者可以兼得吗？ 

**Authors**: Stanislas Laborde, Martin Cousseau, Antoun Yaacoub, Lionel Prevost  

**Link**: [PDF](https://arxiv.org/pdf/2505.07289)  

**Abstract**: The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate. 

**Abstract (ZH)**: 大型语言模型部署的指数增长加剧了对高效模型压缩技术的需求以降低计算和内存成本。虽然剪枝和量化显示了潜力，但它们的联合潜力尚未充分探索。本文探讨了联合压缩，并研究了如何战略性地结合剪枝和量化以获得优于单一方法的性能-压缩比。鉴于准确评估大型语言模型性能的挑战，本文解决了先前评价框架的关键局限性，并引入了语义保留压缩率（SrCr），这是一种新的度量标准，用于量化模型压缩与语义保留之间的权衡，以便优化剪枝-量化配置。实验表明，我们建议的组合在相同理论压缩率下，平均可实现20%的性能提升，相比于仅量化模型。 

---
# Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule 

**Title (ZH)**: 基于模态特定最优时间表的结构导向药物设计探索 

**Authors**: Keyue Qiu, Yuxuan Song, Zhehuan Fan, Peidong Liu, Zhe Zhang, Mingyue Zheng, Hao Zhou, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07286)  

**Abstract**: Structure-Based Drug Design (SBDD) is crucial for identifying bioactive molecules. Recent deep generative models are faced with challenges in geometric structure modeling. A major bottleneck lies in the twisted probability path of multi-modalities -- continuous 3D positions and discrete 2D topologies -- which jointly determine molecular geometries. By establishing the fact that noise schedules decide the Variational Lower Bound (VLB) for the twisted probability path, we propose VLB-Optimal Scheduling (VOS) strategy in this under-explored area, which optimizes VLB as a path integral for SBDD. Our model effectively enhances molecular geometries and interaction modeling, achieving state-of-the-art PoseBusters passing rate of 95.9% on CrossDock, more than 10% improvement upon strong baselines, while maintaining high affinities and robust intramolecular validity evaluated on held-out test set. 

**Abstract (ZH)**: 基于结构的药物设计（SBDD）对于识别生物活性分子至关重要。近期的深度生成模型在几何结构建模方面面临挑战。瓶颈在于多模态下的曲折概率路径——连续的3D位置和离散的2D拓扑——它们共同决定了分子几何结构。通过确立噪声调度决定曲折概率路径的变分下界（VLB），我们在此领域提出了变分下界最优调度（VOS）策略，以路径积分优化VLB，以提升SBDD。我们的模型有效增强了分子几何结构和相互作用建模，在CrossDock上的PoseBusters通过率达到了95.9%，比强基线提高了超过10%，同时在保留高亲和力和稳健的分子内有效性方面，在保留的测试集上表现良好。 

---
# Predicting Music Track Popularity by Convolutional Neural Networks on Spotify Features and Spectrogram of Audio Waveform 

**Title (ZH)**: 基于Spotify特征和音频波形谱图的卷积神经网络音乐轨道流行度预测 

**Authors**: Navid Falah, Behnam Yousefimehr, Mehdi Ghatee  

**Link**: [PDF](https://arxiv.org/pdf/2505.07280)  

**Abstract**: In the digital streaming landscape, it's becoming increasingly challenging for artists and industry experts to predict the success of music tracks. This study introduces a pioneering methodology that uses Convolutional Neural Networks (CNNs) and Spotify data analysis to forecast the popularity of music tracks. Our approach takes advantage of Spotify's wide range of features, including acoustic attributes based on the spectrogram of audio waveform, metadata, and user engagement metrics, to capture the complex patterns and relationships that influence a track's popularity. Using a large dataset covering various genres and demographics, our CNN-based model shows impressive effectiveness in predicting the popularity of music tracks. Additionally, we've conducted extensive experiments to assess the strength and adaptability of our model across different musical styles and time periods, with promising results yielding a 97\% F1 score. Our study not only offers valuable insights into the dynamic landscape of digital music consumption but also provides the music industry with advanced predictive tools for assessing and predicting the success of music tracks. 

**Abstract (ZH)**: 数字流媒体 landscapes 中，艺术家和行业专家越来越难以预测音乐轨道的成功。本研究介绍了一种开创性的方法，该方法利用卷积神经网络（CNNs）和Spotify数据分析来预测音乐轨道的受欢迎程度。我们的方法利用了Spotify广泛的特征，包括基于音频波形光谱图的声学属性、元数据和用户参与度指标，以捕捉影响轨道受欢迎程度的复杂模式和关系。使用涵盖各种流派和人口统计学的大数据集，我们的基于CNN的模型在预测音乐轨道的受欢迎程度方面表现出色。此外，我们进行了广泛的实验，评估了该模型在不同音乐风格和时间段的强度和适应性，取得了令人鼓舞的结果，F1分数达到97%。本研究不仅提供了数字音乐消费动态景观的宝贵见解，还为音乐行业提供了先进的预测工具，用于评估和预测音乐轨道的成功。 

---
# On the Robustness of Reward Models for Language Model Alignment 

**Title (ZH)**: 关于奖励模型在语言模型对齐中的健壮性研究 

**Authors**: Jiwoo Hong, Noah Lee, Eunki Kim, Guijin Son, Woojin Chung, Aman Gupta, Shao Tang, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2505.07271)  

**Abstract**: The Bradley-Terry (BT) model is widely practiced in reward modeling for reinforcement learning with human feedback (RLHF). Despite its effectiveness, reward models (RMs) trained with BT model loss are prone to over-optimization, losing generalizability to unseen input distributions. In this paper, we study the cause of over-optimization in RM training and its downstream effects on the RLHF procedure, accentuating the importance of distributional robustness of RMs in unseen data. First, we show that the excessive dispersion of hidden state norms is the main source of over-optimization. Then, we propose batch-wise sum-to-zero regularization (BSR) to enforce zero-centered reward sum per batch, constraining the rewards with extreme magnitudes. We assess the impact of BSR in improving robustness in RMs through four scenarios of over-optimization, where BSR consistently manifests better robustness. Subsequently, we compare the plain BT model and BSR on RLHF training and empirically show that robust RMs better align the policy to the gold preference model. Finally, we apply BSR to high-quality data and models, which surpasses state-of-the-art RMs in the 8B scale by adding more than 5% in complex preference prediction tasks. By conducting RLOO training with 8B RMs, AlpacaEval 2.0 reduces generation length by 40% while adding a 7% increase in win rate, further highlighting that robustness in RMs induces robustness in RLHF training. We release the code, data, and models: this https URL. 

**Abstract (ZH)**: 布拉德利-特里(BT)模型在具有人类反馈的强化学习(RLHF)中的奖励模型训练中广泛应用。尽管其有效，但使用BT模型损失训练的奖励模型(RMs)容易过度优化，失去对未见输入分布的泛化能力。本文研究了RMs训练中的过度优化原因及其对RLHF流程的下游影响，强调了RMs对未见数据分布鲁棒性的的重要性。首先，我们表明隐藏状态范数的过度分散是过度优化的主要来源。然后，我们提出了批次累和为零正则化(BSR)，以确保每批次的奖励总和为中心，限制极端幅度的奖励。通过在四种过度优化情景下评估BSR在提高RMs鲁棒性方面的影响，我们发现BSR表现出更好的鲁棒性。接着，我们将BSR与原始BT模型在RLHF训练中进行比较，并实验证明鲁棒的RMs更好地对齐了策略与黄金偏好模型。最后，我们将BSR应用于高质量的数据和模型，在8B规模的任务中实现了超过5%的复杂偏好预测性能提升，并通过RLOO训练AlpacaEval 2.0减少了生成长度40%的同时增加了7%的胜率，进一步强调了RMs的鲁棒性对RLHF训练的鲁棒性所起的作用。我们开源了代码、数据和模型：https://github.com/alibaba/Qwen 

---
# CHD: Coupled Hierarchical Diffusion for Long-Horizon Tasks 

**Title (ZH)**: CHD: 耦合层级扩散模型用于长期任务 

**Authors**: Ce Hao, Anxing Xiao, Zhiwei Xue, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2505.07261)  

**Abstract**: Diffusion-based planners have shown strong performance in short-horizon tasks but often fail in complex, long-horizon settings. We trace the failure to loose coupling between high-level (HL) sub-goal selection and low-level (LL) trajectory generation, which leads to incoherent plans and degraded performance. We propose Coupled Hierarchical Diffusion (CHD), a framework that models HL sub-goals and LL trajectories jointly within a unified diffusion process. A shared classifier passes LL feedback upstream so that sub-goals self-correct while sampling proceeds. This tight HL-LL coupling improves trajectory coherence and enables scalable long-horizon diffusion planning. Experiments across maze navigation, tabletop manipulation, and household environments show that CHD consistently outperforms both flat and hierarchical diffusion baselines. 

**Abstract (ZH)**: 基于扩散的规划在短时域任务中表现出强大的性能，但在复杂、长时域设置中经常失效。我们追踪失败原因归结为高层（HL）子目标选择与低层（LL）轨迹生成之间的松散耦合，导致不一致的计划和性能下降。我们提出了一种耦合层次扩散（CHD）框架，该框架在统一的扩散过程中联合建模高层子目标和低层轨迹。共享分类器将低层反馈传递至上层，使子目标在采样过程中自我修正。这种紧密的 HL-LL 耦合提高了轨迹的一致性，并使长时域扩散规划更具扩展性。跨迷宫导航、桌面操作和家庭环境的实验表明，CHD 一致地优于平面和层次扩散基准。 

---
# UMoE: Unifying Attention and FFN with Shared Experts 

**Title (ZH)**: UMoE：统一注意力与前馈网络的共享专家模块 

**Authors**: Yuanhang Yang, Chaozheng Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.07260)  

**Abstract**: Sparse Mixture of Experts (MoE) architectures have emerged as a promising approach for scaling Transformer models. While initial works primarily incorporated MoE into feed-forward network (FFN) layers, recent studies have explored extending the MoE paradigm to attention layers to enhance model performance. However, existing attention-based MoE layers require specialized implementations and demonstrate suboptimal performance compared to their FFN-based counterparts. In this paper, we aim to unify the MoE designs in attention and FFN layers by introducing a novel reformulation of the attention mechanism, revealing an underlying FFN-like structure within attention modules. Our proposed architecture, UMoE, achieves superior performance through attention-based MoE layers while enabling efficient parameter sharing between FFN and attention components. 

**Abstract (ZH)**: 基于注意力的稀疏混合专家统一设计：UMoE架构 

---
# No Query, No Access 

**Title (ZH)**: 无查询，无访问。 

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07258)  

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.
Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at this https URL 

**Abstract (ZH)**: 基于 Victim 数据的对抗攻击（VDBA）：仅使用 Victim 文本误导 NLP 模型 

---
# Incomplete In-context Learning 

**Title (ZH)**: 部分在上下文学习 

**Authors**: Wenqiang Wang, Yangshijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07251)  

**Abstract**: Large vision language models (LVLMs) achieve remarkable performance through Vision In-context Learning (VICL), a process that depends significantly on demonstrations retrieved from an extensive collection of annotated examples (retrieval database). Existing studies often assume that the retrieval database contains annotated examples for all labels. However, in real-world scenarios, delays in database updates or incomplete data annotation may result in the retrieval database containing labeled samples for only a subset of classes. We refer to this phenomenon as an \textbf{incomplete retrieval database} and define the in-context learning under this condition as \textbf{Incomplete In-context Learning (IICL)}. To address this challenge, we propose \textbf{Iterative Judgments and Integrated Prediction (IJIP)}, a two-stage framework designed to mitigate the limitations of IICL. The Iterative Judgments Stage reformulates an \(\boldsymbol{m}\)-class classification problem into a series of \(\boldsymbol{m}\) binary classification tasks, effectively converting the IICL setting into a standard VICL scenario. The Integrated Prediction Stage further refines the classification process by leveraging both the input image and the predictions from the Iterative Judgments Stage to enhance overall classification accuracy. IJIP demonstrates considerable performance across two LVLMs and two datasets under three distinct conditions of label incompleteness, achieving the highest accuracy of 93.9\%. Notably, even in scenarios where labels are fully available, IJIP still achieves the best performance of all six baselines. Furthermore, IJIP can be directly applied to \textbf{Prompt Learning} and is adaptable to the \textbf{text domain}. 

**Abstract (ZH)**: 大型视觉语言模型通过视觉上下文学习（VICL）实现显著性能，这一过程依赖于从大量标注示例集合（检索数据库）中检索的示例。现有研究通常假设检索数据库包含所有标签的标注示例。然而，在实际场景中，数据库更新延迟或数据标注不完整可能导致检索数据库仅包含部分类别的标注样本。我们称这种现象为“不完整检索数据库”，并在该情况下定义的上下文学习为“不完整上下文学习（IICL）”。为应对这一挑战，我们提出了一种两阶段框架“迭代判断与综合预测（IJIP）”，旨在缓解IICL的限制。迭代判断阶段将\(\boldsymbol{m}\)-类分类问题重新表述为\(\boldsymbol{m}\)个二元分类任务，有效将IICL设置转化为标准的VICL场景。综合预测阶段进一步通过结合输入图像和迭代判断阶段的预测结果来优化分类过程，提高整体分类准确性。IJIP在两个大型视觉语言模型和两个数据集的三种不同条件下的标签不完整性情况下均表现出显著性能，最高准确率达到93.9%。即使在标签完全可用的情况下，IJIP仍优于所有六个基准方法。此外，IJIP可以直接应用于提示学习并在文本域中具有适应性。 

---
# SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models 

**Title (ZH)**: SAS-Bench：一种细粒度的短答评分评估基准（Large Language Models版本） 

**Authors**: Peichao Lai, Kexuan Zhang, Yi Lin, Linyihan Zhang, Feiyang Ye, Jinhao Yan, Yanwei Xu, Conghui He, Yilei Wang, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07247)  

**Abstract**: Subjective Answer Grading (SAG) plays a crucial role in education, standardized testing, and automated assessment systems, particularly for evaluating short-form responses in Short Answer Scoring (SAS). However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems. 

**Abstract (ZH)**: 主观回答评分（SAG）在教育、标准化测试和自动化评估系统中起着重要作用，特别是在简短答案评分（SAS）中评估简短形式的响应方面。虽然现有的方法通常产生粗粒度的评分并且缺乏详细的理由说明，尽管大型语言模型（LLMs）在零样本评估中显示出潜在能力，但仍易受偏见、人类判断不一致和评分决策透明度低的影响。为克服这些局限性，我们引入了SAS-Bench，一个专门针对基于LLM的SAS任务的基准测试。SAS-Bench提供了细粒度、分步骤的评分、专家标注的错误类别以及源自真实世界学科特定考试的多样化问题类型。该基准测试促进了对模型推理过程和可解释性的详细评估。我们还发布了包含1,030个问题和4,109个学生回应的开源数据集，每个问题和回应都由领域专家标注。此外，我们进行了全面的实验，使用了多种LLM，确定了在评分科学相关问题时的主要挑战，并强调了少样本提示在提高评分准确性方面的有效性。我们的工作为开发更稳健、公平且教育意义更强的基于LLM的评估系统提供了宝贵的洞察。 

---
# REMEDI: Relative Feature Enhanced Meta-Learning with Distillation for Imbalanced Prediction 

**Title (ZH)**: REMEDI: 相对特征增强的元学习与蒸馏方法在不平衡预测中的应用 

**Authors**: Fei Liu, Huanhuan Ren, Yu Guan, Xiuxu Wang, Wang Lv, Zhiqiang Hu, Yaxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07245)  

**Abstract**: Predicting future vehicle purchases among existing owners presents a critical challenge due to extreme class imbalance (<0.5% positive rate) and complex behavioral patterns. We propose REMEDI (Relative feature Enhanced Meta-learning with Distillation for Imbalanced prediction), a novel multi-stage framework addressing these challenges. REMEDI first trains diverse base models to capture complementary aspects of user behavior. Second, inspired by comparative op-timization techniques, we introduce relative performance meta-features (deviation from ensemble mean, rank among peers) for effective model fusion through a hybrid-expert architecture. Third, we distill the ensemble's knowledge into a single efficient model via supervised fine-tuning with MSE loss, enabling practical deployment. Evaluated on approximately 800,000 vehicle owners, REMEDI significantly outperforms baseline approaches, achieving the business target of identifying ~50% of actual buyers within the top 60,000 recommendations at ~10% precision. The distilled model preserves the ensemble's predictive power while maintaining deployment efficiency, demonstrating REMEDI's effectiveness for imbalanced prediction in industry settings. 

**Abstract (ZH)**: 预测现有车主的未来购车行为面临着严重挑战，主要由于极不平衡的类别分布（阳性率<0.5%）和复杂的用户行为模式。我们提出了一种名为REMEDI（相对特征增强元学习与蒸馏不平衡预测）的新型多阶段框架，以应对这些挑战。REMEDI首先训练多样化的基础模型以捕捉用户行为的不同方面。其次，借鉴比较优化技术，我们引入了相对性能元特征（相对于集合平均值的偏差、在同侪中的排名）以通过混合专家架构实现有效的模型融合。第三，通过具有MSE损失的监督微调过程，将集成的知识蒸馏到单个高效模型中，从而实现实际部署。在约800,000名汽车车主上评估，REMEDI显著优于基线方法，实现业务目标，在前60,000推荐中识别约50%的实际买家，且精确率为10%左右。蒸馏后的模型保留了集成的预测能力，同时保持了部署效率，展示了REMEDI在工业环境中不平衡预测的有效性。 

---
# Comet: Accelerating Private Inference for Large Language Model by Predicting Activation Sparsity 

**Title (ZH)**: Comet: 通过预测激活稀疏性加速大型语言模型的隐私推理 

**Authors**: Guang Yan, Yuhui Zhang, Zimu Guo, Lutan Zhao, Xiaojun Chen, Chen Wang, Wenhao Wang, Dan Meng, Rui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07239)  

**Abstract**: With the growing use of large language models (LLMs) hosted on cloud platforms to offer inference services, privacy concerns about the potential leakage of sensitive information are escalating. Secure multi-party computation (MPC) is a promising solution to protect the privacy in LLM inference. However, MPC requires frequent inter-server communication, causing high performance overhead.
Inspired by the prevalent activation sparsity of LLMs, where most neuron are not activated after non-linear activation functions, we propose an efficient private inference system, Comet. This system employs an accurate and fast predictor to predict the sparsity distribution of activation function output. Additionally, we introduce a new private inference protocol. It efficiently and securely avoids computations involving zero values by exploiting the spatial locality of the predicted sparse distribution. While this computation-avoidance approach impacts the spatiotemporal continuity of KV cache entries, we address this challenge with a low-communication overhead cache refilling strategy that merges miss requests and incorporates a prefetching mechanism. Finally, we evaluate Comet on four common LLMs and compare it with six state-of-the-art private inference systems. Comet achieves a 1.87x-2.63x speedup and a 1.94x-2.64x communication reduction. 

**Abstract (ZH)**: 基于云平台的大语言模型推理服务中，隐私泄露担忧加剧，安全多方计算是保护隐私的 promising 解决方案。然而，安全多方计算需要频繁的服务器间通信，导致高性能开销。受大语言模型中普遍存在的激活稀疏性启发，大多数神经元在非线性激活函数后未被激活，我们提出了一种高效的隐私推理系统 Comet。该系统采用准确且快速的预测器来预测激活函数输出的稀疏性分布。此外，我们引入了一种新的隐私推理协议。该协议通过利用预测稀疏分布的空间局部性高效且安全地避免涉及零值的计算。尽管这种计算避免方法影响 KV 缓存条目的时空间连续性，我们通过一种低通信开销的缓存补充策略解决了这一挑战，该策略合并了缺失请求并结合了一个预取机制。最后，我们在四种常见的大语言模型上评估了 Comet，并将其与六种最先进的隐私推理系统进行了比较。Comet 实现了 1.87-2.63 倍的加速和 1.94-2.64 倍的通信量减少。 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Title (ZH)**: UAV-CodeAgents: 通过多智能体ReAct和视觉-语言推理实现可扩展的无人机任务规划 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

**Abstract (ZH)**: UAV-CodeAgents：基于大型语言和多模态模型的可扩展多Agent自主无人机任务生成框架 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Title (ZH)**: 动态RAG：将大规模语言模型的输出作为反馈用于检索增强生成的动态重排名 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）系统结合了大规模语言模型（LLMs）与外部知识检索，使其在知识密集型任务中表现出色。这些系统中的关键但常常被忽视的组件是再排序器，它通过对检索到的文档进行细化来提升生成质量和可解释性。如何选择最优的文档数量（k）这一挑战仍未解决：数量过少可能导致重要信息缺失，而数量过多则会引入噪音和低效。尽管近期研究已经探索了基于LLM的再排序器，但这些研究主要利用了模型内部的知识，而忽视了LLM能够提供的丰富监督信号，如使用响应质量作为反馈来优化再排序决策。在本文中，我们提出了DynamicRAG，这是一种新颖的RAG框架，其中再排序器能够根据查询动态调整检索到的文档的数量和顺序。我们通过强化学习（RL）将再排序器建模为一个优化代理，并使用来源于LLM输出质量的奖励。在七个知识密集型数据集上，DynamicRAG展示了优越的性能，达到了最先进的成果。模型、数据和代码可在以下链接获取。 

---
# Towards user-centered interactive medical image segmentation in VR with an assistive AI agent 

**Title (ZH)**: 面向用户的交互式医学图像分割在VR中的辅助AI代理助手 

**Authors**: Pascal Spiegler, Arash Harirpoush, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07214)  

**Abstract**: Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user-feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent that assists users with localizing, segmenting, and visualizing 3D medical concepts in VR. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks. 

**Abstract (ZH)**: 基于最新放射学AI基础模型和虚拟现实的交互辅助医学影像手动分割与可视化方法：SAMIRA 

---
# Internet of Agents: Fundamentals, Applications, and Challenges 

**Title (ZH)**: 代理互联网：基础、应用与挑战 

**Authors**: Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2505.07176)  

**Abstract**: With the rapid proliferation of large language models and vision-language models, AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the Internet of Agents (IoA) as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems. 

**Abstract (ZH)**: 随着大规模语言模型和视觉语言模型的迅速 proliferation，AI 代理从孤立的任务特定系统演变为无需人类干预即可感知、推理和行动的自主交互实体。随着这些代理在虚拟和物理环境中普及，从虚拟助手到具身机器人，构建统一的以代理为中心的基础架构变得至关重要。在本文综述中，我们介绍了代理互联网（IoA）作为一种基础框架，使大规模异构代理能够实现无缝互联、动态发现和协作编排。我们首先概述了通用的 IoA 架构，强调其分层组织、相对于传统互联网的独特特性以及新兴应用。接着，我们分析了 IoA 的关键操作使能技术，包括能力通知和发现、自适应通信协议、动态任务匹配、共识和冲突解决机制以及激励模型。最后，我们确定了朝着构建稳健且可信赖的 IoA 生态系统的研究方向。 

---
# Towards Scalable IoT Deployment for Visual Anomaly Detection via Efficient Compression 

**Title (ZH)**: 面向视觉异常检测的高效压缩驱动可扩展物联网部署 

**Authors**: Arianna Stropeni, Francesco Borsatti, Manuel Barusco, Davide Dalle Pezze, Marco Fabris, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07119)  

**Abstract**: Visual Anomaly Detection (VAD) is a key task in industrial settings, where minimizing waste and operational costs is essential. Deploying deep learning models within Internet of Things (IoT) environments introduces specific challenges due to the limited computational power and bandwidth of edge devices. This study investigates how to perform VAD effectively under such constraints by leveraging compact and efficient processing strategies. We evaluate several data compression techniques, examining the trade-off between system latency and detection accuracy. Experiments on the MVTec AD benchmark demonstrate that significant compression can be achieved with minimal loss in anomaly detection performance compared to uncompressed data. 

**Abstract (ZH)**: 视觉异常检测（VAD）是工业环境中的一项关键任务，其中减少浪费和运营成本至关重要。在物联网（IoT）环境中部署深度学习模型由于边缘设备的计算能力和带宽有限而引入了特定的挑战。本研究探讨如何在这种约束条件下有效进行VAD，通过利用紧凑且高效的处理策略。我们评估了几种数据压缩技术，研究了系统延迟和检测准确性之间的权衡。在MVTec AD基准测试上的实验表明，与未压缩数据相比，可以实现显著的压缩且异常检测性能的下降可以忽略不计。 

---
# X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real 

**Title (ZH)**: X-Sim：通过实境到仿真再到实境的跨体态学习 

**Authors**: Prithwish Dan, Kushal Kedia, Angela Chao, Edward Weiyi Duan, Maximus Adrian Pace, Wei-Chiu Ma, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.07096)  

**Abstract**: Human videos offer a scalable way to train robot manipulation policies, but lack the action labels needed by standard imitation learning algorithms. Existing cross-embodiment approaches try to map human motion to robot actions, but often fail when the embodiments differ significantly. We propose X-Sim, a real-to-sim-to-real framework that uses object motion as a dense and transferable signal for learning robot policies. X-Sim starts by reconstructing a photorealistic simulation from an RGBD human video and tracking object trajectories to define object-centric rewards. These rewards are used to train a reinforcement learning (RL) policy in simulation. The learned policy is then distilled into an image-conditioned diffusion policy using synthetic rollouts rendered with varied viewpoints and lighting. To transfer to the real world, X-Si introduces an online domain adaptation technique that aligns real and simulated observations during deployment. Importantly, X-Sim does not require any robot teleoperation data. We evaluate it across 5 manipulation tasks in 2 environments and show that it: (1) improves task progress by 30% on average over hand-tracking and sim-to-real baselines, (2) matches behavior cloning with 10x less data collection time, and (3) generalizes to new camera viewpoints and test-time changes. Code and videos are available at this https URL. 

**Abstract (ZH)**: 人类视频为机器人操作策略训练提供了可扩展的方法，但缺乏标准imitation learning算法所需的动作标签。现有的跨体态方法尝试将人类动作映射到机器人动作，但在体态差异显著时往往失败。我们提出X-Sim，一种从真实到模拟再到真实的世界框架，使用物体运动作为密集且可转移的信号来学习机器人策略。X-Sim 首先从RGBD人类视频中重构逼真的模拟，并跟踪物体轨迹以定义以物体为中心的奖励。这些奖励用于在模拟中训练强化学习（RL）策略。学习到的策略随后通过多样视角和光照渲染合成轨迹提炼为条件图像扩散策略。为了转移到真实世界，X-Si 引入了一种在线领域适应技术，在部署过程中对真实和模拟观察进行对齐。重要的是，X-Sim 不需要任何机器人远程操作数据。我们在两个环境中对5个操作任务进行了评估，并展示了它：(1) 平均在手部追踪和sim-to-real基线方法上提高任务进度30%，(2) 用1/10的数据收集时间匹配行为克隆，(3) 能够适应新的相机视角和测试时的变化。代码和视频可在以下链接获取。 

---
# Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? 

**Title (ZH)**: 基于LLM的金融投资策略能否在长期内超越市场？ 

**Authors**: Weixian Waylon Li, Hyeonjun Kim, Mihai Cucuringu, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07078)  

**Abstract**: Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity. 

**Abstract (ZH)**: 大规模语言模型（LLMs） recently has been利用于资产定价任务和股票交易应用，使AI代理能够从非结构化的财务数据中生成投资决策。然而，大多数基于时间的大规模语言模型投资策略评估都在较窄的时间框架和有限的股票 universes 中进行，这夸大了其效果，原因在于生存偏差和数据淘金偏见。我们通过提出FINSABER回测框架，系统地评估这些策略在更长时期和更大股票 universes 中的一般化能力和稳健性。二十年和100多种股票的系统回测揭示，之前报告的大规模语言模型的优势在更广泛的横截面和更长时间的评估下大幅减弱。进一步的市场环境分析表明，大规模语言模型策略在牛市中过于保守，表现不及被动基准，在熊市中则过于激进，造成重大损失。这些发现强调了开发能够优先考虑趋势检测和环境感知风险控制的大规模语言模型策略的重要性，而不仅仅是提升框架复杂性。 

---
# ParaView-MCP: An Autonomous Visualization Agent with Direct Tool Use 

**Title (ZH)**: ParaView-MCP：一个具备直接工具使用的自主可视化代理 

**Authors**: Shusen Liu, Haichao Miao, Peer-Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07064)  

**Abstract**: While powerful and well-established, tools like ParaView present a steep learning curve that discourages many potential users. This work introduces ParaView-MCP, an autonomous agent that integrates modern multimodal large language models (MLLMs) with ParaView to not only lower the barrier to entry but also augment ParaView with intelligent decision support. By leveraging the state-of-the-art reasoning, command execution, and vision capabilities of MLLMs, ParaView-MCP enables users to interact with ParaView through natural language and visual inputs. Specifically, our system adopted the Model Context Protocol (MCP) - a standardized interface for model-application communication - that facilitates direct interaction between MLLMs with ParaView's Python API to allow seamless information exchange between the user, the language model, and the visualization tool itself. Furthermore, by implementing a visual feedback mechanism that allows the agent to observe the viewport, we unlock a range of new capabilities, including recreating visualizations from examples, closed-loop visualization parameter updates based on user-defined goals, and even cross-application collaboration involving multiple tools. Broadly, we believe such an agent-driven visualization paradigm can profoundly change the way we interact with visualization tools. We expect a significant uptake in the development of such visualization tools, in both visualization research and industry. 

**Abstract (ZH)**: ParaView-MCP：基于现代多模态大型语言模型的自主代理增强可视化工具 

---
# Seed1.5-VL Technical Report 

**Title (ZH)**: Seed1.5-VL 技术报告 

**Authors**: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07062)  

**Abstract**: We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible at this https URL (Volcano Engine Model ID: doubao-1-5-thinking-vision-pro-250428) 

**Abstract (ZH)**: 我们介绍Seed1.5-VL，一个设计用于推进通用多模态理解与推理的视觉-语言基础模型。Seed1.5-VL由一个包含532M参数的视觉编码器和一个具有20B激活参数的专家混排（MoE）大型语言模型组成。尽管其架构相对紧凑，但在广泛公共VLM基准测试和内部评估套件中均展示了卓越的性能，共在60个公共基准中的38个上达到最佳性能。此外，在以代理为中心的任务，如GUI控制和游戏玩法中，Seed1.5-VL也优于包括OpenAI CUA和Claude 3.7在内的其他多模态系统。超越视觉和视频理解，它还展示了强大的推理能力，对于多模态推理挑战如视觉谜题尤其有效。我们相信这些能力将促进跨多种任务的应用。在本报告中，我们主要提供在模型设计、数据构建和各个阶段训练过程中的全面经验回顾，希望这份报告能够激励进一步的研究。Seed1.5-VL现在可通过以下链接访问：https://volcanoengine.com/model/doubao-1-5-thinking-vision-pro-250428。 

---
# Empirical Analysis of Asynchronous Federated Learning on Heterogeneous Devices: Efficiency, Fairness, and Privacy Trade-offs 

**Title (ZH)**: 异步步联邦学习在异构设备上的实证分析：效率、公平性和隐私权权衡 

**Authors**: Samaneh Mohammadi, Iraklis Symeonidis, Ali Balador, Francesco Flammini  

**Link**: [PDF](https://arxiv.org/pdf/2505.07041)  

**Abstract**: Device heterogeneity poses major challenges in Federated Learning (FL), where resource-constrained clients slow down synchronous schemes that wait for all updates before aggregation. Asynchronous FL addresses this by incorporating updates as they arrive, substantially improving efficiency. While its efficiency gains are well recognized, its privacy costs remain largely unexplored, particularly for high-end devices that contribute updates more frequently, increasing their cumulative privacy exposure. This paper presents the first comprehensive analysis of the efficiency-fairness-privacy trade-off in synchronous vs. asynchronous FL under realistic device heterogeneity. We empirically compare FedAvg and staleness-aware FedAsync using a physical testbed of five edge devices spanning diverse hardware tiers, integrating Local Differential Privacy (LDP) and the Moments Accountant to quantify per-client privacy loss. Using Speech Emotion Recognition (SER) as a privacy-critical benchmark, we show that FedAsync achieves up to 10x faster convergence but exacerbates fairness and privacy disparities: high-end devices contribute 6-10x more updates and incur up to 5x higher privacy loss, while low-end devices suffer amplified accuracy degradation due to infrequent, stale, and noise-perturbed updates. These findings motivate the need for adaptive FL protocols that jointly optimize aggregation and privacy mechanisms based on client capacity and participation dynamics, moving beyond static, one-size-fits-all solutions. 

**Abstract (ZH)**: 设备异质性给联邦学习（FL）带来了重大挑战，资源受限的客户端会减慢需要等待所有更新后再进行聚合的同步方案。异步FL通过在接收到更新时即刻整合更新，显著提高了效率。尽管其效率提升被广泛认可，但其隐私成本尚未得到充分探索，特别是在高端设备贡献更新频率更高，增加其累计隐私暴露的情况下。本文首次在实际设备异质性条件下，全面分析了同步与异步FL在效率-公平-隐私之间的权衡。我们使用包含不同硬件级别的五台边缘设备进行物理测试床比较FedAvg和 staleness-aware FedAsync，结合局部差分隐私（LDP）和矩账户方法量化客户端的隐私损失。使用语音情感识别（SER）作为隐私敏感基准，研究显示， FedAsync可实现高达10倍的更快收敛，但加剧了公平性和隐私差距：高端设备贡献6-10倍更多更新，并可能遭受高达5倍的更高隐私损失，而低端设备则因更新不频繁、过时和噪声扰动更新而加剧准确率下降。这些发现促使我们开发基于客户端能力和参与动态联合优化聚合与隐私机制的自适应FL协议，超越静态的一刀切解决方案。 

---
# Predicting Diabetes Using Machine Learning: A Comparative Study of Classifiers 

**Title (ZH)**: 使用机器学习预测糖尿病：分类器的比较研究 

**Authors**: Mahade Hasan, Farhana Yasmin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07036)  

**Abstract**: Diabetes remains a significant health challenge globally, contributing to severe complications like kidney disease, vision loss, and heart issues. The application of machine learning (ML) in healthcare enables efficient and accurate disease prediction, offering avenues for early intervention and patient support. Our study introduces an innovative diabetes prediction framework, leveraging both traditional ML techniques such as Logistic Regression, SVM, Naïve Bayes, and Random Forest and advanced ensemble methods like AdaBoost, Gradient Boosting, Extra Trees, and XGBoost. Central to our approach is the development of a novel model, DNet, a hybrid architecture combining Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers for effective feature extraction and sequential learning. The DNet model comprises an initial convolutional block for capturing essential features, followed by a residual block with skip connections to facilitate efficient information flow. Batch Normalization and Dropout are employed for robust regularization, and an LSTM layer captures temporal dependencies within the data. Using a Kaggle-sourced real-world diabetes dataset, our model evaluation spans cross-validation accuracy, precision, recall, F1 score, and ROC-AUC. Among the models, DNet demonstrates the highest efficacy with an accuracy of 99.79% and an AUC-ROC of 99.98%, establishing its potential for superior diabetes prediction. This robust hybrid architecture showcases the value of combining CNN and LSTM layers, emphasizing its applicability in medical diagnostics and disease prediction tasks. 

**Abstract (ZH)**: 糖尿病仍然是全球性的健康挑战，导致严重的并发症如肾病、视力丧失和心脏问题。机器学习（ML）在医疗领域的应用能够实现高效的疾病预测，为早期干预和患者支持提供途径。本研究引入了一种创新的糖尿病预测框架，结合了传统的机器学习技术如逻辑回归、支持向量机、朴素贝叶斯和随机森林，以及先进的集成方法如AdaBoost、梯度提升、极端随机森林和XGBoost。本方法的核心在于开发了一种新型模型DNet，这是一种结合卷积神经网络（CNN）和长短期记忆（LSTM）层的混合架构，用于有效的特征提取和序列学习。DNet模型包括一个初始的卷积块以捕获关键特征，随后是一个具有跳跃连接的残差块，以促进高效的信息流动。批量标准化和 dropout 用于实现稳健的正则化，LSTM 层用于捕捉数据中的时间依赖性。使用Kaggle提供的真实世界糖尿病数据集，我们的模型评估涵盖了交叉验证精度、精确度、召回率、F1分数和ROC-AUC。在各种模型中，DNet展示了最高的有效性，精度为99.79%，AUC-ROC为99.98%，证明了其在糖尿病预测中的潜在优势。这种稳健的混合架构展示了结合CNN和LSTM层的价值，强调了其在医疗诊断和疾病预测任务中的适用性。 

---
# Incremental Uncertainty-aware Performance Monitoring with Active Labeling Intervention 

**Title (ZH)**: 增量不确定性意识性能监控与主动标签干预 

**Authors**: Alexander Koebler, Thomas Decker, Ingo Thon, Volker Tresp, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07023)  

**Abstract**: We study the problem of monitoring machine learning models under gradual distribution shifts, where circumstances change slowly over time, often leading to unnoticed yet significant declines in accuracy. To address this, we propose Incremental Uncertainty-aware Performance Monitoring (IUPM), a novel label-free method that estimates performance changes by modeling gradual shifts using optimal transport. In addition, IUPM quantifies the uncertainty in the performance prediction and introduces an active labeling procedure to restore a reliable estimate under a limited labeling budget. Our experiments show that IUPM outperforms existing performance estimation baselines in various gradual shift scenarios and that its uncertainty awareness guides label acquisition more effectively compared to other strategies. 

**Abstract (ZH)**: 我们在渐进分布偏移下研究机器学习模型监控问题，其中环境随时间缓慢变化，常常导致未被察觉但显著的准确性下降。为解决这一问题，我们提出了一种新颖的无需标签性能监控方法 Incremental Uncertainty-aware Performance Monitoring (IUPM)，该方法通过最优运输模型化渐进变化来估计性能变化。此外，IUPM量化的性能预测不确定性并且引入主动标记程序，在有限的标记预算下恢复可靠的估计。我们的实验表明，IUPM在各种渐进偏移场景中优于现有性能估计基准，并且其不确定性意识比其他策略更有效地指导标记获取。 

---
# R-CAGE: A Structural Model for Emotion Output Design in Human-AI Interaction 

**Title (ZH)**: R-CAGE: 人类-人工智能交互中情感输出设计的结构模型 

**Authors**: Suyeon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07020)  

**Abstract**: This paper presents R-CAGE (Rhythmic Control Architecture for Guarding Ego), a theoretical framework for restructuring emotional output in long-term human-AI interaction. While prior affective computing approaches emphasized expressiveness, immersion, and responsiveness, they often neglected the cognitive and structural consequences of repeated emotional engagement. R-CAGE instead conceptualizes emotional output not as reactive expression but as ethical design structure requiring architectural intervention. The model is grounded in experiential observations of subtle affective symptoms such as localized head tension, interpretive fixation, and emotional lag arising from prolonged interaction with affective AI systems. These indicate a mismatch between system-driven emotion and user interpretation that cannot be fully explained by biometric data or observable behavior. R-CAGE adopts a user-centered stance prioritizing psychological recovery, interpretive autonomy, and identity continuity. The framework consists of four control blocks: (1) Control of Rhythmic Expression regulates output pacing to reduce fatigue; (2) Architecture of Sensory Structuring adjusts intensity and timing of affective stimuli; (3) Guarding of Cognitive Framing reduces semantic pressure to allow flexible interpretation; (4) Ego-Aligned Response Design supports self-reference recovery during interpretive lag. By structurally regulating emotional rhythm, sensory intensity, and interpretive affordances, R-CAGE frames emotion not as performative output but as sustainable design unit. The goal is to protect users from oversaturation and cognitive overload while sustaining long-term interpretive agency in AI-mediated environments. 

**Abstract (ZH)**: R-CAGE：护 ego 节律控制架构 

---
# Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization 

**Title (ZH)**: 远程生理传感中基于目标信号约束因子分解的高效稳健多维注意力机制 

**Authors**: Jitesh Joshi, Youngjun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.07013)  

**Abstract**: Remote physiological sensing using camera-based technologies offers transformative potential for non-invasive vital sign monitoring across healthcare and human-computer interaction domains. Although deep learning approaches have advanced the extraction of physiological signals from video data, existing methods have not been sufficiently assessed for their robustness to domain shifts. These shifts in remote physiological sensing include variations in ambient conditions, camera specifications, head movements, facial poses, and physiological states which often impact real-world performance significantly. Cross-dataset evaluation provides an objective measure to assess generalization capabilities across these domain shifts. We introduce Target Signal Constrained Factorization module (TSFM), a novel multidimensional attention mechanism that explicitly incorporates physiological signal characteristics as factorization constraints, allowing more precise feature extraction. Building on this innovation, we present MMRPhys, an efficient dual-branch 3D-CNN architecture designed for simultaneous multitask estimation of photoplethysmography (rPPG) and respiratory (rRSP) signals from multimodal RGB and thermal video inputs. Through comprehensive cross-dataset evaluation on five benchmark datasets, we demonstrate that MMRPhys with TSFM significantly outperforms state-of-the-art methods in generalization across domain shifts for rPPG and rRSP estimation, while maintaining a minimal inference latency suitable for real-time applications. Our approach establishes new benchmarks for robust multitask and multimodal physiological sensing and offers a computationally efficient framework for practical deployment in unconstrained environments. The web browser-based application featuring on-device real-time inference of MMRPhys model is available at this https URL 

**Abstract (ZH)**: 基于摄像头的远程生理传感技术在医疗保健和人机交互领域提供了非侵入性生命体征监测的变革潜力。尽管深度学习方法在从视频数据中提取生理信号方面取得了进展，但现有方法在面对域转移时的稳健性评估尚不满意。远程生理传感中的这些域转移包括环境条件变化、摄像头规格差异、头部移动、面部姿态和生理状态的变化，这些因素往往在实际应用中显著影响性能。跨数据集评估提供了一种客观的手段来衡量在这些域转移下的泛化能力。我们引入了目标信号约束分解模块（TSFM），这是一种新型多维注意力机制，明确地将生理信号特征作为分解约束，从而实现更精确的特征提取。在此基础上，我们提出了MMRPhys，一种高效的双分支3D-CNN架构，旨在同时从多模态RGB和热视频输入中估计光体积描记图（rPPG）和呼吸（rRSP）信号。通过在五个基准数据集上的全面跨数据集评估，我们证明了带有TSFM的MMRPhys在rPPG和rRSP估计的域转移泛化能力上显著优于现有方法，同时保持了适合实时应用的最小推理延迟。我们的方法确立了鲁棒多任务和多模态生理传感的新基准，并提供了一种在不受约束环境中进行现实部署的高效计算框架。基于Web浏览器的应用程序可以在以下链接中实现MMRPhys模型的设备上实时推理功能：[链接]。 

---
# Hand-Shadow Poser 

**Title (ZH)**: 手影Pose生成器 

**Authors**: Hao Xu, Yinqiao Wang, Niloy J. Mitra, Shuaicheng Liu, Pheng-Ann Heng, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07012)  

**Abstract**: Hand shadow art is a captivating art form, creatively using hand shadows to reproduce expressive shapes on the wall. In this work, we study an inverse problem: given a target shape, find the poses of left and right hands that together best produce a shadow resembling the input. This problem is nontrivial, since the design space of 3D hand poses is huge while being restrictive due to anatomical constraints. Also, we need to attend to the input's shape and crucial features, though the input is colorless and textureless. To meet these challenges, we design Hand-Shadow Poser, a three-stage pipeline, to decouple the anatomical constraints (by hand) and semantic constraints (by shadow shape): (i) a generative hand assignment module to explore diverse but reasonable left/right-hand shape hypotheses; (ii) a generalized hand-shadow alignment module to infer coarse hand poses with a similarity-driven strategy for selecting hypotheses; and (iii) a shadow-feature-aware refinement module to optimize the hand poses for physical plausibility and shadow feature preservation. Further, we design our pipeline to be trainable on generic public hand data, thus avoiding the need for any specialized training dataset. For method validation, we build a benchmark of 210 diverse shadow shapes of varying complexity and a comprehensive set of metrics, including a novel DINOv2-based evaluation metric. Through extensive comparisons with multiple baselines and user studies, our approach is demonstrated to effectively generate bimanual hand poses for a large variety of hand shapes for over 85% of the benchmark cases. 

**Abstract (ZH)**: 手影艺术是一种引人入胜的艺术形式，创造性地使用手影在墙上重现有表现力的形状。在本文中，我们研究了一个逆问题：给定一个目标形状，找出左、右手的最佳姿态，使它们共同产生与输入相似的阴影。这个问题非同 trivial，因为3D 手部姿态的设计空间巨大，但受到解剖学限制而受到限制。此外，我们需要关注输入的形状和关键特征，尽管输入是无色且无纹理的。为了解决这些挑战，我们设计了手影姿态生成器（Hand-Shadow Poser），这是一个三阶段流水线，以解耦解剖学约束（由手处理）和语义约束（由阴影形状处理）：（i）生成性手部分配模块，探索多样的但合理的左手/右手形状假设；（ii）通用手部-手影对齐模块，通过相似性驱动的选择策略推断粗略的手部姿态；（iii）基于手影特征的细化模块，优化手部姿态以保持物理合理性并保留手影特征。此外，我们设计了该流水线可以使用通用的公开手部数据进行训练，从而避免使用任何专门的训练数据集。为了方法验证，我们构建了一个包含210个不同复杂度的手影形状的数据集，并提供了一套全面的指标，包括一个新的基于DINOv2的评估指标。通过与多个基线方法和用户研究的广泛比较，我们的方法证明可以在超过85%的基准案例中有效地生成多样手形的双手姿势。 

---
# Towards the Three-Phase Dynamics of Generalization Power of a DNN 

**Title (ZH)**: 探索DNN泛化能力的三相动态 

**Authors**: Yuxuan He, Junpeng Zhang, Hongyuan Zhang, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06993)  

**Abstract**: This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses. 

**Abstract (ZH)**: 本文提出了一种分析深度神经网络（DNN）泛化能力的新视角，即直接拆分和分析DNN在训练过程中编码的可泛化和不可泛化的相互作用的动力学。具体而言，本文基于近期可解释人工智能领域的理论成就，该成就证明了DNN的详细推理逻辑可以严格地重写为少数几种AND-OR相互作用模式。基于此，我们提出了一种有效的方法来量化每个相互作用的泛化能力，并发现相互作用在训练过程中具有独特的三阶段动态。特别是，在训练的早期阶段通常会去除噪声和不可泛化的相互作用，并学习简单的和可泛化的相互作用。而在第二和第三阶段，倾向于捕捉越来越复杂且更难泛化的相互作用。实验结果验证了非可泛化相互作用的习得是训练损失与测试损失之间差距的直接原因。 

---
# Convert Language Model into a Value-based Strategic Planner 

**Title (ZH)**: 将语言模型转换为价值为基础的战略规划器 

**Authors**: Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Xiaokai Chen, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.06987)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines. 

**Abstract (ZH)**: 情绪支持对话（ESC）旨在通过有效的对话缓解个体的情绪困扰。尽管大型语言模型（LLMs）在ESC方面取得了显著进展，但大多数研究可能并未从状态模型的角度定义该过程，因此可能无法提供长期满意的解决方案。为解决这一问题，我们利用Q-learning技术，提出了一个名为straQ*的框架。该框架允许插拔式的LLMs在ESC过程中进行规划，基于长期回报确定最优策略，并最终指导LLM进行响应。在ESC数据集上的大量实验表明，straQ*在多种基线方法（包括直接推理、自我完善、逻辑推理、微调和有限状态机）中表现更优。 

---
# Reinforcement Learning-Based Monocular Vision Approach for Autonomous UAV Landing 

**Title (ZH)**: 基于强化学习的单目视觉自主无人机着陆方法 

**Authors**: Tarik Houichime, Younes EL Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06963)  

**Abstract**: This paper introduces an innovative approach for the autonomous landing of Unmanned Aerial Vehicles (UAVs) using only a front-facing monocular camera, therefore obviating the requirement for depth estimation cameras. Drawing on the inherent human estimating process, the proposed method reframes the landing task as an optimization problem. The UAV employs variations in the visual characteristics of a specially designed lenticular circle on the landing pad, where the perceived color and form provide critical information for estimating both altitude and depth. Reinforcement learning algorithms are utilized to approximate the functions governing these estimations, enabling the UAV to ascertain ideal landing settings via training. This method's efficacy is assessed by simulations and experiments, showcasing its potential for robust and accurate autonomous landing without dependence on complex sensor setups. This research contributes to the advancement of cost-effective and efficient UAV landing solutions, paving the way for wider applicability across various fields. 

**Abstract (ZH)**: 本文提出了一种仅使用前置单目摄像头进行自主降落的无人机创新方法，从而省去了深度估计摄像头的需要。该方法借鉴了人类固有的估算过程，将降落任务重新定义为一个优化问题。无人机利用着陆垫上特制透镜圆环的视觉特征的变化，通过感知颜色和形态来估算高度和深度。利用强化学习算法近似这些估算函数，使无人机通过训练确定理想的着陆设置。该方法的有效性通过模拟和实验进行了评估，展示了其在无需复杂传感器配置的情况下实现稳健准确自主降落的潜力。该研究为低成本高效的无人机降落解决方案的发展做出了贡献，为在各种领域的广泛应用铺平了道路。 

---
# AI-Powered Inverse Design of Ku-Band SIW Resonant Structures by Iterative Residual Correction Network 

**Title (ZH)**: 基于迭代残差校正网络的AI驱动Ku波段SIW谐振结构逆设计 

**Authors**: Mohammad Mashayekhi, Kamran Salehian  

**Link**: [PDF](https://arxiv.org/pdf/2505.06936)  

**Abstract**: Inverse electromagnetic modeling has emerged as a powerful approach for designing complex microwave structures with high accuracy and efficiency. In this study, we propose an Iterative Residual Correction Network (IRC-Net) for the inverse design of Ku-band Substrate Integrated Waveguide (SIW) components based on multimode resonators. We use a multimode resonance structure to demonstrate that it is possible to control the resonances of the structure. Therefore, these structures can be used for resonant components and smart filter design. The proposed deep learning architecture leverages residual neural networks to overcome the limitations of traditional inverse design techniques, such as the Feedforward Inverse Model (FIM), offering improved generalization and prediction accuracy. The approach begins with a FIM to generate initial design estimates, followed by an iterative correction strategy inspired by the Hybrid Inverse-Forward Residual Refinement Network (HiFR\textsuperscript{2}-Net), which we call IRC-Net. Experiments demonstrate that the IRC-Net achieves substantial improvements in prediction accuracy compared to traditional single-stage networks, validated through statistical metrics, full-wave electromagnetic simulations, and measurements. To validate the proposed framework, we first design and fabricate a three-resonance SIW structure. Next, we apply the trained IRC-Net model to predict the geometry of a four-resonance structure based on its desired frequency response. Both designs are fabricated and tested, showing strong agreement between the simulated, predicted, and measured results, confirming the effectiveness and practicality of the proposed method. 

**Abstract (ZH)**: 基于多模谐振器的Ku波段集成波导组件的逆向设计的迭代残差校正网络 

---
# RedTeamLLM: an Agentic AI framework for offensive security 

**Title (ZH)**: RedTeamLLM: 一种用于进攻性安全的自主AI框架 

**Authors**: Brian Challita, Pierre Parrend  

**Link**: [PDF](https://arxiv.org/pdf/2505.06913)  

**Abstract**: From automated intrusion testing to discovery of zero-day attacks before software launch, agentic AI calls for great promises in security engineering. This strong capability is bound with a similar threat: the security and research community must build up its models before the approach is leveraged by malicious actors for cybercrime. We therefore propose and evaluate RedTeamLLM, an integrated architecture with a comprehensive security model for automatization of pentest tasks. RedTeamLLM follows three key steps: summarizing, reasoning and act, which embed its operational capacity. This novel framework addresses four open challenges: plan correction, memory management, context window constraint, and generality vs. specialization. Evaluation is performed through the automated resolution of a range of entry-level, but not trivial, CTF challenges. The contribution of the reasoning capability of our agentic AI framework is specifically evaluated. 

**Abstract (ZH)**: 从自动化入侵测试到软件发布前发现零日攻击，自主人工智能在安全工程领域寄予厚望。这一强大能力伴随着相似的威胁：安全与研究领域必须在该方法被恶意行为者用于网络犯罪之前建立健全其模型。因此，我们提出并评估了RedTeamLLM，这是一种集成架构，具有全面的安全模型以自动化渗透测试任务。RedTeamLLM 遵循三个关键步骤：总结、推理和执行，这嵌入了其操作能力。该新颖框架解决了四个开放挑战：计划修正、内存管理、上下文窗口约束以及通用性与专门化之间的权衡。通过自动化解决一系列入门级但不简单的CTF挑战来对其实验评估。特别评估了我们自主人工智能框架的推理能力贡献。 

---
# MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning 

**Title (ZH)**: MMiC: 缓解集群联邦学习中的模态不完整性 

**Authors**: Lishan Yang, Wei Zhang, Quan Z. Sheng, Weitong Chen, Lina Yao, Weitong Chen, Ali Shakeri  

**Link**: [PDF](https://arxiv.org/pdf/2505.06911)  

**Abstract**: In the era of big data, data mining has become indispensable for uncovering hidden patterns and insights from vast and complex datasets. The integration of multimodal data sources further enhances its potential. Multimodal Federated Learning (MFL) is a distributed approach that enhances the efficiency and quality of multimodal learning, ensuring collaborative work and privacy protection. However, missing modalities pose a significant challenge in MFL, often due to data quality issues or privacy policies across the clients. In this work, we present MMiC, a framework for Mitigating Modality incompleteness in MFL within the Clusters. MMiC replaces partial parameters within client models inside clusters to mitigate the impact of missing modalities. Furthermore, it leverages the Banzhaf Power Index to optimize client selection under these conditions. Finally, MMiC employs an innovative approach to dynamically control global aggregation by utilizing Markovitz Portfolio Optimization. Extensive experiments demonstrate that MMiC consistently outperforms existing federated learning architectures in both global and personalized performance on multimodal datasets with missing modalities, confirming the effectiveness of our proposed solution. 

**Abstract (ZH)**: 在大数据时代，数据挖掘已成为从庞大而复杂的多模态数据集中发现隐藏模式和洞察力不可或缺的工具。多模态联邦学习（MFL）是一种分布式方法，它提高了多模态学习的效率和质量，同时确保协作工作和隐私保护。然而，缺失的模态在MFL中构成了重大挑战，通常是由于客户端的数据质量问题或隐私政策所致。在本文中，我们提出了MMiC框架，用于在簇内缓解多模态数据缺失性问题。MMiC通过在客户端模型中替换部分参数来减轻缺失模态的影响。此外，它利用Banzhaf权力指数优化在这些条件下选择客户端。最后，MMiC采用了利用马克维兹资产组合优化的创新方法，动态控制全局聚合。 extensive实验证明，MMiC在存在缺失模态的多模态数据集上的一般性和个性化性能上均优于现有的联邦学习架构，证实了我们所提出的解决方案的有效性。 

---
# NeuGen: Amplifying the 'Neural' in Neural Radiance Fields for Domain Generalization 

**Title (ZH)**: NeuGen: 强化神经辐射场中的“神经”元素以实现领域泛化 

**Authors**: Ahmed Qazi, Abdul Basit, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06894)  

**Abstract**: Neural Radiance Fields (NeRF) have significantly advanced the field of novel view synthesis, yet their generalization across diverse scenes and conditions remains challenging. Addressing this, we propose the integration of a novel brain-inspired normalization technique Neural Generalization (NeuGen) into leading NeRF architectures which include MVSNeRF and GeoNeRF. NeuGen extracts the domain-invariant features, thereby enhancing the models' generalization capabilities. It can be seamlessly integrated into NeRF architectures and cultivates a comprehensive feature set that significantly improves accuracy and robustness in image rendering. Through this integration, NeuGen shows improved performance on benchmarks on diverse datasets across state-of-the-art NeRF architectures, enabling them to generalize better across varied scenes. Our comprehensive evaluations, both quantitative and qualitative, confirm that our approach not only surpasses existing models in generalizability but also markedly improves rendering quality. Our work exemplifies the potential of merging neuroscientific principles with deep learning frameworks, setting a new precedent for enhanced generalizability and efficiency in novel view synthesis. A demo of our study is available at this https URL. 

**Abstract (ZH)**: 基于神经辐射场的新型视图合成中，神经泛化（NeuGen）的脑启发规范化技术在多样化场景和条件下的泛化能力提升 

---
# IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method 

**Title (ZH)**: IM-BERT：通过隐式欧拉方法增强BERT的鲁棒性 

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06889)  

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.
To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.
Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.
Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy. 

**Abstract (ZH)**: 预训练语言模型（PLMs）通过预训练和微调在多种NLP任务中取得了显著性能。然而，使用大量参数对下游数据集进行微调往往会导致模型对 adversarial 攻击的脆弱性，造成模型在标准数据集上的过拟合。
为解决这些问题，我们从动态系统的角度提出了 IM-BERT，将其一层 BERT 视作常微分方程（ODEs）的解。在初始值扰动的情况下，我们分析了两种主要的数值 ODE 解算器——显式和隐式欧拉方法的数值稳定性。
基于这些分析，我们引入了一种数值稳健的 IM 连接，该策略增强了 PLMs 对 adversarial 攻击的鲁棒性，即使在低资源场景下，也不会引入额外参数或 adversarial 训练策略。
在 adversarial GLUE（AdvGLUE）数据集上的实验结果验证了 IM-BERT 在各种条件下的鲁棒性。与原始 BERT 相比，IM-BERT 在 AdvGLUE 数据集上的性能提高了约 8.3%p。此外，在低资源场景中，IM-BERT 获得了 5.9%p 的更高准确率。 

---
# Mice to Machines: Neural Representations from Visual Cortex for Domain Generalization 

**Title (ZH)**: 从老鼠到机器：视觉皮层的神经表示在领域泛化中的应用 

**Authors**: Ahmed Qazi, Hamd Jalil, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06886)  

**Abstract**: The mouse is one of the most studied animal models in the field of systems neuroscience. Understanding the generalized patterns and decoding the neural representations that are evoked by the diverse range of natural scene stimuli in the mouse visual cortex is one of the key quests in computational vision. In recent years, significant parallels have been drawn between the primate visual cortex and hierarchical deep neural networks. However, their generalized efficacy in understanding mouse vision has been limited. In this study, we investigate the functional alignment between the mouse visual cortex and deep learning models for object classification tasks. We first introduce a generalized representational learning strategy that uncovers a striking resemblance between the functional mapping of the mouse visual cortex and high-performing deep learning models on both top-down (population-level) and bottom-up (single cell-level) scenarios. Next, this representational similarity across the two systems is further enhanced by the addition of Neural Response Normalization (NeuRN) layer, inspired by the activation profile of excitatory and inhibitory neurons in the visual cortex. To test the performance effect of NeuRN on real-world tasks, we integrate it into deep learning models and observe significant improvements in their robustness against data shifts in domain generalization tasks. Our work proposes a novel framework for comparing the functional architecture of the mouse visual cortex with deep learning models. Our findings carry broad implications for the development of advanced AI models that draw inspiration from the mouse visual cortex, suggesting that these models serve as valuable tools for studying the neural representations of the mouse visual cortex and, as a result, enhancing their performance on real-world tasks. 

**Abstract (ZH)**: 小鼠是系统神经科学领域中研究最多的小型动物模型之一。理解小鼠视觉皮层在面对各种自然 scenes 刺激时表现出的通用模式及其神经表征解码是计算视觉中的关键任务之一。近年来，猕猴视觉皮层与层次化深度神经网络之间的类比关系越来越多，然而这在解释小鼠视觉方面的作用有限。本研究探讨了小鼠视觉皮层与深度学习模型在物体分类任务中的功能对齐。我们首先介绍了一种通用的表征学习策略，揭示了小鼠视觉皮层的功能映射与高性能深度学习模型之间的显著相似性，这一发现适用于自上而下（群体层面）和自下而上（单细胞层面）两种情况。然后，通过引入神经响应归一化（NeuRN）层，进一步加强了两种系统之间的表征相似性，该层受视觉皮层兴奋性和抑制性神经元激活特征的启发。为了测试NeuRN在实际任务中的性能影响，我们将其整合到深度学习模型中，并观察到在跨域泛化任务中对其鲁棒性的显著提升。本研究提出了一个新型框架，用于比较小鼠视觉皮层的功能架构与深度学习模型的对比。我们的发现对基于小鼠视觉皮层构建高级人工智能模型的发展具有广泛的影响，表明这些模型是研究小鼠视觉皮层神经表征的重要工具，并且能够提升其在实际任务中的性能。 

---
# FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots 

**Title (ZH)**: FACET：基于阻抗参考跟踪的力自适应控制方法应用于腿式机器人 

**Authors**: Botian Xu, Haoyang Weng, Qingzhou Lu, Yang Gao, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06883)  

**Abstract**: Reinforcement learning (RL) has made significant strides in legged robot control, enabling locomotion across diverse terrains and complex loco-manipulation capabilities. However, the commonly used position or velocity tracking-based objectives are agnostic to forces experienced by the robot, leading to stiff and potentially dangerous behaviors and poor control during forceful interactions. To address this limitation, we present \emph{Force-Adaptive Control via Impedance Reference Tracking} (FACET). Inspired by impedance control, we use RL to train a control policy to imitate a virtual mass-spring-damper system, allowing fine-grained control under external forces by manipulating the virtual spring. In simulation, we demonstrate that our quadruped robot achieves improved robustness to large impulses (up to 200 Ns) and exhibits controllable compliance, achieving an 80% reduction in collision impulse. The policy is deployed to a physical robot to showcase both compliance and the ability to engage with large forces by kinesthetic control and pulling payloads up to 2/3 of its weight. Further extension to a legged loco-manipulator and a humanoid shows the applicability of our method to more complex settings to enable whole-body compliance control. Project Website: this https URL 

**Abstract (ZH)**: 基于阻抗参考跟踪的力自适应控制（Force-Adaptive Control via Impedance Reference Tracking） 

---
# NeuRN: Neuro-inspired Domain Generalization for Image Classification 

**Title (ZH)**: NeuRN: 基于神经启发的领域泛化图像分类 

**Authors**: Hamd Jalil, Ahmed Qazi, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06881)  

**Abstract**: Domain generalization in image classification is a crucial challenge, with models often failing to generalize well across unseen datasets. We address this issue by introducing a neuro-inspired Neural Response Normalization (NeuRN) layer which draws inspiration from neurons in the mammalian visual cortex, which aims to enhance the performance of deep learning architectures on unseen target domains by training deep learning models on a source domain. The performance of these models is considered as a baseline and then compared against models integrated with NeuRN on image classification tasks. We perform experiments across a range of deep learning architectures, including ones derived from Neural Architecture Search and Vision Transformer. Additionally, in order to shortlist models for our experiment from amongst the vast range of deep neural networks available which have shown promising results, we also propose a novel method that uses the Needleman-Wunsch algorithm to compute similarity between deep learning architectures. Our results demonstrate the effectiveness of NeuRN by showing improvement against baseline in cross-domain image classification tasks. Our framework attempts to establish a foundation for future neuro-inspired deep learning models. 

**Abstract (ZH)**: 基于神经元启发的神经响应归一化层在图像分类中的泛化能力研究 

---
# Enhancing Time Series Forecasting via a Parallel Hybridization of ARIMA and Polynomial Classifiers 

**Title (ZH)**: 基于ARIMA与多项式分类器并行混合的时序预测增强方法 

**Authors**: Thanh Son Nguyen, Van Thanh Nguyen, Dang Minh Duc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06874)  

**Abstract**: Time series forecasting has attracted significant attention, leading to the de-velopment of a wide range of approaches, from traditional statistical meth-ods to advanced deep learning models. Among them, the Auto-Regressive Integrated Moving Average (ARIMA) model remains a widely adopted linear technique due to its effectiveness in modeling temporal dependencies in economic, industrial, and social data. On the other hand, polynomial classifi-ers offer a robust framework for capturing non-linear relationships and have demonstrated competitive performance in domains such as stock price pre-diction. In this study, we propose a hybrid forecasting approach that inte-grates the ARIMA model with a polynomial classifier to leverage the com-plementary strengths of both models. The hybrid method is evaluated on multiple real-world time series datasets spanning diverse domains. Perfor-mance is assessed based on forecasting accuracy and computational effi-ciency. Experimental results reveal that the proposed hybrid model consist-ently outperforms the individual models in terms of prediction accuracy, al-beit with a modest increase in execution time. 

**Abstract (ZH)**: 时间序列预测吸引了广泛的关注，推动了从传统统计方法到先进深度学习模型的广泛应用。其中，自动回归积分移动平均（ARIMA）模型由于在建模经济、工业和社会数据的时间依赖性方面效果显著，仍然是广泛采用的线性技术之一。另一方面，多项式分类器提供了一种稳健的框架来捕获非线性关系，并在股票价格预测等领域展示了竞争力。在本研究中，我们提出了一种将ARIMA模型与多项式分类器相结合的混合预测方法，以发挥两种模型的互补优势。该混合方法在多个涵盖不同领域的实际时间序列数据集上进行了评估。性能评估基于预测准确性与计算效率。实验结果表明，所提出的混合模型在预测准确性方面始终优于单一模型，尽管执行时间略有增加。 

---
# Efficient Robotic Policy Learning via Latent Space Backward Planning 

**Title (ZH)**: 通过潜在空间逆向规划实现高效的机器人政策学习 

**Authors**: Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06861)  

**Abstract**: Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: this https URL 

**Abstract (ZH)**: 当前的机器人规划方法往往依赖于预测多帧具有全像素细节的图像。虽然这种精细的方法可以作为通用的世界模型，但对于下游策略学习来说，它引入了两个重大挑战：巨大的计算成本阻碍了实时部署，以及累积的不准确数据可能会误导动作提取。使用粗粒度的子目标进行规划部分缓解了效率问题。然而，其前瞻性的规划方案仍然可能由于累积误差导致离任务目标的预测，从而与长期目标产生偏差。这提出了一个关键问题：机器人规划能否在长时间多阶段任务中既高效又足够准确以实现实时控制？为了解决这个问题，我们提出了一种潜空间反向规划方案（LBP），该方案首先将任务细化为最终的潜目标，然后通过递归预测与当前状态更接近的中间子目标。最终目标的先验性使得反向子目标规划总是能够保持对任务完成的意识，促进在整个规划时段内的任务相关预测。子目标条件策略结合了一个可学习的标记来总结子目标序列，并决定每个子目标如何引导动作提取。通过大量的模拟实验和长时间的真实机器人实验，我们展示了LBP在性能上优于现有的精细和前瞻规划方法，达到了目前最佳的性能。项目页面: [this URL](this https URL) 

---
# DP-TRAE: A Dual-Phase Merging Transferable Reversible Adversarial Example for Image Privacy Protection 

**Title (ZH)**: DP-TRAE: 一种双阶段合并可移植可逆 adversarial 示例的图像隐私保护方法 

**Authors**: Xia Du, Jiajie Zhu, Jizhe Zhou, Chi-man Pun, Zheng Lin, Cong Wu, Zhe Chen, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06860)  

**Abstract**: In the field of digital security, Reversible Adversarial Examples (RAE) combine adversarial attacks with reversible data hiding techniques to effectively protect sensitive data and prevent unauthorized analysis by malicious Deep Neural Networks (DNNs). However, existing RAE techniques primarily focus on white-box attacks, lacking a comprehensive evaluation of their effectiveness in black-box scenarios. This limitation impedes their broader deployment in complex, dynamic environments. Further more, traditional black-box attacks are often characterized by poor transferability and high query costs, significantly limiting their practical applicability. To address these challenges, we propose the Dual-Phase Merging Transferable Reversible Attack method, which generates highly transferable initial adversarial perturbations in a white-box model and employs a memory augmented black-box strategy to effectively mislead target mod els. Experimental results demonstrate the superiority of our approach, achieving a 99.0% attack success rate and 100% recovery rate in black-box scenarios, highlighting its robustness in privacy protection. Moreover, we successfully implemented a black-box attack on a commercial model, further substantiating the potential of this approach for practical use. 

**Abstract (ZH)**: 在数字安全领域，可逆对抗样本（RAE）结合了对抗攻击与可逆数据隐藏技术，有效保护敏感数据并防止恶意深度神经网络（DNN）的未经授权分析。然而，现有RAE技术主要关注白盒攻击，缺乏对黑盒场景有效性的全面评估。这一局限性阻碍了其在复杂动态环境中的广泛应用。此外，传统的黑盒攻击通常表现出较差的迁移性和较高的查询成本，显著限制了其实用性。为应对这些挑战，我们提出了双阶段合并可迁移可逆攻击方法，该方法在白盒模型中生成高度可迁移的初始对抗扰动，并采用记忆增强的黑盒策略有效地误导目标模型。实验结果证明了该方法的优势，在黑盒场景中实现99.0%的攻击成功率和100%的数据恢复率，突显了其在隐私保护方面的稳健性。此外，我们成功对一个商用模型进行了黑盒攻击，进一步证明了该方法在实际应用中的潜力。 

---
# Optimizing Recommendations using Fine-Tuned LLMs 

**Title (ZH)**: 使用微调后的大语言模型优化推荐系统 

**Authors**: Prabhdeep Cheema, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2505.06841)  

**Abstract**: As digital media platforms strive to meet evolving user expectations, delivering highly personalized and intuitive movies and media recommendations has become essential for attracting and retaining audiences. Traditional systems often rely on keyword-based search and recommendation techniques, which limit users to specific keywords and a combination of keywords. This paper proposes an approach that generates synthetic datasets by modeling real-world user interactions, creating complex chat-style data reflective of diverse preferences. This allows users to express more information with complex preferences, such as mood, plot details, and thematic elements, in addition to conventional criteria like genre, title, and actor-based searches. In today's search space, users cannot write queries like ``Looking for a fantasy movie featuring dire wolves, ideally set in a harsh frozen world with themes of loyalty and survival.''
Building on these contributions, we evaluate synthetic datasets for diversity and effectiveness in training and benchmarking models, particularly in areas often absent from traditional datasets. This approach enhances personalization and accuracy by enabling expressive and natural user queries. It establishes a foundation for the next generation of conversational AI-driven search and recommendation systems in digital entertainment. 

**Abstract (ZH)**: 随着数字媒体平台努力满足不断演变的用户期望，提供高度个性化和直观的内容推荐已成为吸引和保留观众的关键。传统系统通常依赖于基于关键词的搜索和推荐技术，这限制了用户只能使用特定的关键词及其组合。本文提出了一种方法，通过模拟真实世界的用户交互来生成合成数据集，创建反映多样化偏好的复杂聊天风格数据。这使用户能够通过复杂偏好表达更多信息，包括情绪、情节细节和主题元素，而不仅仅是传统的类别搜索，如类型的、标题的和演员的搜索。在当今的搜索空间中，用户不能编写类似于“寻找一部特色为Direwolves的奇幻电影，理想情况下设定在一个严酷的冰冻世界中，主题涉及忠诚和生存”的查询。
基于这些贡献，我们评估合成数据集在多样性和有效性方面的表现，特别是在传统数据集中经常缺乏的领域，用以训练和基准测试模型。这种方法通过启用富有表现力和自然的用户查询来增强个性化和准确性。它为下一代基于对话AI的搜索和推荐系统在数字娱乐中的应用奠定了基础。 

---
# The power of fine-grained experts: Granularity boosts expressivity in Mixture of Experts 

**Title (ZH)**: 细粒度专家的力量：粒度增强混合专家模型的表达能力 

**Authors**: Enric Boix-Adsera, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2505.06839)  

**Abstract**: Mixture-of-Experts (MoE) layers are increasingly central to frontier model architectures. By selectively activating parameters, they reduce computational cost while scaling total parameter count. This paper investigates the impact of the number of active experts, termed granularity, comparing architectures with many (e.g., 8 per layer in DeepSeek) to those with fewer (e.g., 1 per layer in Llama-4 models). We prove an exponential separation in network expressivity based on this design parameter, suggesting that models benefit from higher granularity. Experimental results corroborate our theoretical findings and illustrate this separation. 

**Abstract (ZH)**: 混合专家层（MoE）在前沿模型架构中日益占据核心地位。通过选择性激活参数，它们在增加总参数量的同时降低计算成本。本文探讨了激活专家数量（称为精细度）对网络表达能力的影响，对比了每层具有多个专家（如DeepSeek中的每层8个专家）的架构与每层具有较少专家（如Llama-4模型中的每层1个专家）的架构。我们证明了基于此设计参数的网络表达能力存在指数级差异，表明模型从中受益于更高精细度。实验结果证实了我们的理论发现，并展示了这种差异。 

---
# Sandcastles in the Storm: Revisiting the (Im)possibility of Strong Watermarking 

**Title (ZH)**: 风暴中的沙堡：重访强水印的（不）可能性 

**Authors**: Fabrice Y Harel-Canada, Boran Erol, Connor Choi, Jason Liu, Gary Jiarui Song, Nanyun Peng, Amit Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06827)  

**Abstract**: Watermarking AI-generated text is critical for combating misuse. Yet recent theoretical work argues that any watermark can be erased via random walk attacks that perturb text while preserving quality. However, such attacks rely on two key assumptions: (1) rapid mixing (watermarks dissolve quickly under perturbations) and (2) reliable quality preservation (automated quality oracles perfectly guide edits). Through large-scale experiments and human-validated assessments, we find mixing is slow: 100% of perturbed texts retain traces of their origin after hundreds of edits, defying rapid mixing. Oracles falter, as state-of-the-art quality detectors misjudge edits (77% accuracy), compounding errors during attacks. Ultimately, attacks underperform: automated walks remove watermarks just 26% of the time -- dropping to 10% under human quality review. These findings challenge the inevitability of watermark removal. Instead, practical barriers -- slow mixing and imperfect quality control -- reveal watermarking to be far more robust than theoretical models suggest. The gap between idealized attacks and real-world feasibility underscores the need for stronger watermarking methods and more realistic attack models. 

**Abstract (ZH)**: 人工智能生成文本的水印对于防止滥用至关重要。然而，近期的理论工作认为任何水印都可以通过随机游走攻击被消除，这些攻击在扰动文本的同时保持了质量。然而，这类攻击依赖于两个关键假设：（1）快速混合（水印在扰动下迅速消散）和（2）可靠的质量保持（自动化质量仲裁器完美地指导编辑）。通过大规模实验和人类验证的评估，我们发现混合缓慢：经过数百次编辑后，100%的扰动文本仍保留其来源的痕迹，反驳了快速混合的假设。仲裁器表现不佳，最新质量检测器对编辑的判断有误（准确率为77%），在攻击过程中累积了错误。最终，攻击表现不佳：自动化游走只能去除水印26%的时间，在人工质量审阅下这一比例降至10%。这些发现挑战了水印去除的必然性。相反，实际障碍——缓慢的混合和不完美的质量控制——表明水印技术比理论模型所设想的更为 robust。理想化的攻击与现实可行性之间的差距凸显了更强大水印方法和更现实攻击模型的必要性。 

---
# ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification 

**Title (ZH)**: ThreatLens: LLM引导的硬件安全验证中的威胁建模与测试计划生成 

**Authors**: Dipayan Saha, Hasan Al Shaikh, Shams Tarek, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06821)  

**Abstract**: Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios. 

**Abstract (ZH)**: 当前的硬件安全验证过程主要依赖于手工威胁建模和测试计划生成，这既费时又容易出错，并且难以适应日益复杂的设计和不断演变的攻击方法。为应对这些挑战，我们提出了一种基于LLM的多agent框架——ThreatLens，该框架自动进行硬件安全验证中的威胁建模和测试计划生成。ThreatLens通过检索增强生成（RAG）提取相关安全知识，利用LLM进行威胁评估，并结合交互式用户反馈来确保生成的测试计划具有实际操作性。通过自动化这些过程，该框架减少了手工验证的工作量，提升了覆盖率，并确保了一种结构化且适应性强的安全验证方法。我们通过对NEORV32 SoC的评估，展示了该框架通过结构化测试计划实现自动化安全验证的能力，并验证了其在实际应用中的有效性。 

---
# Overview of the NLPCC 2025 Shared Task 4: Multi-modal, Multilingual, and Multi-hop Medical Instructional Video Question Answering Challenge 

**Title (ZH)**: NLPCC 2025 共享任务 4 概览：多模态、多语言、多跳医疗教学视频问答挑战 

**Authors**: Bin Li, Shenxi Liu, Yixuan Weng, Yue Du, Yuhang Tian, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06814)  

**Abstract**: Following the successful hosts of the 1-st (NLPCC 2023 Foshan) CMIVQA and the 2-rd (NLPCC 2024 Hangzhou) MMIVQA challenges, this year, a new task has been introduced to further advance research in multi-modal, multilingual, and multi-hop medical instructional question answering (M4IVQA) systems, with a specific focus on medical instructional videos. The M4IVQA challenge focuses on evaluating models that integrate information from medical instructional videos, understand multiple languages, and answer multi-hop questions requiring reasoning over various modalities. This task consists of three tracks: multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Single Video (M4TAGSV), multi-modal, multilingual, and multi-hop Video Corpus Retrieval (M4VCR) and multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Video Corpus (M4TAGVC). Participants in M4IVQA are expected to develop algorithms capable of processing both video and text data, understanding multilingual queries, and providing relevant answers to multi-hop medical questions. We believe the newly introduced M4IVQA challenge will drive innovations in multimodal reasoning systems for healthcare scenarios, ultimately contributing to smarter emergency response systems and more effective medical education platforms in multilingual communities. Our official website is this https URL 

**Abstract (ZH)**: 在成功举办了第一届（NLPCC 2023 福州）CMIVQA和第二届（NLPCC 2024 杭州）MMIVQA挑战赛后，今年引入了一个新的任务，旨在进一步推动多模态、多语言、多跳医疗指导性问答（M4IVQA）系统的研究，特别关注医疗指导性视频。M4IVQA挑战关注的是评估能够整合医疗指导性视频信息、理解多种语言并回答需要在多种模态上进行推理的多跳问题的模型。该任务包含三个赛道：多模态、多语言、多跳单视频时间答案定位（M4TAGSV）、多模态、多语言、多跳视频数据集检索（M4VCR）和多模态、多语言、多跳视频数据集时间答案定位（M4TAGVC）。M4IVQA参赛者需开发能够处理视频和文本数据、理解多语言查询并提供相关答案以应对多跳医疗问题的算法。我们相信新引入的M4IVQA挑战将推动医疗场景中多模态推理系统的创新，最终为多语言社区贡献更智能的应急响应系统和更有效的医疗教育平台。我们的官方网站是这个 https URL。 

---
# Quantum Observers: A NISQ Hardware Demonstration of Chaotic State Prediction Using Quantum Echo-state Networks 

**Title (ZH)**: 量子观测者：使用量子回声状态网络预测混沌态的NISQ硬件演示 

**Authors**: Erik L. Connerty, Ethan N. Evans, Gerasimos Angelatos, Vignesh Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06799)  

**Abstract**: Recent advances in artificial intelligence have highlighted the remarkable capabilities of neural network (NN)-powered systems on classical computers. However, these systems face significant computational challenges that limit scalability and efficiency. Quantum computers hold the potential to overcome these limitations and increase processing power beyond classical systems. Despite this, integrating quantum computing with NNs remains largely unrealized due to challenges posed by noise, decoherence, and high error rates in current quantum hardware. Here, we propose a novel quantum echo-state network (QESN) design and implementation algorithm that can operate within the presence of noise on current IBM hardware. We apply classical control-theoretic response analysis to characterize the QESN, emphasizing its rich nonlinear dynamics and memory, as well as its ability to be fine-tuned with sparsity and re-uploading blocks. We validate our approach through a comprehensive demonstration of QESNs functioning as quantum observers, applied in both high-fidelity simulations and hardware experiments utilizing data from a prototypical chaotic Lorenz system. Our results show that the QESN can predict long time-series with persistent memory, running over 100 times longer than the median T}1 and T2 of the IBM Marrakesh QPU, achieving state-of-the-art time-series performance on superconducting hardware. 

**Abstract (ZH)**: 近年来，人工智能力量的神经网络（NN）驱动系统在经典计算机上展现了令人瞩目的能力。然而，这些系统面临显著的计算挑战，限制了其可扩展性和效率。量子计算机有可能克服这些限制，并在处理能力上超越经典系统。尽管如此，将量子计算与神经网络相结合仍因当前量子硬件中的噪声、退相干和高错误率等挑战而难以实现。在此，我们提出了一种新型量子回声状态网络（QESN）的设计和实现算法，该算法能够在当前IBM硬件中的噪声环境中运行。我们通过经典的控制理论响应分析来表征QESN，强调其丰富的非线性动力学和记忆特性，以及其通过稀疏性与重加载块进行微调的能力。我们通过全面演示QESN作为量子观测器的功能来验证我们的方法，这些演示在高保真模拟和使用IBM Marrakesh QPU数据进行的硬件实验中进行。结果表明，QESN可以在保留长期记忆的情况下预测长时序列，并在超导硬件上实现了最先进的时序性能，运行时间超过IBM Marrakesh QPU的中位T1和T2时间的100多倍。 

---
# Decoding Futures Price Dynamics: A Regularized Sparse Autoencoder for Interpretable Multi-Horizon Forecasting and Factor Discovery 

**Title (ZH)**: 解码期货价格动态：一种正则化稀疏自编码器用于可解释的多horizon预测和因子发现 

**Authors**: Abhijit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06795)  

**Abstract**: Commodity price volatility creates economic challenges, necessitating accurate multi-horizon forecasting. Predicting prices for commodities like copper and crude oil is complicated by diverse interacting factors (macroeconomic, supply/demand, geopolitical, etc.). Current models often lack transparency, limiting strategic use. This paper presents a Regularized Sparse Autoencoder (RSAE), a deep learning framework for simultaneous multi-horizon commodity price prediction and discovery of interpretable latent market drivers. The RSAE forecasts prices at multiple horizons (e.g., 1-day, 1-week, 1-month) using multivariate time series. Crucially, L1 regularization ($\|\mathbf{z}\|_1$) on its latent vector $\mathbf{z}$ enforces sparsity, promoting parsimonious explanations of market dynamics through learned factors representing underlying drivers (e.g., demand, supply shocks). Drawing from energy-based models and sparse coding, the RSAE optimizes predictive accuracy while learning sparse representations. Evaluated on historical Copper and Crude Oil data with numerous indicators, our findings indicate the RSAE offers competitive multi-horizon forecasting accuracy and data-driven insights into price dynamics via its interpretable latent space, a key advantage over traditional black-box approaches. 

**Abstract (ZH)**: 商品价格波动创造经济挑战，亟需准确的多时域预测。本论文提出一种正则化稀疏自编码器（RSAE），这是一种用于同时进行多时域商品价格预测及可解释潜在市场驱动因素发现的深度学习框架。RSAE 利用多变量时间序列预测多个时域的价格（例如，1 天、1 周、1 个月）。关键地，其潜在向量 $\mathbf{z}$ 上的 L1 正则化（$\|\mathbf{z}\|_1$）促进稀疏性，通过学习代表潜在驱动因素（如需求、供给冲击）的因素来简洁地解释市场动态。借鉴能量模型和稀疏编码，RSAE 在提高预测准确性的同时学习稀疏表示。在包含多种指标的历史铜和原油数据上进行评估，我们的研究结果表明，RSAE 提供了具有竞争力的多时域预测准确度，并通过其可解释的潜在空间提供了数据驱动的价格动态见解，这一优势使其有别于传统的黑盒方法。 

---
# Symbolic Rule Extraction from Attention-Guided Sparse Representations in Vision Transformers 

**Title (ZH)**: 基于注意力引导稀疏表示的视觉变换器中符号规则提取 

**Authors**: Parth Padalkar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06745)  

**Abstract**: Recent neuro-symbolic approaches have successfully extracted symbolic rule-sets from CNN-based models to enhance interpretability. However, applying similar techniques to Vision Transformers (ViTs) remains challenging due to their lack of modular concept detectors and reliance on global self-attention mechanisms. We propose a framework for symbolic rule extraction from ViTs by introducing a sparse concept layer inspired by Sparse Autoencoders (SAEs). This linear layer operates on attention-weighted patch representations and learns a disentangled, binarized representation in which individual neurons activate for high-level visual concepts. To encourage interpretability, we apply a combination of L1 sparsity, entropy minimization, and supervised contrastive loss. These binarized concept activations are used as input to the FOLD-SE-M algorithm, which generates a rule-set in the form of logic programs. Our method achieves a 5.14% better classification accuracy than the standard ViT while enabling symbolic reasoning. Crucially, the extracted rule-set is not merely post-hoc but acts as a logic-based decision layer that operates directly on the sparse concept representations. The resulting programs are concise and semantically meaningful. This work is the first to extract executable logic programs from ViTs using sparse symbolic representations. It bridges the gap between transformer-based vision models and symbolic logic programming, providing a step forward in interpretable and verifiable neuro-symbolic AI. 

**Abstract (ZH)**: Recent神经符号方法已成功从基于CNN的模型中提取符号规则集以增强可解释性。但由于ViTs缺乏模块化概念检测器且依赖全局自注意力机制，将其上类似的技术应用仍然具有挑战性。我们提出了一种从ViTs提取符号规则的新框架，通过引入灵感源于稀疏自动编码器（SAEs）的稀疏概念层。该线性层作用于注意力加权片段表示，并学习一个彼此独立的二值化表示，其中单个神经元为高级视觉概念激活。为促进可解释性，我们应用L1稀疏性、熵最小化和监督对比丢失的组合。这些二值化概念激活被用作FOLD-SE-M算法的输入，该算法生成逻辑程序形式的规则集。该方法在标准ViT的基础上提高了5.14%的分类精度同时支持符号推理。至关重要的是，提取的规则集不仅仅是事后解释性的，而是作为基于逻辑的决策层直接作用于稀疏概念表示上。生成的程序简洁且语义有意义。这项工作首次使用稀疏符号表示从ViTs中提取可执行的逻辑程序，填补了基于变压器的视觉模型与符号逻辑编程之间的空白，为可解释和可验证的神经符号AI迈出了重要一步。 

---
# TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility 

**Title (ZH)**: TPK: 可信赖的轨迹预测集成先验知识以提高可解释性和动力学可行性 

**Authors**: Marius Baden, Ahmed Abouelazm, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06743)  

**Abstract**: Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics. 

**Abstract (ZH)**: 轨迹预测对于自动驾驶至关重要，能够通过预见周围道路使用者的运动来安全导航。然而，当前的深度学习模型往往缺乏可信度，因为它们的预测可能在物理上是不可能的，且不符合人类逻辑。为了使预测更加可信，最近的研究将先验知识纳入其中，如使用社会力模型来建模交互，使用动力学模型来实现物理现实。然而，这些方法专注于适用于车辆或行人的先验知识，无法推广到包含多种代理类别的交通环境中。我们提出了一种结合所有代理类别（车辆、行人在内和自行车）的交互和动力学先验知识的方法，并使用类别特定的交互层来捕捉代理行为的差异。为了提高代理交互的可解释性，我们引入了一种基于规则的交互重要性评分DG-SFM，以指导交互层。为了确保预测的物理可行性，我们为所有代理类别提出了合适的动力学模型，并采用了一种新颖的行人类动力学模型。我们使用最先进的Transformer HPTR作为基准，在Argoverse 2数据集上对我们的方法进行了评估。实验表明，我们的方法提高了交互的可解释性，揭示了错误预测与偏离我们交互先验之间的关联。尽管结合动力学模型略微降低了准确性，但它们消除了数据集和基准模型中存在的不可行轨迹。因此，我们的方法促进了轨迹预测的信任，因为其交互推理是可解释的，并且其预测符合物理原理。 

---
# Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving 

**Title (ZH)**: 边界引导的道路感知与物理可行的自主驾驶轨迹预测 

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06740)  

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions. 

**Abstract (ZH)**: 准确预测周围道路用户的轨迹对于实现安全高效的自动驾驶至关重要。尽管深度学习模型已有改进，但仍存在防止道路外预测和确保动力学可行性的挑战。现有方法整合了道路意识模块并施加动力学约束，但缺乏可行性保证，并且常在复杂性和灵活性之间引入权衡。本文提出了一种新型框架，将轨迹预测公式化为由允许的驾驶方向及其边界引导的约束回归问题。利用代理当前状态和高精度地图，我们的方法定义有效边界并确保道路内预测，通过训练网络学习边界多边线间的叠加路径。为了保证可行性，模型预测加速度分布，这些加速度分布确定车辆沿这些路径的行驶距离，并遵守动力学约束。我们通过Argoverse-2数据集将我们的方法与HPTR基线进行对比评估。虽然我们的方法在基准指标上略有下降，但在最终位移误差和去除不可行轨迹方面表现优异。此外，所提出的方法在不常见的操作和未见过的分布外场景中表现出更优的泛化能力，将对抗攻击下的道路外率从66%降至仅1%。这些结果突显了我们方法在生成可行和稳健预测方面的有效性。 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Title (ZH)**: 平衡进展与安全：自主驾驶中强化学习的新颖风险感知目标 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

**Abstract (ZH)**: 基于强化学习的自动驾驶奖励函数设计与风险意识目标研究：减少无信号交叉口碰撞促进安全驾驶 

---
# Deeply Explainable Artificial Neural Network 

**Title (ZH)**: 深度可解释的人工神经网络 

**Authors**: David Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06731)  

**Abstract**: While deep learning models have demonstrated remarkable success in numerous domains, their black-box nature remains a significant limitation, especially in critical fields such as medical image analysis and inference. Existing explainability methods, such as SHAP, LIME, and Grad-CAM, are typically applied post hoc, adding computational overhead and sometimes producing inconsistent or ambiguous results. In this paper, we present the Deeply Explainable Artificial Neural Network (DxANN), a novel deep learning architecture that embeds explainability ante hoc, directly into the training process. Unlike conventional models that require external interpretation methods, DxANN is designed to produce per-sample, per-feature explanations as part of the forward pass. Built on a flow-based framework, it enables both accurate predictions and transparent decision-making, and is particularly well-suited for image-based tasks. While our focus is on medical imaging, the DxANN architecture is readily adaptable to other data modalities, including tabular and sequential data. DxANN marks a step forward toward intrinsically interpretable deep learning, offering a practical solution for applications where trust and accountability are essential. 

**Abstract (ZH)**: 深度可解释人工神经网络（DxANN）：基于先验嵌入的可解释深度学习架构 

---
# Underwater object detection in sonar imagery with detection transformer and Zero-shot neural architecture search 

**Title (ZH)**: 基于检测变换器和零样本神经架构搜索的声纳图像 underwater目标检测 

**Authors**: XiaoTong Gu, Shengyu Tang, Yiming Cao, Changdong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06694)  

**Abstract**: Underwater object detection using sonar imagery has become a critical and rapidly evolving research domain within marine technology. However, sonar images are characterized by lower resolution and sparser features compared to optical images, which seriously degrades the performance of object this http URL address these challenges, we specifically propose a Detection Transformer (DETR) architecture optimized with a Neural Architecture Search (NAS) approach called NAS-DETR for object detection in sonar images. First, an improved Zero-shot Neural Architecture Search (NAS) method based on the maximum entropy principle is proposed to identify a real-time, high-representational-capacity CNN-Transformer backbone for sonar image detection. This method enables the efficient discovery of high-performance network architectures with low computational and time overhead. Subsequently, the backbone is combined with a Feature Pyramid Network (FPN) and a deformable attention-based Transformer decoder to construct a complete network architecture. This architecture integrates various advanced components and training schemes to enhance overall performance. Extensive experiments demonstrate that this architecture achieves state-of-the-art performance on two Representative datasets, while maintaining minimal overhead in real-time efficiency and computational complexity. Furthermore, correlation analysis between the key parameters and differential entropy-based fitness function is performed to enhance the interpretability of the proposed framework. To the best of our knowledge, this is the first work in the field of sonar object detection to integrate the DETR architecture with a NAS search mechanism. 

**Abstract (ZH)**: 基于声呐图像的目标检测：NAS-DETR在marine技术中的应用 

---
# FNBench: Benchmarking Robust Federated Learning against Noisy Labels 

**Title (ZH)**: FNBench: 在嘈杂标签环境下评估联邦学习的鲁棒性 

**Authors**: Xuefeng Jiang, Jia Li, Nannan Wu, Zhiyuan Wu, Xujing Li, Sheng Sun, Gang Xu, Yuwei Wang, Qi Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06684)  

**Abstract**: Robustness to label noise within data is a significant challenge in federated learning (FL). From the data-centric perspective, the data quality of distributed datasets can not be guaranteed since annotations of different clients contain complicated label noise of varying degrees, which causes the performance degradation. There have been some early attempts to tackle noisy labels in FL. However, there exists a lack of benchmark studies on comprehensively evaluating their practical performance under unified settings. To this end, we propose the first benchmark study FNBench to provide an experimental investigation which considers three diverse label noise patterns covering synthetic label noise, imperfect human-annotation errors and systematic errors. Our evaluation incorporates eighteen state-of-the-art methods over five image recognition datasets and one text classification dataset. Meanwhile, we provide observations to understand why noisy labels impair FL, and additionally exploit a representation-aware regularization method to enhance the robustness of existing methods against noisy labels based on our observations. Finally, we discuss the limitations of this work and propose three-fold future directions. To facilitate related communities, our source code is open-sourced at this https URL. 

**Abstract (ZH)**: 联邦学习中数据标签噪声鲁棒性研究：FNBench基准研究 

---
# A Short Overview of Multi-Modal Wi-Fi Sensing 

**Title (ZH)**: 多模Wi-Fi传感简要概述 

**Authors**: Zijian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06682)  

**Abstract**: Wi-Fi sensing has emerged as a significant technology in wireless sensing and Integrated Sensing and Communication (ISAC), offering benefits such as low cost, high penetration, and enhanced privacy. Currently, it is widely utilized in various applications, including action recognition, human localization, and crowd counting. However, Wi-Fi sensing also faces challenges, such as low robustness and difficulties in data collection. Recently, there has been an increasing focus on multi-modal Wi-Fi sensing, where other modalities can act as teachers, providing ground truth or robust features for Wi-Fi sensing models to learn from, or can be directly fused with Wi-Fi for enhanced sensing capabilities. Although these methods have demonstrated promising results and substantial value in practical applications, there is a lack of comprehensive surveys reviewing them. To address this gap, this paper reviews the multi-modal Wi-Fi sensing literature \textbf{from the past 24 months} and highlights the current limitations, challenges and future directions in this field. 

**Abstract (ZH)**: Wi-Fi sensing从多模态角度在过去24个月内的发展：当前限制、挑战与未来方向 

---
# Enfoque Odychess: Un método dialéctico, constructivista y adaptativo para la enseñanza del ajedrez con inteligencias artificiales generativas 

**Title (ZH)**: Odychess 方法：一种辩证、建构主义和适应性的象棋教学方法，涉及生成性人工智能 

**Authors**: Ernesto Giralt Hernandez, Lazaro Antonio Bueno Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06652)  

**Abstract**: Chess teaching has evolved through different approaches, however, traditional methodologies, often based on memorization, contrast with the new possibilities offered by generative artificial intelligence, a technology still little explored in this field. This study seeks to empirically validate the effectiveness of the Odychess Approach in improving chess knowledge, strategic understanding, and metacognitive skills in students. A quasi-experimental study was conducted with a pre-test/post-test design and a control group (N=60). The experimental intervention implemented the Odychess Approach, incorporating a Llama 3.3 language model that was specifically adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques to act as a Socratic chess tutor. Quantitative assessment instruments were used to measure chess knowledge, strategic understanding, and metacognitive skills before and after the intervention. The results of the quasi-experimental study showed significant improvements in the experimental group compared to the control group in the three variables analyzed: chess knowledge, strategic understanding, and metacognitive skills. The complementary qualitative analysis revealed greater analytical depth, more developed dialectical reasoning, and increased intrinsic motivation in students who participated in the Odychess method-based intervention. The Odychess Approach represents an effective pedagogical methodology for teaching chess, demonstrating the potential of the synergistic integration of constructivist and dialectical principles with generative artificial intelligence. The implications of this work are relevant for educators and institutions interested in adopting innovative pedagogical technologies and for researchers in the field of AI applied to education, highlighting the transferability of the language model adaptation methodology to other educational domains. 

**Abstract (ZH)**: 棋类教学通过不同的方法演化，然而，传统的基于记忆的方法与生成式人工智能技术提供的新可能性形成对比，而该技术在这一领域仍较少被探索。本研究旨在通过定量实验证实Odychess方法在提高学生棋类知识、战略理解以及元认知技能方面的有效性。采用准实验设计，包括预测试/后测试以及对照组（N=60），实验组实施了结合Parameter-Efficient Fine-Tuning (PEFT) 技术适配的Llama 3.3语言模型，以此作为苏格拉底式棋类导师进行干预。使用定量评估工具在干预前和干预后分别测量棋类知识、战略理解以及元认知技能。准实验研究的结果显示，与对照组相比，实验组在三个分析变量上均表现出显著改善：棋类知识、战略理解以及元认知技能。补充的定性分析表明，采用Odychess方法为基础的干预措施的学生表现出更深层次的分析能力、更发达的辩证推理能力和更高的内在动机。Odychess方法代表了一种有效的教学方法，展示了结合建构主义和辩证原则与生成式人工智能的协同集成的潜力。该研究对关注采用创新教学技术的教育工作者和机构以及研究人工智能在教育应用领域的研究人员具有重要意义，强调了语言模型适配方法在其他教育领域中的可转移性。 

---
# Dyn-D$^2$P: Dynamic Differentially Private Decentralized Learning with Provable Utility Guarantee 

**Title (ZH)**: Dyn-D$^2$P: 动态差分隐私去中心化学习及其可证明的效用保证 

**Authors**: Zehan Zhu, Yan Huang, Xin Wang, Shouling Ji, Jinming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06651)  

**Abstract**: Most existing decentralized learning methods with differential privacy (DP) guarantee rely on constant gradient clipping bounds and fixed-level DP Gaussian noises for each node throughout the training process, leading to a significant accuracy degradation compared to non-private counterparts. In this paper, we propose a new Dynamic Differentially Private Decentralized learning approach (termed Dyn-D$^2$P) tailored for general time-varying directed networks. Leveraging the Gaussian DP (GDP) framework for privacy accounting, Dyn-D$^2$P dynamically adjusts gradient clipping bounds and noise levels based on gradient convergence. This proposed dynamic noise strategy enables us to enhance model accuracy while preserving the total privacy budget. Extensive experiments on benchmark datasets demonstrate the superiority of Dyn-D$^2$P over its counterparts employing fixed-level noises, especially under strong privacy guarantees. Furthermore, we provide a provable utility bound for Dyn-D$^2$P that establishes an explicit dependency on network-related parameters, with a scaling factor of $1/\sqrt{n}$ in terms of the number of nodes $n$ up to a bias error term induced by gradient clipping. To our knowledge, this is the first model utility analysis for differentially private decentralized non-convex optimization with dynamic gradient clipping bounds and noise levels. 

**Abstract (ZH)**: 一种新的动态差分隐私去中心化学习方法（Dyn-D$^2$P）：针对一般时间varying有向网络的设计 

---
# AI-Powered Anomaly Detection with Blockchain for Real-Time Security and Reliability in Autonomous Vehicles 

**Title (ZH)**: 基于区块链的AI驱动异常检测技术及其在自主车辆实时安全与可靠性中的应用 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06632)  

**Abstract**: Autonomous Vehicles (AV) proliferation brings important and pressing security and reliability issues that must be dealt with to guarantee public safety and help their widespread adoption. The contribution of the proposed research is towards achieving more secure, reliable, and trustworthy autonomous transportation system by providing more capabilities for anomaly detection, data provenance, and real-time response in safety critical AV deployments. In this research, we develop a new framework that combines the power of Artificial Intelligence (AI) for real-time anomaly detection with blockchain technology to detect and prevent any malicious activity including sensor failures in AVs. Through Long Short-Term Memory (LSTM) networks, our approach continually monitors associated multi-sensor data streams to detect anomalous patterns that may represent cyberattacks as well as hardware malfunctions. Further, this framework employs a decentralized platform for securely storing sensor data and anomaly alerts in a blockchain ledger for data incorruptibility and authenticity, while offering transparent forensic features. Moreover, immediate automated response mechanisms are deployed using smart contracts when anomalies are found. This makes the AV system more resilient to attacks from both cyberspace and hardware component failure. Besides, we identify potential challenges of scalability in handling high frequency sensor data, computational constraint in resource constrained environment, and of distributed data storage in terms of privacy. 

**Abstract (ZH)**: 自主驾驶车辆的普及带来了重要的紧迫的安全性和可靠性问题，必须解决这些问题以确保公众安全并促进其广泛应用。本研究的贡献在于通过结合人工智能（AI）实现即时异常检测与区块链技术，提供更多的异常检测能力、数据溯源能力和在关键安全场景下自主驾驶车辆的即时响应能力，以实现更安全、可靠和可信赖的自主交通系统。在本研究中，我们开发了一个新的框架，该框架结合了基于长短期记忆网络（LSTM）的实时异常检测能力与区块链技术，用于检测和防止包括传感器故障在内的任何恶意活动。进一步地，该框架采用去中心化的平台在区块链账本中安全地存储传感器数据和异常警报，以确保数据的不可篡改性和真实性，并提供透明的取证功能。此外，当检测到异常时，将部署智能合约以实现即时自动化响应机制，使自主驾驶车辆系统更具抗攻击性，可以从网络空间和硬件组件故障中恢复。此外，我们还识别了在高频率传感器数据处理中可能面临的扩展性挑战、受限资源环境下的计算约束挑战，以及分布式数据存储中的隐私问题。 

---
# Dynamic Domain Information Modulation Algorithm for Multi-domain Sentiment Analysis 

**Title (ZH)**: 多域情感分析的动态领域信息调制算法 

**Authors**: Chunyi Yue, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06630)  

**Abstract**: Multi-domain sentiment classification aims to mitigate poor performance models due to the scarcity of labeled data in a single domain, by utilizing data labeled from various domains. A series of models that jointly train domain classifiers and sentiment classifiers have demonstrated their advantages, because domain classification helps generate necessary information for sentiment classification. Intuitively, the importance of sentiment classification tasks is the same in all domains for multi-domain sentiment classification; but domain classification tasks are different because the impact of domain information on sentiment classification varies across different fields; this can be controlled through adjustable weights or hyper parameters. However, as the number of domains increases, existing hyperparameter optimization algorithms may face the following challenges: (1) tremendous demand for computing resources, (2) convergence problems, and (3) high algorithm complexity. To efficiently generate the domain information required for sentiment classification in each domain, we propose a dynamic information modulation algorithm. Specifically, the model training process is divided into two stages. In the first stage, a shared hyperparameter, which would control the proportion of domain classification tasks across all fields, is determined. In the second stage, we introduce a novel domain-aware modulation algorithm to adjust the domain information contained in the input text, which is then calculated based on a gradient-based and loss-based method. In summary, experimental results on a public sentiment analysis dataset containing 16 domains prove the superiority of the proposed method. 

**Abstract (ZH)**: 多领域情感分类旨在通过利用来自多个领域的标注数据来缓解单一领域标注数据稀少导致的模型性能不佳问题。一系列同时训练领域分类器和情感分类器的模型展示了其优势，因为领域分类有助于为情感分类生成必要的信息。直觉上，多领域情感分类中所有领域的感情分类任务的重要性相同；但领域分类任务不同，因为领域信息对情感分类的影响因领域而异；这可以通过可调节的权重或超参数来控制。然而，随着领域数量的增加，现有的超参数优化算法可能会面临以下挑战：（1）对计算资源的巨大需求，（2）收敛问题，（3）高算法复杂度。为高效生成用于每个领域的情感分类所需要的领域信息，我们提出了一种动态信息调制算法。具体而言，模型训练过程分为两个阶段。在第一阶段，确定一个共享的超参数，该超参数将控制各个领域领域分类任务的比例。在第二阶段，我们引入了一种新颖的领域意识调制算法来调整输入文本中的领域信息，并基于梯度和损失方法进行计算。总之，公共情感分析数据集上的16个领域实验结果证明了所提方法的优越性。 

---
# CaMDN: Enhancing Cache Efficiency for Multi-tenant DNNs on Integrated NPUs 

**Title (ZH)**: CaMDN: 提升集成NPUs上多租户DNN缓存效率 

**Authors**: Tianhao Cai, Liang Wang, Limin Xiao, Meng Han, Zeyu Wang, Lin Sun, Xiaojian Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06625)  

**Abstract**: With the rapid development of DNN applications, multi-tenant execution, where multiple DNNs are co-located on a single SoC, is becoming a prevailing trend. Although many methods are proposed in prior works to improve multi-tenant performance, the impact of shared cache is not well studied. This paper proposes CaMDN, an architecture-scheduling co-design to enhance cache efficiency for multi-tenant DNNs on integrated NPUs. Specifically, a lightweight architecture is proposed to support model-exclusive, NPU-controlled regions inside shared cache to eliminate unexpected cache contention. Moreover, a cache scheduling method is proposed to improve shared cache utilization. In particular, it includes a cache-aware mapping method for adaptability to the varying available cache capacity and a dynamic allocation algorithm to adjust the usage among co-located DNNs at runtime. Compared to prior works, CaMDN reduces the memory access by 33.4% on average and achieves a model speedup of up to 2.56$\times$ (1.88$\times$ on average). 

**Abstract (ZH)**: 基于共享缓存的多租户DNN架构-调度协同设计CaMDN 

---
# Integrating Explainable AI in Medical Devices: Technical, Clinical and Regulatory Insights and Recommendations 

**Title (ZH)**: 将可解释的AI集成到医疗设备中：技术、临床和监管洞察与建议 

**Authors**: Dima Alattal, Asal Khoshravan Azar, Puja Myles, Richard Branson, Hatim Abdulhussein, Allan Tucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06620)  

**Abstract**: There is a growing demand for the use of Artificial Intelligence (AI) and Machine Learning (ML) in healthcare, particularly as clinical decision support systems to assist medical professionals. However, the complexity of many of these models, often referred to as black box models, raises concerns about their safe integration into clinical settings as it is difficult to understand how they arrived at their predictions. This paper discusses insights and recommendations derived from an expert working group convened by the UK Medicine and Healthcare products Regulatory Agency (MHRA). The group consisted of healthcare professionals, regulators, and data scientists, with a primary focus on evaluating the outputs from different AI algorithms in clinical decision-making contexts. Additionally, the group evaluated findings from a pilot study investigating clinicians' behaviour and interaction with AI methods during clinical diagnosis. Incorporating AI methods is crucial for ensuring the safety and trustworthiness of medical AI devices in clinical settings. Adequate training for stakeholders is essential to address potential issues, and further insights and recommendations for safely adopting AI systems in healthcare settings are provided. 

**Abstract (ZH)**: 人工智能和机器学习在医疗保健中的应用：黑箱模型的复杂性及其在临床决策支持中的安全整合探究——英国 Medicine and Healthcare products Regulatory Agency (MHRA) 专家工作组的见解与建议 

---
# Burger: Robust Graph Denoising-augmentation Fusion and Multi-semantic Modeling in Social Recommendation 

**Title (ZH)**: Burger：社交推荐中的鲁棒图去噪、增强融合与多语义建模 

**Authors**: Yuqin Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06612)  

**Abstract**: In the era of rapid development of social media, social recommendation systems as hybrid recommendation systems have been widely applied. Existing methods capture interest similarity between users to filter out interest-irrelevant relations in social networks that inevitably decrease recommendation accuracy, however, limited research has a focus on the mutual influence of semantic information between the social network and the user-item interaction network for further improving social recommendation. To address these issues, we introduce a social \underline{r}ecommendation model with ro\underline{bu}st g\underline{r}aph denoisin\underline{g}-augmentation fusion and multi-s\underline{e}mantic Modeling(Burger). Specifically, we firstly propose to construct a social tensor in order to smooth the training process of the model. Then, a graph convolutional network and a tensor convolutional network are employed to capture user's item preference and social preference, respectively. Considering the different semantic information in the user-item interaction network and the social network, a bi-semantic coordination loss is proposed to model the mutual influence of semantic information. To alleviate the interference of interest-irrelevant relations on multi-semantic modeling, we further use Bayesian posterior probability to mine potential social relations to replace social noise. Finally, the sliding window mechanism is utilized to update the social tensor as the input for the next iteration. Extensive experiments on three real datasets show Burger has a superior performance compared with the state-of-the-art models. 

**Abstract (ZH)**: 社交媒体快速发展时代的社交推荐系统robust图去噪增强融合多语义建模(Burger)研究 

---
# Feature Representation Transferring to Lightweight Models via Perception Coherence 

**Title (ZH)**: 基于感知一致性的小型模型特征表示转移 

**Authors**: Hai-Vy Nguyen, Fabrice Gamboa, Sixin Zhang, Reda Chhaibi, Serge Gratton, Thierry Giaccone  

**Link**: [PDF](https://arxiv.org/pdf/2505.06595)  

**Abstract**: In this paper, we propose a method for transferring feature representation to lightweight student models from larger teacher models. We mathematically define a new notion called \textit{perception coherence}. Based on this notion, we propose a loss function, which takes into account the dissimilarities between data points in feature space through their ranking. At a high level, by minimizing this loss function, the student model learns to mimic how the teacher model \textit{perceives} inputs. More precisely, our method is motivated by the fact that the representational capacity of the student model is weaker than the teacher model. Hence, we aim to develop a new method allowing for a better relaxation. This means that, the student model does not need to preserve the absolute geometry of the teacher one, while preserving global coherence through dissimilarity ranking. Our theoretical insights provide a probabilistic perspective on the process of feature representation transfer. Our experiments results show that our method outperforms or achieves on-par performance compared to strong baseline methods for representation transferring. 

**Abstract (ZH)**: 本文提出了一种将大型教师模型的特征表示转移到轻量级学生模型的方法。我们从数学上定义了一个新的概念叫作“感知一致性”。基于这一概念，我们提出了一种损失函数，该损失函数通过数据点在特征空间中的排名来考虑其差异性。总体而言，通过最小化该损失函数，学生模型学会模仿教师模型如何“感知”输入。更精确地说，我们的方法动机在于学生的表示能力弱于教师模型，因此我们旨在开发一种新的方法使得学生模型能够更有效地放松约束。这意味着，学生模型不需要完全保留教师模型的绝对几何结构，同时通过差异性排名保持全局一致性。我们的理论洞察提供了特征表示转移过程的概率视角。我们的实验结果表明，与强基准方法相比，我们的方法在特征表示转移上表现出更优或相当的性能。 

---
# Optimal Transport for Machine Learners 

**Title (ZH)**: 机器学习中的最优传输 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2505.06589)  

**Abstract**: Optimal Transport is a foundational mathematical theory that connects optimization, partial differential equations, and probability. It offers a powerful framework for comparing probability distributions and has recently become an important tool in machine learning, especially for designing and evaluating generative models. These course notes cover the fundamental mathematical aspects of OT, including the Monge and Kantorovich formulations, Brenier's theorem, the dual and dynamic formulations, the Bures metric on Gaussian distributions, and gradient flows. It also introduces numerical methods such as linear programming, semi-discrete solvers, and entropic regularization. Applications in machine learning include topics like training neural networks via gradient flows, token dynamics in transformers, and the structure of GANs and diffusion models. These notes focus primarily on mathematical content rather than deep learning techniques. 

**Abstract (ZH)**: 最优运输是连接优化、偏微分方程和概率的基础数学理论。它提供了一种强大的框架来比较概率分布，并 recently 成为机器学习中一个重要的工具，特别是在设计和评估生成模型方面。这些课程笔记涵盖了最优运输的基本数学方面，包括蒙格和坎托罗维奇公式、布伦耐尔定理、对偶和动力学公式、高斯分布上的布勒斯度量以及梯度流。它还介绍了数值方法，如线性规划、半离散求解器和熵正则化。在机器学习中的应用包括通过梯度流训练神经网络、变压器中的标记动力学以及生成对抗网络和扩散模型的结构等内容。这些笔记主要关注数学内容而非深度学习技术。 

---
# JAEGER: Dual-Level Humanoid Whole-Body Controller 

**Title (ZH)**: JAEGER: 双层次 humanoid 整体体动控制算法 

**Authors**: Ziluo Ding, Haobin Jiang, Yuxuan Wang, Zhenguo Sun, Yu Zhang, Xiaojie Niu, Ming Yang, Weishuai Zeng, Xinrun Xu, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06584)  

**Abstract**: This paper presents JAEGER, a dual-level whole-body controller for humanoid robots that addresses the challenges of training a more robust and versatile policy. Unlike traditional single-controller approaches, JAEGER separates the control of the upper and lower bodies into two independent controllers, so that they can better focus on their distinct tasks. This separation alleviates the dimensionality curse and improves fault tolerance. JAEGER supports both root velocity tracking (coarse-grained control) and local joint angle tracking (fine-grained control), enabling versatile and stable movements. To train the controller, we utilize a human motion dataset (AMASS), retargeting human poses to humanoid poses through an efficient retargeting network, and employ a curriculum learning approach. This method performs supervised learning for initialization, followed by reinforcement learning for further exploration. We conduct our experiments on two humanoid platforms and demonstrate the superiority of our approach against state-of-the-art methods in both simulation and real environments. 

**Abstract (ZH)**: JAEGER： humanoid机器人的双层次全身控制器及其训练方法 

---
# Two-Stage Random Alternation Framework for Zero-Shot Pansharpening 

**Title (ZH)**: 两阶段随机交替框架用于零样本 pansharpening 

**Authors**: Haorui Chen, Zeyu Ren, Jiaxuan Ren, Ran Ran, Jinliang Shao, Jie Huang, Liangjian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06576)  

**Abstract**: In recent years, pansharpening has seen rapid advancements with deep learning methods, which have demonstrated impressive fusion quality. However, the challenge of acquiring real high-resolution images limits the practical applicability of these methods. To address this, we propose a two-stage random alternating framework (TRA-PAN) that effectively integrates strong supervision constraints from reduced-resolution images with the physical characteristics of full-resolution images. The first stage introduces a pre-training procedure, which includes Degradation-Aware Modeling (DAM) to capture spatial-spectral degradation mappings, alongside a warm-up procedure designed to reduce training time and mitigate the negative effects of reduced-resolution data. In the second stage, Random Alternation Optimization (RAO) is employed, where random alternating training leverages the strengths of both reduced- and full-resolution images, further optimizing the fusion model. By primarily relying on full-resolution images, our method enables zero-shot training with just a single image pair, obviating the need for large datasets. Experimental results demonstrate that TRA-PAN outperforms state-of-the-art (SOTA) methods in both quantitative metrics and visual quality in real-world scenarios, highlighting its strong practical applicability. 

**Abstract (ZH)**: 基于两阶段随机交替框架的强监督 pansharpening 方法（TRA-PAN）：结合低分辨率图像的监督约束与高分辨率图像的物理特性 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Title (ZH)**: MacRAG: 压缩、切片和扩展以实现多尺度自适应上下文检索增强生成 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

**Abstract (ZH)**: 多尺度自适应上下文检索增强生成（MacRAG）：一种高效的复杂多跳推理解决方案 

---
# Quadrupedal Robot Skateboard Mounting via Reverse Curriculum Learning 

**Title (ZH)**: 基于逆序课程学习的四足机器人滑板搭载 

**Authors**: Danil Belov, Artem Erkhov, Elizaveta Pestova, Ilya Osokin, Dzmitry Tsetserukou, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06561)  

**Abstract**: The aim of this work is to enable quadrupedal robots to mount skateboards using Reverse Curriculum Reinforcement Learning. Although prior work has demonstrated skateboarding for quadrupeds that are already positioned on the board, the initial mounting phase still poses a significant challenge. A goal-oriented methodology was adopted, beginning with the terminal phases of the task and progressively increasing the complexity of the problem definition to approximate the desired objective. The learning process was initiated with the skateboard rigidly fixed within the global coordinate frame and the robot positioned directly above it. Through gradual relaxation of these initial conditions, the learned policy demonstrated robustness to variations in skateboard position and orientation, ultimately exhibiting a successful transfer to scenarios involving a mobile skateboard. The code, trained models, and reproducible examples are available at the following link: this https URL 

**Abstract (ZH)**: 本工作旨在利用反向 Curriculum 强化学习使四足机器人能够骑上滑板。尽管之前的工作已经展示了四足机器人在滑板上定位好之后进行滑板骑行，初始上板阶段仍然存在重大挑战。采用了一种以目标为导向的方法，从任务的最终阶段开始，逐步增加问题定义的复杂性，以逼近所需目标。学习过程从滑板刚性固定在全局坐标系中且机器人直接在其上方开始。通过逐渐放宽这些初始条件，学习到的策略显示了对滑板位置和方向变化的鲁棒性，并最终成功地转移到涉及移动滑板的场景中。相关代码、训练模型及可重复实验示例可在以下链接获取：this https URL 

---
# dcFCI: Robust Causal Discovery Under Latent Confounding, Unfaithfulness, and Mixed Data 

**Title (ZH)**: dcFCI: 在潜在混杂因素、不忠实性和混合数据下的稳健因果发现 

**Authors**: Adèle H. Ribeiro, Dominik Heider  

**Link**: [PDF](https://arxiv.org/pdf/2505.06542)  

**Abstract**: Causal discovery is central to inferring causal relationships from observational data. In the presence of latent confounding, algorithms such as Fast Causal Inference (FCI) learn a Partial Ancestral Graph (PAG) representing the true model's Markov Equivalence Class. However, their correctness critically depends on empirical faithfulness, the assumption that observed (in)dependencies perfectly reflect those of the underlying causal model, which often fails in practice due to limited sample sizes. To address this, we introduce the first nonparametric score to assess a PAG's compatibility with observed data, even with mixed variable types. This score is both necessary and sufficient to characterize structural uncertainty and distinguish between distinct PAGs. We then propose data-compatible FCI (dcFCI), the first hybrid causal discovery algorithm to jointly address latent confounding, empirical unfaithfulness, and mixed data types. dcFCI integrates our score into an (Anytime)FCI-guided search that systematically explores, ranks, and validates candidate PAGs. Experiments on synthetic and real-world scenarios demonstrate that dcFCI significantly outperforms state-of-the-art methods, often recovering the true PAG even in small and heterogeneous datasets. Examining top-ranked PAGs further provides valuable insights into structural uncertainty, supporting more robust and informed causal reasoning and decision-making. 

**Abstract (ZH)**: 因果发现是从观察数据中推断因果关系的核心。在潜在共因存在的情况下，快速因果推理（FCI）算法学习一个部分祖先图（PAG），代表真实模型的马尔可夫等价类。然而，它们的正确性严格依赖于经验忠实性假设，即观测到的（无）相关性完美地反映了潜在因果模型中的（无）相关性，但受限于样本量有限，这一假设在实践中经常失效。为解决这一问题，我们引入了第一个非参数分数，以评估PAG与观测数据的兼容性，即使变量类型混合也不例外。该分数既是表征结构不确定性所必需的，也是区分不同PAG所充分的。然后，我们提出了数据兼容性FCI（dcFCI），这是第一个同时解决潜在共因、经验不忠实性和混合数据类型的混合因果发现算法。dcFCI将我们的分数整合到一个由FCI引导的搜索中，系统地探索、排名和验证候选的PAGs。实验结果显示，dcFCI显著优于现有方法，在小规模和异质数据集中经常能够恢复真实的PAG。进一步研究排名靠前的PAGs提供了关于结构不确定性的宝贵见解，支持更稳健和知情的因果推理和决策。 

---
# ProFashion: Prototype-guided Fashion Video Generation with Multiple Reference Images 

**Title (ZH)**: 基于原型指导的多参考图像时尚视频生成 

**Authors**: Xianghao Kong, Qiaosong Qi, Yuanbin Wang, Anyi Rao, Biaolong Chen, Aixi Zhang, Si Liu, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06537)  

**Abstract**: Fashion video generation aims to synthesize temporally consistent videos from reference images of a designated character. Despite significant progress, existing diffusion-based methods only support a single reference image as input, severely limiting their capability to generate view-consistent fashion videos, especially when there are different patterns on the clothes from different perspectives. Moreover, the widely adopted motion module does not sufficiently model human body movement, leading to sub-optimal spatiotemporal consistency. To address these issues, we propose ProFashion, a fashion video generation framework leveraging multiple reference images to achieve improved view consistency and temporal coherency. To effectively leverage features from multiple reference images while maintaining a reasonable computational cost, we devise a Pose-aware Prototype Aggregator, which selects and aggregates global and fine-grained reference features according to pose information to form frame-wise prototypes, which serve as guidance in the denoising process. To further enhance motion consistency, we introduce a Flow-enhanced Prototype Instantiator, which exploits the human keypoint motion flow to guide an extra spatiotemporal attention process in the denoiser. To demonstrate the effectiveness of ProFashion, we extensively evaluate our method on the MRFashion-7K dataset we collected from the Internet. ProFashion also outperforms previous methods on the UBC Fashion dataset. 

**Abstract (ZH)**: 基于多参考图像的时尚视频生成：实现增强的视角一致性和时间连贯性 

---
# TACFN: Transformer-based Adaptive Cross-modal Fusion Network for Multimodal Emotion Recognition 

**Title (ZH)**: 基于Transformer的自适应跨模态融合网络多模态情感识别 

**Authors**: Feng Liu, Ziwang Fu, Yunlong Wang, Qijian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06536)  

**Abstract**: The fusion technique is the key to the multimodal emotion recognition task. Recently, cross-modal attention-based fusion methods have demonstrated high performance and strong robustness. However, cross-modal attention suffers from redundant features and does not capture complementary features well. We find that it is not necessary to use the entire information of one modality to reinforce the other during cross-modal interaction, and the features that can reinforce a modality may contain only a part of it. To this end, we design an innovative Transformer-based Adaptive Cross-modal Fusion Network (TACFN). Specifically, for the redundant features, we make one modality perform intra-modal feature selection through a self-attention mechanism, so that the selected features can adaptively and efficiently interact with another modality. To better capture the complementary information between the modalities, we obtain the fused weight vector by splicing and use the weight vector to achieve feature reinforcement of the modalities. We apply TCAFN to the RAVDESS and IEMOCAP datasets. For fair comparison, we use the same unimodal representations to validate the effectiveness of the proposed fusion method. The experimental results show that TACFN brings a significant performance improvement compared to other methods and reaches the state-of-the-art. All code and models could be accessed from this https URL. 

**Abstract (ZH)**: 多模态情感识别任务中的融合技术是关键。近年来，跨模态注意力基融合方法显示出高性能和强鲁棒性。然而，跨模态注意力存在冗余特征问题，并不能很好地捕捉互补特征。我们发现，在跨模态交互过程中，并非必须使用另一种模态的全部信息来强化自身，能够强化一种模态的特征可能仅包含其部分信息。为此，我们设计了一种创新的基于Transformer的自适应跨模态融合网络(TACFN)。具体来说，对于冗余特征，使一种模态通过自我注意力机制进行模内特征选择，从而使选定的特征能够适应性和高效地与另一种模态交互。为了更好地捕捉模态间的互补信息，我们通过拼接获得融合权向量，并利用此权向量实现模态特征的强化。我们将TACFN应用于RAVDESS和IEMOCAP数据集。为了公平比较，我们使用相同的单模态表示验证所提融合方法的有效性。实验结果表明，TACFN相比其他方法带来了显著的性能提升，并达到当前最佳水平。所有代码和模型均可从此链接访问。 

---
# Improving Generalization of Medical Image Registration Foundation Model 

**Title (ZH)**: 改进医学图像配准基础模型的泛化能力 

**Authors**: Jing Hu, Kaiwei Yu, Hongjiang Xian, Shu Hu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06527)  

**Abstract**: Deformable registration is a fundamental task in medical image processing, aiming to achieve precise alignment by establishing nonlinear correspondences between images. Traditional methods offer good adaptability and interpretability but are limited by computational efficiency. Although deep learning approaches have significantly improved registration speed and accuracy, they often lack flexibility and generalizability across different datasets and tasks. In recent years, foundation models have emerged as a promising direction, leveraging large and diverse datasets to learn universal features and transformation patterns for image registration, thus demonstrating strong cross-task transferability. However, these models still face challenges in generalization and robustness when encountering novel anatomical structures, varying imaging conditions, or unseen modalities. To address these limitations, this paper incorporates Sharpness-Aware Minimization (SAM) into foundation models to enhance their generalization and robustness in medical image registration. By optimizing the flatness of the loss landscape, SAM improves model stability across diverse data distributions and strengthens its ability to handle complex clinical scenarios. Experimental results show that foundation models integrated with SAM achieve significant improvements in cross-dataset registration performance, offering new insights for the advancement of medical image registration technology. Our code is available at this https URL}{this https URL\_sam. 

**Abstract (ZH)**: 变形注册是医学图像处理中的一个基本任务，旨在通过建立图像之间的非线性对应关系实现精确对齐。传统的注册方法提供了良好的适应性和可解释性，但在计算效率上受到限制。尽管深度学习方法显著提高了注册速度和准确性，但在不同数据集和任务上的灵活性和普遍性仍然不足。近年来，基础模型作为一种有前途的方向出现，利用大规模和多样化的数据集学习图像注册中的通用特征和变换模式，从而展示了强大的跨任务迁移能力。然而，这些模型在遇到新型解剖结构、变化的成像条件或未见过的模态时，依然面临泛化能力和鲁棒性的挑战。为了解决这些局限性，本文将Sharpness-Aware Minimization (SAM)纳入基础模型中，以增强其在医学图像注册中的泛化能力和鲁棒性。通过优化损失景观的平坦度，SAM提高了模型在不同数据分布下的稳定性，并增强了其处理复杂临床场景的能力。实验结果表明，集成SAM的基础模型在跨数据集的注册性能上取得了显著提升，为医学图像注册技术的进步提供了新的见解。我们的代码可在此处访问。 

---
# PRUNE: A Patching Based Repair Framework for Certiffable Unlearning of Neural Networks 

**Title (ZH)**: PRUNE: 基于 patches 的可验证遗忘神经网络修复框架 

**Authors**: Xuran Li, Jingyi Wang, Xiaohan Yuan, Peixin Zhang, Zhan Qin, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.06520)  

**Abstract**: It is often desirable to remove (a.k.a. unlearn) a speciffc part of the training data from a trained neural network model. A typical application scenario is to protect the data holder's right to be forgotten, which has been promoted by many recent regulation rules. Existing unlearning methods involve training alternative models with remaining data, which may be costly and challenging to verify from the data holder or a thirdparty auditor's perspective. In this work, we provide a new angle and propose a novel unlearning approach by imposing carefully crafted "patch" on the original neural network to achieve targeted "forgetting" of the requested data to delete. Speciffcally, inspired by the research line of neural network repair, we propose to strategically seek a lightweight minimum "patch" for unlearning a given data point with certiffable guarantee. Furthermore, to unlearn a considerable amount of data points (or an entire class), we propose to iteratively select a small subset of representative data points to unlearn, which achieves the effect of unlearning the whole set. Extensive experiments on multiple categorical datasets demonstrates our approach's effectiveness, achieving measurable unlearning while preserving the model's performance and being competitive in efffciency and memory consumption compared to various baseline methods. 

**Abstract (ZH)**: 从训练神经网络模型中移除特定部分的训练数据往往是有益的（即去学习）。一个典型的应用场景是保护数据持有者的被遗忘权，这已被许多最近的法规推广。现有的去学习方法涉及使用剩余数据训练替代模型，这从数据持有者或第三方审核员的角度来看可能是成本高且难以验证的。在本工作中，我们提供了一个新的视角，并提出了一种新的去学习方法，通过对原始神经网络施加精心设计的“补丁”以实现对指定数据的“遗忘”。具体而言，受神经网络修复研究方向的启发，我们提出了一种战略上寻找轻量级最小“补丁”来去学习给定数据点，同时具有可验证的保证。此外，为了去学习大量数据点（或整个类别），我们提出迭代选择一小部分有代表性的数据点来去学习，从而实现对整个数据集去学习的效果。在多个分类数据集上的广泛实验表明，我们的方法在提高去学习效果的同时保持了模型性能，并且在效率和内存消耗方面与各种基线方法相比具有竞争力。 

---
# Attention Mechanisms in Dynamical Systems: A Case Study with Predator-Prey Models 

**Title (ZH)**: 动态系统中的注意力机制：以捕食者-猎物模型为例 

**Authors**: David Balaban  

**Link**: [PDF](https://arxiv.org/pdf/2505.06503)  

**Abstract**: Attention mechanisms are widely used in artificial intelligence to enhance performance and interpretability. In this paper, we investigate their utility in modeling classical dynamical systems -- specifically, a noisy predator-prey (Lotka-Volterra) system. We train a simple linear attention model on perturbed time-series data to reconstruct system trajectories. Remarkably, the learned attention weights align with the geometric structure of the Lyapunov function: high attention corresponds to flat regions (where perturbations have small effect), and low attention aligns with steep regions (where perturbations have large effect). We further demonstrate that attention-based weighting can serve as a proxy for sensitivity analysis, capturing key phase-space properties without explicit knowledge of the system equations. These results suggest a novel use of AI-derived attention for interpretable, data-driven analysis and control of nonlinear systems. For example our framework could support future work in biological modeling of circadian rhythms, and interpretable machine learning for dynamical environments. 

**Abstract (ZH)**: 注意力机制在增强人工智能性能和可解释性方面广泛应用。本文探讨了其在建模经典动力系统中的作用——具体而言，是噪声扰动的捕食者-猎物（Lotka-Volterra）系统。我们训练一个简单的线性注意力模型来重构系统轨迹。值得注意的是，学习到的注意力权重与李雅普诺夫函数的几何结构相一致：高注意力对应平坦区域（扰动影响小），低注意力对应陡峭区域（扰动影响大）。此外，我们还展示了基于注意力的加权可以作为灵敏度分析的代理，无需显式了解系统方程即可捕捉相空间的关键特性。这些结果表明，AI提取的注意力在非线性系统可解释的数据驱动分析和控制中具有新颖的应用。例如，我们的框架可以支持未来关于生物节奏建模的工作，以及动态环境中的可解释机器学习。 

---
# xGen-small Technical Report 

**Title (ZH)**: xGen-small 技术报告 

**Authors**: Erik Nijkamp, Bo Pang, Egor Pakhomov, Akash Gokul, Jin Qu, Silvio Savarese, Yingbo Zhou, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06496)  

**Abstract**: We introduce xGen-small, a family of 4B and 9B Transformer decoder models optimized for long-context applications. Our vertically integrated pipeline unites domain-balanced, frequency-aware data curation; multi-stage pre-training with quality annealing and length extension to 128k tokens; and targeted post-training via supervised fine-tuning, preference learning, and online reinforcement learning. xGen-small delivers strong performance across various tasks, especially in math and coding domains, while excelling at long context benchmarks. 

**Abstract (ZH)**: xGen-small: 一种优化用于长上下文应用的4B和9B Transformer解码器模型 

---
# System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection 

**Title (ZH)**: 系统提示中毒：超出用户注入的大语言模型持续攻击 

**Authors**: Jiawei Guo, Haipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06493)  

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其强大的生成能力在各种应用中得到了广泛应用。它们的即插即用特性使得开发者和最终用户可以通过简单的提示与这些模型进行交互。然而，随着LLMs在各个领域中越来越多地集成到各种系统中，对其安全性的担忧也在增长。现有研究主要关注来自用户提示（如提示注入攻击）和模型输出（如模型反向工程攻击）的威胁，而系统提示的安全性则被很大程度上忽视。本工作填补了这一关键空白。我们介绍了系统提示投毒这一新的攻击向量，与传统的用户提示注入不同，系统提示投毒攻击旨在持久影响所有后续的用户交互和模型响应。我们系统地研究了四种实用的攻击策略在各种投毒场景下的表现。通过在生成和推理两种大型语言模型上的演示，我们表明，系统提示投毒在无需使用越狱技术的情况下是高度可行的，并且在广泛的任务中具有有效性，包括数学、编程、逻辑推理和自然语言处理任务。重要的是，我们的结果揭示出，即使用户提示使用了链式思考（CoT）等高级提示技术，攻击仍然有效。我们还展示了这些技术，包括CoT和检索增强生成（RAG），在许多任务中已被证明能够显著提高大型语言模型的性能，但通过系统提示投毒却显著削弱了其有效性。 

---
# Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach 

**Title (ZH)**: 基于模型的视频增强离线强化学习 

**Authors**: Minting Pan, Yitao Zheng, Jiajian Li, Yunbo Wang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06482)  

**Abstract**: Offline reinforcement learning (RL) enables policy optimization in static datasets, avoiding the risks and costs of real-world exploration. However, it struggles with suboptimal behavior learning and inaccurate value estimation due to the lack of environmental interaction. In this paper, we present Video-Enhanced Offline RL (VeoRL), a model-based approach that constructs an interactive world model from diverse, unlabeled video data readily available online. Leveraging model-based behavior guidance, VeoRL transfers commonsense knowledge of control policy and physical dynamics from natural videos to the RL agent within the target domain. Our method achieves substantial performance gains (exceeding 100% in some cases) across visuomotor control tasks in robotic manipulation, autonomous driving, and open-world video games. 

**Abstract (ZH)**: 视频增强的离线强化学习（VeoRL） 

---
# Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles 

**Title (ZH)**: 使用误差界和解集改进物理引导神经网络中的不确定性量化 

**Authors**: Pablo Flores, Olga Graf, Pavlos Protopapas, Karim Pichara  

**Link**: [PDF](https://arxiv.org/pdf/2505.06459)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have been widely used to obtain solutions to various physical phenomena modeled as Differential Equations. As PINNs are not naturally equipped with mechanisms for Uncertainty Quantification, some work has been done to quantify the different uncertainties that arise when dealing with PINNs. In this paper, we use a two-step procedure to train Bayesian Neural Networks that provide uncertainties over the solutions to differential equation systems provided by PINNs. We use available error bounds over PINNs to formulate a heteroscedastic variance that improves the uncertainty estimation. Furthermore, we solve forward problems and utilize the obtained uncertainties when doing parameter estimation in inverse problems in cosmology. 

**Abstract (ZH)**: 物理导向的神经网络（PINNs）已经被广泛应用于解决各种由微分方程描述的物理现象。由于PINNs本身不具备不确定性量化机制，一些工作致力于量化处理PINNs时出现的不同不确定性。在本文中，我们采用两步训练程序构建贝叶斯神经网络，为PINNs提供的微分方程系统解提供不确定性。我们利用PINNs的可用误差边界来制定异方差方差，以改进不确定性估计。此外，我们在宇宙学中利用获得的不确定性解决正向问题，并进行参数估计的逆向问题。 

---
# My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing 

**Title (ZH)**: 你在脸上的情绪：面部关键点检测在潜在空间编辑中保留情绪的应用 

**Authors**: Jingrui He, Andrew Stephen McGough  

**Link**: [PDF](https://arxiv.org/pdf/2505.06436)  

**Abstract**: Generative Adversarial Network approaches such as StyleGAN/2 provide two key benefits: the ability to generate photo-realistic face images and possessing a semantically structured latent space from which these images are created. Many approaches have emerged for editing images derived from vectors in the latent space of a pre-trained StyleGAN/2 models by identifying semantically meaningful directions (e.g., gender or age) in the latent space. By moving the vector in a specific direction, the ideal result would only change the target feature while preserving all the other features. Providing an ideal data augmentation approach for gesture research as it could be used to generate numerous image variations whilst keeping the facial expressions intact. However, entanglement issues, where changing one feature inevitably affects other features, impacts the ability to preserve facial expressions. To address this, we propose the use of an addition to the loss function of a Facial Keypoint Detection model to restrict changes to the facial expressions. Building on top of an existing model, adding the proposed Human Face Landmark Detection (HFLD) loss, provided by a pre-trained Facial Keypoint Detection model, to the original loss function. We quantitatively and qualitatively evaluate the existing and our extended model, showing the effectiveness of our approach in addressing the entanglement issue and maintaining the facial expression. Our approach achieves up to 49% reduction in the change of emotion in our experiments. Moreover, we show the benefit of our approach by comparing with state-of-the-art models. By increasing the ability to preserve the facial gesture and expression during facial transformation, we present a way to create human face images with fixed expression but different appearances, making it a reliable data augmentation approach for Facial Gesture and Expression research. 

**Abstract (ZH)**: 生成对抗网络方法，如StyleGAN/2，提供两项关键优势：生成照片级真实的人脸图像和拥有一个语义上结构化的潜在空间，从该空间中生成这些图像。通过在预训练StyleGAN/2模型的潜在空间中的向量中识别语义上有意义的方向（例如性别或年龄），已出现了许多图像编辑方法。通过在特定方向上移动向量，理想的结果是仅改变目标特征同时保留所有其他特征。这为手势研究提供了一种理想的數據增强方法，因为可以生成大量图像变体同时保持面部表情不变。然而，特征纠缠问题（改变一个特征不可避免地会影响其他特征）影响了保持面部表情的能力。为此，我们提出在面部关键点检测模型的损失函数中添加一项，以限制对面部表情的更改。基于现有模型，将预训练面部关键点检测模型提供的提议的人脸地标检测（HFLD）损失添加到原始损失函数中。我们从定量和定性两方面评估现有和扩展后的模型，展示了我们的方法在解决特征纠缠问题和保持面部表情方面的有效性。在我们的实验中，我们的方法实现了最多49%的情绪变化减少。此外，我们通过与最先进的模型进行对比，展示了我们方法的优势。通过增强在面部变换过程中保持面部手势和表情的能力，我们提出了一种方法，可以创建具有固定表情但不同外观的人脸图像，使之成为面部手势和表情研究的可靠数据增强方法。 

---
# What Do People Want to Know About Artificial Intelligence (AI)? The Importance of Answering End-User Questions to Explain Autonomous Vehicle (AV) Decisions 

**Title (ZH)**: 人们最想了解的人工智能（AI）是什么？解答最终用户的问题以解释自动驾驶车辆（AV）的决策的重要性 

**Authors**: Somayeh Molaei, Lionel P. Robert, Nikola Banovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.06428)  

**Abstract**: Improving end-users' understanding of decisions made by autonomous vehicles (AVs) driven by artificial intelligence (AI) can improve utilization and acceptance of AVs. However, current explanation mechanisms primarily help AI researchers and engineers in debugging and monitoring their AI systems, and may not address the specific questions of end-users, such as passengers, about AVs in various scenarios. In this paper, we conducted two user studies to investigate questions that potential AV passengers might pose while riding in an AV and evaluate how well answers to those questions improve their understanding of AI-driven AV decisions. Our initial formative study identified a range of questions about AI in autonomous driving that existing explanation mechanisms do not readily address. Our second study demonstrated that interactive text-based explanations effectively improved participants' comprehension of AV decisions compared to simply observing AV decisions. These findings inform the design of interactions that motivate end-users to engage with and inquire about the reasoning behind AI-driven AV decisions. 

**Abstract (ZH)**: 提高最终用户对由人工智能驱动的自主车辆（AVs）所做的决策的理解可以提高自主车辆的使用率和接受度。然而，当前的解释机制主要有助于人工智能研究人员和工程师调试和监控其人工智能系统，可能无法解决最终用户（如乘客）在各种场景中对自主车辆的具体问题。本文通过两项用户研究调查了潜在自主车辆乘客在乘坐自主车辆时可能会提出的问题，并评估了这些问题的回答如何提高他们对人工智能驱动的自主车辆决策的理解。初步形成性研究确定了一类现有解释机制无法轻松解答的关于自主驾驶中人工智能的问题。第二项研究证明，互动的基于文本的解释比仅观察自主车辆的决策更有效地提高了参与者对自主车辆决策的理解。这些发现为设计能够促使最终用户参与并了解人工智能驱动的自主车辆决策原因的交互界面提供了指导。 

---
# Natural Reflection Backdoor Attack on Vision Language Model for Autonomous Driving 

**Title (ZH)**: 自然反射后门攻击对自动驾驶视觉语言模型的影响 

**Authors**: Ming Liu, Siyuan Liang, Koushik Howlader, Liwen Wang, Dacheng Tao, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06413)  

**Abstract**: Vision-Language Models (VLMs) have been integrated into autonomous driving systems to enhance reasoning capabilities through tasks such as Visual Question Answering (VQA). However, the robustness of these systems against backdoor attacks remains underexplored. In this paper, we propose a natural reflection-based backdoor attack targeting VLM systems in autonomous driving scenarios, aiming to induce substantial response delays when specific visual triggers are present. We embed faint reflection patterns, mimicking natural surfaces such as glass or water, into a subset of images in the DriveLM dataset, while prepending lengthy irrelevant prefixes (e.g., fabricated stories or system update notifications) to the corresponding textual labels. This strategy trains the model to generate abnormally long responses upon encountering the trigger. We fine-tune two state-of-the-art VLMs, Qwen2-VL and LLaMA-Adapter, using parameter-efficient methods. Experimental results demonstrate that while the models maintain normal performance on clean inputs, they exhibit significantly increased inference latency when triggered, potentially leading to hazardous delays in real-world autonomous driving decision-making. Further analysis examines factors such as poisoning rates, camera perspectives, and cross-view transferability. Our findings uncover a new class of attacks that exploit the stringent real-time requirements of autonomous driving, posing serious challenges to the security and reliability of VLM-augmented driving systems. 

**Abstract (ZH)**: 基于自然反射的视觉语言模型后门攻击：针对自动驾驶场景的响应延迟诱导 

---
# MAGE:A Multi-stage Avatar Generator with Sparse Observations 

**Title (ZH)**: MAGE：一种基于稀疏观测的多阶段_avatar生成器 

**Authors**: Fangyu Du, Yang Yang, Xuehao Gao, Hongye Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06411)  

**Abstract**: Inferring full-body poses from Head Mounted Devices, which capture only 3-joint observations from the head and wrists, is a challenging task with wide AR/VR applications. Previous attempts focus on learning one-stage motion mapping and thus suffer from an over-large inference space for unobserved body joint motions. This often leads to unsatisfactory lower-body predictions and poor temporal consistency, resulting in unrealistic or incoherent motion sequences. To address this, we propose a powerful Multi-stage Avatar GEnerator named MAGE that factorizes this one-stage direct motion mapping learning with a progressive prediction strategy. Specifically, given initial 3-joint motions, MAGE gradually inferring multi-scale body part poses at different abstract granularity levels, starting from a 6-part body representation and gradually refining to 22 joints. With decreasing abstract levels step by step, MAGE introduces more motion context priors from former prediction stages and thus improves realistic motion completion with richer constraint conditions and less ambiguity. Extensive experiments on large-scale datasets verify that MAGE significantly outperforms state-of-the-art methods with better accuracy and continuity. 

**Abstract (ZH)**: 从头戴设备推断全身体态：一种逐步Avatar生成器MAGE的方法 

---
# Engineering Risk-Aware, Security-by-Design Frameworks for Assurance of Large-Scale Autonomous AI Models 

**Title (ZH)**: 为大规模自主AI模型提供保障的风险意识与设计安全框架 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2505.06409)  

**Abstract**: As AI models scale to billions of parameters and operate with increasing autonomy, ensuring their safe, reliable operation demands engineering-grade security and assurance frameworks. This paper presents an enterprise-level, risk-aware, security-by-design approach for large-scale autonomous AI systems, integrating standardized threat metrics, adversarial hardening techniques, and real-time anomaly detection into every phase of the development lifecycle. We detail a unified pipeline - from design-time risk assessments and secure training protocols to continuous monitoring and automated audit logging - that delivers provable guarantees of model behavior under adversarial and operational stress. Case studies in national security, open-source model governance, and industrial automation demonstrate measurable reductions in vulnerability and compliance overhead. Finally, we advocate cross-sector collaboration - uniting engineering teams, standards bodies, and regulatory agencies - to institutionalize these technical safeguards within a resilient, end-to-end assurance ecosystem for the next generation of AI. 

**Abstract (ZH)**: 随着AI模型参数数量达到十亿级别并展现出越来越高的自主性，确保其安全可靠的运行需要工程级别的安全和保证框架。本文提出了一种面向企业的、具备风险意识的设计安全方法，用于大规模自主AI系统，该方法将标准化威胁度量、对抗性强化技术和实时异常检测整合到开发生命周期的每一个阶段。我们详细阐述了一条统一的管线——从设计时的风险评估和安全训练协议到持续监控和自动审计日志记录——以在对抗和运营压力下提供可验证的模型行为保证。在国家安全、开源模型治理和工业自动化领域的案例研究证明了可测量的漏洞和合规开销减少。最后，我们倡导跨领域的合作——结合工程团队、标准机构和监管机构——在具有韧性的端到端保证生态系统中制度化这些技术保护措施，以为下一代AI提供服务。 

---
# Camera Control at the Edge with Language Models for Scene Understanding 

**Title (ZH)**: 边缘节点上的相机控制以语言模型实现场景理解 

**Authors**: Alexiy Buynitsky, Sina Ehsani, Bhanu Pallakonda, Pragyana Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.06402)  

**Abstract**: In this paper, we present Optimized Prompt-based Unified System (OPUS), a framework that utilizes a Large Language Model (LLM) to control Pan-Tilt-Zoom (PTZ) cameras, providing contextual understanding of natural environments. To achieve this goal, the OPUS system improves cost-effectiveness by generating keywords from a high-level camera control API and transferring knowledge from larger closed-source language models to smaller ones through Supervised Fine-Tuning (SFT) on synthetic data. This enables efficient edge deployment while maintaining performance comparable to larger models like GPT-4. OPUS enhances environmental awareness by converting data from multiple cameras into textual descriptions for language models, eliminating the need for specialized sensory tokens. In benchmark testing, our approach significantly outperformed both traditional language model techniques and more complex prompting methods, achieving a 35% improvement over advanced techniques and a 20% higher task accuracy compared to closed-source models like Gemini Pro. The system demonstrates OPUS's capability to simplify PTZ camera operations through an intuitive natural language interface. This approach eliminates the need for explicit programming and provides a conversational method for interacting with camera systems, representing a significant advancement in how users can control and utilize PTZ camera technology. 

**Abstract (ZH)**: 本文提出了优化提示统一系统（OPUS），这是一种利用大规模语言模型（LLM）控制PTZ摄像头的框架，提供对自然环境的上下文理解。为了实现这一目标，OPUS系统通过从高级摄像头控制API生成关键词，并通过监督微调（SFT）在合成数据上的训练，将大型专有语言模型的知识转移到较小的模型上，从而提高成本效益，实现高效的边缘部署，同时保持与GPT-4等大型模型相当的性能。OPUS通过将多个摄像头的数据转换为文本描述以供语言模型使用，增强了环境意识，消除了对专门感知识别标记的需求。在基准测试中，我们的方法显著优于传统的语言模型技术以及更为复杂的提示方法，相较于高级技术实现了35%的改进，并且与Gemini Pro等专有模型相比，任务准确率提高了20%。该系统展示了OPUS通过直观的自然语言界面简化PTZ摄像头操作的能力，这一方法消除了显式编程的需求，并提供了一种与摄像头系统交互的对话方式，代表了用户控制和利用PTZ摄像头技术的一个重要进步。 

---
# Towards AI-Driven Human-Machine Co-Teaming for Adaptive and Agile Cyber Security Operation Centers 

**Title (ZH)**: 面向人工智能驱动的人机协同操作的自适应敏捷网络安全运营中心 

**Authors**: Massimiliano Albanese, Xinming Ou, Kevin Lybarger, Daniel Lende, Dmitry Goldgof  

**Link**: [PDF](https://arxiv.org/pdf/2505.06394)  

**Abstract**: Security Operations Centers (SOCs) face growing challenges in managing cybersecurity threats due to an overwhelming volume of alerts, a shortage of skilled analysts, and poorly integrated tools. Human-AI collaboration offers a promising path to augment the capabilities of SOC analysts while reducing their cognitive overload. To this end, we introduce an AI-driven human-machine co-teaming paradigm that leverages large language models (LLMs) to enhance threat intelligence, alert triage, and incident response workflows. We present a vision in which LLM-based AI agents learn from human analysts the tacit knowledge embedded in SOC operations, enabling the AI agents to improve their performance on SOC tasks through this co-teaming. We invite SOCs to collaborate with us to further develop this process and uncover replicable patterns where human-AI co-teaming yields measurable improvements in SOC productivity. 

**Abstract (ZH)**: 安全运营中心（SOC）面临日益严峻的网络安全威胁管理挑战，由于警报量巨大、熟练分析师短缺以及工具集成不良。人机协作为增强SOC分析师能力、减轻其认知负担提供了前景。为此，我们提出了一种基于AI的人机协同训练 paradigm，利用大型语言模型（LLMs）提升威胁情报、警报分诊和事件响应工作流程。我们提出了一种愿景：基于LLM的AI代理从SOC分析师那里学习隐含在SOC运营中的 tacit 知识，使AI代理能够通过这种人机协作提高其在SOC任务上的表现。我们邀请SOC与我们合作，进一步开发这一过程，并发现人机协同训练在提高SOC生产率方面可复制的模式。 

---
# Offensive Security for AI Systems: Concepts, Practices, and Applications 

**Title (ZH)**: AI系统进攻性安全：概念、实践与应用 

**Authors**: Josh Harguess, Chris M. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.06380)  

**Abstract**: As artificial intelligence (AI) systems become increasingly adopted across sectors, the need for robust, proactive security strategies is paramount. Traditional defensive measures often fall short against the unique and evolving threats facing AI-driven technologies, making offensive security an essential approach for identifying and mitigating risks. This paper presents a comprehensive framework for offensive security in AI systems, emphasizing proactive threat simulation and adversarial testing to uncover vulnerabilities throughout the AI lifecycle. We examine key offensive security techniques, including weakness and vulnerability assessment, penetration testing, and red teaming, tailored specifically to address AI's unique susceptibilities. By simulating real-world attack scenarios, these methodologies reveal critical insights, informing stronger defensive strategies and advancing resilience against emerging threats. This framework advances offensive AI security from theoretical concepts to practical, actionable methodologies that organizations can implement to strengthen their AI systems against emerging threats. 

**Abstract (ZH)**: 随着人工智能（AI）系统在各领域的广泛应用，建立 robust、 proactive 的安全策略变得至关重要。传统防御措施往往无法有效应对 AI 驱动技术所面临的独特且不断演变的威胁，因此，采取主动防御安全策略以识别和减轻风险变得必不可少。本文提出了一个全面的 AI 系统主动防御安全框架，强调在 AI 生命周期中进行积极的威胁模拟和对抗性测试以揭示漏洞。我们探讨了关键的主动防御安全技术，包括脆弱性和漏洞评估、渗透测试和红队演练，这些技术特别针对解决 AI 的独特脆弱性进行了定制。通过模拟真实世界的攻击场景，这些方法论揭示了关键见解，为制定更强的防御策略并提高对新兴威胁的抵御能力奠定了基础。该框架将主动 AI 安全从理论概念推进到可操作的实际方法论，为企业提供实施以加强其 AI 系统抵御新兴威胁的策略。 

---
# Bi-LSTM based Multi-Agent DRL with Computation-aware Pruning for Agent Twins Migration in Vehicular Embodied AI Networks 

**Title (ZH)**: 基于Bi-LSTM的考虑计算量的多Agent DRL与代理双胞胎迁移修剪方法在车载实体AI网络中的应用 

**Authors**: Yuxiang Wei, Zhuoqi Zeng, Yue Zhong, Jiawen Kang, Ryan Wen Liu, M. Shamim Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.06378)  

**Abstract**: With the advancement of large language models and embodied Artificial Intelligence (AI) in the intelligent transportation scenarios, the combination of them in intelligent transportation spawns the Vehicular Embodied AI Network (VEANs). In VEANs, Autonomous Vehicles (AVs) are typical agents whose local advanced AI applications are defined as vehicular embodied AI agents, enabling capabilities such as environment perception and multi-agent collaboration. Due to computation latency and resource constraints, the local AI applications and services running on vehicular embodied AI agents need to be migrated, and subsequently referred to as vehicular embodied AI agent twins, which drive the advancement of vehicular embodied AI networks to offload intensive tasks to Roadside Units (RSUs), mitigating latency problems while maintaining service quality. Recognizing workload imbalance among RSUs in traditional approaches, we model AV-RSU interactions as a Stackelberg game to optimize bandwidth resource allocation for efficient migration. A Tiny Multi-Agent Bidirectional LSTM Proximal Policy Optimization (TMABLPPO) algorithm is designed to approximate the Stackelberg equilibrium through decentralized coordination. Furthermore, a personalized neural network pruning algorithm based on Path eXclusion (PX) dynamically adapts to heterogeneous AV computation capabilities by identifying task-critical parameters in trained models, reducing model complexity with less performance degradation. Experimental validation confirms the algorithm's effectiveness in balancing system load and minimizing delays, demonstrating significant improvements in vehicular embodied AI agent deployment. 

**Abstract (ZH)**: 基于大规模语言模型和具身人工智能的智能交通场景下具身人工智能网络（VEANs）：自治车辆与道路侧单元的工作负载优化 

---
# The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization 

**Title (ZH)**: ML.ENERGY基准：迈向自动化推理能耗测量与优化 

**Authors**: Jae-Won Chung, Jiachen Liu, Jeff J. Ma, Ruofan Wu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.06371)  

**Abstract**: As the adoption of Generative AI in real-world services grow explosively, energy has emerged as a critical bottleneck resource. However, energy remains a metric that is often overlooked, under-explored, or poorly understood in the context of building ML systems. We present the this http URL Benchmark, a benchmark suite and tool for measuring inference energy consumption under realistic service environments, and the corresponding this http URL Leaderboard, which have served as a valuable resource for those hoping to understand and optimize the energy consumption of their generative AI services. In this paper, we explain four key design principles for benchmarking ML energy we have acquired over time, and then describe how they are implemented in the this http URL Benchmark. We then highlight results from the latest iteration of the benchmark, including energy measurements of 40 widely used model architectures across 6 different tasks, case studies of how ML design choices impact energy consumption, and how automated optimization recommendations can lead to significant (sometimes more than 40%) energy savings without changing what is being computed by the model. The this http URL Benchmark is open-source and can be easily extended to various customized models and application scenarios. 

**Abstract (ZH)**: 随着生成式AI在实际服务中的应用爆炸式增长，能源已成为一个关键的瓶颈资源。然而，在构建机器学习系统的过程中，能源仍然是一个常被忽视、未充分探索或不完全理解的指标。我们介绍了this http URL基准测试，这是一个用于在实际服务环境中测量推理能耗的基准套件和工具，以及相应的this http URL排行榜，它们已成为希望了解并优化其生成式AI服务能耗的研究人员的重要资源。在本文中，我们解释了我们在长期研究中获得的四个关键设计原则，并描述了这些原则在this http URL基准测试中的实现方式。然后，我们强调了基准测试最新迭代的结果，包括40种广泛使用的模型架构在6种不同任务上的能耗测量、ML设计选择对能耗影响的案例研究，以及自动优化建议如何在不改变模型计算内容的情况下实现显著（有时超过40%）的能耗节省。this http URL基准测试是开源的，并且可以轻松扩展到各种定制模型和应用场景。 

---
# Learning Sequential Kinematic Models from Demonstrations for Multi-Jointed Articulated Objects 

**Title (ZH)**: 基于演示学习多关节 articulated 对象的序列运动学模型 

**Authors**: Anmol Gupta, Weiwei Gu, Omkar Patil, Jun Ki Lee, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06363)  

**Abstract**: As robots become more generalized and deployed in diverse environments, they must interact with complex objects, many with multiple independent joints or degrees of freedom (DoF) requiring precise control. A common strategy is object modeling, where compact state-space models are learned from real-world observations and paired with classical planning. However, existing methods often rely on prior knowledge or focus on single-DoF objects, limiting their applicability. They also fail to handle occluded joints and ignore the manipulation sequences needed to access them. We address this by learning object models from human demonstrations. We introduce Object Kinematic Sequence Machines (OKSMs), a novel representation capturing both kinematic constraints and manipulation order for multi-DoF objects. To estimate these models from point cloud data, we present Pokenet, a deep neural network trained on human demonstrations. We validate our approach on 8,000 simulated and 1,600 real-world annotated samples. Pokenet improves joint axis and state estimation by over 20 percent on real-world data compared to prior methods. Finally, we demonstrate OKSMs on a Sawyer robot using inverse kinematics-based planning to manipulate multi-DoF objects. 

**Abstract (ZH)**: 随着机器人在多样化环境中变得更加通用，它们必须与复杂对象交互，这些对象往往具有多个独立关节或自由度，需要精确控制。一种常见策略是对象建模，即从实际观察中学习紧凑的状态空间模型，并与经典规划相结合。然而，现有方法往往依赖先验知识或专注于单自由度对象，这限制了其适用性。它们也无法处理被遮挡的关节，并且忽略了解析它们所需的操控序列。我们通过从人类演示中学习对象模型来解决这个问题。我们提出了对象运动序列机（OKSMs），这是一种新颖的表现形式，可以捕捉多自由度对象的运动约束及其操作顺序。为了从点云数据中估算这些模型，我们呈现了Pokenet，这是一种基于人类演示训练的深度神经网络。我们在8,000个模拟和1,600个实际标注样本上验证了该方法的有效性。与先前方法相比，Pokenet在实际数据中的关节轴线和状态估计提高了超过20%。最后，我们在Sawyer机器人上展示了OKSMs，通过基于逆向运动学的规划来操控多自由度对象。 

---
# Quantum State Preparation via Large-Language-Model-Driven Evolution 

**Title (ZH)**: 大型语言模型驱动的量子态准备方法 

**Authors**: Qing-Hong Cao, Zong-Yue Hou, Ying-Ying Li, Xiaohui Liu, Zhuo-Yang Song, Liang-Qi Zhang, Shutao Zhang, Ke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06347)  

**Abstract**: We propose an automated framework for quantum circuit design by integrating large-language models (LLMs) with evolutionary optimization to overcome the rigidity, scalability limitations, and expert dependence of traditional ones in variational quantum algorithms. Our approach (FunSearch) autonomously discovers hardware-efficient ansätze with new features of scalability and system-size-independent number of variational parameters entirely from scratch. Demonstrations on the Ising and XY spin chains with n = 9 qubits yield circuits containing 4 parameters, achieving near-exact energy extrapolation across system sizes. Implementations on quantum hardware (Zuchongzhi chip) validate practicality, where two-qubit quantum gate noises can be effectively mitigated via zero-noise extrapolations for a spin chain system as large as 20 sites. This framework bridges algorithmic design and experimental constraints, complementing contemporary quantum architecture search frameworks to advance scalable quantum simulations. 

**Abstract (ZH)**: 一种将大型语言模型与进化优化集成的自动化量子电路设计框架：FunSearch方法及其在可扩展量子模拟中的应用 

---
# Remote Rowhammer Attack using Adversarial Observations on Federated Learning Clients 

**Title (ZH)**: 面向联邦学习客户端的基于对抗观察的远程Rowhammer攻击 

**Authors**: Jinsheng Yuan, Yuhang Hao, Weisi Guo, Yun Wu, Chongyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06335)  

**Abstract**: Federated Learning (FL) has the potential for simultaneous global learning amongst a large number of parallel agents, enabling emerging AI such as LLMs to be trained across demographically diverse data. Central to this being efficient is the ability for FL to perform sparse gradient updates and remote direct memory access at the central server. Most of the research in FL security focuses on protecting data privacy at the edge client or in the communication channels between the client and server. Client-facing attacks on the server are less well investigated as the assumption is that a large collective of clients offer resilience.
Here, we show that by attacking certain clients that lead to a high frequency repetitive memory update in the server, we can remote initiate a rowhammer attack on the server memory. For the first time, we do not need backdoor access to the server, and a reinforcement learning (RL) attacker can learn how to maximize server repetitive memory updates by manipulating the client's sensor observation. The consequence of the remote rowhammer attack is that we are able to achieve bit flips, which can corrupt the server memory. We demonstrate the feasibility of our attack using a large-scale FL automatic speech recognition (ASR) systems with sparse updates, our adversarial attacking agent can achieve around 70\% repeated update rate (RUR) in the targeted server model, effectively inducing bit flips on server DRAM. The security implications are that can cause disruptions to learning or may inadvertently cause elevated privilege. This paves the way for further research on practical mitigation strategies in FL and hardware design. 

**Abstract (ZH)**: 联邦学习中的远程行hammer攻击研究 

---
# NSF-MAP: Neurosymbolic Multimodal Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines 

**Title (ZH)**: NSF-MAP: 基于神经符号多模态融合的装配流水线稳健可解释异常预测 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Fadi El Kalach, Revathy Venkataramanan, Ramtin Zand, Ramy Harik, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06333)  

**Abstract**: In modern assembly pipelines, identifying anomalies is crucial in ensuring product quality and operational efficiency. Conventional single-modality methods fail to capture the intricate relationships required for precise anomaly prediction in complex predictive environments with abundant data and multiple modalities. This paper proposes a neurosymbolic AI and fusion-based approach for multimodal anomaly prediction in assembly pipelines. We introduce a time series and image-based fusion model that leverages decision-level fusion techniques. Our research builds upon three primary novel approaches in multimodal learning: time series and image-based decision-level fusion modeling, transfer learning for fusion, and knowledge-infused learning. We evaluate the novel method using our derived and publicly available multimodal dataset and conduct comprehensive ablation studies to assess the impact of our preprocessing techniques and fusion model compared to traditional baselines. The results demonstrate that a neurosymbolic AI-based fusion approach that uses transfer learning can effectively harness the complementary strengths of time series and image data, offering a robust and interpretable approach for anomaly prediction in assembly pipelines with enhanced performance. \noindent The datasets, codes to reproduce the results, supplementary materials, and demo are available at this https URL. 

**Abstract (ZH)**: 现代装配生产线中多模态异常检测的关键性及其神经符号AI和融合方法研究 

---
# Mask-PINNs: Regulating Feature Distributions in Physics-Informed Neural Networks 

**Title (ZH)**: Mask-PINNs：调节物理信息神经网络中特征分布 

**Authors**: Feilong Jiang, Xiaonan Hou, Jianqiao Ye, Min Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06331)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve partial differential equations by incorporating physical laws directly into the loss function. However, the internal covariate shift, which has been largely overlooked, hinders the effective utilization of neural network capacity in PINNs. To this end, we propose Mask-PINNs, a novel architecture designed to address this issue in PINNs. Unlike traditional normalization methods such as BatchNorm or LayerNorm, we introduce a learnable, nonlinear mask function that constrains the feature distributions without violating underlying physics. The experimental results show that the proposed method significantly improves feature distribution stability, accuracy, and robustness across various activation functions and PDE benchmarks. Furthermore, it enables the stable and efficient training of wider networks a capability that has been largely overlooked in PINNs. 

**Abstract (ZH)**: 基于物理的神经网络（Mask-PINNs）：一种解决内部协变移位问题的新架构 

---
# Prompting Large Language Models for Training-Free Non-Intrusive Load Monitoring 

**Title (ZH)**: prompting大型语言模型进行无需训练的非侵入式负荷监测 

**Authors**: Junyu Xue, Xudong Wang, Xiaoling He, Shicheng Liu, Yi Wang, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06330)  

**Abstract**: Non-intrusive Load Monitoring (NILM) aims to disaggregate aggregate household electricity consumption into individual appliance usage, enabling more effective energy management. While deep learning has advanced NILM, it remains limited by its dependence on labeled data, restricted generalization, and lack of interpretability. In this paper, we introduce the first prompt-based NILM framework that leverages Large Language Models (LLMs) with in-context learning. We design and evaluate prompt strategies that integrate appliance features, timestamps and contextual information, as well as representative time-series examples, using the REDD dataset. With optimized prompts, LLMs achieve competitive state detection accuracy, reaching an average F1-score of 0.676 on unseen households, and demonstrate robust generalization without the need for fine-tuning. LLMs also enhance interpretability by providing clear, human-readable explanations for their predictions. Our results show that LLMs can reduce data requirements, improve adaptability, and provide transparent energy disaggregation in NILM applications. 

**Abstract (ZH)**: 基于提示的非侵入式负荷监测框架：利用大型语言模型实现高效能源管理 

---
# Enterprise Architecture as a Dynamic Capability for Scalable and Sustainable Generative AI adoption: Bridging Innovation and Governance in Large Organisations 

**Title (ZH)**: 企业架构作为可扩展和可持续生成式AI adoption的动态能力：在大型组织中bridging创新与治理 

**Authors**: Alexander Ettinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.06326)  

**Abstract**: Generative Artificial Intelligence is a powerful new technology with the potential to boost innovation and reshape governance in many industries. Nevertheless, organisations face major challenges in scaling GenAI, including technology complexity, governance gaps and resource misalignments. This study explores how Enterprise Architecture Management can meet the complex requirements of GenAI adoption within large enterprises. Based on a systematic literature review and the qualitative analysis of 16 semi-structured interviews with experts, it examines the relationships between EAM, dynamic capabilities and GenAI adoption. The review identified key limitations in existing EA frameworks, particularly their inability to fully address the unique requirements of GenAI. The interviews, analysed using the Gioia methodology, revealed critical enablers and barriers to GenAI adoption across industries. The findings indicate that EAM, when theorised as sensing, seizing and transforming dynamic capabilities, can enhance GenAI adoption by improving strategic alignment, governance frameworks and organisational agility. However, the study also highlights the need to tailor EA frameworks to GenAI-specific challenges, including low data governance maturity and the balance between innovation and compliance. Several conceptual frameworks are proposed to guide EA leaders in aligning GenAI maturity with organisational readiness. The work contributes to academic understanding and industry practice by clarifying the role of EA in bridging innovation and governance in disruptive technology environments. 

**Abstract (ZH)**: 生成式人工智能是一种强大的新技术，有可能促进创新并重塑许多行业的治理。然而，组织在扩大生成式人工智能的应用方面面临重大挑战，包括技术复杂性、治理缺口和资源配置不匹配。本研究探讨了企业架构管理如何满足大型企业中生成式人工智能采纳的复杂要求。基于系统文献综述及对16名专家进行的半结构化访谈的定性分析，研究了企业架构管理、动态能力和生成式人工智能采纳之间的关系。文献综述发现现有企业架构框架的关键局限性，特别是其无法充分解决生成式人工智能的独特要求。访谈结果（采用Goia方法分析）揭示了跨行业生成式人工智能采纳的关键促进因素和障碍。研究表明，当理论化为感知、捕获和转换动态能力时，企业架构管理可以提高生成式人工智能采纳的战略对齐、治理框架和组织敏捷性。然而，研究也指出了需要根据生成式人工智能的特定挑战来调整企业架构框架的需求，包括数据治理成熟度低以及创新与合规之间的平衡。提出了几个概念框架以指导企业架构领导者将生成式人工智能成熟度与组织准备度进行对齐。该工作通过阐明企业在颠覆性技术环境中促进创新和治理的角色，为学术研究和行业实践做出了贡献。 

---
# Human in the Latent Loop (HILL): Interactively Guiding Model Training Through Human Intuition 

**Title (ZH)**: 人类干预潜在循环（HILL）：通过人类直觉交互指导模型训练 

**Authors**: Daniel Geissler, Lars Krupp, Vishal Banwari, David Habusch, Bo Zhou, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2505.06325)  

**Abstract**: Latent space representations are critical for understanding and improving the behavior of machine learning models, yet they often remain obscure and intricate. Understanding and exploring the latent space has the potential to contribute valuable human intuition and expertise about respective domains. In this work, we present HILL, an interactive framework allowing users to incorporate human intuition into the model training by interactively reshaping latent space representations. The modifications are infused into the model training loop via a novel approach inspired by knowledge distillation, treating the user's modifications as a teacher to guide the model in reshaping its intrinsic latent representation. The process allows the model to converge more effectively and overcome inefficiencies, as well as provide beneficial insights to the user. We evaluated HILL in a user study tasking participants to train an optimal model, closely observing the employed strategies. The results demonstrated that human-guided latent space modifications enhance model performance while maintaining generalization, yet also revealing the risks of including user biases. Our work introduces a novel human-AI interaction paradigm that infuses human intuition into model training and critically examines the impact of human intervention on training strategies and potential biases. 

**Abstract (ZH)**: 隐空间表示对于理解并改进机器学习模型的行为至关重要，但往往晦涩难懂。理解和探索隐空间有可能为相应领域贡献宝贵的直觉和专业知识。在本工作中，我们提出了HILL，一个交互式框架，允许用户通过交互式重塑隐空间表示来将人类直觉融入模型训练中。通过借鉴知识蒸馏的方法，将用户的修改视为教师，指导模型重塑其固有的隐空间表示。该过程使模型能够更有效地收敛，克服效率低下，并为用户提供有益的见解。我们在一项用户研究中评估了HILL，要求参与者训练最优模型，并密切观察所采用的策略。结果表明，人类指导下的隐空间修改可以提升模型性能并保持泛化能力，但也揭示了纳入用户偏见的风险。我们的工作引入了一种新的人类-人工智能交互范式，将人类直觉融入模型训练，并批判性地探讨了人类干预对训练策略和潜在偏见的影响。 

---
# Document Attribution: Examining Citation Relationships using Large Language Models 

**Title (ZH)**: 文档归属性分析：使用大规模语言模型探究引用关系 

**Authors**: Vipula Rawte, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2505.06324)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied to document-based tasks - such as document summarization, question answering, and information extraction - where user requirements focus on retrieving information from provided documents rather than relying on the model's parametric knowledge, ensuring the trustworthiness and interpretability of these systems has become a critical concern. A central approach to addressing this challenge is attribution, which involves tracing the generated outputs back to their source documents. However, since LLMs can produce inaccurate or imprecise responses, it is crucial to assess the reliability of these citations.
To tackle this, our work proposes two techniques. (1) A zero-shot approach that frames attribution as a straightforward textual entailment task. Our method using flan-ul2 demonstrates an improvement of 0.27% and 2.4% over the best baseline of ID and OOD sets of AttributionBench, respectively. (2) We also explore the role of the attention mechanism in enhancing the attribution process. Using a smaller LLM, flan-t5-small, the F1 scores outperform the baseline across almost all layers except layer 4 and layers 8 through 11. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在文档任务中的应用与信任与可解释性的保障：一种新颖的归因方法及其关注点 

---
# Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Learning 

**Title (ZH)**: 基于图学习提升大语言模型推理能力：学会思考 

**Authors**: Hang Gao, Chenhao Zhang, Tie Wang, Junsuo Zhao, Fengge Wu, Changwen Zheng, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06321)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在多个领域取得了显著成功。然而，它们仍然面临重大挑战，包括高昂的训练计算成本以及解决复杂推理问题的局限性。尽管现有的方法通过结构化范式扩展了LLMs的推理能力，但这些方法往往依赖于特定任务的提示和预定义的推理过程，这限制了它们的灵活性和泛化能力。为解决这些局限性，我们提出了一种新的框架，利用图学习来增强LLMs的更灵活和自适应的推理能力。具体而言，该方法将问题的推理过程建模为图，并利用基于LLM的图学习来指导每一步推理的自适应生成。为进一步增强模型的适应性，我们引入了一个图神经网络（GNN）模块，对生成的推理过程进行表示学习，从而能够实时调整模型和提示。实验结果表明，该方法在多个任务上显著提高了推理性能，无需额外训练或特定任务的提示设计。代码可在以下链接找到：https://...。 

---
# Divide (Text) and Conquer (Sentiment): Improved Sentiment Classification by Constituent Conflict Resolution 

**Title (ZH)**: 分而治之（情感冲突决议改进情感分类） 

**Authors**: Jan Kościałkowski, Paweł Marcinkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.06320)  

**Abstract**: Sentiment classification, a complex task in natural language processing, becomes even more challenging when analyzing passages with multiple conflicting tones. Typically, longer passages exacerbate this issue, leading to decreased model performance. The aim of this paper is to introduce novel methodologies for isolating conflicting sentiments and aggregating them to effectively predict the overall sentiment of such passages. One of the aggregation strategies involves a Multi-Layer Perceptron (MLP) model which outperforms baseline models across various datasets, including Amazon, Twitter, and SST while costing $\sim$1/100 of what fine-tuning the baseline would take. 

**Abstract (ZH)**: 情感分类，自然语言处理中的一个复杂任务，在分析具有多种矛盾基调的段落时变得更加具有挑战性。通常，更长的段落会加剧这一问题，导致模型性能下降。本文旨在介绍新的方法来隔离矛盾的情感并将其聚合，以有效预测此类段落的整体情感。一种聚合策略涉及多层感知机（MLP）模型，该模型在包括Amazon、Twitter和SST在内的各种数据集上表现优于基线模型，成本仅为基线微调的1/100左右。 

---
# Threat Modeling for AI: The Case for an Asset-Centric Approach 

**Title (ZH)**: AI的安全威胁建模：资产中心化方法的必要性 

**Authors**: Jose Sanchez Vicarte, Marcin Spoczynski, Mostafa Elsaid  

**Link**: [PDF](https://arxiv.org/pdf/2505.06315)  

**Abstract**: Recent advances in AI are transforming AI's ubiquitous presence in our world from that of standalone AI-applications into deeply integrated AI-agents. These changes have been driven by agents' increasing capability to autonomously make decisions and initiate actions, using existing applications; whether those applications are AI-based or not. This evolution enables unprecedented levels of AI integration, with agents now able to take actions on behalf of systems and users -- including, in some cases, the powerful ability for the AI to write and execute scripts as it deems necessary. With AI systems now able to autonomously execute code, interact with external systems, and operate without human oversight, traditional security approaches fall short.
This paper introduces an asset-centric methodology for threat modeling AI systems that addresses the unique security challenges posed by integrated AI agents. Unlike existing top-down frameworks that analyze individual attacks within specific product contexts, our bottom-up approach enables defenders to systematically identify how vulnerabilities -- both conventional and AI-specific -- impact critical AI assets across distributed infrastructures used to develop and deploy these agents. This methodology allows security teams to: (1) perform comprehensive analysis that communicates effectively across technical domains, (2) quantify security assumptions about third-party AI components without requiring visibility into their implementation, and (3) holistically identify AI-based vulnerabilities relevant to their specific product context. This approach is particularly relevant for securing agentic systems with complex autonomous capabilities. By focusing on assets rather than attacks, our approach scales with the rapidly evolving threat landscape while accommodating increasingly complex and distributed AI development pipelines. 

**Abstract (ZH)**: 近期人工智能技术的发展正将人工智能在世界上的无处不在从独立的人工智能应用转变为深度整合的人工智能代理。这些变化是由代理日益增强的自主决策和行动能力驱动的，无论这些行动是由基于人工智能的应用还是非基于人工智能的应用触发的。这种演变使得前所未有的高水平人工智能整合成为可能，代理现在可以代表系统和用户采取行动——包括在某些情况下，代理具有编写和执行必要脚本的强大能力。随着人工智能系统现在能够自主执行代码、与外部系统交互并运行而无需人工监督，传统的安全方法已不再有效。本文提出了一种以资产为中心的威胁建模方法，以应对整合人工智能代理所带来的独特安全挑战。与现有的自上而下框架仅在特定产品背景下分析单一攻击不同，我们自下而上的方法允许防守者系统地识别传统漏洞和人工智能特定漏洞如何影响分布式基础设施中的关键人工智能资产。这种方法使安全团队能够：(1) 进行全面分析，有效跨越技术领域沟通，(2) 不要求对第三方人工智能组件的实现可见性即可量化安全假设，以及(3) 从具体的产品环境中全面识别人工智能相关的漏洞。这种方法特别适用于保护具有复杂自主能力的代理系统。通过关注资产而非攻击，我们的方法能够适应迅速演变的威胁环境，并适应日益复杂和分布的人工智能开发流水线。 

---
# A4L: An Architecture for AI-Augmented Learning 

**Title (ZH)**: A4L：一种AI增强学习架构 

**Authors**: Ashok Goel, Ploy Thajchayapong, Vrinda Nandan, Harshvardhan Sikka, Spencer Rugaber  

**Link**: [PDF](https://arxiv.org/pdf/2505.06314)  

**Abstract**: AI promises personalized learning and scalable education. As AI agents increasingly permeate education in support of teaching and learning, there is a critical and urgent need for data architectures for collecting and analyzing data on learning, and feeding the results back to teachers, learners, and the AI agents for personalization of learning at scale. At the National AI Institute for Adult Learning and Online Education, we are developing an Architecture for AI-Augmented Learning (A4L) for supporting adult learning through online education. We present the motivations, goals, requirements of the A4L architecture. We describe preliminary applications of A4L and discuss how it advances the goals of making learning more personalized and scalable. 

**Abstract (ZH)**: AI承诺个性化学习与规模化教育：为支持成人在线教育，我们正在开发AI增强学习架构（A4L），以推动学习更加个性化和规模化。 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Title (ZH)**: AI方法在北约团结行动定性与定量新闻分析中的应用 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

**Abstract (ZH)**: 使用GPT模型结合检索增强生成（RAG）方法对北约 sentiment、团结和Article 5信任意见分数进行定性和定量分析：基于Google Search API检索的新闻网站、YouTube视频评论和Reddit讨论。采用GPT-4.1模型运用RAG方法分析与北约相关的新闻摘要。使用零-shot提示生成定性新闻概述和定量意见分数，并在第二层生成摘要的概述；通过贝叶斯回归分析生成的意见分数趋势，评估新闻意见分数趋势的不确定性。研究结果表明，分析分数呈下降趋势。该方法旨在为复杂的分析方法提供基于AI的工具，结果表明使用GPT模型进行新闻分析可以提供有用的信息和定量见解。基于神经常微分方程的动态模型用于 modelling 公众意见，该方法能够分析公众意见演变的不同情景。 

---
# Responsibility Gap in Collective Decision Making 

**Title (ZH)**: 集体决策中的责任缺口 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06312)  

**Abstract**: The responsibility gap is a set of outcomes of a collective decision-making mechanism in which no single agent is individually responsible. In general, when designing a decision-making process, it is desirable to minimise the gap.
The paper proposes a concept of an elected dictatorship. It shows that, in a perfect information setting, the gap is empty if and only if the mechanism is an elected dictatorship. It also proves that in an imperfect information setting, the class of gap-free mechanisms is positioned strictly between two variations of the class of elected dictatorships. 

**Abstract (ZH)**: 责任缺口是集体决策机制的一种结果，在这种机制中，没有单一代理个体承担责任。通常，在设计决策过程时，应尽量减小这种缺口。
本文提出了当选制独裁的概念。它证明，在完美信息的环境下，当且仅当机制是当选制独裁时，缺口为空。此外，它还证明，在不完美信息的环境下，无缺口机制类严格位于两种当选制独裁类的变体之间。 

---
# Defending against Indirect Prompt Injection by Instruction Detection 

**Title (ZH)**: 防御间接提示注入攻击：指令检测方法 

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06311)  

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark. 

**Abstract (ZH)**: 大型语言模型与外部数据源的集成变得越来越普遍，检索增强生成（RAG）是其中的一个典型例子。然而，这种集成引入了间接提示注入（IPI）攻击的安全漏洞，隐藏在外部数据中的指令可以操控大型语言模型执行未预期或有害的操作。我们认识到，IPI攻击的成功与否从根本上依赖于存在于外部内容中的隐蔽指令，这些指令可以改变大型语言模型的行为状态。有效检测这种状态变化是否能帮助我们抵御IPI攻击？在本文中，我们提出了一种新的方法，将外部数据作为输入，并利用大型语言模型在前向和反向传播过程中的行为状态，来检测潜在的IPI攻击。具体而言，我们证明了中间层的隐藏状态和梯度提供了一种高度区分性的特征，用于指令检测。通过有效结合这些特征，我们的方法在领域内设置实现了99.60%的检测准确率，在领域外设置实现了96.90%的检测准确率，并将BIPIA基准测试中的攻击成功率降低到仅0.12%。 

---
# Large Language Model-driven Security Assistant for Internet of Things via Chain-of-Thought 

**Title (ZH)**: 大型语言模型驱动的物联网安全助手通过思维链 

**Authors**: Mingfei Zeng, Ming Xie, Xixi Zheng, Chunhai Li, Chuan Zhang, Liehuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06307)  

**Abstract**: The rapid development of Internet of Things (IoT) technology has transformed people's way of life and has a profound impact on both production and daily activities. However, with the rapid advancement of IoT technology, the security of IoT devices has become an unavoidable issue in both research and applications. Although some efforts have been made to detect or mitigate IoT security vulnerabilities, they often struggle to adapt to the complexity of IoT environments, especially when dealing with dynamic security scenarios. How to automatically, efficiently, and accurately understand these vulnerabilities remains a challenge. To address this, we propose an IoT security assistant driven by Large Language Model (LLM), which enhances the LLM's understanding of IoT security vulnerabilities and related threats. The aim of the ICoT method we propose is to enable the LLM to understand security issues by breaking down the various dimensions of security vulnerabilities and generating responses tailored to the user's specific needs and expertise level. By incorporating ICoT, LLM can gradually analyze and reason through complex security scenarios, resulting in more accurate, in-depth, and personalized security recommendations and solutions. Experimental results show that, compared to methods relying solely on LLM, our proposed LLM-driven IoT security assistant significantly improves the understanding of IoT security issues through the ICoT approach and provides personalized solutions based on the user's identity, demonstrating higher accuracy and reliability. 

**Abstract (ZH)**: 物联网技术的快速发展改变了人们的生活方式，并对生产和日常活动产生了深远影响。然而，随着物联网技术的快速进步，物联网设备的安全性已经成为研究和应用中无法回避的问题，尤其是在处理动态安全场景时更为明显。如何自动、高效、准确地理解和应对这些安全漏洞仍然是一项挑战。为此，我们提出了由大规模语言模型（Large Language Model，LLM）驱动的物联网安全助手，以增强LLM对物联网安全漏洞及其相关威胁的理解。我们所提出的ICoT方法旨在通过分解安全漏洞的各种维度并生成针对用户具体需求和专业水平的响应，使LLM能够理解安全问题。通过整合ICoT方法，LLM可以逐步分析和推理复杂的安全场景，从而提供更准确、深入且个性化的安全建议和解决方案。实验结果表明，与仅依赖于LLM的方法相比，我们提出的大规模语言模型驱动的物联网安全助手通过ICoT方法显著提高了对物联网安全问题的理解，并基于用户的身份提供个性化的解决方案，显示出更高的准确性和可靠性。 

---
# User Behavior Analysis in Privacy Protection with Large Language Models: A Study on Privacy Preferences with Limited Data 

**Title (ZH)**: 使用大规模语言模型保护隐私中的用户行为分析：基于有限数据的隐私偏好研究 

**Authors**: Haowei Yang, Qingyi Lu, Yang Wang, Sibei Liu, Jiayun Zheng, Ao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06305)  

**Abstract**: With the widespread application of large language models (LLMs), user privacy protection has become a significant research topic. Existing privacy preference modeling methods often rely on large-scale user data, making effective privacy preference analysis challenging in data-limited environments. This study explores how LLMs can analyze user behavior related to privacy protection in scenarios with limited data and proposes a method that integrates Few-shot Learning and Privacy Computing to model user privacy preferences. The research utilizes anonymized user privacy settings data, survey responses, and simulated data, comparing the performance of traditional modeling approaches with LLM-based methods. Experimental results demonstrate that, even with limited data, LLMs significantly improve the accuracy of privacy preference modeling. Additionally, incorporating Differential Privacy and Federated Learning further reduces the risk of user data exposure. The findings provide new insights into the application of LLMs in privacy protection and offer theoretical support for advancing privacy computing and user behavior analysis. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的广泛应用，用户隐私保护已成为一个重要研究课题。现有的隐私偏好建模方法往往依赖大量用户数据，这在数据受限环境中使得有效的隐私偏好分析变得具有挑战性。本研究探索了在数据有限的情景下，LLM 如何分析与隐私保护相关的用户行为，并提出了一种结合少样本学习和隐私计算的用户隐私偏好建模方法。研究利用匿名化的用户隐私设置数据、调查响应和模拟数据，比较了传统建模方法与基于LLM的方法的性能。实验结果表明，即使在数据有限的情况下，LLM 也显著提高了隐私偏好建模的准确性。此外，结合差分隐私和联邦学习进一步降低了用户数据暴露的风险。研究结果为 LLM 在隐私保护中的应用提供了新的见解，并为推进隐私计算和用户行为分析提供了理论支持。 

---
# Collaborative Multi-LoRA Experts with Achievement-based Multi-Tasks Loss for Unified Multimodal Information Extraction 

**Title (ZH)**: 基于成就导向多任务损失的协作多LoRA专家统一多模态信息提取 

**Authors**: Li Yuan, Yi Cai, Xudong Shen, Qing Li, Qingbao Huang, Zikun Deng, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06303)  

**Abstract**: Multimodal Information Extraction (MIE) has gained attention for extracting structured information from multimedia sources. Traditional methods tackle MIE tasks separately, missing opportunities to share knowledge across tasks. Recent approaches unify these tasks into a generation problem using instruction-based T5 models with visual adaptors, optimized through full-parameter fine-tuning. However, this method is computationally intensive, and multi-task fine-tuning often faces gradient conflicts, limiting performance. To address these challenges, we propose collaborative multi-LoRA experts with achievement-based multi-task loss (C-LoRAE) for MIE tasks. C-LoRAE extends the low-rank adaptation (LoRA) method by incorporating a universal expert to learn shared multimodal knowledge from cross-MIE tasks and task-specific experts to learn specialized instructional task features. This configuration enhances the model's generalization ability across multiple tasks while maintaining the independence of various instruction tasks and mitigating gradient conflicts. Additionally, we propose an achievement-based multi-task loss to balance training progress across tasks, addressing the imbalance caused by varying numbers of training samples in MIE tasks. Experimental results on seven benchmark datasets across three key MIE tasks demonstrate that C-LoRAE achieves superior overall performance compared to traditional fine-tuning methods and LoRA methods while utilizing a comparable number of training parameters to LoRA. 

**Abstract (ZH)**: 多模态信息提取中的协作多LoRA专家与成就导向的多任务损失（C-LoRAE） 

---
# QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives 

**Title (ZH)**: QiMeng-TensorOp: 通过硬件 primitives 自动生成高性能张量操作符 

**Authors**: Xuzhi Zhang, Shaohui Peng, Qirui Zhou, Yuanbo Wen, Qi Guo, Ruizhi Chen, Xinguo Zhu, Weiqiang Xiong, Haixin Chen, Congying Ma, Ke Gao, Chen Zhao, Yanjun Wu, Yunji Chen, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06302)  

**Abstract**: Computation-intensive tensor operators constitute over 90\% of the computations in Large Language Models (LLMs) and Deep Neural this http URL and efficiently generating high-performance tensor operators with hardware primitives is crucial for diverse and ever-evolving hardware architectures like RISC-V, ARM, and GPUs, as manually optimized implementation takes at least months and lacks this http URL excel at generating high-level language codes, but they struggle to fully comprehend hardware characteristics and produce high-performance tensor operators. We introduce a tensor-operator auto-generation framework with a one-line user prompt (QiMeng-TensorOp), which enables LLMs to automatically exploit hardware characteristics to generate tensor operators with hardware primitives, and tune parameters for optimal performance across diverse hardware. Experimental results on various hardware platforms, SOTA LLMs, and typical tensor operators demonstrate that QiMeng-TensorOp effectively unleashes the computing capability of various hardware platforms, and automatically generates tensor operators of superior performance. Compared with vanilla LLMs, QiMeng-TensorOp achieves up to $1291 \times$ performance improvement. Even compared with human experts, QiMeng-TensorOp could reach $251 \%$ of OpenBLAS on RISC-V CPUs, and $124 \%$ of cuBLAS on NVIDIA GPUs. Additionally, QiMeng-TensorOp also significantly reduces development costs by $200 \times$ compared with human experts. 

**Abstract (ZH)**: 计算密集型张量操作构成了大规模语言模型（LLMs）和深度神经网络中超过90%的计算量，高效地使用硬件 primitives 生成高性能张量操作对于如RISC-V、ARM和GPU等多样和不断演进的硬件架构至关重要。由于手工优化实现需要至少几个月的时间且缺乏一致性，虽然现有的LLMs在生成高级语言代码方面表现出色，但它们在完全理解硬件特性并产生高性能张量操作方面存在困难。我们引入了一种带有单一用户提示（QiMeng-TensorOp）的张量操作自动生成框架，使LLMs能够自动利用硬件特性，使用硬件 primitives 生成张量操作，并针对多种硬件进行参数调整以实现最优性能。实验结果表明，QiMeng-TensorOp 有效地释放了各种硬件平台的计算能力，并自动生成了高性能的张量操作。与 vanilla LLMs 相比，QiMeng-TensorOp 实现了高达1291倍的性能改进。即使与人类专家相比，QiMeng-TensorOp 在RISC-V CPU上的性能也可达OpenBLAS的251%，在NVIDIA GPU上的性能则可达cuBLAS的124%。此外，与人类专家相比，QiMeng-TensorOp 还将开发成本降低了200倍。 

---
# Domain-Adversarial Anatomical Graph Networks for Cross-User Human Activity Recognition 

**Title (ZH)**: 跨用户人体活动识别的领域对抗解剖图网络 

**Authors**: Xiaozhou Ye, Kevin I-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06301)  

**Abstract**: Cross-user variability in Human Activity Recognition (HAR) remains a critical challenge due to differences in sensor placement, body dynamics, and behavioral patterns. Traditional methods often fail to capture biomechanical invariants that persist across users, limiting their generalization capability. We propose an Edge-Enhanced Graph-Based Adversarial Domain Generalization (EEG-ADG) framework that integrates anatomical correlation knowledge into a unified graph neural network (GNN) architecture. By modeling three biomechanically motivated relationships together-Interconnected Units, Analogous Units, and Lateral Units-our method encodes domain-invariant features while addressing user-specific variability through Variational Edge Feature Extractor. A Gradient Reversal Layer (GRL) enforces adversarial domain generalization, ensuring robustness to unseen users. Extensive experiments on OPPORTUNITY and DSADS datasets demonstrate state-of-the-art performance. Our work bridges biomechanical principles with graph-based adversarial learning by integrating information fusion techniques. This fusion of information underpins our unified and generalized model for cross-user HAR. 

**Abstract (ZH)**: 跨用户的动作识别（HAR）由于传感器放置、身体动力学和行为模式的差异仍是一项关键挑战。传统的方法往往无法捕捉到在不同用户间保持不变的生物力学不变量，限制了其泛化能力。我们提出了一种基于图的对抗域泛化增强边缘（EEG-ADG）框架，将解剖学相关知识融合到统一的图神经网络（GNN）架构中。通过一起建模三种生物力学驱动的关系—互联单元、同源单元和侧向单元—我们的方法在编码域不变特征的同时，通过变分边缘特征提取器解决用户特定的变异性。 gradient reversal层（GRL）促进对抗域泛化，确保对未见用户具有鲁棒性。在OPPORTUNITY和DSADS数据集上的广泛实验显示出最先进的性能。我们的工作通过集成信息融合技术将生物力学原理与图基对抗学习相结合，支撑了我们针对跨用户HAR的统一和泛化模型。 

---
# ARDNS-FN-Quantum: A Quantum-Enhanced Reinforcement Learning Framework with Cognitive-Inspired Adaptive Exploration for Dynamic Environments 

**Title (ZH)**: ARDNS-FN-量子增强强化学习框架：认知启发式自适应探索用于动态环境 

**Authors**: Umberto Gonçalves de Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06300)  

**Abstract**: Reinforcement learning (RL) has transformed sequential decision making, yet traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents ARDNS-FN-Quantum (Adaptive Reward-Driven Neural Simulator with Quantum enhancement), a novel framework that integrates a 2-qubit quantum circuit for action selection, a dual-memory system inspired by human cognition, and adaptive exploration strategies modulated by reward variance and curiosity. Evaluated in a 10X10 grid-world over 20,000 episodes, ARDNS-FN-Quantum achieves a 99.5% success rate (versus 81.3% for DQN and 97.0% for PPO), a mean reward of 9.0528 across all episodes (versus 1.2941 for DQN and 7.6196 for PPO), and an average of 46.7 steps to goal (versus 135.9 for DQN and 62.5 for PPO). In the last 100 episodes, it records a mean reward of 9.1652 (versus 7.0916 for DQN and 9.0310 for PPO) and 37.2 steps to goal (versus 52.7 for DQN and 53.4 for PPO). Graphical analyses, including learning curves, steps-to-goal trends, reward variance, and reward distributions, demonstrate ARDNS-FN-Quantum's superior stability (reward variance 5.424 across all episodes versus 252.262 for DQN and 76.583 for PPO) and efficiency. By bridging quantum computing, cognitive science, and RL, ARDNS-FN-Quantum offers a scalable, human-like approach to adaptive learning in uncertain environments, with potential applications in robotics, autonomous systems, and decision-making under uncertainty. 

**Abstract (ZH)**: 强化学习（RL）已重塑序列决策过程，但传统的算法如深度Q网络（DQN）和渐近策略优化（PPO）往往在动态环境中的高效探索、稳定性和适应性方面存在问题。本文提出了一种名为ARDNS-FN-量子（Adaptive Reward-Driven Neural Simulator with Quantum Enhancement）的新颖框架，该框架整合了用于动作选择的2-qubit量子电路、受人类认知启发的双记忆系统以及由奖励方差和好奇性调节的自适应探索策略。在10×10的网格世界中经过20,000个episode的评估，ARDNS-FN-量子实现了99.5%的成功率（相比之下，DQN为81.3%，PPO为97.0%）、平均每回合9.0528的奖励（相比之下，DQN为1.2941，PPO为7.6196）以及平均46.7步达到目标的性能（相比之下，DQN为135.9，PPO为62.5）。最后100个episode中，平均奖励为9.1652（相比之下，DQN为7.0916，PPO为9.0310），平均达到目标步数为37.2（相比之下，DQN为52.7，PPO为53.4）。图形分析包括学习曲线、达到目标步数趋势、奖励方差和奖励分布，证明了ARDNS-FN-量子在稳定性和效率方面的优越性能。通过结合量子计算、认知科学和强化学习，ARDNS-FN-量子提供了一种可扩展、类人的适应性学习方法，适用于不确定环境下的机器人、自主系统和不确定性决策，具有潜在应用价值。 

---
# Input-Specific and Universal Adversarial Attack Generation for Spiking Neural Networks in the Spiking Domain 

**Title (ZH)**: 输入特定性和普遍性的对抗攻击生成用于突触神经网络的突触域 

**Authors**: Spyridon Raptis, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.06299)  

**Abstract**: As Spiking Neural Networks (SNNs) gain traction across various applications, understanding their security vulnerabilities becomes increasingly important. In this work, we focus on the adversarial attacks, which is perhaps the most concerning threat. An adversarial attack aims at finding a subtle input perturbation to fool the network's decision-making. We propose two novel adversarial attack algorithms for SNNs: an input-specific attack that crafts adversarial samples from specific dataset inputs and a universal attack that generates a reusable patch capable of inducing misclassification across most inputs, thus offering practical feasibility for real-time deployment. The algorithms are gradient-based operating in the spiking domain proving to be effective across different evaluation metrics, such as adversarial accuracy, stealthiness, and generation time. Experimental results on two widely used neuromorphic vision datasets, NMNIST and IBM DVS Gesture, show that our proposed attacks surpass in all metrics all existing state-of-the-art methods. Additionally, we present the first demonstration of adversarial attack generation in the sound domain using the SHD dataset. 

**Abstract (ZH)**: 随着脉冲神经网络（SNNs）在各种应用中逐渐受到重视，理解其安全漏洞变得越来越重要。在本文中，我们重点关注敌对攻击，这可能是最令人关切的威胁。敌对攻击旨在找到一种微妙的输入扰动以迷惑网络的决策过程。我们提出了两种针对SNNs的新颖敌对攻击算法：一种是输入特定攻击，从特定数据集输入中创建敌对样本；另一种是通用攻击，生成一个可重用的补丁，能够在大多数输入中诱发错误分类，从而为实时部署提供了实用性。这些算法基于梯度，工作在脉冲域，证明在不同的评估指标（如敌对准确性、隐蔽性和生成时间）上都是有效的。实验结果表明，我们提出的攻击在所有指标上都超过了所有现有的前沿方法。此外，我们还首次在SHD数据集中展示了声音域中的敌对攻击生成。 

---
# Terahertz Spatial Wireless Channel Modeling with Radio Radiance Field 

**Title (ZH)**: 太赫兹空间无线信道建模与射流场理论 

**Authors**: John Song, Lihao Zhang, Feng Ye, Haijian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.06277)  

**Abstract**: Terahertz (THz) communication is a key enabler for 6G systems, offering ultra-wide bandwidth and unprecedented data rates. However, THz signal propagation differs significantly from lower-frequency bands due to severe free space path loss, minimal diffraction and specular reflection, and prominent scattering, making conventional channel modeling and pilot-based estimation approaches inefficient. In this work, we investigate the feasibility of applying radio radiance field (RRF) framework to the THz band. This method reconstructs a continuous RRF using visual-based geometry and sparse THz RF measurements, enabling efficient spatial channel state information (Spatial-CSI) modeling without dense sampling. We first build a fine simulated THz scenario, then we reconstruct the RRF and evaluate the performance in terms of both reconstruction quality and effectiveness in THz communication, showing that the reconstructed RRF captures key propagation paths with sparse training samples. Our findings demonstrate that RRF modeling remains effective in the THz regime and provides a promising direction for scalable, low-cost spatial channel reconstruction in future 6G networks. 

**Abstract (ZH)**: 太赫兹（THz）通信是6G系统的关键使能技术，提供超宽带和空前的数据速率。然而，THz信号传播由于严重的自由空间路径损耗、最小的衍射和镜面反射以及显著的散射与较低频率 band 有所不同，使得传统的信道建模和基于导频的估计方法效率低下。在本文中，我们探讨了将无线电辐射场（RRF）框架应用于THz频段的可能性。该方法使用基于视觉的几何和稀疏的THz射频测量重构连续的RRF，使得无需密集采样即可高效建模空间信道状态信息（Spatial-CSI）。我们首先构建了一个精细模拟的THz场景，然后重构RRF并从重构质量及在THz通信中的有效性两个方面评估其性能，表明重构的RRF能够在稀疏训练样本下捕捉到关键的传播路径。我们的研究结果表明，RRF建模在THz频段仍然有效，并为未来6G网络中具备扩展性和低成本的空间信道重构提供了一个有前景的方向。 

---
# Attonsecond Streaking Phase Retrieval Via Deep Learning Methods 

**Title (ZH)**: 亚飞秒级 streaking 相位恢复基于深度学习方法 

**Authors**: Yuzhou Zhu, Zheng Zhang, Ruyi Zhang, Liang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06275)  

**Abstract**: Attosecond streaking phase retrieval is essential for resolving electron dynamics on sub-femtosecond time scales yet traditional algorithms rely on iterative minimization and central momentum approximations that degrade accuracy for broadband pulses. In this work phase retrieval is reformulated as a supervised computer-vision problem and four neural architectures are systematically compared. A convolutional network demonstrates strong sensitivity to local streak edges but lacks global context; a vision transformer captures long-range delay-energy correlations at the expense of local inductive bias; a hybrid CNN-ViT model unites local feature extraction and full-graph attention; and a capsule network further enforces spatial pose agreement through dynamic routing. A theoretical analysis introduces local, global and positional sensitivity measures and derives surrogate error bounds that predict the strict ordering $CNN<ViT<Hybrid<Capsule$. Controlled experiments on synthetic streaking spectrograms confirm this hierarchy, with the capsule network achieving the highest retrieval fidelity. Looking forward, embedding the strong-field integral into physics-informed neural networks and exploring photonic hardware implementations promise pathways toward real-time attosecond pulse characterization under demanding experimental conditions. 

**Abstract (ZH)**: 阿斯皮秒级相位检索对于亚飞秒时间尺度的电子动力学解析至关重要，但传统算法依赖于迭代最小化和质心动量近似，这会降低宽带脉冲的准确性。本工作中，相位检索被重新表述为监督计算机视觉问题，并系统对比了四种神经网络架构。卷积网络对局部条形边缘表现出强烈的敏感性，但缺乏全局上下文；视觉变换器捕捉长程延迟-能量相关性，但以牺牲局部归纳偏见为代价；混合CNN-ViT模型结合了局部特征提取和全图注意力；胶囊网络进一步通过动态路由确保空间姿态的一致性。理论分析引入了局部、全局和位置敏感性度量，并推导出预测严格排序的替代误差界：$CNN<ViT<Hybrid<Capsule$。受控实验在合成条形光谱图上确认了该层级关系，胶囊网络实现了最高的检索保真度。展望未来，将强场积分嵌入物理感知神经网络并在光子硬件实现中的探索为在苛刻实验条件下实时阿斯皮秒脉冲表征指明了路径。 

---
# PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model 

**Title (ZH)**: PARM：基于偏好意识自回归奖励模型的多目标测试时对齐方法 

**Authors**: Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06274)  

**Abstract**: Multi-objective test-time alignment aims to adapt large language models (LLMs) to diverse multi-dimensional user preferences during inference while keeping LLMs frozen. Recently, GenARM (Xu et al., 2025) first independently trains Autoregressive Reward Models (ARMs) for each preference dimension without awareness of each other, then combines their outputs based on user-specific preference vectors during inference to achieve multi-objective test-time alignment, leading to two key limitations: the need for \textit{multiple} ARMs increases the inference cost, and the separate training of ARMs causes the misalignment between the guided generation and the user preferences. To address these issues, we propose Preference-aware ARM (PARM), a single unified ARM trained across all preference dimensions. PARM uses our proposed Preference-Aware Bilinear Low-Rank Adaptation (PBLoRA), which employs a bilinear form to condition the ARM on preference vectors, enabling it to achieve precise control over preference trade-offs during inference. Experiments demonstrate that PARM reduces inference costs and achieves better alignment with preference vectors compared with existing methods. Additionally, PARM enables weak-to-strong guidance, allowing a smaller PARM to guide a larger frozen LLM without expensive training, making multi-objective alignment accessible with limited computing resources. The code is available at this https URL. 

**Abstract (ZH)**: 多目标测试时对齐旨在适应大规模语言模型（LLMs）在推理过程中多元的多维度用户偏好，同时保持LLMs冻结。近期，GenARM（Xu等，2025）率先独立训练每个偏好维度的自回归奖励模型（ARMs），而彼此之间无意识关系，然后在推理过程中基于用户的特定偏好向量结合它们的输出以实现多目标测试时对齐，导致两个关键限制：需要多个ARM增加了推理成本，独立训练ARM导致导向生成与用户偏好之间的错位。为解决这些问题，我们提出了一种偏好意识ARM（PARM），这是一种在所有偏好维度上统一训练的ARM。PARM使用我们提出的偏好意识双线性低秩适应（PBLoRA），该方法采用双线性形式在ARM上条件化偏好向量，使其能够在推理过程中实现精确的偏好权衡控制。实验表明，PARM减少了推理成本并比现有方法更好地与偏好向量对齐。此外，PARM支持从弱到强的指导，允许较小的PARM在不进行昂贵训练的情况下引导较大的冻结LLM，从而在有限计算资源下实现多目标对齐。代码可在以下链接获取：this https URL。 

---
# Policy-labeled Preference Learning: Is Preference Enough for RLHF? 

**Title (ZH)**: 基于政策标注的偏好学习：偏好足够用于RLHF吗？ 

**Authors**: Taehyun Cho, Seokhun Ju, Seungyub Han, Dohyeong Kim, Kyungjae Lee, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.06273)  

**Abstract**: To design rewards that align with human goals, Reinforcement Learning from Human Feedback (RLHF) has emerged as a prominent technique for learning reward functions from human preferences and optimizing policies via reinforcement learning algorithms. However, existing RLHF methods often misinterpret trajectories as being generated by an optimal policy, causing inaccurate likelihood estimation and suboptimal learning. Inspired by Direct Preference Optimization framework which directly learns optimal policy without explicit reward, we propose policy-labeled preference learning (PPL), to resolve likelihood mismatch issues by modeling human preferences with regret, which reflects behavior policy information. We also provide a contrastive KL regularization, derived from regret-based principles, to enhance RLHF in sequential decision making. Experiments in high-dimensional continuous control tasks demonstrate PPL's significant improvements in offline RLHF performance and its effectiveness in online settings. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）中的奖励设计：直接偏好优化及其在顺序决策中的应用 

---
# A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning 

**Title (ZH)**: 基于灵敏度驱动的专家分配方法在LoRA-MoE中的高效微调 

**Authors**: Junzhou Xu, Boyu Diao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06272)  

**Abstract**: As deep learning models expand, the pre-training-fine-tuning paradigm has become the standard approach for handling various downstream tasks. However, shared parameters can lead to diminished performance when dealing with complex datasets involving multiple tasks. While introducing Mixture-of-Experts (MoE) methods has alleviated this issue to some extent, it also significantly increases the number of parameters required for fine-tuning and training time, introducing greater parameter redundancy. To address these challenges, we propose a method for allocating expert numbers based on parameter sensitivity LoRA-SMoE (A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning). This method rapidly assesses the sensitivity of different tasks to parameters by sampling a small amount of data and using gradient information. It then adaptively allocates expert numbers within a given budget. The process maintains comparable memory consumption to LoRA (Low-Rank Adaptation) while ensuring an efficient and resource-friendly fine-tuning procedure. Experimental results demonstrate that compared to SOTA fine-tuning methods, our LoRA-SMoE approach can enhance model performance while reducing the number of trainable parameters. This significantly improves model performance in resource-constrained environments. Additionally, due to its efficient parameter sensitivity evaluation mechanism, LoRA-SMoE requires minimal computational overhead to optimize expert allocation, making it particularly suitable for scenarios with limited computational resources. All the code in this study will be made publicly available following the acceptance of the paper for publication. Source code is at this https URL 

**Abstract (ZH)**: 基于参数灵敏度的LoRA-SMoE专家分配方法：一种在LoRA-MoE中的灵敏度驱动专家分配方法以提高高效微调性能 

---
# Tri-MTL: A Triple Multitask Learning Approach for Respiratory Disease Diagnosis 

**Title (ZH)**: 三重多任务学习：一种呼吸系统疾病诊断方法 

**Authors**: June-Woo Kim, Sanghoon Lee, Miika Toikkanen, Daehwan Hwang, Kyunghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06271)  

**Abstract**: Auscultation remains a cornerstone of clinical practice, essential for both initial evaluation and continuous monitoring. Clinicians listen to the lung sounds and make a diagnosis by combining the patient's medical history and test results. Given this strong association, multitask learning (MTL) can offer a compelling framework to simultaneously model these relationships, integrating respiratory sound patterns with disease manifestations. While MTL has shown considerable promise in medical applications, a significant research gap remains in understanding the complex interplay between respiratory sounds, disease manifestations, and patient metadata attributes. This study investigates how integrating MTL with cutting-edge deep learning architectures can enhance both respiratory sound classification and disease diagnosis. Specifically, we extend recent findings regarding the beneficial impact of metadata on respiratory sound classification by evaluating its effectiveness within an MTL framework. Our comprehensive experiments reveal significant improvements in both lung sound classification and diagnostic performance when the stethoscope information is incorporated into the MTL architecture. 

**Abstract (ZH)**: 听诊 remains 临床实践的基础，对于初步评估和持续监测都至关重要。医护人员通过结合患者的病史和检测结果来听诊肺部声音并进行诊断。鉴于这一紧密关联，多任务学习（MTL）可以提供一个引人入胜的框架，同时建模这些关系，整合呼吸音模式与疾病表现。尽管 MTL 在医疗应用中展现出巨大潜力，但仍存在关于呼吸音、疾病表现和患者元数据属性之间复杂互动关系理解的研究空白。本研究探讨了将 MTL 与前沿的深度学习架构结合如何增强呼吸音分类和疾病诊断。具体而言，我们扩展了关于元数据对呼吸音分类有益影响的近期发现，评估其在 MTL 框架内的有效性。我们全面的实验结果显示，将听诊器信息纳入 MTL 架构时，呼吸音分类和诊断性能均能显著提高。 

---
# Importance Analysis for Dynamic Control of Balancing Parameter in a Simple Knowledge Distillation Setting 

**Title (ZH)**: 简单知识精简设置中平衡参数动态控制的重要性分析 

**Authors**: Seongmin Kim, Kwanho Kim, Minseung Kim, Kanghyun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06270)  

**Abstract**: Although deep learning models owe their remarkable success to deep and complex architectures, this very complexity typically comes at the expense of real-time performance. To address this issue, a variety of model compression techniques have been proposed, among which knowledge distillation (KD) stands out for its strong empirical performance. The KD contains two concurrent processes: (i) matching the outputs of a large, pre-trained teacher network and a lightweight student network, and (ii) training the student to solve its designated downstream task. The associated loss functions are termed the distillation loss and the downsteam-task loss, respectively. Numerous prior studies report that KD is most effective when the influence of the distillation loss outweighs that of the downstream-task loss. The influence(or importance) is typically regulated by a balancing parameter. This paper provides a mathematical rationale showing that in a simple KD setting when the loss is decreasing, the balancing parameter should be dynamically adjusted 

**Abstract (ZH)**: 尽管深度学习模型因其深度和复杂架构取得了显著的成功，但这种复杂性通常会牺牲实时性能。为了解决这一问题，已经提出了多种模型压缩技术，其中知识蒸馏（KD）因其强大的实证性能而脱颖而出。KD包含两个并发过程：(i) 匹配预先训练的大规模教师网络和轻量级学生网络的输出，(ii) 训练学生解决其指定的下游任务。相关的损失函数分别称为蒸馏损失和下游任务损失。许多先前的研究报告称，当蒸馏损失的影响超过下游任务损失时，KD的效果最佳。影响（或重要性）通常通过一个平衡参数来调节。本文提供了一个数学理由，证明在简单KD设置中，当损失在减少时，平衡参数应动态调整。 

---
# Cluster-Aware Multi-Round Update for Wireless Federated Learning in Heterogeneous Environments 

**Title (ZH)**: 面向异构环境的集群感知多轮更新无线联邦学习 

**Authors**: Pengcheng Sun, Erwu Liu, Wei Ni, Kanglei Yu, Rui Wang, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.06268)  

**Abstract**: The aggregation efficiency and accuracy of wireless Federated Learning (FL) are significantly affected by resource constraints, especially in heterogeneous environments where devices exhibit distinct data distributions and communication capabilities. This paper proposes a clustering strategy that leverages prior knowledge similarity to group devices with similar data and communication characteristics, mitigating performance degradation from heterogeneity. On this basis, a novel Cluster- Aware Multi-round Update (CAMU) strategy is proposed, which treats clusters as the basic units and adjusts the local update frequency based on the clustered contribution threshold, effectively reducing update bias and enhancing aggregation accuracy. The theoretical convergence of the CAMU strategy is rigorously validated. Meanwhile, based on the convergence upper bound, the local update frequency and transmission power of each cluster are jointly optimized to achieve an optimal balance between computation and communication resources under constrained conditions, significantly improving the convergence efficiency of FL. Experimental results demonstrate that the proposed method effectively improves the model performance of FL in heterogeneous environments and achieves a better balance between communication cost and computational load under limited resources. 

**Abstract (ZH)**: 无线联邦学习中资源约束对聚合效率和准确性的显著影响，尤其是在异构环境中设备展现出不同的数据分布和通信能力。本文提出了一种利用先验知识相似性进行聚类的策略，以分组具有相似数据和通信特性的设备，从而减轻异构性带来的性能下降。在此基础上，提出了一种新的聚类意识多轮更新（CAMU）策略，将簇作为基本单位，并根据聚类贡献阈值调整本地更新频率，有效减少了更新偏差并提高了聚合准确性。CAMU策略的理论收敛性得到了严格验证。同时，基于收敛上界，联合优化每个簇的本地更新频率和传输功率，在受限条件下实现计算和通信资源的最优平衡，显著提高了联邦学习的收敛效率。实验结果表明，所提出的方法有效地提高了联邦学习在异构环境中的模型性能，并在有限资源下实现了较好的通信成本和计算负载平衡。 

---
# AKD : Adversarial Knowledge Distillation For Large Language Models Alignment on Coding tasks 

**Title (ZH)**: AKD：面向编码任务的大语言模型对抗知识精炼iếc
user
Transformer模型的工作原理及在自然语言处理任务中的应用。 

**Authors**: Ilyas Oulkadda, Julien Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06267)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) for code generation, exemplified by GitHub Copilot\footnote{A coding extension powered by a Code-LLM to assist in code completion tasks} surpassing a million users, highlights the transformative potential of these tools in improving developer productivity. However, this rapid growth also underscores critical concerns regarding the quality, safety, and reliability of the code they generate. As Code-LLMs evolve, they face significant challenges, including the diminishing returns of model scaling and the scarcity of new, high-quality training data. To address these issues, this paper introduces Adversarial Knowledge Distillation (AKD), a novel approach that leverages adversarially generated synthetic datasets to distill the capabilities of larger models into smaller, more efficient ones. By systematically stress-testing and refining the reasoning capabilities of Code-LLMs, AKD provides a framework for enhancing model robustness, reliability, and security while improving their parameter-efficiency. We believe this work represents a critical step toward ensuring dependable automated code generation within the constraints of existing data and the cost-efficiency of model execution. 

**Abstract (ZH)**: 大型语言模型在代码生成中的广泛采用，以GitHub Copilot为例，用户数超过百万，凸显了这些工具在提升开发者生产力方面的变革潜力。然而，这种快速增长也引起了对其生成的代码质量、安全性和可靠性的关键关注。随着代码LLM的发展，它们面临着模型扩展回报递减和高质量训练数据稀缺的重大挑战。为了解决这些问题，本文提出了对抗性知识蒸馏（AKD）这一新颖方法，利用对抗生成的合成数据集将大模型的能力提炼至更小、更高效的模型中。通过系统性地压力测试和优化代码LLM的推理能力，AKD提供了一种框架，用于增强模型的鲁棒性、可靠性和安全性，同时提高其参数效率。我们认为，这项工作代表了确保在现有数据约束和模型执行成本效益下的可靠自动化代码生成的一个关键步骤。 

---
# Knowledge Guided Encoder-Decoder Framework Integrating Multiple Physical Models for Agricultural Ecosystem Modeling 

**Title (ZH)**: 知识引导的编码-解码框架整合多种物理模型进行农业生态系统建模 

**Authors**: Qi Cheng, Licheng Liu, Zhang Yao, Hong Mu, Shiyuan Luo, Zhenong Jin, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06266)  

**Abstract**: Agricultural monitoring is critical for ensuring food security, maintaining sustainable farming practices, informing policies on mitigating food shortage, and managing greenhouse gas emissions. Traditional process-based physical models are often designed and implemented for specific situations, and their parameters could also be highly uncertain. In contrast, data-driven models often use black-box structures and does not explicitly model the inter-dependence between different ecological variables. As a result, they require extensive training data and lack generalizability to different tasks with data distribution shifts and inconsistent observed variables. To address the need for more universal models, we propose a knowledge-guided encoder-decoder model, which can predict key crop variables by leveraging knowledge of underlying processes from multiple physical models. The proposed method also integrates a language model to process complex and inconsistent inputs and also utilizes it to implement a model selection mechanism for selectively combining the knowledge from different physical models. Our evaluations on predicting carbon and nitrogen fluxes for multiple sites demonstrate the effectiveness and robustness of the proposed model under various scenarios. 

**Abstract (ZH)**: 农业监测对于确保粮食安全、维持可持续农业实践、制定缓解粮食短缺的政策以及管理温室气体排放至关重要。传统的基于过程的物理模型通常为特定情况设计和实施，且其参数可能高度不确定。相比之下，数据驱动模型往往采用黑盒结构，不明确建模不同生态变量之间的相互依赖关系。因此，它们需要大量的训练数据，并且在数据分布转移和观测变量不一致的情况下缺乏普适性。为应对需要更加通用的模型的需求，我们提出了一种知识引导的编码解码模型，该模型通过从多个物理模型中汲取基础过程的知识来预测关键作物变量。该方法还集成了一种语言模型来处理复杂且不一致的输入，并利用其实施一种模型选择机制，以选择性地结合不同物理模型的知识。我们在多个站点预测碳和氮通量的评估表明，该模型在各种场景下具有有效性与鲁棒性。 

---
# Prediction of Delirium Risk in Mild Cognitive Impairment Using Time-Series data, Machine Learning and Comorbidity Patterns -- A Retrospective Study 

**Title (ZH)**: 使用时间序列数据、机器学习和共病模式预测轻度认知 impairment 患者的谵妄风险：一项回顾性研究 

**Authors**: Santhakumar Ramamoorthy, Priya Rani, James Mahon, Glenn Mathews, Shaun Cloherty, Mahdi Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2505.06264)  

**Abstract**: Delirium represents a significant clinical concern characterized by high morbidity and mortality rates, particularly in patients with mild cognitive impairment (MCI). This study investigates the associated risk factors for delirium by analyzing the comorbidity patterns relevant to MCI and developing a longitudinal predictive model leveraging machine learning methodologies. A retrospective analysis utilizing the MIMIC-IV v2.2 database was performed to evaluate comorbid conditions, survival probabilities, and predictive modeling outcomes. The examination of comorbidity patterns identified distinct risk profiles for the MCI population. Kaplan-Meier survival analysis demonstrated that individuals with MCI exhibit markedly reduced survival probabilities when developing delirium compared to their non-MCI counterparts, underscoring the heightened vulnerability within this cohort. For predictive modeling, a Long Short-Term Memory (LSTM) ML network was implemented utilizing time-series data, demographic variables, Charlson Comorbidity Index (CCI) scores, and an array of comorbid conditions. The model demonstrated robust predictive capabilities with an AUROC of 0.93 and an AUPRC of 0.92. This study underscores the critical role of comorbidities in evaluating delirium risk and highlights the efficacy of time-series predictive modeling in pinpointing patients at elevated risk for delirium development. 

**Abstract (ZH)**: 轻微认知障碍患者谵妄的相关风险因素研究：基于acomorbidity模式的纵向预测模型 

---
# Dialz: A Python Toolkit for Steering Vectors 

**Title (ZH)**: Dialz: 一个用于引导向量的Python工具包 

**Authors**: Zara Siddique, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2505.06262)  

**Abstract**: We introduce Dialz, a framework for advancing research on steering vectors for open-source LLMs, implemented in Python. Steering vectors allow users to modify activations at inference time to amplify or weaken a 'concept', e.g. honesty or positivity, providing a more powerful alternative to prompting or fine-tuning. Dialz supports a diverse set of tasks, including creating contrastive pair datasets, computing and applying steering vectors, and visualizations. Unlike existing libraries, Dialz emphasizes modularity and usability, enabling both rapid prototyping and in-depth analysis. We demonstrate how Dialz can be used to reduce harmful outputs such as stereotypes, while also providing insights into model behaviour across different layers. We release Dialz with full documentation, tutorials, and support for popular open-source models to encourage further research in safe and controllable language generation. Dialz enables faster research cycles and facilitates insights into model interpretability, paving the way for safer, more transparent, and more reliable AI systems. 

**Abstract (ZH)**: Dialz：一种用于开源LLM引导向量研究的框架 

---
# Modeling supply chain compliance response strategies based on AI synthetic data with structural path regression: A Simulation Study of EU 2027 Mandatory Labor Regulations 

**Title (ZH)**: 基于AI合成数据的结构路径回归建模：欧盟2027强制劳动规定下的供应链合规响应策略仿真研究 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06261)  

**Abstract**: In the context of the new mandatory labor compliance in the European Union (EU), which will be implemented in 2027, supply chain enterprises face stringent working hour management requirements and compliance risks. In order to scientifically predict the enterprises' coping behaviors and performance outcomes under the policy impact, this paper constructs a methodological framework that integrates the AI synthetic data generation mechanism and structural path regression modeling to simulate the enterprises' strategic transition paths under the new regulations. In terms of research methodology, this paper adopts high-quality simulation data generated based on Monte Carlo mechanism and NIST synthetic data standards to construct a structural path analysis model that includes multiple linear regression, logistic regression, mediation effect and moderating effect. The variable system covers 14 indicators such as enterprise working hours, compliance investment, response speed, automation level, policy dependence, etc. The variable set with explanatory power is screened out through exploratory data analysis (EDA) and VIF multicollinearity elimination. The findings show that compliance investment has a significant positive impact on firm survival and its effect is transmitted through the mediating path of the level of intelligence; meanwhile, firms' dependence on the EU market significantly moderates the strength of this mediating effect. It is concluded that AI synthetic data combined with structural path modeling provides an effective tool for high-intensity regulatory simulation, which can provide a quantitative basis for corporate strategic response, policy design and AI-assisted decision-making in the pre-prediction stage lacking real scenario data. Keywords: AI synthetic data, structural path regression modeling, compliance response strategy, EU 2027 mandatory labor regulation 

**Abstract (ZH)**: 欧盟2027年强制劳动合规政策背景下基于AI合成数据及结构路径回归模型的企业应对策略研究 

---
# Fair Clustering with Clusterlets 

**Title (ZH)**: 公平聚类与簇集/grouplet 

**Authors**: Mattia Setzu, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.06259)  

**Abstract**: Given their widespread usage in the real world, the fairness of clustering methods has become of major interest. Theoretical results on fair clustering show that fairness enjoys transitivity: given a set of small and fair clusters, a trivial centroid-based clustering algorithm yields a fair clustering. Unfortunately, discovering a suitable starting clustering can be computationally expensive, rather complex or arbitrary.
In this paper, we propose a set of simple \emph{clusterlet}-based fuzzy clustering algorithms that match single-class clusters, optimizing fair clustering. Matching leverages clusterlet distance, optimizing for classic clustering objectives, while also regularizing for fairness. Empirical results show that simple matching strategies are able to achieve high fairness, and that appropriate parameter tuning allows to achieve high cohesion and low overlap. 

**Abstract (ZH)**: 基于集群块的公平聚类算法研究 

---
# ABE: A Unified Framework for Robust and Faithful Attribution-Based Explainability 

**Title (ZH)**: ABE：一种统一的基于归因的解释性鲁棒且忠实的框架 

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Fang Chen, Jianlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06258)  

**Abstract**: Attribution algorithms are essential for enhancing the interpretability and trustworthiness of deep learning models by identifying key features driving model decisions. Existing frameworks, such as InterpretDL and OmniXAI, integrate multiple attribution methods but suffer from scalability limitations, high coupling, theoretical constraints, and lack of user-friendly implementations, hindering neural network transparency and interoperability. To address these challenges, we propose Attribution-Based Explainability (ABE), a unified framework that formalizes Fundamental Attribution Methods and integrates state-of-the-art attribution algorithms while ensuring compliance with attribution axioms. ABE enables researchers to develop novel attribution techniques and enhances interpretability through four customizable modules: Robustness, Interpretability, Validation, and Data & Model. This framework provides a scalable, extensible foundation for advancing attribution-based explainability and fostering transparent AI systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于 attribution 的可解释性框架 (ABE)：提升深度学习模型的可解释性和可信度 

---
# Beyond Attention: Toward Machines with Intrinsic Higher Mental States 

**Title (ZH)**: 超越注意力：迈向拥有内在高级心理状态的机器 

**Authors**: Ahsan Adeel  

**Link**: [PDF](https://arxiv.org/pdf/2505.06257)  

**Abstract**: Attending to what is relevant is fundamental to both the mammalian brain and modern machine learning models such as Transformers. Yet, determining relevance remains a core challenge, traditionally offloaded to learning algorithms like backpropagation. Inspired by recent cellular neurobiological evidence linking neocortical pyramidal cells to distinct mental states, this work shows how models (e.g., Transformers) can emulate high-level perceptual processing and awake thought (imagination) states to pre-select relevant information before applying attention. Triadic neuronal-level modulation loops among questions ($Q$), clues (keys, $K$), and hypotheses (values, $V$) enable diverse, deep, parallel reasoning chains at the representation level and allow a rapid shift from initial biases to refined understanding. This leads to orders-of-magnitude faster learning with significantly reduced computational demand (e.g., fewer heads, layers, and tokens), at an approximate cost of $\mathcal{O}(N)$, where $N$ is the number of input tokens. Results span reinforcement learning (e.g., CarRacing in a high-dimensional visual setup), computer vision, and natural language question answering. 

**Abstract (ZH)**: 关注相关信息是哺乳动物大脑和现代机器学习模型如变换器的基本要素。然而，确定相关性仍然是一个核心挑战，传统上由反向传播等学习算法承担。受最近细胞神经生物学证据启发，该工作展示了如何使模型（如变换器）模拟高级感知处理和清醒思考（想象）状态，在应用注意力之前预先筛选相关信息。由问题（$Q$）、线索（键，$K$）和假设（值，$V$）三元神经级调节回路实现多样、深入、并行的表示层面推理链，并允许从初始偏见迅速转向精炼理解。这导致了比传统方法快得多的学习速度，计算需求显著降低（如较少的头部、层和标记），并以约$\mathcal{O}(N)$的成本，其中$N$为输入标记的数量。结果涵盖强化学习（例如，高维视觉设置中的CarRacing）、计算机视觉和自然语言问答领域。 

---
# SpectrumFM: A Foundation Model for Intelligent Spectrum Management 

**Title (ZH)**: SpectrumFM：智能频谱管理的基座模型 

**Authors**: Fuhui Zhou, Chunyu Liu, Hao Zhang, Wei Wu, Qihui Wu, Derrick Wing Kwan Ng, Tony Q. S. Quek, Chan-Byoung Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.06256)  

**Abstract**: Intelligent spectrum management is crucial for improving spectrum efficiency and achieving secure utilization of spectrum resources. However, existing intelligent spectrum management methods, typically based on small-scale models, suffer from notable limitations in recognition accuracy, convergence speed, and generalization, particularly in the complex and dynamic spectrum environments. To address these challenges, this paper proposes a novel spectrum foundation model, termed SpectrumFM, establishing a new paradigm for spectrum management. SpectrumFM features an innovative encoder architecture that synergistically exploits the convolutional neural networks and the multi-head self-attention mechanisms to enhance feature extraction and enable robust representation learning. The model is pre-trained via two novel self-supervised learning tasks, namely masked reconstruction and next-slot signal prediction, which leverage large-scale in-phase and quadrature (IQ) data to achieve comprehensive and transferable spectrum representations. Furthermore, a parameter-efficient fine-tuning strategy is proposed to enable SpectrumFM to adapt to various downstream spectrum management tasks, including automatic modulation classification (AMC), wireless technology classification (WTC), spectrum sensing (SS), and anomaly detection (AD). Extensive experiments demonstrate that SpectrumFM achieves superior performance in terms of accuracy, robustness, adaptability, few-shot learning efficiency, and convergence speed, consistently outperforming conventional methods across multiple benchmarks. Specifically, SpectrumFM improves AMC accuracy by up to 12.1% and WTC accuracy by 9.3%, achieves an area under the curve (AUC) of 0.97 in SS at -4 dB signal-to-noise ratio (SNR), and enhances AD performance by over 10%. 

**Abstract (ZH)**: 智能频谱管理对于提高频谱效率和实现频谱资源的安全利用至关重要。现有基于小规模模型的智能频谱管理方法在识别准确性、收敛速度和泛化能力方面存在明显不足，特别是在复杂的动态频谱环境中。为应对这些挑战，本文提出了一种新型频谱基础模型——SpectrumFM，建立了一种新的频谱管理范式。SpectrumFM 特设了一种创新的编码器架构，结合了卷积神经网络和多头自注意力机制，以增强特征提取并实现稳健的表示学习。该模型通过两个新颖的自我监督学习任务——掩码重建和下一槽信号预测——进行了预训练，利用大规模同相和正交（IQ）数据实现全面和可迁移的频谱表示。此外，提出了一种参数高效微调策略，使SpectrumFM能够适应各种下游频谱管理任务，包括自动调制分类（AMC）、无线技术分类（WTC）、频谱感知（SS）和异常检测（AD）。广泛实验表明，SpectrumFM 在准确率、鲁棒性、适应性、少样本学习效率和收敛速度方面表现优异，全面优于传统方法。具体而言，SpectrumFM 将AMC准确性提高了12.1%，WTC准确性提高了9.3%，在-4 dB信噪比（SNR）下SS的曲线下面积（AUC）达到了0.97，且异常检测性能提升了超过10%。 

---
# DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion 

**Title (ZH)**: DeltaDPD：利用循环神经网络中的动态时间稀疏性实现宽带数字预-distortion的能效提升 

**Authors**: Yizhuo Wu, Yi Zhu, Kun Qian, Qinyu Chen, Anding Zhu, John Gajadharsing, Leo C. N. de Vreede, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06250)  

**Abstract**: Digital Predistortion (DPD) is a popular technique to enhance signal quality in wideband RF power amplifiers (PAs). With increasing bandwidth and data rates, DPD faces significant energy consumption challenges during deployment, contrasting with its efficiency goals. State-of-the-art DPD models rely on recurrent neural networks (RNN), whose computational complexity hinders system efficiency. This paper introduces DeltaDPD, exploring the dynamic temporal sparsity of input signals and neuronal hidden states in RNNs for energy-efficient DPD, reducing arithmetic operations and memory accesses while preserving satisfactory linearization performance. Applying a TM3.1a 200MHz-BW 256-QAM OFDM signal to a 3.5 GHz GaN Doherty RF PA, DeltaDPD achieves -50.03 dBc in Adjacent Channel Power Ratio (ACPR), -37.22 dB in Normalized Mean Square Error (NMSE) and -38.52 dBc in Error Vector Magnitude (EVM) with 52% temporal sparsity, leading to a 1.8X reduction in estimated inference power. The DeltaDPD code will be released after formal publication at this https URL. 

**Abstract (ZH)**: 基于输入信号和RNN神经隐藏状态的动态时域稀疏性实现高效数字预失真 

---
# United States Road Accident Prediction using Random Forest Predictor 

**Title (ZH)**: 美国道路事故预测基于随机森林预测模型 

**Authors**: Dominic Parosh Yamarthi, Haripriya Raman, Shamsad Parvin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06246)  

**Abstract**: Road accidents significantly threaten public safety and require in-depth analysis for effective prevention and mitigation strategies. This paper focuses on predicting accidents through the examination of a comprehensive traffic dataset covering 49 states in the United States. The dataset integrates information from diverse sources, including transportation departments, law enforcement, and traffic sensors. This paper specifically emphasizes predicting the number of accidents, utilizing advanced machine learning models such as regression analysis and time series analysis. The inclusion of various factors, ranging from environmental conditions to human behavior and infrastructure, ensures a holistic understanding of the dynamics influencing road safety. Temporal and spatial analysis further allows for the identification of trends, seasonal variations, and high-risk areas. The implications of this research extend to proactive decision-making for policymakers and transportation authorities. By providing accurate predictions and quantifiable insights into expected accident rates under different conditions, the paper aims to empower authorities to allocate resources efficiently and implement targeted interventions. The goal is to contribute to the development of informed policies and interventions that enhance road safety, creating a safer environment for all road users. Keywords: Machine Learning, Random Forest, Accident Prediction, AutoML, LSTM. 

**Abstract (ZH)**: 道路事故显著威胁公共安全，需要进行深入分析以制定有效的预防和缓解策略。本文通过分析覆盖美国49个州的综合交通数据集，重点关注事故预测。该数据集整合了来自交通部门、执法机构和交通传感器的多种信息。本文特别强调利用先进的机器学习模型，如回归分析和时间序列分析来预测事故数量。包括从环境条件到人类行为和基础设施的各种因素，确保对影响道路安全的动力学有一个全面的理解。通过时空分析，进一步识别出趋势、季节性变化和高风险区域。本文的研究结果对政策制定者和交通管理部门的前瞻性决策具有重要意义。通过提供准确的预测和不同条件下预期事故率的量化洞察，本文旨在帮助当局有效分配资源并实施有针对性的干预措施。目标是促进制定基于数据的政策和干预措施，提高道路安全，为所有道路使用者创造更安全的环境。关键词：机器学习，随机森林，事故预测，AutoML，LSTM。 

---
# Low-Complexity CNN-Based Classification of Electroneurographic Signals 

**Title (ZH)**: 基于CNN的低复杂度Electroneurographic信号分类 

**Authors**: Arek Berc Gokdag, Silvia Mura, Antonio Coviello, Michele Zhu, Maurizio Magarini, Umberto Spagnolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.06241)  

**Abstract**: Peripheral nerve interfaces (PNIs) facilitate neural recording and stimulation for treating nerve injuries, but real-time classification of electroneurographic (ENG) signals remains challenging due to constraints on complexity and latency, particularly in implantable devices. This study introduces MobilESCAPE-Net, a lightweight architecture that reduces computational cost while maintaining and slightly improving classification performance. Compared to the state-of-the-art ESCAPE-Net, MobilESCAPE-Net achieves comparable accuracy and F1-score with significantly lower complexity, reducing trainable parameters by 99.9\% and floating point operations per second by 92.47\%, enabling faster inference and real-time processing. Its efficiency makes it well-suited for low-complexity ENG signal classification in resource-constrained environments such as implantable devices. 

**Abstract (ZH)**: 基于移动设备的轻量级ENG信号分类网络（MobilESCAPE-Net）：一种降低复杂性和保持分类性能的方法 

---
