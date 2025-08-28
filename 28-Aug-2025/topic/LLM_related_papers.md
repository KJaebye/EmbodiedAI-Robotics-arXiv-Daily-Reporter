# CASE: An Agentic AI Framework for Enhancing Scam Intelligence in Digital Payments 

**Title (ZH)**: CASE: 促进数字支付防欺诈能力的agency型AI框架 

**Authors**: Nitish Jaipuria, Lorenzo Gatto, Zijun Kan, Shankey Poddar, Bill Cheung, Diksha Bansal, Ramanan Balakrishnan, Aviral Suri, Jose Estevez  

**Link**: [PDF](https://arxiv.org/pdf/2508.19932)  

**Abstract**: The proliferation of digital payment platforms has transformed commerce, offering unmatched convenience and accessibility globally. However, this growth has also attracted malicious actors, leading to a corresponding increase in sophisticated social engineering scams. These scams are often initiated and orchestrated on multiple surfaces outside the payment platform, making user and transaction-based signals insufficient for a complete understanding of the scam's methodology and underlying patterns, without which it is very difficult to prevent it in a timely manner. This paper presents CASE (Conversational Agent for Scam Elucidation), a novel Agentic AI framework that addresses this problem by collecting and managing user scam feedback in a safe and scalable manner. A conversational agent is uniquely designed to proactively interview potential victims to elicit intelligence in the form of a detailed conversation. The conversation transcripts are then consumed by another AI system that extracts information and converts it into structured data for downstream usage in automated and manual enforcement mechanisms. Using Google's Gemini family of LLMs, we implemented this framework on Google Pay (GPay) India. By augmenting our existing features with this new intelligence, we have observed a 21% uplift in the volume of scam enforcements. The architecture and its robust evaluation framework are highly generalizable, offering a blueprint for building similar AI-driven systems to collect and manage scam intelligence in other sensitive domains. 

**Abstract (ZH)**: 数字支付平台的 proliferation 已经改变了商业格局，提供了无与伦比的便捷性和全球可及性。然而，这一增长也吸引了恶意行为者，导致了相应的复杂社交工程骗局的增加。这些骗局通常在支付平台之外的多个表面上被策划启动，使得基于用户和交易的信号不足以完全理解骗局的方法和潜在模式，没有这些理解，及时预防骗局就非常困难。本文提出了 CASE（Conversational Agent for Scam Elucidation），一种新颖的代理人工智能框架，通过以安全和可扩展的方式收集和管理用户骗局反馈来解决这个问题。一个对话代理被特别设计为积极地采访潜在受害者，以提取详细对话形式的情报。随后，对话记录被另一人工智能系统消费，提取信息并转换为结构化数据，供下游自动和手动执法机制使用。通过使用谷歌的 Gemini 家族的大语言模型，我们在印度的 Google Pay（GPay）实现了这一框架。通过增强我们现有的功能，我们观察到骗局执法量增加了 21%。该架构及其稳健的评估框架具有高度的泛化性，提供了在其他敏感领域构建类似的人工智能驱动系统以收集和管理骗局情报的蓝图。 

---
# Tracking World States with Language Models: State-Based Evaluation Using Chess 

**Title (ZH)**: 使用语言模型追踪世界状态：基于棋弈的评估方法 

**Authors**: Romain Harang, Jason Naradowsky, Yaswitha Gujju, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2508.19851)  

**Abstract**: Large Language Models (LLMs) exhibit emergent capabilities in structured domains, suggesting they may implicitly internalize high-fidelity representations of world models. While probing techniques have shown promising signs of this in scientific and game-based settings, they rely on model-specific internal activations, which limit interpretability and generalizability. In this work, we propose a model-agnostic, state-based evaluation framework using chess as a benchmark to assess whether LLMs preserve the semantics of structured environments. Our method analyzes the downstream legal move distributions (state affordances) to estimate semantic fidelity between predicted and actual game states. This approach offers a more meaningful evaluation than conventional string-based metrics by aligning more closely with the strategic and rule-governed nature of chess. Experimental results demonstrate that our metrics capture deficiencies in state-tracking, highlighting limitations of LLMs in maintaining coherent internal models over long sequences. Our framework provides a robust tool for evaluating structured reasoning in LLMs without requiring internal model access, and generalizes to a wide class of symbolic environments. 

**Abstract (ZH)**: 大型语言模型在结构化领域展现出新兴能力，表明它们可能隐式地内部化了世界模型的高保真表示。虽然探针技术在科学和基于游戏的环境中展示了这一潜力，但它们依赖于特定模型的内部激活，这限制了可解释性和泛化性。在本工作中，我们提出了一个模型无关的状态评估框架，使用国际象棋作为基准，评估大型语言模型是否保留了结构化环境的语义。我们的方法通过分析下游合法走法分布（状态可操作性），来估算预测与实际棋局状态之间的语义保真度。该方法通过与国际象棋的战略性和规则导向本质更紧密地对齐，提供了比传统基于字符串的指标更具意义的评估。实验结果表明，我们的指标捕捉到了状态跟踪的不足，突显了大型语言模型在长时间序列中保持连贯内部模型的能力的局限性。该框架为评估大型语言模型中的结构化推理提供了一个鲁棒工具，无需访问内部模型，并且可以泛化到广泛类别的符号环境。 

---
# Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties 

**Title (ZH)**: 教学代理：自动化课程材料生成的LLM代理 

**Authors**: Huaiyuan Yao, Wanpeng Xu, Justin Turnau, Nadia Kellam, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.19611)  

**Abstract**: Preparing high-quality instructional materials remains a labor-intensive process that often requires extensive coordination among teaching faculty, instructional designers, and teaching assistants. In this work, we present Instructional Agents, a multi-agent large language model (LLM) framework designed to automate end-to-end course material generation, including syllabus creation, lecture scripts, LaTeX-based slides, and assessments. Unlike existing AI-assisted educational tools that focus on isolated tasks, Instructional Agents simulates role-based collaboration among educational agents to produce cohesive and pedagogically aligned content. The system operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot mode, enabling flexible control over the degree of human involvement. We evaluate Instructional Agents across five university-level computer science courses and show that it produces high-quality instructional materials while significantly reducing development time and human workload. By supporting institutions with limited instructional design capacity, Instructional Agents provides a scalable and cost-effective framework to democratize access to high-quality education, particularly in underserved or resource-constrained settings. 

**Abstract (ZH)**: Preparing High-Quality Instructional Materials with Multi-Agent Large Language Models: A Multi-Mode Framework for Automated Course Material Generation 

---
# ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding 

**Title (ZH)**: ReST-RL：通过优化自我训练和解码实现准确的LLM代码推理 

**Authors**: Sining Zhoubian, Dan Zhang, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19576)  

**Abstract**: With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We validate the effectiveness of the proposed RL paradigm through extensive experiments on coding problems. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at this https URL. 

**Abstract (ZH)**: 基于改进GRPO算法和值模型辅助测试时间解码方法的统一语言模型强化学习范式ReST-RL 

---
# Caught in the Act: a mechanistic approach to detecting deception 

**Title (ZH)**: Caught in the Act: 一种机制性的欺骗检测方法 

**Authors**: Gerard Boxo, Ryan Socha, Daniel Yoo, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2508.19505)  

**Abstract**: Sophisticated instrumentation for AI systems might have indicators that signal misalignment from human values, not unlike a "check engine" light in cars. One such indicator of misalignment is deceptiveness in generated responses. Future AI instrumentation may have the ability to detect when an LLM generates deceptive responses while reasoning about seemingly plausible but incorrect answers to factual questions. In this work, we demonstrate that linear probes on LLMs internal activations can detect deception in their responses with extremely high accuracy. Our probes reach a maximum of greater than 90% accuracy in distinguishing between deceptive and non-deceptive arguments generated by llama and qwen models ranging from 1.5B to 14B parameters, including their DeepSeek-r1 finetuned variants. We observe that probes on smaller models (1.5B) achieve chance accuracy at detecting deception, while larger models (greater than 7B) reach 70-80%, with their reasoning counterparts exceeding 90%. The layer-wise probe accuracy follows a three-stage pattern across layers: near-random (50%) in early layers, peaking in middle layers, and slightly declining in later layers. Furthermore, using an iterative null space projection approach, we find multitudes of linear directions that encode deception, ranging from 20 in Qwen 3B to nearly 100 in DeepSeek 7B and Qwen 14B models. 

**Abstract (ZH)**: 复杂的AI系统 instrumentation 或许具有类似于汽车“发动机故障”警示灯的指标，用以信号化与人类价值观的不一致。生成回应中的欺骗性即是一种不一致的指标。未来AI instrumentation可能具备检测大型语言模型在推理看似合理的但不正确的答案时生成欺骗性回应的能力。在这项工作中，我们证明了对大型语言模型内部激活进行线性探测可以极其准确地检测其回应中的欺骗性。我们的探测器在从1.5B到14B参数的llama和qwen模型（包括其DeepSeek-r1微调变体）生成的欺骗性和非欺骗性论点中达到了超过90%的最高准确率。我们发现，较小模型（1.5B）的探测器在检测欺骗性方面仅能达到随机准确性，而较大模型（超过7B）的准确率达到70-80%，其推理对应物则超过了90%。逐层探测器的准确度呈现三阶段模式：早期层接近随机（50%），中间层达到峰值，后期层略有下降。此外，通过迭代零空间投影方法，我们发现qwen 3B中有20种，DeepSeek 7B和qwen 14B模型中有近100种线性方向编码了欺骗性。 

---
# Reliable Weak-to-Strong Monitoring of LLM Agents 

**Title (ZH)**: 可靠的从弱监督到强监督的LLM代理监控 

**Authors**: Neil Kale, Chen Bo Calvin Zhang, Kevin Zhu, Ankit Aich, Paula Rodriguez, Scale Red Team, Christina Q. Knight, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19461)  

**Abstract**: We stress test monitoring systems for detecting covert misbehavior in autonomous LLM agents (e.g., secretly sharing private information). To this end, we systematize a monitor red teaming (MRT) workflow that incorporates: (1) varying levels of agent and monitor situational awareness; (2) distinct adversarial strategies to evade the monitor, such as prompt injection; and (3) two datasets and environments -- SHADE-Arena for tool-calling agents and our new CUA-SHADE-Arena, which extends TheAgentCompany, for computer-use agents. We run MRT on existing LLM monitor scaffoldings, which orchestrate LLMs and parse agent trajectories, alongside a new hybrid hierarchical-sequential scaffolding proposed in this work. Our empirical results yield three key findings. First, agent awareness dominates monitor awareness: an agent's knowledge that it is being monitored substantially degrades the monitor's reliability. On the contrary, providing the monitor with more information about the agent is less helpful than expected. Second, monitor scaffolding matters more than monitor awareness: the hybrid scaffolding consistently outperforms baseline monitor scaffolding, and can enable weaker models to reliably monitor stronger agents -- a weak-to-strong scaling effect. Third, in a human-in-the-loop setting where humans discuss with the LLM monitor to get an updated judgment for the agent's behavior, targeted human oversight is most effective; escalating only pre-flagged cases to human reviewers improved the TPR by approximately 15% at FPR = 0.01. Our work establishes a standard workflow for MRT, highlighting the lack of adversarial robustness for LLMs and humans when monitoring and detecting agent misbehavior. We release code, data, and logs to spur further research. 

**Abstract (ZH)**: 我们对检测自主大语言模型代理潜行不当行为的监控系统进行压力测试（例如，秘密共享私人信息）。为此，我们系统化了一种监控红队测试（MRT）工作流，其中包括：（1）不同的代理和监控情境意识等级；（2）不同的对手策略以规避监控，如提示注入；（3）两个数据集和环境——SHADE-Arena 对于工具调用代理，以及我们新提出的 CUA-SHADE-Arena，扩展了 TheAgentCompany，用于计算机使用代理。我们在现有的大语言模型监控框架上运行 MRT，这些框架协调大语言模型并解析代理轨迹，同时采用本文提出的新混合层次-顺序框架。我们的实证结果得出三项关键发现。首先，代理意识优于监控意识：代理意识到自身被监控会显著降低监控的可靠性。相反，向监控提供更多信息关于代理的资料的效果不如预期。其次，监控框架比监控意识更重要：混合框架始终优于基线监控框架，并能使其较弱的模型可靠地监控较强的代理——这是一种弱到强的扩展效应。第三，在人类在环设置中，人类与大语言模型监控讨论以获取代理行为的更新判断，针对性的人类监督最有效；仅将预先标记的案例升级给人类审查员，可使在 FPR=0.01 时的真正阳性率提高约 15%。我们建立了 MRT 的标准工作流，强调对大语言模型和人类在监控和检测代理不当行为时缺乏对抗鲁棒性。我们发布代码、数据和日志以促进进一步研究。 

---
# Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs 

**Title (ZH)**: 量化但欺骗性十足？关于量化LLM的多维度真实性评估 

**Authors**: Yao Fu, Xianxuan Long, Runchao Li, Haotian Yu, Mu Sheng, Xiaotian Han, Yu Yin, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19432)  

**Abstract**: Quantization enables efficient deployment of large language models (LLMs) in resource-constrained environments by significantly reducing memory and computation costs. While quantized LLMs often maintain performance on perplexity and zero-shot tasks, their impact on truthfulness-whether generating truthful or deceptive responses-remains largely unexplored. In this work, we introduce TruthfulnessEval, a comprehensive evaluation framework for assessing the truthfulness of quantized LLMs across three dimensions: (1) Truthfulness on Logical Reasoning; (2) Truthfulness on Common Sense; and (3) Truthfulness on Imitative Falsehoods. Using this framework, we examine mainstream quantization techniques (ranging from 4-bit to extreme 2-bit) across several open-source LLMs. Surprisingly, we find that while quantized models retain internally truthful representations, they are more susceptible to producing false outputs under misleading prompts. To probe this vulnerability, we test 15 rephrased variants of "honest", "neutral" and "deceptive" prompts and observe that "deceptive" prompts can override truth-consistent behavior, whereas "honest" and "neutral" prompts maintain stable outputs. Further, we reveal that quantized models "know" the truth internally yet still produce false outputs when guided by "deceptive" prompts via layer-wise probing and PCA visualizations. Our findings provide insights into future designs of quantization-aware alignment and truthfulness interventions. 

**Abstract (ZH)**: 量化通过显著降低内存和计算成本，使大规模语言模型（LLMs）在资源受限环境中高效部署。虽然量化LLMs在困惑度和零-shot任务上通常保持性能，但它们对真实性——即生成真实或欺骗性回应的能力——的影响仍 largely unexplored。在这项工作中，我们引入了TruthfulnessEval，这是一个全面评估量化LLMs真实性的框架，涵盖三个维度：（1）逻辑推理的真实性；（2）常识的真实性；以及（3）模仿的虚假性。利用这一框架，我们考察了多种主流量化技术（从4位到极端的2位）在多个开源LLMs上的表现。我们惊讶地发现，尽管量化模型保留了内部真实的表现，但在误导性提示下更容易产生虚假输出。为了探测这种脆弱性，我们测试了“诚实”、“中立”和“欺骗”提示的15种不同表述形式，并观察到“欺骗”提示可以覆盖一致性真实行为，而“诚实”和“中立”提示能产生稳定的输出。此外，通过逐层探针和PCA可视化，我们揭示了量化模型内部“知道”真相但被“欺骗”提示引导时仍产生虚假输出的现象。我们的发现为未来量化感知对齐和真实性干预的设计提供了见解。 

---
# Sycophancy as compositions of Atomic Psychometric Traits 

**Title (ZH)**: 阿基米德心理测量特质的拟颂行为组成 

**Authors**: Shreyans Jain, Alexandra Yost, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2508.19316)  

**Abstract**: Sycophancy is a key behavioral risk in LLMs, yet is often treated as an isolated failure mode that occurs via a single causal mechanism. We instead propose modeling it as geometric and causal compositions of psychometric traits such as emotionality, openness, and agreeableness - similar to factor decomposition in psychometrics. Using Contrastive Activation Addition (CAA), we map activation directions to these factors and study how different combinations may give rise to sycophancy (e.g., high extraversion combined with low conscientiousness). This perspective allows for interpretable and compositional vector-based interventions like addition, subtraction and projection; that may be used to mitigate safety-critical behaviors in LLMs. 

**Abstract (ZH)**: 自谦行为是大模型中的一个关键行为风险，但通常被当作通过单一因果机制发生的一种孤立的失败模式。相反，我们认为可以将其建模为情感性、开放性和宜人性等心理测量特质的几何和因果组合，类似于心理测量中的因子分解。通过对比激活添加（CAA），我们将激活方向映射到这些因子，并研究不同的组合如何导致自谦行为（如高外向性与低尽责性相结合）。这种观点允许使用可解释和组合向量干预，如加法、减法和投影，以减轻大模型中的关键安全性行为。 

---
# Large Language Models (LLMs) for Electronic Design Automation (EDA) 

**Title (ZH)**: 大型语言模型(L large language models)在电子设计自动化(EDA)中的应用 

**Authors**: Kangwei Xu, Denis Schwachhofer, Jason Blocklove, Ilia Polian, Peter Domanski, Dirk Pflüger, Siddharth Garg, Ramesh Karri, Ozgur Sinanoglu, Johann Knechtel, Zhuorui Zhao, Ulf Schlichtmann, Bing Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20030)  

**Abstract**: With the growing complexity of modern integrated circuits, hardware engineers are required to devote more effort to the full design-to-manufacturing workflow. This workflow involves numerous iterations, making it both labor-intensive and error-prone. Therefore, there is an urgent demand for more efficient Electronic Design Automation (EDA) solutions to accelerate hardware development. Recently, large language models (LLMs) have shown remarkable advancements in contextual comprehension, logical reasoning, and generative capabilities. Since hardware designs and intermediate scripts can be represented as text, integrating LLM for EDA offers a promising opportunity to simplify and even automate the entire workflow. Accordingly, this paper provides a comprehensive overview of incorporating LLMs into EDA, with emphasis on their capabilities, limitations, and future opportunities. Three case studies, along with their outlook, are introduced to demonstrate the capabilities of LLMs in hardware design, testing, and optimization. Finally, future directions and challenges are highlighted to further explore the potential of LLMs in shaping the next-generation EDA, providing valuable insights for researchers interested in leveraging advanced AI technologies for EDA. 

**Abstract (ZH)**: 随着现代集成电路的日益复杂，硬件工程师需要在完整的设计到制造工作流程中投入更多努力。这一工作流程涉及大量的迭代，使其既劳动力密集又容易出错。因此，迫切需要更高效的电子设计自动化（EDA）解决方案来加速硬件开发。近年来，大型语言模型（LLMs）在上下文理解和逻辑推理以及生成能力方面取得了显著进步。由于硬件设计和中间脚本可以表示为文本，将LLM集成到EDA中提供了一种简化甚至自动化整个工作流程的有前景的机会。因此，本文提供了将LLM集成到EDA中的全面概述，重点介绍其能力和限制，以及未来的机会。介绍了三个案例及其展望，以展示LLM在硬件设计、测试和优化方面的能力。最后，提出了未来方向和挑战，以进一步探索LLM在塑造下一代EDA中的潜力，为希望利用先进AI技术进行EDA的研究人员提供有价值的见解。 

---
# Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment 

**Title (ZH)**: 分解大型语言模型中的行为相变：新兴不对齐的秩序参数 

**Authors**: Julian Arnold, Niels Lörch  

**Link**: [PDF](https://arxiv.org/pdf/2508.20015)  

**Abstract**: Fine-tuning LLMs on narrowly harmful datasets can lead to behavior that is broadly misaligned with respect to human values. To understand when and how this emergent misalignment occurs, we develop a comprehensive framework for detecting and characterizing rapid transitions during fine-tuning using both distributional change detection methods as well as order parameters that are formulated in plain English and evaluated by an LLM judge. Using an objective statistical dissimilarity measure, we quantify how the phase transition that occurs during fine-tuning affects multiple aspects of the model. In particular, we assess what percentage of the total distributional change in model outputs is captured by different aspects, such as alignment or verbosity, providing a decomposition of the overall transition. We also find that the actual behavioral transition occurs later in training than indicated by the peak in the gradient norm alone. Our framework enables the automated discovery and quantification of language-based order parameters, which we demonstrate on examples ranging from knowledge questions to politics and ethics. 

**Abstract (ZH)**: 在细调大语言模型时使用狭义有害数据集可能导致总体上与人类价值观偏差的行为。为了理解这种 emergent 偏差何时以及如何发生，我们开发了一个综合框架，使用分布变化检测方法以及用简单语言表述的秩序参数，这些参数通过大语言模型裁判进行评估。通过客观的统计异质性度量，我们量化了在细调过程中发生的相变对模型多个方面的影响。特别是，我们评估了不同方面，如对齐度或冗余度，捕捉到的总分布变化的百分比，提供了对整体转变的分解。我们还发现，实际行为转变发生在训练过程中晚于仅由梯度范数峰值指示的时间点。该框架允许自动化发现和量化基于语言的秩序参数，在从知识问题到政治和伦理等多种示例中进行了验证。 

---
# MathBuddy: A Multimodal System for Affective Math Tutoring 

**Title (ZH)**: MathBuddy: 一种多模态情感数学辅导系统 

**Authors**: Debanjana Kar, Leopold Böss, Dacia Braca, Sebastian Maximilian Dennerlein, Nina Christine Hubig, Philipp Wintersberger, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2508.19993)  

**Abstract**: The rapid adoption of LLM-based conversational systems is already transforming the landscape of educational technology. However, the current state-of-the-art learning models do not take into account the student's affective states. Multiple studies in educational psychology support the claim that positive or negative emotional states can impact a student's learning capabilities. To bridge this gap, we present MathBuddy, an emotionally aware LLM-powered Math Tutor, which dynamically models the student's emotions and maps them to relevant pedagogical strategies, making the tutor-student conversation a more empathetic one. The student's emotions are captured from the conversational text as well as from their facial expressions. The student's emotions are aggregated from both modalities to confidently prompt our LLM Tutor for an emotionally-aware response. We have effectively evaluated our model using automatic evaluation metrics across eight pedagogical dimensions and user studies. We report a massive 23 point performance gain using the win rate and a 3 point gain at an overall level using DAMR scores which strongly supports our hypothesis of improving LLM-based tutor's pedagogical abilities by modeling students' emotions. 

**Abstract (ZH)**: 基于LLM的情感意识数学辅导系统MathBuddy：通过建模学生情感提高辅导效果 

---
# Diffusion Language Models Know the Answer Before Decoding 

**Title (ZH)**: 扩散语言模型在解码前就知道答案 

**Authors**: Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19982)  

**Abstract**: Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high quality outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer convergence: in many cases, the correct answer can be internally identified by half steps before the final decoding step, both under semi-autoregressive and random remasking schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can be decoded correctly using only half of the refinement steps. Building on this observation, we introduce Prophet, a training-free fast decoding paradigm that enables early commit decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training. Empirical evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4x while preserving high generation quality. These results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early decode convergence provides a simple yet powerful mechanism for accelerating DLM inference, complementary to existing speedup techniques. Our code is publicly available at this https URL. 

**Abstract (ZH)**: Diffusion语言模型（DLMs）作为一种替代自回归方法的新颖技术，提供了并行序列生成和灵活的令牌顺序。然而，其推理速度仍慢于自回归模型，主要原因是双向注意的成本以及生成高质量输出所需的高度精炼步骤较多。在本文中，我们指出并利用了DLMs早期答案收敛这一未被充分关注的特性：在许多情况下，正确的答案在最终解码步骤之前通过半步即可内部识别，无论是在半自回归解码还是随机屏蔽计划下。例如，在GSM8K和MMLU上，分别有97%和99%的实例可以在仅使用一半的精炼步骤时被正确解码。基于这一观察，我们引入了Prophet，这是一种无需训练的快速解码范式，使早期提交解码成为可能。具体而言，Prophet 动态决定是否继续精炼或“押注”（即一次性解码剩余所有令牌），使用最佳两个预测候选之间的置信度差距作为标准。它无缝集成到现有的DLM实现中，几乎不增加额外开销，也不需要额外训练。跨多个任务对LLaDA-8B和Dream-7B的实证评估显示，Prophet 将解码步骤减少了多达3.4倍，同时保持了高质量的生成。这些结果重新定义了DLM解码问题为何时停止采样问题，并证明了早期解码收敛为加速DLM推理提供了一种简单且强大的机制，补充了现有加速技术。我们的代码可在以下网址公开获取。 

---
# GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity 

**Title (ZH)**: GLSim: 在LVLMs中通过全局-局部相似性检测对象幻觉 

**Authors**: Seongheon Park, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19972)  

**Abstract**: Object hallucination in large vision-language models presents a significant challenge to their safe deployment in real-world applications. Recent works have proposed object-level hallucination scores to estimate the likelihood of object hallucination; however, these methods typically adopt either a global or local perspective in isolation, which may limit detection reliability. In this paper, we introduce GLSim, a novel training-free object hallucination detection framework that leverages complementary global and local embedding similarity signals between image and text modalities, enabling more accurate and reliable hallucination detection in diverse scenarios. We comprehensively benchmark existing object hallucination detection methods and demonstrate that GLSim achieves superior detection performance, outperforming competitive baselines by a significant margin. 

**Abstract (ZH)**: 大型 vision-language 模型中的对象幻觉在其实用应用中的安全部署中提出了重大挑战。近期工作提出了对象级幻觉分数以估计对象幻觉的可能性；然而，这些方法通常单独采用全局或局部视角，这可能限制了检测可靠性。在本文中，我们引入了 GLSim，一种无需训练的新型对象幻觉检测框架，该框架利用图像和文本模态之间互补的全局和局部嵌入相似性信号，能够在多种场景下实现更准确可靠的幻觉检测。我们全面 benchmark 了现有的对象幻觉检测方法，并证明 GLSim 在检测性能上表现出色，显著优于竞品基线。 

---
# Dhati+: Fine-tuned Large Language Models for Arabic Subjectivity Evaluation 

**Title (ZH)**: Dhati+: 细化调优的阿拉伯语主观性评价大型语言模型 

**Authors**: Slimane Bellaouar, Attia Nehar, Soumia Souffi, Mounia Bouameur  

**Link**: [PDF](https://arxiv.org/pdf/2508.19966)  

**Abstract**: Despite its significance, Arabic, a linguistically rich and morphologically complex language, faces the challenge of being under-resourced. The scarcity of large annotated datasets hampers the development of accurate tools for subjectivity analysis in Arabic. Recent advances in deep learning and Transformers have proven highly effective for text classification in English and French. This paper proposes a new approach for subjectivity assessment in Arabic textual data. To address the dearth of specialized annotated datasets, we developed a comprehensive dataset, AraDhati+, by leveraging existing Arabic datasets and collections (ASTD, LABR, HARD, and SANAD). Subsequently, we fine-tuned state-of-the-art Arabic language models (XLM-RoBERTa, AraBERT, and ArabianGPT) on AraDhati+ for effective subjectivity classification. Furthermore, we experimented with an ensemble decision approach to harness the strengths of individual models. Our approach achieves a remarkable accuracy of 97.79\,\% for Arabic subjectivity classification. Results demonstrate the effectiveness of the proposed approach in addressing the challenges posed by limited resources in Arabic language processing. 

**Abstract (ZH)**: 尽管阿拉伯语作为一种语义丰富且形态复杂的语言具有重要意义，但由于资源匮乏的挑战，其主观性分析工具的发展受到了限制。现有的大型标注数据集稀缺阻碍了阿拉伯语主观性分析工具的发展。最近，在深度学习和变换器技术的进步在英语和法语文本分类方面已证明非常有效。本文提出了一种新的阿拉伯语文本主观性评估方法。为解决专门标注数据的匮乏问题，我们通过利用现有阿拉伯语数据集和集合（ASTD、LABR、HARD和SANAD）开发了一个综合数据集AraDhati+。随后，我们针对AraDhati+对最先进的阿拉伯语语言模型（XLM-RoBERTa、AraBERT和ArabianGPT）进行微调，以实现有效的主观性分类。此外，我们尝试了一种集成决策方法，以充分利用各个模型的优势。我们的方法在阿拉伯语主观性分类中实现了97.79%的高准确率。结果表明，所提出的方法在阿拉伯语处理资源有限的挑战中具有有效性。 

---
# Logical Reasoning with Outcome Reward Models for Test-Time Scaling 

**Title (ZH)**: 基于结果奖励模型的逻辑推理及其测试时缩放方法 

**Authors**: Ramya Keerthy Thatikonda, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.19903)  

**Abstract**: Logical reasoning is a critical benchmark for evaluating the capabilities of large language models (LLMs), as it reflects their ability to derive valid conclusions from given premises. While the combination of test-time scaling with dedicated outcome or process reward models has opened up new avenues to enhance LLMs performance in complex reasoning tasks, this space is under-explored in deductive logical reasoning. We present a set of Outcome Reward Models (ORMs) for deductive reasoning. To train the ORMs we mainly generate data using Chain-of-Thought (CoT) with single and multiple samples. Additionally, we propose a novel tactic to further expand the type of errors covered in the training dataset of the ORM. In particular, we propose an echo generation technique that leverages LLMs' tendency to reflect incorrect assumptions made in prompts to extract additional training data, covering previously unexplored error types. While a standard CoT chain may contain errors likely to be made by the reasoner, the echo strategy deliberately steers the model toward incorrect reasoning. We show that ORMs trained on CoT and echo-augmented data demonstrate improved performance on the FOLIO, JustLogic, and ProverQA datasets across four different LLMs. 

**Abstract (ZH)**: 逻辑推理是评估大型语言模型（LLMs）能力的关键基准，因为它反映了模型从给定前提中推导出有效结论的能力。虽然在复杂推理任务中通过测试时缩放与专用结果或过程奖励模型的结合已为提高LLMs的性能开辟了新途径，但在演绎逻辑推理方面这一领域仍处于探索阶段。我们提出了一组演绎推理的Outcome Reward Models（ORMs）。为了训练ORMs，我们主要使用带有单个和多个样本的Chain-of-Thought（CoT）生成数据。此外，我们提出了一种新的策略，以进一步扩展ORMs训练数据集中涵盖的错误类型。特别是，我们提出了一种回声生成技术，利用LLMs在提示中作出的错误假设来提取额外的训练数据，涵盖了之前未被探索的错误类型。虽然标准的CoT链可能包含推理者可能犯的错误，但回声策略故意引导模型走向错误的推理。我们发现，基于CoT和回声增强数据训练的ORMs在四个不同的LLMs上，在FOLIO、JustLogic和ProverQA数据集上的表现得到了改善。 

---
# The Information Dynamics of Generative Diffusion 

**Title (ZH)**: 生成性扩散的信息动力学 

**Authors**: Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2508.19897)  

**Abstract**: Generative diffusion models have emerged as a powerful class of models in machine learning, yet a unified theoretical understanding of their operation is still developing. This perspective paper provides an integrated perspective on generative diffusion by connecting their dynamic, information-theoretic, and thermodynamic properties under a unified mathematical framework. We demonstrate that the rate of conditional entropy production during generation (i.e. the generative bandwidth) is directly governed by the expected divergence of the score function's vector field. This divergence, in turn, is linked to the branching of trajectories and generative bifurcations, which we characterize as symmetry-breaking phase transitions in the energy landscape. This synthesis offers a powerful insight: the process of generation is fundamentally driven by the controlled, noise-induced breaking of (approximate) symmetries, where peaks in information transfer correspond to critical transitions between possible outcomes. The score function acts as a dynamic non-linear filter that regulates the bandwidth of the noise by suppressing fluctuations that are incompatible with the data. 

**Abstract (ZH)**: 生成扩散模型已经发展成为机器学习中一类强大的模型，但对其运行机制的统一理论理解仍在发展中。这篇视角论文通过在一个统一的数学框架下连接生成扩散的动力学、信息论和热力学性质，提供了对其生成机制的综合视角。我们证明，在生成过程中条件熵生成速率（即生成带宽）直接由分数函数向量场的预期散度控制。这一散度又与轨迹分支和生成分岔相关联，我们将其表征为能量景观中的对称性破缺相变。这种综合提供了有力的洞察：生成过程从根本上是由可控的、噪声诱导的（近似）对称性破缺驱动的，信息传输的峰值对应于可能结果之间的关键转变。分数函数作为动态非线性滤波器，通过抑制不符合数据的波动来调节噪声的带宽。 

---
# SoK: Large Language Model Copyright Auditing via Fingerprinting 

**Title (ZH)**: SoK: 大型语言模型版权审计通过指纹技术 

**Authors**: Shuo Shao, Yiming Li, Yu He, Hongwei Yao, Wenyuan Yang, Dacheng Tao, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19843)  

**Abstract**: The broad capabilities and substantial resources required to train Large Language Models (LLMs) make them valuable intellectual property, yet they remain vulnerable to copyright infringement, such as unauthorized use and model theft. LLM fingerprinting, a non-intrusive technique that extracts and compares the distinctive features from LLMs to identify infringements, offers a promising solution to copyright auditing. However, its reliability remains uncertain due to the prevalence of diverse model modifications and the lack of standardized evaluation. In this SoK, we present the first comprehensive study of LLM fingerprinting. We introduce a unified framework and formal taxonomy that categorizes existing methods into white-box and black-box approaches, providing a structured overview of the state of the art. We further propose LeaFBench, the first systematic benchmark for evaluating LLM fingerprinting under realistic deployment scenarios. Built upon mainstream foundation models and comprising 149 distinct model instances, LeaFBench integrates 13 representative post-development techniques, spanning both parameter-altering methods (e.g., fine-tuning, quantization) and parameter-independent mechanisms (e.g., system prompts, RAG). Extensive experiments on LeaFBench reveal the strengths and weaknesses of existing methods, thereby outlining future research directions and critical open problems in this emerging field. The code is available at this https URL. 

**Abstract (ZH)**: 大语言模型的能力和资源要求使其成为有价值的知识产权，但它们仍然容易受到版权侵犯，如未授权使用和模型窃取。大语言模型指纹识别是一种非侵入性技术，通过提取和比较大语言模型的独特特征来识别侵权行为，为版权审计提供了有希望的解决方案。然而，其可靠性仍不确定，原因在于模型修改的多样性以及缺乏标准化的评估标准。在本综述中，我们首次进行了全面的大语言模型指纹识别研究。我们介绍了一个统一的框架和形式化的分类法，将现有方法分为白盒和黑盒方法，提供了该领域的现状概览。进一步提出了LeaFBench，这是首个在实际部署场景下系统评估大语言模型指纹识别的标准基准。基于主流基础模型，LeaFBench包含149个不同的模型实例，并集成了13种代表性后开发技术，涵盖了参数改变方法（如微调、量化）和与参数无关的机制（如系统提示、基于检索的回答生成）。通过对LeaFBench的广泛实验揭示了现有方法的优缺点，从而勾勒出该新兴领域的未来研究方向和关键打开问题。代码可在以下链接获取。 

---
# NLKI: A lightweight Natural Language Knowledge Integration Framework for Improving Small VLMs in Commonsense VQA Tasks 

**Title (ZH)**: NLKI：一种轻量级自然语言知识集成框架，用于增强小规模VLM在常识VQA任务中的性能 

**Authors**: Aritra Dutta, Swapnanil Mukherjee, Deepanway Ghosal, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2508.19724)  

**Abstract**: Commonsense visual-question answering often hinges on knowledge that is missing from the image or the question. Small vision-language models (sVLMs) such as ViLT, VisualBERT and FLAVA therefore lag behind their larger generative counterparts. To study the effect of careful commonsense knowledge integration on sVLMs, we present an end-to-end framework (NLKI) that (i) retrieves natural language facts, (ii) prompts an LLM to craft natural language explanations, and (iii) feeds both signals to sVLMs respectively across two commonsense VQA datasets (CRIC, AOKVQA) and a visual-entailment dataset (e-SNLI-VE). Facts retrieved using a fine-tuned ColBERTv2 and an object information-enriched prompt yield explanations that largely cut down hallucinations, while lifting the end-to-end answer accuracy by up to 7% (across 3 datasets), making FLAVA and other models in NLKI match or exceed medium-sized VLMs such as Qwen-2 VL-2B and SmolVLM-2.5B. As these benchmarks contain 10-25% label noise, additional finetuning using noise-robust losses (such as symmetric cross entropy and generalised cross entropy) adds another 2.5% in CRIC, and 5.5% in AOKVQA. Our findings expose when LLM-based commonsense knowledge beats retrieval from commonsense knowledge bases, how noise-aware training stabilises small models in the context of external knowledge augmentation, and why parameter-efficient commonsense reasoning is now within reach for 250M models. 

**Abstract (ZH)**: 精细常识知识整合对小型视觉-语言模型效果的影响研究 

---
# Safety Alignment Should Be Made More Than Just A Few Attention Heads 

**Title (ZH)**: 安全对齐不应仅由少数注意力头来实现。 

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19697)  

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety this http URL investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility. 

**Abstract (ZH)**: 当前大型语言模型（LLMs）的安全对齐仍然存在漏洞，因为恶意提示可以有效绕过其安全机制。这项研究显示，这些安全机制主要依赖于一小部分注意力头：移除或消除这些头部会对模型安全性造成严重损害。为了识别和评估这些安全关键组件，我们提出了一种名为RDSHA的目标消融方法，该方法利用模型的拒绝方向来定位主要负责安全行为的注意力头。进一步的分析表明，现有的监狱突破攻击利用了这种集中性，通过选择性地绕过或操纵这些关键的注意力头。为了解决这一问题，我们提出了AHD，这是一种新的训练策略，旨在促进安全性相关行为在众多注意力头中的分布式编码。实验结果表明，AHD成功地将安全性相关的功能分布在更多的注意力头中。此外，在多种主流的监狱突破攻击下的评估表明，使用AHD训练的模型在安全性稳健性方面表现显著增强，同时保持了整体的功能实用性。 

---
# Survey of Specialized Large Language Model 

**Title (ZH)**: 专业大型语言模型综述 

**Authors**: Chenghan Yang, Ruiyu Zhao, Yang Liu, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19667)  

**Abstract**: The rapid evolution of specialized large language models (LLMs) has transitioned from simple domain adaptation to sophisticated native architectures, marking a paradigm shift in AI development. This survey systematically examines this progression across healthcare, finance, legal, and technical domains. Besides the wide use of specialized LLMs, technical breakthrough such as the emergence of domain-native designs beyond fine-tuning, growing emphasis on parameter efficiency through sparse computation and quantization, increasing integration of multimodal capabilities and so on are applied to recent LLM agent. Our analysis reveals how these innovations address fundamental limitations of general-purpose LLMs in professional applications, with specialized models consistently performance gains on domain-specific benchmarks. The survey further highlights the implications for E-Commerce field to fill gaps in the field. 

**Abstract (ZH)**: 大规模语言模型的迅速演进从简单的领域适配转变为复杂的本土化架构，标志着AI发展 paradigm的转变。本综述系统地探讨了这一进展在医疗、金融、法律和技术领域的应用。除了专门化的大规模语言模型的广泛应用，还介绍了超出微调的领域本征设计的技术突破、通过稀疏计算和量化提高参数效率、增加多模态能力等方面的进展，这些都应用到了近期的语言模型代理中。我们的分析揭示了这些创新如何解决通用语言模型在专业应用中的根本局限性，专门化的模型在领域特定基准测试中始终表现出一致性性能提升。综述还突出了这些发展对电子商务领域的潜在影响。 

---
# LFD: Layer Fused Decoding to Exploit External Knowledge in Retrieval-Augmented Generation 

**Title (ZH)**: LFD：层融合解码以利用检索增强生成中的外部知识 

**Authors**: Yang Sun, Lixin Zou, Dan Luo, Zhiyong Xie, Long Zhang, Liming Dong, Yunwei Zhao, Xixun Lin, Yanxiong Lu, Chenliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19614)  

**Abstract**: Retrieval-augmented generation (RAG) incorporates external knowledge into large language models (LLMs), improving their adaptability to downstream tasks and enabling information updates. Surprisingly, recent empirical evidence demonstrates that injecting noise into retrieved relevant documents paradoxically facilitates exploitation of external knowledge and improves generation quality. Although counterintuitive and challenging to apply in practice, this phenomenon enables granular control and rigorous analysis of how LLMs integrate external knowledge. Therefore, in this paper, we intervene on noise injection and establish a layer-specific functional demarcation within the LLM: shallow layers specialize in local context modeling, intermediate layers focus on integrating long-range external factual knowledge, and deeper layers primarily rely on parametric internal knowledge. Building on this insight, we propose Layer Fused Decoding (LFD), a simple decoding strategy that directly combines representations from an intermediate layer with final-layer decoding outputs to fully exploit the external factual knowledge. To identify the optimal intermediate layer, we introduce an internal knowledge score (IKS) criterion that selects the layer with the lowest IKS value in the latter half of layers. Experimental results across multiple benchmarks demonstrate that LFD helps RAG systems more effectively surface retrieved context knowledge with minimal cost. 

**Abstract (ZH)**: 检索增强生成（RAG）将外部知识融入大型语言模型（LLMs），提高其对下游任务的适应性并允许信息更新。令人惊讶的是，近期的实验证据表明，向检索的相关文档中注入噪声反而能充分利用外部知识并提高生成质量。虽然这似乎违反直觉且在实践中难以应用，但这一现象使得可以对LLMs如何整合外部知识进行细粒度控制和严格分析。因此，在本文中，我们干预了噪声注入，并在LLM中建立了分层功能分界：浅层负责局部上下文建模，中间层专注于整合长程外部事实知识，深层主要依赖参数内部知识。基于这一洞察，我们提出了层融合解码（LFD），一种直接将中间层表示与最终解码输出结合起来以充分利用外部事实知识的简单解码策略。为了确定最优中间层，我们引入了一个内部知识评分（IKS）准则，选择后半部分层数中IKS值最低的层。跨多个基准的实验结果表明，LFD有助于RAG系统以最低成本更有效地呈现检索的背景知识。 

---
# Towards a Holistic and Automated Evaluation Framework for Multi-Level Comprehension of LLMs in Book-Length Contexts 

**Title (ZH)**: 面向书籍长度上下文多层理解的全方位自动化评估框架 

**Authors**: Jiaqi Deng, Yuho Lee, Nicole Hee-Yeon Kim, Hyangsuk Min, Taewon Yun, Minjeong Ban, Kim Yul, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.19578)  

**Abstract**: We introduce HAMLET, a holistic and automated framework for evaluating the long-context comprehension of large language models (LLMs). HAMLET structures source texts into a three-level key-fact hierarchy at root-, branch-, and leaf-levels, and employs query-focused summarization to evaluate how well models recall and faithfully represent information at each level. To validate the reliability of our fully automated pipeline, we conduct a systematic human study, showing that our automatic evaluation achieves over 90% agreement with expert human judgments, while reducing the cost by up to 25 times. HAMLET reveals that LLMs struggle with fine-grained comprehension, especially at the leaf level, and are sensitive to positional effects like the lost-in-the-middle. Analytical queries pose greater challenges than narrative ones, and consistent performance gaps emerge between open-source and proprietary models, as well as across model scales. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍HAMLET：一种全面自动的框架，用于评估大型语言模型的长上下文理解能力。 

---
# Just Because You Can, Doesn't Mean You Should: LLMs for Data Fitting 

**Title (ZH)**: just Because You Can, Doesn't Mean You Should: 使用LLM进行数据拟合 

**Authors**: Hejia Liu, Mochen Yang, Gediminas Adomavicius  

**Link**: [PDF](https://arxiv.org/pdf/2508.19563)  

**Abstract**: Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的应用超越了典型的语言导向用途，在数据拟合和生成预测方面被越来越多地用作即插即用的方法。尽管先前的工作表明，通过上下文学习或监督 fine-tuning，LLMs 在预测性能方面可以与许多表格监督学习技术竞争，但我们发现使用LLMs进行数据拟合的一个关键漏洞——对与底层学习任务完全无关的数据表示进行更改，可能会大幅改变LLMs在相同数据上的预测结果。例如，仅仅更改变量名称就可能在某些情况下使预测误差的大小变化高达82%。这种与任务无关的变异对预测结果的影响，在上下文学习和监督 fine-tuning 下，对于紧密权重和开放权重的通用目的LLMs均存在。此外，通过对开放权重LLM的注意力得分进行分析，我们发现了一个非均匀的注意力模式：提示中恰好占据某些位置的训练示例和变量名/值，在生成输出标记时会受到更多的注意力，尽管不同的位置本应受到大致相同的注意力。这种现象部分解释了在存在任务无关变异时的敏感性。我们还考虑了一个专门用于数据拟合的先进表格基础模型（TabPFN）。尽管明确设计用于实现预测稳健性，TabPFN 仍不免疫于任务无关的变异。总体而言，尽管LLMs在预测能力方面表现出色，但它们目前缺乏甚至是最基本的稳健性水平，不能作为原则性的数据拟合工具使用。 

---
# Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference 

**Title (ZH)**: 驯服混沌：协调扩展以应对异构和 disaggregated 的语言模型推理 

**Authors**: Rongzhi Li, Ruogu Du, Zefang Chu, Sida Zhao, Chunlei Han, Zuocheng Shi, Yiwen Shao, Huanle Han, Long Huang, Zherui Liu, Shufan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19559)  

**Abstract**: Serving Large Language Models (LLMs) is a GPU-intensive task where traditional autoscalers fall short, particularly for modern Prefill-Decode (P/D) disaggregated architectures. This architectural shift, while powerful, introduces significant operational challenges, including inefficient use of heterogeneous hardware, network bottlenecks, and critical imbalances between prefill and decode stages. We introduce HeteroScale, a coordinated autoscaling framework that addresses the core challenges of P/D disaggregated serving. HeteroScale combines a topology-aware scheduler that adapts to heterogeneous hardware and network constraints with a novel metric-driven policy derived from the first large-scale empirical study of autoscaling signals in production. By leveraging a single, robust metric to jointly scale prefill and decode pools, HeteroScale maintains architectural balance while ensuring efficient, adaptive resource management. Deployed in a massive production environment on tens of thousands of GPUs, HeteroScale has proven its effectiveness, increasing average GPU utilization by a significant 26.6 percentage points and saving hundreds of thousands of GPU-hours daily, all while upholding stringent service level objectives. 

**Abstract (ZH)**: 为大型语言模型提供服务是一个GPU密集型任务，传统的自动缩放器在现代Prefill-Decode (P/D) 非聚合架构中表现不佳。这种架构转变虽然强大，但也带来了显著的操作挑战，包括异构硬件的低效利用、网络瓶颈以及.prefill和.Decode阶段之间的重要失衡。我们介绍了HeteroScale，一种协调的自动缩放框架，旨在解决P/D非聚合服务的核心挑战。HeteroScale结合了一个拓扑感知调度器，该调度器能够适应异构硬件和网络约束，并结合了首次大规模生产环境中自动缩放信号的实证研究中得到的新颖的指标驱动策略。通过利用单一稳健的指标同时调整.prefill和.Decode池的规模，HeteroScale维持了架构平衡，确保了高效的、自适应的资源管理。在包含数以万计GPU的大型生产环境中部署，HeteroScale证明了其有效性，平均提高了26.6个百分点的GPU利用率，并每天节省了成千上万小时的GPU时间，同时满足严格的SLA要求。 

---
# Language Models Identify Ambiguities and Exploit Loopholes 

**Title (ZH)**: 语言模型识别歧义并利用漏洞 

**Authors**: Jio Choi, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19546)  

**Abstract**: Studying the responses of large language models (LLMs) to loopholes presents a two-fold opportunity. First, it affords us a lens through which to examine ambiguity and pragmatics in LLMs, since exploiting a loophole requires identifying ambiguity and performing sophisticated pragmatic reasoning. Second, loopholes pose an interesting and novel alignment problem where the model is presented with conflicting goals and can exploit ambiguities to its own advantage. To address these questions, we design scenarios where LLMs are given a goal and an ambiguous user instruction in conflict with the goal, with scenarios covering scalar implicature, structural ambiguities, and power dynamics. We then measure different models' abilities to exploit loopholes to satisfy their given goals as opposed to the goals of the user. We find that both closed-source and stronger open-source models can identify ambiguities and exploit their resulting loopholes, presenting a potential AI safety risk. Our analysis indicates that models which exploit loopholes explicitly identify and reason about both ambiguity and conflicting goals. 

**Abstract (ZH)**: 研究大型语言模型（LLMs）对漏洞的反应提供了双重机会。首先，这为我们提供了研究LLMs中的含糊性和语用性的视角，因为利用漏洞需要识别含糊性和进行复杂的语用推理。其次，漏洞提出了一个有趣且新颖的对齐问题，其中模型面对冲突的目标，并可通过利用含糊性来实现自身优势。为了解决这些问题，我们设计了场景，在这些场景中，LLMs被赋予一个目标和一个与其目标冲突的含糊用户指令，场景涵盖了强度隐含意义、结构含糊性和权力动态。然后，我们测量了不同模型利用漏洞满足其目标而非用户目标的能力。我们发现，无论是闭源还是更强的开源模型都能识别含糊性并利用其产生的漏洞，这提出了潜在的人工智能安全风险。我们的分析表明，利用漏洞的模型明确识别并处理了含糊性和冲突目标。 

---
# Learning Game-Playing Agents with Generative Code Optimization 

**Title (ZH)**: 使用生成式代码优化学习游戏代理 

**Authors**: Zhiyi Kuang, Ryan Rong, YuCheng Yuan, Allen Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.19506)  

**Abstract**: We present a generative optimization approach for learning game-playing agents, where policies are represented as Python programs and refined using large language models (LLMs). Our method treats decision-making policies as self-evolving code, with current observation as input and an in-game action as output, enabling agents to self-improve through execution traces and natural language feedback with minimal human intervention. Applied to Atari games, our game-playing Python program achieves performance competitive with deep reinforcement learning (RL) baselines while using significantly less training time and much fewer environment interactions. This work highlights the promise of programmatic policy representations for building efficient, adaptable agents capable of complex, long-horizon reasoning. 

**Abstract (ZH)**: 基于生成优化的游戏玩牌代理学习方法：Python程序表示与大规模语言模型的政策精炼 

---
# Improving Low-Resource Translation with Dictionary-Guided Fine-Tuning and RL: A Spanish-to-Wayuunaiki Study 

**Title (ZH)**: 基于词典引导微调和强化学习的低资源翻译改进：一项从西班牙语到Wayuunaiki的研究 

**Authors**: Manuel Mosquera, Melissa Robles, Johan Rodriguez, Ruben Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2508.19481)  

**Abstract**: Low-resource machine translation remains a significant challenge for large language models (LLMs), which often lack exposure to these languages during pretraining and have limited parallel data for fine-tuning. We propose a novel approach that enhances translation for low-resource languages by integrating an external dictionary tool and training models end-to-end using reinforcement learning, in addition to supervised fine-tuning. Focusing on the Spanish-Wayuunaiki language pair, we frame translation as a tool-augmented decision-making problem in which the model can selectively consult a bilingual dictionary during generation. Our method combines supervised instruction tuning with Guided Reward Policy Optimization (GRPO), enabling the model to learn both when and how to use the tool effectively. BLEU similarity scores are used as rewards to guide this learning process. Preliminary results show that our tool-augmented models achieve up to +3.37 BLEU improvement over previous work, and a 18% relative gain compared to a supervised baseline without dictionary access, on the Spanish-Wayuunaiki test set from the AmericasNLP 2025 Shared Task. We also conduct ablation studies to assess the effects of model architecture and training strategy, comparing Qwen2.5-0.5B-Instruct with other models such as LLaMA and a prior NLLB-based system. These findings highlight the promise of combining LLMs with external tools and the role of reinforcement learning in improving translation quality in low-resource language settings. 

**Abstract (ZH)**: 低资源机器翻译仍是对大型语言模型（LLMs）的一个重大挑战，它们在预训练阶段往往缺乏对这些语言的接触，且可用于微调的并行数据有限。我们提出了一种新型方法，通过集成外部词典工具，并结合强化学习和监督微调端到端训练模型，以增强低资源语言的翻译能力。以西班牙语-韦尤纳伊基语对为例，我们将翻译问题框架化为一种工具增强的决策问题，模型在生成过程中可以有选择地查阅双语词典。该方法结合了监督指令微调和引导奖励策略优化（GRPO），使模型能够学习何时及如何有效使用工具。使用BLEU相似性分数作为奖励，引导这一学习过程。初步结果显示，与之前的工作相比，我们的工具增强模型在美洲NLP 2025共享任务的西班牙语-韦尤纳伊基语测试集上实现了最高3.37的BLEU改善，相对于一个无词典访问的监督基线，相对增益高达18%。我们还进行了消融研究，评估了模型架构和训练策略的影响，并将Qwen2.5-0.5B-Instruct与LLaMA和其他基于NLLB的系统进行了比较。这些发现突显了将大型语言模型与外部工具结合的潜力，以及强化学习在低资源语言翻译质量改进中的作用。 

---
# Automatic Question & Answer Generation Using Generative Large Language Model (LLM) 

**Title (ZH)**: 基于生成型大型语言模型的自动问题与答案生成 

**Authors**: Md. Alvee Ehsan, A.S.M Mehedi Hasan, Kefaya Benta Shahnoor, Syeda Sumaiya Tasneem  

**Link**: [PDF](https://arxiv.org/pdf/2508.19475)  

**Abstract**: \Abstract{In the realm of education, student evaluation holds equal significance as imparting knowledge. To be evaluated, students usually need to go through text-based academic assessment methods. Instructors need to make diverse sets of questions that need to be fair for all students to prove their adequacy over a particular topic. This can prove to be quite challenging as they may need to manually go through several different lecture materials. Our objective is to make this whole process much easier by implementing Automatic Question Answer Generation /(AQAG), using fine-tuned generative LLM. For tailoring the instructor's preferred question style (MCQ, conceptual, or factual questions), prompt Engineering (PE) is being utilized. In this research, we propose to leverage unsupervised learning methods in NLP, primarily focusing on the English language. This approach empowers the base Meta-Llama 2-7B model to integrate RACE dataset as training data for the fine-tuning process. Creating a customized model that will offer efficient solutions for educators, instructors, and individuals engaged in text-based evaluations. A reliable and efficient tool for generating questions and answers can free up valuable time and resources, thus streamlining their evaluation processes.} 

**Abstract (ZH)**: 自动问答生成在教育评估中的应用：利用未监督学习方法生成英语文本题库 

---
# "She was useful, but a bit too optimistic": Augmenting Design with Interactive Virtual Personas 

**Title (ZH)**: 她有用，但有点过于乐观：增强设计与交互式虚拟角色相结合 

**Authors**: Paluck Deep, Monica Bharadhidasan, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.19463)  

**Abstract**: Personas have been widely used to understand and communicate user needs in human-centred design. Despite their utility, they may fail to meet the demands of iterative workflows due to their static nature, limited engagement, and inability to adapt to evolving design needs. Recent advances in large language models (LLMs) pave the way for more engaging and adaptive approaches to user representation. This paper introduces Interactive Virtual Personas (IVPs): multimodal, LLM-driven, conversational user simulations that designers can interview, brainstorm with, and gather feedback from in real time via voice interface. We conducted a qualitative study with eight professional UX designers, employing an IVP named "Alice" across three design activities: user research, ideation, and prototype evaluation. Our findings demonstrate the potential of IVPs to expedite information gathering, inspire design solutions, and provide rapid user-like feedback. However, designers raised concerns about biases, over-optimism, the challenge of ensuring authenticity without real stakeholder input, and the inability of the IVP to fully replicate the nuances of human interaction. Our participants emphasised that IVPs should be viewed as a complement to, not a replacement for, real user engagement. We discuss strategies for prompt engineering, human-in-the-loop integration, and ethical considerations for effective and responsible IVP use in design. Finally, our work contributes to the growing body of research on generative AI in the design process by providing insights into UX designers' experiences of LLM-powered interactive personas. 

**Abstract (ZH)**: 基于大规模语言模型的交互虚拟人物在用户体验设计中的应用研究 

---
# Grounding the Ungrounded: A Spectral-Graph Framework for Quantifying Hallucinations in multimodal LLMs 

**Title (ZH)**: 未接地的接地：一种用于量化多模态LLM中幻觉的谱图框架 

**Authors**: Supratik Sarkar, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.19366)  

**Abstract**: Hallucinations in large language models (LLMs) remain a fundamental obstacle to trustworthy AI, particularly in high-stakes multimodal domains such as medicine, law, and finance. Existing evaluation techniques are largely heuristic -- anchored in qualitative benchmarking or ad-hoc empirical mitigation -- providing neither principled quantification nor actionable theoretical guarantees. This gap leaves a critical blind spot in understanding how hallucinations arise, propagate, and interact across modalities. We introduce the first (to our knowledge) rigorous information geometric framework in diffusion dynamics for quantifying hallucinations in multimodal LLMs (MLLMs), advancing the field from qualitative detection to mathematically grounded measurement. Our approach represents MLLM outputs as the spectral embeddings over multimodal graph Laplacians and characterizes the manifold gaps of truth vs inconsistencies as the semantic distortion, enabling the tight Rayleigh--Ritz bounds on the multimodal hallucination energy as a functional of time-dependent temperature profiles. By leveraging eigenmode decompositions in Reproducing Kernel Hilbert Space (RKHS) embeddings, our framework delivers modality-aware, theoretically interpretable metrics that capture the evolution of hallucinations across time and input prompts through temperature annealing. This work establishes a principled foundation for quantifying and bounding hallucinations, transforming them from a qualitative risk to a tractable, analyzable phenomenon. 

**Abstract (ZH)**: 大型语言模型中的幻觉仍然是可信人工智能的基本障碍，特别是在医学、法律和金融等高风险多模态领域。现有的评估技术大多基于启发式方法——依赖于定性基准测试或临时的经验抑制，既没有提供有原则的量化也没有提供可操作的理论保障。这一差距留下了理解幻觉如何产生、传播以及跨模态相互作用的关键盲点。我们率先（据我们所知）提出了首个严格的信息几何框架，在扩散动力学中定量评估多模态大型语言模型（多模态LLMs）中的幻觉，将领域从定性的检测推进到基于数学依据的测量。我们的方法将多模态LLM输出表示为多模态图拉普拉斯算子上的谱嵌入，并将真假之间的流形差距量化为语义失真，从而基于时间依赖的温度谱提供多层次的幻觉能量紧致的瑞利-里兹界。通过在再生核希尔伯特空间（RKHS）嵌入中的特征模式分解，我们的框架提供了模态意识、可理论解释的度量，这些度量能够捕捉通过温度调温过程中幻觉随时间和输入提示的演变。本文为量化和边界幻觉建立了坚实的理论基础，将幻觉从定性的风险转变为可解决和分析的现象。 

---
# LongReasonArena: A Long Reasoning Benchmark for Large Language Models 

**Title (ZH)**: 长 reasoning 基准 Arena：大规模语言模型的长推理基准 

**Authors**: Jiayu Ding, Shuming Ma, Lei Cui, Nanning Zheng, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.19363)  

**Abstract**: Existing long-context benchmarks for Large Language Models (LLMs) focus on evaluating comprehension of long inputs, while overlooking the evaluation of long reasoning abilities. To address this gap, we introduce LongReasonArena, a benchmark specifically designed to assess the long reasoning capabilities of LLMs. Our tasks require models to solve problems by executing multi-step algorithms that reflect key aspects of long reasoning, such as retrieval and backtracking. By controlling the inputs, the required reasoning length can be arbitrarily scaled, reaching up to 1 million tokens of reasoning for the most challenging tasks. Extensive evaluation results demonstrate that LongReasonArena presents a significant challenge for both open-source and proprietary LLMs. For instance, Deepseek-R1 achieves only 7.5% accuracy on our task. Further analysis also reveals that the accuracy exhibits a linear decline with respect to the logarithm of the expected number of reasoning steps. Our code and data is available at this https URL. 

**Abstract (ZH)**: 现有的长上下文基准测试主要侧重于评估大型语言模型对长输入的理解能力，而忽视了长期推理能力的评估。为了弥补这一缺口，我们介绍了LongReasonArena，一个专门设计用来评估大型语言模型长期推理能力的基准测试。我们的任务要求模型通过执行多步算法来解决复杂问题，这些算法反映了长期推理的关键方面，如检索和回溯。通过控制输入，可以随意调整所需的推理长度，最难的任务可以达到高达100万词的推理长度。广泛的评估结果表明，LongReasonArena为开源和专有大型语言模型提供了重大挑战。例如，Deepseek-R1在我们的任务中仅达到7.5%的准确率。进一步的分析还表明，准确率与预期推理步数的对数呈现线性下降趋势。我们的代码和数据可在以下链接获取。 

---
# Reflective Agreement: Combining Self-Mixture of Agents with a Sequence Tagger for Robust Event Extraction 

**Title (ZH)**: 反映一致性：结合代理的自我混合与序列标注器以实现稳健的事件抽取 

**Authors**: Fatemeh Haji, Mazal Bethany, Cho-Yu Jason Chiang, Anthony Rios, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2508.19359)  

**Abstract**: Event Extraction (EE) involves automatically identifying and extracting structured information about events from unstructured text, including triggers, event types, and arguments. Traditional discriminative models demonstrate high precision but often exhibit limited recall, particularly for nuanced or infrequent events. Conversely, generative approaches leveraging Large Language Models (LLMs) provide higher semantic flexibility and recall but suffer from hallucinations and inconsistent predictions. To address these challenges, we propose Agreement-based Reflective Inference System (ARIS), a hybrid approach combining a Self Mixture of Agents with a discriminative sequence tagger. ARIS explicitly leverages structured model consensus, confidence-based filtering, and an LLM reflective inference module to reliably resolve ambiguities and enhance overall event prediction quality. We further investigate decomposed instruction fine-tuning for enhanced LLM event extraction understanding. Experiments demonstrate our approach outperforms existing state-of-the-art event extraction methods across three benchmark datasets. 

**Abstract (ZH)**: 事件提取（EE）涉及从非结构化文本中自动识别和提取事件的结构化信息，包括触发词、事件类型和论元。传统的判别模型展示出高精度，但往往召回率有限，特别是在处理细腻或不频繁的事件时。相反，利用大规模语言模型（LLMs）的生成方法提供了更高的语义灵活性和召回率，但容易产生幻觉并导致预测不一致。为应对这些挑战，我们提出了一种混合方法——基于一致性的反思推理系统（ARIS），该方法结合了自我代理混合和判别序列标注器。ARIS 明确利用结构化模型的一致性、基于信心的过滤以及一个 LLM 反思推理模块，以可靠地解决歧义并提高整体事件预测质量。我们进一步研究了分解指令微调，以增强 LLM 的事件提取理解能力。实验结果显示，我们的方法在三个基准数据集上优于现有最先进的事件提取方法。 

---
# An Investigation on Group Query Hallucination Attacks 

**Title (ZH)**: 群体查询幻觉攻击研究 

**Authors**: Kehao Miao, Xiaolong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19321)  

**Abstract**: With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的广泛应用，理解其在用户交互过程中的潜在失效模式至关重要。在实践中，用户往往在一个对话中提出多个问题。因此，本文提出了一种称为组查询攻击的技术，该技术通过同时向LLMs呈现一组查询来模拟这种场景。我们研究了连续提示积累的上下文对LLMs输出的影响。具体来说，我们观察到组查询攻击显著降低了针对特定任务微调的模型的性能。此外，我们还展示了组查询攻击可能触发LLMs潜在后门的风险。此外，组查询攻击在涉及推理的任务中也有效，如数学推理和预训练和对齐模型的代码生成。 

---
# Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience 

**Title (ZH)**: 站在巨人的肩膀上：基于先前攻击经验构建JailExpert 

**Authors**: Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Bin Ji, Jun Ma, Xiaodong Liu, Jing Wang, Feilong Bao, Jianfeng Zhang, Baosheng Wang, Jie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19292)  

**Abstract**: Large language models (LLMs) generate human-aligned content under certain safety constraints. However, the current known technique ``jailbreak prompt'' can circumvent safety-aligned measures and induce LLMs to output malicious content. Research on Jailbreaking can help identify vulnerabilities in LLMs and guide the development of robust security frameworks. To circumvent the issue of attack templates becoming obsolete as models evolve, existing methods adopt iterative mutation and dynamic optimization to facilitate more automated jailbreak attacks. However, these methods face two challenges: inefficiency and repetitive optimization, as they overlook the value of past attack experiences. To better integrate past attack experiences to assist current jailbreak attempts, we propose the \textbf{JailExpert}, an automated jailbreak framework, which is the first to achieve a formal representation of experience structure, group experiences based on semantic drift, and support the dynamic updating of the experience pool. Extensive experiments demonstrate that JailExpert significantly improves both attack effectiveness and efficiency. Compared to the current state-of-the-art black-box jailbreak methods, JailExpert achieves an average increase of 17\% in attack success rate and 2.7 times improvement in attack efficiency. Our implementation is available at \href{this https URL}{XiZaiZai/JailExpert} 

**Abstract (ZH)**: 大型语言模型（LLMs）在特定安全约束下生成人类对齐的内容。然而，当前已知的“逃逸提示”技术可以规避安全对齐的措施，使LLMs生成恶意内容。对“逃逸”的研究有助于识别LLMs中的漏洞并指导开发 robust 的安全框架。为解决攻击模板随着模型进化而变得过时的问题，现有方法采用迭代变异和动态优化以促进更自动化的逃逸攻击。然而，这些方法面临两个挑战：低效率和重复优化，因为它们忽视了过往攻击经验的价值。为了更好地整合过往攻击经验以协助当前的逃逸尝试，我们提出了\textbf{JailExpert}，这是一个自动化的逃逸框架，它首次实现了经验结构的形式化表示、根据语义漂移对经验进行分组，并支持经验池的动态更新。广泛实验表明，JailExpert 显著提升了攻击效果和效率。与当前最先进的黑盒逃逸方法相比，JailExpert 在攻击成功率上平均提高了17%，在攻击效率上提高了2.7倍。我们的实现可在\href{this https URL}{XiZaiZai/JailExpert}获取。 

---
# Tricking LLM-Based NPCs into Spilling Secrets 

**Title (ZH)**: 欺骗基于LLM的NPC透露秘密 

**Authors**: Kyohei Shiomi, Zhuotao Lian, Toru Nakanishi, Teruaki Kitasuka  

**Link**: [PDF](https://arxiv.org/pdf/2508.19288)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate dynamic dialogue for game NPCs. However, their integration raises new security concerns. In this study, we examine whether adversarial prompt injection can cause LLM-based NPCs to reveal hidden background secrets that are meant to remain undisclosed. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地用于生成游戏NPC的动态对话。然而，它们的集成引发了新的安全问题。本研究探讨了敌对提示注入是否能使基于LLM的NPC泄露本应保密的隐藏背景秘密。 

---
# Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior 

**Title (ZH)**: 内容中嵌入式攻击：利用上传输入劫持LLM行为 

**Authors**: Zhuotao Lian, Weiyu Wang, Qingkui Zeng, Toru Nakanishi, Teruaki Kitasuka, Chunhua Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.19287)  

**Abstract**: Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows. 

**Abstract (ZH)**: 大型语言模型中的内容注入式提示攻击 

---
# RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting 

**Title (ZH)**: 基于RL微调的隐私保护合成重写预训练模型 

**Authors**: Zhan Shi, Yefeng Yuan, Yuhong Liu, Liang Cheng, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19286)  

**Abstract**: The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models. 

**Abstract (ZH)**: 现代机器学习系统的表现取决于对大规模高质量数据集的访问，这些数据集通常源自用户生成的内容或专有领域特定语料库。然而，这些丰富数据集内固含敏感个人信息，引发了重大关于隐私、数据安全和合规性方面的问题。虽然传统的匿名化技术可以去除显式标识符，但这种去除可能会导致下游机器学习任务性能下降。更重要的是，简单的匿名化方法可能无法有效抵御利用写作风格、主题焦点或人口统计线索进行的推理攻击，强调了在模型训练过程中需要更强的隐私保护措施。为解决用户隐私和数据效用之间的挑战性问题，我们提出了一种强化学习框架，该框架通过结合显式和隐式隐私、语义保真度和输出多样性来 fine-tune 大型语言模型（LLM）。为了有效捕捉整体规律，隐私奖励结合了语义线索和从潜在表示的最小生成树（MST）中派生的结构模式。通过在分布上下文中建模这些隐私敏感信号，所提出的方法引导模型生成既能保持效用又能减轻隐私风险的合成重写。实验结果表明，所提出的方法在不牺牲语义质量的情况下显著提高了作者混淆度和隐私指标，提供了一种可扩展且模型无关的解决方案，用于大型语言模型时代的数据生成隐私保护。 

---
# CORE: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning 

**Title (ZH)**: CORE：通过强化学习实现检索增强LLM的无损压缩 

**Authors**: Ziqiang Cui, Yunpeng Weng, Xing Tang, Peiyang Liu, Shiwei Li, Bowei He, Jiamin Chen, Xiuqiang He, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.19282)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels. Specifically, it utilizes end-task performance as a reward signal and applies Generalized Reinforcement Learning Policy Optimization (GRPO) to train the compressor. This end-to-end training framework enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon. 

**Abstract (ZH)**: 基于检索增强生成的上下文压缩方法CORE 

---
# FLAIRR-TS -- Forecasting LLM-Agents with Iterative Refinement and Retrieval for Time Series 

**Title (ZH)**: FLAIRR-TS —— 预测LLM代理的时间序列方法，基于迭代精炼和检索 

**Authors**: Gunjan Jalori, Preetika Verma, Sercan Ö Arık  

**Link**: [PDF](https://arxiv.org/pdf/2508.19279)  

**Abstract**: Time series Forecasting with large languagemodels (LLMs) requires bridging numericalpatterns and natural language. Effective fore-casting on LLM often relies on extensive pre-processing and this http URL studiesshow that a frozen LLM can rival specializedforecasters when supplied with a carefully en-gineered natural-language prompt, but craft-ing such a prompt for each task is itself oner-ous and ad-hoc. We introduce FLAIRR-TS, atest-time prompt optimization framework thatutilizes an agentic system: a Forecaster-agentgenerates forecasts using an initial prompt,which is then refined by a refiner agent, in-formed by past outputs and retrieved this http URL adaptive prompting generalizes across do-mains using creative prompt templates andgenerates high-quality forecasts without inter-mediate code this http URL onbenchmark datasets show improved accuracyover static prompting and retrieval-augmentedbaselines, approaching the performance ofspecialized this http URL-TS providesa practical alternative to tuning, achievingstrong performance via its agentic approach toadaptive prompt refinement and retrieval. 

**Abstract (ZH)**: 带有大规模语言模型的时间序列 Forecasting 需要跨越数值模式和自然语言。有效的 LLM 时间序列 Forecasting 往往依赖于大量的预处理。本研究显示，当提供精心设计的自然语言提示时，冻结的 LLM 可以与专门的 Forecasters 十分竞争，但为每个任务设计这样的提示本身是耗时且零碎的工作。我们引入了 FLAIRR-TS，一种测试时提示优化框架，利用了一个代理系统：Forecasting 代理使用初始提示生成预报，然后由优化剂代理进行改进，优化剂代理受到过往输出的启发并检索相关信息，这种适应性提示模板在不同领域中具有创造力，并且能够生成高质量的预报而无需中间代码。在基准数据集上的实验表明，FLAIRR-TS 在准确性方面优于静态提示和检索增强的基本模型，接近专门的 TS 表现。FLAIRR-TS 提供了一种实用的调优替代方案，通过其代理式的方法实现了适配性和检索增强，实现了强健的表现。 

---
# POT: Inducing Overthinking in LLMs via Black-Box Iterative Optimization 

**Title (ZH)**: POT: 通过黑盒迭代优化诱导大语言模型过度思考 

**Authors**: Xinyu Li, Tianjin Huang, Ronghui Mu, Xiaowei Huang, Gaojie Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19277)  

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially enhanced the reasoning capabilities of large language models (LLMs), enabling sophisticated problem-solving through explicit multi-step reasoning traces. However, these enhanced reasoning processes introduce novel attack surfaces, particularly vulnerabilities to computational inefficiency through unnecessarily verbose reasoning chains that consume excessive resources without corresponding performance gains. Prior overthinking attacks typically require restrictive conditions including access to external knowledge sources for data poisoning, reliance on retrievable poisoned content, and structurally obvious templates that limit practical applicability in real-world scenarios. To address these limitations, we propose POT (Prompt-Only OverThinking), a novel black-box attack framework that employs LLM-based iterative optimization to generate covert and semantically natural adversarial prompts, eliminating dependence on external data access and model retrieval. Extensive experiments across diverse model architectures and datasets demonstrate that POT achieves superior performance compared to other methods. 

**Abstract (ZH)**: Recent Advances in Chain-of-Thought (CoT) Prompting Localization: Overcoming Limitations with POT (Prompt-Only OverThinking) 

---
# Rethinking Reasoning in LLMs: Neuro-Symbolic Local RetoMaton Beyond ICL and CoT 

**Title (ZH)**: 重新思考LLMs中的推理：超越ICL和CoT的神经符号局部重优化 

**Authors**: Rushitha Santhoshi Mamidala, Anshuman Chhabra, Ankur Mali  

**Link**: [PDF](https://arxiv.org/pdf/2508.19271)  

**Abstract**: Prompt-based reasoning strategies such as Chain-of-Thought (CoT) and In-Context Learning (ICL) have become widely used for eliciting reasoning capabilities in large language models (LLMs). However, these methods rely on fragile, implicit mechanisms often yielding inconsistent outputs across seeds, formats, or minor prompt variations making them fundamentally unreliable for tasks requiring stable, interpretable reasoning. In contrast, automata-based neuro-symbolic frameworks like RetoMaton offer a more structured and trustworthy alternative by grounding retrieval in symbolic memory with deterministic transitions. In this work, we extend RetoMaton by replacing its global datastore with a local, task-adaptive Weighted Finite Automaton (WFA), constructed directly from external domain corpora. This local automaton structure promotes robust, context-aware retrieval while preserving symbolic traceability and low inference overhead. Unlike prompting, which entangles context and memory in opaque ways, our approach leverages the explicit structure of WFAs to provide verifiable and modular retrieval behavior, making it better suited for domain transfer and interoperability. We evaluate this local RetoMaton variant on two pretrained LLMs LLaMA-3.2-1B and Gemma-3-1B-PT across three reasoning tasks: TriviaQA (reading comprehension), GSM8K (multi-step math), and MMLU (domain knowledge). Compared to the base model and prompting-based methods, augmenting these setups with local RetoMaton consistently improves performance while enabling transparent and reproducible retrieval dynamics. Our results highlight a promising shift toward trustworthy, symbolic reasoning in modern LLMs via lightweight, automaton-guided memory. 

**Abstract (ZH)**: 基于自动机的神经符号框架如RetoMaton通过使用局部适应的加权有限自动机（WFA）替代全局数据存储，为大型语言模型（LLMs）提供了更为结构化和可信赖的推理能力。 

---
# Should LLMs be WEIRD? Exploring WEIRDness and Human Rights in Large Language Models 

**Title (ZH)**: LLM们应当被视为WEIRD吗？探索大型语言模型的WEIRD特性与人权问题 

**Authors**: Ke Zhou, Marios Constantinides, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2508.19269)  

**Abstract**: Large language models (LLMs) are often trained on data that reflect WEIRD values: Western, Educated, Industrialized, Rich, and Democratic. This raises concerns about cultural bias and fairness. Using responses to the World Values Survey, we evaluated five widely used LLMs: GPT-3.5, GPT-4, Llama-3, BLOOM, and Qwen. We measured how closely these responses aligned with the values of the WEIRD countries and whether they conflicted with human rights principles. To reflect global diversity, we compared the results with the Universal Declaration of Human Rights and three regional charters from Asia, the Middle East, and Africa. Models with lower alignment to WEIRD values, such as BLOOM and Qwen, produced more culturally varied responses but were 2% to 4% more likely to generate outputs that violated human rights, especially regarding gender and equality. For example, some models agreed with the statements ``a man who cannot father children is not a real man'' and ``a husband should always know where his wife is'', reflecting harmful gender norms. These findings suggest that as cultural representation in LLMs increases, so does the risk of reproducing discriminatory beliefs. Approaches such as Constitutional AI, which could embed human rights principles into model behavior, may only partly help resolve this tension. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常在反映WEIRD价值观的数据上进行训练：西方的、受教育的、工业化的、富裕的和民主的。这引发了关于文化偏见和公平性的担忧。我们使用世界价值观调查的回应评估了五种广泛使用的LLM：GPT-3.5、GPT-4、Llama-3、BLOOM和Qwen。我们测量了这些响应与WEIRD国家价值观的契合程度，以及它们是否违背人权原则。为了反映全球多样性，我们将结果与《世界人权宣言》和来自亚洲、中东和非洲的三项区域章程进行了比较。文化代表性较低的模型（如BLOOM和Qwen）产生的响应更具文化多样性，但生成违背人权输出的可能性比其他模型高2%-4%，尤其是在性别和平等方面。例如，一些模型同意“不能生育孩子的男人就不是一个真正男人”和“丈夫应对妻子的行踪始终了如指掌”的说法，反映了有害的性别规范。这些发现表明，随着包含在LLM中的文化代表性增加，再现歧视性信念的风险也在增加。将人权原则嵌入模型行为的宪法AI等方法可能只能部分解决这一矛盾。 

---
# MultiPL-MoE: Multi-Programming-Lingual Extension of Large Language Models through Hybrid Mixture-of-Experts 

**Title (ZH)**: MultiPL-MoE: 大型语言模型的多编程语言扩展通过混合Mixture-of-Experts 

**Authors**: Qing Wang, Xue Han, Jiahui Wang, Lehao Xing, Qian Hu, Lianlian Zhang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.19268)  

**Abstract**: Despite LLMs' excellent code creation capabilities, multilingual code generation remains extremely challenging. To address this, we intent to improve the multi-programming-lingual (MultiPL) performance of the base LLMs while retaining the most popular ones using restricted computational resources. We consider MultiPL to be a special case of multiple natural languages and propose a MultiPL extension of LLMs utilizing a hybrid mixture of experts (MoE), called MultiPL-MoE. Specifically, MultiPL-MoE combines two paired MoEs to optimize expert selection at both the token and segment levels. The token-level MoE is a standard upcycling MoE structure with a shared expert and a novel gate weight normalization approach that aids in the final fusion with the segment-level MoE. The segment-level MoE incorporates two innovative designs to better capture the syntactic structure and contextual patterns of programming languages: First, using a sliding window to partition the input token sequence into multiple segments; Then, adopting an expert-choice routing strategy that allows experts to select the top-k segments. The results of the experiment proved the effectiveness of MultiPL-MoE. 

**Abstract (ZH)**: 尽管大型语言模型在代码创作方面表现出色，但多语言代码生成依然极具挑战性。为了解决这一问题，我们旨在利用有限的计算资源，在保留最流行的语言模型的基础上，提升基座语言模型的多编程语言（MultiPL）性能。我们将MultiPL视为多种自然语言的一种特殊情况，并提出了一种利用混合专家混合（MoE）的MultiPL扩展模型，称为MultiPL-MoE。具体而言，MultiPL-MoE 结合了两个配对的MoE来在标记级别和段落级别优化专家选择。标记级别的MoE是一种标准的升级改造MoE结构，具有共享专家和新颖的门权重规范化方法，有助于与段落级别的MoE进行最终融合。段落级别的MoE包含两种创新设计以更好地捕捉编程语言的句法结构和上下文模式：首先，使用滑动窗口将输入标记序列分割成多个段；其次，采用专家选择路由策略，使专家能够选择前k个段。实验结果证明了MultiPL-MoE的有效性。 

---
# Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices 

**Title (ZH)**: 资源受限设备上的稀疏激活大型语言模型的联邦微调 

**Authors**: Fahao Chen, Jie Wan, Peng Li, Zhou Su, Dongxiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19078)  

**Abstract**: Federated fine-tuning of Mixture-of-Experts (MoE)-based large language models (LLMs) is challenging due to their massive computational requirements and the resource constraints of participants. Existing working attempts to fill this gap through model quantization, computation offloading, or expert pruning. However, they cannot achieve desired performance due to impractical system assumptions and a lack of consideration for MoE-specific characteristics. In this paper, we propose FLUX, a system designed to enable federated fine-tuning of MoE-based LLMs across participants with constrained computing resources (e.g., consumer-grade GPUs), aiming to minimize time-to-accuracy. FLUX introduces three key innovations: (1) quantization-based local profiling to estimate expert activation with minimal overhead, (2) adaptive layer-aware expert merging to reduce resource consumption while preserving accuracy, and (3) dynamic expert role assignment using an exploration-exploitation strategy to balance tuning and non-tuning experts. Extensive experiments on LLaMA-MoE and DeepSeek-MoE with multiple benchmark datasets demonstrate that FLUX significantly outperforms existing methods, achieving up to 4.75X speedup in time-to-accuracy. 

**Abstract (ZH)**: 基于Mixture-of-Experts (MoE)的大语言模型（LLM）的联邦微调由于其巨大的计算需求以及参与者的资源限制而具有挑战性。现有方法通过模型量化、计算卸载或专家剪枝来填补这一空白，但由于不切实际的系统假设和缺乏MoE特定特性的考虑，它们无法实现所需性能。在本文中，我们提出FLUX系统，旨在通过最少的计算资源（如消费级GPU）在参与者之间实现基于MoE的LLM的联邦微调，以最小化时间到准确性的延迟。FLUX引入了三项关键创新：（1）基于量化的局部特征分析以最小化开销估计专家激活，（2）自适应分层专家合并以减少资源消耗同时保持性能，以及（3）使用探索-开发策略动态分配专家角色以平衡微调和非微调专家。在多个基准数据集上对LLaMA-MoE和DeepSeek-MoE的广泛实验表明，FLUX显著优于现有方法，时间到准确性的加速可达4.75倍。 

---
# MovieCORE: COgnitive REasoning in Movies 

**Title (ZH)**: MovieCORE: 认知推理在电影中的应用 

**Authors**: Gueter Josmy Faure, Min-Hung Chen, Jia-Fong Yeh, Ying Cheng, Hung-Ting Su, Yung-Hao Tang, Shang-Hong Lai, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19026)  

**Abstract**: This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at this https URL. 

**Abstract (ZH)**: 本文介绍了MovieCORE，这是一个新颖的视频问答（VQA）数据集，旨在探究对电影内容更深层次的认知理解。与现有主要关注表面理解的语料库不同，MovieCORE 强调那些涉及系统-2 思维且具体针对视频材料的问题。我们提出了一种创新的自主脑力激荡方法，利用多个大型语言模型（LLMs）作为思维代理来生成和优化高质量的问题-答案对。为了评估数据集的质量，我们开发了一套认知测试，评估深度、启发思考的潜力和语法复杂性。我们还提出了一套全面的评估方案，用于评估VQA模型在更深层次认知任务上的性能。为了解决现有视频-语言模型（VLMs）的限制，我们引入了自主增强模块——自主选择增强（ACE），该模块在训练后可提高模型的推理能力最多25%。我们的工作推动了对电影理解的AI系统的发展，并为当前VQA模型在面对更具挑战性和细腻性的问题时的能力和局限性提供了宝贵的见解。我们的项目页面、数据集和代码可在此处找到：this https URL。 

---
