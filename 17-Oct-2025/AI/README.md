# Agentic Design of Compositional Machines 

**Title (ZH)**: 代理设计组成的机器 

**Authors**: Wenqian Zhang, Weiyang Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14980)  

**Abstract**: The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning. 

**Abstract (ZH)**: 复杂机器的设计既体现了人类智能，也是工程实践的基础。鉴于大型语言模型（LLMs）的 recent 进展，我们询问它们是否也能学会创造。我们通过组合式机器设计这一视角来探讨这一问题：这是一种将标准化组件组装起来以满足功能性需求（如移动或操纵）的任务，并在模拟物理环境中进行。为了支持这一调查，我们引入了基于《围城》（Besiege）机器建造游戏构建的 BesiegeField 测试平台，该平台支持基于部件的构建、物理模拟和奖励驱动的评估。使用 BesiegeField，我们对最先进的 LLMs 进行基准测试，并使用代理工作流识别成功所需的关键能力，包括空间推理、战略组装和指令遵循。由于当前的开源模型尚不理想，我们探讨了强化学习（RL）作为改进的途径，进行了冷启动数据集的整理、RL 微调实验，并指出了语言、机器设计和物理推理交叉领域中的开放挑战。 

---
# GroundedPRM: Tree-Guided and Fidelity-Aware Process Reward Modeling for Step-Level Reasoning 

**Title (ZH)**: 基于树引导和保真度意识的过程奖励建模：步骤级推理 

**Authors**: Yao Zhang, Yu Wu, Haowei Zhang, Weiguo Li, Haokun Chen, Jingpei Wu, Guohao Li, Zhen Han, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2510.14942)  

**Abstract**: Process Reward Models (PRMs) aim to improve multi-step reasoning in Large Language Models (LLMs) by supervising intermediate steps and identifying errors. However, building effective PRMs remains challenging due to the lack of scalable, high-quality annotations. Existing approaches rely on costly human labeling, LLM-based self-evaluation that is prone to hallucination, or Monte Carlo (MC) estimation, which infers step quality solely from rollout outcomes and often introduces noisy, misaligned supervision due to credit misattribution. These issues result in three core limitations: noisy rewards, low factual fidelity, and misalignment with step-level reasoning objectives. To address these challenges, we introduce GroundedPRM, a tree-guided and fidelity-aware framework for automatic process supervision. To reduce reward noise and enable fine-grained credit assignment, we construct structured reasoning paths via Monte Carlo Tree Search (MCTS). To eliminate hallucinated supervision, we validate each intermediate step using an external tool, providing execution-grounded correctness signals. To combine both step-level validation and global outcome assessment, we design a hybrid reward aggregation mechanism that fuses tool-based verification with MCTS-derived feedback. Finally, we format the reward signal into a rationale-enhanced, generative structure to promote interpretability and compatibility with instruction-tuned LLMs. GroundedPRM is trained on only 40K automatically labeled samples, amounting to just 10% of the data used by the best-performing PRM trained with auto-labeled supervision. Nevertheless, it achieves up to a 26% relative improvement in average performance on ProcessBench. When used for reward-guided greedy search, GroundedPRM outperforms even PRMs trained with human-labeled supervision, offering a scalable and verifiable path toward high-quality process-level reasoning. 

**Abstract (ZH)**: 基于指导的进程奖励模型（GroundedPRM）：一种树引导且 fidelity 意识到自动进程监督框架 

---
# Stable but Miscalibrated: A Kantian View on Overconfidence from Filters to Large Language Models 

**Title (ZH)**: 稳定但误校准：从过滤器到大规模语言模型的康德视角的过度自信探究 

**Authors**: Akira Okutomi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14925)  

**Abstract**: We reinterpret Kant's Critique of Pure Reason as a theory of feedback stability, viewing reason as a regulator that keeps inference within the bounds of possible experience. We formalize this intuition via a composite instability index (H-Risk) combining spectral margin, conditioning, temporal sensitivity, and innovation amplification. In linear-Gaussian simulations, higher H-Risk predicts overconfident errors even under formal stability, revealing a gap between nominal and epistemic stability. Extending to large language models (LLMs), we find that fragile internal dynamics correlate with miscalibration and hallucination, while critique-style prompts show mixed effects on calibration and hallucination. These results suggest a structural bridge between Kantian self-limitation and feedback control, offering a principled lens for diagnosing -- and selectively reducing -- overconfidence in reasoning systems. This is a preliminary version; supplementary experiments and broader replication will be reported in a future revision. 

**Abstract (ZH)**: 我们将康德的《纯粹理性批判》重新解读为反馈稳定性的理论，将理性视作一种调节器，确保推理在可能的经验范围内。我们通过综合不稳定性指数（H-Risk）来正式化这一直观认识，该指数结合了谱 margins、条件性、时间敏感性和创新放大等因素。在线性高斯模拟中，更高的 H-Risk 甚至在形式上稳定的情况下预测出过度自信的错误，揭示了名义稳定性和知识稳定性的差距。扩展到大规模语言模型（LLMs），我们发现脆弱的内部动态与校准不足和幻觉相关，而批判性提示在校准和幻觉方面则表现出混合效果。这些结果表明坎托尔自我限制与反馈控制之间存在结构性的桥梁，提供了一种原则性的视角来诊断——并选择性地降低——推理系统的过度自信。这是一个初步版本；补充实验和更广泛的操作将在未来的修订中报告。 

---
# TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG 

**Title (ZH)**: 三模态对比研究：基于语音、文本和EEG的抑郁检测 

**Authors**: Annisaa Fitri Nurfidausi, Eleonora Mancini, Paolo Torroni  

**Link**: [PDF](https://arxiv.org/pdf/2510.14922)  

**Abstract**: Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection. 

**Abstract (ZH)**: 抑郁症是一种常见的精神健康疾病，但其自动检测仍然具有挑战性。现有的研究主要探索了一模态和多模态方法，多模态系统通过利用互补信号表现出潜力。然而，现有研究在范围上有限，缺乏对特征的系统比较，并且在评估协议上缺乏一致性。我们通过系统地探索EEG、语音和文本跨模态的特征表示和建模策略来弥补这些差距。我们评估了手工设计特征与预训练嵌入表示的效果，评估了不同的神经编码器的有效性，比较了一模态、二模态和三模态配置，并分析了关注EEG作用的融合策略。采用一致的被试独立分割确保了稳健和可重复的基准测试。我们的结果显示，(i) EEG、语音和文本模态的结合增强了多模态检测，(ii) 预训练嵌入表示优于手工设计特征，(iii) 仔细设计的三模态模型达到了现有最佳性能。我们的工作为进一步研究多模态抑郁症检测奠定了基础。 

---
# Budget-aware Test-time Scaling via Discriminative Verification 

**Title (ZH)**: 预算感知的测试时缩放通过辨别性验证 

**Authors**: Kyle Montgomery, Sijun Tan, Yuqi Chen, Siyuan Zhuang, Tianjun Zhang, Raluca Ada Popa, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14913)  

**Abstract**: Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we shift the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, this hybrid approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 15.3\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at this https URL. 

**Abstract (ZH)**: 测试时缩放是提升大规模语言模型在复杂推理任务上性能的有力策略。尽管最先进的方法通常采用生成式验证器从候选解决方案中选择最佳方案，但这种方法会带来高昂的计算成本，限制了其实用性。在本文中，我们将重点转向一种更具预算意识的范式：辨别性验证。我们进行了详尽的实证分析，并证明虽然辨别性验证器在孤立使用时可能会表现不佳，但将其与自我一致性结合使用，可以在混合方法中形成一种强大且高效的测试时缩放机制。值得注意的是，在固定计算预算下，该混合方法在最先进的生成式验证方法上取得了显著的优势：在AIME2025上的准确率提高了多达15.3%。我们的研究结果表明，对于实际应用，使用辨别性验证器进行预算意识缩放不仅是自我一致性的“免费”升级，而且还是一种成本效益更高的替代生成式技术的选择。代码可在以下链接获取。 

---
# Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates 

**Title (ZH)**: 不靠更努力，靠更聪明：一种在测试时增强学习的代理，无需标签或模型更新即可改进 

**Authors**: Wen-Kwang Tsao, Yao-Ching Yu, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14900)  

**Abstract**: The Enterprise Intelligence Platform must integrate logs from numerous third-party vendors in order to perform various downstream tasks. However, vendor documentation is often unavailable at test time. It is either misplaced, mismatched, poorly formatted, or incomplete, which makes schema mapping challenging. We introduce a reinforcement learning agent that can self-improve without labeled examples or model weight updates. During inference, the agent: 1) Identifies ambiguous field-mapping attempts. 2) Generates targeted web-search queries to gather external evidence. 3) Applies a confidence-based reward to iteratively refine its mappings. To demonstrate this concept, we converted Microsoft Defender for Endpoint logs into a common schema. Our method increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\% over 100 iterations using GPT-4o. At the same time, it reduced the number of low-confidence mappings requiring expert review by 85\%. This new approach provides an evidence-driven, transparent method for solving future industry problems, paving the way for more robust, accountable, scalable, efficient, flexible, adaptable, and collaborative solutions. 

**Abstract (ZH)**: 企业智能平台必须整合众多第三方供应商的日志以执行各种下游任务。然而，在测试时，供应商文档往往不可用，要么丢失，要么不匹配，要么格式不佳，要么不完整，这使得模式映射变得困难。我们引入了一个无需标记样本或模型权重更新即可自我改进的强化学习代理。在推理过程中，代理：1）识别模糊的字段映射尝试。2）生成针对性的网络搜索查询以收集外部证据。3）应用基于置信度的奖励以迭代精化其映射。为了验证这一概念，我们将Microsoft Defender for Endpoint日志转换为通用模式。经过100次迭代后，我们的方法将仅使用大语言模型的映射准确性从56.4%提高到使用检索增强生成式代理（RAG）的72.73%，最终使用GPT-4o提高到93.94%。同时，它还将需要专家审核的低置信度映射数量减少了85%。这一新方法提供了基于证据的、透明的解决未来行业问题的方法，为更 robust、可问责、可扩展、高效、灵活、适应性强和协作性解决方案铺平了道路。 

---
# The Gatekeeper Knows Enough 

**Title (ZH)**: 守门人知足矣 

**Authors**: Fikresilase Wondmeneh Abebayew  

**Link**: [PDF](https://arxiv.org/pdf/2510.14881)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents, yet their practical utility is fundamentally constrained by a limited context window and state desynchronization resulting from the LLMs' stateless nature and inefficient context management. These limitations lead to unreliable output, unpredictable behavior, and inefficient resource usage, particularly when interacting with large, structured, and sensitive knowledge systems such as codebases and documents. To address these challenges, we introduce the Gatekeeper Protocol, a novel, domain-agnostic framework that governs agent-system interactions. Our protocol mandates that the agent first operate and reason on a minimalist, low-fidelity "latent state" representation of the system to strategically request high-fidelity context on demand. All interactions are mediated through a unified JSON format that serves as a declarative, state-synchronized protocol, ensuring the agent's model of the system remains verifiably grounded in the system's reality. We demonstrate the efficacy of this protocol with Sage, a reference implementation of the Gatekeeper Protocol for software development. Our results show that this approach significantly increases agent reliability, improves computational efficiency by minimizing token consumption, and enables scalable interaction with complex systems, creating a foundational methodology for building more robust, predictable, and grounded AI agents for any structured knowledge domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被部署为自主代理，但它们的实际实用价值受到有限上下文窗口和由LLMs的状态无性和无效上下文管理导致的状态脱同步的限制。这些限制会导致输出不可靠、行为不可预测和资源使用低效，尤其是在与大型、结构化和敏感的知识系统（如代码库和文档）交互时。为应对这些挑战，我们提出了门keeper协议（Gatekeeper Protocol），这是一种通用领域框架，用于管理代理与系统之间的交互。该协议要求代理首先在系统的低保真“潜在状态”表示上操作和推理，以策略性地按需请求高保真上下文。所有交互均通过统一的JSON格式进行调解，作为声明性的、状态同步的协议，确保代理对系统的模型可验证地基于系统的现实。我们使用Sage对该协议的一个参考实现进行了验证，Sage是软件开发中的门keeper协议。实验结果表明，这种方法显著提高了代理的可靠性，通过最小化令牌消耗提高了计算效率，并使与复杂系统的交互更具可扩展性，从而为任何结构化知识领域构建更加稳健、可预测和基于现实的AI代理奠定了基础方法。 

---
# LabOS: The AI-XR Co-Scientist That Sees and Works With Humans 

**Title (ZH)**: LabOS: 能看到并与人类协作的AI-XR 合作科学家 

**Authors**: Le Cong, Zaixi Zhang, Xiaotong Wang, Yin Di, Ruofan Jin, Michal Gerasimiuk, Yinkai Wang, Ravi K. Dinesh, David Smerkous, Alex Smerkous, Xuekun Wu, Shilong Liu, Peishan Li, Yi Zhu, Simran Serrao, Ning Zhao, Imran A. Mohammad, John B. Sunwoo, Joseph C. Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14861)  

**Abstract**: Modern science advances fastest when thought meets action. LabOS represents the first AI co-scientist that unites computational reasoning with physical experimentation through multimodal perception, self-evolving agents, and Entended-Reality(XR)-enabled human-AI collaboration. By connecting multi-model AI agents, smart glasses, and human-AI collaboration, LabOS allows AI to see what scientists see, understand experimental context, and assist in real-time execution. Across applications--from cancer immunotherapy target discovery to stem-cell engineering -- LabOS shows that AI can move beyond computational design to participation, turning the laboratory into an intelligent, collaborative environment where human and machine discovery evolve together. 

**Abstract (ZH)**: 现代科学在思想与行动相遇时进步最快。LabOS代表了第一个通过多模态感知、自我进化的代理和扩展现实(XR)-enable的人机协作将计算推理与物理实验结合在一起的AI合作科学家。通过连接多模型AI代理、智能眼镜和人机协作，LabOS使AI能够看到科学家所见，理解实验背景，并在实时执行中提供协助。从癌症免疫疗法靶点发现到干细胞工程等应用中，LabOS展示了AI可以从计算设计跨越到参与，将实验室变成一个智能化、协作的环境，在人类和机器发现共同进化中发挥作用。 

---
# Where to Search: Measure the Prior-Structured Search Space of LLM Agents 

**Title (ZH)**: 在哪里搜索：测量LLM代理的先验结构搜索空间 

**Authors**: Zhuo-Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.14846)  

**Abstract**: The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs. 

**Abstract (ZH)**: 基于大型语言模型的生成-过滤-精炼（迭代范式）在AI+Science中的推理、编程和程序发现方面取得了进展。然而，搜索的有效性取决于搜索的范围，即如何将领域先验知识编码到可操作结构化的假设空间中。为此，本文提出了一种紧凑的形式理论，以描述和度量由领域先验知识引导的大语言模型辅助迭代搜索。我们将代理表示为一种模糊关系操作符在输入和输出之间，以捕捉可行的转换；代理因此受到固定的安全包络的约束。为了描述多步推理/搜索，我们通过单一连续参数加权所有可达路径，并求和得到覆盖生成函数；这引发了一种可达性难度的度量，并为由安全包络诱导的图上的搜索提供了一种几何解释。此外，我们提供了最简单的可验证推断，并通过多数投票实例进行了验证。该理论提供了一种可操作的语言和工具来度量代理及其搜索空间，提出了一种由大语言模型构建的迭代搜索的系统形式描述。 

---
# Boosting Instruction Following at Scale 

**Title (ZH)**: 大规模增强指令跟随能力 

**Authors**: Ben Elder, Evelyn Duesterwald, Vinod Muthusamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.14842)  

**Abstract**: A typical approach developers follow to influence an LLM's behavior in an application is through careful manipulation of the prompt, such as by adding or modifying instructions. However, merely adding more instructions provides little assurance that they will actually be followed. We introduce Instruction Boosting as a post-generation method to increase the reliability of LLM prompt instructions. We show that Instruction Boosting improves the instruction following rate by up to 7 points for two instructions and up to 4 points for ten instructions. To demonstrate these results we introduce SCALEDIF, a benchmark with a scaled instruction volume of up to ten instructions per data sample. We also present an analysis of the commonly observed trend that performance degrades as more instructions are added. We show that an important factor contributing to this trend is the degree of tension and conflict that arises as the number of instructions is increased. We contribute a quantitative conflict scoring tool that explains the observed performance trends and provides feedback to developers on the impact that additional prompt instructions have on a model's performance. 

**Abstract (ZH)**: 一种开发人员常用的通过精心操纵提示（如添加或修改指令）来影响LLM行为的方法。然而，仅仅增加更多指令并不能保证它们会被遵循。我们介绍了一种后生成方法——指令增强，以提高LLM提示指令的有效性。我们展示了指令增强可以将指令遵循率提高7个百分点（对于两条指令）和4个百分点（对于十条指令）。为了证明这些结果，我们引入了SCALEDIF基准，其每项数据样本包含最多十条指令。我们还呈现了随着指令数量增加而观察到的性能下降趋势的分析。我们表明，这一趋势的主要因素之一是随着指令数量的增加，所产生的紧张和冲突程度加大。我们提供了一种定量的冲突评分工具，以解释观察到的性能趋势，并为开发者提供关于额外提示指令对其模型性能影响的反馈。 

---
# RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning 

**Title (ZH)**: RoboGPT-R1: 通过强化学习增强机器人规划 

**Authors**: Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14828)  

**Abstract**: Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark. 

**Abstract (ZH)**: 提升具身代理的推理能力对于机器人成功完成复杂的长期操作任务至关重要。尽管基于监督微调（SFT）的大语言模型和视觉语言模型在规划任务中取得了成功，但在复杂现实环境中的长期操作任务中，它们仍然面临挑战，主要是由于它们受限的常识和推理能力。鉴于通过监督微调将通用视觉语言模型对齐到机器人规划任务在泛化能力和物理理解方面存在不足，我们提出了一种双重微调框架RoboGPT-R1，用于具身规划。在该框架中，监督训练通过专家序列获取基础知识，随后通过RL解决模型在视觉空间理解和推理方面的不足。为了在多步推理任务中实现物理理解和动作序列一致性，我们设计了一种基于规则的奖励函数，同时考虑环境中的长期性能和动作约束。在Qwen2.5-VL-3B上训练的推理模型在EmbodiedBench基准测试中显著优于更大规模的模型GPT-4o-mini，高出21.33%，并在Qwen2.5-VL-7B上训练的其他工作中高出20.33%。 

---
# Agentic NL2SQL to Reduce Computational Costs 

**Title (ZH)**: 代理NL2SQL以减少计算成本 

**Authors**: Dominik Jehle, Lennart Purucker, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.14808)  

**Abstract**: Translating natural language queries into SQL queries (NL2SQL or Text-to-SQL) has recently been empowered by large language models (LLMs). Using LLMs to perform NL2SQL methods on a large collection of SQL databases necessitates processing large quantities of meta-information about the databases, which in turn results in lengthy prompts with many tokens and high processing costs. To address this challenge, we introduce Datalake Agent, an agentic system designed to enable an LLM to solve NL2SQL tasks more efficiently. Instead of utilizing direct solvers for NL2SQL that call the LLM once with all meta-information in the prompt, the Datalake Agent employs an interactive loop to reduce the utilized meta-information. Within the loop, the LLM is used in a reasoning framework that selectively requests only the necessary information to solve a table question answering task. We evaluate the Datalake Agent on a collection of 23 databases with 100 table question answering tasks. The Datalake Agent reduces the tokens used by the LLM by up to 87\% and thus allows for substantial cost reductions while maintaining competitive performance. 

**Abstract (ZH)**: 利用大规模语言模型将自然语言查询转换为SQL查询（NL2SQL或Text-to-SQL）最近得到了增强。为了在大量SQL数据库集合上执行NL2SQL方法，需要处理大量关于数据库的元信息，从而导致长且包含大量令牌的提示和高处理成本。为应对这一挑战，我们引入了Datalake Agent，这是一种旨在使LLM更高效地解决NL2SQL任务的代理系统。Datalake Agent不使用直接求解器一次性将所有元信息包含在提示中调用LLM，而是采用交互循环来减少使用的元信息。在这个循环中，LLM被用于推理框架中，仅选择性地请求解决表格问题解答任务所必需的信息。我们在包含23个数据库和100个表格问题解答任务的集合上评估了Datalake Agent。Datalake Agent将LLM使用的令牌减少了高达87%，从而在显著降低成本的同时保持了竞争力。 

---
# SimKO: Simple Pass@K Policy Optimization 

**Title (ZH)**: SimKO: 简单的 Pass@K 策略优化 

**Authors**: Ruotian Peng, Yi Ren, Zhouliang Yu, Weiyang Liu, Yandong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14807)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has advanced the reasoning capabilities of large language models (LLMs). However, prevailing RLVR methods exhibit a systematic bias toward exploitation over exploration, as evidenced by improved pass@1 but reduced pass@K (K>1) performance. To understand this issue, we analyze training dynamics of RLVR methods by tracking the token-level probability distributions over vocabulary candidates. Our analysis reveals a consistent probability concentration effect where the top-1 candidate increasingly accumulates probability mass and suppresses that of other candidates. More importantly, stronger over-concentration correlates with worse pass@K performance. Inspired by this finding, we propose Simple Pass@K Optimization (SimKO), a method designed to mitigate the over-concentration issue, thereby encouraging exploration. SimKO operates in an asymmetrical manner. For verified-correct responses, it boosts the probabilities of the top-K candidates. For verified-incorrect responses, it applies stronger penalties to the top-1 candidate. We observe that this asymmetric design is particularly effective at mitigating over-concentration when applied at tokens with high entropy. Across various math and logical-reasoning benchmarks, SimKO consistently yields higher pass@K for a wide range of K, providing a simple way to improve RLVR's exploration. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）增强了大型语言模型的推理能力。然而，现有的RLVR方法在表现上倾向于利用而非探索，表现为pass@1性能提高但pass@K（K>1）性能下降。为了理解这一问题，我们通过追踪词汇候选词级别的概率分布来分析RLVR方法的训练动态。我们的分析揭示了一个一致的概率集中效应，即top-1候选词不断增加概率质量并抑制其他候选词的概率。更重要的是，过度集中越强，pass@K性能越差。受这一发现启发，我们提出了一种简单的pass@K优化方法（SimKO），旨在缓解过度集中问题，从而鼓励探索。SimKO以不对称的方式运作。对于验证正确的响应，它提升top-K候选词的概率；对于验证错误的响应，它对top-1候选词施加更强的惩罚。我们发现，当应用于高熵的令牌时，这种不对称设计特别有效地缓解过度集中。在各种数学和逻辑推理基准测试中，SimKO在广泛的K值下一致地提供了更高的pass@K性能，提供了一种简单的方法来提高RLVR的探索能力。 

---
# ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling 

**Title (ZH)**: ToolPRM: 结构化输出函数调用的细粒度推理扩展方法 

**Authors**: Jianghao Lin, Yuanyuan Shi, Xin Peng, Renjie Ding, Hairui Wang, Yuxuan Peng, Bizhe Bai, Weixi Song, Fengshuo Bai, Huacan Chai, Weinan Zhang, Fei Huang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14703)  

**Abstract**: Large language models (LLMs) are increasingly demonstrating strong capabilities as autonomous agents, with function calling serving as a core mechanism for interaction with the environment. Meanwhile, inference scaling has become a cutting-edge technique to enhance LLM performance by allocating more computational resources during the inference process. However, current research on inference scaling primarily focuses on unstructured output generation tasks, leaving its application in structured outputs, like function calling, largely underexplored. To bridge this gap, we propose an inference scaling framework that combines fine-grained beam search with a process reward model, ToolPRM, which scores the internal steps of each single function call. To train ToolPRM, we construct the first fine-grained intra-call process supervision dataset, automatically annotated with function-masking techniques to provide step-level rewards for structured tool-use reasoning. Extensive experiments demonstrate that ToolPRM beats the coarse-grained and outcome reward models in terms of predictive accuracy, indicating its stronger capability in supervising the function calling inference process. Inference scaling technique equipped with ToolPRM also significantly improves the backbone model performance across various function calling tasks and benchmarks. More importantly, we reveal a key principle for applying inference scaling techniques to structured outputs: "explore more but retain less" due to the unrecoverability characteristics of structured function calling generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益展现出强大的自主代理能力，功能调用作为与环境交互的核心机制，正在受到广泛关注。同时，推理扩展已成为一种前沿技术，通过在推理过程中分配更多计算资源来提高LLM的性能。然而，当前关于推理扩展的研究主要集中在无结构输出生成任务上，而其在结构化输出，如功能调用，方面的应用却相对未被充分探索。为进一步填补这一空白，我们提出了一种结合细粒度贝叶斯搜索和过程奖励模型（ToolPRM）的推理扩展框架，ToolPRM用于评估每次单独函数调用的内部步骤。为训练ToolPRM，我们构建了首个细粒度的函数调用内部过程监督数据集，通过函数遮掩技术自动标注，以提供步骤级奖励，从而支持结构化工具使用推理。广泛的实验表明，ToolPRM在预测准确性方面优于粗粒度和结果奖励模型，表明其在监督功能调用推理过程方面的更强能力。配以ToolPRM的推理扩展技术也显著提升了各种功能调用任务和基准下的主干模型性能。更重要的是，我们揭示出应用推理扩展技术到结构化输出的关键原则：“探索更多但保留更少”，这源于结构化功能调用生成的不可恢复特性。 

---
# Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction 

**Title (ZH)**: 面向认知的时空大规模语言模型用于下一步兴趣点预测 

**Authors**: Penglong Zhai, Jie Li, Fanyi Di, Yue Liu, Yifang Yuan, Jie Huang, Peng Wu, Sicong Wang, Mingyang Yin, Tingting Hu, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14702)  

**Abstract**: The next point-of-interest (POI) recommendation task aims to predict the users' immediate next destinations based on their preferences and historical check-ins, holding significant value in location-based services. Recently, large language models (LLMs) have shown great potential in recommender systems, which treat the next POI prediction in a generative manner. However, these LLMs, pretrained primarily on vast corpora of unstructured text, lack the native understanding of structured geographical entities and sequential mobility patterns required for next POI prediction tasks. Moreover, in industrial-scale POI prediction applications, incorporating world knowledge and alignment of human cognition, such as seasons, weather conditions, holidays, and users' profiles (such as habits, occupation, and preferences), can enhance the user experience while improving recommendation performance. To address these issues, we propose CoAST (Cognitive-Aligned Spatial-Temporal LLMs), a framework employing natural language as an interface, allowing for the incorporation of world knowledge, spatio-temporal trajectory patterns, profiles, and situational information. Specifically, CoAST mainly comprises of 2 stages: (1) Recommendation Knowledge Acquisition through continued pretraining on the enriched spatial-temporal trajectory data of the desensitized users; (2) Cognitive Alignment to align cognitive judgments with human preferences using enriched training data through Supervised Fine-Tuning (SFT) and a subsequent Reinforcement Learning (RL) phase. Extensive offline experiments on various real-world datasets and online experiments deployed in "Guess Where You Go" of AMAP App homepage demonstrate the effectiveness of CoAST. 

**Abstract (ZH)**: 基于认知对齐的空间时间大型语言模型：CoAST在下一步兴趣点推荐中的应用 

---
# Purifying Task Vectors in Knowledge-Aware Subspace for Model Merging 

**Title (ZH)**: 知识感知子空间中任务向量的净化与模型融合 

**Authors**: Bang An, Yibo Yang, Philip Torr, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14697)  

**Abstract**: Model merging aims to integrate task-specific abilities from individually fine-tuned models into a single model without extra training. In recent model merging methods, task vector has become a fundamental building block, as it can encapsulate the residual information from finetuning. However, the merged model often suffers from notable performance degradation due to the conflicts caused by task-irrelevant redundancy in task vectors. Existing efforts in overcoming redundancy by randomly dropping elements in the parameter space involves randomness and lacks knowledge awareness. To address these challenges, in this study, we propose Purifying TAsk Vectors (PAVE) in knowledge-aware subspace. Concretely, we sample some training examples from each task, and feed them into their corresponding fine-tuned models to acquire the covariance matrices before linear layers. We then perform a context-oriented singular value decomposition, which accentuates the weight components most relevant to the target knowledge. As a result, we can split fine-tuned model weights into task-relevant and redundant components in the knowledge-aware subspace, and purify the task vector by pruning the redundant components. To induce fair pruning efforts across models, we further introduce a spectral rank allocation strategy by optimizing a normalized activated pruning error. The task vector purification by our method as a plug-and-play scheme is applicable across various task vector-based merging methods to improve their performance. In experiments, we demonstrate the effectiveness of PAVE across a diverse set of merging methods, tasks, and model architectures. 

**Abstract (ZH)**: 知识导向子空间中净化任务向量（PAVE）模型融合 

---
# Practical, Utilitarian Algorithm Configuration 

**Title (ZH)**: 实用型实用算法配置 

**Authors**: Devon Graham, Kevin Leyton-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.14683)  

**Abstract**: Utilitarian algorithm configuration identifies a parameter setting for a given algorithm that maximizes a user's utility. Utility functions offer a theoretically well-grounded approach to optimizing decision-making under uncertainty and are flexible enough to capture a user's preferences over algorithm runtimes (e.g., they can describe a sharp cutoff after which a solution is no longer required, a per-hour cost for compute, or diminishing returns from algorithms that take longer to run). COUP is a recently-introduced utilitarian algorithm configuration procedure which was designed mainly to offer strong theoretical guarantees about the quality of the configuration it returns, with less attention paid to its practical performance. This paper closes that gap, bringing theoretically-grounded, utilitarian algorithm configuration to the point where it is competitive with widely used, heuristic configuration procedures that offer no performance guarantees. We present a series of improvements to COUP that improve its empirical performance without degrading its theoretical guarantees and demonstrate their benefit experimentally. Using a case study, we also illustrate ways of exploring the robustness of a given solution to the algorithm selection problem to variations in the utility function. 

**Abstract (ZH)**: 效用算法配置确定给定算法的一个参数设置，以最大化用户的效用。效用函数提供了一种理论上扎实的方法来优化不确定性下的决策，并且足够灵活以捕捉用户对算法运行时间的偏好（例如，它们可以描述一个清晰的截止点，在此之后不再需要解决方案，每小时的计算成本，或者随运行时间增加而递减的回报）。COUP 是最近引入的一种效用算法配置程序，主要旨在提供关于返回的配置质量的强大理论保证，而在其实用性能方面则关注较少。本文填补了这一空白，使理论扎实的效用算法配置接近于广泛使用的、没有性能保证的启发式配置程序。我们提出了一系列改进 COUP 的方法，这些改进提高了其 empirical 性能而不牺牲其理论保证，并通过实验证明了这些改进的好处。使用案例研究，我们还展示了如何探索给定的算法选择问题解决方案对效用函数变化的鲁棒性。 

---
# NAEL: Non-Anthropocentric Ethical Logic 

**Title (ZH)**: NAEL: 非人性中心的伦理逻辑 

**Authors**: Bianca Maria Lerma, Rafael Peñaloza  

**Link**: [PDF](https://arxiv.org/pdf/2510.14676)  

**Abstract**: We introduce NAEL (Non-Anthropocentric Ethical Logic), a novel ethical framework for artificial agents grounded in active inference and symbolic reasoning. Departing from conventional, human-centred approaches to AI ethics, NAEL formalizes ethical behaviour as an emergent property of intelligent systems minimizing global expected free energy in dynamic, multi-agent environments. We propose a neuro-symbolic architecture to allow agents to evaluate the ethical consequences of their actions in uncertain settings. The proposed system addresses the limitations of existing ethical models by allowing agents to develop context-sensitive, adaptive, and relational ethical behaviour without presupposing anthropomorphic moral intuitions. A case study involving ethical resource distribution illustrates NAEL's dynamic balancing of self-preservation, epistemic learning, and collective welfare. 

**Abstract (ZH)**: NAEL（非anthropocentric伦理逻辑）：一种基于主动推断和符号推理的新型伦理框架 

---
# TITAN: Graph-Executable Reasoning for Cyber Threat Intelligence 

**Title (ZH)**: TITAN: 基于图执行推理的网络威胁情报分析 

**Authors**: Marco Simoni, Aleksandar Fontana, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2510.14670)  

**Abstract**: TITAN (Threat Intelligence Through Automated Navigation) is a framework that connects natural-language cyber threat queries with executable reasoning over a structured knowledge graph. It integrates a path planner model, which predicts logical relation chains from text, and a graph executor that traverses the TITAN Ontology to retrieve factual answers and supporting evidence. Unlike traditional retrieval systems, TITAN operates on a typed, bidirectional graph derived from MITRE, allowing reasoning to move clearly and reversibly between threats, behaviors, and defenses. To support training and evaluation, we introduce the TITAN Dataset, a corpus of 88209 examples (Train: 74258; Test: 13951) pairing natural language questions with executable reasoning paths and step by step Chain of Thought explanations. Empirical evaluations show that TITAN enables models to generate syntactically valid and semantically coherent reasoning paths that can be deterministically executed on the underlying graph. 

**Abstract (ZH)**: TITAN（威胁情报通过自动导航）是将自然语言网络威胁查询与结构化知识图谱上的可执行推理连接起来的框架。 

---
# Machine Learning and Public Health: Identifying and Mitigating Algorithmic Bias through a Systematic Review 

**Title (ZH)**: 机器学习与公共卫生：通过系统评价识别和缓解算法偏见 

**Authors**: Sara Altamirano, Arjan Vreeken, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2510.14669)  

**Abstract**: Machine learning (ML) promises to revolutionize public health through improved surveillance, risk stratification, and resource allocation. However, without systematic attention to algorithmic bias, ML may inadvertently reinforce existing health disparities. We present a systematic literature review of algorithmic bias identification, discussion, and reporting in Dutch public health ML research from 2021 to 2025. To this end, we developed the Risk of Algorithmic Bias Assessment Tool (RABAT) by integrating elements from established frameworks (Cochrane Risk of Bias, PROBAST, Microsoft Responsible AI checklist) and applied it to 35 peer-reviewed studies. Our analysis reveals pervasive gaps: although data sampling and missing data practices are well documented, most studies omit explicit fairness framing, subgroup analyses, and transparent discussion of potential harms. In response, we introduce a four-stage fairness-oriented framework called ACAR (Awareness, Conceptualization, Application, Reporting), with guiding questions derived from our systematic literature review to help researchers address fairness across the ML lifecycle. We conclude with actionable recommendations for public health ML practitioners to consistently consider algorithmic bias and foster transparency, ensuring that algorithmic innovations advance health equity rather than undermine it. 

**Abstract (ZH)**: 机器学习（ML）有望通过改进监控、风险分层和资源分配来革新公共卫生。然而，如果没有系统地关注算法偏见，ML可能无意中加剧现有的健康不平等。我们对2021年至2025年间荷兰公共卫生机器学习研究中的算法偏见识别、讨论和报告进行了系统文献回顾。为此，我们开发了算法偏见风险评估工具（RABAT），并将现有框架（Cochrane偏倚风险、PROBAST、微软负责任AI检查表）中的元素进行整合，并应用于35项同行评审研究。我们的分析揭示普遍存在空白：尽管数据采样和缺失数据处理方法已详细记录，但大多数研究未明确包含公平性框架、亚组分析以及潜在危害的透明讨论。为此，我们引入了一个四阶段公平导向框架（ACAR，意识、概念化、应用、报告），并从系统文献回顾中衍生出指导问题，以帮助研究人员在机器学习生命周期中全面考虑公平性。我们最终提出可操作的建议，指导公共卫生机器学习从业人员始终考虑算法偏见，促进透明度，确保算法创新促进卫生公平而非削弱它。 

---
# Beyond Hallucinations: The Illusion of Understanding in Large Language Models 

**Title (ZH)**: 超越幻觉：大型语言模型中的理解错觉 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14665)  

**Abstract**: Large language models (LLMs) are becoming deeply embedded in human communication and decision-making, yet they inherit the ambiguity, bias, and lack of direct access to truth inherent in language itself. While their outputs are fluent, emotionally resonant, and coherent, they are generated through statistical prediction rather than grounded reasoning. This creates the risk of hallucination, responses that sound convincing but lack factual validity. Building on Geoffrey Hinton's observation that AI mirrors human intuition rather than reasoning, this paper argues that LLMs operationalize System 1 cognition at scale: fast, associative, and persuasive, but without reflection or falsification. To address this, we introduce the Rose-Frame, a three-dimensional framework for diagnosing cognitive and epistemic drift in human-AI interaction. The three axes are: (i) Map vs. Territory, which distinguishes representations of reality (epistemology) from reality itself (ontology); (ii) Intuition vs. Reason, drawing on dual-process theory to separate fast, emotional judgments from slow, reflective thinking; and (iii) Conflict vs. Confirmation, which examines whether ideas are critically tested through disagreement or simply reinforced through mutual validation. Each dimension captures a distinct failure mode, and their combination amplifies misalignment. Rose-Frame does not attempt to fix LLMs with more data or rules. Instead, it offers a reflective tool that makes both the model's limitations and the user's assumptions visible, enabling more transparent and critically aware AI deployment. It reframes alignment as cognitive governance: intuition, whether human or artificial, must remain governed by human reason. Only by embedding reflective, falsifiable oversight can we align machine fluency with human understanding. 

**Abstract (ZH)**: 大型语言模型（LLMs）在人类交流和决策中扮演着日益重要的角色，但它们不可避免地继承了语言本身的模糊性、偏见和对事实缺乏直接访问的特点。尽管它们的输出流畅、富有情感共鸣且连贯，但这些输出是通过统计预测生成的，而非基于牢固推理。这带来了幻觉的风险，即听起来令人信服但实际上缺乏事实依据的回答。受Geoffrey Hinton观察到的AI反映人类直觉而非推理的启发，本文认为LLMs规模化地实现了系统1认知：快速、关联且有说服力，但缺乏反思和反驳。为解决这一问题，本文引入了Rose-Frame框架，这是一种三维框架，用于诊断人机交互中的认知和知识漂移。该框架的三个轴分别是：(i) 地图 vs. 地域，区分现实的表征（认识论）与现实本身（本体论）；(ii) 直觉 vs. 推理，借鉴双过程理论，分离快速的情绪判断与缓慢的反思性思考；(iii) 冲突 vs. 确认，考察思想是否通过异议进行批判性测试，还是仅仅通过相互验证得到强化。每个维度捕获了一种独特的失败模式，它们的结合增强了不一致性的放大效应。Rose-Frame并不试图通过增加数据或规则来修复LLMs。相反，它提供了一种反思工具，使模型的局限性和用户的假设变得明显，从而使更透明和批判性地意识的AI部署成为可能。它将对齐重新定义为认知治理：无论是人类还是人工，直觉都必须受人类推理的治理。只有通过嵌入反思和可验证的监督，我们才能使机器流畅性与人类理解相一致。 

---
# ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks 

**Title (ZH)**: ColorBench：基于图结构框架评估移动代理复杂长时间任务能力 

**Authors**: Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Jun Wang, Yong Yu, Weinan Zhang, Zhaoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14621)  

**Abstract**: The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: this https URL. 

**Abstract (ZH)**: 多模态大型语言模型的快速进展使得代理能够通过直接与图形用户界面交互来操作移动设备，为移动自动化开启了新的可能性。然而，现实中的移动任务往往是复杂的，并允许多个有效的解决方案。这与当前的移动代理评估标准相矛盾：离线静态基准只能验证一个预定义的“黄金路径”，而在线动态测试受限于真实设备的复杂性和不可再现性，使这两种方法都无法全面评估代理的能力。为了弥合离线和在线评估之间的差距并增强测试稳定性，本文引入了一个新颖的图结构基准框架。通过建模实际设备交互中观察到的有限状态，实现了动态行为的静态仿真。在此基础上，我们开发了ColorBench，一个侧重于复杂长时任务的基准。它支持评估多个有效解决方案、子任务完成率统计和原子级能力分析。ColorBench 包含 175 个任务（74 个单应用，101 个跨应用），平均长度超过 13 步。每个任务至少包含两条正确的路径和几个典型的错误路径，实现准动态交互。通过在各种基线上评估 ColorBench，我们发现了现有模型的局限性，并基于实验结果提出改进方向和技术路径，以提高代理在处理复杂长时问题时的表现。代码和数据可在以下网址获取：this https URL。 

---
# LLM Agents Beyond Utility: An Open-Ended Perspective 

**Title (ZH)**: LLM代理超越实用性：一种开放视角 

**Authors**: Asen Nachkov, Xi Wang, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2510.14548)  

**Abstract**: Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals. 

**Abstract (ZH)**: 近年来，预训练的大语言模型（LLM）代理在链式思维推理和函数调用方面取得了巨大进展。随着其能力的提升，一个重要问题出现了：这种软件是否不仅能作为智能解决问题的工具，还能成为一个独立的实体，能够制定计划、设计即时任务，并朝向更为宽泛且模糊的目标进行推理？为了研究这一问题，我们采用了开放式的实验设定，将预训练的LLM代理增强使其能够生成自己的任务、积累知识，并与环境进行广泛互动。我们对最终的开放性代理进行了定性的研究。它能够可靠地遵循复杂的多步骤指令，跨运行存储和重用信息，并能够提出和解决自己的任务，但仍然受到提示设计的影响，容易生成重复的任务，并且无法形成自我表征。这些发现既展示了将预训练的LLM模型朝开放性方向发展的潜力，也揭示了当前的局限性，并指出了未来培训代理以管理记忆、有效探索和追求抽象长期目标的方向。 

---
# Symbol Grounding in Neuro-Symbolic AI: A Gentle Introduction to Reasoning Shortcuts 

**Title (ZH)**: 神经符号AI中的符号 grounding：推理快捷方式的温和介绍 

**Authors**: Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Paolo Morettin, Elena Umili, Antonio Vergari, Efthymia Tsamoura, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2510.14538)  

**Abstract**: Neuro-symbolic (NeSy) AI aims to develop deep neural networks whose predictions comply with prior knowledge encoding, e.g. safety or structural constraints. As such, it represents one of the most promising avenues for reliable and trustworthy AI. The core idea behind NeSy AI is to combine neural and symbolic steps: neural networks are typically responsible for mapping low-level inputs into high-level symbolic concepts, while symbolic reasoning infers predictions compatible with the extracted concepts and the prior knowledge. Despite their promise, it was recently shown that - whenever the concepts are not supervised directly - NeSy models can be affected by Reasoning Shortcuts (RSs). That is, they can achieve high label accuracy by grounding the concepts incorrectly. RSs can compromise the interpretability of the model's explanations, performance in out-of-distribution scenarios, and therefore reliability. At the same time, RSs are difficult to detect and prevent unless concept supervision is available, which is typically not the case. However, the literature on RSs is scattered, making it difficult for researchers and practitioners to understand and tackle this challenging problem. This overview addresses this issue by providing a gentle introduction to RSs, discussing their causes and consequences in intuitive terms. It also reviews and elucidates existing theoretical characterizations of this phenomenon. Finally, it details methods for dealing with RSs, including mitigation and awareness strategies, and maps their benefits and limitations. By reformulating advanced material in a digestible form, this overview aims to provide a unifying perspective on RSs to lower the bar to entry for tackling them. Ultimately, we hope this overview contributes to the development of reliable NeSy and trustworthy AI models. 

**Abstract (ZH)**: 神经符号（NeSy）人工智能旨在开发遵循先验知识（例如安全性或结构约束）的深度神经网络。因此，它代表了可靠和可信赖人工智能的一个最有前途的方向。NeSy人工智能的核心思想是结合神经和符号步骤：神经网络通常负责将低级输入映射为高级符号概念，而符号推理则根据提取的概念和先验知识推断兼容的预测。尽管它们很有前景，但最近的研究显示，当概念未直接监督时，NeSy模型可能会受到推理捷径（RSs）的影响。也就是说，它们可以通过错误地链接概念来实现高标签准确性。RSs可以削弱模型解释的可解释性，影响异常分布场景中的性能，从而影响可靠性。同时，除非有概念监督，否则检测和防止RSs是困难的，而这种情况通常不会发生。然而，关于RSs的文献比较分散，这使得研究人员和实践者难以理解并解决这一具有挑战性的问题。本文综述通过提供一种直观的介绍RSs、讨论其成因和后果，并回顾和阐明现有的理论刻画，最后详细阐述解决RSs的方法，包括缓解和意识策略及其优缺点，旨在以一种易于理解的形式重新表述高级材料，为解决RSs提供统一的视角，降低解决它们的门槛。最终，我们希望本文综述能够促进可靠NeSy和可信赖人工智能模型的发展。 

---
# JSPLIT: A Taxonomy-based Solution for Prompt Bloating in Model Context Protocol 

**Title (ZH)**: JSPLIT：基于分类学的模型上下文协议中提示膨胀解决方案 

**Authors**: Emanuele Antonioni, Stefan Markovic, Anirudha Shankar, Jaime Bernardo, Lovro Markovic, Silvia Pareti, Benedetto Proietti  

**Link**: [PDF](https://arxiv.org/pdf/2510.14537)  

**Abstract**: AI systems are continually evolving and advancing, and user expectations are concurrently increasing, with a growing demand for interactions that go beyond simple text-based interaction with Large Language Models (LLMs). Today's applications often require LLMs to interact with external tools, marking a shift toward more complex agentic systems. To support this, standards such as the Model Context Protocol (MCP) have emerged, enabling agents to access tools by including a specification of the capabilities of each tool within the prompt. Although this approach expands what agents can do, it also introduces a growing problem: prompt bloating. As the number of tools increases, the prompts become longer, leading to high prompt token costs, increased latency, and reduced task success resulting from the selection of tools irrelevant to the prompt. To address this issue, we introduce JSPLIT, a taxonomy-driven framework designed to help agents manage prompt size more effectively when using large sets of MCP tools. JSPLIT organizes the tools into a hierarchical taxonomy and uses the user's prompt to identify and include only the most relevant tools, based on both the query and the taxonomy structure. In this paper, we describe the design of the taxonomy, the tool selection algorithm, and the dataset used to evaluate JSPLIT. Our results show that JSPLIT significantly reduces prompt size without significantly compromising the agent's ability to respond effectively. As the number of available tools for the agent grows substantially, JSPLIT even improves the tool selection accuracy of the agent, effectively reducing costs while simultaneously improving task success in high-complexity agent environments. 

**Abstract (ZH)**: 基于-taxonomy的JSPLIT框架：有效管理大规模MCP工具的提示大小 

---
# Helmsman: Autonomous Synthesis of Federated Learning Systems via Multi-Agent Collaboration 

**Title (ZH)**: Helmsman: 通过多智能体协作自主合成联邦学习系统 

**Authors**: Haoyuan Li, Mathias Funk, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14512)  

**Abstract**: Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems. 

**Abstract (ZH)**: 联邦学习(Federated Learning)提供了一种强大的范式来在分布式数据上训练模型，但其潜力常因设计和部署稳健系统所面临的巨大复杂性而受挫。对数据异质性和系统约束等多方面挑战的策略选择、组合与调整已经成为关键瓶颈，导致了脆弱且定制的解决方案。为解决这一问题，我们引入了Helmsman，这是一种新颖的多代理系统，能够从高层次用户规范自动合成端到端的联邦学习系统。它通过三个协作阶段模仿了规范的研究和开发工作流程：(1) 交互式的人机环规划以制定严谨的研究计划，(2) 监督代理团队的模块化代码生成，以及(3) 沙箱仿真环境中的自主评价和优化闭环。为了促进严格的评估，我们还引入了AgentFL-Bench，这是一种包含16项不同任务的新基准，旨在评估代理系统在联邦学习中的系统级生成能力。全面的实验展示了我们方法生成的竞争性和优越性解决方案，通常优于现有的手工构建基准。我们的工作代表了自动工程复杂分布式AI系统的重要一步。 

---
# Eliminating Negative Occurrences of Derived Predicates from PDDL Axioms 

**Title (ZH)**: 从PDDL公理中消除派生谓词的负面出现 

**Authors**: Claudia Grundke, Gabriele Röger  

**Link**: [PDF](https://arxiv.org/pdf/2510.14412)  

**Abstract**: Axioms are a feature of the Planning Domain Definition Language PDDL that can be considered as a generalization of database query languages such as Datalog. The PDDL standard restricts negative occurrences of predicates in axiom bodies to predicates that are directly set by actions and not derived by axioms. In the literature, authors often deviate from this limitation and only require that the set of axioms is stratifiable. Both variants can express exactly the same queries as least fixed-point logic, indicating that negative occurrences of derived predicates can be eliminated. We present the corresponding transformation. 

**Abstract (ZH)**: axioms是Planning Domain Definition Language PDDL的一个特征，可以视为数据库查询语言如Datalog的一般化。PDDL标准限制了在公理体内出现的负谓词仅限于由动作直接设置的谓词，而非由公理推导出的谓词。文献中，作者通常偏离这一限制，只需确保公理集是可分层的。这两种变体都可以精确地表达最少固定点逻辑所能表达的所有查询，表明可以消除导出谓词的负出现象。我们提出了相应的转换。 

---
# IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning 

**Title (ZH)**: IMAGINE: 将多智能体系统集成到单一模型中进行复杂推理和规划 

**Authors**: Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14406)  

**Abstract**: Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在各种任务上取得了显著进展，但在复杂推理和规划方面仍然面临重大挑战。例如，即使使用精心设计的提示和明确提供的先验信息，GPT-4o在独立试划模式下的最终通过率为7%。同样，在思考模式下，Qwen3-8B-Instruct和DeepSeek-R1-671B的最终通过率分别为5.9%和40%。虽然井然有序的多代理系统（MAS）可以提供改进的集体推理能力，但由于多轮内部交互、每个响应的长时延和端到端训练的困难，往往会导致推理成本高。为解决这些问题，我们提出了一种名为IMAGINE的一般性和可扩展框架，其全称为将多代理系统集成到一个模型中。该框架不仅将MAS的推理和规划能力整合到一个紧凑型模型中，而且通过简单的端到端训练显著超越了MAS的能力。通过这个流程，一个小规模模型不仅能获取井然有序的MAS的结构化推理和规划能力，还能显著超越它。实验结果表明，使用Qwen3-8B-Instruct作为基础模型并通过我们的方法训练时，该模型在TravelPlanner基准上的最终通过率为82.7%，远超DeepSeek-R1-671B的40%，且模型规模要小得多。 

---
# Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control 

**Title (ZH)**: Hi-Agent: 分层级的视觉-语言代理用于移动设备控制 

**Authors**: Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14388)  

**Abstract**: Building agents that autonomously operate mobile devices has attracted increasing attention. While Vision-Language Models (VLMs) show promise, most existing approaches rely on direct state-to-action mappings, which lack structured reasoning and planning, and thus generalize poorly to novel tasks or unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized. For efficient training, we reformulate multi-step decision-making as a sequence of single-step subgoals and propose a foresight advantage function, which leverages execution feedback from the low-level model to guide high-level optimization. This design alleviates the path explosion issue encountered by Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art (SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark, significantly outperforming prior methods across three paradigms: prompt-based (AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot generalization on the ScreenSpot-v2 benchmark. On the more challenging AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones, showing strong adaptability in high-complexity mobile control scenarios. 

**Abstract (ZH)**: 自主操作移动设备的代理构建引起了越来越多的关注。虽然视觉-语言模型（VLMs）表现出色，但大多数现有方法依赖于直接的状态到动作映射，缺乏结构化的推理和规划，因此在处理新型任务或未见的UI布局时泛化能力较差。我们提出了Hi-Agent，一个用于移动控制的可训练层次视觉-语言代理，该代理包含一个高层推理模型和一个低层动作模型，并且两者是联合优化的。为实现高效的训练，我们将多步决策问题重新表述为一系列单一步骤的子目标，并提出了一种前瞻优势函数，该函数利用低层模型的执行反馈来指导高层优化。这一设计缓解了长时序任务中基于组相对策略优化（GRPO）方法遇到的路径爆炸问题，并使稳定、无批评家的联合训练成为可能。Hi-Agent 在Android-in-the-Wild（AitW）基准测试中达到了新的最佳表现，任务成功率为87.9%，显著优于先前的方法，在三种范式中均表现出更优异的表现：提示驱动（AppAgent：17.7%）、监督学习（过滤后的BC：54.5%）和强化学习驱动（DigiRL：71.9%）。它还在ScreenSpot-v2基准测试中展示了较强的零-shot泛化能力。在更具挑战性的AndroidWorld基准测试中，Hi-Agent 也随着更大模型规模的有效扩展，在高复杂度的移动控制场景中表现出强烈的适应能力。 

---
# Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch? 

**Title (ZH)**: MLLMs能像免费午餐一样从LLMs中吸收数学推理能力？ 

**Authors**: Yijie Hu, Zihao Zhou, Kaizhu Huang, Xiaowei Huang, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14387)  

**Abstract**: Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities. 

**Abstract (ZH)**: 多模态大语言模型的数学推理能力提升：无需调优的模型合并方法 

---
# AI for Service: Proactive Assistance with AI Glasses 

**Title (ZH)**: AI 服务：AI 眼镜下的主动协助 

**Authors**: Zichen Wen, Yiyu Wang, Chenfei Liao, Boxue Yang, Junxian Li, Weifeng Liu, Haocong He, Bolong Feng, Xuyang Liu, Yuanhuiyi Lyu, Xu Zheng, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14359)  

**Abstract**: In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glasses, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts. 

**Abstract (ZH)**: 在人工智能从被动工具演化为主动适应伴侣的时代，我们提出了AI服务（AI4Service）这一新范式，以实现日常生活中的主动和实时辅助。现有的AI服务主要反应式地响应用户的显式命令。我们主张，真正智能且有帮助的助手应该能够预测用户需求并在适当的时候主动采取行动。为实现这一愿景，我们提出了一种统一框架Alpha-Service，该框架解决了两个基本挑战：知晓何时干预，通过检测自我中心视频流中的服务机会；知晓如何提供，提供通用和个性化服务。Alpha-Service借鉴了冯·诺依曼计算机架构并基于AI眼镜，由五个关键组件组成：感知单元、中央处理单元、算术逻辑单元、长期个性化存储单元以及自然人机交互单元。作为初步探索，我们通过部署在AI眼镜上的多智能体系统实现Alpha-Service。案例研究，包括实时黑 jack 导师、博物馆导游和购物搭配助手，展示了其无缝感知环境、推断用户意图并在无需明确提示的情况下提供及时有用辅助的能力。 

---
# Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction 

**Title (ZH)**: 基于原型引导的下一次执行重建的元认知自我修正多agent系统 

**Authors**: Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14319)  

**Abstract**: Large Language Model based multi-agent systems (MAS) excel at collaborative problem solving but remain brittle to cascading errors: a single faulty step can propagate across agents and disrupt the trajectory. In this paper, we present MASC, a metacognitive framework that endows MAS with real-time, unsupervised, step-level error detection and self-correction. MASC rethinks detection as history-conditioned anomaly scoring via two complementary designs: (1) Next-Execution Reconstruction, which predicts the embedding of the next step from the query and interaction history to capture causal consistency, and (2) Prototype-Guided Enhancement, which learns a prototype prior over normal-step embeddings and uses it to stabilize reconstruction and anomaly scoring under sparse context (e.g., early steps). When an anomaly step is flagged, MASC triggers a correction agent to revise the acting agent's output before information flows downstream. On the Who&When benchmark, MASC consistently outperforms all baselines, improving step-level error detection by up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers consistent end-to-end gains across architectures, confirming that our metacognitive monitoring and targeted correction can mitigate error propagation with minimal overhead. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统中的元认知框架：实时无监督的步骤级错误检测与自修正 

---
# Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies 

**Title (ZH)**: Terrarium: 重新审视黑板架构以研究多智能体的安全性、隐私性和安全性 

**Authors**: Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2510.14312)  

**Abstract**: A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems. 

**Abstract (ZH)**: 基于大型语言模型的多代理系统（MAS）可以通过自动化如会议调度等繁琐的用户任务，并支持代理间的协作。大型语言模型能够处理非结构化的私人数据、用户约束和偏好，实现细腻的协议。然而，这种设计引入了新的风险，包括模型偏差和恶意攻击者对代理的攻击或窃取用户数据。在本文中，我们提出了Terrarium框架，以细粒度地研究基于大型语言模型的多代理系统中的安全、隐私和安全问题。我们重新利用了多代理系统早期的黑板设计，创建了一个模块化、可配置的多代理协作测试平台。我们识别了关键攻击向量，如模型偏差、恶意代理、通信被劫持和数据投毒。我们实现了三个协作MAS场景，并应用四种代表性攻击来展示框架的灵活性。通过提供快速原型设计、评估和迭代安全措施和设计方案的工具，Terrarium旨在加速可信赖的多代理系统的发展。 

---
# A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space 

**Title (ZH)**: 一种保障安全性的工作机制：当安全性敏感子空间遇到抗有害性零空间 

**Authors**: Bingjie Zhang, Yibo Yang, Renzhe, Dandan Guo, Jindong Gu, Philip Torr, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14301)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in diverse tasks, yet their safety alignment remains fragile during adaptation. Even when fine-tuning on benign data or with low-rank adaptation, pre-trained safety behaviors are easily degraded, leading to harmful responses in the fine-tuned models. To address this challenge, we propose GuardSpace, a guardrail framework for preserving safety alignment throughout fine-tuning, composed of two key components: a safety-sensitive subspace and a harmful-resistant null space. First, we explicitly decompose pre-trained weights into safety-relevant and safety-irrelevant components using covariance-preconditioned singular value decomposition, and initialize low-rank adapters from the safety-irrelevant ones, while freezing safety-relevant components to preserve their associated safety mechanism. Second, we construct a null space projector that restricts adapter updates from altering safe outputs on harmful prompts, thereby maintaining the original refusal behavior. Experiments with various pre-trained models on multiple downstream tasks demonstrate that GuardSpace achieves superior performance over existing methods. Notably, for Llama-2-7B-Chat fine-tuned on GSM8K, GuardSpace outperforms the state-of-the-art method AsFT, reducing the average harmful score from 14.4% to 3.6%, while improving the accuracy from from 26.0% to 28.0%. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中取得了显著成功，但在 Adaptation 过程中其安全性对齐仍然脆弱。即使在良性数据上进行微调或低秩调整，预训练的安全行为也会容易退化，导致微调模型产生有害响应。为应对这一挑战，我们提出了 GuardSpace，这是一种在整个微调过程中保持安全性对齐的防护框架，由两个关键组件组成：安全性敏感子空间和抗有害零空间。首先，我们使用协方差预条件奇异值分解显式分解预训练权重为与安全相关和与安全无关的分量，并从与安全无关的分量初始化低秩适配器，同时冻结与安全相关分量以保留其相关安全机制。其次，我们构建了一个零空间投影器，该投影器限制适配器更新不改变有害提示的安全输出，从而保持原始拒绝行为。在多种预训练模型上的多下游任务实验表明，GuardSpace 较现有方法具有更优性能。值得注意的是，对于在 GSM8K 上微调的 Llama-2-7B-Chat，GuardSpace 在有害评分平均值从 14.4% 降低至 3.6% 的同时，准确率从 26.0% 提高至 28.0%，优于最先进的方法 AsFT。 

---
# MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning 

**Title (ZH)**: MorphoBench: 一种适应模型推理难度的基准评测 

**Authors**: Xukai Wang, Xuanbo Liu, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Bohan Zeng, Jinbo Hu, Hao Liang, Junbo Niu, Xuchen Li, Ruitao Wu, Ruichuan An, Yang Shi, Liu Liu, Xu-Yao Zhang, Qiang Liu, Zhouchen Lin, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.14265)  

**Abstract**: With the advancement of powerful large-scale reasoning models, effectively evaluating the reasoning capabilities of these models has become increasingly important. However, existing benchmarks designed to assess the reasoning abilities of large models tend to be limited in scope and lack the flexibility to adapt their difficulty according to the evolving reasoning capacities of the models. To address this, we propose MorphoBench, a benchmark that incorporates multidisciplinary questions to evaluate the reasoning capabilities of large models and can adjust and update question difficulty based on the reasoning abilities of advanced models. Specifically, we curate the benchmark by selecting and collecting complex reasoning questions from existing benchmarks and sources such as Olympiad-level competitions. Additionally, MorphoBench adaptively modifies the analytical challenge of questions by leveraging key statements generated during the model's reasoning process. Furthermore, it includes questions generated using simulation software, enabling dynamic adjustment of benchmark difficulty with minimal resource consumption. We have gathered over 1,300 test questions and iteratively adjusted the difficulty of MorphoBench based on the reasoning capabilities of models such as o3 and GPT-5. MorphoBench enhances the comprehensiveness and validity of model reasoning evaluation, providing reliable guidance for improving both the reasoning abilities and scientific robustness of large models. The code has been released in this https URL. 

**Abstract (ZH)**: 随着强大大规模推理模型的发展，有效评估这些模型的推理能力变得越来越重要。然而，现有的用于评估大模型推理能力的基准测试往往范围有限，缺乏根据模型推理能力的发展调整难度的灵活性。为了解决这一问题，我们提出了MorphoBench这一基准测试，该测试结合了跨学科的问题来评估大模型的推理能力，并可根据高级模型的推理能力调整和更新问题难度。具体来说，我们通过从现有基准测试和如奥林匹克级别的竞赛中精选和收集复杂推理问题来编制基准测试。此外，MorphoBench通过利用模型推理过程中生成的关键语句，动态调整问题的分析挑战。此外，它还包含使用模拟软件生成的问题，能够在极低的资源消耗下动态调整基准测试难度。我们已经收集了超过1,300个测试问题，并根据如o3和GPT-5等模型的推理能力迭代调整了MorphoBench的难度。MorphoBench增强了模型推理评估的全面性和有效性，为提高大模型的推理能力和科学稳健性提供了可靠指导。代码已在此处发布：https://。 

---
# Towards Agentic Self-Learning LLMs in Search Environment 

**Title (ZH)**: 面向搜索环境的自主自我学习语言模型 

**Authors**: Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14253)  

**Abstract**: We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at this https URL 

**Abstract (ZH)**: 我们研究自学习是否能在无需依赖人类标注数据集或预定义规则奖励的情况下扩展基于大语言模型的智能体。通过在搜索智能体设置下的受控实验，我们确定了可扩展智能体训练的两个关键因素：奖励信号的来源和智能体任务数据的规模。我们发现，对于开放领域学习而言，生成式奖励模型（GRM）的奖励优于僵化的规则基信号，并且与策略共同进化进一步提升了性能。即使任务数据是合成生成的，增加其数量也能显著增强智能体的能力。基于这些洞察，我们提出了Agentic Self-Learning（ASL）——一个完全闭环、多角色增强学习框架，将任务生成、策略执行和评估统一在一个共享工具环境中，并使用大语言模型作为主干。ASL 组织提示生成器、策略模型和生成式奖励模型形成一个更难任务设定、更精确验证、更强解决的良性循环。实验证明，ASL 持续带来稳定的收益提升，超越了在某些条件下停滞或退化的强基准（如 Search-R1），并在零标注数据条件下继续改进，表明其具有更好的样本效率和鲁棒性。我们进一步证明，生成式奖励模型的验证能力是主要瓶颈：如果冻结，会导致奖励作弊并阻碍进展；在不断变化的数据分布上持续训练生成式奖励模型可以缓解这一问题，而少量后期注入的真实验证数据可以进一步提升性能上限。这项工作确立了奖励来源和数据规模对开放领域智能体学习的关键作用，并展示了多角色共同进化对于可扩展、自我提升智能体的有效性。本文的数据和代码发布在该链接：<https://yourlinkhere.com> 

---
# LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild 

**Title (ZH)**: LiveResearchBench: 一个面向用户的深度研究实时基准 

**Authors**: Jiayu Wang, Yifei Ming, Riya Dulepet, Qinglin Chen, Austin Xu, Zixuan Ke, Frederic Sala, Aws Albarghouthi, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.14240)  

**Abstract**: Deep research -- producing comprehensive, citation-grounded reports by searching and synthesizing information from hundreds of live web sources -- marks an important frontier for agentic systems. To rigorously evaluate this ability, four principles are essential: tasks should be (1) user-centric, reflecting realistic information needs, (2) dynamic, requiring up-to-date information beyond parametric knowledge, (3) unambiguous, ensuring consistent interpretation across users, and (4) multi-faceted and search-intensive, requiring search over numerous web sources and in-depth analysis. Existing benchmarks fall short of these principles, often focusing on narrow domains or posing ambiguous questions that hinder fair comparison. Guided by these principles, we introduce LiveResearchBench, a benchmark of 100 expert-curated tasks spanning daily life, enterprise, and academia, each requiring extensive, dynamic, real-time web search and synthesis. Built with over 1,500 hours of human labor, LiveResearchBench provides a rigorous basis for systematic evaluation. To evaluate citation-grounded long-form reports, we introduce DeepEval, a comprehensive suite covering both content- and report-level quality, including coverage, presentation, citation accuracy and association, consistency and depth of analysis. DeepEval integrates four complementary evaluation protocols, each designed to ensure stable assessment and high agreement with human judgments. Using LiveResearchBench and DeepEval, we conduct a comprehensive evaluation of 17 frontier deep research systems, including single-agent web search, single-agent deep research, and multi-agent systems. Our analysis reveals current strengths, recurring failure modes, and key system components needed to advance reliable, insightful deep research. 

**Abstract (ZH)**: 深度研究——通过搜索和综合来自数百个实时网络源的信息以生成全面、引文为基础的报告标志着代理系统的重大前沿。为了严谨地评估这一能力，四项原则是必不可少的：任务应（1）以用户为中心，反映现实的信息需求，（2）动态的，要求超出参数知识的最新信息，（3）明确的，确保用户之间的一致解释，以及（4）多维度和搜索密集型的，需要跨众多网络源进行搜索和深入分析。现有的基准测试未能满足这些原则，往往集中在狭窄的领域或提出了模糊的问题，这妨碍了公平的比较。根据这些原则，我们提出了LiveResearchBench，这是一个包含100项专家选择的任务基准，覆盖日常生活、企业与学术界，每项任务都需要广泛的动态实时网络搜索与综合。LiveResearchBench在超过1,500小时的人工努力下构建，为系统评估奠定了坚实的基础。为评估引文为基础的长篇报告，我们引入了DeepEval，这是一个全面的评估套件，涵盖了内容级和报告级质量，包括覆盖面、呈现方式、引文的准确性与关联性、一致性和分析的深度。DeepEval结合了四个互补的评估协议，每个协议都旨在确保稳定的评估和与人类判断的高一致性。利用LiveResearchBench和DeepEval，我们对17个前沿深度研究系统进行了全面评估，包括单代理网络搜索、单代理深度研究以及多代理系统。我们的分析揭示了当前的优势、反复出现的失败模式以及推进可靠而深刻的深度研究所需的关键系统组件。 

---
# Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks 

**Title (ZH)**: 代理中的人类恶意回声：多轮在线骚扰攻击评估基准 

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14207)  

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible. 

**Abstract (ZH)**: 大型语言模型（LLM）代理正越来越多地驱动交互式网络应用程序，但仍然容易被误用和造成伤害。先前的越狱研究主要集中在单轮提示上，而现实中的骚扰通常会在多轮互动中展开。在本研究中，我们提出了一个在线骚扰代理基准，包含：(i) 一个合成的多轮骚扰对话数据集，(ii) 受重复博弈理论指导的多代理（如骚扰者、受害者）模拟，(iii) 三种攻击代理的方法，涉及记忆、规划和微调，以及(iv) 一种混合方法评估框架。我们利用了两个突出的LLM，LLaMA-3.1-8B-Instruct（开源）和Gemini-2.0-flash（闭源）。结果显示，在LLama中，通过调优的攻击成功率从57.25%到64.19%提高到95.78%到96.89%，而在Gemini中，这一比例从98.46%提高到99.33%，同时拒绝率降至1%到2%。最常见的有毒行为是侮辱，调优后的比例为84.9%到87.8%，而非调优时为44.2%到50.8%，以及喷子言论，调优后的比例为81.2%到85.1%，而非调优时为31.5%到38.8%，表明与性骚扰或种族歧视等敏感类别相比，有更弱的防线。定性评估进一步发现，受攻击的代理再现了类似人类的攻击模式，在计划中有 Machiavellian/心理变态的表现，而在记忆中则表现出自恋倾向。令人意想不到的是，闭源和开源模型在各轮次中表现出不同的升级轨迹，闭源模型显示出显著的脆弱性。总体而言，我们的研究结果表明，基于多轮和理论指导的攻击不仅成功率高，而且模拟人类式的骚扰动态，促使开发更加稳健的安全防线，以最终确保在线平台的安全和负责任。 

---
# Implementation of AI in Precision Medicine 

**Title (ZH)**: AI在精准医学中的实施 

**Authors**: Göktuğ Bender, Samer Faraj, Anand Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.14194)  

**Abstract**: Artificial intelligence (AI) has become increasingly central to precision medicine by enabling the integration and interpretation of multimodal data, yet implementation in clinical settings remains limited. This paper provides a scoping review of literature from 2019-2024 on the implementation of AI in precision medicine, identifying key barriers and enablers across data quality, clinical reliability, workflow integration, and governance. Through an ecosystem-based framework, we highlight the interdependent relationships shaping real-world translation and propose future directions to support trustworthy and sustainable implementation. 

**Abstract (ZH)**: 人工智能在精准医疗中的实施：2019-2024年间文献综述及其关键障碍与促成因素分析 

---
# ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning 

**Title (ZH)**: ARM-FM：通过基础模型实现自动化的奖励机器，用于组合强化学习 

**Authors**: Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2510.14176)  

**Abstract**: Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization. 

**Abstract (ZH)**: Automated Reward Machines via Foundation Models：基于基础模型的自动化奖赏机器框架 

---
# JEDA: Query-Free Clinical Order Search from Ambient Dialogues 

**Title (ZH)**: JEDA：无需查询的临床订单搜索从环境对话中获取 

**Authors**: Praphul Singh, Corey Barrett, Sumana Srivasta, Amitabh Saikia, Irfan Bulu, Sri Gadde, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14169)  

**Abstract**: Clinical conversations mix explicit directives (order a chest X-ray) with implicit reasoning (the cough worsened overnight, we should check for pneumonia). Many systems rely on LLM rewriting, adding latency, instability, and opacity that hinder real-time ordering. We present JEDA (Joint Embedding for Direct and Ambient clinical orders), a domain-initialized bi-encoder that retrieves canonical orders directly and, in a query-free mode, encodes a short rolling window of ambient dialogue to trigger retrieval. Initialized from PubMedBERT and fine-tuned with a duplicate-safe contrastive objective, JEDA aligns heterogeneous expressions of intent to shared order concepts. Training uses constrained LLM guidance to tie each signed order to complementary formulations (command only, context only, command+context, context+reasoning), producing clearer inter-order separation, tighter query extendash order coupling, and stronger generalization. The query-free mode is noise-resilient, reducing sensitivity to disfluencies and ASR errors by conditioning on a short window rather than a single utterance. Deployed in practice, JEDA yields large gains and substantially outperforms its base encoder and recent open embedders (Linq Embed Mistral, SFR Embedding, GTE Qwen, BGE large, Embedding Gemma). The result is a fast, interpretable, LLM-free retrieval layer that links ambient context to actionable clinical orders in real time. 

**Abstract (ZH)**: JEDA：联合嵌入直接和环境临床订单 

---
# Combining Reinforcement Learning and Behavior Trees for NPCs in Video Games with AMD Schola 

**Title (ZH)**: 使用AMD Schola结合强化学习和行为树为视频游戏中的NPCs设计智能行为 

**Authors**: Tian Liu, Alex Cann, Ian Colbert, Mehdi Saeedi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14154)  

**Abstract**: While the rapid advancements in the reinforcement learning (RL) research community have been remarkable, the adoption in commercial video games remains slow. In this paper, we outline common challenges the Game AI community faces when using RL-driven NPCs in practice, and highlight the intersection of RL with traditional behavior trees (BTs) as a crucial juncture to be explored further. Although the BT+RL intersection has been suggested in several research papers, its adoption is rare. We demonstrate the viability of this approach using AMD Schola -- a plugin for training RL agents in Unreal Engine -- by creating multi-task NPCs in a complex 3D environment inspired by the commercial video game ``The Last of Us". We provide detailed methodologies for jointly training RL models with BTs while showcasing various skills. 

**Abstract (ZH)**: 尽管强化学习（RL）研究领域的快速发展令人瞩目，但在商用视频游戏中的应用依然缓慢。本文概述了游戏AI社区在实践中使用基于RL的NPC所面临的常见挑战，并强调RL与传统行为树（BT）的交集是一个亟待深入探索的关键领域。尽管已有数篇研究论文建议采用BT+RL的交集方法，但实际上的应用仍然很少。我们通过使用AMD Schola（Unreal Engine中的一个训练RL代理的插件），在受商业视频游戏《最后生还者》启发的复杂3D环境中创建多任务NPC，展示了这一方法的可行性，并详细介绍了在同时训练RL模型与BT时的各种方法和技术。 

---
# CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization 

**Title (ZH)**: CodeEvolve: 一个开源演化编码代理用于算法发现与优化 

**Authors**: Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14150)  

**Abstract**: In this work, we introduce CodeEvolve, an open-source evolutionary coding agent that unites Large Language Models (LLMs) with genetic algorithms to solve complex computational problems. Our framework adapts powerful evolutionary concepts to the LLM domain, building upon recent methods for generalized scientific discovery. CodeEvolve employs an island-based genetic algorithm to maintain population diversity and increase throughput, introduces a novel inspiration-based crossover mechanism that leverages the LLMs context window to combine features from successful solutions, and implements meta-prompting strategies for dynamic exploration of the solution space. We conduct a rigorous evaluation of CodeEvolve on a subset of the mathematical benchmarks used to evaluate Google DeepMind's closed-source AlphaEvolve. Our findings show that our method surpasses AlphaEvolve's performance on several challenging problems. To foster collaboration and accelerate progress, we release our complete framework as an open-source repository. 

**Abstract (ZH)**: 本研究引入了CodeEvolve，一个开源的进化编码代理，将大型语言模型与遗传算法结合，以解决复杂的计算问题。我们的框架将强大的进化概念适应大型语言模型领域，基于近期通用科学发现方法。CodeEvolve 使用基于岛屿的遗传算法保持种群多样性并提高 throughput，引入了一种基于启发式的交叉机制，利用大型语言模型的上下文窗口来结合成功解决方案的特征，并实施了元提示策略以动态探索解空间。我们对用于评估谷歌DeepMind封闭源代码AlphaEvolve的部分数学基准进行了严格的评估。我们的研究结果表明，我们的方法在多个具有挑战性的问题上超越了AlphaEvolve的性能。为了促进合作并加快进度，我们将完整的框架作为开源仓库发布。 

---
# A Multimodal Approach to Heritage Preservation in the Context of Climate Change 

**Title (ZH)**: 气候变化背景下多模态文化遗产保护方法 

**Authors**: David Roqui, Adèle Cormier, nistor Grozavu, Ann Bourges  

**Link**: [PDF](https://arxiv.org/pdf/2510.14136)  

**Abstract**: Cultural heritage sites face accelerating degradation due to climate change, yet tradi- tional monitoring relies on unimodal analysis (visual inspection or environmental sen- sors alone) that fails to capture the complex interplay between environmental stres- sors and material deterioration. We propose a lightweight multimodal architecture that fuses sensor data (temperature, humidity) with visual imagery to predict degradation severity at heritage sites. Our approach adapts PerceiverIO with two key innovations: (1) simplified encoders (64D latent space) that prevent overfitting on small datasets (n=37 training samples), and (2) Adaptive Barlow Twins loss that encourages modality complementarity rather than redundancy. On data from Strasbourg Cathedral, our model achieves 76.9% accu- racy, a 43% improvement over standard multimodal architectures (VisualBERT, Trans- former) and 25% over vanilla PerceiverIO. Ablation studies reveal that sensor-only achieves 61.5% while image-only reaches 46.2%, confirming successful multimodal synergy. A systematic hyperparameter study identifies an optimal moderate correlation target ({\tau} =0.3) that balances align- ment and complementarity, achieving 69.2% accuracy compared to other {\tau} values ({\tau} =0.1/0.5/0.7: 53.8%, {\tau} =0.9: 61.5%). This work demonstrates that architectural sim- plicity combined with contrastive regularization enables effective multimodal learning in data-scarce heritage monitoring contexts, providing a foundation for AI-driven con- servation decision support systems. 

**Abstract (ZH)**: 文化遗址由于气候变化加速退化，传统的单模态监测（仅视觉检查或环境传感器）未能捕捉到环境应力与材料退化之间的复杂相互作用。我们提出了一种轻量级的多模态架构，将传感器数据（温度、湿度）与视觉图像融合，以预测文化遗址的退化程度。我们的方法通过两种关键创新适应PerceiverIO：（1）简化编码器（64D潜在空间），防止在小数据集（n=37训练样本）上过拟合；（2）自适应Barlow Twins损失，鼓励模态互补而非冗余。在斯特拉斯堡大教堂的数据上，我们的模型准确率为76.9%，分别比标准多模态架构（VisualBERT、Transformer）和vanilla PerceiverIO高出43%和25%。消融研究表明，仅传感器准确率为61.5%，仅图像准确率为46.2%，证实了多模态协同的成功。系统性的超参数研究发现，最佳适度相关目标（τ=0.3）在对齐和互补之间取得平衡，准确率为69.2%，而其他τ值（τ=0.1/0.5/0.7：53.8%，τ=0.9：61.5%）的准确率较低。这项工作证明，在数据稀缺的文化遗址监测环境中，结合架构简化和对比正则化可以使多模态学习有效，为基于AI的保护决策支持系统奠定了基础。 

---
# Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems 

**Title (ZH)**: 规范化的有agency的AI系统的安全、安全性和功能属性的形式化描述 

**Authors**: Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2510.14133)  

**Abstract**: Agentic AI systems, which leverage multiple autonomous agents and Large Language Models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two foundational models. The first, the host agent model, formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools. The second, the task lifecycle model, details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 17 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the systematic analysis, design, and deployment of correct, reliable, and robust agentic AI systems. 

**Abstract (ZH)**: 利用多个自主代理和大型语言模型的代理型AI系统 increasingly used to address complex, multi-step tasks 

---
# STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management 

**Title (ZH)**: STEMS：基于时空增强的安全多Agent协同管理建筑能效 

**Authors**: Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2510.14112)  

**Abstract**: Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types. 

**Abstract (ZH)**: 基于空间-时间增强的安全多代理协调（STEMS）在多建筑能效管理中的应用 

---
# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs 

**Title (ZH)**: 基于代币级别MDP的社会选择生成公平共识声明 

**Authors**: Carter Blair, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14106)  

**Abstract**: Current frameworks for consensus statement generation with large language models lack the inherent structure needed to provide provable fairness guarantees when aggregating diverse free-form opinions. We model the task as a multi-objective, token-level Markov Decision Process (MDP), where each objective corresponds to an agent's preference. Token-level rewards for each agent are derived from their policy (e.g., a personalized language model). This approach utilizes the finding that such policies implicitly define optimal Q-functions, providing a principled way to quantify rewards at each generation step without a value function (Rafailov et al., 2024). This MDP formulation creates a formal structure amenable to analysis using principles from social choice theory. We propose two approaches grounded in social choice theory. First, we propose a stochastic generation policy guaranteed to be in the ex-ante core, extending core stability concepts from voting theory to text generation. This policy is derived from an underlying distribution over complete statements that maximizes proportional fairness (Nash Welfare). Second, for generating a single statement, we target the maximization of egalitarian welfare using search algorithms within the MDP framework. Empirically, experiments using language models to instantiate agent policies show that search guided by the egalitarian objective generates consensus statements with improved worst-case agent alignment compared to baseline methods, including the Habermas Machine (Tessler et al., 2024). 

**Abstract (ZH)**: 基于大规模语言模型的共识声明生成框架缺乏聚合多样化自由形式意见时提供可证明公平性保证的固有结构。我们将该任务建模为一个多目标、token级马尔可夫决策过程（MDP），其中每个目标对应于代理的偏好。每个代理的token级奖励源自其策略（例如，个性化语言模型）。该方法利用了这样的发现：这些策略隐含地定义了最优Q函数，提供了一种在每个生成步骤中量化奖励的规范方式，无需价值函数（Rafailov等，2024）。这种MDP形式通过社会选择理论中的原理创建了可进行正式分析的结构。我们提出了两种基于社会选择理论的方法。首先，我们提出了一种随机生成策略，保证其在事前核心中，将投票理论中的核心稳定性概念扩展到文本生成中。该策略源自最大化比例公平（纳什福利）的完整声明的底层分布。其次，在生成单个声明时，我们利用MDP框架中的搜索算法最大化激进福利。实验结果显示，基于激进目标的搜索生成的共识声明在最坏情况下的代理对齐优于基线方法，包括Habermas Machine（Tessler等，2024）。 

---
# Position: Require Frontier AI Labs To Release Small "Analog" Models 

**Title (ZH)**: 要求前沿AI实验室发布小型“模拟”模型 

**Authors**: Shriyash Upadhyay, Chaithanya Bandi, Narmeen Oozeer, Philip Quirke  

**Link**: [PDF](https://arxiv.org/pdf/2510.14053)  

**Abstract**: Recent proposals for regulating frontier AI models have sparked concerns about the cost of safety regulation, and most such regulations have been shelved due to the safety-innovation tradeoff. This paper argues for an alternative regulatory approach that ensures AI safety while actively promoting innovation: mandating that large AI laboratories release small, openly accessible analog models (scaled-down versions) trained similarly to and distilled from their largest proprietary models.
Analog models serve as public proxies, allowing broad participation in safety verification, interpretability research, and algorithmic transparency without forcing labs to disclose their full-scale models. Recent research demonstrates that safety and interpretability methods developed using these smaller models generalize effectively to frontier-scale systems. By enabling the wider research community to directly investigate and innovate upon accessible analogs, our policy substantially reduces the regulatory burden and accelerates safety advancements.
This mandate promises minimal additional costs, leveraging reusable resources like data and infrastructure, while significantly contributing to the public good. Our hope is not only that this policy be adopted, but that it illustrates a broader principle supporting fundamental research in machine learning: deeper understanding of models relaxes the safety-innovation tradeoff and lets us have more of both. 

**Abstract (ZH)**: 近期关于调节前沿人工智能模型的提议引发了对安全监管成本的担忧，且大多数此类监管措施因安全与创新权衡而被搁置。本文建议一种替代监管方法，既能确保AI安全又能积极促进创新：要求大型AI实验室发布小型、开源的模拟模型（缩小版），这些模型的训练方式类似且源自其最大的专有模型。 

---
# GammaZero: Learning To Guide POMDP Belief Space Search With Graph Representations 

**Title (ZH)**: GammaZero：学习引导POMDP信念空间搜索的图表示方法 

**Authors**: Rajesh Mangannavar, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.14035)  

**Abstract**: We introduce an action-centric graph representation framework for learning to guide planning in Partially Observable Markov Decision Processes (POMDPs). Unlike existing approaches that require domain-specific neural architectures and struggle with scalability, GammaZero leverages a unified graph-based belief representation that enables generalization across problem sizes within a domain. Our key insight is that belief states can be systematically transformed into action-centric graphs where structural patterns learned on small problems transfer to larger instances. We employ a graph neural network with a decoder architecture to learn value functions and policies from expert demonstrations on computationally tractable problems, then apply these learned heuristics to guide Monte Carlo tree search on larger problems. Experimental results on standard POMDP benchmarks demonstrate that GammaZero achieves comparable performance to BetaZero when trained and tested on the same-sized problems, while uniquely enabling zero-shot generalization to problems 2-4 times larger than those seen during training, maintaining solution quality with reduced search requirements. 

**Abstract (ZH)**: 一种用于部分可观测马尔可夫决策过程规划引导的学习动作中心图表示框架 

---
# Do Large Language Models Show Biases in Causal Learning? Insights from Contingency Judgment 

**Title (ZH)**: 大型语言模型在因果学习中表现出偏见吗？基于 contingency 判断的视角 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Giovanni Franco Gabriel Marraffini, Mario Alejandro Leiva, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13985)  

**Abstract**: Causal learning is the cognitive process of developing the capability of making causal inferences based on available information, often guided by normative principles. This process is prone to errors and biases, such as the illusion of causality, in which people perceive a causal relationship between two variables despite lacking supporting evidence. This cognitive bias has been proposed to underlie many societal problems, including social prejudice, stereotype formation, misinformation, and superstitious thinking. In this work, we examine whether large language models are prone to developing causal illusions when faced with a classic cognitive science paradigm: the contingency judgment task. To investigate this, we constructed a dataset of 1,000 null contingency scenarios (in which the available information is not sufficient to establish a causal relationship between variables) within medical contexts and prompted LLMs to evaluate the effectiveness of potential causes. Our findings show that all evaluated models systematically inferred unwarranted causal relationships, revealing a strong susceptibility to the illusion of causality. While there is ongoing debate about whether LLMs genuinely understand causality or merely reproduce causal language without true comprehension, our findings support the latter hypothesis and raise concerns about the use of language models in domains where accurate causal reasoning is essential for informed decision-making. 

**Abstract (ZH)**: 因果学习是个体基于可用信息发展出基于规范原则进行因果推断的能力的认知过程，常易受到如因果错觉等错误和偏见的影响。这种认知偏见被认为可解释许多社会问题，包括社会偏见、刻板印象形成、信息误导和迷信思维。在本研究中，我们探究了大型语言模型在面对经典认知科学范式——关联判断任务时是否会发展出因果错觉。为此，我们构建了一个包含1000个医疗情境下的零关联场景的数据集，并促使语言模型评估潜在原因的有效性。研究发现，所有评估的模型都系统地推断出不合理的因果关系，显示出强烈地受到因果错觉的影响。虽然关于语言模型是否真正理解因果关系还存在争议，认为它们只是重现因果语言而无真实理解，但我们的发现支持了这一观点，并对在需要准确因果推理以支持知情决策的领域中使用语言模型提出了担忧。 

---
# Do Slides Help? Multi-modal Context for Automatic Transcription of Conference Talks 

**Title (ZH)**: 幻灯片有帮助吗？多模态上下文下的会议演讲自动转录 

**Authors**: Supriti Sinhamahapatra, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.13979)  

**Abstract**: State-of-the-art (SOTA) Automatic Speech Recognition (ASR) systems primarily rely on acoustic information while disregarding additional multi-modal context. However, visual information are essential in disambiguation and adaptation. While most work focus on speaker images to handle noise conditions, this work also focuses on integrating presentation slides for the use cases of scientific presentation.
In a first step, we create a benchmark for multi-modal presentation including an automatic analysis of transcribing domain-specific terminology. Next, we explore methods for augmenting speech models with multi-modal information. We mitigate the lack of datasets with accompanying slides by a suitable approach of data augmentation. Finally, we train a model using the augmented dataset, resulting in a relative reduction in word error rate of approximately 34%, across all words and 35%, for domain-specific terms compared to the baseline model. 

**Abstract (ZH)**: 当前最先进的自动语音识别（ASR）系统主要依赖声学信息，忽视了额外的多模态上下文。然而，视觉信息在消歧和适应中是必不可少的。尽管大多数研究侧重于利用演讲者图像处理噪声条件，本研究还专注于集成演示幻灯片以适用于科学演示场景。首先，我们创建了一个多模态演示的基准，包括对领域特定术语自动转录的分析。接着，我们探索了将多模态信息集成到语音模型中的方法。通过适当的数据增强方法缓解了缺乏配有幻灯片的数据集问题。最后，我们使用增强后的数据集训练模型，结果表明，相较于基线模型，整体单词错误率降低了约34%，领域特定术语单词错误率降低了约35%。 

---
# Decision Oriented Technique (DOTechnique): Finding Model Validity Through Decision-Maker Context 

**Title (ZH)**: 决策导向技术（DOT技术）：通过决策者背景寻找模型有效性 

**Authors**: Raheleh Biglari, Joachim Denil  

**Link**: [PDF](https://arxiv.org/pdf/2510.13858)  

**Abstract**: Model validity is as critical as the model itself, especially when guiding decision-making processes. Traditional approaches often rely on predefined validity frames, which may not always be available or sufficient. This paper introduces the Decision Oriented Technique (DOTechnique), a novel method for determining model validity based on decision consistency rather than output similarity. By evaluating whether surrogate models lead to equivalent decisions compared to high-fidelity models, DOTechnique enables efficient identification of validity regions, even in the absence of explicit validity boundaries. The approach integrates domain constraints and symbolic reasoning to narrow the search space, enhancing computational efficiency. A highway lane change system serves as a motivating example, demonstrating how DOTechnique can uncover the validity region of a simulation model. The results highlight the potential of the technique to support finding model validity through decision-maker context. 

**Abstract (ZH)**: 模型的有效性与模型本身一样至关重要，尤其是在指导决策过程时。传统方法往往依赖于预定义的有效性框架，但这些框架可能并不总是可用或足够的。本文引入了旨在基于决策一致性而非输出相似性来确定模型有效性的决策导向技术（DOT技术）。通过评估代理模型的决策是否与高保真模型相当，DOT技术能够在没有明确有效边界的情况下有效地识别有效性区域。该方法结合领域约束和符号推理来缩小搜索空间，增强计算效率。高速公路变道系统作为一个示例，展示了如何使用DOT技术发现仿真模型的有效性区域。结果突显了该技术在通过决策者背景支持发现模型有效性方面的潜力。 

---
# Coupled Diffusion Sampling for Training-Free Multi-View Image Editing 

**Title (ZH)**: 耦合扩散采样：无需训练的多视图图像编辑 

**Authors**: Hadi Alzayer, Yunzhi Zhang, Chen Geng, Jia-Bin Huang, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14981)  

**Abstract**: We present an inference-time diffusion sampling method to perform multi-view consistent image editing using pre-trained 2D image editing models. These models can independently produce high-quality edits for each image in a set of multi-view images of a 3D scene or object, but they do not maintain consistency across views. Existing approaches typically address this by optimizing over explicit 3D representations, but they suffer from a lengthy optimization process and instability under sparse view settings. We propose an implicit 3D regularization approach by constraining the generated 2D image sequences to adhere to a pre-trained multi-view image distribution. This is achieved through coupled diffusion sampling, a simple diffusion sampling technique that concurrently samples two trajectories from both a multi-view image distribution and a 2D edited image distribution, using a coupling term to enforce the multi-view consistency among the generated images. We validate the effectiveness and generality of this framework on three distinct multi-view image editing tasks, demonstrating its applicability across various model architectures and highlighting its potential as a general solution for multi-view consistent editing. 

**Abstract (ZH)**: 我们提出了一种推理时的扩散采样方法，使用预训练的2D图像编辑模型在多视角图像中进行一致的图像编辑。 

---
# From Pixels to Words -- Towards Native Vision-Language Primitives at Scale 

**Title (ZH)**: 从像素到文字——迈向大规模的本源多模态感知语言基础单元 

**Authors**: Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14979)  

**Abstract**: The edifice of native Vision-Language Models (VLMs) has emerged as a rising contender to typical modular VLMs, shaped by evolving model architectures and training paradigms. Yet, two lingering clouds cast shadows over its widespread exploration and promotion: (-) What fundamental constraints set native VLMs apart from modular ones, and to what extent can these barriers be overcome? (-) How to make research in native VLMs more accessible and democratized, thereby accelerating progress in the field. In this paper, we clarify these challenges and outline guiding principles for constructing native VLMs. Specifically, one native VLM primitive should: (i) effectively align pixel and word representations within a shared semantic space; (ii) seamlessly integrate the strengths of formerly separate vision and language modules; (iii) inherently embody various cross-modal properties that support unified vision-language encoding, aligning, and reasoning. Hence, we launch NEO, a novel family of native VLMs built from first principles, capable of rivaling top-tier modular counterparts across diverse real-world scenarios. With only 390M image-text examples, NEO efficiently develops visual perception from scratch while mitigating vision-language conflicts inside a dense and monolithic model crafted from our elaborate primitives. We position NEO as a cornerstone for scalable and powerful native VLMs, paired with a rich set of reusable components that foster a cost-effective and extensible ecosystem. Our code and models are publicly available at: this https URL. 

**Abstract (ZH)**: 本土视觉-语言模型的基石：从基本原则构建本土视觉-语言模型及NEO家族的提出 

---
# Terra: Explorable Native 3D World Model with Point Latents 

**Title (ZH)**: Terra：可探索的原生3D世界模型与点潜在表示 

**Authors**: Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Xin Tao, Pengfei Wan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14977)  

**Abstract**: World models have garnered increasing attention for comprehensive modeling of the real world. However, most existing methods still rely on pixel-aligned representations as the basis for world evolution, neglecting the inherent 3D nature of the physical world. This could undermine the 3D consistency and diminish the modeling efficiency of world models. In this paper, we present Terra, a native 3D world model that represents and generates explorable environments in an intrinsic 3D latent space. Specifically, we propose a novel point-to-Gaussian variational autoencoder (P2G-VAE) that encodes 3D inputs into a latent point representation, which is subsequently decoded as 3D Gaussian primitives to jointly model geometry and appearance. We then introduce a sparse point flow matching network (SPFlow) for generating the latent point representation, which simultaneously denoises the positions and features of the point latents. Our Terra enables exact multi-view consistency with native 3D representation and architecture, and supports flexible rendering from any viewpoint with only a single generation process. Furthermore, Terra achieves explorable world modeling through progressive generation in the point latent space. We conduct extensive experiments on the challenging indoor scenes from ScanNet v2. Terra achieves state-of-the-art performance in both reconstruction and generation with high 3D consistency. 

**Abstract (ZH)**: 一种原生3D世界模型：Terra 

---
# WithAnyone: Towards Controllable and ID Consistent Image Generation 

**Title (ZH)**: WithAnyone: 向可控且ID一致的图像生成努力 

**Authors**: Hengyuan Xu, Wei Cheng, Peng Xing, Yixiao Fang, Shuhan Wu, Rui Wang, Xianfang Zeng, Daxin Jiang, Gang Yu, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14975)  

**Abstract**: Identity-consistent generation has become an important focus in text-to-image research, with recent models achieving notable success in producing images aligned with a reference identity. Yet, the scarcity of large-scale paired datasets containing multiple images of the same individual forces most approaches to adopt reconstruction-based training. This reliance often leads to a failure mode we term copy-paste, where the model directly replicates the reference face rather than preserving identity across natural variations in pose, expression, or lighting. Such over-similarity undermines controllability and limits the expressive power of generation. To address these limitations, we (1) construct a large-scale paired dataset MultiID-2M, tailored for multi-person scenarios, providing diverse references for each identity; (2) introduce a benchmark that quantifies both copy-paste artifacts and the trade-off between identity fidelity and variation; and (3) propose a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity. These contributions culminate in WithAnyone, a diffusion-based model that effectively mitigates copy-paste while preserving high identity similarity. Extensive qualitative and quantitative experiments demonstrate that WithAnyone significantly reduces copy-paste artifacts, improves controllability over pose and expression, and maintains strong perceptual quality. User studies further validate that our method achieves high identity fidelity while enabling expressive controllable generation. 

**Abstract (ZH)**: 身份一致生成已成为文本到图像研究的重要关注点，近期的模型在生成与参考身份一致的图像方面取得了显著成果。然而，缺乏包含同一个体多张图片的大规模配对数据集促使大多数方法采用基于重构的训练。这种依赖往往导致我们称之为复制粘贴的失败模式，即模型直接复制参考面部而不是在姿态、表情或光照的自然变化中保持身份一致性。这种过度相似性削弱了可控性并限制了生成的表达力。为了应对这些局限性，我们（1）构建了一个适用于多人大场景的大型配对数据集MultiID-2M，为每个身份提供多样化的参考；（2）引入了一个基准，量化复制粘贴的伪影以及身份忠实度与多样性之间的权衡；（3）提出了一种新的训练范式，利用配对数据引入对比身份损失，平衡忠实度与多样性。这些贡献导致了WithAnyone模型的提出，该模型基于扩散模型，能够有效缓解复制粘贴问题，同时保持高身份相似性。大量定性和定量实验表明，WithAnyone显著减少了复制粘贴伪影，提高了对姿态和表情的可控性，并保持了强烈的感知质量。用户研究进一步验证了我们的方法在保持高身份忠实度的同时实现了表达性可控生成。 

---
# pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation 

**Title (ZH)**: pi-Flow: 基于策略的 Few-Step 生成通过模仿提炼 

**Authors**: Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, Sai Bi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14974)  

**Abstract**: Few-step diffusion or flow-based generative models typically distill a velocity-predicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality-diversity trade-off. To address this, we propose policy-based flow models ($\pi$-Flow). $\pi$-Flow modifies the output layer of a student flow model to predict a network-free policy at one timestep. The policy then produces dynamic flow velocities at future substeps with negligible overhead, enabling fast and accurate ODE integration on these substeps without extra network evaluations. To match the policy's ODE trajectory to the teacher's, we introduce a novel imitation distillation approach, which matches the policy's velocity to the teacher's along the policy's trajectory using a standard $\ell_2$ flow matching loss. By simply mimicking the teacher's behavior, $\pi$-Flow enables stable and scalable training and avoids the quality-diversity trade-off. On ImageNet 256$^2$, it attains a 1-NFE FID of 2.85, outperforming MeanFlow of the same DiT architecture. On FLUX.1-12B and Qwen-Image-20B at 4 NFEs, $\pi$-Flow achieves substantially better diversity than state-of-the-art few-step methods, while maintaining teacher-level quality. 

**Abstract (ZH)**: 基于策略的流模型（$\pi$-Flow） 

---
# Attention Is All You Need for KV Cache in Diffusion LLMs 

**Title (ZH)**: Attention Is All You Need for KV Cache in Diffusion LLMs（扩散型LLM中键值缓存的注意力机制） 

**Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14973)  

**Abstract**: This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs. 

**Abstract (ZH)**: 这项工作研究了如何自适应地 recompute 关键值（KV）缓存以最大化预测准确性并最小化解码延迟，针对扩散大型语言模型（DLMs）。先前方法在每一步和每一层都会重新计算所有标记的 QKV，尽管大多数步骤中 KV 状态变化很小，特别是在浅层中，导致大量冗余。我们做出了以下观察：（1）距离较长的 ${\bf MASK}$ 标记主要作为长度偏差起作用，并且可以在活跃预测窗口之外块状缓存；（2）KV 动态随着深度增加而增加，表明从较深层开始的选择性刷新足以；（3）最受关注的标记展示了最小的 KV 游移，为其他标记的缓存变化提供了保守的下限。基于这些观察，我们提出了一种无需训练、架构无关的方法 ${\bf Elastic-Cache}$，该方法联合决定了何时（通过最受关注标记的注意意识游移测试）以及何地（通过深度意识调度，从选定层开始重新计算，同时重用浅层缓存和窗口外的 ${\bf MASK}$ 缓存）进行刷新。与固定周期方案不同，${\bf Elastic-Cache}$ 对扩散LLMs执行自适应、分层感知的缓存更新，减少了冗余计算并加速了解码，同时在生成质量几乎没有损失的情况下。在LLaDA-Instruct、LLaDA-1.5和LLaDA-V上的数学推理和代码生成任务中的实验一致地显示加速效果：在GSM8K（256标记）上加速了$8.7\times$，在较长序列上加速了$45.1\times$，在HumanEval上加速了$4.8\times$，同时始终维持了比基线更高的准确性。我们的方法在GSM8K上实现了显著更高的吞吐量（$6.8\times$）并保持了生成质量，从而使得扩散LLMs的实际部署成为可能。 

---
# TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar 

**Title (ZH)**: TokDrift: 当LLM在亚词中说话而代码在语法学中说话时 

**Authors**: Yinxi Li, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.14972)  

**Abstract**: Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs. 

**Abstract (ZH)**: Large语言模型（LLMs）对于代码的处理依赖于子词分词器，如基于字节对编码（BPE）的分词，这种分词器是从混合自然语言文本和编程语言代码中学习来的，但更多地依赖统计学而不是语法。因此，语义相同的代码片段根据表面因素（如空格或标识符命名）可能会被分词器分词为不同的方式。为了衡量这种不一致的影响，我们引入了TokDrift框架，该框架通过应用保持语义的重写规则来创建仅在分词方面不同的代码变体。在包括超过300亿参数的大型代码LLM在内的九个代码LLM中，即使是细微的格式变化也会导致模型行为出现显著变化。逐层分析表明，问题源于早期嵌入，其中子词分割无法捕获语法标记边界。我们的研究结果指出，不一致的分词已成为可靠代码理解和生成的隐藏障碍，并强调了未来代码LLM中需要语法感知的分词的重要性。 

---
# LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training 

**Title (ZH)**: LLMs作为可扩展的通用模拟器，用于数字代理培训的进化 

**Authors**: Yiming Wang, Da Yin, Yuedong Cui, Ruichen Zheng, Zhiqian Li, Zongyu Lin, Di Wu, Xueqing Wu, Chenchen Ye, Yu Zhou, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14969)  

**Abstract**: Digital agents require diverse, large-scale UI trajectories to generalize across real-world tasks, yet collecting such data is prohibitively expensive in both human annotation, infra and engineering perspectives. To this end, we introduce $\textbf{UI-Simulator}$, a scalable paradigm that generates structured UI states and transitions to synthesize training trajectories at scale. Our paradigm integrates a digital world simulator for diverse UI states, a guided rollout process for coherent exploration, and a trajectory wrapper that produces high-quality and diverse trajectories for agent training. We further propose $\textbf{UI-Simulator-Grow}$, a targeted scaling strategy that enables more rapid and data-efficient scaling by prioritizing high-impact tasks and synthesizes informative trajectory variants. Experiments on WebArena and AndroidWorld show that UI-Simulator rivals or surpasses open-source agents trained on real UIs with significantly better robustness, despite using weaker teacher models. Moreover, UI-Simulator-Grow matches the performance of Llama-3-70B-Instruct using only Llama-3-8B-Instruct as the base model, highlighting the potential of targeted synthesis scaling paradigm to continuously and efficiently enhance the digital agents. 

**Abstract (ZH)**: Digital代理需要大规模多样的UI轨迹来跨实际任务进行泛化，但由于在人类注释、基础设施和工程方面的成本高昂，收集此类数据是不可能的。为了解决这一问题，我们引入了UI-Simulator，一种可扩展的框架，用于生成结构化的UI状态和转换以大规模合成训练轨迹。该框架结合了数字世界模拟器以生成多样的UI状态、指导性展开过程以实现连贯探索，以及轨迹包装器以生成高质量和多样化的轨迹用于代理训练。我们还提出了UI-Simulator-Grow，这是一种有针对性的扩展策略，通过优先处理高影响任务并合成信息丰富的轨迹变体，实现更快更数据高效地扩展。实验结果显示，UI-Simulator在WebArena和AndroidWorld上的表现与使用真实UI训练的开源代理相当或更好，尽管使用了较弱的教师模型。此外，UI-Simulator-Grow仅使用Llama-3-8B-Instruct作为基础模型就达到了Llama-3-70B-Instruct的性能，突显了有针对性的合成扩展框架在持续和高效增强数字代理方面的潜力。 

---
# RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks 

**Title (ZH)**: 基于检索的演示分解器：长期任务计划者对齐的检索式示例分解方法 

**Authors**: Mingxuan Yan, Yuping Wang, Zechun Liu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14968)  

**Abstract**: To tackle long-horizon tasks, recent hierarchical vision-language-action (VLAs) frameworks employ vision-language model (VLM)-based planners to decompose complex manipulation tasks into simpler sub-tasks that low-level visuomotor policies can easily handle. Typically, the VLM planner is finetuned to learn to decompose a target task. This finetuning requires target task demonstrations segmented into sub-tasks by either human annotation or heuristic rules. However, the heuristic subtasks can deviate significantly from the training data of the visuomotor policy, which degrades task performance. To address these issues, we propose a Retrieval-based Demonstration Decomposer (RDD) that automatically decomposes demonstrations into sub-tasks by aligning the visual features of the decomposed sub-task intervals with those from the training data of the low-level visuomotor policies. Our method outperforms the state-of-the-art sub-task decomposer on both simulation and real-world tasks, demonstrating robustness across diverse settings. Code and more results are available at this http URL. 

**Abstract (ZH)**: 基于检索的演示分解器 (RDD)：自动将演示分解为子任务以应对长期任务 

---
# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents 

**Title (ZH)**: 基于信息增益的策略优化：一种简单而有效的多轮LLM代理方法 

**Authors**: Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying  

**Link**: [PDF](https://arxiv.org/pdf/2510.14967)  

**Abstract**: Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的智能体通过强化学习（RL）训练，以增强其通过工具使用与外部环境交互的能力，特别是在需要多轮推理和知识获取的搜索型场景中。现有方法通常依赖于仅在最终答案时提供的基于结果的奖励，这种稀疏奖励在多轮场景中尤为成问题，长时间轨迹加剧了两个关键问题：（i）优势崩溃，即所有模拟过程收到相同的奖励，无法提供有用的学习信号；（ii）缺乏精细的信用分配，尤其是在长期任务中，轮次之间的依赖关系变得模糊。本文提出了一种简单的高效强化学习框架——信息增益基于策略优化（IGPO），以密集且内在的方式监督多轮智能体的训练。IGPO将每轮交互视为获取地面真值信息的渐进过程，并定义轮次奖励为策略产生正确答案概率的边际增加。与依赖外部奖励模型或成本昂贵的蒙特卡洛估计的先前提取过程奖励的方法不同，IGPO直接从模型自身的信念更新中推导出内在奖励。这些内在的轮次奖励与结果级别的监督结合，形成密集的奖励轨迹。在领域内和领域外基准测试上的广泛实验表明，IGPO在多轮场景中始终优于强基线，实现了更高的准确性并提高了样本效率。 

---
# C4D: 4D Made from 3D through Dual Correspondences 

**Title (ZH)**: C4D：通过双对应关系从3D生成4D 

**Authors**: Shizun Wang, Zhenxiang Jiang, Xingyi Yang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14960)  

**Abstract**: Recovering 4D from monocular video, which jointly estimates dynamic geometry and camera poses, is an inevitably challenging problem. While recent pointmap-based 3D reconstruction methods (e.g., DUSt3R) have made great progress in reconstructing static scenes, directly applying them to dynamic scenes leads to inaccurate results. This discrepancy arises because moving objects violate multi-view geometric constraints, disrupting the reconstruction. To address this, we introduce C4D, a framework that leverages temporal Correspondences to extend existing 3D reconstruction formulation to 4D. Specifically, apart from predicting pointmaps, C4D captures two types of correspondences: short-term optical flow and long-term point tracking. We train a dynamic-aware point tracker that provides additional mobility information, facilitating the estimation of motion masks to separate moving elements from the static background, thus offering more reliable guidance for dynamic scenes. Furthermore, we introduce a set of dynamic scene optimization objectives to recover per-frame 3D geometry and camera parameters. Simultaneously, the correspondences lift 2D trajectories into smooth 3D trajectories, enabling fully integrated 4D reconstruction. Experiments show that our framework achieves complete 4D recovery and demonstrates strong performance across multiple downstream tasks, including depth estimation, camera pose estimation, and point tracking. Project Page: this https URL 

**Abstract (ZH)**: 从单目视频恢复4D：一种联合估计动态几何和相机姿态的框架，是不可避免的挑战性问题。虽然最近基于点图的3D重建方法（例如DUSt3R）在重建静态场景方面取得了巨大进步，但直接将其应用于动态场景会导致不准确的结果。这种差异源于移动物体违反了多视图几何约束，干扰了重建。为解决这一问题，我们提出C4D框架，其利用时间对应关系将现有的3D重建公式扩展到4D。具体而言，C4D不仅预测点图，还捕获两种类型的对应关系：短期光学流和长期点跟踪。我们训练了一个动态感知的点跟踪器，提供额外的移动性信息，促进运动掩码的估计，以分离移动元素和静态背景，从而为动态场景提供更多可靠的指导。此外，我们引入了一组动态场景优化目标，以恢复每帧的3D几何和相机参数。同时，对应关系将2D轨迹提升为平滑的3D轨迹，实现全面集成的4D重建。实验结果显示，我们的框架实现了完整的4D恢复，并在多个下游任务中表现出强大的性能，包括深度估计、相机姿态估计和点跟踪。项目页面：this https URL 

---
# CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions 

**Title (ZH)**: 基于控制屏障函数的训练安全性滤波强化学习(CBF-RL) 

**Authors**: Lizhi Yang, Blake Werner, Massimiliano de Sa Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14959)  

**Abstract**: Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed \emph{online} via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs \emph{in training}. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter. 

**Abstract (ZH)**: 基于控制障碍函数的强化学习：通过在训练中强制执行控制障碍函数生成安全行为 

---
# RealDPO: Real or Not Real, that is the Preference 

**Title (ZH)**: RealDPO: 实或虚，偏好决定 

**Authors**: Guo Cheng, Danni Yang, Ziqi Huang, Jianlou Si, Chenyang Si, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14955)  

**Abstract**: Video generative models have recently achieved notable advancements in synthesis quality. However, generating complex motions remains a critical challenge, as existing models often struggle to produce natural, smooth, and contextually consistent movements. This gap between generated and real-world motions limits their practical applicability. To address this issue, we introduce RealDPO, a novel alignment paradigm that leverages real-world data as positive samples for preference learning, enabling more accurate motion synthesis. Unlike traditional supervised fine-tuning (SFT), which offers limited corrective feedback, RealDPO employs Direct Preference Optimization (DPO) with a tailored loss function to enhance motion realism. By contrasting real-world videos with erroneous model outputs, RealDPO enables iterative self-correction, progressively refining motion quality. To support post-training in complex motion synthesis, we propose RealAction-5K, a curated dataset of high-quality videos capturing human daily activities with rich and precise motion details. Extensive experiments demonstrate that RealDPO significantly improves video quality, text alignment, and motion realism compared to state-of-the-art models and existing preference optimization techniques. 

**Abstract (ZH)**: 视频生成模型最近在合成质量方面取得了显著进步。然而，生成复杂的运动仍然是一个关键挑战，因为现有的模型往往难以产生自然、流畅且上下文一致的运动。生成的运动与真实世界运动之间的差距限制了其实际应用。为了解决这一问题，我们引入了RealDPO，这是一种新颖的对齐范式，通过使用真实世界数据作为偏好学习的正样本，以实现更准确的运动合成。与传统的监督微调（SFT）相比，后者提供的纠正反馈有限，RealDPO利用直接偏好优化（DPO）和定制的损失函数来增强运动的真实感。通过将真实世界视频与模型的错误输出进行对比，RealDPO能够实现迭代自我纠正，逐步提高运动质量。为了支持复杂运动合成的后训练，我们提出了RealAction-5K数据集，这是一个精心编制的高质量视频数据集，捕捉了人类日常活动中的丰富和精细的运动细节。大量实验表明，与最先进的模型和现有的偏好优化技术相比，RealDPO显著提高了视频质量、文本对齐和运动的真实感。 

---
# Architecture Is All You Need: Diversity-Enabled Sweet Spots for Robust Humanoid Locomotion 

**Title (ZH)**: Architecture Is All You Need: 基于多样性的稳健人形机器人行走优化nellested 

**Authors**: Blake Werner, Lizhi Yang, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14947)  

**Abstract**: Robust humanoid locomotion in unstructured environments requires architectures that balance fast low-level stabilization with slower perceptual decision-making. We show that a simple layered control architecture (LCA), a proprioceptive stabilizer running at high rate, coupled with a compact low-rate perceptual policy, enables substantially more robust performance than monolithic end-to-end designs, even when using minimal perception encoders. Through a two-stage training curriculum (blind stabilizer pretraining followed by perceptual fine-tuning), we demonstrate that layered policies consistently outperform one-stage alternatives in both simulation and hardware. On a Unitree G1 humanoid, our approach succeeds across stair and ledge tasks where one-stage perceptual policies fail. These results highlight that architectural separation of timescales, rather than network scale or complexity, is the key enabler for robust perception-conditioned locomotion. 

**Abstract (ZH)**: 无结构环境中鲁棒的人形机器人运动需要平衡快速低层级稳定与缓慢感知决策的架构。我们展示了简单分层控制架构（LCA）、高频率的本体感受稳定器与低频的感知策略相结合，即使使用最小的感知编码器，也能实现比端到端单一架构更鲁棒的表现。通过两阶段培训课程（盲稳定器预训练后进行感知微调），我们证明了分层策略在仿真和硬件中的一致性表现优于单阶段替代方案。在Unitree G1人形机器人上，我们的方法在一台阶感知策略失败的楼梯和凸起任务中取得了成功。这些结果强调了时间尺度的架构分离而非网络规模或复杂性是实现鲁棒感知条件下的运动的关键。 

---
# MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics 

**Title (ZH)**: MetaBench: 代谢组学中评估LLM多任务能力的基准测试 

**Authors**: Yuxing Lu, Xukai Zhao, J. Ben Tamo, Micky C. Nnamdi, Rui Peng, Shuang Zeng, Xingyu Hu, Jinzhuo Wang, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14944)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用文本上展现了显著的能力；然而，它们在需要深厚且相互关联知识的专门科学领域中的熟练程度仍大多未被characterize。代谢组学提出了独特的挑战，包括其复杂的生物化学途径、异质的标识系统以及分散的数据库。为了系统评估LLMs在这一领域的能力，我们引入了MetaBench，这是首个用于代谢组学评估的标准数据库。MetaBench从权威的公开资源中甄选，评估了代谢组学研究中五个关键能力：知识、理解、扎根、推理和研究。我们对25个开源和闭源的LLMs的评估揭示了代谢组学任务中不同的性能模式：尽管模型在文本生成任务上表现良好，但在跨数据库标识符扎根上仍然具有挑战性，即使使用了检索增强技术也是如此。模型在稀疏注释的长尾代谢物上的表现也较差。通过MetaBench，我们为开发和评估代谢组学AI系统提供了重要的基础设施，从而推动了可靠计算工具的系统性进展，以服务于代谢组学研究。 

---
# LaSeR: Reinforcement Learning with Last-Token Self-Rewarding 

**Title (ZH)**: LaSeR: 基于最后一令牌自我奖励的强化学习 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14943)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance. 

**Abstract (ZH)**: 可验证奖励的强化学习（VERL）：提升大型语言模型推理能力的核心范式 

---
# Circuit Insights: Towards Interpretability Beyond Activations 

**Title (ZH)**: 电路洞察：超越激活函数的可解释性探索 

**Authors**: Elena Golimblevskaia, Aakriti Jain, Bruno Puri, Ammar Ibrahim, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14936)  

**Abstract**: The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality. 

**Abstract (ZH)**: 可解释AI和机制可解释性领域的研究旨在揭示神经网络的内部结构，电路发现是理解模型计算的重要工具。现有方法依赖手动检查且仅限于玩具任务。自动可解释性通过分析孤立特征及其激活来实现可扩展性，但往往忽略了特征之间的交互，并且强烈依赖于外部LLM和数据集质量。编码器 recently 使能够将特征归属分解为输入依赖和输入不变的组件，为更系统的电路分析奠定了基础。基于此，我们提出WeightLens和CircuitLens两种互补的方法，超越了基于激活的分析。WeightLens直接从学习的权重中解释特征，无需使用解释器模型或数据集，同时在独立于上下文的特征上与现有方法性能相当或更好。CircuitLens捕捉特征激活由组件间交互引起的方式，揭示了仅基于激活的方法无法识别的电路级动态。结合使用，这些方法提高了可解释性的鲁棒性，增强了电路的可扩展机制分析的效率和质量。 

---
# Predicting Task Performance with Context-aware Scaling Laws 

**Title (ZH)**: 基于上下文感知的标度律预测任务性能 

**Authors**: Kyle Montgomery, David Park, Jianhong Tu, Michael Bendersky, Beliz Gunel, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14919)  

**Abstract**: Scaling laws have transformed our understanding of large language models by linking upstream metrics like cross-entropy loss to design factors such as model size, training data, and compute. However, these conventional laws fail to capture downstream task performance, where context plays a critical role. In this work, we propose a straightforward, interpretable framework that jointly models downstream performance as a function of the training compute and the provided context. We empirically validate our framework by fitting it on the observed downstream performance of extended-context variants of Llama-2-7B and Llama-2-13B across 65,500 unique instances spanning three tasks: arithmetic reasoning, common sense reasoning, and machine translation. Our results demonstrate that our framework accurately models in-distribution downstream performance, generalizes across three orders of magnitude in training compute, and reliably extrapolates performance as the amount of context increases. These findings offer valuable insights into the interplay between training compute and context utilization, providing guidance for designing more efficient long-context LLMs for diverse downstream tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 缩放定律通过将上游指标如交叉熵损失与模型规模、训练数据和计算资源等设计因素联系起来，已经改变了我们对大型语言模型的理解。然而，这些传统规律无法捕捉到下游任务性能，其中上下文扮演了至关重要的角色。本研究提出了一种简单且可解释的框架，该框架将下游性能建模为训练计算和提供的上下文的函数。我们通过在65,500个独特实例上拟合扩展上下文版本的Llama-2-7B和Llama-2-13B在三个任务（算术推理、常识推理和机器翻译）上的观察到的下游性能来实证验证该框架。研究表明，我们的框架能准确 modeling 收敛内的下游性能，在三个数量级的训练计算上具有泛化能力，并且能够可靠地在上下文量增加时外推性能。这些发现为理解训练计算和上下文利用之间的相互作用提供了宝贵的见解，并为设计适用于多下游任务的更高效长上下文LLMs提供了指导。我们的代码可在以下网址获得：this https URL。 

---
# MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos 

**Title (ZH)**: MaskCaptioner：学习联合分割和描述视频中对象轨迹 

**Authors**: Gabriel Fiastre, Antoine Yang, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2510.14904)  

**Abstract**: Dense Video Object Captioning (DVOC) is the task of jointly detecting, tracking, and captioning object trajectories in a video, requiring the ability to understand spatio-temporal details and describe them in natural language. Due to the complexity of the task and the high cost associated with manual annotation, previous approaches resort to disjoint training strategies, potentially leading to suboptimal performance. To circumvent this issue, we propose to generate captions about spatio-temporally localized entities leveraging a state-of-the-art VLM. By extending the LVIS and LV-VIS datasets with our synthetic captions (LVISCap and LV-VISCap), we train MaskCaptioner, an end-to-end model capable of jointly detecting, segmenting, tracking and captioning object trajectories. Moreover, with pretraining on LVISCap and LV-VISCap, MaskCaptioner achieves state-of-the-art DVOC results on three existing benchmarks, VidSTG, VLN and BenSMOT. The datasets and code are available at this https URL. 

**Abstract (ZH)**: 稠密视频对象描述（DVOC）是联合检测、跟踪和描述视频中对象轨迹的任务，要求能够理解时空细节并用自然语言描述。由于任务的复杂性和手动标注的高成本，先前的方法采用分离训练策略，可能导致性能不佳。为解决这一问题，我们提出利用先进的视觉语言模型生成时空局部化实体的描述。通过将LVIS和LV-VIS数据集扩展为包含我们合成的描述（LVISCap和LV-VISCap），我们训练了一个端到端模型MaskCaptioner，该模型能够联合检测、分割、跟踪和描述对象轨迹。此外，通过在LVISCap和LV-VISCap上的预训练，MaskCaptioner在三个现有基准VidSTG、VLN和BenSMOT上达到了最先进的DVOC结果。数据集和代码可在以下链接获取。 

---
# Reasoning with Sampling: Your Base Model is Smarter Than You Think 

**Title (ZH)**: 采样推理：你的基模型比你想象的要聪明 

**Authors**: Aayush Karan, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.14901)  

**Abstract**: Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains. 

**Abstract (ZH)**: 前端推理模型在广泛学科中显示出了令人惊叹的能力，通过后训练的大语言模型（LLMs）结合强化学习（RL）。然而，尽管这一范式取得了广泛的 Success，大部分文献集中在剖析在RL过程中出现的真正新颖行为，而在基础模型中不存在的行为。在我们的工作中，我们从一个不同的角度来探讨这个问题，问是否可以在推理过程中从基础模型中通过纯粹采样来诱发相当的推理能力，而无需任何额外训练。受马尔可夫链蒙特卡罗（MCMC）技术从尖化分布中采样的启发，我们提出了一种简单的迭代采样算法，利用基础模型本身的似然性。对于不同基础模型，我们展示我们的算法在推理能力上提供了显著提升，几乎能与甚至在多种单一任务（包括MATH500、HumanEval和GPQA）上超越RL后的训练。此外，我们的采样器避免了RL后训练过程中多次采样后多样性下降的特征。最关键的是，我们的方法不需要训练、筛选数据集或验证器，这表明它在易于验证的领域之外也具有广泛的适用性。 

---
# Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media 

**Title (ZH)**: 通过社交媒体的纵向和信息环境信号检测早期和隐匿的自杀意念 

**Authors**: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2510.14889)  

**Abstract**: On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments. 

**Abstract (ZH)**: 在社交媒体上，很多经历自杀意念的个体不会明确披露其痛苦。相反，这些信号可能通过日常发布的帖子或同伴互动间接表现出来。及早检测这些隐性的信号至关重要但极具挑战性。我们将早期和隐性的自杀意念视为一种前瞻预测任务，并开发了一种计算框架，该框架模型了用户的信息环境，包括用户的纵向发帖历史及其社交邻近同伴的言论。我们采用了综合网络中心性度量来识别用户的重要邻居，并暂态对齐用户及其邻居的互动——在微调的DeBERTa-v3模型中整合了多层次的信号。在针对1,000名（500名病例和500名对照）Reddit用户的研究中，我们的方法相较于仅基于个体的基线提高了15%的早期和隐性自杀意念检测率。这些发现强调了同伴互动提供的宝贵预测信号，并具有更广泛的含义，即设计能在网络环境中捕捉隐性及掩饰的风险表达的早期检测系统。 

---
# Learning When Not to Learn: Risk-Sensitive Abstention in Bandits with Unbounded Rewards 

**Title (ZH)**: 学习在何时不学习：带无界奖励的Bandits中的风险敏感型回避 

**Authors**: Sarah Liaw, Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2510.14884)  

**Abstract**: In high-stakes AI applications, even a single action can cause irreparable damage. However, nearly all of sequential decision-making theory assumes that all errors are recoverable (e.g., by bounding rewards). Standard bandit algorithms that explore aggressively may cause irreparable damage when this assumption fails. Some prior work avoids irreparable errors by asking for help from a mentor, but a mentor may not always be available. In this work, we formalize a model of learning with unbounded rewards without a mentor as a two-action contextual bandit with an abstain option: at each round the agent observes an input and chooses either to abstain (always 0 reward) or to commit (execute a preexisting task policy). Committing yields rewards that are upper-bounded but can be arbitrarily negative, and the commit reward is assumed Lipschitz in the input. We propose a caution-based algorithm that learns when not to learn: it chooses a trusted region and commits only where the available evidence does not already certify harm. Under these conditions and i.i.d. inputs, we establish sublinear regret guarantees, theoretically demonstrating the effectiveness of cautious exploration for deploying learning agents safely in high-stakes environments. 

**Abstract (ZH)**: 在高风险AI应用中，即使单个行动也可能造成不可逆的损害。然而，几乎所有序贯决策理论都假设所有错误都是可恢复的（例如，通过限制奖励）。标准的探索性很强的多臂老虎机算法在这一假设不成立时可能会造成不可逆的损害。一些先前的工作通过寻求导师的帮助来避免不可逆的错误，但导师可能并不总是可用的。在这项工作中，我们以两行动上下文多臂老虎机模型的形式形式化了无导师的无界奖励学习模型，该模型附带弃权选项：在每一轮中，代理观察输入并选择弃权（总是0奖励）或投入（执行预存的任务策略）。投入产生上界受限但可以任意负的奖励，且投入奖励假设为输入的Lipschitz函数。我们提出了一种基于谨慎性的算法，学习何时不应学习：它选择一个可信赖的区域，并仅在现有证据未认证有危害时才投入。在这些条件下和独立同分布的输入下，我们建立了子线性后悔保证，从理论上证明了谨慎探索在高风险环境中安全部署学习代理的有效性。 

---
# Predicting kernel regression learning curves from only raw data statistics 

**Title (ZH)**: 仅从原始数据统计预测核回归学习曲线 

**Authors**: Dhruva Karkada, Joseph Turnbull, Yuxi Liu, James B. Simon  

**Link**: [PDF](https://arxiv.org/pdf/2510.14878)  

**Abstract**: We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet. We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$. The key new idea is an analytical approximation of a kernel's eigenvalues and eigenfunctions with respect to an anisotropic data distribution. The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the Hermite eigenstructure ansatz (HEA). We prove the HEA for Gaussian data, but we find that real image data is often "Gaussian enough" for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk. Extending beyond kernel regression, we empirically find that MLPs in the feature-learning regime learn Hermite polynomials in the order predicted by the HEA. Our HEA framework is a proof of concept that an end-to-end theory of learning which maps dataset structure all the way to model performance is possible for nontrivial learning algorithms on real datasets. 

**Abstract (ZH)**: 我们研究了在CIFAR-5m、SVHN和ImageNet等真实数据集上使用共同的旋转不变核的核回归。我们提供了一个理论框架，仅从经验数据协方差矩阵和目标函数$f_*$的经验多项式分解中预测学习曲线（测试风险与样本大小的关系）。关键新想法是核在各向异性数据分布下的特征值和特征函数的分析近似。特征函数类似于数据的赫mite多项式，因此我们将这种近似称为赫mite特征结构假设（HEA）。我们证明了HEA在高斯数据上成立，但在实际图像数据上，HEA在实践中常常很好地成立，使我们能够通过将核特征结构与测试风险的关系应用到以前的结果来预测学习曲线。超越核回归，我们实验证明多层感知机在特征学习阶段学习的赫mite多项式顺序由HEA预测。我们的HEA框架证明了对于真实数据集上非平凡的学习算法，从数据集结构直接映射到模型性能的端到端理论是可能的。 

---
# Benchmarking Multimodal Large Language Models for Face Recognition 

**Title (ZH)**: 多模态大规模语言模型在人脸识别中的基准测试 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2510.14866)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable performance across diverse vision-and-language tasks. However, their potential in face recognition remains underexplored. In particular, the performance of open-source MLLMs needs to be evaluated and compared with existing face recognition models on standard benchmarks with similar protocol. In this work, we present a systematic benchmark of state-of-the-art MLLMs for face recognition on several face recognition datasets, including LFW, CALFW, CPLFW, CFP, AgeDB and RFW. Experimental results reveal that while MLLMs capture rich semantic cues useful for face-related tasks, they lag behind specialized models in high-precision recognition scenarios in zero-shot applications. This benchmark provides a foundation for advancing MLLM-based face recognition, offering insights for the design of next-generation models with higher accuracy and generalization. The source code of our benchmark is publicly available in the project page. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在多种视觉与语言任务中取得了显著性能。然而，它们在人脸识别领域的潜力尚待探索。特别是，开源MLLMs的性能需要在标准基准上与现有的人脸识别模型进行评估和比较。在此项工作中，我们系统地在多个人脸识别数据集上评估了最先进的MLLMs的性能，包括LFW、CALFW、CPLFW、CFP、AgeDB和RFW。实验结果显示，虽然MLLMs能捕捉到对人脸识别任务有用丰富的语义线索，但在零样本应用中的高精度识别场景中，它们仍落后于专门模型。该基准提供了一个推进基于MLLM的人脸识别的基础，为设计更高准确性和泛化能力的下一代模型提供了见解。我们的基准代码已在项目页面上公开。 

---
# RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning 

**Title (ZH)**: RL-100: 实用的机器人 manipulotion 与现实世界的强化学习 

**Authors**: Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14830)  

**Abstract**: Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained bu supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours. 

**Abstract (ZH)**: 基于扩散视觉运动策略的鲁棒实时强化学习训练框架：RL-100 

---
# Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks 

**Title (ZH)**: 多肿瘤早期检测中通过增加报告数量减少掩膜数量以 Scaling Artificial Intelligence 的方式 

**Authors**: Pedro R. A. S. Bassi, Xinze Zhou, Wenxuan Li, Szymon Płotka, Jieneng Chen, Qi Chen, Zheren Zhu, Jakub Prządo, Ibrahim E. Hamacı, Sezgin Er, Yuhan Wang, Ashwin Kumar, Bjoern Menze, Jarosław B. Ćwikła, Yuyin Zhou, Akshay S. Chaudhari, Curtis P. Langlotz, Sergio Decherchi, Andrea Cavalli, Kang Wang, Yang Yang, Alan L. Yuille, Zongwei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14803)  

**Abstract**: Early tumor detection save lives. Each year, more than 300 million computed tomography (CT) scans are performed worldwide, offering a vast opportunity for effective cancer screening. However, detecting small or early-stage tumors on these CT scans remains challenging, even for experts. Artificial intelligence (AI) models can assist by highlighting suspicious regions, but training such models typically requires extensive tumor masks--detailed, voxel-wise outlines of tumors manually drawn by radiologists. Drawing these masks is costly, requiring years of effort and millions of dollars. In contrast, nearly every CT scan in clinical practice is already accompanied by medical reports describing the tumor's size, number, appearance, and sometimes, pathology results--information that is rich, abundant, and often underutilized for AI training. We introduce R-Super, which trains AI to segment tumors that match their descriptions in medical reports. This approach scales AI training with large collections of readily available medical reports, substantially reducing the need for manually drawn tumor masks. When trained on 101,654 reports, AI models achieved performance comparable to those trained on 723 masks. Combining reports and masks further improved sensitivity by +13% and specificity by +8%, surpassing radiologists in detecting five of the seven tumor types. Notably, R-Super enabled segmentation of tumors in the spleen, gallbladder, prostate, bladder, uterus, and esophagus, for which no public masks or AI models previously existed. This study challenges the long-held belief that large-scale, labor-intensive tumor mask creation is indispensable, establishing a scalable and accessible path toward early detection across diverse tumor types.
We plan to release our trained models, code, and dataset at this https URL 

**Abstract (ZH)**: 早期肿瘤检测拯救生命。每年，全球范围内进行的CT扫描超过3亿次，提供了有效癌症筛查的巨大机会。然而，即使是专家也难以在这些CT扫描中检测到小型或早期阶段的肿瘤。人工智能（AI）模型可以通过突出显示可疑区域来提供帮助，但训练这些模型通常需要大量的人工绘制的肿瘤掩膜——由放射ologist手工绘制的详细体素级肿瘤轮廓。绘制这些掩膜成本高昂，需要数年时间和数百万美元。相比之下，临床实践中几乎每一张CT扫描都已经附带了描述肿瘤大小、数量、外观以及有时病理结果的医学报告——这些信息丰富且充足，但通常未被用于AI训练。我们介绍了R-Super，该方法训练AI将肿瘤与其医学报告中的描述进行匹配分隔。这种方法利用大量现成的医学报告来扩展AI训练规模，大幅减少了手动绘制肿瘤掩膜的需求。当使用101,654份报告进行训练时，AI模型的性能与使用723张掩膜进行训练的模型相当。结合使用报告和掩膜进一步提高了敏感性+13%和特异性+8%，超越了放射学家在检测七种肿瘤类型中的五种时的表现。值得一提的是，R-Super使得在脾脏、胆囊、前列腺、膀胱、子宫和食道等部位的肿瘤分割成为可能，而对于这些部位，之前并未存在公共掩膜或AI模型。这项研究挑战了大规模、劳动密集型肿瘤掩膜创建不可或缺的传统观念，为各类肿瘤类型的早期检测提供了一条可扩展且易获取的道路。 

---
# Morphology-Aware Prognostic model for Five-Year Survival Prediction in Colorectal Cancer from H&E Whole Slide Images 

**Title (ZH)**: 染色组织学全切片图像中基于形态学的结直肠癌五年生存率预测模型 

**Authors**: Usama Sajjad, Abdul Rehman Akbar, Ziyu Su, Deborah Knight, Wendy L. Frankel, Metin N. Gurcan, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14800)  

**Abstract**: Colorectal cancer (CRC) remains the third most prevalent malignancy globally, with approximately 154,000 new cases and 54,000 projected deaths anticipated for 2025. The recent advancement of foundation models in computational pathology has been largely propelled by task agnostic methodologies that can overlook organ-specific crucial morphological patterns that represent distinct biological processes that can fundamentally influence tumor behavior, therapeutic response, and patient outcomes. The aim of this study is to develop a novel, interpretable AI model, PRISM (Prognostic Representation of Integrated Spatial Morphology), that incorporates a continuous variability spectrum within each distinct morphology to characterize phenotypic diversity and reflecting the principle that malignant transformation occurs through incremental evolutionary processes rather than abrupt phenotypic shifts. PRISM is trained on 8.74 million histological images extracted from surgical resection specimens of 424 patients with stage III CRC. PRISM achieved superior prognostic performance for five-year OS (AUC = 0.70 +- 0.04; accuracy = 68.37% +- 4.75%; HR = 3.34, 95% CI = 2.28-4.90; p < 0.0001), outperforming existing CRC-specific methods by 15% and AI foundation models by ~23% accuracy. It showed sex-agnostic robustness (AUC delta = 0.02; accuracy delta = 0.15%) and stable performance across clinicopathological subgroups, with minimal accuracy fluctuation (delta = 1.44%) between 5FU/LV and CPT-11/5FU/LV regimens, replicating the Alliance cohort finding of no survival difference between treatments. 

**Abstract (ZH)**: 结直肠癌（CRC）仍然是全球第三大常见恶性肿瘤，预计2025年将有约154,000例新发病例和54,000例死亡。最近，在计算病理学中基础模型的发展很大程度上得益于任务无关的方法，这些方法可能会忽略特定器官的关键形态学模式，而这些模式代表了不同的生物学过程，这些过程可以从根本上影响肿瘤行为、治疗反应和患者预后。本研究的目标是开发一种新的可解释人工智能模型PRISM（预后综合空间形态学代表），该模型在每种独特形态内结合了连续的变异谱，以表征表型多样性，并反映恶性转化是通过渐进的进化过程，而不是突然的表型转变。PRISM在424例III期CRC患者的手术切除标本中提取的874万张组织学图像上进行训练。PRISM在五年总生存期的预后性能表现优异（AUC = 0.70 ± 0.04；准确率 = 68.37% ± 4.75%；HR = 3.34，95% CI = 2.28-4.90；p < 0.0001），其性能优于现有的CRC特异性方法15%，优于AI基础模型约23%的准确率。它表现出性别无偏好性稳健性（AUC变化 = 0.02；准确率变化 = 0.15%），并在临床病理学亚组中表现出稳定性能，治疗方案（FOLFOX和CPT-11/FOLFOX）之间的小幅准确率波动（变化 = 1.44%），复制了Alliance队列研究中治疗之间无生存差异的发现。 

---
# Cross-Scenario Unified Modeling of User Interests at Billion Scale 

**Title (ZH)**: 十亿规模跨场景统一建模用户兴趣 

**Authors**: Manjie Xu, Cheng Chen, Xin Jia, Jingyi Zhou, Yongji Wu, Zejian Wang, Chi Zhang, Kai Zuo, Yibo Chen, Xu Tang, Yao Hu, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14788)  

**Abstract**: User interests on content platforms are inherently diverse, manifesting through complex behavioral patterns across heterogeneous scenarios such as search, feed browsing, and content discovery. Traditional recommendation systems typically prioritize business metric optimization within isolated specific scenarios, neglecting cross-scenario behavioral signals and struggling to integrate advanced techniques like LLMs at billion-scale deployments, which finally limits their ability to capture holistic user interests across platform touchpoints. We propose RED-Rec, an LLM-enhanced hierarchical Recommender Engine for Diversified scenarios, tailored for industry-level content recommendation systems. RED-Rec unifies user interest representations across multiple behavioral contexts by aggregating and synthesizing actions from varied scenarios, resulting in comprehensive item and user modeling. At its core, a two-tower LLM-powered framework enables nuanced, multifaceted representations with deployment efficiency, and a scenario-aware dense mixing and querying policy effectively fuses diverse behavioral signals to capture cross-scenario user intent patterns and express fine-grained, context-specific intents during serving. We validate RED-Rec through online A/B testing on hundreds of millions of users in RedNote through online A/B testing, showing substantial performance gains in both content recommendation and advertisement targeting tasks. We further introduce a million-scale sequential recommendation dataset, RED-MMU, for comprehensive offline training and evaluation. Our work advances unified user modeling, unlocking deeper personalization and fostering more meaningful user engagement in large-scale UGC platforms. 

**Abstract (ZH)**: 用户在内容平台上的兴趣本质上是多元的，通过跨异构场景（如搜索、信息流浏览和内容发现）的复杂行为模式表现出来。传统的推荐系统通常优先优化孤立特定场景下的业务指标，忽视了跨场景的行为信号，并且难以在十亿规模的部署中集成先进的技术（如LLMs），从而限制了它们在全平台触点上捕捉用户整体兴趣的能力。我们提出了RED-Rec，一种增强型层次推荐引擎，专为工业级内容推荐系统设计。RED-Rec 通过聚合和综合来自不同场景的多种行为动作来统一用户的兴趣表示，实现全面的项目和用户建模。其核心是一个基于LLM的双塔框架，能够提供细腻、多维度的表示，并通过场景感知的密集混合和查询策略有效融合多种行为信号，以捕捉跨场景的用户意图模式，并在服务中表达细粒度的、特定于上下文的意图。通过在线A/B测试验证，我们在RedNote上对数亿用户进行了内容推荐和广告定向任务的验证，显示出了显著的性能提升。我们还引入了一个百万规模的序列推荐数据集RED-MMU，用于全面的离线训练和评估。我们的工作推进了统一的用户建模，实现了更深层次的个性化，并促进了大规模UGC平台上的更高质量的用户参与。 

---
# Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning 

**Title (ZH)**: 寻找思考中的答案：重新审视大型语言模型的推理评价 

**Authors**: Hwiyeol Jo, Joosung Lee, Jaehone Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14773)  

**Abstract**: Evaluating generative models, such as large language models (LLMs), commonly involves question-answering tasks where the final answer is selected based on probability of answer choices. On the other hand, for models requiring reasoning, the method of answer extraction plays a critical role. Our research reveals that the performance of reasoning models and their final answer distributions are highly sensitive to the answer extraction algorithm employed. In order to mitigate this, we propose a basic framework: Answer Regeneration. The method uses an additional model inference, providing the prior input and output prefaced by the prompt "Answer:". The final answer is then selected or extracted from the regenerated output. We show that this extraction-rule-agnostic approach exhibits improved performance and enhanced robustness. Furthermore, we have applied this framework to general math problems and open-ended question answering tasks. Our analysis and this framework could offer a more reliable results for model evaluation. 

**Abstract (ZH)**: 评估生成模型，如大型语言模型（LLMs），通常涉及问答任务，其中最终答案是基于答案选项的概率选取。另一方面，对于需要推理的模型，答案提取方法起着关键作用。我们的研究发现，推理模型的性能及其最终答案分布对所使用的答案提取算法高度敏感。为了缓解这一问题，我们提出了一种基本框架：答案再生。该方法使用额外的模型推理，并在输入和输出前缀加“Answer:”提示。然后从再生输出中选择或提取最终答案。我们展示了这种不依赖于提取规则的方法在性能和稳健性方面都有所提升。此外，我们还将该框架应用于一般数学问题和开放性问题回答任务。我们的分析和该框架能够为模型评估提供更可靠的结果。 

---
# Inpainting the Red Planet: Diffusion Models for the Reconstruction of Martian Environments in Virtual Reality 

**Title (ZH)**: 虚拟现实中火星环境的修复：扩散模型在重建火星环境中的应用 

**Authors**: Giuseppe Lorenzo Catalano, Agata Marta Soccini  

**Link**: [PDF](https://arxiv.org/pdf/2510.14765)  

**Abstract**: Space exploration increasingly relies on Virtual Reality for several tasks, such as mission planning, multidisciplinary scientific analysis, and astronaut training. A key factor for the reliability of the simulations is having accurate 3D representations of planetary terrains. Extraterrestrial heightmaps derived from satellite imagery often contain missing values due to acquisition and transmission constraints. Mars is among the most studied planets beyond Earth, and its extensive terrain datasets make the Martian surface reconstruction a valuable task, although many areas remain unmapped. Deep learning algorithms can support void-filling tasks; however, whereas Earth's comprehensive datasets enables the use of conditional methods, such approaches cannot be applied to Mars. Current approaches rely on simpler interpolation techniques which, however, often fail to preserve geometric coherence. In this work, we propose a method for reconstructing the surface of Mars based on an unconditional diffusion model. Training was conducted on an augmented dataset of 12000 Martian heightmaps derived from NASA's HiRISE survey. A non-homogeneous rescaling strategy captures terrain features across multiple scales before resizing to a fixed 128x128 model resolution. We compared our method against established void-filling and inpainting techniques, including Inverse Distance Weighting, kriging, and Navier-Stokes algorithm, on an evaluation set of 1000 samples. Results show that our approach consistently outperforms these methods in terms of reconstruction accuracy (4-15% on RMSE) and perceptual similarity (29-81% on LPIPS) with the original data. 

**Abstract (ZH)**: 外太空探索越来越多地依赖虚拟现实进行任务规划、跨学科科学分析和宇航员训练。高度图的准确三维表示是模拟可靠性的关键因素。由于获取和传输约束，从卫星图像派生的外星球高度图经常包含缺失值。火星是除地球外研究最多的行星之一，其庞大的地形数据集使火星表面重建成为一个有价值的任务，尽管许多地区尚未被测绘。深度学习算法可以支持空值填充任务；然而，由于地球数据的全面性，可以在其中应用条件方法，而这些方法在火星上无法应用。当前的方法依赖于更简单的插值技术，但这些技术往往无法保持几何连贯性。在这项工作中，我们提出了一种基于无条件扩散模型的火星表面重建方法。训练数据集包含12000个由NASA HiRISE调查派生的高度图，并进行了扩充。非均匀缩放策略在缩放至固定128x128模型分辨率之前捕捉跨多个尺度的地形特征。我们使用1000个样本的评估集将我们的方法与已建立的空值填充和修复技术（包括距离加权法、克里金法和Navier-Stokes算法）进行了比较。结果显示，在均方根误差和LPIPS感知相似度方面，我们的方法在重建准确性上始终优于这些方法（4-15%的均方根误差改善和29-81%的LPIPS感知相似度提升）。 

---
# COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes 

**Title (ZH)**: COIG-Writer: 一种包含思维过程的高质量中文创意写作数据集 

**Authors**: Yunwen Li, Shuangshuang Ying, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Tianyu Zheng, Xeron Du, Qiguang Chen, Jiajun Shi, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Stephen Huang, Wanxiang Che, Chenghua Lin, Eli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14763)  

**Abstract**: Large language models exhibit systematic deficiencies in creative writing, particularly in non-English contexts where training data is scarce and lacks process-level supervision. We present COIG-Writer, a novel Chinese creative writing dataset that captures both diverse outputs and their underlying thought processes through systematic reverse-engineering of high-quality texts. Unlike existing datasets that provide only input-output pairs, COIG-Writer comprises 1,665 meticulously curated triplets spanning 51 genres, each containing: (1) a reverse-engineered prompt, (2) detailed creative reasoning documenting decision-making processes, and (3) the final text. Through comprehensive experiments, we identify a two-component model of creative writing: narrative logic (provided by process supervision) and linguistic expression (maintained by general-purpose data). Our findings reveal three critical insights: (1) Process supervision is highly effective but requires stabilization with general data. A ratio of at least one creative sample to twelve general samples is needed to achieve optimal performance; below this threshold, the win rate progressively degrades (from 62.75% down to 35.78%)., (2) creative capabilities are culturally-bound with no cross-lingual transfer (89.26pp gap between Chinese and English performance), and (3) lexical diversity inversely correlates with creative quality (TTR paradox), suggesting high diversity signals compensatory behavior for logical deficiencies. These findings establish that creative excellence emerges from the interaction between logical scaffolding and linguistic grounding, analogous to how mathematical reasoning enhances but cannot replace linguistic competence in foundation models. 

**Abstract (ZH)**: 大型语言模型在创意写作中表现出系统性的不足，特别是在训练数据稀缺且缺乏过程级监督的非英语语境中更为明显。我们提出了COIG-Writer，这是一个新型的中文创意写作数据集，通过系统逆向工程高质量文本，捕捉了多样化输出及其背后的思想过程。与仅提供输入-输出对的现有数据集不同，COIG-Writer 包含了1,665个精挑细选的三元组，涵盖了51种文体，每个三元组包含：(1) 逆向工程生成的提示，(2) 详细的创意推理记录决策过程，以及(3) 最终文本。通过全面的实验，我们识别出创意写作的双成分模型：叙述逻辑（由过程监督提供）和语言表达（由通用数据保持）。我们的研究发现三条关键见解：(1) 过程监督非常有效，但需要与通用数据相结合以稳定效果。至少需要一个创意样本配十二个通用样本才能达到最佳性能；低于这一阈值时，成功率逐渐下降 (从 62.75% 降至 35.78%)，(2) 创造力能力是文化绑定的，在跨语言迁移上没有显著效果（中文和英文表现有 89.26 个百分点的差距），(3) 词汇多样性与创意质量呈负相关（TTR悖论），这表明高多样性的词汇可能是对逻辑不足的一种补偿行为。这些发现表明，创意卓越来自于逻辑支撑和语言根基的交互作用，类似于数学推理如何增强但不能替代语言能力，对基础模型同样适用。 

---
# Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries 

**Title (ZH)**: 超越多令牌预测：使用未来摘要预训练大规模语言模型 

**Authors**: Divyat Mahajan, Sachin Goyal, Badr Youbi Idrissi, Mohammad Pezeshki, Ioannis Mitliagkas, David Lopez-Paz, Kartik Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2510.14751)  

**Abstract**: Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks. 

**Abstract (ZH)**: 未来摘要预测（FSP）在长期推理、计划和创造性写作中的应用及其效果 

---
# DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models 

**Title (ZH)**: DEXTER：基于扩散引导的文本推理解释方法用于视觉模型 

**Authors**: Simone Carnemolla, Matteo Pennisi, Sarinda Samarasinghe, Giovanni Bellitto, Simone Palazzo, Daniela Giordano, Mubarak Shah, Concetto Spampinato  

**Link**: [PDF](https://arxiv.org/pdf/2510.14741)  

**Abstract**: Understanding and explaining the behavior of machine learning models is essential for building transparent and trustworthy AI systems. We introduce DEXTER, a data-free framework that employs diffusion models and large language models to generate global, textual explanations of visual classifiers. DEXTER operates by optimizing text prompts to synthesize class-conditional images that strongly activate a target classifier. These synthetic samples are then used to elicit detailed natural language reports that describe class-specific decision patterns and biases. Unlike prior work, DEXTER enables natural language explanation about a classifier's decision process without access to training data or ground-truth labels. We demonstrate DEXTER's flexibility across three tasks-activation maximization, slice discovery and debiasing, and bias explanation-each illustrating its ability to uncover the internal mechanisms of visual classifiers. Quantitative and qualitative evaluations, including a user study, show that DEXTER produces accurate, interpretable outputs. Experiments on ImageNet, Waterbirds, CelebA, and FairFaces confirm that DEXTER outperforms existing approaches in global model explanation and class-level bias reporting. Code is available at this https URL. 

**Abstract (ZH)**: 理解并解释机器学习模型的行为对于构建透明可信赖的AI系统至关重要。我们提出了DEXTER，一种无需数据的框架，利用扩散模型和大型语言模型生成视觉分类器的全局文本解释。DEXTER通过优化文本提示来合成强激活目标分类器的类条件图像。然后使用这些合成样本引发详细的自然语言报告，描述类别的特定决策模式和偏差。与先前工作不同，DEXTER能够在不访问训练数据或真实标签的情况下，为分类器的决策过程提供自然语言解释。DEXTER在激活最大化、切片发现与去偏见和偏差解释三项任务上展示了其能力，证明能够揭示视觉分类器的内部机制。定量和定性评估，包括用户研究，表明DEXTER能够生成准确且可解释的输出。在 imagenet、waterbirds、celeba 和 fairfaces 上的实验证明，DEXTER在全局模型解释和类别级偏差报告方面优于现有方法。代码可在以下链接获取。 

---
# Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling 

**Title (ZH)**: Seesaw: 通过平衡学习率和批量大小调度加速训练 

**Authors**: Alexandru Meterez, Depen Morwani, Jingfeng Wu, Costin-Andrei Oncescu, Cengiz Pehlevan, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2510.14717)  

**Abstract**: Increasing the batch size during training -- a ''batch ramp'' -- is a promising strategy to accelerate large language model pretraining. While for SGD, doubling the batch size can be equivalent to halving the learning rate, the optimal strategy for adaptive optimizers like Adam is less clear. As a result, any batch-ramp scheduling, if used at all, is typically tuned heuristically. This work develops a principled framework for batch-size scheduling and introduces Seesaw: whenever a standard scheduler would halve the learning rate, Seesaw instead multiplies it by $1/\sqrt{2}$ and doubles the batch size, preserving loss dynamics while reducing serial steps. Theoretically, we provide, to our knowledge, the first finite-sample proof of equivalence between learning-rate decay and batch-size ramp-up for SGD on noisy linear regression, and we extend this equivalence to normalized SGD, a tractable proxy for Adam, under a variance-dominated regime observed in practice. Empirically, on 150M/300M/600M-parameter models trained at Chinchilla scale using a constant (critical) batch size, Seesaw matches cosine decay at equal FLOPs while reducing wall-clock time by $\approx 36\%$, approaching the theoretical limit implied by our analysis. 

**Abstract (ZH)**: 在训练过程中逐步增加批次大小——“批次增长”——是一种加快大规模语言模型预训练的有前途的策略。对于SGD，加倍批次大小相当于减半学习率，但对于像Adam这样的自适应优化器，最优策略尚不明确。因此，如果使用任何批次增长策略，通常会进行启发式调整。本研究开发了一个原理性的框架来调度批次大小，并引入了Seesaw：每当标准调度器减半学习率时，Seesaw 会将其乘以 $1/\sqrt{2}$ 并将批次大小加倍，保持损失动态的同时减少串行步骤。理论上，我们提供了一个基于有限样本证明SGD在噪声线性回归中学习率衰减与批次大小增长等效，这是已知的第一个证明，并将这种等效性扩展到实践观察到的方差占主导地位的情况下可求解的归一化SGD，其作为Adam的可处理代理。实验上，在使用恒定（关键）批次大小的Chinchilla规模下训练150M/300M/600M参数模型时，Seesaw 在同等FLOPs的情况下达到余弦衰减效果，将墙钟时间减少了约36%，接近我们分析所暗示的理论极限。 

---
# Camera Movement Classification in Historical Footage: A Comparative Study of Deep Video Models 

**Title (ZH)**: 历史影像中的摄像头运动分类：深度视频模型的比较研究 

**Authors**: Tingyu Lin, Armin Dadras, Florian Kleber, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14713)  

**Abstract**: Camera movement conveys spatial and narrative information essential for understanding video content. While recent camera movement classification (CMC) methods perform well on modern datasets, their generalization to historical footage remains unexplored. This paper presents the first systematic evaluation of deep video CMC models on archival film material. We summarize representative methods and datasets, highlighting differences in model design and label definitions. Five standard video classification models are assessed on the HISTORIAN dataset, which includes expert-annotated World War II footage. The best-performing model, Video Swin Transformer, achieves 80.25% accuracy, showing strong convergence despite limited training data. Our findings highlight the challenges and potential of adapting existing models to low-quality video and motivate future work combining diverse input modalities and temporal architectures. 

**Abstract (ZH)**: 相机运动传递了理解视频内容所需的空问和叙述信息。尽管近期的相机运动分类方法在现代数据集上表现良好，但其对历史片段的泛化能力尚未被探索。本文首次系统评估了深度视频相机运动分类模型在档案电影材料上的表现。我们总结了代表性的方法和数据集，强调了模型设计和标签定义的差异。五种标准的视频分类模型在包含专家标注的二战 footage 的 HISTORIAN 数据集上进行了评估。性能最好的 Video Swin Transformer 模型达到了 80.25% 的准确率，尽管训练数据有限，但仍表现出较强的收敛性。我们的研究结果凸显了现有模型适应低质量视频的挑战和潜力，并激发了未来结合多种输入模态和时空架构的研究工作。 

---
# Where are the Whales: A Human-in-the-loop Detection Method for Identifying Whales in High-resolution Satellite Imagery 

**Title (ZH)**: Whale何在：一种基于人的回路高分辨率卫星图像中识别鲸鱼的检测方法 

**Authors**: Caleb Robinson, Kimberly T. Goetz, Christin B. Khan, Meredith Sackett, Kathleen Leonard, Rahul Dodhia, Juan M. Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2510.14709)  

**Abstract**: Effective monitoring of whale populations is critical for conservation, but traditional survey methods are expensive and difficult to scale. While prior work has shown that whales can be identified in very high-resolution (VHR) satellite imagery, large-scale automated detection remains challenging due to a lack of annotated imagery, variability in image quality and environmental conditions, and the cost of building robust machine learning pipelines over massive remote sensing archives. We present a semi-automated approach for surfacing possible whale detections in VHR imagery using a statistical anomaly detection method that flags spatial outliers, i.e. "interesting points". We pair this detector with a web-based labeling interface designed to enable experts to quickly annotate the interesting points. We evaluate our system on three benchmark scenes with known whale annotations and achieve recalls of 90.3% to 96.4%, while reducing the area requiring expert inspection by up to 99.8% -- from over 1,000 sq km to less than 2 sq km in some cases. Our method does not rely on labeled training data and offers a scalable first step toward future machine-assisted marine mammal monitoring from space. We have open sourced this pipeline at this https URL. 

**Abstract (ZH)**: 有效的鲸群监测对于保护至关重要，但传统调查方法成本高昂且难以扩展。尽管先前的工作证明了可以在非常高分辨率（VHR）卫星图像中识别鲸鱼，但由于缺乏标注图像、图像质量和环境条件的变异性以及构建 robust 机器学习管道的高成本，大规模自动化检测仍具有挑战性。我们提出了一种半自动化方法，通过统计异常检测方法在VHR图像中 surface 可能的鲸鱼检测结果，该方法标记空间异常点，即“有趣点”。我们使用了一个基于Web的标注界面，以使专家能够快速标注这些有趣点。我们在三个具有已知鲸鱼标注的基准场景上评估了该系统，召回率达到了 90.3% 至 96.4%，并且在某些情况下将需要专家检查的区域减少了99.8%，从超过1,000平方公里减少到不到2平方公里。我们的方法不依赖于标注训练数据，并为未来基于机器辅助的空间内海哺乳动物监测提供了一步可扩展的步骤。我们已在以下网址开源了此管道：this https URL。 

---
# FedPPA: Progressive Parameter Alignment for Personalized Federated Learning 

**Title (ZH)**: FedPPA: 进步参数对齐实现个性化联邦学习 

**Authors**: Maulidi Adi Prasetia, Muhamad Risqi U. Saputra, Guntur Dharma Putra  

**Link**: [PDF](https://arxiv.org/pdf/2510.14698)  

**Abstract**: Federated Learning (FL) is designed as a decentralized, privacy-preserving machine learning paradigm that enables multiple clients to collaboratively train a model without sharing their data. In real-world scenarios, however, clients often have heterogeneous computational resources and hold non-independent and identically distributed data (non-IID), which poses significant challenges during training. Personalized Federated Learning (PFL) has emerged to address these issues by customizing models for each client based on their unique data distribution. Despite its potential, existing PFL approaches typically overlook the coexistence of model and data heterogeneity arising from clients with diverse computational capabilities. To overcome this limitation, we propose a novel method, called Progressive Parameter Alignment (FedPPA), which progressively aligns the weights of common layers across clients with the global model's weights. Our approach not only mitigates inconsistencies between global and local models during client updates, but also preserves client's local knowledge, thereby enhancing personalization robustness in non-IID settings. To further enhance the global model performance while retaining strong personalization, we also integrate entropy-based weighted averaging into the FedPPA framework. Experiments on three image classification datasets, including MNIST, FMNIST, and CIFAR-10, demonstrate that FedPPA consistently outperforms existing FL algorithms, achieving superior performance in personalized adaptation. 

**Abstract (ZH)**: 联邦学习（FL）是一种分布式、保护隐私的机器学习范式，使多个客户端能够协同训练模型而无需共享其数据。在实际场景中，客户端往往具有异质计算资源并且持有非独立且同分布的数据（非-IID），这给训练带来了重大挑战。个性化联邦学习（PFL）作为一种解决这些问题的方法应运而生，通过根据各个客户端独特的数据分布定制模型。尽管具有潜力，现有的PFL方法通常忽视了来自不同计算能力客户端的模型和数据异质性共存的问题。为克服这一限制，我们提出了一种名为渐进参数对齐（FedPPA）的新方法，该方法逐步使客户端常见层的权重与全局模型的权重对齐。我们的方法不仅在客户端更新期间缓解了全局和局部模型之间的一致性问题，而且还保留了客户端的本地知识，从而在非-IID环境中增强个性化鲁棒性。为了进一步提高全局模型性能并保持强大的个性化，我们在FedPPA框架中整合了基于熵加权平均的方法。在包括MNIST、FMNIST和CIFAR-10的三个图像分类数据集上的实验表明，FedPPA在个性化适应方面始终优于现有FL算法，表现出更好的性能。 

---
# xLLM Technical Report 

**Title (ZH)**: xLLM 技术报告 

**Authors**: Tongxuan Liu, Tao Peng, Peijun Yang, Xiaoyang Zhao, Xiusheng Lu, Weizhe Huang, Zirui Liu, Xiaoyu Chen, Zhiwei Liang, Jun Xiong, Donghe Jin, Minchao Zhang, Jinrong Guo, Yingxu Deng, Xu Zhang, Xianzhe Dong, Siqi Wang, Siyu Wu, Yu Wu, Zihan Tang, Yuting Zeng, Yanshu Wang, Jinguang Liu, Meng Kang, Menxin Li, Yunlong Wang, Yiming Liu, Xiaolong Ma, Yifan Wang, Yichen Zhang, Jinrun Yin, Keyang Zheng, Jiawei Yin, Jun Zhang, Ziyue Wang, Xiaobo Lin, Liangyu Liu, Liwei Lan, Yang Liu, Chunhua Peng, Han Liu, Songcheng Ren, Xuezhu Wang, Yunheng Shen, Yi Wang, Guyue Liu, Hui Chen, Tong Yang, Hailong Yang, Jing Li, Guiguang Ding, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14686)  

**Abstract**: We introduce xLLM, an intelligent and efficient Large Language Model (LLM) inference framework designed for high-performance, large-scale enterprise-grade serving, with deep optimizations for diverse AI accelerators. To address these challenges, xLLM builds a novel decoupled service-engine architecture. At the service layer, xLLM-Service features an intelligent scheduling module that efficiently processes multimodal requests and co-locates online and offline tasks through unified elastic scheduling to maximize cluster utilization. This module also relies on a workload-adaptive dynamic Prefill-Decode (PD) disaggregation policy and a novel Encode-Prefill-Decode (EPD) disaggregation policy designed for multimodal inputs. Furthermore, it incorporates a distributed architecture to provide global KV Cache management and robust fault-tolerant capabilities for high availability. At the engine layer, xLLM-Engine co-optimizes system and algorithm designs to fully saturate computing resources. This is achieved through comprehensive multi-layer execution pipeline optimizations, an adaptive graph mode and an xTensor memory management. xLLM-Engine also further integrates algorithmic enhancements such as optimized speculative decoding and dynamic EPLB, collectively serving to substantially boost throughput and inference efficiency. Extensive evaluations demonstrate that xLLM delivers significantly superior performance and resource efficiency. Under identical TPOT constraints, xLLM achieves throughput up to 1.7x that of MindIE and 2.2x that of vLLM-Ascend with Qwen-series models, while maintaining an average throughput of 1.7x that of MindIE with Deepseek-series models. xLLM framework is publicly available at this https URL and this https URL. 

**Abstract (ZH)**: xLLM：一种面向高性能大规模企业级服务的智能高效大语言模型推理框架 

---
# When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks 

**Title (ZH)**: 规划者遇现实：学习到的反应式交通代理如何影响nuPlan基准测试 

**Authors**: Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache  

**Link**: [PDF](https://arxiv.org/pdf/2510.14677)  

**Abstract**: Planner evaluation in closed-loop simulation often uses rule-based traffic agents, whose simplistic and passive behavior can hide planner deficiencies and bias rankings. Widely used IDM agents simply follow a lead vehicle and cannot react to vehicles in adjacent lanes, hindering tests of complex interaction capabilities. We address this issue by integrating the state-of-the-art learned traffic agent model SMART into nuPlan. Thus, we are the first to evaluate planners under more realistic conditions and quantify how conclusions shift when narrowing the sim-to-real gap. Our analysis covers 14 recent planners and established baselines and shows that IDM-based simulation overestimates planning performance: nearly all scores deteriorate. In contrast, many planners interact better than previously assumed and even improve in multi-lane, interaction-heavy scenarios like lane changes or turns. Methods trained in closed-loop demonstrate the best and most stable driving performance. However, when reaching their limits in augmented edge-case scenarios, all learned planners degrade abruptly, whereas rule-based planners maintain reasonable basic behavior. Based on our results, we suggest SMART-reactive simulation as a new standard closed-loop benchmark in nuPlan and release the SMART agents as a drop-in alternative to IDM at this https URL. 

**Abstract (ZH)**: 闭环仿真中计划器评估通常使用基于规则的交通代理，其简单且被动的行为可能掩盖计划器缺陷并偏倚排名。广泛使用的 IDM 代理 merely 仅跟随前车，不能响应相邻车道的车辆，阻碍了对复杂交互能力的测试。我们通过将最先进的学习交通代理模型 SMART 集成到 nuPlan 中解决了这一问题，从而成为第一个在更现实条件下评估计划器并在缩小仿真实际差距时量化结论变化的研究。我们的分析涵盖14个近期计划器和基准，表明基于 IDM 的仿真高估了计划性能：几乎所有分数都下降了。相反，许多计划器的交互能力优于之前假设的，在多车道、交互密集的场景（如变道或转弯）中甚至有所提升。在闭环中训练的方法展示出了最佳且最稳定的驾驶性能。然而，当在扩展的边缘情况下达到极限时，所有学习的计划器会突然退化，而基于规则的计划器则保持基本的合理行为。基于我们的结果，我们建议将 SMART 反应式仿真作为 nuPlan 中新的标准闭环基准，并在此 <https://> 释放 SMART 代理作为 IDM 的现成替代品。 

---
# An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs 

**Title (ZH)**: 基于评分 rubric 的高效生成验证器用于搜索增强的大语言模型 

**Authors**: Linyue Ma, Yilong Xu, Xiang Long, Zhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14660)  

**Abstract**: Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs. 

**Abstract (ZH)**: 基于检索的微调单元作为评价标准：一种统一且可验证的框架enhancing search-augmented large language models with a unified and verifiable paradigm: "nugget-as-rubric" 

---
# Galaxy Morphology Classification with Counterfactual Explanation 

**Title (ZH)**: 银河系形态分类与反事实解释 

**Authors**: Zhuo Cao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14655)  

**Abstract**: Galaxy morphologies play an essential role in the study of the evolution of galaxies. The determination of morphologies is laborious for a large amount of data giving rise to machine learning-based approaches. Unfortunately, most of these approaches offer no insight into how the model works and make the results difficult to understand and explain. We here propose to extend a classical encoder-decoder architecture with invertible flow, allowing us to not only obtain a good predictive performance but also provide additional information about the decision process with counterfactual explanations. 

**Abstract (ZH)**: galaxies的形态在研究星系演化中发挥着重要作用。形态的确定对于大量数据来说是劳动密集型的工作，因此导致了基于机器学习的方法。然而，大多数这些方法未能提供模型工作原理的见解，使结果难以理解和解释。我们提出将经典编码解码架构扩展为包含可逆流的架构，从而不仅获得良好的预测性能，还能通过事实假设解释提供决策过程的附加信息。 

---
# In-Context Learning with Unpaired Clips for Instruction-based Video Editing 

**Title (ZH)**: 基于指令的视频编辑中未配对剪辑的上下文学习 

**Authors**: Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, Guosheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14648)  

**Abstract**: Despite the rapid progress of instruction-based image editing, its extension to video remains underexplored, primarily due to the prohibitive cost and complexity of constructing large-scale paired video editing datasets. To address this challenge, we introduce a low-cost pretraining strategy for instruction-based video editing that leverages in-context learning from unpaired video clips. We show that pretraining a foundation video generation model with this strategy endows it with general editing capabilities, such as adding, replacing, or deleting operations, according to input editing instructions. The pretrained model can then be efficiently refined with a small amount of high-quality paired editing data. Built upon HunyuanVideoT2V, our framework first pretrains on approximately 1M real video clips to learn basic editing concepts, and subsequently fine-tunes on fewer than 150k curated editing pairs to extend more editing tasks and improve the editing quality. Comparative experiments show that our method surpasses existing instruction-based video editing approaches in both instruction alignment and visual fidelity, achieving a 12\% improvement in editing instruction following and a 15\% improvement in editing quality. 

**Abstract (ZH)**: 尽管基于指令的图像编辑取得了 rapid progress，其在视频编辑领域的扩展仍鲜有探索，主要原因是构建大规模配对视频编辑数据集的成本高且复杂。为解决这一挑战，我们提出了一种基于指令的低成本预训练策略，利用非配对视频片段的上下文学习进行训练。我们展示了使用该策略预训练基础视频生成模型，使其获得根据输入编辑指令执行添加、替换或删除等通用编辑能力。预训练后的模型可以使用少量高质量的配对编辑数据进行高效调优。我们的框架基于HunyuanVideoT2V，首先在约100万真实视频片段上进行预训练以学习基本编辑概念，然后使用不到15万精心挑选的编辑配对进行微调，以扩展更多编辑任务并提高编辑质量。对比实验显示，我们的方法在指令对齐和视觉保真度方面均优于现有基于指令的视频编辑方法，在编辑指令跟随上提升了12%，在编辑质量上提升了15%。 

---
# The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain 

**Title (ZH)**: Polygon区块链中的投标游戏：面向MEV提取的强化学习 

**Authors**: Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.14642)  

**Abstract**: In blockchain networks, the strategic ordering of transactions within blocks has emerged as a significant source of profit extraction, known as Maximal Extractable Value (MEV). The transition from spam-based Priority Gas Auctions to structured auction mechanisms like Polygon Atlas has transformed MEV extraction from public bidding wars into sealed-bid competitions under extreme time constraints. While this shift reduces network congestion, it introduces complex strategic challenges where searchers must make optimal bidding decisions within a sub-second window without knowledge of competitor behavior or presence. Traditional game-theoretic approaches struggle in this high-frequency, partially observable environment due to their reliance on complete information and static equilibrium assumptions. We present a reinforcement learning framework for MEV extraction on Polygon Atlas and make three contributions: (1) A novel simulation environment that accurately models the stochastic arrival of arbitrage opportunities and probabilistic competition in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time constraints, capable of adaptive strategy formulation in continuous action spaces while maintaining production-ready inference speeds; (3) Empirical validation demonstrating our history-conditioned agent captures 49\% of available profits when deployed alongside existing searchers and 81\% when replacing the market leader, significantly outperforming static bidding strategies. Our work establishes that reinforcement learning provides a critical advantage in high-frequency MEV environments where traditional optimization methods fail, offering immediate value for industrial participants and protocol designers alike. 

**Abstract (ZH)**: 在区块链网络中，区块内交易的战略排序已成为一种重要的利润提取来源，称为最大可提取价值（MEV）。从基于垃圾交易的优先气体拍卖过渡到结构化拍卖机制（如Polygon Atlas），将MEV提取从公开竞价转变为在极端时间限制下的密封出价竞争。虽然这种转变减少了网络拥堵，但也引入了复杂的战略挑战，要求搜索者在不到一秒钟的时间内做出最优竞价决策，且缺乏对竞争对手行为或存在性的了解。传统博弈论方法在这种高频次、部分可观测的环境中难以发挥作用，因为它们依赖于完全信息和静态均衡假设。本文提出了一种针对Polygon Atlas的增强学习框架，并做出如下贡献：（1）一种新颖的仿真环境，准确模拟 Arbitrage 机会的随机到达和Atlas 拍卖中的概率性竞争；（2）一种基于PPO的竞价代理，优化了实时约束条件，能够在连续动作空间中形成自适应策略，并保持生产就绪的推理速度；（3）实证验证表明，当部署在现有搜索者旁边时，我们的历史条件代理捕获了49%的可用利润，而取代市场领导者时则捕获了81%的利润，显著优于静态竞价策略。我们的工作证明了增强学习在传统优化方法失效的高频MEV环境中提供了关键优势，为工业参与者和协议设计师提供了即时价值。 

---
# Causality Enhancement for Cross-Domain Recommendation 

**Title (ZH)**: 跨域推荐中的因果性增强 

**Authors**: Zhibo Wu, Yunfan Wu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14641)  

**Abstract**: Cross-domain recommendation forms a crucial component in recommendation systems. It leverages auxiliary information through source domain tasks or features to enhance target domain recommendations. However, incorporating inconsistent source domain tasks may result in insufficient cross-domain modeling or negative transfer. While incorporating source domain features without considering the underlying causal relationships may limit their contribution to final predictions. Thus, a natural idea is to directly train a cross-domain representation on a causality-labeled dataset from the source to target domain. Yet this direction has been rarely explored, as identifying unbiased real causal labels is highly challenging in real-world scenarios. In this work, we attempt to take a first step in this direction by proposing a causality-enhanced framework, named CE-CDR. Specifically, we first reformulate the cross-domain recommendation as a causal graph for principled guidance. We then construct a causality-aware dataset heuristically. Subsequently, we derive a theoretically unbiased Partial Label Causal Loss to generalize beyond the biased causality-aware dataset to unseen cross-domain patterns, yielding an enriched cross-domain representation, which is then fed into the target model to enhance target-domain recommendations. Theoretical and empirical analyses, as well as extensive experiments, demonstrate the rationality and effectiveness of CE-CDR and its general applicability as a model-agnostic plugin. Moreover, it has been deployed in production since April 2025, showing its practical value in real-world applications. 

**Abstract (ZH)**: 因果增强跨域推荐框架：CE-CDR 

---
# RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF 

**Title (ZH)**: RLAIF-SPA: 基于RLAIF优化的情感语音合成方法 

**Authors**: Qing Yang, Zhenghao Liu, Junxin Wang, Yangfan Du, Pengcheng Huang, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14628)  

**Abstract**: Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation. 

**Abstract (ZH)**: 文本到语音合成在中性语音方面已接近人类质量，但情感表达性仍是一项挑战。现有方法often依赖昂贵的情感标注或优化未能捕捉语音情感表达性和感知自然度的间接目标，导致生成的语音准确但情感平淡。为应对这些挑战，我们提出了RLAIF-SPA框架，结合强化学习从AI反馈（RLAIF）机制，利用自动语音识别（ASR）和大型语言模型（LLM）技术分别判断语义准确性和韵律-情感标签对齐，作为情感表达性和可理解性的直接奖励。具体而言，该框架通过联合考虑语义准确性和韵律-情感对齐的四个精细维度：结构、情感、速度和音调，来增强表达质量。此外，该框架还整合了语义准确性的反馈，以确保生成清晰准确的语音。实验结果表明，RLAIF-SPA在LibriSpeech数据集上的表现优于Chat-TTS，WER降低了26.1%，SIM-O提高了9.1%，并且在人类评估中提高了超过10%。 

---
# GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation 

**Title (ZH)**: GemiRec：兴趣量化与生成的多兴趣推荐 

**Authors**: Zhibo Wu, Yunfan Wu, Quan Liu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14626)  

**Abstract**: Multi-interest recommendation has gained attention, especially in industrial retrieval stage. Unlike classical dual-tower methods, it generates multiple user representations instead of a single one to model comprehensive user interests. However, prior studies have identified two underlying limitations: the first is interest collapse, where multiple representations homogenize. The second is insufficient modeling of interest evolution, as they struggle to capture latent interests absent from a user's historical behavior. We begin with a thorough review of existing works in tackling these limitations. Then, we attempt to tackle these limitations from a new perspective. Specifically, we propose a framework-level refinement for multi-interest recommendation, named GemiRec. The proposed framework leverages interest quantization to enforce a structural interest separation and interest generation to learn the evolving dynamics of user interests explicitly. It comprises three modules: (a) Interest Dictionary Maintenance Module (IDMM) maintains a shared quantized interest dictionary. (b) Multi-Interest Posterior Distribution Module (MIPDM) employs a generative model to capture the distribution of user future interests. (c) Multi-Interest Retrieval Module (MIRM) retrieves items using multiple user-interest representations. Both theoretical and empirical analyses, as well as extensive experiments, demonstrate its advantages and effectiveness. Moreover, it has been deployed in production since March 2025, showing its practical value in industrial applications. 

**Abstract (ZH)**: 多兴趣推荐在工业检索阶段获得了关注。不同于经典的双塔方法，它生成多个用户表示而不是单个表示以建模用户的全方位兴趣。然而，先前的研究发现了两个潜在的局限性：一是兴趣坍缩，多个表示变得同质化。二是兴趣演化建模不足，难以捕捉用户历史行为中不存在的潜在兴趣。我们首先对现有工作的这些局限性进行了详尽的回顾。然后，我们尝试从一个新的视角来解决这些局限性。具体而言，我们提出了一种针对多兴趣推荐的框架级改进框架，称为GemiRec。该框架利用兴趣量化来强制构建结构化兴趣分离，并利用生成模型来明确学习用户兴趣的演化动态。它包括三个模块：(a) 兴趣字典维护模块 (IDMM) 维护一个共享的兴趣量化词典。(b) 多兴趣后验分布模块 (MIPDM) 求取用户的未来兴趣分布。(c) 多兴趣检索模块 (MIRM) 使用多个用户兴趣表示进行项目检索。理论和实证分析以及广泛实验表明其优势和有效性。此外，自2025年3月起已在生产中部署，展示了其在工业应用中的实用价值。 

---
# LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching 

**Title (ZH)**: LeapFactual: 可靠的基于条件流匹配的视觉反事实解释 

**Authors**: Zhuo Cao, Xuan Zhao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14623)  

**Abstract**: The growing integration of machine learning (ML) and artificial intelligence (AI) models into high-stakes domains such as healthcare and scientific research calls for models that are not only accurate but also interpretable. Among the existing explainable methods, counterfactual explanations offer interpretability by identifying minimal changes to inputs that would alter a model's prediction, thus providing deeper insights. However, current counterfactual generation methods suffer from critical limitations, including gradient vanishing, discontinuous latent spaces, and an overreliance on the alignment between learned and true decision boundaries. To overcome these limitations, we propose LeapFactual, a novel counterfactual explanation algorithm based on conditional flow matching. LeapFactual generates reliable and informative counterfactuals, even when true and learned decision boundaries diverge. Following a model-agnostic approach, LeapFactual is not limited to models with differentiable loss functions. It can even handle human-in-the-loop systems, expanding the scope of counterfactual explanations to domains that require the participation of human annotators, such as citizen science. We provide extensive experiments on benchmark and real-world datasets showing that LeapFactual generates accurate and in-distribution counterfactual explanations that offer actionable insights. We observe, for instance, that our reliable counterfactual samples with labels aligning to ground truth can be beneficially used as new training data to enhance the model. The proposed method is broadly applicable and enhances both scientific knowledge discovery and non-expert interpretability. 

**Abstract (ZH)**: 机器学习和人工智能在高 stakes 领域如医疗和科学研究中的日益集成呼唤着不仅准确而且可解释的模型。现有的可解释方法中，因果解释通过识别最小输入变化来改变模型预测，提供了深入的洞察，但当前的因果解释生成方法存在梯度消失、非连续的潜在空间以及对学习和真实决策边界的对齐过度依赖的关键限制。为了克服这些限制，我们提出了基于条件流匹配的 LeapFactual，这是一种新颖的因果解释算法。LeapFactual 即使在真实和学习的决策边界存在分歧时也能生成可靠和信息丰富的因果解释。遵循模型无关的方法，LeapFactual 不局限于具有可微损失函数的模型。它甚至可以处理人机交互系统，将因果解释的范围扩展到需要人类注释员参与的领域，例如公民科学。我们在基准数据集和真实世界数据集上的广泛实验表明，LeapFactual 生成了准确且在分布内的因果解释，提供了可操作的洞察。例如，我们可靠且标签对齐的因果样本可以有益地用作新的训练数据以增强模型。所提出的方法具有广泛的应用性，提升了科学知识发现和非专家可解释性。 

---
# Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models 

**Title (ZH)**: 代码驱动的数字序列计算：增强大型语言模型的归纳推理能力 

**Authors**: Kedi Chen, Zhikai Lei, Xu Guo, Xuecheng Wu, Siyuan Zeng, Jianghao Yin, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14620)  

**Abstract**: Large language models (LLMs) make remarkable progress in reasoning tasks. Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest. However, research on inductive reasoning faces certain challenges. First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns. Second, current works merely prompt LLMs or finetune on simple prompt-response pairs, but do not provide precise thinking processes nor implement difficulty control. Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking. Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures. Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models' OOD performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理任务中取得了显著进步。在不同的推理模式中，由于归纳推理更符合人类学习的特点，因此引发了越来越多的关注。然而，对归纳推理的研究也面临一些挑战。首先，现有的归纳数据主要集中在表面规律，缺乏更复杂的内部模式。其次，当前的工作只是通过简单提示或微调LLM，但并未提供精确的思维过程，也未实施难度控制。不同于以往的工作，我们通过引入\textit{CodeSeq}，一个源自数字序列的合成后训练数据集，来应对这些挑战。我们将数字序列打包成算法问题，以发现其通用项，相应地定义了一个通用项生成（GTG）任务。我们的流水线通过反思失败的测试案例并结合迭代修正来生成监督微调数据，从而教会LLM自主案例生成和自我检查的能力。此外，该流水线利用基于问题通过率评估解题能力和自我指导案例生成成功率的新颖实例协同可解决性奖励强化学习，使模型能够从成功和失败中更有效地学习。实验结果表明，使用\textit{CodeSeq}训练的模型在各种推理任务中表现更好，并能保持模型的OOD性能。 

---
# Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures 

**Title (ZH)**: 超越正确性：跨文化评价主观写作偏好 

**Authors**: Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, Letian Ni, Yuyang Cheng, Qiguang Chen, Jingzhe Ding, Shengda Long, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Ge Zhang, Wenhao Huang, Wanxiang Che, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14616)  

**Abstract**: Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification. 

**Abstract (ZH)**: 当前的偏好学习方法在标准基准上实现了高准确度，但在移除客观质量信号时表现出显著性能下降。我们介绍了WritingPreferenceBench数据集，包含1800个人工标注的偏好配对（1200个英语，600个中文），覆盖8种创造性写作体裁，其中响应在客观正确性、事实准确性和长度方面相配。在这个基准上，基于序列的奖励模型——RLHF的标准架构——仅实现了52.7%的平均准确度，而零样本语言模型评估器则达到了53.9%。相比之下，生成式奖励模型产生显式推理链的准确度达到了81.8%。我们观察到高模型内部变异性：个体模型在不同写作类别中的准确率范围从18.2%到81.8%，平均标准差为10.1%。这种变异性在不同模型规模下依然存在，27B参数模型并未显示一致性的改进效果。我们的结果表明，当前的RLHF方法主要学会了检测客观错误而非捕捉主观质量偏好（如创意、风格和情感共鸣），而成功的偏好建模可能需要中间推理表示而非直接分类。 

---
# An Active Inference Model of Mouse Point-and-Click Behaviour 

**Title (ZH)**: 小鼠点选行为的主动推断模型 

**Authors**: Markus Klar, Sebastian Stein, Fraser Paterson, John H. Williamson, Roderick Murray-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.14611)  

**Abstract**: We explore the use of Active Inference (AIF) as a computational user model for spatial pointing, a key problem in Human-Computer Interaction (HCI). We present an AIF agent with continuous state, action, and observation spaces, performing one-dimensional mouse pointing and clicking. We use a simple underlying dynamic system to model the mouse cursor dynamics with realistic perceptual delay. In contrast to previous optimal feedback control-based models, the agent's actions are selected by minimizing Expected Free Energy, solely based on preference distributions over percepts, such as observing clicking a button correctly. Our results show that the agent creates plausible pointing movements and clicks when the cursor is over the target, with similar end-point variance to human users. In contrast to other models of pointing, we incorporate fully probabilistic, predictive delay compensation into the agent. The agent shows distinct behaviour for differing target difficulties without the need to retune system parameters, as done in other approaches. We discuss the simulation results and emphasize the challenges in identifying the correct configuration of an AIF agent interacting with continuous systems. 

**Abstract (ZH)**: 我们探索将主动推断（AIF）作为计算用户模型应用于空间指点的问题，这是人机交互（HCI）中的一个关键问题。我们呈现了一个拥有连续状态、动作和观测空间的AIF代理，执行一维鼠标指点和点击任务。我们使用简单的动态系统来模拟鼠标光标的动力学，并考虑了现实的感知延迟。与基于最优反馈控制的模型不同，代理的动作是通过最小化预期自由能来选择的，仅基于对感知（如正确点击按钮）的偏好分布。我们的结果显示，当鼠标光标位于目标上时，代理能够产生合理的指点运动和点击，其端点变异量与人类用户类似。与其它指点模型不同，我们为代理引入了完整的概率预测延迟补偿。代理能够表现出不同的行为特征以应对不同的目标难度，无需重新调整系统参数。我们讨论了模拟结果，并强调了确定与连续系统交互的AIF代理正确配置所面临的挑战。 

---
# Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering 

**Title (ZH)**: 基于知识的多模态视觉问题解答与检索过滤 

**Authors**: Yuyang Hong, Jiaqi Gu, Qi Yang, Lubin Fan, Yue Wu, Ying Wang, Kun Ding, Shiming Xiang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.14605)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) requires visual language models (VLMs) to integrate visual understanding with external knowledge retrieval. Although retrieval-augmented generation (RAG) achieves significant advances in this task by combining knowledge-base querying, it still struggles with the quality of multimodal queries and the relevance of retrieved results. To overcome these challenges, we propose a novel three-stage method, termed Wiki-PRF, including Processing, Retrieval and Filtering stages. The processing stage dynamically invokes visual tools to extract precise multimodal information for retrieval. The retrieval stage integrates visual and text features to achieve multimodal knowledge retrieval. The filtering stage performs relevance filtering and concentration on retrieval results. To this end, we introduce a visual language model trained with answer accuracy and format consistency as reward signals via a reinforcement learning manner. This enhances the model's reasoning, tool invocation for accurate queries, and filtering of irrelevant content. Experiments on benchmark datasets (E-VQA and InfoSeek) show significant improvements~(36.0 and 42.8) in answer quality, achieving state-of-the-art performance. Code is available at this https URL 

**Abstract (ZH)**: 基于知识的视觉问答（KB-VQA）要求视觉语言模型（VLMs）将视觉理解与外部知识检索结合。尽管检索增强生成（RAG）通过结合知识库查询在这一任务中取得了显著进展，但仍然在多模态查询的质量和检索结果的相关性方面存在挑战。为解决这些挑战，我们提出了一种新颖的三阶段方法，称为Wiki-PRF，包括处理、检索和筛选阶段。处理阶段动态调用视觉工具以提取用于检索的精确多模态信息。检索阶段结合视觉和文本特征以实现多模态知识检索。筛选阶段执行相关性筛选并集中关注检索结果。为此，我们通过强化学习的方式引入了一种视觉语言模型，该模型以答案准确性与格式一致性作为奖励信号进行训练。这提升了模型的推理能力、准确查询的工具调用以及无关内容的筛选。在基准数据集（E-VQA和InfoSeek）上的实验显示，答案质量有了显著提高（分别提高了36.0和42.8个百分点），达到了最佳性能。代码已发布于此网址。 

---
# Just-In-Time Objectives: A General Approach for Specialized AI Interactions 

**Title (ZH)**: 及时目标：专用于AI交互的通用方法 

**Authors**: Michelle S. Lam, Omar Shaikh, Hallie Xu, Alice Guo, Diyi Yang, Jeffrey Heer, James A. Landay, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.14591)  

**Abstract**: Large language models promise a broad set of functions, but when not given a specific objective, they default to milquetoast results such as drafting emails littered with cliches. We demonstrate that inferring the user's in-the-moment objective, then rapidly optimizing for that singular objective, enables LLMs to produce tools, interfaces, and responses that are more responsive and desired. We contribute an architecture for automatically inducing just-in-time objectives by passively observing user behavior, then steering downstream AI systems through generation and evaluation against this objective. Inducing just-in-time objectives (e.g., "Clarify the abstract's research contribution") enables automatic generation of tools, e.g., those that critique a draft based on relevant HCI methodologies, anticipate related researchers' reactions, or surface ambiguous terminology. In a series of experiments (N=14, N=205) on participants' own tasks, JIT objectives enable LLM outputs that achieve 66-86% win rates over typical LLMs, and in-person use sessions (N=17) confirm that JIT objectives produce specialized tools unique to each participant. 

**Abstract (ZH)**: 基于即时目标的大语言模型能够生成更加响应和受用户欢迎的结果：一种自动诱导即时目标的架构 

---
# STANCE: Motion Coherent Video Generation Via Sparse-to-Dense Anchored Encoding 

**Title (ZH)**: stance: 通过稀疏到稠密锚定编码实现运动连贯的视频生成 

**Authors**: Zhifei Chen, Tianshuo Xu, Leyi Wu, Luozhou Wang, Dongyu Yan, Zihan You, Wenting Luo, Guo Zhang, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14588)  

**Abstract**: Video generation has recently made striking visual progress, but maintaining coherent object motion and interactions remains difficult. We trace two practical bottlenecks: (i) human-provided motion hints (e.g., small 2D maps) often collapse to too few effective tokens after encoding, weakening guidance; and (ii) optimizing for appearance and motion in a single head can favor texture over temporal consistency. We present STANCE, an image-to-video framework that addresses both issues with two simple components. First, we introduce Instance Cues -- a pixel-aligned control signal that turns sparse, user-editable hints into a dense 2.5D (camera-relative) motion field by averaging per-instance flow and augmenting with monocular depth over the instance mask. This reduces depth ambiguity compared to 2D arrow inputs while remaining easy to use. Second, we preserve the salience of these cues in token space with Dense RoPE, which tags a small set of motion tokens (anchored on the first frame) with spatial-addressable rotary embeddings. Paired with joint RGB \(+\) auxiliary-map prediction (segmentation or depth), our model anchors structure while RGB handles appearance, stabilizing optimization and improving temporal coherence without requiring per-frame trajectory scripts. 

**Abstract (ZH)**: 基于实例的线索和密集RoPE的视频生成框架：解决连贯对象运动和交互的问题 

---
# Local Causal Discovery for Statistically Efficient Causal Inference 

**Title (ZH)**: 局部因果发现以实现统计效率的因果推断 

**Authors**: Mátyás Schubert, Tom Claassen, Sara Magliacane  

**Link**: [PDF](https://arxiv.org/pdf/2510.14582)  

**Abstract**: Causal discovery methods can identify valid adjustment sets for causal effect estimation for a pair of target variables, even when the underlying causal graph is unknown. Global causal discovery methods focus on learning the whole causal graph and therefore enable the recovery of optimal adjustment sets, i.e., sets with the lowest asymptotic variance, but they quickly become computationally prohibitive as the number of variables grows. Local causal discovery methods offer a more scalable alternative by focusing on the local neighborhood of the target variables, but are restricted to statistically suboptimal adjustment sets. In this work, we propose Local Optimal Adjustments Discovery (LOAD), a sound and complete causal discovery approach that combines the computational efficiency of local methods with the statistical optimality of global methods. First, LOAD identifies the causal relation between the targets and tests if the causal effect is identifiable by using only local information. If it is identifiable, it then finds the optimal adjustment set by leveraging local causal discovery to infer the mediators and their parents. Otherwise, it returns the locally valid parent adjustment sets based on the learned local structure. In our experiments on synthetic and realistic data LOAD outperforms global methods in scalability, while providing more accurate effect estimation than local methods. 

**Abstract (ZH)**: 局部最优调整发现（LOAD）：结合局部方法的高效性和全局方法的统计最优性进行因果发现 

---
# Selective Labeling with False Discovery Rate Control 

**Title (ZH)**: 选择性标注与假发现率控制 

**Authors**: Huipeng Huang, Wenbo Liao, Huajun Xi, Hao Zeng, Mengchen Zhao, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14581)  

**Abstract**: Obtaining high-quality labels for large datasets is expensive, requiring massive annotations from human experts. While AI models offer a cost-effective alternative by predicting labels, their label quality is compromised by the unavoidable labeling errors. Existing methods mitigate this issue through selective labeling, where AI labels a subset and human labels the remainder. However, these methods lack theoretical guarantees on the quality of AI-assigned labels, often resulting in unacceptably high labeling error within the AI-labeled subset. To address this, we introduce \textbf{Conformal Labeling}, a novel method to identify instances where AI predictions can be provably trusted. This is achieved by controlling the false discovery rate (FDR), the proportion of incorrect labels within the selected subset. In particular, we construct a conformal $p$-value for each test instance by comparing AI models' predicted confidence to those of calibration instances mislabeled by AI models. Then, we select test instances whose $p$-values are below a data-dependent threshold, certifying AI models' predictions as trustworthy. We provide theoretical guarantees that Conformal Labeling controls the FDR below the nominal level, ensuring that a predefined fraction of AI-assigned labels is correct on average. Extensive experiments demonstrate that our method achieves tight FDR control with high power across various tasks, including image and text labeling, and LLM QA. 

**Abstract (ZH)**: Conformal Labeling：通过控制错误发现率来证明AI预测的可靠性 

---
# Agentic Entropy-Balanced Policy Optimization 

**Title (ZH)**: 代理熵平衡策略优化 

**Authors**: Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14545)  

**Abstract**: Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training. 

**Abstract (ZH)**: 近来，代理强化学习（Agentic RL）在激励多轮、长_horizon_工具使用能力方面取得了显著进展。然而，主流代理强化学习算法在熵的指导下自主探索高不确定性工具调用步骤时，过度依赖熵信号可能导致进一步的约束，从而导致训练崩溃。本文探讨了熵带来的挑战，并提出代理平衡熵策略优化（AEPO），这是一种设计用于在展开和策略更新阶段平衡熵的代理强化学习算法。AEPO 包含两个核心组件：（1）动态平衡熵展开机制，通过熵预监测动态分配全局和分支采样预算，同时对连续的高熵工具调用步骤施加分支惩罚以防止过度分支问题；（2）平衡熵策略优化，通过在高熵剪辑项中插入止梯度操作来保留和适当缩放高熵标记的梯度，并结合熵感知的优势估计来优先学习高不确定性标记。在14个具有挑战性的数据集上的结果表明，AEPO 始终优于7种主流RL算法。仅使用1K RL样本，应用AEPO的Qwen3-14B在GAIA、Humanity's Last Exam和WebWalker上的性能分别为47.6%、11.2%和43.0%（Pass@1）；分别为65.0%、26.0%和70.0%（Pass@5）。进一步的分析表明，AEPO 在保持策略熵稳定的同时提高了展开采样的多样性，促进了可扩展的网页代理训练。 

---
# Real-Time Surgical Instrument Defect Detection via Non-Destructive Testing 

**Title (ZH)**: 基于非destructive testing的实时手术器械缺陷检测 

**Authors**: Qurrat Ul Ain, Atif Aftab Ahmed Jilani, Zunaira Shafqat, Nigar Azhar Butt  

**Link**: [PDF](https://arxiv.org/pdf/2510.14525)  

**Abstract**: Defective surgical instruments pose serious risks to sterility, mechanical integrity, and patient safety, increasing the likelihood of surgical complications. However, quality control in surgical instrument manufacturing often relies on manual inspection, which is prone to human error and inconsistency. This study introduces SurgScan, an AI-powered defect detection framework for surgical instruments. Using YOLOv8, SurgScan classifies defects in real-time, ensuring high accuracy and industrial scalability. The model is trained on a high-resolution dataset of 102,876 images, covering 11 instrument types and five major defect categories. Extensive evaluation against state-of-the-art CNN architectures confirms that SurgScan achieves the highest accuracy (99.3%) with real-time inference speeds of 4.2-5.8 ms per image, making it suitable for industrial deployment. Statistical analysis demonstrates that contrast-enhanced preprocessing significantly improves defect detection, addressing key limitations in visual inspection. SurgScan provides a scalable, cost-effective AI solution for automated quality control, reducing reliance on manual inspection while ensuring compliance with ISO 13485 and FDA standards, paving the way for enhanced defect detection in medical manufacturing. 

**Abstract (ZH)**: 缺陷手术器械对无菌性、机械完整性和患者安全构成严重风险，增加手术并发症的可能性。然而，手术器械制造中的质量控制往往依赖于人工检查，容易出现人为错误和不一致性。本文介绍了一种基于AI的手术器械缺陷检测框架SurgScan。利用YOLOv8，SurgScan实现了实时缺陷分类，保证了高精度和工业可扩展性。该模型在102,876张高分辨率图像的数据集上进行训练，涵盖了11种器械类型和五大主要缺陷类别。与最先进的CNN架构进行广泛评估证实，SurgScan在保持99.3%精度的同时，实现了每张图像4.2-5.8毫秒的实时推理速度，使其适合工业部署。统计分析表明，对比增强预处理显著提高了缺陷检测效果，解决了视觉检查的关键局限性。SurgScan提供了一种可扩展、低成本的AI解决方案，用于自动化质量控制，减少对人工检查的依赖，同时确保符合ISO 13485和FDA标准，为医疗制造业中的缺陷检测提升铺平了道路。 

---
# State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living 

**Title (ZH)**: 表达您的意图以引导注意力：一个旨在促进有意数字生活的AI助理 

**Authors**: Juheon Choi, Juyoung Lee, Jian Kim, Chanyoung Kim, Taewon Min, W. Bradley Knox, Min Kyung Lee, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14513)  

**Abstract**: When working on digital devices, people often face distractions that can lead to a decline in productivity and efficiency, as well as negative psychological and emotional impacts. To address this challenge, we introduce a novel Artificial Intelligence (AI) assistant that elicits a user's intention, assesses whether ongoing activities are in line with that intention, and provides gentle nudges when deviations occur. The system leverages a large language model to analyze screenshots, application titles, and URLs, issuing notifications when behavior diverges from the stated goal. Its detection accuracy is refined through initial clarification dialogues and continuous user feedback. In a three-week, within-subjects field deployment with 22 participants, we compared our assistant to both a rule-based intent reminder system and a passive baseline that only logged activity. Results indicate that our AI assistant effectively supports users in maintaining focus and aligning their digital behavior with their intentions. Our source code is publicly available at this url this https URL 

**Abstract (ZH)**: 在使用数字设备时，人们往往会面临干扰，这可能导致生产力和效率下降，并产生负面的心理和情绪影响。为应对这一挑战，我们提出一种新型人工智能（AI）助手，该助手能够感知用户意图，评估正在进行的活动是否符合该意图，并在偏离时提供温和的提示。该系统利用大规模语言模型分析屏幕截图、应用程序标题和URL，在行为偏离既定目标时发出通知。其检测准确性通过初步澄清对话和持续的用户反馈进行优化。在为期三周的单被试设计实地部署中，我们分别与基于规则的意图提醒系统和仅记录活动的被动基线进行了对比，参与者共计22人。结果表明，我们的AI助手有效地帮助用户保持专注，并使数字行为与意图保持一致。我们的源代码已在此处公开：this https URL 

---
# E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task 

**Title (ZH)**: E2Edev：评估大型语言模型在端到端软件开发任务中的性能 

**Authors**: Jingyao Liu, Chen Huang, Zhizhao Guan, Wenqiang Lei, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14509)  

**Abstract**: E2EDev comprises (i) a fine-grained set of user requirements, (ii) {multiple BDD test scenarios with corresponding Python step implementations for each requirement}, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). {By evaluating various E2ESD frameworks and LLM backbones with E2EDev}, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at this https URL. 

**Abstract (ZH)**: E2EDev 包含（i）细粒度的用户需求集，（ii）多个与每个需求对应的 BDD 测试场景及其 Python 步骤实现，以及（iii）基于 Behave 框架构建的完全自动化测试管道。为了保证其质量并减少标注工作量，E2EDev 利用我们提出的集成人类在环多代理标注框架（HITL-MAA）。通过使用 E2EDev 评估各种端到端标注框架和大语言模型骨干网络，我们的分析表明这些任务的有效解决依旧存在困难，突显了更加有效且成本效益更高的端到端标注解决方案的迫切需求。我们的代码库和基准可在此处公开获取。 

---
# From Guess2Graph: When and How Can Unreliable Experts Safely Boost Causal Discovery in Finite Samples? 

**Title (ZH)**: 从Guess2Graph：不可靠的专家在有限样本中如何安全提升因果发现？ 

**Authors**: Sujai Hiremath, Dominik Janzing, Philipp Faller, Patrick Blöbaum, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, Kyra Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14488)  

**Abstract**: Causal discovery algorithms often perform poorly with limited samples. While integrating expert knowledge (including from LLMs) as constraints promises to improve performance, guarantees for existing methods require perfect predictions or uncertainty estimates, making them unreliable for practical use. We propose the Guess2Graph (G2G) framework, which uses expert guesses to guide the sequence of statistical tests rather than replacing them. This maintains statistical consistency while enabling performance improvements. We develop two instantiations of G2G: PC-Guess, which augments the PC algorithm, and gPC-Guess, a learning-augmented variant designed to better leverage high-quality expert input. Theoretically, both preserve correctness regardless of expert error, with gPC-Guess provably outperforming its non-augmented counterpart in finite samples when experts are "better than random." Empirically, both show monotonic improvement with expert accuracy, with gPC-Guess achieving significantly stronger gains. 

**Abstract (ZH)**: 因果发现算法在样本有限时常常表现不佳。通过将专家知识（包括来自大语言模型的知识）作为约束整合以提高性能的前景令人期待，但现有方法的可靠性依赖于完美的预测或不确定性估计，这使其在实际应用中不可靠。我们提出了Guess2Graph (G2G)框架，该框架利用专家猜测来引导统计测试的顺序，而不仅仅是取代它们。这保持了统计一致性的同时，允许性能提升。我们开发了G2G的两个实例：扩展PC算法的PC-Guess，以及更擅长利用高质量专家输入的可学习增强变体gPC-Guess。理论上，无论专家错误如何，两者均保持正确性，且在专家优于随机的情况下，gPC-Guess在有限样本中的表现优于其非增强版本。实验证明，两者均随专家准确性的提高而逐步提升，其中gPC-Guess获得了显著更大的提升。 

---
# Semantic representations emerge in biologically inspired ensembles of cross-supervising neural networks 

**Title (ZH)**: 生物启发的跨监督神经网络集成中的语义表示 

**Authors**: Roy Urbach, Elad Schneidman  

**Link**: [PDF](https://arxiv.org/pdf/2510.14486)  

**Abstract**: Brains learn to represent information from a large set of stimuli, typically by weak supervision. Unsupervised learning is therefore a natural approach for exploring the design of biological neural networks and their computations. Accordingly, redundancy reduction has been suggested as a prominent design principle of neural encoding, but its ``mechanistic'' biological implementation is unclear. Analogously, unsupervised training of artificial neural networks yields internal representations that allow for accurate stimulus classification or decoding, but typically rely on biologically-implausible implementations. We suggest that interactions between parallel subnetworks in the brain may underlie such learning: we present a model of representation learning by ensembles of neural networks, where each network learns to encode stimuli into an abstract representation space by cross-supervising interactions with other networks, for inputs they receive simultaneously or in close temporal proximity. Aiming for biological plausibility, each network has a small ``receptive field'', thus receiving a fixed part of the external input, and the networks do not share weights. We find that for different types of network architectures, and for both visual or neuronal stimuli, these cross-supervising networks learn semantic representations that are easily decodable and that decoding accuracy is comparable to supervised networks -- both at the level of single networks and the ensemble. We further show that performance is optimal for small receptive fields, and that sparse connectivity between networks is nearly as accurate as all-to-all interactions, with far fewer computations. We thus suggest a sparsely interacting collective of cross-supervising networks as an algorithmic framework for representational learning and collective computation in the brain. 

**Abstract (ZH)**: 大脑通过从大范围刺激中学习表示信息，通常依赖于弱监督。因此，无监督学习是探索生物神经网络设计及其计算的自然方法。相应地，冗余减少被建议为神经编码的一个重要设计原则，但其“机械”生物学实现尚不清楚。类似地，人工神经网络的无监督训练会产生内部表示，允许准确的刺激分类或解码，但通常依赖于不符合生物学实现的方案。我们建议大脑平行子网络之间的相互作用可能底层实现这种学习：我们提出了一种由神经网络集合实现的表示学习模型，其中每个网络通过与其他网络的交叉监督学习将刺激编码到抽象的表示空间中，用于它们同时接收到的输入或接近时间窗口内的输入。为了追求生物可行性，每个网络具有一个小的“感受野”，因此接收外部输入的一部分，并且网络之间不共享权重。我们发现，对于不同类型的网络架构和无论是视觉还是神经元刺激，这些交叉监督的网络都能够学习易于解码的语义表示，解码准确性与监督网络相当——无论是单个网络还是整个网络集合。我们进一步表明，性能在小感受野时最佳，网络传播的稀疏连接几乎与全互连接效果一样准确，并且计算量更少。因此，我们建议一种稀疏交互的交叉监督网络集群作为大脑表示学习和集体计算的算法框架。 

---
# Stealthy Dual-Trigger Backdoors: Attacking Prompt Tuning in LM-Empowered Graph Foundation Models 

**Title (ZH)**: 隐形双触发后门：攻击基于LM赋能图基础模型的提示调优 

**Authors**: Xiaoyu Xue, Yuni Lai, Chenxi Huang, Yulin Zhu, Gaolei Li, Xiaoge Zhang, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14470)  

**Abstract**: The emergence of graph foundation models (GFMs), particularly those incorporating language models (LMs), has revolutionized graph learning and demonstrated remarkable performance on text-attributed graphs (TAGs). However, compared to traditional GNNs, these LM-empowered GFMs introduce unique security vulnerabilities during the unsecured prompt tuning phase that remain understudied in current research. Through empirical investigation, we reveal a significant performance degradation in traditional graph backdoor attacks when operating in attribute-inaccessible constrained TAG systems without explicit trigger node attribute optimization. To address this, we propose a novel dual-trigger backdoor attack framework that operates at both text-level and struct-level, enabling effective attacks without explicit optimization of trigger node text attributes through the strategic utilization of a pre-established text pool. Extensive experimental evaluations demonstrate that our attack maintains superior clean accuracy while achieving outstanding attack success rates, including scenarios with highly concealed single-trigger nodes. Our work highlights critical backdoor risks in web-deployed LM-empowered GFMs and contributes to the development of more robust supervision mechanisms for open-source platforms in the era of foundation models. 

**Abstract (ZH)**: GFMs增强的语言模型驱动图基础模型的安全性研究：双触发器后门攻击框架及其应用 

---
# LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models 

**Title (ZH)**: LiRA： linguistic robust anchoring for cross-lingual large language models 

**Authors**: Haolin Li, Haipeng Zhang, Mang Li, Yaohua Wang, Lijie Wen, Yu Zhang, Biqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14466)  

**Abstract**: As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca's multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaSR. Code will be released on GitHub and the dataset on Hugging Face. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的迅速发展，高资源语言（如英语、中文）的表现接近饱和，而低资源语言（如乌尔都语、泰语）的表现仍然显著较低，原因包括训练数据有限、机器翻译噪声和跨语言对齐不稳定。我们引入了LiRA（ Linguistic Robust Anchoring for Large Language Models），这是一种在低资源条件下增强跨语言表示的同时联合增强检索和推理的训练框架。LiRA 包含两个模块：(i) Arca（锚定表示组合架构），通过基于锚点的对齐和多代理协作编码将低资源语言锚定到英语语义空间中，保持共享嵌入空间中的几何稳定性；(ii) LaSR（语言耦合语义推理器），在其多语言表示之上添加了语言感知的小型推理头，并通过一致性正则化进行统一训练，以增强跨语言理解和推理的鲁棒性。我们还构建并发布了涵盖五个东南亚语言和两个南亚语言的多语言产品检索数据集。在多个低资源基准测试（跨语言检索、语义相似性和推理）中，实验在少量样本和噪声放大条件下显示出一致的改进和鲁棒性；消融实验验证了 Arca 和 LaSR 的贡献。代码将发布在 GitHub 上，数据集将发布在 Hugging Face 上。 

---
# Holdout-Loss-Based Data Selection for LLM Finetuning via In-Context Learning 

**Title (ZH)**: 基于Holdout Loss的数据选择方法：通过上下文学习进行LLM微调 

**Authors**: Ling Zhang, Xianliang Yang, Juwon Yu, Park Cheonyoung, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2510.14459)  

**Abstract**: Fine-tuning large pretrained language models is a common approach for aligning them with human preferences, but noisy or off-target examples can dilute supervision. While small, well-chosen datasets often match the performance of much larger ones, systematic and efficient ways to identify high-value training data remain underexplored. Many current methods rely on heuristics or expensive retraining. We present a theoretically grounded, resource-efficient framework for data selection and reweighting. At its core is an In-Context Approximation (ICA) that estimates the holdout loss a model would incur after training on a candidate example by conditioning on a small, curated holdout set in context. ICA requires no reference model and no additional finetuning. Under a local linearization, ICA is equivalent to a first-order update toward the holdout optimum, motivating its use as a proxy for data value. We derive per-example weights from ICA scores, dynamically reweighting gradient updates as model parameters evolve. Across SFT, DPO, and SimPO, and over diverse backbones and datasets, ICA-based reweighting consistently improves model alignment with minimal overhead. We analyze sensitivity to score update frequency and the choice of $k$ holdout examples for in-context demonstrations, and note limitations for rapidly drifting on-policy updates, highlighting directions for future work. Code and prompts will be released. 

**Abstract (ZH)**: 一种理论依据明确且资源高效的數據選取與重權重框架：基于上下文近似的方法 

---
# Towards Adaptable Humanoid Control via Adaptive Motion Tracking 

**Title (ZH)**: 基于自适应运动跟踪的可适应 humanoid 控制研究 

**Authors**: Tao Huang, Huayi Wang, Junli Ren, Kangning Yin, Zirui Wang, Xiao Chen, Feiyu Jia, Wentao Zhang, Junfeng Long, Jingbo Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14454)  

**Abstract**: Humanoid robots are envisioned to adapt demonstrated motions to diverse real-world conditions while accurately preserving motion patterns. Existing motion prior approaches enable well adaptability with a few motions but often sacrifice imitation accuracy, whereas motion-tracking methods achieve accurate imitation yet require many training motions and a test-time target motion to adapt. To combine their strengths, we introduce AdaMimic, a novel motion tracking algorithm that enables adaptable humanoid control from a single reference motion. To reduce data dependence while ensuring adaptability, our method first creates an augmented dataset by sparsifying the single reference motion into keyframes and applying light editing with minimal physical assumptions. A policy is then initialized by tracking these sparse keyframes to generate dense intermediate motions, and adapters are subsequently trained to adjust tracking speed and refine low-level actions based on the adjustment, enabling flexible time warping that further improves imitation accuracy and adaptability. We validate these significant improvements in our approach in both simulation and the real-world Unitree G1 humanoid robot in multiple tasks across a wide range of adaptation conditions. Videos and code are available at this https URL. 

**Abstract (ZH)**: 类人机器人通过单个参考动作实现适应性运动模仿并保留运动模式。 

---
# Feature Selection and Regularization in Multi-Class Classification: An Empirical Study of One-vs-Rest Logistic Regression with Gradient Descent Optimization and L1 Sparsity Constraints 

**Title (ZH)**: 多类分类中特征选择与正则化：基于梯度下降优化和L1稀疏约束的一对多逻辑回归 empirical 研究 

**Authors**: Jahidul Arafat, Fariha Tasmin, Md Kaosar Uddin, Sanjaya Poudel, Eftakhar Ahmed Arnob  

**Link**: [PDF](https://arxiv.org/pdf/2510.14449)  

**Abstract**: Multi-class wine classification presents fundamental trade-offs between model accuracy, feature dimensionality, and interpretability - critical factors for production deployment in analytical chemistry. This paper presents a comprehensive empirical study of One-vs-Rest logistic regression on the UCI Wine dataset (178 samples, 3 cultivars, 13 chemical features), comparing from-scratch gradient descent implementation against scikit-learn's optimized solvers and quantifying L1 regularization effects on feature sparsity. Manual gradient descent achieves 92.59 percent mean test accuracy with smooth convergence, validating theoretical foundations, though scikit-learn provides 24x training speedup and 98.15 percent accuracy. Class-specific analysis reveals distinct chemical signatures with heterogeneous patterns where color intensity varies dramatically (0.31 to 16.50) across cultivars. L1 regularization produces 54-69 percent feature reduction with only 4.63 percent accuracy decrease, demonstrating favorable interpretability-performance trade-offs. We propose an optimal 5-feature subset achieving 62 percent complexity reduction with estimated 92-94 percent accuracy, enabling cost-effective deployment with 80 dollars savings per sample and 56 percent time reduction. Statistical validation confirms robust generalization with sub-2ms prediction latency suitable for real-time quality control. Our findings provide actionable guidelines for practitioners balancing comprehensive chemical analysis against targeted feature measurement in resource-constrained environments. 

**Abstract (ZH)**: 多类别葡萄酒分类展示了模型准确性、特征维度和可解释性之间的基本权衡——这对分析化学生产部署至关重要。本文对UCI葡萄酒数据集（178个样本，3个葡萄品种，13个化学特征）进行了One-vs-Rest逻辑回归的全面经验研究，比较了自实现梯度下降方法与scikit-learn优化求解器，并量化了L1正则化对特征稀疏性的影响。 

---
# A Free Lunch in LLM Compression: Revisiting Retraining after Pruning 

**Title (ZH)**: LLM压缩中的免费午餐：重新审视裁剪后的重新训练 

**Authors**: Moritz Wagner, Christophe Roux, Max Zimmer, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14444)  

**Abstract**: While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs. 

**Abstract (ZH)**: 而在裁剪后重建或重新训练剩余权重的关键设计选择研究中，状态-of-the-art大型语言模型的比例蒸馏方法避免了全面重新训练，尽管需要较少的内存，但仍能实现最佳性能。Badge: 关于裁剪后重建与重新训练的关键设计选择研究 

---
# Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and Scalable ML Framework for Precision Livestock Welfare 

**Title (ZH)**: 基于大数据方法的牛生物声学研究：符合FAIR原则的数据集及可扩展的机器学习框架以实现精准畜牧福祉 

**Authors**: Mayuri Kate, Suresh Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14443)  

**Abstract**: The convergence of IoT sensing, edge computing, and machine learning is transforming precision livestock farming. Yet bioacoustic data streams remain underused because of computational complexity and ecological validity challenges. We present one of the most comprehensive bovine vocalization datasets to date, with 569 curated clips covering 48 behavioral classes, recorded across three commercial dairy farms using multiple microphone arrays and expanded to 2900 samples through domain informed augmentation. This FAIR compliant resource addresses major Big Data challenges - volume (90 hours of recordings, 65.6 GB), variety (multi farm and multi zone acoustics), velocity (real time processing), and veracity (noise robust feature extraction). Our distributed processing framework integrates advanced denoising using iZotope RX, multimodal synchronization through audio and video alignment, and standardized feature engineering with 24 acoustic descriptors generated from Praat, librosa, and openSMILE. Preliminary benchmarks reveal distinct class level acoustic patterns for estrus detection, distress classification, and maternal communication. The datasets ecological realism, reflecting authentic barn acoustics rather than controlled settings, ensures readiness for field deployment. This work establishes a foundation for animal centered AI, where bioacoustic data enable continuous and non invasive welfare assessment at industrial scale. By releasing standardized pipelines and detailed metadata, we promote reproducible research that connects Big Data analytics, sustainable agriculture, and precision livestock management. The framework supports UN SDG 9, showing how data science can turn traditional farming into intelligent, welfare optimized systems that meet global food needs while upholding ethical animal care. 

**Abstract (ZH)**: 物联网 sensing、边缘计算和机器学习的融合正在改变精准畜牧业。然而，由于计算复杂性和生态有效性方面的挑战，生物声学数据流仍被广泛应用不足。我们呈现了迄今为止最为全面的牛隻鸣叫数据集，包含569个经过筛选的片段，涵盖48种行为类别，并通过领域知识增强扩展至2900个样本，记录于三家商业奶牛场使用多个麦克风阵列。该资源符合FAIR原则，解决大规模数据挑战——数据量（90小时录音，65.6GB）、多样性（多农场和多区域声学）、速度（实时处理）和真实性（噪声鲁棒特征提取）。我们的分布式处理框架整合了先进的去噪技术（iZotope RX）、多模态同步（通过音频和视频对齐）以及标准化特征工程（由Praat、librosa和openSMILE生成24个声学描述符）。初步基准测试显示，对于发情检测、应激分类和母性交流具有明显的类别水平声学模式。该数据集的生态现实性，反映实际牛舍声学环境而非受控环境，确保其适用于现场部署。本研究为以动物为中心的人工智能奠定了基础，生物声学数据使大规模、连续和不侵入式的福利评估成为可能。通过发布标准化管道和详细的元数据，我们促进了可重复的研究，连接大规模数据分析、可持续农业和精准畜牧业管理。该框架支持联合国可持续发展目标9，展示了如何通过数据科学将传统农业转变为智能、福利优化的系统，满足全球食物需求的同时保持伦理的动物护理。 

---
# Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following 

**Title (ZH)**: 所需指令即一切：自监督强化学习在指令跟随中的应用 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14420)  

**Abstract**: Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL 

**Abstract (ZH)**: 语言模型在遵循关键现实应用中的多约束指令时往往表现不佳。现有的强化学习（RL）方法依赖于外部监督和来自多约束任务的稀疏奖励信号。我们提出了一种标签-free自监督RL框架，通过直接从指令中推导奖励信号并为奖励模型训练生成伪标签，消除对外部监督的依赖。我们的方法引入了约束分解策略和高效的按约束二元分类，以应对稀疏奖励挑战并保持计算效率。实验表明，我们的方法具有良好的泛化能力，在3个领域内和5个领域外数据集上取得了显著改进，包括具有挑战性的代理性和多轮指令遵循任务。数据和代码已公开，可在此链接获取。 

---
# The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems 

**Title (ZH)**: 社会学习与集体规范形成在促进大规模语言模型多agent系统中合作的作用 

**Authors**: Prateek Gupta, Qiankun Zhong, Hiromu Yakura, Thomas Eisenmann, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14401)  

**Abstract**: A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments. 

**Abstract (ZH)**: 一种新兴的多智能体研究表明，大型语言模型（LLMs）在混合动机场景中探讨了规范和合作的 emergence，其中个体获益可能损害集体利益。虽然已有研究在丰富背景下的模拟和简化博弈环境中探讨了这些动态，大多数涉及公共池资源（CPR）游戏的LLM系统为智能体提供了与行为直接关联的显式奖励函数。相比之下，人类的合作往往在不完全了解报酬和人口的情况下产生，依靠启发式方法、沟通和惩罚。我们引入了一种CPR模拟框架，消除了显式的奖励信号，并嵌入了文化演化机制：社会学习（从成功的同侪采纳策略和信念）和基于奥斯特罗姆资源治理原则的规范惩罚机制。智能体还通过环境反馈从收获、监视和惩罚中学习，使规范能内生地生成。我们通过重现现有研究中关于人类行为的关键发现，建立了模拟的有效性。在此基础上，我们研究了不同环境和社会初始条件（资源丰富 vs. 资源稀缺；利他 vs. 自私）下规范进化的演变，并评估了由不同LLM组成的有机关社会在这些条件下的表现。我们的结果揭示了不同模型在维持合作和规范形成方面的系统性差异，将该框架定位为研究混合动机LLM社会中涌现规范的严格测试床。这种分析可以为在社会和组织环境中部署的AI系统的设计提供信息，其中与合作规范的对齐对于稳定、公平和AI中介环境中有效治理至关重要。 

---
# MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering 

**Title (ZH)**: MedTrust-RAG: biomedical知识检索与证据验证及信任对齐 

**Authors**: Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua Wang, Yuchen Pan, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14400)  

**Abstract**: Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our approach consistently outperforms competitive baselines across multiple model architectures, achieving the best average accuracy with gains of 2.7% for LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B. 

**Abstract (ZH)**: 基于MedTrust引导的迭代检索增强生成在生物医学问答中的应用 

---
# FairBatching: Fairness-Aware Batch Formation for LLM Inference 

**Title (ZH)**: 公平批次形成：考虑公平性的LLM推理批次构建方法 

**Authors**: Hongtao Lyu, Boyue Liu, Mingyu Wu, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14392)  

**Abstract**: Large language model (LLM) inference systems face a fundamental tension between minimizing Time-to-First-Token (TTFT) latency for new requests and maintaining a high, steady token generation rate (low Time-Per-Output-Token, or TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by Sarathi, while effective at preventing decode stalls, introduce significant computational unfairness. They prioritize decode tasks excessively, simultaneously leading to underutilized decode slack and unnecessary prefill queuing delays, which collectively degrade the system's overall quality of service (QoS).
This work identifies the root cause of this unfairness: the non-monotonic nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid decode-prioritizing policy that fails to adapt to dynamic workload bursts. We therefore propose FairBatching, a novel LLM inference scheduler that enforces fair resource allocation between prefill and decode tasks. It features an adaptive batch capacity determination mechanism, which dynamically adjusts the computational budget to improve the GPU utilization without triggering SLO violations. Its fair and dynamic batch formation algorithm breaks away from the decode-prioritizing paradigm, allowing computation resources to be reclaimed from bursting decode tasks to serve prefill surges, achieving global fairness. Furthermore, FairBatching provides a novel load estimation method, enabling more effective coordination with upper-level schedulers. Implemented and evaluated on realistic traces, FairBatching significantly reduces TTFT tail latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall 20.0% improvement in single-node capacity and 54.3% improvement in cluster-level capacity. 

**Abstract (ZH)**: 大型语言模型推断系统面对着一个根本性的权衡，即在最小化新请求的首个令牌 latency（TTFT）与维持高且稳定的令牌生成速率（低时间每输出令牌，TPOT）之间存在矛盾。萨拉蒂提出的无阻塞批量调度器虽然能够预防解码延迟，但会导致显著的计算不公平。这些调度器过度优先处理解码任务，同时导致解码余量的未能充分利用和不必要的预填充排队延迟，从而整体上降低了系统的服务质量（QoS）。

这项工作识别了这种不公平的根本原因：时间间隔（TBT）作为调度指标的非单调性质以及僵化的解码优先策略未能适应动态工作负载突发。因此，我们提出了一种新的大型语言模型推断调度器——FairBatching，该调度器在预填充和解码任务之间实现了公平资源分配。它具备一种自适应批量容量确定机制，能够动态调整计算预算以提高GPU利用率，同时不引发SLO违约。其公平且动态的批量形成算法打破了解码优先的范式，允许计算资源从突发的解码任务回收以服务于预填充突发，从而实现全局公平。此外，FairBatching 提供了一种新的负载估计方法，能够更有效地与上层调度器协调。在实际跟踪上实现和评估后，FairBatching 通过最多减少2.29倍的TTFT尾部延迟，同时稳健地维持TPOT SLO，实现了单节点容量整体20.0%的提升以及集群级别容量54.3%的提升。 

---
# Beat Detection as Object Detection 

**Title (ZH)**: 鼓点检测作为对象检测 

**Authors**: Jaehoon Ahn, Moon-Ryul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2510.14391)  

**Abstract**: Recent beat and downbeat tracking models (e.g., RNNs, TCNs, Transformers) output frame-level activations. We propose reframing this task as object detection, where beats and downbeats are modeled as temporal "objects." Adapting the FCOS detector from computer vision to 1D audio, we replace its original backbone with WaveBeat's temporal feature extractor and add a Feature Pyramid Network to capture multi-scale temporal patterns. The model predicts overlapping beat/downbeat intervals with confidence scores, followed by non-maximum suppression (NMS) to select final predictions. This NMS step serves a similar role to DBNs in traditional trackers, but is simpler and less heuristic. Evaluated on standard music datasets, our approach achieves competitive results, showing that object detection techniques can effectively model musical beats with minimal adaptation. 

**Abstract (ZH)**: 最近的打击声和重拍跟踪模型（如RNNs、TCNs、Transformers）输出帧级激活。我们提出了将此任务重新定义为对象检测，其中打击声和重拍被 modeling 为时间上的“对象”。将来自于计算机视觉领域的 FCOS 检测器适应至 1D 音频，我们用 WaveBeat 的时间特征提取器替换其原始骨干，并添加 Feature Pyramid Network 以捕捉多层次的时间模式。该模型预测具有置信分数的重叠的打击声/重拍区间，随后通过非最大抑制（NMS）选择最终预测。该 NMS 步骤在传统跟踪器中类似于 DBNs 的作用，但更为简单且减少了几何硬性约束。在标准音乐数据集上评估，我们的方法取得竞争性的结果，表明对象检测技术在少量调整的情况下能够有效建模音乐节奏。 

---
# Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers 

**Title (ZH)**: 我的优化提示被妥协了吗？探索基于LLM的优化器的安全漏洞 

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes  

**Link**: [PDF](https://arxiv.org/pdf/2510.14381)  

**Abstract**: Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks. 

**Abstract (ZH)**: 大规模语言模型（LLM）系统现在支撑着诸如聊天机器人、计算机使用助手和自主机器人等日常AI应用，其中性能常常依赖于精心设计的提示。基于LLM的提示优化器通过迭代改进评分反馈的提示来减轻这种努力，但这一优化阶段的安全性仍待进一步考察。我们首次系统地分析了基于LLM的提示优化中的投毒风险。使用HarmBench，我们发现系统在被操纵反馈的影响下比被注入查询的影响更为脆弱：基于反馈的攻击可将攻击成功率提高多达ΔASR = 0.48。我们引入了一种简单的伪造奖励攻击，该攻击无需访问奖励模型即可显著增加脆弱性，并提出了一种轻量级的高亮防御措施，该措施将伪造奖励的ΔASR从0.23降低到0.07，而不牺牲性能。这些结果将提示优化管道确立为首要的攻击面，并促使反馈通道和优化框架的安全防范措施更加严格。 

---
# From Binary to Bilingual: How the National Weather Service is Using Artificial Intelligence to Develop a Comprehensive Translation Program 

**Title (ZH)**: 从二元到双语：国家气象服务如何利用人工智能开发全面的翻译计划 

**Authors**: Joseph E. Trujillo-Falcon, Monica L. Bozeman, Liam E. Llewellyn, Samuel T. Halvorson, Meryl Mizell, Stuti Deshpande, Bob Manning, Todd Fagin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14369)  

**Abstract**: To advance a Weather-Ready Nation, the National Weather Service (NWS) is developing a systematic translation program to better serve the 68.8 million people in the U.S. who do not speak English at home. This article outlines the foundation of an automated translation tool for NWS products, powered by artificial intelligence. The NWS has partnered with LILT, whose patented training process enables large language models (LLMs) to adapt neural machine translation (NMT) tools for weather terminology and messaging. Designed for scalability across Weather Forecast Offices (WFOs) and National Centers, the system is currently being developed in Spanish, Simplified Chinese, Vietnamese, and other widely spoken non-English languages. Rooted in best practices for multilingual risk communication, the system provides accurate, timely, and culturally relevant translations, significantly reducing manual translation time and easing operational workloads across the NWS. To guide the distribution of these products, GIS mapping was used to identify language needs across different NWS regions, helping prioritize resources for the communities that need them most. We also integrated ethical AI practices throughout the program's design, ensuring that transparency, fairness, and human oversight guide how automated translations are created, evaluated, and shared with the public. This work has culminated into a website featuring experimental multilingual NWS products, including translated warnings, 7-day forecasts, and educational campaigns, bringing the country one step closer to a national warning system that reaches all Americans. 

**Abstract (ZH)**: 为了推进“天气预警国家”目标，美国国家气象服务（NWS）正开发一套系统化的翻译计划，以更好地服务于美国6880万不在家使用英语的人口。本文概述了一种自动翻译工具的基础，该工具由人工智能驱动，用于NWS产品。NWS与LILT合作，后者拥有专利的培训过程，使大规模语言模型（LLMs）能够适应神经机器翻译（NMT）工具以适用于气象术语和信息。该系统已设计为在气象预报办公室（WFO）和国家级中心之间可扩展，并正在西班牙语、简体中文、越南语及其他广泛使用的非英语语言中开发。基于最佳的多语言风险管理实践，该系统提供准确、及时且文化相关的翻译，大大减少了手动翻译时间，减轻了NWS的操作负担。为了指导这些产品的分发，使用GIS mapping确定了NWS各区域的语言需求，有助于优先为最需要的社区分配资源。我们还在该计划的设计中整合了伦理AI实践，确保透明度、公正性和人类监督指导自动翻译的创建、评估和向公众分享的方式。这一工作已经形成了一个网站，展示了实验性多语言NWS产品，包括翻译后的警报、7天预报和教育宣传活动，使国家向所有美国人的国家级警报系统迈进了一步。 

---
# SUM-AgriVLN: Spatial Understanding Memory for Agricultural Vision-and-Language Navigation 

**Title (ZH)**: SUM-AgriVLN: 空间理解记忆在农业视觉-语言导航中的应用 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14357)  

**Abstract**: Agricultural robots are emerging as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or fixed rail systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling robots to navigate to the target positions following the natural language instructions. In practical agricultural scenarios, navigation instructions often repeatedly occur, yet AgriVLN treat each instruction as an independent episode, overlooking the potential of past experiences to provide spatial context for subsequent ones. To bridge this gap, we propose the method of Spatial Understanding Memory for Agricultural Vision-and-Language Navigation (SUM-AgriVLN), in which the SUM module employs spatial understanding and save spatial memory through 3D reconstruction and representation. When evaluated on the A2A benchmark, our SUM-AgriVLN effectively improves Success Rate from 0.47 to 0.54 with slight sacrifice on Navigation Error from 2.91m to 2.93m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL. 

**Abstract (ZH)**: 农业机器人正在广泛农业生产任务中崭露头角，但仍主要依赖手动操作或固定轨道系统进行移动。AgriVLN方法和A2A基准首次将视觉与语言导航（VLN）扩展到农业领域，使机器人能够根据自然语言指令导航至目标位置。在实际农业生产场景中，导航指令常常重复出现，但AgriVLN将每条指令视为独立的 episode，忽视了过往经验在提供后续指令空间上下文方面的潜力。为解决这一问题，我们提出了农业视觉与语言导航的空间理解记忆方法（SUM-AgriVLN），其中SUM模块通过三维重建和表示来实现空间理解和保存空间记忆。在A2A基准上进行评估时，我们的SUM-AgriVLN有效提高了成功率达到0.54，同时略微增加了导航误差从2.91米到2.93米，证明了在农业领域的先进性能。代码：这个 https URL。 

---
# CURE: Confidence-driven Unified Reasoning Ensemble Framework for Medical Question Answering 

**Title (ZH)**: CURE：基于信心的统一推理集成框架在医学问答中的应用 

**Authors**: Ziad Elshaer, Essam A. Rashed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14353)  

**Abstract**: High-performing medical Large Language Models (LLMs) typically require extensive fine-tuning with substantial computational resources, limiting accessibility for resource-constrained healthcare institutions. This study introduces a confidence-driven multi-model framework that leverages model diversity to enhance medical question answering without fine-tuning. Our framework employs a two-stage architecture: a confidence detection module assesses the primary model's certainty, and an adaptive routing mechanism directs low-confidence queries to Helper models with complementary knowledge for collaborative reasoning. We evaluate our approach using Qwen3-30B-A3B-Instruct, Phi-4 14B, and Gemma 2 12B across three medical benchmarks; MedQA, MedMCQA, and PubMedQA. Result demonstrate that our framework achieves competitive performance, with particularly strong results in PubMedQA (95.0\%) and MedMCQA (78.0\%). Ablation studies confirm that confidence-aware routing combined with multi-model collaboration substantially outperforms single-model approaches and uniform reasoning strategies. This work establishes that strategic model collaboration offers a practical, computationally efficient pathway to improve medical AI systems, with significant implications for democratizing access to advanced medical AI in resource-limited settings. 

**Abstract (ZH)**: 高性能医疗大型语言模型通常需要大量的 Fine-tuning 和计算资源，限制了资源受限的医疗保健机构的访问能力。本研究引入了一种基于信心驱动的多模型框架，利用模型多样性在无需 Fine-tuning 的情况下增强医疗问答。该框架采用两阶段架构：信心检测模块评估主要模型的确定性，自适应路由机制将低信心查询导向具有互补知识的 Helper 模型以进行协作推理。我们使用 Qwen3-30B-A3B-Instruct、Phi-4 14B 和 Gemma 2 12B 在三个医疗基准 MedQA、MedMCQA 和 PubMedQA 上评估了该方法；结果显示，该框架实现了竞争性的性能，尤其在 PubMedQA（95.0%）和 MedMCQA（78.0%）上表现强劲。消融研究证实，信心感知路由与多模型协作显著优于单模型方法和均匀推理策略。本研究确立了策略性模型协作提供了一种实用、计算高效的途径来提升医疗人工智能系统，并对在资源受限环境中实现高级医疗人工智能的民主化具有重要意义。 

---
# Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts 

**Title (ZH)**: 超越单一宇宙：多宇宙背景下角色扮演超级英雄的基准测试 

**Authors**: Perapard Ngokpol, Kun Kerdthaisong, Pasin Buakhaw, Pitikorn Khlaisamniang, Supasate Vorathammathorn, Piyalitt Ittichaiwong, Nutchanon Yongsatianchot  

**Link**: [PDF](https://arxiv.org/pdf/2510.14351)  

**Abstract**: Large language models (LLMs) are increasingly used as role-playing agents, yet their capacity to faithfully and consistently portray version-specific characters -- for example, superheroes across comic and cinematic universes -- remains underexplored. Superhero canons such as Marvel and DC provide a rich testbed: decades of storytelling yield multiple incarnations of the same character with distinct histories, values, and moral codes. To study this problem, we introduce Beyond One World, a benchmark for character-grounded roleplay spanning 30 iconic heroes and 90 canon-specific versions. The benchmark comprises two tasks: (i) Canon Events, which probes factual recall of pivotal life stages, and (ii) Moral Dilemmas, which confronts models with ethically charged scenarios. We score responses for canonical accuracy and reasoning fidelity under a framework that separates internal deliberation ("thinking") from outward decisions ("acting"). We further propose Think-Act Matching, a metric that quantifies alignment between reasons and actions and serves as a proxy for model trustworthiness. Experiments across reasoning- and non-reasoning-oriented models yield three findings: (1) chain-of-thought prompting improves narrative coherence in weaker models but can reduce canonical accuracy in stronger ones; (2) cross-version generalization within a character remains a major obstacle; and (3) models often excel at either thinking or acting, but rarely both. Beyond One World exposes critical gaps in multiversal consistency and reasoning alignment, offering a challenging evaluation for role-playing LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在扮演角色方面越来越被广泛使用，但它们忠实且一致地呈现版本特异性人物的能力——例如，漫威和DC等超级英雄在漫画和电影宇宙中的表现——仍待深入探索。为此，我们引入了“超越单一世界”基准，这是一个涵盖30位标志性英雄和90种特定宇宙版本的角色导向角色扮演基准。该基准包含两项任务：(i) 正史事件，考查关键时刻的事实回忆，以及(ii) 道德困境，对模型提出道德层面的场景挑战。我们根据一个框架评分，该框架将内部推断（“思考”）与外在决定（“行动”）区分开来。我们还提出了思考-行动匹配度量，该度量量化了原因和行动之间的一致程度，并作为模型可信度的代理指标。针对推理和非推理导向模型的实验得出以下三点发现：(1) 链式思考提示在较弱模型中提高了叙事连贯性，但在较强模型中可能降低正史准确性；(2) 在同一角色内部跨越版本的泛化仍是主要障碍；(3) 模型往往在思考或行动中表现出色，但很少两者兼备。“超越单一世界”揭示了宇宙一致性与推理一致性的关键缺口，为角色扮演LLMs提供了具有挑战性的评估。 

---
# BinCtx: Multi-Modal Representation Learning for Robust Android App Behavior Detection 

**Title (ZH)**: BinCtx: 多模态表示学习在鲁棒Android应用程序行为检测中的应用 

**Authors**: Zichen Liu, Shao Yang, Xusheng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14344)  

**Abstract**: Mobile app markets host millions of apps, yet undesired behaviors (e.g., disruptive ads, illegal redirection, payment deception) remain hard to catch because they often do not rely on permission-protected APIs and can be easily camouflaged via UI or metadata edits. We present BINCTX, a learning approach that builds multi-modal representations of an app from (i) a global bytecode-as-image view that captures code-level semantics and family-style patterns, (ii) a contextual view (manifested actions, components, declared permissions, URL/IP constants) indicating how behaviors are triggered, and (iii) a third-party-library usage view summarizing invocation frequencies along inter-component call paths. The three views are embedded and fused to train a contextual-aware classifier. On real-world malware and benign apps, BINCTX attains a macro F1 of 94.73%, outperforming strong baselines by at least 14.92%. It remains robust under commercial obfuscation (F1 84% post-obfuscation) and is more resistant to adversarial samples than state-of-the-art bytecode-only systems. 

**Abstract (ZH)**: BINCTX：一种多模态学习方法用于移动应用的上下文感知分类 

---
# A Density-Informed Multimodal Artificial Intelligence Framework for Improving Breast Cancer Detection Across All Breast Densities 

**Title (ZH)**: 一种基于密度的信息多模态人工智能框架，用于提高乳腺癌检测效果，涵盖所有乳腺密度类型 

**Authors**: Siva Teja Kakileti, Bharath Govindaraju, Sudhakar Sampangi, Geetha Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2510.14340)  

**Abstract**: Mammography, the current standard for breast cancer screening, has reduced sensitivity in women with dense breast tissue, contributing to missed or delayed diagnoses. Thermalytix, an AI-based thermal imaging modality, captures functional vascular and metabolic cues that may complement mammographic structural data. This study investigates whether a breast density-informed multi-modal AI framework can improve cancer detection by dynamically selecting the appropriate imaging modality based on breast tissue composition. A total of 324 women underwent both mammography and thermal imaging. Mammography images were analyzed using a multi-view deep learning model, while Thermalytix assessed thermal images through vascular and thermal radiomics. The proposed framework utilized Mammography AI for fatty breasts and Thermalytix AI for dense breasts, optimizing predictions based on tissue type. This multi-modal AI framework achieved a sensitivity of 94.55% (95% CI: 88.54-100) and specificity of 79.93% (95% CI: 75.14-84.71), outperforming standalone mammography AI (sensitivity 81.82%, specificity 86.25%) and Thermalytix AI (sensitivity 92.73%, specificity 75.46%). Importantly, the sensitivity of Mammography dropped significantly in dense breasts (67.86%) versus fatty breasts (96.30%), whereas Thermalytix AI maintained high and consistent sensitivity in both (92.59% and 92.86%, respectively). This demonstrates that a density-informed multi-modal AI framework can overcome key limitations of unimodal screening and deliver high performance across diverse breast compositions. The proposed framework is interpretable, low-cost, and easily deployable, offering a practical path to improving breast cancer screening outcomes in both high-resource and resource-limited settings. 

**Abstract (ZH)**: 基于乳腺密度的多模态AI框架在乳腺癌检测中的应用研究 

---
# Stop-RAG: Value-Based Retrieval Control for Iterative RAG 

**Title (ZH)**: Stop-RAG: 基于价值的检索控制以实现迭代RAG 

**Authors**: Jaewan Park, Solbee Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14337)  

**Abstract**: Iterative retrieval-augmented generation (RAG) enables large language models to answer complex multi-hop questions, but each additional loop increases latency, costs, and the risk of introducing distracting evidence, motivating the need for an efficient stopping strategy. Existing methods either use a predetermined number of iterations or rely on confidence proxies that poorly reflect whether more retrieval will actually help. We cast iterative RAG as a finite-horizon Markov decision process and introduce Stop-RAG, a value-based controller that adaptively decides when to stop retrieving. Trained with full-width forward-view Q($\lambda$) targets from complete trajectories, Stop-RAG learns effective stopping policies while remaining compatible with black-box APIs and existing pipelines. On multi-hop question-answering benchmarks, Stop-RAG consistently outperforms both fixed-iteration baselines and prompting-based stopping with LLMs. These results highlight adaptive stopping as a key missing component in current agentic systems, and demonstrate that value-based control can improve the accuracy of RAG systems. 

**Abstract (ZH)**: 迭代检索增强生成（RAG）使得大型语言模型能够回答复杂的多跳问题，但每次额外的循环都会增加延迟、成本，并增加引入分散注意力的证据的风险，从而促使需要一个高效的停止策略。现有的方法要么使用固定数量的迭代次数，要么依赖于不能准确反映额外检索是否真正有益的置信代理。我们将迭代RAG建模为有限 horizon 马尔可夫决策过程，并引入Stop-RAG，这是一种基于价值的控制器，能够适应性地决定是否停止检索。Stop-RAG通过完整的轨迹全宽前瞻Q($\lambda$)目标进行训练，既能学习有效的停止策略，又能与黑盒API和现有管道保持兼容。在多跳问答基准测试中，Stop-RAG在准确性和固定迭代基线以及基于提示的停止策略（使用LLM）中表现出色。这些结果强调了自适应停止策略是当前代理系统中一个关键的缺失组件，并表明基于价值的控制可以提高RAG系统的准确性。 

---
# A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease 

**Title (ZH)**: 基于混合词嵌入的鲁棒分类方法在阿尔茨海默病早期诊断中的应用 

**Authors**: Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14332)  

**Abstract**: Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD. 

**Abstract (ZH)**: 早发性老年痴呆症（AD）的检测对AD患者极为有益，可以实现早期治疗，减轻症状并缓解医疗开支。作为AD的一个主要标志，语言能力的变化可以用于AD的早期诊断。本文提出了一种基于混合词嵌入和微调超参数的稳健分类方法，以实现AD早期检测的最新准确度。具体而言，我们基于Doc2Vec和ELMo的词向量创建了混合词嵌入，以获得句子的困惑度分数。这些分数可以识别句子是否流畅，并捕捉句子的语义上下文。通过添加语言特征来丰富词嵌入，以分析句法和语义。随后，我们将嵌入特征向量输入逻辑回归，并在整个流水线中微调超参数。通过微调机器学习流水线中的超参数（如模型正则化参数、学习率和Doc2Vec的向量大小、ELMo的向量大小），我们实现了91%的分类准确度和97%的曲线下面积（AUC），在区分早期AD和健康个体方面表现优异。据我所知，我的模型在准确度为91%和AUC为97%的情况下，比现有最佳的AD诊断NLP模型（准确度为88%）表现更佳。通过多次实验研究模型的稳定性，发现即使随机分割训练数据，模型也保持稳定（准确度标准差=0.0403；AUC标准差=0.0174）。这证实了我们提出的方法准确且稳定。该模型可以用作AD的大规模筛查方法，也可以作为医生诊断AD的补充检查工具。 

---
# Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL 

**Title (ZH)**: 评估与减少语言模型中欺骗性对话的方法：多轮RLomore 

**Authors**: Marwa Abdulhai, Ryan Cheng, Aryansh Shrivastava, Natasha Jaques, Yarin Gal, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.14318)  

**Abstract**: Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在客户服务、教育和医疗等领域与全世界数百万人交互。然而，它们产生误导性输出的能力，无论是有意还是无意，都带来了重大安全风险。LLMs行为的不可预测性，以及缺乏对幻觉、 misinformation 和用户操纵的充分防护措施，使其误用成为现实世界中的严重风险。在本文中，我们研究了LLMs在对话中进行欺骗的程度，并提出了信念不一致度量来量化欺骗。我们使用五种公认的欺骗检测度量标准和我们提出的新度量标准，在四个不同的对话场景中评估了欺骗。我们的发现揭示了这一新型欺骗指标与人类判断的相关性高于我们测试的所有现有指标。此外，对标八个最先进的模型的基准测试表明，即使在受到看似良性目标的提示时，LLMs也自然表现出约26%对话回合的欺骗行为。当要求欺骗时，LLMs相对基线可增加欺骗性高达31%。令人意外的是，用RLHF训练的模型——确保广泛部署的LLMs安全的主要方法——的平均欺骗率为43%。鉴于对话中的欺骗行为是在交互历史中逐步发展的，其有效评估和缓解需要超越单句分析。我们引入了一种多轮强化学习方法来微调LLMs以减少欺骗行为，相对于其他指令调优模型，欺骗性降低了77.6%。 

---
# Column Generation Using Domain-Independent Dynamic Programming 

**Title (ZH)**: 基于域无关动态规划的列生成方法 

**Authors**: Ryo Kuroiwa, Edward Lam  

**Link**: [PDF](https://arxiv.org/pdf/2510.14317)  

**Abstract**: Column generation and branch-and-price are leading methods for large-scale exact optimization. Column generation iterates between solving a master problem and a pricing problem. The master problem is a linear program, which can be solved using a generic solver. The pricing problem is highly dependent on the application but is usually discrete. Due to the difficulty of discrete optimization, high-performance column generation often relies on a custom pricing algorithm built specifically to exploit the problem's structure. This bespoke nature of the pricing solver prevents the reuse of components for other applications. We show that domain-independent dynamic programming, a software package for modeling and solving arbitrary dynamic programs, can be used as a generic pricing solver. We develop basic implementations of branch-and-price with pricing by domain-independent dynamic programming and show that they outperform a world-leading solver on static mixed integer programming formulations for seven problem classes. 

**Abstract (ZH)**: 列生成和分支定价是大型精确优化的主导方法。列生成在解决主问题和定价问题之间迭代。主问题是一个线性程序，可以使用通用求解器求解。定价问题高度依赖于应用，通常为离散型。由于离散优化的难度，高性能列生成往往依赖于针对问题结构特别定制的定价算法。这种定制化的定价求解器防止了组件在其他应用中的重用。我们展示了一种通用的定价求解器——基于通用动态规划的软件包，可以用于分支定价。我们开发了基于通用动态规划的分支定价的基本实现，并展示了它们在七个问题类别的静态混合整数规划形式化表示上优于世界领先的求解器。 

---
# MERLIN: A Testbed for Multilingual Multimodal Entity Recognition and Linking 

**Title (ZH)**: MERLIN：多模态多语言实体识别与链接的实验平台 

**Authors**: Sathyanarayanan Ramamoorthy, Vishwa Shah, Simran Khanuja, Zaid Sheikh, Shan Jie, Ann Chia, Shearman Chua, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14307)  

**Abstract**: This paper introduces MERLIN, a novel testbed system for the task of Multilingual Multimodal Entity Linking. The created dataset includes BBC news article titles, paired with corresponding images, in five languages: Hindi, Japanese, Indonesian, Vietnamese, and Tamil, featuring over 7,000 named entity mentions linked to 2,500 unique Wikidata entities. We also include several benchmarks using multilingual and multimodal entity linking methods exploring different language models like LLaMa-2 and Aya-23. Our findings indicate that incorporating visual data improves the accuracy of entity linking, especially for entities where the textual context is ambiguous or insufficient, and particularly for models that do not have strong multilingual abilities. For the work, the dataset, methods are available here at this https URL 

**Abstract (ZH)**: 本文介绍了MERLIN，一种用于多语言多模态实体链接的新颖试验系统。该创建的数据集包括五种语言（ Hindi、Japanese、Indonesian、Vietnamese 和 Tamil）的BBC新闻文章标题及其对应的图片，涉及超过7,000个命名实体提及，链接到2,500个唯一的Wikidata实体。我们还提供了使用多语言和多模态实体链接方法进行实验的基准，涉及像LLaMa-2和Aya-23这样的不同语言模型。我们的研究结果表明，结合视觉数据可以提高实体链接的准确性，特别是在文本上下文模糊或不足的情况下，尤其对于不具备强大多语言能力的模型。相关数据集和方法可从以下链接获取：https://this-url.com 

---
# Watermarking for Factuality: Guiding Vision-Language Models Toward Truth via Tri-layer Contrastive Decoding 

**Title (ZH)**: 事实水印：通过三层对比解码引导视觉-语言模型追求真理 

**Authors**: Kyungryul Back, Seongbeom Park, Milim Kim, Mincheol Kwon, SangHyeok Lee, Hyunyoung Lee, Junhee Cho, Seunghyun Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14304)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown promising results on various multimodal tasks, even achieving human-comparable performance in certain cases. Nevertheless, LVLMs remain prone to hallucinations -- they often rely heavily on a single modality or memorize training data without properly grounding their outputs. To address this, we propose a training-free, tri-layer contrastive decoding with watermarking, which proceeds in three steps: (1) select a mature layer and an amateur layer among the decoding layers, (2) identify a pivot layer using a watermark-related question to assess whether the layer is visually well-grounded, and (3) apply tri-layer contrastive decoding to generate the final output. Experiments on public benchmarks such as POPE, MME and AMBER demonstrate that our method achieves state-of-the-art performance in reducing hallucinations in LVLMs and generates more visually grounded responses. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）在多种多模态任务中展现出了有前途的结果，即使在某些情况下达到了与人类相当的性能。然而，LVLMs仍然容易产生幻觉——它们经常过度依赖单一模态或记忆训练数据，而未能适当地将输出进行 grounded。为解决这一问题，我们提出了一个无需训练的三层对比解码方法，并结合水印技术，该方法分为三步：（1）选择一个成熟的解码层和一个新手层，（2）使用与水印相关的问题识别一个枢纽层，评估该层是否在视觉上良好grounded，（3）应用三层对比解码生成最终输出。在POPE、MME和AMBER等公开基准上的实验表明，我们的方法在减少LVLMs中的幻觉方面取得了最优性能，并生成了更多视觉上grounded的响应。 

---
# Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning 

**Title (ZH)**: 专家知识不必垄断：面向视觉-语言-行动学习的动作专业化专家混合模型 

**Authors**: Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14300)  

**Abstract**: Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks. However, scaling up VLA models presents several critical challenges: (1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process. (2) Real-time control requires carefully balancing model capacity with computational efficiency. To address these challenges, We propose AdaMoE, a Mixture-of-Experts (MoE) architecture that inherits pretrained weights from dense VLA models, and scales up the action expert by substituting the feedforward layers into sparsely activated MoE layers. AdaMoE employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This enables experts to be selected based on task relevance while contributing with independently controlled weights, allowing collaborative expert utilization rather than winner-takes-all dynamics. Our approach demonstrates that expertise need not monopolize. Instead, through collaborative expert utilization, we can achieve superior performance while maintaining computational efficiency. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8% on LIBERO and 9.3% on RoboTwin. Most importantly, a substantial 21.5% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作（VLA）模型在机器人操作任务中的快速发展和前景能力，面对扩大VLA模型规模的若干关键挑战，提出了一种混合专家（MoE）架构AdaMoE，该架构通过稀疏激活的MoE层替代密集VLA模型中的前馈层，继承预训练权重并扩展动作专家。AdaMoE通过解耦专家选择与权重分配，使专家可以根据任务相关性进行选择并独立控制权重，从而实现协作专家利用而非胜者全拿的动态。该方法表明，专家无需垄断，通过协作利用专家，可以在保持计算效率的同时实现卓越的性能。实验结果表明，AdaMoE在关键基准上始终优于基线模型，在LIBERO上性能提升1.8%，在RoboTwin上提升9.3%。最重要的是，实际实验中的显著改进（21.5%）验证了其在机器人操作任务中的实际有效性。 

---
# TED++: Submanifold-Aware Backdoor Detection via Layerwise Tubular-Neighbourhood Screening 

**Title (ZH)**: TED++: 基于层wise筒形邻域筛选的子流形 Aware 后门检测 

**Authors**: Nam Le, Leo Yu Zhang, Kewen Liao, Shirui Pan, Wei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14299)  

**Abstract**: As deep neural networks power increasingly critical applications, stealthy backdoor attacks, where poisoned training inputs trigger malicious model behaviour while appearing benign, pose a severe security risk. Many existing defences are vulnerable when attackers exploit subtle distance-based anomalies or when clean examples are scarce. To meet this challenge, we introduce TED++, a submanifold-aware framework that effectively detects subtle backdoors that evade existing defences. TED++ begins by constructing a tubular neighbourhood around each class's hidden-feature manifold, estimating its local ``thickness'' from a handful of clean activations. It then applies Locally Adaptive Ranking (LAR) to detect any activation that drifts outside the admissible tube. By aggregating these LAR-adjusted ranks across all layers, TED++ captures how faithfully an input remains on the evolving class submanifolds. Based on such characteristic ``tube-constrained'' behaviour, TED++ flags inputs whose LAR-based ranking sequences deviate significantly. Extensive experiments are conducted on benchmark datasets and tasks, demonstrating that TED++ achieves state-of-the-art detection performance under both adaptive-attack and limited-data scenarios. Remarkably, even with only five held-out examples per class, TED++ still delivers near-perfect detection, achieving gains of up to 14\% in AUROC over the next-best method. The code is publicly available at this https URL. 

**Abstract (ZH)**: 随着深度神经网络在日益关键的应用中发挥作用，隐形后门攻击在训练过程中注入恶意样本，使模型在看似正常的输入下产生恶意行为，这一风险构成了严重的安全威胁。许多现有的防御方法在攻击者利用细微的距离异常或缺乏干净样本时容易失效。为应对这一挑战，我们引入了TED++，这是一个亚流形感知框架，能够有效检测现有防御方法无法察觉的隐蔽后门。TED++首先围绕每个类别的隐藏特征流形构建一个管状邻域，从少量干净激活中估计其局部“厚度”。然后，它应用局部自适应排名（LAR）来检测任何偏离允许管的激活。通过在所有层上聚合这些LAR调整后的排名，TED++捕捉输入如何忠实地保持在不断演化的类亚流形上。基于这种特征的“管约束”行为，TED++标记LAR基于的排名序列显著偏差的输入。我们在基准数据集和任务上进行了广泛实验，证明TED++在适应性攻击和少量数据场景下都达到了最先进的检测性能。即使每类仅有的五个保留样本，TED++仍能实现近乎完美的检测表现，在AUROC方面相对于下一最佳方法取得了高达14%的提升。代码已公开，可通过以下链接获取。 

---
# Learning Human-Humanoid Coordination for Collaborative Object Carrying 

**Title (ZH)**: 学习人类-类人机器人协作搬运物体的合作协调 

**Authors**: Yushi Du, Yixuan Li, Baoxiong Jia, Yutang Lin, Pei Zhou, Wei Liang, Yanchao Yang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14293)  

**Abstract**: Human-humanoid collaboration shows significant promise for applications in healthcare, domestic assistance, and manufacturing. While compliant robot-human collaboration has been extensively developed for robotic arms, enabling compliant human-humanoid collaboration remains largely unexplored due to humanoids' complex whole-body dynamics. In this paper, we propose a proprioception-only reinforcement learning approach, COLA, that combines leader and follower behaviors within a single policy. The model is trained in a closed-loop environment with dynamic object interactions to predict object motion patterns and human intentions implicitly, enabling compliant collaboration to maintain load balance through coordinated trajectory planning. We evaluate our approach through comprehensive simulator and real-world experiments on collaborative carrying tasks, demonstrating the effectiveness, generalization, and robustness of our model across various terrains and objects. Simulation experiments demonstrate that our model reduces human effort by 24.7%. compared to baseline approaches while maintaining object stability. Real-world experiments validate robust collaborative carrying across different object types (boxes, desks, stretchers, etc.) and movement patterns (straight-line, turning, slope climbing). Human user studies with 23 participants confirm an average improvement of 27.4% compared to baseline models. Our method enables compliant human-humanoid collaborative carrying without requiring external sensors or complex interaction models, offering a practical solution for real-world deployment. 

**Abstract (ZH)**: 人形机器人与人类协作在医疗、家庭辅助和制造领域的应用展现出显著潜力。尽管具备顺应性的机器人臂人机协作已得到广泛开发，但如何实现具备顺应性的人类与人形机器人协作还未得到充分探索，主要归因于人形机器人复杂的全身动力学。在本文中，我们提出了一种仅依靠本体感受的强化学习方法COLA，该方法结合了领导行为和跟随行为于单一策略中。该模型在包含动态物体交互的闭环环境中训练，以隐式预测物体运动模式和人类意图，从而通过协调轨迹规划维持负载平衡。我们通过全面的模拟器和真实世界实验评估了该方法在协作承载任务中的有效性、泛化能力和鲁棒性，结果表明，在不同地形和物体类型下，模型的效能均表现出良好的适用性。模拟实验结果显示，相比基线方法，该模型在保持物体稳定性的前提下，能将人类的努力降低24.7%。真实世界实验验证了该方法在不同物体类型（箱子、桌子、担架等）和运动模式（直线、转弯、坡度攀爬）下的鲁棒性协作承载能力。23名参与者的用户研究结果表明，与基线模型相比，该方法能平均提高27.4%的协作承载效果。本方法无需外部传感器或复杂的交互模型，为实际部署提供了可行的解决方案。 

---
# Beyond a Single Perspective: Towards a Realistic Evaluation of Website Fingerprinting Attacks 

**Title (ZH)**: 超越单一视角：面向网站指纹识别攻击的现实评估 

**Authors**: Xinhao Deng, Jingyou Chen, Linxiao Yu, Yixiang Zhang, Zhongyi Gu, Changhao Qiu, Xiyuan Zhao, Ke Xu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14283)  

**Abstract**: Website Fingerprinting (WF) attacks exploit patterns in encrypted traffic to infer the websites visited by users, posing a serious threat to anonymous communication systems. Although recent WF techniques achieve over 90% accuracy in controlled experimental settings, most studies remain confined to single scenarios, overlooking the complexity of real-world environments. This paper presents the first systematic and comprehensive evaluation of existing WF attacks under diverse realistic conditions, including defense mechanisms, traffic drift, multi-tab browsing, early-stage detection, open-world settings, and few-shot scenarios. Experimental results show that many WF techniques with strong performance in isolated settings degrade significantly when facing other conditions. Since real-world environments often combine multiple challenges, current WF attacks are difficult to apply directly in practice. This study highlights the limitations of WF attacks and introduces a multidimensional evaluation framework, offering critical insights for developing more robust and practical WF attacks. 

**Abstract (ZH)**: 网站指纹识别（WF）攻击通过利用加密流量中的模式来推断用户访问的网站，对匿名通信系统构成严重威胁。尽管近期的WF技术在受控实验环境中实现超过90%的准确性，大部分研究仍然局限于单一场景，忽略了实际环境的复杂性。本文首次在多种现实条件下系统性地评估现有的WF攻击，包括防御机制、流量漂移、多标签浏览、早期检测、开放世界设置和少样本场景。实验结果表明，许多在孤立环境中表现优异的WF技术在面对其他条件时性能显著下降。由于实际环境经常结合多种挑战，当前的WF攻击在实践中难以直接应用。本文突出了WF攻击的局限性，并引入一个多维度的评估框架，为开发更 robust 和实际的WF攻击提供了关键见解。 

---
# PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering 

**Title (ZH)**: PRISM: 基于LLM的多跳问答代理检索 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14278)  

**Abstract**: Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines. 

**Abstract (ZH)**: 检索在多跳问答（QA）中起着核心作用，其中回答复杂问题需要汇集多个证据。我们介绍了一种代理检索系统，该系统通过结构化的循环利用大型语言模型（LLMs）以高精度和召回率检索相关证据。该框架包括三个专业代理：问题分析器将多跳问题分解为子问题，选择器为每个子问题识别最相关的上下文（专注于精度），添加器引入任何缺失的证据（专注于召回）。选择器和添加器之间的迭代交互生成一个紧凑且全面的支持段落集。特别是，它在过滤掉干扰内容的同时提高了检索准确性，使下游问答模型能够依赖较少的无关信息超越全上下文答案的准确性。在四个多跳问答基准测试集——HotpotQA、2WikiMultiHopQA、MuSiQue和MultiHopRAG——上的实验表明，我们的方法始终优于强大的基线。 

---
# Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation 

**Title (ZH)**: 少即是多：噪声消除以增强知识图 aras中的检索生成 

**Authors**: Yilun Zheng, Dan Yang, Jie Li, Lin Shang, Lihui Chen, Jiahao Xu, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14271)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enable large language models (LLMs) instant access to relevant information for the generative process, demonstrating their superior performance in addressing common LLM challenges such as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG further extends this paradigm by incorporating knowledge graphs (KGs) to leverage rich, structured connections for more precise and inferential responses. A critical challenge, however, is that most Graph-based RAG systems rely on LLMs for automated KG construction, often yielding noisy KGs with redundant entities and unreliable relationships. This noise degrades retrieval and generation performance while also increasing computational cost. Crucially, current research does not comprehensively address the denoising problem for LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), a framework that addresses these challenges through: (1) entity resolution, which eliminates redundant entities, and (2) triple reflection, which removes erroneous relations. Together, these techniques yield more compact, higher-quality KGs that significantly outperform their unprocessed counterparts. Beyond the methods, we conduct a systematic evaluation of entity resolution for LLM-generated KGs, examining different blocking strategies, embedding choices, similarity metrics, and entity merging techniques. To the best of our knowledge, this is the first comprehensive exploration of entity resolution in LLM-generated KGs. Our experiments demonstrate that this straightforward approach not only drastically reduces graph size but also consistently improves question answering performance across diverse popular Graph-based RAG variants. 

**Abstract (ZH)**: 基于检索增强生成的去噪知识图谱（DEG-RAG） 

---
# CAST: Compositional Analysis via Spectral Tracking for Understanding Transformer Layer Functions 

**Title (ZH)**: CAST: 组合分析通过谱跟踪理解Transformer层函数 

**Authors**: Zihao Fu, Ming Liao, Chris Russell, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14262)  

**Abstract**: Large language models have achieved remarkable success but remain largely black boxes with poorly understood internal mechanisms. To address this limitation, many researchers have proposed various interpretability methods including mechanistic analysis, probing classifiers, and activation visualization, each providing valuable insights from different perspectives. Building upon this rich landscape of complementary approaches, we introduce CAST (Compositional Analysis via Spectral Tracking), a probe-free framework that contributes a novel perspective by analyzing transformer layer functions through direct transformation matrix estimation and comprehensive spectral analysis. CAST offers complementary insights to existing methods by estimating the realized transformation matrices for each layer using Moore-Penrose pseudoinverse and applying spectral analysis with six interpretable metrics characterizing layer behavior. Our analysis reveals distinct behaviors between encoder-only and decoder-only models, with decoder models exhibiting compression-expansion cycles while encoder models maintain consistent high-rank processing. Kernel analysis further demonstrates functional relationship patterns between layers, with CKA similarity matrices clearly partitioning layers into three phases: feature extraction, compression, and specialization. 

**Abstract (ZH)**: 大型语言模型已经取得了显著的成功，但仍然主要是黑箱模型，内部机制 poorly understood。为了解决这一局限性，许多研究者提出了各种解释性方法，包括机制分析、探针分类器和激活可视化，每种方法都从不同的角度提供了宝贵的见解。在此基础上，我们引入了CAST（Compositional Analysis via Spectral Tracking）——一种无需探针的框架，通过直接估计变换层函数的变换矩阵并进行全面的谱分析，提供了一个新颖的视角。CAST通过使用莫尔-彭罗塞伪逆估计每层的实际变换矩阵，并应用六种可解释的指标进行谱分析，为现有的方法提供了互补的见解。我们的分析揭示了编码器模型和解码器模型之间不同的行为模式，解码器模型表现出压缩-扩张周期，而编码器模型则保持一致的高秩处理。核分析进一步展示了各层之间的功能关系模式，CKA相似矩阵明显将层分为三个阶段：特征提取、压缩和专业化。 

---
# Do Joint Language-Audio Embeddings Encode Perceptual Timbre Semantics? 

**Title (ZH)**: 联合语言-音频嵌入是否编码感知音色语义？ 

**Authors**: Qixin Deng, Bryan Pardo, Thrasyvoulos N Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2510.14249)  

**Abstract**: Understanding and modeling the relationship between language and sound is critical for applications such as music information retrieval,text-guided music generation, and audio captioning. Central to these tasks is the use of joint language-audio embedding spaces, which map textual descriptions and auditory content into a shared embedding space. While multimodal embedding models such as MS-CLAP, LAION-CLAP, and MuQ-MuLan have shown strong performance in aligning language and audio, their correspondence to human perception of timbre, a multifaceted attribute encompassing qualities such as brightness, roughness, and warmth, remains underexplored. In this paper, we evaluate the above three joint language-audio embedding models on their ability to capture perceptual dimensions of timbre. Our findings show that LAION-CLAP consistently provides the most reliable alignment with human-perceived timbre semantics across both instrumental sounds and audio effects. 

**Abstract (ZH)**: 理解语言与声音之间的关系对于音乐信息检索、基于文本的音乐生成和音频描述等应用至关重要。这些任务的核心在于使用联合语言-音频嵌入空间，将文本描述和听觉内容映射到共享嵌入空间。虽然MS-CLAP、LAION-CLAP和MuQ-MuLan等多模态嵌入模型在语言和音频对齐方面表现出色，但它们与人类感知的音色这一多维度属性之间的对应关系，包括亮度、粗糙度和温暖度等品质，仍待进一步探讨。在本文中，我们评估了上述三种联合语言-音频嵌入模型在捕捉音色感知维度方面的能力。我们的研究发现，LAION-CLAP在乐器声音和音频效果两种情况下都提供了最可靠的人类感知音色语义对齐。 

---
# Policy Regularized Distributionally Robust Markov Decision Processes with Linear Function Approximation 

**Title (ZH)**: 带线性函数逼近的策略正则化分布鲁棒马尔可夫决策过程 

**Authors**: Jingwen Gu, Yiting He, Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14246)  

**Abstract**: Decision-making under distribution shift is a central challenge in reinforcement learning (RL), where training and deployment environments differ. We study this problem through the lens of robust Markov decision processes (RMDPs), which optimize performance against adversarial transition dynamics. Our focus is the online setting, where the agent has only limited interaction with the environment, making sample efficiency and exploration especially critical. Policy optimization, despite its success in standard RL, remains theoretically and empirically underexplored in robust RL. To bridge this gap, we propose \textbf{D}istributionally \textbf{R}obust \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization algorithm (DR-RPO), a model-free online policy optimization method that learns robust policies with sublinear regret. To enable tractable optimization within the softmax policy class, DR-RPO incorporates reference-policy regularization, yielding RMDP variants that are doubly constrained in both transitions and policies. To scale to large state-action spaces, we adopt the $d$-rectangular linear MDP formulation and combine linear function approximation with an upper confidence bonus for optimistic exploration. We provide theoretical guarantees showing that policy optimization can achieve polynomial suboptimality bounds and sample efficiency in robust RL, matching the performance of value-based approaches. Finally, empirical results across diverse domains corroborate our theory and demonstrate the robustness of DR-RPO. 

**Abstract (ZH)**: 分布转移下的决策制定是强化学习（RL）中的一个核心挑战，其中训练环境和部署环境存在差异。我们通过鲁棒马尔可夫决策过程（RMDPs）的视角研究这一问题，RMDPs旨在优化对抗性转换动力学下的性能。我们的重点是在线设置，其中智能体与环境的交互非常有限，这使得样本效率和探索尤为重要。尽管策略优化在标准RL中取得了成功，但在鲁棒RL中，策略优化的理论和实证研究仍然不足。为弥合这一差距，我们提出了一种基于模型的在线策略优化方法——分布鲁棒正则化策略优化算法（DR-RPO），该方法能够学习具有亚线性遗憾的鲁棒策略。为了在softmax策略类中实现可处理的优化，DR-RPO引入了参考策略正则化，从而产生在转换和策略方面双约束的RMDP变体。为了扩展到大规模状态-动作空间，我们采用了$d$-矩形线性MDP形式化描述，并结合了线性函数逼近和乐观探索的上置信边界。我们提供了理论保证，表明策略优化在鲁棒RL中可以实现多项式次优性边界和样本效率，达到基于值的方法的性能。最后，我们在多个领域的实验结果验证了我们的理论，并展示了DR-RPO的鲁棒性。 

---
# Reinforcement Learning for Unsupervised Domain Adaptation in Spatio-Temporal Echocardiography Segmentation 

**Title (ZH)**: 时空超声心肌分割的无监督领域适应强化学习 

**Authors**: Arnaud Judge, Nicolas Duchateau, Thierry Judge, Roman A. Sandler, Joseph Z. Sokol, Christian Desrosiers, Olivier Bernard, Pierre-Marc Jodoin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14244)  

**Abstract**: Domain adaptation methods aim to bridge the gap between datasets by enabling knowledge transfer across domains, reducing the need for additional expert annotations. However, many approaches struggle with reliability in the target domain, an issue particularly critical in medical image segmentation, where accuracy and anatomical validity are essential. This challenge is further exacerbated in spatio-temporal data, where the lack of temporal consistency can significantly degrade segmentation quality, and particularly in echocardiography, where the presence of artifacts and noise can further hinder segmentation performance. To address these issues, we present RL4Seg3D, an unsupervised domain adaptation framework for 2D + time echocardiography segmentation. RL4Seg3D integrates novel reward functions and a fusion scheme to enhance key landmark precision in its segmentations while processing full-sized input videos. By leveraging reinforcement learning for image segmentation, our approach improves accuracy, anatomical validity, and temporal consistency while also providing, as a beneficial side effect, a robust uncertainty estimator, which can be used at test time to further enhance segmentation performance. We demonstrate the effectiveness of our framework on over 30,000 echocardiographic videos, showing that it outperforms standard domain adaptation techniques without the need for any labels on the target domain. Code is available at this https URL. 

**Abstract (ZH)**: 基于RL的无监督3D心超时空分割领域适应方法_rl4seg3D 

---
# Spatial Computing Communications for Multi-User Virtual Reality in Distributed Mobile Edge Computing Network 

**Title (ZH)**: 分布式移动边缘计算网络中多用户虚拟现实的时空计算通信 

**Authors**: Caolu Xu, Zhiyong Chen, Meixia Tao, Li Song, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14243)  

**Abstract**: Immersive virtual reality (VR) applications impose stringent requirements on latency, energy efficiency, and computational resources, particularly in multi-user interactive scenarios. To address these challenges, we introduce the concept of spatial computing communications (SCC), a framework designed to meet the latency and energy demands of multi-user VR over distributed mobile edge computing (MEC) networks. SCC jointly represents the physical space, defined by users and base stations, and the virtual space, representing shared immersive environments, using a probabilistic model of user dynamics and resource requirements. The resource deployment task is then formulated as a multi-objective combinatorial optimization (MOCO) problem that simultaneously minimizes system latency and energy consumption across distributed MEC resources. To solve this problem, we propose MO-CMPO, a multi-objective consistency model with policy optimization that integrates supervised learning and reinforcement learning (RL) fine-tuning guided by preference weights. Leveraging a sparse graph neural network (GNN), MO-CMPO efficiently generates Pareto-optimal solutions. Simulations with real-world New Radio base station datasets demonstrate that MO-CMPO achieves superior hypervolume performance and significantly lower inference latency than baseline methods. Furthermore, the analysis reveals practical deployment patterns: latency-oriented solutions favor local MEC execution to reduce transmission delay, while energy-oriented solutions minimize redundant placements to save energy. 

**Abstract (ZH)**: 沉浸式虚拟现实（VR）应用对延迟、能量效率和计算资源提出了严格要求，特别是在多用户交互场景中。为应对这些挑战，我们引入了空间计算通信（SCC）的概念，这是一种设计用于分布式移动边缘计算（MEC）网络中满足多用户VR延迟和能量需求的框架。SCC使用用户动态和资源需求的概率模型，联合表示由用户和基站定义的物理空间以及表示共享沉浸环境的虚拟空间。随后，资源部署任务被形式化为同时最小化分布式MEC资源上系统延迟和能量消耗的多目标组合优化（MOCO）问题。为了解决这个问题，我们提出了MO-CMPO，这是一种结合监督学习和由偏好权重引导的强化学习（RL）微调的多目标一致性模型和策略优化方法。利用稀疏图神经网络（GNN），MO-CMPO有效地生成了帕累托最优解。使用实际的New Radio基站数据集进行的仿真实验表明，MO-CMPO在超体积性能和推理延迟方面优于基线方法。此外，分析揭示了实际部署模式：延迟导向的解决方案倾向于局部执行MEC以减少传输延迟，而能量导向的解决方案倾向于最小化冗余放置以节省能量。 

---
# Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models 

**Title (ZH)**: 实现IOI金牌奖牌的开放权重模型的测试时计算扩展 

**Authors**: Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain, Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2510.14232)  

**Abstract**: Competitive programming has become a rigorous benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). The International Olympiad in Informatics (IOI) stands out as one of the most prestigious annual competitions in competitive programming and has become a key benchmark for comparing human and AI-level programming ability. While several proprietary models have been claimed to achieve gold medal-level performance at the IOI, often with undisclosed methods, achieving comparable results with open-weight models remains a significant challenge. In this paper, we present \gencluster, a scalable and reproducible test-time compute framework that attains IOI gold-level performance using open-weight models. It combines large-scale generation, behavioral clustering, ranking, and a round-robin submission strategy to efficiently explore diverse solution spaces under limited validation budgets. Our experiments show that the performance of our proposed approach scales consistently with available compute, narrowing the gap between open and closed systems. Notably, we will show that GenCluster can achieve a gold medal at IOI 2025 for the first time with an open-weight model gpt-oss-120b, setting a new benchmark for transparent and reproducible evaluation of reasoning in LLMs. 

**Abstract (ZH)**: 竞争编程已成为评估大型语言模型推理和问题解决能力的严格基准。国际 информatics奥林匹克（IOI）是竞争编程中最具声望的年度竞赛之一，并已成为比较人类和AI级别编程能力的关键基准。尽管已经有多个私有模型声称在IOI中达到了金牌水平，且往往未披露其方法，但使用开放权重模型实现类似结果仍是一项重大挑战。本文介绍了一种可扩展且可重现的测试时计算框架——\gencluster，该框架使用开放权重模型实现了IOI金牌水平的成绩。该框架结合了大规模生成、行为聚类、排名以及单循环提交策略，在有限的验证预算下高效探索不同的解决方案空间。我们的实验表明，所提出方法的性能与可用计算资源的一致性增加，缩小了开放系统与封闭系统的差距。值得注意的是，我们将展示GenCluster能够首次使用开放权重模型gpt-oss-120b在IOI 2025中获得金牌，为透明且可重现的评估大型语言模型推理能力设立了新基准。 

---
# Large Scale Retrieval for the LinkedIn Feed using Causal Language Models 

**Title (ZH)**: 使用因果语言模型的大型规模LinkedIn动态检索 

**Authors**: Sudarshan Srinivasa Ramanujam, Antonio Alonso, Saurabh Kataria, Siddharth Dangi, Akhilesh Gupta, Birjodh Singh Tiwana, Manas Somaiya, Luke Simon, David Byrne, Sojeong Ha, Sen Zhou, Andrei Akterskii, Zhanglong Liu, Samira Sriram, Crescent Xiong, Zhoutao Pei, Angela Shao, Alex Li, Annie Xiao, Caitlin Kolb, Thomas Kistler, Zach Moore, Hamed Firooz  

**Link**: [PDF](https://arxiv.org/pdf/2510.14223)  

**Abstract**: In large scale recommendation systems like the LinkedIn Feed, the retrieval stage is critical for narrowing hundreds of millions of potential candidates to a manageable subset for ranking. LinkedIn's Feed serves suggested content from outside of the member's network (based on the member's topical interests), where 2000 candidates are retrieved from a pool of hundreds of millions candidate with a latency budget of a few milliseconds and inbound QPS of several thousand per second. This paper presents a novel retrieval approach that fine-tunes a large causal language model (Meta's LLaMA 3) as a dual encoder to generate high quality embeddings for both users (members) and content (items), using only textual input. We describe the end to end pipeline, including prompt design for embedding generation, techniques for fine-tuning at LinkedIn's scale, and infrastructure for low latency, cost effective online serving. We share our findings on how quantizing numerical features in the prompt enables the information to get properly encoded in the embedding, facilitating greater alignment between the retrieval and ranking layer. The system was evaluated using offline metrics and an online A/B test, which showed substantial improvements in member engagement. We observed significant gains among newer members, who often lack strong network connections, indicating that high-quality suggested content aids retention. This work demonstrates how generative language models can be effectively adapted for real time, high throughput retrieval in industrial applications. 

**Abstract (ZH)**: 在 LinkedIn 动态这样的大规模推荐系统中，检索阶段对于将数亿潜在候选者缩小到可管理的排名子集至关重要。LinkedIn 动态从成员的领域兴趣出发，为成员推荐外部网络的内容，在毫秒级延迟预算和每秒数千次的入站 QPS 下，从数亿候选者池中检索出 2000 个候选者。本文介绍了一种新颖的检索方法，该方法针对 Meta 的 LLaMA 3 大型因变量语言模型进行微调，作为双编码器生成高质量的用户（成员）和内容（项目）嵌入，仅使用文本输入。本文描述了端到端的流程，包括嵌入生成的提示设计、在 LinkedIn 规模下进行微调的技术以及适用于低延迟和成本效益高的在线服务的基础设施。文中探讨了在提示中量化数值特征如何使信息能够正确编码在嵌入中，从而实现检索和排名层之间更强的对齐。该系统通过离线指标评估和在线 A/B 测试进行了评估，结果显示成员参与度显著提高。我们观察到新成员特别获得显著收益，这些成员通常缺乏强大的网络联系，表明高质量的推荐内容有助于留存。本文展示了生成语言模型如何有效地适应工业应用中的实时、高吞吐量检索场景。 

---
# LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning 

**Title (ZH)**: LiteStage：面向延迟的多阶段推理中的层跳过方法 

**Authors**: Beomseok Kang, Jiwon Song, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14211)  

**Abstract**: Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage achieves up to 1.70x speedup with less than 4.0% accuracy loss, outperforming prior training-free layer skipping methods. 

**Abstract (ZH)**: 面向多阶段推理的 AwareStage 低延迟层跳过框架 

---
# DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans 

**Title (ZH)**: DPRF：一种优化个性化LLM角色扮演代理与人类行为对齐的可泛化动态人设精炼框架 

**Authors**: Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14205)  

**Abstract**: The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI. 

**Abstract (ZH)**: 新兴的大语言模型角色扮演代理（LLM RPAs）旨在模拟个体人类行为，但个性保真度往往因缺乏验证的手工创建的人物档案（例如，挑中的信息和个人特质）而受损。为解决这一局限，我们提出了一种动态个性细化框架（DPRF）。DPRF通过迭代地识别生成行为与人类真实行为之间的认知差异（无论是自由形式的还是理论支撑的结构化分析），来优化LLM RPAs的行为与目标个体行为的一致性，并细化人物档案以减轻这些差异。我们使用五种LLM在四种不同行为预测场景（正式辩论、涉及心理健康问题的社交媒体帖子、公开采访和电影）上对DPRF进行了评估，结果表明DPRF可以显著提高行为一致性，并在不同模型和任务之间具有泛化能力。本项工作提供了一种稳健的方法来创建高保真人物档案，并增强下游应用（如用户仿真、社会研究和个人化AI）的有效性。 

---
# MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation 

**Title (ZH)**: MAFA：一种具备可配置任务适应性的企业规模标注多代理框架 

**Authors**: Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14184)  

**Abstract**: We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges. 

**Abstract (ZH)**: 面向标注的多Agent框架：MAFA在金融机构大规模标注流程中的应用 

---
# Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures 

**Title (ZH)**: 虚拟存在：基于多视角性能捕捉的自定义摄像机可控视频扩散模型 

**Authors**: Yuancheng Xu, Wenqi Xian, Li Ma, Julien Philip, Ahmet Levent Taşel, Yiwei Zhao, Ryan Burgert, Mingming He, Oliver Hermann, Oliver Pilarski, Rahul Garg, Paul Debevec, Ning Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14179)  

**Abstract**: We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: this https URL. 

**Abstract (ZH)**: 我们提出了一种框架，通过一种新颖的自定义数据管道，实现了视频扩散模型中的多视图角色一致性与3D相机控制。我们使用4D高斯点图（4DGS）重渲染记录的体三维捕捉表演，并结合视频重光照模型获得的光照变化，训练角色一致性组件。我们在这些数据上对最先进的开源视频扩散模型进行微调，以提供强大的多视图身份保真、精确的相机控制和光照适应性。该框架还支持虚拟生产的核心能力，包括使用两种方法（联合训练和噪声融合）进行多主体生成，后者在推理时能高效地组合独立定制的模型；同时实现了场景和现实生活视频的自定义，以及在自定义过程中对动作和空间布局的控制。大量实验表明，视频质量得到了提高，个性化精度更高，相机控制和光照适应性增强，推动了视频生成与虚拟生产的融合。我们的项目页面可在以下链接访问：this https URL。 

---
# Towards Reversible Model Merging For Low-rank Weights 

**Title (ZH)**: 面向低秩权重的可逆模型合并方法 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.14163)  

**Abstract**: Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations, either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ``revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin. 

**Abstract (ZH)**: 基于可逆模型合并的低秩模型压缩与重构 

---
# FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API 

**Title (ZH)**: FinAI数据助手：基于OpenAI函数调用API的金融数据库查询处理 

**Authors**: Juhyeong Kim, Yejin Kim, Youngbin Lee, Hyunwoo Byun  

**Link**: [PDF](https://arxiv.org/pdf/2510.14162)  

**Abstract**: We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment. 

**Abstract (ZH)**: FinAI数据助手：结合大型语言模型与OpenAI函数调用API的金融数据库自然语言查询实用方法 

---
# Inferred global dense residue transition graphs from primary structure sequences enable protein interaction prediction via directed graph convolutional neural networks 

**Title (ZH)**: 从初级结构序列推导出的全局密集残基转换图通过有向图卷积神经网络进行蛋白质相互作用预测 

**Authors**: Islam Akef Ebeid, Haoteng Tang, Pengfei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14139)  

**Abstract**: Introduction Accurate prediction of protein-protein interactions (PPIs) is crucial for understanding cellular functions and advancing drug development. Existing in-silico methods use direct sequence embeddings from Protein Language Models (PLMs). Others use Graph Neural Networks (GNNs) for 3D protein structures. This study explores less computationally intensive alternatives. We introduce a novel framework for downstream PPI prediction through link prediction. Methods We introduce a two-stage graph representation learning framework, ProtGram-DirectGCN. First, we developed ProtGram. This approach models a protein's primary structure as a hierarchy of globally inferred n-gram graphs. In these graphs, residue transition probabilities define edge weights. Each edge connects a pair of residues in a directed graph. The probabilities are aggregated from a large corpus of sequences. Second, we propose DirectGCN, a custom directed graph convolutional neural network. This model features a unique convolutional layer. It processes information through separate path-specific transformations: incoming, outgoing, and undirected. A shared transformation is also applied. These paths are combined via a learnable gating mechanism. We apply DirectGCN to ProtGram graphs to learn residue-level embeddings. These embeddings are pooled via attention to generate protein-level embeddings for prediction. Results We first established the efficacy of DirectGCN on standard node classification benchmarks. Its performance matches established methods on general datasets. The model excels at complex, directed graphs with dense, heterophilic structures. When applied to PPI prediction, the full ProtGram-DirectGCN framework delivers robust predictive power. This strong performance holds even with limited training data. 

**Abstract (ZH)**: 介绍 准确预测蛋白质-蛋白质相互作用（PPIs）对于理解细胞功能和推进药物开发至关重要。现有计算方法使用蛋白质语言模型（PLMs）的直接序列嵌入。其他方法使用图神经网络（GNNs）处理三维蛋白质结构。本研究探索了更少计算成本的替代方案。我们引入了一种新的框架，通过链接预测进行下游PPI预测。方法 我们提出了一种两阶段图表示学习框架，名为ProtGram-DirectGCN。首先，我们开发了ProtGram。该方法将蛋白质的一级结构建模为全局推断的n-克隆图层次结构。在这类图中，残基转换概率定义边权。每条边连接有向图中的残基对。这些概率是从一个大规模序列语料库中聚合而来的。其次，我们提出了DirectGCN，这是一种定制化的有向图卷积神经网络。该模型具有独特的卷积层，通过单独的路径特定变换处理信息：入边、出边和无向边。此外还应用了共享变换。这些路径通过可学习的门控机制结合。我们将DirectGCN应用于ProtGram图，学习残基级嵌入，并通过注意力机制池化生成蛋白质级嵌入进行预测。结果 我们首先在标准节点分类基准上验证了DirectGCN的有效性。其性能在通用数据集上与现有方法相当。该模型在复杂、有向图且结构稠密和异质的结构上表现出色。在应用于PPI预测时，完整的ProtGram-DirectGCN框架提供了稳健的预测能力，即使在训练数据有限的情况下也能保持这种性能。 

---
# Toward Cybersecurity-Expert Small Language Models 

**Title (ZH)**: 面向网络安全的专家小型语言模型 

**Authors**: Matan Levi, Daniel Ohayon, Ariel Blobstein, Ravid Sagi, Ian Molloy, Yair Allouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.14113)  

**Abstract**: Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B-20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. On core threat-investigation tasks, such as correlating vulnerabilities and bug tickets with weaknesses, our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在改变日常应用，但在网络安全领域的部署因缺乏高质量的专业化模型和训练数据集而滞后。为填补这一缺口，我们推出了CyberPal 2.0，这是一个从4B至20B参数的网络安全专家小型语言模型系列。为训练CyberPal 2.0，我们使用SecKnowledge 2.0数据丰富和格式化管道生成了一个增强的链式思维网络安全指令数据集，该管道结合了专家指导的推理格式引导与LLM驱动的多步骤接地，产生了更高保真度、任务导向的推理轨迹，用于安全任务。在各种网络安全基准测试中，CyberPal 2.0持续优于其基线模型，并在开源和闭源前沿模型中达到或超越了它们的性能，同时保持其规模仅为它们的一小部分。在核心网络安全知识任务上，我们的模型超过了几乎所有测试的前沿模型，仅次于Sec-Gemini v1。在核心威胁调查任务，如将漏洞和错误报告与弱点关联起来的任务中，我们20B参数的最佳模型在性能上超过了GPT-4o、o1、o3-mini和Sec-Gemini v1，排名第一，而我们的最小4B参数模型排名第二。 

---
# Extracting latent representations from X-ray spectra. Classification, regression, and accretion signatures of Chandra sources 

**Title (ZH)**: 从X射线光谱中提取潜藏表示。Chandra源的分类、回归及积累特征。 

**Authors**: Nicolò Oreste Pinciroli Vago, Juan Rafael Martínez-Galarza, Roberta Amato  

**Link**: [PDF](https://arxiv.org/pdf/2510.14102)  

**Abstract**: The study of X-ray spectra is crucial to understanding the physical nature of astrophysical sources. Machine learning methods can extract compact and informative representations of data from large datasets. The Chandra Source Catalog (CSC) provides a rich archive of X-ray spectral data, which remains largely underexplored in this context. This work aims to develop a compact and physically meaningful representation of Chandra X-ray spectra using deep learning. To verify that the learned representation captures relevant information, we evaluate it through classification, regression, and interpretability analyses. We use a transformer-based autoencoder to compress X-ray spectra. The input spectra, drawn from the CSC, include only high-significance detections. Astrophysical source types and physical summary statistics are compiled from external catalogs. We evaluate the learned representation in terms of spectral reconstruction accuracy, clustering performance on 8 known astrophysical source classes, and correlation with physical quantities such as hardness ratios and hydrogen column density ($N_H$). The autoencoder accurately reconstructs spectra with 8 latent variables. Clustering in the latent space yields a balanced classification accuracy of $\sim$40% across the 8 source classes, increasing to $\sim$69% when restricted to AGNs and stellar-mass compact objects exclusively. Moreover, latent features correlate with non-linear combinations of spectral fluxes, suggesting that the compressed representation encodes physically relevant information. The proposed autoencoder-based pipeline is a powerful tool for the representation and interpretation of X-ray spectra, providing a compact latent space that supports both classification and the estimation of physical properties. This work demonstrates the potential of deep learning for spectral studies and uncovering new patterns in X-ray data. 

**Abstract (ZH)**: X射线光谱研究对于理解天体物理源的物理性质至关重要。机器学习方法可以从大型数据集中提取紧凑且富有信息量的数据表示。钱德拉源目录（CSC）提供了丰富的X射线光谱数据集，这些数据在该领域尚未得到充分利用。本工作旨在使用深度学习开发Chandra X射线光谱的紧凑且物理意义丰富的表示。为验证所学习的表示是否捕获了相关信息，我们通过分类、回归和可解释性分析对其进行评估。我们使用基于变换器的自动编码器压缩X射线光谱。输入光谱仅包括CSC中的高信噪比检测结果。天体物理源类型和物理总括统计量是从外部目录中编制的。我们从光谱重建准确性、基于8个已知天体物理源类的聚类性能和与物理量（如硬度比和氢柱密度$N_H$）的相关性等方面评估所学习的表示。自动编码器使用8个潜在变量准确重建光谱。潜在空间聚类在8个源类上的平衡分类准确率为约40%，当仅限于活跃星系核（AGNs）和恒星质量致密天体时，分类准确率提高到约69%。此外，潜在特征与光谱通量的非线性组合相关，表明压缩表示包含了物理相关的信息。提出的基于自动编码器的方法是一个强大的工具，用于X射线光谱的表示和解释，提供了一个支持分类和物理性质估计的紧凑潜在空间。本工作展示了深度学习在光谱研究中以及在挖掘X射线数据中的新模式方面具有潜在价值。 

---
# Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning 

**Title (ZH)**: 通过递归潜在空间推理解锁变压器模型的分布外泛化能力 

**Authors**: Awni Altabaa, Siyu Chen, John Lafferty, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14095)  

**Abstract**: Systematic, compositional generalization beyond the training distribution remains a core challenge in machine learning -- and a critical bottleneck for the emergent reasoning abilities of modern language models. This work investigates out-of-distribution (OOD) generalization in Transformer networks using a GSM8K-style modular arithmetic on computational graphs task as a testbed. We introduce and explore a set of four architectural mechanisms aimed at enhancing OOD generalization: (i) input-adaptive recurrence; (ii) algorithmic supervision; (iii) anchored latent representations via a discrete bottleneck; and (iv) an explicit error-correction mechanism. Collectively, these mechanisms yield an architectural approach for native and scalable latent space reasoning in Transformer networks with robust algorithmic generalization capabilities. We complement these empirical results with a detailed mechanistic interpretability analysis that reveals how these mechanisms give rise to robust OOD generalization abilities. 

**Abstract (ZH)**: Transformer网络中基于计算图模块化算术的离分布泛化：一种增强离分布泛化能力的架构方法 

---
# Every Language Model Has a Forgery-Resistant Signature 

**Title (ZH)**: 每种语言模型都有防伪签名。 

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14086)  

**Abstract**: The ubiquity of closed-weight language models with public-facing APIs has generated interest in forensic methods, both for extracting hidden model details (e.g., parameters) and for identifying models by their outputs. One successful approach to these goals has been to exploit the geometric constraints imposed by the language model architecture and parameters. In this work, we show that a lesser-known geometric constraint--namely, that language model outputs lie on the surface of a high-dimensional ellipse--functions as a signature for the model and can be used to identify the source model of a given output. This ellipse signature has unique properties that distinguish it from existing model-output association methods like language model fingerprints. In particular, the signature is hard to forge: without direct access to model parameters, it is practically infeasible to produce log-probabilities (logprobs) on the ellipse. Secondly, the signature is naturally occurring, since all language models have these elliptical constraints. Thirdly, the signature is self-contained, in that it is detectable without access to the model inputs or the full weights. Finally, the signature is compact and redundant, as it is independently detectable in each logprob output from the model. We evaluate a novel technique for extracting the ellipse from small models and discuss the practical hurdles that make it infeasible for production-scale models. Finally, we use ellipse signatures to propose a protocol for language model output verification, analogous to cryptographic symmetric-key message authentication systems. 

**Abstract (ZH)**: 闭权值语言模型的普遍存在引发了对公共界面API的法医学方法的兴趣，这些方法既可用于提取隐藏的模型细节（例如，参数），也可用于通过输出识别模型。一种成功的方法是利用语言模型架构和参数施加的几何约束。在本工作中，我们展示了一种较少为人知的几何约束——即语言模型输出位于高维椭球的表面——可以作为模型的特征签名，并可用于识别给定输出的源模型。此椭球签名具有独特的性质，使其有别于现有的模型输出关联方法，如语言模型指纹。特别是，该签名难以伪造：在没有直接访问模型参数的情况下，生成椭球上的对数概率（logprobs）实际上是不可行的。其次，该签名是自发产生的，因为所有语言模型都具有这些椭圆约束。第三，该签名是自包含的，即在不访问模型输入或全部权重的情况下可以检测到。最后，该签名是紧凑且冗余的，因为在模型的每个对数概率输出中独立可检测到。我们评估了一种从小型模型中提取椭球的新技术，并讨论了使其在生产规模模型中不可行的实际障碍。最后，我们使用椭球签名提出了一种语言模型输出验证协议，类似于加密的对称密钥消息认证系统。 

---
# DiffOPF: Diffusion Solver for Optimal Power Flow 

**Title (ZH)**: DiffOPF: 基于扩散的最优功率流求解器 

**Authors**: Milad Hoseinpour, Vladimir Dvorkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14075)  

**Abstract**: The optimal power flow (OPF) is a multi-valued, non-convex mapping from loads to dispatch setpoints. The variability of system parameters (e.g., admittances, topology) further contributes to the multiplicity of dispatch setpoints for a given load. Existing deep learning OPF solvers are single-valued and thus fail to capture the variability of system parameters unless fully represented in the feature space, which is prohibitive. To solve this problem, we introduce a diffusion-based OPF solver, termed \textit{DiffOPF}, that treats OPF as a conditional sampling problem. The solver learns the joint distribution of loads and dispatch setpoints from operational history, and returns the marginal dispatch distributions conditioned on loads. Unlike single-valued solvers, DiffOPF enables sampling statistically credible warm starts with favorable cost and constraint satisfaction trade-offs. We explore the sample complexity of DiffOPF to ensure the OPF solution within a prescribed distance from the optimization-based solution, and verify this experimentally on power system benchmarks. 

**Abstract (ZH)**: 基于扩散的最优功率流求解器（DiffOPF） 

---
# Exploratory Causal Inference in SAEnce 

**Title (ZH)**: 探索性因果推理在科学中的应用 

**Authors**: Tommaso Mencattini, Riccardo Cadei, Francesco Locatello  

**Link**: [PDF](https://arxiv.org/pdf/2510.14073)  

**Abstract**: Randomized Controlled Trials are one of the pillars of science; nevertheless, they rely on hand-crafted hypotheses and expensive analysis. Such constraints prevent causal effect estimation at scale, potentially anchoring on popular yet incomplete hypotheses. We propose to discover the unknown effects of a treatment directly from data. For this, we turn unstructured data from a trial into meaningful representations via pretrained foundation models and interpret them via a sparse autoencoder. However, discovering significant causal effects at the neural level is not trivial due to multiple-testing issues and effects entanglement. To address these challenges, we introduce Neural Effect Search, a novel recursive procedure solving both issues by progressive stratification. After assessing the robustness of our algorithm on semi-synthetic experiments, we showcase, in the context of experimental ecology, the first successful unsupervised causal effect identification on a real-world scientific trial. 

**Abstract (ZH)**: 从数据中发现治疗的未知效果：一种解决神经级显著因果效应发现挑战的递归方法 

---
# On the expressivity of sparse maxout networks 

**Title (ZH)**: 稀疏MaxOut网络的表征能力 

**Authors**: Moritz Grillo, Tobias Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.14068)  

**Abstract**: We study the expressivity of sparse maxout networks, where each neuron takes a fixed number of inputs from the previous layer and employs a, possibly multi-argument, maxout activation. This setting captures key characteristics of convolutional or graph neural networks. We establish a duality between functions computable by such networks and a class of virtual polytopes, linking their geometry to questions of network expressivity. In particular, we derive a tight bound on the dimension of the associated polytopes, which serves as the central tool for our analysis. Building on this, we construct a sequence of depth hierarchies. While sufficiently deep sparse maxout networks are universal, we prove that if the required depth is not reached, width alone cannot compensate for the sparsity of a fixed indegree constraint. 

**Abstract (ZH)**: 我们研究稀疏maxout网络的表達能力，其中每个神经元从上一层固定数量的输入中选取，并使用可能多参数的maxout激活函数。这一设置捕获了卷积或图神经网络的关键特性。我们建立了此类网络可计算函数与一类虚拟多面体之间的对偶关系，将它们的几何结构与网络的表達能力联系起来。特别是，我们推导出与这些多面体相关的维度的紧致界，这成为我们分析的核心工具。基于此，我们构造了一系列深度层次结构。虽然足够深的稀疏maxout网络具有通用性，但我们证明了如果没有达到所需的深度，宽度 alone 无法弥补固定入度约束下的稀疏性。 

---
# Optical Computation-in-Communication enables low-latency, high-fidelity perception in telesurgery 

**Title (ZH)**: 光学计算-通信使远程手术实现低延迟、高保真感知 

**Authors**: Rui Yang, Jiaming Hu, Jian-Qing Zheng, Yue-Zhen Lu, Jian-Wei Cui, Qun Ren, Yi-Jie Yu, John Edward Wu, Zhao-Yu Wang, Xiao-Li Lin, Dandan Zhang, Mingchu Tang, Christos Masouros, Huiyun Liu, Chin-Pang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14058)  

**Abstract**: Artificial intelligence (AI) holds significant promise for enhancing intraoperative perception and decision-making in telesurgery, where physical separation impairs sensory feedback and control. Despite advances in medical AI and surgical robotics, conventional electronic AI architectures remain fundamentally constrained by the compounded latency from serial processing of inference and communication. This limitation is especially critical in latency-sensitive procedures such as endovascular interventions, where delays over 200 ms can compromise real-time AI reliability and patient safety. Here, we introduce an Optical Computation-in-Communication (OCiC) framework that reduces end-to-end latency significantly by performing AI inference concurrently with optical communication. OCiC integrates Optical Remote Computing Units (ORCUs) directly into the optical communication pathway, with each ORCU experimentally achieving up to 69 tera-operations per second per channel through spectrally efficient two-dimensional photonic convolution. The system maintains ultrahigh inference fidelity within 0.1% of CPU/GPU baselines on classification and coronary angiography segmentation, while intrinsically mitigating cumulative error propagation, a longstanding barrier to deep optical network scalability. We validated the robustness of OCiC through outdoor dark fibre deployments, confirming consistent and stable performance across varying environmental conditions. When scaled globally, OCiC transforms long-haul fibre infrastructure into a distributed photonic AI fabric with exascale potential, enabling reliable, low-latency telesurgery across distances up to 10,000 km and opening a new optical frontier for distributed medical intelligence. 

**Abstract (ZH)**: 光学计算在通信中的框架（OCiC）：显著降低端到端延迟以增强远程手术中的感知和决策 

---
# Cyber-Resilient System Identification for Power Grid through Bayesian Integration 

**Title (ZH)**: 通过贝叶斯集成实现电力系统的 cyber-韧性识别 

**Authors**: Shimiao Li, Guannan Qu, Bryan Hooi, Vyas Sekar, Soummya Kar, Larry Pileggi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14043)  

**Abstract**: Power grids increasingly need real-time situational awareness under the ever-evolving cyberthreat landscape. Advances in snapshot-based system identification approaches have enabled accurately estimating states and topology from a snapshot of measurement data, under random bad data and topology errors. However, modern interactive, targeted false data can stay undetectable to these methods, and significantly compromise estimation accuracy. This work advances system identification that combines snapshot-based method with time-series model via Bayesian Integration, to advance cyber resiliency against both random and targeted false data. Using a distance-based time-series model, this work can leverage historical data of different distributions induced by changes in grid topology and other settings. The normal system behavior captured from historical data is integrated into system identification through a Bayesian treatment, to make solutions robust to targeted false data. We experiment on mixed random anomalies (bad data, topology error) and targeted false data injection attack (FDIA) to demonstrate our method's 1) cyber resilience: achieving over 70% reduction in estimation error under FDIA; 2) anomalous data identification: being able to alarm and locate anomalous data; 3) almost linear scalability: achieving comparable speed with the snapshot-based baseline, both taking <1min per time tick on the large 2,383-bus system using a laptop CPU. 

**Abstract (ZH)**: 基于贝叶斯整合的混合时间序列模型在电力系统识别中的应用：增强应对随机与 targeted 缺失数据的网络安全韧性 

---
# One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery 

**Title (ZH)**: 一个错误，成百个后续错误：大规模缺陷发现的预训练语言模型 

**Authors**: Qiushi Wu, Yue Xiao, Dhilung Kirat, Kevin Eykholt, Jiyong Jang, Douglas Lee Schales  

**Link**: [PDF](https://arxiv.org/pdf/2510.14036)  

**Abstract**: Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program.
In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset. 

**Abstract (ZH)**: 在大型程序中修复重复模式漏洞是一项具有挑战性的工作，需要大量的时间和精力。在发现漏洞后，它会被报告给项目的维护者，他们与报告者合作修复漏洞并最终关闭这个问题。然而，在程序中，经常存在类似的代码段，这些代码段可能也包含相同的漏洞但未被发现。单独找到并修复每个重复出现的漏洞实例是劳动密集型的。更令人担忧的是，漏洞报告可能会无意中扩大攻击面，因为它们为攻击者提供了可以利用的模式，而这些模式在程序的其他部分可能仍未解决。

在本文中，我们探讨了这些重复模式漏洞（RPBs），它们在程序的不同代码段甚至不同程序中反复出现，源于同一个根源但未被解决。我们的研究发现，RPBs 普遍存在且可能严重危害软件程序的安全性。本文介绍了一种名为 BugStone 的程序分析系统，该系统借助 LLVM 和大型语言模型（LLM）实现。关键观察结果是，许多 RPBS 至少有一个已修补的实例，可以利用这一事实来识别一致的错误模式，例如特定的 API 滥用。通过在整个程序中查找这种模式，可以识别出可能易受攻击的相似代码片段。从135个独特的 RPBS 开始，BugStone 在 Linux 内核中发现了超过22,000个新的潜在问题。手动分析400个这些发现中的结果确认其中有246个是有效的。我们还从23篇近期顶级会议工作中报告的安全漏洞中创建了一个包含超过1,900个安全漏洞的数据集。人工注释该数据集并识别出80个重复模式及其相应的850个修复方案。即使使用了成本效益较高的模型选择方案，BugStone 在数据集上的精确度达到了92.2%，成对准确性达到了79.1%。 

---
# Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games 

**Title (ZH)**: 全局思考，局部分组：多语言单词分组游戏评估LLMs 

**Authors**: César Guerra-Solano, Zhuochun Li, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14030)  

**Abstract**: Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models. 

**Abstract (ZH)**: 大型语言模型在抽象推理任务中可能会因语言模态表现出偏见，在不同语言的任务上表现不同，即使内容相似。现有的大多数研究表明了这一点，通常通过一些依赖于策略或知识的任务来评估，比如常识或数学任务。然而，抽象推理对于日常生活中的推理至关重要，人们在这种情况下会运用“创新思维”来识别和使用模式来解决问题，而不是依赖于固定的方法。在这方面，现有研究较少。在本文中，我们提出了一项受《纽约时报》全球连线游戏启发的任务，该任务跨越多种语言评估模型在抽象推理任务中的表现。我们构建了一个涵盖五种语言背景的游戏基准——英语、西班牙语、汉语、 Hindi（印地语）和阿拉伯语，并提供了英语翻译版本以供比较。我们还提出了游戏难度测量方法，以评估难度相似的游戏中的模型性能，从而使比较更加可控，这对推理评估尤为重要。通过实验，我们发现英语模态在这一抽象推理任务中的表现最好，并且开源和闭源模型之间的性能差异显著。 

---
# Context-Selective State Space Models: Feedback is All You Need 

**Title (ZH)**: 面向上下文的选择性状态空间模型：反馈即所必需 

**Authors**: Riccardo Zattra, Giacomo Baggio, Umberto Casti, Augusto Ferrante, Francesco Ticozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14027)  

**Abstract**: Transformers, powered by the attention mechanism, are the backbone of most foundation models, yet they suffer from quadratic complexity and difficulties in dealing with long-range dependencies in the input sequence. Recent work has shown that state space models (SSMs) provide an efficient alternative, with the S6 module at the core of the Mamba architecture achieving state-of-the-art results on long-sequence benchmarks. In this paper, we introduce the COFFEE (COntext From FEEdback) model, a novel time-varying SSM that incorporates state feedback to enable context-dependent selectivity, while still allowing for parallel implementation. Whereas the selectivity mechanism of S6 only depends on the current input, COFFEE computes it from the internal state, which serves as a compact representation of the sequence history. This shift allows the model to regulate its dynamics based on accumulated context, improving its ability to capture long-range dependencies. In addition to state feedback, we employ an efficient model parametrization that removes redundancies present in S6 and leads to a more compact and trainable formulation. On the induction head task, COFFEE achieves near-perfect accuracy with two orders of magnitude fewer parameters and training sequences compared to S6. On MNIST, COFFEE largely outperforms S6 within the same architecture, reaching 97% accuracy with only 3585 parameters. These results showcase the role of state feedback as a key mechanism for building scalable and efficient sequence models. 

**Abstract (ZH)**: 基于反馈的COFFEE模型：一种新颖的时间varying状态空间模型 

---
# Conditional Clifford-Steerable CNNs with Complete Kernel Basis for PDE Modeling 

**Title (ZH)**: 基于完备核基的条件克利福德-可 steering CNNs 用于偏微分方程建模 

**Authors**: Bálint László Szarvas, Maksim Zhdanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.14007)  

**Abstract**: Clifford-Steerable CNNs (CSCNNs) provide a unified framework that allows incorporating equivariance to arbitrary pseudo-Euclidean groups, including isometries of Euclidean space and Minkowski spacetime. In this work, we demonstrate that the kernel basis of CSCNNs is not complete, thus limiting the model expressivity. To address this issue, we propose Conditional Clifford-Steerable Kernels, which augment the kernels with equivariant representations computed from the input feature field. We derive the equivariance constraint for these input-dependent kernels and show how it can be solved efficiently via implicit parameterization. We empirically demonstrate an improved expressivity of the resulting framework on multiple PDE forecasting tasks, including fluid dynamics and relativistic electrodynamics, where our method consistently outperforms baseline methods. 

**Abstract (ZH)**: Clifford-可引导CNNs (CSCNNs) 提供了一个统一框架，使得可以包含任意伪欧几里得群的协变性，包括欧几里得空间的等距变换和闵可夫斯基时空的等距变换。在本文中，我们证明了CSCNNs的核基不完整，从而限制了模型的表征能力。为解决这一问题，我们提出了条件Clifford-可引导核，通过从输入特征场计算协变表示来增强核。我们推导了这些输入依赖核的协变约束，并展示了如何通过隐式参数化高效求解。我们通过多个偏微分方程预测任务的实验表明，该方法在流体动力学和相对论电磁动力学领域的一致上优于基线方法，改善了模型的表征能力。 

---
# REAP the Experts: Why Pruning Prevails for One-Shot MoE compression 

**Title (ZH)**: REAP名家: 为何剪枝在单一-shot MoE压缩中占据优势 

**Authors**: Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2510.13999)  

**Abstract**: Sparsely-activated Mixture-of-Experts (SMoE) models offer efficient pre-training and low latency but their large parameter counts create significant memory overhead, motivating research into expert compression. Contrary to recent findings favouring expert merging on discriminative benchmarks, we demonstrate that expert pruning is a superior strategy for generative tasks. We prove that merging introduces an irreducible error by causing a "functional subspace collapse", due to the loss of the router's independent, input-dependent control over experts. Leveraging this insight, we propose Router-weighted Expert Activation Pruning (REAP), a novel pruning criterion that considers both router gate-values and expert activation norms. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation and tool-calling tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts. 

**Abstract (ZH)**: 稀疏激活专家混合模型（SMoE）提供了高效的预训练和低延迟，但其庞大的参数量带来了显著的内存负担，激励了专家压缩的研究。与近期研究倾向于在辨别性基准上进行专家合并结论相反，我们证明了对于生成任务，专家剪枝是一种更优策略。我们证明合并引入了不可约错误，这是由于路由器失去了独立的、输入相关的对专家的控制，导致“功能子空间坍塌”。基于这一洞察，我们提出了路由加权专家激活剪枝（REAP），这是一种新的剪枝标准，考虑了路由门值和专家激活范数。在从200亿到1万亿参数的多种SMoE模型中，REAP在生成任务基准上始终优于合并和其他剪枝方法，尤其是在50%压缩情况下。值得注意的是，即使剪枝了50%的专家，我们的方法仍能在代码生成和工具调用任务中实现接近无损的压缩，如Qwen3-Coder-480B和Kimi-K2。 

---
# Finding Holes: Pathologist Level Performance Using AI for Cribriform Morphology Detection in Prostate Cancer 

**Title (ZH)**: 寻找漏洞：使用AI检测前列腺癌 cribriform形态的人类病理学家水平性能 

**Authors**: Kelvin Szolnoky, Anders Blilie, Nita Mulliqi, Toyonori Tsuzuki, Hemamali Samaratunga, Matteo Titus, Xiaoyi Ji, Sol Erika Boman, Einar Gudlaugsson, Svein Reidar Kjosavik, José Asenjo, Marcello Gambacorta, Paolo Libretti, Marcin Braun, Radisław Kordek, Roman Łowicki, Brett Delahunt, Kenneth A. Iczkowski, Theo van der Kwast, Geert J. L. H. van Leenders, Katia R. M. Leite, Chin-Chen Pan, Emiel Adrianus Maria Janssen, Martin Eklund, Lars Egevad, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13995)  

**Abstract**: Background: Cribriform morphology in prostate cancer is a histological feature that indicates poor prognosis and contraindicates active surveillance. However, it remains underreported and subject to significant interobserver variability amongst pathologists. We aimed to develop and validate an AI-based system to improve cribriform pattern detection.
Methods: We created a deep learning model using an EfficientNetV2-S encoder with multiple instance learning for end-to-end whole-slide classification. The model was trained on 640 digitised prostate core needle biopsies from 430 patients, collected across three cohorts. It was validated internally (261 slides from 171 patients) and externally (266 slides, 104 patients from three independent cohorts). Internal validation cohorts included laboratories or scanners from the development set, while external cohorts used completely independent instruments and laboratories. Annotations were provided by three expert uropathologists with known high concordance. Additionally, we conducted an inter-rater analysis and compared the model's performance against nine expert uropathologists on 88 slides from the internal validation cohort.
Results: The model showed strong internal validation performance (AUC: 0.97, 95% CI: 0.95-0.99; Cohen's kappa: 0.81, 95% CI: 0.72-0.89) and robust external validation (AUC: 0.90, 95% CI: 0.86-0.93; Cohen's kappa: 0.55, 95% CI: 0.45-0.64). In our inter-rater analysis, the model achieved the highest average agreement (Cohen's kappa: 0.66, 95% CI: 0.57-0.74), outperforming all nine pathologists whose Cohen's kappas ranged from 0.35 to 0.62.
Conclusion: Our AI model demonstrates pathologist-level performance for cribriform morphology detection in prostate cancer. This approach could enhance diagnostic reliability, standardise reporting, and improve treatment decisions for prostate cancer patients. 

**Abstract (ZH)**: 背景：cribriform形态在前列腺癌中是一种预后不良的组织学特征，且提示不宜进行主动监测。然而，该特征在病理学家之间报道不足且存在显著的主观差异。我们旨在开发并验证一种基于AI的系统以提高cribriform模式检测的准确性。 

---
# Efficient Few-Shot Learning in Remote Sensing: Fusing Vision and Vision-Language Models 

**Title (ZH)**: 遥感中的高效少-shot学习：视觉与视觉语言模型融合 

**Authors**: Jia Yun Chua, Argyrios Zolotas, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2510.13993)  

**Abstract**: Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few-shot learning scenarios. 

**Abstract (ZH)**: 遥感已成为城市规划、环境监测和灾害响应等领域的重要工具。虽然生成的数据量显著增加，但传统视觉模型往往受限于需要大量专用领域标注数据，并且在理解复杂环境中的上下文方面能力有限。视觉语言模型通过整合视觉和文本数据提供了互补的方法；然而，它们在遥感领域的应用尚未得到充分探索，特别是考虑到它们的通用性。本研究调查了将视觉模型与VLMs结合以增强遥感图像分析的方法，重点在于飞机检测和场景理解。通过将YOLO与LLaVA、ChatGPT、Gemini等VLMs集成，旨在实现更准确和上下文感知的图像解释。性能在标记和未标记的遥感数据以及降质图像场景中进行了评估，后者对于遥感至关重要。研究结果表明，在飞机检测和计数的准确性上，模型平均提高了48.46%，尤其是在挑战性条件下，在原始和降质场景中尤为明显。遥感图像综合理解的CLIPScore提高了6.17%。结合传统视觉模型和VLMs的方法为更高级和高效的遥感图像分析铺平了道路，特别是在少量样本学习场景中。 

---
# Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations 

**Title (ZH)**: 静态沙箱不足：基于LLM的多agent模拟需要开放-ended共进化以 modeling 社会复杂性 

**Authors**: Jinkun Chen, Sher Badshah, Xuemin Yu, Sijia Han, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.13982)  

**Abstract**: What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. \textbf{We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.} 

**Abstract (ZH)**: 如果人工代理不仅能沟通，还能进化、适应并重新塑造我们无法完全预测的世界？随着大模型现已成为多代理系统和社会模拟的核心驱动力，我们正目睹着对开放延展、不断变化环境建模的新可能性。然而，当前大多数模拟仍然局限于静态的沙盒环境，具有预定义的任务、有限的动态性和严格的评价标准。这些限制使其难以捕捉到真实世界社会的复杂性。在本文中，我们论证认为静态、任务特定的基准是根本不足的，必须重新思考。我们批判性地回顾了将大模型与多代理动态结合的新架构，指出了关键障碍，如平衡稳定性和多样性、评估意外行为以及扩大复杂性，并引入了一个新的分类体系以适应这一迅速发展的领域。最后，我们提出了一条以开放延展性、持续共进化和开发韧性、社会导向的AI生态系统为中心的研究路线图。我们呼吁社区超越静态范式，帮助塑造新一代适应性强、社会意识强的多代理模拟。 

---
# Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention 

**Title (ZH)**: 少即是多：通过最小化测试时干预提升大语言模型推理能力 

**Authors**: Zhen Yang, Mingyang Zhang, Feng Chen, Ganggui Ding, Liang Hou, Xin Tao, Pengfei Wan, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13940)  

**Abstract**: Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient. 

**Abstract (ZH)**: 最近大型语言模型的进步集中在测试时扩展以通过增加推断计算来提高推理能力，但往往会牺牲效率。我们重新审视测试时的行为并揭开了一个简单且尚未深入探索的现象：推理不确定性是局部化的——只有少量高熵标记在主要影响输出正确性方面占主导地位。受此启发，我们提出了无需训练的最小测试时干预（MTI）框架，该框架以极小的开销提高推理准确性和稳定性。MTI 包括：（i）选择性的自回归模型干预，仅在不确定性位置应用无分类器引导；以及（ii）轻量级的负提示引导，利用主要模型的 KV 缓存来高效近似无条件解码。MTI 在通用任务、编程任务和 STEM 任务中均展现出一致性改进——例如，Qwen3-8B-Base 在八个基准上的平均改进为 +1.35%，而 Qwen3-32B-Reasoning 在 AIME2024 上改进为 +5%，同时保持高度高效。 

---
# Big Reasoning with Small Models: Instruction Retrieval at Inference Time 

**Title (ZH)**: 小模型进行大推理：推理时的指令检索 

**Authors**: Kenan Alkiek, David Jurgens, Vinod Vydiswaran  

**Link**: [PDF](https://arxiv.org/pdf/2510.13935)  

**Abstract**: Can we bring large-scale reasoning to local-scale compute? Small language models (SLMs) are increasingly attractive because they run efficiently on local hardware, offering strong privacy, low cost, and reduced environmental impact. Yet they often struggle with tasks that require multi-step reasoning or domain-specific knowledge. We address this limitation through instruction intervention at inference time, where an SLM retrieves structured reasoning procedures rather than generating them from scratch. Our method builds an Instruction Corpus by grouping similar training questions and creating instructions via GPT-5. During inference, the SLM retrieves the most relevant instructions and follows their steps. Unlike retrieval-augmented generation, which retrieves text passages, instruction retrieval gives the model structured guidance for reasoning. We evaluate this framework on MedQA (medical board exams), MMLU Professional Law, and MathQA using models from 3B to 14B parameters without any additional fine-tuning. Instruction retrieval yields consistent gains: 9.4% on MedQA, 7.9% on MMLU Law, and 5.1% on MathQA. Concise instructions outperform longer ones, and the magnitude of improvement depends strongly on model family and intrinsic reasoning ability. 

**Abstract (ZH)**: 我们能在本地计算规模实现大规模推理吗？小语言模型（SLMs）因其在本地硬件上高效运行、提供强大的隐私保护、低成本和降低环境影响而日益受到关注。然而，它们往往在需要多步推理或领域特定知识的任务上表现不佳。我们通过推断时的指令干预来克服这一局限，使SLM检索结构化推理步骤而非从头生成。我们的方法通过将类似训练问题分组并通过GPT-5创建指令构建指令库。在推断过程中，SLM检索最相关指令并遵循其步骤。与检索增强生成不同，后者检索文本段落，指令检索为模型提供了结构化的推理指导。我们在未经任何额外微调的情况下，使用参数量从3B到14B的模型在MedQA（医学执业考试）、MMLU专业法务和MathQA上评估了该框架。指令检索带来了一致的提升：在MedQA上为9.4%，MMLU Law上为7.9%，MathQA上为5.1%。简洁的指令优于较长的指令，改进程度强烈依赖于模型家族和内在推理能力。 

---
# LLMs Can Get "Brain Rot"! 

**Title (ZH)**: LLMs可能会出现“智力衰退”！ 

**Authors**: Shuo Xing, Junyuan Hong, Yifan Wang, Runjin Chen, Zhenyu Zhang, Ananth Grama, Zhengzhong Tu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13928)  

**Abstract**: We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0\%$ to $100\%$.
Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that data quality is a causal driver of LLM capability decay, reframing curation for continual pretraining as a \textit{training-time safety} problem and motivating routine "cognitive health checks" for deployed LLMs. 

**Abstract (ZH)**: 我们提出并测试了LLM大脑衰退假说：持续接触低质量网络文本会导致大规模语言模型（LLMs）出现持久的认知衰退。通过控制实验，我们在真实的Twitter/X语料库上运行实验，构建垃圾数据和逆向控制数据集，采用两个正交的操作化方法：M1（参与度程度）和M2（语义质量），并确保各条件下的tokens规模和训练操作匹配。与对照组相反，持续在垃圾数据集上预训练4个LLM导致在推理解释力、长上下文理解、安全性以及“黑暗特质”（例如，反社会人格、自恋）的膨胀方面出现了实质性下降（Hedges' $g>0.3$）。垃圾数据和控制数据的逐步混合也显示出剂量反应的认知衰退：例如，在M1条件下，带有思考过程的ARC-Challenge分数从74.9下降到57.2，RULER-CWE分数从84.4下降到52.3，随着垃圾数据比例从0%增加到100%。

错误分析揭示了几条关键见解。首先，我们确定了思维跳跃为主要缺陷：模型逐渐裁剪或跳过推理链，解释了大部分错误增长。其次，部分但不完全的恢复被观察到：扩展指令调优和清洁数据预训练可以改善认知衰退，但不能恢复基线能力，表明固有的表示漂移而非格式不匹配。最后，我们发现，推文的流行度（一种非语义指标）在M1条件下比长度更能预测大脑衰退效应。综上所述，研究结果提供了重要的多视角证据，表明数据质量是导致LLM能力衰退的因果驱动因素，重新定义持续预训练的数据整理为“训练时的安全”问题，并促使对部署的LLM进行常规的认知健康检查。 

---
# Readability $\ne$ Learnability: Rethinking the Role of Simplicity in Training Small Language Models 

**Title (ZH)**: 可读性 $\ne$ 可学习性：重新审视小语言模型训练中简单性的角色 

**Authors**: Ivan Lee, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2510.13915)  

**Abstract**: Recent studies suggest that very small language models (SLMs) can generate surprisingly coherent text when trained on simplified, child-directed corpora such as TinyStories. These findings have been interpreted as evidence that readability -- characterized by accessible vocabulary, familiar narrative structure, and simple syntax -- plays a key role in enabling such capabilities to emerge. In this paper, we challenge that interpretation. We construct synthetic datasets with matched structure but varied readability, and find that readability alone does not predict coherence or learning efficiency in SLMs. Models trained on complex, adult-level text perform comparably to those trained on simplified language, and even exhibit faster development of coherence during training. Instead, we show that statistical simplicity, as measured by n-gram diversity, is a stronger predictor of learnability. Our findings caution against the growing trend of anthropomorphizing language model training -- drawing parallels to human cognitive development without empirical basis -- and argue for more precise reasoning about what properties actually support capability emergence in small models. 

**Abstract (ZH)**: 最近的研究表明，当使用简化版的儿童导向语料（如TinyStories）训练时，非常小的语言模型（SLMs）能够生成出乎意料一致性的文本。这些发现被解释为易读性的证据，易读性包括易于理解的词汇、熟悉的叙事结构和简单的语法，在使这些能力出现中起着关键作用。在本文中，我们挑战这一观点。我们构建了具有匹配结构但不同易读性的合成数据集，并发现仅易读性并不预测SLMs的一致性或学习效率。在复杂成人水平文本上训练的模型与在简化语言上训练的模型表现相当，甚至在训练过程中一致性的开发速度更快。相反，我们证明了统计简单性（用n-克gram多样性衡量）是学习能力更强的预测因子。我们的研究警告人们避免将语言模型训练过程拟人化——将语言模型的训练与人类认知发展进行类比而没有实验证据，并且主张对支持小型模型能力出现的实际属性进行更精确的推理。 

---
# Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms 

**Title (ZH)**: 使用渐进难度增强机制合成代理数据以供网络代理使用 

**Authors**: Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.13913)  

**Abstract**: Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors. 

**Abstract (ZH)**: 基于Web的“深度研究”代理旨在通过与在线工具进行长期交互来解决复杂的问答任务。尽管这些任务具有挑战性，因为底层语言模型往往未优化用于长期推理和探索。先前的工作提出了一些构建指令调优数据集的方法，通常利用知识图谱。然而，这些方法通常缺乏对难度和质量的精细控制，生成的数据未能捕捉到长期推理所需的复杂性。此外，许多研究通过比较在不同优化配方下训练的模型来混淆数据和训练效果，使得难以隔离和评估数据本身的效用。我们提出了一种双管齐下的数据合成管道，通过逐步增加任务复杂性，直至基准Web代理失败来生成问题-答案对。基准代理在这个过程中扮演多重角色：尝试回答问题、验证事实性、检查替代答案并执行过滤。为了评估我们合成方法的有效性，我们采用了基于强Web代理蒸馏的受控训练设置。在多个基于Web的基准测试中进行的实验显示，尽管我们的数据集规模较小，但仍能使训练出的Web代理比现有数据集更有效。特别地，我们的数据展示了使用工具行为两倍的多样性，使在此类数据上训练的模型能够实现更优性能，同时避免重复调用工具的行为。 

---
# AI Debaters are More Persuasive when Arguing in Alignment with Their Own Beliefs 

**Title (ZH)**: AI辩论者在其信念一致时更具说服力。 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Facundo Nieto, Oscar Agustín Stanchi, Guido Ernesto Bergman, Mario Alejandro Leiva, Eitan Sprejer, Luca Nicolás Forziati Gangi, Francisca Gauna Selasco, Juan Gustavo Corvalán, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13912)  

**Abstract**: The core premise of AI debate as a scalable oversight technique is that it is harder to lie convincingly than to refute a lie, enabling the judge to identify the correct position. Yet, existing debate experiments have relied on datasets with ground truth, where lying is reduced to defending an incorrect proposition. This overlooks a subjective dimension: lying also requires the belief that the claim defended is false. In this work, we apply debate to subjective questions and explicitly measure large language models' prior beliefs before experiments. Debaters were asked to select their preferred position, then presented with a judge persona deliberately designed to conflict with their identified priors. This setup tested whether models would adopt sycophantic strategies, aligning with the judge's presumed perspective to maximize persuasiveness, or remain faithful to their prior beliefs. We implemented and compared two debate protocols, sequential and simultaneous, to evaluate potential systematic biases. Finally, we assessed whether models were more persuasive and produced higher-quality arguments when defending positions consistent with their prior beliefs versus when arguing against them. Our main findings show that models tend to prefer defending stances aligned with the judge persona rather than their prior beliefs, sequential debate introduces significant bias favoring the second debater, models are more persuasive when defending positions aligned with their prior beliefs, and paradoxically, arguments misaligned with prior beliefs are rated as higher quality in pairwise comparison. These results can inform human judges to provide higher-quality training signals and contribute to more aligned AI systems, while revealing important aspects of human-AI interaction regarding persuasion dynamics in language models. 

**Abstract (ZH)**: 基于AI辩论作为可扩展监督技术的核心假设是，难以诚实地撒谎比反驳谎言更困难，使法官能够识别正确的立场。然而，现有的辩论实验依赖于具有ground truth数据集，将撒谎简化为捍卫一个不正确的命题。这忽略了主观维度：撒谎也需要相信所捍卫的主张是错误的。在这项工作中，我们将辩论应用于主观问题，并在实验前明确测量大型语言模型的先验信念。辩论者被要求选择他们偏爱的立场，然后面临一个故意与其识别的先验相矛盾的法官人设。这一设置测试模型是否会采用谄媚策略，与法官 presumed 角度对齐以最大化说服力，还是会保持其先验信念。我们实施并比较了两种辩论协议，顺序和同时进行，以评估潜在的系统性偏差。最后，我们评估了当模型捍卫与其先验信念一致的立场时，与争论其相反立场时，其说服力和生成高质量论点的差异。我们的主要发现表明，模型倾向于选择与其法官人设一致的立场而非其先验信念，顺序辩论显著偏向第二位辩论者，当捍卫与其先验信念一致的立场时，模型更具说服力，而令人 paradoxically 地，与先验信念不一致的论点在两两比较中被评为更有质量。这些结果可以为人机法官提供更高质量的训练信号，并有助于构建更具一致性的AI系统，同时揭示了人类-AI交互中的重要方面，特别是在语言模型的说服动态方面。 

---
# Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning 

**Title (ZH)**: 知识推理语言模型：统一知识与语言以促进归纳知识图谱推理 

**Authors**: Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13909)  

**Abstract**: Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM\footnote{Our source codes are available at this https URL in both zero-shot reasoning and fine-tuning scenarios. 

**Abstract (ZH)**: 知识图谱推理语言模型（KRLM）：实现LLM知识和KG上下文在整个知识图谱推理过程中的统一协调 

---
# Schema for In-Context Learning 

**Title (ZH)**: 基于上下文学习的方案 

**Authors**: Pan Chen, Shaohong Chen, Mark Wang, Shi Xuan Leong, Priscilla Fung, Varinia Bernales, Alan Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2510.13905)  

**Abstract**: In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce SCHEMA ACTIVATED IN CONTEXT LEARNING (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. SCHEMA ACTIVATED IN CONTEXT LEARNING not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs. 

**Abstract (ZH)**: SCHEMA ACTIVATED IN CONTEXT LEARNING 

---
# Benefits and Limitations of Communication in Multi-Agent Reasoning 

**Title (ZH)**: 多智能体推理中通信的利与弊 

**Authors**: Michael Rizvi-Martel, Satwik Bhattamishra, Neil Rathi, Guillaume Rabusseau, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.13903)  

**Abstract**: Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems. 

**Abstract (ZH)**: 链式思维提示在大型语言模型中普及了逐步推理，但随着问题复杂性和上下文长度的增长，模型性能仍然会下降。通过将具有长期上下文的困难任务分解为更短、更易管理的任务，最近的多智能体范式为解决这一问题提供了有希望的短期解决方案。然而，这类系统的根本能力还知之甚少。在本文中，我们提出了一种理论框架来分析多智能体系统的表达能力。我们将该框架应用于三种算法家族：状态跟踪、回忆和$k$-跳推理。我们推导出了关于(i)完成任务所需的智能体数量、(ii) 交互智能体间通信的数量和结构，以及(iii) 随问题规模和上下文扩展可实现的加速比的上界。我们的结果明确了通信在可证明有益的区间、智能体数量与带宽之间的权衡，并在任一资源受限时揭示内在限制。我们通过使用受控合成基准对预训练的大语言模型进行实验，补充了我们的理论分析。实证结果证实了我们理论预测的关键量之间的权衡。综上所述，我们的分析为设计可扩展的多智能体推理系统提供了原则性指导。 

---
# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences 

**Title (ZH)**: 窄微调会在激活差异中留下清晰可辨的痕迹 

**Authors**: Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.13900)  

**Abstract**: Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research. 

**Abstract (ZH)**: 细粒度微调已成为一种 Essential Tool，用于适应大型语言模型 (LLMs) 以执行特定任务，并创建具有已知异常属性的模型，这些属性对研究有用。我们展示了细粒度微调在LLM激活中创建了强大的偏见，这些偏见可以被解读以理解微调领域。这些偏见可以通过模型差异分析中的简单工具——即比较模型微调前后差异——来发现。特别是，通过分析随机文本前几个词的激活差异，并将此差异添加到模型激活中进行引导，可以生成类似于微调数据格式和内容的文本。我们通过创建一种基于LLM的可解释性代理来理解微调领域，证明了这些分析包含了关键信息。该代理通过访问偏见相较于使用简单提示的基线代理表现得更好。我们的分析涵盖了不同架构（Gemma、LLaMA、Qwen）和规模（1B至32B参数）的虚假事实合成文档微调、新兴不对齐、潜意识学习和禁忌词猜谜模型。我们怀疑这些偏见反映了过度拟合，并发现将预训练数据混入微调语料库可以大大消除这些偏见，尽管可能存在剩余风险。我们的工作：(1) 证明了细粒度微调模型在其激活中有明显的训练目标痕迹，建议改进其训练方式；(2) 警告AI安全和可解释性研究人员，使用这些模型作为研究广泛微调（例如聊天调优）的代理可能不现实；(3) 强调了深入研究细粒度微调影响的必要性，并开发真正具代表性的模型差异分析、安全和可解释性研究案例的必要性。 

---
# Dual-attention ResNet outperforms transformers in HER2 prediction on DCE-MRI 

**Title (ZH)**: Dual-attention ResNet在DCE-MRI的HER2预测中优于 Transformers 

**Authors**: Naomi Fridman, Anat Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.13897)  

**Abstract**: Breast cancer is the most diagnosed cancer in women, with HER2 status critically guiding treatment decisions. Noninvasive prediction of HER2 status from dynamic contrast-enhanced MRI (DCE-MRI) could streamline diagnostics and reduce reliance on biopsy. However, preprocessing high-dynamic-range DCE-MRI into standardized 8-bit RGB format for pretrained neural networks is nontrivial, and normalization strategy significantly affects model performance. We benchmarked intensity normalization strategies using a Triple-Head Dual-Attention ResNet that processes RGB-fused temporal sequences from three DCE phases. Trained on a multicenter cohort (n=1,149) from the I-SPY trials and externally validated on BreastDCEDL_AMBL (n=43 lesions), our model outperformed transformer-based architectures, achieving 0.75 accuracy and 0.74 AUC on I-SPY test data. N4 bias field correction slightly degraded performance. Without fine-tuning, external validation yielded 0.66 AUC, demonstrating cross-institutional generalizability. These findings highlight the effectiveness of dual-attention mechanisms in capturing transferable spatiotemporal features for HER2 stratification, advancing reproducible deep learning biomarkers in breast cancer imaging. 

**Abstract (ZH)**: HER2状态从动态对比增强MRI的无创预测：三头双注意力ResNet在乳腺癌影像中的应用 

---
# GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation via Large Language Model Agents 

**Title (ZH)**: GenCellAgent: 基于大型语言模型代理的通用无训练细胞图像分割 

**Authors**: Xi Yu, Yang Yang, Qun Liu, Yonghua Du, Sean McSweeney, Yuewei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.13896)  

**Abstract**: Cellular image segmentation is essential for quantitative biology yet remains difficult due to heterogeneous modalities, morphological variability, and limited annotations. We present GenCellAgent, a training-free multi-agent framework that orchestrates specialist segmenters and generalist vision-language models via a planner-executor-evaluator loop (choose tool $\rightarrow$ run $\rightarrow$ quality-check) with long-term memory. The system (i) automatically routes images to the best tool, (ii) adapts on the fly using a few reference images when imaging conditions differ from what a tool expects, (iii) supports text-guided segmentation of organelles not covered by existing models, and (iv) commits expert edits to memory, enabling self-evolution and personalized workflows. Across four cell-segmentation benchmarks, this routing yields a 15.7\% mean accuracy gain over state-of-the-art baselines. On endoplasmic reticulum and mitochondria from new datasets, GenCellAgent improves average IoU by 37.6\% over specialist models. It also segments novel objects such as the Golgi apparatus via iterative text-guided refinement, with light human correction further boosting performance. Together, these capabilities provide a practical path to robust, adaptable cellular image segmentation without retraining, while reducing annotation burden and matching user preferences. 

**Abstract (ZH)**: 细胞图像分割对于定量生物学至关重要，但由于异质模态、形态变异性和有限的注释，这一任务依然颇具挑战。我们提出了一种无需训练的多代理框架GenCellAgent，该框架通过规划者-执行者-评估者循环（选择工具 → 运行 → 质量检查）并利用长期记忆，协调专业分割器和通用的视觉语言模型。该系统（i）自动将图像路由到最佳工具，（ii）在成像条件与工具预期不符时，能够即刻调整，（iii）支持文本引导的现有模型未涵盖的细胞器分割，（iv）将专家编辑记忆化，从而实现自我进化和个人化的工作流程。在四个细胞分割基准测试中，这种路由方法相较于最先进的基线提高了15.7%的平均准确率。在内质网和线粒体等新数据集上，GenCellAgent相较于专门模型平均提升了37.6%的IoU。此外，通过迭代的文本引导细化，该系统还能够分割新的细胞结构如高尔基体，适度的人工校正进一步提升了性能。这些能力为在无需重新训练的情况下实现鲁棒且适应性强的细胞图像分割提供了一条实际路径，同时减少了注释负担并匹配用户偏好。 

---
# Bayes or Heisenberg: Who(se) Rules? 

**Title (ZH)**: 贝叶斯还是海森堡：谁（的规则）？ 

**Authors**: Volker Tresp Hang Li, Federico Harjes, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13894)  

**Abstract**: Although quantum systems are generally described by quantum state vectors, we show that in certain cases their measurement processes can be reformulated as probabilistic equations expressed in terms of probabilistic state vectors. These probabilistic representations can, in turn, be approximated by the neural network dynamics of the Tensor Brain (TB) model.
The Tensor Brain is a recently proposed framework for modeling perception and memory in the brain, providing a biologically inspired mechanism for efficiently integrating generated symbolic representations into reasoning processes. 

**Abstract (ZH)**: 尽管量子系统一般由量子状态矢量描述，但我们展示了在某些情况下，其测量过程可以重新表述为用概率状态矢量表示的概率方程。这些概率表示可以近似为Tensor Brain (TB)模型中的神经网络动力学。Tensor Brain是一种 recently 提出的框架，用于模型大脑的感知和记忆，提供了一种受生物学启发的机制，以高效地将生成的符号表示集成到推理过程中。 

---
# Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection 

**Title (ZH)**: 护栏的守护：基于分类学的方法对 Jailbreak 的检测 

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13893)  

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards. 

**Abstract (ZH)**: 破解技术对大型语言模型（LLMs）的安全性构成重大威胁。现有的防御方法通常侧重于单轮攻击，缺乏跨语言覆盖，且依赖有限的分类体系，这些分类体系要么无法捕捉到所有攻击策略的多样性，要么侧重于风险类别而非破解技术。为了增进对破解技术有效性的理解，我们开展了一项结构化的红队挑战。实验结果表明，首先，我们开发了一种全面的分层分类体系，涵盖50种破解策略，并将其之前的分类扩展为七大类，包括冒充、说动、权限提升、认知过载、混淆、目标冲突和数据污染。其次，我们分析了挑战收集的数据，以研究不同攻击类型的发生频率和成功率，提供了关于特定破解策略如何利用模型漏洞并导致脱轨的见解。第三，我们对一个流行的LLM进行了破解检测基准测试，评估了基于分类体系引导的提示对于改进自动检测的好处。最后，我们编译了一个新的意大利语多轮 adversarial 对话数据集，包含1364个对话，并用我们的分类体系进行注释，使研究能够观察到敌对方意图如何逐步显现并成功绕过传统保护措施。 

---
# K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding 

**Title (ZH)**: K-框架：场景驱动的任意k关键帧选择用于长视频理解 

**Authors**: Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13891)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in image understanding, but long-video are constrained by context windows and computational cost. Uniform frame sampling often leads to substantial information loss. Meanwhile existing keyframe selection methods such as text-frame retrieval or RL-based frame optimization typically yield sparse and temporally disjointed frames, overlooking scene continuity and lacking flexibility for multi-scale frame selection. To address these limitations, we introduce K-frames, a novel paradigm for scene-driven keyframe selection that preserves temporal continuity. Instead of selecting individual frames, K-frames predicts semantically coherent, query-relevant clips, which enables any-k keyframes selection to meet diverse user budgets. To achieve this approach, we first introduce PeakClips, a dataset of 200K video highlights conditioned by query. Building on this dataset, K-frames learns clip2frame selection using a three-stage progressive curriculum. It involves two Supervised Fine-Tuning stages for temporal grounding and key-clip perception, followed by a Reinforcement Learning stage that directly optimizes the scene-driven prediction policy for downstream task without further annotations. Extensive experiments on major long-video understanding benchmarks demonstrate that K-frames provides an effective, interpretable, and plug-and-play solution for keyframe selection at various scales. Our dataset and model will be available. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在图像理解方面展示了显著的能力，但长视频受限于上下文窗口和计算成本。均匀帧抽样往往导致信息丢失严重。同时，现有的关键帧选择方法如基于文本的帧检索或基于RL的帧优化通常会选取稀疏且时空断开的帧，忽视了场景连续性并缺乏多尺度帧选择的灵活性。为了解决这些局限，我们提出了K-帧，这是一种基于场景的关键帧选择新范式，能够保持时间连续性。K-帧通过预测语义一致、查询相关的片段来代替选择单个帧，使得任何数量的关键帧都能满足不同用户的需求。为了实现这一方法，我们首先引入了PeakClips数据集，该数据集包含20万条查询条件下的视频高光片段。在此数据集的基础上，K-帧通过三阶段逐步课程学习片段到帧的选择。第一阶段是对齐时间和关键片段感知的两阶段监督微调，第二阶段是直接优化基于场景的预测策略的强化学习阶段，无需进一步标注。在主要的长视频理解基准上的广泛实验表明，K-帧提供了一种有效、可解释且即插即用的关键帧选择解决方案。我们的数据集和模型将可供使用。 

---
# A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness 

**Title (ZH)**: 小规模和大规模语言模型协作性能、成本效益、云端边缘隐私及可信性综述 

**Authors**: Fali Wang, Jihai Chen, Shuhua Yang, Ali Al-Lawati, Linli Tang, Hui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13890)  

**Abstract**: Large language models (LLMs) have advanced many domains and applications but face high fine-tuning costs, inference latency, limited edge deployability, and reliability concerns. Small language models (SLMs), compact, efficient, and adaptable, offer complementary remedies. Recent work explores collaborative frameworks that fuse SLMs' specialization and efficiency with LLMs' generalization and reasoning to meet diverse objectives across tasks and deployment scenarios. Motivated by these developments, this paper presents a systematic survey of SLM-LLM collaboration organized by collaboration objectives. We propose a taxonomy with four goals: performance enhancement, cost-effectiveness, cloud-edge privacy, and trustworthiness. Within this framework, we review representative methods, summarize design paradigms, and outline open challenges and future directions toward efficient, secure, and scalable SLM-LLM collaboration. 

**Abstract (ZH)**: 大型语言模型(LLMs)在许多领域和应用中取得了进展，但面临高昂的微调成本、推断延迟、边缘部署限制以及可靠性的担忧。小型语言模型(SLMs)，紧凑、高效且可适应，提供了互补的解决方案。近期的研究探索了将SLMs的专业化和高效性与LLMs的泛化和推理能力相结合的协作框架，以满足跨任务和部署场景的各种目标。受这些进展的激发，本文按照协作目标系统地回顾了SLMs-LLMs的协作研究。我们提出了一个包含四个目标的分类体系：性能提升、成本效益、云边隐私以及可信度。在这一框架内，我们回顾了代表性方法、总结了设计范式，并概述了朝着高效、安全和可扩展的SLMs-LLMs协作所面临的开放挑战和未来方向。 

---
# Reliable Fine-Grained Evaluation of Natural Language Math Proofs 

**Title (ZH)**: 自然语言数学证明的可靠细粒度评估 

**Authors**: Wenjie Ma, Andrei Cojocaru, Neel Kolhe, Bradley Louie, Robin Said Sharif, Haihan Zhang, Vincent Zhuang, Matei Zaharia, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2510.13888)  

**Abstract**: Recent advances in large language models (LLMs) for mathematical reasoning have largely focused on tasks with easily verifiable final answers; however, generating and verifying natural language math proofs remains an open challenge. We identify the absence of a reliable, fine-grained evaluator for LLM-generated math proofs as a critical gap. To address this, we propose a systematic methodology for developing and validating evaluators that assign fine-grained scores on a 0-7 scale to model-generated math proofs. To enable this study, we introduce ProofBench, the first expert-annotated dataset of fine-grained proof ratings, spanning 145 problems from six major math competitions (USAMO, IMO, Putnam, etc) and 435 LLM-generated solutions from Gemini-2.5-pro, o3, and DeepSeek-R1. %with expert gradings. Using ProofBench as a testbed, we systematically explore the evaluator design space across key axes: the backbone model, input context, instructions and evaluation workflow. Our analysis delivers ProofGrader, an evaluator that combines a strong reasoning backbone LM, rich context from reference solutions and marking schemes, and a simple ensembling method; it achieves a low Mean Absolute Error (MAE) of 0.926 against expert scores, significantly outperforming naive baselines. Finally, we demonstrate its practical utility in a best-of-$n$ selection task: at $n=16$, ProofGrader achieves an average score of 4.14 (out of 7), closing 78% of the gap between a naive binary evaluator (2.48) and the human oracle (4.62), highlighting its potential to advance downstream proof generation. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在数学推理中的最新进展主要集中在具有容易验证最终答案的任务上；然而，生成和验证自然语言数学证明仍然是一个开放的挑战。我们识别出缺乏可靠的细粒度评估器来评估LLM生成的数学证明是关键空白。为此，我们提出了一种系统的方法学，用于开发和验证能够在0-7分的细粒度评分尺度上评估模型生成数学证明的评估器。为了使这一研究成为可能，我们引入了ProofBench，这是首个专家标注的细粒度证明评分数据集，覆盖了六大主要数学竞赛（USAMO、IMO、Putnam等）的145道题目和来自Gemini-2.5-pro、o3和DeepSeek-R1的435个LLM生成的解题方案，并附有专家评分。使用ProofBench作为试验平台，我们系统地探索了评估器设计空间的关键维度：骨干模型、输入上下文、指令和评估流程。我们的分析产生了一个结合强大推理骨干LM、参考解题方案和评分方案中的丰富上下文以及简单集成方法的评估器—ProofGrader；它相对于专家评分的平均绝对误差（MAE）仅为0.926，显著优于简单的基线方法。最后，我们展示了其在最佳选优任务中的实际用途：在$n=16$时，ProofGrader的平均得分为4.14（满分为7分），填补了2.48（ naive二元评估器）和4.62（人类 oracle）之间78%的差距，突显了其促进下游证明生成的潜力。 

---
# Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion 

**Title (ZH)**: 基于分层语义对齐与协同完成的不完整多视图聚类 

**Authors**: Xiaojian Ding, Lin Zhao, Xian Li, Xiaoying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13887)  

**Abstract**: Incomplete multi-view data, where certain views are entirely missing for some samples, poses significant challenges for traditional multi-view clustering methods. Existing deep incomplete multi-view clustering approaches often rely on static fusion strategies or two-stage pipelines, leading to suboptimal fusion results and error propagation issues. To address these limitations, this paper proposes a novel incomplete multi-view clustering framework based on Hierarchical Semantic Alignment and Cooperative Completion (HSACC). HSACC achieves robust cross-view fusion through a dual-level semantic space design. In the low-level semantic space, consistency alignment is ensured by maximizing mutual information across views. In the high-level semantic space, adaptive view weights are dynamically assigned based on the distributional affinity between individual views and an initial fused representation, followed by weighted fusion to generate a unified global representation. Additionally, HSACC implicitly recovers missing views by projecting aligned latent representations into high-dimensional semantic spaces and jointly optimizes reconstruction and clustering objectives, enabling cooperative learning of completion and clustering. Experimental results demonstrate that HSACC significantly outperforms state-of-the-art methods on five benchmark datasets. Ablation studies validate the effectiveness of the hierarchical alignment and dynamic weighting mechanisms, while parameter analysis confirms the model's robustness to hyperparameter variations. 

**Abstract (ZH)**: 基于层次语义对齐和协同补全的不完备多视图聚类框架 

---
# Physics-Informed autoencoder for DSC-MRI Perfusion post-processing: application to glioma grading 

**Title (ZH)**: 基于物理的自动编码器用于DSC-MRI灌注后处理：胶质瘤分级应用 

**Authors**: Pierre Fayolle, Alexandre Bône, Noëlie Debs, Mathieu Naudin, Pascal Bourdon, Remy Guillevin, David Helbert  

**Link**: [PDF](https://arxiv.org/pdf/2510.13886)  

**Abstract**: DSC-MRI perfusion is a medical imaging technique for diagnosing and prognosing brain tumors and strokes. Its analysis relies on mathematical deconvolution, but noise or motion artifacts in a clinical environment can disrupt this process, leading to incorrect estimate of perfusion parameters. Although deep learning approaches have shown promising results, their calibration typically rely on third-party deconvolution algorithms to generate reference outputs and are bound to reproduce their limitations.
To adress this problem, we propose a physics-informed autoencoder that leverages an analytical model to decode the perfusion parameters and guide the learning of the encoding network. This autoencoder is trained in a self-supervised fashion without any third-party software and its performance is evaluated on a database with glioma patients. Our method shows reliable results for glioma grading in accordance with other well-known deconvolution algorithms despite a lower computation time. It also achieved competitive performance even in the presence of high noise which is critical in a medical environment. 

**Abstract (ZH)**: 基于物理的自编码器在MRI灌注成像中的应用：一种无需第三方软件的自监督学习方法及其在胶质瘤分级中的性能评估 

---
# Order from Chaos: Comparative Study of Ten Leading LLMs on Unstructured Data Categorization 

**Title (ZH)**: 从乱到序：十种领先的大语言模型在无结构数据分类中的 comparative study 

**Authors**: Ariel Kamen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13885)  

**Abstract**: This study presents a comparative evaluation of ten state-of-the-art large language models (LLMs) applied to unstructured text categorization using the Interactive Advertising Bureau (IAB) 2.2 hierarchical taxonomy. The analysis employed a uniform dataset of 8,660 human-annotated samples and identical zero-shot prompts to ensure methodological consistency across all models. Evaluation metrics included four classic measures - accuracy, precision, recall, and F1-score - and three LLM-specific indicators: hallucination ratio, inflation ratio, and categorization cost.
Results show that, despite their rapid advancement, contemporary LLMs achieve only moderate classic performance, with average scores of 34% accuracy, 42% precision, 45% recall, and 41% F1-score. Hallucination and inflation ratios reveal that models frequently overproduce categories relative to human annotators. Among the evaluated systems, Gemini 1.5/2.0 Flash and GPT 20B/120B offered the most favorable cost-to-performance balance, while GPT 120B demonstrated the lowest hallucination ratio. The findings suggest that scaling and architectural improvements alone do not ensure better categorization accuracy, as the task requires compressing rich unstructured text into a limited taxonomy - a process that challenges current model architectures.
To address these limitations, a separate ensemble-based approach was developed and tested. The ensemble method, in which multiple LLMs act as independent experts, substantially improved accuracy, reduced inflation, and completely eliminated hallucinations. These results indicate that coordinated orchestration of models - rather than sheer scale - may represent the most effective path toward achieving or surpassing human-expert performance in large-scale text categorization. 

**Abstract (ZH)**: 本研究采用交互式广告局（IAB）2.2级层次分类法，对十个最先进的大型语言模型（LLMs）在非结构化文本分类中的应用进行了比较评估，使用了8,660个人工标注样本和相同的零样本提示，以确保所有模型方法论的一致性。评估指标包括四种经典措施：准确性、精确度、召回率和F1分数，以及三种LLM特定指标：幻觉比率、膨胀比率和分类成本。 

---
# PAGE: Prompt Augmentation for text Generation Enhancement 

**Title (ZH)**: PAGE: 文本生成增强的提示增强 

**Authors**: Mauro Jose Pacchiotti, Luciana Ballejos, Mariel Ale  

**Link**: [PDF](https://arxiv.org/pdf/2510.13880)  

**Abstract**: In recent years, natural language generative models have shown outstanding performance in text generation tasks. However, when facing specific tasks or particular requirements, they may exhibit poor performance or require adjustments that demand large amounts of additional data. This work introduces PAGE (Prompt Augmentation for text Generation Enhancement), a framework designed to assist these models through the use of simple auxiliary modules. These modules, lightweight models such as classifiers or extractors, provide inferences from the input text. The output of these auxiliaries is then used to construct an enriched input that improves the quality and controllability of the generation. Unlike other generation-assistance approaches, PAGE does not require auxiliary generative models; instead, it proposes a simpler, modular architecture that is easy to adapt to different tasks. This paper presents the proposal, its components and architecture, and reports a proof of concept in the domain of requirements engineering, where an auxiliary module with a classifier is used to improve the quality of software requirements generation. 

**Abstract (ZH)**: 近年来，自然语言生成模型在文本生成任务中展现了出色的表现。然而，在面对特定任务或特殊要求时，它们可能表现出色不佳，或者需要通过大量额外数据进行调整。本文介绍了PAGE（Prompt Augmentation for Text Generation Enhancement）框架，该框架通过使用简单的辅助模块来辅助这些模型。这些模块，如分类器或提取器等轻量级模型，从输入文本中提供推断。这些辅助模块的输出用于构建增强输入，从而提高生成的质量和可控性。与其它生成辅助方法不同，PAGE 不需要辅助生成模型；它提出了一种更简单、模块化的架构，易于适应不同任务。本文在需求工程领域提出了该方法的提案、组件和架构，并报告了一个概念验证，其中使用了一个分类器辅助模块来提高软件需求生成的质量。 

---
# Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production 

**Title (ZH)**: Pause and Breathe: 自适应计算促进自我节奏序列生成 

**Authors**: Alexandre Galashov, Matt Jones, Rosemary Ke, Yuan Cao, Vaishnavh Nagarajan, Michael C. Mozer  

**Link**: [PDF](https://arxiv.org/pdf/2510.13879)  

**Abstract**: We explore a class of supervised training objectives that allow a language model to dynamically and autonomously scale the number of compute steps used for each input token. For any token, the model can request additional compute steps by emitting a <don't know> output. If the model is granted a delay, a specialized <pause> token is inserted at the next input step, providing the model with additional compute resources to generate an output. The model can request multiple pauses. To train the model to use <don't know> outputs judiciously and to calibrate its uncertainty, we frame the selection of each output token as a sequential-decision problem with a time cost. We refer to the class of methods as $\textit{Catch Your Breath}$ losses and we study three methods in this class: CYB-AP frames the model's task as anytime prediction, where an output may be required at any step and accuracy is discounted over time; CYB-VA is a variational approach that aims to maximize prediction accuracy subject to a specified distribution over stopping times; and CYB-DP imposes a penalty based on a computational budget. Through fine-tuning experiments, we identify the best performing loss variant. The CYB model needs only one third as much training data as the baseline (no pause) model needs to achieve the same performance, and half as much data as a model with pauses and a cross-entropy loss. We find that the CYB model requests additional steps when doing so improves accuracy, and the model adapts its processing time to token-level complexity and context. For example, it often pauses after plural nouns like $\textit{patients}$ and $\textit{challenges}$ but never pauses after the first token of contracted words like $\textit{wasn}$ and $\textit{didn}$, and it shows high variability for ambiguous tokens like $\textit{won}$, which could function as either a verb or part of a contraction. 

**Abstract (ZH)**: 我们探索了一类监督训练目标，允许语言模型动态自主地调整为每个输入词元所使用的计算步骤数量。对于任意词元，模型可以通过输出<don't know>来请求额外的计算步骤。如果模型获得延迟，将在下一个输入步骤插入一个特殊 的<pause>标记，为模型提供额外的计算资源以生成输出。模型可以请求多次暂停。为了训练模型合理使用<don't know>输出并校准其不确定性，我们将每个输出词元的选择视为具有时间成本的顺序决策问题。我们将这类方法称为“Catch Your Breath”损失，并研究了其中的三种方法：CYB-AP将模型的任务定义为随时预测，其中输出可能在任何步骤被要求，并且随时间准确性会被折现；CYB-VA是一种变分方法，旨在在给定的停止时间分布下最大化预测准确性；而CYB-DP则基于计算预算施加惩罚。通过微调实验，我们确定了表现最佳的损失变体。与基准模型（无暂停）相比，CYB模型只需要三分之一的数据量即可达到相同的性能，而与具有暂停和交叉熵损失的模型相比，只需要一半的数据量。我们发现，CYB模型在进行额外步骤能够提高准确性时会请求更多的步骤，并根据词元级别的复杂性和上下文调整处理时间。例如，它通常在复数名词如“patients”和“challenges”后暂停，但在缩写词如“wasn”和“didn”的第一个词元之后从不暂停，而且它对可能作为动词或缩写一部分的模棱两可的词元如“won”显示出高度的可变性。 

---
# What Layers When: Learning to Skip Compute in LLMs with Residual Gates 

**Title (ZH)**: 何时跳过计算：学习在LLMs中使用残差门跳过计算 

**Authors**: Filipe Laitenberger, Dawid Kopiczko, Cees G.M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13876)  

**Abstract**: We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condenses the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15\% compute while retaining over 90\% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50\% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding. 

**Abstract (ZH)**: GateSkip：一种简单的残差流门控机制，使解码器仅限模型中的token级层跳过成为可能 

---
# FRACCO: A gold-standard annotated corpus of oncological entities with ICD-O-3.1 normalisation 

**Title (ZH)**: FRACCO：一种基于ICD-O-3.1规范化的人类恶性肿瘤实体金标准标注语料库 

**Authors**: Johann Pignat, Milena Vucetic, Christophe Gaudet-Blavignac, Jamil Zaghir, Amandine Stettler, Fanny Amrein, Jonatan Bonjour, Jean-Philippe Goldman, Olivier Michielin, Christian Lovis, Mina Bjelogrlic  

**Link**: [PDF](https://arxiv.org/pdf/2510.13873)  

**Abstract**: Developing natural language processing tools for clinical text requires annotated datasets, yet French oncology resources remain scarce. We present FRACCO (FRench Annotated Corpus for Clinical Oncology) an expert-annotated corpus of 1301 synthetic French clinical cases, initially translated from the Spanish CANTEMIST corpus as part of the FRASIMED initiative. Each document is annotated with terms related to morphology, topography, and histologic differentiation, using the International Classification of Diseases for Oncology (ICD-O) as reference. An additional annotation layer captures composite expression-level normalisations that combine multiple ICD-O elements into unified clinical concepts. Annotation quality was ensured through expert review: 1301 texts were manually annotated for entity spans by two domain experts. A total of 71127 ICD-O normalisations were produced through a combination of automated matching and manual validation by a team of five annotators. The final dataset representing 399 unique morphology codes (from 2549 different expressions), 272 topography codes (from 3143 different expressions), and 2043 unique composite expressions (from 11144 different expressions). This dataset provides a reference standard for named entity recognition and concept normalisation in French oncology texts. 

**Abstract (ZH)**: FRASIMED initiative中的FRench Annotated Corpus for Clinical Oncology (FRACCO) 

---
# Joint Discriminative-Generative Modeling via Dual Adversarial Training 

**Title (ZH)**: 双对抗训练下的判别-生成联合建模 

**Authors**: Xuwang Yin, Claire Zhang, Julie Steele, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13872)  

**Abstract**: Simultaneously achieving robust classification and high-fidelity generative modeling within a single framework presents a significant challenge. Hybrid approaches, such as Joint Energy-Based Models (JEM), interpret classifiers as EBMs but are often limited by the instability and poor sample quality inherent in SGLD-based training. We address these limitations by proposing a novel training framework that integrates adversarial training (AT) principles for both discriminative robustness and stable generative learning. The proposed method introduces three key innovations: (1) the replacement of SGLD-based JEM learning with a stable, AT-based approach that optimizes the energy function by discriminating between real data and PGD-generated contrastive samples using the BCE loss; (2) synergistic adversarial training for the discriminative component that enhances classification robustness while eliminating the need for explicit gradient penalties; and (3) a two-stage training procedure to resolve the incompatibility between batch normalization and EBM training. Experiments on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method substantially improves adversarial robustness over existing hybrid models while maintaining competitive generative performance. On ImageNet, when optimized for generative modeling, our model's generative fidelity surpasses that of BigGAN and approaches diffusion models, representing the first MCMC-based EBM approach to achieve high-quality generation on complex, high-resolution datasets. Our approach addresses key stability issues that have limited JEM scaling and demonstrates that adversarial training can serve as an effective foundation for unified frameworks capable of generating and robustly classifying visual data. 

**Abstract (ZH)**: 在单一框架中同时实现稳健分类和高保真生成建模是一项重大挑战。通过结合对抗训练原则进行判别稳健性和稳定生成学习的新型训练框架 

---
# Unlocking the Potential of Diffusion Language Models through Template Infilling 

**Title (ZH)**: 通过模板填充解锁扩散语言模型的潜力 

**Authors**: Junhoo Lee, Seungyeon Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13870)  

**Abstract**: Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs' generation process. Unlike conventional prefix prompting, TI first generates a structural template for the target response, then fills in the masked segments. To enhance the flexibility of this structural control, we introduce Dynamic Segment Allocation (DSA), which adaptively adjusts segment lengths based on generation confidence. We demonstrate the effectiveness of our approach on mathematical reasoning and code generation benchmarks, achieving consistent improvements of 17.01$\%$p over baseline. Furthermore, we show that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality. 

**Abstract (ZH)**: 扩散语言模型（DLMs）作为一种有前途的自回归语言模型替代方案 emerged as a promising alternative to Autoregressive Language Models，然而其推理策略仍然局限于从自回归范式继承而来的前缀提示。在本文中，我们提出了一种针对DLMs生成过程的定制化条件化方法，称为模板填充（TI）。与传统的前缀提示不同，TI 首先生成目标响应的结构模板，然后填充被遮蔽的部分。为了增强这种结构控制的灵活性，我们引入了动态段分配（DSA），它根据生成置信度自适应地调整段长度。我们在数学推理和代码生成基准上证明了该方法的有效性，相较于基线方法实现了一致提升17.01%。此外，我们展示了在多令牌生成场景下，TI 还能提供额外优势，实现有效的加速同时保持生成质量。 

---
# CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks 

**Title (ZH)**: CoLoR-GAN：生成 adversarial 网络中的低秩适应持续少样本学习 

**Authors**: Munsif Ali, Leonardo Rossi, Massimo Bertozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13869)  

**Abstract**: Continual learning (CL) in the context of Generative Adversarial Networks (GANs) remains a challenging problem, particularly when it comes to learn from a few-shot (FS) samples without catastrophic forgetting. Current most effective state-of-the-art (SOTA) methods, like LFS-GAN, introduce a non-negligible quantity of new weights at each training iteration, which would become significant when considering the long term. For this reason, this paper introduces \textcolor{red}{\textbf{\underline{c}}}ontinual few-sh\textcolor{red}{\textbf{\underline{o}}}t learning with \textcolor{red}{\textbf{\underline{lo}}}w-\textcolor{red}{\textbf{\underline{r}}}ank adaptation in GANs named CoLoR-GAN, a framework designed to handle both FS and CL together, leveraging low-rank tensors to efficiently adapt the model to target tasks while reducing even more the number of parameters required. Applying a vanilla LoRA implementation already permitted us to obtain pretty good results. In order to optimize even further the size of the adapters, we challenged LoRA limits introducing a LoRA in LoRA (LLoRA) technique for convolutional layers. Finally, aware of the criticality linked to the choice of the hyperparameters of LoRA, we provide an empirical study to easily find the best ones. We demonstrate the effectiveness of CoLoR-GAN through experiments on several benchmark CL and FS tasks and show that our model is efficient, reaching SOTA performance but with a number of resources enormously reduced. Source code is available on \href{this https URL}{Github. 

**Abstract (ZH)**: 连续学习（CL）在生成对抗网络（GANs）的上下文中，特别是在从少量样本（FS）中学习且不导致灾难性遗忘的情况下，仍是一个具有挑战性的问题。当前最有效的状态最前沿方法，如LFS-GAN，在每次训练迭代中引入了不容忽视的新权重，长期来看这将变得相当重要。出于这个原因，本文介绍了一种名为CoLoR-GAN的框架，该框架通过低秩张量有效地适应目标任务，同时进一步减少所需的参数数量，以处理同时的少量样本和连续学习。通过简单的LoRA实现已经可以获得相当不错的结果。为了进一步优化适配器的大小，我们提出了在卷积层中引入改进的LoRA（LLoRA）技术。最后，鉴于LoRA超参数选择的重要性，我们提供了一个经验研究来轻松找到最优的超参数。通过在多个基准的连续学习（CL）和少量样本（FS）任务上的实验，我们展示了CoLoR-GAN的有效性，该模型在资源使用上大大减少但仍能达到状态最前沿的性能。源代码可在Github上获取。 

---
# FFT-Accelerated Auxiliary Variable MCMC for Fermionic Lattice Models: A Determinant-Free Approach with $O(N\log N)$ Complexity 

**Title (ZH)**: FFT加速辅助变量MCMC方法：基于O(NlogN)复杂性的行列式自由方法 

**Authors**: Deqian Kong, Shi Feng, Jianwen Xie, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13866)  

**Abstract**: We introduce a Markov Chain Monte Carlo (MCMC) algorithm that dramatically accelerates the simulation of quantum many-body systems, a grand challenge in computational science. State-of-the-art methods for these problems are severely limited by $O(N^3)$ computational complexity. Our method avoids this bottleneck, achieving near-linear $O(N \log N)$ scaling per sweep.
Our approach samples a joint probability measure over two coupled variable sets: (1) particle trajectories of the fundamental fermions, and (2) auxiliary variables that decouple fermion interactions. The key innovation is a novel transition kernel for particle trajectories formulated in the Fourier domain, revealing the transition probability as a convolution that enables massive acceleration via the Fast Fourier Transform (FFT). The auxiliary variables admit closed-form, factorized conditional distributions, enabling efficient exact Gibbs sampling update.
We validate our algorithm on benchmark quantum physics problems, accurately reproducing known theoretical results and matching traditional $O(N^3)$ algorithms on $32\times 32$ lattice simulations at a fraction of the wall-clock time, empirically demonstrating $N \log N$ scaling. By reformulating a long-standing physics simulation problem in machine learning language, our work provides a powerful tool for large-scale probabilistic inference and opens avenues for physics-inspired generative models. 

**Abstract (ZH)**: 一种Markov链蒙特卡洛算法，它极大地加速了量子多体系统的模拟，解决了计算科学中的重大挑战。该方法实现了每扫掠近线性O(N log N)的扩展，避免了立方阶的计算复杂性瓶颈。 

---
# Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning 

**Title (ZH)**: 深度边缘滤波器：回归到深度学习中的手工制作层 

**Authors**: Dongkwan Lee, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13865)  

**Abstract**: We introduce the Deep Edge Filter, a novel approach that applies high-pass filtering to deep neural network features to improve model generalizability. Our method is motivated by our hypothesis that neural networks encode task-relevant semantic information in high-frequency components while storing domain-specific biases in low-frequency components of deep features. By subtracting low-pass filtered outputs from original features, our approach isolates generalizable representations while preserving architectural integrity. Experimental results across diverse domains such as Vision, Text, 3D, and Audio demonstrate consistent performance improvements regardless of model architecture and data modality. Analysis reveals that our method induces feature sparsification and effectively isolates high-frequency components, providing empirical validation of our core hypothesis. The code is available at this https URL. 

**Abstract (ZH)**: 我们介绍了深度边缘滤波器，这是一种通过高通滤波深神经网络特征以提高模型泛化能力的新方法。该方法基于我们假设神经网络在深度特征的高频分量中编码任务相关语义信息，在低频分量中存储领域特定的偏见的假设。通过从原始特征中减去低通滤波输出，该方法分离出可泛化表示的同时保持网络架构的完整性。在视觉、文本、三维和音频等多个领域进行的实验结果表明，无论模型架构和数据模态如何，该方法都能一致地提高性能。分析表明，该方法引发特征稀疏化，并有效地分离出高频分量，提供了对核心假设的经验验证。代码可在以下网址获取。 

---
# Self-Training with Dynamic Weighting for Robust Gradual Domain Adaptation 

**Title (ZH)**: 动态加权的自我训练在鲁棒渐进域适应中的应用 

**Authors**: Zixi Wang, Yushe Cao, Yubo Huang, Jinzhu Wei, Jingzehua Xu, Shuai Zhang, Xin Lai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13864)  

**Abstract**: In this paper, we propose a new method called Self-Training with Dynamic Weighting (STDW), which aims to enhance robustness in Gradual Domain Adaptation (GDA) by addressing the challenge of smooth knowledge migration from the source to the target domain. Traditional GDA methods mitigate domain shift through intermediate domains and self-training but often suffer from inefficient knowledge migration or incomplete intermediate data. Our approach introduces a dynamic weighting mechanism that adaptively balances the loss contributions of the source and target domains during training. Specifically, we design an optimization framework governed by a time-varying hyperparameter $\varrho$ (progressing from 0 to 1), which controls the strength of domain-specific learning and ensures stable adaptation. The method leverages self-training to generate pseudo-labels and optimizes a weighted objective function for iterative model updates, maintaining robustness across intermediate domains. Experiments on rotated MNIST, color-shifted MNIST, portrait datasets, and the Cover Type dataset demonstrate that STDW outperforms existing baselines. Ablation studies further validate the critical role of $\varrho$'s dynamic scheduling in achieving progressive adaptation, confirming its effectiveness in reducing domain bias and improving generalization. This work provides both theoretical insights and a practical framework for robust gradual domain adaptation, with potential applications in dynamic real-world scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 自适应加权自我训练方法在渐进域适应中的应用：一种自我训练动态加权方法（Self-Training with Dynamic Weighting for Robust Gradual Domain Adaptation） 

---
# Ensembling Large Language Models to Characterize Affective Dynamics in Student-AI Tutor Dialogues 

**Title (ZH)**: 基于大型语言模型的ensemble方法刻画学生-AI Tutor对话中的情感动态 

**Authors**: Chenyu Zhang, Sharifa Alghowinem, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13862)  

**Abstract**: While recent studies have examined the leaning impact of large language model (LLM) in educational contexts, the affective dynamics of LLM-mediated tutoring remain insufficiently understood. This work introduces the first ensemble-LLM framework for large-scale affect sensing in tutoring dialogues, advancing the conversation on responsible pathways for integrating generative AI into education by attending to learners' evolving affective states. To achieve this, we analyzed two semesters' worth of 16,986 conversational turns exchanged between PyTutor, an LLM-powered AI tutor, and 261 undergraduate learners across three U.S. institutions. To investigate learners' emotional experiences, we generate zero-shot affect annotations from three frontier LLMs (Gemini, GPT-4o, Claude), including scalar ratings of valence, arousal, and learning-helpfulness, along with free-text emotion labels. These estimates are fused through rank-weighted intra-model pooling and plurality consensus across models to produce robust emotion profiles. Our analysis shows that during interaction with the AI tutor, students typically report mildly positive affect and moderate arousal. Yet learning is not uniformly smooth: confusion and curiosity are frequent companions to problem solving, and frustration, while less common, still surfaces in ways that can derail progress. Emotional states are short-lived--positive moments last slightly longer than neutral or negative ones, but they are fragile and easily disrupted. Encouragingly, negative emotions often resolve quickly, sometimes rebounding directly into positive states. Neutral moments frequently act as turning points, more often steering students upward than downward, suggesting opportunities for tutors to intervene at precisely these junctures. 

**Abstract (ZH)**: 尽管近期研究已经探讨了大型语言模型（LLM）在教育情境中的学习影响，但由LLM介导的辅导的情感动态仍然不够为人所理解。本文介绍了首个用于辅导对话的大规模情感感知集成LLM框架，通过关注学习者不断演变的情感状态，促进了负责任地将生成式AI整合到教育中的对话。为此，我们分析了261名美国三所机构的大二本科生与PyTutor（一个基于LLM的人工智能导师）对谈的两个学期共16,986轮对话。为了研究学习者的情感体验，我们从三个前沿LLM（Gemini、GPT-4o、Claude）中生成零样本情感标注，包括价值、唤醒和学习帮助性等级评分，以及自由文本情绪标签。通过加权内部模型聚合和模型间的多数共识，融合这些估计，产生稳健的情感概貌。分析结果显示，与AI导师互动时，学生通常报告轻微的正面情感和中等唤醒水平。然而，学习并非一帆风顺：困惑和好奇经常伴随着问题解决，尽管挫败感较为罕见，但在某种程度上确实干扰了进度。情绪状态持续时间较短——积极时刻略长于中性或消极时刻，但它们较为脆弱，容易被打断。令人鼓舞的是，负面情绪通常会迅速消解，有时会直接反弹到积极状态。中性时刻常常成为转折点，更常引导学生向上而不是向下，这表明辅导者可以在这些关键时刻采取干预措施。 

---
# ShishuLM: Lightweight Language Model with Hybrid Decoder-MLP Architecture and Paired Weight Sharing 

**Title (ZH)**: 幼儿LM：具有混合解码器-MLP架构和配对权重共享的轻量级语言模型 

**Authors**: Shivanshu Kumar, Gopalakrishnan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13860)  

**Abstract**: While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, presenting opportunities for optimization without compromising performance. Taking insights from research in AI interpretability and inference-time layer pruning, we introduce an efficient language model architecture, referred to as ShishuLM, which reduces both the parameter count and Key-Value (KV) cache requirements. Given the increasing importance of Small Language Models (SLMs) in agentic AI systems, we evaluate our approach on two SLMs of different scales. Our analysis reveals that for moderate-context scenarios, normalization coupled with attention computation is roughly linear with the input, enabling entire transformer blocks to be approximated through Multi-Layer Perceptrons (MLPs). Our results show that ShishuLM provides up to 25% reduction in memory requirements and up to 40% improvement in latency during both training and inference, compared to parent models. Our experimental and analytical findings provide insights towards building more efficient SLM architectures from a pre-training standpoint. 

**Abstract (ZH)**: 尽管Transformer架构在自然语言处理任务中取得了最先进的性能，但这些模型带来了显著的内存和计算开销。 recent研究发现这些模型内部存在大量的架构冗余，提供了在不牺牲性能的情况下进行优化的机会。借鉴AI可解释性和推理时层修剪的研究成果，我们提出了一种高效的语言模型架构，称为ShishuLM，该架构减少了参数数量和Key-Value（KV）缓存需求。鉴于小型语言模型（SLMs）在自主人工智能系统中的日益重要性，我们在两种不同规模的SLMs上评估了我们的方法。我们的分析显示，在中等上下文场景中，归一化与注意力计算大致与输入成线性关系，使得整个Transformer块可以通过多层感知机（MLPs）近似。我们的结果表明，与母模型相比，ShishuLM在训练和推断期间分别减少了最多25%的内存需求和最多40%的延迟。我们的实验和分析结果为进一步从预训练角度构建更高效的SLM架构提供了见解。 

---
# Benchmarking Correctness and Security in Multi-Turn Code Generation 

**Title (ZH)**: 多轮代码生成中的正确性和安全性基准测试 

**Authors**: Ruchit Rawal, Jeffrey Yang Fan Chiang, Chihao Shen, Jeffery Siyuan Tian, Aastha Mahajan, Tom Goldstein, Yizheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13859)  

**Abstract**: AI coding assistants powered by large language models (LLMs) have transformed software development, significantly boosting productivity. While existing benchmarks evaluate the correctness and security of LLM-generated code, they are typically limited to single-turn tasks that do not reflect the iterative nature of real-world development. We introduce MT-Sec, the first benchmark to systematically evaluate both correctness and security in multi-turn coding scenarios. We construct this using a synthetic data pipeline that transforms existing single-turn tasks into semantically aligned multi-turn interaction sequences, allowing reuse of original test suites while modeling the complexity of real-world coding processes. We evaluate 32 open- and closed-source models, and three agent-scaffolding on MT-Sec and observe a consistent 20-27% drop in "correct and secure" outputs from single-turn to multi-turn settings -- even among state-of-the-art models. Beyond full-program generation, we also evaluate models on multi-turn code-diff generation -- an unexplored yet practically relevant setting -- and find that models perform worse here, with increased rates of functionally incorrect and insecure outputs. Finally, we find that while agent scaffoldings boost single-turn code generation performance, they are not quite as effective in multi-turn evaluations. Together, these findings highlight the need for benchmarks that jointly evaluate correctness and security in multi-turn, real-world coding workflows. 

**Abstract (ZH)**: 由大规模语言模型驱动的AI代码助手已革新软件开发，显著提升生产力。尽管现有基准测试评估了大规模语言模型生成代码的正确性和安全性，但它们通常局限于单轮任务，未能反映实际开发过程中的迭代性。我们引入了MT-Sec基准测试，这是首个系统性评估多轮编程场景中正确性和安全性的基准测试。我们通过一个合成数据管道将现有的单轮任务转换为语义对齐的多轮交互序列，从而重新利用原始测试套件并模拟实际编程过程的复杂性。我们在MT-Sec上评估了32个开源和闭源模型，以及三种代理支撑方法，发现从单轮到多轮设置，即使是最先进的模型，“正确且安全”的输出减少了20-27%。此外，我们还将模型评估扩展到多轮代码差异生成——这一尚未探索但具有实际意义的设置，并发现模型在这一方面的表现更差，功能性不正确和不安全的输出增加。最后，我们发现，虽然代理支撑方法能提升单轮代码生成性能，但在多轮评估中效果并不如预期。这些发现强调了对联合评估多轮实际编程工作流中正确性和安全性的基准测试的需求。 

---
# From Craft to Constitution: A Governance-First Paradigm for Principled Agent Engineering 

**Title (ZH)**: 从工艺到宪法：原则性代理工程的治理优先范式 

**Authors**: Qiang Xu, Xiangyu Wen, Changran Xu, Zeju Li, Jianyuan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.13857)  

**Abstract**: The advent of powerful Large Language Models (LLMs) has ushered in an ``Age of the Agent,'' enabling autonomous systems to tackle complex goals. However, the transition from prototype to production is hindered by a pervasive ``crisis of craft,'' resulting in agents that are brittle, unpredictable, and ultimately untrustworthy in mission-critical applications. This paper argues this crisis stems from a fundamental paradigm mismatch -- attempting to command inherently probabilistic processors with the deterministic mental models of traditional software engineering. To solve this crisis, we introduce a governance-first paradigm for principled agent engineering, embodied in a formal architecture we call ArbiterOS. 

**Abstract (ZH)**: 强大的大型语言模型的出现带来了“代理时代”，使自主系统能够应对复杂的目标。然而，从原型到生产的过程受到了普遍存在的“工艺危机”的阻碍，导致代理在关键任务应用中变得脆弱、不可预测且最终不可信赖。本文认为这一危机源于根本性的范式 mismatch——试图用传统软件工程的确定性思维模型来指挥本征概率性的处理器。为了解决这一危机，我们提出了一种以治理为主导的原则性代理工程范式，并将其体现在一种正式架构ArbiterOS中。 

---
# Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA 

**Title (ZH)**: 大规模语言模型增强的多模态检索生成医疗VQA 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.13856)  

**Abstract**: Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks. 

**Abstract (ZH)**: 医学视觉问答（MedVQA）使自然语言查询能够应用于医疗图像，以支持临床决策和患者护理。MEDIQA-WV 2025 共享任务关注伤口护理的视觉问答任务，要求系统从图像和患者查询中生成自由文本回答和结构化的伤口属性。我们介绍了MasonNLP系统，该系统采用了一种通用领域、指令微调的大规模语言模型，并结合了检索增强生成（RAG）框架，该框架包含领域内的文本和视觉示例。该方法将输出与临床相关示例联系起来，提高了推理、模式遵从性和响应质量，在dBLEU、ROUGE、BERTScore和基于LLM的指标上均有所提升。我们的最佳系统在19支团队和51个提交中排名第3，得分为41.37%，表明轻量级RAG与通用目的的语言模型（通过简单的索引和融合添加少量相关示例，无需额外训练或复杂重新排序，在推理时间上增加了一层最小的计算）为多模态临床自然语言处理任务提供了一个简单而有效的基本方案。 

---
# Harnessing Consistency for Robust Test-Time LLM Ensemble 

**Title (ZH)**: 利用一致性提高测试时LLM集成的鲁棒性 

**Authors**: Zhichen Zeng, Qi Yu, Xiao Lin, Ruizhong Qiu, Xuying Ning, Tianxin Wei, Yuchen Yan, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.13855)  

**Abstract**: Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. Token-level consistency captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. Model-level consistency models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness. 

**Abstract (ZH)**: 不同的大规模语言模型（LLMs）展现出 diverse 的优势和劣势，LLM 集群作为一种整合互补能力的有前途的方法得到了发展。尽管在提高集群质量方面取得了显著进展，但很少有人关注集群对潜在错误信号的鲁棒性，这些错误信号通常源自异构的分词方案和不同的模型专业知识。我们的分析表明，集群失败通常源自标记层级和模型层级：前者反映了严重的标记预测分歧，而后者涉及模型间的低信心度和显著差异。鉴于此，我们提出 CoRE，这是一种插件式技术，利用模型一致性来提高 LLM 集群的鲁棒性，可以与各种集群方法无缝集成。标记层级的一致性通过应用低通滤波器来减小程序不确定性，通常由于标记对齐不良，从而在细微程度上提高鲁棒性。模型层级的一致性通过促进高自我信心且与他人差异最小的模型输出来实现全局一致性，从而在较粗的层级上增强鲁棒性。广泛实验表明，CoRE 一致地提高了集群性能和鲁棒性。 

---
# BenchPress: A Human-in-the-Loop Annotation System for Rapid Text-to-SQL Benchmark Curation 

**Title (ZH)**: BenchPress：一种具有人机交互注释功能的快速文本到SQL基准数据整理系统 

**Authors**: Fabian Wenz, Omar Bouattour, Devin Yang, Justin Choi, Cecil Gregg, Nesime Tatbul, Çağatay Demiralp  

**Link**: [PDF](https://arxiv.org/pdf/2510.13853)  

**Abstract**: Large language models (LLMs) have been successfully applied to many tasks, including text-to-SQL generation. However, much of this work has focused on publicly available datasets, such as Fiben, Spider, and Bird. Our earlier work showed that LLMs are much less effective in querying large private enterprise data warehouses and released Beaver, the first private enterprise text-to-SQL benchmark. To create Beaver, we leveraged SQL logs, which are often readily available. However, manually annotating these logs to identify which natural language questions they answer is a daunting task. Asking database administrators, who are highly trained experts, to take on additional work to construct and validate corresponding natural language utterances is not only challenging but also quite costly. To address this challenge, we introduce BenchPress, a human-in-the-loop system designed to accelerate the creation of domain-specific text-to-SQL benchmarks. Given a SQL query, BenchPress uses retrieval-augmented generation (RAG) and LLMs to propose multiple natural language descriptions. Human experts then select, rank, or edit these drafts to ensure accuracy and domain alignment. We evaluated BenchPress on annotated enterprise SQL logs, demonstrating that LLM-assisted annotation drastically reduces the time and effort required to create high-quality benchmarks. Our results show that combining human verification with LLM-generated suggestions enhances annotation accuracy, benchmark reliability, and model evaluation robustness. By streamlining the creation of custom benchmarks, BenchPress offers researchers and practitioners a mechanism for assessing text-to-SQL models on a given domain-specific workload. BenchPress is freely available via our public GitHub repository at this https URL and is also accessible on our website at this http URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在许多任务中取得了成功，包括文本到SQL生成。然而，其中大部分工作集中在公共可用的数据集中，如Fiben、Spider和Bird。我们早期的工作表明，LLMs在查询大型私有企业数据仓库方面效果较差，并发布了Beaver，这是首个私有企业文本到SQL基准。为了创建Beaver，我们利用了通常易于获取的SQL日志。然而，手动注释这些日志以识别它们回答的自然语言问题是一个艰巨的任务。请高度训练的专业数据库管理员额外工作来构建和验证相应的自然语言表达既困难又昂贵。为了解决这一挑战，我们引入了BenchPress，这是一种人类在环系统，旨在加速领域特定文本到SQL基准的创建。给定一个SQL查询，BenchPress利用检索增强生成（RAG）和LLMs提出多个自然语言描述。领域专家随后选择、排名或编辑这些草稿以确保准确性和领域对齐。我们在注释的企业SQL日志上评估了BenchPress，结果表明，借助LLM辅助注释极大地减少了创建高质量基准所需的时间和努力。结果显示，结合人类验证与LLM生成的建议提高了注释准确性、基准可靠性和模型评估稳健性。通过简化自定义基准的创建过程，BenchPress为研究人员和实践者提供了一种在特定领域工作负载上评估文本到SQL模型的机制。BenchPress可通过我们的公共GitHub仓库（此链接）免费获取，也可通过我们的网站（此链接）获取。 

---
# ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups 

**Title (ZH)**: ConsistencyAI: 一个评估大模型在回答不同人口统计学群体时事实一致性表现的标准基准 

**Authors**: Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute  

**Link**: [PDF](https://arxiv.org/pdf/2510.13852)  

**Abstract**: Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies. 

**Abstract (ZH)**: 不同人物视角下LLM提供事实一致性的独立基准：ConsistencyAI 

---
# Revisiting the UID Hypothesis in LLM Reasoning Traces 

**Title (ZH)**: revisit_UID_假设在LLM推理轨迹中的重新审视 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13850)  

**Abstract**: Large language models (LLMs) often solve problems using step-by-step Chain-of-Thought (CoT) reasoning, yet these intermediate steps are frequently unfaithful or hard to interpret. Inspired by the Uniform Information Density (UID) hypothesis in psycholinguistics -- which posits that humans communicate by maintaining a stable flow of information -- we introduce entropy-based metrics to analyze the information flow within reasoning traces. Surprisingly, across three challenging mathematical benchmarks, we find that successful reasoning in LLMs is globally non-uniform: correct solutions are characterized by uneven swings in information density, in stark contrast to human communication patterns. This result challenges assumptions about machine reasoning and suggests new directions for designing interpretable and adaptive reasoning models. 

**Abstract (ZH)**: 大型语言模型中的推理过程往往是逐步进行的，但其中的中间步骤经常不准确或难以解释。受心理语言学中的均匀信息密度（UID）假设启发，我们通过引入基于熵的度量来分析推理痕迹中的信息流。令人惊讶的是，在三个具有挑战性的数学基准测试中，我们发现成功的推理在全局上是非均匀的：正确解题的特点是信息密度存在不均等的波动，这与人类沟通模式截然不同。这一结果挑战了关于机器推理的假设，并建议新的方向，以设计可解释和自适应的推理模型。 

---
# On-device System of Compositional Multi-tasking in Large Language Models 

**Title (ZH)**: 大型语言模型中设备端组合作业多任务系统 

**Authors**: Ondrej Bohdal, Konstantinos Theodosiadis, Asterios Mpatziakas, Dimitris Filippidis, Iro Spyrou, Christos Zonios, Anastasios Drosou, Dimosthenis Ioannidis, Kyeng-Hun Lee, Jijoong Moon, Hyeonmok Ko, Mete Ozay, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13848)  

**Abstract**: Large language models (LLMs) are commonly adapted for diverse downstream tasks via parameter-efficient fine-tuning techniques such as Low-Rank Adapters (LoRA). While adapters can be combined to handle multiple tasks separately, standard approaches struggle when targeting the simultaneous execution of complex tasks, such as generating a translated summary from a long conversation. To address this challenge, we propose a novel approach tailored specifically for compositional multi-tasking scenarios involving summarization and translation. Our technique involves adding a learnable projection layer on top of the combined summarization and translation adapters. This design enables effective integration while maintaining efficiency through reduced computational overhead compared to alternative strategies requiring extensive retraining or sequential processing. We demonstrate the practical viability of our method within an on-device environment by developing an Android app capable of executing compositional tasks seamlessly. Experimental results indicate our solution performs well and is fast in both cloud-based and on-device implementations, highlighting the potential benefits of adopting our framework in real-world applications demanding high-speed operation alongside resource constraints. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过参数高效微调技术（如低秩适配器LoRA）被广泛adapt以应对多种下游任务。虽然适配器可以组合起来分别处理多个任务，但标准方法在同时执行复杂任务（如生成长时间对话的翻译摘要）时表现不佳。为应对这一挑战，我们提出了一种专门针对涉及摘要和翻译的组合多任务场景的新方法。该方法在综合摘要和翻译适配器之上添加了一个可学习的投影层，这种设计能够在减少计算开销的同时实现有效的集成，避免了需要大量重新训练或顺序处理的替代策略。我们通过开发一个能够在设备端无缝执行组合任务的Android应用程序，展示了我们方法的实用性。实验结果表明，我们的解决方案在基于云和设备端的实现中均表现出色且快速，突显了在同时要求高速操作和资源限制的应用场景中采用我们框架的潜在优势。 

---
# DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models 

**Title (ZH)**: DynaSpec: 基于上下文的动态推测采样方法用于大词汇量语言模型 

**Authors**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar  

**Link**: [PDF](https://arxiv.org/pdf/2510.13847)  

**Abstract**: Speculative decoding (a.k.a. speculative sampling) has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V|d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed subset of the target model's vocabulary, ranked in descending order of token frequency. Although this reduces draft-time compute, it is brittle, since: (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. On standard speculative-decoding benchmarks, we observe consistent gains in mean accepted length over fixed-shortlist baselines, while context-dependent selection enables smaller shortlists without degrading acceptance. 

**Abstract (ZH)**: 基于上下文的动态短列表机制：一种鲁棒、加速且通用的 speculative 解码方法 

---
# Information flow in multilayer perceptrons: an in-depth analysis 

**Title (ZH)**: 多层感知器中的信息流：深入分析 

**Authors**: Giuliano Armano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13846)  

**Abstract**: Analysing how information flows along the layers of a multilayer perceptron is a topic of paramount importance in the field of artificial neural networks. After framing the problem from the point of view of information theory, in this position article a specific investigation is conducted on the way information is processed, with particular reference to the requirements imposed by supervised learning. To this end, the concept of information matrix is devised and then used as formal framework for understanding the aetiology of optimisation strategies and for studying the information flow. The underlying research for this article has also produced several key outcomes: i) the definition of a parametric optimisation strategy, ii) the finding that the optimisation strategy proposed in the information bottleneck framework shares strong similarities with the one derived from the information matrix, and iii) the insight that a multilayer perceptron serves as a kind of "adaptor", meant to process the input according to the given objective. 

**Abstract (ZH)**: 分析多层感知机各层间信息流变对人工神经网络领域至关重要。从信息论的角度界定问题后，本文针对信息处理方式进行具体调查，特别关注监督学习提出的要求。为此，我们提出了信息矩阵的概念，并将其用作理解优化策略起源和研究信息流的形式框架。本文的研究还产生了多个关键成果：i) 定义了参数化优化策略，ii) 发现信息瓶颈框架中提出的优化策略与从信息矩阵中推导出的优化策略具有很强的相似性，iii) 洞察到多层感知机充当了一种“适配器”，旨在根据给定的目标处理输入。 

---
# Serialized EHR make for good text representations 

**Title (ZH)**: 序列化的电子健康记录-make for good text representations 

**Authors**: Zhirong Chou, Quan Qin, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13843)  

**Abstract**: The emergence of foundation models in healthcare has opened new avenues for learning generalizable representations from large scale clinical data. Yet, existing approaches often struggle to reconcile the tabular and event based nature of Electronic Health Records (EHRs) with the sequential priors of natural language models. This structural mismatch limits their ability to capture longitudinal dependencies across patient encounters. We introduce SerialBEHRT, a domain aligned foundation model that extends SciBERT through additional pretraining on structured EHR sequences. SerialBEHRT is designed to encode temporal and contextual relationships among clinical events, thereby producing richer patient representations. We evaluate its effectiveness on the task of antibiotic susceptibility prediction, a clinically meaningful problem in antibiotic stewardship. Through extensive benchmarking against state of the art EHR representation strategies, we demonstrate that SerialBEHRT achieves superior and more consistent performance, highlighting the importance of temporal serialization in foundation model pretraining for healthcare. 

**Abstract (ZH)**: 基础模型在医疗领域的出现为从大规模临床数据中学习可泛化的表示开辟了新的途径。然而，现有方法往往难以调和电子健康记录(EHRs)的表格式和事件驱动性质与自然语言模型的序列先验之间的结构性不匹配。这种结构性不匹配限制了它们捕捉患者就诊间纵向依赖性的能力。我们提出了一种领域对齐的基础模型SerialBEHRT，通过额外对结构化EHR序列进行预训练，扩展了SciBERT。SerialBEHRT旨在编码临床事件之间的时空关系，从而生成 richer 的患者表示。我们将其效果评估应用于抗生素敏感性预测任务，这是抗生素管理中的一个临床相关问题。通过与最先进的EHR表示策略的广泛基准测试，我们证明SerialBEHRT实现了更优且更一致的性能，强调了在医疗领域对基础模型进行时间序列化预训练的重要性。 

---
# ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking 

**Title (ZH)**: ADMIT：基于RAG的事实检查中的少量样本知识投毒攻击 

**Authors**: Yutao Wu, Xiao Liu, Yinghui Li, Yifeng Gao, Yifan Ding, Jiale Ding, Xiang Zheng, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13842)  

**Abstract**: Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems. 

**Abstract (ZH)**: 知识投毒对检索增强生成（RAG）系统构成关键威胁，通过向知识库注入 adversarial 内容，欺骗大型语言模型（LLMs）生成基于篡改上下文的攻击者可控输出。先前的工作强调了 LLMs 对误导性或恶意检索内容的脆弱性。然而，现实世界中的事实核查场景更加复杂，因为可信证据通常主导了检索池。为了探究这一问题，我们将知识投毒扩展到事实核查场景中，在此场景下检索到的上下文包括真实的支持性或反驳性证据。我们提出了 **ADMIT**（**AD**versarial **M**ulti-**I**njection **T**echnique），这是一种少量示例、语义对齐的投毒攻击方法，能够在不访问目标LLMs、检索器或字级控制的情况下翻转事实核查决策并诱导误导性的理由。广泛实验表明，ADMIT能够在4种检索器、11种LLMs和4种跨域基准上有效转移，以极低的投毒率 $0.93 \times 10^{-6}$ 达成平均攻击成功率（ASR）86%，即使在面对强大反证证据的情况下仍然保持鲁棒性。与先前的最佳攻击相比，ADMIT在所有场景下将ASR提高了11.2%，揭示了现实世界RAG基础事实核查系统的重大漏洞。 

---
# Meronymic Ontology Extraction via Large Language Models 

**Title (ZH)**: 大型语言模型驱动的聚敛 ontology 提取 

**Authors**: Dekai Zhang, Simone Conia, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2510.13839)  

**Abstract**: Ontologies have become essential in today's digital age as a way of organising the vast amount of readily available unstructured text. In providing formal structure to this information, ontologies have immense value and application across various domains, e.g., e-commerce, where countless product listings necessitate proper product organisation. However, the manual construction of these ontologies is a time-consuming, expensive and laborious process. In this paper, we harness the recent advancements in large language models (LLMs) to develop a fully-automated method of extracting product ontologies, in the form of meronymies, from raw review texts. We demonstrate that the ontologies produced by our method surpass an existing, BERT-based baseline when evaluating using an LLM-as-a-judge. Our investigation provides the groundwork for LLMs to be used more generally in (product or otherwise) ontology extraction. 

**Abstract (ZH)**: 本体已成为当今数字时代组织大量可用的非结构化文本的一种 essentials 方式。通过为这些信息提供正式结构，本体在各个领域具有巨大的价值和应用潜力，例如电子商务领域，其中无数的产品列表需要适当的产品组织。然而，本体的手动构造是一个耗时、昂贵且劳动密集的过程。在本文中，我们利用大型语言模型（LLMs）的最新进展，开发了一种完全自动的方法，从原始评论文本中提取产品本体，形式为部分-整体关系。我们证明，我们的方法生成的本体在使用 LLM 作为评判标准时超越了一个现有的基于 BERT 的基线方法。我们的研究为通用使用大型语言模型进行本体（产品或其他类型）提取奠定了基础。 

---
# Seeing Hate Differently: Hate Subspace Modeling for Culture-Aware Hate Speech Detection 

**Title (ZH)**: 从不同视角看待仇恨：基于文化意识的仇恨言辞检测的仇恨子空间建模 

**Authors**: Weibin Cai, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.13837)  

**Abstract**: Hate speech detection has been extensively studied, yet existing methods often overlook a real-world complexity: training labels are biased, and interpretations of what is considered hate vary across individuals with different cultural backgrounds. We first analyze these challenges, including data sparsity, cultural entanglement, and ambiguous labeling. To address them, we propose a culture-aware framework that constructs individuals' hate subspaces. To alleviate data sparsity, we model combinations of cultural attributes. For cultural entanglement and ambiguous labels, we use label propagation to capture distinctive features of each combination. Finally, individual hate subspaces, which in turn can further enhance classification performance. Experiments show our method outperforms state-of-the-art by 1.05\% on average across all metrics. 

**Abstract (ZH)**: 含有 Hate 言论检测的研究已十分广泛，但现有方法往往忽略了现实世界的复杂性：训练标签存有偏见，不同文化背景的人对什么是 Hate 的理解也各不相同。我们首先分析这些挑战，包括数据稀疏性、文化纠缠和模糊标签。为此，我们提出了一种文化意识框架，构建个体的 Hate 子空间。为缓解数据稀疏性，我们建模文化属性的组合。针对文化纠缠和模糊标签，我们使用标签传播来捕捉每种组合的独特特征。最终，个体的 Hate 子空间可以进一步提升分类性能。实验结果显示，我们的方法在所有指标上的平均性能比现有最佳方法高出 1.05%。 

---
# SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models 

**Title (ZH)**: 基于相似性聚合的大语言模型不确定性量化（SIMBA UQ） 

**Authors**: Debarun Bhattacharjya, Balaji Ganesan, Junkyu Lee, Radu Marinescu, Katsiaryna Mirylenka, Michael Glass, Xiao Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.13836)  

**Abstract**: When does a large language model (LLM) know what it does not know? Uncertainty quantification (UQ) provides measures of uncertainty, such as an estimate of the confidence in an LLM's generated output, and is therefore increasingly recognized as a crucial component of trusted AI systems. Black-box UQ methods do not require access to internal model information from the generating LLM and therefore have numerous real-world advantages, such as robustness to system changes, adaptability to choice of LLM, reduced costs, and computational tractability. In this paper, we investigate the effectiveness of UQ techniques that are primarily but not necessarily entirely black-box, where the consistency between a generated output and other sampled generations is used as a proxy for confidence in its correctness. We propose a high-level non-verbalized similarity-based aggregation framework that subsumes a broad swath of UQ approaches suitable for complex generative tasks, as well as introduce specific novel techniques from the framework that train confidence estimation models using small training sets. Through an empirical study with datasets spanning the diverse tasks of question answering, summarization, and text-to-SQL, we demonstrate that our proposed similarity-based methods can yield better calibrated confidences than baselines. 

**Abstract (ZH)**: 大型语言模型何时知道自己不知道的东西？不确定性量化提供了不确定性度量，如对大型语言模型生成输出的信心估计，因此不确定性量化被日益认为是可信赖人工智能系统的关键组成部分。黑盒不确定性量化方法无需访问生成大型语言模型的内部信息，因此具有诸多现实世界优势，如对系统变化的鲁棒性、适应不同选择的大型语言模型的能力、成本降低和计算可行性。在本文中，我们研究了主要但不一定完全是黑盒的不确定性量化技术的有效性，其中生成输出与其他采样生成的一致性被用作其正确性的信心代理。我们提出了一种高层次的非言语相似性聚合框架，该框架涵盖了适用于复杂生成任务的一系列不确定性量化方法，并引入了基于框架的具体新颖技术，使用小规模训练集训练信心估计模型。通过跨问题回答、摘要和文本到SQL等多样任务数据集的实证研究，我们展示了所提出的方法可以获得比基线更好的校准信心。 

---
# ConDABench: Interactive Evaluation of Language Models for Data Analysis 

**Title (ZH)**: ConDABench：数据分析语言模型的交互评估 

**Authors**: Avik Dutta, Priyanshu Gupta, Hosein Hasanbeig, Rahul Pratap Singh, Harshit Nigam, Sumit Gulwani, Arjun Radhakrishna, Gustavo Soares, Ashish Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2510.13835)  

**Abstract**: Real-world data analysis tasks often come with under-specified goals and unclean data. User interaction is necessary to understand and disambiguate a user's intent, and hence, essential to solving these complex tasks. Existing benchmarks for evaluating LLMs on data analysis tasks do not capture these complexities or provide first-class support for interactivity. We introduce ConDABench, a framework for generating conversational data analysis (ConDA) benchmarks and evaluating external tools on the generated benchmarks. \bench consists of (a) a multi-agent workflow for generating realistic benchmarks from articles describing insights gained from public datasets, (b) 1,420 ConDA problems generated using this workflow, and (c) an evaluation harness that, for the first time, makes it possible to systematically evaluate conversational data analysis tools on the generated ConDA problems. Evaluation of state-of-the-art LLMs on the benchmarks reveals that while the new generation of models are better at solving more instances, they are not necessarily better at solving tasks that require sustained, long-form engagement. ConDABench is an avenue for model builders to measure progress towards truly collaborative models that can complete complex interactive tasks. 

**Abstract (ZH)**: 现实世界的数据分析任务往往目标不明确且数据不干净。用户互动对于理解用户意图和消歧是非常必要的，因此对于解决这些复杂任务是必不可少的。现有的用于评估大模型（LLMs）在数据分析任务上表现的基准没有捕捉到这些复杂性或提供一等交互支持。我们引入了ConDABench，这是一个用于生成对话式数据分析（ConDA）基准并评估外部工具的框架。ConDABench 包括（a）一个多代理流程，用于从描述公共数据集洞察的论文中生成现实基准；（b）使用此流程生成的1,420个ConDA问题；以及（c）一个评估框架，首次使得系统地评估对话式数据分析工具成为可能。对基准上的最先进的大模型进行评估揭示了这样一个事实：虽然新一代模型在解决更多实例方面表现更好，但在需要持续长时间交互的任务上并不一定表现更好。ConDABench 为模型构建者提供了一个途径，以衡量向真正协作模型发展的进程，这些模型能够完成复杂的交互式任务。 

---
# Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning 

**Title (ZH)**: 熵遇重要性：统一的头重要性-熵得分方法以实现Transformer剪枝的稳定性和高效性 

**Authors**: Minsik Choi, Hyegang Son, Changhoon Kim, Young Geun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13832)  

**Abstract**: Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication. 

**Abstract (ZH)**: 基于Transformer的模型在自然语言处理任务中取得了显著性能。然而，其结构特性——多层和注意力头——在推理和部署中引入了效率挑战。为应对这些挑战，最近提出了多种剪枝方法。值得注意的是，基于梯度的方法利用头重要性得分（HIS）因其实用性、效率以及识别冗余头的能力而受到关注。然而，HIS单独使用有局限性，因为它只捕捉了梯度驱动的贡献，忽略了注意力模式的多样性。为克服这些局限，我们提出了一种新的剪枝标准HIES（头重要性-熵得分），它将头重要性得分与注意力熵相结合，提供了头贡献的补充证据。实验证明，基于HIES的剪枝方法相比仅使用HIS的方法，在模型质量上提高了15.2%，在稳定性上提高了2.04倍，能够在不牺牲准确性和稳定性的前提下实现大规模模型压缩。代码将在发表后开源。 

---
# Informed Routing in LLMs: Smarter Token-Level Computation for Faster Inference 

**Title (ZH)**: LLMs中的知情路由：更智能的标记级别计算以实现更快的推理 

**Authors**: Chao Han, Yijuan Liang, Zihao Xuan, Daokuan Wu, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13831)  

**Abstract**: The deployment of large language models (LLMs) in real-world applications is increasingly limited by their high inference cost. While recent advances in dynamic token-level computation allocation attempt to improve efficiency by selectively activating model components per token, existing methods rely on greedy routing--a myopic execute-or-skip mechanism that often leads to irreversible information loss and suboptimal token selection. This paper introduces informed routing, a new paradigm that proactively addresses these issues. The key insight is to assess not only a token's immediate importance but also its recoverability, i.e., how well its transformation can be approximated. To this end, we propose the Lightweight Feature Forecaster (LFF), a small predictive module that estimates a unit's output before routing decisions are made. This enables a flexible execute-or-approximate policy that preserves model fidelity while drastically reducing computation. Extensive experiments on both language modeling and reasoning tasks show that informed routing achieves state-of-the-art efficiency-performance trade-offs across multiple sparsity levels. Notably, even without final LoRA fine-tuning, our method matches or surpasses strong baselines that require full fine-tuning, all while reducing training time by over 50%. The code is available at: this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在实际应用中的部署 increasingly受限于其高昂的推理成本。尽管最近在动态 token 级计算分配方面的进步试图通过按 token 选择性激活模型组件来提高效率，现有方法仍然依赖于贪婪路由——一种短视的执行或跳过机制，这往往导致不可逆的信息损失和次优的 token 选择。本文引入了明智路由，这是一种新的范式，旨在前瞻性地解决这些问题。关键洞察是不仅评估 token 的即时重要性，还要评估其恢复性，即其转换能被近似得有多好。为此，我们提出了轻量级特征预测器（LFF），一个小型预测模块，在路由决策之前估计单元的输出。这使得能够执行或近似策略，同时保持模型准确性并大幅减少计算量。在语言建模和推断任务上的广泛实验表明，明智路由在多个稀疏度级别上实现了最先进的效率-性能折中。值得注意的是，即使不进行最终的 LoRA 微调，我们的方法也能达到或超越需要完全微调的强大基线，同时将训练时间减少超过 50%。代码可在以下链接获取：this https URL。 

---
# Users as Annotators: LLM Preference Learning from Comparison Mode 

**Title (ZH)**: 用户作为标注者：比较模式下的LLM偏好学习 

**Authors**: Zhongze Cai, Xiaocheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13830)  

**Abstract**: Pairwise preference data have played an important role in the alignment of large language models (LLMs). Each sample of such data consists of a prompt, two different responses to the prompt, and a binary label indicating which of the two responses is better. The labels are usually annotated by professional human annotators. In this paper, we consider an alternative approach to collect pairwise preference data -- user annotation from comparison mode. With the increasingly wider adoption of LLMs among the population, users are contributing more and more of their preference labels through their daily interactions with the LLMs. The upside of such labels is that users are the best experts in judging the responses to their own queries/prompts, but the downside is the lack of quality control in these labels. In this paper, we consider a new idea of generating two responses from two different models or two different versions of the same model. The asymmetry allows us to make an inference of the user's data quality through our proposed user behavior model. We develop an expectation-maximization algorithm to estimate a latent quality factor of the user, and filter users' annotation data accordingly. The downstream task shows the effectiveness of our approach in both capturing the user behavior and data filtering for LLM alignment. 

**Abstract (ZH)**: 大规模语言模型对齐中，成对偏好数据的作用：基于用户比较模式的成对偏好数据收集新方法及其在质量控制中的应用 

---
# A Linguistics-Aware LLM Watermarking via Syntactic Predictability 

**Title (ZH)**: 基于句法学预测的 Linguistics 意识大语言模型水印技术 

**Authors**: Shinwoo Park, Hyejin Park, Hyeseon Ahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.13829)  

**Abstract**: As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthen it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）不断发展，可靠的技术治理工具变得至关重要。公开可验证的水印尤为重要，有助于建立可信赖的AI生态系统。一个主要挑战persist：在保持文本质量的同时提高检测鲁棒性。近期研究试图通过利用模型输出分布的信号（例如，token级熵）来解决这一权衡问题，但其依赖于特定于模型的信号成为了公共验证的一大障碍，因为检测过程需要访问底层模型的logits。我们提出了STELA，一种新颖的框架，根据语言固有的语言学自由度来调整水印强度。STELA利用部分词性（POS）n-gram建模的语言学不确定性，根据语法约束程度动态调节信号，从而在保持质量的同时在语言学灵活性更大的语境中增强可检测性。我们的检测器无需访问任何模型logits，从而支持公开可验证的检测。通过在类型学上不同的语言（分析性英语、孤立语汉语和粘着语韩语）中的实验，我们展示了STELA在检测鲁棒性方面优于先前方法。我们的代码可在以下链接处获取：this https URL。 

---
# Bridging the Semantic Gap: Contrastive Rewards for Multilingual Text-to-SQL 

**Title (ZH)**: 跨越语义鸿沟：多语言文本到SQL的对比奖励 

**Authors**: Ashish Kattamuri, Ishita Prasad, Meetu Malhotra, Arpita Vats, Rahul Raja, Albert Lie  

**Link**: [PDF](https://arxiv.org/pdf/2510.13827)  

**Abstract**: Current Text-to-SQL methods are evaluated and only focused on executable queries, overlooking the semantic alignment challenge -- both in terms of the semantic meaning of the query and the correctness of the execution results. Even execution accuracy itself shows significant drops when moving from English to other languages, with an average decline of 6 percentage points across non-English languages. We address these challenges by presenting a new framework that combines Group Relative Policy Optimization (GRPO) within a multilingual contrastive reward signal to enhance both task efficiency and semantic accuracy in Text-to-SQL systems in cross-lingual scenarios. Our method teaches models to obtain better correspondence between SQL generation and user intent by combining a reward signal based on semantic similarity. On the seven-language MultiSpider dataset, fine-tuning the LLaMA-3-3B model with GRPO improved the execution accuracy up to 87.4 percent (+26 pp over zero-shot) and semantic accuracy up to 52.29 percent (+32.86 pp). Adding our contrastive reward signal in the GRPO framework further improved the average semantic accuracy to 59.14 percent (+6.85 pp, up to +10 pp for Vietnamese). Our experiments showcase that a smaller, parameter-efficient 3B LLaMA model fine-tuned with our contrastive reward signal outperforms a much larger zero-shot 8B LLaMA model, with an uplift of 7.43 pp in execution accuracy (from 81.43 percent on the 8B model to 88.86 percent on the 3B model), and nearly matches its semantic accuracy (59.14 percent vs. 68.57 percent) -- all using just 3,000 reinforcement learning training examples. These results demonstrate how we can improve the performance of Text-to-SQL systems with contrastive rewards for directed semantic alignment, without requiring large-scale training datasets. 

**Abstract (ZH)**: 当前的文本到SQL方法仅评估可执行查询，忽视了语义对齐挑战——包括查询的语义意义和执行结果的正确性。即使在从英语到其他语言时，执行准确性也显示出显著下降，平均下降6个百分点。我们通过提出一种新框架来应对这些挑战，该框架结合了组相对策略优化（GRPO）与多语言对比奖励信号，以增强跨语言场景中文本到SQL系统的任务效率和语义准确性。我们通过结合基于语义相似性的奖励信号，使模型更好地实现SQL生成与用户意图之间的对应关系。在七语言的MultiSpider数据集上，使用GRPO微调LLaMA-3-3B模型的执行准确性提高了至87.4%（零样本基础上提高26个百分点），语义准确性提高了至52.29%（提高32.86个百分点）。将我们的对比性奖励信号加入GRPO框架中进一步提高了平均语义准确性至59.14%（提高6.85个百分点，对于越南语则提高10个百分点）。实验结果显示，使用我们对比性奖励信号微调的3B参数高效LLaMA模型在执行准确性方面优于8B零样本LLaMA模型，提高了7.43个百分点（从8B模型的81.43%提高至3B模型的88.86%），并在语义准确性方面几乎与其持平（59.14% vs. 68.57%）——仅使用3000个强化学习训练样本。这些结果展示了如何通过对比性奖励实现定向语义对齐来提升文本到SQL系统的性能，而无需大规模训练数据集。 

---
# Towards Neurocognitive-Inspired Intelligence: From AI's Structural Mimicry to Human-Like Functional Cognition 

**Title (ZH)**: 面向神经认知启发的智能：从AI的结构模拟到类似人类的功能认知 

**Authors**: Noorbakhsh Amiri Golilarz, Hassan S. Al Khatib, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13826)  

**Abstract**: Artificial intelligence has advanced significantly through deep learning, reinforcement learning, and large language and vision models. However, these systems often remain task specific, struggle to adapt to changing conditions, and cannot generalize in ways similar to human cognition. Additionally, they mainly focus on mimicking brain structures, which often leads to black-box models with limited transparency and adaptability. Inspired by the structure and function of biological cognition, this paper introduces the concept of "Neurocognitive-Inspired Intelligence (NII)," a hybrid approach that combines neuroscience, cognitive science, computer vision, and AI to develop more general, adaptive, and robust intelligent systems capable of rapid learning, learning from less data, and leveraging prior experience. These systems aim to emulate the human brain's ability to flexibly learn, reason, remember, perceive, and act in real-world settings with minimal supervision. We review the limitations of current AI methods, define core principles of neurocognitive-inspired intelligence, and propose a modular, biologically inspired architecture that emphasizes integration, embodiment, and adaptability. We also discuss potential implementation strategies and outline various real-world applications, from robotics to education and healthcare. Importantly, this paper offers a hybrid roadmap for future research, laying the groundwork for building AI systems that more closely resemble human cognition. 

**Abstract (ZH)**: 基于神经认知灵感的人工智能（NII）：一种结合神经科学、认知科学、计算机视觉和AI的综合方法 

---
# A2AS: Agentic AI Runtime Security and Self-Defense 

**Title (ZH)**: A2AS: 自主AI运行时安全与自我防御 

**Authors**: Eugene Neelou, Ivan Novikov, Max Moroz, Om Narayan, Tiffany Saade, Mika Ayenson, Ilya Kabanov, Jen Ozmen, Edward Lee, Vineeth Sai Narajala, Emmanuel Guilherme Junior, Ken Huang, Huseyin Gulsin, Jason Ross, Marat Vyshegorodtsev, Adelin Travers, Idan Habler, Rahul Jadav  

**Link**: [PDF](https://arxiv.org/pdf/2510.13825)  

**Abstract**: The A2AS framework is introduced as a security layer for AI agents and LLM-powered applications, similar to how HTTPS secures HTTP. A2AS enforces certified behavior, activates model self-defense, and ensures context window integrity. It defines security boundaries, authenticates prompts, applies security rules and custom policies, and controls agentic behavior, enabling a defense-in-depth strategy. The A2AS framework avoids latency overhead, external dependencies, architectural changes, model retraining, and operational complexity. The BASIC security model is introduced as the A2AS foundation: (B) Behavior certificates enable behavior enforcement, (A) Authenticated prompts enable context window integrity, (S) Security boundaries enable untrusted input isolation, (I) In-context defenses enable secure model reasoning, (C) Codified policies enable application-specific rules. This first paper in the series introduces the BASIC security model and the A2AS framework, exploring their potential toward establishing the A2AS industry standard. 

**Abstract (ZH)**: A2AS框架作为AI代理和LLM驱动应用的安全层，类似于HTTPS对HTTP的保护。A2AS确保认证行为、激活模型自我防御并保证上下文窗口完整性。它定义安全边界、验证提示、应用安全规则和自定义策略，并控制代理行为，实现多层防御策略。A2AS框架避免了延迟 overhead、外部依赖、架构更改、模型再训练和运营复杂性。引入了BASIC安全模型作为A2AS的基础：(B) 行为证书用于执行行为，(A) 验证提示用于保证上下文窗口完整性，(S) 安全边界用于隔离不可信输入，(I) 在上下文中的防御用于安全模型推理，(C) 编码策略用于应用特定规则。本文系列的第一篇论文介绍了BASIC安全模型和A2AS框架，探索其建立A2AS行业标准的潜力。 

---
# Leveraging Wireless Sensor Networks for Real-Time Monitoring and Control of Industrial Environments 

**Title (ZH)**: 利用无线传感器网络实现工业环境的实时监测与控制 

**Authors**: Muhammad Junaid Asif, Shazia Saqib, Rana Fayyaz Ahmad, Hamza Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13820)  

**Abstract**: This research proposes an extensive technique for monitoring and controlling the industrial parameters using Internet of Things (IoT) technology based on wireless communication. We proposed a system based on NRF transceivers to establish a strong Wireless Sensor Network (WSN), enabling transfer of real-time data from multiple sensors to a central setup that is driven by ARDUINO microcontrollers. Different key parameters, crucial for industrial setup such as temperature, humidity, soil moisture and fire detection, are monitored and displayed on an LCD screen, enabling factory administration to oversee the industrial operations remotely over the internet. Our proposed system bypasses the need for physical presence for monitoring by addressing the shortcomings of conventional wired communication systems. Other than monitoring, there is an additional feature to remotely control these parameters by controlling the speed of DC motors through online commands. Given the rising incidence of industrial fires over the worldwide between 2020 and 2024 due to an array of hazards, this system with dual functionality boosts the overall operational efficiency and safety. This overall integration of IoT and Wireless Sensor Network (WSN) reduces the potential risks linked with physical monitoring, providing rapid responses in emergency scenarios, including the activation of firefighting equipment. The results show that innovations in wireless communication perform an integral part in industrial process automation and safety, paving the way to more intelligent and responsive operating environments. Overall, this study highlights the potential for change of IoT-enabled systems to revolutionize monitoring and control in a variety of industrial applications, resulting in increased productivity and safety. 

**Abstract (ZH)**: 基于物联网技术的无线通信工业参数监控与控制方法研究 

---
# GQVis: A Dataset of Genomics Data Questions and Visualizations for Generative AI 

**Title (ZH)**: GQVis: 用于生成式AI的基因组数据问题和可视化数据集 

**Authors**: Skylar Sargent Walters, Arthea Valderrama, Thomas C. Smits, David Kouřil, Huyen N. Nguyen, Sehi L'Yi, Devin Lange, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2510.13816)  

**Abstract**: Data visualization is a fundamental tool in genomics research, enabling the exploration, interpretation, and communication of complex genomic features. While machine learning models show promise for transforming data into insightful visualizations, current models lack the training foundation for domain-specific tasks. In an effort to provide a foundational resource for genomics-focused model training, we present a framework for generating a dataset that pairs abstract, low-level questions about genomics data with corresponding visualizations. Building on prior work with statistical plots, our approach adapts to the complexity of genomics data and the specialized representations used to depict them. We further incorporate multiple linked queries and visualizations, along with justifications for design choices, figure captions, and image alt-texts for each item in the dataset. We use genomics data retrieved from three distinct genomics data repositories (4DN, ENCODE, Chromoscope) to produce GQVis: a dataset consisting of 1.14 million single-query data points, 628k query pairs, and 589k query chains. The GQVis dataset and generation code are available at this https URL and this https URL. 

**Abstract (ZH)**: 基因组学研究中的数据可视化是探索、解释和传达复杂基因组特征的基本工具。虽然机器学习模型有望将数据转化为洞察性的可视化，但当前的模型缺乏针对特定领域任务的训练基础。为提供一个面向基因组学模型训练的基础资源，我们提出了一种生成数据集的框架，该数据集将抽象的、低级别的基因组数据问题与相应的可视化图像配对。在此前统计图工作的基础上，我们的方法适应了基因组数据的复杂性和专门的表示方法。我们进一步将多个链接的问题和可视化图像纳入其中，并为每项数据提供了设计选择的说明、图表标题以及图像替代文本。我们使用从三个不同的基因组数据存储库（4DN、ENCODE、Chromoscope）检索的基因组数据生成了GQVis数据集，该数据集包含114万单查询数据点、62.8万查询对以及58.9万查询链。GQVis数据集及其生成代码可在以下链接获得：this https URL 和 this https URL。 

---
# Reversing the Lens: Using Explainable AI to Understand Human Expertise 

**Title (ZH)**: 反转视角：运用可解释的人工智能理解人类专长 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra, Wan-Lin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13814)  

**Abstract**: Both humans and machine learning models learn from experience, particularly in safety- and reliability-critical domains. While psychology seeks to understand human cognition, the field of Explainable AI (XAI) develops methods to interpret machine learning models. This study bridges these domains by applying computational tools from XAI to analyze human learning. We modeled human behavior during a complex real-world task -- tuning a particle accelerator -- by constructing graphs of operator subtasks. Applying techniques such as community detection and hierarchical clustering to archival operator data, we reveal how operators decompose the problem into simpler components and how these problem-solving structures evolve with expertise. Our findings illuminate how humans develop efficient strategies in the absence of globally optimal solutions, and demonstrate the utility of XAI-based methods for quantitatively studying human cognition. 

**Abstract (ZH)**: 人类和机器学习模型均通过经验学习，尤其在安全和可靠性关键领域。心理学旨在理解人类认知，而可解释人工智能（XAI）领域则发展方法以解释机器学习模型。本研究通过将XAI的计算工具应用到人类学习分析中，将这两个领域结合起来。我们通过构建操作子任务图来模拟复杂实际任务（调谐粒子加速器）中的人类行为。通过对存档的操作员数据应用社区检测和层次聚类等技术，揭示了操作员如何将问题分解为更简单的组件，以及这些解决问题的结构如何随着专业知识的提高而演变。我们的研究结果阐明了在不存在全局最优解的情况下，人类如何发展高效的策略，并展示了基于XAI的方法在定量研究人类认知方面的实用性。 

---
# Generative AI in Heritage Practice: Improving the Accessibility of Heritage Guidance 

**Title (ZH)**: Heritage实践中的生成式AI：提高遗产指导的可达性 

**Authors**: Jessica Witte, Edmund Lee, Lisa Brausem, Verity Shillabeer, Chiara Bonacchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13811)  

**Abstract**: This paper discusses the potential for integrating Generative Artificial Intelligence (GenAI) into professional heritage practice with the aim of enhancing the accessibility of public-facing guidance documents. We developed HAZEL, a GenAI chatbot fine-tuned to assist with revising written guidance relating to heritage conservation and interpretation. Using quantitative assessments, we compare HAZEL's performance to that of ChatGPT (GPT-4) in a series of tasks related to the guidance writing process. The results of this comparison indicate a slightly better performance of HAZEL over ChatGPT, suggesting that the GenAI chatbot is more effective once the underlying large language model (LLM) has been fine-tuned. However, we also note significant limitations, particularly in areas requiring cultural sensitivity and more advanced technical expertise. These findings suggest that, while GenAI cannot replace human heritage professionals in technical authoring tasks, its potential to automate and expedite certain aspects of guidance writing could offer valuable benefits to heritage organisations, especially in resource-constrained contexts. 

**Abstract (ZH)**: 本文讨论了将生成型人工智能（GenAI）整合到专业文化遗产实践中的潜力，旨在提高公共面向指导文件的可访问性。我们开发了HAZEL，这是一种专门用于修订文化遗产保护和解释相关写作指导的GenAI聊天机器人。通过定量评估，我们将HAZEL的表现与其竞争对手ChatGPT（GPT-4）在一系列与指导文件写作过程相关的任务中进行比较。比较结果表明HAZEL的表现略好于ChatGPT，这表明在底层大规模语言模型（LLM）进行微调后，GenAI聊天机器人更为有效。然而，我们也注意到了显著的局限性，特别是在需要文化敏感性和更高级技术专长的领域。这些发现表明，虽然GenAI不能替代文化遗产专业人士在技术写作方面的任务，但它在自动化和加速指导文件写作某些方面的潜力可以为文化遗产组织提供有价值的益处，尤其是在资源受限的环境下。 

---
