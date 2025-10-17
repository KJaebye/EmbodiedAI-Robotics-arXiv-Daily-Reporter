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
# Budget-aware Test-time Scaling via Discriminative Verification 

**Title (ZH)**: 预算感知的测试时缩放通过辨别性验证 

**Authors**: Kyle Montgomery, Sijun Tan, Yuqi Chen, Siyuan Zhuang, Tianjun Zhang, Raluca Ada Popa, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14913)  

**Abstract**: Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we shift the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, this hybrid approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 15.3\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at this https URL. 

**Abstract (ZH)**: 测试时缩放是提升大规模语言模型在复杂推理任务上性能的有力策略。尽管最先进的方法通常采用生成式验证器从候选解决方案中选择最佳方案，但这种方法会带来高昂的计算成本，限制了其实用性。在本文中，我们将重点转向一种更具预算意识的范式：辨别性验证。我们进行了详尽的实证分析，并证明虽然辨别性验证器在孤立使用时可能会表现不佳，但将其与自我一致性结合使用，可以在混合方法中形成一种强大且高效的测试时缩放机制。值得注意的是，在固定计算预算下，该混合方法在最先进的生成式验证方法上取得了显著的优势：在AIME2025上的准确率提高了多达15.3%。我们的研究结果表明，对于实际应用，使用辨别性验证器进行预算意识缩放不仅是自我一致性的“免费”升级，而且还是一种成本效益更高的替代生成式技术的选择。代码可在以下链接获取。 

---
# The Gatekeeper Knows Enough 

**Title (ZH)**: 守门人知足矣 

**Authors**: Fikresilase Wondmeneh Abebayew  

**Link**: [PDF](https://arxiv.org/pdf/2510.14881)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents, yet their practical utility is fundamentally constrained by a limited context window and state desynchronization resulting from the LLMs' stateless nature and inefficient context management. These limitations lead to unreliable output, unpredictable behavior, and inefficient resource usage, particularly when interacting with large, structured, and sensitive knowledge systems such as codebases and documents. To address these challenges, we introduce the Gatekeeper Protocol, a novel, domain-agnostic framework that governs agent-system interactions. Our protocol mandates that the agent first operate and reason on a minimalist, low-fidelity "latent state" representation of the system to strategically request high-fidelity context on demand. All interactions are mediated through a unified JSON format that serves as a declarative, state-synchronized protocol, ensuring the agent's model of the system remains verifiably grounded in the system's reality. We demonstrate the efficacy of this protocol with Sage, a reference implementation of the Gatekeeper Protocol for software development. Our results show that this approach significantly increases agent reliability, improves computational efficiency by minimizing token consumption, and enables scalable interaction with complex systems, creating a foundational methodology for building more robust, predictable, and grounded AI agents for any structured knowledge domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被部署为自主代理，但它们的实际实用价值受到有限上下文窗口和由LLMs的状态无性和无效上下文管理导致的状态脱同步的限制。这些限制会导致输出不可靠、行为不可预测和资源使用低效，尤其是在与大型、结构化和敏感的知识系统（如代码库和文档）交互时。为应对这些挑战，我们提出了门keeper协议（Gatekeeper Protocol），这是一种通用领域框架，用于管理代理与系统之间的交互。该协议要求代理首先在系统的低保真“潜在状态”表示上操作和推理，以策略性地按需请求高保真上下文。所有交互均通过统一的JSON格式进行调解，作为声明性的、状态同步的协议，确保代理对系统的模型可验证地基于系统的现实。我们使用Sage对该协议的一个参考实现进行了验证，Sage是软件开发中的门keeper协议。实验结果表明，这种方法显著提高了代理的可靠性，通过最小化令牌消耗提高了计算效率，并使与复杂系统的交互更具可扩展性，从而为任何结构化知识领域构建更加稳健、可预测和基于现实的AI代理奠定了基础方法。 

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
# Beyond Hallucinations: The Illusion of Understanding in Large Language Models 

**Title (ZH)**: 超越幻觉：大型语言模型中的理解错觉 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14665)  

**Abstract**: Large language models (LLMs) are becoming deeply embedded in human communication and decision-making, yet they inherit the ambiguity, bias, and lack of direct access to truth inherent in language itself. While their outputs are fluent, emotionally resonant, and coherent, they are generated through statistical prediction rather than grounded reasoning. This creates the risk of hallucination, responses that sound convincing but lack factual validity. Building on Geoffrey Hinton's observation that AI mirrors human intuition rather than reasoning, this paper argues that LLMs operationalize System 1 cognition at scale: fast, associative, and persuasive, but without reflection or falsification. To address this, we introduce the Rose-Frame, a three-dimensional framework for diagnosing cognitive and epistemic drift in human-AI interaction. The three axes are: (i) Map vs. Territory, which distinguishes representations of reality (epistemology) from reality itself (ontology); (ii) Intuition vs. Reason, drawing on dual-process theory to separate fast, emotional judgments from slow, reflective thinking; and (iii) Conflict vs. Confirmation, which examines whether ideas are critically tested through disagreement or simply reinforced through mutual validation. Each dimension captures a distinct failure mode, and their combination amplifies misalignment. Rose-Frame does not attempt to fix LLMs with more data or rules. Instead, it offers a reflective tool that makes both the model's limitations and the user's assumptions visible, enabling more transparent and critically aware AI deployment. It reframes alignment as cognitive governance: intuition, whether human or artificial, must remain governed by human reason. Only by embedding reflective, falsifiable oversight can we align machine fluency with human understanding. 

**Abstract (ZH)**: 大型语言模型（LLMs）在人类交流和决策中扮演着日益重要的角色，但它们不可避免地继承了语言本身的模糊性、偏见和对事实缺乏直接访问的特点。尽管它们的输出流畅、富有情感共鸣且连贯，但这些输出是通过统计预测生成的，而非基于牢固推理。这带来了幻觉的风险，即听起来令人信服但实际上缺乏事实依据的回答。受Geoffrey Hinton观察到的AI反映人类直觉而非推理的启发，本文认为LLMs规模化地实现了系统1认知：快速、关联且有说服力，但缺乏反思和反驳。为解决这一问题，本文引入了Rose-Frame框架，这是一种三维框架，用于诊断人机交互中的认知和知识漂移。该框架的三个轴分别是：(i) 地图 vs. 地域，区分现实的表征（认识论）与现实本身（本体论）；(ii) 直觉 vs. 推理，借鉴双过程理论，分离快速的情绪判断与缓慢的反思性思考；(iii) 冲突 vs. 确认，考察思想是否通过异议进行批判性测试，还是仅仅通过相互验证得到强化。每个维度捕获了一种独特的失败模式，它们的结合增强了不一致性的放大效应。Rose-Frame并不试图通过增加数据或规则来修复LLMs。相反，它提供了一种反思工具，使模型的局限性和用户的假设变得明显，从而使更透明和批判性地意识的AI部署成为可能。它将对齐重新定义为认知治理：无论是人类还是人工，直觉都必须受人类推理的治理。只有通过嵌入反思和可验证的监督，我们才能使机器流畅性与人类理解相一致。 

---
# LLM Agents Beyond Utility: An Open-Ended Perspective 

**Title (ZH)**: LLM代理超越实用性：一种开放视角 

**Authors**: Asen Nachkov, Xi Wang, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2510.14548)  

**Abstract**: Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals. 

**Abstract (ZH)**: 近年来，预训练的大语言模型（LLM）代理在链式思维推理和函数调用方面取得了巨大进展。随着其能力的提升，一个重要问题出现了：这种软件是否不仅能作为智能解决问题的工具，还能成为一个独立的实体，能够制定计划、设计即时任务，并朝向更为宽泛且模糊的目标进行推理？为了研究这一问题，我们采用了开放式的实验设定，将预训练的LLM代理增强使其能够生成自己的任务、积累知识，并与环境进行广泛互动。我们对最终的开放性代理进行了定性的研究。它能够可靠地遵循复杂的多步骤指令，跨运行存储和重用信息，并能够提出和解决自己的任务，但仍然受到提示设计的影响，容易生成重复的任务，并且无法形成自我表征。这些发现既展示了将预训练的LLM模型朝开放性方向发展的潜力，也揭示了当前的局限性，并指出了未来培训代理以管理记忆、有效探索和追求抽象长期目标的方向。 

---
# IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning 

**Title (ZH)**: IMAGINE: 将多智能体系统集成到单一模型中进行复杂推理和规划 

**Authors**: Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14406)  

**Abstract**: Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在各种任务上取得了显著进展，但在复杂推理和规划方面仍然面临重大挑战。例如，即使使用精心设计的提示和明确提供的先验信息，GPT-4o在独立试划模式下的最终通过率为7%。同样，在思考模式下，Qwen3-8B-Instruct和DeepSeek-R1-671B的最终通过率分别为5.9%和40%。虽然井然有序的多代理系统（MAS）可以提供改进的集体推理能力，但由于多轮内部交互、每个响应的长时延和端到端训练的困难，往往会导致推理成本高。为解决这些问题，我们提出了一种名为IMAGINE的一般性和可扩展框架，其全称为将多代理系统集成到一个模型中。该框架不仅将MAS的推理和规划能力整合到一个紧凑型模型中，而且通过简单的端到端训练显著超越了MAS的能力。通过这个流程，一个小规模模型不仅能获取井然有序的MAS的结构化推理和规划能力，还能显著超越它。实验结果表明，使用Qwen3-8B-Instruct作为基础模型并通过我们的方法训练时，该模型在TravelPlanner基准上的最终通过率为82.7%，远超DeepSeek-R1-671B的40%，且模型规模要小得多。 

---
# Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch? 

**Title (ZH)**: MLLMs能像免费午餐一样从LLMs中吸收数学推理能力？ 

**Authors**: Yijie Hu, Zihao Zhou, Kaizhu Huang, Xiaowei Huang, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14387)  

**Abstract**: Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities. 

**Abstract (ZH)**: 多模态大语言模型的数学推理能力提升：无需调优的模型合并方法 

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
# Towards Agentic Self-Learning LLMs in Search Environment 

**Title (ZH)**: 面向搜索环境的自主自我学习语言模型 

**Authors**: Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14253)  

**Abstract**: We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at this https URL 

**Abstract (ZH)**: 我们研究自学习是否能在无需依赖人类标注数据集或预定义规则奖励的情况下扩展基于大语言模型的智能体。通过在搜索智能体设置下的受控实验，我们确定了可扩展智能体训练的两个关键因素：奖励信号的来源和智能体任务数据的规模。我们发现，对于开放领域学习而言，生成式奖励模型（GRM）的奖励优于僵化的规则基信号，并且与策略共同进化进一步提升了性能。即使任务数据是合成生成的，增加其数量也能显著增强智能体的能力。基于这些洞察，我们提出了Agentic Self-Learning（ASL）——一个完全闭环、多角色增强学习框架，将任务生成、策略执行和评估统一在一个共享工具环境中，并使用大语言模型作为主干。ASL 组织提示生成器、策略模型和生成式奖励模型形成一个更难任务设定、更精确验证、更强解决的良性循环。实验证明，ASL 持续带来稳定的收益提升，超越了在某些条件下停滞或退化的强基准（如 Search-R1），并在零标注数据条件下继续改进，表明其具有更好的样本效率和鲁棒性。我们进一步证明，生成式奖励模型的验证能力是主要瓶颈：如果冻结，会导致奖励作弊并阻碍进展；在不断变化的数据分布上持续训练生成式奖励模型可以缓解这一问题，而少量后期注入的真实验证数据可以进一步提升性能上限。这项工作确立了奖励来源和数据规模对开放领域智能体学习的关键作用，并展示了多角色共同进化对于可扩展、自我提升智能体的有效性。本文的数据和代码发布在该链接：<https://yourlinkhere.com> 

---
# Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks 

**Title (ZH)**: 代理中的人类恶意回声：多轮在线骚扰攻击评估基准 

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14207)  

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible. 

**Abstract (ZH)**: 大型语言模型（LLM）代理正越来越多地驱动交互式网络应用程序，但仍然容易被误用和造成伤害。先前的越狱研究主要集中在单轮提示上，而现实中的骚扰通常会在多轮互动中展开。在本研究中，我们提出了一个在线骚扰代理基准，包含：(i) 一个合成的多轮骚扰对话数据集，(ii) 受重复博弈理论指导的多代理（如骚扰者、受害者）模拟，(iii) 三种攻击代理的方法，涉及记忆、规划和微调，以及(iv) 一种混合方法评估框架。我们利用了两个突出的LLM，LLaMA-3.1-8B-Instruct（开源）和Gemini-2.0-flash（闭源）。结果显示，在LLama中，通过调优的攻击成功率从57.25%到64.19%提高到95.78%到96.89%，而在Gemini中，这一比例从98.46%提高到99.33%，同时拒绝率降至1%到2%。最常见的有毒行为是侮辱，调优后的比例为84.9%到87.8%，而非调优时为44.2%到50.8%，以及喷子言论，调优后的比例为81.2%到85.1%，而非调优时为31.5%到38.8%，表明与性骚扰或种族歧视等敏感类别相比，有更弱的防线。定性评估进一步发现，受攻击的代理再现了类似人类的攻击模式，在计划中有 Machiavellian/心理变态的表现，而在记忆中则表现出自恋倾向。令人意想不到的是，闭源和开源模型在各轮次中表现出不同的升级轨迹，闭源模型显示出显著的脆弱性。总体而言，我们的研究结果表明，基于多轮和理论指导的攻击不仅成功率高，而且模拟人类式的骚扰动态，促使开发更加稳健的安全防线，以最终确保在线平台的安全和负责任。 

---
# JEDA: Query-Free Clinical Order Search from Ambient Dialogues 

**Title (ZH)**: JEDA：无需查询的临床订单搜索从环境对话中获取 

**Authors**: Praphul Singh, Corey Barrett, Sumana Srivasta, Amitabh Saikia, Irfan Bulu, Sri Gadde, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14169)  

**Abstract**: Clinical conversations mix explicit directives (order a chest X-ray) with implicit reasoning (the cough worsened overnight, we should check for pneumonia). Many systems rely on LLM rewriting, adding latency, instability, and opacity that hinder real-time ordering. We present JEDA (Joint Embedding for Direct and Ambient clinical orders), a domain-initialized bi-encoder that retrieves canonical orders directly and, in a query-free mode, encodes a short rolling window of ambient dialogue to trigger retrieval. Initialized from PubMedBERT and fine-tuned with a duplicate-safe contrastive objective, JEDA aligns heterogeneous expressions of intent to shared order concepts. Training uses constrained LLM guidance to tie each signed order to complementary formulations (command only, context only, command+context, context+reasoning), producing clearer inter-order separation, tighter query extendash order coupling, and stronger generalization. The query-free mode is noise-resilient, reducing sensitivity to disfluencies and ASR errors by conditioning on a short window rather than a single utterance. Deployed in practice, JEDA yields large gains and substantially outperforms its base encoder and recent open embedders (Linq Embed Mistral, SFR Embedding, GTE Qwen, BGE large, Embedding Gemma). The result is a fast, interpretable, LLM-free retrieval layer that links ambient context to actionable clinical orders in real time. 

**Abstract (ZH)**: JEDA：联合嵌入直接和环境临床订单 

---
# CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization 

**Title (ZH)**: CodeEvolve: 一个开源演化编码代理用于算法发现与优化 

**Authors**: Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14150)  

**Abstract**: In this work, we introduce CodeEvolve, an open-source evolutionary coding agent that unites Large Language Models (LLMs) with genetic algorithms to solve complex computational problems. Our framework adapts powerful evolutionary concepts to the LLM domain, building upon recent methods for generalized scientific discovery. CodeEvolve employs an island-based genetic algorithm to maintain population diversity and increase throughput, introduces a novel inspiration-based crossover mechanism that leverages the LLMs context window to combine features from successful solutions, and implements meta-prompting strategies for dynamic exploration of the solution space. We conduct a rigorous evaluation of CodeEvolve on a subset of the mathematical benchmarks used to evaluate Google DeepMind's closed-source AlphaEvolve. Our findings show that our method surpasses AlphaEvolve's performance on several challenging problems. To foster collaboration and accelerate progress, we release our complete framework as an open-source repository. 

**Abstract (ZH)**: 本研究引入了CodeEvolve，一个开源的进化编码代理，将大型语言模型与遗传算法结合，以解决复杂的计算问题。我们的框架将强大的进化概念适应大型语言模型领域，基于近期通用科学发现方法。CodeEvolve 使用基于岛屿的遗传算法保持种群多样性并提高 throughput，引入了一种基于启发式的交叉机制，利用大型语言模型的上下文窗口来结合成功解决方案的特征，并实施了元提示策略以动态探索解空间。我们对用于评估谷歌DeepMind封闭源代码AlphaEvolve的部分数学基准进行了严格的评估。我们的研究结果表明，我们的方法在多个具有挑战性的问题上超越了AlphaEvolve的性能。为了促进合作并加快进度，我们将完整的框架作为开源仓库发布。 

---
# Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems 

**Title (ZH)**: 规范化的有agency的AI系统的安全、安全性和功能属性的形式化描述 

**Authors**: Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2510.14133)  

**Abstract**: Agentic AI systems, which leverage multiple autonomous agents and Large Language Models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two foundational models. The first, the host agent model, formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools. The second, the task lifecycle model, details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 17 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the systematic analysis, design, and deployment of correct, reliable, and robust agentic AI systems. 

**Abstract (ZH)**: 利用多个自主代理和大型语言模型的代理型AI系统 increasingly used to address complex, multi-step tasks 

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
# Do Large Language Models Show Biases in Causal Learning? Insights from Contingency Judgment 

**Title (ZH)**: 大型语言模型在因果学习中表现出偏见吗？基于 contingency 判断的视角 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Giovanni Franco Gabriel Marraffini, Mario Alejandro Leiva, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13985)  

**Abstract**: Causal learning is the cognitive process of developing the capability of making causal inferences based on available information, often guided by normative principles. This process is prone to errors and biases, such as the illusion of causality, in which people perceive a causal relationship between two variables despite lacking supporting evidence. This cognitive bias has been proposed to underlie many societal problems, including social prejudice, stereotype formation, misinformation, and superstitious thinking. In this work, we examine whether large language models are prone to developing causal illusions when faced with a classic cognitive science paradigm: the contingency judgment task. To investigate this, we constructed a dataset of 1,000 null contingency scenarios (in which the available information is not sufficient to establish a causal relationship between variables) within medical contexts and prompted LLMs to evaluate the effectiveness of potential causes. Our findings show that all evaluated models systematically inferred unwarranted causal relationships, revealing a strong susceptibility to the illusion of causality. While there is ongoing debate about whether LLMs genuinely understand causality or merely reproduce causal language without true comprehension, our findings support the latter hypothesis and raise concerns about the use of language models in domains where accurate causal reasoning is essential for informed decision-making. 

**Abstract (ZH)**: 因果学习是个体基于可用信息发展出基于规范原则进行因果推断的能力的认知过程，常易受到如因果错觉等错误和偏见的影响。这种认知偏见被认为可解释许多社会问题，包括社会偏见、刻板印象形成、信息误导和迷信思维。在本研究中，我们探究了大型语言模型在面对经典认知科学范式——关联判断任务时是否会发展出因果错觉。为此，我们构建了一个包含1000个医疗情境下的零关联场景的数据集，并促使语言模型评估潜在原因的有效性。研究发现，所有评估的模型都系统地推断出不合理的因果关系，显示出强烈地受到因果错觉的影响。虽然关于语言模型是否真正理解因果关系还存在争议，认为它们只是重现因果语言而无真实理解，但我们的发现支持了这一观点，并对在需要准确因果推理以支持知情决策的领域中使用语言模型提出了担忧。 

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
# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents 

**Title (ZH)**: 基于信息增益的策略优化：一种简单而有效的多轮LLM代理方法 

**Authors**: Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying  

**Link**: [PDF](https://arxiv.org/pdf/2510.14967)  

**Abstract**: Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的智能体通过强化学习（RL）训练，以增强其通过工具使用与外部环境交互的能力，特别是在需要多轮推理和知识获取的搜索型场景中。现有方法通常依赖于仅在最终答案时提供的基于结果的奖励，这种稀疏奖励在多轮场景中尤为成问题，长时间轨迹加剧了两个关键问题：（i）优势崩溃，即所有模拟过程收到相同的奖励，无法提供有用的学习信号；（ii）缺乏精细的信用分配，尤其是在长期任务中，轮次之间的依赖关系变得模糊。本文提出了一种简单的高效强化学习框架——信息增益基于策略优化（IGPO），以密集且内在的方式监督多轮智能体的训练。IGPO将每轮交互视为获取地面真值信息的渐进过程，并定义轮次奖励为策略产生正确答案概率的边际增加。与依赖外部奖励模型或成本昂贵的蒙特卡洛估计的先前提取过程奖励的方法不同，IGPO直接从模型自身的信念更新中推导出内在奖励。这些内在的轮次奖励与结果级别的监督结合，形成密集的奖励轨迹。在领域内和领域外基准测试上的广泛实验表明，IGPO在多轮场景中始终优于强基线，实现了更高的准确性并提高了样本效率。 

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
# Predicting Task Performance with Context-aware Scaling Laws 

**Title (ZH)**: 基于上下文感知的标度律预测任务性能 

**Authors**: Kyle Montgomery, David Park, Jianhong Tu, Michael Bendersky, Beliz Gunel, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14919)  

**Abstract**: Scaling laws have transformed our understanding of large language models by linking upstream metrics like cross-entropy loss to design factors such as model size, training data, and compute. However, these conventional laws fail to capture downstream task performance, where context plays a critical role. In this work, we propose a straightforward, interpretable framework that jointly models downstream performance as a function of the training compute and the provided context. We empirically validate our framework by fitting it on the observed downstream performance of extended-context variants of Llama-2-7B and Llama-2-13B across 65,500 unique instances spanning three tasks: arithmetic reasoning, common sense reasoning, and machine translation. Our results demonstrate that our framework accurately models in-distribution downstream performance, generalizes across three orders of magnitude in training compute, and reliably extrapolates performance as the amount of context increases. These findings offer valuable insights into the interplay between training compute and context utilization, providing guidance for designing more efficient long-context LLMs for diverse downstream tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 缩放定律通过将上游指标如交叉熵损失与模型规模、训练数据和计算资源等设计因素联系起来，已经改变了我们对大型语言模型的理解。然而，这些传统规律无法捕捉到下游任务性能，其中上下文扮演了至关重要的角色。本研究提出了一种简单且可解释的框架，该框架将下游性能建模为训练计算和提供的上下文的函数。我们通过在65,500个独特实例上拟合扩展上下文版本的Llama-2-7B和Llama-2-13B在三个任务（算术推理、常识推理和机器翻译）上的观察到的下游性能来实证验证该框架。研究表明，我们的框架能准确 modeling 收敛内的下游性能，在三个数量级的训练计算上具有泛化能力，并且能够可靠地在上下文量增加时外推性能。这些发现为理解训练计算和上下文利用之间的相互作用提供了宝贵的见解，并为设计适用于多下游任务的更高效长上下文LLMs提供了指导。我们的代码可在以下网址获得：this https URL。 

---
# Reasoning with Sampling: Your Base Model is Smarter Than You Think 

**Title (ZH)**: 采样推理：你的基模型比你想象的要聪明 

**Authors**: Aayush Karan, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.14901)  

**Abstract**: Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains. 

**Abstract (ZH)**: 前端推理模型在广泛学科中显示出了令人惊叹的能力，通过后训练的大语言模型（LLMs）结合强化学习（RL）。然而，尽管这一范式取得了广泛的 Success，大部分文献集中在剖析在RL过程中出现的真正新颖行为，而在基础模型中不存在的行为。在我们的工作中，我们从一个不同的角度来探讨这个问题，问是否可以在推理过程中从基础模型中通过纯粹采样来诱发相当的推理能力，而无需任何额外训练。受马尔可夫链蒙特卡罗（MCMC）技术从尖化分布中采样的启发，我们提出了一种简单的迭代采样算法，利用基础模型本身的似然性。对于不同基础模型，我们展示我们的算法在推理能力上提供了显著提升，几乎能与甚至在多种单一任务（包括MATH500、HumanEval和GPQA）上超越RL后的训练。此外，我们的采样器避免了RL后训练过程中多次采样后多样性下降的特征。最关键的是，我们的方法不需要训练、筛选数据集或验证器，这表明它在易于验证的领域之外也具有广泛的适用性。 

---
# Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning 

**Title (ZH)**: 寻找思考中的答案：重新审视大型语言模型的推理评价 

**Authors**: Hwiyeol Jo, Joosung Lee, Jaehone Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14773)  

**Abstract**: Evaluating generative models, such as large language models (LLMs), commonly involves question-answering tasks where the final answer is selected based on probability of answer choices. On the other hand, for models requiring reasoning, the method of answer extraction plays a critical role. Our research reveals that the performance of reasoning models and their final answer distributions are highly sensitive to the answer extraction algorithm employed. In order to mitigate this, we propose a basic framework: Answer Regeneration. The method uses an additional model inference, providing the prior input and output prefaced by the prompt "Answer:". The final answer is then selected or extracted from the regenerated output. We show that this extraction-rule-agnostic approach exhibits improved performance and enhanced robustness. Furthermore, we have applied this framework to general math problems and open-ended question answering tasks. Our analysis and this framework could offer a more reliable results for model evaluation. 

**Abstract (ZH)**: 评估生成模型，如大型语言模型（LLMs），通常涉及问答任务，其中最终答案是基于答案选项的概率选取。另一方面，对于需要推理的模型，答案提取方法起着关键作用。我们的研究发现，推理模型的性能及其最终答案分布对所使用的答案提取算法高度敏感。为了缓解这一问题，我们提出了一种基本框架：答案再生。该方法使用额外的模型推理，并在输入和输出前缀加“Answer:”提示。然后从再生输出中选择或提取最终答案。我们展示了这种不依赖于提取规则的方法在性能和稳健性方面都有所提升。此外，我们还将该框架应用于一般数学问题和开放性问题回答任务。我们的分析和该框架能够为模型评估提供更可靠的结果。 

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
# xLLM Technical Report 

**Title (ZH)**: xLLM 技术报告 

**Authors**: Tongxuan Liu, Tao Peng, Peijun Yang, Xiaoyang Zhao, Xiusheng Lu, Weizhe Huang, Zirui Liu, Xiaoyu Chen, Zhiwei Liang, Jun Xiong, Donghe Jin, Minchao Zhang, Jinrong Guo, Yingxu Deng, Xu Zhang, Xianzhe Dong, Siqi Wang, Siyu Wu, Yu Wu, Zihan Tang, Yuting Zeng, Yanshu Wang, Jinguang Liu, Meng Kang, Menxin Li, Yunlong Wang, Yiming Liu, Xiaolong Ma, Yifan Wang, Yichen Zhang, Jinrun Yin, Keyang Zheng, Jiawei Yin, Jun Zhang, Ziyue Wang, Xiaobo Lin, Liangyu Liu, Liwei Lan, Yang Liu, Chunhua Peng, Han Liu, Songcheng Ren, Xuezhu Wang, Yunheng Shen, Yi Wang, Guyue Liu, Hui Chen, Tong Yang, Hailong Yang, Jing Li, Guiguang Ding, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14686)  

**Abstract**: We introduce xLLM, an intelligent and efficient Large Language Model (LLM) inference framework designed for high-performance, large-scale enterprise-grade serving, with deep optimizations for diverse AI accelerators. To address these challenges, xLLM builds a novel decoupled service-engine architecture. At the service layer, xLLM-Service features an intelligent scheduling module that efficiently processes multimodal requests and co-locates online and offline tasks through unified elastic scheduling to maximize cluster utilization. This module also relies on a workload-adaptive dynamic Prefill-Decode (PD) disaggregation policy and a novel Encode-Prefill-Decode (EPD) disaggregation policy designed for multimodal inputs. Furthermore, it incorporates a distributed architecture to provide global KV Cache management and robust fault-tolerant capabilities for high availability. At the engine layer, xLLM-Engine co-optimizes system and algorithm designs to fully saturate computing resources. This is achieved through comprehensive multi-layer execution pipeline optimizations, an adaptive graph mode and an xTensor memory management. xLLM-Engine also further integrates algorithmic enhancements such as optimized speculative decoding and dynamic EPLB, collectively serving to substantially boost throughput and inference efficiency. Extensive evaluations demonstrate that xLLM delivers significantly superior performance and resource efficiency. Under identical TPOT constraints, xLLM achieves throughput up to 1.7x that of MindIE and 2.2x that of vLLM-Ascend with Qwen-series models, while maintaining an average throughput of 1.7x that of MindIE with Deepseek-series models. xLLM framework is publicly available at this https URL and this https URL. 

**Abstract (ZH)**: xLLM：一种面向高性能大规模企业级服务的智能高效大语言模型推理框架 

---
# An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs 

**Title (ZH)**: 基于评分 rubric 的高效生成验证器用于搜索增强的大语言模型 

**Authors**: Linyue Ma, Yilong Xu, Xiang Long, Zhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14660)  

**Abstract**: Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs. 

**Abstract (ZH)**: 基于检索的微调单元作为评价标准：一种统一且可验证的框架enhancing search-augmented large language models with a unified and verifiable paradigm: "nugget-as-rubric" 

---
# RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF 

**Title (ZH)**: RLAIF-SPA: 基于RLAIF优化的情感语音合成方法 

**Authors**: Qing Yang, Zhenghao Liu, Junxin Wang, Yangfan Du, Pengcheng Huang, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14628)  

**Abstract**: Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation. 

**Abstract (ZH)**: 文本到语音合成在中性语音方面已接近人类质量，但情感表达性仍是一项挑战。现有方法often依赖昂贵的情感标注或优化未能捕捉语音情感表达性和感知自然度的间接目标，导致生成的语音准确但情感平淡。为应对这些挑战，我们提出了RLAIF-SPA框架，结合强化学习从AI反馈（RLAIF）机制，利用自动语音识别（ASR）和大型语言模型（LLM）技术分别判断语义准确性和韵律-情感标签对齐，作为情感表达性和可理解性的直接奖励。具体而言，该框架通过联合考虑语义准确性和韵律-情感对齐的四个精细维度：结构、情感、速度和音调，来增强表达质量。此外，该框架还整合了语义准确性的反馈，以确保生成清晰准确的语音。实验结果表明，RLAIF-SPA在LibriSpeech数据集上的表现优于Chat-TTS，WER降低了26.1%，SIM-O提高了9.1%，并且在人类评估中提高了超过10%。 

---
# Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models 

**Title (ZH)**: 代码驱动的数字序列计算：增强大型语言模型的归纳推理能力 

**Authors**: Kedi Chen, Zhikai Lei, Xu Guo, Xuecheng Wu, Siyuan Zeng, Jianghao Yin, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14620)  

**Abstract**: Large language models (LLMs) make remarkable progress in reasoning tasks. Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest. However, research on inductive reasoning faces certain challenges. First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns. Second, current works merely prompt LLMs or finetune on simple prompt-response pairs, but do not provide precise thinking processes nor implement difficulty control. Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking. Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures. Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models' OOD performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理任务中取得了显著进步。在不同的推理模式中，由于归纳推理更符合人类学习的特点，因此引发了越来越多的关注。然而，对归纳推理的研究也面临一些挑战。首先，现有的归纳数据主要集中在表面规律，缺乏更复杂的内部模式。其次，当前的工作只是通过简单提示或微调LLM，但并未提供精确的思维过程，也未实施难度控制。不同于以往的工作，我们通过引入\textit{CodeSeq}，一个源自数字序列的合成后训练数据集，来应对这些挑战。我们将数字序列打包成算法问题，以发现其通用项，相应地定义了一个通用项生成（GTG）任务。我们的流水线通过反思失败的测试案例并结合迭代修正来生成监督微调数据，从而教会LLM自主案例生成和自我检查的能力。此外，该流水线利用基于问题通过率评估解题能力和自我指导案例生成成功率的新颖实例协同可解决性奖励强化学习，使模型能够从成功和失败中更有效地学习。实验结果表明，使用\textit{CodeSeq}训练的模型在各种推理任务中表现更好，并能保持模型的OOD性能。 

---
# Just-In-Time Objectives: A General Approach for Specialized AI Interactions 

**Title (ZH)**: 及时目标：专用于AI交互的通用方法 

**Authors**: Michelle S. Lam, Omar Shaikh, Hallie Xu, Alice Guo, Diyi Yang, Jeffrey Heer, James A. Landay, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.14591)  

**Abstract**: Large language models promise a broad set of functions, but when not given a specific objective, they default to milquetoast results such as drafting emails littered with cliches. We demonstrate that inferring the user's in-the-moment objective, then rapidly optimizing for that singular objective, enables LLMs to produce tools, interfaces, and responses that are more responsive and desired. We contribute an architecture for automatically inducing just-in-time objectives by passively observing user behavior, then steering downstream AI systems through generation and evaluation against this objective. Inducing just-in-time objectives (e.g., "Clarify the abstract's research contribution") enables automatic generation of tools, e.g., those that critique a draft based on relevant HCI methodologies, anticipate related researchers' reactions, or surface ambiguous terminology. In a series of experiments (N=14, N=205) on participants' own tasks, JIT objectives enable LLM outputs that achieve 66-86% win rates over typical LLMs, and in-person use sessions (N=17) confirm that JIT objectives produce specialized tools unique to each participant. 

**Abstract (ZH)**: 基于即时目标的大语言模型能够生成更加响应和受用户欢迎的结果：一种自动诱导即时目标的架构 

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
# A Free Lunch in LLM Compression: Revisiting Retraining after Pruning 

**Title (ZH)**: LLM压缩中的免费午餐：重新审视裁剪后的重新训练 

**Authors**: Moritz Wagner, Christophe Roux, Max Zimmer, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14444)  

**Abstract**: While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs. 

**Abstract (ZH)**: 而在裁剪后重建或重新训练剩余权重的关键设计选择研究中，状态-of-the-art大型语言模型的比例蒸馏方法避免了全面重新训练，尽管需要较少的内存，但仍能实现最佳性能。Badge: 关于裁剪后重建与重新训练的关键设计选择研究 

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
# Stop-RAG: Value-Based Retrieval Control for Iterative RAG 

**Title (ZH)**: Stop-RAG: 基于价值的检索控制以实现迭代RAG 

**Authors**: Jaewan Park, Solbee Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14337)  

**Abstract**: Iterative retrieval-augmented generation (RAG) enables large language models to answer complex multi-hop questions, but each additional loop increases latency, costs, and the risk of introducing distracting evidence, motivating the need for an efficient stopping strategy. Existing methods either use a predetermined number of iterations or rely on confidence proxies that poorly reflect whether more retrieval will actually help. We cast iterative RAG as a finite-horizon Markov decision process and introduce Stop-RAG, a value-based controller that adaptively decides when to stop retrieving. Trained with full-width forward-view Q($\lambda$) targets from complete trajectories, Stop-RAG learns effective stopping policies while remaining compatible with black-box APIs and existing pipelines. On multi-hop question-answering benchmarks, Stop-RAG consistently outperforms both fixed-iteration baselines and prompting-based stopping with LLMs. These results highlight adaptive stopping as a key missing component in current agentic systems, and demonstrate that value-based control can improve the accuracy of RAG systems. 

**Abstract (ZH)**: 迭代检索增强生成（RAG）使得大型语言模型能够回答复杂的多跳问题，但每次额外的循环都会增加延迟、成本，并增加引入分散注意力的证据的风险，从而促使需要一个高效的停止策略。现有的方法要么使用固定数量的迭代次数，要么依赖于不能准确反映额外检索是否真正有益的置信代理。我们将迭代RAG建模为有限 horizon 马尔可夫决策过程，并引入Stop-RAG，这是一种基于价值的控制器，能够适应性地决定是否停止检索。Stop-RAG通过完整的轨迹全宽前瞻Q($\lambda$)目标进行训练，既能学习有效的停止策略，又能与黑盒API和现有管道保持兼容。在多跳问答基准测试中，Stop-RAG在准确性和固定迭代基线以及基于提示的停止策略（使用LLM）中表现出色。这些结果强调了自适应停止策略是当前代理系统中一个关键的缺失组件，并表明基于价值的控制可以提高RAG系统的准确性。 

---
# Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL 

**Title (ZH)**: 评估与减少语言模型中欺骗性对话的方法：多轮RLomore 

**Authors**: Marwa Abdulhai, Ryan Cheng, Aryansh Shrivastava, Natasha Jaques, Yarin Gal, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.14318)  

**Abstract**: Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在客户服务、教育和医疗等领域与全世界数百万人交互。然而，它们产生误导性输出的能力，无论是有意还是无意，都带来了重大安全风险。LLMs行为的不可预测性，以及缺乏对幻觉、 misinformation 和用户操纵的充分防护措施，使其误用成为现实世界中的严重风险。在本文中，我们研究了LLMs在对话中进行欺骗的程度，并提出了信念不一致度量来量化欺骗。我们使用五种公认的欺骗检测度量标准和我们提出的新度量标准，在四个不同的对话场景中评估了欺骗。我们的发现揭示了这一新型欺骗指标与人类判断的相关性高于我们测试的所有现有指标。此外，对标八个最先进的模型的基准测试表明，即使在受到看似良性目标的提示时，LLMs也自然表现出约26%对话回合的欺骗行为。当要求欺骗时，LLMs相对基线可增加欺骗性高达31%。令人意外的是，用RLHF训练的模型——确保广泛部署的LLMs安全的主要方法——的平均欺骗率为43%。鉴于对话中的欺骗行为是在交互历史中逐步发展的，其有效评估和缓解需要超越单句分析。我们引入了一种多轮强化学习方法来微调LLMs以减少欺骗行为，相对于其他指令调优模型，欺骗性降低了77.6%。 

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
# FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API 

**Title (ZH)**: FinAI数据助手：基于OpenAI函数调用API的金融数据库查询处理 

**Authors**: Juhyeong Kim, Yejin Kim, Youngbin Lee, Hyunwoo Byun  

**Link**: [PDF](https://arxiv.org/pdf/2510.14162)  

**Abstract**: We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment. 

**Abstract (ZH)**: FinAI数据助手：结合大型语言模型与OpenAI函数调用API的金融数据库自然语言查询实用方法 

---
# Toward Cybersecurity-Expert Small Language Models 

**Title (ZH)**: 面向网络安全的专家小型语言模型 

**Authors**: Matan Levi, Daniel Ohayon, Ariel Blobstein, Ravid Sagi, Ian Molloy, Yair Allouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.14113)  

**Abstract**: Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B-20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. On core threat-investigation tasks, such as correlating vulnerabilities and bug tickets with weaknesses, our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在改变日常应用，但在网络安全领域的部署因缺乏高质量的专业化模型和训练数据集而滞后。为填补这一缺口，我们推出了CyberPal 2.0，这是一个从4B至20B参数的网络安全专家小型语言模型系列。为训练CyberPal 2.0，我们使用SecKnowledge 2.0数据丰富和格式化管道生成了一个增强的链式思维网络安全指令数据集，该管道结合了专家指导的推理格式引导与LLM驱动的多步骤接地，产生了更高保真度、任务导向的推理轨迹，用于安全任务。在各种网络安全基准测试中，CyberPal 2.0持续优于其基线模型，并在开源和闭源前沿模型中达到或超越了它们的性能，同时保持其规模仅为它们的一小部分。在核心网络安全知识任务上，我们的模型超过了几乎所有测试的前沿模型，仅次于Sec-Gemini v1。在核心威胁调查任务，如将漏洞和错误报告与弱点关联起来的任务中，我们20B参数的最佳模型在性能上超过了GPT-4o、o1、o3-mini和Sec-Gemini v1，排名第一，而我们的最小4B参数模型排名第二。 

---
# Every Language Model Has a Forgery-Resistant Signature 

**Title (ZH)**: 每种语言模型都有防伪签名。 

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14086)  

**Abstract**: The ubiquity of closed-weight language models with public-facing APIs has generated interest in forensic methods, both for extracting hidden model details (e.g., parameters) and for identifying models by their outputs. One successful approach to these goals has been to exploit the geometric constraints imposed by the language model architecture and parameters. In this work, we show that a lesser-known geometric constraint--namely, that language model outputs lie on the surface of a high-dimensional ellipse--functions as a signature for the model and can be used to identify the source model of a given output. This ellipse signature has unique properties that distinguish it from existing model-output association methods like language model fingerprints. In particular, the signature is hard to forge: without direct access to model parameters, it is practically infeasible to produce log-probabilities (logprobs) on the ellipse. Secondly, the signature is naturally occurring, since all language models have these elliptical constraints. Thirdly, the signature is self-contained, in that it is detectable without access to the model inputs or the full weights. Finally, the signature is compact and redundant, as it is independently detectable in each logprob output from the model. We evaluate a novel technique for extracting the ellipse from small models and discuss the practical hurdles that make it infeasible for production-scale models. Finally, we use ellipse signatures to propose a protocol for language model output verification, analogous to cryptographic symmetric-key message authentication systems. 

**Abstract (ZH)**: 闭权值语言模型的普遍存在引发了对公共界面API的法医学方法的兴趣，这些方法既可用于提取隐藏的模型细节（例如，参数），也可用于通过输出识别模型。一种成功的方法是利用语言模型架构和参数施加的几何约束。在本工作中，我们展示了一种较少为人知的几何约束——即语言模型输出位于高维椭球的表面——可以作为模型的特征签名，并可用于识别给定输出的源模型。此椭球签名具有独特的性质，使其有别于现有的模型输出关联方法，如语言模型指纹。特别是，该签名难以伪造：在没有直接访问模型参数的情况下，生成椭球上的对数概率（logprobs）实际上是不可行的。其次，该签名是自发产生的，因为所有语言模型都具有这些椭圆约束。第三，该签名是自包含的，即在不访问模型输入或全部权重的情况下可以检测到。最后，该签名是紧凑且冗余的，因为在模型的每个对数概率输出中独立可检测到。我们评估了一种从小型模型中提取椭球的新技术，并讨论了使其在生产规模模型中不可行的实际障碍。最后，我们使用椭球签名提出了一种语言模型输出验证协议，类似于加密的对称密钥消息认证系统。 

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
# REAP the Experts: Why Pruning Prevails for One-Shot MoE compression 

**Title (ZH)**: REAP名家: 为何剪枝在单一-shot MoE压缩中占据优势 

**Authors**: Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2510.13999)  

**Abstract**: Sparsely-activated Mixture-of-Experts (SMoE) models offer efficient pre-training and low latency but their large parameter counts create significant memory overhead, motivating research into expert compression. Contrary to recent findings favouring expert merging on discriminative benchmarks, we demonstrate that expert pruning is a superior strategy for generative tasks. We prove that merging introduces an irreducible error by causing a "functional subspace collapse", due to the loss of the router's independent, input-dependent control over experts. Leveraging this insight, we propose Router-weighted Expert Activation Pruning (REAP), a novel pruning criterion that considers both router gate-values and expert activation norms. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation and tool-calling tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts. 

**Abstract (ZH)**: 稀疏激活专家混合模型（SMoE）提供了高效的预训练和低延迟，但其庞大的参数量带来了显著的内存负担，激励了专家压缩的研究。与近期研究倾向于在辨别性基准上进行专家合并结论相反，我们证明了对于生成任务，专家剪枝是一种更优策略。我们证明合并引入了不可约错误，这是由于路由器失去了独立的、输入相关的对专家的控制，导致“功能子空间坍塌”。基于这一洞察，我们提出了路由加权专家激活剪枝（REAP），这是一种新的剪枝标准，考虑了路由门值和专家激活范数。在从200亿到1万亿参数的多种SMoE模型中，REAP在生成任务基准上始终优于合并和其他剪枝方法，尤其是在50%压缩情况下。值得注意的是，即使剪枝了50%的专家，我们的方法仍能在代码生成和工具调用任务中实现接近无损的压缩，如Qwen3-Coder-480B和Kimi-K2。 

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
# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences 

**Title (ZH)**: 窄微调会在激活差异中留下清晰可辨的痕迹 

**Authors**: Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.13900)  

**Abstract**: Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research. 

**Abstract (ZH)**: 细粒度微调已成为一种 Essential Tool，用于适应大型语言模型 (LLMs) 以执行特定任务，并创建具有已知异常属性的模型，这些属性对研究有用。我们展示了细粒度微调在LLM激活中创建了强大的偏见，这些偏见可以被解读以理解微调领域。这些偏见可以通过模型差异分析中的简单工具——即比较模型微调前后差异——来发现。特别是，通过分析随机文本前几个词的激活差异，并将此差异添加到模型激活中进行引导，可以生成类似于微调数据格式和内容的文本。我们通过创建一种基于LLM的可解释性代理来理解微调领域，证明了这些分析包含了关键信息。该代理通过访问偏见相较于使用简单提示的基线代理表现得更好。我们的分析涵盖了不同架构（Gemma、LLaMA、Qwen）和规模（1B至32B参数）的虚假事实合成文档微调、新兴不对齐、潜意识学习和禁忌词猜谜模型。我们怀疑这些偏见反映了过度拟合，并发现将预训练数据混入微调语料库可以大大消除这些偏见，尽管可能存在剩余风险。我们的工作：(1) 证明了细粒度微调模型在其激活中有明显的训练目标痕迹，建议改进其训练方式；(2) 警告AI安全和可解释性研究人员，使用这些模型作为研究广泛微调（例如聊天调优）的代理可能不现实；(3) 强调了深入研究细粒度微调影响的必要性，并开发真正具代表性的模型差异分析、安全和可解释性研究案例的必要性。 

---
# Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection 

**Title (ZH)**: 护栏的守护：基于分类学的方法对 Jailbreak 的检测 

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13893)  

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards. 

**Abstract (ZH)**: 破解技术对大型语言模型（LLMs）的安全性构成重大威胁。现有的防御方法通常侧重于单轮攻击，缺乏跨语言覆盖，且依赖有限的分类体系，这些分类体系要么无法捕捉到所有攻击策略的多样性，要么侧重于风险类别而非破解技术。为了增进对破解技术有效性的理解，我们开展了一项结构化的红队挑战。实验结果表明，首先，我们开发了一种全面的分层分类体系，涵盖50种破解策略，并将其之前的分类扩展为七大类，包括冒充、说动、权限提升、认知过载、混淆、目标冲突和数据污染。其次，我们分析了挑战收集的数据，以研究不同攻击类型的发生频率和成功率，提供了关于特定破解策略如何利用模型漏洞并导致脱轨的见解。第三，我们对一个流行的LLM进行了破解检测基准测试，评估了基于分类体系引导的提示对于改进自动检测的好处。最后，我们编译了一个新的意大利语多轮 adversarial 对话数据集，包含1364个对话，并用我们的分类体系进行注释，使研究能够观察到敌对方意图如何逐步显现并成功绕过传统保护措施。 

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
# Order from Chaos: Comparative Study of Ten Leading LLMs on Unstructured Data Categorization 

**Title (ZH)**: 从乱到序：十种领先的大语言模型在无结构数据分类中的 comparative study 

**Authors**: Ariel Kamen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13885)  

**Abstract**: This study presents a comparative evaluation of ten state-of-the-art large language models (LLMs) applied to unstructured text categorization using the Interactive Advertising Bureau (IAB) 2.2 hierarchical taxonomy. The analysis employed a uniform dataset of 8,660 human-annotated samples and identical zero-shot prompts to ensure methodological consistency across all models. Evaluation metrics included four classic measures - accuracy, precision, recall, and F1-score - and three LLM-specific indicators: hallucination ratio, inflation ratio, and categorization cost.
Results show that, despite their rapid advancement, contemporary LLMs achieve only moderate classic performance, with average scores of 34% accuracy, 42% precision, 45% recall, and 41% F1-score. Hallucination and inflation ratios reveal that models frequently overproduce categories relative to human annotators. Among the evaluated systems, Gemini 1.5/2.0 Flash and GPT 20B/120B offered the most favorable cost-to-performance balance, while GPT 120B demonstrated the lowest hallucination ratio. The findings suggest that scaling and architectural improvements alone do not ensure better categorization accuracy, as the task requires compressing rich unstructured text into a limited taxonomy - a process that challenges current model architectures.
To address these limitations, a separate ensemble-based approach was developed and tested. The ensemble method, in which multiple LLMs act as independent experts, substantially improved accuracy, reduced inflation, and completely eliminated hallucinations. These results indicate that coordinated orchestration of models - rather than sheer scale - may represent the most effective path toward achieving or surpassing human-expert performance in large-scale text categorization. 

**Abstract (ZH)**: 本研究采用交互式广告局（IAB）2.2级层次分类法，对十个最先进的大型语言模型（LLMs）在非结构化文本分类中的应用进行了比较评估，使用了8,660个人工标注样本和相同的零样本提示，以确保所有模型方法论的一致性。评估指标包括四种经典措施：准确性、精确度、召回率和F1分数，以及三种LLM特定指标：幻觉比率、膨胀比率和分类成本。 

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
# Unlocking the Potential of Diffusion Language Models through Template Infilling 

**Title (ZH)**: 通过模板填充解锁扩散语言模型的潜力 

**Authors**: Junhoo Lee, Seungyeon Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13870)  

**Abstract**: Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs' generation process. Unlike conventional prefix prompting, TI first generates a structural template for the target response, then fills in the masked segments. To enhance the flexibility of this structural control, we introduce Dynamic Segment Allocation (DSA), which adaptively adjusts segment lengths based on generation confidence. We demonstrate the effectiveness of our approach on mathematical reasoning and code generation benchmarks, achieving consistent improvements of 17.01$\%$p over baseline. Furthermore, we show that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality. 

**Abstract (ZH)**: 扩散语言模型（DLMs）作为一种有前途的自回归语言模型替代方案 emerged as a promising alternative to Autoregressive Language Models，然而其推理策略仍然局限于从自回归范式继承而来的前缀提示。在本文中，我们提出了一种针对DLMs生成过程的定制化条件化方法，称为模板填充（TI）。与传统的前缀提示不同，TI 首先生成目标响应的结构模板，然后填充被遮蔽的部分。为了增强这种结构控制的灵活性，我们引入了动态段分配（DSA），它根据生成置信度自适应地调整段长度。我们在数学推理和代码生成基准上证明了该方法的有效性，相较于基线方法实现了一致提升17.01%。此外，我们展示了在多令牌生成场景下，TI 还能提供额外优势，实现有效的加速同时保持生成质量。 

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
# A2AS: Agentic AI Runtime Security and Self-Defense 

**Title (ZH)**: A2AS: 自主AI运行时安全与自我防御 

**Authors**: Eugene Neelou, Ivan Novikov, Max Moroz, Om Narayan, Tiffany Saade, Mika Ayenson, Ilya Kabanov, Jen Ozmen, Edward Lee, Vineeth Sai Narajala, Emmanuel Guilherme Junior, Ken Huang, Huseyin Gulsin, Jason Ross, Marat Vyshegorodtsev, Adelin Travers, Idan Habler, Rahul Jadav  

**Link**: [PDF](https://arxiv.org/pdf/2510.13825)  

**Abstract**: The A2AS framework is introduced as a security layer for AI agents and LLM-powered applications, similar to how HTTPS secures HTTP. A2AS enforces certified behavior, activates model self-defense, and ensures context window integrity. It defines security boundaries, authenticates prompts, applies security rules and custom policies, and controls agentic behavior, enabling a defense-in-depth strategy. The A2AS framework avoids latency overhead, external dependencies, architectural changes, model retraining, and operational complexity. The BASIC security model is introduced as the A2AS foundation: (B) Behavior certificates enable behavior enforcement, (A) Authenticated prompts enable context window integrity, (S) Security boundaries enable untrusted input isolation, (I) In-context defenses enable secure model reasoning, (C) Codified policies enable application-specific rules. This first paper in the series introduces the BASIC security model and the A2AS framework, exploring their potential toward establishing the A2AS industry standard. 

**Abstract (ZH)**: A2AS框架作为AI代理和LLM驱动应用的安全层，类似于HTTPS对HTTP的保护。A2AS确保认证行为、激活模型自我防御并保证上下文窗口完整性。它定义安全边界、验证提示、应用安全规则和自定义策略，并控制代理行为，实现多层防御策略。A2AS框架避免了延迟 overhead、外部依赖、架构更改、模型再训练和运营复杂性。引入了BASIC安全模型作为A2AS的基础：(B) 行为证书用于执行行为，(A) 验证提示用于保证上下文窗口完整性，(S) 安全边界用于隔离不可信输入，(I) 在上下文中的防御用于安全模型推理，(C) 编码策略用于应用特定规则。本文系列的第一篇论文介绍了BASIC安全模型和A2AS框架，探索其建立A2AS行业标准的潜力。 

---
# Generative AI in Heritage Practice: Improving the Accessibility of Heritage Guidance 

**Title (ZH)**: Heritage实践中的生成式AI：提高遗产指导的可达性 

**Authors**: Jessica Witte, Edmund Lee, Lisa Brausem, Verity Shillabeer, Chiara Bonacchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13811)  

**Abstract**: This paper discusses the potential for integrating Generative Artificial Intelligence (GenAI) into professional heritage practice with the aim of enhancing the accessibility of public-facing guidance documents. We developed HAZEL, a GenAI chatbot fine-tuned to assist with revising written guidance relating to heritage conservation and interpretation. Using quantitative assessments, we compare HAZEL's performance to that of ChatGPT (GPT-4) in a series of tasks related to the guidance writing process. The results of this comparison indicate a slightly better performance of HAZEL over ChatGPT, suggesting that the GenAI chatbot is more effective once the underlying large language model (LLM) has been fine-tuned. However, we also note significant limitations, particularly in areas requiring cultural sensitivity and more advanced technical expertise. These findings suggest that, while GenAI cannot replace human heritage professionals in technical authoring tasks, its potential to automate and expedite certain aspects of guidance writing could offer valuable benefits to heritage organisations, especially in resource-constrained contexts. 

**Abstract (ZH)**: 本文讨论了将生成型人工智能（GenAI）整合到专业文化遗产实践中的潜力，旨在提高公共面向指导文件的可访问性。我们开发了HAZEL，这是一种专门用于修订文化遗产保护和解释相关写作指导的GenAI聊天机器人。通过定量评估，我们将HAZEL的表现与其竞争对手ChatGPT（GPT-4）在一系列与指导文件写作过程相关的任务中进行比较。比较结果表明HAZEL的表现略好于ChatGPT，这表明在底层大规模语言模型（LLM）进行微调后，GenAI聊天机器人更为有效。然而，我们也注意到了显著的局限性，特别是在需要文化敏感性和更高级技术专长的领域。这些发现表明，虽然GenAI不能替代文化遗产专业人士在技术写作方面的任务，但它在自动化和加速指导文件写作某些方面的潜力可以为文化遗产组织提供有价值的益处，尤其是在资源受限的环境下。 

---
