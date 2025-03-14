# SCOOP: A Framework for Proactive Collaboration and Social Continual Learning through Natural Language Interaction andCausal Reasoning 

**Title (ZH)**: SCOOP：一种通过自然语言交互和因果推理实现主动协作和社会连续学习的框架 

**Authors**: Dimitri Ognibene, Sabrina Patania, Luca Annese, Cansu Koyuturk, Franca Garzotto, Giuseppe Vizzari, Azzurra Ruggeri, Simone Colombani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10241)  

**Abstract**: Multimodal information-gathering settings, where users collaborate with AI in dynamic environments, are increasingly common. These involve complex processes with textual and multimodal interactions, often requiring additional structural information via cost-incurring requests. AI helpers lack access to users' true goals, beliefs, and preferences and struggle to integrate diverse information effectively.
We propose a social continual learning framework for causal knowledge acquisition and collaborative decision-making. It focuses on autonomous agents learning through dialogues, question-asking, and interaction in open, partially observable environments. A key component is a natural language oracle that answers the agent's queries about environmental mechanisms and states, refining causal understanding while balancing exploration or learning, and exploitation or knowledge use.
Evaluation tasks inspired by developmental psychology emphasize causal reasoning and question-asking skills. They complement benchmarks by assessing the agent's ability to identify knowledge gaps, generate meaningful queries, and incrementally update reasoning. The framework also evaluates how knowledge acquisition costs are amortized across tasks within the same environment.
We propose two architectures: 1) a system combining Large Language Models (LLMs) with the ReAct framework and question-generation, and 2) an advanced system with a causal world model, symbolic, graph-based, or subsymbolic, for reasoning and decision-making. The latter builds a causal knowledge graph for efficient inference and adaptability under constraints. Challenges include integrating causal reasoning into ReAct and optimizing exploration and question-asking in error-prone scenarios. Beyond applications, this framework models developmental processes combining causal reasoning, question generation, and social learning. 

**Abstract (ZH)**: 多模态信息收集场景下的社会连续学习框架：因果知识获取与协作决策 

---
# Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search 

**Title (ZH)**: 围城：基于树搜索的自主多轮大语言模型越狱 

**Authors**: Andy Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10619)  

**Abstract**: We introduce Siege, a multi-turn adversarial framework that models the gradual erosion of Large Language Model (LLM) safety through a tree search perspective. Unlike single-turn jailbreaks that rely on one meticulously engineered prompt, Siege expands the conversation at each turn in a breadth-first fashion, branching out multiple adversarial prompts that exploit partial compliance from previous responses. By tracking these incremental policy leaks and re-injecting them into subsequent queries, Siege reveals how minor concessions can accumulate into fully disallowed outputs. Evaluations on the JailbreakBench dataset show that Siege achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4 in a single multi-turn run, using fewer queries than baselines such as Crescendo or GOAT. This tree search methodology offers an in-depth view of how model safeguards degrade over successive dialogue turns, underscoring the urgency of robust multi-turn testing procedures for language models. 

**Abstract (ZH)**: 我们引入了Siege，一种多轮对抗框架，从树搜索的角度建模大型语言模型（LLM）安全性逐渐下降的过程。 

---
# LLM Agents Display Human Biases but Exhibit Distinct Learning Patterns 

**Title (ZH)**: LLM代理表现出人类偏见但展现出不同的学习模式 

**Authors**: Idan Horowitz, Ori Plonsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.10248)  

**Abstract**: We investigate the choice patterns of Large Language Models (LLMs) in the context of Decisions from Experience tasks that involve repeated choice and learning from feedback, and compare their behavior to human participants. We find that on the aggregate, LLMs appear to display behavioral biases similar to humans: both exhibit underweighting rare events and correlation effects. However, more nuanced analyses of the choice patterns reveal that this happens for very different reasons. LLMs exhibit strong recency biases, unlike humans, who appear to respond in more sophisticated ways. While these different processes may lead to similar behavior on average, choice patterns contingent on recent events differ vastly between the two groups. Specifically, phenomena such as ``surprise triggers change" and the ``wavy recency effect of rare events" are robustly observed in humans, but entirely absent in LLMs. Our findings provide insights into the limitations of using LLMs to simulate and predict humans in learning environments and highlight the need for refined analyses of their behavior when investigating whether they replicate human decision making tendencies. 

**Abstract (ZH)**: 我们探究了在经验决策任务中大型语言模型（LLMs）的决策模式，这些任务涉及重复选择和从反馈中学习，并将其行为与人类参与者的行为进行比较。我们发现总体而言，LLMs 表现出与人类相似的行为偏差：两者都对稀有事件给予不当权重并受到相关性效应的影响。然而，更细致的决策模式分析揭示了这些差异背后的原因有所不同。LLMs 表现出强烈的近期效应偏差，而人类则以更为复杂的方式作出反应。尽管这两种过程在平均表现上可能导致相似的行为，但两种群体基于近期事件的决策模式差异巨大。具体来说，如“惊奇触发变化”和“稀有事件的波状近期效应”等现象在人类中普遍存在，但在LLMs中完全不存在。我们的研究结果为使用LLMs模拟和预测学习环境中的人类行为提供了见解，并强调了在研究它们是否复制人类决策倾向时需要对其行为进行细致分析的必要性。 

---
# StepMathAgent: A Step-Wise Agent for Evaluating Mathematical Processes through Tree-of-Error 

**Title (ZH)**: StepMathAgent: 一种基于错误树的分步骤评估数学过程的代理模型 

**Authors**: Shu-Xun Yang, Cunxiang Wang, Yidong Wang, Xiaotao Gu, Minlie Huang, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10105)  

**Abstract**: Evaluating mathematical capabilities is critical for assessing the overall performance of large language models (LLMs). However, existing evaluation methods often focus solely on final answers, resulting in highly inaccurate and uninterpretable evaluation outcomes, as well as their failure to assess proof or open-ended problems. To address these issues, we propose a novel mathematical process evaluation agent based on Tree-of-Error, called StepMathAgent. This agent incorporates four internal core operations: logical step segmentation, step scoring, score aggregation and error tree generation, along with four external extension modules: difficulty calibration, simplicity evaluation, completeness validation and format assessment. Furthermore, we introduce StepMathBench, a benchmark comprising 1,000 step-divided process evaluation instances, derived from 200 high-quality math problems grouped by problem type, subject category and difficulty level. Experiments on StepMathBench show that our proposed StepMathAgent outperforms all state-of-the-art methods, demonstrating human-aligned evaluation preferences and broad applicability to various scenarios. Our data and code are available at this https URL. 

**Abstract (ZH)**: 评估数学能力是评估大型语言模型整体性能的关键。然而，现有的评估方法往往只关注最终答案，导致评估结果高度不准确和不可解释，同时也无法评估证明或开放性问题。为解决这些问题，我们提出一种基于错误树的新型数学过程评估代理，称为StepMathAgent。该代理包含四种内部核心操作：逻辑步骤分割、步骤评分、评分聚合和错误树生成，以及四种外部扩展模块：难度校准、简洁性评估、完整性验证和格式评估。此外，我们介绍了StepMathBench，这是一个包含1000个步骤划分过程评估实例的基准，这些实例来自200道高质量数学问题，按问题类型、主题类别和难度级别分组。实验表明，我们的StepMathAgent在StepMathBench上优于所有最先进的方法，展示了与人类一致的评估偏好并适用于各种场景。我们的数据和代码可在以下链接获取。 

---
# Advanced Tool Learning and Selection System (ATLASS): A Closed-Loop Framework Using LLM 

**Title (ZH)**: 基于LLM的闭环框架：高级工具学习与选择系统（ATLASS） 

**Authors**: Mohd Ariful Haque, Justin Williams, Sunzida Siddique, Md. Hujaifa Islam, Hasmot Ali, Kishor Datta Gupta, Roy George  

**Link**: [PDF](https://arxiv.org/pdf/2503.10071)  

**Abstract**: The combination of LLM agents with external tools enables models to solve complex tasks beyond their knowledge base. Human-designed tools are inflexible and restricted to solutions within the scope of pre-existing tools created by experts. To address this problem, we propose ATLASS, an advanced tool learning and selection system designed as a closed-loop framework. It enables the LLM to solve problems by dynamically generating external tools on demand. In this framework, agents play a crucial role in orchestrating tool selection, execution, and refinement, ensuring adaptive problem-solving capabilities. The operation of ATLASS follows three phases: The first phase, Understanding Tool Requirements, involves the Agents determining whether tools are required and specifying their functionality; the second phase, Tool Retrieval/Generation, involves the Agents retrieving or generating tools based on their availability; and the third phase, Task Solving, involves combining all the component tools necessary to complete the initial task. The Tool Dataset stores the generated tools, ensuring reusability and minimizing inference cost. Current LLM-based tool generation systems have difficulty creating complex tools that need APIs or external packages. In ATLASS, we solve the problem by automatically setting up the environment, fetching relevant API documentation online, and using a Python interpreter to create a reliable, versatile tool that works in a wider range of situations. OpenAI GPT-4.0 is used as the LLM agent, and safety and ethical concerns are handled through human feedback before executing generated code. By addressing the limitations of predefined toolsets and enhancing adaptability, ATLASS serves as a real-world solution that empowers users with dynamically generated tools for complex problem-solving. 

**Abstract (ZH)**: LLM代理与外部工具的结合使模型能够解决超出其知识库的复杂任务。ATLASS：一种先进的工具学习和选择系统，设计为闭环框架，使LLM能够通过按需动态生成外部工具来解决问题。 

---
# OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model 

**Title (ZH)**: OR-LLM-Agent: 利用推理大规模语言模型自动建模和求解运筹优化问题 

**Authors**: Bowen Zhang, Pengcheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.10009)  

**Abstract**: Operations Research (OR) has been widely applied in various fields such as resource allocation, production planning, and supply chain management. However, addressing real-world OR problems requires OR experts to perform mathematical modeling and programmers to develop solution algorithms. This traditional method, heavily reliant on experts, is costly and has long development cycles, severely limiting the widespread adoption of OR techniques. Few have considered using Artificial Intelligence (AI) to replace professionals to achieve fully automated solutions for OR problems. We propose OR-LLM-Agent, the first AI agent that enables end-to-end automation for solving real-world OR problems. OR-LLM-Agent leverages the Chain-of-Thought (CoT) reasoning capabilities of Large Language Models (LLMs) to translate natural language problem descriptions into formal mathematical models and automatically generate Gurobi solver code. In OR-LLM-Agent, OR-CodeAgent is designed to automate code execution and repair within a sandbox environment, facilitating the derivation of the final solution. Due to the lack of dedicated benchmark datasets for evaluating the automated solving of OR problems, we construct a benchmark dataset comprising 83 real-world OR problems described in natural language. We conduct comparative experiments with state-of-the-art (SOTA) reasoning LLMs, including GPT-o3-mini, DeepSeek-R1, and Gemini 2.0 Flash Thinking. The OR-LLM-Agent achieved the highest pass rate of 100% and the highest solution accuracy of 85%, demonstrating the feasibility of automated OR problem-solving. Data and code have been publicly available at this https URL. 

**Abstract (ZH)**: OR-LLM-Agent：一种用于解决实际运筹学问题的端到端自动化AI代理 

---
# Media and responsible AI governance: a game-theoretic and LLM analysis 

**Title (ZH)**: 媒体与负责任的AI治理：博弈论与大规模语言模型分析 

**Authors**: Nataliya Balabanova, Adeela Bashir, Paolo Bova, Alessio Buscemi, Theodor Cimpeanu, Henrique Correia da Fonseca, Alessandro Di Stefano, Manh Hong Duong, Elias Fernandez Domingos, Antonio Fernandes, Anh Han, Marcus Krellner, Ndidi Bianca Ogbo, Simon T. Powers, Daniele Proverbio, Fernando P. Santos, Zia Ush Shamszaman, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.09858)  

**Abstract**: This paper investigates the complex interplay between AI developers, regulators, users, and the media in fostering trustworthy AI systems. Using evolutionary game theory and large language models (LLMs), we model the strategic interactions among these actors under different regulatory regimes. The research explores two key mechanisms for achieving responsible governance, safe AI development and adoption of safe AI: incentivising effective regulation through media reporting, and conditioning user trust on commentariats' recommendation. The findings highlight the crucial role of the media in providing information to users, potentially acting as a form of "soft" regulation by investigating developers or regulators, as a substitute to institutional AI regulation (which is still absent in many regions). Both game-theoretic analysis and LLM-based simulations reveal conditions under which effective regulation and trustworthy AI development emerge, emphasising the importance of considering the influence of different regulatory regimes from an evolutionary game-theoretic perspective. The study concludes that effective governance requires managing incentives and costs for high quality commentaries. 

**Abstract (ZH)**: 本文探讨了AI开发者、监管者、用户和媒体之间复杂的相互作用，以促进可信的AI系统。利用演化博弈理论和大规模语言模型（LLMs），我们构建了在不同监管环境下这些行为者之间战略互动的模型。研究探索了实现负责任治理、安全AI开发和安全AI采纳的两种关键机制：通过媒体报道激励有效的监管，以及将用户信任依赖于评论社群的建议。研究结果强调了媒体在向用户提供信息方面的关键作用，可能作为一种“软”监管形式，通过调查开发者或监管者来取代机构化的AI监管（在许多地区仍未实现）。博弈论分析和基于LLMs的模拟揭示了哪些条件下有效的监管和可信的AI开发能够出现，强调了从演化博弈理论视角考虑不同监管环境影响的重要性。研究得出结论，有效的治理需要管理高质量评论的利益和成本。 

---
# AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents 

**Title (ZH)**: AgentDAM：自主网络代理的隐私泄露评估 

**Authors**: Arman Zharmagambetov, Chuan Guo, Ivan Evtimov, Maya Pavlova, Ruslan Salakhutdinov, Kamalika Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2503.09780)  

**Abstract**: LLM-powered AI agents are an emerging frontier with tremendous potential to increase human productivity. However, empowering AI agents to take action on their user's behalf in day-to-day tasks involves giving them access to potentially sensitive and private information, which leads to a possible risk of inadvertent privacy leakage when the agent malfunctions. In this work, we propose one way to address that potential risk, by training AI agents to better satisfy the privacy principle of data minimization. For the purposes of this benchmark, by "data minimization" we mean instances where private information is shared only when it is necessary to fulfill a specific task-relevant purpose. We develop a benchmark called AgentDAM to evaluate how well existing and future AI agents can limit processing of potentially private information that we designate "necessary" to fulfill the task. Our benchmark simulates realistic web interaction scenarios and is adaptable to all existing web navigation agents. We use AgentDAM to evaluate how well AI agents built on top of GPT-4, Llama-3 and Claude can limit processing of potentially private information when unnecessary, and show that these agents are often prone to inadvertent use of unnecessary sensitive information. We finally propose a prompting-based approach that reduces this. 

**Abstract (ZH)**: LLM赋能的AI代理是提升人类生产力的新兴前沿领域，但赋予其在日常任务中代表用户采取行动可能涉及对其潜在敏感和私人信息的访问，这可能导致代理故障时无意中泄露隐私的风险。在此工作中，我们提出了一种应对该潜在风险的方法，即训练AI代理更好地满足数据最小化隐私原则。为本次基准测试的目的，“数据最小化”是指仅在履行特定任务相关目的必要时分享私人信息。我们开发了一个名为AgentDAM的基准测试，以评估现有和未来AI代理如何限制处理我们指定为“必要”的可能涉及私人信息的处理。我们的基准测试模拟了现实的网络交互场景，并适应所有现有的网络导航代理。我们使用AgentDAM评估基于GPT-4、Llama-3和Claude构建的AI代理在不必要的情况下如何限制处理可能涉及私人信息，并展示这些代理经常无意中使用不必要的敏感信息。最后，我们提出了一种基于提示的方法来减少这一问题。 

---
# A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1 

**Title (ZH)**: 令人沮丧的简单但极其有效的攻击基线：针对GPT-4.5/4001的强大黑盒模型成功率超过90% 

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10635)  

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at this https URL. 

**Abstract (ZH)**: 尽管开源大型视觉-语言模型（LVLMs）表现出色，基于转移的针对性攻击经常在面向商业的黑盒LVLMs上失败。分析失败的对抗性扰动发现，学习到的扰动通常源自均匀分布且缺乏明确的语义细节，导致意外的响应。这种语义信息的缺失使商业LVLMs要么完全忽略扰动，要么错误解释其嵌入的语义，从而导致攻击失败。为克服这些问题，我们注意到，训练模型时识别核心语义对象是一个关键目标。这一洞察促使我们提出一种方法，通过在局部区域中编码明确的语义细节来提高语义清晰度，从而确保互操作性和捕捉更细腻的特征，同时将修改集中在语义丰富的区域而不是均匀应用。为实现这一目标，我们提出了一种简单而有效的解决方案：在每次优化步骤中，随机裁剪对抗性图像的控制方面比和比例，重新调整大小，并在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们使用局部聚集扰动创建的对抗性样本专注于关键区域，在商业LVLMs，包括GPT-4.5、GPT-4o、Gemini-2.0-flash、Claude-3.5-sonnet、Claude-3.7-sonnet以及推理模型如o1、Claude-3.7-thinking和Gemini-2.0-flash-thinking中表现出惊人的转移性。我们的方法在GPT-4.5、4o和o1上的成功率超过90%，显著优于所有先前的最佳攻击方法。我们的优化对抗性样本在不同配置和训练代码下的版本可在以下链接获得。 

---
# SciVerse: Unveiling the Knowledge Comprehension and Visual Reasoning of LMMs on Multi-modal Scientific Problems 

**Title (ZH)**: SciVerse: 揭示LMMs在多模态科学问题上的知识理解与视觉推理能力 

**Authors**: Ziyu Guo, Ray Zhang, Hao Chen, Jialin Gao, Dongzhi Jiang, Jiaze Wang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10627)  

**Abstract**: The rapid advancement of Large Multi-modal Models (LMMs) has enabled their application in scientific problem-solving, yet their fine-grained capabilities remain under-explored. In this paper, we introduce SciVerse, a multi-modal scientific evaluation benchmark to thoroughly assess LMMs across 5,735 test instances in five distinct versions. We aim to investigate three key dimensions of LMMs: scientific knowledge comprehension, multi-modal content interpretation, and Chain-of-Thought (CoT) reasoning. To unveil whether LMMs possess sufficient scientific expertise, we first transform each problem into three versions containing different levels of knowledge required for solving, i.e., Knowledge-free, -lite, and -rich. Then, to explore how LMMs interpret multi-modal scientific content, we annotate another two versions, i.e., Vision-rich and -only, marking more question information from texts to diagrams. Comparing the results of different versions, SciVerse systematically examines the professional knowledge stock and visual perception skills of LMMs in scientific domains. In addition, to rigorously assess CoT reasoning, we propose a new scientific CoT evaluation strategy, conducting a step-wise assessment on knowledge and logical errors in model outputs. Our extensive evaluation of different LMMs on SciVerse reveals critical limitations in their scientific proficiency and provides new insights into future developments. Project page: this https URL 

**Abstract (ZH)**: 大规模多模态模型的 rapid advancement 已使其在科学研究中得以应用，但其精细能力仍亟待深入探索。本文介绍了一种多模态科学评估基准 SciVerse，以全面评估 LMMs 在五个不同版本中的 5,735 个测试实例中的性能。我们旨在研究 LMMs 的三个关键维度：科学知识理解、多模态内容解释和思维链（CoT）推理。为了揭示 LMMs 是否具备足够的科学专业知识，我们首先将每个问题转化为包含不同知识要求的三个版本，即知识无、轻和丰富版本。然后，为了探索 LMMs 如何解释多模态科学内容，我们标注了另外两个版本，即视觉丰富和仅视觉版本，从文本中更多地标记问题信息为图表。通过比较不同版本的结果，SciVerse 系统性地检查了 LMMs 在科学领域的专业知识储备和视觉感知能力。此外，为了严格评估 CoT 推理，我们提出了一种新的科学 CoT 评估策略，在模型输出的知识和逻辑错误方面进行逐步评估。我们在 SciVerse 上对不同 LMMs 的广泛评估揭示了它们在科学专业性方面的关键局限性，并为未来的发展提供了新的见解。项目页面：这个 https URL。 

---
# Compositional Subspace Representation Fine-tuning for Adaptive Large Language Models 

**Title (ZH)**: compositional 子空间表示微调以实现自适应大型语言模型 

**Authors**: Andy Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10617)  

**Abstract**: Adapting large language models to multiple tasks can cause cross-skill interference, where improvements for one skill degrade another. While methods such as LoRA impose orthogonality constraints at the weight level, they do not fully address interference in hidden-state representations. We propose Compositional Subspace Representation Fine-tuning (CS-ReFT), a novel representation-based approach that learns multiple orthonormal subspace transformations, each specializing in a distinct skill, and composes them via a lightweight router. By isolating these subspace edits in the hidden state, rather than weight matrices, CS-ReFT prevents cross-task conflicts more effectively. On the AlpacaEval benchmark, applying CS-ReFT to Llama-2-7B achieves a 93.94% win rate, surpassing GPT-3.5 Turbo (86.30%) while requiring only 0.0098% of model parameters. These findings show that specialized representation edits, composed via a simple router, significantly enhance multi-task instruction following with minimal overhead. 

**Abstract (ZH)**: 一种新的表示方法：组成子空间表示微调（CS-ReFT）在多个任务指令跟随中的应用 

---
# TruthPrInt: Mitigating LVLM Object Hallucination Via Latent Truthful-Guided Pre-Intervention 

**Title (ZH)**: TruthPrInt: 通过潜在事实引导预干预减轻LVLM对象幻觉 

**Authors**: Jinhao Duan, Fei Kong, Hao Cheng, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10602)  

**Abstract**: Object Hallucination (OH) has been acknowledged as one of the major trustworthy challenges in Large Vision-Language Models (LVLMs). Recent advancements in Large Language Models (LLMs) indicate that internal states, such as hidden states, encode the "overall truthfulness" of generated responses. However, it remains under-explored how internal states in LVLMs function and whether they could serve as "per-token" hallucination indicators, which is essential for mitigating OH. In this paper, we first conduct an in-depth exploration of LVLM internal states in relation to OH issues and discover that (1) LVLM internal states are high-specificity per-token indicators of hallucination behaviors. Moreover, (2) different LVLMs encode universal patterns of hallucinations in common latent subspaces, indicating that there exist "generic truthful directions" shared by various LVLMs. Based on these discoveries, we propose Truthful-Guided Pre-Intervention (TruthPrInt) that first learns the truthful direction of LVLM decoding and then applies truthful-guided inference-time intervention during LVLM decoding. We further propose ComnHallu to enhance both cross-LVLM and cross-data hallucination detection transferability by constructing and aligning hallucination latent subspaces. We evaluate TruthPrInt in extensive experimental settings, including in-domain and out-of-domain scenarios, over popular LVLMs and OH benchmarks. Experimental results indicate that TruthPrInt significantly outperforms state-of-the-art methods. Codes will be available at this https URL. 

**Abstract (ZH)**: Object Hallucination的内部状态探索及其缓解：基于可信方向的预干预（TruthPrInt）和跨模型幻觉检测增强（ComnHallu） 

---
# Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More 

**Title (ZH)**: 语言模型、图搜索与监督污染：何时更多的监督会导致效果下降以及如何使监督更加有效 

**Authors**: Arvid Frydenlund  

**Link**: [PDF](https://arxiv.org/pdf/2503.10542)  

**Abstract**: This work concerns the path-star task, a minimal example of searching over a graph. The graph, $G$, is star-shaped with $D$ arms radiating from a start node, $s$. A language model (LM) is given $G$, $s$, and a target node $t$, which ends one of the arms and is tasked with generating the arm containing $t$. The minimal nature of this task means only a single choice needs to be made: which of the $D$ arms contains $t$?
Decoder-only LMs fail to solve this elementary task above $1/D$ chance due to a learned shortcut that absorbs training supervision. We show how this pathology is caused by excess supervision and we present a series of solutions demonstrating that the task is solvable via decoder-only LMs. We find that the task's minimal nature causes its difficulty, as it prevents task decomposition. Our solutions provide insight into the pathology and its implications for LMs trained via next-token prediction. 

**Abstract (ZH)**: 本工作关注路径-星图任务，这是在图上搜索的一个最小范例。图 $G$ 是星形结构，从起始节点 $s$ 辐射出 $D$ 条臂。给定一个语言模型 (LM)，以及目标节点 $t$ （其位于其中一条臂的末端），任务是生成包含 $t$ 的那条臂。由于任务的最小性质，只需要做出一个选择：哪一条臂包含 $t$。解码器仅模型在超过 $1/D$ 的随机猜测概率下无法解决这一基本任务，这是由于学习到的一个捷径吸收了训练监督。我们展示了这种病理现象是如何由过度的监督引起的，并提供了一系列解决方案，证明此类任务可以通过解码器仅模型解决。我们发现任务的最小性质使其困难，因为它阻止了任务的分解。我们的解决方案为理解此类病理现象及其对通过下一个单词预测训练的模型的影响提供了见解。 

---
# PiSA: A Self-Augmented Data Engine and Training Strategy for 3D Understanding with Large Models 

**Title (ZH)**: PiSA: 一种自我增强的数据引擎和训练策略，用于大型模型的三维理解 

**Authors**: Zilu Guo, Hongbin Lin, Zhihao Yuan, Chaoda Zheng, Pengshuo Qiu, Dongzhi Jiang, Renrui Zhang, Chun-Mei Feng, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10529)  

**Abstract**: 3D Multimodal Large Language Models (MLLMs) have recently made substantial advancements. However, their potential remains untapped, primarily due to the limited quantity and suboptimal quality of 3D datasets. Current approaches attempt to transfer knowledge from 2D MLLMs to expand 3D instruction data, but still face modality and domain gaps. To this end, we introduce PiSA-Engine (Point-Self-Augmented-Engine), a new framework for generating instruction point-language datasets enriched with 3D spatial semantics. We observe that existing 3D MLLMs offer a comprehensive understanding of point clouds for annotation, while 2D MLLMs excel at cross-validation by providing complementary information. By integrating holistic 2D and 3D insights from off-the-shelf MLLMs, PiSA-Engine enables a continuous cycle of high-quality data generation. We select PointLLM as the baseline and adopt this co-evolution training framework to develop an enhanced 3D MLLM, termed PointLLM-PiSA. Additionally, we identify limitations in previous 3D benchmarks, which often feature coarse language captions and insufficient category diversity, resulting in inaccurate evaluations. To address this gap, we further introduce PiSA-Bench, a comprehensive 3D benchmark covering six key aspects with detailed and diverse labels. Experimental results demonstrate PointLLM-PiSA's state-of-the-art performance in zero-shot 3D object captioning and generative classification on our PiSA-Bench, achieving significant improvements of 46.45% (+8.33%) and 63.75% (+16.25%), respectively. We will release the code, datasets, and benchmark. 

**Abstract (ZH)**: 3D多模态大语言模型（MLLMs） recently取得了重大进展。然而，其潜力尚未完全开发，主要是由于3D数据集的数量有限且质量不佳。目前的方法试图从2D MLLMs转移知识以扩展3D指令数据，但仍面临模态和领域差距。为了解决这一问题，我们提出了PiSA-Engine（点自我增强引擎），这是一种新的框架，用于生成包含3D空间语义的指令点语言数据集。我们观察到现有的3D MLLMs在注解方面提供了全面的点云理解，而2D MLLMs则通过提供补充信息在交叉验证方面表现出色。通过结合现成的2D和3D MLLMs的整体见解，PiSA-Engine使高品质数据生成形成连续循环。我们选择PointLLM作为基线，并采用这种共同演化训练框架开发了一种增强的3D MLLM，称为PointLLM-PiSA。此外，我们还指出了之前3D基准的局限性，这些基准通常语言标注粗糙且分类多样性不足，导致评估不准确。为此，我们进一步引入了PiSA-Bench，这是一种涵盖六个关键方面的全面3D基准，具有详细的多样标签。实验结果表明，PointLLM-PiSA在我们的PiSA-Bench上的零样本3D对象描述和生成分类方面表现出最先进的性能，分别取得了46.45%（+8.33%）和63.75%（+16.25%）的重大提升。我们将发布代码、数据集和基准。 

---
# LLMs in Disease Diagnosis: A Comparative Study of DeepSeek-R1 and O3 Mini Across Chronic Health Conditions 

**Title (ZH)**: LLMs在疾病诊断中的比较研究：慢性健康条件下DeepSeek-R1和O3 Mini的对比分析 

**Authors**: Gaurav Kumar Gupta, Pranal Pande  

**Link**: [PDF](https://arxiv.org/pdf/2503.10486)  

**Abstract**: Large Language Models (LLMs) are revolutionizing medical diagnostics by enhancing both disease classification and clinical decision-making. In this study, we evaluate the performance of two LLM- based diagnostic tools, DeepSeek R1 and O3 Mini, using a structured dataset of symptoms and diagnoses. We assessed their predictive accuracy at both the disease and category levels, as well as the reliability of their confidence scores. DeepSeek R1 achieved a disease-level accuracy of 76% and an overall accuracy of 82%, outperforming O3 Mini, which attained 72% and 75% respectively. Notably, DeepSeek R1 demonstrated exceptional performance in Mental Health, Neurological Disorders, and Oncology, where it reached 100% accuracy, while O3 Mini excelled in Autoimmune Disease classification with 100% accuracy. Both models, however, struggled with Respiratory Disease classification, recording accuracies of only 40% for DeepSeek R1 and 20% for O3 Mini. Additionally, the analysis of confidence scores revealed that DeepSeek R1 provided high-confidence predictions in 92% of cases, compared to 68% for O3 Mini. Ethical considerations regarding bias, model interpretability, and data privacy are also discussed to ensure the responsible integration of LLMs into clinical practice. Overall, our findings offer valuable insights into the strengths and limitations of LLM-based diagnostic systems and provide a roadmap for future enhancements in AI-driven healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在通过增强疾病分类和临床决策来革新医疗诊断。在本研究中，我们使用症状和诊断的结构化数据评估了两种基于LLM的诊断工具DeepSeek R1和O3 Mini的表现。我们评估了它们在疾病和类别水平上的预测准确性，以及它们置信度评分的可靠性。DeepSeek R1在疾病水平上的准确率为76%，整体准确率为82%，优于O3 Mini的72%和75%。值得注意的是，DeepSeek R1在精神健康、神经系统疾病和肿瘤学领域表现出色，达到100%的准确率，而O3 Mini在自身免疫病分类方面的准确率为100%。然而，这两种模型在呼吸系统疾病的分类方面都表现不佳，DeepSeek R1的准确率为40%，O3 Mini为20%。此外，置信度评分的分析显示，DeepSeek R1在92%的情况下提供了高置信度预测，而O3 Mini为68%。我们还讨论了关于偏见、模型可解释性和数据隐私的伦理考虑，以确保LLM在临床实践中的负责任集成。本研究结果为LLM基于的诊断系统的强项和不足提供了有价值的见解，并提供了未来AI驱动医疗保健改进的路线图。 

---
# DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation 

**Title (ZH)**: DynaCode: 一种动态复杂性意识代码基准，用于评估代码生成中的大型语言模型 

**Authors**: Wenhao Hu, Jinhao Duan, Chunchen Wei, Li Zhang, Yue Zhang, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10452)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly improved their performance in code generation tasks. However, existing code benchmarks remain static, consisting of fixed datasets with predefined problems. This makes them vulnerable to memorization during training, where LLMs recall specific test cases instead of generalizing to new problems, leading to data contamination and unreliable evaluation results. To address these issues, we introduce DynaCode, a dynamic, complexity-aware benchmark that overcomes the limitations of static datasets. DynaCode evaluates LLMs systematically using a complexity-aware metric, incorporating both code complexity and call-graph structures. DynaCode achieves large-scale diversity, generating up to 189 million unique nested code problems across four distinct levels of code complexity, referred to as units, and 16 types of call graphs. Results on 12 latest LLMs show an average performance drop of 16.8% to 45.7% compared to MBPP+, a static code generation benchmark, with performance progressively decreasing as complexity increases. This demonstrates DynaCode's ability to effectively differentiate LLMs. Additionally, by leveraging call graphs, we gain insights into LLM behavior, particularly their preference for handling subfunction interactions within nested code. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的迅速发展显著提高了其在代码生成任务中的性能。然而，现有的代码基准仍然保持静态，由固定数据集和预定义的问题组成。这使得它们在训练过程中容易出现记忆现象，即LLMs回忆特定测试案例而不是泛化到新问题，导致数据污染和不可靠的评估结果。为解决这些问题，我们提出了DynaCode，一种动态、复杂性感知的基准，克服了静态数据集的局限性。DynaCode使用复杂性感知度量系统地评估LLMs，结合代码复杂性和调用图结构。DynaCode实现了大规模多样性，生成了高达1.89亿个不同复杂度级别的唯一嵌套代码问题，涵盖四个复杂度单位和16种调用图类型。在12个最新的LLMs上的结果表明，与静态代码生成基准MBPP+相比，平均性能下降16.8%至45.7%，随着复杂性的增加，性能逐渐下降。这证明了DynaCode有效地区分LLMs的能力。此外，通过利用调用图，我们能够深入了解LLM的行为，尤其是它们处理嵌套代码中子函数交互的偏好。 

---
# RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing 

**Title (ZH)**: RoMA: 基于Mamba的基础模型在遥感领域的规模化应用 

**Authors**: Fengxiang Wang, Hongzhen Wang, Yulin Wang, Di Wang, Mingshuo Chen, Haiyan Zhao, Yangang Sun, Shuo Wang, Long Lan, Wenjing Yang, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10392)  

**Abstract**: Recent advances in self-supervised learning for Vision Transformers (ViTs) have fueled breakthroughs in remote sensing (RS) foundation models. However, the quadratic complexity of self-attention poses a significant barrier to scalability, particularly for large models and high-resolution images. While the linear-complexity Mamba architecture offers a promising alternative, existing RS applications of Mamba remain limited to supervised tasks on small, domain-specific datasets. To address these challenges, we propose RoMA, a framework that enables scalable self-supervised pretraining of Mamba-based RS foundation models using large-scale, diverse, unlabeled data. RoMA enhances scalability for high-resolution images through a tailored auto-regressive learning strategy, incorporating two key innovations: 1) a rotation-aware pretraining mechanism combining adaptive cropping with angular embeddings to handle sparsely distributed objects with arbitrary orientations, and 2) multi-scale token prediction objectives that address the extreme variations in object scales inherent to RS imagery. Systematic empirical studies validate that Mamba adheres to RS data and parameter scaling laws, with performance scaling reliably as model and data size increase. Furthermore, experiments across scene classification, object detection, and semantic segmentation tasks demonstrate that RoMA-pretrained Mamba models consistently outperform ViT-based counterparts in both accuracy and computational efficiency. The source code and pretrained models will be released at this https URL. 

**Abstract (ZH)**: Recent Advances in Scalable Self-Supervised Pretraining of Mamba-Based Remote Sensing Foundation Models Using RoMA 

---
# G-Boost: Boosting Private SLMs with General LLMs 

**Title (ZH)**: G-Boost: 通过通用大语言模型增强私有SLMs 

**Authors**: Yijiang Fan, Yuren Mao, Longbin Lai, Ying Zhang, Zhengping Qian, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10367)  

**Abstract**: Due to the limited computational resources, most Large Language Models (LLMs) developers can only fine-tune Small Language Models (SLMs) on their own data. These private SLMs typically have limited effectiveness. To boost the performance of private SLMs, this paper proposes to ask general LLMs for help. The general LLMs can be APIs or larger LLMs whose inference cost the developers can afford. Specifically, we propose the G-Boost framework where a private SLM adaptively performs collaborative inference with a general LLM under the guide of process reward. Experiments demonstrate that our framework can significantly boost the performance of private SLMs. 

**Abstract (ZH)**: 由于计算资源有限，大多数大型语言模型开发者只能在其自有数据上 fine-tune 小型语言模型。这些私有小型语言模型通常效果有限。为了提升私有小型语言模型的性能，本文提出请求通用大型语言模型提供帮助的方法。通用大型语言模型可以是 API 或者开发者负担得起推理成本的更大模型。具体而言，我们提出了 G-Boost 框架，在该框架下，私有小型语言模型在过程奖励的指导下与通用大型语言模型协作推理。实验结果表明，我们的框架能够显著提升私有小型语言模型的性能。 

---
# KV-Distill: Nearly Lossless Learnable Context Compression for LLMs 

**Title (ZH)**: KV-精炼：几乎无损的学习可微上下文压缩 for 大型语言模型 

**Authors**: Vivek Chari, Guanghui Qin, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2503.10337)  

**Abstract**: Sequence-to-sequence tasks often benefit from long contexts, but the quadratic complexity of self-attention in standard Transformers renders this non-trivial. During generation, temporary representations -stored in the so-called KV cache-account for a large portion of GPU memory usage and scale linearly with context length. We introduce KV-Distill, a Transformer compression framework that distills long context KV caches into significantly shorter representations in a question-independent fashion. KV-Distill can be trained as a parameter-efficient adaptor for pretrained models, and enables the compression of arbitrary spans of a context while preserving pre-trained model capabilities. We treat a compressed-uncompressed cache as a student-teacher pairing and apply a KL-type divergence to match the generated outputs. KV-Distill outperforms other compression techniques in worst-case extractive tasks and approaches uncompressed performance in long context question answering and summarization, and it can be fine-tuned on domain-specific contexts to reduce lengths by up to 99% while preserving downstream performance. We demonstrate the generalizability of KV-Distill across various model sizes and architectures. 

**Abstract (ZH)**: 基于序列的任务往往可以从长上下文中受益，但标准Transformer中的自注意机制的二次复杂性使得这一点并不简单。在生成过程中，临时表示——存储在所谓的KV缓存中——占用了大量的GPU内存，并且与上下文长度成线性关系。我们引入了KV-Distill，这是一种Transformer压缩框架，能够在问题无关的情况下将长上下文的KV缓存压缩为显著较短的表示。KV-Distill可以作为参数高效适配器进行预训练模型的训练，并能够压缩上下文的任意部分同时保留预训练模型的能力。我们将压缩-未压缩缓存视为学生-教师配对，并应用KL型散度来匹配生成的输出。KV-Distill在最坏情况下的提取任务中表现出色，并在长上下文问答和总结任务中接近未压缩性能，同时可以通过特定领域上下文的微调最多减少99%的长度而不影响下游性能。KV-Distill在不同模型大小和架构中展示了通用性。 

---
# MinorBench: A hand-built benchmark for content-based risks for children 

**Title (ZH)**: MinorBench: 一个手工构建的内容相关儿童风险基准测试 

**Authors**: Shaun Khoo, Gabriel Chua, Rachel Shong  

**Link**: [PDF](https://arxiv.org/pdf/2503.10242)  

**Abstract**: Large Language Models (LLMs) are rapidly entering children's lives - through parent-driven adoption, schools, and peer networks - yet current AI ethics and safety research do not adequately address content-related risks specific to minors. In this paper, we highlight these gaps with a real-world case study of an LLM-based chatbot deployed in a middle school setting, revealing how students used and sometimes misused the system. Building on these findings, we propose a new taxonomy of content-based risks for minors and introduce MinorBench, an open-source benchmark designed to evaluate LLMs on their ability to refuse unsafe or inappropriate queries from children. We evaluate six prominent LLMs under different system prompts, demonstrating substantial variability in their child-safety compliance. Our results inform practical steps for more robust, child-focused safety mechanisms and underscore the urgency of tailoring AI systems to safeguard young users. 

**Abstract (ZH)**: 大型语言模型（LLMs）正迅速进入儿童的生活——通过父母的采用、学校和同龄人网络——然而当前的AI伦理和安全研究未能充分应对特定于未成年人的内容相关风险。在这篇论文中，我们通过一个中学校园环境中的LLM聊天机器人案例研究，揭示了学生如何使用并有时误用该系统，从而指出现有研究中的缺口。基于这些发现，我们提出了一种新的针对未成年人的内容相关风险分类，并介绍了MinorBench，一个开源基准，用于评估LLM在拒绝儿童发出的不安全或不适当查询方面的能力。我们对六种主流LLM在不同系统提示下的表现进行了评估，展示了它们在儿童安全合规性方面的显著差异。我们的结果为构建更 robust、更关注儿童的安全机制提供了实用步骤，并强调了调整AI系统以保护年轻用户的重要性。 

---
# Efficient Federated Fine-Tuning of Large Language Models with Layer Dropout 

**Title (ZH)**: 高效分层dropout在联邦细调大规模语言模型中的应用 

**Authors**: Shilong Wang, Jianchun Liu, Hongli Xu, Jiaming Yan, Xianjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10217)  

**Abstract**: Fine-tuning plays a crucial role in enabling pre-trained LLMs to evolve from general language comprehension to task-specific expertise. To preserve user data privacy, federated fine-tuning is often employed and has emerged as the de facto paradigm. However, federated fine-tuning is prohibitively inefficient due to the tension between LLM complexity and the resource constraint of end devices, incurring unaffordable fine-tuning overhead. Existing literature primarily utilizes parameter-efficient fine-tuning techniques to mitigate communication costs, yet computational and memory burdens continue to pose significant challenges for developers. This work proposes DropPEFT, an innovative federated PEFT framework that employs a novel stochastic transformer layer dropout method, enabling devices to deactivate a considerable fraction of LLMs layers during training, thereby eliminating the associated computational load and memory footprint. In DropPEFT, a key challenge is the proper configuration of dropout ratios for layers, as overhead and training performance are highly sensitive to this setting. To address this challenge, we adaptively assign optimal dropout-ratio configurations to devices through an exploration-exploitation strategy, achieving efficient and effective fine-tuning. Extensive experiments show that DropPEFT can achieve a 1.3-6.3\times speedup in model convergence and a 40%-67% reduction in memory footprint compared to state-of-the-art methods. 

**Abstract (ZH)**: Fine-tuning plays a crucial role in enabling pre-trained LLMs to evolve from general language comprehension to task-specific expertise. To preserve user data privacy, federated fine-tuning is often employed and has emerged as the de facto paradigm. However, federated fine-tuning is prohibitively inefficient due to the tension between LLM complexity and the resource constraint of end devices, incurring unaffordable fine-tuning overhead. Existing literature primarily utilizes parameter-efficient fine-tuning techniques to mitigate communication costs, yet computational and memory burdens continue to pose significant challenges for developers. This work proposes DropPEFT, an innovative federated PEFT framework that employs a novel stochastic transformer layer dropout method, enabling devices to deactivate a considerable fraction of LLMs layers during training, thereby eliminating the associated computational load and memory footprint. In DropPEFT, a key challenge is the proper configuration of dropout ratios for layers, as overhead and training performance are highly sensitive to this setting. To address this challenge, we adaptively assign optimal dropout-ratio configurations to devices through an exploration-exploitation strategy, achieving efficient and effective fine-tuning. Extensive experiments show that DropPEFT can achieve a 1.3-6.3倍模型收敛速度提升和40%-67%的内存占用减少， compared to state-of-the-art methods。 

---
# Robustness Tokens: Towards Adversarial Robustness of Transformers 

**Title (ZH)**: 鲁棒性令牌：迈向Transformer的对抗鲁棒性 

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.10191)  

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances. 

**Abstract (ZH)**: 近期，大型预训练基础模型已被机器学习 practitioner 广泛用于多种任务。鉴于此类模型是公开可用的，依赖其作为下游任务骨干模型的使用可能会导致高易受使用相同公开模型构造的对抗攻击的影响。在本文中，我们提出了一种专门针对transformer 架构的 Robustness Tokens 新颖方法，该方法以较低的计算要求微调少量额外的私有 tokens，而传统对抗训练则是调整模型参数。我们展示了 Robustness Tokens 使 Vision Transformer 模型在面对白盒对抗攻击时显著更加稳健，同时保持原始下游任务的性能。 

---
# Retrieval-Augmented Generation with Hierarchical Knowledge 

**Title (ZH)**: 具有层级知识增强的检索生成 

**Authors**: Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10150)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) methods have significantly enhanced the performance of large language models (LLMs) in domain-specific tasks. However, existing RAG methods do not adequately utilize the naturally inherent hierarchical knowledge in human cognition, which limits the capabilities of RAG systems. In this paper, we introduce a new RAG approach, called HiRAG, which utilizes hierarchical knowledge to enhance the semantic understanding and structure capturing capabilities of RAG systems in the indexing and retrieval processes. Our extensive experiments demonstrate that HiRAG achieves significant performance improvements over the state-of-the-art baseline methods. The code of our proposed method is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于图的层次知识增强检索增强生成（HiRAG）方法在特定领域任务中显著提升了大型语言模型的性能。然而，现有的RAG方法并未充分利用人类认知中自然具有的层次知识，这限制了RAG系统的功能。在本文中，我们提出了一种新的RAG方法，称为HiRAG，该方法利用层次知识在索引和检索过程中增强RAG系统的语义理解和结构捕获能力。我们的大量实验证明，HiRAG在性能上显著优于最先进的基线方法。所提出方法的代码可在\href{this https URL}{this https URL}获取。 

---
# Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding 

**Title (ZH)**: 九尾狐：一种混合架构，用于在推测性解码中优先处理早期令牌 

**Authors**: Jinze Li, Yixing Xu, Haiduo Huang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10135)  

**Abstract**: Speculative decoding (SPD) aims to accelerate the auto-regressive token generation process of a target Large Language Model (LLM). Some approaches employ a draft model with multiple heads to predict a sequence of future tokens, where each head handles a token in the sequence. The target LLM verifies the predicted sequence and accepts aligned tokens, enabling efficient multi-token generation. However, existing methods assume that all tokens within a sequence are equally important, employing identical head structures and relying on a single-generation paradigm, either serial or parallel. To this end, we theoretically demonstrate that initial tokens in the draft sequence are more important than later ones. Building on this insight, we propose Gumiho, a hybrid model combining serial and parallel heads. Specifically, given the critical importance of early tokens, we employ a sophisticated Transformer architecture for the early draft heads in a serial configuration to improve accuracy. For later tokens, we utilize multiple lightweight MLP heads operating in parallel to enhance efficiency. By allocating more advanced model structures and longer running times to the early heads, Gumiho achieves improved overall performance. The experimental results demonstrate that our method outperforms existing approaches, fully validating its effectiveness. 

**Abstract (ZH)**: 推测性解码（SPD）旨在加速目标大型语言模型（LLM）的自回归 token 生成过程。一些方法通过使用具有多个头的草稿模型来预测未来 token 的序列，每个头处理序列中的一个 token。目标 LLM 验证预测的序列并接受对齐的 token，从而实现高效的多 token 生成。然而，现有方法假设序列中的所有 token 具有同等的重要性，采用相同的头结构并依赖于串行或并行的一次生成 paradigm。基于此，我们理论证明初始 token 在草稿序列中比后来的 token 更重要。据此，我们提出了一种结合串行和并行头的混合模型 Gumiho。具体而言，鉴于早期 token 的关键重要性，我们采用复杂的 Transformer 架构在串行配置中为早期草稿头提供更高的准确性。对于后续 token，我们使用多个轻量级的 MLP 头在并行配置中进行处理以提高效率。通过为早期头分配更先进的模型结构和更长的运行时间，Gumiho 实现了整体性能的提升。实验结果表明，我们的方法优于现有方法，充分验证了其有效性。 

---
# Cognitive-Mental-LLM: Leveraging Reasoning in Large Language Models for Mental Health Prediction via Online Text 

**Title (ZH)**: 认知-心理大语言模型：通过在线文本利用推理预测心理健康 

**Authors**: Avinash Patil, Amardeep Kour Gedhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10095)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential in predicting mental health outcomes from online text, yet traditional classification methods often lack interpretability and robustness. This study evaluates structured reasoning techniques-Chain-of-Thought (CoT), Self-Consistency (SC-CoT), and Tree-of-Thought (ToT)-to improve classification accuracy across multiple mental health datasets sourced from Reddit. We analyze reasoning-driven prompting strategies, including Zero-shot CoT and Few-shot CoT, using key performance metrics such as Balanced Accuracy, F1 score, and Sensitivity/Specificity. Our findings indicate that reasoning-enhanced techniques improve classification performance over direct prediction, particularly in complex cases. Compared to baselines such as Zero Shot non-CoT Prompting, and fine-tuned pre-trained transformers such as BERT and Mental-RoBerta, and fine-tuned Open Source LLMs such as Mental Alpaca and Mental-Flan-T5, reasoning-driven LLMs yield notable gains on datasets like Dreaddit (+0.52\% over M-LLM, +0.82\% over BERT) and SDCNL (+4.67\% over M-LLM, +2.17\% over BERT). However, performance declines in Depression Severity, and CSSRS predictions suggest dataset-specific limitations, likely due to our using a more extensive test set. Among prompting strategies, Few-shot CoT consistently outperforms others, reinforcing the effectiveness of reasoning-driven LLMs. Nonetheless, dataset variability highlights challenges in model reliability and interpretability. This study provides a comprehensive benchmark of reasoning-based LLM techniques for mental health text classification. It offers insights into their potential for scalable clinical applications while identifying key challenges for future improvements. 

**Abstract (ZH)**: 大型语言模型在预测从在线文本中获得的心理健康结果方面展现了潜力，但传统分类方法通常缺乏可解释性和稳健性。本研究评估了结构化推理技术——链式推理（CoT）、自我一致性（SC-CoT）和思维方式树（ToT）——以提高跨多个来自Reddit的心理健康数据集的分类准确性。我们使用均衡准确性、F1分数和敏感性/特异性等关键性能指标，分析了推理驱动的提示策略，包括零样本链式推理和少量样本链式推理。研究结果表明，增强推理的技术在复杂情况下提高了分类性能，与零样本非CoT提示、微调预训练变换器（如BERT和Mental-RoBerta）、以及微调的开源大型语言模型（如Mental Alpaca和Mental-Flan-T5）相比，具有显著优势。例如，在Dreaddit（+0.52%）和SDCNL（+4.67%）等数据集上，推理驱动的大型语言模型的性能优于基线。然而，在抑郁症严重程度和CSSRS预测中，性能下降，这可能反映了数据集的特定限制。在提示策略中，少量样本链式推理表现最佳，增强了推理驱动大型语言模型的有效性。然而，数据集的变异性突显了模型可靠性及解释性方面的问题。本研究为心理健康的文本分类提供了综合的推理驱动大型语言模型基准，提供了其在可扩展临床应用方面的潜在见解，并指出了未来改进的关键挑战。 

---
# Compute Optimal Scaling of Skills: Knowledge vs Reasoning 

**Title (ZH)**: 计算技能的最佳缩放比例：知识 vs 原理推理 

**Authors**: Nicholas Roberts, Niladri Chatterji, Sharan Narang, Mike Lewis, Dieuwke Hupkes  

**Link**: [PDF](https://arxiv.org/pdf/2503.10061)  

**Abstract**: Scaling laws are a critical component of the LLM development pipeline, most famously as a way to forecast training decisions such as 'compute-optimally' trading-off parameter count and dataset size, alongside a more recent growing list of other crucial decisions. In this work, we ask whether compute-optimal scaling behaviour can be skill-dependent. In particular, we examine knowledge and reasoning-based skills such as knowledge-based QA and code generation, and we answer this question in the affirmative: $\textbf{scaling laws are skill-dependent}$. Next, to understand whether skill-dependent scaling is an artefact of the pretraining datamix, we conduct an extensive ablation of different datamixes and find that, also when correcting for datamix differences, $\textbf{knowledge and code exhibit fundamental differences in scaling behaviour}$. We conclude with an analysis of how our findings relate to standard compute-optimal scaling using a validation set, and find that $\textbf{a misspecified validation set can impact compute-optimal parameter count by nearly 50%,}$ depending on its skill composition. 

**Abstract (ZH)**: 缩放定律是大语言模型开发管道中的关键组成部分，最著名的是作为一种方法来预测训练决策，如“计算最优”权衡参数数量和数据集大小，以及近年来越来越多的其他关键决策。在本工作中，我们探讨计算最优缩放行为是否与技能相关。特别是，我们研究了基于知识和推理的技能，如基于知识的问答和代码生成，并得出结论：缩放定律是技能相关的。接下来，为了理解技能相关的缩放是否是预训练数据混合的产物，我们进行了广泛的数据混合消融实验，并发现，即使校正了数据混合差异，知识和代码在缩放行为上仍表现出根本性的差异。最后，我们将我们的发现与使用验证集的标准计算最优缩放进行了分析，并发现验证集的设定不当可能会影响计算最优参数数量高达近50%，这取决于其技能组成。 

---
# Exploring Mutual Empowerment Between Wireless Networks and RL-based LLMs: A Survey 

**Title (ZH)**: 无线网络与基于RL的LLM间相互赋能探索：一个综述 

**Authors**: Yu Qiao, Phuong-Nam Tran, Ji Su Yoon, Loc X. Nguyen, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09956)  

**Abstract**: Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have gained significant attention for their exceptional capabilities in natural language processing and multimodal data understanding. Meanwhile, the rapid expansion of information services has driven the growing need for intelligence, efficient, and adaptable wireless networks. Wireless networks require the empowerment of RL-based LLMs while these models also benefit from wireless networks to broaden their application scenarios. Specifically, RL-based LLMs can enhance wireless communication systems through intelligent resource allocation, adaptive network optimization, and real-time decision-making. Conversely, wireless networks provide a vital infrastructure for the efficient training, deployment, and distributed inference of RL-based LLMs, especially in decentralized and edge computing environments. This mutual empowerment highlights the need for a deeper exploration of the interplay between these two domains. We first review recent advancements in wireless communications, highlighting the associated challenges and potential solutions. We then discuss the progress of RL-based LLMs, focusing on key technologies for LLM training, challenges, and potential solutions. Subsequently, we explore the mutual empowerment between these two fields, highlighting key motivations, open challenges, and potential solutions. Finally, we provide insights into future directions, applications, and their societal impact to further explore this intersection, paving the way for next-generation intelligent communication systems. Overall, this survey provides a comprehensive overview of the relationship between RL-based LLMs and wireless networks, offering a vision where these domains empower each other to drive innovations. 

**Abstract (ZH)**: 基于强化学习的大型语言模型与无线网络的相互赋能：推动下一代智能通信系统的创新 

---
# Generative AI for Named Entity Recognition in Low-Resource Language Nepali 

**Title (ZH)**: 生成式AI在低资源语言尼泊尔语命名实体识别中的应用 

**Authors**: Sameer Neupane, Jeevan Chapagain, Nobal B. Niraula, Diwa Koirala  

**Link**: [PDF](https://arxiv.org/pdf/2503.09822)  

**Abstract**: Generative Artificial Intelligence (GenAI), particularly Large Language Models (LLMs), has significantly advanced Natural Language Processing (NLP) tasks, such as Named Entity Recognition (NER), which involves identifying entities like person, location, and organization names in text. LLMs are especially promising for low-resource languages due to their ability to learn from limited data. However, the performance of GenAI models for Nepali, a low-resource language, has not been thoroughly evaluated. This paper investigates the application of state-of-the-art LLMs for Nepali NER, conducting experiments with various prompting techniques to assess their effectiveness. Our results provide insights into the challenges and opportunities of using LLMs for NER in low-resource settings and offer valuable contributions to the advancement of NLP research in languages like Nepali. 

**Abstract (ZH)**: 生成式人工智能（GenAI），尤其是大型语言模型（LLMs），在自然语言处理（NLP）任务中取得了显著进展，例如命名实体识别（NER），该任务涉及识别文本中的人名、地名和组织名等实体。LLMs特别适合低资源语言，因为它们能够从有限的数据中学习。然而，生成式人工智能模型在尼泊尔语这一低资源语言中的性能尚未得到充分评估。本文探讨了最先进的LLMs在尼泊尔语NER中的应用，通过使用各种提示技术进行实验，评估其效果。我们的结果提供了关于在低资源环境中使用LLMs进行NER所面临挑战和机遇的见解，并为尼泊尔语等语言的NLP研究进展提供了宝贵贡献。 

---
# Can A Society of Generative Agents Simulate Human Behavior and Inform Public Health Policy? A Case Study on Vaccine Hesitancy 

**Title (ZH)**: 生成型代理社会能否模拟人类行为并为公共卫生政策提供信息？以疫苗犹豫为例的研究案例 

**Authors**: Abe Bohan Hou, Hongru Du, Yichen Wang, Jingyu Zhang, Zixiao Wang, Paul Pu Liang, Daniel Khashabi, Lauren Gardner, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2503.09639)  

**Abstract**: Can we simulate a sandbox society with generative agents to model human behavior, thereby reducing the over-reliance on real human trials for assessing public policies? In this work, we investigate the feasibility of simulating health-related decision-making, using vaccine hesitancy, defined as the delay in acceptance or refusal of vaccines despite the availability of vaccination services (MacDonald, 2015), as a case study. To this end, we introduce the VacSim framework with 100 generative agents powered by Large Language Models (LLMs). VacSim simulates vaccine policy outcomes with the following steps: 1) instantiate a population of agents with demographics based on census data; 2) connect the agents via a social network and model vaccine attitudes as a function of social dynamics and disease-related information; 3) design and evaluate various public health interventions aimed at mitigating vaccine hesitancy. To align with real-world results, we also introduce simulation warmup and attitude modulation to adjust agents' attitudes. We propose a series of evaluations to assess the reliability of various LLM simulations. Experiments indicate that models like Llama and Qwen can simulate aspects of human behavior but also highlight real-world alignment challenges, such as inconsistent responses with demographic profiles. This early exploration of LLM-driven simulations is not meant to serve as definitive policy guidance; instead, it serves as a call for action to examine social simulation for policy development. 

**Abstract (ZH)**: 能否通过生成代理模拟沙盒社会来模仿人类行为，从而减少对实际人类试验的依赖以评估公共政策？本研究探讨使用疫苗犹豫（MacDonald, 2015）作为案例研究，利用人口统计学数据构建代理群体，通过社会网络构建代理并模拟疫苗态度作为社会动态和疾病相关信息的函数，设计并评估旨在减少疫苗犹豫的各种公共卫生干预措施，从而模拟健康相关决策。为使模拟结果与实际情况相符，我们引入了模拟热身和态度调节。我们提出了一系列评估，以评估各种大型语言模型模拟的可靠性。实验表明，如Llama和Qwen等模型可以模拟人类行为的某些方面，但也指出了现实世界对齐的挑战，如与人口统计特征不一致的反应。本早期探索性研究并非旨在提供最终政策指导，而是呼吁通过社会模拟来研究政策开发。 

---
# Exploiting Edited Large Language Models as General Scientific Optimizers 

**Title (ZH)**: 利用编辑后的大型语言模型作为通用科学优化器 

**Authors**: Qitan Lv, Tianyu Liu, Hong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09620)  

**Abstract**: Large language models (LLMs) have been widely adopted in mathematical optimization in scientific scenarios for their extensive knowledge and advanced reasoning capabilities. Existing methods mainly focus on utilizing LLMs to solve optimization problems in a prompt-based manner, which takes observational feedback as additional textual descriptions. However, due to LLM's \textbf{high sensitivity to the prompts} and \textbf{tendency to get lost in lengthy prompts}, these methods struggle to effectively utilize the {observational} feedback from each optimization step, which severely hinders the applications for real-world scenarios. To address these challenges, we propose a conceptually simple and general {bi-level} optimization method, namely \textbf{G}eneral \textbf{S}cientific \textbf{O}ptimizers (GSO). Specifically, GSO first utilizes inner-level simulators as experimental platforms to evaluate the current solution and provide observational feedback. Then, LLMs serve as knowledgeable and versatile scientists, generating new solutions by refining potential errors from the feedback as the outer-level optimization. Finally, simulations together with the expert knowledge in LLMs are jointly updated with bi-level interactions via model editing. Extensive experiments show that GSO consistently outperforms existing state-of-the-art methods using \textit{six} different LLM backbones on \textit{seven} different tasks, demonstrating the effectiveness and a wide range of applications. 

**Abstract (ZH)**: 大型语言模型在科学场景中的数学优化应用：一种通用双层优化方法（GSO） 

---
# A Unified Framework with Novel Metrics for Evaluating the Effectiveness of XAI Techniques in LLMs 

**Title (ZH)**: 一种新颖度量指标下的统一框架：评估解释性AI技术在大语言模型中的有效性 

**Authors**: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Hassan shakil, Ali Al shami, Sanghyun Byun, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2503.05050)  

**Abstract**: The increasing complexity of LLMs presents significant challenges to their transparency and interpretability, necessitating the use of eXplainable AI (XAI) techniques to enhance trustworthiness and usability. This study introduces a comprehensive evaluation framework with four novel metrics for assessing the effectiveness of five XAI techniques across five LLMs and two downstream tasks. We apply this framework to evaluate several XAI techniques LIME, SHAP, Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Attention Mechanism Visualization (AMV) using the IMDB Movie Reviews and Tweet Sentiment Extraction datasets. The evaluation focuses on four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. Our results show that LIME consistently achieves high scores across multiple LLMs and evaluation metrics, while AMV demonstrates superior Robustness and near-perfect Consistency. LRP excels in Contrastivity, particularly with more complex models. Our findings provide valuable insights into the strengths and limitations of different XAI methods, offering guidance for developing and selecting appropriate XAI techniques for LLMs. 

**Abstract (ZH)**: LLMs复杂性的增加对其实现透明性和可解释性提出了重大挑战，亟需使用可解释AI（XAI）技术来增强其可信度和实用性。本研究引入了一个全面的评估框架，包含四个新型指标，用于评估五种XAI技术在五个LLM和两项下游任务中的有效性。我们应用该框架，使用IMDB电影评论和推文情感提取数据集，评估了LIME、SHAP、整合梯度、层相关性传播（LRP）和注意力机制可视化（AMV）等多种XAI技术。评估主要聚焦于四个关键指标：人类推理一致性（HA）、鲁棒性、一致性与对比度性。研究结果显示，LIME在多个LLM和评估指标上持续获得高分，而AMV在鲁棒性和一致性方面表现出色。LRP在对比度性上表现突出，尤其是在更复杂模型中。我们的研究成果提供了不同XAI方法优缺点的有价值的见解，为开发和选择适合LLM的XAI技术提供了指导。 

---
