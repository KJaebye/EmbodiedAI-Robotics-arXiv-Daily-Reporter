# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Title (ZH)**: 基于多模态大语言模型的.context-aware人类行为预测：挑战与见解 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

**Abstract (ZH)**: 共享环境中的人类行为预测对于安全高效的机器人交互至关重要。传统的数据驱动方法依赖于特定领域的数据集、活动类型和预测时间窗口进行预训练。相比之下，大型语言模型（LLMs）的近期突破允诺了跨领域的开放性泛化能力，以描述各种人类活动并在任何上下文中进行预测。特别是多模态LLMs能够从多种来源整合信息，从而实现更强的上下文意识和更优的场景理解。直接将通用的多模态LLMs应用于预测的一大困难在于它们处理大型输入序列的能力有限、对提示设计高度敏感且微调成本高昂。在本文中，我们提出了一个系统化的分析，探讨预训练的多模态LLMs在上下文感知的人类行为预测中的应用。为此，我们引入了一个模块化的多模态人类活动预测框架，使我们能够对各种LLMs、输入变化、增量式学习（ICL）和自回归技术进行基准测试。我们的评估表明，最佳框架配置能够在目标帧中达到92.8%的语义相似度和66.1%的精确标签准确性。 

---
# Contextualized Autonomous Drone Navigation using LLMs Deployed in Edge-Cloud Computing 

**Title (ZH)**: 基于边缘-云计算部署的LLM驱动的上下文自主无人机导航 

**Authors**: Hongqian Chen, Yun Tang, Antonios Tsourdos, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00607)  

**Abstract**: Autonomous navigation is usually trained offline in diverse scenarios and fine-tuned online subject to real-world experiences. However, the real world is dynamic and changeable, and many environmental encounters/effects are not accounted for in real-time due to difficulties in describing them within offline training data or hard to describe even in online scenarios. However, we know that the human operator can describe these dynamic environmental encounters through natural language, adding semantic context. The research is to deploy Large Language Models (LLMs) to perform real-time contextual code adjustment to autonomous navigation. The challenge not evaluated in literature is what LLMs are appropriate and where should these computationally heavy algorithms sit in the computation-communication edge-cloud computing architectures. In this paper, we evaluate how different LLMs can adjust both the navigation map parameters dynamically (e.g., contour map shaping) and also derive navigation task instruction sets. We then evaluate which LLMs are most suitable and where they should sit in future edge-cloud of 6G telecommunication architectures. 

**Abstract (ZH)**: 基于大语言模型的自主导航实时上下文代码调整研究 

---
# SACA: A Scenario-Aware Collision Avoidance Framework for Autonomous Vehicles Integrating LLMs-Driven Reasoning 

**Title (ZH)**: 面向场景的自主车辆碰撞避免框架：结合LLM驱动的推理方法 

**Authors**: Shiyue Zhao, Junzhi Zhang, Neda Masoud, Heye Huang, Xingpeng Xia, Chengkun He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00115)  

**Abstract**: Reliable collision avoidance under extreme situations remains a critical challenge for autonomous vehicles. While large language models (LLMs) offer promising reasoning capabilities, their application in safety-critical evasive maneuvers is limited by latency and robustness issues. Even so, LLMs stand out for their ability to weigh emotional, legal, and ethical factors, enabling socially responsible and context-aware collision avoidance. This paper proposes a scenario-aware collision avoidance (SACA) framework for extreme situations by integrating predictive scenario evaluation, data-driven reasoning, and scenario-preview-based deployment to improve collision avoidance decision-making. SACA consists of three key components. First, a predictive scenario analysis module utilizes obstacle reachability analysis and motion intention prediction to construct a comprehensive situational prompt. Second, an online reasoning module refines decision-making by leveraging prior collision avoidance knowledge and fine-tuning with scenario data. Third, an offline evaluation module assesses performance and stores scenarios in a memory bank. Additionally, A precomputed policy method improves deployability by previewing scenarios and retrieving or reasoning policies based on similarity and confidence levels. Real-vehicle tests show that, compared with baseline methods, SACA effectively reduces collision losses in extreme high-risk scenarios and lowers false triggering under complex conditions. Project page: this https URL. 

**Abstract (ZH)**: 可靠的极端情况下碰撞规避仍然是自动驾驶车辆面临的关键挑战。尽管大规模语言模型（LLMs）提供了有希望的推理能力，但其在安全关键的规避操作中的应用受限于延迟和鲁棒性问题。即便如此，LLMs 由于其在权衡情感、法律和伦理因素方面的能力，使得碰撞规避能够更加社会负责和情境意识。本文提出了一种情境感知碰撞规避（SACA）框架，通过融合预测情景评估、数据驱动推理和基于情景预览的部署，以改进碰撞规避决策制定。SACA 包含三个关键组件。首先，预测情景分析模块利用障碍可达性分析和运动意图预测来构建全面的情景提示。其次，在线推理模块利用先验碰撞规避知识和基于情景数据的微调来优化决策制定。第三，离线评估模块评估性能并将情景存储在记忆库中。此外，通过预计算策略方法改进可部署性，通过预览情景并基于相似性和置信度水平检索或推理策略。实车测试表明，与基线方法相比，SACA 在极端高风险情景中有效减少了碰撞损失，并在复杂条件下降低了误触发率。项目页面：this https URL。 

---
# Investigating Large Language Models in Diagnosing Students' Cognitive Skills in Math Problem-solving 

**Title (ZH)**: 探究大型语言模型在数学问题解决中诊断学生认知技能的应用 

**Authors**: Hyoungwook Jin, Yoonsu Kim, Dongyun Jung, Seungju Kim, Kiyoon Choi, Jinho Son, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00843)  

**Abstract**: Mathematics learning entails mastery of both content knowledge and cognitive processing of knowing, applying, and reasoning with it. Automated math assessment primarily has focused on grading students' exhibition of content knowledge by finding textual evidence, such as specific numbers, formulas, and statements. Recent advancements in problem-solving, image recognition, and reasoning capabilities of large language models (LLMs) show promise for nuanced evaluation of students' cognitive skills. Diagnosing cognitive skills needs to infer students' thinking processes beyond textual evidence, which is an underexplored task in LLM-based automated assessment. In this work, we investigate how state-of-the-art LLMs diagnose students' cognitive skills in mathematics. We constructed MathCog, a novel benchmark dataset comprising 639 student responses to 110 expert-curated middle school math problems, each annotated with detailed teachers' diagnoses based on cognitive skill checklists. Using MathCog, we evaluated 16 closed and open LLMs of varying model sizes and vendors. Our evaluation reveals that even the state-of-the-art LLMs struggle with the task, all F1 scores below 0.5, and tend to exhibit strong false confidence for incorrect cases ($r_s=.617$). We also found that model size positively correlates with the diagnosis performance ($r_s=.771$). Finally, we discuss the implications of these findings, the overconfidence issue, and directions for improving automated cognitive skill diagnosis. 

**Abstract (ZH)**: 数学学习涉及内容知识的掌握和认知处理能力的培养，包括应用和推理。自动数学评估主要侧重于通过查找特定数字、公式和陈述等文本证据来评估学生对内容知识的展示。大型语言模型（LLMs）在解决问题、图像识别和推理能力方面的最新进展显示出对学生的认知技能进行细致评估的潜力。诊断认知技能需要推断学生的思想过程，这在基于LLM的自动评估中是一个未被充分探索的任务。在本文中，我们研究了最先进的LLMs如何诊断学生在数学中的认知技能。我们构建了MathCog，这是一个新颖的基准数据集，包含639名学生的110个中学数学问题的响应，每个问题都基于认知技能清单进行了详细的教师诊断标注。使用MathCog，我们评估了16个不同模型大小和供应商的封闭和开放LLMs。评估结果显示，即使是最先进的LLMs也难以完成这项任务，所有F1分数均低于0.5，并且在错误情况下表现出强烈的事后自信（$r_s=.617$）。我们还发现，模型大小与诊断性能正相关（$r_s=.771$）。最后，我们讨论了这些发现的意义、过自信问题以及改进自动认知技能诊断的方向。 

---
# Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scale Test-Time Compute 

**Title (ZH)**: 我们需要这么多样本吗？多模态大语言模型重复采样高效扩展测试时计算量 

**Authors**: Jianhao Chen, Zishuo Xun, Bocheng Zhou, Han Qi, Qiaosheng Zhang, Yang Chen, Wei Hu, Yuzhong Qu, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00762)  

**Abstract**: This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm. 

**Abstract (ZH)**: 本文提出了一种简单、有效且成本效益高的策略，通过扩展测试时计算来提升大语言模型的性能。该策略基于重复采样后再投票的框架，并引入了一个新的元素：结合多个模型，即使是最弱的模型，以利用它们从多样化的训练数据和范式中可能产生的互补优势。通过一致性作为信号，该策略动态切换模型。理论分析突显了该策略的效率和性能优势。在六个数据集上的广泛实验表明，该策略不仅优于自一致性以及最先进的多代理辩论方法，而且显著降低了推理成本。此外，ModelSwitch仅需少量可比的大语言模型即可实现最佳性能，并可通过验证方法扩展，展示了利用多个大语言模型在生成-验证范式中的潜力。 

---
# LLM-Guided Search for Deletion-Correcting Codes 

**Title (ZH)**: LLM引导的删除纠错码搜索 

**Authors**: Franziska Weindel, Reinhard Heckel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00613)  

**Abstract**: Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes. 

**Abstract (ZH)**: 寻找具有最大规模的删除纠错码的问题已有超过70年未解之谜，即使是单删除的情况亦是如此。在本文中，我们提出了一种新的构建删除纠错码的方法。一种代码是由满足某些约束条件的序列集合组成，我们通过根据优先级函数逐次添加最高优先级的序列来构建它。为了找到良好的优先级函数，我们利用Romera等人于2024年提出的由大语言模型（LLM）引导的进化搜索FunSearch。FunSearch通过迭代生成、评估和改进优先级函数来构建大规模的删除纠错码。对于单删除情况，我们的进化搜索发现的函数构建的码越大，可与已知的最大规模相匹配；对于最大值未知的汉森-特内戈尔茨码，其规模达到已知最大（猜想最优）的规模；并且能够独立地以等效形式重新发现它们。对于双删除情况，我们发现的函数构建的码长度分别为\( n = 12, 13 \)和\( 16 \)时，具有新的最佳已知规模，从而建立了改进的下界。这些结果展示了LLM引导的搜索在信息理论和码设计中的潜力，并代表了此类方法首次用于构建纠错码的应用。 

---
# Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Title (ZH)**: Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Authors**: Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00509)  

**Abstract**: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surpassing human intelligence. However, is the LLMs' remarkable reasoning ability indeed comes from true intelligence by human standards, or are they simply reciting solutions witnessed during training at an Internet level? To study this problem, we propose RoR-Bench, a novel, multi-modal benchmark for detecting LLM's recitation behavior when asked simple reasoning problems but with conditions subtly shifted, and conduct empirical analysis on our benchmark. Surprisingly, we found existing cutting-edge LLMs unanimously exhibits extremely severe recitation behavior; by changing one phrase in the condition, top models such as OpenAI-o1 and DeepSeek-R1 can suffer $60\%$ performance loss on elementary school-level arithmetic and reasoning problems. Such findings are a wake-up call to the LLM community that compels us to re-evaluate the true intelligence level of cutting-edge LLMs. 

**Abstract (ZH)**: 近年来，大语言模型基准从小学级问题到前沿问题难度的迅速升级，为研究人员编织了一个奇迹般的前景，即我们几乎达到了超越人类智能的水平。然而，大语言模型卓越的推理能力究竟是基于人类标准的真正智能，还是仅仅在网上训练时回忆并重现的解决方案？为研究这一问题，我们提出了RoR-Bench这一新颖的多模态基准，用于检测大语言模型在回答简单推理问题但在条件上细微变化时的回忆行为，并对基准进行了实证分析。令人惊讶的是，我们发现现有最先进的大语言模型普遍表现出极其严重的回忆行为；仅通过改变一个条件短语，如OpenAI-o1和DeepSeek-R1等顶级模型在小学级算术和推理问题上的性能损失高达60%。这一发现是对大语言模型社区的警醒，促使我们重新评估当前最先进大语言模型的真实智能水平。 

---
# Hawkeye:Efficient Reasoning with Model Collaboration 

**Title (ZH)**: 慧眼：模型协作的高效推理 

**Authors**: Jianshu She, Zhuohao Li, Zhemin Huang, Qi Li, Peiran Xu, Haonan Li, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.00424)  

**Abstract**: Chain-of-Thought (CoT) reasoning has demonstrated remarkable effectiveness in enhancing the reasoning abilities of large language models (LLMs). However, its efficiency remains a challenge due to the generation of excessive intermediate reasoning tokens, which introduce semantic redundancy and overly detailed reasoning steps. Moreover, computational expense and latency are significant concerns, as the cost scales with the number of output tokens, including those intermediate steps. In this work, we observe that most CoT tokens are unnecessary, and retaining only a small portion of them is sufficient for producing high-quality responses. Inspired by this, we propose HAWKEYE, a novel post-training and inference framework where a large model produces concise CoT instructions to guide a smaller model in response generation. HAWKEYE quantifies redundancy in CoT reasoning and distills high-density information via reinforcement learning. By leveraging these concise CoTs, HAWKEYE is able to expand responses while reducing token usage and computational cost significantly. Our evaluation shows that HAWKEYE can achieve comparable response quality using only 35% of the full CoTs, while improving clarity, coherence, and conciseness by approximately 10%. Furthermore, HAWKEYE can accelerate end-to-end reasoning by up to 3.4x on complex math tasks while reducing inference cost by up to 60%. HAWKEYE will be open-sourced and the models will be available soon. 

**Abstract (ZH)**: HAWKEYE：一种新型的后训练和推理框架，用于精简链式推理指令以提高生成质量并加速推理 

---
# CyberBOT: Towards Reliable Cybersecurity Education via Ontology-Grounded Retrieval Augmented Generation 

**Title (ZH)**: CyberBOT: 基于本体导向检索增强生成的可靠网络安全教育探索 

**Authors**: Chengshuai Zhao, Riccardo De Maria, Tharindu Kumarage, Kumar Satvik Chaudhary, Garima Agrawal, Yiwen Li, Jongchan Park, Yuli Deng, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00389)  

**Abstract**: Advancements in large language models (LLMs) have enabled the development of intelligent educational tools that support inquiry-based learning across technical domains. In cybersecurity education, where accuracy and safety are paramount, systems must go beyond surface-level relevance to provide information that is both trustworthy and domain-appropriate. To address this challenge, we introduce CyberBOT, a question-answering chatbot that leverages a retrieval-augmented generation (RAG) pipeline to incorporate contextual information from course-specific materials and validate responses using a domain-specific cybersecurity ontology. The ontology serves as a structured reasoning layer that constrains and verifies LLM-generated answers, reducing the risk of misleading or unsafe guidance. CyberBOT has been deployed in a large graduate-level course at Arizona State University (ASU), where more than one hundred students actively engage with the system through a dedicated web-based platform. Computational evaluations in lab environments highlight the potential capacity of CyberBOT, and a forthcoming field study will evaluate its pedagogical impact. By integrating structured domain reasoning with modern generative capabilities, CyberBOT illustrates a promising direction for developing reliable and curriculum-aligned AI applications in specialized educational contexts. 

**Abstract (ZH)**: 大型语言模型的进步使开发支持基于探究的学习的智能教育工具成为可能，这些工具适用于各类技术领域。在信息安全教育中，由于准确性与安全性至关重要，系统必须超越表面相关性，提供既可信又符合特定领域的信息。为应对这一挑战，我们引入了CyberBOT，这是一种利用检索增强生成（RAG）pipeline的问答聊天机器人，它可以整合课程特定材料的上下文信息，并使用特定领域的安全Ontology验证响应。此Ontology作为结构化推理层，限制并验证由大型语言模型生成的答案，从而降低误导性或不安全指导的风险。CyberBOT已在亚利桑那州立大学（ASU）的一门大型研究生课程中部署，超过百名学生通过专用的WEB平台积极与该系统互动。实验室环境下的计算评估突显了CyberBOT的潜在容量，而即将进行的实地研究将评估其教学影响。通过将结构化领域推理与现代生成能力相结合，CyberBOT展示了在特定教育环境中开发可靠且符合课程要求的AI应用的一个有前景的方向。 

---
# Collaborative LLM Numerical Reasoning with Local Data Protection 

**Title (ZH)**: 本地数据保护下的协作大语言模型数值推理 

**Authors**: Min Zhang, Yuzhe Lu, Yun Zhou, Panpan Xu, Lin Lee Cheong, Chang-Tien Lu, Haozhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00299)  

**Abstract**: Numerical reasoning over documents, which demands both contextual understanding and logical inference, is challenging for low-capacity local models deployed on computation-constrained devices. Although such complex reasoning queries could be routed to powerful remote models like GPT-4, exposing local data raises significant data leakage concerns. Existing mitigation methods generate problem descriptions or examples for remote assistance. However, the inherent complexity of numerical reasoning hinders the local model from generating logically equivalent queries and accurately inferring answers with remote guidance. In this paper, we present a model collaboration framework with two key innovations: (1) a context-aware synthesis strategy that shifts the query domains while preserving logical consistency; and (2) a tool-based answer reconstruction approach that reuses the remote-generated problem-solving pattern with code snippets. Experimental results demonstrate that our method achieves better reasoning accuracy than solely using local models while providing stronger data protection than fully relying on remote models. Furthermore, our method improves accuracy by 16.2% - 43.6% while reducing data leakage by 2.3% - 44.6% compared to existing data protection approaches. 

**Abstract (ZH)**: 基于文档的数值推理需要同时具备上下文理解与逻辑推理能力，对于部署在计算受限设备上的低容量本地模型来说是一个挑战。虽然这类复杂的推理查询可以路由到强大的远程模型（如GPT-4）进行处理，但将本地数据暴露出去引发了显著的数据泄露问题。现有缓解方法生成问题描述或示例以寻求远程协助。然而，数值推理的固有复杂性阻碍了本地模型生成逻辑等价的查询以及在远程指导下的准确推理。在本文中，我们提出了一种模型协作框架，包含两个关键创新：（1）一种上下文感知的合成策略，能够在保持逻辑一致性的同时转换查询域；（2）一种基于工具的答案重构方法，该方法利用代码片段重用远程生成的解决问题模式。实验结果表明，与仅使用本地模型相比，我们的方法在推理准确性上表现更好，同时在数据保护强度上优于完全依赖远程模型的方法。此外，与现有数据保护方法相比，我们的方法将准确性提高了16.2% - 43.6%，同时将数据泄露减少了2.3% - 44.6%。 

---
# Large Language Models in Numberland: A Quick Test of Their Numerical Reasoning Abilities 

**Title (ZH)**: 大语言模型在数 land 的快速测试：它们的数值推理能力 

**Authors**: Roussel Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.00226)  

**Abstract**: An essential element of human mathematical reasoning is our number sense -- an abstract understanding of numbers and their relationships -- which allows us to solve problems involving vast number spaces using limited computational resources. Mathematical reasoning of Large Language Models (LLMs) is often tested on high-level problems (such as Olympiad challenges, geometry, word problems, and puzzles), but their low-level number sense remains less explored. We introduce "Numberland," a 100-problem test to evaluate the numerical reasoning abilities of LLM-based agents. The tasks -- basic operations, advanced calculations (e.g., exponentiation, complex numbers), prime number checks, and the 24 game -- aim to test elementary skills and their integration in solving complex and uncertain problems. We evaluated five LLM-based agents: OpenAI's o1 and o1-mini, Google Gemini, Microsoft Copilot, and Anthropic Claude. They scored 74-95% on the first three tasks that allow deterministic steps to solutions. In the 24 game, which needs trial-and-error search, performance dropped to 10-73%. We tested the top 24 solver (o1 with 73% accuracy) on 25 harder problems, and its score fell to 27%, confirming search as a bottleneck. These results, along with the types of mistakes, suggest a fragile number of LLMs, which is a bit surprising given their prowess in challenging benchmarks. The limits of LLM numerical reasoning highlight the scope of simple, targeted tests to evaluate and explain LLM math skills to ensure safe use. 

**Abstract (ZH)**: 一种人类数学推理的关键元素是我们的数感——一种对数字及其关系的抽象理解——它允许我们在有限的计算资源下解决涉及巨大数字空间的问题。大规模语言模型（LLMs）的数学推理经常通过奥林匹克挑战、几何学、文字问题和谜题等高级问题进行测试，但其低级数感的研究相对较少。我们引入了“Numberland”，一个包含100个问题的测试，以评估基于LLM的代理的数值推理能力。任务包括基本操作、高级计算（如幂运算、复数）、素数检验和24点游戏，旨在测试基础技能及其在解决复杂和不确定问题中的整合。我们评估了五种基于LLM的代理：OpenAI的o1和o1-mini、Google Gemini、Microsoft Copilot和Anthropic Claude。它们在前三项允许确定步骤的问题上获得了74-95%的分数。在需要试错搜索的24点游戏中，性能下降到10-73%。我们测试了最准确的24点游戏的求解器（o1，准确率为73%）在25个更难的问题上，其得分降至27%，确认了搜索是一个瓶颈。这些结果以及错误类型表明，LLMs在数值方面的脆弱性，这在考虑到它们在具有挑战性的基准测试中的表现时有些出人意料。LLMs数值推理的局限性强调了对简单和针对性测试的需求，以评估和解释LLMs的数学技能，确保其安全使用。 

---
# LLMs for Explainable AI: A Comprehensive Survey 

**Title (ZH)**: 可解释AI中的LLMs综述：一项全面调研 

**Authors**: Ahsan Bilal, David Ebert, Beiyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00125)  

**Abstract**: Large Language Models (LLMs) offer a promising approach to enhancing Explainable AI (XAI) by transforming complex machine learning outputs into easy-to-understand narratives, making model predictions more accessible to users, and helping bridge the gap between sophisticated model behavior and human interpretability. AI models, such as state-of-the-art neural networks and deep learning models, are often seen as "black boxes" due to a lack of transparency. As users cannot fully understand how the models reach conclusions, users have difficulty trusting decisions from AI models, which leads to less effective decision-making processes, reduced accountabilities, and unclear potential biases. A challenge arises in developing explainable AI (XAI) models to gain users' trust and provide insights into how models generate their outputs. With the development of Large Language Models, we want to explore the possibilities of using human language-based models, LLMs, for model explainabilities. This survey provides a comprehensive overview of existing approaches regarding LLMs for XAI, and evaluation techniques for LLM-generated explanation, discusses the corresponding challenges and limitations, and examines real-world applications. Finally, we discuss future directions by emphasizing the need for more interpretable, automated, user-centric, and multidisciplinary approaches for XAI via LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）为增强可解释人工智能（XAI）提供了有前途的方法，通过将复杂的机器学习输出转换为易于理解的故事，使模型预测更加易于用户的理解，并有助于弥合高级模型行为与人类可解释性之间的差距。由于透明度不足，AI模型，如最先进的神经网络和深度学习模型，常被视为“黑箱”。由于用户无法完全理解模型如何得出结论，用户难以信任AI模型的决策，从而导致决策过程效率降低、责任减少以及潜在偏见不清晰。在开发可解释AI（XAI）模型以赢得用户信任并提供有关模型生成输出的见解方面存在挑战。随着大型语言模型的发展，我们旨在探索使用基于人类语言的模型—LLMs—提高模型可解释性的可能性。本文综述了现有用于XAI的LLM方法及其评估技术，讨论了相应挑战和限制，并分析了实际应用。最后，我们强调通过LLMs实现XAI的更可解释性、自动化、用户中心和跨学科方法的重要性，指出未来的研究方向。 

---
# When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning 

**Title (ZH)**: 何时求解，何时验证：面向LLM推理的计算最优问题求解与生成验证 

**Authors**: Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.01005)  

**Abstract**: Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at this https URL. 

**Abstract (ZH)**: 扩展测试时计算量已成为增强大型语言模型（LLMs）推理能力的关键策略，特别是在数学问题求解等任务中。传统的自一致性（SC）方法生成多个问题解决方案，并通过 majority voting 选择最常见的答案。另一种常见方法是对每个解决方案打分（使用验证器），并选择得分最高的。最新的生成奖励模型（GenRM）将验证重新构想为下一个词预测任务，从而可以在新的轴上扩展推理。具体而言，GenRM 为每个解决方案生成多个验证链式思考来评分。在有限的推理预算下，这引入了一个基本的权衡：你是否应该将预算用于通过 SC 扩展解决方案的数量，还是生成更少的解决方案并为验证分配计算资源通过 GenRM？为了解决这个问题，我们在固定推理预算下评估了 GenRM 对 SC 的表现。有趣的是，我们发现，在大多数实际推理预算下，SC 在各种模型和数据集上的计算效率高于 GenRM。例如，GenRM 需要消耗高达 8 倍的推理计算量才能与 SC 相当，并且需要显著更多的计算量才能优于 SC。此外，我们推导了 GenRM 模式下的推理扩展法则，发现计算最优的推理倾向于更激进地扩展解决方案生成而不是扩展验证的数量。我们的工作提供了优化测试时扩展的实际指导，通过平衡解决方案生成和验证。代码可在以下链接获取。 

---
# Token embeddings violate the manifold hypothesis 

**Title (ZH)**: Token嵌入违反流形假说 

**Authors**: Michael Robinson, Sourya Dey, Tony Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01002)  

**Abstract**: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions about the LLM is likely flawed, regardless of its architecture. Here, we elucidate the structure of the token embeddings, the input domain for LLMs, both empirically and theoretically. We present a generalized and statistically testable model where the neighborhood of each token splits into well-defined signal and noise dimensions.
This model is based on a generalization of a manifold called a fiber bundle, so we denote our hypothesis test as the ``fiber bundle null.'' Failing to reject the null is uninformative, but rejecting it at a specific token indicates that token has a statistically significant local structure, and so is of interest to us. By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the token subspace is provably not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, and if one prompt contains a token implicated by our test, that prompt will likely exhibit more output variability proportional to the local signal dimension of the token. 

**Abstract (ZH)**: 要全面理解大型语言模型（LLM）的行为，需要我们理解其输入空间。如果这个输入空间与我们的假设不同，那么我们对LLM的理解和结论很可能是有缺陷的，无论其架构如何。在这里，我们详述了token嵌入的结构，即LLM的输入域，通过实证和理论的方法。我们提出了一个一般化且可统计检验的模型，其中每个token的邻域被分为明确的信号维度和噪声维度。

该模型基于纤维丛这一流形的一般化概念，因此我们将我们的假设检验称为“纤维丛零假设”。未能拒绝零假设是无信息性的，但特定token处拒绝零假设表明该token具有统计显著的局部结构，从而引起我们的兴趣。通过在多个开源LLM上运行我们的测试，每种LLM具有独特的token嵌入，我们发现零假设经常被拒绝，因此token子空间不是纤维丛，也不一定是流形。据此，当LLM接收到两个语义等价的提示时，如果一个提示包含我们的测试所指的token，则该提示的输出变异性很可能会与该token的局部信号维度成比例。 

---
# Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models 

**Title (ZH)**: 零样本基准测试：一种灵活可扩展的语言模型自动评估框架 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.01001)  

**Abstract**: As language models improve and become capable of performing more complex tasks across modalities, evaluating them automatically becomes increasingly challenging. Developing strong and robust task-specific automatic metrics gets harder, and human-annotated test sets -- which are expensive to create -- saturate more quickly. A compelling alternative is to design reliable strategies to automate the creation of test data and evaluation, but previous attempts either rely on pre-existing data, or focus solely on individual tasks. We present Zero-shot Benchmarking (ZSB), a framework for creating high-quality benchmarks for any task by leveraging language models for both synthetic test data creation and evaluation. ZSB is simple and flexible: it requires only the creation of a prompt for data generation and one for evaluation; it is scalable to tasks and languages where collecting real-world data is costly or impractical; it is model-agnostic, allowing the creation of increasingly challenging benchmarks as models improve. To assess the effectiveness of our framework, we create benchmarks for five text-only tasks and a multi-modal one: general capabilities in four languages (English, Chinese, French, and Korean), translation, and general vision-language capabilities in English. We then rank a broad range of open and closed systems on our benchmarks. ZSB rankings consistently correlate strongly with human rankings, outperforming widely-adopted standard benchmarks. Through ablations, we find that strong benchmarks can be created with open models, and that judge model size and dataset variety are crucial drivers of performance. We release all our benchmarks, and code to reproduce our experiments and to produce new benchmarks. 

**Abstract (ZH)**: 基于语言模型的零样本基准测试框架：创建多模态任务的高质量基准 

---
# MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs 

**Title (ZH)**: MedReason: 通过知识图谱提取LLM中的事实医学推理步骤 

**Authors**: Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, Yihan Cao, Hui Ren, Xiang Li, Xiaoxiao Li, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00993)  

**Abstract**: Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available. 

**Abstract (ZH)**: 大规模高质量医学推理数据集MedReason：提升大型语言模型的医学问题解决能力 

---
# SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching 

**Title (ZH)**: SentenceKV：通过句子级语义KV缓存实现高效的LLM推理 

**Authors**: Yuxuan Zhu, Ali Falahati, David H. Yang, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.00970)  

**Abstract**: Large language models face significant computational and memory challenges when processing long contexts. During inference, efficient management of the key-value (KV) cache, which stores intermediate activations for autoregressive generation, is critical to reducing memory overhead and improving computational efficiency. Traditional token-level efficient KV caching methods overlook semantic information, treating tokens independently without considering their semantic relationships. Meanwhile, existing semantic-preserving KV cache management approaches often suffer from substantial memory usage and high time-to-first-token. To address these limitations, we propose SentenceKV, a novel sentence-level semantic KV caching approach designed to enhance inference efficiency while preserving semantic coherence. During prefilling, SentenceKV groups tokens based on sentence-level semantic similarity, compressing sentence representations into concise semantic vectors stored directly on the GPU, while individual KV pairs are offloaded to CPU. During decoding, SentenceKV generates tokens by selectively retrieving semantically relevant sentence-level KV entries, leveraging the semantic similarity between the prefilling-stage semantic vectors and decoding-stage queries. This ensures efficient and contextually accurate predictions, minimizing the loading of redundant or irrelevant data into GPU memory and significantly reducing memory overhead while maintaining stable inference latency, even for extremely long contexts. Extensive evaluations on benchmarks including PG-19, LongBench, and Needle-In-A-Haystack demonstrate that SentenceKV significantly outperforms state-of-the-art methods in both efficiency and memory usage, without compromising model accuracy. 

**Abstract (ZH)**: SentenceKV：面向长上下文的句级语义键值缓存方法 

---
# CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models 

**Title (ZH)**: CrackSQL: 由大规模语言模型驱动的混合SQL方言翻译系统 

**Authors**: Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00882)  

**Abstract**: Dialect translation plays a key role in enabling seamless interaction across heterogeneous database systems. However, translating SQL queries between different dialects (e.g., from PostgreSQL to MySQL) remains a challenging task due to syntactic discrepancies and subtle semantic variations. Existing approaches including manual rewriting, rule-based systems, and large language model (LLM)-based techniques often involve high maintenance effort (e.g., crafting custom translation rules) or produce unreliable results (e.g., LLM generates non-existent functions), especially when handling complex queries. In this demonstration, we present CrackSQL, the first hybrid SQL dialect translation system that combines rule and LLM-based methods to overcome these limitations. CrackSQL leverages the adaptability of LLMs to minimize manual intervention, while enhancing translation accuracy by segmenting lengthy complex SQL via functionality-based query processing. To further improve robustness, it incorporates a novel cross-dialect syntax embedding model for precise syntax alignment, as well as an adaptive local-to-global translation strategy that effectively resolves interdependent query operations. CrackSQL supports three translation modes and offers multiple deployment and access options including a web console interface, a PyPI package, and a command-line prompt, facilitating adoption across a variety of real-world use cases 

**Abstract (ZH)**: 方言翻译在实现异构数据库系统之间无缝交互中起着关键作用。然而，不同方言之间的SQL查询翻译（例如，从PostgreSQL到MySQL）由于句法差异和微妙的语义变化仍是一项具有挑战性的任务。现有的方法包括手动重写、基于规则的系统和基于大型语言模型（LLM）的技术，往往需要高维护 effort（例如，制定自定义翻译规则）或产生不可靠的结果（例如，LLM生成不存在的函数），特别是在处理复杂查询时。在这项演示中，我们介绍了CrackSQL，这是一种结合规则和LLM方法的首个混合SQL方言翻译系统，以克服这些限制。CrackSQL利用LLM的适应性来减少人工干预，通过基于功能的查询处理将长而复杂的SQL拆分，以提高翻译准确性。为了进一步提高鲁棒性，它引入了一种新的跨方言句法嵌入模型进行精确的语法对齐，以及一种有效的解决相互依赖查询操作的自适应局部到全局翻译策略。CrackSQL支持三种翻译模式，并提供多种部署和访问选项，包括Web控制台接口、PyPI包和命令行提示，便于在各种实际应用案例中采用。 

---
# m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models 

**Title (ZH)**: M1: 在大规模语言模型中释放测试时缩放的潜力以促进医疗推理 

**Authors**: Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00869)  

**Abstract**: Test-time scaling has emerged as a powerful technique for enhancing the reasoning capabilities of large language models. However, its effectiveness in medical reasoning remains uncertain, as the medical domain fundamentally differs from mathematical tasks in terms of knowledge representation and decision-making processes. In this paper, we provide the first comprehensive investigation of test-time scaling for medical reasoning and present m1, a simple yet effective approach that increases a model's medical reasoning capability at inference. Our evaluation across diverse medical tasks demonstrates that test-time scaling consistently enhances medical reasoning, enabling lightweight fine-tuned models under 10B parameters to establish new state-of-the-art performance, while our 32B model rivals previous 70B-scale medical LLMs. However, we identify an optimal reasoning token budget of approximately 4K, beyond which performance may degrade due to overthinking. Budget forcing, which extends test-time computation through iterative prompts, helps models double-check answers but does not necessarily improve the overall medical QA performance and, in some cases, even introduces errors into previously correct responses. Our case-by-case analysis identifies insufficient medical knowledge as a key bottleneck that prevents further performance gains through test-time scaling. We find that increasing data scale, improving data quality, and expanding model capacity consistently enhance medical knowledge grounding, enabling continued performance improvements, particularly on challenging medical benchmarks where smaller models reach saturation. These findings underscore fundamental differences between medical and mathematical reasoning in LLMs, highlighting that enriched medical knowledge, other than increased reasoning depth alone, is essential for realizing the benefits of test-time scaling. 

**Abstract (ZH)**: 测试时缩放已成为增强大规模语言模型推理能力的有力技术，但在医疗推理领域的有效性尚不确定，因为医疗领域在知识表示和决策过程中与数学任务存在根本性差异。本文首次全面探讨了医疗推理中的测试时缩放，并提出了m1，这是一种简单而有效的方法，可以提高模型在推理时的医疗推理能力。我们的跨多种医疗任务评估表明，测试时缩放能够一致地增强医疗推理能力，使轻量级细调模型在少于10B参数的情况下取得新的最佳性能，而我们的32B模型能与之前的70B规模的医疗LLM相媲美。然而，我们发现了一个大约4K的最优推理令牌预算，超过这个预算可能会因为过度推理而导致性能下降。预算强制策略通过迭代提示扩展测试时计算，有助于模型复查答案，但并不一定能改善整体医疗问答性能，在某些情况下甚至会引入错误到先前正确的回答中。我们的逐案分析发现，缺乏医疗知识是阻碍通过测试时缩放进一步性能提升的关键瓶颈。我们发现，增加数据规模、提高数据质量以及扩大模型容量能够一致地增强医疗知识基础，使模型在难以处理的医疗基准上持续取得性能提升。这些发现强调了LLM中的医疗与数学推理之间根本差异，并突出了除了增加推理深度之外，丰富医疗知识对于实现测试时缩放效益的必要性。 

---
# LLMs4SchemaDiscovery: A Human-in-the-Loop Workflow for Scientific Schema Mining with Large Language Models 

**Title (ZH)**: LLMs4SchemaDiscovery：大型语言模型辅助的科学模式发现人类在环工作流 

**Authors**: Sameer Sadruddin, Jennifer D'Souza, Eleni Poupaki, Alex Watkins, Hamed Babaei Giglou, Anisa Rula, Bora Karasulu, Sören Auer, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2504.00752)  

**Abstract**: Extracting structured information from unstructured text is crucial for modeling real-world processes, but traditional schema mining relies on semi-structured data, limiting scalability. This paper introduces schema-miner, a novel tool that combines large language models with human feedback to automate and refine schema extraction. Through an iterative workflow, it organizes properties from text, incorporates expert input, and integrates domain-specific ontologies for semantic depth. Applied to materials science--specifically atomic layer deposition--schema-miner demonstrates that expert-guided LLMs generate semantically rich schemas suitable for diverse real-world applications. 

**Abstract (ZH)**: 从无结构文本中提取结构化信息对于建模现实世界过程至关重要，但传统的模式挖掘依赖于半结构化数据，限制了其可扩展性。本文介绍了schema-miner，这是一种结合大型语言模型和人类反馈来自动化和精炼模式提取的新工具。通过迭代工作流，它组织文本中的属性，整合专家输入，并结合领域特定本体以实现语义深度。应用于材料科学，特别是原子层沉积，schema-miner 显示出在专家指导下，大型语言模型生成适合多种实际应用场景的语义丰富的模式。 

---
# Command A: An Enterprise-Ready Large Language Model 

**Title (ZH)**: 企业级大型语言模型 Command A 

**Authors**: Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas, Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D'souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yannis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking  

**Link**: [PDF](https://arxiv.org/pdf/2504.00698)  

**Abstract**: In this report we describe the development of Command A, a powerful large language model purpose-built to excel at real-world enterprise use cases. Command A is an agent-optimised and multilingual-capable model, with support for 23 languages of global business, and a novel hybrid architecture balancing efficiency with top of the range performance. It offers best-in-class Retrieval Augmented Generation (RAG) capabilities with grounding and tool use to automate sophisticated business processes. These abilities are achieved through a decentralised training approach, including self-refinement algorithms and model merging techniques. We also include results for Command R7B which shares capability and architectural similarities to Command A. Weights for both models have been released for research purposes. This technical report details our original training pipeline and presents an extensive evaluation of our models across a suite of enterprise-relevant tasks and public benchmarks, demonstrating excellent performance and efficiency. 

**Abstract (ZH)**: 本报告描述了Command A的开发，Command A是一款专为 excellence于真实企业应用场景设计的强大语言模型。Command A是一款优化过的代理模型，支持23种全球商业语言，并具备一种新颖的混合架构，平衡了效率与顶级性能。该模型具备最佳的检索增强生成（RAG）能力，支持接地和工具使用以自动化复杂业务流程。这些能力是通过去中心化的训练方法实现的，包括自我改进算法和模型合并技术。我们还介绍了与Command A具有能力和架构相似性的Command R7B的成果。这两个模型的权重均已发布用于研究目的。本技术报告详细介绍了我们的原始训练管道，并对模型在一系列企业相关任务和公共基准上的进行了广泛评估，展示了出色的表现和效率。 

---
# DynMoLE: Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism 

**Title (ZH)**: DynMoLE: 结合混合路由机制提升LoRA专家混合模型微调性能 

**Authors**: Dengchun Li, Naizheng Wang, Zihao Zhang, Haoyang Yin, Lei Duan, Meng Xiao, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00661)  

**Abstract**: Instruction-based fine-tuning of large language models (LLMs) has achieved remarkable success in various natural language processing (NLP) tasks. Parameter-efficient fine-tuning (PEFT) methods, such as Mixture of LoRA Experts (MoLE), combine the efficiency of Low-Rank Adaptation (LoRA) with the versatility of Mixture of Experts (MoE) models, demonstrating significant potential for handling multiple downstream tasks. However, the existing routing mechanisms for MoLE often involve a trade-off between computational efficiency and predictive accuracy, and they fail to fully address the diverse expert selection demands across different transformer layers. In this work, we propose DynMoLE, a hybrid routing strategy that dynamically adjusts expert selection based on the Tsallis entropy of the router's probability distribution. This approach mitigates router uncertainty, enhances stability, and promotes more equitable expert participation, leading to faster convergence and improved model performance. Additionally, we introduce an auxiliary loss based on Tsallis entropy to further guide the model toward convergence with reduced uncertainty, thereby improving training stability and performance. Our extensive experiments on commonsense reasoning benchmarks demonstrate that DynMoLE achieves substantial performance improvements, outperforming LoRA by 9.6% and surpassing the state-of-the-art MoLE method, MoLA, by 2.3%. We also conduct a comprehensive ablation study to evaluate the contributions of DynMoLE's key components. 

**Abstract (ZH)**: 基于指令的大型语言模型细调已在多种自然语言处理任务中取得了显著成功。参数效率细调（PEFT）方法，如Mixture of LoRA Experts (MoLE)，结合了LoRA的高效性和MoE模型的灵活性，展示了处理多种下游任务的巨大潜力。然而，MoLE现有的路由机制往往在计算效率和预测准确性之间存在权衡，并未能充分解决不同Transformer层的多样化专家选择需求。在本文中，我们提出DynMoLE，这是一种基于Tsallis熵的动态路由策略，根据路由器概率分布的Tsallis熵动态调整专家选择。这种方法减轻了路由器不确定性，提升了稳定性，并促进了更加公平的专家参与，从而加快了收敛速度并提高了模型性能。此外，我们引入了基于Tsallis熵的辅助损失，进一步引导模型以降低不确定性的方式向收敛目标发展，从而提高训练稳定性和性能。我们在常识推理基准上的 extensive 实验表明，DynMoLE取得了显著的性能提升，比LoRA高出9.6%，并优于最先进的MoLE方法MoLA 2.3%。我们还进行了全面的消融研究以评估DynMoLE关键组件的贡献。 

---
# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation 

**Title (ZH)**: 多语言上下文利用在检索增强生成中的一致性研究 

**Authors**: Jirui Qi, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00597)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from out-language passages, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements. 

**Abstract (ZH)**: 基于大型语言模型的检索增强生成（RAG）在多语言问答任务中的表现通过利用从语料库中检索的相关段落得到了显著提升。在多语言RAG（mRAG）中，检索到的段落可以是与用户查询语言不同的语言，这使得大型语言模型难以有效利用提供的信息。最近的研究表明，从多语言语料库中检索段落可以提高RAG的表现，尤其是对于低资源语言。然而，大型语言模型利用不同类型的多语言上下文生成准确答案的能力，在不依赖于检索质量的情况下，仍需进一步研究。本文对四种大型语言模型在三个涵盖48种语言的问答数据集上的性能进行了广泛的评估，结果显示，大型语言模型具有从非本族语言段落中提取相关信息的令人惊讶的能力，但在生成正确语言的完整答案方面能力较弱。我们的分析表明，不相关段落无论其语言如何，都会负面影响答案质量，但查询语言的不相关段落影响更为显著。综上所述，我们的研究结果加深了对大型语言模型在多语言RAG系统中利用上下文的理解，并为未来改进指明了方向。 

---
# Enhancing Negation Awareness in Universal Text Embeddings: A Data-efficient and Computational-efficient Approach 

**Title (ZH)**: 提高通用文本嵌入中的否定意识：一种数据高效且计算高效的方法 

**Authors**: Hongliu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00584)  

**Abstract**: Negation plays an important role in various natural language processing tasks such as Natural Language Inference and Sentiment Analysis tasks. Numerous prior studies have found that contextual text embedding models such as BERT, ELMO, RoBERTa or XLNet face challenges in accurately understanding negation. Recent advancements in universal text embeddings have demonstrated superior performance over contextual text embeddings in various tasks. However, due to the bias in popular evaluation benchmarks, the negation awareness capacity of these models remains unclear. To bridge the gap in existing literature, an in-depth analysis is initiated in this work to study the negation awareness of cutting-edge universal text embedding models. Our findings reveal a significant lack of negation awareness in these models, often interpreting negated text pairs as semantically similar. To efficiently deal with the conflict that different tasks need different trade-offs between topic and negation information among other semantic information, a data-efficient and computational-efficient embedding re-weighting method is proposed without modifying the parameters of text embedding models. The proposed solution is able to improve text embedding models' negation awareness significantly on both simple negation understanding task and complex negation understanding task. Furthermore, the proposed solution can also significantly improve the negation awareness of Large Language Model based task-specific high dimensional universal text embeddings. 

**Abstract (ZH)**: 否定在自然语言推理和情感分析等自然语言处理任务中发挥着重要作用。尽管先前研究表明，诸如BERT、ELMO、RoBERTa或XLNet等上下文文本嵌入模型在理解否定方面面临挑战，近期的通用文本嵌入进展在各种任务中显示出更好的性能。然而，由于流行评估基准中的偏差，这些模型的否定感知能力仍不清楚。为弥合现有文献的差距，本研究深入分析了最新通用文本嵌入模型的否定感知能力。我们的发现揭示了这些模型在否定感知方面存在显著不足，经常将否定文本对错误地解释为语义相似。为进一步高效解决不同任务在主题和否定信息等其他语义信息之间需要不同权衡的问题，我们提出了一种无需修改文本嵌入模型参数的高效且计算效率高的嵌入重权方法。该解决方案能够显著提高文本嵌入模型在简单否定理解任务和复杂否定理解任务中的否定感知能力，并且还能显著提高基于大型语言模型的任务特定高维通用文本嵌入的否定感知能力。 

---
# Automated detection of atomicity violations in large-scale systems 

**Title (ZH)**: 大规模系统中原子性违例的自动检测 

**Authors**: Hang He, Yixing Luo, Chengcheng Wan, Ting Su, Haiying Sun, Geguang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00521)  

**Abstract**: Atomicity violations in interrupt-driven programs pose a significant threat to software safety in critical systems. These violations occur when the execution sequence of operations on shared resources is disrupted by asynchronous interrupts. Detecting atomicity violations is challenging due to the vast program state space, application-level code dependencies, and complex domain-specific knowledge. We propose Clover, a hybrid framework that integrates static analysis with large language model (LLM) agents to detect atomicity violations in real-world programs. Clover first performs static analysis to extract critical code snippets and operation information. It then initiates a multi-agent process, where the expert agent leverages domain-specific knowledge to detect atomicity violations, which are subsequently validated by the judge agent. Evaluations on RaceBench 2.1, SV-COMP, and RWIP demonstrate that Clover achieves a precision/recall of 92.3%/86.6%, outperforming existing approaches by 27.4-118.2% on F1-score. 

**Abstract (ZH)**: 中断驱动程序中的原子性违反对关键系统软件安全性构成了重大威胁。这些违反发生在共享资源操作的执行顺序被异步中断打断时。检测原子性违反由于巨大的程序状态空间、应用程序级代码依赖关系和复杂的领域特定知识而具有挑战性。我们提出了一种名为Clover的混合框架，该框架结合了静态分析和大型语言模型（LLM）代理以检测实际程序中的原子性违反。Clover首先进行静态分析以提取关键代码片段和操作信息。然后启动一个多代理过程，其中专家代理利用领域特定知识检测原子性违反，随后由裁判代理进行验证。在RaceBench 2.1、SV-COMP和RWIP上的评估表明，Clover的精确度/召回率达到了92.3%/86.6%，在F1分数上比现有方法高出27.4%-118.2%。 

---
# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning 

**Title (ZH)**: 记忆不足：通过推理注入深度知识 

**Authors**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00472)  

**Abstract**: Although large language models (LLMs) excel in knowledge recall and reasoning, their static nature leads to outdated information as the real world evolves or when adapting to domain-specific knowledge, highlighting the need for effective knowledge injection. However, current research on knowledge injection remains superficial, mainly focusing on knowledge memorization and retrieval. This paper proposes a four-tier knowledge injection framework that systematically defines the levels of knowledge injection: memorization, retrieval, reasoning, and association. Based on this framework, we introduce DeepKnowledge, a synthetic experimental testbed designed for fine-grained evaluation of the depth of knowledge injection across three knowledge types (novel, incremental, and updated). We then explore various knowledge injection scenarios and evaluate the depth of knowledge injection for each scenario on the benchmark. Experimental results reveal key factors to reach each level of knowledge injection for LLMs and establish a mapping between the levels of knowledge injection and the corresponding suitable injection methods, aiming to provide a comprehensive approach for efficient knowledge injection across various levels. 

**Abstract (ZH)**: 尽管大型语言模型在知识回忆和推理方面表现出色，但由于其静态特性，在现实世界发展或适应领域特定知识时会出现过时信息的问题，凸显了有效知识注入的必要性。然而，当前的知识注入研究仍然停留在表面，主要集中在知识的记忆和检索。本文提出了一种四层知识注入框架，系统定义了知识注入的四个层级：记忆、检索、推理和关联。基于此框架，我们介绍了DeepKnowledge，这是一种合成实验测试床，用于对不同类型知识（新颖、增量和更新）的知识注入深度进行细粒度评估。随后，我们探索了各种知识注入场景，并在基准测试上评估了每个场景的知识注入深度。实验结果揭示了达到大型语言模型每个层级知识注入的关键因素，并建立了知识注入层级与相应适当注入方法之间的映射，旨在为不同层级的有效知识注入提供全面的方法。 

---
# No Free Lunch with Guardrails 

**Title (ZH)**: 有护栏也不能免于困境 

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00441)  

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.
Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and our own custom-built guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和生成式AI的广泛应用，护栏已成为确保其安全使用的关键工具。然而，增加护栏并非没有取舍；增强的安全措施可能会降低易用性，而更灵活的系统可能会留下对抗性攻击的漏洞。在本研究中，我们探讨当前护栏是否能够有效防止滥用的同时保持其实用性。我们引入了一种评估这些取舍的框架，衡量不同护栏如何平衡风险、安全和易用性，并构建了一个高效的护栏。 

---
# LLM-Assisted Proactive Threat Intelligence for Automated Reasoning 

**Title (ZH)**: LLM辅助主动威胁情报以实现自动化推理 

**Authors**: Shuva Paul, Farhad Alemi, Richard Macwan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00428)  

**Abstract**: Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices. 

**Abstract (ZH)**: 成功防御动态演变的网络威胁需要先进的复杂技术。本研究提出了一种通过将大型语言模型（LLMs）和检索增强生成（RAG）系统与持续的威胁情报流相结合来增强实时网络安全威胁检测和响应的新方法。利用最新的LLM发展，特别是GPT-4o，并创新地应用RAG技术，该方法通过集成动态和实时数据源解决了传统静态威胁分析的局限性。我们利用RAG实现实时获取最新的威胁情报，这是现有的GPT-4o模型所不具备的能力。我们使用Patrowl框架自动化了多种网络安全威胁情报流的检索，包括通用漏洞和暴露（CVE）、通用弱点枚举（CWE）、漏洞利用预测评分系统（EPSS）和已知利用漏洞（KEV）数据库，并将这些数据与all-mpnet-base-v2模型结合，用于高维向量嵌入，并在Milvus中存储和查询。我们通过一系列案例研究展示了该系统的有效性，相对于基线GPT-4o，显著提高了对最近披露的漏洞、KEVs和高EPSS评分CVE的处理能力。本研究不仅推动了LLMs在网络安全领域的应用，还为自动智能网络威胁信息管理系统的开发奠定了坚实的基础，填补了当前网络安全实践中的关键空白。 

---
# Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding 

**Title (ZH)**: 语义掌握：通过高级自然语言理解增强LLMs 

**Authors**: Mohanakrishnan Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00409)  

**Abstract**: Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding. 

**Abstract (ZH)**: 大型语言模型（LLMs）在执行NLP任务方面的能力有显著提升，但仍难以获得更深层次的语义理解、上下文连贯性和更为细致的推理能力。本文讨论了采用更先进的自然语言理解技术（如语义解析、知识集成和上下文强化学习）来推进LLMs的前沿方法。我们分析了结构化知识图谱的应用、检索增强生成（RAG）以及与人类级理解相匹配的微调策略。此外，我们还探讨了基于变压器的架构、对比学习以及混合符号-神经方法在解决复杂NLP任务中的幻觉、模糊性和事实一致性问题方面的作用。我们的研究结果强调了语义精确性对于增强AI驱动的语言系统的关键作用，并提出了未来研究方向，以弥合统计语言模型与真正自然语言理解之间的差距。 

---
# VerifiAgent: a Unified Verification Agent in Language Model Reasoning 

**Title (ZH)**: VerifiAgent：语言模型推理中的统一验证代理 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00406)  

**Abstract**: Large language models demonstrate remarkable reasoning capabilities but often produce unreliable or incorrect responses. Existing verification methods are typically model-specific or domain-restricted, requiring significant computational resources and lacking scalability across diverse reasoning tasks. To address these limitations, we propose VerifiAgent, a unified verification agent that integrates two levels of verification: meta-verification, which assesses completeness and consistency in model responses, and tool-based adaptive verification, where VerifiAgent autonomously selects appropriate verification tools based on the reasoning type, including mathematical, logical, or commonsense reasoning. This adaptive approach ensures both efficiency and robustness across different verification scenarios. Experimental results show that VerifiAgent outperforms baseline verification methods (e.g., deductive verifier, backward verifier) among all reasoning tasks. Additionally, it can further enhance reasoning accuracy by leveraging feedback from verification results. VerifiAgent can also be effectively applied to inference scaling, achieving better results with fewer generated samples and costs compared to existing process reward models in the mathematical reasoning domain. Code is available at this https URL 

**Abstract (ZH)**: 大规模语言模型展示了卓越的推理能力，但往往会产生不可靠或错误的响应。现有的验证方法通常是模型特定的或领域受限的，需要大量的计算资源，并且缺乏跨多样化推理任务的扩展性。为解决这些局限性，我们提出VerifiAgent，这是一种统一的验证代理，整合了两个层次的验证：元验证，评估模型响应的完整性和一致性；以及基于工具的自适应验证，VerifiAgent根据推理类型（包括数学、逻辑或常识推理）自主选择合适的验证工具。这种自适应方法确保了在不同的验证场景中兼具效率和鲁棒性。实验结果显示，VerifiAgent在所有推理任务中优于基线验证方法（如演绎验证器、回溯验证器）。此外，它还可以通过利用验证结果的反馈进一步提升推理准确性。在数学推理领域，VerifiAgent还可以有效应用于推理扩展，相较于现有的过程奖励模型，能在更少的生成样本和成本下取得更好的结果。代码可在此处访问。 

---
# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Title (ZH)**: 当说服力 overriding 事实真相在多代理大规模语言模型辩论中发生时：引入一种信心加权说服力 overriding 率（CW-POR） 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

**Abstract (ZH)**: 在多种真实场景中，单一大型语言模型（LLM）可能遇到相互矛盾的断言——有些准确而另一些则是强制性的错误，并必须判断哪一个为真。我们在此研究单轮多代理辩论框架中的这一风险：一个基于LLM的代理提供一个事实性答案，另一个则极力辩护一个谬误，而相同的LLM架构担任裁判。我们引入了置信加权说服否决率（CW-POR），该指标不仅量化裁判被误导的频率，还衡量其对错误选择的信念强度。我们在五个开源LLM（3B-14B参数）上进行的实验中系统地变化代理的详尽程度（30-300词），揭示即使是较小的模型也能构建强有力的论据以超越真实的答案——并常常抱着很高的信心。这些发现强调了对LLM进行鲁棒校准和对抗性测试的重要性，以防止它们自信地推广谬误信息。 

---
# Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments 

**Title (ZH)**: 基于LLM的集成入侵检测与安全切片xApp相结合，以保障O-RAN使能的无线网络部署安全性 

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic  

**Link**: [PDF](https://arxiv.org/pdf/2504.00341)  

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy. 

**Abstract (ZH)**: 基于大规模语言模型的O-RAN环境下的 intrusion detection 方法研究 

---
# VNJPTranslate: A comprehensive pipeline for Vietnamese-Japanese translation 

**Title (ZH)**: VNJPTranslate: 一个全面的越南语-日语翻译管道 

**Authors**: Hoang Hai Phan, Nguyen Duc Minh Vu, Nam Dang Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2504.00339)  

**Abstract**: Neural Machine Translation (NMT) driven by Transformer architectures has advanced significantly, yet faces challenges with low-resource language pairs like Vietnamese-Japanese (Vi-Ja). Issues include sparse parallel data and handling linguistic/cultural nuances. Recent progress in Large Language Models (LLMs) with strong reasoning, often refined via Reinforcement Learning (RL), enables high-quality synthetic data generation. We introduce VNJPTranslate, a pipeline designed to systematically address the Vi-Ja translation task. It features a targeted data augmentation strategy using advanced LLMs with Chain-of-Thought prompting for challenging segments identified via corpus analysis. Subsequently, we employ efficient fine-tuning techniques (Unsloth with QLoRA) on a capable, low-parameter autoregressive model (specifically, a fine-tuned version of the 1.8B parameter Sailor model, which is based on the Qwen architecture) to create a practical and high-performing translation system. This integrated approach aims to improve Vi-Ja translation quality significantly over existing baselines. 

**Abstract (ZH)**: 基于Transformer架构的神经机器翻译（NMT）在低资源语言对（如越南语-日语）翻译任务（Vi-Ja）中获得了显著进步，但仍面临并行数据稀疏和处理语言/文化差异的挑战。大规模语言模型（LLMs）借助强烈的推理能力，并通过强化学习（RL）进行优化，能够生成高质量的合成数据。我们提出VNJPTranslate流水线，旨在系统地解决越南语-日语翻译任务。该流水线采用先进LLMs的链式思考提示策略进行有针对性的数据增强，并通过语料库分析识别具有挑战性的段落。随后，我们采用高效的微调技术（UnSloth与QLoRA）对一个具有强大自回归能力但参数较少的模型（具体而言，是基于Qwen架构、已微调的1.8B参数Sailor模型）进行微调，以创建一个实用且高性能的翻译系统。该综合方法旨在显著提高越南语-日语翻译质量，超越现有基线。 

---
# Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training 

**Title (ZH)**: 通过知识图谱增强训练检测和缓解LLM中的偏见 

**Authors**: Rajeev Kumar, Harishankar Kumar, Kumari Shalini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00310)  

**Abstract**: Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications. 

**Abstract (ZH)**: 大型语言模型通过其令人惊讶的语言理解与生成人类文本的能力， revolutionized 自然语言处理。然而，这些模型继承并进一步放大了训练数据中存在的偏见，引发了伦理与公平性方面的关注。检测与缓解这些偏见对于确保大型语言模型在多领域中负责任且公平地行动至关重要。本文探讨了知识图谱增强训练（KGAT）作为缓解大型语言模型偏见的新方法。通过使用现实世界知识图谱中的结构化领域特定知识，我们改进了模型的理解并减少了有偏见的输出。用于偏见评估的公开数据集包括 Gender Shades、Bias in Bios 和 FairFace，而人口统计平等等指标促使我们进行严格的检测。我们还实施了有针对性的缓解策略以纠正有偏见的关联，显著减少了有偏见的输出并改善了偏见指标。配备了现实世界数据集和知识图谱，我们的框架既具扩展性又有效，为在敏感和高风险应用中负责任地部署奠定了基础。 

---
# Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead 

**Title (ZH)**: 复杂任务下的推理时伸缩性：当前状况与未来展望 

**Authors**: Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, Shivam Garg, Neel Joshi, Yash Lara, John Langford, Besmira Nushi, Vibhav Vineet, Yue Wu, Safoora Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00294)  

**Abstract**: Inference-time scaling can enhance the reasoning capabilities of large language models (LLMs) on complex problems that benefit from step-by-step problem solving. Although lengthening generated scratchpads has proven effective for mathematical tasks, the broader impact of this approach on other tasks remains less clear. In this work, we investigate the benefits and limitations of scaling methods across nine state-of-the-art models and eight challenging tasks, including math and STEM reasoning, calendar planning, NP-hard problems, navigation, and spatial reasoning. We compare conventional models (e.g., GPT-4o) with models fine-tuned for inference-time scaling (e.g., o1) through evaluation protocols that involve repeated model calls, either independently or sequentially with feedback. These evaluations approximate lower and upper performance bounds and potential for future performance improvements for each model, whether through enhanced training or multi-model inference systems. Our extensive empirical analysis reveals that the advantages of inference-time scaling vary across tasks and diminish as problem complexity increases. In addition, simply using more tokens does not necessarily translate to higher accuracy in these challenging regimes. Results from multiple independent runs with conventional models using perfect verifiers show that, for some tasks, these models can achieve performance close to the average performance of today's most advanced reasoning models. However, for other tasks, a significant performance gap remains, even in very high scaling regimes. Encouragingly, all models demonstrate significant gains when inference is further scaled with perfect verifiers or strong feedback, suggesting ample potential for future improvements. 

**Abstract (ZH)**: 推理时的扩展可以增强大型语言模型在复杂问题上的推理能力，这些问题是步骤式问题解决所获益的。虽然延长生成草稿区片段对数学任务的有效性已经得到了证明，但这种方法对其他任务的更广泛影响仍不明确。在本工作中，我们调查了九个最先进的模型和八个具有挑战性的任务（包括数学和STEM推理、日历规划、NP难问题、导航和空间推理）中扩展方法的好处和限制。我们将传统模型（如GPT-4o）与针对推理时扩展进行微调的模型（如o1）进行了比较，评估协议包括重复调用模型，要么独立地要么在反馈情况下顺序调用。这些评估接近每个模型的下界和上界性能，并且反馈可能对未来性能改进的潜力提供估算。广泛的实证分析显示，推理时扩展的优势随着任务复杂度的增加而变化并在复杂问题上减弱。此外，在这些具有挑战性的环境中，简单地使用更多标记并不必然意味着更高的准确性。使用完美验证器对传统模型进行多次独立运行的评估表明，对于某些任务，这些模型可以达到接近当今最先进的推理模型平均性能的水平。然而，对于其他任务，在非常高的扩展范围内仍然存在显著的性能差距。令人鼓舞的是，所有模型在使用完美验证器或强反馈进一步扩展推理时都显示出显著的性能提升，这表明未来改进的潜力巨大。 

---
# Do Chinese models speak Chinese languages? 

**Title (ZH)**: 中国人使用的模型说中文吗？ 

**Authors**: Andrea W Wen-Yi, Unso Eun Seo Jo, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2504.00289)  

**Abstract**: The release of top-performing open-weight LLMs has cemented China's role as a leading force in AI development. Do these models support languages spoken in China? Or do they speak the same languages as Western models? Comparing multilingual capabilities is important for two reasons. First, language ability provides insights into pre-training data curation, and thus into resource allocation and development priorities. Second, China has a long history of explicit language policy, varying between inclusivity of minority languages and a Mandarin-first policy. To test whether Chinese LLMs today reflect an agenda about China's languages, we test performance of Chinese and Western open-source LLMs on Asian regional and Chinese minority languages. Our experiments on Information Parity and reading comprehension show Chinese models' performance across these languages correlates strongly (r=0.93) with Western models', with the sole exception being better Mandarin. Sometimes, Chinese models cannot identify languages spoken by Chinese minorities such as Kazakh and Uyghur, even though they are good at French and German. These results provide a window into current development priorities, suggest options for future development, and indicate guidance for end users. 

**Abstract (ZH)**: 顶级开放权重LLM的发布已确立了中国在AI发展中的领导地位。这些模型支持中国的语言吗？还是说它们使用与西方模型相同的语言？比较多语言能力对于两个原因非常重要。首先，语言能力提供了有关预训练数据收集的见解，从而反映了资源分配和开发优先级。其次，中国有着悠久的语言政策历史，政策之间在少数语言包容性和汉语优先政策之间存在差异。为了测试当前的Chinese LLMs是否反映了关于中国语言的议程，我们测试了Chinese和Western开源LLM在亚洲区域和中国少数民族语言上的性能。我们的Information Parity和阅读理解实验结果显示，Chinese模型在这几种语言上的性能与Western模型高度相关（r=0.93），唯一例外是在普通话上稍好。有时，Chinese模型无法识别如哈萨克语和乌兹别克语等中国少数民族的语言，尽管它们在法语和德语上表现很好。这些结果提供了一个了解当前开发优先级的窗口，提出了未来发展的选项，并指出了终端用户的应用指导。 

---
# SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers 

**Title (ZH)**: SciReplicate-Bench：基于代理驱动算法再现从研究论文中评估LLMs的效果 

**Authors**: Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00255)  

**Abstract**: This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's this http URL analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at this https URL. 

**Abstract (ZH)**: 本研究评估了大型语言模型在从近期NLP论文中的算法描述生成代码方面的性能。该任务要求具备两个关键能力：（1）算法理解：从论文和学术文献中综合信息以理解实现逻辑，（2）编程技能：识别依赖关系并正确实现必要的API。为确保严格的评估，我们引入了SciReplicate-Bench这一基准，包括来自2024年36篇NLP论文的100个任务，附有详细注释和全面的测试案例。基于SciReplicate-Bench，我们提出了一种多代理框架Sci-Reproducer，其中包括一个论文代理，负责从文献中解释算法概念，以及一个代码代理，负责从代码库中检索依赖关系并实现解决方案。为了评估算法理解，我们引入了推理图准确性，该指标量化了生成的推理图与来自代码注释和结构的参考推理图之间的相似性。为了评估实现质量，我们采用了执行准确性、CodeBLEU以及代码库依赖关系/API召回率指标。在实验中，我们评估了多种强大的非推理型LLM和推理型LLM作为基础模型。使用Sci-Reproducer的性能最佳模型仅实现39%的执行准确性，突显了基准的挑战性。分析指出，缺失或不一致的算法描述是成功复现的主要障碍。我们将在GitHub开源我们的基准和代码。 

---
# Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy 

**Title (ZH)**: 使用大语言模型合成公众意见：角色创建、影响及对eDemocracy的未来展望 

**Authors**: Rabimba Karanjai, Boris Shor, Amanda Austin, Ryan Kennedy, Yang Lu, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00241)  

**Abstract**: This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data, addressing challenges in traditional survey methods like declining response rates and non-response bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-context learning that leverages RAG and specified personality profiles from the HEXACO model and demographic information, and uses that for dynamically generated prompts. This method allows LLMs to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare our results with pre-trained models with standard few-shot prompts. Experiments using questions from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly improves the alignment of LLM-generated opinions with real-world human survey responses, increasing answer adherence. In addition, we discuss challenges, limitations and future research directions. 

**Abstract (ZH)**: 本文探讨了使用大型语言模型（LLMs）合成公众意见数据，解决传统调查方法中如响应率下降和无响应偏差等挑战。我们提出了一种新颖的技术：基于知识注入的角色创建，这是一种利用RAG和HEXACO模型指定的人格特征以及人口统计信息的上下文学习方式，并用于动态生成提示。该方法使LLMs能够比现有的提示工程方法更准确地模拟多样化的观点。我们还将我们的结果与使用标准少样本提示的预训练模型进行了比较。使用合作选举研究（CES）的问题进行的实验表明，我们的角色创建方法显著提高了LLM生成观点与实际人类调查响应的一致性，增加了答案的符合度。此外，我们讨论了挑战、局限性和未来的研究方向。 

---
# $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks 

**Title (ZH)**: 《四面受敌的智能体》：通过优化提示攻击打破实用多智能体LLM系统 

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00218)  

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms. 

**Abstract (ZH)**: 大多数关于大型语言模型（LLM）安全性的讨论集中在单代理设置上，但多代理LLM系统现在创造了新的对抗风险，因为其行为依赖于代理之间的通信和分散式推理。在此项工作中，我们创新性地将攻击目标集中在具有有限令牌带宽、消息传递延迟等约束的实用系统上，并设计了一种具有传递不变性对抗攻击，旨在优化跨延迟和带宽约束网络拓扑的提示分布，以绕过系统内的分布式安全机制。我们将攻击路径表述为最大流最小成本问题，并结合新型传递不变性规避损失（PIEL），利用图基优化方法在最大化攻击成功率的同时最小化检测风险。在包括Llama、Mistral、Gemma、DeepSeek及其他变体的多种模型上，在JailBreakBench和AdversarialBench等不同数据集上评估，我们的方法比传统攻击高出高达7倍的效果，揭示了多代理系统中的关键漏洞。此外，我们证明现有的防御措施，包括Llama-Guard和PromptGuard的变体，无法阻止我们的攻击，强调了针对多代理系统的特定安全机制的迫切需求。 

---
# Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency 

**Title (ZH)**: RAG系统中的矛盾检测：评估LLM作为上下文验证器以提高信息一致性 

**Authors**: Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.00180)  

**Abstract**: Retrieval Augmented Generation (RAG) systems have emerged as a powerful method for enhancing large language models (LLMs) with up-to-date information. However, the retrieval step in RAG can sometimes surface documents containing contradictory information, particularly in rapidly evolving domains such as news. These contradictions can significantly impact the performance of LLMs, leading to inconsistent or erroneous outputs. This study addresses this critical challenge in two ways. First, we present a novel data generation framework to simulate different types of contradictions that may occur in the retrieval stage of a RAG system. Second, we evaluate the robustness of different LLMs in performing as context validators, assessing their ability to detect contradictory information within retrieved document sets. Our experimental results reveal that context validation remains a challenging task even for state-of-the-art LLMs, with performance varying significantly across different types of contradictions. While larger models generally perform better at contradiction detection, the effectiveness of different prompting strategies varies across tasks and model architectures. We find that chain-of-thought prompting shows notable improvements for some models but may hinder performance in others, highlighting the complexity of the task and the need for more robust approaches to context validation in RAG systems. 

**Abstract (ZH)**: 基于检索增强生成（RAG）系统的矛盾信息缓解研究 

---
# Does "Reasoning" with Large Language Models Improve Recognizing, Generating, and Reframing Unhelpful Thoughts? 

**Title (ZH)**: 大型语言模型进行“推理”能否改善识别、生成和重新框定无帮助思维？ 

**Authors**: Yilin Qi, Dong Won Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.00163)  

**Abstract**: Cognitive Reframing, a core element of Cognitive Behavioral Therapy (CBT), helps individuals reinterpret negative experiences by finding positive meaning. Recent advances in Large Language Models (LLMs) have demonstrated improved performance through reasoning-based strategies. This inspires a promising direction of leveraging the reasoning capabilities of LLMs to improve CBT and mental reframing by simulating the process of critical thinking, potentially enabling more effective recognition, generation, and reframing of cognitive distortions. In this work, we investigate the role of various reasoning methods, including pre-trained reasoning LLMs and augmented reasoning strategies such as CoT and self-consistency in enhancing LLMs' ability to perform cognitive reframing tasks. We find that augmented reasoning methods, even when applied to "outdated" LLMs like GPT-3.5, consistently outperform state-of-the-art pretrained reasoning models on recognizing, generating and reframing unhelpful thoughts. 

**Abstract (ZH)**: 基于推理的认知重框理论：利用大型语言模型增强认知行为疗法中的认知重框技术 

---
# Assessing Code Understanding in LLMs 

**Title (ZH)**: 评估LLM中的代码理解能力 

**Authors**: Cosimo Laneve, Alvise Spanò, Dalila Ressi, Sabina Rossi, Michele Bugliesi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00065)  

**Abstract**: We present an empirical evaluation of Large Language Models in code understanding associated with non-trivial, semantic-preserving program transformations such as copy propagation or constant folding. Our findings show that LLMs fail to judge semantic equivalence in approximately 41\% of cases when no context is provided and in 29\% when given a simple generic context. To improve accuracy, we advocate integrating LLMs with code-optimization tools to enhance training and facilitate more robust program understanding. 

**Abstract (ZH)**: 我们对大型语言模型在代码理解方面的实证评估，特别是在处理诸如复制传播或常数折叠等非平凡、语义保持的程序变换方面的能力。研究发现，在没有提供上下文的情况下，大型语言模型在约41%的案例中未能判断语义等价性；即使提供了简单的通用上下文，这一比例也达到了29%。为了提高准确性，我们建议将大型语言模型与代码优化工具集成，以增强训练并促进更稳健的程序理解。 

---
# Evaluating the Feasibility and Accuracy of Large Language Models for Medical History-Taking in Obstetrics and Gynecology 

**Title (ZH)**: 评估大型语言模型在妇产科学科医疗病史采集中的可行性与准确性 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Tian Tang, Rong Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00061)  

**Abstract**: Effective physician-patient communications in pre-diagnostic environments, and most specifically in complex and sensitive medical areas such as infertility, are critical but consume a lot of time and, therefore, cause clinic workflows to become inefficient. Recent advancements in Large Language Models (LLMs) offer a potential solution for automating conversational medical history-taking and improving diagnostic accuracy. This study evaluates the feasibility and performance of LLMs in those tasks for infertility cases. An AI-driven conversational system was developed to simulate physician-patient interactions with ChatGPT-4o and ChatGPT-4o-mini. A total of 70 real-world infertility cases were processed, generating 420 diagnostic histories. Model performance was assessed using F1 score, Differential Diagnosis (DDs) Accuracy, and Accuracy of Infertility Type Judgment (ITJ). ChatGPT-4o-mini outperformed ChatGPT-4o in information extraction accuracy (F1 score: 0.9258 vs. 0.9029, p = 0.045, d = 0.244) and demonstrated higher completeness in medical history-taking (97.58% vs. 77.11%), suggesting that ChatGPT-4o-mini is more effective in extracting detailed patient information, which is critical for improving diagnostic accuracy. In contrast, ChatGPT-4o performed slightly better in differential diagnosis accuracy (2.0524 vs. 2.0048, p > 0.05). ITJ accuracy was higher in ChatGPT-4o-mini (0.6476 vs. 0.5905) but with lower consistency (Cronbach's $\alpha$ = 0.562), suggesting variability in classification reliability. Both models demonstrated strong feasibility in automating infertility history-taking, with ChatGPT-4o-mini excelling in completeness and extraction accuracy. In future studies, expert validation for accuracy and dependability in a clinical setting, AI model fine-tuning, and larger datasets with a mix of cases of infertility have to be prioritized. 

**Abstract (ZH)**: 有效诊断前的医患沟通在预诊环境中至关重要，尤其是在不孕不育等复杂和敏感的医疗领域，但这些沟通耗费大量时间，导致 clinics 工作流程效率低下。近年来大语言模型（LLMs）的进展为自动化问诊和提高诊断准确性提供了潜在解决方案。本研究评估了大语言模型在不孕不育病例中进行任务的可行性和性能。开发了一个以ChatGPT-4o和ChatGPT-4o-mini为基础的AI驱动对话系统，处理了70例实际的不孕不育病例，生成了420份诊断病史。模型性能通过F1分数、鉴别诊断准确性和不孕不育类型判断准确率（ITJ）进行评估。ChatGPT-4o-mini在信息提取准确性（F1分数：0.9258 vs. 0.9029，p=0.045，d=0.244）和医学病史提取完整性（97.58% vs. 77.11%）方面表现更优，表明ChatGPT-4o-mini在提取详细患者信息方面更有效，这对于提高诊断准确性至关重要。相比之下，ChatGPT-4o在鉴别诊断准确率方面略优（2.0524 vs. 2.0048，p>0.05）。ChatGPT-4o-mini在ITJ准确率方面更高（0.6476 vs. 0.5905），但一致性较低（Cronbach's α = 0.562），表明分类可靠性存在变异性。两种模型在自动化不孕不育病史提取方面具有较强的可行性，ChatGPT-4o-mini在完整性与提取准确性方面表现更优。未来研究需优先考虑临床环境下的专家验证、AI模型微调以及包含不同不孕不育病例的更大数据集。 

---
# Integrating Large Language Models with Human Expertise for Disease Detection in Electronic Health Records 

**Title (ZH)**: 将大型语言模型与人类专长集成以检测电子健康记录中的疾病 

**Authors**: Jie Pan, Seungwon Lee, Cheligeer Cheligeer, Elliot A. Martin, Kiarash Riazi, Hude Quan, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00053)  

**Abstract**: Objective: Electronic health records (EHR) are widely available to complement administrative data-based disease surveillance and healthcare performance evaluation. Defining conditions from EHR is labour-intensive and requires extensive manual labelling of disease outcomes. This study developed an efficient strategy based on advanced large language models to identify multiple conditions from EHR clinical notes. Methods: We linked a cardiac registry cohort in 2015 with an EHR system in Alberta, Canada. We developed a pipeline that leveraged a generative large language model (LLM) to analyze, understand, and interpret EHR notes by prompts based on specific diagnosis, treatment management, and clinical guidelines. The pipeline was applied to detect acute myocardial infarction (AMI), diabetes, and hypertension. The performance was compared against clinician-validated diagnoses as the reference standard and widely adopted International Classification of Diseases (ICD) codes-based methods. Results: The study cohort accounted for 3,088 patients and 551,095 clinical notes. The prevalence was 55.4%, 27.7%, 65.9% and for AMI, diabetes, and hypertension, respectively. The performance of the LLM-based pipeline for detecting conditions varied: AMI had 88% sensitivity, 63% specificity, and 77% positive predictive value (PPV); diabetes had 91% sensitivity, 86% specificity, and 71% PPV; and hypertension had 94% sensitivity, 32% specificity, and 72% PPV. Compared with ICD codes, the LLM-based method demonstrated improved sensitivity and negative predictive value across all conditions. The monthly percentage trends from the detected cases by LLM and reference standard showed consistent patterns. 

**Abstract (ZH)**: 目标：电子健康记录（EHR）广泛可用，可以补充基于行政数据的疾病监测和医疗服务绩效评价。从EHR中定义条件劳动强度大，需要进行广泛的疾病结果手工标注。本研究基于先进的大语言模型开发了一种高效策略，用于识别EHR临床笔记中的多种条件。方法：我们将2015年的冠心病注册队列与加拿大的艾伯塔省EHR系统连接。我们开发了一个流水线，利用生成性大语言模型（LLM）通过基于特定诊断、治疗管理及临床指南的提示来分析、理解和解释EHR笔记。该流水线应用于检测急性心肌梗死（AMI）、糖尿病和高血压。性能与临床验证诊断和广泛应用的国际疾病分类（ICD）编码方法进行了比较。结果：研究队列包括3088名患者和551095份临床笔记。AMIs、糖尿病和高血压的患病率分别为55.4%、27.7%和65.9%。基于LLM的流水线在检测条件方面的性能不同：AMI的敏感性为88%，特异性为63%，阳性预测值为77%；糖尿病的敏感性为91%，特异性为86%，阳性预测值为71%；高血压的敏感性为94%，特异性为32%，阳性预测值为72%。与ICD编码相比，基于LLM的方法在所有条件下均显示提高了敏感性和阴性预测值。通过LLM和参考标准检测的确诊病例的月百分比趋势显示出一致的模式。 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Title (ZH)**: JudgeLRM：大型推理模型作为法官 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

**Abstract (ZH)**: 大规模语言模型作为评估者崛起提供了规模化的人类标注替代方案，但在需要复杂推理的领域，现有的监督微调（SFT）方法往往表现不佳。本研究探索大规模语言模型评估者是否真的可以从增强的推理能力中受益。通过对评估任务中推理需求的详细分析，我们揭示了SFT性能改进与推理需求样本比例之间的负相关性，突显了SFT在这些场景中的局限性。为了解决这一问题，我们提出了一种使用基于奖励的强化学习（RL）训练的JudgeLRM系列判断导向的大规模语言模型。JudgeLRM模型在判断任务中的一致性评估中表现出色，优于SFT调优的大规模语言模型和最先进的推理模型。值得注意的是，JudgeLRM-3B超过了GPT-4，而JudgeLRM-7B在F1分数上比DeepSeek-R1高出了2.79%，特别是在需要深度推理的判断任务中表现尤为出色。 

---
# Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs 

**Title (ZH)**: Distill-C: 通过LLM辅助精简定制增强NL2SQL 

**Authors**: Cong Duy Vu Hoang, Gioacchino Tangari, Clemence Lanfranchi, Dalu Guo, Paul Cayet, Steve Siu, Don Dharmasiri, Yuan-Fang Li, Long Duong, Damien Hilloulin, Rhicheek Patra, Sungpack Hong, Hassan Chafi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00048)  

**Abstract**: The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost. 

**Abstract (ZH)**: 大型语言模型在企业应用中的不断 adoption 已加剧了对自然语言转结构查询语言（NL2SQL）解决方案的需求，这类解决方案在高性能和效率之间存在竞争需求。特定领域和客户的要求进一步复杂化了这一问题。为应对这一挑战，我们引入了 Distill-C，这是一种针对 NL2SQL 任务定制的精简框架。Distill-C 利用大型教师语言模型通过一个稳健且可扩展的流水线生成高质量的合成数据。在这些合成数据上对较小的开源语言模型进行微调，使它们能够与或超越比它们大一个数量级的教师模型。在多个具有挑战性的基准测试中，Distill-C 的执行准确率平均提高了 36%，相对于三个不同语言模型家族的基础模型。此外，在三个内部客户基准测试中，Distill-C 的性能比基础模型提高了 22.6%。我们的结果表明，Distill-C 是一种有效的、高性能且可泛化的轻量级但强大的 NL2SQL 模型部署方法，能够在保持较低计算成本的同时实现出色的效果。 

---
# Multi-Stakeholder Disaster Insights from Social Media Using Large Language Models 

**Title (ZH)**: 使用大型语言模型从社交媒体获取多方灾害见解 

**Authors**: Loris Belcastro, Cristian Cosentino, Fabrizio Marozzo, Merve Gündüz-Cüre, Şule Öztürk-Birim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00046)  

**Abstract**: In recent years, social media has emerged as a primary channel for users to promptly share feedback and issues during disasters and emergencies, playing a key role in crisis management. While significant progress has been made in collecting and analyzing social media content, there remains a pressing need to enhance the automation, aggregation, and customization of this data to deliver actionable insights tailored to diverse stakeholders, including the press, police, EMS, and firefighters. This effort is essential for improving the coordination of activities such as relief efforts, resource distribution, and media communication. This paper presents a methodology that leverages the capabilities of LLMs to enhance disaster response and management. Our approach combines classification techniques with generative AI to bridge the gap between raw user feedback and stakeholder-specific reports. Social media posts shared during catastrophic events are analyzed with a focus on user-reported issues, service interruptions, and encountered challenges. We employ full-spectrum LLMs, using analytical models like BERT for precise, multi-dimensional classification of content type, sentiment, emotion, geolocation, and topic. Generative models such as ChatGPT are then used to produce human-readable, informative reports tailored to distinct audiences, synthesizing insights derived from detailed classifications. We compare standard approaches, which analyze posts directly using prompts in ChatGPT, to our advanced method, which incorporates multi-dimensional classification, sub-event selection, and tailored report generation. Our methodology demonstrates superior performance in both quantitative metrics, such as text coherence scores and latent representations, and qualitative assessments by automated tools and field experts, delivering precise insights for diverse disaster response stakeholders. 

**Abstract (ZH)**: 近年来，社交媒体已成为用户在灾难和紧急事件中及时分享反馈和问题的主要渠道，在危机管理中发挥着关键作用。尽管在收集和分析社交媒体内容方面取得了显著进展，但仍迫切需要增强这些数据的自动化、聚合和定制化，以提供针对不同利益相关者的个性化行动建议，包括媒体、警察、急救服务和消防员。这种努力对于提高如救援工作、资源分配和媒体沟通等活动的协调性至关重要。本文提出了一种利用大型语言模型（LLM）能力的方法，以增强灾难响应和管理。我们的方法结合了分类技术和生成AI，以填补原始用户反馈与特定利益相关者报告之间的差距。在灾难事件期间共享的社交媒体帖子被分析，重点关注用户报告的问题、服务中断以及遇到的挑战。我们使用全谱LLMs，采用如BERT的分析模型对内容类型、情感、情绪、地理位置和主题进行精确的多维度分类。然后使用如ChatGPT的生成模型生成针对不同受众的人类可读且信息丰富的报告，综合详细的分类洞察。我们将基于ChatGPT直接分析帖子的标准方法与我们的方法进行了比较，该方法结合了多维度分类、分事件选择和定制报告生成。我们的方法在定量指标（如文本连贯性评分和潜在表示）和自动化工具及现场专家的定性评估中都展现了更优的表现，为不同的灾难响应相关方提供精确的洞察。 

---
# CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation 

**Title (ZH)**: CrossWordBench：通过可控谜题生成评估大语言模型和大型向量语言模型的推理能力 

**Authors**: Jixuan Leng, Chengsong Huang, Langlin Huang, Bill Yuchen Lin, William W. Cohen, Haohan Wang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00043)  

**Abstract**: Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations. 

**Abstract (ZH)**: 现有的大规模语言模型(LLMs)和大规模视觉-语言模型(LVLMs)推理评估框架主要评估文本推理或视觉-语言理解能力，但文本和视觉约束之间的动态交互有限。为解决这一问题，我们引入了CrossWordBench，这是一个通过填字谜题评估LLMs和LVLMs推理能力的基准，填字谜题需要同时遵循文本线索的语义约束和图形结构的交叉约束。CrossWordBench 利用了一个可控的谜题生成框架，生成多种格式（文本和图像）的谜题，并提供从直接解谜到互动模式的不同评估策略。我们的广泛评估表明，具备推理能力的LLMs在利用交叉字母约束方面显著优于不具备推理能力的模型。此外，我们还展示了LVLMs在任务中的困难，并发现它们的解谜性能与图形解析准确性之间存在强烈相关性。我们的研究结果揭示了当前LLMs和LVLMs推理能力的局限性，并为未来评估提供了有效的多模态约束任务设计方法。 

---
# MiZero: The Shadowy Defender Against Text Style Infringements 

**Title (ZH)**: MiZero: 阴影中的守护者——对抗文本风格侵权 

**Authors**: Ziwei Zhang, Juan Wen, Wanli Peng, Zhengxian Wu, Yinghan Zhou, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.00035)  

**Abstract**: In-Context Learning (ICL) and efficient fine-tuning methods significantly enhanced the efficiency of applying Large Language Models (LLMs) to downstream tasks. However, they also raise concerns about the imitation and infringement of personal creative data. Current methods for data copyright protection primarily focuses on content security but lacks effectiveness in protecting the copyrights of text styles. In this paper, we introduce a novel implicit zero-watermarking scheme, namely MiZero. This scheme establishes a precise watermark domain to protect the copyrighted style, surpassing traditional watermarking methods that distort the style characteristics. Specifically, we employ LLMs to extract condensed-lists utilizing the designed instance delimitation mechanism. These lists guide MiZero in generating the watermark. Extensive experiments demonstrate that MiZero effectively verifies text style copyright ownership against AI imitation. 

**Abstract (ZH)**: 基于上下文学习(ICL)和高效微调方法显著提升了大型语言模型(LLMs)在下游任务中的应用效率。然而，它们也引发了对个人创作品复制和侵权的担忧。当前的数据版权保护方法主要集中在内容安全，但在保护文本风格版权方面效果有限。本文介绍了一种新颖的隐式零水印方案，即MiZero。该方案建立了精确的水印域，保护版权风格，超越了传统会扭曲风格特征的水印方法。具体来说，我们利用LLMs通过设计的实例分隔机制提取压缩列表，这些列表指导MiZero生成水印。广泛的经验实验表明，MiZero有效地验证了文本风格的版权归属，对抗AI模仿。 

---
# Leaking LoRa: An Evaluation of Password Leaks and Knowledge Storage in Large Language Models 

**Title (ZH)**: 泄露的LoRa：大规模语言模型中密码泄露和知识存储的评估 

**Authors**: Ryan Marinelli, Magnus Eckhoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.00031)  

**Abstract**: To effectively deploy Large Language Models (LLMs) in application-specific settings, fine-tuning techniques are applied to enhance performance on specialized tasks. This process often involves fine-tuning on user data data, which may contain sensitive information. Although not recommended, it is not uncommon for users to send passwords in messages, and fine-tuning models on this could result in passwords being leaked. In this study, a Large Language Model is fine-tuned with customer support data and passwords from the RockYou password wordlist using Low-Rank Adaptation (LoRA). Out of the first 200 passwords from the list, 37 were successfully recovered. Further, causal tracing is used to identify that password information is largely located in a few layers. Lastly, Rank One Model Editing (ROME) is used to remove the password information from the model, resulting in the number of passwords recovered going from 37 to 0. 

**Abstract (ZH)**: 在特定应用场景中有效部署大规模语言模型：通过低秩适应（LoRA）微调客户支持数据和RockYou密码词表中的密码，去除密码信息以提升安全性 

---
# Token-Driven GammaTune: Adaptive Calibration for Enchanced Speculative Decoding 

**Title (ZH)**: 基于令牌驱动的 GammaTune：增强推测性解码的自适应校准 

**Authors**: Aayush Gautam, Susav Shrestha, Narasimha Annapareddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.00030)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, selecting an optimal speculation length is critical for maximizing speedup while minimizing wasted computation. We introduce \textit{GammaTune} and \textit{GammaTune+}, training-free adaptive algorithms that dynamically adjust speculation length based on token acceptance rates using a heuristic-based switching mechanism. Evaluated on SpecBench across multiple tasks and model pairs, our method outperforms other heuristic-based approaches and fixed-length speculative decoding, achieving an average speedup of 15\% ($\pm$5\%) with \textit{GammaTune} and 16\% ($\pm$3\%) with \textit{GammaTune+}, while reducing performance variance. This makes \textit{GammaTune} a robust and efficient solution for real-world deployment. 

**Abstract (ZH)**: GammaTune 和 GammaTune+: 基于启发式动态调整猜想长度的无训练适应算法 

---
# Generating Structured Plan Representation of Procedures with LLMs 

**Title (ZH)**: 使用大语言模型生成程序的结构化计划表示 

**Authors**: Deepeka Garg, Sihan Zeng, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2504.00029)  

**Abstract**: In this paper, we address the challenges of managing Standard Operating Procedures (SOPs), which often suffer from inconsistencies in language, format, and execution, leading to operational inefficiencies. Traditional process modeling demands significant manual effort, domain expertise, and familiarity with complex languages like Business Process Modeling Notation (BPMN), creating barriers for non-techincal users. We introduce SOP Structuring (SOPStruct), a novel approach that leverages Large Language Models (LLMs) to transform SOPs into decision-tree-based structured representations. SOPStruct produces a standardized representation of SOPs across different domains, reduces cognitive load, and improves user comprehension by effectively capturing task dependencies and ensuring sequential integrity. Our approach enables leveraging the structured information to automate workflows as well as empower the human users. By organizing procedures into logical graphs, SOPStruct facilitates backtracking and error correction, offering a scalable solution for process optimization. We employ a novel evaluation framework, combining deterministic methods with the Planning Domain Definition Language (PDDL) to verify graph soundness, and non-deterministic assessment by an LLM to ensure completeness. We empirically validate the robustness of our LLM-based structured SOP representation methodology across SOPs from different domains and varying levels of complexity. Despite the current lack of automation readiness in many organizations, our research highlights the transformative potential of LLMs to streamline process modeling, paving the way for future advancements in automated procedure optimization. 

**Abstract (ZH)**: 本文介绍了管理标准操作程序（SOPs）所面临的挑战，由于SOPs在语言、格式和执行方面存在不一致性，导致操作效率低下。传统的流程建模需要大量的手动努力、领域专业知识以及熟悉复杂语言（如业务流程建模符号BPMN）的能力，这为非技术人员设置了障碍。我们提出了SOP结构化（SOPStruct）这一新颖的方法，利用大规模语言模型（LLMs）将SOPs转换为基于决策树的结构化表示。SOPStruct为不同领域的SOPs提供了标准化的表示形式，减少了认知负荷，通过有效捕捉任务依赖关系和确保顺序完整性，提高了用户理解能力。我们的方法能够利用结构化信息自动化工作流，同时增强人类用户的操作能力。通过将程序组织成逻辑图，SOPStruct支持回溯和错误修正，提供了一种可扩展的流程优化解决方案。我们采用了一种新的评估框架，结合确定性方法和规划领域定义语言（PDDL）验证图形的一致性，并通过LLM进行非确定性评估以确保完整性。我们通过跨不同领域和复杂程度的SOP的实证验证了我们基于LLM的结构化SOP表示方法的鲁棒性。尽管许多组织目前尚未准备好自动化，但我们的研究展示了LLMs在简化流程建模方面的变革潜力，为未来自动流程优化的进步铺平了道路。 

---
# Celler:A Genomic Language Model for Long-Tailed Single-Cell Annotation 

**Title (ZH)**: CELLER：一种针对长尾单细胞注释的基因组语言模型 

**Authors**: Huan Zhao, Yiming Liu, Jina Yao, Ling Xiong, Zexin Zhou, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00020)  

**Abstract**: Recent breakthroughs in single-cell technology have ushered in unparalleled opportunities to decode the molecular intricacy of intricate biological systems, especially those linked to diseases unique to humans. However, these progressions have also ushered in novel obstacles-specifically, the efficient annotation of extensive, long-tailed single-cell data pertaining to disease conditions. To effectively surmount this challenge, we introduce Celler, a state-of-the-art generative pre-training model crafted specifically for the annotation of single-cell data. Celler incorporates two groundbreaking elements: First, we introduced the Gaussian Inflation (GInf) Loss function. By dynamically adjusting sample weights, GInf Loss significantly enhances the model's ability to learn from rare categories while reducing the risk of overfitting for common categories. Secondly, we introduce an innovative Hard Data Mining (HDM) strategy into the training process, specifically targeting the challenging-to-learn minority data samples, which significantly improved the model's predictive accuracy. Additionally, to further advance research in this field, we have constructed a large-scale single-cell dataset: Celler-75, which encompasses 40 million cells distributed across 80 human tissues and 75 specific diseases. This dataset provides critical support for comprehensively exploring the potential of single-cell technology in disease research. Our code is available at this https URL. 

**Abstract (ZH)**: 近期单细胞技术的突破为解读复杂生物系统，尤其是与人类特有疾病相关的分子复杂性，带来了前所未有的机会。然而，这些进展也带来了一个新型挑战——高效标注广泛的长尾单细胞疾病数据。为有效克服这一挑战，我们引入了Celler，一种专为单细胞数据标注设计的先进生成预训练模型。Celler结合了两项创新元素：首先，我们引入了高斯膨胀损失函数（GInf Loss）。通过动态调整样本权重，GInf Loss显著增强了模型从稀有类别学习的能力，同时降低了对常见类别的过拟合风险。其次，我们在训练过程中引入了一种创新的难样本挖掘策略（HDM），专门针对难以学习的小众数据样本，显著提高了模型的预测准确性。此外，为了进一步推动该领域的研究，我们构建了一个大规模单细胞数据集——Celler-75，其中包括分布在80个人体组织和75种特定疾病中的4000万细胞。该数据集为全面探索单细胞技术在疾病研究中的潜力提供了关键支持。我们的代码可在以下链接获取：this https URL。 

---
# ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding 

**Title (ZH)**: ObscuraCoder: 通过混淆基础实现高效代码LM预训练 

**Authors**: Indraneil Paul, Haoyi Yang, Goran Glavaš, Kristian Kersting, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2504.00019)  

**Abstract**: Language models (LMs) have become a staple of the code-writing toolbox. Their pre-training recipe has, however, remained stagnant over recent years, barring the occasional changes in data sourcing and filtering strategies. In particular, research exploring modifications to Code-LMs' pre-training objectives, geared towards improving data efficiency and better disentangling between syntax and semantics, has been noticeably sparse, especially compared with corresponding efforts in natural language LMs. In this work, we examine grounding on obfuscated code as a means of helping Code-LMs look beyond the surface-form syntax and enhance their pre-training sample efficiency. To this end, we compile ObscuraX, a dataset of approximately 55M source and obfuscated code pairs in seven languages. Subsequently, we pre-train ObscuraCoder models, ranging in size from 255M to 2.8B parameters, on a 272B-token corpus that includes ObscuraX and demonstrate that our obfuscation-based pre-training recipe leads to consistent improvements in Code-LMs' abilities compared to both vanilla autoregressive pre-training as well as existing de-obfuscation (DOBF) objectives. ObscuraCoder demonstrates sizeable gains across multiple tests of syntactic and semantic code understanding, along with improved capabilities in multilingual code completion, multilingual code commit summarization, and multi-purpose library-oriented code generation. 

**Abstract (ZH)**: 语言模型（LMs）已成为代码编写工具箱中的标准工具。然而，它们的预训练配方近年来尚未发生变化，仅偶尔在数据来源和筛选策略上有所改变。特别是针对代码-LMs的预训练目标进行改进的研究，以提高数据效率并更好地分离语法和语义的研究相对较少，尤其与自然语言LMs相应领域的努力相比。在这项工作中，我们研究将混淆代码作为手段，帮助代码-LMs超越表面语法结构，并提高其预训练样本效率。为此，我们编译了一个包含约5500万对源代码和混淆代码的ObscuraX数据集，涉及七种编程语言。随后，我们使用包含ObscuraX的272亿词语料库，对从2.55亿到28亿参数不等的ObscuraCoder模型进行了预训练，并证明基于混淆代码的预训练配方在与传统的自回归预训练以及现有的去混淆（DOBF）目标相比时，能够显著提高代码-LMs的能力。ObscuraCoder在多种语法和语义代码理解测试中表现出显著的进步，并且在多语言代码补全、多语言代码提交总结以及多用途面向库的代码生成方面也展现了增强的能力。 

---
# Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices 

**Title (ZH)**: 我们到了吗？移动设备上大语言模型应用的效率测量研究 

**Authors**: Xiao Yan, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.00002)  

**Abstract**: Recent advancements in large language models (LLMs) have prompted interest in deploying these models on mobile devices to enable new applications without relying on cloud connectivity. However, the efficiency constraints of deploying LLMs on resource-limited devices present significant challenges. In this paper, we conduct a comprehensive measurement study to evaluate the efficiency tradeoffs between mobile-based, edge-based, and cloud-based deployments for LLM applications. We implement AutoLife-Lite, a simplified LLM-based application that analyzes smartphone sensor data to infer user location and activity contexts. Our experiments reveal that: (1) Only small-size LLMs (<4B parameters) can run successfully on powerful mobile devices, though they exhibit quality limitations compared to larger models; (2) Model compression is effective in lower the hardware requirement, but may lead to significant performance degradation; (3) The latency to run LLMs on mobile devices with meaningful output is significant (>30 seconds), while cloud services demonstrate better time efficiency (<10 seconds); (4) Edge deployments offer intermediate tradeoffs between latency and model capabilities, with different results on CPU-based and GPU-based settings. These findings provide valuable insights for system designers on the current limitations and future directions for on-device LLM applications. 

**Abstract (ZH)**: 近期大规模语言模型的进步激发了在移动设备上部署这些模型的兴趣，以在无需依靠云连接的情况下启用新的应用程序。然而，资源受限设备上部署大规模语言模型的效率限制带来了重大挑战。本文开展了一项全面的测量研究，评估了在移动设备、边缘设备和云端部署大规模语言模型应用之间的效率权衡。我们实现了AutoLife-Lite，这是一个简化的大规模语言模型应用，分析智能手机传感器数据以推断用户的位置和活动背景。实验结果显示：（1）只有参数量小于4B的小型语言模型能在强大的移动设备上成功运行，但其在质量上存在局限性，相比大型模型；（2）模型压缩有效降低了硬件需求，但可能导致性能显著下降；（3）使用移动设备运行大规模语言模型以获得有意义输出的延迟显著（>30秒），而云服务表现出更好的时间效率（<10秒）；（4）边缘部署在延迟和模型能力之间提供了中间权衡，不同结果分别在基于CPU和基于GPU的设置下。这些发现为系统设计者提供了关于当前限制和未来方向的重要见解，以支持设备端的大规模语言模型应用。 

---
